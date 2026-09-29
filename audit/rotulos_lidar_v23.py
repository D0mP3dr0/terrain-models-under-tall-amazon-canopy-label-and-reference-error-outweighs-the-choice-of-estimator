"""Per-cell ground labels from GEDI L2A and ICESat-2 ATL08.

Shots are quality-filtered (GEDI `quality_flag` and waveform modes; ATL08 photon count
and agreement between terrain estimators), their ground elevation is converted from the
WGS84 ellipsoid to EGM2008 orthometric height, and the label is DSM(GLO-30) - ground.
Values that are physically implausible for the biome are rejected before per-cell
aggregation.

Output: one `.npz` per (biome, quadrant), matched by node index; no `.pt` file is
rewritten.

    idx_no             node index within the quadrant (0..12,959,999)
    n_disparos         number of approved shots in the cell
    y_novo             DSM(GLO-30) - ground (LiDAR ground already orthometric)
    y_bruto            same without datum conversion, kept for traceability
    n_geoide           EGM2008 undulation applied at the shot, in metres
    peso               quality weight in [0,1]; 0 = discard
    bom                boolean mask of the hard criterion
    h_dossel           measured canopy height (top - ground)
    rh50 rh75 rh90 rh98  relative height profile above ground
    dispersao_solo     GEDI: std of the six ground estimates a1..a6; ATL08: |best_fit - median|
    declividade_graus  ATL08 `terrain_slope`, the reference slope label
    rugosidade_m       from the collinear triplet in `fila_aquisicao_alvos`
    fonte              1 GEDI, 2 ATL08, 3 both

DATUM
-----
GEDI and ATL08 give height above the WGS84 ellipsoid; Copernicus GLO-30 gives height
above the EGM2008 geoid. Left in the label, the difference would make the model learn
geoid undulation instead of DSM error.

The conversion is applied per shot with the official EGM2008 grid (NGA, via PROJ;
see `geoide_v23`), not as a per-biome constant: undulation varies within a biome box,
and a constant offset would leave a smooth large-scale ramp in the label. No assumption
about Copernicus accuracy enters the conversion.

BARE-GROUND RESIDUAL
--------------------
After the geoid correction, the median of (DSM - ground) over bare ground is the vertical
bias of Copernicus on exposed terrain. It is reported, not subtracted, so the model still
learns to correct it. `y_bruto` keeps the unconverted value so the chain can be traced.

    python rotulos_lidar_v23.py                    # all biomes
    python rotulos_lidar_v23.py --biomas amazonia
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

from fila_aquisicao_alvos import (  # noqa: E402
    BIOMAS, GRID, SAIDA, Painel, gravar_atomico,
)

GEEDIR = Path(r"D:\GNN_TOPO\SATELITES\gee")
LATDIR = Path(r"D:\GNN_TOPO\SATELITES\laterais")
QUAD = GRID // 2
QUADS = {"Q1": (0, 0), "Q2": (0, 1), "Q3": (1, 0), "Q4": (1, 1)}

FILL = 1e38                 # GEDI and ATL08 fill missing values with ~3.4e38
SOLO_NU_M = 3.0             # canopy below this: bare ground, where DSM = DTM
MIN_SOLO_NU = 500           # fewer bare-ground shots than this: bias not estimated

# ---- GEDI hard criterion
#
# A shot passes if `quality_flag == 1` and at least GEDI_NUM_MODOS_MIN waveform modes
# were detected (a waveform with no mode has no usable ground return). The reference
# for this choice is bare ground (Landsat tree cover <= 10%), where DSM - ground should
# be zero. `sensitivity`, `surface_flag`, `degrade_flag` and a hard cut on the spread of
# the six ground estimates are not applied as filters; the spread enters the weight.
GEDI_NUM_MODOS_MIN = 1

# The spread of the six ground estimates is treated as uncertainty, not as a gate:
# weight = 1 / (1 + spread / DISP_ESCALA), i.e. 0.5 at DISP_ESCALA metres and never
# zero, so an uncertain shot counts less instead of being dropped.
DISP_ESCALA = 4.0

# ---- ATL08 hard criterion
#
# A segment passes with at least ATL_FOTONS_MIN terrain photons, agreement within
# ATL_DISCORD_MAX between the product's two terrain estimators (`h_te_best_fit` and
# `h_te_median`), no cloud and no saturation. `terrain_flg` is not used because its
# polarity is not established; `msw_flag` is treated as equivalent to `cloud_flag_atm`;
# `h_te_uncertainty` is not comparable across biomes, so it enters the weight only.
ATL_FOTONS_MIN = 50         # terrain photons in the 100 m segment
ATL_DISCORD_MAX = 2.0       # metres between h_te_best_fit and h_te_median

RH_GUARDAR = [50, 75, 90, 98]

# ---- label plausibility limits, derived per biome (see `limites_do_bioma`).
#
# A fixed bound does not transfer across biomes: the target is DSM - ground, and DEM
# error grows with slope because horizontal geolocation error becomes vertical error on
# steep terrain, widening the tail of the target more than its centre. The GLO-30 DSM
# (TanDEM-X) has its phase centre between ground and canopy top, so the upper bound is
#
#     max canopy height of the biome  +  K_GORDURA * max DEM error
#
# with canopy height from the p99.99 of `h_dossel` of approved shots and DEM error from
# the p99.9 of |target|. The lower bound is minus the same error term, since ground can
# lie above the DSM where the DEM underestimates. K_GORDURA is the only chosen constant;
# it places the bound in the sparsely populated upper tail of the target.
K_GORDURA = 3.0

# Geographic sanity bound, independent of any distribution: orthometric ground must lie
# within the Copernicus elevation range of the box widened by this margin.
MARGEM_ALTITUDE_M = 500.0

DOSSEL_MIN_M = -1.0        # zero with a noise margin; negative canopy height is not physical
RH_MIN_M = -10.0           # a relative-height percentile may fall slightly below ground


def limites_do_bioma(d, dsm_faixa: tuple, painel) -> dict:
    """Plausibility limits derived from this biome's own shots (see the block above).

    `d` holds the shots with `y_novo`, `h_dossel` and `bom`; `dsm_faixa` is the (min, max)
    Copernicus elevation in the box. Returns the limits and the quantities behind them.
    """
    bom = d["bom"].to_numpy().astype(bool)
    y = d["y_novo"].to_numpy(np.float64)[bom]
    h = d["h_dossel"].to_numpy(np.float64)[bom]
    y, h = y[np.isfinite(y)], h[np.isfinite(h)]
    if y.size < 5000 or h.size < 5000:
        raise RuntimeError("amostra pequena demais para derivar limites do bioma")

    dossel_max = float(np.percentile(h, 99.99))
    # DEM error shows up in the tail of the target. Slope is not available per shot here,
    # so the global p99.9 of |target| is used; that tail is dominated by steep terrain.
    erro_dem = float(np.percentile(np.abs(y), 99.9))
    teto = dossel_max + K_GORDURA * erro_dem
    piso = -K_GORDURA * erro_dem
    lim = {"dossel_p9999_m": round(dossel_max, 2),
           "erro_dem_p999_m": round(erro_dem, 2), "k_gordura": K_GORDURA,
           "y_teto_m": round(teto, 2), "y_piso_m": round(piso, 2),
           "dossel_teto_m": round(dossel_max + K_GORDURA * erro_dem, 2),
           "solo_min_m": round(dsm_faixa[0] - MARGEM_ALTITUDE_M, 1),
           "solo_max_m": round(dsm_faixa[1] + MARGEM_ALTITUDE_M, 1),
           "n_amostra": int(y.size)}
    painel.log(f"  limites derivados: dossel p99,99 {dossel_max:.1f} m + {K_GORDURA:g}x "
               f"erro do DEM {erro_dem:.1f} m -> alvo em [{piso:.1f}, {teto:.1f}] m; "
               f"solo em [{lim['solo_min_m']:.0f}, {lim['solo_max_m']:.0f}] m")
    return lim
COLS = ["idx_no", "n_disparos", "y_novo", "y_bruto", "peso", "bom", "h_dossel",
        "rh50", "rh75", "rh90", "rh98", "dispersao_solo", "declividade_graus",
        "rugosidade_m", "n_geoide", "fonte",
        # Direction of the slope measurement, so it acts as a directional-derivative
        # constraint rather than a biased scalar target. See `rugosidade_por_celula`.
        # `decl_err_graus` gives the angular error behind `decl_conf` in degrees.
        "decl_sin", "decl_cos", "decl_conf", "decl_err_graus", "decl_base_m"]

# Angular-error scale for the slope confidence, in degrees: the confidence is 0.5 when
# the angular error equals this value. See `rugosidade_por_celula`.
ESCALA_CONF_GRAUS = 2.0


def _valido(a: np.ndarray) -> np.ndarray:
    return np.isfinite(a) & (np.abs(a) < FILL)


def celula(lat, lon, bb):
    """(row, col) in the biome's 7200x7200 grid; points outside the box get -1."""
    w, s, e, n = bb
    r = np.floor((n - lat) / (n - s) * GRID).astype(np.int64)
    c = np.floor((lon - w) / (e - w) * GRID).astype(np.int64)
    fora = (r < 0) | (r >= GRID) | (c < 0) | (c >= GRID)
    r[fora], c[fora] = -1, -1
    return r, c


# ------------------------------------------------------------------ GEDI
def preparar_gedi(bioma: str, painel: Painel) -> pd.DataFrame | None:
    # Prefer the deduplicated file: the raw download lists each orbit under two collection
    # versions (GEDI02_A.002 and .003), so `shot_number` repeats and every shot would
    # otherwise be counted twice.
    arq = SAIDA / f"gedi02_a_{bioma}_dedup.parquet"
    if not arq.exists():
        arq = SAIDA / f"gedi02_a_{bioma}.parquet"
    if not arq.exists():
        arq = SAIDA / f"gedi_{bioma}.parquet"
    if not arq.exists():
        painel.log(f"  {bioma}: sem parquet do GEDI — a tarefa `gedi` nao terminou")
        return None
    d = pd.read_parquet(arq)
    n0 = len(d)

    solo = d["elev_solo"].to_numpy(np.float64)
    topo = d["elev_topo"].to_numpy(np.float64)
    ok = _valido(solo) & _valido(topo)

    # Spread across the six ground-detection algorithm settings: a direct measure of how
    # ambiguous ground detection was for this shot.
    cols_a = [c for c in ("solo_a1", "solo_a2", "solo_a3", "solo_a4", "solo_a5",
                          "solo_a6") if c in d.columns]
    if len(cols_a) >= 3:
        # `copy=True` is required: with homogeneous dtypes pandas may return a read-only
        # view of its internal block, and the assignment below would fail.
        A = d[cols_a].to_numpy(np.float64, copy=True)
        A[~_valido(A)] = np.nan
        with np.errstate(invalid="ignore"):
            disp = np.nanstd(A, axis=1)
        disp = np.where(np.isfinite(disp), disp, np.inf)
    else:
        painel.log(f"  {bioma}: GEDI sem as seis estimativas a1..a6 — dispersao nao "
                   f"entra no criterio (colunas presentes: {cols_a})")
        disp = np.zeros(len(d))

    dura = (
        ok
        & (d.get("quality_flag", pd.Series(1, index=d.index)).to_numpy() == 1)
        & (d.get("num_modos", pd.Series(1, index=d.index)).to_numpy()
           >= GEDI_NUM_MODOS_MIN)
    )

    # Weight = uncertainty from the spread of the six ground estimates:
    # 1 / (1 + spread / DISP_ESCALA) inside the hard criterion, 0 outside.
    # `sensitivity` is used neither in the criterion nor in the weight.
    d_eff = np.where(np.isfinite(disp), disp, 0.0)
    peso = np.where(dura, 1.0 / (1.0 + d_eff / DISP_ESCALA), 0.0)

    r = pd.DataFrame({
        "lat": d["lat"].to_numpy(np.float64), "lon": d["lon"].to_numpy(np.float64),
        "solo": solo, "h_dossel": topo - solo,
        "dispersao_solo": np.where(np.isfinite(disp), disp, np.nan),
        "bom": dura, "peso": peso, "fonte": 1,
    })
    for q in RH_GUARDAR:
        col = f"rh{q}"
        r[col] = d[col].to_numpy(np.float64) if col in d.columns else np.nan
        r.loc[~_valido(r[col].to_numpy()), col] = np.nan
    r = r[ok].reset_index(drop=True)
    painel.log(f"  {bioma}: GEDI {n0:,} disparos -> {len(r):,} com solo e topo validos; "
               f"{int(r.bom.sum()):,} passam no criterio duro "
               f"({100*r.bom.mean():.1f}%)")
    return r


# ------------------------------------------------------------------ ATL08
def preparar_atl08(bioma: str, painel: Painel) -> pd.DataFrame | None:
    arq = SAIDA / f"atl08_{bioma}.parquet"
    if not arq.exists():
        painel.log(f"  {bioma}: sem parquet do ATL08")
        return None
    d = pd.read_parquet(arq)
    n0 = len(d)

    solo = d["h_terreno"].to_numpy(np.float64)
    mediana = d.get("h_terreno_mediana", pd.Series(np.nan, index=d.index)).to_numpy(np.float64)
    ok = _valido(solo) & _valido(mediana)
    # disagreement between the product's two terrain estimators
    discord = np.where(ok, np.abs(solo - mediana), np.inf)
    inc = d.get("h_terreno_inc", pd.Series(np.nan, index=d.index)).to_numpy(np.float64)
    inc = np.where(_valido(inc), inc, np.nan)
    nfot = d.get("n_fotons_terreno", pd.Series(1e9, index=d.index)).to_numpy(np.float64)
    dossel = d.get("h_dossel", pd.Series(np.nan, index=d.index)).to_numpy(np.float64)
    dossel = np.where(_valido(dossel), dossel, np.nan)
    decl = d.get("declividade", pd.Series(np.nan, index=d.index)).to_numpy(np.float64)
    decl = np.where(_valido(decl), np.degrees(np.arctan(np.abs(decl))), np.nan)

    dura = (
        ok
        & (nfot >= ATL_FOTONS_MIN)
        & (discord <= ATL_DISCORD_MAX)
        # `nuvem == 0` also implies `espalhamento_flag == 0`, so only one is tested.
        & (d.get("nuvem", pd.Series(0, index=d.index)).to_numpy() == 0)
        & (d.get("saturacao_flag", pd.Series(0, index=d.index)).to_numpy() == 0)
    )
    # Weight: mean of estimator agreement, photon count and declared uncertainty. The
    # declared uncertainty only weights, never rejects; it counts as 0.5 when missing.
    p_disc = np.clip(1.0 - discord / ATL_DISCORD_MAX, 0, 1)
    p_fot = np.clip(np.log10(np.maximum(nfot, 1)) / np.log10(500.0), 0, 1)
    p_inc = np.where(np.isfinite(inc), np.clip(1.0 - inc / 5.0, 0, 1), 0.5)
    peso = np.where(dura, (p_disc + p_fot + p_inc) / 3.0, 0.0)

    r = pd.DataFrame({
        "lat": d["lat"].to_numpy(np.float64), "lon": d["lon"].to_numpy(np.float64),
        "solo": solo, "h_dossel": dossel,
        "dispersao_solo": np.where(np.isfinite(discord), discord, np.nan),
        "declividade_graus": decl, "bom": dura, "peso": peso, "fonte": 2,
    })
    for q in RH_GUARDAR:
        r[f"rh{q}"] = np.nan
    r = r[ok].reset_index(drop=True)
    painel.log(f"  {bioma}: ATL08 {n0:,} segmentos -> {len(r):,} com terreno valido; "
               f"{int(r.bom.sum()):,} passam no criterio duro "
               f"({100*r.bom.mean():.1f}%)")
    return r


# ------------------------------------------------------------------ roughness
def rugosidade_por_celula(bioma: str, bb) -> pd.DataFrame | None:
    """Along-track slope and roughness aggregated per cell, keeping the direction.

    Along-track slope is the component along one direction, not the steepest slope, so
    as a scalar target it would bias the model low. With its direction it becomes a
    constraint on the directional derivative of the corrected surface:

        grad(z') . u  =  s        with  u = (sin azimuth, cos azimuth)

    Direction is stored as sin/cos rather than azimuth, because azimuth is circular and
    cannot be averaged directly; the per-cell aggregation averages unit vectors.

    Slope is signed: positive means uphill in the azimuth direction (from point a to c).
    The direction therefore spans 360 degrees and the ordinary circular mean is used,
    without angle doubling.

    `base_m` is kept because the constraint applies to the surface smoothed over the
    measurement baseline (much longer than the 30 m cell), not to the local gradient.
    """
    arq = SAIDA / f"declividade_trilha_{bioma}.parquet"
    if not arq.exists():
        return None
    d = pd.read_parquet(arq, columns=["lat", "lon", "rugosidade_m", "declividade_graus",
                                      "azimute_graus", "base_m"])
    r, c = celula(d["lat"].to_numpy(np.float64), d["lon"].to_numpy(np.float64), bb)
    d = d.assign(r=r, c=c)
    d = d[(d.r >= 0)]

    # Ordinary circular mean of the azimuth: with signed slope the direction spans 360
    # degrees, so averaging sin and cos of the plain angle is correct.
    az = np.radians(d["azimute_graus"].to_numpy(np.float64))
    d = d.assign(_s=np.sin(az), _c=np.cos(az))

    g = d.groupby(["r", "c"], sort=False).agg(
        rugosidade_m=("rugosidade_m", "median"),
        decl_trilha=("declividade_graus", "median"),
        base_m=("base_m", "median"),
        _s=("_s", "mean"), _c=("_c", "mean"),
        n_trilha=("declividade_graus", "size")).reset_index()

    ang = np.arctan2(g["_s"].to_numpy(), g["_c"].to_numpy())
    g["decl_sin"] = np.sin(ang)
    g["decl_cos"] = np.cos(ang)

    # `decl_conf`: confidence of the slope constraint, from its angular error. The mean
    # resultant length of the azimuths is not used: it is 1 by construction in a cell
    # with a single measurement.
    #
    # Angular error = atan(2 * |roughness| / baseline): roughness is the deviation of the
    # middle point from the line joining the flanks, and a shorter baseline turns the same
    # height deviation into a larger angle. The confidence has the same form as the label
    # weight, 1 / (1 + error / ESCALA_CONF_GRAUS): 1 for a clean measurement, 0.5 when
    # the error equals the scale.
    rug = g["rugosidade_m"].to_numpy(np.float64)
    base = np.maximum(g["base_m"].to_numpy(np.float64), 1.0)
    err_ang = np.degrees(np.arctan(np.abs(2.0 * rug) / base))
    g["decl_conf"] = (1.0 / (1.0 + err_ang / ESCALA_CONF_GRAUS)).astype(np.float32)
    g["decl_err_graus"] = err_ang.astype(np.float32)
    return g.drop(columns=["_s", "_c"])


# ------------------------------------------------------------------ assembly
def montar_bioma(bioma: str, painel: Painel, prog=None, prefixo: str = "rotulos") -> dict:
    import rasterio

    bb = BIOMAS[bioma]
    partes = [x for x in (preparar_gedi(bioma, painel),
                          preparar_atl08(bioma, painel)) if x is not None]
    if not partes:
        raise RuntimeError(f"{bioma}: nem GEDI nem ATL08 disponiveis")
    d = pd.concat(partes, ignore_index=True)
    if "declividade_graus" not in d.columns:
        d["declividade_graus"] = np.nan

    dsm_arq = GEEDIR / f"glo30_{bioma}.tif"
    if not dsm_arq.exists():
        raise RuntimeError(
            f"{dsm_arq.name} ausente. O rotulo novo e DSM - solo, e o DSM na NOSSA grade "
            f"vem da tarefa `dems`. Sem ele nao ha rotulo — rode `dems` antes.")
    with rasterio.open(dsm_arq) as src:
        if src.width != GRID or src.height != GRID:
            raise RuntimeError(f"{dsm_arq.name} e {src.width}x{src.height}, "
                               f"esperado {GRID}x{GRID}")
        dsm = src.read(1).astype(np.float32)

    r, c = celula(d["lat"].to_numpy(), d["lon"].to_numpy(), bb)
    d = d.assign(r=r, c=c)
    d = d[d.r >= 0].reset_index(drop=True)
    if d.empty:
        raise RuntimeError(f"{bioma}: nenhum disparo dentro da caixa")

    z = dsm[d.r.to_numpy(), d.c.to_numpy()]
    d["y_bruto"] = z.astype(np.float64) - d["solo"].to_numpy()
    d = d[np.isfinite(d.y_bruto.to_numpy())].reset_index(drop=True)

    # ---- datum: ellipsoid -> geoid, per shot
    from geoide_v23 import undulacao
    N = undulacao(d.lon.to_numpy(), d.lat.to_numpy())
    d["n_geoide"] = N
    d["solo_orto"] = d["solo"].to_numpy() - N
    z2 = dsm[d.r.to_numpy(), d.c.to_numpy()].astype(np.float64)
    d["y_novo"] = z2 - d["solo_orto"].to_numpy()
    painel.log(f"  {bioma}: ondulacao do geoide de {N.min():+.2f} a {N.max():+.2f} m "
               f"(amplitude {N.max()-N.min():.2f} m) — aplicada por disparo")

    # ---- physical plausibility cut
    #
    # The quality criterion uses processing flags only and does not check whether the
    # resulting value is possible, so shots with ground far from the terrain can pass
    # and would give the target an extreme tail. The cut applies to every field derived
    # from the shot (target, raw target, canopy height, RH percentiles, orthometric
    # ground), since an implausible value in any of them marks a faulty shot. Rejected
    # shots keep their row with `bom = False` and `peso = 0`, so they stay traceable.
    lim = limites_do_bioma(d, (float(np.nanmin(dsm)), float(np.nanmax(dsm))), painel)
    impossivel = np.zeros(len(d), dtype=bool)
    for col in ("y_novo", "y_bruto"):
        v = d[col].to_numpy(np.float64)
        impossivel |= ~(np.isfinite(v) & (v >= lim["y_piso_m"]) & (v <= lim["y_teto_m"]))
    hd = d["h_dossel"].to_numpy(np.float64)
    impossivel |= np.isfinite(hd) & ((hd < DOSSEL_MIN_M) | (hd > lim["dossel_teto_m"]))
    for q in RH_GUARDAR:
        c = f"rh{q}"
        if c in d.columns:
            v = d[c].to_numpy(np.float64)
            impossivel |= np.isfinite(v) & ((v < RH_MIN_M) | (v > lim["dossel_teto_m"]))
    # Geographic sanity bound, independent of any distribution.
    so = d["solo_orto"].to_numpy(np.float64)
    impossivel |= ~(np.isfinite(so) & (so >= lim["solo_min_m"]) & (so <= lim["solo_max_m"]))
    n_imp = int((impossivel & d.bom.to_numpy().astype(bool)).sum())
    lim["reprovados"] = n_imp
    d.loc[impossivel, "bom"] = False
    d.loc[impossivel, "peso"] = 0.0
    painel.log(f"  {bioma}: corte de plausibilidade fisica reprovou {n_imp:,} disparos "
               f"que passavam no criterio de qualidade "
               f"({100*n_imp/max(int(d.bom.to_numpy().sum())+n_imp,1):.2f}% dos aprovados)")

    # ---- Copernicus bias on bare ground: measured and reported, not removed
    nu = d.bom.to_numpy() & (d.h_dossel.to_numpy() < SOLO_NU_M)
    n_nu = int(np.nansum(nu))
    if n_nu >= MIN_SOLO_NU:
        vies = float(np.nanmedian(d.y_novo.to_numpy()[nu]))
        disp = float(np.nanpercentile(np.abs(d.y_novo.to_numpy()[nu] - vies), 68))
        nota_vies = (f"mediana de {n_nu:,} disparos com dossel < {SOLO_NU_M} m; "
                     f"dispersao (p68 do desvio) {disp:.2f} m")
        painel.log(f"  {bioma}: vies do Copernicus em solo exposto {vies:+.2f} m "
                   f"— MEDIDO, nao removido ({n_nu:,} disparos)")
        if abs(vies) > 3.0:
            painel.log(f"  {bioma}: ATENCAO — vies de {vies:+.2f} m em solo exposto e "
                       f"grande demais para ser so ruido do Copernicus. Conferir a "
                       f"cadeia de datum antes de treinar com este rotulo.")
    else:
        vies, disp = float("nan"), float("nan")
        nota_vies = (f"apenas {n_nu:,} disparos de solo nu (minimo {MIN_SOLO_NU}) — "
                     f"vies nao estimavel neste bioma")
        painel.log(f"  {bioma}: {nota_vies}")

    rug = rugosidade_por_celula(bioma, bb)
    if rug is not None:
        d = d.merge(rug, on=["r", "c"], how="left")
        d["declividade_graus"] = d["declividade_graus"].fillna(d["decl_trilha"])
    else:
        d["rugosidade_m"] = np.nan

    # ---- per-cell aggregation on the biome grid
    #
    # Cell values (medians) are computed from approved shots only, so rejected shots
    # cannot shift the cell label. `bom` and `peso` come from all shots in the cell, as
    # they are the criterion itself; `n_disparos` counts approved shots and
    # `n_total_disparos` all shots.
    d["fonte_g"] = (d.fonte == 1).astype(np.int8)
    d["fonte_a"] = (d.fonte == 2).astype(np.int8)
    m_bom = d["bom"].to_numpy().astype(bool)
    d_bom = d[m_bom]
    n_celulas_com_bom = len(d_bom.groupby(["r", "c"]).size())
    painel.log(f"  {bioma}: agregando {len(d_bom):,} disparos APROVADOS de {len(d):,} "
               f"em {n_celulas_com_bom:,} celulas; a mediana da celula deixa de "
               f"misturar aprovado com reprovado")
    agg = {
        "y_novo": ("y_novo", "median"), "y_bruto": ("y_bruto", "median"),
        "n_geoide": ("n_geoide", "median"),
        "peso": ("peso", "max"), "bom": ("bom", "max"),
        "h_dossel": ("h_dossel", "median"),
        "dispersao_solo": ("dispersao_solo", "median"),
        "declividade_graus": ("declividade_graus", "median"),
        "rugosidade_m": ("rugosidade_m", "median"),
        # Direction and scale of the slope measurement were already aggregated per cell
        # in `rugosidade_por_celula`; every row of a cell carries the same value.
        "decl_sin": ("decl_sin", "first"),
        "decl_cos": ("decl_cos", "first"),
        "decl_conf": ("decl_conf", "first"),
        "decl_err_graus": ("decl_err_graus", "first"),
        "decl_base_m": ("base_m", "first"),
        "n_disparos": ("y_novo", "size"),
        "tem_g": ("fonte_g", "max"), "tem_a": ("fonte_a", "max"),
    }
    for q in RH_GUARDAR:
        agg[f"rh{q}"] = (f"rh{q}", "median")

    # Values come from approved shots only; the criterion (`bom`, `peso`) from all shots,
    # since "does this cell hold any reliable measurement?" must consider all of them.
    g = d_bom.groupby(["r", "c"], sort=False).agg(**agg).reset_index()
    crit = (d.groupby(["r", "c"], sort=False)
            .agg(bom_qq=("bom", "max"), peso_qq=("peso", "max"),
                 n_total=("y_novo", "size")).reset_index())
    g = g.merge(crit, on=["r", "c"], how="outer")
    g["bom"] = g["bom_qq"].fillna(False)
    g["peso"] = g["peso_qq"].fillna(0.0)
    g["n_total_disparos"] = g["n_total"].fillna(0)
    g["n_disparos"] = g["n_disparos"].fillna(0)
    g = g.drop(columns=["bom_qq", "peso_qq", "n_total"])
    # A cell without an approved shot is kept as rejected, with null values rather than
    # the median of rejected shots.
    sem = ~g["bom"].to_numpy().astype(bool)
    for c in ("y_novo", "y_bruto", "h_dossel", *[f"rh{q}" for q in RH_GUARDAR]):
        if c in g.columns:
            g.loc[sem, c] = np.nan
    g["fonte"] = (g.tem_g.fillna(0).to_numpy() + 2 * g.tem_a.fillna(0).to_numpy())

    LATDIR.mkdir(parents=True, exist_ok=True)
    diag = {"bioma": bioma, "limites_derivados": lim,
            "fontes": sorted(set(d.fonte.tolist())),
            "datum": {"fonte": "EGM2008 via PROJ, por disparo",
                      "n_min_m": float(N.min()), "n_max_m": float(N.max()),
                      "amplitude_m": float(N.max() - N.min())},
            "vies_copernicus_solo_exposto_m": vies,
            "vies_dispersao_p68_m": disp, "nota_vies": nota_vies,
            "n_solo_nu": n_nu,
            "n_disparos": int(len(d)), "n_celulas": int(len(g)),
            "criterio": {"gedi_regras": ["quality_flag == 1",
                                         f"num_modos >= {GEDI_NUM_MODOS_MIN}"],
                         "gedi_peso": f"1 / (1 + dispersao / {DISP_ESCALA})",
                         "gedi_criterio_reescrito_em": "2026-08-05, a partir da regua de "
                                                       "solo exposto nos quatro biomas",
                         "atl_fotons_min": ATL_FOTONS_MIN,
                         "atl_discordancia_max_m": ATL_DISCORD_MAX,
                         "atl_nao_filtrados": ["terrain_flg (polaridade nao "
                                               "estabelecida)",
                                               "h_te_uncertainty (nao comparavel "
                                               "entre biomas)",
                                               "msw_flag (identico a cloud_flag_atm)"]},
            "quadrantes": {}}

    for q, (qr, qc) in QUADS.items():
        m = ((g.r >= qr * QUAD) & (g.r < (qr + 1) * QUAD) &
             (g.c >= qc * QUAD) & (g.c < (qc + 1) * QUAD)).to_numpy()
        sub = g[m]
        if sub.empty:
            painel.log(f"  {bioma}/{q}: nenhuma celula com rotulo")
            continue
        idx_no = ((sub.r.to_numpy() - qr * QUAD) * QUAD +
                  (sub.c.to_numpy() - qc * QUAD)).astype(np.int64)
        dados = {"idx_no": idx_no}
        for col in COLS[1:]:
            v = sub[col].to_numpy() if col in sub.columns else np.full(len(sub), np.nan)
            dados[col] = v.astype(np.float32)
        alvo = LATDIR / f"{prefixo}_{bioma}_{q}.npz"
        # The temporary name must end in `.npz`: `savez_compressed` appends the extension
        # otherwise, and `replace` would not find the file.
        tmp = alvo.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **dados)
        tmp.replace(alvo)
        bons = int(np.nansum(dados["bom"] > 0.5))
        diag["quadrantes"][q] = {
            "n_celulas": int(len(sub)), "n_bons": bons,
            "frac_boa": float(bons / max(len(sub), 1)),
            "mae_y_novo_m": float(np.nanmedian(np.abs(dados["y_novo"]))),
            "dossel_mediano_m": float(np.nanmedian(dados["h_dossel"])),
            "arquivo": str(alvo),
        }
        painel.log(f"  {bioma}/{q}: {len(sub):,} celulas, {bons:,} boas "
                   f"({100*bons/max(len(sub),1):.1f}%), |y| mediano "
                   f"{diag['quadrantes'][q]['mae_y_novo_m']:.2f} m")
        if prog:
            prog(len(diag["quadrantes"]), 4, f"{bioma}/{q}")
    return diag


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--biomas", nargs="+", default=list(BIOMAS))
    ap.add_argument("--prefixo", default="rotulos",
                    help="nome dos .npz de saida. Use outro para uma prova de cadeia "
                         "sem sobrescrever o rotulo de treino — foi assim que a "
                         "cadeia foi conferida so com ATL08, antes de o GEDI fechar.")
    a = ap.parse_args()

    painel = Painel([f"rotulos:{b}" for b in a.biomas])
    saida = {}
    for b in a.biomas:
        painel.comecar(f"rotulos:{b}")
        t0 = time.time()
        saida[b] = montar_bioma(b, painel, painel.progresso(f"rotulos:{b}"),
                                prefixo=a.prefixo)
        saida[b]["segundos"] = round(time.time() - t0, 1)
        painel.terminar(f"rotulos:{b}", True)
    (RAIZ / "results").mkdir(exist_ok=True)
    gravar_atomico(RAIZ / "results" / f"{a.prefixo}_lidar_v23.json",
                   json.dumps(saida, indent=2, ensure_ascii=False, default=str))
    print(f"\n  rotulos de {len(saida)} bioma(s) -> {LATDIR}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
