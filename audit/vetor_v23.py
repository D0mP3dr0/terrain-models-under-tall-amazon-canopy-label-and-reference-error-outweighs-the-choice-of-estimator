"""V23 feature vector: declared here, frozen to `vetor_v23.json`, and loaded by node index.

The layout is checked on every load, so a missing or reshaped lateral file makes the load
fail instead of silently training on zeros.

Design choices:
- Each data source is a separate branch with its own encoder: topo, optico, agua, alos_l,
  palsar2, s1_umida, s1_seca, s2_amplitude, plus the regime gate. `alos_l` (2006-2011)
  and `palsar2` (2019-2021) are both L-band but kept apart because only the first is
  contemporaneous with the target. New branches inherit the gate prior column of their
  physical family (`HERANCA_PRIOR`); the prior matrix remains trainable.
- Copernicus GLO-30 is the only DEM input. FABDEM is excluded because it was already
  corrected against the same GLO-30 with the same ICESat-2/GEDI labels (it was the
  strongest of 49 columns, rho = -0.486, above elevation itself); NASADEM and AW3D30 are
  excluded by design. All three, plus ANADEM and GEDTM30, remain as benchmarks.
- Anchors that fail the quality criterion are removed from every split, including test
  and hold-out. The partition is computed before filtering with the same seed, so the
  evaluated set is a subset of the baseline's, and `comparacao_v23.py` re-scores the
  baseline on it from stored per-node predictions.

Usage:
    python vetor_v23.py --congelar        # inspect lateral files and write the specification
    python vetor_v23.py --conferir        # check the frozen specification against disk
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import os
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

# Data directory, overridable with the V23_LATDIR environment variable. An environment
# variable is used because monkeypatching the imported module does not reach the script
# when it runs as `__main__` (it has its own globals).
LATDIR = Path(os.environ.get("V23_LATDIR", r"D:\GNN_TOPO\SATELITES\laterais"))
ESPEC = RAIZ / "vetor_v23.json"
BIOMAS = ["amazonia", "cerrado", "mata_atlantica", "pantanal"]
QUADS = ["Q1", "Q2", "Q3", "Q4"]

# ---------------------------------------------------------------- terrain geometry
#
# The eight topographic columns come from the V18.5 graph, which computed them with a
# geometry defect: the same spacing was used in both directions.
#
#     generate_graph_gpu.py:171   px = cell_size * 111_320.0
#     generate_graph_gpu.py:176   g_y, g_x = torch.gradient(dem, spacing=px)
#
# In EPSG:4326 one degree of longitude spans 111,320 m only at the equator; at latitude
# `phi` it spans `111,320 * cos(phi)`. Because the true east-west spacing is SMALLER than
# the one used, `dz/dx` is underestimated by the factor `cos(phi)`: under 0.25% in the
# Amazon (2-4 S) and 5-6% in the Atlantic Forest (18-20 S). The error propagates to slope,
# aspect and curvature.
#
# The fix does not require regenerating the graph (~500 GB). Elevation (column 0) is
# correct and derivatives are invariant to a constant offset, so `x[:,0] * elev_std` is
# enough as elevation in metres without recovering `elev_mean`. Nodes are in raster order
# within the quadrant, so the 3600x3600 grid can be rebuilt and derivatives recomputed.
BBOX = {
    "amazonia": [-60.0, -4.0, -58.0, -2.0],
    "cerrado": [-48.0, -16.0, -46.0, -14.0],
    "mata_atlantica": [-42.0, -20.0, -40.0, -18.0],
    "pantanal": [-58.0, -20.0, -56.0, -18.0],
}
GRID, QUAD_N = 7200, 3600          # biome grid size and quadrant side (cells)
GRAU_M = 111_320.0
SLOPE_MAX = 90.0                   # same normalisation as the graph: degrees / 90

# ---- NEW BRANCHES. Each entry is a branch with its OWN encoder in the model.
#
# Per-column normalisation:
#   "dif_glo30"  minus GLO-30, divided by elev_std -> dimensionless
#   "robusta"    (x - median) / IQR with frozen dataset-wide statistics (`_norm_fixa`)
#   "db_jaxa"    JAXA DN -> gamma0 in dB -> robust scaling
#   "agua_jrc"   maps the JRC -128 "no water observed" code to 0 and scales to 0..1
#
# The last two come from checking each branch's mean and std on Amazon Q1, where two
# branches were off:
#
#   agua      mean -126.8    ->  -128 is JRC's "no water observed" code and covers 86.6%
#                                of the box. Used as a number, a branch with mean -127
#                                dominates anything added to it. Physically, -128 means
#                                occurrence = 0.
#
#   palsar2   535 to 65,396  ->  the annual JAXA mosaic on Earth Engine is in DN, not dB.
#                                Same family as ALOS, converted with
#                                gamma0[dB] = 10*log10(DN^2) - 83.0. Median/IQR scaling of
#                                DN is not equivalent: the power distribution is skewed and
#                                the dB one is not, so the same terrain would land at
#                                different positions on the scale.
CF_JAXA_DB = -83.0          # JAXA calibration factor, same as in alos_para_lateral.py
#
# Normalisation is per COLUMN, not per branch, because columns within one branch can have
# different scales.
RAMOS_NOVOS = [
    # The `dem_ind` branch (DEM differences `d_nasadem`, `d_aw3d30`, `d_fabdem`, plus
    # `glo30_norm` and the derived `dem_desvio`) was removed:
    #   - FABDEM: contamination. It is a DTM that already corrected the same Copernicus
    #     GLO-30 with the same ICESat-2 and GEDI labels, so `FABDEM - GLO-30` is already an
    #     estimate of the target (strongest of 49 columns in the Amazon, rho = -0.486,
    #     above elevation itself). FABDEM stays on disk as a benchmark, together with
    #     ANADEM and GEDTM30, which never entered the model.
    #   - NASADEM and AW3D30: design choice of a single DEM source. Both are independent
    #     (C-band InSAR from 2000, PRISM optical stereo) and not trained on our labels;
    #     the model corrects Copernicus from SAR and optical data, not from DEM consensus.
    #   - `glo30_norm`: redundant with `elevation` in `topo` (identical rho = +0.483).
    #   - `dem_desvio`: with only two differences left it reduces to half the absolute
    #     difference between them.
    # The `nasadem`, `aw3d30` and `fabdem` lateral files remain on disk for comparison.
    # Branch `agua`: JRC occurrence and seasonality, in 0-100 and 0-12.
    ("agua", [
        ("agua_jrc", "occurrence",  "agua_jrc", "agua_ocorrencia"),
        ("agua_jrc", "seasonality", "agua_jrc", "agua_sazonal"),
    ]),
    # Branch `alos_l`: L-band, 2006-2011, contemporaneous with the DEM being corrected.
    # Ascending only: PALSAR-1 acquired no fine-beam descending passes over these boxes
    # (ASF search: 268 FBD and 202 FBS ascending, zero fine-beam descending; all 204
    # descending scenes are ~100 m ScanSAR). Declaring `l_hh_desc` would create always-empty
    # columns that become zero after normalisation and enter the ablation as if informative.
    ("alos_l", [
        ("alos", "l_hh_asc",  "robusta", "l_hh_asc"),
        ("alos", "l_hv_asc",  "robusta", "l_hv_asc"),
        ("alos", "l_rvi_asc", "robusta", "l_rvi_asc"),
        ("alos", "l_n_asc",   "robusta", "l_n_asc"),
    ]),
    # Branch `palsar2`: L-band again, 2019-2021 annual mosaic. Kept separate from `alos_l`
    # because the epoch differs: merging them would make the model sum, with the same
    # weights, a measurement contemporaneous with the target and one 12 years later.
    ("palsar2", [
        ("palsar2", "HH", "db_jaxa", "p2_hh"),
        ("palsar2", "HV", "db_jaxa", "p2_hv"),
    ]),
    # Branches `s1_umida` and `s1_seca`: 2021-2023 wet- and dry-season C-band composites.
    #
    # The structure is IDENTICAL in all four biomes on purpose: one model is trained per
    # biome and one across all four, which requires a fixed-width vector even when a
    # column is weak in one biome.
    #
    # Two columns are known to be weak in ONE biome and are kept:
    #
    #   `angulo`  in the Pantanal correlates -0.944 with the raster column because there is
    #             only ONE descending track, so the angle is a pure range ramp. In the other
    #             three biomes two tracks break the ramp (correlation +0.006 to +0.36) and
    #             the angle reflects true viewing geometry.
    #
    #   `vh`      in the Atlantic Forest correlates 0.919 with `vv`; in the other three
    #             biomes all pairs stay below 0.90.
    #
    # The `sd` (temporal standard deviation) bands justify the seasonal composite: they are
    # the ONLY SAR bands whose sign is stable across the four biomes (rho -0.16 to -0.22
    # against the target), whereas `vv` flips from -0.21 in the Atlantic Forest to +0.25 in
    # the Pantanal, and even between Amazon quadrants (-0.211 in Q1, +0.212 in Q3). A
    # feature that flips sign between quadrants harms the chained transfer of experiment B2.
    ("s1_umida", [
        ("s1_umida", "VV_dB",    "escala_fixa", "c_vv_u"),
        ("s1_umida", "VH_dB",    "escala_fixa", "c_vh_u"),
        ("s1_umida", "RVI",      "escala_fixa", "c_rvi_u"),
        ("s1_umida", "sd_VV_dB", "escala_fixa", "c_sdvv_u"),
        ("s1_umida", "sd_VH_dB", "escala_fixa", "c_sdvh_u"),
        ("s1_umida", "angulo",   "escala_fixa", "c_ang_u"),
        ("s1_umida", "n_obs",    "escala_fixa", "c_nobs_u"),
    ]),
    ("s1_seca", [
        ("s1_seca", "VV_dB",    "escala_fixa", "c_vv_s"),
        ("s1_seca", "VH_dB",    "escala_fixa", "c_vh_s"),
        ("s1_seca", "RVI",      "escala_fixa", "c_rvi_s"),
        ("s1_seca", "sd_VV_dB", "escala_fixa", "c_sdvv_s"),
        ("s1_seca", "sd_VH_dB", "escala_fixa", "c_sdvh_s"),
        ("s1_seca", "angulo",   "escala_fixa", "c_ang_s"),
        ("s1_seca", "n_obs",    "escala_fixa", "c_nobs_s"),
    ]),
    # Branch `s2_amplitude`: what Sentinel-2 adds beyond the median composites.
    #
    # The four reflectances and two median indices do NOT enter here; they enter through
    # the `optico` branch as composites, as before. What is new are the percentiles and
    # the observation count.
    #
    # Amplitude, not level, carries the information. In dense forest NDVI saturates near
    # 0.85 all year and does not discriminate within the Amazon, while senescing Cerrado
    # grass has a large amplitude. Measured on the product: Amazon p10-p90 = 0.535-0.855,
    # Cerrado 0.285-0.696, i.e. 28% more amplitude in a biome with a LOWER median.
    # Amplitude separates regimes, and the regime determines where the phase centre sits.
    #
    # `n_obs` is a per-pixel uncertainty. It matters little for the (robust) median and a
    # lot for the percentiles: with 29 observations p10 is about the 3rd smallest value,
    # with 154 about the 15th.
    ("s2_amplitude", [
        ("s2", "ndvi_p10", "escala_fixa", "o_ndvi_p10"),
        ("s2", "ndvi_p90", "escala_fixa", "o_ndvi_p90"),
        ("s2", "ndwi_p10", "escala_fixa", "o_ndwi_p10"),
        ("s2", "ndwi_p90", "escala_fixa", "o_ndwi_p90"),
        ("s2", "n_obs",    "escala_fixa", "o_nobs"),
    ]),
]
# No derived column in the vector. The only one, `dem_desvio` (standard deviation across
# DEM differences), was removed with the `dem_ind` branch. `None` rather than an empty
# list so that the frozen specification records its absence instead of omitting it.
DERIVADA = None

# Branches carried over from the graph and Sentinel-1, in column order.
RAMOS_BASE = [
    ("topo",   (0, 8),   ["elevation", "slope", "aspect_cos", "aspect_sin",
                          # `amplitude3x3` was formerly named `tri`. The graph computes
                          # `max3x3 - min3x3` (generate_graph_gpu.py:197-199), i.e. local
                          # range, NOT Riley's TRI (squared differences to the eight
                          # neighbours). The quantity is kept; only the name changed, so
                          # that tables do not label it as a different measure.
                          "curvature", "tpi", "amplitude3x3", "roughness"]),
    ("optico", (8, 13),  ["ndvi", "ndwi", "bsi", "bright", "shadow"]),
    ("sar_c",  (13, 18), ["vv", "vh", "rvi", "sar_diff", "sar_sum"]),
]
NOMES_REGIME = ["agua_regime", "solo_exposto", "dossel_denso", "dossel_esparso"]


def corrigir_topo(X: np.ndarray, bioma: str, quad: str, elev_std: float) -> tuple:
    """Recompute slope, aspect and curvature with the correct latitude-dependent spacing.

    Returns `(X_corrected, diagnostics)`. Works on a copy; `X` is not modified.

    What changes, and why only this:
      x[1] slope, x[2] aspect_cos, x[3] aspect_sin, x[4] curvature
          depend on `dz/dx`, which was underestimated by `cos(latitude)`.
      x[4] is also rescaled: it was raw 1/m, incompatible with the other columns, which
          are all dimensionless or divided by `elev_std`.
      x[0] elevation, x[5] TPI, x[6] amplitude3x3, x[7] roughness
          come from 3x3 windows on elevation without spacing; the defect does not affect
          them, and recomputing them would only add rounding differences.

    Deliberately excluded: `flow_accum` is a target (`y[8]`) and is not in the V23 vector,
    which has a single output; `x[16]` was allocated but never written in the graph. Both
    are defects of the graph artefact, not of this code path.
    """
    lon_min, lat_min, lon_max, lat_max = BBOX[bioma]
    n = QUAD_N * QUAD_N
    if X.shape[0] != n:
        raise ValueError(f"{bioma}/{quad}: {X.shape[0]} nos, esperado {n} — a correcao "
                         f"de geometria assume a grade 3600x3600 em ordem de raster")

    # Latitude of each ROW of the quadrant. Raster row 0 is the northernmost, and Q1/Q2
    # occupy the northern half of the biome. `cos` varies within the quadrant, so the factor
    # is per row, not a single value (up to 0.5% over 1 degree in the Atlantic Forest).
    passo = (lat_max - lat_min) / GRID
    linha0 = 0 if quad in ("Q1", "Q2") else QUAD_N
    lat = lat_max - (linha0 + np.arange(QUAD_N) + 0.5) * passo
    cos_lat = np.cos(np.radians(lat)).astype(np.float64)

    py = passo * GRAU_M                      # north-south: constant
    px = py * cos_lat                        # east-west: shrinks with latitude

    z = (X[:, 0].astype(np.float64) * elev_std).reshape(QUAD_N, QUAD_N)

    # `np.gradient` takes a scalar spacing per axis; the east-west spacing varies by row,
    # so the division is done afterwards, row by row.
    gy = np.gradient(z, py, axis=0)
    gx = np.gradient(z, axis=1) / px[:, None]

    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    aspect = np.remainder(np.arctan2(-gx, gy) + 2 * np.pi, 2 * np.pi)
    gxx = np.gradient(gx, axis=1) / px[:, None]
    gyy = np.gradient(gy, py, axis=0)
    curv = gxx + gyy

    Y = X.copy()
    antes_slope = float(np.nanmedian(X[:, 1])) * SLOPE_MAX
    Y[:, 1] = (slope / SLOPE_MAX).reshape(-1).astype(X.dtype)
    Y[:, 2] = np.cos(aspect).reshape(-1).astype(X.dtype)
    Y[:, 3] = np.sin(aspect).reshape(-1).astype(X.dtype)
    # Curvature is scaled by a PHYSICAL scale, not by data quantiles. Dividing by the
    # quadrant's own curvature IQR would give each quadrant a different scale, which breaks
    # the Q1 -> Q2 -> Q3 weight transfer in `b2_transferencia.py`.
    #
    # `curv` is in 1/m. Multiplying by `px*py` gives metres (the elevation change implied by
    # the curvature over one cell); dividing by `elev_std` makes it dimensionless, consistent
    # with `tpi` and `roughness`, which the graph normalises the same way. None of the three
    # factors depends on the data distribution.
    area = py * float(px.mean())
    Y[:, 4] = (curv * area / max(elev_std, 1e-9)).reshape(-1).astype(X.dtype)

    diag = {
        "bioma": bioma, "quad": quad,
        "lat_min": round(float(lat.min()), 3), "lat_max": round(float(lat.max()), 3),
        "cos_lat_medio": round(float(cos_lat.mean()), 5),
        "subestimacao_dz_dx_pct": round(100 * (1 - float(cos_lat.mean())), 3),
        "slope_mediana_antes_graus": round(antes_slope, 4),
        "slope_mediana_depois_graus": round(float(np.nanmedian(slope)), 4),
        "curvatura_escala_m2_por_elev_std": round(area / max(elev_std, 1e-9), 8),
        "verificado_em": "verificar_correcoes.py — grade reproduz `roughness` com erro "
                         "mediano de 0,003% a 0,010%; `elev_std` reproduz a declividade "
                         "antiga com erro de 2e-05%",
    }
    return Y, diag


def _ler_lateral(chave: str, bioma: str, quad: str) -> dict:
    arq = LATDIR / f"{chave}_{bioma}_{quad}.npz"
    if not arq.exists():
        raise FileNotFoundError(
            f"lateral ausente: {arq.name}. O vetor V23 declara o bloco `{chave}`; sem "
            f"ele o treino usaria zeros e a ablacao mediria outra coisa. Rode a tarefa "
            f"de integracao correspondente.")
    z = np.load(arq)
    return {k: z[k] for k in z.files}


# No-data sentinel of the JAXA annual mosaic: 146 to 559 nodes per Amazon quadrant hold
# exactly 9999. It is not backscatter: useful PALSAR-2 DN values are in the low thousands
# and the measured p99 does not exceed ~7,000. Unmasked, 9999 becomes
# 10*log10(9999^2) - 83 = -3.0 dB, a plausible-looking value that would also bias the
# frozen scale.
JAXA_SENTINELA = 9999.0


def _para_db(v: np.ndarray) -> np.ndarray:
    """JAXA DN -> gamma0 in dB. DN <= 0 means no data, not zero backscatter."""
    d = np.asarray(v, dtype=np.float64)
    d = np.where(d == JAXA_SENTINELA, np.nan, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(np.maximum(d, 1.0) ** 2) + CF_JAXA_DB
    return np.where(np.isfinite(d) & (d > 0), db, np.nan)


def _norm_agua(v: np.ndarray, teto: float) -> np.ndarray:
    """JRC -128 means `no water observed` -> 0; the value is then scaled to a fraction."""
    d = np.asarray(v, dtype=np.float32)
    d = np.where(d < 0, 0.0, d)
    return np.clip(d / teto, 0.0, 1.0).astype(np.float32)


_ESCALAS: dict | None = None


def escalas() -> dict:
    """Frozen scales from `escalas_v23.json`, loaded once.

    Fails loudly if the file is missing. There is deliberately no fallback to per-quadrant
    median/IQR: that produced a vector that trained without complaint but transferred
    wrongly between quadrants. Run `escalas_v23.py` to create the file.
    """
    global _ESCALAS
    if _ESCALAS is None:
        p = Path(__file__).resolve().parent / "escalas_v23.json"
        if not p.exists():
            raise FileNotFoundError(
                "escalas_v23.json ausente. Ele congela mediana e IQR sobre os 16 "
                "quadrantes juntos, que e o que permite transferir pesos entre "
                "quadrantes e treinar um modelo unico para os quatro biomas. "
                "Rode `python escalas_v23.py`.")
        _ESCALAS = json.loads(p.read_text(encoding="utf-8"))["colunas"]
    return _ESCALAS


def _escala_elev() -> dict:
    """Median and standard deviation of elevation over the sixteen quadrants.

    Used to scale five of the eight topographic columns. A per-quadrant scale would map a
    5 m TPI to 0.1 in a flat quadrant and to 0.02 in a rugged one.
    """
    p = Path(__file__).resolve().parent / "escalas_v23.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    if "elevacao" not in d:
        raise KeyError("escalas_v23.json sem a entrada `elevacao` — versao antiga do "
                       "arquivo. Rode `python escalas_v23.py` de novo.")
    return d["elevacao"]


def _norm_fixa(v: np.ndarray, apelido: str) -> np.ndarray:
    """(v - median) / IQR using dataset-wide statistics, not per-quadrant ones.

    See the header of `escalas_v23.py`. Per-quadrant median/IQR gave each of the sixteen
    quadrants its own scale, so the same column meant different things in each (for
    `glo30_norm` it erased absolute elevation).
    """
    e = escalas().get(apelido)
    if e is None:
        raise KeyError(f"`{apelido}` nao tem escala congelada em escalas_v23.json — "
                       f"coluna nova? rode `python escalas_v23.py` de novo")
    return np.nan_to_num((v - e["mediana"]) / e["iqr"],
                         nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def chaves_de_lateral() -> list:
    vistas = []
    for _, cols in RAMOS_NOVOS:
        for chave, *_ in cols:
            if chave not in vistas:
                vistas.append(chave)
    return vistas


def montar_ramos(bioma: str, quad: str, n_nodes: int, elev_std: float) -> tuple:
    """Return ({branch: matrix}, {branch: names}) for the NEW branches."""
    dados = {c: _ler_lateral(c, bioma, quad) for c in chaves_de_lateral()}

    def coluna(chave, banda):
        d = dados[chave]
        if banda in d:
            return d[banda].astype(np.float32)
        if len(d) == 1:                       # GeoTIFF without band description
            return next(iter(d.values())).astype(np.float32)
        raise KeyError(f"{chave}_{bioma}_{quad}.npz nao tem a banda `{banda}`; "
                       f"tem {sorted(d)}")

    # GLO-30 is loaded only if some column requests `dif_glo30`; none does since the
    # `dem_ind` branch was removed, which saves loading 12.96 million values per quadrant.
    # Elevation itself is read in `carregar_v23`, which builds the topographic block.
    glo30 = (coluna("glo30", "DEM")
             if any(m == "dif_glo30" for _, cols in RAMOS_NOVOS for *_, m, _ in
                    [(c, b, m, a) for c, b, m, a in cols]) else None)
    mats, nomes, difs = {}, {}, []
    for ramo, cols in RAMOS_NOVOS:
        pilha, rot = [], []
        for chave, banda, modo, apelido in cols:
            v = coluna(chave, banda)
            if v.shape[0] != n_nodes:
                raise ValueError(f"{chave}_{bioma}_{quad}: {v.shape[0]} valores para "
                                 f"{n_nodes} nos — grade incompativel")
            if modo == "db_jaxa":
                col = _norm_fixa(_para_db(v), apelido)
            elif modo == "agua_jrc":
                col = _norm_agua(v, 100.0 if "ocorr" in apelido else 12.0)
            elif modo == "dif_glo30":
                # Already dimensionless: DEM difference divided by `elev_std`, the same
                # scale as the topographic block, so no frozen scale is needed.
                col = np.nan_to_num((v - glo30) / max(elev_std, 1e-6),
                                    nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                difs.append(col)
            elif modo in ("robusta", "escala_fixa"):
                col = _norm_fixa(v, apelido)
            else:
                col = np.nan_to_num(v, nan=0.0, posinf=0.0,
                                    neginf=0.0).astype(np.float32)
            pilha.append(col)
            rot.append(apelido)
        mats[ramo] = np.stack(pilha, axis=1)
        nomes[ramo] = rot

    if DERIVADA is not None and difs:
        ramo_d, nome_d = DERIVADA
        d = np.std(np.stack(difs, axis=1), axis=1).astype(np.float32)[:, None]
        mats[ramo_d] = np.concatenate([mats[ramo_d], d], axis=1)
        nomes[ramo_d] = nomes[ramo_d] + [nome_d]
    return mats, nomes


def optico_e_regime(bioma: str, quad: str, n_nodes: int) -> tuple:
    """Five optical composites and four regime memberships, built from lateral files.

    Reflectances previously lived in graph columns 8-13 (aggregated 3x3 from 10 m to
    30 m). The composite formulas are unchanged: `regime.composicoes` is called with the
    array layout it expects, so any difference from the baseline is due to the data
    source only, not to the formulas.

    The SAR input to the regime is the MEAN OF BOTH SEASONS in dB. The regime classifies
    surface type (water, bare soil, dense canopy, sparse canopy), a property of the place,
    not of the season; a single season would let seasonally flooded areas switch class
    and open and close the model's gate by calendar.
    """
    import regime as R

    s2 = _ler_lateral("s2", bioma, quad)
    faltando = [b for b in ("B2", "B3", "B4", "B8", "ndvi", "ndwi") if b not in s2]
    if faltando:
        raise KeyError(f"s2_{bioma}_{quad}.npz nao tem {faltando}; tem {sorted(s2)}")

    # `regime.composicoes` reads fixed indices (COL_B02=8 ... COL_NDWI=13); building the
    # array with that convention is safer than re-implementing the formulas here.
    x = np.zeros((n_nodes, 14), dtype=np.float32)
    for i, b in enumerate(("B2", "B3", "B4", "B8", "ndvi", "ndwi"), start=8):
        v = s2[b].astype(np.float32)
        if v.shape[0] != n_nodes:
            raise ValueError(f"s2_{bioma}_{quad}: {v.shape[0]} valores para {n_nodes} "
                             f"nos — grade incompativel")
        x[:, i] = v
    del s2

    vv = np.mean([_ler_lateral(f"s1_{e}", bioma, quad)["VV_dB"] for e in ("umida", "seca")],
                 axis=0).astype(np.float32)
    vh = np.mean([_ler_lateral(f"s1_{e}", bioma, quad)["VH_dB"] for e in ("umida", "seca")],
                 axis=0).astype(np.float32)

    comp = R.composicoes(x, vv, vh)
    del x, vv, vh
    optico = np.stack([comp[k] for k in ("ndvi", "ndwi", "bsi", "bright", "shadow")],
                      axis=1).astype(np.float32)
    reg = R.regime(comp, T=1.0).astype(np.float32)
    diag = R.diagnostico(reg, comp)
    return optico, reg, diag


def carregar_v23(bioma: str, quad: str, usar_sar: bool = True,
                 bloco_por_bioma: bool = True) -> dict:
    """Build a full quadrant from lateral files, without `.pt` graph files.

    Components:
        topology    `grade_v23.arestas`: the grid is regular, the eight neighbours of a
                    cell are the surrounding cells, and distances use the correct
                    geometry (30.922 m and `cos(latitude)`), not a fixed 30.0 m
        topography  `topo_v23.montar`, directly from the `glo30` lateral file
        labels      lateral file `rotulos_*`

    Scales: elevation, dB values and counts are divided by statistics frozen in
    `escalas_v23.json`, computed once over the sixteen quadrants. Per-quadrant
    normalisation would erase between-biome differences and prevent both the chained
    transfer of B2 and a single model across the four biomes.

    Returns the loader dictionary, plus:
        cols        {branch: (start, end)}; each branch is a CONTIGUOUS block
        nomes_col   name of each column, in order
        qualidade   per anchor: weight in [0, 1]; 0 = failed the hard criterion
    """
    import grade_v23 as GR
    import topo_v23 as TP
    # Block size comes from the FROZEN B1 file, not a local copy. The baseline comparison
    # changes features and target only: geography, seed and block stay fixed. Two block
    # definitions would give two partitions and break the per-quadrant paired comparison.
    from b1_kriging_indutivo import BLOCO, BLOCO_POR_BIOMA

    t0 = time.time()
    n_nodes = QUAD_N * QUAD_N
    esc = escalas()
    el = _escala_elev()

    z = _ler_lateral("glo30", bioma, quad)["DEM"].astype(np.float32)
    if z.shape[0] != n_nodes:
        raise ValueError(f"glo30_{bioma}_{quad}: {z.shape[0]} valores para {n_nodes} nos")
    elev_std = float(el["elev_std"])

    topo, nomes_topo = TP.montar(bioma, quad, z, elev_std, float(el["mediana"]))
    optico, reg, diag = optico_e_regime(bioma, quad, n_nodes)
    novos, nomes_novos = montar_ramos(bioma, quad, n_nodes, elev_std)

    # Order: topo, optico, the new branches in declared order, and the regime last.
    # Each branch is contiguous because the model gives each one its own encoder.
    partes, nomes, cols, k = [], [], {}, 0
    for ramo, M, rot in [("topo", topo, nomes_topo),
                         ("optico", optico, ["ndvi", "ndwi", "bsi", "bright", "shadow"])]:
        partes.append(M); nomes += rot
        cols[ramo] = (k, k + M.shape[1]); k += M.shape[1]
    for ramo, _ in RAMOS_NOVOS:
        M = novos[ramo]
        partes.append(M); nomes += nomes_novos[ramo]
        cols[ramo] = (k, k + M.shape[1]); k += M.shape[1]
    partes.append(reg); nomes += NOMES_REGIME
    cols["regime"] = (k, k + 4)

    X = np.nan_to_num(np.concatenate(partes, axis=1).astype(np.float32),
                      nan=0.0, posinf=0.0, neginf=0.0)
    del partes, topo, optico, novos
    if X.shape[1] != cols["regime"][1] or len(nomes) != X.shape[1]:
        raise ValueError(f"layout inconsistente: X tem {X.shape[1]} colunas, "
                         f"{len(nomes)} nomes, regime termina em {cols['regime'][1]}")

    # -- anchors and target, from the label lateral file --
    r = np.load(LATDIR / f"rotulos_{bioma}_{quad}.npz")
    idx = r["idx_no"].astype(np.int64)
    y = r["y_novo"].astype(np.float32)
    ordem = np.argsort(idx, kind="stable")      # natural node order, as in the graph
    idx, y = idx[ordem], y[ordem]
    r.close()

    row, col = idx // QUAD_N, idx % QUAD_N
    blc = BLOCO_POR_BIOMA.get(bioma, BLOCO) if bloco_por_bioma else BLOCO
    bloco = ((row // blc) * (QUAD_N // blc + 1) + (col // blc)).astype(np.int32)
    pos_anc = np.stack([row, col], axis=1).astype(np.float32)
    pos_anc_m = GR.posicao_metros(bioma, quad, idx)

    edge_index, edge_attr = GR.arestas(bioma, quad, z.astype(np.float64))
    del z

    ctx_anc = {
        "slope_norm": X[idx, cols["topo"][0] + 1],
        "tpi": X[idx, cols["topo"][0] + 5],
        "ndvi": X[idx, cols["optico"][0]],
        "ndwi": X[idx, cols["optico"][0] + 1],
        "elev_norm": X[idx, cols["topo"][0]],
        "slope_deg": X[idx, cols["topo"][0] + 1] * 90.0,
        "pos_m": pos_anc_m,
        "elev_std": elev_std,
        "regime": reg[idx],
    }

    dd = {"X": X, "edge_index": edge_index, "edge_attr": edge_attr,
          "idx": idx, "y": y, "bloco": bloco,
          "pos_anc": pos_anc, "pos_anc_m": pos_anc_m, "ctx_anc": ctx_anc,
          "n_nodes": n_nodes, "elev_std": elev_std, "diag_regime": diag,
          "feat_nomes": ["v23"], "cols": cols, "nomes_col": nomes,
          "escalas_parciais": [c for c in nomes if "PROVISORIA_biomas" in esc.get(c, {})],
          "segundos": round(time.time() - t0, 1)}
    dd["qualidade"], dd["diag_rotulo"] = _rotulo_novo(dd, bioma, quad)
    dd["decl_trilha"] = _restricao_declividade(bioma, quad)
    return dd


def _restricao_declividade(bioma: str, quad: str) -> dict | None:
    """Build the directional-derivative constraint from along-track slope.

    The measured slope is not the maximum hillside slope but its component along the
    track, over a ~114 m baseline; as a scalar target it would teach the model to
    underestimate. With the direction (stored as `decl_sin`/`decl_cos`, since azimuths
    are circular and cannot be averaged directly) it becomes an exact constraint:

        z'(p+) - z'(p-)  =  tan(slope) * separation

    where `p+` and `p-` are the nodes half a baseline ahead of and behind the point along
    the direction, `p+` on the side the azimuth points to. The constraint holds AT THE
    SCALE OF MEASUREMENT (~114 m), not on the 30 m edges: the local gradient is a
    different quantity.

    The slope is SIGNED (the extraction does not take `|z_c - z_a|`), so the constraint
    distinguishes a slope from its mirror image: 44.3% of slopes are negative in the
    Cerrado and 48.7% in the Atlantic Forest.

    Convention: `azimuth = atan2(delta_east, delta_north)` at extraction, so
    `u = (sin az, cos az)` and a positive slope means going UP along `u`, i.e. from `p-`
    to `p+`.

    Returns node indices already resolved, so the loss does not redo geometry per batch.
    """
    arq = LATDIR / f"rotulos_{bioma}_{quad}.npz"
    if not arq.exists():
        return None
    z = np.load(arq)
    if "decl_sin" not in z.files:
        z.close()
        return None

    bom = z["bom"] > 0.5
    s = z["declividade_graus"][bom].astype(np.float64)
    sin_a = z["decl_sin"][bom].astype(np.float64)
    cos_a = z["decl_cos"][bom].astype(np.float64)
    base = z["decl_base_m"][bom].astype(np.float64)
    # `decl_conf` is the confidence from the angular error propagated from roughness over
    # the baseline. A circular-coherence weight would be 1.000 in 99.55% of cells (97.8%
    # have a single shot, and the circular mean of one direction is 1 by construction),
    # so it would weight nothing. See `rotulos_lidar_v23`.
    conf = (z["decl_conf"][bom].astype(np.float64) if "decl_conf" in z.files
            else np.ones(int(bom.sum())))
    idx = z["idx_no"][bom].astype(np.int64)
    z.close()

    # `s` is signed and can be negative; filtering by `>= 0` would discard about half of
    # the measurements.
    ok = (np.isfinite(s) & np.isfinite(sin_a) & np.isfinite(cos_a)
          & np.isfinite(base) & (base > 0))
    if not ok.any():
        return None
    s, sin_a, cos_a, base, conf, idx = (v[ok] for v in
                                       (s, sin_a, cos_a, base, conf, idx))

    lon_min, lat_min, lon_max, lat_max = BBOX[bioma]
    passo = (lat_max - lat_min) / GRID
    lin = idx // QUAD_N
    col = idx % QUAD_N
    lin0 = 0 if quad in ("Q1", "Q2") else QUAD_N
    col0 = 0 if quad in ("Q1", "Q3") else QUAD_N
    lat = lat_max - (lin0 + lin + 0.5) * passo
    m_por_celula_y = passo * GRAU_M
    m_por_celula_x = m_por_celula_y * np.cos(np.radians(lat))

    # Half baseline along the direction. Row index grows SOUTHWARD, so the north component
    # (cos of the azimuth) enters with a flipped sign.
    dx = (base / 2.0) * sin_a / m_por_celula_x
    dy = -(base / 2.0) * cos_a / m_por_celula_y

    def _no(sinal: int):
        l = np.rint(lin + sinal * dy).astype(np.int64)
        c = np.rint(col + sinal * dx).astype(np.int64)
        dentro = (l >= 0) & (l < QUAD_N) & (c >= 0) & (c < QUAD_N)
        return l * QUAD_N + c, dentro

    i_mais, d1 = _no(+1)
    i_menos, d2 = _no(-1)
    # A degenerate pair constrains nothing: both nodes fall in the same cell.
    val = d1 & d2 & (i_mais != i_menos)
    if not val.any():
        return None

    # The denominator is the ACTUAL separation, not the nominal baseline. Neighbour nodes
    # are chosen by rounding to whole cells, which shortens the span: in Cerrado Q1 the
    # median separation was 2.8 cells (~85 m) against a nominal baseline of 114.1 m, so
    # using the baseline would misstate the slope by ~25% relative to the sampled
    # geometry. The distance between the two chosen nodes is exact, so it is used.
    lm, cm = i_mais[val] // QUAD_N, i_mais[val] % QUAD_N
    ln, cn = i_menos[val] // QUAD_N, i_menos[val] % QUAD_N
    dxm = (cm - cn) * m_por_celula_x[val]
    dym = (lm - ln) * m_por_celula_y
    sep = np.hypot(dxm, dym)

    # `dz_alvo` is the elevation difference in METRES the corrected surface must have
    # between the two nodes (metres, not degrees, so the loss avoids per-batch trigonometry).
    dz_alvo = np.tan(np.radians(s[val])) * sep

    # This is what the loss asks of the model DIRECTLY. Since z' = DSM - Delta:
    #
    #   z'(p+) - z'(p-)                        = dz_alvo
    #   (DSM+ - Delta+) - (DSM- - Delta-)      = dz_alvo
    #   Delta+ - Delta-                        = (DSM+ - DSM-) - dz_alvo
    #
    # The right-hand side does not depend on the model and is precomputed once. The loss
    # is `| (p[i+] - p[i-]) - alvo_dp |`, with no DSM or trigonometry in the loop.
    glo = np.load(LATDIR / f"glo30_{bioma}_{quad}.npz")["DEM"].astype(np.float64)
    dsm_dif = glo[i_mais[val]] - glo[i_menos[val]]
    alvo_dp = dsm_dif - dz_alvo

    return {"i_mais": i_mais[val], "i_menos": i_menos[val],
            "alvo_dp_m": alvo_dp.astype(np.float32),
            "dsm_dif_m": dsm_dif.astype(np.float32),
            "dz_alvo_m": dz_alvo.astype(np.float32),
            "s_graus": s[val].astype(np.float32),
            "sep_m": sep.astype(np.float32),
            "base_nominal_m": base[val].astype(np.float32),
            "confianca": conf[val].astype(np.float32),
            "n": int(val.sum()), "n_descartado_por_geometria": int((~val).sum()),
            "sep_mediana_m": round(float(np.median(sep)), 1),
            "base_nominal_mediana_m": round(float(np.median(base[val])), 1),
            "frac_negativa": round(float((s[val] < 0).mean()), 4)}


def _rotulo_novo(dd: dict, bioma: str, quad: str) -> tuple:
    """Replace `y` with the filtered label and return the per-anchor quality weight.

    Where there is no new label the old one is kept, so NO anchor disappears and the
    partition stays identical to the baseline's. Only training membership changes.
    """
    arq = LATDIR / f"rotulos_{bioma}_{quad}.npz"
    if not arq.exists():
        raise FileNotFoundError(
            f"{arq.name} ausente — rode `rotulos_lidar`. Treinar sem ele seria repetir "
            f"o baseline com features novas, que nao e o que o V23 se propos.")
    z = np.load(arq)
    idx_r = z["idx_no"].astype(np.int64)

    n = dd["n_nodes"]
    y_map = np.full(n, np.nan, dtype=np.float32)
    p_map = np.zeros(n, dtype=np.float32)
    y_map[idx_r] = z["y_novo"]
    p_map[idx_r] = np.where(z["bom"] > 0.5, np.maximum(z["peso"], 1e-3), 0.0)

    idx = dd["idx"]
    y_nova = y_map[idx]
    tem = np.isfinite(y_nova)
    y_ant = dd["y"].copy()
    dd["y"] = np.where(tem, y_nova, y_ant).astype(np.float32)
    qual = p_map[idx].astype(np.float32)

    diag = {
        "n_ancoras": int(len(idx)),
        "n_com_rotulo_novo": int(tem.sum()),
        "frac_com_rotulo_novo": float(tem.mean()),
        "n_aprovadas": int((qual > 0).sum()),
        "frac_aprovadas": float((qual > 0).mean()),
        "delta_mediano_rotulo_m": float(np.nanmedian(np.abs(y_nova[tem] - y_ant[tem])))
        if tem.any() else None,
        "nota": "onde nao ha rotulo novo o antigo permanece; nenhuma ancora sai do "
                "conjunto, so do treino",
    }
    return qual, diag


def aplicar_qualidade(lab: np.ndarray, qual: np.ndarray) -> tuple:
    """Remove failed anchors from EVERY split: train, validation, test and hold-out.

    An anchor that fails the quality criterion is a point where the LiDAR did not find the
    ground: its value is not a noisy target but a WRONG one, and scoring the model against
    it says nothing about the model. Removing it only from train/validation would keep the
    evaluation cells identical to the baseline at the cost of measurement validity.

    Pairing is preserved: the partition is computed BEFORE filtering, on the full set and
    with the same seed, so geography, block draw and buffer match the baseline. The
    evaluated population becomes a SUBSET of the baseline's, and `comparacao_v23.py`
    re-scores the baseline on exactly those nodes from its stored per-node predictions
    (a recount, not a retraining).
    """
    ruim = qual <= 0.0
    antes = {r: int((lab == r).sum()) for r in (0, 1, 2, 3)}
    lab = lab.copy()
    lab[ruim & (lab >= 0)] = -1
    depois = {r: int((lab == r).sum()) for r in (0, 1, 2, 3)}

    if depois[0] < 1000:
        raise RuntimeError(
            f"filtro de qualidade deixou {depois[0]} ancoras de treino — abaixo do "
            f"minimo utilizavel. Revise o criterio em rotulos_lidar_v23.py antes de "
            f"aceitar este numero.")
    if depois[3] < 200:
        raise RuntimeError(
            f"filtro de qualidade deixou {depois[3]} ancoras na reserva. Avaliar "
            f"generalizacao com menos que isso nao sustenta intervalo de confianca.")
    return lab, {
        "removidas": {"treino": antes[0] - depois[0], "validacao": antes[1] - depois[1],
                      "teste": antes[2] - depois[2], "reserva": antes[3] - depois[3]},
        "restam": {"treino": depois[0], "validacao": depois[1],
                   "teste": depois[2], "reserva": depois[3]},
        "frac_reserva_mantida": depois[3] / max(antes[3], 1),
        "aplicado_a": "todos os conjuntos, inclusive teste e reserva",
    }


# ------------------------------------------------------------------ freezing
def inspecionar() -> dict:
    """Check the lateral files on disk and return the effective specification."""
    esp = {"ramos": [r for r, *_ in RAMOS_BASE] + [r for r, _ in RAMOS_NOVOS]
           + ["regime"], "blocos": [], "quadrantes": {}, "faltando": []}
    for ramo, _, rot in RAMOS_BASE:
        esp["blocos"].append({"ramo": ramo, "origem": "grafo v18.5 + Sentinel-1",
                              "nomes": rot})
    for ramo, cols in RAMOS_NOVOS:
        esp["blocos"].append({"ramo": ramo,
                              "laterais": sorted({c for c, *_ in cols}),
                              "normalizacao": sorted({m for _, _, m, _ in cols}),
                              "nomes": [a for *_, a in cols]})
    esp["derivadas"] = ([] if DERIVADA is None else
                        [{"nome": DERIVADA[1], "ramo": DERIVADA[0]}])
    for b in BIOMAS:
        for q in QUADS:
            faltam = [c for c in chaves_de_lateral()
                      if not (LATDIR / f"{c}_{b}_{q}.npz").exists()]
            if not (LATDIR / f"rotulos_{b}_{q}.npz").exists():
                faltam.append("rotulos")
            esp["quadrantes"][f"{b}_{q}"] = {"faltando": faltam, "pronto": not faltam}
            # Store (block, quadrant) separately: rebuilding the block name from the file
            # name fails both ways (`agua_jrc` would become `agua`, and `mata_atlantica`
            # itself contains an underscore).
            esp["faltando"] += [{"bloco": c, "quadrante": f"{b}_{q}"} for c in faltam]
    esp["rotulo"] = _proveniencia_rotulo()
    esp["pronto"] = not esp["faltando"] and esp["rotulo"]["completo"]
    return esp


def _proveniencia_rotulo() -> dict:
    """Which sensors the label was built from, and whether one present on disk is unused.

    Guards against freezing a label built from ATL08 alone while GEDI data exist: GEDI
    provides most of the shots and the only vertical profile. The check asks "is there
    GEDI data on disk that the label did not use?", which does not rely on anyone
    remembering to reprocess.
    """
    import numpy as np
    from fila_aquisicao_alvos import SAIDA

    fontes, faltam_arq = set(), []
    for b in BIOMAS:
        for q in QUADS:
            arq = LATDIR / f"rotulos_{b}_{q}.npz"
            if not arq.exists():
                faltam_arq.append(f"{b}_{q}")
                continue
            f = np.load(arq)["fonte"].astype(np.int64)
            if (f & 1).any():
                fontes.add("GEDI")
            if (f & 2).any():
                fontes.add("ATL08")

    no_disco = set()
    if any((SAIDA / f"gedi02_a_{b}.parquet").exists()
           or (SAIDA / f"gedi_{b}.parquet").exists() for b in BIOMAS):
        no_disco.add("GEDI")
    if any((SAIDA / f"atl08_{b}.parquet").exists() for b in BIOMAS):
        no_disco.add("ATL08")

    ignoradas = sorted(no_disco - fontes)
    return {"fontes_usadas": sorted(fontes), "fontes_no_disco": sorted(no_disco),
            "ignoradas": ignoradas, "quadrantes_sem_rotulo": faltam_arq,
            "completo": not ignoradas and not faltam_arq,
            "nota": (f"ha {', '.join(ignoradas)} no disco que o rotulo nao usou — "
                     f"rode `rotulos_lidar` de novo") if ignoradas else "ok"}


def dimensao_declarada() -> int:
    """Width of X exactly as `carregar_v23` builds it: topo + optico +
    RAMOS_NOVOS + regime.

    `sar_c` is NOT counted: it is declared in RAMOS_BASE for compatibility, but
    `carregar_v23` never stacks it. `dem_desvio` is NOT counted (`DERIVADA = None`).
    The two observation channels enter in B2, outside this vector: model d_in = this
    value + 2.
    """
    k = sum(b - a for ramo, (a, b), _ in RAMOS_BASE if ramo != "sar_c")
    for _, cols in RAMOS_NOVOS:
        k += len(cols)
    return k + 4                                           # + regime


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--congelar", action="store_true")
    ap.add_argument("--conferir", action="store_true")
    a = ap.parse_args()

    esp = inspecionar()
    esp["d_base_declarado"] = dimensao_declarada()
    esp["d_in_com_obs"] = esp["d_base_declarado"] + 2
    prontos = sum(1 for v in esp["quadrantes"].values() if v["pronto"])
    print(f"\n  vetor V23: {esp['d_base_declarado']} colunas base "
          f"(+2 de observacao = {esp['d_in_com_obs']})")
    print(f"  quadrantes prontos: {prontos}/16")
    if esp["faltando"]:
        from collections import Counter
        c = Counter(x["bloco"] for x in esp["faltando"])
        print("  blocos incompletos: " +
              ", ".join(f"{k} ({v}/16)" for k, v in sorted(c.items())))

    rot = esp["rotulo"]
    print(f"  rotulo: fontes {rot['fontes_usadas'] or 'nenhuma'} | {rot['nota']}")

    if a.congelar:
        if not esp["pronto"]:
            if esp["faltando"]:
                print("\n  NAO congelado: ha lateral faltando. Congelar um vetor "
                      "incompleto e assinar um contrato que o disco nao cumpre.\n")
            else:
                print(f"\n  NAO congelado: {rot['nota']}\n")
            return 1
        ESPEC.write_text(json.dumps(esp, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  congelado em {ESPEC.name}\n")
        return 0

    if a.conferir:
        if not ESPEC.exists():
            print("\n  sem especificacao congelada; rode --congelar\n")
            return 1
        antes = json.loads(ESPEC.read_text(encoding="utf-8"))
        if antes.get("d_base_declarado") != esp["d_base_declarado"]:
            print(f"\n  DIVERGENCIA: congelado com {antes.get('d_base_declarado')} "
                  f"colunas, o codigo agora declara {esp['d_base_declarado']}\n")
            return 1
        print("\n  especificacao confere com o disco\n")
        return 0
    print()
    return 0 if esp["pronto"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
