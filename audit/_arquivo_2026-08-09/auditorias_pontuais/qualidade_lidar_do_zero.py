"""Which GEDI variable predicts label error: measured, not inherited.

Method
  The lidar label has no independent reference in general, except over EXPOSED SOIL:
  there is no canopy between the DSM and the ground, so `DSM - ground` should be zero.
  The median exposed-soil residual is -0.11 m in the Amazon and -0.18 m in the
  Pantanal, against +12.06 m under canopy.

  Restricted to exposed soil, `|DSM - ground|` therefore IS the label error. This gives an
  external yardstick to ask of each quality variable: does it predict error, and from
  which value on?

  The question answered is not "which threshold" but "which variable deserves to
  enter", the same form as the questions posed for the SAR and optical inputs.

Caveats
  Datum: GEDI `elev_solo` is WGS84 ellipsoidal and GLO-30 is EGM2008 orthometric. The
  conversion is per point, via `geoide_v23.ortometrica`; without it the geoid undulation,
  which ranges from -5 to -30 m in Brazil, would enter the "error" in full.

  Exposed soil comes from `land_cover_data/landsat_treecover`, carried by the GEDI granule
  itself: Landsat tree cover, external to GEDI and to the DSM, and independent of every
  quality variable tested here. It does not use the graph or the `regime` layer derived from it.

Usage
  python qualidade_lidar_do_zero.py
  python qualidade_lidar_do_zero.py --bioma cerrado --quad Q1
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

ALVOS = Path(r"D:\GNN_TOPO\SATELITES\alvos")
SAIDA = Path(__file__).resolve().parent / "qualidade_lidar_do_zero.json"
GRID, QUAD_N = 7200, 3600
BBOX = {
    "amazonia": [-60.0, -4.0, -58.0, -2.0],
    "cerrado": [-48.0, -16.0, -46.0, -14.0],
    "mata_atlantica": [-42.0, -20.0, -40.0, -18.0],
    "pantanal": [-58.0, -20.0, -56.0, -18.0],
}
# Landsat tree cover below which a shot counts as exposed soil. 10% is the same cut-off
# FABDEM uses to decide where forest is removed (Hawker et al., 2022).
TREECOVER_MAX = 10.0
LATDIR = Path(r"D:\GNN_TOPO\SATELITES\laterais")
COLS = ["lat", "lon", "elev_solo", "elev_topo", "quality_flag", "degrade_flag",
        "sensitivity", "surface_flag", "num_modos", "solar_elevation",
        "cobertura_landsat"] + [f"solo_a{i}" for i in range(1, 7)]


def _quadrante_de(lat, lon, bbox):
    lon_min, lat_min, lon_max, lat_max = bbox
    linha = np.clip(((lat_max - lat) / (lat_max - lat_min) * GRID).astype(int), 0, GRID - 1)
    col = np.clip(((lon - lon_min) / (lon_max - lon_min) * GRID).astype(int), 0, GRID - 1)
    return linha, col


def carregar(bioma: str, quad: str) -> dict:
    from geoide_v23 import ortometrica

    # DSM taken directly from the Copernicus covariate, already on the project grid and in
    # EGM2008 orthometric height.
    glo = np.load(LATDIR / f"glo30_{bioma}_{quad}.npz")["DEM"].astype(np.float64)

    d = pq.read_table(ALVOS / f"gedi02_a_{bioma}_dedup.parquet", columns=COLS).to_pandas()
    lat = d.lat.to_numpy(np.float64)
    lon = d.lon.to_numpy(np.float64)
    linha, colu = _quadrante_de(lat, lon, BBOX[bioma])
    norte = linha < QUAD_N
    oeste = colu < QUAD_N
    dentro = {"Q1": norte & oeste, "Q2": norte & ~oeste,
              "Q3": ~norte & oeste, "Q4": ~norte & ~oeste}[quad]
    d = d[dentro].reset_index(drop=True)
    no = ((linha[dentro] % QUAD_N) * QUAD_N + (colu[dentro] % QUAD_N)).astype(np.int64)

    solo_no = d.cobertura_landsat.to_numpy(np.float64)
    dsm = glo[no]

    h_orto = ortometrica(d.lon.to_numpy(), d.lat.to_numpy(),
                         d.elev_solo.to_numpy(np.float64))
    resid = dsm - h_orto

    M = d[[f"solo_a{i}" for i in range(1, 7)]].to_numpy(np.float64, copy=True)
    M[M == -9999] = np.nan
    with np.errstate(invalid="ignore"):
        disp = np.nanstd(M, axis=1)

    return {"d": d, "resid": resid, "disp": disp, "solo": solo_no,
            "n_total": int(len(d))}


def perfil(rot: str, v: np.ndarray, err: np.ndarray, bins) -> list:
    """Absolute label error in bins of the quality variable."""
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = np.isfinite(v) & (v >= lo) & (v < hi) & np.isfinite(err)
        n = int(m.sum())
        if n < 200:
            out.append({"faixa": f"[{lo:g}, {hi:g})", "n": n, "insuficiente": True})
            continue
        e = np.abs(err[m])
        out.append({"faixa": f"[{lo:g}, {hi:g})", "n": n,
                    "erro_mediano_m": round(float(np.median(e)), 3),
                    "erro_p90_m": round(float(np.percentile(e, 90)), 2),
                    "vies_mediano_m": round(float(np.median(err[m])), 3)})
    return out


def main() -> int:
    args = sys.argv[1:]
    bioma = args[args.index("--bioma") + 1] if "--bioma" in args else "cerrado"
    quad = args[args.index("--quad") + 1] if "--quad" in args else "Q1"

    print(f"  carregando {bioma}/{quad}", flush=True)
    c = carregar(bioma, quad)
    d, resid, disp, solo = c["d"], c["resid"], c["disp"], c["solo"]

    nu = np.isfinite(solo) & (solo <= TREECOVER_MAX)
    print(f"  {c['n_total']:,} disparos no quadrante | "
          f"{int(nu.sum()):,} sobre solo exposto (cobertura arborea <= {TREECOVER_MAX}%)", flush=True)
    if nu.sum() < 2000:
        print("  amostra de solo exposto insuficiente neste quadrante")
        return 1

    e = resid[nu]
    rel = {"gerado_em": datetime.now().isoformat(timespec="seconds"),
           "script": str(Path(__file__).resolve()),
           "bioma": bioma, "quad": quad,
           "disparos_no_quadrante": c["n_total"],
           "disparos_em_solo_exposto": int(nu.sum()),
           "treecover_max_pct": TREECOVER_MAX,
           "erro_de_referencia": {
               "mediano_abs_m": round(float(np.median(np.abs(e))), 3),
               "vies_mediano_m": round(float(np.median(e)), 3),
               "p90_abs_m": round(float(np.percentile(np.abs(e), 90)), 2)},
           "perfis": {}}

    rel["perfis"]["sensitivity"] = perfil(
        "sensitivity", d.sensitivity.to_numpy(np.float64)[nu], e,
        [0.0, 0.8, 0.85, 0.90, 0.93, 0.95, 0.97, 1.01])
    rel["perfis"]["dispersao_m"] = perfil(
        "dispersao", disp[nu], e, [0, 0.5, 1, 2, 4, 8, 16, 1e9])
    rel["perfis"]["num_modos"] = perfil(
        "num_modos", d.num_modos.to_numpy(np.float64)[nu], e, [0, 1, 2, 3, 5, 100])

    for nome, v in (("quality_flag", d.quality_flag.to_numpy(np.float64)[nu]),
                    ("degrade_flag", d.degrade_flag.to_numpy(np.float64)[nu]),
                    ("surface_flag", d.surface_flag.to_numpy(np.float64)[nu])):
        g = {}
        for val in np.unique(v[np.isfinite(v)])[:6]:
            m = v == val
            if m.sum() < 200:
                continue
            g[str(val)] = {"n": int(m.sum()),
                           "erro_mediano_m": round(float(np.median(np.abs(e[m]))), 3),
                           "erro_p90_m": round(float(np.percentile(np.abs(e[m]), 90)), 2)}
        rel["perfis"][nome] = g

    SAIDA.write_text(json.dumps(rel, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  relatorio -> {SAIDA}\n")
    r = rel["erro_de_referencia"]
    print(f"  ERRO DE ROTULO sobre solo exposto — o zero fisico")
    print(f"    mediana |erro| = {r['mediano_abs_m']} m | vies {r['vies_mediano_m']} m | "
          f"p90 {r['p90_abs_m']} m\n")

    for nome in ("sensitivity", "dispersao_m", "num_modos"):
        print(f"  {nome}")
        for l in rel["perfis"][nome]:
            if l.get("insuficiente"):
                continue
            print(f"    {l['faixa']:14s} n={l['n']:>8,d}  |erro| mediano "
                  f"{l['erro_mediano_m']:6.2f} m   p90 {l['erro_p90_m']:7.2f} m")
        print()
    for nome in ("quality_flag", "degrade_flag", "surface_flag"):
        print(f"  {nome}: " + " | ".join(
            f"{k}: n={v['n']:,} erro {v['erro_mediano_m']:.2f} m"
            for k, v in rel["perfis"][nome].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
