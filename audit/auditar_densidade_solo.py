"""Ground-return density by canopy class across the EBA tiles.

WHAT IT MEASURES. For each cell of the reference parquet: density
(pt/m^2) of three NAMED populations — n_all (all returns), n_cl2 (the
producer's ASPRS class 2, SUPER-INCLUSIVE in this survey), and n_casca
(returns in the lower crust, z <= p2 + 1 m, the population used by the
ground estimator) — over the cell's REAL area at its latitude (geographic
grid of 2/7200 degrees; ~948 m^2, NOT 900). Stratifies by EXOGENOUS canopy
class (ETH, cob_eth >= 0.8; primary) and by dossel_v2 (circular;
sensitivity check). Records medians, quartiles, the fraction of cells
below declared thresholds, and the crust/floor ratio, where
floor = 0.02 * n_all/A is the density that the 2nd percentile imposes on
the crust by construction (n_casca >= 0.02 n_all).

CRITERIA/DEFINITIONS (recorded in the JSON ahead of the numbers):
  area  A(phi) = (PASSO*m_lat(phi)) * (PASSO*m_lon(phi)), WGS84, phi = latitude of the cell center
  declared thresholds: 0.40 and 0.50 pt/m^2 (order of magnitude from a
        temperate-zone study cited in the literature; NOT directly
        transferable; used only as corroboration)
  the manuscript text must NAME the population ("returns within the lowest
        metre of the cloud"), never a bare "ground returns": n_cl2/A and
        n_casca/A differ by ~13x under tall canopy.

Usage: python auditar_densidade_solo.py [--parquet ...] [--saida auditar_densidade_solo.json]

This is v2: thresholds apply only to the density columns, not to the
dimensionless ratio, and the ratio-of-medians field is recorded alongside
the median-of-ratios field. The manuscript text cites the MEDIAN OF THE
RATIO (1.50).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import proveniencia as PROV   # noqa: E402

LAT = Path(r"D:\GNN_TOPO\SATELITES\laterais")
GRID = 7200; PASSO = 2.0 / GRID; LAT1 = -2.0
FAIXAS = [-1, 3, 10, 20, 30, 500]; ROT_F = ["0-3m", "3-10m", "10-20m", "20-30m", ">30m"]
LIMIARES = [0.40, 0.50]
QUADS = ["Q1", "Q2", "Q3", "Q4"]


def log(m=""):
    print(m, flush=True)


def metros_por_grau(phi_deg: np.ndarray):
    """Length of 1 degree of latitude and longitude in WGS84 (standard formulas)."""
    p = np.deg2rad(phi_deg)
    m_lat = 111132.954 - 559.822 * np.cos(2 * p) + 1.175 * np.cos(4 * p)
    m_lon = 111412.84 * np.cos(p) - 93.5 * np.cos(3 * p) + 0.118 * np.cos(5 * p)
    return m_lat, m_lon


def resumo(dens: pd.DataFrame, col: str, limiares=LIMIARES) -> dict:
    """Thresholds in pt/m2 apply only to density columns; the ratio
    (dimensionless) is called with limiares=()."""
    v = dens[col].to_numpy(); v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"n": 0}
    return {"n": int(len(v)), "mediana": float(np.median(v)), "q1": float(np.percentile(v, 25)),
            "q3": float(np.percentile(v, 75)), "media": float(v.mean()),
            **{f"frac_abaixo_de_{t:.2f}": float((v < t).mean()) for t in limiares}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=r"D:\GNN_TOPO\SATELITES\lidar_eba\solo_lidar_30m_v2.parquet")
    ap.add_argument("--saida", default="auditar_densidade_solo.json")
    a = ap.parse_args()
    parquet = Path(a.parquet); saida = RAIZ / a.saida
    d = pd.read_parquet(parquet)
    for c in ("n_all", "n_cl2", "n_casca", "lin", "col", "quad", "idx_no", "dossel_v2", "transecto"):
        if c not in d.columns:
            raise RuntimeError(f"parquet sem coluna {c}")
    # real cell area at the center latitude (project grid)
    phi = LAT1 - (d["lin"].to_numpy() + 0.5) * PASSO
    m_lat, m_lon = metros_por_grau(phi)
    d["area_m2"] = (PASSO * m_lat) * (PASSO * m_lon)
    for c in ("n_all", "n_cl2", "n_casca"):
        d[f"dens_{c}"] = d[c] / d["area_m2"]
    d["piso"] = 0.02 * d["dens_n_all"]
    d["razao_casca_sobre_piso"] = d["dens_n_casca"] / d["piso"].where(d["piso"] > 0)
    d["frac_cl2"] = d["n_cl2"] / d["n_all"]
    # exogenous canopy (ETH) per cell
    fontes = [parquet]
    h = np.full(len(d), np.nan, np.float32); cob = np.full(len(d), np.nan, np.float32)
    for q in QUADS:
        f = LAT / f"dossel_eth_amazonia_{q}.npz"; z = np.load(f); fontes.append(f)
        m = (d["quad"] == q).to_numpy(); idx = d.loc[m, "idx_no"].to_numpy()
        h[m] = z["h_dossel"][idx]; cob[m] = z["frac_valida"][idx]; z.close()
    d["h_eth"] = h; d["cob_eth"] = cob
    d["faixa_eth"] = pd.cut(d["h_eth"].where(d["cob_eth"] >= 0.8), FAIXAS, labels=ROT_F)
    d["faixa_v2"] = pd.cut(d["dossel_v2"], FAIXAS, labels=ROT_F)
    out = {"definicoes": {
        "area": "A(phi) = (PASSO*m_lat)*(PASSO*m_lon), WGS84, phi = latitude do centro da celula; PASSO = 2/7200 graus",
        "populacoes": {"n_all": "todos os retornos com -5 < z < 400 m", "n_cl2": "classe 2 ASPRS do produtor (super-inclusiva)",
                       "n_casca": "retornos com z <= p2_all + 1 m (populacao do estimador de chao)"},
        "piso": "0,02 * dens_n_all — densidade que o p2 impoe a casca por construcao",
        "limiares_declarados_pt_m2": LIMIARES,
        "estratificacao_principal": "ETH exogeno (h_eth, cob_eth >= 0,8)", "sensibilidade": "dossel_v2 (circular)",
        "nota": "a frase do artigo tem de nomear a populacao; nunca 'ground returns' sem qualificador"},
        "area_celula_m2": {"mediana": float(d["area_m2"].median()), "min": float(d["area_m2"].min()), "max": float(d["area_m2"].max()),
                           "erro_relativo_se_900": float(d["area_m2"].median() / 900.0 - 1.0)},
        "n_celulas": int(len(d)), "n_celulas_com_eth": int(d["faixa_eth"].notna().sum()),
        "por_faixa": {}}
    for nome, col in (("ETH", "faixa_eth"), ("dossel_v2", "faixa_v2")):
        out["por_faixa"][nome] = {}
        for rot in ROT_F + ["TODAS"]:
            s = d if rot == "TODAS" else d[d[col] == rot]
            if nome == "ETH" and rot == "TODAS":
                s = d[d["faixa_eth"].notna()]
            reg = {"n_celulas": int(len(s)), "n_transectos": int(s["transecto"].nunique()),
                   "dens_n_all": resumo(s, "dens_n_all"), "dens_n_cl2": resumo(s, "dens_n_cl2"),
                   "dens_n_casca": resumo(s, "dens_n_casca"),
                   "razao_casca_sobre_piso": resumo(s, "razao_casca_sobre_piso", limiares=()),
                   # The ratio OF THE medians exists as a field (1.51 in the >30 m band) alongside the median OF THE ratio (1.50) — cite the median of the ratio
                   "razao_das_medianas_casca_sobre_piso": (float(np.nanmedian(s["dens_n_casca"]) / np.nanmedian(s["piso"])) if len(s) and np.nanmedian(s["piso"]) > 0 else float("nan")),
                   "frac_cl2_mediana": float(s["frac_cl2"].median()) if len(s) else float("nan")}
            out["por_faixa"][nome][rot] = reg
            if nome == "ETH":
                log(f"  ETH {rot:6s} n={len(s):7,d} t={reg['n_transectos']:2d} | casca {reg['dens_n_casca'].get('mediana', float('nan')):.3f} pt/m2 "
                    f"(<0,40: {reg['dens_n_casca'].get('frac_abaixo_de_0.40', float('nan')):.3f}) | cl2 {reg['dens_n_cl2'].get('mediana', float('nan')):.3f} | "
                    f"casca/piso {reg['razao_casca_sobre_piso'].get('mediana', float('nan')):.2f} | frac_cl2 {reg['frac_cl2_mediana']:.2f}")
    out["contra_auditoria"] = ("v1 checked by an independent recomputation (geodesic area, populations, "
                               "floor and numbers recomputed independently); v2 applies D-1 and D-2a")
    PROV.gravar(saida, out, fontes_lidas=fontes, script=__file__)
    log(f"  -> {saida.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
