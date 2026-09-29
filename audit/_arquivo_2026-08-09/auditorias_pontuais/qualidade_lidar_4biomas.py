"""Repeats in the four biomes the exposed-soil label-quality profile first measured in the Cerrado.

What is tested
  Measured over ~408,000 GEDI shots on exposed soil in Cerrado Q1, where the residual
  `DSM - ground` should be zero and therefore IS the label error (median |error|):

    quality_flag       good       - 1.4 m with the flag vs. 108 m without it
    num_modos          good       - 1.2 m with one mode vs. 1709 m with zero modes,
                                    and NOT part of the current filter
    sensitivity >=0.95 useless    - error is flat from 0.8 upward; it rises only below 0.8
    dispersion <=2.0 m inverted   - the perfect-agreement band is the worst, because
                                    perfect agreement is a symptom of degeneracy
    surface_flag       redundant  - what it rejects, quality_flag already rejects

  A result in one quadrant of one biome does not fix a rule. This script repeats the
  measurement in the four biomes, pooling the four quadrants of each, because exposed
  soil is scarce in the Amazon and a single quadrant would not give enough samples.

Usage
  python qualidade_lidar_4biomas.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qualidade_lidar_do_zero import BBOX, TREECOVER_MAX, carregar

SAIDA = Path(__file__).resolve().parent / "qualidade_lidar_4biomas.json"
MIN_AMOSTRA = 500


def med(e: np.ndarray) -> float:
    return round(float(np.median(e)), 2) if e.size else float("nan")


def de_um(bioma: str) -> dict:
    e_l, disp_l, qf_l, nm_l, se_l, sf_l = [], [], [], [], [], []
    n_tot = 0
    for q in ("Q1", "Q2", "Q3", "Q4"):
        c = carregar(bioma, q)
        d, resid, disp, solo = c["d"], c["resid"], c["disp"], c["solo"]
        nu = np.isfinite(solo) & (solo <= TREECOVER_MAX) & np.isfinite(resid)
        n_tot += c["n_total"]
        if not nu.any():
            continue
        e_l.append(np.abs(resid[nu]))
        disp_l.append(disp[nu])
        qf_l.append(d.quality_flag.to_numpy(np.float64)[nu])
        nm_l.append(d.num_modos.to_numpy(np.float64)[nu])
        se_l.append(d.sensitivity.to_numpy(np.float64)[nu])
        sf_l.append(d.surface_flag.to_numpy(np.float64)[nu])
        del c, d, resid, disp, solo

    e = np.concatenate(e_l); disp = np.concatenate(disp_l)
    qf = np.concatenate(qf_l); nm = np.concatenate(nm_l)
    se = np.concatenate(se_l); sf = np.concatenate(sf_l)

    r = {"bioma": bioma, "disparos_no_bioma": n_tot,
         "em_solo_exposto": int(e.size),
         "erro_mediano_bruto_m": med(e)}
    if e.size < MIN_AMOSTRA:
        r["insuficiente"] = True
        return r

    r["quality_flag"] = {"0": med(e[qf == 0]), "1": med(e[qf == 1]),
                         "n0": int((qf == 0).sum()), "n1": int((qf == 1).sum())}
    r["num_modos"] = {"0": med(e[nm == 0]), ">=1": med(e[nm >= 1]),
                      "n0": int((nm == 0).sum()), "n1": int((nm >= 1).sum())}
    r["surface_flag"] = {"0": med(e[sf == 0]), "1": med(e[sf == 1])}

    r["sensitivity"] = {}
    for lo, hi in ((0, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.01)):
        m = (se >= lo) & (se < hi)
        if m.sum() >= MIN_AMOSTRA:
            r["sensitivity"][f"[{lo:g},{hi:g})"] = {"n": int(m.sum()), "erro": med(e[m])}

    # Central question: after the two good rules, does dispersion still carry information?
    ok = (qf == 1) & (nm >= 1)
    r["apos_qf_e_nm"] = {"n": int(ok.sum()), "erro": med(e[ok]), "dispersao": {}}
    for lo, hi in ((0, 0.5), (0.5, 2), (2, 4), (4, 8), (8, 1e9)):
        m = ok & (disp >= lo) & (disp < hi)
        if m.sum() >= MIN_AMOSTRA:
            r["apos_qf_e_nm"]["dispersao"][f"[{lo:g},{hi:g})"] = {
                "n": int(m.sum()), "erro": med(e[m]),
                "p90": round(float(np.percentile(e[m], 90)), 1)}

    # How much each filter regime would let through, and with what error
    reg = {"nenhum": np.ones_like(qf, bool),
           "atual (5 regras)": (qf == 1) & (se >= 0.95) & (disp <= 2.0) & (sf == 1),
           "proposto (qf + num_modos)": ok,
           "proposto + sens>=0.8": ok & (se >= 0.8)}
    r["regimes"] = {k: {"passa_pct": round(100 * float(v.mean()), 2),
                        "erro_mediano_m": med(e[v]),
                        "p90_m": round(float(np.percentile(e[v], 90)), 1) if v.any() else None}
                    for k, v in reg.items()}
    return r


def main() -> int:
    rel = {"gerado_em": datetime.now().isoformat(timespec="seconds"),
           "script": str(Path(__file__).resolve()),
           "regua": "|DSM - solo| sobre solo exposto (cobertura arborea Landsat <= 10%)",
           "biomas": []}
    for b in BBOX:
        print(f"  {b}", flush=True)
        rel["biomas"].append(de_um(b))
    SAIDA.write_text(json.dumps(rel, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  relatorio -> {SAIDA}\n")

    print(f"  {'bioma':16s} {'solo exposto':>13s} {'qf=0':>9s} {'qf=1':>8s} "
          f"{'nm=0':>10s} {'nm>=1':>8s}")
    for r in rel["biomas"]:
        if r.get("insuficiente"):
            print(f"  {r['bioma']:16s} {r['em_solo_exposto']:13,d}  amostra insuficiente")
            continue
        q, n = r["quality_flag"], r["num_modos"]
        print(f"  {r['bioma']:16s} {r['em_solo_exposto']:13,d} {q['0']:9.2f} {q['1']:8.2f} "
              f"{n['0']:10.2f} {n['>=1']:8.2f}")

    print(f"\n  sensitivity — erro mediano por faixa (m)")
    for r in rel["biomas"]:
        if r.get("insuficiente"):
            continue
        s = " | ".join(f"{k} {v['erro']:.2f}" for k, v in r["sensitivity"].items())
        print(f"    {r['bioma']:16s} {s}")

    print(f"\n  depois de quality_flag==1 e num_modos>=1, por faixa de dispersao (m)")
    for r in rel["biomas"]:
        if r.get("insuficiente"):
            continue
        s = " | ".join(f"{k} {v['erro']:.2f}" for k, v in r["apos_qf_e_nm"]["dispersao"].items())
        print(f"    {r['bioma']:16s} {s}")

    print(f"\n  regimes de filtro — quanto passa e com que erro")
    for r in rel["biomas"]:
        if r.get("insuficiente"):
            continue
        print(f"    {r['bioma']}")
        for k, v in r["regimes"].items():
            print(f"      {k:26s} passa {v['passa_pct']:6.2f}%  erro {v['erro_mediano_m']:6.2f} m"
                  f"  p90 {v['p90_m']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
