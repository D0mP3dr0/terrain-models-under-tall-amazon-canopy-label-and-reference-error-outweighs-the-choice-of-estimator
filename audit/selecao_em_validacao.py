"""Hyperparameter selection redone WITHOUT touching the final holdout set.

WHY THIS EXISTS. A review found that ~16 OFAT choices (smoothing weight,
boost K, canopy weight) had read `reserva_uniao.skill` -- guard G3 ("the
holdout set is never used for model selection") was violated at the
hyperparameter level. The training runs do not need to be redone: each run
already recorded the VALIDATION of the final stage (`treino.val_mae_melhor`
of the union) and the predictions on `teste_em_dist`, populations that were
already part of the train/test protocol and are NOT the holdout set. This
script reapplies the decision rules reading only those populations, and
answers: DOES THE ADOPTED CHOICE STILL HOLD?

GUARDRAIL TRANSLATION, declared before looking at the numbers: the original
guardrail was 0.010 of skill on the holdout set, equivalent to 0.010 x
trivial MAE (8.09 m) = 0.081 m of MAE. We apply the SAME value in meters to
val_mae: smoothing cannot cost more than 0.081 m of validation against the
zero-smoothing point.

ARTIFACT AXIS UNCHANGED: sinkholes created >2 m do not depend on the
holdout set (they are an unsupervised count over the surface); the values
come from the already recorded comparar_* artifacts. Watch out for a
documented naming collision: the key "16.0" in
`comparar_suavidade_fina_ba4.json` is the BOOST K=16 arm, not lambda=16 --
this script does not read it.

FAMILIES:
  fine smoothing (base boost K=4): lambda 0 / 0.25 / 0.35 / 0.5
  boost K (base pd137 = K=1):      K 1 / 2 / 4 / 16, decided by ATL08 MAE
                                   on teste_em_dist (not on the holdout set)

Usage:  python selecao_em_validacao.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import statistics as st
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
RESULTS = RAIZ / "results"
PRED = RAIZ / "predicoes_b2"
LAT = Path(os.environ.get("V23_LATDIR", r"D:\GNN_TOPO\SATELITES\laterais"))
SAIDA = RAIZ / "selecao_em_validacao.json"
QUADS = ["Q1", "Q2", "Q3", "Q4"]
SEEDS = {42, 123, 7, 2024, 31}
GUARDA_VAL_M = 0.081          # 0,010 de skill x trivial 8,09 m — ver docstring

SUAVIDADE = {                 # lambda -> (results, prediction tag)
    0.0: ("ofat_boost_ba4.json", "ba4"),
    0.25: ("ofat_suavfina_b4s025.json", "b4s025"),
    0.35: ("ofat_suavfina_b4s035.json", "b4s035"),
    0.5: ("ofat_suavfina_b4s05.json", "b4s05"),
}
CRIADOS_2M = {                # artifact axis, from the recorded comparar_* files
    0.0: ("comparar_suavidade_fina_ba4.json", "0.0"),
    0.25: ("comparar_suavidade_fina_ba4.json", "0.25"),
    0.35: ("comparar_ponto_s035.json", "0.35"),
    0.5: ("comparar_suavidade_fina_ba4.json", "0.5"),
}
BOOST = {                     # K -> (results, prediction tag)
    1: ("ofat_dossel_pd137.json", "pd137"),
    2: ("ofat_boost_ba2.json", "ba2"),
    4: ("ofat_boost_ba4.json", "ba4"),
    16: ("ofat_boost_ba16.json", "ba16"),
}


def log(m=""):
    print(m, flush=True)


def val_por_semente(arq: str) -> dict[int, float]:
    """val_mae_melhor of the FINAL STAGE (union Q1..Q4), per seed."""
    d = json.loads((RESULTS / arq).read_text(encoding="utf-8"))
    out = {}
    for k, v in d.items():
        if not (isinstance(v, dict) and v.get("quadrantes")):
            continue
        m = re.search(r"seed(\d+)", k)
        if not m or int(m.group(1)) not in SEEDS:
            continue
        out[int(m.group(1))] = v["quadrantes"]["Q4"]["treino"]["val_mae_melhor"]
    return out


META = None


def meta_fontes() -> pd.DataFrame:
    global META
    if META is None:
        partes = []
        for q in QUADS:
            z = np.load(LAT / f"rotulos_amazonia_{q}.npz")
            bom = z["bom"] > 0.5
            partes.append(pd.DataFrame({
                "quad": q, "idx": z["idx_no"][bom].astype(np.int64),
                "fonte": z["fonte"][bom].astype(np.int8)}))
            z.close()
        META = pd.concat(partes, ignore_index=True)
    return META


def mae_atl08_teste(tag: str) -> dict[int, float]:
    """ATL08 MAE on teste_em_dist, per seed -- test population, not the holdout set."""
    pad = re.compile(rf"amazonia_(Q\d)_seed(\d+)_.+_v23_ampliacao_{tag}"
                     rf"_teste_em_dist\.npz$")
    ls = []
    for f in sorted(glob.glob(str(PRED / f"*_{tag}_teste_em_dist.npz"))):
        m = pad.search(os.path.basename(f))
        if not m or int(m.group(2)) not in SEEDS:
            continue
        z = np.load(f)
        ls.append(pd.DataFrame({
            "quad": m.group(1), "seed": int(m.group(2)),
            "idx": z["idx"].astype(np.int64),
            "e": np.abs(z["p"] - z["y"]).astype(np.float32)}))
        z.close()
    dd = pd.concat(ls, ignore_index=True).merge(meta_fontes(),
                                                on=["quad", "idx"], how="inner")
    atl = dd[dd["fonte"] >= 2]
    return {int(s): float(v) for s, v in atl.groupby("seed")["e"].mean().items()}


def main() -> int:
    out = {"traducao_guarda_corpo": f"{GUARDA_VAL_M} m de val_mae "
                                    "(= 0,010 de skill x trivial 8,09 m)",
           "populacoes_usadas": ["treino.val_mae_melhor (etapa final, uniao)",
                                 "teste_em_dist (MAE ATL08)"],
           "reserva_tocada": False, "familias": {}}

    # ── family 1: fine smoothing
    log("  ===== SUAVIDADE (base boost K=4) — guarda-corpo em VALIDACAO =====")
    vals = {lam: val_por_semente(arq) for lam, (arq, _) in SUAVIDADE.items()}
    base = vals[0.0]
    fam = {}
    log(f"  {'lambda':>7s}{'val_mae':>9s}{'dp':>7s}{'d_vs_0':>9s}"
        f"{'guarda':>8s}{'criados>2m':>12s}{'corte':>7s}")
    cr0 = None
    for lam in sorted(SUAVIDADE):
        v = vals[lam]
        comuns = sorted(set(v) & set(base))
        serie = [v[s] for s in comuns]
        dvs = [v[s] - base[s] for s in comuns]
        arq_c, chave = CRIADOS_2M[lam]
        cr = json.loads((RAIZ / arq_c).read_text(encoding="utf-8"))[
            "por_peso"][chave]["criados_2m_soma"]
        if lam == 0.0:
            cr0 = cr
        corte = 1.0 - cr / cr0 if cr0 else 0.0
        dentro = st.mean(dvs) <= GUARDA_VAL_M
        fam[str(lam)] = {"val_mae": st.mean(serie), "dp": st.stdev(serie),
                         "delta_vs_0": st.mean(dvs), "n_sementes": len(comuns),
                         "dentro_guarda_val": dentro,
                         "criados_2m": cr, "corte_criados": corte}
        log(f"  {lam:>7.2f}{st.mean(serie):>9.3f}{st.stdev(serie):>7.3f}"
            f"{st.mean(dvs):>+9.3f}{str(dentro):>8s}{cr:>12,.0f}{corte:>7.1%}")
    candidatos = [lam for lam in sorted(SUAVIDADE) if lam > 0
                  and fam[str(lam)]["dentro_guarda_val"]
                  and fam[str(lam)]["corte_criados"] >= 0.5]
    eleito = candidatos[0] if candidatos else None
    fam["_regra"] = ("menor lambda com corte >=50% dos criados >2m e "
                     f"delta de val_mae <= {GUARDA_VAL_M} m")
    fam["_eleito"] = eleito
    fam["_adotado_sobrevive"] = (eleito == 0.35)
    log(f"  regra em validacao elege: {eleito} | adotado 0,35 "
        f"{'SOBREVIVE' if eleito == 0.35 else 'NAO e o eleito'}")
    out["familias"]["suavidade"] = fam

    # ── family 2: boost K
    log("\n  ===== BOOST K — MAE ATL08 no TESTE (nao na reserva) =====")
    fam2 = {}
    log(f"  {'K':>4s}{'ATL08_teste':>12s}{'dp':>7s}{'val_mae':>9s}{'dp':>7s}")
    for K in sorted(BOOST):
        arq, tag = BOOST[K]
        atl = mae_atl08_teste(tag)
        vv = val_por_semente(arq)
        comuns = sorted(set(atl) & set(vv))
        fam2[str(K)] = {"mae_atl08_teste": st.mean([atl[s] for s in comuns]),
                        "dp_atl08": st.stdev([atl[s] for s in comuns]),
                        "val_mae": st.mean([vv[s] for s in comuns]),
                        "dp_val": st.stdev([vv[s] for s in comuns]),
                        "n_sementes": len(comuns)}
        f2 = fam2[str(K)]
        log(f"  {K:>4d}{f2['mae_atl08_teste']:>12.3f}{f2['dp_atl08']:>7.3f}"
            f"{f2['val_mae']:>9.3f}{f2['dp_val']:>7.3f}")
    # saturation: smallest K whose additional ATL08 gain is within 1 sd
    ks = sorted(BOOST)
    eleito_k = ks[-1]
    for i, K in enumerate(ks[:-1]):
        ganho_restante = (fam2[str(K)]["mae_atl08_teste"]
                          - min(fam2[str(k2)]["mae_atl08_teste"]
                                for k2 in ks[i + 1:]))
        if ganho_restante <= fam2[str(K)]["dp_atl08"]:
            eleito_k = K
            break
    fam2["_regra"] = ("menor K alem do qual o ganho de MAE ATL08 no teste "
                      "fica dentro de 1 dp entre sementes")
    fam2["_eleito"] = eleito_k
    fam2["_adotado_sobrevive"] = (eleito_k == 4)
    log(f"  regra em teste elege: K={eleito_k} | adotado K=4 "
        f"{'SOBREVIVE' if eleito_k == 4 else 'NAO e o eleito'}")
    out["familias"]["boost"] = fam2

    SAIDA.write_text(json.dumps(out, indent=2, ensure_ascii=False,
                                default=float), encoding="utf-8")
    log(f"\n  -> {SAIDA.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
