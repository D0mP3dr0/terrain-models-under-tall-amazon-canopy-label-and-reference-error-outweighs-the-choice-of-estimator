"""TREE ARM -- the same recipe without any neural network.

WHY IT EXISTS. The paper claims that data structure and curation decide the
outcome, not the regressor architecture. Today this is demonstrated with
GATv2 and MLP: two neural networks, same optimizer, same training loop, same
sampler. A reviewer reading this could reasonably call it two
parameterizations of the same family. A gradient-boosted tree ensemble has a
genuinely different inductive bias -- it is not differentiable, does not do
gradient descent, and shares nothing of the framework. If it lands in the
same place under the same recipe, the claim upgrades from "two networks
agree" to "three regressor families land in the same place".

THE FAIRNESS CONTRACT, written BEFORE running:

  SAME DATA. X, y, the partition and the weights come from `preparar_peca` in
  B2 itself -- the SAME function the neural arms use, not a reimplementation.
  Two separate copies could silently diverge, a known failure mode.

  SAME COLUMNS. The comparison arms are all `_semobs`, and `_semobs` keeps
  the two observation columns ZEROED OUT. Constant columns carry no
  information for any regressor, so the tree receives the same 44 real
  columns. Declared: this is not an advantage, it is the same information.

  SAME WEIGHTED SUPERVISION. The per-source boost (ATL08 x K, GEDI x1)
  transplants EXACTLY as `sample_weight`. It is the same arithmetic B2's
  `wfull` builds.

  SAME CURRICULUM. The growing window transplants as model continuation:
  train on Q1, continue the SAME ensemble on Q1+Q2, and so on -- the direct
  analogue of carrying weights forward between stages. Same tree budget per
  stage, proportional to B2's epoch budget per stage (45/25/18/14).

  WHAT DOES NOT TRANSPLANT, DECLARED. Graph total-variation smoothness and
  the physical losses act on the predicted field during optimization; a tree
  has no differentiable optimization to couple them to. That is why the
  comparison is run at the ZERO POINT of the frontier: boosting on,
  smoothness OFF, for all three arms. Regressor is compared against
  regressor, and the TV effect remains a declared component of the recipe
  that applies to the differentiable arms. References on disk:
  `ofat_boost_ba4` (GATv2, boost 4, lambda 0) and the equivalent MLP arm
  from Phase 2.

  FROZEN SCALES. A tree is invariant to a monotonic per-column transform, so
  scale freezing does not affect it -- a justified difference, contract item
  5, verifiable by also running it on the raw columns.

  SEARCH BUDGET. The neural arms consumed ~16 OFAT points. The tree gets the
  same budget: 16 configurations, searched on VALIDATION, never on the
  reserve.

USAGE   python braco_arvores.py --seeds 42 123 7 2024 31
        python braco_arvores.py --seeds 42 --busca      # config search only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
SAIDA = RAIZ / "results" / "f2_arvores.json"
QUADS = ["Q1", "Q2", "Q3", "Q4"]
NTHREAD = max(1, (os.cpu_count() or 8) // 2)   # see note in `treinar`
# tree budget per stage, proportional to B2's epoch budget per stage
ARVORES_ETAPA = [450, 250, 180, 140]
# 16 configurations -- the same search budget as the neural arms
GRADE = [
    {"max_depth": d, "learning_rate": lr, "min_child_weight": mcw,
     "subsample": ss, "colsample_bytree": cs}
    for d in (6, 8)
    for lr in (0.05, 0.10)
    for mcw in (10, 50)
    for ss, cs in ((0.8, 0.8), (1.0, 0.6))
]


def log(m=""):
    print(m, flush=True)


def carregar_tudo(seed: int, boost: float):
    """X, y, partition and weight for each quadrant, via B2's ACTUAL code path."""
    import b2_transferencia as B2
    B2.BOOST_ATL08 = boost          # the same global that B2 uses
    B2.PESO_SUAVE = 0.0             # zero point of the frontier, declared
    pecas = {}
    for q in QUADS:
        dd, lab, prep, _ = B2.preparar_peca(
            "amazonia", q, seed, B2.BLOCO, "v23",
            True, True, False, False, False, True)   # sem_obs=True, like the neural arms
        idx = dd["idx"]
        w = np.ones(len(idx), dtype=np.float32)
        if boost > 0 and dd.get("fonte_anc") is not None:
            f = np.asarray(dd["fonte_anc"])
            w = np.where(f >= 2, float(boost), 1.0).astype(np.float32)
        pecas[q] = {"X": dd["X"][idx], "y": dd["y"], "lab": lab, "w": w,
                    "elev_std": float(dd["elev_std"])}
        del dd, prep
    return pecas


def blocos(pecas, quads, rot):
    """Concatenates the `rot` population (0 train, 1 val, 2 test, 3 reserve)."""
    X = np.concatenate([pecas[q]["X"][pecas[q]["lab"] == rot] for q in quads])
    y = np.concatenate([pecas[q]["y"][pecas[q]["lab"] == rot] for q in quads])
    w = np.concatenate([pecas[q]["w"][pecas[q]["lab"] == rot] for q in quads])
    return X, y, w


def treinar(pecas, cfg, seed: int = 0, quads_ordem=QUADS):
    """Growing-window curriculum via CONTINUATION of the same ensemble."""
    import xgboost as xgb
    modelo = None
    for etapa, n_arv in enumerate(ARVORES_ETAPA):
        janela = quads_ordem[:etapa + 1]
        Xt, yt, wt = blocos(pecas, janela, 0)
        Xv, yv, wv = blocos(pecas, janela, 1)
        d_tr = xgb.DMatrix(Xt, label=yt, weight=wt)
        # WEIGHTED VALIDATION. The previous version built `d_va` without
        # `weight`, so early stopping selected the number of trees by an
        # UNweighted validation error, while B2 selects by weighted
        # validation (`b2_transferencia.py:1729-1734`). This tree arm exists
        # to be compared against the neural arm; selecting by a different
        # criterion breaks exactly what the fairness contract asks to match.
        d_va = xgb.DMatrix(Xv, label=yv, weight=wv)
        # LIMITED THREADS. `nthread=0` claims all 24 logical cores and
        # competes with the GPU queue's neighborhood sampling, which is
        # CPU-bound -- the tree arm would slow down the neural arm it exists
        # to be compared against. Half the machine runs the ensemble in
        # minutes and leaves the other half for the neural runs.
        par = {"objective": "reg:absoluteerror", "tree_method": "hist",
               "eval_metric": "mae", "nthread": NTHREAD,
               # PER-RUN SEED: without this xgboost defaults to 0 and the "5
               # seeds" would share the same subsample randomness, so the
               # standard deviation would only reflect the partition.
               "seed": seed, **cfg}
        modelo = xgb.train(par, d_tr, num_boost_round=n_arv,
                           evals=[(d_va, "val")], xgb_model=modelo,
                           early_stopping_rounds=30, verbose_eval=False)
        # ACTUAL EARLY STOPPING. `early_stopping_rounds` marks the best
        # iteration but does NOT restore it: in xgboost 3.x `predict` uses
        # the full model and `save_best` defaults to False, so trees added
        # past the optimum stayed in the ensemble and were inherited by the
        # next stage. B2 restores the best state
        # (`b2_transferencia.py:1749-1753`); the faithful analogue here is
        # slicing the booster.
        if getattr(modelo, "best_iteration", None) is not None:
            modelo = modelo[: modelo.best_iteration + 1]
    return modelo


def avaliar(modelo, pecas, rot: int) -> dict:
    """Metric IDENTICAL to B2's -- same trivial reference, same aggregation.

    An earlier version used
    `trivial = median of the EVALUATED subset itself`, pooling the 4
    quadrants. B2 uses `constant = MEAN of the TRAINING labels`, per
    QUADRANT, aggregated by n-weighted mean
    (`b2_transferencia.py:1899-1908`). Three divergences: statistic (median
    vs mean), population (evaluated vs training) and aggregation.

    Error magnitude, measured on the SAME predictions: a skill bias of
    -0.0835 -- 52x the GATv2 vs MLP TOST delta (0.0016) and larger than the
    curriculum's interval (0.069). A skill value that large is not
    comparable to anything. Worse: taking the reference from the evaluated
    subset is exactly what B2's own comment (`:1899-1901`) forbids as a
    leakage.
    """
    import xgboost as xgb
    maes, ns, skills = [], [], []
    for q in QUADS:
        pq, lab = pecas[q], pecas[q]["lab"]
        m = lab == rot
        if not m.any():
            continue
        p = modelo.predict(xgb.DMatrix(pq["X"][m]))
        y = pq["y"][m]
        # trivial reference: mean of that quadrant's OBSERVED TRAINING
        # labels -- never from the evaluated subset
        c_triv = float(pq["y"][lab == 0].mean())
        mae = float(np.abs(p - y).mean())
        triv = float(np.abs(y - c_triv).mean())
        maes.append(mae); ns.append(int(m.sum()))
        skills.append(1.0 - mae / max(triv, 1e-9))
    n_tot = sum(ns)
    return {"n": n_tot,
            "mae_media_ponderada_por_n_m": float(np.average(maes, weights=ns)),
            "mae_media_simples_m": float(np.mean(maes)),
            "skill_media_ponderada_por_n": float(np.average(skills, weights=ns)),
            "por_quadrante": {q: {"mae": m_, "n": n_, "skill": s_}
                              for q, m_, n_, s_ in zip(QUADS, maes, ns, skills)}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=[42, 123, 7, 2024, 31])
    ap.add_argument("--boost", type=float, default=4.0)
    ap.add_argument("--busca", action="store_true",
                    help="so busca a configuracao na validacao da 1a semente")
    a = ap.parse_args()
    t0 = time.time()

    log(f"  carregando a semente {a.seeds[0]} pelo caminho do B2 ...")
    pecas = carregar_tudo(a.seeds[0], a.boost)
    n_tr = sum(int((pecas[q]["lab"] == 0).sum()) for q in QUADS)
    log(f"  {n_tr:,} ancoras de treino | {pecas['Q1']['X'].shape[1]} colunas "
        f"| boost {a.boost:g} | suavidade 0 (ponto zero declarado)")

    # ── SEARCH on validation, never on the reserve
    f_cfg = RAIZ / "f2_arvores_config.json"
    if a.busca or not f_cfg.exists():
        log(f"\n  busca de configuracao: {len(GRADE)} pontos, criterio = MAE de "
            f"VALIDACAO da uniao (a reserva nao e tocada)")
        melhor, melhor_v = None, float("inf")
        for i, cfg in enumerate(GRADE, 1):
            m = treinar(pecas, cfg, seed=a.seeds[0])
            v = avaliar(m, pecas, 1)["mae_media_ponderada_por_n_m"]
            marca = ""
            if v < melhor_v:
                melhor_v, melhor, marca = v, cfg, "  <- melhor"
            log(f"    {i:2d}/{len(GRADE)} val MAE {v:.4f} | {cfg}{marca}")
        f_cfg.write_text(json.dumps({"config": melhor, "val_mae": melhor_v,
                                     "grade": len(GRADE),
                                     "criterio": "MAE de validacao da uniao"},
                                    indent=2), encoding="utf-8")
        log(f"  -> {f_cfg.name}: {melhor}")
        if a.busca:
            return 0
    cfg = json.loads(f_cfg.read_text(encoding="utf-8"))["config"]
    log(f"\n  configuracao adotada: {cfg}")

    # ── the seeds
    out = {"contrato": {
        "dados": "preparar_peca do B2 (mesma funcao dos bracos neurais)",
        "colunas": "44 reais; as 2 de observacao sao zeradas em todos os "
                   "bracos `_semobs`",
        "boost": a.boost, "suavidade": 0.0,
        "curriculo": f"continuacao do ensemble, tetos {ARVORES_ETAPA}",
        "busca": f"{len(GRADE)} configuracoes na validacao",
        "nao_transplantado": "TV de grafo e perdas fisicas (nao diferenciavel) "
                             "— por isso a comparacao e no ponto zero da fronteira"},
        "config": cfg, "por_semente": {}}
    for s in a.seeds:
        if s != a.seeds[0]:
            pecas = carregar_tudo(s, a.boost)
        m = treinar(pecas, cfg, seed=s)
        r = {"validacao": avaliar(m, pecas, 1),
             "teste": avaliar(m, pecas, 2),
             "reserva": avaliar(m, pecas, 3)}
        out["por_semente"][str(s)] = r
        log(f"  seed {s:>4d}: val {r['validacao']['mae_media_ponderada_por_n_m']:.4f}"
            f" | reserva MAE {r['reserva']['mae_media_ponderada_por_n_m']:.4f}"
            f" skill {r['reserva']['skill_media_ponderada_por_n']:+.4f}"
            f" | {time.time()-t0:.0f} s")

    sk = [v["reserva"]["skill_media_ponderada_por_n"] for v in out["por_semente"].values()]
    out["resumo"] = {"skill_reserva_media": float(np.mean(sk)),
                     "skill_reserva_dp": float(np.std(sk, ddof=1)),
                     "n_sementes": len(sk)}
    log(f"\n  skill na reserva: {np.mean(sk):+.4f} +- {np.std(sk, ddof=1):.4f} "
        f"({len(sk)} sementes)")
    SAIDA.parent.mkdir(exist_ok=True)
    SAIDA.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float),
                     encoding="utf-8")
    log(f"  -> {SAIDA.name} ({time.time()-t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
