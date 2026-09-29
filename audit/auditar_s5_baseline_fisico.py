"""S5 -- physical baseline of a single variable and external products on the
SAME reserve skill scale (eq. 1) and against the lidar.

Runs ONCE; a reproduction guard failure -> RuntimeError and stops (it is
not patched to fit).

WHAT IT DOES
  Part 1 -- reserve (eq. 1 of the manuscript, S = 1 - MAE(D^)/MAE(D-), D- =
  mean of the target on the TRAINING anchors). The partition, the target,
  the training anchors (lab==0), the reserve anchors (lab==3, 157,847) and
  the per-source weights (ATL08 x4, GEDI x1) come from
  `b2_transferencia.preparar_peca`, IMPORTED (the same path as
  `braco_arvores.carregar_tudo`); the skill aggregation is the one from
  `b2_transferencia.avaliar` / `braco_arvores.avaliar`: per quadrant, with
  the constant = UNweighted mean of y on the quadrant's lab==0 anchors, and
  the union = n-weighted mean of the per-quadrant skills.
    * baseline: D^ = a + b*h, h = ETH canopy (laterais_nativa_diretas,
      `h_dossel`, the SAME file that gives the ETH class in the judge), a
      and b by weighted least squares (per-source weights), only on the
      union's training anchors across the 4 quadrants, per seed. No
      hyperparameter.
    * products: D^ = GLO-30 - product (FABDEM, GEDTM30, ANADEM), no
      adjustment.
    * networks with persisted predictions (D8 x4 GATv2/MLP, Table II;
      b4s035 GATv2/MLP, the adopted recipe) recomputed by the SAME
      function.
    * tree: no persisted prediction (that is S7) -> only the hash-stamped
      value.
  Coverage per arm is recorded; the main comparison uses the common
  intersection.

  Part 2 -- against the lidar: v23 x baseline with the
  `auditar_nmad_pareado.py` pipeline (REF.testar: median of the per-unit
  differences, B=10,000 bootstrap, seed 42) over the judge's
  nativa_diretas population, built by
  `auditar_nmad_pareado_nativa_anadem.montar_populacao_com_anadem`
  (IMPORTED). Transect and footprint units (footprint map read from
  auditar_rotulo_regua_v2.json). MAE and NMAD per ETH class.

GUARDS (fail-loud)
  G0 sha256 of the ANADEM inputs == the value checked by the ANADEM
     auditor.
  G1 the skill recomputed by this script's own function reproduces the
     hash-stamped value of the neural arms in Table II (D8 x4 GATv2 and
     MLP), seed by seed, to <= 0.002, and the per-quadrant trivial
     constant to 1e-4; the idx and y of the persisted predictions match
     preparar_peca's reserve.
  G2 the same pipeline against the lidar reproduces
     auditar_nmad_pareado_nativa_diretas.json (4 original pairs, pool and
     test tables) to 1e-6 -- including v23 vs fabdem MAE >30m median -- and
     the v23 vs mlp ceiling from auditar_rotulo_regua_v2.json to 1e-6.

PRE-DECLARED READING: LS_u = max(|lo|,|hi|) of the IC95 of the median of
dMAE(v23 - baseline) above 30 m. Outcome 1 if, in BOTH the transect AND
footprint units, LS_u < the lower bound of A (mean AND median) in the same
unit; otherwise outcome 2.

Does not interpret the result for the paper, does not audit itself.

USAGE
  /trabalho/ambientes/s33_amb_virtual/.venv/bin/python auditar_s5_baseline_fisico.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# No GPU; V23 paths on Linux.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("V23_LATDIR", "/trabalho/GNN_TOPO/SATELITES/laterais")
os.environ.setdefault("V23_RAIZ", str(Path(__file__).resolve().parent))

import json                                           # noqa: E402

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import proveniencia as PROV                           # noqa: E402

SATEL = Path("/trabalho/GNN_TOPO/SATELITES")
LAT_TREINO = Path(os.environ["V23_LATDIR"])           # training laterais (preparar_peca)
LAT_NATIVA = SATEL / "laterais_nativa_diretas"       # node grid: products, ETH, judge
RECORTE_ANADEM = SATEL / "anadem"
PARQUET = SATEL / "lidar_eba" / "solo_lidar_30m_v3_nativa.parquet"
JUIZ = RAIZ / "juiz_grade_nativa_diretas.json"
REF_NMAD = RAIZ / "auditar_nmad_pareado_nativa_diretas.json"
REGUA_V2 = RAIZ / "auditar_rotulo_regua_v2.json"
ARVORE = RAIZ / "results" / "f2_arvores.json"
PREDS = RAIZ / "predicoes_b2"
SAIDA = RAIZ / "auditar_s5_baseline_fisico.json"

QUADS = ["Q1", "Q2", "Q3", "Q4"]
SEEDS = [42, 123, 7, 2024, 31]
BOOST = 4.0                   # the same point as the arms (boost 4, smoothness 0)
TOL_SKILL = 0.002
TOL_CTRIV = 1e-4
TOL_LIDAR = 1e-6

# neural arms with persisted predictions: name -> (results json, arm, tag)
NEURAIS = {
    "gatv2_x4_D8 (Tabela II)": ("results/d8_triangulo_fator4_gatv2_2_semobs.json",
                                "gatv2_2_semobs", "d8tri"),
    "mlp_x4_D8 (Tabela II)": ("results/d8_triangulo_fator4_mlp_semobs.json",
                              "mlp_semobs", "d8tri"),
    "v23_gatv2_b4s035 (receita adotada)": ("results/ofat_suavfina_b4s035.json",
                                           "gatv2_2_semobs", "b4s035"),
    "mlp_b4s035": ("results/baseline_mlp_b4s035.json", "mlp_semobs", "b4s035"),
}
GUARDA_G1 = ("gatv2_x4_D8 (Tabela II)", "mlp_x4_D8 (Tabela II)")
PRODUTOS_EXT = {"fabdem": "b1", "gedtm30": "DTM", "anadem": "DTM"}


def log(m=""):
    print(m, flush=True)


# ─────────────────────────── Part 1: held-out anchors ───────────────────────────
def carregar_ancoras(seed: int) -> dict:
    """idx, y, lab and per-source weight for each quadrant, via B2's ACTUAL
    code path (same arguments as braco_arvores.carregar_tudo)."""
    import b2_transferencia as B2
    B2.BOOST_ATL08 = BOOST
    B2.PESO_SUAVE = 0.0
    out = {}
    for q in QUADS:
        dd, lab, prep, _ = B2.preparar_peca(
            "amazonia", q, seed, B2.BLOCO, "v23",
            True, True, False, False, False, True)
        idx = np.asarray(dd["idx"]).astype(np.int64)
        w = np.ones(len(idx), dtype=np.float64)
        if dd.get("fonte_anc") is not None:
            f = np.asarray(dd["fonte_anc"])
            w = np.where(f >= 2, BOOST, 1.0).astype(np.float64)
        out[q] = {"idx": idx, "y": np.asarray(dd["y"], dtype=np.float64),
                  "lab": np.asarray(lab).copy(), "w": w}
        del dd, prep
    return out


def skill_reserva(anc: dict, pred: dict, mascara: dict | None = None) -> dict:
    """Eq. (1), aggregation identical to b2_transferencia.avaliar/braco_arvores.avaliar.

    pred[q]: Delta prediction aligned with anc[q]['idx'][lab==3].
    mascara[q]: boolean (same alignment) restricting the evaluated set; the
    trivial constant is always the mean of the quadrant's TRAINING anchors
    (never of the evaluated set)."""
    maes, ns, skills, por_q = [], [], [], {}
    for q in QUADS:
        a = anc[q]
        res = a["lab"] == 3
        y_r = a["y"][res]
        p = np.asarray(pred[q], dtype=np.float64)
        m = np.ones(len(y_r), bool) if mascara is None else mascara[q]
        m = m & np.isfinite(p)
        if not m.any():
            continue
        c_triv = float(a["y"][a["lab"] == 0].mean())
        mae = float(np.abs(p[m] - y_r[m]).mean())
        triv = float(np.abs(y_r[m] - c_triv).mean())
        s = 1.0 - mae / max(triv, 1e-9)
        maes.append(mae); ns.append(int(m.sum())); skills.append(s)
        por_q[q] = {"n": int(m.sum()), "mae_m": mae, "mae_trivial_m": triv,
                    "constante_trivial_de_treino_m": c_triv, "skill": s}
    return {"n": int(sum(ns)),
            "skill_media_ponderada_por_n": float(np.average(skills, weights=ns)),
            "mae_media_ponderada_por_n_m": float(np.average(maes, weights=ns)),
            "por_quadrante": por_q}


def ler_lateral(nome: str, chave: str, q: str) -> np.ndarray:
    z = np.load(LAT_NATIVA / f"{nome}_amazonia_{q}.npz")
    a = z[chave].astype(np.float64)
    z.close()
    return a


def ajustar_baseline(anc: dict, h: dict) -> dict:
    """WLS fit of y = a + b*h on the TRAINING anchors (lab==0) across the
    union of the 4 quadrants, per-source weights; only where h is finite."""
    Y, H, W, n_nan = [], [], [], 0
    for q in QUADS:
        a = anc[q]
        tr = a["lab"] == 0
        hq = h[q][a["idx"][tr]]
        ok = np.isfinite(hq)
        n_nan += int((~ok).sum())
        Y.append(a["y"][tr][ok]); H.append(hq[ok]); W.append(a["w"][tr][ok])
    Y, H, W = np.concatenate(Y), np.concatenate(H), np.concatenate(W)
    X = np.c_[np.ones_like(H), H]
    XtW = X.T * W
    coef = np.linalg.solve(XtW @ X, XtW @ Y)
    return {"a_m": float(coef[0]), "b_m_por_m": float(coef[1]),
            "n_ancoras_treino_usadas": int(len(Y)),
            "n_ancoras_treino_sem_h_eth": n_nan,
            "n_atl08_peso4": int((W > 1).sum()),
            "soma_pesos": float(W.sum())}


def resumo_sementes(vals: list[float]) -> dict:
    v = np.asarray(vals, dtype=float)
    return {"media": float(v.mean()), "dp": float(v.std(ddof=1)) if len(v) > 1 else None,
            "por_semente": [float(x) for x in v]}


def parte1(fontes: list) -> dict:
    log("=== Parte 1: reserva (eq. 1) ===")
    carimbo_neural = {}
    for nome, (arq, braco, tag) in NEURAIS.items():
        d = json.loads((RAIZ / arq).read_text(encoding="utf-8"))
        carimbo_neural[nome] = {}
        for s in SEEDS:
            k = f"amazonia_seed{s}_{braco}_ampliacao_{tag}_v23"
            if k not in d:
                raise RuntimeError(f"{arq}: chave {k} ausente")
            ru = d[k]["reserva_uniao"]
            carimbo_neural[nome][s] = {
                "skill": float(ru["skill_media_ponderada_por_n"]),
                "n": int(ru["n_total"]),
                "ctriv": {q: float(d[k]["quadrantes"][q]["reserva_final"]["comparabilidade"]
                                   ["constante_trivial_de_treino_m"]) for q in QUADS}}
        fontes.append(RAIZ / arq)
    arv = json.loads(ARVORE.read_text(encoding="utf-8"))
    fontes.append(ARVORE)

    # laterais on the node grid (ETH h and products) -- loaded once
    lat = {q: {"h_eth": ler_lateral("dossel_eth", "h_dossel", q),
               "glo30": ler_lateral("glo30", "DEM", q),
               **{p: ler_lateral(p, k, q) for p, k in PRODUTOS_EXT.items()}}
           for q in QUADS}
    h_all = {q: lat[q]["h_eth"] for q in QUADS}

    por_semente, guarda_g1, coef = {}, {}, {}
    cobertura = None
    for s in SEEDS:
        t0 = time.time()
        anc = carregar_ancoras(s)
        n_res = sum(int((anc[q]["lab"] == 3).sum()) for q in QUADS)
        log(f"  semente {s}: ancoras carregadas ({time.time() - t0:.0f} s), reserva n={n_res}")
        if n_res != 157847:
            raise RuntimeError(f"semente {s}: reserva com {n_res} ancoras, esperado 157.847")

        # per-arm predictions, aligned with the reserve
        preds, valid = {}, {}
        for q in QUADS:
            ir = anc[q]["idx"][anc[q]["lab"] == 3]
            L = lat[q]
            valid.setdefault("h_eth", {})[q] = np.isfinite(L["h_eth"][ir])
            for p in PRODUTOS_EXT:
                dp_ = L["glo30"][ir] - L[p][ir]
                preds.setdefault(p, {})[q] = dp_
                valid.setdefault(p, {})[q] = np.isfinite(dp_)
        # baseline
        cf = ajustar_baseline(anc, h_all)
        coef[s] = cf
        for q in QUADS:
            ir = anc[q]["idx"][anc[q]["lab"] == 3]
            preds.setdefault("baseline", {})[q] = cf["a_m"] + cf["b_m_por_m"] * h_all[q][ir]
        valid["baseline"] = valid["h_eth"]
        inter = {q: valid["h_eth"][q] & valid["fabdem"][q] & valid["gedtm30"][q]
                 & valid["anadem"][q] for q in QUADS}

        # neural: persisted predictions + G1
        g1s = {}
        for nome, (arq, braco, tag) in NEURAIS.items():
            pq = {}
            for q in QUADS:
                f = PREDS / f"amazonia_{q}_seed{s}_{braco}_v23_ampliacao_{tag}_reserva_final.npz"
                z = np.load(f)
                ir = anc[q]["idx"][anc[q]["lab"] == 3]
                yr = anc[q]["y"][anc[q]["lab"] == 3]
                if not np.array_equal(np.asarray(z["idx"]).astype(np.int64), ir):
                    raise RuntimeError(f"G1: {f.name} idx != reserva de preparar_peca")
                if np.abs(z["y"].astype(np.float64) - yr).max() > 1e-4:
                    raise RuntimeError(f"G1: {f.name} y != y de preparar_peca")
                pq[q] = z["p"].astype(np.float64)
                z.close()
                fontes.append(f)
            preds[nome] = pq
            r = skill_reserva(anc, pq)
            c = carimbo_neural[nome][s]
            dif = r["skill_media_ponderada_por_n"] - c["skill"]
            dct = max(abs(r["por_quadrante"][q]["constante_trivial_de_treino_m"] - c["ctriv"][q])
                      for q in QUADS)
            ok = abs(dif) <= TOL_SKILL and dct <= TOL_CTRIV and r["n"] == c["n"]
            g1s[nome] = {"recalculado": r["skill_media_ponderada_por_n"],
                         "carimbado": c["skill"], "diferenca": dif,
                         "max_dif_constante_trivial_m": dct, "n": r["n"], "passou": bool(ok)}
            if nome in GUARDA_G1 and not ok:
                raise RuntimeError(f"G1 FALHOU: {nome} semente {s}: recalculado "
                                   f"{r['skill_media_ponderada_por_n']:.6f} vs carimbado "
                                   f"{c['skill']:.6f} (dif {dif:+.2e}), dif constante {dct:.2e}. PARANDO.")
        guarda_g1[s] = g1s
        log("  G1 semente %d: " % s + "; ".join(
            f"{k.split(' ')[0]} {v['diferenca']:+.1e}{'' if v['passou'] else ' FALHOU'}"
            for k, v in g1s.items()))

        # per-arm skills: full population (wherever each predicts) and the intersection
        bracos = ["baseline", "fabdem", "gedtm30", "anadem"] + list(NEURAIS)
        reg = {}
        for b in bracos:
            reg[b] = {"completa": skill_reserva(anc, preds[b]),
                      "intersecao": skill_reserva(anc, preds[b], inter)}
        por_semente[s] = reg
        if cobertura is None:
            cobertura = {
                "n_reserva": n_res,
                "n_validas_por_braco": {
                    k: int(sum(valid[k][q].sum() for q in QUADS))
                    for k in ("h_eth", "fabdem", "gedtm30", "anadem")},
                "n_intersecao_comum": int(sum(inter[q].sum() for q in QUADS)),
                "n_intersecao_por_quadrante": {q: int(inter[q].sum()) for q in QUADS},
                "nota": "baseline valido onde h ETH e finito; redes predizem em toda a "
                        "reserva; produtos onde GLO-30 - produto e finito. A reserva e "
                        "fixa entre sementes (geometrica); a cobertura e a mesma nas 5."}
        log(f"  semente {s}: a={cf['a_m']:+.4f} b={cf['b_m_por_m']:+.5f} | skill (intersecao) "
            + " ".join(f"{b.split(' ')[0]} {reg[b]['intersecao']['skill_media_ponderada_por_n']:+.4f}"
                       for b in bracos))
        del anc

    # per-arm summary (mean +- sd across the 5 seeds)
    resumo = {}
    for b in por_semente[SEEDS[0]]:
        resumo[b] = {pop: {
            "skill": resumo_sementes([por_semente[s][b][pop]["skill_media_ponderada_por_n"]
                                      for s in SEEDS]),
            "mae_m": resumo_sementes([por_semente[s][b][pop]["mae_media_ponderada_por_n_m"]
                                      for s in SEEDS]),
            "n": por_semente[SEEDS[0]][b][pop]["n"]} for pop in ("completa", "intersecao")}
    resumo["arvore (carimbado, sem predicao persistida)"] = {
        "completa": {"skill": {"media": float(arv["resumo"]["skill_reserva_media"]),
                               "dp": float(arv["resumo"]["skill_reserva_dp"]),
                               "por_semente": [float(arv["por_semente"][str(s)]["reserva"]
                                                     ["skill_media_ponderada_por_n"]) for s in SEEDS]},
                     "n": int(arv["por_semente"]["42"]["reserva"]["n"]),
                     "fonte": "results/f2_arvores.json (carimbado; nao recalculado aqui)"},
        "intersecao": "nao calculavel: a arvore nao tem predicao persistida (S7)"}
    ab = {"a_m": resumo_sementes([coef[s]["a_m"] for s in SEEDS]),
          "b_m_por_m": resumo_sementes([coef[s]["b_m_por_m"] for s in SEEDS]),
          "por_semente": {str(s): coef[s] for s in SEEDS}}
    return {"coeficientes_baseline": ab, "cobertura": cobertura,
            "guarda_G1": {str(s): v for s, v in guarda_g1.items()},
            "skill_por_braco_resumo": resumo,
            "skill_por_braco_por_semente": {str(s): v for s, v in por_semente.items()},
            "_lat": lat}


# ─────────────────────────── Parte 2: lidar ───────────────────────────
def parte2(a_bar: float, b_bar: float, fontes: list) -> dict:
    log("\n=== Parte 2: contra o lidar ===")
    import juiz_lidar_v4 as J
    import auditar_nmad_pareado as REF
    import auditar_nmad_pareado_nativa_anadem as ANA
    J.LAT = LAT_NATIVA
    ANA.conferir_sha256_anadem(LAT_NATIVA, RECORTE_ANADEM)            # G0
    juiz = json.loads(JUIZ.read_text(encoding="utf-8"))
    ref = json.loads(REF_NMAD.read_text(encoding="utf-8"))
    regua = json.loads(REGUA_V2.read_text(encoding="utf-8"))

    flor, info = ANA.montar_populacao_com_anadem(PARQUET, juiz, LAT_NATIVA)
    conf = REF.guardas_populacao(flor, juiz)
    log(f"  populacao reproduz tabela_regua_k0 do juiz em {len(conf)} celulas produto x faixa")

    # G2: the pipeline reproduces the hash-stamped reference to 1e-6
    base = ANA.PRODUTOS_BASE
    pool_ex = ANA.tabela_pool_generic(flor, "faixa_ex", base)
    pool_v2 = ANA.tabela_pool_generic(flor, "faixa", base)
    mat_ex = ANA.por_transecto_faixa_generic(flor, "faixa_ex", base)
    mat_v2 = ANA.por_transecto_faixa_generic(flor, "faixa", base)
    principal, sens = ANA.rodar_par(mat_ex, mat_v2, ANA.PARES_BASE)
    g2 = ANA.guarda_reproduz_referencia(pool_ex, pool_v2, principal, sens, ref)
    r_fab = principal["v23 vs fabdem"]["mae"][">30m"]["mediana_m"]
    ref_fab = ref["testes_principal_ETH"]["v23 vs fabdem"]["mae"][">30m"]["mediana_m"]
    if abs(r_fab - ref_fab) > TOL_LIDAR:
        raise RuntimeError(f"G2 FALHOU: v23 vs fabdem MAE >30m {r_fab} != {ref_fab}")
    tetos = {}
    for rot in ("20-30m", ">30m"):
        lo, hi = principal["v23 vs mlp"]["mae"][rot]["ic95_bootstrap_transecto_mediana"]
        meu = max(abs(lo), abs(hi))
        alvo = [c for c in regua["tabela_orcamento_P1"][rot]["contrastes_lidar_detalhe"]
                if c["par"] == "v23 vs mlp"][0]["teto_abs_ddiferenca_m"]
        if abs(meu - alvo) > TOL_LIDAR:
            raise RuntimeError(f"G2 FALHOU: teto v23 vs mlp {rot} {meu} != {alvo}")
        tetos[rot] = {"recalculado_m": meu, "carimbado_m": alvo, "diferenca_m": meu - alvo}
    log(f"  G2: {g2['status']}; v23 vs fabdem MAE >30m mediana {r_fab:.10f} (ref {ref_fab:.10f}); "
        f"teto v23 vs mlp >30m {tetos['>30m']['recalculado_m']:.6f}")

    # baseline as a product: z = GLO-30 - (a + b*h_eth), h from the SAME file as the ETH class
    flor = flor.copy()
    flor["baseline"] = flor["glo30"] - (a_bar + b_bar * flor["h_eth"])
    n_sem_h = int(flor["baseline"].isna().sum())
    n_sem_h_classe = int((flor["baseline"].isna() & flor["faixa_ex"].notna()).sum())
    prods = ["v23", "mlp", "baseline"]
    pool_b_ex = ANA.tabela_pool_generic(flor, "faixa_ex", ["baseline"])
    pool_b_v2 = ANA.tabela_pool_generic(flor, "faixa", ["baseline"])

    # transect unit
    mt_ex = ANA.por_transecto_faixa_generic(flor, "faixa_ex", prods)
    mt_v2 = ANA.por_transecto_faixa_generic(flor, "faixa", prods)
    # footprint unit: cells from transects of the same footprint grouped together
    mapa = regua["populacao"]["unidade_pegada"]["mapa"]
    fp = flor.copy()
    if set(fp["transecto"].astype(str).unique()) - set(mapa):
        raise RuntimeError("transecto sem pegada no mapa de auditar_rotulo_regua_v2.json")
    fp["transecto"] = fp["transecto"].astype(str).map(mapa)
    mp_ex = ANA.por_transecto_faixa_generic(fp, "faixa_ex", prods)
    mp_v2 = ANA.por_transecto_faixa_generic(fp, "faixa", prods)

    testes = {"transecto": {}, "pegada": {}}
    for u, (m_ex, m_v2) in (("transecto", (mt_ex, mt_v2)), ("pegada", (mp_ex, mp_v2))):
        for a_, b_ in (("v23", "baseline"), ("mlp", "baseline")):
            testes[u][f"{a_} vs {b_}"] = {
                "principal_ETH": {"mae": REF.testar(m_ex, a_, b_, None, "mae"),
                                  "nmad": REF.testar(m_ex, a_, b_, None, "nmad")},
                "sensibilidade_dossel_v2": {"mae": REF.testar(m_v2, a_, b_, None, "mae"),
                                            "nmad": REF.testar(m_v2, a_, b_, None, "nmad")}}
    # v23 vs mlp on the footprint (family ceiling on the footprint unit, informative)
    testes["pegada"]["v23 vs mlp"] = {"principal_ETH": {
        "mae": REF.testar(mp_ex, "v23", "mlp", None, "mae"),
        "nmad": REF.testar(mp_ex, "v23", "mlp", None, "nmad")}}

    for u in ("transecto", "pegada"):
        for rot in ("20-30m", ">30m"):
            for met in ("mae", "nmad"):
                r = testes[u]["v23 vs baseline"]["principal_ETH"][met][rot]
                if r.get("testado"):
                    lo, hi = r["ic95_bootstrap_transecto_mediana"]
                    log(f"  {u:9s} {rot:6s} {met:4s} v23-baseline: mediana {r['mediana_m']:+.3f} "
                        f"IC95 [{lo:+.3f}, {hi:+.3f}] vence {r['vence']}/{r['total']} "
                        f"p_w {r['p_wilcoxon']:.4f}")
                else:
                    log(f"  {u} {rot} {met}: nao testado ({r.get('motivo')})")

    # pre-declared reading
    checks, ok1 = {}, True
    for u in ("transecto", "pegada"):
        r = testes[u]["v23 vs baseline"]["principal_ETH"]["mae"][">30m"]
        if not r.get("testado"):
            raise RuntimeError(f"leitura: v23 vs baseline MAE >30m nao testado na unidade {u}")
        lo, hi = r["ic95_bootstrap_transecto_mediana"]
        ls = max(abs(lo), abs(hi))
        A = regua["por_classe_eth"][">30m"]["por_unidade"][u]["A"]
        red = regua["por_classe_eth"][">30m"]["por_unidade"][u]["reducao_mae_equivalente"]
        li_media, li_mediana = A["ic95_pond_celula"][0], A["ic95_mediana"][0]
        c = {"LS_abs_dMAE_v23_menos_baseline_m": ls, "ic95_dMAE_m": [lo, hi],
             "mediana_dMAE_m": r["mediana_m"],
             "LI_A_media_m": li_media, "LI_A_mediana_m": li_mediana,
             "LS_abaixo_LI_A_media": bool(ls < li_media),
             "LS_abaixo_LI_A_mediana": bool(ls < li_mediana),
             "informativo_LI_reducao_mae_equivalente_m": red["ic95_pond_celula"][0],
             "informativo_LS_abaixo_LI_reducao": bool(ls < red["ic95_pond_celula"][0])}
        checks[u] = c
        ok1 = ok1 and c["LS_abaixo_LI_A_media"] and c["LS_abaixo_LI_A_mediana"]
    rt = testes["transecto"]["v23 vs baseline"]["principal_ETH"]["mae"][">30m"]
    if ok1:
        saida = {"vale": "saida_1",
                 "texto_pre_declarado": "nem uma correcao fisica de uma variavel sai do termo conjunto"}
    else:
        saida = {"vale": "saida_2",
                 "texto_pre_declarado": "o claim vale entre as familias aprendidas e a correcao de "
                                        "uma variavel e pior por X m",
                 "X_m": {"mediana_dMAE_v23_menos_baseline_transecto_m": rt["mediana_m"],
                         "ic95_m": rt["ic95_bootstrap_transecto_mediana"],
                         "nota": "dMAE = MAE(v23) - MAE(baseline); negativo = baseline pior "
                                 "(v23 com MAE menor); positivo = baseline melhor"}}
    log(f"  LEITURA: {saida['vale']} | " + "; ".join(
        f"{u}: LS {c['LS_abs_dMAE_v23_menos_baseline_m']:.3f} vs LI_A media {c['LI_A_media_m']:.3f} "
        f"/ mediana {c['LI_A_mediana_m']:.3f}" for u, c in checks.items()))

    fontes += [PARQUET, JUIZ, REF_NMAD, REGUA_V2, RAIZ / "juiz_lidar_v4.py",
               RAIZ / "auditar_nmad_pareado.py", RAIZ / "auditar_nmad_pareado_nativa_anadem.py",
               RECORTE_ANADEM / "anadem_v1_21M_recorte_amazonia_2x2.tif"]
    fontes += [J.SUP / f"delta_amazonia_ampliacao_b4s035{arm}_seed{sd}.npz"
               for arm in ("", "_mlp_semobs") for sd in J.SEEDS_PAR]
    fontes += [LAT_NATIVA / f"{n}_amazonia_{q}.npz" for q in QUADS
               for n in ("agua_jrc", "rotulos")]
    return {
        "populacao": {**info, "n_celulas_sem_h_eth_populacao": n_sem_h,
                      "n_celulas_sem_h_eth_dentro_de_classe_ETH": n_sem_h_classe,
                      "unidade_pegada_mapa": mapa,
                      "n_pegadas": len(set(mapa.values()))},
        "guarda_G2": {"reproduz_auditar_nmad_pareado_nativa_diretas_1e-6": g2,
                      "v23_vs_fabdem_mae_maior30m_mediana": {"recalculado_m": r_fab,
                                                             "carimbado_m": ref_fab,
                                                             "diferenca_m": r_fab - ref_fab},
                      "teto_v23_vs_mlp_auditar_rotulo_regua_v2": tetos,
                      "reproducao_tabela_regua_k0_juiz_celulas": len(conf),
                      "passou": True},
        "produto_baseline": {"definicao": "z = GLO-30 - (a_bar + b_bar * h_eth), h_eth = "
                                          "laterais_nativa_diretas/dossel_eth h_dossel (mesma "
                                          "coluna da classe ETH); a_bar, b_bar = media das 5 "
                                          "sementes (= media das 5 predicoes, como v23/mlp)",
                             "a_bar_m": a_bar, "b_bar_m_por_m": b_bar},
        "tabela_pool_ETH_baseline": pool_b_ex["baseline"],
        "tabela_pool_dossel_v2_baseline": pool_b_v2["baseline"],
        "tabela_pool_ETH_v23_mlp_referencia": {p: pool_ex[p] for p in ("v23", "mlp")},
        "testes": testes,
        "leitura_pre_declarada": {
            "regra": ("LS_u = max(|lo|,|hi|) of the 95% CI of the median of dMAE(v23 - "
                      "baseline) in >30 m; saida_1 if, in the transect AND footprint units, "
                      "LS_u < the lower bound of A (mean AND median) in the same unit "
                      "(auditar_rotulo_regua_v2.json); otherwise saida_2 (rule recorded before "
                      "running)"),
            "checagens_por_unidade": checks, **saida},
    }


def main() -> int:
    t0 = time.time()
    fontes: list = [Path(__file__), RAIZ / "b2_transferencia.py", RAIZ / "vetor_v23.py",
                    RAIZ / "braco_arvores.py"]
    fontes += [LAT_TREINO / f"{n}_amazonia_{q}.npz" for q in QUADS for n in ("rotulos", "glo30")]
    fontes += [LAT_NATIVA / f"{n}_amazonia_{q}.npz" for q in QUADS
               for n in ("dossel_eth", "glo30", "fabdem", "gedtm30", "anadem")]

    p1 = parte1(fontes)
    lat = p1.pop("_lat")
    del lat
    a_bar = p1["coeficientes_baseline"]["a_m"]["media"]
    b_bar = p1["coeficientes_baseline"]["b_m_por_m"]["media"]
    p2 = parte2(a_bar, b_bar, fontes)

    vistos, fontes_u = set(), []
    for f in fontes:
        f = Path(f)
        if str(f) not in vistos:
            vistos.add(str(f)); fontes_u.append(f)
    PROV.gravar(SAIDA, {
        "protocolo": "Pre-declared design note, section S5; runs ONCE",
        "pergunta": "onde fica um piso fisico de uma variavel, e onde ficam os produtos "
                    "externos, na mesma escala de skill da reserva e contra o lidar",
        "definicoes": {
            "skill": "eq. (1): S = 1 - MAE(D^)/MAE(D-), D- = media nao ponderada de y nas "
                     "ancoras lab==0 do quadrante (semente); uniao = media dos S por quadrante "
                     "ponderada por n (b2_transferencia.avaliar, braco_arvores.avaliar)",
            "ancoras": "b2_transferencia.preparar_peca(amazonia, q, seed, BLOCO, v23, True, "
                       "True, False, False, False, True), BOOST_ATL08=4 (como braco_arvores)",
            "pesos_por_fonte": "ATL08 (fonte>=2) x4, demais x1",
            "baseline": "D^ = a + b*h_eth, WLS nas ancoras de treino da uniao, por semente",
            "produtos": "D^ = GLO-30 - produto (laterais_nativa_diretas)",
            "sementes": SEEDS},
        "reserva": p1,
        "lidar": p2,
        "segundos": round(time.time() - t0, 1),
        "contra_auditoria": "PENDENTE -- nao feita por quem escreveu este script",
    }, fontes_lidas=fontes_u, script=__file__)
    log(f"\n  -> {SAIDA.name} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
