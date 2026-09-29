"""D8 — the fair triangle comparison.

Motivation. The earlier triangle comparison was not fair: the lambda_tv=0
GATv2 branch was the most epoch-truncated run in the project (14/20 stages
exhausting the epoch cap; 7/20 with the validation optimum pinned to the
cap; mean best/cap ratio 0.936 — measured by this script, v2) against a
tree model with 10x larger caps and genuine early stopping. D8 reruns both
networks at the SAME point as the tree (boost 4, lambda_tv 0) with a 4x
epoch factor and the tree's 5 seeds. This script decides what the
manuscript text is allowed to say.

PRE-REGISTERED CRITERION (transcribed from the run-queue header, written
BEFORE running):
  Delta_T = skill(XGBoost) - skill(GATv2_factor4), paired, 5 seeds.
  * 95% CI of Delta_T entirely ABOVE +0.010 -> CAPACITY: the tree's lead
    survives the generous budget; the text may state the triangle with the
    tree ahead.
  * 95% CI entirely WITHIN +-0.010 -> BUDGET: the earlier lead was a budget
    artifact; the text states that the three families sit at the same
    level under an adequate budget.
  * Any other case -> UNDETERMINED: the text reports the range and the
    budget qualification, without an ordering.
  ALWAYS TOGETHER: physical admissibility (canopy violation > 1 m, p95 and
  max) reported alongside every skill number from this branch.

WHAT IT READS (nothing is recomputed from raster data; this is a witness
reader):
  * `results/d8_triangulo_fator4_{gatv2_2_semobs,mlp_semobs}.json` — D8.
  * `results/f2_arvores.json` — XGBoost, 5 seeds, same point.
  * `results/ofat_boost_ba4.json` and `results/noite_mlp_boost4_suave0.json`
    — the earlier run (same point, 1x budget), to measure what the budget
    changed.
  Skill per run = `reserva_uniao.skill_media_ponderada_por_n` (networks) and
  `por_semente[s].reserva.skill_media_ponderada_por_n` (tree) — the SAME
  aggregation used in `auditar_e2_e5.py`, over the same 157,847 nodes.
  Admissibility per run = `auditar_admissibilidade.por_corrida` (fields
  `G_raiox.{Q}.F_dossel` / `E_admissibilidade` recorded during training).

GUARDS (fail loud): D8 config with fator_epocas 4, peso_suavidade 0.0,
boost_atl08 4.0, frac_ancoras_treino 1.0, fator_fases 1, batch None; the
earlier-era config with boost 4, batch None, and WITHOUT the newer-era
fields; per-stage epoch caps (45/25/18/14 vs 180/100/72/56) read from the
run scope; effective lambda_tv == 0 verified per WITNESS
(`calibragem_por_epoca[*].pesos.suave`) on every neural run; identical
seeds across the three branches; identical n_total (157,847) on every run;
`ablacao.sem_canal_observacao` True; `etapas` and `quadrantes.treino` agree
on the budget; the run-queue header file itself is listed in `_fontes`.

Usage: python auditar_d8_triangulo.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

import proveniencia as PROV
from auditar_admissibilidade import por_corrida as adm_por_corrida
from auditar_admissibilidade import resumo as adm_resumo

RAIZ = Path(__file__).resolve().parent
SAIDA = RAIZ / "auditar_d8_triangulo.json"
QUADS = ["Q1", "Q2", "Q3", "Q4"]

D8 = {"gatv2": "results/d8_triangulo_fator4_gatv2_2_semobs.json",
      "mlp": "results/d8_triangulo_fator4_mlp_semobs.json"}
E2 = {"gatv2": "results/ofat_boost_ba4.json",
      "mlp": "results/noite_mlp_boost4_suave0.json"}
ARVORE = "results/f2_arvores.json"
FILA = "fila_2026-08-16_d8_f2_f1.sh"     # the pre-registered criterion lives here

MARGEM = 0.010          # same margin used by the pre-registered criterion and the project's TOST
SEMENTES = {42, 123, 7, 2024, 31}
N_RESERVA = 157847

CONFIG_D8_ESPERADA = {"fator_epocas": 4, "peso_suavidade": 0.0,
                      "boost_atl08": 4.0, "frac_ancoras_treino": 1.0,
                      "fator_fases": 1, "batch": None}
# The earlier era (E2) does NOT declare `fator_*`; that absence is part of
# the contract and is therefore VERIFIED, not assumed. `peso_suavidade` is
# left out of this guard because `ofat_boost_ba4.json` stores null with an
# effective value of 0.0 — the effective lambda is checked via a scope
# WITNESS (below), not via config.
CONFIG_E2_ESPERADA = {"boost_atl08": 4.0, "batch": None, "vetor": "v23",
                      "regime": "ampliacao"}
PROIBIDAS_E2 = ("fator_epocas", "fator_fases", "frac_ancoras_treino")
# Per-stage epoch caps: witness the budget independently of the config.
TETOS_X1 = {"etapa1": 45, "etapa2": 25, "etapa3": 18, "etapa4": 14}
TETOS_X4 = {"etapa1": 180, "etapa2": 100, "etapa3": 72, "etapa4": 56}


def log(m=""):
    print(m, flush=True)


# ── readers ─────────────────────────────────────────────────────────────────

def lambda_efetivo(escopo: dict) -> float:
    """MAXIMUM of `pesos.suave` over epochs and the 4 quadrants.

    The `config` field is not reliable: `ofat_boost_ba4.json` stores
    `peso_suavidade: null` yet ran with an effective value of 0.0. The
    witness lives in the scope, recorded by training epoch by epoch
    (`quadrantes.Q*.treino.calibragem_por_epoca[*].pesos.suave`); the
    MAXIMUM is the aggregator because under curriculum training the weight
    ramps in and the first epoch is always 0 (same reading as
    `registro_canonico.era_efetiva`).
    """
    vals = []
    for q in QUADS:
        tre = escopo["quadrantes"][q].get("treino") or {}
        cal = tre.get("calibragem_por_epoca") or []
        if not cal:
            raise RuntimeError(f"{q}: sem calibragem_por_epoca — lambda "
                               "efetivo nao testemunhado, o ponto nao entra")
        for e in cal:
            if "suave" not in (e.get("pesos") or {}):
                raise RuntimeError(f"{q}: pesos sem a chave 'suave'")
            vals.append(float(e["pesos"]["suave"]))
    return max(vals)


def ler_neural(caminho: str, exigir_config: dict, tetos: dict,
               proibidas: tuple = ()) -> dict:
    p = RAIZ / caminho
    d = json.loads(p.read_text(encoding="utf-8"))
    cfg = d.get("config") or {}
    for k, v in exigir_config.items():
        if k not in cfg:
            raise RuntimeError(f"{p.name}: config sem a chave '{k}'")
        if cfg[k] != v:
            raise RuntimeError(f"{p.name}: config.{k} = {cfg[k]!r}, "
                               f"esperado {v!r} — fora do ponto esperado")
    for k in proibidas:
        if cfg.get(k) is not None:
            raise RuntimeError(f"{p.name}: config.{k} = {cfg[k]!r} — a era "
                               "antiga nao pode declarar este campo")
    out = {}
    for k, v in d.items():
        if not (isinstance(v, dict) and v.get("reserva_uniao")):
            continue
        ru = v["reserva_uniao"]
        s = int(v["seed"])
        if not bool(v["ablacao"]["sem_canal_observacao"]):
            raise RuntimeError(f"{p.name}/{k}: canal de observacao LIGADO — "
                               "fora do contrato `_semobs`")
        if int(ru["n_total"]) != N_RESERVA:
            raise RuntimeError(f"{p.name}/{k}: n_total {ru['n_total']} != "
                               f"{N_RESERVA} — reserva diferente, nao pareia")
        lam = lambda_efetivo(v)
        if lam != 0.0:
            raise RuntimeError(f"{p.name}/{k}: lambda_tv efetivo {lam} != 0 — "
                               "fora do ponto casado do triangulo")
        if v["epocas"] != tetos:
            raise RuntimeError(f"{p.name}/{k}: tetos {v['epocas']} nao sao os "
                               f"do orcamento esperado {tetos}")
        etapas = v["etapas"]
        # CAP-BINDING WITNESS. `parou_antes` does NOT measure truncation: the
        # best-epoch weights are ALWAYS restored (b2_transferencia.py:1836-1837).
        # What measures binding is `melhor_epoca` pinned to the cap — the
        # source of the 0.936 figure cited above — and it is not in
        # `etapas`, only in `quadrantes[Q].treino`.
        tre = {q: v["quadrantes"][q]["treino"] for q in QUADS}
        for e in etapas:
            t = tre[e["entrou"]]
            if (t["epocas_usadas"] != e["epocas_usadas"]
                    or t["epocas_teto"] != e["epocas_teto"]):
                raise RuntimeError(f"{p.name}/{k}: etapas e quadrantes"
                                   f"[{e['entrou']}].treino divergem no orcamento")
        out[s] = {
            "skill": float(ru["skill_media_ponderada_por_n"]),
            "mae_m": float(ru["mae_media_ponderada_por_n_m"]),
            "n": int(ru["n_total"]),
            "lambda_efetivo": lam,
            "skill_por_quadrante": {
                q: float(v["quadrantes"][q]["reserva_final"]
                         ["comparabilidade"]["skill"]) for q in QUADS},
            "etapas_no_teto": int(sum(1 for e in etapas if not e["parou_antes"])),
            "n_etapas": len(etapas),
            "epocas_usadas": int(sum(e["epocas_usadas"] for e in etapas)),
            "epocas_teto": int(sum(e["epocas_teto"] for e in etapas)),
            "melhor_epoca_por_etapa": [int(tre[e["entrou"]]["melhor_epoca"])
                                       for e in etapas],
            "etapas_com_melhor_no_teto": int(sum(
                1 for e in etapas
                if tre[e["entrou"]]["melhor_epoca"] >= e["epocas_teto"])),
            "razao_melhor_sobre_teto": [
                tre[e["entrou"]]["melhor_epoca"] / e["epocas_teto"]
                for e in etapas],
        }
    if set(out) != SEMENTES:
        raise RuntimeError(f"{p.name}: sementes {sorted(out)} != "
                           f"{sorted(SEMENTES)}")
    return out


def ler_arvore(caminho: str) -> dict:
    p = RAIZ / caminho
    d = json.loads(p.read_text(encoding="utf-8"))
    c = d["contrato"]
    if float(c["boost"]) != 4.0 or float(c["suavidade"]) != 0.0:
        raise RuntimeError(f"{p.name}: contrato boost/suavidade fora do ponto")
    out = {}
    for s, v in d["por_semente"].items():
        r = v["reserva"]
        if int(r["n"]) != N_RESERVA:
            raise RuntimeError(f"{p.name}/{s}: n {r['n']} != {N_RESERVA}")
        out[int(s)] = {
            "skill": float(r["skill_media_ponderada_por_n"]),
            "mae_m": float(r["mae_media_ponderada_por_n_m"]),
            "n": int(r["n"]),
            "skill_por_quadrante": {q: float(r["por_quadrante"][q]["skill"])
                                    for q in QUADS}}
    if set(out) != SEMENTES:
        raise RuntimeError(f"{p.name}: sementes {sorted(out)}")
    return out


def testemunha_orcamento_arvore(caminho: str) -> dict:
    """The tree ALSO needs a budget witness. `f2_arvores.json` does not
    record `best_iteration` per stage; without it there is no way to tell
    whether early stopping kicked in, or whether it used the full
    [450,250,180,140] caps — the same evidence required from the networks.
    The absence is DECLARED explicitly."""
    d = json.loads((RAIZ / caminho).read_text(encoding="utf-8"))
    tem = "arvores_usadas_por_etapa" in (d.get("resumo") or {})
    return {
        "testemunha_gravada": tem,
        "tetos_declarados": d["contrato"].get("curriculo"),
        "busca_na_validacao": d["contrato"].get("busca"),
        "nota": (None if tem else
                 "f2_arvores.json nao grava best_iteration por etapa; o vinculo "
                 "do teto [450,250,180,140] NAO esta auditado (braco_arvores.py); "
                 "a arvore teve busca de 16 configuracoes na validacao no "
                 "orcamento em que e avaliada, as redes usam ETAPAS_LR de x1 "
                 "transportada para x4 — a folga de justica corre a favor da "
                 "arvore")}


# ── statistics ──────────────────────────────────────────────────────────────

def par(a: dict, b: dict, campo: str = "skill") -> dict:
    comuns = sorted(set(a) & set(b))
    d = np.array([a[s][campo] - b[s][campo] for s in comuns], float)
    n = len(d)
    md, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    t, p = stats.ttest_1samp(d, 0.0)
    tc = stats.t.ppf(0.975, n - 1)
    tc90 = stats.t.ppf(0.95, n - 1)
    # TOST (two one-sided tests at 5%, equivalent to a 90% CI within the
    # margin): informative only, NOT the pre-registered criterion (which
    # requires a 95% CI). Both are reported for completeness.
    t_lo = (md + MARGEM) / se
    t_hi = (md - MARGEM) / se
    p_tost = float(max(1 - stats.t.cdf(t_lo, n - 1), stats.t.cdf(t_hi, n - 1)))
    return {"n": n, "sementes": comuns, "media": md, "dp": sd, "se": se,
            "t": float(t), "p": float(p),
            "ic95": [float(md - tc * se), float(md + tc * se)],
            "ic90": [float(md - tc90 * se), float(md + tc90 * se)],
            "tost_margem": MARGEM, "tost_p": p_tost,
            "tost_equivalente_a_5pct": bool(p_tost < 0.05),
            "sinal_consistente": int(np.sum(np.sign(d) == np.sign(md))),
            "por_semente": {str(s): float(x) for s, x in zip(comuns, d)}}


def desc(d: dict, campo: str = "skill") -> dict:
    x = np.array([v[campo] for v in d.values()], float)
    return {"media": float(x.mean()), "dp": float(x.std(ddof=1)),
            "min": float(x.min()), "max": float(x.max()), "n": len(x)}


def veredito(ic: list[float]) -> str:
    """The THREE branches of the pre-registered criterion, verbatim. The
    case "95% CI entirely below -margin" (network beats the tree) was NOT
    pre-registered and falls into UNDETERMINED per rule 3; it is flagged
    separately in `sub_caso_nao_pre_registrado` so it is not read as an
    inverted ordering."""
    lo, hi = ic
    if lo > MARGEM:
        return "CAPACIDADE"
    if lo > -MARGEM and hi < MARGEM:
        return "ORCAMENTO"
    return "INDETERMINADO"


def sub_caso(ic: list[float]) -> str | None:
    lo, hi = ic
    if hi < -MARGEM:
        return (("Entire CI95 BELOW -margin: the network beats the tree beyond the margin "
                 "— a case not foreseen by the criterion; the text does not rank them; "
                 "requires separate review"))
    return None


def por_quadrante(a: dict, b: dict) -> dict:
    """Conservative reading: paired per-seed contrast WITHIN each quadrant,
    and the sign per quadrant of the mean across seeds (floor p=0.125 with
    4 quadrants — reported as a range, not as a test)."""
    out = {}
    for q in QUADS:
        d = np.array([a[s]["skill_por_quadrante"][q]
                      - b[s]["skill_por_quadrante"][q] for s in sorted(a)])
        t, p = stats.ttest_1samp(d, 0.0)
        out[q] = {"media": float(d.mean()), "dp": float(d.std(ddof=1)),
                  "t": float(t), "p": float(p),
                  "sinal_consistente": int(np.sum(np.sign(d) == np.sign(d.mean())))}
    medias = [out[q]["media"] for q in QUADS]
    out["quadrantes_com_media_positiva"] = int(sum(1 for m in medias if m > 0))
    out["nota"] = ("4 quadrantes: sinal unanime tem p bicaudal 0,125 — "
                   "indecidivel no nivel do quadrante; ler como amplitude. Os p "
                   "por quadrante sao exploratorios (5 sementes dentro do "
                   "quadrante), sem correcao de multiplicidade")
    return out


def orcamento(runs: dict) -> dict:
    ks = sorted(runs)
    tetos = {runs[s]["epocas_teto"] for s in ks}
    if len(tetos) != 1:
        raise RuntimeError(f"tetos diferentes entre sementes: {sorted(tetos)}")
    raz = [r for s in ks for r in runs[s]["razao_melhor_sobre_teto"]]
    return {
        "etapas_no_teto_total": int(sum(runs[s]["etapas_no_teto"] for s in ks)),
        "etapas_total": int(sum(runs[s]["n_etapas"] for s in ks)),
        "razao_epocas_usadas_sobre_teto": float(
            sum(runs[s]["epocas_usadas"] for s in ks)
            / sum(runs[s]["epocas_teto"] for s in ks)),
        "epocas_teto_por_corrida": int(tetos.pop()),
        # the cap only BINDS when the validation optimum is pinned to it
        "etapas_com_melhor_no_teto": int(sum(
            runs[s]["etapas_com_melhor_no_teto"] for s in ks)),
        "etapas_com_melhor_acima_de_090_do_teto": int(
            sum(1 for r in raz if r >= 0.90)),
        "razao_melhor_sobre_teto_media": float(np.mean(raz)),
        "nota": (("`etapas_no_teto` (parou_antes False) does NOT measure truncation — the "
                  "best-epoch weights are always restored (b2_transferencia.py:1836-1837); "
                  "the binding witness is `melhor_epoca/teto`"))}


def main() -> int:
    fontes = [RAIZ / FILA]          # the pre-declared criterion is also a recorded source
    g8 = ler_neural(D8["gatv2"], CONFIG_D8_ESPERADA, TETOS_X4)
    m8 = ler_neural(D8["mlp"], CONFIG_D8_ESPERADA, TETOS_X4)
    g2 = ler_neural(E2["gatv2"], CONFIG_E2_ESPERADA, TETOS_X1, PROIBIDAS_E2)
    m2 = ler_neural(E2["mlp"], CONFIG_E2_ESPERADA, TETOS_X1, PROIBIDAS_E2)
    x = ler_arvore(ARVORE)
    arv_orc = testemunha_orcamento_arvore(ARVORE)
    fontes += [RAIZ / D8["gatv2"], RAIZ / D8["mlp"], RAIZ / E2["gatv2"],
               RAIZ / E2["mlp"], RAIZ / ARVORE]

    # ── 1. triangle table at both budgets
    log("  ponto (boost 4, lambda_tv 0)     braco   skill medio   dp      MAE(m)")
    for nome, rot, d in (("E2 x1", "gatv2", g2), ("E2 x1", "mlp", m2),
                         ("D8 x4", "gatv2", g8), ("D8 x4", "mlp", m8),
                         ("arvore", "xgb", x)):
        a, mm = desc(d), desc(d, "mae_m")
        log(f"  {nome:30s} {rot:6s}  {a['media']:+.4f}   {a['dp']:.4f}  "
            f"{mm['media']:.4f}")

    # ── 2. the pre-registered criterion (skill) plus the same contrast in meters
    dT = par(x, g8)
    dT_mae = par(x, g8, "mae_m")
    ver = veredito(dT["ic95"])
    sub = sub_caso(dT["ic95"])
    log(f"\n  Delta_T = XGB - GATv2_fator4: {dT['media']:+.5f} "
        f"IC95 [{dT['ic95'][0]:+.5f}, {dT['ic95'][1]:+.5f}] "
        f"(t {dT['t']:+.2f}, p {dT['p']:.3g}, {dT['sinal_consistente']}/{dT['n']})"
        f"  -> {ver}")
    log(f"     em metros (MAE): {dT_mae['media']:+.4f} m "
        f"IC95 [{dT_mae['ic95'][0]:+.4f}, {dT_mae['ic95'][1]:+.4f}]"
        f"   | TOST(skill, ±{MARGEM}) p {dT['tost_p']:.4f}, IC90 "
        f"[{dT['ic90'][0]:+.5f}, {dT['ic90'][1]:+.5f}] (informativo)")
    dT_mlp = par(x, m8)
    log(f"  XGB - MLP_fator4:            {dT_mlp['media']:+.5f} "
        f"IC95 [{dT_mlp['ic95'][0]:+.5f}, {dT_mlp['ic95'][1]:+.5f}] "
        f"(t {dT_mlp['t']:+.2f}, {dT_mlp['sinal_consistente']}/{dT_mlp['n']})"
        f"  [exploratorio]")
    dGM = par(g8, m8)
    log(f"  GATv2 - MLP (ambos fator4):  {dGM['media']:+.5f} "
        f"IC95 [{dGM['ic95'][0]:+.5f}, {dGM['ic95'][1]:+.5f}] "
        f"(t {dGM['t']:+.2f}, {dGM['sinal_consistente']}/{dGM['n']})"
        f"  [exploratorio]")

    # ── 3. what the budget changed for each network (D8 - E2, same point)
    dG = par(g8, g2)
    dM = par(m8, m2)
    dT_e2 = par(x, g2)
    log(f"\n  orcamento x4 - x1 no GATv2 (lambda 0): {dG['media']:+.5f} "
        f"IC95 [{dG['ic95'][0]:+.5f}, {dG['ic95'][1]:+.5f}] "
        f"({dG['sinal_consistente']}/{dG['n']})")
    log(f"  orcamento x4 - x1 no MLP   (lambda 0): {dM['media']:+.5f} "
        f"IC95 [{dM['ic95'][0]:+.5f}, {dM['ic95'][1]:+.5f}] "
        f"({dM['sinal_consistente']}/{dM['n']})")
    log(f"  (E2, para memoria) XGB - GATv2_x1:      {dT_e2['media']:+.5f} "
        f"IC95 [{dT_e2['ic95'][0]:+.5f}, {dT_e2['ic95'][1]:+.5f}]")

    # ── 4. cap-binding witness (the reason D8 exists)
    orc = {"gatv2_e2_x1": orcamento(g2), "mlp_e2_x1": orcamento(m2),
           "gatv2_d8_x4": orcamento(g8), "mlp_d8_x4": orcamento(m8)}
    log("\n  vinculo do teto: melhor==teto/etapas | melhor>=0,90 teto | "
        "melhor/teto medio | (parou_antes False)/etapas | usadas/teto")
    for k, o in orc.items():
        n = o["etapas_total"]
        log(f"    {k:14s} {o['etapas_com_melhor_no_teto']:>2d}/{n}   "
            f"{o['etapas_com_melhor_acima_de_090_do_teto']:>2d}/{n}   "
            f"{o['razao_melhor_sobre_teto_media']:.3f}   "
            f"{o['etapas_no_teto_total']:>2d}/{n}   "
            f"{o['razao_epocas_usadas_sobre_teto']:.3f}   "
            f"teto/corrida {o['epocas_teto_por_corrida']}")
    log(f"    arvore: testemunha gravada? {arv_orc['testemunha_gravada']} — "
        f"{arv_orc['tetos_declarados']}")

    # ── 5. admissibility reported ALONGSIDE
    adm = {}
    for rot, arq in (("gatv2_e2_x1", E2["gatv2"]), ("mlp_e2_x1", E2["mlp"]),
                     ("gatv2_d8_x4", D8["gatv2"]), ("mlp_d8_x4", D8["mlp"])):
        adm[rot] = adm_resumo(adm_por_corrida(RAIZ / arq))
    # same key schema for the tree, filled with None
    adm["xgb"] = {k: None for k in adm["mlp_d8_x4"]}
    adm["xgb"]["nota"] = ("f2_arvores.json nao grava G_raiox: a admissibilidade "
                          "da arvore nao esta medida neste artefato (E3, raster "
                          "do XGBoost no eixo hidrologico, e o caminho)")
    log("\n  admissibilidade      viol>1m(med)  min..max   p95(m)* max(m)  "
        "quadrantes zero-por-testemunha + medidos (somam 20)")
    for k in ("gatv2_e2_x1", "mlp_e2_x1", "gatv2_d8_x4", "mlp_d8_x4"):
        r = adm[k]
        log(f"    {k:16s} {r['violacoes_acima_1m_media']:10.1f}  "
            f"{r['violacoes_acima_1m_min']:>4d}..{r['violacoes_acima_1m_max']:<5d} "
            f"{r['violacao_p95_m_max_entre_quadrantes_media']:6.3f}  "
            f"{r['violacao_max_m_pior']:6.2f}   "
            f"{r['quadrantes_zero_por_testemunha_total']:>2d} + "
            f"{r['quadrantes_com_medicao_total']:<2d}")
    log("    xgb              nao medida no artefato (ver nota)")
    log("    * p95 = MAX over quadrants of the within-quadrant p95, not the p95 of the whole run")

    # ── 6. conservative per-quadrant reading
    pq_T = por_quadrante(x, g8)
    pq_GM = por_quadrante(g8, m8)
    log("\n  por quadrante, XGB - GATv2_fator4 (media entre sementes):")
    log("    " + "  ".join(f"{q} {pq_T[q]['media']:+.4f}" for q in QUADS)
        + f"   quadrantes positivos {pq_T['quadrantes_com_media_positiva']}/4")

    # ── 7. the claim the criterion authorizes (copied, not interpreted)
    frases = {
        "CAPACIDADE": ("a lideranca da arvore sobrevive ao orcamento generoso: e "
                       "capacidade; o texto pode afirmar o triangulo com a "
                       "arvore a frente"),
        "ORCAMENTO": ("a lideranca era orcamento; o texto afirma tres familias "
                      "no mesmo patamar sob orcamento adequado"),
        "INDETERMINADO": ("indeterminado; o texto reporta a amplitude e a "
                          "qualificacao de orcamento, sem ordenacao"),
    }
    log(f"\n  VEREDITO D8: {ver} — {frases[ver]}")
    if sub:
        log(f"  sub-caso nao pre-registrado: {sub}")
    log("  (com a coluna de admissibilidade obrigatoria ao lado)")

    PROV.gravar(SAIDA, {
        "criterio_pre_declarado": {
            "fonte": f"{FILA}, cabecalho, secao D8 (arquivo em _fontes)",
            "contraste": "Delta_T = skill(XGBoost) - skill(GATv2_fator4), "
                         "pareado por semente, 5 sementes",
            "margem": MARGEM,
            "regra": {"CAPACIDADE": "IC95 inteiro acima de +margem",
                      "ORCAMENTO": "IC95 inteiro dentro de +-margem",
                      "INDETERMINADO": "qualquer outro caso"},
            "junto_sempre": "admissibilidade (viol. dossel >1 m, p95, max) ao "
                            "lado de todo skill deste braco"},
        "veredito": ver,
        "sub_caso_nao_pre_registrado": sub,
        "frase_autorizada": frases[ver],
        "delta_T_xgb_menos_gatv2_fator4": dT,
        "delta_T_em_metros_mae": dT_mae,
        "exploratorios_sem_pre_registro": {
            "nota": "so Delta_T e pre-declarado; estes contrastes nao tem "
                    "correcao de multiplicidade nem veredito",
            "xgb_menos_mlp_fator4": dT_mlp,
            "gatv2_menos_mlp_fator4": dGM,
            "gatv2_x4_menos_x1": dG, "mlp_x4_menos_x1": dM,
            "xgb_menos_gatv2_x1_E2_para_memoria": dT_e2},
        "resumo_por_braco": {
            "gatv2_e2_x1": {"skill": desc(g2), "mae_m": desc(g2, "mae_m")},
            "mlp_e2_x1": {"skill": desc(m2), "mae_m": desc(m2, "mae_m")},
            "gatv2_d8_x4": {"skill": desc(g8), "mae_m": desc(g8, "mae_m")},
            "mlp_d8_x4": {"skill": desc(m8), "mae_m": desc(m8, "mae_m")},
            "xgb": {"skill": desc(x), "mae_m": desc(x, "mae_m")}},
        "por_semente": {str(s): {"gatv2_d8": g8[s]["skill"],
                                 "mlp_d8": m8[s]["skill"],
                                 "gatv2_e2": g2[s]["skill"],
                                 "mlp_e2": m2[s]["skill"],
                                 "xgb": x[s]["skill"]} for s in sorted(SEMENTES)},
        "vinculo_do_teto": {**orc, "arvore": arv_orc},
        "admissibilidade": {
            "definicao_p95": "violacao_p95_m_max_entre_quadrantes = MAX entre quadrantes do p95 "
                             "intraquadrante (nao p95 da corrida); media entre "
                             "sementes",
            **adm},
        "por_quadrante": {"xgb_menos_gatv2_fator4": pq_T,
                          "gatv2_menos_mlp_fator4": pq_GM},
        "guardas_verificadas": {
            "config_d8": CONFIG_D8_ESPERADA, "config_e2": CONFIG_E2_ESPERADA,
            "proibidas_e2": list(PROIBIDAS_E2),
            "tetos_x1": TETOS_X1, "tetos_x4": TETOS_X4,
            "lambda_efetivo_por_testemunha": "max de pesos.suave em "
                "calibragem_por_epoca == 0.0 nas 20 corridas neurais",
            "concordancia_etapas_vs_quadrantes_treino": "epocas_usadas e "
                "epocas_teto identicos nos dois blocos, por etapa (falha alto)",
            "tetos_uniformes_entre_sementes": "orcamento() exige um unico "
                "epocas_teto por braco (falha alto)",
            "criterio_em_fontes": f"{FILA} e a primeira entrada de _fontes "
                "(sha256), para que edicao do criterio seja detectavel",
            "sementes": sorted(SEMENTES),
            "n_reserva": N_RESERVA, "sem_canal_observacao": True,
            "agregacao": "reserva_uniao.skill_media_ponderada_por_n (redes) / "
                         "por_semente.reserva.skill_media_ponderada_por_n "
                         "(arvore); identica a auditar_e2_e5.py"},
        "ressalvas": [
            "5 sementes: o IC95 usa t(4); a decisao e sobre a media entre "
            "sementes, nao sobre quadrantes (92% da variancia e entre "
            "quadrantes — auditar_unidade_quadrante.json)",
            "A regra pre-declarada exige IC95 dentro da margem, mais dura que o "
            "TOST padrao (IC90); os dois sao reportados, o veredito segue a "
            "regra pre-declarada (reetiquetar seria post-hoc)",
            "SO Delta_T e pre-declarado; os demais contrastes sao exploratorios",
            ("Tuning asymmetry: the tree had a search of 16 configurations on "
             "validation at the budget at which it is evaluated; the networks use "
             "ETAPAS_LR from x1 carried over to x4; the fairness slack runs in favor "
             "of the tree"),
            "a arvore nao tem admissibilidade gravada nem testemunha de "
            "orcamento; as colunas ficam declaradas como ausentes",
        ],
        "contra_auditoria": {
            "data": "2026-08-16",
            "pareceres": ["_pareceres_2026-08-16/fisico_mat_d8.md",
                          "_pareceres_2026-08-16/eng_dados_d8.md",
                          "_pareceres_2026-08-16/eng_ia_d8.md"],
            "veredito_dos_tres": "aprovado com correcoes — aplicadas nesta versao",
        },
    }, fontes_lidas=fontes, script=__file__)
    log(f"  -> {SAIDA.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
