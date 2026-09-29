"""F2 — the cell (fraction 0.25 x factor 1) that deconfounds D2.

Motivation. D2's sample-efficiency curve confounded anchor fraction with
epoch factor BY DESIGN (100% -> x1, 50% -> x2, 25% -> x4). D1 (100%, x4)
was the fourth cell that revealed the cancellation. F2 (25%, x1) closes
the 2x2 {100%, 25%} x {x1, x4} design and lets the effect of fraction at a
fixed budget, the effect of factor at a fixed fraction, and their
interaction all be estimated separately. This script decides what the
text can say about supervision dependence.

PRE-REGISTERED CRITERION (transcribed from the header of
`fila_2026-08-16_d8_f2_f1.sh`, section F2, written BEFORE running):
  Delta_frac(branch) = skill(0.25, x1) - skill(1.00, x1), paired, 5 seeds.
  * If Delta_frac(GATv2) is NEGATIVE beyond -0.010 and more negative than
    that of the MLP, this confirms that, at a comparable budget, the graph
    DEPENDS more on supervision.
  * Otherwise, differential dependence is not established and the text
    stays with the common-degradation reading.
  Operationalization (fixed here, before reading the MLP numbers): "beyond
  -0.010" = 95% CI of Delta_frac(GATv2) entirely below -0.010; "more
  negative than the MLP's" = 95% CI of the paired difference
  [Delta_frac(GATv2) - Delta_frac(MLP)] entirely below 0. Both conditions
  together -> DIFFERENTIAL DEPENDENCE; any other combination -> COMMON
  DEGRADATION (the text reports both degradations and the difference with
  a CI, without claiming differential dependence).

The 2x2 (every cell at the adopted point: boost 4, lambda_tv 0.35, default
batch, `_semobs`):
  (1.00, x1) adopted   : results/ofat_suavfina_b4s035.json / baseline_mlp_b4s035.json
  (0.25, x1) F2        : results/f2cel_frac025_fator1_{gatv2_2_semobs,mlp_semobs}.json
  (1.00, x4) D1        : results/d1_orcamento_{...}.json (10 seeds; uses the 5 common ones)
  (0.25, x4) D2        : results/d2_frac025_{...}.json
Skill per run = `reserva_uniao.skill_media_ponderada_por_n` (same
aggregation as auditar_e2_e5 / auditar_d8_triangulo).

GUARDS (fail loud, via scope WITNESS, not via config): per-stage epoch
caps (45/25/18/14 for x1, 180/100/72/56 for x4); effective lambda_tv = max
of `calibragem_por_epoca[*].pesos.suave` == 0.35; `particao.
frac_ancoras_treino_pedida` == 0.25 and `realizada` within 0.25 +- 0.002
in the 25% cells, and absent/1.0 in the 100% cells; `sem_canal_observacao`;
n_total 157,847; 5 seeds {42,123,7,2024,31} present in every cell; declared
config coherent where present (newer era). Admissibility reported
alongside via `auditar_admissibilidade.por_corrida`.

Usage: python auditar_f2_celula.py
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
SAIDA = RAIZ / "auditar_f2_celula.json"
FILA = "fila_2026-08-16_d8_f2_f1.sh"
QUADS = ["Q1", "Q2", "Q3", "Q4"]
MARGEM = 0.010
SEMENTES = {42, 123, 7, 2024, 31}
N_RESERVA = 157847
LAMBDA = 0.35
TOL_FRAC = 0.002
# exact text written by b2_transferencia.py (garantias.peso_rotulo.origem)
BOOST_ORIGEM_ESPERADA = "boost por fonte — ATL08 x4, GEDI x1"
TETOS = {1: {"etapa1": 45, "etapa2": 25, "etapa3": 18, "etapa4": 14},
         4: {"etapa1": 180, "etapa2": 100, "etapa3": 72, "etapa4": 56}}

CELULAS = {  # (fraction, factor) -> {arm: file}
    (1.0, 1): {"gatv2": "results/ofat_suavfina_b4s035.json",
               "mlp": "results/baseline_mlp_b4s035.json"},
    (0.25, 1): {"gatv2": "results/f2cel_frac025_fator1_gatv2_2_semobs.json",
                "mlp": "results/f2cel_frac025_fator1_mlp_semobs.json"},
    (1.0, 4): {"gatv2": "results/d1_orcamento_gatv2_2_semobs.json",
               "mlp": "results/d1_orcamento_mlp_semobs.json"},
    (0.25, 4): {"gatv2": "results/d2_frac025_gatv2_2_semobs.json",
                "mlp": "results/d2_frac025_mlp_semobs.json"},
}
# declared config that the NEWER era must carry (the older/adopted era
# does not declare factor/fraction; the scope witness covers both)
CONFIG_ERA_NOVA = {(0.25, 1): {"fator_epocas": 1, "frac_ancoras_treino": 0.25,
                               "peso_suavidade": LAMBDA, "boost_atl08": 4.0,
                               "batch": None},
                   (1.0, 4): {"fator_epocas": 4, "frac_ancoras_treino": 1.0,
                              "peso_suavidade": LAMBDA, "boost_atl08": 4.0,
                              "batch": None},
                   (0.25, 4): {"fator_epocas": 4, "frac_ancoras_treino": 0.25,
                               "peso_suavidade": LAMBDA, "boost_atl08": 4.0,
                               "batch": None}}


def log(m=""):
    print(m, flush=True)


def lambda_efetivo(escopo: dict) -> float:
    vals = []
    for q in QUADS:
        cal = (escopo["quadrantes"][q].get("treino") or {}).get(
            "calibragem_por_epoca") or []
        if not cal:
            raise RuntimeError(f"{q}: sem calibragem_por_epoca")
        for e in cal:
            if "suave" not in (e.get("pesos") or {}):
                raise RuntimeError(f"{q}: pesos sem 'suave'")
            vals.append(float(e["pesos"]["suave"]))
    return max(vals)


def ler(caminho: str, fracao: float, fator: int, celula) -> dict:
    p = RAIZ / caminho
    d = json.loads(p.read_text(encoding="utf-8"))
    cfg = d.get("config") or {}
    if celula in CONFIG_ERA_NOVA:
        for k, v in CONFIG_ERA_NOVA[celula].items():
            if k not in cfg:
                raise RuntimeError(f"{p.name}: config sem '{k}'")
            if cfg[k] != v:
                raise RuntimeError(f"{p.name}: config.{k}={cfg[k]!r}, "
                                   f"esperado {v!r}")
    else:
        for k in ("fator_epocas", "frac_ancoras_treino", "fator_fases"):
            if cfg.get(k) is not None:
                raise RuntimeError(f"{p.name}: era antiga declara {k}")
        if cfg.get("batch") is not None:
            raise RuntimeError(f"{p.name}: batch explicito na era antiga")
    out = {}
    for k, v in d.items():
        if not (isinstance(v, dict) and v.get("reserva_uniao")):
            continue
        s = int(v["seed"])
        if s not in SEMENTES:
            continue                       # D1 has 10 seeds; use the 5 shared ones
        ru = v["reserva_uniao"]
        if not bool(v["ablacao"]["sem_canal_observacao"]):
            raise RuntimeError(f"{p.name}/{k}: canal de observacao ligado")
        if int(ru["n_total"]) != N_RESERVA:
            raise RuntimeError(f"{p.name}/{k}: n_total {ru['n_total']}")
        if v["epocas"] != TETOS[fator]:
            raise RuntimeError(f"{p.name}/{k}: tetos {v['epocas']} != "
                               f"{TETOS[fator]} (fator {fator})")
        lam = lambda_efetivo(v)
        if abs(lam - LAMBDA) > 1e-9:
            raise RuntimeError(f"{p.name}/{k}: lambda efetivo {lam} != {LAMBDA}")
        n_base = {}
        for q in QUADS:
            pa = v["quadrantes"][q].get("particao") or {}
            ped = pa.get("frac_ancoras_treino_pedida")
            rea = pa.get("frac_ancoras_treino_realizada")
            # BOOST WITNESS: `garantias.peso_rotulo.origem` is formatted from
            # the effective variable (b2_transferencia.py:1179-1180); the
            # older era does not declare boost in the config.
            origem = ((v["quadrantes"][q].get("garantias") or {})
                      .get("peso_rotulo") or {}).get("origem")
            # EXACT equality (a substring match on "ATL08 x4" would also
            # match x40/x4.5)
            if origem != BOOST_ORIGEM_ESPERADA:
                raise RuntimeError(f"{p.name}/{k}/{q}: peso_rotulo.origem = "
                                   f"{origem!r} — boost 4 nao testemunhado")
            if fracao == 1.0:
                if ped not in (None, 1.0):
                    raise RuntimeError(f"{p.name}/{k}/{q}: fracao pedida {ped} "
                                       "numa celula de 100%")
                # The absence of the field is NOT proof of a full fraction: the
                # positive witness is n_observada, compared across cells in
                # `verificar_pool_comum` (it must equal
                # n_observada_antes_da_subamostragem of the 25% cells).
                if "n_observada" not in pa:
                    raise RuntimeError(f"{p.name}/{k}/{q}: particao sem "
                                       "n_observada")
                n_base[q] = int(pa["n_observada"])
            else:
                if "n_observada_antes_da_subamostragem" not in pa:
                    raise RuntimeError(f"{p.name}/{k}/{q}: particao sem "
                                       "n_observada_antes_da_subamostragem")
                n_base[q] = int(pa["n_observada_antes_da_subamostragem"])
                if ped != fracao:
                    raise RuntimeError(f"{p.name}/{k}/{q}: fracao pedida {ped} "
                                       f"!= {fracao}")
                if rea is None or abs(float(rea) - fracao) > TOL_FRAC:
                    raise RuntimeError(f"{p.name}/{k}/{q}: fracao realizada "
                                       f"{rea} fora de {fracao}+-{TOL_FRAC}")
        if s in out:
            raise RuntimeError(f"{p.name}: semente {s} duplicada")
        # cap-binding via the correct witness: melhor_epoca/teto from
        # `quadrantes[Q].treino`, agreeing with `etapas`
        tre = {q: v["quadrantes"][q]["treino"] for q in QUADS}
        etapas = v["etapas"]
        for e in etapas:
            tq = tre[e["entrou"]]
            if (tq["epocas_usadas"] != e["epocas_usadas"]
                    or tq["epocas_teto"] != e["epocas_teto"]):
                raise RuntimeError(f"{p.name}/{k}: etapas x treino divergem")
        out[s] = {
            "skill": float(ru["skill_media_ponderada_por_n"]),
            "mae_m": float(ru["mae_media_ponderada_por_n_m"]),
            "lambda_efetivo": lam,
            "n_base_por_quadrante": n_base,
            "etapas_com_melhor_no_teto": int(sum(
                1 for e in etapas
                if tre[e["entrou"]]["melhor_epoca"] >= e["epocas_teto"])),
            "razao_melhor_sobre_teto": [tre[e["entrou"]]["melhor_epoca"]
                                        / e["epocas_teto"] for e in etapas],
            "skill_por_quadrante": {
                q: float(v["quadrantes"][q]["reserva_final"]
                         ["comparabilidade"]["skill"]) for q in QUADS}}
    if set(out) != SEMENTES:
        raise RuntimeError(f"{p.name}: sementes {sorted(out)} != "
                           f"{sorted(SEMENTES)}")
    return out


def par(a: dict, b: dict, campo: str = "skill") -> dict:
    comuns = sorted(set(a) & set(b))
    d = np.array([a[s][campo] - b[s][campo] for s in comuns], float)
    n = len(d)
    md, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    t, p = stats.ttest_1samp(d, 0.0)
    tc = stats.t.ppf(0.975, n - 1)
    return {"n": n, "sementes": comuns, "media": md, "dp": sd, "se": se,
            "t": float(t), "p": float(p),
            "ic95": [float(md - tc * se), float(md + tc * se)],
            "sinal_consistente": int(np.sum(np.sign(d) == np.sign(md))),
            "por_semente": {str(s): float(x) for s, x in zip(comuns, d)}}


def par_de_dif(da: dict, db: dict) -> dict:
    """CI of the paired difference between two paired differences (same seeds)."""
    comuns = sorted(set(da["por_semente"]) & set(db["por_semente"]))
    d = np.array([da["por_semente"][s] - db["por_semente"][s] for s in comuns])
    n = len(d)
    md, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    t, p = stats.ttest_1samp(d, 0.0)
    tc = stats.t.ppf(0.975, n - 1)
    return {"n": n, "media": md, "dp": sd, "se": se, "t": float(t),
            "p": float(p), "ic95": [float(md - tc * se), float(md + tc * se)],
            "sinal_consistente": int(np.sum(np.sign(d) == np.sign(md))),
            "por_semente": {str(s): float(x) for s, x in zip(comuns, d)}}


def verificar_pool_comum(cel: dict) -> dict:
    """The anchor pool (n_observada in the 100% cells; n_observada_antes_da_
    subamostragem in the 25% cells) must be EQUAL across every cell and
    branch, per seed and quadrant — this is the positive witness that the
    100% and 25% cells start from the same set."""
    ref = {}
    conferidos = 0
    for (fr, ft), bb in cel.items():
        for br, runs in bb.items():
            for s in sorted(SEMENTES):
                nb = runs[s]["n_base_por_quadrante"]
                if s not in ref:
                    ref[s] = nb
                elif ref[s] != nb:
                    raise RuntimeError(f"pool de ancoras difere na celula "
                                       f"({fr},{ft}) {br} semente {s}: {nb} != "
                                       f"{ref[s]}")
                else:
                    conferidos += len(QUADS)   # quadrant values cross-checked
    return {"valores_de_quadrante_confrontados": conferidos,
            "nota": "1a celula de cada semente e a referencia; as demais 7 "
                    "celulas x 4 quadrantes sao confrontadas contra ela",
            "n_base_por_semente_quadrante": {str(s): v for s, v in ref.items()}}


def truncamento(runs: dict) -> dict:
    ks = sorted(runs)
    raz = [r for s in ks for r in runs[s]["razao_melhor_sobre_teto"]]
    return {"etapas_com_melhor_no_teto": int(sum(
                runs[s]["etapas_com_melhor_no_teto"] for s in ks)),
            "etapas_total": int(sum(len(runs[s]["razao_melhor_sobre_teto"])
                                    for s in ks)),
            "razao_melhor_sobre_teto_media": float(np.mean(raz))}


def por_quadrante_dif(a25: dict, a100: dict, b25: dict, b100: dict) -> dict:
    """Delta_frac per quadrant in each arm and the difference between arms."""
    out = {}
    for q in QUADS:
        da = np.array([a25[s]["skill_por_quadrante"][q]
                       - a100[s]["skill_por_quadrante"][q]
                       for s in sorted(SEMENTES)])
        db = np.array([b25[s]["skill_por_quadrante"][q]
                       - b100[s]["skill_por_quadrante"][q]
                       for s in sorted(SEMENTES)])
        d = da - db
        n = len(d)
        md = float(d.mean())
        se = float(d.std(ddof=1)) / math.sqrt(n)
        tc = stats.t.ppf(0.975, n - 1)
        out[q] = {"delta_frac_gatv2": float(da.mean()),
                  "delta_frac_mlp": float(db.mean()),
                  "dif_gatv2_menos_mlp": md,
                  "ic95": [md - tc * se, md + tc * se],
                  "sinal_consistente": int(np.sum(np.sign(d) == np.sign(md)))}
    return out


def desc(d: dict, campo: str = "skill") -> dict:
    x = np.array([v[campo] for v in d.values()], float)
    return {"media": float(x.mean()), "dp": float(x.std(ddof=1)),
            "min": float(x.min()), "max": float(x.max()), "n": len(x)}


def main() -> int:
    fontes = [RAIZ / FILA]
    cel = {}
    for (fr, ft), bracos in CELULAS.items():
        cel[(fr, ft)] = {}
        for br, arq in bracos.items():
            p = RAIZ / arq
            if not p.exists():
                raise RuntimeError(f"ausente: {arq}")
            fontes.append(p)
            cel[(fr, ft)][br] = ler(arq, fr, ft, (fr, ft))

    pool = verificar_pool_comum(cel)
    rot = lambda fr, ft: f"({int(fr*100):>3d}%, x{ft})"
    log("  celula          braco   skill medio   dp      MAE(m)")
    for (fr, ft), bb in cel.items():
        for br, d in bb.items():
            a, mm = desc(d), desc(d, "mae_m")
            log(f"  {rot(fr, ft):14s}  {br:6s}  {a['media']:+.4f}   {a['dp']:.4f}  "
                f"{mm['media']:.4f}")

    # ── the criterion: effect of fraction at factor 1
    dfg = par(cel[(0.25, 1)]["gatv2"], cel[(1.0, 1)]["gatv2"])
    dfm = par(cel[(0.25, 1)]["mlp"], cel[(1.0, 1)]["mlp"])
    dif = par_de_dif(dfg, dfm)
    cond1 = dfg["ic95"][1] < -MARGEM
    cond2 = dif["ic95"][1] < 0.0
    ver = "DEPENDENCIA_DIFERENCIAL" if (cond1 and cond2) else "DEGRADACAO_COMUM"
    log(f"\n  Delta_frac(GATv2) a x1: {dfg['media']:+.5f} IC95 "
        f"[{dfg['ic95'][0]:+.5f}, {dfg['ic95'][1]:+.5f}] "
        f"({dfg['sinal_consistente']}/{dfg['n']})   cond1 (IC inteiro < -0,010): {cond1}")
    log(f"  Delta_frac(MLP)   a x1: {dfm['media']:+.5f} IC95 "
        f"[{dfm['ic95'][0]:+.5f}, {dfm['ic95'][1]:+.5f}] "
        f"({dfm['sinal_consistente']}/{dfm['n']})")
    log(f"  GATv2 - MLP (dif. das degradacoes): {dif['media']:+.5f} IC95 "
        f"[{dif['ic95'][0]:+.5f}, {dif['ic95'][1]:+.5f}] "
        f"({dif['sinal_consistente']}/{dif['n']})   cond2 (IC inteiro < 0): {cond2}")
    log(f"  -> VEREDITO F2: {ver}")

    # ── the same contrast in meters of MAE (physical unit)
    dfg_m = par(cel[(0.25, 1)]["gatv2"], cel[(1.0, 1)]["gatv2"], "mae_m")
    dfm_m = par(cel[(0.25, 1)]["mlp"], cel[(1.0, 1)]["mlp"], "mae_m")
    dif_m = par_de_dif(dfg_m, dfm_m)
    log(f"  em metros de MAE: GATv2 {dfg_m['media']:+.4f} m, MLP "
        f"{dfm_m['media']:+.4f} m, diferenca {dif_m['media']:+.4f} m "
        f"[{dif_m['ic95'][0]:+.4f}, {dif_m['ic95'][1]:+.4f}]")

    # ── the full 2x2 (exploratory, no multiplicity correction)
    ex = {}
    for br in ("gatv2", "mlp"):
        ex[br] = {
            "frac_a_x1": par(cel[(0.25, 1)][br], cel[(1.0, 1)][br]),
            "frac_a_x4": par(cel[(0.25, 4)][br], cel[(1.0, 4)][br]),
            "fator_a_100": par(cel[(1.0, 4)][br], cel[(1.0, 1)][br]),
            "fator_a_025": par(cel[(0.25, 4)][br], cel[(0.25, 1)][br]),
        }
        ex[br]["interacao_frac_x_fator"] = par_de_dif(ex[br]["frac_a_x4"],
                                                      ex[br]["frac_a_x1"])
    log("\n  2x2 (exploratorio)   frac@x1    frac@x4    fator@100  fator@25   interacao")
    for br in ("gatv2", "mlp"):
        e = ex[br]
        log(f"  {br:6s}              {e['frac_a_x1']['media']:+.4f}    "
            f"{e['frac_a_x4']['media']:+.4f}    {e['fator_a_100']['media']:+.4f}    "
            f"{e['fator_a_025']['media']:+.4f}    "
            f"{e['interacao_frac_x_fator']['media']:+.4f} "
            f"[{e['interacao_frac_x_fator']['ic95'][0]:+.4f}, "
            f"{e['interacao_frac_x_fator']['ic95'][1]:+.4f}]")

    # ── additional contrasts recorded in the artifact (exploratory)
    dif_x4 = par_de_dif(ex["gatv2"]["frac_a_x4"], ex["mlp"]["frac_a_x4"])
    dif_inter = par_de_dif(ex["gatv2"]["interacao_frac_x_fator"],
                           ex["mlp"]["interacao_frac_x_fator"])
    intracel = {rot(fr, ft): par(cel[(fr, ft)]["gatv2"], cel[(fr, ft)]["mlp"])
                for (fr, ft) in CELULAS}
    trunc = {f"{rot(fr, ft)} {br}": truncamento(cel[(fr, ft)][br])
             for (fr, ft) in CELULAS for br in ("gatv2", "mlp")}
    pq = por_quadrante_dif(cel[(0.25, 1)]["gatv2"], cel[(1.0, 1)]["gatv2"],
                           cel[(0.25, 1)]["mlp"], cel[(1.0, 1)]["mlp"])
    log(f"\n  diferenca das degradacoes a x4 (GATv2-MLP): {dif_x4['media']:+.5f} "
        f"IC95 [{dif_x4['ic95'][0]:+.5f}, {dif_x4['ic95'][1]:+.5f}] "
        f"({dif_x4['sinal_consistente']}/{dif_x4['n']})")
    log(f"  diferenca das interacoes entre bracos:    {dif_inter['media']:+.5f} "
        f"IC95 [{dif_inter['ic95'][0]:+.5f}, {dif_inter['ic95'][1]:+.5f}] "
        f"(p {dif_inter['p']:.3f})")
    log("  ordenacao intracelula GATv2-MLP:")
    for k, v in intracel.items():
        log(f"    {k:12s} {v['media']:+.5f} [{v['ic95'][0]:+.5f}, "
            f"{v['ic95'][1]:+.5f}] ({v['sinal_consistente']}/{v['n']})")
    log("  vinculo do teto (melhor==teto / etapas; melhor/teto medio):")
    for k, v in trunc.items():
        log(f"    {k:18s} {v['etapas_com_melhor_no_teto']:>2d}/{v['etapas_total']}"
            f"   {v['razao_melhor_sobre_teto_media']:.3f}")
    log("  diferencial por quadrante a x1 (GATv2-MLP):  " + "  ".join(
        f"{q} {pq[q]['dif_gatv2_menos_mlp']:+.4f} ({pq[q]['sinal_consistente']}/5)"
        for q in QUADS))

    # ── admissibility reported alongside
    adm = {}
    for (fr, ft), bracos in CELULAS.items():
        for br, arq in bracos.items():
            # SAME seeds as the skill numbers (D1 has 10; admissibility is
            # also restricted to the 5 common ones, otherwise the
            # side-by-side column would not pair up)
            runs = {s: v for s, v in adm_por_corrida(RAIZ / arq).items()
                    if s in SEMENTES}
            if set(runs) != SEMENTES:
                raise RuntimeError(f"{arq}: admissibilidade sem as 5 sementes")
            adm[f"{rot(fr, ft)} {br}"] = adm_resumo(runs)
    log("\n  admissibilidade         viol>1m(med)  max(m)  zero-por-test + medidos")
    for k, r in adm.items():
        log(f"    {k:20s} {r['violacoes_acima_1m_media']:10.1f}  "
            f"{r['violacao_max_m_pior']:6.2f}   "
            f"{r['quadrantes_zero_por_testemunha_total']:>2d} + "
            f"{r['quadrantes_com_medicao_total']:<2d}")

    frases = {
        "DEPENDENCIA_DIFERENCIAL": (
            "sob o mesmo teto nominal de epocas, o grafo depende mais da "
            "supervisao: cortar as ancoras a um quarto degrada o GATv2 alem "
            "da margem e mais do que o MLP (em tres dos quatro quadrantes; "
            "nao no Q3)"),
        "DEGRADACAO_COMUM": (
            "a dependencia diferencial nao se estabelece; o texto reporta as "
            "duas degradacoes e a diferenca com IC, sem afirmar que o grafo "
            "depende mais da supervisao"),
    }
    log(f"\n  frase autorizada: {frases[ver]}")

    PROV.gravar(SAIDA, {
        "criterio_pre_declarado": {
            "fonte": f"{FILA}, cabecalho, secao F2 (arquivo em _fontes)",
            "contraste": "Delta_frac(braco) = skill(0,25, x1) - skill(1,00, x1), "
                         "pareado, 5 sementes",
            "regra": "DEPENDENCIA_DIFERENCIAL se IC95 de Delta_frac(GATv2) "
                     "inteiro < -0,010 E IC95 de [Delta_frac(GATv2) - "
                     "Delta_frac(MLP)] inteiro < 0; senao DEGRADACAO_COMUM",
            "operacionalizacao_fixada_em": "este script, antes de ler o braco "
                                           "MLP do F2 (docstring)",
            "margem": MARGEM},
        "veredito": ver, "condicoes": {"cond1_gatv2_alem_da_margem": cond1,
                                       "cond2_mais_negativo_que_mlp": cond2},
        "frase_autorizada": frases[ver],
        "delta_frac_gatv2_x1": dfg, "delta_frac_mlp_x1": dfm,
        "diferenca_das_degradacoes_gatv2_menos_mlp": dif,
        "em_metros_de_mae": {"delta_frac_gatv2_x1": dfg_m,
                             "delta_frac_mlp_x1": dfm_m,
                             "diferenca_gatv2_menos_mlp": dif_m},
        "exploratorios_2x2_sem_pre_registro": {
            **ex,
            "diferenca_das_degradacoes_a_x4_gatv2_menos_mlp": dif_x4,
            "diferenca_das_interacoes_entre_bracos": dif_inter,
            "ordenacao_intracelula_gatv2_menos_mlp": intracel,
            "diferencial_por_quadrante_a_x1": pq},
        "vinculo_do_teto": trunc,
        "pool_de_ancoras_comum": pool,
        "resumo_por_celula": {f"{rot(fr, ft)} {br}": {
            "skill": desc(d), "mae_m": desc(d, "mae_m")}
            for (fr, ft), bb in cel.items() for br, d in bb.items()},
        "por_semente": {str(s): {f"{rot(fr, ft)} {br}": cel[(fr, ft)][br][s]["skill"]
                                 for (fr, ft) in CELULAS for br in ("gatv2", "mlp")}
                        for s in sorted(SEMENTES)},
        "admissibilidade": adm,
        "guardas_verificadas": {
            "tetos_por_fator": {str(k): v for k, v in TETOS.items()},
            "lambda_efetivo_por_testemunha": LAMBDA,
            "fracao_por_testemunha": f"pedida == fracao; realizada +- {TOL_FRAC}",
            "sementes": sorted(SEMENTES), "n_reserva": N_RESERVA,
            "d1_com_10_sementes": "usa so as 5 comuns (pareamento)",
            "config_era_nova": {f"{k}": v for k, v in
                                {str(k): v for k, v in CONFIG_ERA_NOVA.items()}.items()},
            "criterio_em_fontes": FILA,
            "boost_por_testemunha": "garantias.peso_rotulo.origem contem "
                                    "'ATL08 x4' em todo quadrante",
            "fracao_cheia_por_testemunha": "n_observada (100%) == "
                "n_observada_antes_da_subamostragem (25%) por semente e "
                "quadrante, em todas as celulas e bracos"},
        "ressalvas": [
            "5 sementes, t(4); decisao sobre a media entre sementes",
            "so Delta_frac a x1 e o contraste pre-declarado; o 2x2 e "
            "exploratorio, sem correcao de multiplicidade",
            ("the operationalization of 'beyond -0.010' and 'more negative' (entire "
             "95% CI) was fixed in this script before reading the F2 MLP result, but "
             "was NOT written in the queue header"),
            "baseline_mlp_b4s035.json (era antiga) nao declara peso_suavidade "
            "nem boost no config; lambda e boost conferidos por testemunha de "
            "escopo (0,35; 'ATL08 x4')",
            "a decisao conjunta (cond1 E cond2) e um teste de intersecao-uniao: "
            "nao pede correcao de multiplicidade; a operacionalizacao IC95 e "
            "unilateral a 2,5% e enviesa CONTRA o claim (fisico-mat)",
            "o pareamento da subamostragem entre bracos e dedutivo (mesma "
            "seed/geometria, b2_transferencia.py:1039-1042) e conferido por "
            "contagem (n_observada bit a bit igual); nao ha hash do conjunto "
            "de ancoras mantidas no artefato (eng-dados D1)",
            "`mae_trivial_m` deriva das ancoras de treino e se move entre "
            "celulas (vies medido 3,0e-5 em skill, fisico-mat): declarar em "
            "Methods, nao corrigir",
            "'a orcamento comparavel' significa o mesmo TETO nominal de epocas; "
            "a x1 o GATv2 ainda tem etapas com o otimo colado no teto (ver "
            "vinculo_do_teto) e o MLP nao — a magnitude do diferencial a x4 "
            "cai pela metade (ver diferenca_das_degradacoes_a_x4)",
        ],
        "contra_auditoria": {
            "data": "2026-08-16",
            "pareceres": ["_pareceres_2026-08-16/eng_ia_f2.md",
                          "_pareceres_2026-08-16/eng_dados_f2.md",
                          "_pareceres_2026-08-16/fisico_mat_f2.md"],
            "veredito_dos_tres": "aprovado com correcoes — aplicadas nesta "
                                 "versao 2 (script congelado durante a leitura "
                                 "da v1, sha256 aaee07dc...)"},
    }, fontes_lidas=fontes, script=__file__)
    log(f"  -> {SAIDA.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
