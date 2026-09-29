"""F1 — attributing the D1-1536 surplus (phase RAMP scaled together).

Motivation. D1 (4x epoch factor, phase RAMP NOT scaled) produced
GATv2-MLP = +0.0180, EXCEEDING the 1536-batch Delta (+0.0128) by +0.0052
(p 0.011; `auditar_orcamento_d1.json`, attribution left open). Two
candidates: the phase ramp (which in D1 stayed at 10/15/20/25/30 epochs
while the caps quadrupled) and the batch size. F1 reruns D1 with the ramp
SCALED (`--fator-fases 4`: transitions at 40/60/80/100/120), 10 seeds, two
branches.

PRE-REGISTERED CRITERION (transcribed from the header of
`fila_2026-08-16_d8_f2_f1.sh`, section F1, written BEFORE running):
  Delta_F1 = skill(GATv2) - skill(MLP) in this branch.
  * 95% CI of Delta_F1 containing +0.0128 (the 1536 Delta) -> the surplus
    was the ramp; the batch-size axis loses support and the text does not
    mention batch size.
  * Delta_F1 >= +0.0170 -> the surplus was NOT the ramp; a large batch at
    matched steps becomes a result of its own and F1 earns a follow-up.
  In both cases, a conservative per-quadrant reading is reported alongside.
  Operationalization (fixed here, before reading the numbers): the two
  rules are evaluated LITERALLY and independently; if both are true or
  both are false the verdict is UNDETERMINED and the text reports both
  numbers without attribution. "+0.0128" is read from
  `auditar_lote1536.json` (delta.media) and re-checked against the raw
  1536 artifacts using the same aggregation.

PRE-REGISTERED ALERT: the ramp only acts in stage 1, and the MLP's
degradation under budget is dominated by Q2 — F1 as designed could confuse
the ramp with differential forgetting. For this reason the script also
reports F1-D1 per branch (the ramp's effect on each network) and the
per-quadrant differential.

WITNESSES (fail loud): F1 config with fator_epocas 4, fator_fases 4,
peso_suavidade 0.35, boost 4, frac 1.0, batch None; caps 180/100/72/56;
effective lambda 0.35 (max of pesos.suave); phase transitions read from
`calibragem_por_epoca[*].fase` equal to 4x those of D1 in the same seed
and quadrant; boost via `peso_rotulo.origem`; `sem_canal_observacao`;
n_total 157,847; 10 identical seeds across F1, D1 and 1536; `etapas` and
`quadrantes.treino` agree.

Usage: python auditar_f1_rampa.py
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
SAIDA = RAIZ / "auditar_f1_rampa.json"
FILA = "fila_2026-08-16_d8_f2_f1.sh"
REF_1536 = "auditar_lote1536.json"
QUADS = ["Q1", "Q2", "Q3", "Q4"]
SEMENTES = {42, 123, 7, 2024, 31, 100, 200, 300, 400, 500}
N_RESERVA = 157847
LAMBDA = 0.35
TETOS_X4 = {"etapa1": 180, "etapa2": 100, "etapa3": 72, "etapa4": 56}
LIMIAR_CAPACIDADE = 0.0170

ARQ = {
    "f1": {"gatv2": "results/f1_rampa_gatv2_2_semobs.json",
           "mlp": "results/f1_rampa_mlp_semobs.json"},
    "d1": {"gatv2": "results/d1_orcamento_gatv2_2_semobs.json",
           "mlp": "results/d1_orcamento_mlp_semobs.json"},
    "l1536": {"gatv2": "results/noite_lote1536_gatv2_2_semobs.json",
              "mlp": "results/noite_lote1536_mlp_semobs.json"},
}
CONFIG = {
    "f1": {"fator_epocas": 4, "fator_fases": 4, "peso_suavidade": LAMBDA,
           "boost_atl08": 4.0, "frac_ancoras_treino": 1.0, "batch": None},
    "d1": {"fator_epocas": 4, "peso_suavidade": LAMBDA, "boost_atl08": 4.0,
           "frac_ancoras_treino": 1.0, "batch": None},
    "l1536": {"batch": 1536},
}
# The 1536 branch is from an earlier era: these fields may be ABSENT or
# hold the neutral value; any value other than the neutral one is an error.
ESPERADO_1536 = {"fator_epocas": (None, 1), "fator_fases": (None, 1),
                 "frac_ancoras_treino": (None, 1.0)}
# The D1-1536 surplus is CALCULATED in main() via par_de_dif(dD1, d1536) on
# the same 10 seeds (not hardcoded) and checked against
# auditar_orcamento_d1.json when that field exists.
TETOS = {"f1": TETOS_X4, "d1": TETOS_X4,
         "l1536": {"etapa1": 45, "etapa2": 25, "etapa3": 18, "etapa4": 14}}


FRASE_CONSERVADORA = ("o contraste direto Delta_F1 - Delta_1536 nao distingue "
                      "de zero e o desenho nao tem poder para o excedente: o "
                      "texto NAO atribui o excedente (nem a rampa, nem a lote) "
                      "e nao menciona lote como resultado; a leitura por "
                      "quadrante e parte do criterio")


def log(m=""):
    print(m, flush=True)


def lambda_efetivo(escopo: dict) -> float:
    vals = []
    for q in QUADS:
        cal = (escopo["quadrantes"][q].get("treino") or {}).get(
            "calibragem_por_epoca") or []
        if not cal:
            raise RuntimeError(f"{q}: sem calibragem_por_epoca")
        vals += [float(e["pesos"]["suave"]) for e in cal]
    return max(vals)


def transicoes(escopo: dict) -> dict:
    """{Q: [epoch each phase starts at]} read from calibragem_por_epoca. The
    `fase` key is REQUIRED in every epoch: without this check, a missing key
    could silently pass as "40 pairs checked"."""
    out = {}
    for q in QUADS:
        cal = escopo["quadrantes"][q]["treino"]["calibragem_por_epoca"]
        if not cal:
            raise RuntimeError(f"{q}: calibragem_por_epoca vazia")
        tr, last = [], object()
        for c in cal:
            if "fase" not in c:
                raise RuntimeError(f"{q}: epoca {c.get('epoca')} sem a chave "
                                   "'fase' — testemunha da rampa ausente")
            if c["fase"] != last:
                tr.append((int(c["epoca"]), c["fase"]))
                last = c["fase"]
        out[q] = tr
    return out


def primeira_epoca_com_peso(escopo: dict) -> dict:
    """NUMERIC witness of the ramp: the first epoch at which any physical
    weight is > 0, per quadrant. Only stage 1 (Q1) has a ramp
    (b2_transferencia.py:617-619); in Q2-Q4 it is 0 in both designs."""
    out = {}
    for q in QUADS:
        cal = escopo["quadrantes"][q]["treino"]["calibragem_por_epoca"]
        pos = [int(c["epoca"]) for c in cal
               if any(float(v) > 0 for v in (c.get("pesos") or {}).values())]
        out[q] = min(pos) if pos else None
    return out


def fase_da_melhor_epoca(escopo: dict) -> dict:
    """SELECTION point per quadrant: the phase in which `melhor_epoca` fell,
    and whether it precedes the first epoch with a physical weight — scaling
    the ramp without scaling patience can move selection into the
    curriculum asymmetrically between branches."""
    out = {}
    for q in QUADS:
        tre = escopo["quadrantes"][q]["treino"]
        cal = tre["calibragem_por_epoca"]
        me = int(tre["melhor_epoca"])          # 1-based (curva_val_mae argmin+1)
        # `calibragem_por_epoca[*].epoca` is 0-based: the best epoch (1-based)
        # is the record at index me-1 — without this correction, seed 42 of
        # F1 MLP Q1 fell on 'solo_dossel' and 5/80 pairs silently became None.
        por_idx = {int(c["epoca"]): c for c in cal}
        if (me - 1) not in por_idx:
            raise RuntimeError(f"{q}: melhor_epoca {me} (base 1) sem registro "
                               f"de calibragem no indice {me - 1}")
        fase = por_idx[me - 1]["fase"]
        prim = primeira_epoca_com_peso(escopo)[q]      # base 0
        out[q] = {"melhor_epoca_base1": me, "fase": fase,
                  "antes_da_fisica": bool(prim is not None and (me - 1) < prim)}
    return out


def ler(chave: str, braco: str) -> dict:
    p = RAIZ / ARQ[chave][braco]
    d = json.loads(p.read_text(encoding="utf-8"))
    cfg = d.get("config") or {}
    for k, v in CONFIG[chave].items():
        if k not in cfg:
            raise RuntimeError(f"{p.name}: config sem '{k}'")
        if cfg[k] != v:
            raise RuntimeError(f"{p.name}: config.{k}={cfg[k]!r}, esperado {v!r}")
    if chave == "d1" and cfg.get("fator_fases") not in (None, 1):
        raise RuntimeError(f"{p.name}: D1 com fator_fases {cfg.get('fator_fases')}")
    if chave == "l1536":
        for k, ok in ESPERADO_1536.items():
            if cfg.get(k) not in ok:
                raise RuntimeError(f"{p.name}: braco 1536 declara {k}="
                                   f"{cfg.get(k)!r}, esperado {ok} — nao e o "
                                   "ponto de referencia")
    out = {}
    for k, v in d.items():
        if not (isinstance(v, dict) and v.get("reserva_uniao")):
            continue
        s = int(v["seed"])
        ru = v["reserva_uniao"]
        if not bool(v["ablacao"]["sem_canal_observacao"]):
            raise RuntimeError(f"{p.name}/{k}: canal de observacao ligado")
        if int(ru["n_total"]) != N_RESERVA:
            raise RuntimeError(f"{p.name}/{k}: n_total {ru['n_total']}")
        if v["epocas"] != TETOS[chave]:
            raise RuntimeError(f"{p.name}/{k}: tetos {v['epocas']}")
        lam = lambda_efetivo(v)
        if abs(lam - LAMBDA) > 1e-9:
            raise RuntimeError(f"{p.name}/{k}: lambda efetivo {lam}")
        for q in QUADS:
            origem = ((v["quadrantes"][q].get("garantias") or {})
                      .get("peso_rotulo") or {}).get("origem")
            if not origem or "ATL08 x4" not in origem:
                raise RuntimeError(f"{p.name}/{k}/{q}: boost nao testemunhado "
                                   f"({origem!r})")
            ped = (v["quadrantes"][q].get("particao") or {}).get(
                "frac_ancoras_treino_pedida")
            if ped not in (None, 1.0):
                raise RuntimeError(f"{p.name}/{k}/{q}: fracao pedida {ped}")
        tre = {q: v["quadrantes"][q]["treino"] for q in QUADS}
        etapas = v["etapas"]
        for e in etapas:
            tq = tre[e["entrou"]]
            if (tq["epocas_usadas"] != e["epocas_usadas"]
                    or tq["epocas_teto"] != e["epocas_teto"]):
                raise RuntimeError(f"{p.name}/{k}: etapas x treino divergem")
        if s in out:
            raise RuntimeError(f"{p.name}: semente {s} duplicada")
        out[s] = {
            "skill": float(ru["skill_media_ponderada_por_n"]),
            "mae_m": float(ru["mae_media_ponderada_por_n_m"]),
            "skill_por_quadrante": {
                q: float(v["quadrantes"][q]["reserva_final"]
                         ["comparabilidade"]["skill"]) for q in QUADS},
            "transicoes": transicoes(v),
            "primeira_epoca_com_peso": primeira_epoca_com_peso(v),
            "fase_da_melhor_epoca": fase_da_melhor_epoca(v),
            "etapas_com_melhor_no_teto": int(sum(
                1 for e in etapas
                if tre[e["entrou"]]["melhor_epoca"] >= e["epocas_teto"])),
            "razao_melhor_sobre_teto": [tre[e["entrou"]]["melhor_epoca"]
                                        / e["epocas_teto"] for e in etapas]}
    if set(out) != SEMENTES:
        raise RuntimeError(f"{p.name}: sementes {sorted(out)} != "
                           f"{sorted(SEMENTES)}")
    return out


def verificar_rampa(f1: dict, d1: dict, rot: str) -> dict:
    """F1's phase transitions must be 4x those of D1, same seed and
    quadrant — witnessing that `--fator-fases 4` actually took effect. Only
    the pairs where a ramp EXISTS (transitions with epoch > 0 in D1) are
    informative: in Q2-Q4 the ramp is null in both designs and the check
    would be a trivial 0 == 4x0. The numeric witness (first epoch with
    physical weight > 0) must also come out at 4x in the same pairs."""
    informativos = tautologicos = 0
    exemplo = None
    for s in sorted(SEMENTES):
        for q in QUADS:
            a = f1[s]["transicoes"][q]
            b = d1[s]["transicoes"][q]
            fa = [x for x in a if x[1] is not None]
            fb = [x for x in b if x[1] is not None]
            if [f for _, f in fa] != [f for _, f in fb]:
                raise RuntimeError(f"{rot}/{s}/{q}: sequencia de fases difere: "
                                   f"{fa} vs {fb}")
            if [e for e, _ in fa] != [4 * e for e, _ in fb]:
                raise RuntimeError(f"{rot}/{s}/{q}: rampa F1 {fa} nao e 4x a do "
                                   f"D1 {fb}")
            tem_rampa = any(e > 0 for e, _ in fb)
            pf = f1[s]["primeira_epoca_com_peso"][q]
            pd = d1[s]["primeira_epoca_com_peso"][q]
            if tem_rampa:
                if pf is None or pd is None or pf != 4 * pd:
                    raise RuntimeError(f"{rot}/{s}/{q}: primeira epoca com peso "
                                       f"F1 {pf} nao e 4x a do D1 {pd}")
                informativos += 1
                if exemplo is None:
                    exemplo = {"f1": fa, "d1": fb, "primeira_epoca_com_peso":
                               {"f1": pf, "d1": pd}}
            else:
                tautologicos += 1
    if informativos != len(SEMENTES):
        raise RuntimeError(f"{rot}: {informativos} pares informativos, esperado "
                           f"{len(SEMENTES)} (um por semente, na etapa 1)")
    return {"pares_informativos": informativos,
            "pares_tautologicos_rampa_nula_nos_dois": tautologicos,
            "exemplo": exemplo}


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


def desc(d: dict, campo: str = "skill") -> dict:
    x = np.array([v[campo] for v in d.values()], float)
    return {"media": float(x.mean()), "dp": float(x.std(ddof=1)),
            "min": float(x.min()), "max": float(x.max()), "n": len(x)}


def por_quadrante(a: dict, b: dict) -> dict:
    out = {}
    for q in QUADS:
        d = np.array([a[s]["skill_por_quadrante"][q] - b[s]["skill_por_quadrante"][q]
                      for s in sorted(SEMENTES)])
        n = len(d); md = float(d.mean()); se = float(d.std(ddof=1)) / math.sqrt(n)
        tc = stats.t.ppf(0.975, n - 1)
        out[q] = {"media": md, "ic95": [md - tc * se, md + tc * se],
                  "sinal_consistente": int(np.sum(np.sign(d) == np.sign(md)))}
    out["quadrantes_com_media_positiva"] = int(sum(1 for q in QUADS if out[q]["media"] > 0))
    return out


def ponto_de_selecao(runs: dict) -> dict:
    ks = sorted(runs)
    out = {}
    for q in QUADS:
        antes = sum(1 for s in ks if runs[s]["fase_da_melhor_epoca"][q]["antes_da_fisica"])
        fases = {}
        for s in ks:
            f = runs[s]["fase_da_melhor_epoca"][q]["fase"]
            fases[f] = fases.get(f, 0) + 1
        out[q] = {"sementes_com_melhor_epoca_antes_da_fisica": int(antes),
                  "n": len(ks), "fase_da_melhor_epoca_contagem": fases}
    return out


def mde_80(se: float, n: int) -> float:
    """Minimum detectable effect at 80% power, two-sided 5%, t(n-1)."""
    return float(se * (stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.80, n - 1)))


def truncamento(runs: dict) -> dict:
    ks = sorted(runs)
    raz = [r for s in ks for r in runs[s]["razao_melhor_sobre_teto"]]
    return {"etapas_com_melhor_no_teto": int(sum(runs[s]["etapas_com_melhor_no_teto"] for s in ks)),
            "etapas_total": 4 * len(ks),
            "razao_melhor_sobre_teto_media": float(np.mean(raz))}


def main() -> int:
    fontes = [RAIZ / FILA, RAIZ / REF_1536]
    R = {ch: {br: ler(ch, br) for br in ("gatv2", "mlp")} for ch in ARQ}
    for ch in ARQ:
        for br in ("gatv2", "mlp"):
            fontes.append(RAIZ / ARQ[ch][br])
    rampa = {br: verificar_rampa(R["f1"][br], R["d1"][br], br) for br in ("gatv2", "mlp")}
    # The reference branch (1536) must have the x1 ramp (same as D1) —
    # positive evidence, not merely the absence of a config field.
    for br in ("gatv2", "mlp"):
        for s in sorted(SEMENTES):
            a = R["l1536"][br][s]; b = R["d1"][br][s]
            if a["transicoes"]["Q1"] != b["transicoes"]["Q1"]:
                raise RuntimeError(f"1536/{br}/{s}: rampa do Q1 {a['transicoes']['Q1']} "
                                   f"difere da do D1 {b['transicoes']['Q1']}")
            if a["primeira_epoca_com_peso"]["Q1"] != b["primeira_epoca_com_peso"]["Q1"]:
                raise RuntimeError(f"1536/{br}/{s}: primeira epoca com peso difere do D1")
    rampa["l1536_igual_a_do_d1"] = {"pares_Q1_conferidos": 2 * len(SEMENTES)}

    # reference +0.0128: read from the audited artifact AND re-checked against the raw data
    ref = json.loads((RAIZ / REF_1536).read_text(encoding="utf-8"))
    ref_1536 = float(ref["delta"]["media"])
    d1536 = par(R["l1536"]["gatv2"], R["l1536"]["mlp"])
    if abs(d1536["media"] - ref_1536) > 1e-9:
        raise RuntimeError(f"Delta 1536 recalculado {d1536['media']} != "
                           f"{ref_1536} de {REF_1536}")

    log("  ponto                     braco   skill medio   dp      MAE(m)")
    for ch, nome in (("l1536", "lote 1536, x1"), ("d1", "D1: x4, rampa x1"),
                     ("f1", "F1: x4, rampa x4")):
        for br in ("gatv2", "mlp"):
            a, mm = desc(R[ch][br]), desc(R[ch][br], "mae_m")
            log(f"  {nome:24s}  {br:6s}  {a['media']:+.4f}   {a['dp']:.4f}  {mm['media']:.4f}")

    dF1 = par(R["f1"]["gatv2"], R["f1"]["mlp"])
    dD1 = par(R["d1"]["gatv2"], R["d1"]["mlp"])
    contem = dF1["ic95"][0] <= ref_1536 <= dF1["ic95"][1]
    acima = dF1["media"] >= LIMIAR_CAPACIDADE
    if contem and not acima:
        ver = "EXCEDENTE_ERA_A_RAMPA"
    elif acima and not contem:
        ver = "EXCEDENTE_NAO_ERA_A_RAMPA"
    else:
        ver = "INDETERMINADO"
    log(f"\n  Delta_F1 = GATv2 - MLP (F1): {dF1['media']:+.5f} IC95 "
        f"[{dF1['ic95'][0]:+.5f}, {dF1['ic95'][1]:+.5f}] "
        f"({dF1['sinal_consistente']}/{dF1['n']}, p {dF1['p']:.2g})")
    log(f"  regra 1: IC95 contem +{ref_1536:.4f} (1536)? {contem}   "
        f"regra 2: Delta_F1 >= +{LIMIAR_CAPACIDADE}? {acima}   -> {ver}")
    log(f"  (D1, para memoria) GATv2 - MLP: {dD1['media']:+.5f} IC95 "
        f"[{dD1['ic95'][0]:+.5f}, {dD1['ic95'][1]:+.5f}]")

    # ramp effect on each network and on the differential (exploratory)
    rG = par(R["f1"]["gatv2"], R["d1"]["gatv2"])
    rM = par(R["f1"]["mlp"], R["d1"]["mlp"])
    rDif = par_de_dif(dF1, dD1)
    log(f"\n  rampa x4 - x1 no GATv2: {rG['media']:+.5f} IC95 [{rG['ic95'][0]:+.5f}, "
        f"{rG['ic95'][1]:+.5f}] ({rG['sinal_consistente']}/{rG['n']})")
    log(f"  rampa x4 - x1 no MLP:   {rM['media']:+.5f} IC95 [{rM['ic95'][0]:+.5f}, "
        f"{rM['ic95'][1]:+.5f}] ({rM['sinal_consistente']}/{rM['n']})")
    log(f"  Delta_F1 - Delta_D1:    {rDif['media']:+.5f} IC95 [{rDif['ic95'][0]:+.5f}, "
        f"{rDif['ic95'][1]:+.5f}] ({rDif['sinal_consistente']}/{rDif['n']})")

    # DIRECT contrast Delta_F1 - Delta_1536, same 10 seeds ("attribution
    # with residual")
    dDir = par_de_dif(dF1, d1536)
    dDir["mde_80pct"] = mde_80(dDir["se"], dDir["n"])
    exced = par_de_dif(dD1, d1536)                     # o excedente, calculado
    EXCEDENTE_D1_1536_REF = float(exced["media"])
    dDir["excedente_d1_1536_calculado"] = {"media": EXCEDENTE_D1_1536_REF,
                                            "ic95": exced["ic95"], "p": exced["p"]}
    # minimum n for powered detection with margin, given the observed sd
    sd = dDir["dp"]
    n_min = None
    for n in range(3, 200):
        if mde_80(sd / math.sqrt(n), n) <= 0.8 * EXCEDENTE_D1_1536_REF:
            n_min = n
            break
    dDir["n_minimo_para_poder_com_folga"] = n_min
    # "Power" only WITH A MARGIN: MDE80 <= 80% of the surplus. Without the
    # margin, an MDE within 2% of the surplus (0.00508 vs 0.00518, this
    # data) would count as "powered" by a razor-thin margin. The 0.8 factor
    # is a choice of THIS version, declared but not pre-registered.
    dDir["poder_com_folga_mde80_le_0p8_excedente"] = bool(
        dDir["mde_80pct"] <= 0.8 * EXCEDENTE_D1_1536_REF)
    cruza_zero = dDir["ic95"][0] < 0 < dDir["ic95"][1]
    if not cruza_zero:
        honesto = "RESIDUO_NAO_NULO"
    elif dDir["poder_com_folga_mde80_le_0p8_excedente"]:
        honesto = "RESIDUO_NULO_COM_PODER"
    else:
        honesto = "SEM_PODER_PARA_ATRIBUICAO_EXCLUSIVA"
    log(f"  contraste direto Delta_F1 - Delta_1536: {dDir['media']:+.5f} IC95 "
        f"[{dDir['ic95'][0]:+.5f}, {dDir['ic95'][1]:+.5f}] (p {dDir['p']:.3f}); "
        f"MDE80 {dDir['mde_80pct']:.5f} vs excedente {EXCEDENTE_D1_1536_REF:.6f} "
        f"(n minimo p/ poder com folga: {n_min})  -> leitura honesta: {honesto}")

    # sensitivity to per-quadrant COMPOSITION: Delta_F1 recomposed with Q3
    # neutralized (Q3 = 0 in the n-weighted mean)
    npq = {}
    for q in QUADS:
        # n per quadrant is identical across branches and seeds (157,847 total)
        npq[q] = None
    d0 = json.loads((RAIZ / ARQ["f1"]["gatv2"]).read_text(encoding="utf-8"))
    for k0, v0 in d0.items():
        if isinstance(v0, dict) and v0.get("reserva_uniao"):
            for q in QUADS:
                npq[q] = int(v0["quadrantes"][q]["reserva_final"]["n"])
            break
    ntot = sum(npq.values())
    def skill_sem_q3(runs, s):
        return sum(runs[s]["skill_por_quadrante"][q] * npq[q]
                   for q in QUADS if q != "Q3") / ntot
    dsem = np.array([skill_sem_q3(R["f1"]["gatv2"], s) - skill_sem_q3(R["f1"]["mlp"], s)
                     for s in sorted(SEMENTES)])
    n = len(dsem); md = float(dsem.mean()); se = float(dsem.std(ddof=1)) / math.sqrt(n)
    tc = stats.t.ppf(0.975, n - 1)
    sens = {"delta_F1_com_Q3_neutro": md, "ic95": [md - tc * se, md + tc * se],
            "regra_1_contem_1536": bool(md - tc * se <= ref_1536 <= md + tc * se),
            "regra_2_acima_de_0p017": bool(md >= LIMIAR_CAPACIDADE),
            "n_por_quadrante": npq,
            "nota": "Q3 contribui negativo; neutraliza-lo inverte o veredito "
                    "literal — o criterio binario e sensivel a composicao"}
    log(f"  sensibilidade a composicao: Delta_F1 com Q3 neutro {md:+.5f} "
        f"[{sens['ic95'][0]:+.5f}, {sens['ic95'][1]:+.5f}]  regra1 {sens['regra_1_contem_1536']} "
        f"regra2 {sens['regra_2_acima_de_0p017']}")

    pq = por_quadrante(R["f1"]["gatv2"], R["f1"]["mlp"])
    sel = {f"{ch} {br}": ponto_de_selecao(R[ch][br]) for ch in ("d1", "f1")
           for br in ("gatv2", "mlp")}
    log("  selecao antes da fisica (sementes/10) por quadrante:")
    for k, v in sel.items():
        log(f"    {k:10s} " + "  ".join(
            f"{q} {v[q]['sementes_com_melhor_epoca_antes_da_fisica']}" for q in QUADS))
    log("  Delta_F1 por quadrante: " + "  ".join(
        f"{q} {pq[q]['media']:+.4f} ({pq[q]['sinal_consistente']}/10)" for q in QUADS)
        + f"   positivos {pq['quadrantes_com_media_positiva']}/4")

    trunc = {f"{ch} {br}": truncamento(R[ch][br]) for ch in ARQ for br in ("gatv2", "mlp")}
    log("  vinculo do teto (melhor==teto/etapas; melhor/teto medio):")
    for k, v in trunc.items():
        log(f"    {k:14s} {v['etapas_com_melhor_no_teto']:>2d}/{v['etapas_total']}   "
            f"{v['razao_melhor_sobre_teto_media']:.3f}")

    adm = {}
    for ch in ("d1", "f1"):
        for br in ("gatv2", "mlp"):
            adm[f"{ch} {br}"] = adm_resumo(adm_por_corrida(RAIZ / ARQ[ch][br]))
    log("  admissibilidade      viol>1m(med)  max(m)  zero-por-test + medidos")
    for k, r in adm.items():
        log(f"    {k:12s} {r['violacoes_acima_1m_media']:10.1f}  {r['violacao_max_m_pior']:6.2f}   "
            f"{r['quadrantes_zero_por_testemunha_total']:>2d} + {r['quadrantes_com_medicao_total']:<2d}")

    frases = {
        "EXCEDENTE_ERA_A_RAMPA": ("o excedente D1-1536 era a rampa; o eixo do lote "
                                  "fica sem suporte e o texto nao menciona lote"),
        "EXCEDENTE_NAO_ERA_A_RAMPA": ("o excedente NAO era a rampa; lote grande a "
                                      "passos casados vira resultado proprio"),
        "INDETERMINADO": ("as duas regras pre-declaradas nao se separam; o texto "
                          "reporta Delta_F1 e Delta_1536 com IC, sem atribuir o "
                          "excedente"),
    }
    log(f"\n  VEREDITO F1 (literal, pre-declarado): {ver} — {frases[ver]}")
    log(f"  FRASE AUTORIZADA (a que o escritor copia): "
        f"{frases[ver] if honesto.startswith('RESIDUO_') else FRASE_CONSERVADORA}")
    log((f"  HONEST READING (independent check): {honesto}; the text does NOT attribute "
         f"the excess to the ramp or to the batch size, and does not mention batch size "
         f"as a result"))

    PROV.gravar(SAIDA, {
        "criterio_pre_declarado": {
            "fonte": f"{FILA}, cabecalho, secao F1 (arquivo em _fontes)",
            "contraste": "Delta_F1 = skill(GATv2) - skill(MLP), pareado, 10 sementes",
            "regra_1": f"IC95 de Delta_F1 contem +{ref_1536:.6f} (Delta do 1536, "
                       f"lido de {REF_1536} e reconferido nos brutos)",
            "regra_2": f"Delta_F1 >= +{LIMIAR_CAPACIDADE}",
            "operacionalizacao": "regras avaliadas literalmente e de forma "
                                 "independente; ambas ou nenhuma -> INDETERMINADO"},
        "veredito": ver, "regra_1_verdadeira": contem, "regra_2_verdadeira": acima,
        "frase_literal_do_criterio": frases[ver],
        "veredito_honesto_pos_contra_auditoria": honesto,
        # the sentence carried to the manuscript: the conservative one, unless the
        # honest reading does not start with RESIDUO_
        "frase_autorizada": (frases[ver] if honesto.startswith("RESIDUO_")
                             else FRASE_CONSERVADORA),
        "sensibilidade_a_composicao": sens,
        "frase_conservadora": FRASE_CONSERVADORA,
        "contraste_direto_delta_F1_menos_delta_1536": dDir,
        "sensibilidade_da_operacionalizacao": {
            "ic95_contem_ambas_as_ancoras": bool(dF1["ic95"][0] <= ref_1536 <= dF1["ic95"][1]
                                              and dF1["ic95"][0] <= LIMIAR_CAPACIDADE <= dF1["ic95"][1]),
            "nota": "regra 1 e de intervalo e regra 2 e de ponto; sob leitura "
                    "simetrica (ambas por intervalo) o veredito seria "
                    "INDETERMINADO (fisico-mat B1, eng-ia risco a)"},
        "por_quadrante_pre_registrado": pq,
        "ponto_de_selecao": sel,
        "delta_F1_gatv2_menos_mlp": dF1,
        "delta_1536_reconferido": d1536,
        "delta_D1_para_memoria": dD1,
        "exploratorios_sem_pre_registro": {
            "rampa_x4_menos_x1_gatv2": rG, "rampa_x4_menos_x1_mlp": rM,
            "delta_F1_menos_delta_D1": rDif},
        "testemunha_da_rampa": rampa,
        "vinculo_do_teto": trunc,
        "admissibilidade": adm,
        "resumo_por_braco": {f"{ch} {br}": {"skill": desc(R[ch][br]),
                                            "mae_m": desc(R[ch][br], "mae_m")}
                             for ch in ARQ for br in ("gatv2", "mlp")},
        "por_semente": {str(s): {f"{ch} {br}": R[ch][br][s]["skill"]
                                 for ch in ARQ for br in ("gatv2", "mlp")}
                        for s in sorted(SEMENTES)},
        "alerta_pre_registrado": (("pre-registered caveat: the ramp acts only in stage 1 and the MLP "
                                   "degradation under budget is dominated by Q2 — F1 may confound the ramp "
                                   "with differential forgetting; read rampa_x4_menos_x1_* and per quadrant")),
        "ressalvas": [
            "so Delta_F1 e as duas regras sao pre-declarados; os demais "
            "contrastes sao exploratorios",
            "a operacionalizacao 'ambas ou nenhuma -> INDETERMINADO' foi fixada "
            "neste script antes de ler os numeros do F1, nao no cabecalho da fila",
            "10 sementes, t(9)",
            "multiplicidade: este JSON publica varios IC95 (Delta_F1, D1, "
            "1536, direto, rampa por braco, F1-D1, 4 quadrantes) sem correcao; "
            "so Delta_F1 e as duas regras sao pre-declarados; Delta_F1-Delta_D1 "
            "(p 0,015) nao passa Bonferroni a 0,0071 (fisico-mat)",
            "a leitura por quadrante E pre-registrada no cabecalho ('leitura "
            "conservadora por quadrante reportada junto') — por isso sai de "
            "'exploratorios' nesta versao",
            ("F1 is not 'D1 with the ramp stretched': scaling the ramp without scaling "
             "patience moves the selection point (see ponto_de_selecao) asymmetrically "
             "between arms"),
            "papel dos escopos no registro_v23.csv: 'revisar' ate "
            "papeis_declarados.csv receber as linhas de D8/F2/F1 — numero "
            "provisorio ate la",
        ],
        "contra_auditoria": {
            "data": "2026-08-17",
            "pareceres": ["_pareceres_2026-08-16/eng_ia_f1.md",
                          "_pareceres_2026-08-16/fisico_mat_f1.md",
                          "_pareceres_2026-08-16/eng_dados_f1.md"],
            "veredito_dos_tres": "aprovado com ressalvas/bloqueantes de metodo "
                                 "— aplicados nesta versao 2 (script congelado "
                                 "durante a leitura da v1, sha256 32950ae8...)"},
    }, fontes_lidas=fontes, script=__file__)
    log(f"  -> {SAIDA.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
