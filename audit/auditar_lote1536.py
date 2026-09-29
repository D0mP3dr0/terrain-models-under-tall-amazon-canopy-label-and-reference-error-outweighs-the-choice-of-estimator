"""Does the GATv2-vs-MLP equivalence
survive outside the adopted batch size?

THE PROBLEM. The manuscript's TOST declares equivalence at batch size
6144. But that batch size was never chosen on merit —
`b2_transferencia.py:164` shows it was sized by available VRAM. An
earlier review measured, in runs already on disk, that reducing
6144 -> 1536 yields +0.0216 skill to GATv2 (t = +13.03) against only
+0.0058 to the MLP: the graph benefits 3.7x more. If this is confirmed at
the adopted training recipe, the sentence "graph message passing adds no
measurable skill" (`jstars/main.tex:65`) is a property of an engineering
choice, not of the problem.

PRE-REGISTRATION (written in `fila_noite_20260814.sh` BEFORE the queue
ran):

  If |Delta(GNN-MLP)| under batch size 1536, at the adopted recipe, 10
  paired seeds, falls OUTSIDE +-0.010 of skill, then the graph's
  non-superiority is a property of the operating point and the text must
  state so. Within +-0.010, the equivalence is demonstrated at TWO batch
  sizes.

The +-0.010 margin is the SAME one used in `auditoria_equivalencia_10s.py`
and has the same declared origin: the skill guard-rail from the smoothing
rule, set before any architecture comparison. It is reused on purpose, so
that the two operating points are judged by the same yardstick.

TEST. The same paired-by-seed TOST as E0 (two one-sided t-tests against
the margin bounds, alpha 0.05), plus the two-sided test that answers the
complementary question: is the graph superior at this batch size? The two
verdicts are reported separately, because they answer different
questions — "fits within the margin" and "differs from zero" are not each
other's negation.

Usage: python auditar_lote1536.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

import proveniencia as PROV

RAIZ = Path(__file__).resolve().parent
SAIDA = RAIZ / "auditar_lote1536.json"
MARGEM_SKILL = 0.010
ALFA = 0.05
BRACOS = {
    "gatv2_2_semobs": "results/noite_lote1536_gatv2_2_semobs.json",
    "mlp_semobs": "results/noite_lote1536_mlp_semobs.json",
}
# reference: the same contrast at the adopted batch size
REF_6144 = RAIZ / "auditoria_equivalencia_10s.json"
# pre-registration source, recorded in _fontes so the criterion is dated
FILA = RAIZ / "fila_noite_20260814.sh"


def log(m=""):
    print(m, flush=True)


def skills(caminho: str):
    """Per-seed reserve-union skill, and the implicit trivial MAE.

    Same extraction logic as `auditoria_equivalencia_10s.py:61-77`, on
    purpose: the contrast is only comparable to the adopted batch size's
    if the metric is read from the same field via the same path.
    """
    p = RAIZ / caminho
    if not p.exists():
        raise FileNotFoundError(f"ausente: {caminho}")
    d = json.loads(p.read_text(encoding="utf-8"))
    sk, triv = {}, {}
    for k, v in (d.get("quadrantes") or d).items():
        if not (isinstance(v, dict) and v.get("reserva_uniao")):
            continue
        ru = v["reserva_uniao"]
        s = v.get("seed")
        if s is None:  # the scope key carries the seed
            import re
            m = re.search(r"seed(\d+)", str(k))
            if not m:
                continue
            s = int(m.group(1))
        s = int(s)
        if s in sk:
            raise RuntimeError(
                f"{caminho}: semente {s} aparece em mais de um escopo — "
                "o mesmo `sorted(...)[:1]` que o registro canonico existe "
                "para impedir. Resolva antes de auditar.")
        sk[s] = float(ru["skill_media_ponderada_por_n"])
        mae = float(ru["mae_media_ponderada_por_n_m"])
        triv[s] = mae / max(1.0 - sk[s], 1e-9)
    if not sk:
        raise RuntimeError(f"{caminho}: nenhum escopo com reserva_uniao")
    return sk, triv


def main() -> int:
    # Pointing both branches at the SAME file would give delta 0 and
    # "EQUIVALENT" with no warning at all — the most silent failure mode
    # this script has.
    if BRACOS["gatv2_2_semobs"] == BRACOS["mlp_semobs"]:
        raise RuntimeError("os dois bracos apontam para o mesmo arquivo")
    g, tg = skills(BRACOS["gatv2_2_semobs"])
    m, tm = skills(BRACOS["mlp_semobs"])
    comuns = sorted(set(g) & set(m))
    if not comuns:
        log("  ABORTADO: nenhuma semente em comum"); return 2
    log(f"  sementes pareadas: {len(comuns)}  {comuns}")
    if len(comuns) < len(g) or len(comuns) < len(m):
        log(f"  ATENCAO: GATv2 tem {len(g)}, MLP tem {len(m)} — "
            f"o teste usa so as {len(comuns)} em comum")

    d = np.array([g[s] - m[s] for s in comuns], dtype=float)
    n = len(d)
    md, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    gl = n - 1
    tcrit = stats.t.ppf(1 - ALFA / 2, gl)      # two-sided: for the 95% CI
    tcrit_uni = stats.t.ppf(1 - ALFA, gl)      # one-sided: for the TOST resolution
    ic = (md - tcrit * se, md + tcrit * se)

    # TOST: the two one-sided tests against the margin bounds
    t_inf = (md + MARGEM_SKILL) / se
    t_sup = (md - MARGEM_SKILL) / se
    p_inf = 1 - stats.t.cdf(t_inf, gl)     # H0: delta <= -margin
    p_sup = stats.t.cdf(t_sup, gl)         # H0: delta >= +margin
    equivalentes = bool(p_inf < ALFA and p_sup < ALFA)

    # two-sided: does the graph differ from zero at this batch size?
    t_bi, p_bi = stats.ttest_1samp(d, 0.0)
    difere = bool(p_bi < ALFA)

    # ACHIEVED RESOLUTION — ONE-SIDED t. An earlier version used the
    # two-sided t, and the symptom was visible: `margem_minima` came out
    # IDENTICAL to the top of the 95% CI, digit for digit, because with
    # md > 0 and a two-sided tcrit, `|md| + tcrit*se` IS the top of the CI.
    # The TOST is composed of two one-sided tests, so the smallest margin
    # that would pass uses `t.ppf(1-ALFA, gl)` — which is what
    # `auditoria_equivalencia_10s.py:100` does for the adopted batch size.
    # Without this, the two operating points would be reported side by
    # side with DIFFERENT yardsticks.
    margem_minima = abs(md) + tcrit_uni * se
    triv_medio = float(np.mean([tg[s] for s in comuns] + [tm[s] for s in comuns]))

    log(f"  skill GATv2 : {np.mean([g[s] for s in comuns]):+.4f}")
    log(f"  skill MLP   : {np.mean([m[s] for s in comuns]):+.4f}")
    log(f"  delta pareado: {md:+.5f} +- {se:.5f} (dp {sd:.5f}, n {n})")
    log(f"  IC95 do delta: [{ic[0]:+.5f}, {ic[1]:+.5f}]")
    log(f"  TOST +-{MARGEM_SKILL}: p_inf {p_inf:.3g} | p_sup {p_sup:.3g} "
        f"-> {'EQUIVALENTES' if equivalentes else 'NAO equivalentes'}")
    log(f"  bilateral vs 0: t {t_bi:+.3f} p {p_bi:.3g} "
        f"-> {'DIFERE de zero' if difere else 'nao difere de zero'}")
    log(f"  menor margem que passaria: +-{margem_minima:.5f} de skill "
        f"({margem_minima * triv_medio:+.3f} m)")

    # contrast with the adopted batch size, if the E0 auditor output is on disk
    # WRONG KEY, AND THE GUARD WAS HIDING IT. An earlier version read
    # `r.get("equivalencia")`, a key that does NOT EXIST in
    # `auditoria_equivalencia_10s.json` — the adopted batch size's TOST
    # lives in `skill.delta_media` and `skill.tost.equivalente`. The
    # `or {}` returned an empty dict, both fields came out `null`, and the
    # `if ... is not None` swallowed the error silently, with no log and
    # no exception. The JSON recorded `delta_6144: null` WHILE listing the
    # file in `_fontes` with its sha256 — giving the appearance that the
    # contrast between the two operating points had been computed. It had
    # not, and the entire pre-registration depends on it. This now fails
    # loud.
    ref = None
    if REF_6144.exists():
        r = json.loads(REF_6144.read_text(encoding="utf-8"))
        sk_ref = r.get("skill") or {}
        tost_ref = sk_ref.get("tost") or {}
        if sk_ref.get("delta_media") is None or tost_ref.get("equivalente") is None:
            raise RuntimeError(
                f"{REF_6144.name}: nao achei `skill.delta_media` e "
                f"`skill.tost.equivalente`. O contraste entre os dois pontos de "
                f"operacao e o proprio objeto do pre-registro; sem ele o juizo "
                f"nao pode ser gravado. Chaves de `skill`: {list(sk_ref)}")
        ref = {"arquivo": REF_6144.name,
               "delta_lote_adotado": float(sk_ref["delta_media"]),
               "equivalentes_lote_adotado": bool(tost_ref["equivalente"]),
               "menor_margem_que_passa_lote_adotado":
                   sk_ref.get("menor_margem_que_passa")}
        log(f"  no lote adotado: delta {ref['delta_lote_adotado']:+.5f}, "
            f"equivalentes={ref['equivalentes_lote_adotado']}, "
            f"menor margem {ref['menor_margem_que_passa_lote_adotado']}")

    # verdict against the pre-declared criterion
    fora = not equivalentes
    if fora and difere and md > 0:
        veredito = (
            f"O PONTO DE OPERACAO DECIDE. Sob lote 1536 o GATv2 supera o MLP em "
            f"{md:+.4f} de skill (IC95 [{ic[0]:+.4f}, {ic[1]:+.4f}], p {p_bi:.2g}), "
            f"fora da margem de +-{MARGEM_SKILL}. A nao-superioridade do grafo "
            f"relatada no lote 6144 e propriedade da escolha de lote, nao do "
            f"problema, e o texto tem de declarar isso.")
    elif fora and difere and md < 0:
        veredito = (
            f"O PONTO DE OPERACAO DECIDE, A FAVOR DO MLP. Sob lote 1536 o MLP "
            f"supera o GATv2 em {-md:+.4f} de skill, fora da margem.")
    elif fora:
        veredito = (
            f"INDETERMINADO. O IC95 [{ic[0]:+.4f}, {ic[1]:+.4f}] nao cabe na "
            f"margem de +-{MARGEM_SKILL} e tambem nao exclui o zero: nao ha "
            f"potencia para decidir neste lote. Mais sementes ou margem maior.")
    else:
        veredito = (
            f"EQUIVALENTES TAMBEM NO LOTE 1536 (delta {md:+.5f}, IC95 "
            f"[{ic[0]:+.4f}, {ic[1]:+.4f}]). A equivalencia passa a estar "
            f"demonstrada em DOIS pontos de operacao — mais forte que hoje.")
    log(f"\n  VEREDITO: {veredito}")

    PROV.gravar(SAIDA, {
        "pre_registro": {
            "origem": "fila_noite_20260814.sh, escrito antes de a fila rodar",
            "criterio": "|delta| fora de +-0.010 de skill => a nao-superioridade "
                        "do grafo e propriedade do ponto de operacao",
            "margem_skill": MARGEM_SKILL, "alfa": ALFA,
            "fonte_da_margem": "guarda-corpo de skill da regra da suavidade, "
                               "anterior a qualquer comparacao de arquitetura; "
                               "a mesma de auditoria_equivalencia_10s.py",
            "lote": 1536, "lote_adotado": 6144,
        },
        "sementes": comuns,
        "skill": {"gatv2": {str(s): g[s] for s in comuns},
                  "mlp": {str(s): m[s] for s in comuns}},
        "delta": {"media": md, "dp": sd, "se": se, "n": n,
                  "ic95": list(ic), "gl": gl},
        "tost": {"t_inf": float(t_inf), "p_inf": float(p_inf),
                 "t_sup": float(t_sup), "p_sup": float(p_sup),
                 "equivalentes": equivalentes},
        "bilateral": {"t": float(t_bi), "p": float(p_bi), "difere_de_zero": difere},
        "resolucao_alcancada": {"margem_minima_skill": margem_minima,
                                "em_metros": margem_minima * triv_medio,
                                "mae_trivial_medio_m": triv_medio},
        "referencia_lote_adotado": ref,
        "veredito": veredito,
    }, fontes_lidas=[RAIZ / v for v in BRACOS.values()] +
       ([REF_6144] if REF_6144.exists() else []) +
       # the queue file is the ONLY evidence that the criterion was written
       # before running; without its sha256 the pre-registration is an
       # assertion, not proof
       ([FILA] if FILA.exists() else []),
       script=__file__)
    log(f"  -> {SAIDA.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
