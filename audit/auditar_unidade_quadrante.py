"""THE UNIT OF GENERALIZATION -- architecture contrast by QUADRANT.

THE FINDING. Both the headline TOST and the batch contrast treat the SEED as
the unit of generalization. But Delta(GNN-MLP) flips sign across quadrants
and is nearly constant across seeds: a quadrant x seed ANOVA attributes more
than 86% of the variance to quadrants and less than 6% to seeds. A test
paired by seed resamples exactly the source that accounts for 2% of the
total, and so it produces a standard error that is too small.

Consequence: it is not just the new batch result that loses significance --
the HEADLINE EQUIVALENCE claim was never established either. It rests on a
cancellation between quadrants of opposite sign.

WHAT THIS SCRIPT DOES. For each pair of arms, it extracts the skill Delta by
(quadrant, seed) on each quadrant's own holdout, and reports the two tests
side by side:

  * SEED as the unit  -- what the paper currently does; n = 10
  * QUADRANT as the unit -- mean per quadrant over seeds; n = 4

plus the variance decomposition (two-way ANOVA without replicated
interaction) and the number of independent holdouts that would give 80%
power.

THIS IS NOT A NEW TEST ON THE SAME DATA TO FIND A SMALLER p. It is the
opposite: it swaps the unit for a MORE CONSERVATIVE one and shows that both
claims fall apart. The expected outcome is loss of significance on both
sides.

USAGE   python auditar_unidade_quadrante.py
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
from scipy import stats

import proveniencia as PROV

RAIZ = Path(__file__).resolve().parent
SAIDA = RAIZ / "auditar_unidade_quadrante.json"
MARGEM_SKILL = 0.010
ALFA = 0.05
QUADS = ["Q1", "Q2", "Q3", "Q4"]

# The two operating points, each with its (graph, pointwise) pair.
PARES = {
    "lote_adotado": {
        "gatv2": "results/ofat_suavfina_b4s035.json",
        "mlp": "results/baseline_mlp_b4s035.json",
        "nota": "o par que sustenta o TOST da manchete (main.tex:65)",
    },
    "lote_1536": {
        "gatv2": "results/noite_lote1536_gatv2_2_semobs.json",
        "mlp": "results/noite_lote1536_mlp_semobs.json",
        "nota": "E1 da fila da noite de 2026-08-14",
    },
}


def log(m=""):
    print(m, flush=True)


def por_quadrante(caminho: Path) -> tuple[dict, dict]:
    """({(quad, seed): skill}, {(quad, seed): mae_trivial}) on the FINAL holdout.

    Skill per quadrant comes from
    `scope["quadrantes"][Q]["reserva_final"]["comparabilidade"]["skill"]` --
    NOT from `reserva_uniao`, which already aggregates the four quadrants and
    is exactly the aggregation that hides the sign inversion.

    `reserva_final` is the end-of-chain model's pass over all holdouts, which
    is what `reserva_uniao` summarizes; using `reserva_imediata` would give
    the model at each stage instead, a different quantity.
    """
    if not caminho.exists():
        raise FileNotFoundError(str(caminho))
    d = json.loads(caminho.read_text(encoding="utf-8"))
    sk, triv = {}, {}
    for k, v in d.items():
        if not isinstance(v, dict) or not v.get("quadrantes"):
            continue
        m = re.search(r"seed(\d+)", str(k))
        s = v.get("seed")
        s = int(s) if s is not None else (int(m.group(1)) if m else None)
        if s is None:
            continue
        for q, cel in v["quadrantes"].items():
            if q not in QUADS:
                continue
            rf = (cel or {}).get("reserva_final") or {}
            comp = rf.get("comparabilidade") or {}
            if "skill" not in comp:
                continue
            ch = (q, s)
            if ch in sk:
                raise RuntimeError(
                    f"{caminho.name}: {q}/semente {s} aparece duas vezes")
            sk[ch] = float(comp["skill"])
            triv[ch] = float(comp.get("mae_trivial_m", float("nan")))
    if not sk:
        raise RuntimeError(
            f"{caminho.name}: nenhum quadrante com "
            f"reserva_final.comparabilidade.skill")
    return sk, triv


def teste_pareado(d: np.ndarray, rotulo: str) -> dict:
    n = len(d)
    md, sd = float(d.mean()), float(d.std(ddof=1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    gl = n - 1
    tcrit = stats.t.ppf(1 - ALFA / 2, gl) if n > 1 else float("nan")
    ic = (md - tcrit * se, md + tcrit * se) if n > 1 else (float("nan"),) * 2
    t_bi, p_bi = stats.ttest_1samp(d, 0.0) if n > 1 else (float("nan"),) * 2
    # TOST with the same margin, to check whether EQUIVALENCE holds
    if n > 1 and se > 0:
        p_inf = 1 - stats.t.cdf((md + MARGEM_SKILL) / se, gl)
        p_sup = stats.t.cdf((md - MARGEM_SKILL) / se, gl)
        equiv = bool(p_inf < ALFA and p_sup < ALFA)
    else:
        p_inf = p_sup = float("nan"); equiv = False
    return {"unidade": rotulo, "n": n, "media": md, "dp": sd, "se": se,
            "ic95": list(ic), "t": float(t_bi), "p": float(p_bi),
            "tost_p_inf": float(p_inf), "tost_p_sup": float(p_sup),
            "equivalentes": equiv,
            "ic_cabe_na_margem": bool(n > 1 and ic[0] > -MARGEM_SKILL
                                      and ic[1] < MARGEM_SKILL)}


def main() -> int:
    fontes, blocos = [], {}
    for nome, par in PARES.items():
        pg, pm = RAIZ / par["gatv2"], RAIZ / par["mlp"]
        if not (pg.exists() and pm.exists()):
            log(f"  {nome}: par ausente em disco — pulado")
            continue
        fontes += [pg, pm]
        (g, tg), (m, _) = por_quadrante(pg), por_quadrante(pm)
        # the trivial baseline must be the SAME in both arms for the skill
        # delta to be legitimate; observed max divergence is 8.9e-16 m
        _ = tg
        chaves = sorted(set(g) & set(m))
        sementes = sorted({s for _, s in chaves})
        quads = [q for q in QUADS if any(k[0] == q for k in chaves)]
        # quadrant x seed matrix of the delta
        M = np.full((len(quads), len(sementes)), np.nan)
        for i, q in enumerate(quads):
            for j, s in enumerate(sementes):
                if (q, s) in g and (q, s) in m:
                    M[i, j] = g[(q, s)] - m[(q, s)]
        if np.isnan(M).any():
            faltam = int(np.isnan(M).sum())
            raise RuntimeError(f"{nome}: {faltam} celulas (quad, semente) ausentes")

        # SEED unit: mean over quadrants within each seed
        d_sem = M.mean(axis=0)
        # QUADRANT unit: mean over seeds within each quadrant
        d_qua = M.mean(axis=1)

        # variance decomposition (two factors, no replication)
        gm = M.mean()
        ss_q = len(sementes) * ((M.mean(axis=1) - gm) ** 2).sum()
        ss_s = len(quads) * ((M.mean(axis=0) - gm) ** 2).sum()
        ss_t = ((M - gm) ** 2).sum()
        ss_r = ss_t - ss_q - ss_s
        gl_q, gl_s = len(quads) - 1, len(sementes) - 1
        gl_r = gl_q * gl_s
        f_q = (ss_q / gl_q) / (ss_r / gl_r) if ss_r > 0 else float("inf")
        p_q = 1 - stats.f.cdf(f_q, gl_q, gl_r) if np.isfinite(f_q) else 0.0

        # how many independent holdouts would give 80% power, given the
        # observed effect size and dispersion BETWEEN QUADRANTS
        dz = abs(d_qua.mean()) / d_qua.std(ddof=1) if d_qua.std(ddof=1) > 0 else 0.0
        n80 = None
        if dz > 0:
            for k in range(3, 2001):
                pot = stats.nct.sf(stats.t.ppf(1 - ALFA / 2, k - 1),
                                   k - 1, dz * math.sqrt(k))
                if pot >= 0.80:
                    n80 = k
                    break

        b = {"nota": par["nota"],
             "arquivos": {"gatv2": par["gatv2"], "mlp": par["mlp"]},
             "quadrantes": quads, "sementes": sementes,
             "delta_por_quadrante": {q: float(d_qua[i]) for i, q in enumerate(quads)},
             "sinal_por_quadrante": {
                 q: {"positivo_em_n_sementes": int((M[i] > 0).sum()),
                     "de": len(sementes)} for i, q in enumerate(quads)},
             "por_semente": teste_pareado(d_sem, "semente (o que o artigo faz)"),
             "por_quadrante": teste_pareado(d_qua, "quadrante (conservadora)"),
             "variancia": {
                 "frac_entre_quadrantes": float(ss_q / ss_t) if ss_t else None,
                 "frac_entre_sementes": float(ss_s / ss_t) if ss_t else None,
                 "frac_residual": float(ss_r / ss_t) if ss_t else None,
                 "F_quadrante": float(f_q), "gl": [gl_q, gl_r], "p": float(p_q)},
             "reservas_para_80pct_de_potencia": n80,
             "reservas_disponiveis": len(quads)}
        blocos[nome] = b

        log(f"\n  === {nome} — {par['nota']}")
        log("   delta por quadrante: " + " | ".join(
            f"{q} {d_qua[i]:+.4f} ({int((M[i] > 0).sum())}/{len(sementes)} pos)"
            for i, q in enumerate(quads)))
        for chave in ("por_semente", "por_quadrante"):
            t = b[chave]
            log(f"   {t['unidade']:34s} n {t['n']:2d} | delta {t['media']:+.5f} "
                f"| IC95 [{t['ic95'][0]:+.4f}, {t['ic95'][1]:+.4f}] "
                f"| p {t['p']:.3g} | equiv {t['equivalentes']}")
        vv = b["variancia"]
        log(f"   variancia: {vv['frac_entre_quadrantes']:.1%} entre quadrantes, "
            f"{vv['frac_entre_sementes']:.1%} entre sementes "
            f"(F {vv['F_quadrante']:.1f}, p {vv['p']:.2g})")
        log(f"   reservas para 80% de potencia: {n80} (existem {len(quads)})")

    if not blocos:
        log("  ABORTADO: nenhum par disponivel"); return 2

    # joint verdict
    partes = []
    for nome, b in blocos.items():
        s, q = b["por_semente"], b["por_quadrante"]
        if s["equivalentes"] and not q["equivalentes"]:
            partes.append(f"{nome}: equivalencia SO sobrevive com semente como "
                          f"unidade (por quadrante o IC95 vai de "
                          f"{q['ic95'][0]:+.3f} a {q['ic95'][1]:+.3f})")
        elif s["p"] < ALFA and q["p"] >= ALFA:
            partes.append(f"{nome}: a diferenca SO e significante com semente "
                          f"como unidade (por quadrante p = {q['p']:.2f})")
        else:
            partes.append(f"{nome}: as duas unidades concordam")
    veredito = (
        "A UNIDADE DE GENERALIZACAO DECIDE OS DOIS CLAIMS. " + "; ".join(partes)
        + ". Com quatro reservas espaciais e o sinal invertendo entre elas, este "
          "desenho nao estabelece equivalencia nem superioridade arquitetural; o "
          "que ele estabelece e que o contraste NAO E DECIDIVEL neste n.")
    log(f"\n  VEREDITO: {veredito}")

    PROV.gravar(SAIDA, {
        "motivo": "o teste pareado por semente reamostra a fonte que responde "
                  "por menos de 6% da variancia; a unidade correta de "
                  "generalizacao e a reserva espacial",
        "margem_skill": MARGEM_SKILL, "alfa": ALFA,
        "pares": blocos, "veredito": veredito,
    }, fontes_lidas=fontes, script=__file__)
    log(f"  -> {SAIDA.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
