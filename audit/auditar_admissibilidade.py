"""Physical admissibility check reported alongside skill metrics.

Under a 4x-larger training-epoch budget, the under-canopy sign constraint
stops holding for the graph branch: the number of violations above 1 m per
run goes from zero (adopted budget) to the hundreds, with the worst case
predicting the ground surface METERS ABOVE the DSM under closed canopy,
while the MLP branch stays at zero violations at any budget. The skill gain
and the admissibility loss track together, because the constraint weight is
a ratio between loss terms, and a ratio between loss terms is not invariant
to the optimization budget.

For each declared operating point, this script reads per-run witness fields
already recorded during training (`G_raiox.{Q}.F_dossel` and
`G_raiox.{Q}.E_admissibilidade`) and aggregates them per run (summing the 4
quadrants for counts, taking the maximum for maxima, and an n-weighted mean
for fractions). It reports per branch: canopy violations above 1 m, the p95
and maximum violation magnitude, wells per km2, and the 99.9th-percentile
slope. Nothing is recomputed from raster data — this is a witness reader,
and therefore cheap and exact.

Usage: python auditar_admissibilidade.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

import proveniencia as PROV

RAIZ = Path(__file__).resolve().parent
SAIDA = RAIZ / "auditar_admissibilidade.json"
QUADS = ["Q1", "Q2", "Q3", "Q4"]
CHAVES_VIOL = ("n_violacoes_dossel", "violacao_p95_m", "violacao_max_m",
               "frac_violacoes_acima_1m", "frac_violacoes_acima_10cm")

PONTOS = {
    "lote_adotado_epocas_x1": {
        "gatv2": "results/ofat_suavfina_b4s035.json",
        "mlp": "results/baseline_mlp_b4s035.json"},
    "lote_1536_epocas_x1": {
        "gatv2": "results/noite_lote1536_gatv2_2_semobs.json",
        "mlp": "results/noite_lote1536_mlp_semobs.json"},
    "lote_adotado_epocas_x4_D1": {
        "gatv2": "results/d1_orcamento_gatv2_2_semobs.json",
        "mlp": "results/d1_orcamento_mlp_semobs.json"},
    "d2_frac025_epocas_x4": {
        "gatv2": "results/d2_frac025_gatv2_2_semobs.json",
        "mlp": "results/d2_frac025_mlp_semobs.json"},
}


def log(m=""):
    print(m, flush=True)


def por_corrida(caminho: Path) -> dict:
    """{seed: aggregated per-run metrics} built from the recorded witness fields."""
    d = json.loads(caminho.read_text(encoding="utf-8"))
    out = {}
    for k, v in d.items():
        if not isinstance(v, dict) or not v.get("G_raiox"):
            continue
        s = v.get("seed")
        if s is None:
            continue
        viol = viol10 = p95s = vmax = 0
        pocos = area = 0.0
        slopes = []
        faltou = []
        zero_por_testemunha = medidos = 0
        for q in QUADS:
            g = (v["G_raiox"] or {}).get(q) or {}
            fd = g.get("F_dossel") or {}
            ea = g.get("E_admissibilidade") or {}
            if not fd or not ea:
                faltou.append(q)
                continue
            # Guard: the error branch of `_delta_sob_dossel` returns
            # {"erro": ...} (a truthy dict), and an unguarded .get(...,0)
            # would report a FAILED measurement as zero violations.
            # `_delta_sob_dossel` (b2_transferencia.py:2875-2909) only writes
            # the violation keys when a dense-canopy cell has a negative
            # Delta — absence IS zero, but only when the zero-witness field
            # is present.
            if "erro" in fd:
                raise RuntimeError(f"{caminho.name}/{k}/{q}: F_dossel gravou "
                                   f"erro ({fd['erro']}) — medicao ausente, "
                                   "NAO e zero")
            faltam = [c for c in CHAVES_VIOL if c not in fd]
            if faltam:
                if (fd.get("frac_neg_no_dossel") != 0.0
                        or fd.get("termo_da_perda_m") != 0.0):
                    raise RuntimeError(
                        f"{caminho.name}/{k}/{q}: F_dossel sem {faltam} e sem "
                        f"a testemunha de zero (frac_neg_no_dossel="
                        f"{fd.get('frac_neg_no_dossel')!r}, termo_da_perda_m="
                        f"{fd.get('termo_da_perda_m')!r})")
                nv, fr, fr10, p95, vmx = 0, 0.0, 0.0, 0.0, 0.0
                zero_por_testemunha += 1
            else:
                nv = int(fd["n_violacoes_dossel"])
                fr = float(fd["frac_violacoes_acima_1m"])
                fr10 = float(fd["frac_violacoes_acima_10cm"])
                p95 = float(fd["violacao_p95_m"])
                vmx = float(fd["violacao_max_m"])
                medidos += 1
            # "violation > 1 m" = n_violacoes * frac_acima_1m, both recorded fields
            viol += int(round(nv * fr))
            viol10 += int(round(nv * fr10))
            p95s = max(p95s, p95)
            vmax = max(vmax, vmx)
            # Strict indexing: these three fields feed citable outputs
            # (pocos_por_km2, slope_p99_9); a missing key fails loudly
            # instead of silently defaulting.
            for c in ("pocos", "area_km2", "slope_p99_9_graus"):
                if c not in ea:
                    raise RuntimeError(f"{caminho.name}/{k}/{q}: "
                                       f"E_admissibilidade sem '{c}'")
            pocos += float(ea["pocos"])
            area += float(ea["area_km2"])
            slopes.append(float(ea["slope_p99_9_graus"]))
        if faltou:
            raise RuntimeError(f"{caminho.name}/{k}: G_raiox sem F_dossel/"
                               f"E_admissibilidade em {faltou} — testemunha "
                               "ausente, o ponto nao pode entrar na tabela")
        out[int(s)] = {
            "violacoes_dossel_acima_1m": viol,
            "violacoes_dossel_acima_10cm": viol10,
            "quadrantes_zero_por_testemunha": zero_por_testemunha,
            "quadrantes_com_medicao": medidos,
            # Intra-quadrant p95 over the violating cells, aggregated per
            # run by the MAXIMUM across quadrants — this is NOT the pooled
            # p95 of the run.
            "violacao_p95_m_max_entre_quadrantes": p95s, "violacao_max_m": vmax,
            "pocos_por_km2": pocos / area if area else float("nan"),
            "slope_p99_9_graus": float(np.nanmax(slopes))}
    if not out:
        raise RuntimeError(f"{caminho.name}: nenhum escopo com G_raiox")
    return out


def resumo(runs: dict) -> dict:
    ks = sorted(runs)
    col = lambda c: [runs[s][c] for s in ks]
    v = col("violacoes_dossel_acima_1m")
    return {
        "n_sementes": len(ks), "sementes": ks,
        "violacoes_acima_1m_media": float(np.mean(v)),
        "violacoes_acima_1m_min": int(min(v)),
        "violacoes_acima_1m_max": int(max(v)),
        "sementes_com_zero_violacoes": int(sum(1 for x in v if x == 0)),
        "quadrantes_zero_por_testemunha_total": int(sum(
            col("quadrantes_zero_por_testemunha"))),
        "quadrantes_com_medicao_total": int(sum(col("quadrantes_com_medicao"))),
        "violacoes_acima_10cm_media": float(np.mean(col("violacoes_dossel_acima_10cm"))),
        "violacao_p95_m_max_entre_quadrantes_media": float(
            np.mean(col("violacao_p95_m_max_entre_quadrantes"))),
        "nota_p95": ("p95 intraquadrante sobre as celulas violadoras; o "
                     "agregado por corrida e o MAXIMO entre quadrantes, nao o "
                     "p95 agrupado — para o p95 agrupado seria preciso o vetor "
                     "de violacoes, nao gravado"),
        "violacao_max_m_pior": float(max(col("violacao_max_m"))),
        "pocos_por_km2_media": float(np.mean(col("pocos_por_km2"))),
        "slope_p99_9_graus_pior": float(max(col("slope_p99_9_graus")))}


def main() -> int:
    fontes, tabela, testes = [], {}, {}
    for ponto, bracos in PONTOS.items():
        tabela[ponto] = {}
        for braco, arq in bracos.items():
            p = RAIZ / arq
            if not p.exists():
                raise RuntimeError(f"ausente: {arq}")
            fontes.append(p)
            tabela[ponto][braco] = resumo(por_corrida(p))
    # Paired test of the jump from the adopted operating point to D1 in the graph branch.
    ga = por_corrida(RAIZ / PONTOS["lote_adotado_epocas_x1"]["gatv2"])
    gd = por_corrida(RAIZ / PONTOS["lote_adotado_epocas_x4_D1"]["gatv2"])
    comuns = sorted(set(ga) & set(gd))
    dif = np.array([gd[s]["violacoes_dossel_acima_1m"]
                    - ga[s]["violacoes_dossel_acima_1m"] for s in comuns])
    if len(dif) > 1 and dif.std(ddof=1) > 0:
        t, pv = stats.ttest_1samp(dif, 0.0)
    else:
        t, pv = float("nan"), float("nan")
    testes["gatv2_adotado_vs_D1"] = {
        "n": len(comuns), "dif_media": float(dif.mean()),
        "sinais_positivos": int((dif > 0).sum()), "t": float(t), "p": float(pv)}

    log("  ponto de operacao            braco   viol>1m(med)  p95max(m) max(m)")
    for ponto, bb in tabela.items():
        for braco, r in bb.items():
            log(f"  {ponto:28s} {braco:6s} {r['violacoes_acima_1m_media']:11.1f}"
                f"  {r['violacao_p95_m_max_entre_quadrantes_media']:6.3f}"
                f"  {r['violacao_max_m_pior']:6.2f}")
    tt = testes["gatv2_adotado_vs_D1"]
    log(f"\n  salto adotado->D1 no grafo: +{tt['dif_media']:.0f} violacoes/corrida "
        f"({tt['sinais_positivos']}/{tt['n']} sementes, p {tt['p']:.2g})")

    PROV.gravar(SAIDA, {
        "regra": ("no skill number of an operating point enters a table without the "
                  "admissibility column beside it"),
        "definicoes": {
            "violacoes_dossel_acima_1m": "n_violacoes_dossel * "
                "frac_violacoes_acima_1m, somado nos 4 quadrantes; campos "
                "gravados pelo proprio treino em G_raiox.F_dossel",
            "agregacao": "soma p/ contagens, max p/ maximos, media entre "
                "sementes no resumo"},
        "tabela": tabela, "testes": testes,
    }, fontes_lidas=fontes, script=__file__)
    log(f"  -> {SAIDA.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
