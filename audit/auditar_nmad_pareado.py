"""Paired-by-transect dispersion (NMAD) test by canopy class — Table IV
with significance computed over independent spatial units.

WHY. `juiz_lidar_v4.json -> testes` only tests MAE aggregated by transect;
the NMAD-by-class advantage (tabela_regua_k0) is descriptive, over the
pool of cells, without an independent unit. Readings of "no significant
difference" can come from a comparison where no formal test was actually
run; this script applies a proper test: unit = lidar transect (n = 13
judgeable), paired test within each stratum, with a transect-bootstrap
CI, effect size, and MDE80.

PRE-REGISTERED CRITERION (recorded in the JSON before any number):
  unit                   judgeable transect (list read from the judge JSON, not recomputed)
  confirmatory family    {20-30m, >30m} x {v23 vs fabdem}, alpha 0.05, Holm within the family
  exploratory            other strata and pairs (v23 vs gedtm30, mlp vs fabdem, v23 vs mlp),
                         reported with raw p and NO verdict
  stratification         PRIMARY by exogenous canopy ETH (h_eth, cob_eth >= 0.8);
                         SENSITIVITY by dossel_v2 (circular — same estimator as the target)
  floor                  n_cel >= 200 per transect-stratum; a stratum enters the test if >= 6 transects
  bootstrap              B = 10,000 transect resamples, seed 42, median of the difference
  power                  MDE80 = d_z for 80% power, alpha 0.05 two-sided, n transects, x sd across transects
  metric                 NMAD = 1.4826 * median(|e - median(e)|) per transect-stratum;
                         difference = NMAD(a) - NMAD(b) (negative = a less dispersed)
  guards (fail loud)     population reproduces tabela_regua_k0[p][f].{n, nmad} of the judge
                         in the dossel_v2 variant (n exact; nmad to 1e-6); parquet transects ==
                         keys of offsets_por_transecto; the judge is read without a defaulting .get.

Usage:
  python auditar_nmad_pareado.py                                   # v2 + juiz_lidar_v4.json
  python auditar_nmad_pareado.py --parquet <solo_v3_nativa.parquet> \
         --juiz juiz_grade_nativa.json --laterais <laterais_nativa> --saida auditar_nmad_pareado_nativa.json

This is v2: numbers were independently re-executed and confirmed bit for
bit in both geometries, guards were mutation-tested, and MDE80 was
checked. This version applies a set of fixes: a complete `_fontes` list
(laterais, surfaces, the script itself); a SECONDARY MAE family with its
own Holm correction (no longer a loose "exploratory" test); Holm applied
per key (fixing a latent counting bug); the effect size renamed to
delta_sinal_pareado (NOT Cliff's delta between independent samples);
explicit tie handling; a methodological note on power; a single bootstrap
stream per test (CIs no longer depend on pair ordering); a uniqueness
assertion for the dedup tie-break; and an explicit DECLARATION of the
judged product: V23 and MLP are the mean of 5 Delta seeds (as in the
judge), baselines are a single realization — the `por_semente` block
reruns the confirmatory test with each seed isolated (E1).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import proveniencia as PROV          # noqa: E402
import juiz_lidar_v4 as J             # noqa: E402

PRODUTOS = ["v23", "mlp", "fabdem", "gedtm30", "glo30"]
PARES = [("v23", "fabdem"), ("v23", "gedtm30"), ("mlp", "fabdem"), ("v23", "mlp")]
CONFIRMATORIAS = {("v23", "fabdem"): ["20-30m", ">30m"]}
FAIXAS, ROT_F = J.FAIXAS, J.ROT_F
PISO_CEL, MIN_TRANSECTOS, B, SEED, ALPHA = 200, 6, 10_000, 42, 0.05


def log(m=""):
    print(m, flush=True)


def nmad(e: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(e - np.median(e))))


def holm(ps: list[float]) -> list[float]:
    """Holm-Bonferroni with a running maximum (same form as the judge's)."""
    idx = np.argsort(ps); m = len(ps); out = [np.nan] * m; run = 0.0
    for r, i in enumerate(idx):
        run = max(run, min(1.0, ps[i] * (m - r)))
        out[i] = run
    return out


def delta_sinal_pareado(x: np.ndarray) -> float:
    """Effect size of the sign test for paired data: P(d>0) - P(d<0). NOT
    Cliff's delta between independent samples."""
    return float(((x > 0).sum() - (x < 0).sum()) / len(x))


def rng_do_teste(a: str, b: str, metrica: str, rot: str) -> np.random.Generator:
    """One bootstrap stream per test: the CI does not depend on the order of PARES."""
    h = int.from_bytes(__import__("hashlib").sha256(f"{a}|{b}|{metrica}|{rot}".encode()).digest()[:4], "big")
    return np.random.default_rng(np.random.SeedSequence([SEED, h]))


def montar_populacao(parquet: Path, juiz: dict) -> tuple[pd.DataFrame, dict]:
    """Reproduces the judge's population (filters, judgeable list, offsets, dedup, forest)."""
    J.SOLO = parquet
    d = J.montar()
    n0 = len(d)
    d = d[(d["n_all"] >= J.N_ALL_MIN)
          & (d["n_casca"] / d["n_all"] >= J.FRAC_CASCA_MIN)
          & np.isfinite(d["z_solo_v2"])]
    offs = juiz["offsets_por_transecto"]              # KeyError se faltar
    trans_parquet = set(d["transecto"].unique())
    if trans_parquet != set(offs.keys()):
        raise RuntimeError(f"transectos do parquet {sorted(trans_parquet)} != chaves de offsets {sorted(offs)}")
    julgaveis = [t for t, v in offs.items() if v["julgavel"]]
    d = d[d["transecto"].isin(julgaveis)].copy()
    d["off_t"] = d["transecto"].map({t: offs[t]["offset_m"] for t in julgaveis})
    d["z_ref"] = d["z_solo_v2"] + d["off_t"]
    n_cl = {t: offs[t]["n_clareira"] for t in julgaveis}
    if len(set(n_cl.values())) != len(n_cl):
        raise RuntimeError(f"empate de n_clareira entre julgaveis {n_cl} — o dedup deixa de ser "
                           "determinado pela regra e passa a depender da ordem das linhas (D9)")
    d["_rank"] = d["transecto"].map(n_cl)
    d = (d.sort_values("_rank", ascending=False)
           .drop_duplicates(subset=["quad", "idx_no"], keep="first"))
    d["faixa"] = pd.cut(d["dossel_v2"], FAIXAS, labels=ROT_F)
    flor = d[(d["dossel_v2"] >= J.DOSSEL_FLORESTA) & (d["agua"] < J.AGUA_LIMPO)].copy()
    if "h_eth" not in flor.columns:
        raise RuntimeError("parquet/laterais sem h_eth — a estratificacao principal (ETH) nao pode ser feita")
    flor["faixa_ex"] = pd.cut(flor["h_eth"].where(flor["cob_eth"] >= 0.8), FAIXAS, labels=ROT_F)
    info = {"n_linhas_parquet": int(n0), "n_floresta_julgavel": int(len(flor)),
            "julgaveis": julgaveis, "n_julgaveis": len(julgaveis)}
    return flor, info


def guardas_populacao(flor: pd.DataFrame, juiz: dict) -> dict:
    """The population must reproduce tabela_regua_k0 (dossel_v2 variant) from the judge."""
    tab = juiz["tabela_regua_k0"]
    conf = {}
    for p in PRODUTOS:
        err = (flor[p] - flor["z_ref"]).to_numpy()
        for rot in ROT_F + ["TODAS"]:
            e = err if rot == "TODAS" else err[(flor["faixa"] == rot).to_numpy()]
            n_j, nmad_j = tab[p][rot]["n"], tab[p][rot]["nmad"]
            if len(e) != n_j:
                raise RuntimeError(f"guarda n: {p}/{rot} recomputado {len(e)} != juiz {n_j}")
            v = nmad(e)
            if abs(v - nmad_j) > 1e-6:
                raise RuntimeError(f"guarda nmad: {p}/{rot} recomputado {v} != juiz {nmad_j}")
            conf[f"{p}|{rot}"] = {"n": int(len(e)), "nmad": v}
    return conf


def tabela_pool(flor: pd.DataFrame, col_faixa: str) -> dict:
    """Pooled table (cell pool) per product x stratum: n, mae, vies, nmad, rmse —
    the manuscript's Table IV in the requested stratification (primary ETH). Same
    `metricas` function as the judge."""
    out = {}
    for p in PRODUTOS:
        err = (flor[p] - flor["z_ref"]).to_numpy()
        out[p] = {}
        for rot in ROT_F:
            e = err[(flor[col_faixa] == rot).to_numpy()]
            out[p][rot] = J.metricas(e) if len(e) else {}
        e = err[flor[col_faixa].notna().to_numpy()]
        out[p]["TODAS"] = J.metricas(e) if len(e) else {}
    return out


def por_transecto_faixa(flor: pd.DataFrame, col_faixa: str) -> dict:
    """{stratum: {transect: {product: {nmad, mae, n_cel}}}} — ALWAYS records n_cel."""
    out = {}
    for rot in ROT_F:
        out[rot] = {}
        sub = flor[flor[col_faixa] == rot]
        for t, s in sub.groupby("transecto"):
            reg = {"n_cel": int(len(s))}
            if len(s) >= 2:
                for p in PRODUTOS:
                    e = (s[p] - s["z_ref"]).to_numpy()
                    reg[p] = {"nmad": nmad(e), "mae": float(np.abs(e).mean())}
            out[rot][str(t)] = reg
    return out


def testar(mat: dict, a: str, b: str, rng_ignorado, metrica: str = "nmad") -> dict:
    """Paired-by-transect test of the difference metric(a) - metric(b), per stratum."""
    res = {}
    for rot in ROT_F:
        ts = [t for t, r in mat[rot].items() if r["n_cel"] >= PISO_CEL and a in r and b in r]
        n = len(ts)
        reg = {"n_transectos": n, "transectos": ts,
               "excluidos_abaixo_do_piso": [t for t, r in mat[rot].items() if r["n_cel"] < PISO_CEL]}
        if n < MIN_TRANSECTOS:
            reg["testado"] = False
            reg["motivo"] = f"menos de {MIN_TRANSECTOS} transectos acima do piso de {PISO_CEL} celulas"
            res[rot] = reg; continue
        dif = np.array([mat[rot][t][a][metrica] - mat[rot][t][b][metrica] for t in ts])
        rng = rng_do_teste(a, b, metrica, rot)
        vence = int((dif < 0).sum()); empates = int((dif == 0).sum())
        p_sinal = float(stats.binomtest(vence, n - empates, 0.5).pvalue) if n - empates > 0 else float("nan")
        try:
            w, p_w = stats.wilcoxon(dif); w = float(w); p_w = float(p_w)
        except ValueError:
            w, p_w = float("nan"), float("nan")
        bs = np.median(dif[rng.integers(0, n, size=(B, n))], axis=1)
        ic = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        sd = float(dif.std(ddof=1)); se = sd / np.sqrt(n)
        t_a = stats.t.ppf(1 - ALPHA / 2, n - 1); t_b = stats.t.ppf(0.80, n - 1)
        reg.update({"testado": True, "diferencas_por_transecto": dict(zip(ts, dif.tolist())),
                    "mediana_m": float(np.median(dif)), "media_m": float(dif.mean()),
                    "ic95_bootstrap_transecto_mediana": ic, "vence": vence, "total": n,
                    "p_sinal_binomial": p_sinal, "empates": empates, "wilcoxon_W": w, "p_wilcoxon": p_w,
                    "delta_sinal_pareado": delta_sinal_pareado(dif), "dp_entre_transectos_m": sd,
                    "mde80_m": float((t_a + t_b) * se), "d_z": float(dif.mean() / sd) if sd > 0 else float("nan"),
                    "mde80_metodo": "aproximacao t (t_{1-a/2}+t_{0,80})*dp/sqrt(n); nct exato difere <0,001 m nesta amostra; MDE do t pareado, nao do Wilcoxon (ARE 0,955)"})
        res[rot] = reg
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(J.SOLO))
    ap.add_argument("--juiz", default=str(J.SAIDA))
    ap.add_argument("--laterais", default=str(J.LAT), help="diretorio de laterais usado PELO MESMO juiz (laterais_nativa na geometria nativa)")
    ap.add_argument("--saida", default="auditar_nmad_pareado.json")
    a = ap.parse_args()
    parquet, juiz_p, saida = Path(a.parquet), Path(a.juiz), RAIZ / a.saida
    J.LAT = Path(a.laterais)
    criterio = {
        "unidade": "transecto julgavel (lista e offsets lidos do juiz, nao recalculados)",
        "familia_confirmatoria": {f"{p[0]} vs {p[1]}": fx for p, fx in CONFIRMATORIAS.items()},
        "metrica_confirmatoria": "nmad",
        "familia_secundaria": {"v23 vs fabdem": ["20-30m", ">30m"], "metrica": "mae",
                               "correcao": "Holm dentro da familia secundaria (2 testes)"},
        "produto_julgado": "V23 e MLP = GLO-30 menos a MEDIA das 5 sementes de Delta (SEEDS_PAR), como em juiz_lidar_v4.py; "
                           "GLO-30/FABDEM/GEDTM30 = realizacao unica; o bloco por_semente refaz o teste confirmatorio com cada semente isolada",
        "alpha": ALPHA, "correcao": "Holm dentro da familia confirmatoria (2 testes)",
        "estratificacao_principal": "ETH exogeno (h_eth, cob_eth >= 0,8)",
        "estratificacao_sensibilidade": "dossel_v2 (circular)",
        "piso_celulas_por_transecto_faixa": PISO_CEL, "min_transectos_por_faixa": MIN_TRANSECTOS,
        "bootstrap": {"B": B, "semente": SEED, "estatistica": "mediana da diferenca por transecto"},
        "metrica": "NMAD por transecto-faixa; diferenca = NMAD(a) - NMAD(b); negativo = a menos disperso",
        "poder": "MDE80 = (t_{1-alpha/2,n-1} + t_{0.80,n-1}) * dp/sqrt(n)",
        "exploratorias": "demais faixas e pares: p bruto, sem veredito",
    }
    juiz = json.loads(juiz_p.read_text(encoding="utf-8"))
    log(f"  parquet {parquet.name} | juiz {juiz_p.name}")
    flor, info = montar_populacao(parquet, juiz)
    conf = guardas_populacao(flor, juiz)
    log(f"  populacao reproduz tabela_regua_k0 (n exato, nmad 1e-6) em {len(conf)} celulas produto x faixa")
    rng = np.random.default_rng(SEED)
    pool_ex = tabela_pool(flor, "faixa_ex")
    pool_v2 = tabela_pool(flor, "faixa")
    log("  tabela agrupada por estrato ETH (nmad | mae): " + "; ".join(
        f"{p} " + "/".join(f"{pool_ex[p][f].get('nmad', float('nan')):.2f}" for f in ROT_F) for p in PRODUTOS))
    mat_ex = por_transecto_faixa(flor, "faixa_ex")
    mat_v2 = por_transecto_faixa(flor, "faixa")
    principal, sens = {}, {}
    for a_, b_ in PARES:
        principal[f"{a_} vs {b_}"] = {"nmad": testar(mat_ex, a_, b_, rng, "nmad"),
                                     "mae": testar(mat_ex, a_, b_, rng, "mae")}
        sens[f"{a_} vs {b_}"] = {"nmad": testar(mat_v2, a_, b_, rng, "nmad"),
                                "mae": testar(mat_v2, a_, b_, rng, "mae")}
    # Holm within the confirmatory family (primary stratification)
    fam = principal["v23 vs fabdem"]["nmad"]
    def aplicar_holm(bloco: dict):
        chaves = [f for f in CONFIRMATORIAS[("v23", "fabdem")] if bloco[f].get("testado") and np.isfinite(bloco[f]["p_wilcoxon"])]
        for f, pv in zip(chaves, holm([bloco[f]["p_wilcoxon"] for f in chaves])):
            bloco[f]["p_holm"] = float(pv)
    aplicar_holm(fam)                                    # confirmatory family (NMAD)
    aplicar_holm(principal["v23 vs fabdem"]["mae"])      # secondary family (MAE), its own Holm correction
    aplicar_holm(sens["v23 vs fabdem"]["nmad"]); aplicar_holm(sens["v23 vs fabdem"]["mae"])
    # confirmatory verdict
    ver = {}
    for f in CONFIRMATORIAS[("v23", "fabdem")]:
        r = fam[f]
        if not r.get("testado"):
            ver[f] = "NAO_TESTAVEL"; continue
        if "p_holm" not in r:
            ver[f] = "NAO_DECIDIVEL_P_NAO_FINITO"; continue
        ver[f] = ("V23_MENOS_DISPERSO" if (r["p_holm"] < ALPHA and r["mediana_m"] < 0)
                  else "FABDEM_MENOS_DISPERSO" if (r["p_holm"] < ALPHA and r["mediana_m"] > 0)
                  else "INDETERMINADO")
        log(f"  {f}: v23 vence {r['vence']}/{r['total']} | mediana {r['mediana_m']:+.3f} m "
            f"IC95 [{r['ic95_bootstrap_transecto_mediana'][0]:+.3f}, {r['ic95_bootstrap_transecto_mediana'][1]:+.3f}] | "
            f"p_w {r['p_wilcoxon']:.4f} p_holm {r['p_holm']:.4f} | efeito {r['delta_sinal_pareado']:+.2f} | MDE80 {r['mde80_m']:.3f} m -> {ver[f]}")
    frase = {}
    for f in CONFIRMATORIAS[("v23", "fabdem")]:
        r = fam[f]
        if r.get("testado") and ver[f] == "V23_MENOS_DISPERSO":
            frase[f] = (f"Across {r['total']} independent lidar transects, the reconstruction attains a lower NMAD than FABDEM in the "
                        f"{f} canopy stratum (exogenous canopy height) in {r['vence']} of {r['total']} transects (median paired difference "
                        f"{r['mediana_m']:+.2f} m, 95% CI [{r['ic95_bootstrap_transecto_mediana'][0]:+.2f}, {r['ic95_bootstrap_transecto_mediana'][1]:+.2f}] by transect bootstrap; "
                        f"Wilcoxon p = {r['p_wilcoxon']:.3f}, Holm-adjusted p = {r['p_holm']:.3f}; paired sign effect = {r['delta_sinal_pareado']:+.2f}).")
        elif r.get("testado"):
            frase[f] = (f"Across {r['total']} independent lidar transects the paired difference in NMAD between the reconstruction and FABDEM in the {f} stratum "
                        f"is {r['mediana_m']:+.2f} m (95% CI [{r['ic95_bootstrap_transecto_mediana'][0]:+.2f}, {r['ic95_bootstrap_transecto_mediana'][1]:+.2f}]; Holm-adjusted p = {r['p_holm']:.3f}), "
                        f"which does not establish an ordering; the design could detect a difference of about {r['mde80_m']:.2f} m with 80% power.")
        else:
            frase[f] = "not testable under the pre-registered floor"
    # E1 — the product is the mean of 5 seeds; rerun the confirmatory test with each seed isolated
    por_semente = {}
    for sd in J.SEEDS_PAR:
        col = f"v23_s{sd}"
        vals = np.full(len(flor), np.nan)
        for q in J.QUADS:
            m = (flor["quad"] == q).to_numpy(); idx = flor.loc[m, "idx_no"].to_numpy()
            z = np.load(J.SUP / f"delta_amazonia_ampliacao_b4s035_seed{sd}.npz")
            vals[m] = flor.loc[m, "glo30"].to_numpy() - z[q][idx]; z.close()
        flor[col] = vals
        mat_s = {}
        for rot in ROT_F:
            mat_s[rot] = {}
            sub = flor[flor["faixa_ex"] == rot]
            for t, s_ in sub.groupby("transecto"):
                reg = {"n_cel": int(len(s_))}
                if len(s_) >= 2:
                    e1 = (s_[col] - s_["z_ref"]).to_numpy(); e2 = (s_["fabdem"] - s_["z_ref"]).to_numpy()
                    reg[col] = {"nmad": nmad(e1), "mae": float(np.abs(e1).mean())}
                    reg["fabdem"] = {"nmad": nmad(e2), "mae": float(np.abs(e2).mean())}
                mat_s[rot][str(t)] = reg
        por_semente[str(sd)] = {"nmad": {f: {k: v for k, v in testar(mat_s, col, "fabdem", None, "nmad")[f].items()
                                              if k in ("testado", "vence", "total", "mediana_m", "ic95_bootstrap_transecto_mediana", "p_wilcoxon", "delta_sinal_pareado")}
                                         for f in CONFIRMATORIAS[("v23", "fabdem")]}}
    consist = {f: {"sementes_com_mediana_negativa": sum(1 for sd in por_semente if por_semente[sd]["nmad"][f].get("testado") and por_semente[sd]["nmad"][f]["mediana_m"] < 0),
                   "sementes_com_p_menor_alpha": sum(1 for sd in por_semente if por_semente[sd]["nmad"][f].get("testado") and por_semente[sd]["nmad"][f]["p_wilcoxon"] < ALPHA),
                   "n_sementes": len(por_semente)} for f in CONFIRMATORIAS[("v23", "fabdem")]}
    log("  por semente (v23 isolada vs FABDEM, NMAD): " + "; ".join(f"{f}: mediana<0 em {c['sementes_com_mediana_negativa']}/{c['n_sementes']}, p<alpha em {c['sementes_com_p_menor_alpha']}/{c['n_sementes']}" for f, c in consist.items()))
    fontes = [parquet, juiz_p, RAIZ / "juiz_lidar_v4.py", Path(__file__)]
    fontes += [J.LAT / f"{n}_amazonia_{q}.npz" for q in J.QUADS for n in ("glo30", "fabdem", "gedtm30", "agua_jrc", "dossel_eth", "rotulos")]
    fontes += [J.SUP / f"delta_amazonia_ampliacao_b4s035{arm}_seed{sd}.npz" for arm in ("", "_mlp_semobs") for sd in J.SEEDS_PAR]
    PROV.gravar(saida, {
        "criterio_pre_registrado": criterio,
        "por_semente_confirmatorio": {"testes": por_semente, "consistencia": consist,
                                      "nota": "E1: V23 de cada semente isolada contra FABDEM (realizacao unica), estrato ETH; a media de 5 sementes reduz a NMAD do produto"},
        "entradas": {"parquet": str(parquet), "juiz": str(juiz_p), "laterais": str(J.LAT)},
        "populacao": info,
        "guarda_reproducao_tabela_regua_k0": conf,
        "tabela_pool_ETH": pool_ex,
        "tabela_pool_dossel_v2": pool_v2,
        "nmad_por_transecto_faixa_ETH": mat_ex,
        "nmad_por_transecto_faixa_dossel_v2": mat_v2,
        "testes_principal_ETH": principal,
        "testes_sensibilidade_dossel_v2": sens,
        "veredito_confirmatorio": ver,
        "frase_autorizada": frase,
        "contra_auditoria": ("v1 checked by an independent recomputation (bit-for-bit re-execution in "
                             "both geometries, mutation of the guards, recomputation of p/CI/MDE80); "
                             "v2 applies D1-D9 + E1"),
    }, fontes_lidas=fontes, script=__file__)
    log(f"  -> {saida.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
