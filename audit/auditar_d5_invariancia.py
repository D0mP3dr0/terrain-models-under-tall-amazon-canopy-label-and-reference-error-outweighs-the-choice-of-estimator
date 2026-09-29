"""D5 — invariance of the MAE ranking to a PER-CELL ruler shift sized by the
native scale delta (ETH class >30 m, judgeable forest).

ANADEM enters as an additional product, read through the same code path as
`auditar_nmad_pareado_nativa_anadem.py`; at the time this run was produced,
independent verification of the ANADEM figures was still in progress, so the
ANADEM results here are reported as conditional on that verification. The
same v23-vs-product rule is applied, with no new criterion.

This script does not edit any existing module: it imports
`auditar_nmad_pareado` (REF) and `auditar_nmad_pareado_nativa_anadem` (REF2)
and REUSES their pure functions (`REF.nmad`, `REF.holm`, `REF.testar`,
`REF.montar_populacao`, `REF2.montar_populacao_com_anadem`) instead of
duplicating logic.

QUESTION. Does the ordering "proposed - FABDEM > 0 in MAE above 30 m",
paired by transect, survive a PER-CELL shift of the ruler sized by delta_c?

DATA. Population from `auditar_nmad_pareado_nativa_diretas.json`: judgeable
forest, ETH class >30 m, per-cell error for each product reconstructed
exactly as the NMAD auditor reconstructs it (`REF.montar_populacao` /
`REF2.montar_populacao_com_anadem`, without rewriting). delta_c comes ONLY
from `delta_escala_celulas_nativa.parquet` (sha256 checked below), joined on
(transecto, quad, idx_no). An area-grid proxy is NOT allowed and is not used
here.

BRANCHES:
  - Perturbed ruler g'_c = g_c + lambda*delta_c, lambda in {0; 0.25; 0.5;
    0.75; 1} (physical sign, ruler below ground).
  - Sign control: lambda in {-0.5; -1}.
  - Stress test: delta_c replaced by the upper bound of the bootstrap CI
    (B=10000, seed 42) of the mean delta PER TRANSECT, applied to ALL cells
    of the transect (constant within-transect), with the same g' = g + delta.
  - Common-shift guard: reproduces an earlier invariance check: a COMMON
    offset `s` (same value for every cell and every product, NOT the
    per-cell delta) from 0 to +4.4109 m (the smallest |bias| in the table,
    FABDEM, in `tabela_pool_ETH`) does not change the MAE ranking among the
    5 ORIGINAL products (ANADEM is excluded because it did not exist at the
    time of that earlier check).
  - Positive control: a SYNTHETIC construction (labeled as such, NOT a
    branch of the physical design) that adds K*delta_c (K=1) to the RAW
    FABDEM value (not to z_ref) to demonstrate that the script does report
    a ranking inversion when one genuinely exists.

STATISTICS: the same as the NMAD auditor, reusing its code — median of the
per-transect MAE differences (`REF.testar`, metric="mae"), bootstrap CI over
transects (B=10,000, seed 42), Wilcoxon, sign test, Holm correction
(singleton here — only one ">30m" band per pair in this run, so Holm does
not change p, but it is applied via `REF.holm` to avoid rewriting the
correction rule).

REPRODUCTION GUARD (fail-loud): lambda=0 must reproduce
`testes_principal_ETH["v23 vs fabdem"].mae[">30m"].mediana_m` = +2.3259 m,
CI [+1.780; +2.616], to within 1e-6. The sha256 of the inputs is checked
BEFORE computing anything.

PRE-REGISTERED CRITERION (copied here verbatim, not reinterpreted):
  - The claim stands as written if, for EVERY lambda in the [0;1] grid, the
    95% CI of the median excludes 0 (median>0, ci_lower>0) AND p_holm < 0.05.
  - If it fails at some lambda, the claim gains the qualifier "for reference
    offsets up to lambda_max*delta", where lambda_max is the largest grid
    point that passes (the grid is not refined after seeing the result).
  - If it already fails at lambda=0.25, the claim is dropped from the
    abstract.

This script does not interpret the result for the manuscript and does not
audit itself: it only produces the logged number.

Usage:
  /trabalho/ambientes/s33_amb_virtual/.venv/bin/python auditar_d5_invariancia.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import proveniencia as PROV                      # noqa: E402
import juiz_lidar_v4 as J                          # noqa: E402
import auditar_nmad_pareado as REF                 # noqa: E402  (read-only reference)
import auditar_nmad_pareado_nativa_anadem as REF2   # noqa: E402  (read-only reference)

# ── fixed paths (same "diretas" variant as the reference population) ──
PARQUET = Path("/trabalho/GNN_TOPO/SATELITES/lidar_eba/solo_lidar_30m_v3_nativa.parquet")
JUIZ_P = RAIZ / "juiz_grade_nativa_diretas.json"
LATERAIS = Path("/trabalho/GNN_TOPO/SATELITES/laterais_nativa_diretas")
RECORTE_DIR = Path("/trabalho/GNN_TOPO/SATELITES/anadem")
DELTA_PARQUET = RAIZ / "delta_escala_celulas_nativa.parquet"
REFERENCIA_JSON = RAIZ / "auditar_nmad_pareado_nativa_diretas.json"
CONTRA_AUDITORIA_ANADEM_MD = (RAIZ / "artigo_v23_2" / "_pareceres_2026-09-28_execucao"
                              / "contra_auditoria_anadem.md")

SHA256_DELTA_ESPERADO = "110d75ca56626f268f32ece2acb2d0719ff7a19ced3e8f2b18fa3c122b2a14c2"

FAIXA = ">30m"
PRODUTOS_BASE = list(REF2.PRODUTOS_BASE)          # ["v23","mlp","fabdem","gedtm30","glo30"]
PRODUTOS_TODOS = list(REF2.PRODUTOS_TODOS)        # + "anadem"
B, SEED, ALPHA = REF.B, REF.SEED, REF.ALPHA
PISO_CEL, MIN_TRANSECTOS = REF.PISO_CEL, REF.MIN_TRANSECTOS

LAMBDA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
LAMBDA_CONTROLE_SENTIDO = [-0.5, -1.0]
GUARDA_S_MAX = None  # filled in at runtime with the exact FABDEM bias


def log(m: str = "") -> None:
    print(m, flush=True)


def conferir_sha256_delta(p: Path) -> str:
    h = PROV.sha256(p)
    if h != SHA256_DELTA_ESPERADO:
        raise RuntimeError(
            (f"GUARD FAILED: sha256 of {p.name} = {h}, expected {SHA256_DELTA_ESPERADO} "
             f"(native delta changed since the independent check; aborting BEFORE computing)"))
    return h


def carregar_populacao() -> tuple[pd.DataFrame, dict]:
    """Judgeable-forest population, ETH class >30 m, with ANADEM merged in
    via the SAME code path as `auditar_nmad_pareado_nativa_anadem.py`, and
    the native per-cell delta joined on (transecto, quad, idx_no). Nothing
    is recomputed by hand: `REF2.montar_populacao_com_anadem` reuses
    `J.montar()` plus the same filters/dedup/stratification as the NMAD
    auditor."""
    J.LAT = LATERAIS
    juiz = json.loads(JUIZ_P.read_text(encoding="utf-8"))
    flor, info = REF2.montar_populacao_com_anadem(PARQUET, juiz, LATERAIS)
    flor30 = flor[flor["faixa_ex"] == FAIXA].copy()
    n_floresta_30 = len(flor30)

    delta = pd.read_parquet(DELTA_PARQUET)[["transecto", "quad", "idx_no", "delta"]]
    n_delta_total = len(delta)
    pop = flor30.merge(delta, on=["transecto", "quad", "idx_no"], how="left")
    n_com_delta = int(pop["delta"].notna().sum())
    if n_com_delta != n_floresta_30:
        # An area-grid proxy is NOT allowed: there is no fallback. This just
        # restricts the population to cells with a valid native delta and
        # records the loss.
        pop = pop[pop["delta"].notna()].copy()

    info_pop = {
        "n_linhas_parquet_raw": info["n_linhas_parquet"],
        "n_floresta_julgavel_total": info["n_floresta_julgavel"],
        "n_floresta_julgavel_eth_>30m": n_floresta_30,
        "n_delta_nativo_parquet_linhas": n_delta_total,
        "n_com_delta_valido": n_com_delta,
        "n_perdidas_por_delta_ausente": n_floresta_30 - n_com_delta,
        "n_julgaveis": info["n_julgaveis"],
        "julgaveis": info["julgaveis"],
        "n_transectos_na_populacao_>30m": int(pop["transecto"].nunique()),
    }
    return pop, info_pop


def matriz(pop: pd.DataFrame, zref: pd.Series, produtos: list[str]) -> dict:
    """{faixa: {transecto: {n_cel, produto: {mae, nmad}}}} — ONLY the
    ">30m" band is populated (the others stay empty just so `REF.testar` can
    iterate `J.ROT_F` without a KeyError; it never accesses the empty
    bands)."""
    out = {rot: {} for rot in J.ROT_F}
    for t, s in pop.groupby("transecto"):
        zr = zref.loc[s.index]
        reg: dict = {"n_cel": int(len(s))}
        for p in produtos:
            e = (s[p] - zr).to_numpy()
            if not np.isfinite(e).all():
                continue
            reg[p] = {"mae": float(np.abs(e).mean()), "nmad": REF.nmad(e)}
        out[FAIXA][str(t)] = reg
    return out


def pool_mae(pop: pd.DataFrame, zref: pd.Series, produtos: list[str]) -> tuple[dict, list]:
    maes = {}
    for p in produtos:
        e = (pop[p] - zref).to_numpy()
        maes[p] = float(np.abs(e).mean()) if np.isfinite(e).all() else float("nan")
    ordem = sorted(maes, key=lambda p: maes[p])
    return maes, ordem


def teste_par(mat: dict, a: str, b: str) -> dict:
    """`REF.testar` plus the Holm correction reused via `REF.holm`
    (singleton here — only one band per run — but called the same way the
    original auditor calls it, to avoid rewriting the correction rule)."""
    r = REF.testar(mat, a, b, None, "mae")[FAIXA]
    if r.get("testado") and np.isfinite(r.get("p_wilcoxon", float("nan"))):
        r["p_holm"] = float(REF.holm([r["p_wilcoxon"]])[0])
    return r


def criterio_passa(r: dict) -> bool:
    """Pre-registered criterion: median>0 (proposed is more wrong than
    FABDEM/ANADEM in MAE, i.e. FABDEM/ANADEM wins), 95% CI excludes 0
    (lower bound>0), and p_holm<0.05."""
    if not r.get("testado") or "p_holm" not in r:
        return False
    ic_inf = r["ic95_bootstrap_transecto_mediana"][0]
    return bool(r["mediana_m"] > 0 and ic_inf > 0 and r["p_holm"] < ALPHA)


def delta_ic_sup_por_transecto(pop: pd.DataFrame) -> dict:
    """Per-transect bootstrap (B=10,000, seed 42) of the MEAN of delta_c
    within that transect; returns the 97.5th percentile (upper CI bound),
    used in the stress branch as a CONSTANT delta per transect."""
    out = {}
    for t, s in pop.groupby("transecto"):
        d = s["delta"].to_numpy()
        h = int.from_bytes(hashlib.sha256(f"delta_ic|{t}".encode()).digest()[:4], "big")
        rng = np.random.default_rng(np.random.SeedSequence([SEED, h]))
        bs = d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1)
        out[str(t)] = float(np.percentile(bs, 97.5))
    return out


def braco_lambda(pop: pd.DataFrame, lam: float) -> dict:
    zref = pop["z_ref"] + lam * pop["delta"]
    mat = matriz(pop, zref, PRODUTOS_TODOS)
    r_fab = teste_par(mat, "v23", "fabdem")
    r_ana = teste_par(mat, "v23", "anadem")
    maes, ordem = pool_mae(pop, zref, PRODUTOS_TODOS)
    return {
        "lambda": lam,
        "n_cel": int(len(pop)), "n_transectos": int(pop["transecto"].nunique()),
        "pool_mae_>30m": maes, "ranking_mae_crescente": ordem,
        "teste_pareado_v23_vs_fabdem_mae": r_fab,
        "teste_pareado_v23_vs_anadem_mae_condicionado_a_contra_auditoria": r_ana,
        "criterio_fabdem_passa": criterio_passa(r_fab),
        "criterio_anadem_passa": criterio_passa(r_ana),
    }


def braco_estresse(pop: pd.DataFrame, ic_sup: dict) -> dict:
    delta_transecto = pop["transecto"].astype(str).map(ic_sup)
    zref = pop["z_ref"] + delta_transecto
    mat = matriz(pop, zref, PRODUTOS_TODOS)
    r_fab = teste_par(mat, "v23", "fabdem")
    r_ana = teste_par(mat, "v23", "anadem")
    maes, ordem = pool_mae(pop, zref, PRODUTOS_TODOS)
    return {
        "descricao": "delta_c trocado pelo limite superior do IC bootstrap (B=10000, "
                      "semente 42) da media de delta POR TRANSECTO, constante intra-transecto",
        "delta_ic_sup_por_transecto_m": ic_sup,
        "n_cel": int(len(pop)), "n_transectos": int(pop["transecto"].nunique()),
        "pool_mae_>30m": maes, "ranking_mae_crescente": ordem,
        "teste_pareado_v23_vs_fabdem_mae": r_fab,
        "teste_pareado_v23_vs_anadem_mae_condicionado_a_contra_auditoria": r_ana,
        "criterio_fabdem_passa": criterio_passa(r_fab),
        "criterio_anadem_passa": criterio_passa(r_ana),
    }


def guarda_deslocamento_comum(pop: pd.DataFrame, ref: dict) -> dict:
    """Reproduces an earlier invariance check: a COMMON offset `s` (same
    value for every cell and every product — NOT the per-cell delta) from 0
    to +4.4109 m (the FABDEM bias in tabela_pool_ETH of the reference file
    itself) does not change the MAE ranking among the 5 ORIGINAL products
    (ANADEM did not exist at the time of that earlier check and is left out
    of this specific block)."""
    s_max = float(ref["tabela_pool_ETH"]["fabdem"][FAIXA]["vies"])
    grid = np.linspace(0.0, s_max, 21)
    err = {p: (pop[p] - pop["z_ref"]).to_numpy() for p in PRODUTOS_BASE}
    ordem_0 = None
    cruzou = False
    s_da_troca = None
    curva = []
    for s in grid:
        maes = {p: float(np.abs(err[p] - s).mean()) for p in PRODUTOS_BASE}
        ordem = sorted(maes, key=lambda p: maes[p])
        curva.append({"s": float(s), "pool_mae": maes, "ordem": ordem})
        if ordem_0 is None:
            ordem_0 = ordem
        elif ordem != ordem_0 and not cruzou:
            cruzou = True
            s_da_troca = float(s)
    return {
        "s_max_grade_m": s_max, "n_pontos_grade": len(grid),
        "produtos_avaliados": PRODUTOS_BASE,
        "ordem_em_s_0": ordem_0, "ordem_cruza_no_intervalo": cruzou,
        "s_onde_cruzou_m": s_da_troca,
        "reproduz_invariancia_13_09": bool(not cruzou),
        "curva": curva,
        "nota": ("Common s (not the per-cell delta_c); guard defined on 13/09. ANADEM is "
                 "left out because it did not exist at that date; see the separate block "
                 "for ANADEM."),
    }


def controle_positivo(pop: pd.DataFrame) -> dict:
    """SYNTHETIC construction (labeled as such, NOT a branch of the
    physical design): adds K*delta_c (K=1) to the RAW FABDEM value (not to
    z_ref, not to the other products) to demonstrate that the script DOES
    report the inversion when one genuinely exists by construction."""
    K = 1.0
    pop2 = pop.copy()
    pop2["fabdem"] = pop2["fabdem"] + K * pop2["delta"]
    zref = pop2["z_ref"]
    mat = matriz(pop2, zref, PRODUTOS_BASE)
    r = teste_par(mat, "v23", "fabdem")
    maes0, ordem0 = pool_mae(pop, pop["z_ref"], PRODUTOS_BASE)
    maes1, ordem1 = pool_mae(pop2, zref, PRODUTOS_BASE)
    inverteu = bool(maes1["fabdem"] > maes1["v23"] and maes0["fabdem"] < maes0["v23"])
    return {
        "construcao": "fabdem_construido = fabdem + K*delta_c, K=1.0, aplicado ao valor "
                      "BRUTO do produto (nao ao z_ref); demais produtos inalterados",
        "pool_mae_antes": maes0, "ordem_antes": ordem0,
        "pool_mae_depois": maes1, "ordem_depois": ordem1,
        "teste_pareado_v23_vs_fabdem_construido_mae": r,
        "inversao_reportada": inverteu,
    }


def main() -> int:
    log("D5 -- invariancia do ranking de MAE a deslocamento por celula (delta nativo)")
    conferir_sha256_delta(DELTA_PARQUET)
    log(f"  guarda sha256 delta nativo: bate ({SHA256_DELTA_ESPERADO[:16]}...)")

    if not REFERENCIA_JSON.exists():
        raise RuntimeError(f"referencia ausente: {REFERENCIA_JSON}")
    ref = json.loads(REFERENCIA_JSON.read_text(encoding="utf-8"))

    pop, info_pop = carregar_populacao()
    log(f"  populacao: {info_pop['n_floresta_julgavel_eth_>30m']} celulas floresta "
        f"julgavel >30m, {info_pop['n_com_delta_valido']} com delta nativo valido "
        f"({info_pop['n_perdidas_por_delta_ausente']} perdidas), "
        f"{info_pop['n_transectos_na_populacao_>30m']} transectos")

    # ── reproduction guard (fail-loud): lambda=0 must reproduce the
    # reference value to within 1e-6 BEFORE any other computation.
    mat0 = matriz(pop, pop["z_ref"], PRODUTOS_BASE)
    r0 = REF.testar(mat0, "v23", "fabdem", None, "mae")[FAIXA]
    ref_r = ref["testes_principal_ETH"]["v23 vs fabdem"]["mae"][FAIXA]
    dif_mediana = abs(r0["mediana_m"] - ref_r["mediana_m"])
    dif_ic = max(abs(r0["ic95_bootstrap_transecto_mediana"][0] - ref_r["ic95_bootstrap_transecto_mediana"][0]),
                 abs(r0["ic95_bootstrap_transecto_mediana"][1] - ref_r["ic95_bootstrap_transecto_mediana"][1]))
    guarda_reproducao = {
        "mediana_m_recomputada": r0["mediana_m"], "mediana_m_esperada": ref_r["mediana_m"],
        "diferenca_abs_mediana": dif_mediana,
        "ic95_recomputado": r0["ic95_bootstrap_transecto_mediana"],
        "ic95_esperado": ref_r["ic95_bootstrap_transecto_mediana"],
        "diferenca_abs_ic_max": dif_ic,
        "tolerancia": 1e-6, "passou": bool(dif_mediana <= 1e-6 and dif_ic <= 1e-6),
    }
    if not guarda_reproducao["passou"]:
        raise RuntimeError(f"GUARDA DE REPRODUCAO FALHOU: {guarda_reproducao}")
    log(f"  guarda de reproducao (lambda=0): mediana {r0['mediana_m']:.6f} m == referencia "
        f"{ref_r['mediana_m']:.6f} m (dif {dif_mediana:.2e}); IC bate (dif max {dif_ic:.2e})")

    # ── design branches
    bracos = {}
    for lam in LAMBDA_GRID:
        b = braco_lambda(pop, lam)
        bracos[f"lambda_{lam:+.2f}_sentido_fisico"] = b
        log(f"  lambda={lam:+.2f}: mediana(v23-fabdem) {b['teste_pareado_v23_vs_fabdem_mae']['mediana_m']:+.4f} m "
            f"-> criterio FABDEM {'PASSA' if b['criterio_fabdem_passa'] else 'FALHA'} | "
            f"ANADEM {'PASSA' if b['criterio_anadem_passa'] else 'FALHA'} (condicionado)")
    for lam in LAMBDA_CONTROLE_SENTIDO:
        b = braco_lambda(pop, lam)
        bracos[f"lambda_{lam:+.2f}_controle_sentido"] = b
        log(f"  lambda={lam:+.2f} (controle de sentido): mediana(v23-fabdem) "
            f"{b['teste_pareado_v23_vs_fabdem_mae']['mediana_m']:+.4f} m")

    ic_sup = delta_ic_sup_por_transecto(pop)
    b_estresse = braco_estresse(pop, ic_sup)
    bracos["estresse_delta_ic_sup_transecto"] = b_estresse
    log(f"  estresse (delta = IC sup por transecto): mediana(v23-fabdem) "
        f"{b_estresse['teste_pareado_v23_vs_fabdem_mae']['mediana_m']:+.4f} m -> "
        f"criterio FABDEM {'PASSA' if b_estresse['criterio_fabdem_passa'] else 'FALHA'}")

    guarda_comum = guarda_deslocamento_comum(pop, ref)
    log(f"  guarda de deslocamento comum (13/09): cruza={guarda_comum['ordem_cruza_no_intervalo']} "
        f"-> reproduz_invariancia_13_09={guarda_comum['reproduz_invariancia_13_09']}")

    ctrl_pos = controle_positivo(pop)
    log(f"  controle positivo (inversao por construcao): inversao_reportada="
        f"{ctrl_pos['inversao_reportada']}")

    # ── per-product verdict, per the pre-registered criterion (lambda_max
    # of the physical grid {0;0.25;0.5;0.75;1} that passes; grid not refined)
    def veredito_produto(chave_criterio: str) -> dict:
        passos = {lam: bracos[f"lambda_{lam:+.2f}_sentido_fisico"][chave_criterio] for lam in LAMBDA_GRID}
        todos_passam = all(passos.values())
        falha_em_025 = not passos[0.25]
        lambda_max = max([lam for lam, p in passos.items() if p], default=None)
        if todos_passam:
            veredito = "FRASE_MANTIDA_SEM_RESTRICAO"
            frase = ("A ordenacao proposto-FABDEM>0 em MAE (>30 m) sobrevive ao deslocamento "
                     "por celula do delta nativo em toda a grade fisica [0;1].")
        elif falha_em_025:
            veredito = "FRASE_SAI_DO_ABSTRACT"
            frase = (("Fails already at lambda=0.25: the sentence leaves the abstract and "
                      "becomes conditional in Results (pre-declared rule)."))
        elif lambda_max is not None:
            veredito = "FRASE_CONDICIONADA_A_LAMBDA_MAX"
            frase = (f"A frase ganha a restricao 'for reference offsets up to "
                      f"{lambda_max:g}*delta' (lambda_max da grade que passa, grade nao refinada).")
        else:
            veredito = "NENHUM_LAMBDA_DA_GRADE_PASSA"
            frase = "Nenhum ponto da grade fisica passa o criterio."
        return {"passa_por_lambda": passos, "todos_passam": todos_passam,
                "falha_em_lambda_0_25": falha_em_025, "lambda_max_que_passa": lambda_max,
                "veredito": veredito, "frase_literal": frase}

    veredito = {
        "fabdem": veredito_produto("criterio_fabdem_passa"),
        "anadem_condicionado_a_contra_auditoria": veredito_produto("criterio_anadem_passa"),
    }
    log(f"  veredito FABDEM: {veredito['fabdem']['veredito']}")
    log(f"  veredito ANADEM (condicionado): {veredito['anadem_condicionado_a_contra_auditoria']['veredito']}")

    anadem_nota = {
        "instrucao_recebida": ("ANADEM results are conditional on a pending independent recomputation; "
                               "same rule, no new criterion."),
        "status_observado_no_disco_desta_corrida": (
            CONTRA_AUDITORIA_ANADEM_MD.read_text(encoding="utf-8").splitlines()[2].strip()
            if CONTRA_AUDITORIA_ANADEM_MD.exists() else "independent check record file not found"),
        "arquivo_contra_auditoria": str(CONTRA_AUDITORIA_ANADEM_MD),
    }

    criterio_pre_registrado = {
        "pergunta": "A ordenacao 'proposto - FABDEM > 0 em MAE acima de 30 m', pareada por "
                    "transecto, sobrevive a um deslocamento POR CELULA da regua do tamanho "
                    "de delta_c?",
        "dado": "populacao de auditar_nmad_pareado_nativa_diretas.json: floresta julgavel, "
                "classe ETH >30 m, erro por celula reconstruido como no auditor NMAD; "
                "delta_c SO de delta_escala_celulas_nativa.parquet, join por "
                "(transecto,quad,idx_no); proxy de grade de area PROIBIDO",
        "bracos": "g'_c=g_c+lambda*delta_c, lambda em {0;0,25;0,5;0,75;1} sentido fisico; "
                  "controle de sentido lambda em {-0,5;-1}; estresse com delta = IC sup do "
                  "transecto; guarda de deslocamento comum reproduz a invariancia de 13/09; "
                  "controle positivo inverte por construcao",
        "estatistica": "mesma do auditor NMAD (REF.testar), mediana das diferencas de MAE "
                       "por transecto, bootstrap B=10000 semente 42, Wilcoxon, teste do "
                       "sinal, Holm (singleton nesta corrida)",
        "guarda_de_reproducao": "lambda=0 reproduz testes_principal_ETH['v23 vs "
                                "fabdem'].mae['>30m'].mediana_m=+2,3259 IC[+1,780;+2,616] a "
                                "1e-6, sha256 conferido antes de calcular",
        "controle_positivo": "deslocamento por celula injetado no FABDEM que inverte a "
                             "ordenacao por construcao; o script reporta a inversao",
        "criterio": "frase mantida se para TODO lambda em [0;1] da grade o IC95 da mediana "
                    "excluir 0 e p_holm<0,05; se falhar nalgum lambda, frase ganha 'for "
                    "reference offsets up to lambda_max*delta' (grade nao refinada); se "
                    "falhar ja em lambda=0,25, a frase sai do abstract",
        "dono_do_desenho": ("Design scope of this check: ranking statistics, native delta, and sign "
                            "convention."),
        "acrescimo_do_dono_28_09": ("ANADEM enters as an additional product, same path as "
                                    "auditar_nmad_pareado_nativa_anadem.py, same rule, no new criterion; "
                                    "results conditional on a pending independent recomputation"),
    }

    fontes = [PARQUET, JUIZ_P, DELTA_PARQUET, REFERENCIA_JSON, RAIZ / "juiz_lidar_v4.py",
              RAIZ / "auditar_nmad_pareado.py", RAIZ / "auditar_nmad_pareado_nativa_anadem.py",
              Path(__file__)]
    fontes += [LATERAIS / f"{n}_amazonia_{q}.npz" for q in J.QUADS
               for n in ("glo30", "fabdem", "gedtm30", "agua_jrc", "dossel_eth", "rotulos", "anadem")]

    saida = RAIZ / "auditar_d5_invariancia.json"
    PROV.gravar(saida, {
        "criterio_pre_registrado": criterio_pre_registrado,
        "guarda_sha256_delta_nativo": {"sha256": SHA256_DELTA_ESPERADO, "arquivo": DELTA_PARQUET.name,
                                       "conferido_antes_de_calcular": True},
        "guarda_reproducao_lambda_0": guarda_reproducao,
        "populacao": info_pop,
        "bracos": bracos,
        "guarda_deslocamento_comum_13_09": guarda_comum,
        "controle_positivo_inversao_por_construcao": ctrl_pos,
        "anadem_condicionado_a_contra_auditoria": anadem_nota,
        "veredito": veredito,
    }, fontes_lidas=fontes, script=__file__)
    log(f"  -> {saida.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
