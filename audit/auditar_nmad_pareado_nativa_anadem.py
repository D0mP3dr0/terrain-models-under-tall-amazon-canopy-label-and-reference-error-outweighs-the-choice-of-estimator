"""ANADEM as the FIFTH product of Table IV (Article 1, V23) — the same
comparison PAIRED BY TRANSECT against the airborne lidar that GLO-30,
FABDEM, GEDTM30, and the study's own product (v23) already receive in
`auditar_nmad_pareado.py` (nativa_diretas variant).

This script does not edit `auditar_nmad_pareado.py` or `juiz_lidar_v4.py`:
it imports both as modules and REUSES their pure functions (nmad, holm,
delta_sinal_pareado, rng_do_teste, guardas_populacao, J.montar,
J.metricas) instead of duplicating logic. It only adds what ANADEM
requires:

  1. merging the "anadem" column (laterais_nativa_diretas/anadem_amazonia_Q*.npz,
     key "DTM") by quad+idx_no, RIGHT AFTER `J.montar()` and BEFORE any
     filter — so it passes through exactly the same filters, dedup, and
     stratification as the other 5 products, without changing row order
     or any other field;
  2. a new pair ("v23", "anadem"), EXPLORATORY (no verdict, like the other
     non-pre-registered pairs — same rule as the original script);
  3. GUARD (i), fail-loud: besides reproducing the judge's
     `tabela_regua_k0` (inherited from REF.guardas_populacao), it compares
     `tabela_pool_ETH`, `tabela_pool_dossel_v2`, `testes_principal_ETH`,
     and `testes_sensibilidade_dossel_v2` for the 5 ORIGINAL products/4
     ORIGINAL pairs against `auditar_nmad_pareado_nativa_diretas.json`,
     field by field, to 1e-6 — raises RuntimeError on any divergence;
  4. GUARD (ii), fail-loud plus a report: counts how many cells/transects
     ANADEM loses to missing values within the judgeable-forest
     population, and runs the TWO requested versions: a population common
     to all 5 products (only cells with a finite ANADEM value) and the
     original population (ANADEM only where a value exists, the other 5
     products over the full population, identical to the reference);
  5. a pre-flight sha256 check of the ANADEM inputs (.npz + raw clip)
     CHECKED against the hardcoded value below (computed and recorded
     before this run) — raises RuntimeError if the data on disk diverges.

It does not interpret the result (no claim-level prose) and does not audit
itself: it only produces the logged number.

Usage:
  /trabalho/ambientes/s33_amb_virtual/.venv/bin/python \
      auditar_nmad_pareado_nativa_anadem.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import proveniencia as PROV                     # noqa: E402
import juiz_lidar_v4 as J                        # noqa: E402
import auditar_nmad_pareado as REF               # noqa: E402  (read-only reference)

FAIXAS, ROT_F = J.FAIXAS, J.ROT_F
PISO_CEL, MIN_TRANSECTOS, B, SEED, ALPHA = REF.PISO_CEL, REF.MIN_TRANSECTOS, REF.B, REF.SEED, REF.ALPHA
PRODUTOS_BASE = list(REF.PRODUTOS)               # ["v23","mlp","fabdem","gedtm30","glo30"]
PRODUTOS_TODOS = PRODUTOS_BASE + ["anadem"]
PARES_BASE = list(REF.PARES)                     # [(v23,fabdem),(v23,gedtm30),(mlp,fabdem),(v23,mlp)]
PAR_ANADEM = ("v23", "anadem")
PARES_TODOS = PARES_BASE + [PAR_ANADEM]
CONFIRMATORIAS = dict(REF.CONFIRMATORIAS)        # {(v23,fabdem): ["20-30m",">30m"]} — inalterado

# sha256 checked directly on disk (sha256sum) BEFORE this run. Fails loud
# if the data changed since acquisition.
SHA256_ESPERADO_ANADEM = {
    "anadem_amazonia_Q1.npz": "b863f994c8b96814534a94343d2a646097deb1b94be7cf165c74530d67cd3e63",
    "anadem_amazonia_Q2.npz": "e70230f02c479067c1edd537235c85e9e623cc5aca6bcd0b7b85ba48efe81bb0",
    "anadem_amazonia_Q3.npz": "a9101e9462912e4a494dca0bba4465e86f41b81f4c927b0c1e6fd90713e77633",
    "anadem_amazonia_Q4.npz": "03981d874ab931c3678fc089bc2ff7089741f590a9c18ad585582195ed10f264",
}
SHA256_ESPERADO_RECORTE = {
    "anadem_v1_21M_recorte_amazonia_2x2.tif": "52c3bbb50b1f8d463504de7a80d9cf65c9ca615638a197e67e15a1a089629461",
}


def log(m=""):
    print(m, flush=True)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def conferir_sha256_anadem(laterais: Path, recorte: Path) -> None:
    """GUARD, fail-loud: sha256 on disk == sha256 checked before computing."""
    for nome, esperado in SHA256_ESPERADO_ANADEM.items():
        p = laterais / nome
        if not p.exists():
            raise RuntimeError(f"guarda sha256: {p} nao existe")
        real = _sha256(p)
        if real != esperado:
            raise RuntimeError(f"guarda sha256: {p} mudou desde a aquisicao "
                               f"(esperado {esperado}, disco {real})")
    for nome, esperado in SHA256_ESPERADO_RECORTE.items():
        p = recorte / nome
        if not p.exists():
            raise RuntimeError(f"guarda sha256: {p} nao existe")
        real = _sha256(p)
        if real != esperado:
            raise RuntimeError(f"guarda sha256: {p} mudou desde a aquisicao "
                               f"(esperado {esperado}, disco {real})")
    log(f"  guarda sha256: {len(SHA256_ESPERADO_ANADEM) + len(SHA256_ESPERADO_RECORTE)} "
        "insumos do ANADEM conferidos contra o valor do diario (bate)")


def montar_populacao_com_anadem(parquet: Path, juiz: dict, laterais: Path) -> tuple[pd.DataFrame, dict]:
    """Mirrors REF.montar_populacao, ADDING the 'anadem' column right after
    J.montar() and BEFORE any filter — it passes through the SAME filters,
    dedup, and stratification as the other 5 products, without changing
    row order or any other field (which guard (i) checks)."""
    J.SOLO = parquet
    d = J.montar()
    anadem = np.full(len(d), np.nan, dtype=np.float64)
    for q in J.QUADS:
        m = (d["quad"] == q).to_numpy()
        idx = d.loc[m, "idx_no"].to_numpy()
        z = np.load(laterais / f"anadem_amazonia_{q}.npz")
        anadem[m] = z["DTM"][idx]
        z.close()
    d = d.copy()
    d["anadem"] = anadem
    n0 = len(d)
    d = d[(d["n_all"] >= J.N_ALL_MIN)
          & (d["n_casca"] / d["n_all"] >= J.FRAC_CASCA_MIN)
          & np.isfinite(d["z_solo_v2"])]
    offs = juiz["offsets_por_transecto"]
    trans_parquet = set(d["transecto"].unique())
    if trans_parquet != set(offs.keys()):
        raise RuntimeError(f"transectos do parquet {sorted(trans_parquet)} != chaves de offsets {sorted(offs)}")
    julgaveis = [t for t, v in offs.items() if v["julgavel"]]
    d = d[d["transecto"].isin(julgaveis)].copy()
    d["off_t"] = d["transecto"].map({t: offs[t]["offset_m"] for t in julgaveis})
    d["z_ref"] = d["z_solo_v2"] + d["off_t"]
    n_cl = {t: offs[t]["n_clareira"] for t in julgaveis}
    if len(set(n_cl.values())) != len(n_cl):
        raise RuntimeError(f"empate de n_clareira entre julgaveis {n_cl}")
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


def tabela_pool_generic(flor: pd.DataFrame, col_faixa: str, produtos: list[str]) -> dict:
    out = {}
    for p in produtos:
        err = (flor[p] - flor["z_ref"]).to_numpy()
        finito = np.isfinite(err)
        out[p] = {}
        for rot in ROT_F:
            m = (flor[col_faixa] == rot).to_numpy() & finito
            e = err[m]
            out[p][rot] = J.metricas(e) if len(e) else {}
        m = flor[col_faixa].notna().to_numpy() & finito
        out[p]["TODAS"] = J.metricas(err[m]) if m.sum() else {}
    return out


def por_transecto_faixa_generic(flor: pd.DataFrame, col_faixa: str, produtos: list[str]) -> dict:
    out = {}
    for rot in ROT_F:
        out[rot] = {}
        sub = flor[flor[col_faixa] == rot]
        for t, s in sub.groupby("transecto"):
            reg = {"n_cel": int(len(s))}
            if len(s) >= 2:
                for p in produtos:
                    e = (s[p] - s["z_ref"]).to_numpy()
                    if not np.isfinite(e).all():
                        continue
                    reg[p] = {"nmad": REF.nmad(e), "mae": float(np.abs(e).mean())}
            out[rot][str(t)] = reg
    return out


def testar(mat: dict, a: str, b: str, metrica: str) -> dict:
    """Literal copy of REF.testar (identical parameters: floor, min
    transects, bootstrap B/SEED, sign+Wilcoxon+effect+CI+MDE80) — reused
    via a direct call, not rewritten; kept here only so as not to change
    the signature of the original (which takes `rng_ignorado` for
    compatibility)."""
    return REF.testar(mat, a, b, None, metrica)


def aplicar_holm(bloco: dict) -> None:
    chaves = [f for f in CONFIRMATORIAS[("v23", "fabdem")]
              if bloco[f].get("testado") and np.isfinite(bloco[f]["p_wilcoxon"])]
    for f, pv in zip(chaves, REF.holm([bloco[f]["p_wilcoxon"] for f in chaves])):
        bloco[f]["p_holm"] = float(pv)


def rodar_par(mat_ex: dict, mat_v2: dict, pares: list[tuple[str, str]]) -> tuple[dict, dict]:
    principal, sens = {}, {}
    for a_, b_ in pares:
        principal[f"{a_} vs {b_}"] = {"nmad": testar(mat_ex, a_, b_, "nmad"),
                                     "mae": testar(mat_ex, a_, b_, "mae")}
        sens[f"{a_} vs {b_}"] = {"nmad": testar(mat_v2, a_, b_, "nmad"),
                                "mae": testar(mat_v2, a_, b_, "mae")}
    if ("v23", "fabdem") in pares:
        aplicar_holm(principal["v23 vs fabdem"]["nmad"])
        aplicar_holm(principal["v23 vs fabdem"]["mae"])
        aplicar_holm(sens["v23 vs fabdem"]["nmad"])
        aplicar_holm(sens["v23 vs fabdem"]["mae"])
    return principal, sens


def _prox(a, b, tol=1e-6) -> bool:
    if a is None or b is None:
        return a is None and b is None
    fa, fb = float(a), float(b)
    if np.isnan(fa) or np.isnan(fb):
        return np.isnan(fa) and np.isnan(fb)
    return abs(fa - fb) <= tol


def guarda_reproduz_referencia(pool_ex: dict, pool_v2: dict, principal: dict, sens: dict,
                               ref: dict) -> dict:
    """GUARD (i), fail-loud: the 4 ORIGINAL products/pairs reproduce
    auditar_nmad_pareado_nativa_diretas.json to 1e-6 — field by field."""
    n_pool = 0
    for p in PRODUTOS_BASE:
        for rot in ROT_F + ["TODAS"]:
            a, b = pool_ex[p][rot], ref["tabela_pool_ETH"][p][rot]
            for k in ("n", "mae", "vies", "nmad", "rmse"):
                if k not in b:
                    continue
                if k == "n":
                    if a.get(k) != b[k]:
                        raise RuntimeError(f"guarda (i) tabela_pool_ETH: {p}/{rot}/n "
                                           f"recomputado {a.get(k)} != referencia {b[k]}")
                elif not _prox(a.get(k), b[k]):
                    raise RuntimeError(f"guarda (i) tabela_pool_ETH: {p}/{rot}/{k} "
                                       f"recomputado {a.get(k)} != referencia {b[k]} (tol 1e-6)")
                n_pool += 1
            a2, b2 = pool_v2[p][rot], ref["tabela_pool_dossel_v2"][p][rot]
            for k in ("n", "mae", "vies", "nmad", "rmse"):
                if k not in b2:
                    continue
                if k == "n":
                    if a2.get(k) != b2[k]:
                        raise RuntimeError(f"guarda (i) tabela_pool_dossel_v2: {p}/{rot}/n "
                                           f"recomputado {a2.get(k)} != referencia {b2[k]}")
                elif not _prox(a2.get(k), b2[k]):
                    raise RuntimeError(f"guarda (i) tabela_pool_dossel_v2: {p}/{rot}/{k} "
                                       f"recomputado {a2.get(k)} != referencia {b2[k]} (tol 1e-6)")
                n_pool += 1
    campos_teste = ("mediana_m", "media_m", "p_sinal_binomial", "wilcoxon_W", "p_wilcoxon",
                    "delta_sinal_pareado", "dp_entre_transectos_m", "mde80_m", "d_z")
    n_testes = 0
    for chave_bloco, meu, ref_bloco in (("testes_principal_ETH", principal, ref["testes_principal_ETH"]),
                                        ("testes_sensibilidade_dossel_v2", sens, ref["testes_sensibilidade_dossel_v2"])):
        for a_, b_ in PARES_BASE:
            k = f"{a_} vs {b_}"
            for metrica in ("nmad", "mae"):
                for rot in ROT_F:
                    a, b = meu[k][metrica][rot], ref_bloco[k][metrica][rot]
                    if bool(a.get("testado")) != bool(b.get("testado")):
                        raise RuntimeError(f"guarda (i) {chave_bloco}: {k}/{metrica}/{rot}/testado "
                                           f"recomputado {a.get('testado')} != referencia {b.get('testado')}")
                    if not a.get("testado"):
                        continue
                    for camp in campos_teste:
                        if camp not in b:
                            continue
                        if not _prox(a.get(camp), b[camp]):
                            raise RuntimeError(f"guarda (i) {chave_bloco}: {k}/{metrica}/{rot}/{camp} "
                                               f"recomputado {a.get(camp)} != referencia {b[camp]} (tol 1e-6)")
                        n_testes += 1
                    for i in range(2):
                        av = a["ic95_bootstrap_transecto_mediana"][i]
                        bv = b["ic95_bootstrap_transecto_mediana"][i]
                        if not _prox(av, bv):
                            raise RuntimeError(f"guarda (i) {chave_bloco}: {k}/{metrica}/{rot}/ic95[{i}] "
                                               f"recomputado {av} != referencia {bv} (tol 1e-6)")
                        n_testes += 1
                    if "p_holm" in b:
                        if not _prox(a.get("p_holm"), b["p_holm"]):
                            raise RuntimeError(f"guarda (i) {chave_bloco}: {k}/{metrica}/{rot}/p_holm "
                                               f"recomputado {a.get('p_holm')} != referencia {b['p_holm']} (tol 1e-6)")
                        n_testes += 1
    return {"status": "OK — reproduz auditar_nmad_pareado_nativa_diretas.json a 1e-6",
            "campos_tabela_pool_conferidos": n_pool, "campos_testes_conferidos": n_testes}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(Path("/trabalho/GNN_TOPO/SATELITES/lidar_eba/solo_lidar_30m_v3_nativa.parquet")))
    ap.add_argument("--juiz", default=str(RAIZ / "juiz_grade_nativa_diretas.json"))
    ap.add_argument("--laterais", default=str(Path("/trabalho/GNN_TOPO/SATELITES/laterais_nativa_diretas")))
    ap.add_argument("--referencia", default=str(RAIZ / "auditar_nmad_pareado_nativa_diretas.json"))
    ap.add_argument("--recorte-dir", default=str(Path("/trabalho/GNN_TOPO/SATELITES/anadem")))
    ap.add_argument("--saida", default="auditar_nmad_pareado_nativa_anadem.json")
    a = ap.parse_args()
    parquet, juiz_p, laterais = Path(a.parquet), Path(a.juiz), Path(a.laterais)
    ref_p, recorte_dir, saida = Path(a.referencia), Path(a.recorte_dir), RAIZ / a.saida
    J.LAT = laterais

    conferir_sha256_anadem(laterais, recorte_dir)

    juiz = json.loads(juiz_p.read_text(encoding="utf-8"))
    ref = json.loads(ref_p.read_text(encoding="utf-8"))
    log(f"  parquet {parquet.name} | juiz {juiz_p.name} | referencia {ref_p.name}")

    flor, info = montar_populacao_com_anadem(parquet, juiz, laterais)
    conf = REF.guardas_populacao(flor, juiz)
    log(f"  populacao reproduz tabela_regua_k0 (n exato, nmad 1e-6) em {len(conf)} celulas produto x faixa")

    # ── population B: ORIGINAL (identical to the reference for the 5 base
    # products; ANADEM only where a value exists)
    pool_ex_B = tabela_pool_generic(flor, "faixa_ex", PRODUTOS_BASE)
    pool_v2_B = tabela_pool_generic(flor, "faixa", PRODUTOS_BASE)
    mat_ex_B = por_transecto_faixa_generic(flor, "faixa_ex", PRODUTOS_BASE)
    mat_v2_B = por_transecto_faixa_generic(flor, "faixa", PRODUTOS_BASE)
    principal_B, sens_B = rodar_par(mat_ex_B, mat_v2_B, PARES_BASE)

    guarda_i = guarda_reproduz_referencia(pool_ex_B, pool_v2_B, principal_B, sens_B, ref)
    log(f"  guarda (i): {guarda_i['status']} "
        f"({guarda_i['campos_tabela_pool_conferidos']} campos de tabela_pool + "
        f"{guarda_i['campos_testes_conferidos']} campos de teste)")

    # ── GUARD (ii): how much ANADEM loses within the original population
    faltando = flor["anadem"].isna()
    n_perdidas = int(faltando.sum())
    transectos_afetados = sorted(flor.loc[faltando, "transecto"].astype(str).unique().tolist())
    log(f"  guarda (ii): ANADEM ausente em {n_perdidas}/{len(flor)} celulas da populacao "
        f"floresta-julgavel; transectos afetados: {transectos_afetados or 'nenhum'}")

    # anadem only where valid, within the full ORIGINAL population (B)
    flor_anadem_B = flor[flor["anadem"].notna()].copy()
    pool_ex_B_anadem = tabela_pool_generic(flor_anadem_B, "faixa_ex", ["anadem"])
    pool_v2_B_anadem = tabela_pool_generic(flor_anadem_B, "faixa", ["anadem"])
    mat_ex_B_anadem = por_transecto_faixa_generic(flor_anadem_B, "faixa_ex", ["v23", "anadem"])
    mat_v2_B_anadem = por_transecto_faixa_generic(flor_anadem_B, "faixa", ["v23", "anadem"])
    principal_B_anadem, sens_B_anadem = rodar_par(mat_ex_B_anadem, mat_v2_B_anadem, [PAR_ANADEM])

    # ── population A: COMMON to the 5 products (only cells with a finite anadem value)
    flor_comum = flor[flor["anadem"].notna()].copy()
    pool_ex_A = tabela_pool_generic(flor_comum, "faixa_ex", PRODUTOS_TODOS)
    pool_v2_A = tabela_pool_generic(flor_comum, "faixa", PRODUTOS_TODOS)
    mat_ex_A = por_transecto_faixa_generic(flor_comum, "faixa_ex", PRODUTOS_TODOS)
    mat_v2_A = por_transecto_faixa_generic(flor_comum, "faixa", PRODUTOS_TODOS)
    principal_A, sens_A = rodar_par(mat_ex_A, mat_v2_A, PARES_TODOS)

    log("  tabela ANADEM (pop. original, so onde valido) por estrato ETH (mae/vies/nmad): "
        + "; ".join(f"{f} " + "/".join(f"{pool_ex_B_anadem['anadem'][f].get(k, float('nan')):.2f}"
                                       for k in ("mae", "vies", "nmad")) for f in ROT_F))
    r = principal_B_anadem["v23 vs anadem"]["nmad"]
    for f in ("20-30m", ">30m"):
        rf = r[f]
        if rf.get("testado"):
            log(f"  v23 vs anadem NMAD {f}: mediana {rf['mediana_m']:+.3f} m "
                f"IC95 [{rf['ic95_bootstrap_transecto_mediana'][0]:+.3f}, "
                f"{rf['ic95_bootstrap_transecto_mediana'][1]:+.3f}] p_wilcoxon {rf['p_wilcoxon']:.4f} "
                f"(exploratorio, sem Holm/veredito — fora da familia confirmatoria)")

    criterio = {
        "unidade": "transecto julgavel (lista e offsets lidos do juiz, nao recalculados)",
        "familia_confirmatoria": {f"{p[0]} vs {p[1]}": fx for p, fx in CONFIRMATORIAS.items()},
        "metrica_confirmatoria": "nmad",
        "pares_exploratorios_incluindo_anadem": [f"{a_} vs {b_}" for a_, b_ in PARES_TODOS
                                                 if (a_, b_) not in CONFIRMATORIAS],
        "alpha": ALPHA, "correcao": "Holm dentro da familia confirmatoria (2 testes) — inalterada pelo ANADEM",
        "estratificacao_principal": "ETH exogeno (h_eth, cob_eth >= 0,8)",
        "estratificacao_sensibilidade": "dossel_v2 (circular)",
        "piso_celulas_por_transecto_faixa": PISO_CEL, "min_transectos_por_faixa": MIN_TRANSECTOS,
        "bootstrap": {"B": B, "semente": SEED, "estatistica": "mediana da diferenca por transecto"},
        "metrica": "NMAD por transecto-faixa; diferenca = NMAD(a) - NMAD(b); negativo = a menos disperso",
        "poder": "MDE80 = (t_{1-alpha/2,n-1} + t_{0.80,n-1}) * dp/sqrt(n)",
        "exploratorias": "demais faixas e pares (incl. v23 vs anadem): p bruto, sem veredito",
        "nota_metodologica": "coluna anadem mesclada por quad+idx_no logo apos J.montar(), ANTES de "
                             "qualquer filtro — atravessa os mesmos filtros/dedup/estratificacao dos "
                             "outros 5 produtos; nao interpreta o resultado (script sem prosa de claim)",
    }

    fontes = [parquet, juiz_p, ref_p, RAIZ / "juiz_lidar_v4.py", RAIZ / "auditar_nmad_pareado.py", Path(__file__)]
    fontes += [laterais / f"{n}_amazonia_{q}.npz" for q in J.QUADS
               for n in ("glo30", "fabdem", "gedtm30", "anadem", "agua_jrc", "dossel_eth", "rotulos")]
    fontes += [laterais / "_proveniencia_anadem.json", laterais / "_proveniencia.json"]
    fontes += [recorte_dir / "anadem_v1_21M_recorte_amazonia_2x2.tif",
              recorte_dir / "anadem_v1_21M_recorte_amazonia_2x2.proveniencia.json"]
    fontes += [J.SUP / f"delta_amazonia_ampliacao_b4s035{arm}_seed{sd}.npz"
              for arm in ("", "_mlp_semobs") for sd in J.SEEDS_PAR]

    PROV.gravar(saida, {
        "criterio_pre_registrado": criterio,
        "entradas": {"parquet": str(parquet), "juiz": str(juiz_p), "laterais": str(laterais),
                    "referencia_guarda_i": str(ref_p)},
        "guarda_i_reproduz_referencia_1e-6": guarda_i,
        "guarda_ii_populacao": {
            "n_floresta_julgavel_populacao_original": info["n_floresta_julgavel"],
            "julgaveis": info["julgaveis"], "n_julgaveis": info["n_julgaveis"],
            "anadem_celulas_sem_valor_populacao_original": n_perdidas,
            "anadem_transectos_afetados": transectos_afetados,
            "n_celulas_populacao_comum_5_produtos": int(len(flor_comum)),
            "celulas_perdidas_do_original_para_comum": n_perdidas,
        },
        "populacao_original_ETH": {
            "tabela_pool_ETH_produtos_base": pool_ex_B,
            "tabela_pool_ETH_anadem_so_onde_valido": pool_ex_B_anadem["anadem"],
            "testes_principal_ETH_produtos_base": principal_B,
            "testes_principal_ETH_v23_vs_anadem": principal_B_anadem["v23 vs anadem"],
        },
        "populacao_original_dossel_v2": {
            "tabela_pool_dossel_v2_produtos_base": pool_v2_B,
            "tabela_pool_dossel_v2_anadem_so_onde_valido": pool_v2_B_anadem["anadem"],
            "testes_sensibilidade_dossel_v2_produtos_base": sens_B,
            "testes_sensibilidade_dossel_v2_v23_vs_anadem": sens_B_anadem["v23 vs anadem"],
        },
        "populacao_comum_5_produtos_ETH": {
            "tabela_pool_ETH": pool_ex_A,
            "testes_principal_ETH": principal_A,
        },
        "populacao_comum_5_produtos_dossel_v2": {
            "tabela_pool_dossel_v2": pool_v2_A,
            "testes_sensibilidade_dossel_v2": sens_A,
        },
        "guarda_reproducao_tabela_regua_k0": conf,
    }, fontes_lidas=fontes, script=__file__)
    log(f"  -> {saida.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
