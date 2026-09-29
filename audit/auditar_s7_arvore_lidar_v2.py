"""S7 (v2) -- ensemble-of-trees arm against the lidar reference, O3 criterion.

WHY v2. The original script reads the training manifest
`treinar_arvore_persistir.json` and requires status OK; that file is the
evidence of a stop triggered by guard g1 (FALHOU_G1) and is not
overwritten. This version reads the manifest of a re-run with a
recalibrated g1' (`treinar_arvore_persistir_v2.json`). The ONLY changes
from the original script: (1) MANIFESTO_TREINO points to the v2 manifest;
(2) guard g0 also requires g1'(a) and g1'(b) to pass in the v2 manifest;
(3) the output JSON records g1' and cites the v2 manifest under
`diferencas_declaradas`. Arithmetic, guards g0/g2, contrasts, O3 and
admissibility are identical to the original (see diff).

QUESTION. Does the third regressor family (tree ensemble) change the
ceiling (O)? The earlier ceiling only covered v23 vs mlp (1.093 m above 30
m; 0.356 m between 20 and 30 m) because the tree's prediction had not been
persisted over the full grid. `treinar_arvore_persistir.py` persisted the
tree's Delta over the whole grid, in the same format as the networks. This
script puts the tree on the SAME ruler.

REUSE (nothing reimplemented; the modules are READ, not edited):
  juiz_lidar_v4 (J)                       montar() -- population and the networks' products
  auditar_nmad_pareado (REF)              montar_populacao, guardas_populacao, testar
  auditar_nmad_pareado_nativa_anadem (ANA) tabela_pool_generic, por_transecto_faixa_generic,
                                          rodar_par, guarda_reproduz_referencia
  auditar_rotulo_regua_v2 (ARV2)          contrastes_lidar (the earlier ceiling)
  auditar_admissibilidade (ADM)           por_corrida, resumo
  b2_transferencia._delta_sob_dossel and raiox_geomorfologico.derivadas/c1_admissibilidade
                                          (the same functions used for the networks' admissibility)

THE "tree" COLUMN. Same code path as J.montar for v23/mlp: a float64
accumulator sums the Delta of the 5 seeds in J.SEEDS_PAR order, the product
is glo30(J.LAT) - acc/5, inserted through a wrapper around J.montar BEFORE
any filter (same as ANADEM).

GUARDS (fail-loud, BEFORE any tree number):
  (g0) sha256 of the hash-stamped inputs checked against the value fixed
       below; sha256 of each tree npz checked against its sidecar and the
       training manifest.
  (g2-i)  the ANADEM guard (i) in full: tabela_pool (ETH and dossel_v2) and
          the tests for the 5 products / 4 original pairs reproduce
          auditar_nmad_pareado_nativa_diretas.json to 1e-6.
  (g2-ii) the v23 x mlp ceiling from ARV2.contrastes_lidar over the
          recomputed tests reproduces
          auditar_rotulo_regua_v2.json tabela_orcamento_P1[c].contrastes_lidar_detalhe
          to 1e-6 (20-30m: 0.356; >30m: 1.093).

O3 CRITERION (fixed by design beforehand; copied unchanged):
  Ceiling = the largest upper bound of |dMAE| among the three family pairs
  (v23 x mlp, tree x v23, tree x mlp), within the class and unit. Above 30
  m: if the lower bounds of A (mean and median) and of the equivalent MAE
  reduction are all above the ceiling in both units -> "demonstrated for
  the three families". Otherwise -> "not demonstrated against the tree",
  and the headline claim is not established.

USAGE
  CUDA_VISIBLE_DEVICES="" V23_LATDIR=/trabalho/GNN_TOPO/SATELITES/laterais \
  /trabalho/ambientes/s33_amb_virtual/.venv/bin/python auditar_s7_arvore_lidar.py
  ... auditar_s7_arvore_lidar.py --so-g2     # only guards g0/g2 for the networks, nothing written
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

LATDIR_TREINO = "/trabalho/GNN_TOPO/SATELITES/laterais"
if os.environ.get("V23_LATDIR") != LATDIR_TREINO:
    raise SystemExit(f"V23_LATDIR tem de ser {LATDIR_TREINO} (glo30 e regime da "
                     f"admissibilidade, como no _raiox_da_corrida das redes)")
if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
    raise SystemExit('CUDA_VISIBLE_DEVICES="" e obrigatorio (nada toca GPU)')

import proveniencia as PROV                     # noqa: E402
import juiz_lidar_v4 as J                        # noqa: E402
import auditar_nmad_pareado as REF               # noqa: E402
import auditar_nmad_pareado_nativa_anadem as ANA  # noqa: E402
import auditar_rotulo_regua_v2 as ARV2           # noqa: E402
import auditar_admissibilidade as ADM            # noqa: E402

PARQUET = Path("/trabalho/GNN_TOPO/SATELITES/lidar_eba/solo_lidar_30m_v3_nativa.parquet")
LAT_NATIVA = Path("/trabalho/GNN_TOPO/SATELITES/laterais_nativa_diretas")
JUIZ = RAIZ / "juiz_grade_nativa_diretas.json"
REF_JSON = RAIZ / "auditar_nmad_pareado_nativa_diretas.json"
REGUA_JSON = RAIZ / "auditar_rotulo_regua_v2.json"
ADM_JSON = RAIZ / "auditar_admissibilidade.json"
MANIFESTO_TREINO = RAIZ / "treinar_arvore_persistir_v2.json"   # reproduction guard g1' (mean and per-seed tolerance)
PREFIXO_TREE = "delta_amazonia_arvore_f2_s7"
SAIDA = RAIZ / "auditar_s7_arvore_lidar.json"
SAIDA_TESTEMUNHAS = RAIZ / "auditar_s7_arvore_lidar_testemunhas_raiox.json"

SHA_FIXADO = {   # checked with sha256sum before running (pre-registered)
    REGUA_JSON.name: "2470115aaf6eed5ff2ea7d878f2dd6a1ac3f7bf682240599102a33fb01129e9a",
    REF_JSON.name: "bb3cce3e12470fb2da78fb3ebcc25fb3beeffee7a10c637726eaf3f7e6f00f2d",
    ADM_JSON.name: "b7ece3993e06263502771234d707a32c911370858bf6ce02e66f6e5082273b86",
    JUIZ.name: "5d8e47397eac39122d7d9b52e3118491c4547cffa6c856e4477d0fe09c6d76f5",
    PARQUET.name: "b01a6101850dfe7fc514c85be556fbe0aae3cc941fba9263b431a54556f92dc5",
}
CLASSES_O3 = ["20-30m", ">30m"]
UNIDADES = ["transecto", "pegada"]
PARES_S7 = [("v23", "mlp"), ("tree", "v23"), ("tree", "mlp")]
PRODUTOS_S7 = ["v23", "mlp", "tree"]
TOL = 1e-6


def log(m=""):
    print(time.strftime("[%H:%M:%S] ") + str(m), flush=True)


def npz_tree(sd: int) -> Path:
    return RAIZ / f"{PREFIXO_TREE}_seed{sd}.npz"


# ------------------------------------------------------------------ g0
def conferir_sha(com_tree: bool) -> dict:
    out = {}
    for p in (REGUA_JSON, REF_JSON, ADM_JSON, JUIZ, PARQUET):
        h = PROV.sha256(p)
        if h != SHA_FIXADO[p.name]:
            raise RuntimeError(f"g0: {p.name} sha256 {h} != fixado {SHA_FIXADO[p.name]}")
        out[p.name] = h
    if com_tree:
        man = json.loads(MANIFESTO_TREINO.read_text(encoding="utf-8"))
        if man.get("status") != "OK":
            raise RuntimeError(f"g0: manifesto do treino com status {man.get('status')}")
        g1a = man["g1_linha_a"]                     # v2: g1' approved in the v2 manifest
        if not (g1a["passou_a"] and g1a["passou_b_todas"]):
            raise RuntimeError(f"g0: g1' nao aprovada no manifesto v2: {g1a}")
        for sd in J.SEEDS_PAR:
            p = npz_tree(sd)
            side = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
            h = PROV.sha256(p)
            if h != side["sha256_npz"] or h != man["artefatos"][str(sd)]["sha256_npz"]:
                raise RuntimeError(f"g0: {p.name} sha256 {h} != sidecar/manifesto")
            out[p.name] = h
    log(f"  g0: sha256 conferido em {len(out)} entradas")
    return out


# ------------------------------------------------------- coluna "tree"
def instalar_tree_em_montar() -> None:
    """Wrapper around J.montar: adds 'tree' BEFORE any filter, using the SAME
    arithmetic that J.montar uses for v23/mlp."""
    original = J.montar

    def montar_com_tree():
        d = original()
        tree = np.full(len(d), np.nan, dtype=np.float64)
        for q in J.QUADS:
            m = (d["quad"] == q).to_numpy()
            if not m.any():
                continue
            idx = d.loc[m, "idx_no"].to_numpy()
            glo = np.load(J.LAT / f"glo30_amazonia_{q}.npz")["DEM"][idx]
            acc = np.zeros(len(idx))
            for sd in J.SEEDS_PAR:
                z = np.load(npz_tree(sd))
                acc += z[q][idx]
                z.close()
            tree[m] = glo - acc / len(J.SEEDS_PAR)
        d = d.copy()
        d["tree"] = tree
        return d

    J.montar = montar_com_tree


# ------------------------------------------------------------------ g2
def guarda_g2(flor, ref: dict, regua: dict) -> tuple[dict, dict]:
    pool_ex = ANA.tabela_pool_generic(flor, "faixa_ex", ANA.PRODUTOS_BASE)
    pool_v2 = ANA.tabela_pool_generic(flor, "faixa", ANA.PRODUTOS_BASE)
    mat_ex = ANA.por_transecto_faixa_generic(flor, "faixa_ex", ANA.PRODUTOS_BASE)
    mat_v2 = ANA.por_transecto_faixa_generic(flor, "faixa", ANA.PRODUTOS_BASE)
    principal, sens = ANA.rodar_par(mat_ex, mat_v2, ANA.PARES_BASE)
    g_i = ANA.guarda_reproduz_referencia(pool_ex, pool_v2, principal, sens, ref)
    log(f"  g2-i: {g_i['status']} ({g_i['campos_tabela_pool_conferidos']} campos de pool + "
        f"{g_i['campos_testes_conferidos']} de teste)")
    g_ii = {}
    for c in CLASSES_O3:
        linhas, _ = ARV2.contrastes_lidar({"testes_principal_ETH": principal}, c)
        meu = next(r for r in linhas if r["par"] == "v23 vs mlp")
        alvo = next(r for r in regua["tabela_orcamento_P1"][c]["contrastes_lidar_detalhe"]
                    if r["par"] == "v23 vs mlp")
        dif = meu["teto_abs_ddiferenca_m"] - alvo["teto_abs_ddiferenca_m"]
        ok = abs(dif) <= TOL
        g_ii[c] = {"recomputado_m": meu["teto_abs_ddiferenca_m"],
                   "carimbado_m": alvo["teto_abs_ddiferenca_m"], "diferenca_m": dif,
                   "tolerancia_m": TOL, "passou": bool(ok)}
        log(f"  g2-ii {c}: teto v23 x mlp recomputado {meu['teto_abs_ddiferenca_m']:.9f} "
            f"carimbado {alvo['teto_abs_ddiferenca_m']:.9f} dif {dif:.2e} -> "
            f"{'PASSOU' if ok else 'FALHOU'}")
        if not ok:
            raise RuntimeError(f"GUARDA g2-ii FALHOU em {c}: dif {dif} > {TOL}. PARANDO.")
    return {"g2_i": g_i, "g2_ii": g_ii}, principal


# ------------------------------------------------------------ contrastes
def contrastes_por_unidade(flor, mapa_pegada: dict) -> tuple[dict, dict]:
    if flor["tree"].isna().any() or not np.isfinite(flor["tree"].to_numpy()).all():
        raise RuntimeError("coluna tree com valor nao finito na populacao floresta-julgavel")
    flor_p = flor.copy()
    flor_p["transecto"] = flor["transecto"].astype(str).map(mapa_pegada)
    if flor_p["transecto"].isna().any():
        faltam = sorted(flor.loc[flor_p["transecto"].isna(), "transecto"].astype(str).unique())
        raise RuntimeError(f"transectos sem pegada no mapa: {faltam}")
    mats = {"transecto": ANA.por_transecto_faixa_generic(flor, "faixa_ex", PRODUTOS_S7),
            "pegada": ANA.por_transecto_faixa_generic(flor_p, "faixa_ex", PRODUTOS_S7)}
    res = {}
    for u, mat in mats.items():
        res[u] = {}
        for a_, b_ in PARES_S7:
            res[u][f"{a_} vs {b_}"] = {"mae": REF.testar(mat, a_, b_, None, "mae"),
                                       "nmad": REF.testar(mat, a_, b_, None, "nmad")}
    return res, mats


def teto_do_no(no: dict) -> float | None:
    if not no.get("testado"):
        return None
    lo, hi = no["ic95_bootstrap_transecto_mediana"]
    return float(max(abs(lo), abs(hi)))


def criterio_O3(res: dict, regua: dict) -> dict:
    out = {}
    for c in CLASSES_O3:
        por_u = {}
        ok_geral = True
        falha_ja_com_duas = False
        for u in UNIDADES:
            tetos = {par: teto_do_no(res[u][par]["mae"][c]) for par in res[u]}
            nao_testados = [p for p, t in tetos.items() if t is None]
            testados = {p: t for p, t in tetos.items() if t is not None}
            teto = max(testados.values()) if testados else None
            par_teto = max(testados, key=testados.get) if testados else None
            pu = regua["por_classe_eth"][c]["por_unidade"][u]
            lo_media = pu["A"]["ic95_pond_celula"][0]
            lo_mediana = pu["A"]["ic95_mediana"][0]
            lo_red = pu["reducao_mae_equivalente"]["ic95_pond_celula"][0]
            chk = {"lim_inf_A_media_m": lo_media, "lim_inf_A_mediana_m": lo_mediana,
                   "lim_inf_reducao_mae_m": lo_red}
            if teto is None or nao_testados:
                ok_u = False
                chk.update(acima_do_teto_media=None, acima_do_teto_mediana=None,
                           acima_do_teto_reducao=None)
            else:
                chk.update(acima_do_teto_media=bool(lo_media > teto),
                           acima_do_teto_mediana=bool(lo_mediana > teto),
                           acima_do_teto_reducao=bool(lo_red > teto))
                ok_u = all((chk["acima_do_teto_media"], chk["acima_do_teto_mediana"],
                            chk["acima_do_teto_reducao"]))
            t2 = tetos.get("v23 vs mlp")
            ok_so_duas = (t2 is not None and lo_media > t2 and lo_mediana > t2 and lo_red > t2)
            falha_ja_com_duas = falha_ja_com_duas or not ok_so_duas
            por_u[u] = {"tetos_por_par_m": tetos, "pares_nao_testados": nao_testados,
                        "teto_O3_m": teto, "par_que_fixa_o_teto": par_teto,
                        **chk, "passa_na_unidade": bool(ok_u),
                        "passaria_so_com_v23_x_mlp_nesta_unidade": bool(ok_so_duas)}
            ok_geral = ok_geral and ok_u
        # descriptive reading: the TRANSECT ceiling applied to both units
        t_tr = por_u["transecto"]["teto_O3_m"]
        desc = {}
        for u in UNIDADES:
            pu = regua["por_classe_eth"][c]["por_unidade"][u]
            vals = [pu["A"]["ic95_pond_celula"][0], pu["A"]["ic95_mediana"][0],
                    pu["reducao_mae_equivalente"]["ic95_pond_celula"][0]]
            desc[u] = bool(t_tr is not None and all(v > t_tr for v in vals))
        out[c] = {
            "por_unidade": por_u,
            "veredito_O3": ("demonstrado para as tres familias" if ok_geral
                            else "nao demonstrado contra a arvore"),
            "falha_ocorreria_ja_so_com_v23_x_mlp": bool(falha_ja_com_duas),
            "descritivo_leitura_28_09_teto_do_transecto_nas_duas_unidades": {
                "teto_m": t_tr, "passa_por_unidade": desc,
                "nota": "so descritivo; o veredito O3 usa o teto na classe E na unidade (texto do desenho)"},
        }
    return out


# --------------------------------------------------------- admissibilidade
def admissibilidade_tree() -> dict:
    """The same functions that recorded G_raiox for the networks, applied
    WITHOUT adaptation to each tree seed's Delta; read by
    ADM.por_corrida/resumo unchanged."""
    import b2_transferencia as B2
    import raiox_geomorfologico as RX
    corrida = {}
    for sd in J.SEEDS_PAR:
        g = {}
        z_npz = np.load(npz_tree(sd))
        for q in J.QUADS:
            delta = z_npz[q]
            dsm = np.load(B2.LATDIR_V23 / f"glo30_amazonia_{q}.npz")["DEM"].astype(np.float32)
            z = (dsm - delta).astype(np.float32)
            slope, _aspect = RX.derivadas("amazonia", q, z)
            g[q] = {"E_admissibilidade": RX.c1_admissibilidade("amazonia", q, z, slope),
                    "frac_delta_negativo": round(float((delta < 0).mean()), 6),
                    "delta_mediano_m": round(float(np.median(delta)), 4),
                    "F_dossel": B2._delta_sob_dossel("amazonia", q, delta)}
            del dsm, z, slope, _aspect
        z_npz.close()
        corrida[f"arvore_f2_s7_seed{sd}"] = {"seed": sd, "G_raiox": g}
        log(f"  admissibilidade seed {sd}: " + "; ".join(
            f"{q} viol={g[q]['F_dossel'].get('n_violacoes_dossel', 0)}" for q in J.QUADS))
    SAIDA_TESTEMUNHAS.write_text(json.dumps(corrida, indent=2, ensure_ascii=False,
                                            default=float), encoding="utf-8")
    runs = ADM.por_corrida(SAIDA_TESTEMUNHAS)
    adm_ref = json.loads(ADM_JSON.read_text(encoding="utf-8"))
    return {"instrumentada": True,
            "como": "b2_transferencia._delta_sob_dossel + raiox_geomorfologico.derivadas/"
                    "c1_admissibilidade sobre z = glo30(laterais) - Delta de cada semente "
                    "(o mesmo que _raiox_da_corrida faz para as redes), lido por "
                    "auditar_admissibilidade.por_corrida/resumo sem mudanca",
            "testemunhas": SAIDA_TESTEMUNHAS.name,
            "arvore_por_semente": {str(k): v for k, v in runs.items()},
            "arvore_resumo": ADM.resumo(runs),
            "redes_mesmo_ponto_b4s035 (auditar_admissibilidade.json, lote_adotado_epocas_x1)":
                adm_ref["tabela"]["lote_adotado_epocas_x1"],
            "nota": "as redes sao b4s035 (TV 0,35); a arvore e suavidade 0 (contrato do braco)"}


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--so-g2", action="store_true",
                    help="so g0/g2 das redes; nao le a arvore e nao grava nada")
    a = ap.parse_args()
    t0 = time.time()
    J.LAT = LAT_NATIVA
    com_tree = not a.so_g2

    shas = conferir_sha(com_tree)
    juiz = json.loads(JUIZ.read_text(encoding="utf-8"))
    ref = json.loads(REF_JSON.read_text(encoding="utf-8"))
    regua = json.loads(REGUA_JSON.read_text(encoding="utf-8"))
    if com_tree:
        instalar_tree_em_montar()

    flor, info = REF.montar_populacao(PARQUET, juiz)
    conf = REF.guardas_populacao(flor, juiz)
    log(f"  populacao: {info['n_floresta_julgavel']} celulas floresta-julgavel, "
        f"{info['n_julgaveis']} transectos; tabela_regua_k0 reproduzida em {len(conf)} celulas")

    guardas, principal = guarda_g2(flor, ref, regua)
    if a.so_g2:
        log(f"  --so-g2: guardas das redes PASSARAM; nada gravado ({time.time()-t0:.0f} s)")
        return 0

    mapa_pegada = regua["populacao"]["unidade_pegada"]["mapa"]
    res, mats = contrastes_por_unidade(flor, mapa_pegada)
    # consistency check: v23 x mlp on the transect == the guard's value (same function, same data)
    for c in CLASSES_O3:
        a1 = res["transecto"]["v23 vs mlp"]["mae"][c]["ic95_bootstrap_transecto_mediana"]
        a2 = principal["v23 vs mlp"]["mae"][c]["ic95_bootstrap_transecto_mediana"]
        if max(abs(a1[0] - a2[0]), abs(a1[1] - a2[1])) > TOL:
            raise RuntimeError(f"inconsistencia v23 x mlp transecto {c}: {a1} != {a2}")
    pool_tree = ANA.tabela_pool_generic(flor, "faixa_ex", PRODUTOS_S7)
    o3 = criterio_O3(res, regua)
    for c in CLASSES_O3:
        for u in UNIDADES:
            pu = o3[c]["por_unidade"][u]
            log(f"  {c} {u}: tetos {{{', '.join(f'{k}: {v:.3f}' if v is not None else f'{k}: n/t' for k, v in pu['tetos_por_par_m'].items())}}} "
                f"-> teto {pu['teto_O3_m']} ({pu['par_que_fixa_o_teto']}) | A lo {pu['lim_inf_A_media_m']:.3f} "
                f"med lo {pu['lim_inf_A_mediana_m']:.3f} red lo {pu['lim_inf_reducao_mae_m']:.3f} "
                f"-> {'passa' if pu['passa_na_unidade'] else 'NAO passa'}")
        log(f"  O3 {c}: {o3[c]['veredito_O3']}")

    adm = admissibilidade_tree()

    fontes = [PARQUET, JUIZ, REF_JSON, REGUA_JSON, ADM_JSON, MANIFESTO_TREINO,
              RAIZ / "treinar_arvore_persistir.json",          # v2: the stop manifest (evidence)
              RAIZ / "juiz_lidar_v4.py", RAIZ / "auditar_nmad_pareado.py",
              RAIZ / "auditar_nmad_pareado_nativa_anadem.py", RAIZ / "auditar_rotulo_regua_v2.py",
              RAIZ / "auditar_admissibilidade.py", RAIZ / "b2_transferencia.py",
              RAIZ / "raiox_geomorfologico.py", Path(__file__)]
    fontes += [npz_tree(sd) for sd in J.SEEDS_PAR]
    fontes += [npz_tree(sd).with_suffix(".json") for sd in J.SEEDS_PAR]
    fontes += [J.SUP / f"delta_amazonia_ampliacao_b4s035{arm}_seed{sd}.npz"
               for arm in ("", "_mlp_semobs") for sd in J.SEEDS_PAR]
    fontes += [LAT_NATIVA / f"{n}_amazonia_{q}.npz" for q in J.QUADS
               for n in ("glo30", "fabdem", "gedtm30", "agua_jrc", "dossel_eth", "rotulos")]
    fontes += [Path(LATDIR_TREINO) / f"glo30_amazonia_{q}.npz" for q in J.QUADS]

    prov = PROV.bloco(__file__)
    prov["comando"] = " ".join([sys.executable] + sys.argv)
    prov["ambiente"] = {k: os.environ.get(k) for k in ("V23_LATDIR", "CUDA_VISIBLE_DEVICES")}
    corpo = {
        "_proveniencia": prov,
        "_fontes": PROV.fontes(fontes),
        "desenho": "Pre-declared design note, sections S7 and 'S7 -- resumption'",
        "criterio_O3_literal": ("Ceiling = largest upper bound of |dMAE| among the three family pairs "
                                "(v23 x mlp, tree x v23, tree x mlp), in the class and in the unit. Above "
                                "30 m: if the lower bounds of A (mean and median) and of the equivalent "
                                "MAE reduction lie above the ceiling in both units -> 'demonstrado para "
                                "as tres familias' (demonstrated for the three families). Otherwise -> "
                                "'nao demonstrado contra a arvore' (not demonstrated against the tree), "
                                "and the title claim must be revisited."),
        "operacionalizacao": {
            "limite_superior_abs_dMAE": "max(|lo|,|hi|) do IC95 bootstrap da mediana da diferenca de MAE "
                                        "por unidade (auditar_nmad_pareado.testar; = contrastes_lidar de 28/09)",
            "bootstrap": {"B": REF.B, "semente": REF.SEED, "cluster": "unidade (transecto ou pegada)",
                          "fluxo": "SeedSequence([42, sha256(a|b|metrica|faixa)[:4]])"},
            "piso": {"celulas_por_unidade_faixa": REF.PISO_CEL, "min_unidades": REF.MIN_TRANSECTOS},
            "estratificacao": "ETH exogeno (h_eth, cob_eth >= 0,8), faixa_ex",
            "unidade_pegada": "mapa de auditar_rotulo_regua_v2.json populacao.unidade_pegada.mapa; "
                              "celulas dos transectos da mesma pegada agrupadas antes de medir MAE/NMAD",
            "limites_inferiores_A_e_reducao": "auditar_rotulo_regua_v2.json por_classe_eth[c].por_unidade[u] "
                                              "(A.ic95_pond_celula[0], A.ic95_mediana[0], "
                                              "reducao_mae_equivalente.ic95_pond_celula[0]); inalterados",
            "par_nao_testavel": "nao demonstrado (nao se demonstra contra o que nao foi medido)",
            "orientacao_dos_pares": "diferenca = metrica(a) - metrica(b); negativo = a com menor erro",
        },
        "produto_tree": "glo30(laterais_nativa_diretas) - media das 5 sementes de Delta da arvore "
                        "(delta_amazonia_arvore_f2_s7_seed*.npz), aritmetica de juiz_lidar_v4.montar",
        "conferencia_sha256": shas,
        "g1_linha_treino": json.loads(MANIFESTO_TREINO.read_text(encoding="utf-8"))["g1_linha_a"],
        "guardas": {**guardas, "populacao_tabela_regua_k0": f"{len(conf)} celulas produto x faixa reproduzidas"},
        "populacao": info,
        "tabela_pool_ETH_v23_mlp_tree": pool_tree,
        "contrastes_lidar": res,
        "por_unidade_faixa_ETH": mats,
        "O3": o3,
        "admissibilidade": adm,
        "diferencas_declaradas": [
            "redes b4s035 (boost 4, TV 0,35); arvore boost 4, suavidade 0 (contrato do braco de arvores)",
            ("tree retrained on 28/09 on Ubuntu (xgboost 3.2.0) with the config from "
             "f2_arvores.json (42/123/7/2024 in the 17:53-18:02 training run, 31 in "
             "the resumed run); g1' (mean <= 0.002, each seed <= 0.0092, recalibrated "
             "before the resumed run) checked in treinar_arvore_persistir_v2.json; the "
             "original g1 (0.002 per seed) failed on seed 31 (+0.00275; "
             "treinar_arvore_persistir.json, FALHOU_G1)"),
            "reducao de MAE equivalente e A sao os de 28/09 (e_modelo do v23), inalterados; so o teto muda",
        ],
        "contra_auditoria": "PENDENTE -- nao feita por quem escreveu este script",
    }
    SAIDA.write_text(json.dumps(corpo, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    log(f"  -> {SAIDA.name} ({time.time()-t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
