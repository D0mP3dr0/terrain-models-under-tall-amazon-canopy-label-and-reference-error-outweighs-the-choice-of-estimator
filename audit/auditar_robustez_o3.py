"""R-O3 -- robustness of the THREE-family (O3) ceiling.

QUESTION: do the three robustness checks (three vertical registrations,
the 2011-2023 forest-loss mask, the decoupled lidar canopy class, and
agreeing cells), which today are judged against the two-network ceiling,
still hold against the O3 (three-family) ceiling?

REUSE (modules READ via import, not edited; no arithmetic rewritten):
  auditar_s7_arvore_lidar_v2 (S7): conferir_sha, instalar_tree_em_montar,
      contrastes_por_unidade, teto_do_no, criterio_O3
  auditar_dt3_criterios_registro (DT3): carregar_base, aplicar_registro,
      estatisticas, z_ref_registro, comparar, normal
  auditar_s8_mascara_disturbio (S8): construir_mascaras, marcar, por_unidade_A
  auditar_s1_estratificacao_lidar (S1, v1) / _v2 (S1V2): recomputar_populacao_e_guardas,
      montar_grade_vizinhos, classificar_lnb_e_horn, estatisticas_braco; dump v2
  auditar_rotulo_regua_v2 (ARV2): veredito_O
  auditar_nmad_pareado (REF), auditar_nmad_pareado_nativa_anadem (ANA): populations
The only line not imported: faixa_C = label when ETH == Lnb (np.where,
identical to S1v2's main), checked by guard (b).

CONDITIONS: C0 GLO-30/no mask/ETH (base); C1 registration ii-b; C2
registration iii; C3 Hansen 2011-2023 mask; C4 decoupled lidar canopy
class (Lnb); C5 agreeing cells (C).

GUARDS (fail loud; any failure -> RuntimeError, nothing is written):
  (g0)  sha256 of S7's inputs (and of the tree's npz files against the
        v2 sidecar/manifest).
  (a)   C0: S7.criterio_O3 reproduces auditar_s7_arvore_lidar.json's O3
        to 1e-9 (per-pair ceilings, ceiling, pair, bounds, per-unit pass,
        verdict).
  (b)   C0..C5: the v23-vs-mlp ceiling (transect; and footprint in D-T3)
        reproduces the source artifact to 1e-6, and ARV2.veredito_O with
        it reproduces the source (O) verdict.
  (A)   the condition's A and reduction bounds reproduce the source
        artifact to 1e-9.

CRITERION (O3 design): above 30 m, "demonstrated" if the lower bounds of
A (mean and median) and of the MAE reduction sit above the condition's O3
ceiling. Primary: ARV2.veredito_O(condition's A, transect O3 ceiling).
Reading: "robustness covers all three families" only if demonstrated in
EVERY condition C1-C5 where the two-network (O) verdict above 30 m is
demonstrated today.

Usage:
  cd /trabalho/GNN_TOPO/V23 && CUDA_VISIBLE_DEVICES="" \
    V23_LATDIR=/trabalho/GNN_TOPO/SATELITES/laterais PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=12 taskset -c 0-11 \
    /trabalho/ambientes/s33_amb_virtual/.venv/bin/python auditar_robustez_o3.py \
    2>&1 | tee auditar_robustez_o3.log
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

LATDIR_TREINO = "/trabalho/GNN_TOPO/SATELITES/laterais"
if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
    raise SystemExit('CUDA_VISIBLE_DEVICES="" e obrigatorio (nada toca GPU)')
if os.environ.get("V23_LATDIR") != LATDIR_TREINO:
    raise SystemExit(f"V23_LATDIR tem de ser {LATDIR_TREINO} (exigido pelo import do S7 v2)")

import proveniencia as PROV                              # noqa: E402
import juiz_lidar_v4 as J                                # noqa: E402
import auditar_nmad_pareado as REF                       # noqa: E402
import auditar_nmad_pareado_nativa_anadem as ANA         # noqa: E402
import auditar_rotulo_regua_v2 as ARV2                   # noqa: E402
import auditar_dt3_criterios_registro as DT3             # noqa: E402
import auditar_s8_mascara_disturbio as S8                # noqa: E402
import auditar_s1_estratificacao_lidar as S1             # noqa: E402
import auditar_s1_estratificacao_lidar_v2 as S1V2        # noqa: E402
import auditar_s7_arvore_lidar_v2 as S7                  # noqa: E402

S7_JSON = RAIZ / "auditar_s7_arvore_lidar.json"
DT3_JSON = RAIZ / "auditar_dt3_criterios_registro.json"
S8_JSON = RAIZ / "auditar_s8_mascara_disturbio.json"
S1V2_JSON = RAIZ / "auditar_s1_estratificacao_lidar_v2.json"
SAIDA = RAIZ / "auditar_robustez_o3.json"

CLASSES = ["20-30m", ">30m"]
UNIDADES = ["transecto", "pegada"]
TOL_A, TOL_B, TOL_A_LADO = 1e-9, 1e-6, 1e-9
TETO_S7_LITERAL = {"20-30m": 0.500, ">30m": 1.442}
DEMONSTRADO = "sustentado"          # label used by ARV2.veredito_O


def log(m=""):
    print(time.strftime("[%H:%M:%S] ") + str(m), flush=True)


def f3(v, sinal=False) -> str:
    """Formatting for the log only; None does not abort the single run."""
    if v is None:
        return "n/t"
    return f"{v:+.3f}" if sinal else f"{v:.3f}"


def cmp_ou_para(meu, alvo, tol, rotulo) -> dict:
    erros, cont = [], [0, 0.0]
    DT3.comparar(DT3.normal(meu), alvo, tol, rotulo, erros, cont)
    if erros:
        raise RuntimeError(f"GUARDA {rotulo} FALHOU (tol {tol}): " + "; ".join(erros[:12])
                           + f" ... ({len(erros)} erros). PARANDO.")
    return {"campos": cont[0], "max_abs_dif": cont[1], "tolerancia": tol, "passou": True}


# ------------------------------------------------------------- one condition
def avaliar_condicao(nome: str, res: dict, pu: dict, teto2_origem: dict,
                     vered2_origem: dict, origem: str) -> dict:
    """res: S7.contrastes_por_unidade for the condition; pu: {class: por_unidade}
    (the condition's A and reduction); teto2_origem: {class: {unit: ceiling}}
    from the source artifact (transect always; footprint only where the
    artifact records it); vered2_origem: {class: (O) verdict against the
    two networks in the source artifact}."""
    o3s7 = S7.criterio_O3(res, {"por_classe_eth": {c: {"por_unidade": pu[c]} for c in CLASSES}})
    out = {"origem_duas_redes": origem, "por_classe": {}}
    guarda_b = {}
    for c in CLASSES:
        teto2 = {u: S7.teto_do_no(res[u]["v23 vs mlp"]["mae"][c]) for u in UNIDADES}
        gb = {}
        for u, alvo in teto2_origem[c].items():
            if teto2[u] is None or alvo is None:
                raise RuntimeError(f"GUARDA (b) {nome} {c} {u}: teto v23 x mlp nao testavel "
                                   f"(recomputado {teto2[u]}, origem {alvo}). PARANDO.")
            dif = teto2[u] - alvo
            gb[u] = {"recomputado_m": teto2[u], "origem_m": alvo, "diferenca_m": dif,
                     "tolerancia_m": TOL_B, "passou": bool(abs(dif) <= TOL_B)}
            if abs(dif) > TOL_B:
                raise RuntimeError(f"GUARDA (b) {nome} {c} {u}: teto v23 x mlp {teto2[u]} != "
                                   f"origem {alvo} (dif {dif}). PARANDO.")
        O2 = ARV2.veredito_O(pu[c], teto2["transecto"])
        if O2["veredito"] != vered2_origem[c]:
            raise RuntimeError(f"GUARDA (b) {nome} {c}: veredito (O) duas redes recomputado "
                               f"{O2['veredito']} != origem {vered2_origem[c]}. PARANDO.")
        gb["veredito_O_duas_redes"] = {"recomputado": O2["veredito"], "origem": vered2_origem[c],
                                       "passou": True}
        guarda_b[c] = gb

        ptr = o3s7[c]["por_unidade"]["transecto"]
        teto3 = ptr["teto_O3_m"]
        if ptr["pares_nao_testados"] or teto3 is None:
            O3p = {"criterio": "(O3)", "veredito": "nao_demonstrado",
                   "motivo": f"par(es) nao testavel(is) no transecto: {ptr['pares_nao_testados']} (D9)",
                   "teto_abs_contrastes_lidar_m": teto3}
        else:
            O3p = ARV2.veredito_O(pu[c], teto3)
        lims = {u: {"lim_inf_A_media_m": pu[c][u]["A"]["ic95_pond_celula"][0],
                    "lim_inf_A_mediana_m": pu[c][u]["A"]["ic95_mediana"][0],
                    "lim_inf_reducao_mae_m": pu[c][u]["reducao_mae_equivalente"]["ic95_pond_celula"][0]}
                for u in UNIDADES}
        vals = [v for u in UNIDADES for v in lims[u].values() if v is not None]
        min_lim = min(vals) if vals else None
        so_transecto = (teto3 is not None and not ptr["pares_nao_testados"]
                        and all(v is not None and v > teto3 for v in lims["transecto"].values()))
        out["por_classe"][c] = {
            "teto_duas_redes_v23_x_mlp_m": teto2,
            "veredito_O_duas_redes": O2["veredito"],
            "tetos_por_par_transecto_m": ptr["tetos_por_par_m"],
            "tetos_por_par_pegada_m": o3s7[c]["por_unidade"]["pegada"]["tetos_por_par_m"],
            "teto_O3_transecto_m": teto3,
            "par_que_fixa_o_teto_transecto": ptr["par_que_fixa_o_teto"],
            "teto_O3_pegada_m": o3s7[c]["por_unidade"]["pegada"]["teto_O3_m"],
            "par_que_fixa_o_teto_pegada": o3s7[c]["por_unidade"]["pegada"]["par_que_fixa_o_teto"],
            "limites_inferiores": lims,
            "menor_limite_inferior_m": min_lim,
            "margem_menor_limite_menos_teto_O3_m": (min_lim - teto3) if (teto3 is not None and min_lim is not None) else None,
            "veredito_O3_primario": O3p["veredito"],
            "O3_primario_detalhe": O3p,
            "secundario_s1_so_transecto_passa": bool(so_transecto),
            "secundario_s2_criterio_O3_teto_por_unidade": {
                "veredito": o3s7[c]["veredito_O3"],
                "passa_por_unidade": {u: o3s7[c]["por_unidade"][u]["passa_na_unidade"] for u in UNIDADES}},
        }
        pc = out["por_classe"][c]
        log(f"  {nome} {c}: teto2 {f3(teto2['transecto'])} ({O2['veredito']}) | tetos "
            + ", ".join(f"{k}={f3(v)}" for k, v in ptr["tetos_por_par_m"].items())
            + f" -> O3 {f3(teto3)} ({ptr['par_que_fixa_o_teto']}) | lim inf tr "
            + "/".join(f3(v) for v in lims["transecto"].values()) + " peg "
            + "/".join(f3(v) for v in lims["pegada"].values())
            + f" -> O3 primario {O3p['veredito']} (s1 {so_transecto}, s2 {pc['secundario_s2_criterio_O3_teto_por_unidade']['veredito']})")
    out["criterio_O3_S7_bruto"] = o3s7
    out["contrastes_lidar_mae"] = {u: {p: res[u][p]["mae"] for p in res[u]} for u in res}
    return out, guarda_b, o3s7


# --------------------------------------------------------------------- main
def main() -> int:
    t0 = time.time()
    log("=== g0: sha256 (S7 v2: entradas + npz da arvore contra sidecar e manifesto v2) ===")
    shas = S7.conferir_sha(True)
    for p in (S7_JSON, DT3_JSON, S8_JSON, S1V2_JSON, S1V2.SAIDA_DUMP, S1V2.V1_JSON):
        shas[p.name] = PROV.sha256(p)
    s7j = json.loads(S7_JSON.read_text(encoding="utf-8"))
    dt3j = json.loads(DT3_JSON.read_text(encoding="utf-8"))
    s8j = json.loads(S8_JSON.read_text(encoding="utf-8"))
    s1j = json.loads(S1V2_JSON.read_text(encoding="utf-8"))
    for nome_art, art in (("S7", s7j), ("D-T3", dt3j), ("S8", s8j), ("S1v2", s1j)):
        if "_proveniencia" not in art:
            raise RuntimeError(f"artefato de origem {nome_art} sem _proveniencia. PARANDO.")

    S7.instalar_tree_em_montar()
    condicoes, guardas = {}, {"g0_sha256": {"entradas_conferidas": len(shas), "passou": True}}
    guardas["A_lado_reproduz_origem_1e-9"] = {}
    guardas["b_teto_duas_redes_e_veredito_1e-6"] = {}

    # ------------------------------------------------ C0, C1, C2 (D-T3)
    log("\n=== C0-C2: registros (D-T3) ===")
    base = DT3.carregar_base()
    df, off, mapa_pegada = base["df"], base["off"], base["mapa_pegada"]
    J.LAT = DT3.LAT_NATIVA
    flor, info = REF.montar_populacao(DT3.PARQUET, base["juiz"])
    conf = REF.guardas_populacao(flor, base["juiz"])
    log(f"  populacao: {info['n_floresta_julgavel']} celulas; tabela_regua_k0 em {len(conf)} celulas")
    rotulos_c = {"i_glo30": "C0_base_glo30_sem_mascara_ETH", "ii_b_gedtm30": "C1_registro_ii_b_gedtm30",
                 "iii_sem_offset": "C2_registro_iii_sem_offset"}
    for nome in DT3.REGISTROS:
        cond = rotulos_c[nome]
        d_arm = DT3.aplicar_registro(df, off["i_glo30"], off[nome])
        est = DT3.estatisticas(d_arm)
        reg_or = dt3j["registros"][nome]["por_classe"]
        ga = {}
        for c in CLASSES:
            for u in UNIDADES:
                for k in ("A", "reducao_mae_equivalente"):
                    ga[f"{c}|{u}|{k}"] = cmp_ou_para(est[c][u][k], reg_or[c]["por_unidade"][u][k],
                                                     TOL_A_LADO, f"A-lado {cond} {c} {u} {k}")
        guardas["A_lado_reproduz_origem_1e-9"][cond] = {"campos": sum(v["campos"] for v in ga.values()),
                                                        "max_abs_dif": max(v["max_abs_dif"] for v in ga.values()),
                                                        "passou": True}
        fl = DT3.z_ref_registro(flor, off[nome])
        res, _ = S7.contrastes_por_unidade(fl, mapa_pegada)
        teto2_or = {c: {u: reg_or[c]["teto_familias_v23_x_mlp"][u]["teto_m"] for u in UNIDADES} for c in CLASSES}
        v2_or = {c: reg_or[c]["criterios"]["O_familias_28_09"]["veredito"] for c in CLASSES}
        r, gb, o3s7 = avaliar_condicao(cond, res, {c: est[c] for c in CLASSES}, teto2_or, v2_or,
                                       f"auditar_dt3_criterios_registro.json registros.{nome}")
        r["populacao"] = {"teto": info["n_floresta_julgavel"],
                          "ancoras_por_classe": {c: est[c]["transecto"]["A"]["n_celulas"] for c in CLASSES}}
        condicoes[cond] = r
        guardas["b_teto_duas_redes_e_veredito_1e-6"][cond] = gb
        if nome == "i_glo30":
            log("  --- guarda (a): C0 reproduz O3 de auditar_s7_arvore_lidar.json a 1e-9 ---")
            ga_ = {}
            campos = ("tetos_por_par_m", "teto_O3_m", "par_que_fixa_o_teto", "lim_inf_A_media_m",
                      "lim_inf_A_mediana_m", "lim_inf_reducao_mae_m", "passa_na_unidade",
                      "passaria_so_com_v23_x_mlp_nesta_unidade")
            for c in CLASSES:
                for u in UNIDADES:
                    for f in campos:
                        ga_[f"{c}|{u}|{f}"] = cmp_ou_para(o3s7[c]["por_unidade"][u][f],
                                                          s7j["O3"][c]["por_unidade"][u][f], TOL_A,
                                                          f"(a) O3 {c} {u} {f}")
                ga_[f"{c}|veredito"] = cmp_ou_para(o3s7[c]["veredito_O3"], s7j["O3"][c]["veredito_O3"],
                                                   TOL_A, f"(a) O3 {c} veredito")
                t = o3s7[c]["por_unidade"]["transecto"]["teto_O3_m"]
                if t is None or round(t, 3) != TETO_S7_LITERAL[c]:
                    raise RuntimeError(f"GUARDA (a) {c}: teto O3 {t} nao arredonda ao literal "
                                       f"{TETO_S7_LITERAL[c]}. PARANDO.")
            guardas["a_C0_reproduz_O3_S7_1e-9"] = {
                "campos": sum(v["campos"] for v in ga_.values()),
                "max_abs_dif": max(v["max_abs_dif"] for v in ga_.values()),
                "teto_O3_literal_m": TETO_S7_LITERAL,
                "teto_O3_recomputado_m": {c: o3s7[c]["por_unidade"]["transecto"]["teto_O3_m"] for c in CLASSES},
                "vereditos": {c: o3s7[c]["veredito_O3"] for c in CLASSES}, "passou": True}
            log(f"  (a) PASSOU: {guardas['a_C0_reproduz_O3_S7_1e-9']['campos']} campos, "
                f"max |dif| {guardas['a_C0_reproduz_O3_S7_1e-9']['max_abs_dif']:.2e}")

    # ------------------------------------------------ C3 (S8)
    log("\n=== C3: mascara Hansen 2011-2023 (S8) ===")
    manifesto = json.loads(S8.MANIFESTO.read_text(encoding="utf-8"))
    for reg in manifesto["tiles"]:
        h = PROV.sha256(Path(reg["arquivo"]))
        if h != reg["sha256"]:
            raise RuntimeError(f"g0 Hansen: {reg['arquivo']} sha256 {h} != manifesto. PARANDO.")
        shas[Path(reg["arquivo"]).name] = h
    ANA.conferir_sha256_anadem(S8.LATERAIS, S8.RECORTE_ANADEM)
    J.LAT = S8.LATERAIS
    flor8, info8 = ANA.montar_populacao_com_anadem(S8.PARQUET_LIDAR, base["juiz"], S8.LATERAIS)
    REF.guardas_populacao(flor8, base["juiz"])
    mascaras, extra = S8.construir_mascaras(manifesto)
    df_mk, g6a = S8.marcar(df, mascaras, extra["brutos"], "ancoras_E4")
    flor_mk, g6b = S8.marcar(flor8, mascaras, extra["brutos"], "floresta_julgavel")
    df_m = df_mk[~df_mk["perda_2011_2023"]].copy()
    flor_m = flor_mk[~flor_mk["perda_2011_2023"]].copy()
    pop8 = {"ancoras_nao_perturbadas": int(len(df_m)), "floresta_nao_perturbada": int(len(flor_m))}
    for k, v in pop8.items():
        if v != s8j["populacao"][k]:
            raise RuntimeError(f"GUARDA populacao C3 {k}: {v} != S8 {s8j['populacao'][k]}. PARANDO.")
    log(f"  g6 {g6a['divergencias_reproject_vs_exato']}/{g6b['divergencias_reproject_vs_exato']} divergencias; {pop8}")
    A_com = {c: S8.por_unidade_A(df_m, c) for c in CLASSES}
    A_or8 = s8j["A_por_classe_eth"]["com_mascara_nao_perturbadas"]
    ga = {}
    for c in CLASSES:
        for u in UNIDADES:
            for k in ("A", "reducao_mae_equivalente"):
                ga[f"{c}|{u}|{k}"] = cmp_ou_para(A_com[c][u][k], A_or8[c][u][k], TOL_A_LADO,
                                                 f"A-lado C3 {c} {u} {k}")
    guardas["A_lado_reproduz_origem_1e-9"]["C3_mascara_hansen_2011_2023"] = {
        "campos": sum(v["campos"] for v in ga.values()),
        "max_abs_dif": max(v["max_abs_dif"] for v in ga.values()), "passou": True}
    res8, _ = S7.contrastes_por_unidade(flor_m, mapa_pegada)
    teto2_or8 = {c: {"transecto": s8j["criterios_S_O"][c]["teto_com_mascara"]["teto_m"]} for c in CLASSES}
    v2_or8 = {c: s8j["criterios_S_O"][c]["com_mascara"]["O_teto_recomputado_primario"]["veredito"]
              for c in CLASSES}
    r, gb, _ = avaliar_condicao("C3_mascara_hansen_2011_2023", res8, A_com, teto2_or8, v2_or8,
                                "auditar_s8_mascara_disturbio.json criterios_S_O[c].com_mascara."
                                "O_teto_recomputado_primario / teto_com_mascara")
    r["populacao"] = {**pop8, "g6": [g6a, g6b]}
    condicoes["C3_mascara_hansen_2011_2023"] = r
    guardas["b_teto_duas_redes_e_veredito_1e-6"]["C3_mascara_hansen_2011_2023"] = gb

    # ------------------------------------------------ C4, C5 (S1)
    log("\n=== C4-C5: classe lidar desacoplada (Lnb) e concordantes (C) (S1) ===")
    shas_s1 = S1.conferir_sha256_entradas()
    flor1, info1, gt1 = S1.recomputar_populacao_e_guardas(shas_s1)
    nat = pd.read_parquet(S1.NATIVA_PARQUET)
    viz = S1.montar_grade_vizinhos(nat)
    flor1 = flor1.reset_index(drop=True)
    m = S1.classificar_lnb_e_horn(flor1[["quad", "idx_no", "transecto", "z_p95_all"]], viz)
    if len(m) != len(flor1) or not (m["idx_no"].to_numpy() == flor1["idx_no"].to_numpy()).all():
        raise RuntimeError("classificar_lnb_e_horn mudou a ordem/tamanho da populacao. PARANDO.")
    flor1["faixa_Lnb"] = m["classe_lnb"].values
    flor1["faixa_C"] = np.where(flor1["faixa_ex"].astype(str) == flor1["faixa_Lnb"].astype(str),
                                flor1["faixa_ex"].astype(str), np.nan)          # D5 (identical to S1v2's main)
    dump1 = pd.read_parquet(S1V2.SAIDA_DUMP)
    for arm, col_anc, cond in (("Lnb", "classe_lnb", "C4_classe_lidar_desacoplada_Lnb"),
                               ("C", "classe_C", "C5_celulas_concordantes_C")):
        est = {c: S1.estatisticas_braco(dump1.assign(_classe_braco=dump1[col_anc]), c)["por_unidade"]
               for c in CLASSES}
        ga = {}
        for c in CLASSES:
            chk_or = s1j["resultado_por_classe"][c]["criterios"][arm]["O"]["checagens_por_unidade"]
            for u in UNIDADES:
                meu = {"lim_inf_A_media_m": est[c][u]["A"]["ic95_pond_celula"][0],
                       "lim_inf_A_mediana_m": est[c][u]["A"]["ic95_mediana"][0],
                       "lim_inf_reducao_mae_m": est[c][u]["reducao_mae_equivalente"]["ic95_pond_celula"][0]}
                ga[f"{c}|{u}"] = cmp_ou_para(meu, {k: chk_or[u][k] for k in meu}, TOL_A_LADO,
                                             f"A-lado {cond} {c} {u}")
            ga[f"{c}|A_media_transecto"] = cmp_ou_para(
                est[c]["transecto"]["A"]["media_pond_celula_m"],
                s1j["resultado_por_classe"][c]["guarda_numeros_batem_com_v1"][f"A_{arm}"]["v2"],
                TOL_A_LADO, f"A-lado {cond} {c} media")
        guardas["A_lado_reproduz_origem_1e-9"][cond] = {
            "campos": sum(v["campos"] for v in ga.values()),
            "max_abs_dif": max(v["max_abs_dif"] for v in ga.values()), "passou": True}
        fl = flor1.copy()
        fl["faixa_ex"] = fl[f"faixa_{arm}"]                   # D5: S7.contrastes reads 'faixa_ex'
        res1, _ = S7.contrastes_por_unidade(fl, mapa_pegada)
        teto2_or1 = {c: {"transecto": s1j["resultado_por_classe"][c]["teto_O_m"][f"{arm}_recomputado"]["teto_m"]}
                     for c in CLASSES}
        v2_or1 = {c: s1j["resultado_por_classe"][c]["criterios"][arm]["O"]["veredito"] for c in CLASSES}
        r, gb, _ = avaliar_condicao(cond, res1, est, teto2_or1, v2_or1,
                                    f"auditar_s1_estratificacao_lidar_v2.json resultado_por_classe[c]."
                                    f"criterios.{arm}.O / teto_O_m.{arm}_recomputado")
        r["populacao"] = {"teto": info1["n_floresta_julgavel"],
                          "celulas_teto_por_classe": {c: int((fl["faixa_ex"].astype(str) == c).sum())
                                                      for c in CLASSES},
                          "ancoras_por_classe": {c: est[c]["transecto"]["A"]["n_celulas"] for c in CLASSES}}
        condicoes[cond] = r
        guardas["b_teto_duas_redes_e_veredito_1e-6"][cond] = gb

    # ------------------------------------------------ table and reading
    tabela = []
    for cond, r in condicoes.items():
        for c in CLASSES:
            pc = r["por_classe"][c]
            tabela.append({
                "condicao": cond, "classe": c,
                "teto_duas_redes_transecto_m": pc["teto_duas_redes_v23_x_mlp_m"]["transecto"],
                "veredito_O_duas_redes_hoje": pc["veredito_O_duas_redes"],
                "teto_O3_transecto_m": pc["teto_O3_transecto_m"],
                "par_que_fixa_transecto": pc["par_que_fixa_o_teto_transecto"],
                "teto_O3_pegada_m": pc["teto_O3_pegada_m"],
                "par_que_fixa_pegada": pc["par_que_fixa_o_teto_pegada"],
                **{f"{k}_{u}": pc["limites_inferiores"][u][k] for u in UNIDADES
                   for k in ("lim_inf_A_media_m", "lim_inf_A_mediana_m", "lim_inf_reducao_mae_m")},
                "margem_menor_limite_menos_teto_O3_m": pc["margem_menor_limite_menos_teto_O3_m"],
                "veredito_O3_primario": pc["veredito_O3_primario"],
                "s1_so_transecto_passa": pc["secundario_s1_so_transecto_passa"],
                "s2_criterio_O3_teto_por_unidade": pc["secundario_s2_criterio_O3_teto_por_unidade"]["veredito"],
            })
    escopo, falham, fora = [], [], []
    for cond, r in condicoes.items():
        if cond.startswith("C0"):
            continue
        pc = r["por_classe"][">30m"]
        if pc["veredito_O_duas_redes"] == DEMONSTRADO:
            escopo.append(cond)
            if pc["veredito_O3_primario"] != DEMONSTRADO:
                falham.append(cond)
        else:
            fora.append(cond)
    leitura = {
        "regra_literal": "'a robustez cobre as tres familias' so se for demonstrado em TODAS as condicoes "
                         "em que hoje e demonstrado contra as duas redes; senao, o texto nomeia a condicao "
                         "em que falha (desenho, R-O3)",
        "condicoes_no_escopo_O_maior30_demonstrado_hoje": escopo,
        "condicoes_fora_do_escopo_O_maior30_nao_demonstrado_hoje": fora,
        "condicoes_em_que_O3_falha": falham,
        "a_robustez_cobre_as_tres_familias": "sim" if (escopo and not falham) else "nao",
        "base_C0_O3_maior30": condicoes["C0_base_glo30_sem_mascara_ETH"]["por_classe"][">30m"]["veredito_O3_primario"],
        "O3_maior30_por_condicao": {k: v["por_classe"][">30m"]["veredito_O3_primario"] for k, v in condicoes.items()},
        "O3_20_30_por_condicao_descritivo": {k: v["por_classe"]["20-30m"]["veredito_O3_primario"]
                                            for k, v in condicoes.items()},
        "secundario_s2_O3_maior30_por_condicao": {
            k: v["por_classe"][">30m"]["secundario_s2_criterio_O3_teto_por_unidade"]["veredito"]
            for k, v in condicoes.items()},
    }
    log("\n=== TABELA (>30m e 20-30m) ===")
    for t in tabela:
        log(f"  {t['condicao']:<34} {t['classe']:<7} teto2 {f3(t['teto_duas_redes_transecto_m'])} "
            f"({t['veredito_O_duas_redes_hoje']}) O3 {f3(t['teto_O3_transecto_m'])} [{t['par_que_fixa_transecto']}] "
            f"margem {f3(t['margem_menor_limite_menos_teto_O3_m'], True)} -> {t['veredito_O3_primario']} "
            f"(s1 {t['s1_so_transecto_passa']}, s2 {t['s2_criterio_O3_teto_por_unidade']})")
    log(f"  LEITURA: a robustez cobre as tres familias = {leitura['a_robustez_cobre_as_tres_familias']} "
        f"(escopo {escopo}; falha em {falham}; fora {fora})")

    fontes = [S7_JSON, DT3_JSON, S8_JSON, S1V2_JSON, S1V2.SAIDA_DUMP, S1V2.V1_JSON,
              DT3.REGUA_JSON, DT3.S2_JSON, DT3.JUIZ, DT3.PARQUET, S7.REF_JSON, S7.ADM_JSON,
              S7.MANIFESTO_TREINO, S8.MANIFESTO, S8.REF_ANADEM,
              ARV2.TABELA3_JSON, ARV2.DUMP_E4_PARQUET, ARV2.NATIVA_JSON, ARV2.NATIVA_PARQUET,
              ARV2.NMAD_JSON, ARV2.D8_JSON,
              RAIZ / "auditar_s7_arvore_lidar_v2.py", RAIZ / "auditar_dt3_criterios_registro.py",
              RAIZ / "auditar_s8_mascara_disturbio.py", RAIZ / "auditar_s1_estratificacao_lidar.py",
              RAIZ / "auditar_s1_estratificacao_lidar_v2.py", RAIZ / "auditar_rotulo_regua_v2.py",
              RAIZ / "auditar_nmad_pareado.py", RAIZ / "auditar_nmad_pareado_nativa_anadem.py",
              RAIZ / "juiz_lidar_v4.py", RAIZ / "preparar_anadem_nativa.py", RAIZ / "proveniencia.py",
              Path(__file__)]
    fontes += [Path(t["arquivo"]) for t in manifesto["tiles"]]
    fontes += [S7.npz_tree(sd) for sd in J.SEEDS_PAR]
    fontes += [S7.npz_tree(sd).with_suffix(".json") for sd in J.SEEDS_PAR]
    fontes += [J.SUP / f"delta_amazonia_ampliacao_b4s035{arm}_seed{sd}.npz"
               for arm in ("", "_mlp_semobs") for sd in J.SEEDS_PAR]
    fontes += [DT3.LAT_NATIVA / f"{n}_amazonia_{q}.npz" for q in J.QUADS
               for n in ("glo30", "fabdem", "gedtm30", "anadem", "agua_jrc", "dossel_eth", "rotulos")]

    corpo = {
        "desenho": ("Pre-declared design note, section 'R-O3' (single execution; nothing goes "
                    "into the text at this stage)"),
        "diario": "Execution log for this run, entries D1-D10 (not distributed).",
        "comando": " ".join([sys.executable] + sys.argv),
        "ambiente": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "V23_LATDIR", "OMP_NUM_THREADS",
                                                    "PYTHONDONTWRITEBYTECODE")},
        "criterio_literal": "Em cada condicao: teto O3 recalculado = maior limite superior de |dMAE| entre os "
                            "tres pares (v23 x mlp, tree x v23, tree x mlp), mesma populacao da condicao, mesmo "
                            "bootstrap (B = 10.000, semente 42), unidade transecto (pegada como leitura "
                            "secundaria, nao confirmacao). Acima de 30 m, 'demonstrado' se os limites inferiores "
                            "de A (media e mediana) e da reducao de MAE ficam acima do teto O3 da condicao.",
        "operacionalizacao": {
            "primario_D3": "ARV2.veredito_O(A e reducao da condicao nas unidades transecto e pegada, teto O3 "
                           "do transecto) -- a mesma funcao dos vereditos (O) contra as duas redes, so com o "
                           "teto trocado; 'sustentado' = demonstrado",
            "secundario_s1": "so a linha transecto (limites do transecto contra o teto O3 do transecto)",
            "secundario_s2": "S7.criterio_O3 (teto por unidade: pegada contra pegada; veredito combinado)",
            "par_nao_testavel_D9": "nao demonstrado",
            "limite_superior_abs_dMAE": "max(|lo|,|hi|) do IC95 bootstrap da mediana da diferenca de MAE por "
                                        "unidade (REF.testar via S7.contrastes_por_unidade)",
            "bootstrap": {"B": REF.B, "semente": REF.SEED, "piso_celulas": REF.PISO_CEL,
                          "min_unidades": REF.MIN_TRANSECTOS},
            "populacoes_D10": {"C0-C2": "REF.montar_populacao + z_ref do registro (DT3.z_ref_registro)",
                               "C3": "ANA.montar_populacao_com_anadem + S8.marcar, celulas e ancoras nao "
                                     "perturbadas (lossyear 2011-2023)",
                               "C4-C5": "S1.recomputar_populacao_e_guardas + S1.classificar_lnb_e_horn; "
                                        "ancoras do dump S1v2 (classe_lnb / classe_C)"},
            "produto_tree": "glo30 - media das 5 sementes (delta_amazonia_arvore_f2_s7_seed*.npz), "
                            "S7.instalar_tree_em_montar",
        },
        "conferencia_sha256_antes_de_calcular": shas,
        "guardas": guardas,
        "condicoes": condicoes,
        "tabela_vereditos": tabela,
        "leitura_pre_declarada": leitura,
        "nao_faz": "nao edita nenhum script/JSON existente; nao retreina; nao interpreta para o artigo; "
                   "nao audita o proprio resultado; nao abre teste novo",
        "contra_auditoria": "PENDENTE -- nao feita por quem escreveu este script",
        "tempo_total_s": round(time.time() - t0, 1),
    }
    PROV.gravar(SAIDA, corpo, fontes_lidas=fontes, script=__file__)
    log(f"  -> {SAIDA.name} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
