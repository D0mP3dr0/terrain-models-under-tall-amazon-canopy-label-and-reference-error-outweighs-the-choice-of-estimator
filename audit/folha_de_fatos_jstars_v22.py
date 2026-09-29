# -*- coding: utf-8 -*-
"""Build part D of the article's fact sheet (folha_de_fatos_jstars_v22.json).

Parts A, B and C are copied verbatim from folha_de_fatos_jstars_v21.json after
its sha256 is checked against a pinned value; no value is retyped. Part D
records, with artefact and field, the facts of two robustness tests.

  S1 (auditar_s1_estratificacao_lidar_v2.json), classes 20-30m and >30m:
    canopy stratification by lidar canopy height decoupled from the ground.
    Arms: E (ETH canopy height), Lnb (lidar canopy decoupled from the cell's
    own ground), C (cells where ETH and lidar agree), L (lidar coupled to the
    ground, descriptive only). Facts: recomputed ceiling (O) of the two
    networks, v23 and MLP (the product-based ceiling is kept as descriptive);
    cell counts per arm; per arm, the 95% CI of A (cell-weighted mean) over
    transect and footprint units (S), the lower bound of mean and median A
    against the ceiling (O) and the 95% CI of A - delta (R); S/O/R verdicts;
    H-S1 verdict with operands; quadratic-weighted kappa (E x Lnb); A for
    arm L; operands of (O) for Lnb at >30m, footprint unit, median.

  S2 (auditar_s2_registro_referencia.json): robustness to the vertical
    registration arm. Facts: A per class in each arm ((i) current GLO-30,
    (ii-b) GEDTM30, (iii) no offset); per-transect offset of (ii-b) vs (i)
    (min and max over 13 transects computed in code, median |.| tau copied);
    verdict counts per class, re-summed in code; the "depende" pairs, found
    by search in code, with their operands; the decisive test GLO-30 family
    (FABDEM, ANADEM) vs GEDTM30 at >30m, arm (ii-b); arm (ii-a), not
    evaluable, with its transect count; MAE difference FABDEM - GEDTM30 at
    20-30m, arm (i).

Checks (fail loudly): G1 unique fact ids across parts A-D; G2 no None/NaN
value at any nesting level (arm ii-a, NaN in the artefact, becomes a
separate fact); G3 every source recorded in _fontes; G4 verdict counts
recomputed from veredito_por_par_e_classe and checked against the
artefact's resumo_vereditos; G5 anchor A >30m = 6.651 reproduced both in
arm E of S1 (part B, jstarsb_e4_A_30m_mais) and in arm (i) of S2.

Usage:  python folha_de_fatos_jstars_v22.py [--saida folha_de_fatos_jstars_v22.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import proveniencia as PROV          # noqa: E402

# Pinned sha256 of folha_de_fatos_jstars_v21.json (parts A, B and C).
SHA256_FOLHA_V21_ESPERADO = (
    "6104492b226225623618f5e1148d7b3aeba4fa3ae72e96352009ffbd5d0013fb")

ARQS = {
    "folha_v21": RAIZ / "folha_de_fatos_jstars_v21.json",
    "s1": RAIZ / "auditar_s1_estratificacao_lidar_v2.json",
    "s2": RAIZ / "auditar_s2_registro_referencia.json",
}

CLASSES_S1 = ["20-30m", ">30m"]
CLASSE_ID = {"20-30m": "20_30m", ">30m": "30m_mais"}
BRACOS_S1 = {
    "E_com_teto_duas_redes": "E (com teto das duas redes)",
    "Lnb": "Lnb (dossel lidar desacoplado do chao da propria celula)",
    "C": "C (celulas em que ETH e lidar concordam)",
}
BRACO_ID = {"E_com_teto_duas_redes": "E", "Lnb": "Lnb", "C": "C"}

CLASSES_S2_A = ["3-10m", "10-20m", "20-30m", ">30m", "TODAS"]
CLASSE_ID_S2 = {"3-10m": "3_10m", "10-20m": "10_20m", "20-30m": "20_30m",
                 ">30m": "30m_mais", "TODAS": "todas"}
BRACOS_S2_A = {"i": ("i_atual (GLO-30, atual)", None),
               "ii-b": ("ii-b (GEDTM30)", "ii-b"),
               "iii": ("iii (sem offset)", "iii")}


def log(m=""):
    print(m, flush=True)


def sha256_arquivo(p: Path) -> str:
    return PROV.sha256(p)


def _sem_nulos(x, trilha):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        raise RuntimeError(f"G2: valor nulo/NaN em {trilha}")
    if isinstance(x, dict):
        for kk, vv in x.items():
            _sem_nulos(vv, f"{trilha}.{kk}")
    elif isinstance(x, (list, tuple)):
        for i, vv in enumerate(x):
            _sem_nulos(vv, f"{trilha}[{i}]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default="folha_de_fatos_jstars_v22.json")
    a = ap.parse_args()

    for k, p in ARQS.items():
        if not p.exists():
            raise FileNotFoundError(f"fonte ausente: {k} -> {p}")

    # ------------------------------------------------------------------
    # G3 (part 1): verify the v21 sheet's sha256 before trusting any field.
    # ------------------------------------------------------------------
    sha_folha_v21 = sha256_arquivo(ARQS["folha_v21"])
    if sha_folha_v21 != SHA256_FOLHA_V21_ESPERADO:
        raise RuntimeError(
            f"folha_de_fatos_jstars_v21.json mudou desde o protocolo "
            f"aprovado: sha256 atual={sha_folha_v21} esperado="
            f"{SHA256_FOLHA_V21_ESPERADO}. Nao prossigo copiando parte_A/"
            f"parte_B/parte_C de uma folha nao conferida.")

    d_folha_v21 = json.loads(ARQS["folha_v21"].read_text(encoding="utf-8"))
    parte_a = d_folha_v21["parte_A"]
    parte_b = d_folha_v21["parte_B"]
    parte_a_parte_b_fonte = d_folha_v21["parte_A_parte_B_fonte"]
    parte_c = d_folha_v21["parte_C"]

    d_s1 = json.loads(ARQS["s1"].read_text(encoding="utf-8"))
    d_s2 = json.loads(ARQS["s2"].read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Fact infrastructure (G1, G2, groups)
    # ------------------------------------------------------------------
    ids = set(f["id"] for f in parte_a["fatos"])              # G1: include part A ids
    for grupo_fatos in parte_b["fatos"].values():              # G1: include part B ids
        for f in grupo_fatos:
            if f["id"] in ids:
                raise RuntimeError(
                    f"G1: id de parte_B colide com parte_A: {f['id']}")
            ids.add(f["id"])
    for grupo_fatos in parte_c["fatos"].values():              # G1: include part C ids
        for f in grupo_fatos:
            if f["id"] in ids:
                raise RuntimeError(
                    f"G1: id de parte_C colide com parte_A/parte_B: {f['id']}")
            ids.add(f["id"])
    grupos: dict[str, list] = {}

    def fato(grupo, id_, valor, unidade, chave_arq, campo, nota=None):
        if id_ in ids:                                         # G1
            raise RuntimeError(f"G1: id duplicado: {id_}")
        ids.add(id_)
        _sem_nulos(valor, id_)                                  # G2
        arq = ARQS[chave_arq]
        linha = {"id": id_, "valor": valor, "unidade": unidade,
                  "artefato": str(arq.name), "campo": campo}
        if nota:
            linha["nota"] = nota
        grupos.setdefault(grupo, []).append(linha)
        return linha

    # ==================================================================
    # D1 -- recomputed ceiling (O), per class
    # ==================================================================
    for c in CLASSES_S1:
        cid = CLASSE_ID[c]
        teto = d_s1["resultado_por_classe"][c]["teto_O_m"]
        fato("d1_teto_recomputado", f"jstarsd_s1_teto_recomputado_{cid}",
             {"teto_duas_redes_v23_mlp_m": teto["E_duas_redes_v23_mlp"],
              "teto_produtos_DESCRITIVO_m": teto["E_teto_produtos_DESCRITIVO"]},
             "m", "s1",
             f"resultado_por_classe.'{c}'.teto_O_m."
             "{E_duas_redes_v23_mlp,E_teto_produtos_DESCRITIVO}",
             nota="correcao desta versao (defeito_1_teto_desalinhado): os "
                  "tres bracos (E, Lnb, C) sao julgados pelo teto das duas "
                  "redes (mesma familia de rotulo); o teto com produtos "
                  "(FABDEM/GEDTM30) fica descritivo, nao apagado")

    # ==================================================================
    # D1 -- cell count per class and arm (E, Lnb, C)
    # ==================================================================
    for c in CLASSES_S1:
        cid = CLASSE_ID[c]
        fato("d1_n_celulas", f"jstarsd_s1_n_celulas_{cid}",
             dict(d_s1["resultado_por_classe"][c]["n_celulas"]),
             "celulas", "s1", f"resultado_por_classe.'{c}'.n_celulas",
             nota="n de celulas julgaveis por braco de estratificacao de "
                  "dossel (E=ETH, Lnb=lidar desacoplado do chao, "
                  "C=celulas em que ETH e lidar concordam)")

    # ==================================================================
    # D1 -- A per arm (E_com_teto_duas_redes, Lnb, C): 95% CI of A (S),
    # lower bound of mean/median A against the ceiling (O), 95% CI of
    # A - delta (R); transect and footprint units
    # ==================================================================
    for c in CLASSES_S1:
        cid = CLASSE_ID[c]
        crit = d_s1["resultado_por_classe"][c]["criterios"]
        for braco_chave, braco_rotulo in BRACOS_S1.items():
            bid = BRACO_ID[braco_chave]
            bloco = crit[braco_chave]
            s = bloco["S"]["checagens_por_unidade"]
            o = bloco["O"]["checagens_por_unidade"]
            r = bloco["R"]["checagens_por_unidade"]
            valor = {
                "braco": braco_rotulo,
                "S_ic95_A_media_pond_celula_m": {
                    "transecto": {"ic95_pond_celula": s["transecto"]["ic95_pond_celula"],
                                  "ic95_t_jackknife": s["transecto"]["ic95_t_jackknife"]},
                    "pegada": {"ic95_pond_celula": s["pegada"]["ic95_pond_celula"],
                               "ic95_t_jackknife": s["pegada"]["ic95_t_jackknife"]},
                },
                "O_teto_abs_contrastes_lidar_m": bloco["O"]["teto_abs_contrastes_lidar_m"],
                "O_limiar_inferior_A_media_mediana_m": {
                    "transecto": {"lim_inf_A_media_m": o["transecto"]["lim_inf_A_media_m"],
                                  "acima_do_teto_media": o["transecto"]["acima_do_teto_media"],
                                  "lim_inf_A_mediana_m": o["transecto"]["lim_inf_A_mediana_m"],
                                  "acima_do_teto_mediana": o["transecto"]["acima_do_teto_mediana"]},
                    "pegada": {"lim_inf_A_media_m": o["pegada"]["lim_inf_A_media_m"],
                               "acima_do_teto_media": o["pegada"]["acima_do_teto_media"],
                               "lim_inf_A_mediana_m": o["pegada"]["lim_inf_A_mediana_m"],
                               "acima_do_teto_mediana": o["pegada"]["acima_do_teto_mediana"]},
                },
                "R_ic95_A_menos_delta_m": {
                    "transecto": {"ic95_pond_celula": r["transecto"]["ic95_pond_celula"],
                                  "ic95_t_jackknife": r["transecto"]["ic95_t_jackknife"]},
                    "pegada": {"ic95_pond_celula": r["pegada"]["ic95_pond_celula"],
                               "ic95_t_jackknife": r["pegada"]["ic95_t_jackknife"]},
                },
            }
            fato("d1_A_por_braco", f"jstarsd_s1_A_{cid}_{bid}", valor, "m",
                 "s1",
                 f"resultado_por_classe.'{c}'.criterios.'{braco_chave}'."
                 "{S,O,R}.checagens_por_unidade.{transecto,pegada}",
                 nota="A = solo_anc - z_ref, media ponderada por celula; "
                      "S = IC95 (percentil e t-jackknife) de A; O = limiar "
                      "inferior (2,5%) de A media e mediana contra o maior "
                      "teto de |contraste| entre estimadores x lidar; R = "
                      "IC95 de A-delta (ordenacao A>delta). Para o braco E, "
                      "os valores pontuais de A (media, mediana) ja estao "
                      "em parte_B (jstarsb_e4_A_* do mesmo artefato "
                      "auditar_rotulo_regua_v2.json); este bloco S1 nao os "
                      "reafirma, so os julga contra o novo teto")

    # ==================================================================
    # D1 -- verdicts of (S), (O), (R) per class and arm
    # ==================================================================
    for c in CLASSES_S1:
        cid = CLASSE_ID[c]
        crit = d_s1["resultado_por_classe"][c]["criterios"]
        for braco_chave, braco_rotulo in BRACOS_S1.items():
            bid = BRACO_ID[braco_chave]
            bloco = crit[braco_chave]
            fato("d1_vereditos", f"jstarsd_s1_veredito_{cid}_{bid}",
                 {"braco": braco_rotulo,
                  "S": bloco["S"]["veredito"],
                  "O": bloco["O"]["veredito"],
                  "R": bloco["R"]["veredito"]},
                 None, "s1",
                 f"resultado_por_classe.'{c}'.criterios.'{braco_chave}'."
                 "{S,O,R}.veredito")

    # ==================================================================
    # D1 -- H-S1 per class (verdict, reason, operands)
    # ==================================================================
    for c in CLASSES_S1:
        cid = CLASSE_ID[c]
        h = d_s1["resultado_por_classe"][c]["H_S1"]
        fato("d1_H_S1", f"jstarsd_s1_H_S1_{cid}",
             {"veredito": h["veredito"], "motivo": h["motivo"],
              "operandos": h["operandos"]},
             None, "s1", f"resultado_por_classe.'{c}'.H_S1")

    # ==================================================================
    # D1 -- ETH x lidar agreement (weighted kappa, cells in C per class)
    # ==================================================================
    conf = d_s1["classificacao_confusao_e_relevo_descritivos"]["confusao_E_x_Lnb_descritiva"]
    fato("d1_concordancia_eth_lidar", "jstarsd_s1_kappa_ponderado_E_x_Lnb",
         {"kappa_ponderado_quadratico": conf["kappa_ponderado_quadratico"],
          "n_par_valido": conf["n_par_valido"]},
         None, "s1",
         "classificacao_confusao_e_relevo_descritivos."
         "confusao_E_x_Lnb_descritiva.{kappa_ponderado_quadratico,n_par_valido}",
         nota="kappa de Cohen ponderado quadratico entre a classe ETH (E) e "
              "a classe do dossel lidar desacoplado do chao (Lnb), todas as "
              "celulas julgaveis (todas as classes)")
    for c in CLASSES_S1:
        cid = CLASSE_ID[c]
        fato("d1_concordancia_eth_lidar",
             f"jstarsd_s1_n_celulas_C_{cid}",
             d_s1["resultado_por_classe"][c]["n_celulas"]["C"],
             "celulas", "s1", f"resultado_por_classe.'{c}'.n_celulas.C",
             nota="n de celulas do braco C (ETH e lidar concordam) nesta "
                  "classe, insumo descritivo da concordancia ETH x lidar")

    # ==================================================================
    # D1 -- mean A with the ground-coupled lidar (arm L, descriptive)
    # ==================================================================
    for c in CLASSES_S1:
        cid = CLASSE_ID[c]
        for unidade_nome in ("transecto", "pegada"):
            aa = d_s1["braco_L_descritivo"]["estatisticas_por_classe"][c][
                "por_unidade"][unidade_nome]["A"]
            fato("d1_A_lidar_acoplado",
                 f"jstarsd_s1_A_lidar_acoplado_{cid}_{unidade_nome}",
                 {"media_pond_celula_m": aa["media_pond_celula_m"],
                  "mediana_m": aa["mediana_m"],
                  "n_celulas": aa["n_celulas"],
                  "n_unidades": aa["n_unidades"],
                  "ic95_pond_celula": aa["ic95_pond_celula"],
                  "ic95_mediana": aa["ic95_mediana"]},
                 "m", "s1",
                 f"braco_L_descritivo.estatisticas_por_classe.'{c}'."
                 f"por_unidade.'{unidade_nome}'.A.{{media_pond_celula_m,"
                 "mediana_m,n_celulas,n_unidades,ic95_pond_celula,ic95_mediana}",
                 nota="braco L = dossel lidar ACOPLADO ao chao da propria "
                      "celula (por oposicao a Lnb, desacoplado); descritivo, "
                      "nao entra nos vereditos (S)/(O)/(R) do H-S1")

    # ==================================================================
    # D1 -- operands of (O) for Lnb at >30m, footprint unit, median vs ceiling
    # ==================================================================
    o_lnb_30m_pegada = d_s1["resultado_por_classe"][">30m"]["criterios"][
        "Lnb"]["O"]
    fato("d1_lnb_O_30m_pegada_mediana",
         "jstarsd_s1_lnb_O_30m_mais_pegada_mediana",
         {"lim_inf_A_mediana_m":
              o_lnb_30m_pegada["checagens_por_unidade"]["pegada"]["lim_inf_A_mediana_m"],
          "teto_abs_contrastes_lidar_m": o_lnb_30m_pegada["teto_abs_contrastes_lidar_m"],
          "acima_do_teto_mediana":
              o_lnb_30m_pegada["checagens_por_unidade"]["pegada"]["acima_do_teto_mediana"],
          "veredito_O_lnb_30m_mais": o_lnb_30m_pegada["veredito"]},
         "m", "s1",
         "resultado_por_classe.'>30m'.criterios.'Lnb'.O."
         "{teto_abs_contrastes_lidar_m,checagens_por_unidade.pegada."
         "{lim_inf_A_mediana_m,acima_do_teto_mediana},veredito}",
         nota="unico ponto do desenho em que a mediana de A, na unidade "
              "pegada, NAO fica acima do teto (lim_inf < teto): motivo do "
              "veredito 'nao_demonstrado' do (O) do Lnb em >30m")

    # ==================================================================
    # D2 -- A per class in each registration arm ((i), (ii-b), (iii))
    # ==================================================================
    efeito = d_s2["efeito_sobre_A_por_classe"]
    for c in CLASSES_S2_A:
        cid = CLASSE_ID_S2[c]
        # Arm (i) is the common baseline: A_braco_i_m must be identical in
        # efeito['ii-b'][c] and efeito['iii'][c]; checked in code (G4).
        a_i_via_iib = efeito["ii-b"][c]["A_braco_i_m"]
        a_i_via_iii = efeito["iii"][c]["A_braco_i_m"]
        if round(a_i_via_iib, 9) != round(a_i_via_iii, 9):
            raise RuntimeError(
                f"G4: A_braco_i_m diverge entre 'ii-b' e 'iii' em {c}: "
                f"{a_i_via_iib} != {a_i_via_iii}")
        valor = {
            "i_atual_GLO30_m": a_i_via_iib,
            "ii_b_gedtm30_m": efeito["ii-b"][c]["A_recomputado_m"],
            "iii_sem_offset_m": efeito["iii"][c]["A_recomputado_m"],
            "n_celulas": efeito["ii-b"][c]["n"],
            "guarda_bate_ii_b": efeito["ii-b"][c]["guarda_bate"],
            "guarda_bate_iii": efeito["iii"][c]["guarda_bate"],
        }
        fato("d2_A_por_braco_registro", f"jstarsd_s2_A_registro_{cid}",
             valor, "m", "s2",
             f"efeito_sobre_A_por_classe.{{'ii-b','iii'}}.'{c}'."
             "{A_braco_i_m,A_recomputado_m,n,guarda_bate}",
             nota="A recomputada sob o braco (ii-a) nao e avaliavel (ver "
                  "d2_braco_ii_a); A_braco_i_m e o valor sob o braco (i) "
                  "atual (GLO-30), identico nos dois blocos de origem "
                  "(conferido em codigo, G4)")

    # ==================================================================
    # D2 -- per-transect offset of (ii-b) vs (i): min and max of the
    # difference (13 transects) computed in code; tau copied
    off_i = d_s2["off_t_por_braco"]["i_atual"]
    off_iib = d_s2["off_t_por_braco"]["ii_b_gedtm30"]
    if set(off_i.keys()) != set(off_iib.keys()):
        raise RuntimeError("G4: transectos de i_atual e ii_b_gedtm30 nao "
                            "coincidem em off_t_por_braco")
    diffs = {t: off_iib[t] - off_i[t] for t in off_i}
    t_min = min(diffs, key=diffs.get)
    t_max = max(diffs, key=diffs.get)
    fato("d2_offset_ii_b_vs_i", "jstarsd_s2_offset_ii_b_vs_i_min_max",
         {"n_transectos": len(diffs),
          "min_m": {"transecto": t_min, "valor_m": diffs[t_min]},
          "max_m": {"transecto": t_max, "valor_m": diffs[t_max]},
          "tau_mediana_abs_m": d_s2["tau_m"]["ii-b"]},
         "m", "s2",
         "off_t_por_braco.{i_atual,ii_b_gedtm30} (diferenca por transecto "
         "calculada em codigo: off_ii_b - off_i; min/max sobre os 13 "
         "transectos); tau_m.'ii-b' copiado verbatim do artefato",
         nota="tau (mediana de |off_ii_b - off_i|) e o proprio criterio "
              "pre-registrado de robustez do desenho S2; min/max dao o "
              "alcance da diferenca assinada")

    # ==================================================================
    # D2 -- arm (ii-a), not evaluable, with its transect count
    # ==================================================================
    dispo = d_s2["braco_ii_a_disponibilidade"]
    fato("d2_braco_ii_a", "jstarsd_s2_braco_ii_a_nao_avaliavel",
         {"avaliavel": dispo["avaliavel"],
          "n_transectos_avaliaveis": dispo["n_transectos_avaliaveis"],
          "motivo": dispo["motivo"]},
         None, "s2",
         "braco_ii_a_disponibilidade.{avaliavel,n_transectos_avaliaveis,motivo}")

    # ==================================================================
    # D2 -- verdict counts (artefact summary + sum recomputed in code, G4)
    # ==================================================================
    resumo = d_s2["resumo_vereditos"]
    total_nao_depende = sum(resumo[c]["nao depende"] for c in resumo)
    total_ressalva = sum(resumo[c]["depende com ressalva"] for c in resumo)
    total_depende = sum(resumo[c]["depende"] for c in resumo)
    fato("d2_contagem_vereditos", "jstarsd_s2_contagem_vereditos",
         {"por_classe": resumo,
          "total_nao_depende": total_nao_depende,
          "total_depende_com_ressalva": total_ressalva,
          "total_depende": total_depende},
         None, "s2", "resumo_vereditos (soma por classe calculada em codigo, G4)")

    # ==================================================================
    # D2 -- pairs with verdict "depende" (found by search in code)
    # ==================================================================
    def _sem_none_ii_a(inverte_por_braco):
        """G2: inverte_por_braco.'ii-a' is null in the artefact when arm
        (ii-a) is not evaluable (see d2_braco_ii_a); replace it with a
        descriptive marker instead of a raw None."""
        out = {}
        for k, v in inverte_por_braco.items():
            out[k] = v if v is not None else {
                "disponivel": False,
                "motivo": "braco (ii-a) nao avaliavel (ver d2_braco_ii_a)"}
        return out

    vpc = d_s2["veredito_por_par_e_classe"]
    pares_depende = []
    for classe, pares in vpc.items():
        for par, info in pares.items():
            if info["veredito"] == "depende":
                info = dict(info)
                info["inverte_por_braco"] = _sem_none_ii_a(info["inverte_por_braco"])
                pares_depende.append({"classe": classe, "par": par, **info})
    if len(pares_depende) != total_depende:
        raise RuntimeError(
            f"G4: {len(pares_depende)} pares 'depende' encontrados por "
            f"busca, mas resumo_vereditos soma {total_depende}")
    fato("d2_pares_depende", "jstarsd_s2_pares_depende", pares_depende, None,
         "s2",
         "veredito_por_par_e_classe.*.*[veredito=='depende'] (busca em "
         "codigo sobre as 30 combinacoes classe x par)",
         nota="o artefato nao grava um campo de motivo em prosa para cada "
              "veredito; os operandos (inverte_por_braco, "
              "fracao_ii_c_mantida, dif_mae_i_m, tau_ii_b_m) sao a "
              "evidencia bruta de por que cada par caiu em 'depende', "
              "conforme criterio_pre_registrado.regra_depende")

    # ==================================================================
    # D2 -- decisive test: GLO-30 family (FABDEM, ANADEM) vs GEDTM30
    # at >30m, arm (ii-b)
    # ==================================================================
    td = d_s2["teste_decisivo_familia_glo30_vs_gedtm30_30m"]

    def _sem_none_veredito(produto):
        p = dict(produto)
        v = dict(p["veredito"])
        v["inverte_por_braco"] = _sem_none_ii_a(v["inverte_por_braco"])
        p["veredito"] = v
        return p

    fato("d2_teste_decisivo", "jstarsd_s2_teste_decisivo_familia_glo30_x_gedtm30_30m",
         {"criterio": d_s2["criterio_pre_registrado"]["teste_decisivo"],
          "fabdem": _sem_none_veredito(td["fabdem"]),
          "anadem": _sem_none_veredito(td["anadem"])},
         None, "s2",
         "criterio_pre_registrado.teste_decisivo + "
         "teste_decisivo_familia_glo30_vs_gedtm30_30m.{fabdem,anadem}",
         nota="vantagem_i_m e vantagem_ii_b_m negativos = GLO-30 (FABDEM/"
              "ANADEM) com MAE menor que GEDTM30; troca_de_sinal_sob_ii_b "
              "= false nos dois produtos")

    # ==================================================================
    # D2 -- MAE difference FABDEM x GEDTM30 at 20-30m, arm (i)
    # ==================================================================
    fabdem_20_30 = d_s2["tabela_pool_ETH"]["i_atual"]["fabdem"]["20-30m"]
    gedtm30_20_30 = d_s2["tabela_pool_ETH"]["i_atual"]["gedtm30"]["20-30m"]
    dif_mae_codigo = fabdem_20_30["mae"] - gedtm30_20_30["mae"]
    dif_mae_artefato = vpc["20-30m"]["fabdem_vs_gedtm30"]["dif_mae_i_m"]
    if round(dif_mae_codigo, 9) != round(dif_mae_artefato, 9):
        raise RuntimeError(
            f"G4: diferenca de MAE FABDEM-GEDTM30 em 20-30m recalculada "
            f"({dif_mae_codigo}) nao bate com dif_mae_i_m do artefato "
            f"({dif_mae_artefato})")
    fato("d2_dif_mae_fabdem_gedtm30_20_30m",
         "jstarsd_s2_dif_mae_fabdem_gedtm30_20_30m",
         {"mae_fabdem_m": fabdem_20_30["mae"],
          "mae_gedtm30_m": gedtm30_20_30["mae"],
          "diferenca_m": dif_mae_codigo, "n": fabdem_20_30["n"]},
         "m", "s2",
         "tabela_pool_ETH.i_atual.{fabdem,gedtm30}.'20-30m'.mae (diferenca "
         "calculada em codigo, conferida contra "
         "veredito_por_par_e_classe.'20-30m'.fabdem_vs_gedtm30.dif_mae_i_m, G4)")

    # ==================================================================
    # G5 -- reproduce the anchor value, fail loudly
    # ==================================================================
    anc_parte_b = next(
        f for f in parte_b["fatos"]["e4_A_por_classe"]
        if f["id"] == "jstarsb_e4_A_30m_mais")["valor"]["media_pond_celula_m"]
    anc_s2_i = efeito["ii-b"][">30m"]["A_braco_i_m"]
    if round(anc_parte_b, 3) != round(6.651, 3):
        raise RuntimeError(
            f"G5: ancora A>30m no braco E (via parte_B) nao reproduz: "
            f"{anc_parte_b} != 6.651")
    if round(anc_s2_i, 3) != round(6.651, 3):
        raise RuntimeError(
            f"G5: ancora A>30m no braco (i) do S2 nao reproduz: "
            f"{anc_s2_i} != 6.651")
    if round(anc_parte_b, 9) != round(anc_s2_i, 9):
        raise RuntimeError(
            f"G5: A>30m no braco E (parte_B, {anc_parte_b}) diverge do "
            f"braco (i) do S2 ({anc_s2_i})")
    log(f"G5 ok: A_30m_mais_braco_E_parteB={anc_parte_b:.6f} == "
        f"A_30m_mais_braco_i_S2={anc_s2_i:.6f} == 6.651...")

    # ==================================================================
    n_fatos_d = sum(len(v) for v in grupos.values())
    saida = {
        "gerado_para": "Artigo 1 do V23 (alvo IEEE JSTARS, pacote_overleaf/main.tex)",
        "regra": ("todo numero citado no texto do paper tem linha nesta "
                  "folha; linha sem uso e permitida, numero sem linha NAO "
                  "EXISTE. Copiar da folha, conferir por diff contra o "
                  "artefato da linha."),
        "protocolo": (("Adds to the V23 Article 1 fact sheet the facts of tests S1 "
                       "(auditar_s1_estratificacao_lidar_v2.json) and S2 "
                       "(auditar_s2_registro_referencia.json)")),
        "parte_A": parte_a,
        "parte_B": parte_b,
        "parte_A_parte_B_fonte": parte_a_parte_b_fonte,
        "parte_C": parte_c,
        "parte_D": {
            "n_fatos": n_fatos_d,
            "grupos": {g: len(v) for g, v in grupos.items()},
            "fatos": grupos,
        },
    }
    fontes_lidas = list(ARQS.values())
    PROV.gravar(RAIZ / a.saida, saida, fontes_lidas=fontes_lidas, script=__file__)
    log(f"{len(parte_a['fatos'])} fatos em parte_A + {parte_b['n_fatos']} "
        f"fatos em parte_B + {parte_c['n_fatos']} fatos em parte_C "
        f"(copiados, identicos) + {n_fatos_d} fatos em parte_D "
        f"({len(grupos)} grupos) gravados em {a.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
