# -*- coding: utf-8 -*-
"""folha_de_fatos_jstars_v23 -- fact sheet for the IEEE JSTARS article (V23
model): copies parts A-D from the v22 sheet and adds part E.

Parts A-D (and parte_A_parte_B_fonte) are read in code from
`folha_de_fatos_jstars_v22.json` (sha256 checked before copying) and
reproduced unchanged. Part E: every fact is read in code, with id
`jstarse_*`, value, unit, artifact, field and stamp (sha256 of the artifact
at read time plus its own _proveniencia). Facts computed from fields carry
`derivado: true` and `operacao` and are checked against a stored
counterpart when one exists (G4).

Guards (fail loudly):
  G1  unique ids across parts A-E.
  G2  no None/NaN in `valor`; null fields in an artifact are pruned
      explicitly (listed in `campos_null_omitidos`), never silently.
  G3  every source listed in `_fontes`; pinned sha256 values checked before
      reading, and all sources re-hashed at the end (none changed mid-run).
  G4  every computed value (orders, counts, differences, sums) is checked
      against the stored field when it exists.
  GI  parts A-D identical to v22, in the built object and in the JSON
      re-read from disk.
  GF  internal guards of the sources (S5, S8, D-T3) report pass/OK.

Usage:  python folha_de_fatos_jstars_v23.py [--saida folha_de_fatos_jstars_v23.json]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import proveniencia as PROV          # noqa: E402

PAR4 = RAIZ / "artigo_v23_2" / "_pareceres_2026-09-28_rodada4"

ARQS = {
    "folha_v22": RAIZ / "folha_de_fatos_jstars_v22.json",
    "s5": RAIZ / "auditar_s5_baseline_fisico.json",
    "s8": RAIZ / "auditar_s8_mascara_disturbio.json",
    "dt3": RAIZ / "auditar_dt3_criterios_registro.json",
    "nmad_anadem": RAIZ / "auditar_nmad_pareado_nativa_anadem.json",
    "rotulo_v2": RAIZ / "auditar_rotulo_regua_v2.json",
    "s2": RAIZ / "auditar_s2_registro_referencia.json",
    # validation-selection sources: at the V23 root, not in results/ (noted in
    # qualidade_das_fontes)
    "selecao": RAIZ / "selecao_em_validacao.json",
    "suav_ba4": RAIZ / "comparar_suavidade_fina_ba4.json",
    "ponto_s035": RAIZ / "comparar_ponto_s035.json",
    # ramp and quadrant-unit audits
    "f1_rampa": RAIZ / "auditar_f1_rampa.json",
    "unid_quad": RAIZ / "auditar_unidade_quadrante.json",
    # raw run results (stamped sources of auditar_unidade_quadrante/auditar_f1_rampa)
    "r_adot_gatv2": RAIZ / "results" / "ofat_suavfina_b4s035.json",
    "r_adot_mlp": RAIZ / "results" / "baseline_mlp_b4s035.json",
    "r_1536_gatv2": RAIZ / "results" / "noite_lote1536_gatv2_2_semobs.json",
    "r_1536_mlp": RAIZ / "results" / "noite_lote1536_mlp_semobs.json",
    "r_d1_gatv2": RAIZ / "results" / "d1_orcamento_gatv2_2_semobs.json",
    "r_d1_mlp": RAIZ / "results" / "d1_orcamento_mlp_semobs.json",
    "r_f1_gatv2": RAIZ / "results" / "f1_rampa_gatv2_2_semobs.json",
    "r_f1_mlp": RAIZ / "results" / "f1_rampa_mlp_semobs.json",
    "engia_t8_t10": PAR4 / "engia_T8_T10.md",
    "ca_s5": PAR4 / "contra_auditoria_S5.md",
    "ca_s8": PAR4 / "contra_auditoria_S8.md",
    "ca_dt3": PAR4 / "contra_auditoria_DT3.md",
}

# pinned sha256 values of the audited inputs
SHA_ESPERADO = {
    "folha_v22": "101d470477b08e307186bb9b41c564b25295b52fb4263ff067cc2ee0c56c0d8d",
    "s5": "8f62b6a346692783614dc806034a36b2aa1773b057c0f74cb818e7f6df9156cf",
    "s8": "943e4c47da8e77a7e8970aa7d55f1cccdfef4cceb92364f84f51d1db8febe92b",
    "dt3": "ecb7f3449e7506b5f4faf84fae13ba85cfd6f5f404bbeaba00f1b1f01b37ddb0",
    "nmad_anadem": "95c726a1a7b703ba0092b74560df34e1b2ebb8225a45567ce7ffbd641aa75b66",
    "rotulo_v2": "2470115aaf6eed5ff2ea7d878f2dd6a1ac3f7bf682240599102a33fb01129e9a",
    "s2": "aa3f0f34b2965da9f879b060f2beccb39d994fb2c2ac7bbb8c5674e33742a7a2",
}

CLASSES = ["20-30m", ">30m"]
CID = {"0-3m": "0_3m", "3-10m": "3_10m", "10-20m": "10_20m",
       "20-30m": "20_30m", ">30m": "30m_mais", "TODAS": "todas",
       "sem_classe": "sem_classe"}
UNIDADES = ["transecto", "pegada"]
PRODUTOS_TAB_IV = ["v23", "mlp", "fabdem", "gedtm30", "glo30", "anadem"]


def log(m=""):
    print(m, flush=True)


def _sem_nulos(x, trilha):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        raise RuntimeError(f"G2: valor nulo/NaN em {trilha}")
    if isinstance(x, dict):
        for kk, vv in x.items():
            _sem_nulos(vv, f"{trilha}.{kk}")
    elif isinstance(x, (list, tuple)):
        for i, vv in enumerate(x):
            _sem_nulos(vv, f"{trilha}[{i}]")


def podar(x, trilha=""):
    """Drop dict keys whose value is None/NaN and return (clean, pruned).
    A list containing None/NaN becomes the marker 'nao_calculavel (...)' and
    its path is recorded in `podadas` (never dropped silently)."""
    podadas = []

    def _p(o, t):
        if isinstance(o, dict):
            out = {}
            for k, v in o.items():
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    podadas.append(f"{t}.{k}" if t else str(k))
                    continue
                out[k] = _p(v, f"{t}.{k}" if t else str(k))
            return out
        if isinstance(o, list):
            if any(v is None or (isinstance(v, float) and math.isnan(v))
                   for v in o):
                podadas.append(f"{t} (lista com null/NaN -> marcador)")
                return "nao_calculavel (null/NaN no artefato)"
            return [_p(v, f"{t}[{i}]") for i, v in enumerate(o)]
        return o
    return _p(x, trilha), podadas


def sel(d, chaves):
    return {k: d[k] for k in chaves if k in d}


def igual(a, b, tol, rotulo):
    if abs(a - b) > tol:
        raise RuntimeError(f"G4: {rotulo}: {a} != {b} (tol {tol})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default="folha_de_fatos_jstars_v23.json")
    a = ap.parse_args()

    for k, p in ARQS.items():
        if not p.exists():
            raise FileNotFoundError(f"fonte ausente: {k} -> {p}")

    # ------------------------------------------------------------------
    # G3: sha256 of every source BEFORE reading; pinned values checked
    # ------------------------------------------------------------------
    sha0 = {k: PROV.sha256(p) for k, p in ARQS.items()}
    for k, esp in SHA_ESPERADO.items():
        if sha0[k] != esp:
            raise RuntimeError(f"G3: {ARQS[k].name} mudou: sha256 atual="
                               f"{sha0[k]} esperado={esp}. Nao prossigo.")

    J = {}
    for k, p in ARQS.items():
        if p.suffix == ".json":
            J[k] = json.loads(p.read_text(encoding="utf-8"))
    TXT = {k: ARQS[k].read_text(encoding="utf-8")
           for k in ("engia_t8_t10", "ca_s5", "ca_s8", "ca_dt3")}

    def carimbo_de(k):
        c = {"sha256": sha0[k]}
        if k in J and isinstance(J[k], dict) and isinstance(
                J[k].get("_proveniencia"), dict):
            pv = J[k]["_proveniencia"]
            c["script_origem"] = pv.get("script") or "AUSENTE (null no _proveniencia)"
            s = pv.get("sha256_script") or pv.get(
                "sha256_script_no_momento_do_carimbo")
            c["sha256_script"] = s or "AUSENTE"
            c["carimbado_em"] = pv.get("em") or "AUSENTE"
            c["retroativo"] = bool(pv.get("retroativo", False))
        elif k in J:
            c["script_origem"] = "AUSENTE (artefato sem _proveniencia)"
        else:
            c["script_origem"] = "nao_se_aplica (parecer .md)"
        return c
    CARIMBO = {k: carimbo_de(k) for k in ARQS}

    # ------------------------------------------------------------------
    # Parts A-D copied from v22
    # ------------------------------------------------------------------
    v22 = J["folha_v22"]
    COPIADAS = ["parte_A", "parte_B", "parte_A_parte_B_fonte", "parte_C",
                "parte_D"]
    copia = {k: json.loads(json.dumps(v22[k])) for k in COPIADAS}
    for k in COPIADAS:                                           # GI (1)
        if copia[k] != v22[k]:
            raise RuntimeError(f"GI: {k} copiada difere da v22")

    ids = set()
    for f in copia["parte_A"]["fatos"]:
        if f["id"] in ids:
            raise RuntimeError(f"G1: id repetido na v22: {f['id']}")
        ids.add(f["id"])
    for parte in ("parte_B", "parte_C", "parte_D"):
        for gf in copia[parte]["fatos"].values():
            for f in gf:
                if f["id"] in ids:
                    raise RuntimeError(f"G1: id repetido na v22: {f['id']}")
                ids.add(f["id"])
    n_ids_v22 = len(ids)

    grupos: dict[str, list] = {}

    def fato(grupo, id_, valor, unidade, chaves, campo, nota=None,
             operacao=None, podadas=None):
        if not id_.startswith("jstarse_"):
            raise RuntimeError(f"id fora do prefixo jstarse_: {id_}")
        if id_ in ids:                                           # G1
            raise RuntimeError(f"G1: id duplicado: {id_}")
        ids.add(id_)
        _sem_nulos(valor, id_)                                   # G2
        if isinstance(chaves, str):
            chaves = [chaves]
        linha = {"id": id_, "valor": valor, "unidade": unidade,
                 "artefato": " + ".join(ARQS[c].relative_to(RAIZ).as_posix()
                                        for c in chaves),
                 "campo": campo,
                 "carimbo": {ARQS[c].relative_to(RAIZ).as_posix(): CARIMBO[c]
                             for c in chaves}}
        if operacao:
            linha["derivado"] = True
            linha["operacao"] = operacao
        if podadas:
            linha["campos_null_omitidos"] = podadas
        if nota:
            linha["nota"] = nota
        grupos.setdefault(grupo, []).append(linha)
        return linha

    # ==================================================================
    # GF -- internal guards of the new sources
    # ==================================================================
    s5, s8, dt3 = J["s5"], J["s8"], J["dt3"]
    if s5["lidar"]["guarda_G2"]["passou"] is not True:
        raise RuntimeError("GF: S5 lidar.guarda_G2.passou != true")
    for g, v in dt3["guardas"].items():
        if g.startswith("G2_"):
            for c, vv in v.items():
                if vv["passou"] is not True:
                    raise RuntimeError(f"GF: DT3 {g}.{c} nao passou")
        elif v["passou"] is not True:
            raise RuntimeError(f"GF: DT3 {g} nao passou")

    # ==================================================================
    # E1 -- S5 physical baseline (auditar_s5_baseline_fisico.json)
    # ==================================================================
    cb = s5["reserva"]["coeficientes_baseline"]
    fato("s5_coeficientes", "jstarse_s5_coef_a",
         sel(cb["a_m"], ["media", "dp", "por_semente"]), "m", "s5",
         "reserva.coeficientes_baseline.a_m.{media,dp,por_semente}",
         nota=s5["definicoes"]["baseline"])
    fato("s5_coeficientes", "jstarse_s5_coef_b",
         sel(cb["b_m_por_m"], ["media", "dp", "por_semente"]), "m/m", "s5",
         "reserva.coeficientes_baseline.b_m_por_m.{media,dp,por_semente}",
         nota="sementes na ordem de definicoes.sementes = "
              f"{s5['definicoes']['sementes']}")
    cob = s5["reserva"]["cobertura"]
    fato("s5_cobertura", "jstarse_s5_n_reserva_intersecao",
         sel(cob, ["n_reserva", "n_intersecao_comum", "n_validas_por_braco",
                   "n_intersecao_por_quadrante"]), "ancoras", "s5",
         "reserva.cobertura.{n_reserva,n_intersecao_comum,"
         "n_validas_por_braco,n_intersecao_por_quadrante}",
         nota=cob["nota"])
    BRACO_ID = {
        "baseline": "baseline", "fabdem": "fabdem", "gedtm30": "gedtm30",
        "anadem": "anadem", "gatv2_x4_D8 (Tabela II)": "gatv2_x4_D8_tabII",
        "mlp_x4_D8 (Tabela II)": "mlp_x4_D8_tabII",
        "v23_gatv2_b4s035 (receita adotada)": "v23_gatv2_b4s035_adotada",
        "mlp_b4s035": "mlp_b4s035",
        "arvore (carimbado, sem predicao persistida)": "arvore_carimbado",
    }
    resumo = s5["reserva"]["skill_por_braco_resumo"]
    if set(resumo) != set(BRACO_ID):
        raise RuntimeError(f"S5: bracos inesperados: {sorted(resumo)}")
    for br, bid in BRACO_ID.items():
        val, pod = podar(resumo[br], br)
        fato("s5_skill_reserva", f"jstarse_s5_skill_{bid}",
             {"braco": br, **val}, "skill (adim.); mae_m em m", "s5",
             f"reserva.skill_por_braco_resumo.'{br}'.{{completa,intersecao}}",
             nota="completa = todas as ancoras de reserva validas no braco; "
                  "intersecao = ancoras comuns a todos os bracos (n_intersecao_"
                  "comum); skill = eq. (1) (" + s5["definicoes"]["skill"][:60]
                  + "...)", podadas=pod)
    # contrasts against the lidar
    testes = s5["lidar"]["testes"]
    CAMPOS_T = ["testado", "motivo", "n_transectos", "mediana_m", "media_m",
                "ic95_bootstrap_transecto_mediana", "vence", "total",
                "empates", "p_sinal_binomial", "p_wilcoxon",
                "dp_entre_transectos_m", "mde80_m"]
    for u in UNIDADES:
        for par, pid in (("v23 vs baseline", "v23_menos_baseline"),
                         ("mlp vs baseline", "mlp_menos_baseline")):
            for met in ("mae", "nmad"):
                for c in CLASSES:
                    t = testes[u][par]["principal_ETH"][met][c]
                    val, pod = podar(sel(t, CAMPOS_T))
                    fato("s5_contrastes_lidar",
                         f"jstarse_s5_{pid}_{met}_{CID[c]}_{u}", val, "m",
                         "s5",
                         f"lidar.testes.{u}.'{par}'.principal_ETH.{met}."
                         f"'{c}'.{{{','.join(CAMPOS_T)}}}",
                         nota="diferenca = metrica(a) - metrica(b) por "
                              "unidade (a,b na ordem do par); vence = n de "
                              "unidades com diferenca < 0 (REF.testar, "
                              "auditar_nmad_pareado.py l.191); IC95 = "
                              "bootstrap da mediana por unidade", podadas=pod)
    # sensitivity: lidar canopy class (dossel_v2), same format as the ETH tests
    for u in UNIDADES:
        for par, pid in (("v23 vs baseline", "v23_menos_baseline"),
                         ("mlp vs baseline", "mlp_menos_baseline")):
            for met in ("mae", "nmad"):
                for c in CLASSES:
                    t = testes[u][par]["sensibilidade_dossel_v2"][met][c]
                    val, pod = podar(sel(t, CAMPOS_T))
                    fato("s5_contrastes_lidar_dossel_v2",
                         f"jstarse_s5_{pid}_{met}_{CID[c]}_{u}_dossel_v2", val,
                         "m", "s5",
                         f"lidar.testes.{u}.'{par}'.sensibilidade_dossel_v2."
                         f"{met}.'{c}'.{{{','.join(CAMPOS_T)}}}",
                         nota="sensibilidade: classes pelo dossel do lidar "
                              "(dossel_v2, circular) em vez do ETH; diferenca "
                              "= metrica(a) - metrica(b) por unidade; vence = "
                              "n de unidades com diferenca < 0 (REF.testar); "
                              "IC95 = bootstrap da mediana por unidade",
                         podadas=pod)
    pb = s5["lidar"]["tabela_pool_ETH_baseline"]
    pr = s5["lidar"]["tabela_pool_ETH_v23_mlp_referencia"]
    for c in CLASSES:
        fato("s5_pool_descritivo", f"jstarse_s5_pool_{CID[c]}",
             {"baseline": sel(pb[c], ["n", "mae", "nmad"]),
              "v23": sel(pr["v23"][c], ["n", "mae", "nmad"]),
              "mlp": sel(pr["mlp"][c], ["n", "mae", "nmad"])},
             "m", "s5",
             f"lidar.tabela_pool_ETH_baseline.'{c}'.{{n,mae,nmad}} + "
             f"lidar.tabela_pool_ETH_v23_mlp_referencia.{{v23,mlp}}.'{c}'."
             "{n,mae,nmad}",
             nota="descritivo (pool de celulas da classe ETH), sem IC")
    lpd = s5["lidar"]["leitura_pre_declarada"]
    fato("s5_leitura_pre_declarada", "jstarse_s5_leitura_pre_declarada",
         {"regra": lpd["regra"], "vale": lpd["vale"],
          "texto_pre_declarado": lpd["texto_pre_declarado"],
          "checagens_por_unidade": lpd["checagens_por_unidade"],
          "X_m": lpd["X_m"]}, "m", "s5",
         "lidar.leitura_pre_declarada.{regra,vale,texto_pre_declarado,"
         "checagens_por_unidade,X_m}")
    pbs = s5["lidar"]["produto_baseline"]
    fato("s5_leitura_pre_declarada", "jstarse_s5_produto_baseline",
         sel(pbs, ["definicao", "a_bar_m", "b_bar_m_por_m"]), "m", "s5",
         "lidar.produto_baseline.{definicao,a_bar_m,b_bar_m_por_m}")

    # ==================================================================
    # E2 -- S8 disturbance mask (auditar_s8_mascara_disturbio.json)
    # ==================================================================
    hz = s8["hansen"]
    fato("s8_hansen", "jstarse_s8_hansen_versao_tiles_janela",
         {"versao": hz["versao"],
          "tiles": [sel(t, ["tile", "sha256", "bytes", "url"])
                    for t in hz["tiles"]],
          "definicao_perturbada": hz["definicao_perturbada"],
          "janela_lossyear_perturbada": hz["procedimento"]["faixas_lossyear"][
              "perda_2011_2023"],
          "metodo_reamostragem": hz["procedimento"]["metodo"],
          "grade_2x2_graus_perda_2011_2023": hz["grade_2x2_graus"][
              "perda_2011_2023"]},
         None, "s8",
         "hansen.{versao,tiles[*].{tile,sha256,bytes,url},definicao_perturbada,"
         "procedimento.faixas_lossyear.perda_2011_2023,procedimento.metodo,"
         "grade_2x2_graus.perda_2011_2023}",
         nota="janela_lossyear em codigos Hansen (11 = 2011, 23 = 2023)")
    fato("s8_populacao", "jstarse_s8_populacao",
         sel(s8["populacao"], ["ancoras_E4", "ancoras_nao_perturbadas",
                               "floresta_julgavel", "floresta_nao_perturbada"]),
         "ancoras / celulas", "s8",
         "populacao.{ancoras_E4,ancoras_nao_perturbadas,floresta_julgavel,"
         "floresta_nao_perturbada}")
    fr = s8["fracao_removida"]
    fa, fc = fr["ancoras_E4_por_classe_eth"], fr["celulas_floresta_julgavel_por_classe_eth"]
    CR = ["n", "removidas", "fracao_removida"]
    classes_rem = ["TODAS"] + sorted(set(fa["por_classe"]) | set(fc["por_classe"]),
                                     key=lambda c: list(CID).index(c))
    for c in classes_rem:
        va = fa["TODAS"] if c == "TODAS" else fa["por_classe"].get(c)
        vc = fc["TODAS"] if c == "TODAS" else fc["por_classe"].get(c)
        val = {}
        if va is not None:
            val["ancoras_E4"] = sel(va, CR)
        else:
            val["ancoras_E4"] = "classe ausente em fracao_removida.ancoras_E4_por_classe_eth"
        if vc is not None:
            val["celulas_floresta_julgavel"] = sel(vc, CR)
        fato("s8_removidas", f"jstarse_s8_removidas_{CID[c]}", val,
             "contagem / fracao", "s8",
             "fracao_removida.{ancoras_E4_por_classe_eth,"
             "celulas_floresta_julgavel_por_classe_eth}."
             + ("TODAS" if c == "TODAS" else f"por_classe.'{c}'")
             + ".{n,removidas,fracao_removida}",
             nota="removida = perturbada (perda Hansen 2011-2023)")
    # G4: per-class sum of removed points = TODAS (anchors and cells)
    for nome, bloco in (("ancoras", fa), ("celulas", fc)):
        soma = sum(v["removidas"] for v in bloco["por_classe"].values())
        if soma != bloco["TODAS"]["removidas"]:
            raise RuntimeError(f"G4: S8 {nome}: soma por classe {soma} != "
                               f"TODAS {bloco['TODAS']['removidas']}")
    # anchors removed from NP_T-0638 >30m (counted in code)
    lista = s8["celulas_removidas"]["ancoras"]
    if len(lista) != fa["TODAS"]["removidas"]:
        raise RuntimeError("G4: len(celulas_removidas.ancoras) != removidas TODAS")
    for c, v in fa["por_classe"].items():
        n_c = sum(1 for x in lista if x["classe_eth"] == c)
        if n_c != v["removidas"]:
            raise RuntimeError(f"G4: S8 ancoras removidas {c}: lista {n_c} != "
                               f"{v['removidas']}")
    n_0638 = sum(1 for x in lista if x["transecto"] == "NP_T-0638")
    if n_0638 != fa["por_transecto"]["NP_T-0638"]["removidas"]:
        raise RuntimeError("G4: S8 NP_T-0638 removidas: lista != por_transecto")
    n_0638_30 = sum(1 for x in lista
                    if x["transecto"] == "NP_T-0638" and x["classe_eth"] == ">30m")
    fato("s8_removidas", "jstarse_s8_ancoras_removidas_NP_T_0638_30m_mais",
         {"transecto": "NP_T-0638", "classe_eth": ">30m",
          "ancoras_removidas": n_0638_30,
          "ancoras_removidas_NP_T_0638_todas_classes": n_0638},
         "ancoras", "s8",
         "celulas_removidas.ancoras[*] com transecto=='NP_T-0638' e "
         "classe_eth=='>30m' (contagem)",
         operacao="contagem em codigo sobre a lista de 227 ancoras removidas; "
                  "G4: contagem por classe e a do transecto inteiro conferidas "
                  "contra fracao_removida.ancoras_E4_por_classe_eth.{por_classe,"
                  "por_transecto.'NP_T-0638'}.removidas")
    # A with the mask
    acm = s8["A_por_classe_eth"]["com_mascara_nao_perturbadas"]
    CA = ["n_celulas", "n_unidades", "media_pond_celula_m", "mediana_m",
          "ic95_pond_celula", "ic95_mediana", "ic95_t_jackknife_pond"]
    CRED = ["n_celulas", "n_unidades", "reducao_mae_equivalente_m",
            "ic95_pond_celula", "ic95_t_jackknife"]
    for c in acm:
        for u in UNIDADES:
            b = acm[c][u]
            val, pod = podar({"A": sel(b["A"], CA),
                              "reducao_mae_equivalente": sel(
                                  b["reducao_mae_equivalente"], CRED)})
            fato("s8_A_com_mascara", f"jstarse_s8_A_com_mascara_{CID[c]}_{u}",
                 val, "m", "s8",
                 f"A_por_classe_eth.com_mascara_nao_perturbadas.'{c}'.{u}."
                 f"{{A.{{{','.join(CA)}}},reducao_mae_equivalente."
                 f"{{{','.join(CRED)}}}}}",
                 nota="celulas nao perturbadas; A = solo_anc - z_ref, "
                      "bootstrap por unidade (B=10.000)", podadas=pod)
    # S and O verdicts with/without the mask
    cso = s8["criterios_S_O"]
    for c in CLASSES:
        b = cso[c]
        val = {
            "S_sem_mascara": b["sem_mascara"]["S"]["veredito"],
            "O_sem_mascara_teto_familias": b["sem_mascara"]["O_teto_familias"]["veredito"],
            "S_com_mascara": b["com_mascara"]["S"]["veredito"],
            "O_com_mascara_teto_recomputado_primario":
                b["com_mascara"]["O_teto_recomputado_primario"]["veredito"],
            "O_com_mascara_teto_fixo_28_09_sensibilidade":
                b["com_mascara"]["O_teto_fixo_28_09_sensibilidade"]["veredito"],
            "teto_sem_mascara": sel(b["teto_sem_mascara"], ["teto_m", "ic95_m", "n_transectos"]),
            "teto_com_mascara": sel(b["teto_com_mascara"], ["teto_m", "ic95_m", "n_transectos"]),
            "O_com_mascara_primario_checagens":
                b["com_mascara"]["O_teto_recomputado_primario"]["checagens_por_unidade"],
            "S_com_mascara_checagens": b["com_mascara"]["S"]["checagens_por_unidade"],
        }
        fato("s8_vereditos_S_O", f"jstarse_s8_vereditos_S_O_{CID[c]}", val,
             "m (tetos e limites)", "s8",
             f"criterios_S_O.'{c}'.{{sem_mascara.{{S,O_teto_familias}}.veredito,"
             "com_mascara.{S,O_teto_recomputado_primario,"
             "O_teto_fixo_28_09_sensibilidade}.veredito,teto_sem_mascara,"
             "teto_com_mascara,com_mascara.{O_teto_recomputado_primario,S}."
             "checagens_por_unidade}")
    # Table IV contrasts with/without the mask
    tiv = s8["tabela_IV"]
    cp = tiv["contrastes_pareados_v23_menos_produto"]
    CC = ["testado", "motivo", "n_transectos", "mediana_m", "media_m", "ic95_m",
          "ic_exclui_zero", "sinal_mediana", "vence_a", "p_wilcoxon", "p_holm"]
    for par in cp:
        pid = par.replace("v23 vs ", "v23_menos_")
        for met in ("mae", "nmad"):
            val, pod = podar({c: {"sem_mascara": sel(cp[par][met][c]["sem_mascara"], CC),
                                  "com_mascara": sel(cp[par][met][c]["com_mascara"], CC)}
                              for c in cp[par][met]})
            fato("s8_contrastes_tab_IV", f"jstarse_s8_{pid}_{met}", val, "m",
                 "s8",
                 f"tabela_IV.contrastes_pareados_v23_menos_produto.'{par}'."
                 f"{met}.*.{{sem_mascara,com_mascara}}.{{{','.join(CC)}}}",
                 nota=tiv["nota"] + "; vence_a = n de transectos em que v23 "
                      "tem metrica menor", podadas=pod)
    # highlighted facts (same fields, own id)
    for met, c, rot in (("mae", "20-30m", "v23_menos_fabdem_mae_20_30m"),
                        ("nmad", "20-30m", "confirmatorio_nmad_v23_menos_fabdem_20_30m"),
                        ("nmad", ">30m", "confirmatorio_nmad_v23_menos_fabdem_30m_mais")):
        b = cp["v23 vs fabdem"][met][c]
        val, pod = podar({"sem_mascara": sel(b["sem_mascara"], CC),
                          "com_mascara": sel(b["com_mascara"], CC)})
        if met == "nmad":
            for k in ("sem_mascara", "com_mascara"):
                if "p_holm" not in val[k]:
                    raise RuntimeError(f"S8: p_holm ausente no confirmatorio {c} {k}")
        fato("s8_destaques", f"jstarse_s8_{rot}", val, "m", "s8",
             f"tabela_IV.contrastes_pareados_v23_menos_produto.'v23 vs fabdem'."
             f"{met}.'{c}'.{{sem_mascara,com_mascara}}",
             nota="Holm so no v23 vs fabdem (familia confirmatoria original)"
                  if met == "nmad" else None, podadas=pod)
    # change of order in 20-30m (derived from the aggregates)
    psem, pcom = tiv["pool_ETH_sem_mascara"], tiv["pool_ETH_com_mascara"]
    for met, (x, y) in (("mae", ("fabdem", "gedtm30")),
                        ("nmad", ("anadem", "mlp"))):
        c = "20-30m"
        ordem_sem = sorted(PRODUTOS_TAB_IV, key=lambda p: psem[p][c][met])
        ordem_com = sorted(PRODUTOS_TAB_IV, key=lambda p: pcom[p][c][met])
        fato("s8_mudanca_ordem_20_30m", f"jstarse_s8_ordem_{met}_20_30m",
             {"metrica": met, "par_destacado": [x, y],
              "valores_sem_mascara_m": {p: psem[p][c][met] for p in PRODUTOS_TAB_IV},
              "valores_com_mascara_m": {p: pcom[p][c][met] for p in PRODUTOS_TAB_IV},
              "n_sem_mascara": psem["v23"][c]["n"],
              "n_com_mascara": pcom["v23"][c]["n"],
              "ordem_crescente_sem_mascara": ordem_sem,
              "ordem_crescente_com_mascara": ordem_com,
              f"{x}_menor_que_{y}_sem_mascara": psem[x][c][met] < psem[y][c][met],
              f"{x}_menor_que_{y}_com_mascara": pcom[x][c][met] < pcom[y][c][met],
              "ordem_do_par_troca": (psem[x][c][met] < psem[y][c][met])
                                    != (pcom[x][c][met] < pcom[y][c][met])},
             "m", "s8",
             f"tabela_IV.{{pool_ETH_sem_mascara,pool_ETH_com_mascara}}.*.'{c}'.{met}",
             operacao="ordenacao crescente dos 6 produtos pelo agregado da "
                      "classe, feita em codigo; o artefato nao grava a ordem "
                      "como campo (G4: sem contraparte gravada)",
             nota="descritivo; fora do criterio 'robusto' do desenho S8")
    vs8 = s8["veredito_S8"]
    fato("s8_veredito", "jstarse_s8_veredito",
         {"veredito": vs8["veredito"], "criterio": vs8["criterio"],
          "o_que_mudou_no_criterio": vs8["o_que_mudou_no_criterio"],
          "mudancas_descritivas_fora_do_criterio": vs8["mudancas_descritivas_fora_do_criterio"],
          "contrastes_tab4_com_ic_excluindo_zero_sem_mascara":
              vs8["contrastes_tab4_com_ic_excluindo_zero_sem_mascara"],
          "O3": vs8["O3"]}, None, "s8",
         "veredito_S8.{veredito,criterio,o_que_mudou_no_criterio,"
         "mudancas_descritivas_fora_do_criterio,"
         "contrastes_tab4_com_ic_excluindo_zero_sem_mascara,O3}")
    if len(vs8["o_que_mudou_no_criterio"]) == 0 and vs8["veredito"] != "ROBUSTO":
        raise RuntimeError("G4: S8 nada mudou no criterio mas veredito != ROBUSTO")

    # ==================================================================
    # E3 -- D-T3 (auditar_dt3_criterios_registro.json)
    # ==================================================================
    REG = {"i_glo30": "glo30", "ii_b_gedtm30": "gedtm30",
           "iii_sem_offset": "sem_offset"}
    tab = dt3["tabela_vereditos_registro_classe_unidade"]
    for reg, rid in REG.items():
        for c in CLASSES:
            b = dt3["registros"][reg]["por_classe"][c]
            cr = b["criterios"]
            A_u = {u: sel(b["por_unidade"][u]["A"],
                          ["n_celulas", "n_unidades", "media_pond_celula_m",
                           "mediana_m", "ic95_pond_celula", "ic95_mediana"])
                   for u in UNIDADES}
            red_u = {u: sel(b["por_unidade"][u]["reducao_mae_equivalente"],
                            ["reducao_mae_equivalente_m", "ic95_pond_celula"])
                     for u in UNIDADES}
            linhas = [sel(x, ["unidade", "S_passa", "O_familias_28_09_passa",
                              "O_teto_por_unidade_passa", "R_passa"])
                      for x in tab if x["registro"] == reg and x["classe"] == c]
            if len(linhas) != 2:
                raise RuntimeError(f"G4: DT3 tabela_vereditos {reg} {c}: "
                                   f"{len(linhas)} linhas (esperado 2)")
            val = {
                "registro": reg,
                "A_por_unidade": A_u,
                "reducao_mae_equivalente_por_unidade": red_u,
                "O_limites_inferiores_por_unidade":
                    cr["O_familias_28_09"]["checagens_por_unidade"],
                "teto_O_familias_28_09_m": cr["O_familias_28_09"]["teto_abs_contrastes_lidar_m"],
                "teto_de": cr["O_familias_28_09"].get("teto_de", "campo ausente"),
                "teto_familias_v23_x_mlp_por_unidade_m":
                    {u: b["teto_familias_v23_x_mlp"][u]["teto_m"] for u in UNIDADES},
                "veredito_S": cr["S"]["veredito"],
                "veredito_O": cr["O_familias_28_09"]["veredito"],
                "veredito_O_leitura": cr["O_familias_28_09"]["veredito_leitura"],
                "veredito_O_secundario_teto_por_unidade":
                    cr["O_familias_teto_por_unidade_secundario"]["veredito_leitura"],
                "veredito_R": cr["R"]["veredito"],
                "vereditos_por_unidade": linhas,
            }
            val, pod = podar(val)
            fato("dt3_registro_classe", f"jstarse_dt3_{rid}_{CID[c]}", val, "m",
                 "dt3",
                 f"registros.'{reg}'.por_classe.'{c}'.{{por_unidade.*.{{A,"
                 "reducao_mae_equivalente}},criterios.{S,R,O_familias_28_09,"
                 "O_familias_teto_por_unidade_secundario},teto_familias_v23_x_mlp}"
                 f" + tabela_vereditos_registro_classe_unidade[registro=='{reg}',"
                 f"classe=='{c}']", podadas=pod)
    # Note: the field
    # estabilidade_dos_vereditos.*.*.registros_que_mudam_o_veredito lists the
    # registration variants that DIFFER from GLO-30, not those where the
    # criterion fails. The list of variants where each criterion FAILS is
    # derived here, in code, from the per-variant verdicts and checked (G4)
    # against the *_passa flags of tabela_vereditos_registro_classe_unidade
    # (pass = both units pass).
    est = dt3["estabilidade_dos_vereditos"]
    PASSA = {"S": "positivo", "O": "demonstrado", "O_sec": "demonstrado",
             "R": "ordenacao_valida_A_menos_delta_maior_que_zero"}
    FLAG = {"S": "S_passa", "O": "O_familias_28_09_passa",
            "O_sec": "O_teto_por_unidade_passa", "R": "R_passa"}
    falhas = {}
    for crit, por_c in est.items():
        if crit not in PASSA:
            raise RuntimeError(f"DT3: criterio inesperado em estabilidade: {crit}")
        falhas[crit] = {}
        for c, x in por_c.items():
            falha = sorted(r for r, v in x["vereditos"].items() if v != PASSA[crit])
            for r in x["vereditos"]:
                linhas_rc = [t for t in tab if t["registro"] == r and t["classe"] == c]
                passa_tab = all(t[FLAG[crit]] for t in linhas_rc)
                if passa_tab == (r in falha):
                    raise RuntimeError(f"G4: DT3 {crit} {c} {r}: veredito "
                                       f"'{x['vereditos'][r]}' incoerente com "
                                       f"{FLAG[crit]} da tabela_vereditos")
            falhas[crit][c] = {
                "vereditos_por_registro": x["vereditos"],
                "veredito_que_passa": PASSA[crit],
                "registros_em_que_o_criterio_falha": falha,
                "registros_que_diferem_do_glo30 (campo registros_que_mudam_o_veredito)":
                    x["registros_que_mudam_o_veredito"],
            }
    fato("dt3_leitura", "jstarse_dt3_registros_em_que_cada_criterio_falha",
         falhas, None, "dt3",
         "estabilidade_dos_vereditos.{S,O,O_sec,R}.{'20-30m','>30m'}.vereditos "
         "(+ tabela_vereditos_registro_classe_unidade[*].*_passa para G4)",
         operacao="registros_em_que_o_criterio_falha = registros cujo veredito "
                  "!= veredito_que_passa, calculado em codigo; G4: coincide com "
                  "not all(*_passa nas duas unidades) da tabela_vereditos",
         nota=("the artifact field registros_que_mudam_o_veredito lists the records that "
               "DIFFER from GLO-30 (i_glo30), NOT the records in which the criterion "
               "fails; do not use it as 'record that fails'"))
    lp = dt3["leitura_pre_declarada"]
    fato("dt3_leitura", "jstarse_dt3_veredito_final",
         {**sel(lp, ["regra_literal", "O_maior30_por_registro",
                     "titulo_nao_depende_do_registro",
                     "O_20_30_por_registro", "O3"]),
          "registros_em_que_falha_O_maior30 (campo registros_em_que_falha)":
              lp["registros_em_que_falha"]},
         None, "dt3",
         "leitura_pre_declarada.{regra_literal,O_maior30_por_registro,"
         "titulo_nao_depende_do_registro,registros_em_que_falha,"
         "O_20_30_por_registro,O3}",
         nota="registros_em_que_falha do artefato refere-se so ao (O) acima de "
              "30 m (escopo da regra_literal); para (S), (R), (O) 20-30 m ver "
              "jstarse_dt3_registros_em_que_cada_criterio_falha")
    if sorted(lp["registros_em_que_falha"]) != falhas["O"][">30m"][
            "registros_em_que_o_criterio_falha"]:
        raise RuntimeError("G4: DT3 registros_em_que_falha (O >30m) != derivado")
    # G4: 'sim' <=> O >30m 'demonstrado' in all three registration variants
    tres = all(v == "demonstrado" for v in lp["O_maior30_por_registro"].values())
    if (lp["titulo_nao_depende_do_registro"] == "sim") != tres:
        raise RuntimeError("G4: DT3 titulo_nao_depende_do_registro incoerente "
                           "com O_maior30_por_registro")

    # ==================================================================
    # E4 -- T7 (Table IV JSON: auditar_nmad_pareado_nativa_anadem.json)
    # ==================================================================
    na = J["nmad_anadem"]
    orig = na["populacao_original_ETH"]["testes_principal_ETH_produtos_base"]
    comum = na["populacao_comum_5_produtos_ETH"]["testes_principal_ETH"]
    pool = na["populacao_comum_5_produtos_ETH"]["tabela_pool_ETH"]
    CT7 = ["testado", "n_transectos", "mediana_m", "media_m",
           "ic95_bootstrap_transecto_mediana", "vence", "total", "empates",
           "p_sinal_binomial", "p_wilcoxon", "p_holm",
           "dp_entre_transectos_m", "mde80_m", "mde80_metodo"]
    for c in CLASSES:
        t = orig["v23 vs fabdem"]["nmad"][c]
        fato("t7_mde", f"jstarse_t7_mde80_confirmatorio_nmad_v23_fabdem_{CID[c]}",
             {**sel(t, ["mde80_m", "mde80_metodo", "dp_entre_transectos_m",
                        "n_transectos", "mediana_m", "p_holm"]),
              "poder_criterio_pre_registrado": na["criterio_pre_registrado"]["poder"],
              "alpha": na["criterio_pre_registrado"]["alpha"],
              "correcao": na["criterio_pre_registrado"]["correcao"]},
             "m", "nmad_anadem",
             "populacao_original_ETH.testes_principal_ETH_produtos_base."
             f"'v23 vs fabdem'.nmad.'{c}'.{{mde80_m,mde80_metodo,"
             "dp_entre_transectos_m,n_transectos,mediana_m,p_holm}} + "
             "criterio_pre_registrado.{poder,alpha,correcao}",
             nota="o que o MDE mede, nas palavras do artefato: "
                  "criterio_pre_registrado.poder e mde80_metodo (acima)")
    for c in CLASSES:
        v, m = pool["v23"][c], pool["mlp"][c]
        fato("t7_nmad_agregado", f"jstarse_t7_nmad_agregado_v23_mlp_{CID[c]}",
             {"v23_nmad_m": v["nmad"], "mlp_nmad_m": m["nmad"], "n": v["n"],
              "diferenca_v23_menos_mlp_m": v["nmad"] - m["nmad"]},
             "m", "nmad_anadem",
             f"populacao_comum_5_produtos_ETH.tabela_pool_ETH.{{v23,mlp}}.'{c}'."
             "{nmad,n}",
             operacao="diferenca_v23_menos_mlp_m = nmad(v23) - nmad(mlp), "
                      "calculada em codigo (os dois operandos sao campos)")
        if v["n"] != m["n"]:
            raise RuntimeError(f"G4: n v23 != n mlp no pool {c}")
        tc = comum["v23 vs mlp"]["nmad"][c]
        to = orig["v23 vs mlp"]["nmad"][c]
        for k in ("mediana_m", "media_m", "mde80_m"):
            igual(tc[k], to[k], 1e-12, f"T7 pareado v23-mlp nmad {c} {k} "
                  "comum vs original")
        if tc["ic95_bootstrap_transecto_mediana"] != to["ic95_bootstrap_transecto_mediana"] \
                or tc["vence"] != to["vence"] or tc["total"] != to["total"]:
            raise RuntimeError(f"G4: T7 pareado v23-mlp nmad {c}: comum != original")
        val, pod = podar(sel(tc, CT7))
        fato("t7_nmad_pareado", f"jstarse_t7_pareado_nmad_v23_menos_mlp_{CID[c]}",
             val, "m", "nmad_anadem",
             f"populacao_comum_5_produtos_ETH.testes_principal_ETH.'v23 vs mlp'."
             f"nmad.'{c}'.{{{','.join(CT7)}}}",
             nota="diferenca = NMAD(v23) - NMAD(mlp) por transecto; vence = n "
                  "de transectos com diferenca < 0; G4: identico a populacao_"
                  "original_ETH.testes_principal_ETH_produtos_base (mesmo "
                  "valor de jstarsc_contraste_mlp_nmad_*); p_holm ausente = "
                  "par fora da familia confirmatoria", podadas=pod)

    # ==================================================================
    # E5 -- T6: v23 vs mlp against the lidar (tabela_orcamento_P1) and
    # aggregated differences of Table IV
    # ==================================================================
    top = J["rotulo_v2"]["tabela_orcamento_P1"]
    for c in CLASSES:
        det = [x for x in top[c]["contrastes_lidar_detalhe"] if x["par"] == "v23 vs mlp"]
        if len(det) != 1:
            raise RuntimeError(f"T6: v23 vs mlp nao unico em tabela_orcamento_P1 {c}")
        d0 = det[0]
        # G4: same MAE contrast in the Table IV JSON (same population)
        tm = orig["v23 vs mlp"]["mae"][c]
        igual(d0["mediana_m"], tm["mediana_m"], 1e-6, f"T6 dMAE mediana {c}")
        for i in (0, 1):
            igual(d0["ic95_bootstrap_transecto_mediana_m"][i],
                  tm["ic95_bootstrap_transecto_mediana"][i], 1e-6, f"T6 dMAE ic {c}")
        fato("t6_v23_mlp_lidar", f"jstarse_t6_dMAE_v23_menos_mlp_lidar_{CID[c]}",
             sel(d0, ["mediana_m", "media_m", "ic95_bootstrap_transecto_mediana_m",
                      "teto_abs_ddiferenca_m", "n_transectos", "testado"]),
             "m", "rotulo_v2",
             f"tabela_orcamento_P1.'{c}'.contrastes_lidar_detalhe[par=='v23 vs "
             "mlp'].{mediana_m,media_m,ic95_bootstrap_transecto_mediana_m,"
             "teto_abs_ddiferenca_m,n_transectos,testado}",
             nota="G4: mediana e IC95 conferidos (1e-6) contra "
                  "auditar_nmad_pareado_nativa_anadem.json populacao_original_ETH."
                  f"testes_principal_ETH_produtos_base.'v23 vs mlp'.mae.'{c}'")
        tn = comum["v23 vs mlp"]["nmad"][c]
        fato("t6_v23_mlp_lidar", f"jstarse_t6_dNMAD_v23_menos_mlp_lidar_{CID[c]}",
             sel(tn, ["mediana_m", "media_m", "ic95_bootstrap_transecto_mediana",
                      "n_transectos", "testado"]),
             "m", "nmad_anadem",
             f"populacao_comum_5_produtos_ETH.testes_principal_ETH.'v23 vs mlp'."
             f"nmad.'{c}'.{{mediana_m,media_m,ic95_bootstrap_transecto_mediana,"
             "n_transectos,testado}",
             nota="tabela_orcamento_P1 de auditar_rotulo_regua_v2.json NAO tem "
                  "campo de NMAD; o dNMAD v23-mlp vem do JSON da Tab. IV "
                  f"(mesmo valor de jstarse_t7_pareado_nmad_v23_menos_mlp_{CID[c]})")
    pool_b = {f["id"]: f["valor"] for f in copia["parte_B"]["fatos"]["tabela_iv_pool"]}
    for c in list(pool["v23"].keys()):
        v, m = pool["v23"][c], pool["mlp"][c]
        # G4: operands equal to those in parte_B (tabela_iv_pool)
        idb = f"jstarsb_tabIV_pool_{CID[c]}"
        if idb in pool_b:
            for met in ("mae", "vies", "nmad"):
                igual(pool_b[idb]["v23"][met], v[met], 0.0, f"T6 {idb} v23 {met}")
                igual(pool_b[idb]["mlp"][met], m[met], 0.0, f"T6 {idb} mlp {met}")
        fato("t6_tab_IV_agregado", f"jstarse_t6_tabIV_v23_menos_mlp_{CID[c]}",
             {"mae_m": v["mae"] - m["mae"], "vies_m": v["vies"] - m["vies"],
              "nmad_m": v["nmad"] - m["nmad"], "n": v["n"],
              "operandos": {"v23": sel(v, ["mae", "vies", "nmad"]),
                            "mlp": sel(m, ["mae", "vies", "nmad"])}},
             "m", "nmad_anadem",
             f"populacao_comum_5_produtos_ETH.tabela_pool_ETH.{{v23,mlp}}.'{c}'."
             "{mae,vies,nmad,n}",
             operacao="diferenca v23 - mlp de cada metrica agregada, em codigo; "
                      "G4: operandos = parte_B " + idb)

    # ==================================================================
    # E6 -- T8: selection on validation and lambda curve
    # ==================================================================
    sv = J["selecao"]
    fam = sv["familias"]
    cfg = J["r_adot_gatv2"]["config"]
    adotado = {"suavidade": cfg["peso_suavidade"], "boost": cfg["boost_atl08"]}
    m_guarda = re.search(r"<=\s*([0-9.]+)\s*m", fam["suavidade"]["_regra"])
    if not m_guarda:
        raise RuntimeError("T8: guarda de val_mae nao encontrada em _regra")
    guarda_m = float(m_guarda.group(1))
    for lam, v in fam["suavidade"].items():
        if lam.startswith("_"):
            continue
        dentro = v["delta_vs_0"] <= guarda_m
        if dentro != v["dentro_guarda_val"]:
            raise RuntimeError(f"G4: T8 dentro_guarda_val lambda={lam} incoerente")
    for nome in ("suavidade", "boost"):
        f_ = fam[nome]
        eleito = f_["_eleito"]
        sobrevive = f_["_adotado_sobrevive"]
        if (float(eleito) == float(adotado[nome])) != sobrevive:
            raise RuntimeError(f"G4: T8 {nome}: eleito {eleito} vs adotado "
                               f"{adotado[nome]} incoerente com _adotado_sobrevive")
        numeros = {k: v for k, v in f_.items() if not k.startswith("_")}
        val = {"familia": nome, "regra": f_["_regra"], "eleito_em_validacao": eleito,
               "adotado": adotado[nome], "adotado_sobrevive": sobrevive,
               "numeros_da_regra": numeros}
        if nome == "suavidade":
            val["guarda_val_mae_m"] = guarda_m
            val["traducao_guarda_corpo"] = sv["traducao_guarda_corpo"]
        fato("t8_selecao", f"jstarse_t8_selecao_{nome}", val, None,
             ["selecao", "r_adot_gatv2"],
             f"selecao_em_validacao.json: familias.{nome}.{{_regra,_eleito,"
             "_adotado_sobrevive,<valores>}" + (", traducao_guarda_corpo"
                                                 if nome == "suavidade" else "")
             + "; results/ofat_suavfina_b4s035.json: config."
             + ("peso_suavidade" if nome == "suavidade" else "boost_atl08"),
             operacao=("guarda_val_mae_m extraida por regex de familias.suavidade."
                       "_regra; G4: dentro_guarda_val == (delta_vs_0 <= guarda) "
                       "para os 4 lambdas; " if nome == "suavidade" else "")
                      + "G4: _adotado_sobrevive == (eleito == adotado)",
             nota="adotado lido da config da corrida da receita adotada "
                  "(ofat_suavfina_b4s035, braco gatv2); populacoes_usadas = "
                  f"{sv['populacoes_usadas']}; reserva_tocada = {sv['reserva_tocada']}")
    ba4 = J["suav_ba4"]["por_peso"]
    p035 = J["ponto_s035"]["por_peso"]
    igual(ba4["0.0"]["skill"], p035["0.0"]["skill"], 0.0, "T8 skill lambda 0 ba4 vs ponto")
    for lam in ("0.0", "0.25", "0.35", "0.5"):
        v = fam["suavidade"][lam]
        fonte_sk, chave_sk = (p035, "ponto_s035") if lam == "0.35" else (ba4, "suav_ba4")
        r = fonte_sk[lam]
        igual(v["criados_2m"], r["criados_2m_soma"], 1e-6,
              f"T8 criados_2m lambda={lam} selecao vs {chave_sk}")
        fato("t8_curva_lambda", f"jstarse_t8_curva_lambda_{lam.replace('.', '_')}",
             {"lambda": float(lam),
              "val_mae_m": v["val_mae"], "val_mae_dp_m": v["dp"],
              "delta_val_mae_vs_0_m": v["delta_vs_0"],
              "dentro_guarda_val": v["dentro_guarda_val"],
              "skill_reserva": r["skill"], "skill_reserva_dp": r["skill_dp"],
              "skill_delta_vs_0": r["skill_delta_vs_p0"],
              "dentro_guarda_skill_reserva": r["dentro_guarda_corpo"],
              "pocos_2m": v["criados_2m"], "corte_pocos_vs_0": v["corte_criados"],
              "n_sementes": v["n_sementes"]},
             "val_mae em m; skill adim.; pocos em contagem", ["selecao", chave_sk],
             f"selecao_em_validacao.json: familias.suavidade.'{lam}'.{{val_mae,dp,"
             "delta_vs_0,dentro_guarda_val,criados_2m,corte_criados,n_sementes}; "
             f"{ARQS[chave_sk].name}: por_peso.'{lam}'.{{skill,skill_dp,"
             "skill_delta_vs_p0,dentro_guarda_corpo}",
             nota="G4: criados_2m (selecao) == criados_2m_soma do artefato de "
                  "reserva; lambda 0,35 so existe em comparar_ponto_s035.json "
                  "(comparar_suavidade_fina_ba4.json nao tem 0,35)")

    # ==================================================================
    # E7 -- T10 (auditar_f1_rampa.json, auditar_unidade_quadrante.json)
    # ==================================================================
    f1 = J["f1_rampa"]
    uq = J["unid_quad"]
    cd = f1["contraste_direto_delta_F1_menos_delta_1536"]
    fato("t10", "jstarse_t10_contraste_F1_menos_1536",
         sel(cd, ["n", "media", "ic95", "p", "sinal_consistente"]), "skill (adim.)",
         "f1_rampa",
         "contraste_direto_delta_F1_menos_delta_1536.{n,media,ic95,p,sinal_consistente}")
    ex = cd["excedente_d1_1536_calculado"]
    igual(ex["media"], f1["delta_D1_para_memoria"]["media"]
          - f1["delta_1536_reconferido"]["media"], 1e-12, "T10 excedente D1-1536")
    fato("t10", "jstarse_t10_excedente_D1_menos_1536", sel(ex, ["media", "ic95", "p"]),
         "skill (adim.)", "f1_rampa",
         "contraste_direto_delta_F1_menos_delta_1536.excedente_d1_1536_calculado."
         "{media,ic95,p}",
         nota="G4: media == delta_D1_para_memoria.media - delta_1536_reconferido.media")
    fato("t10", "jstarse_t10_mde80_F1_menos_1536",
         sel(cd, ["mde_80pct", "n_minimo_para_poder_com_folga",
                  "poder_com_folga_mde80_le_0p8_excedente"]),
         "skill (adim.)", "f1_rampa",
         "contraste_direto_delta_F1_menos_delta_1536.{mde_80pct,"
         "n_minimo_para_poder_com_folga,poder_com_folga_mde80_le_0p8_excedente}")
    fato("t10", "jstarse_t10_deltas_F1_D1_1536",
         {k: sel(f1[k], ["n", "media", "ic95", "sinal_consistente"])
          for k in ("delta_F1_gatv2_menos_mlp", "delta_D1_para_memoria",
                    "delta_1536_reconferido")},
         "skill (adim.)", "f1_rampa",
         "{delta_F1_gatv2_menos_mlp,delta_D1_para_memoria,delta_1536_reconferido}."
         "{n,media,ic95,sinal_consistente}",
         nota="skill da uniao ponderada por n (delta = gatv2 - mlp, pareado por semente)")
    fato("t10", "jstarse_t10_n_sementes_por_par",
         {par: {"n": len(uq["pares"][par]["sementes"]),
                "sementes": uq["pares"][par]["sementes"],
                "n_por_semente": uq["pares"][par]["por_semente"]["n"]}
          for par in ("lote_adotado", "lote_1536")},
         "sementes", "unid_quad",
         "pares.{lote_adotado,lote_1536}.{sementes,por_semente.n}",
         operacao="n = len(sementes); G4: == por_semente.n")
    for par in ("lote_adotado", "lote_1536"):
        if len(uq["pares"][par]["sementes"]) != uq["pares"][par]["por_semente"]["n"]:
            raise RuntimeError(f"G4: T10 n sementes {par}")

    # Q3: number of seeds where the graph model loses (gatv2 - mlp < 0), per arm
    def por_semente(chave, campo):
        out = {}
        for k, r in J[chave].items():
            if not k.startswith("amazonia"):
                continue
            s = r["seed"]
            if s in out:
                raise RuntimeError(f"semente repetida em {chave}: {s}")
            out[s] = campo(r)
        return out

    q3 = lambda r: r["quadrantes"]["Q3"]["reserva_final"]["comparabilidade"]["skill"]  # noqa: E731
    uni = lambda r: r["reserva_uniao"]["skill_media_ponderada_por_n"]  # noqa: E731
    BRACOS_Q3 = {"ponto_adotado_cap1x": ("r_adot_gatv2", "r_adot_mlp"),
                 "lote_1536": ("r_1536_gatv2", "r_1536_mlp"),
                 "D1_cap4x_rampa_x1": ("r_d1_gatv2", "r_d1_mlp"),
                 "F1_cap4x_rampa_x4": ("r_f1_gatv2", "r_f1_mlp")}
    q3_val = {}
    for nome, (kg, km) in BRACOS_Q3.items():
        g, m = por_semente(kg, q3), por_semente(km, q3)
        if set(g) != set(m):
            raise RuntimeError(f"T10 Q3 {nome}: sementes nao pareiam")
        dif = {s: g[s] - m[s] for s in g}
        q3_val[nome] = {"perde_em": sum(1 for d in dif.values() if d < 0),
                        "de": len(dif), "sementes": sorted(dif)}
    # G4 against the stored fields
    for nome, par in (("ponto_adotado_cap1x", "lote_adotado"), ("lote_1536", "lote_1536")):
        sq = uq["pares"][par]["sinal_por_quadrante"]["Q3"]
        if q3_val[nome]["de"] != sq["de"] or \
                q3_val[nome]["perde_em"] != sq["de"] - sq["positivo_em_n_sementes"]:
            raise RuntimeError(f"G4: T10 Q3 {nome} nao bate com auditar_unidade_"
                               f"quadrante pares.{par}.sinal_por_quadrante.Q3")
    pq3 = f1["por_quadrante_pre_registrado"]["Q3"]
    if pq3["media"] < 0 and q3_val["F1_cap4x_rampa_x4"]["perde_em"] != pq3["sinal_consistente"]:
        raise RuntimeError("G4: T10 Q3 F1 nao bate com auditar_f1_rampa."
                           "por_quadrante_pre_registrado.Q3.sinal_consistente")
    fato("t10", "jstarse_t10_Q3_grafo_perde_por_braco", q3_val, "sementes",
         ["r_adot_gatv2", "r_adot_mlp", "r_1536_gatv2", "r_1536_mlp",
          "r_d1_gatv2", "r_d1_mlp", "r_f1_gatv2", "r_f1_mlp"],
         "<run>.quadrantes.Q3.reserva_final.comparabilidade.skill (gatv2 - mlp "
         "por semente pareada, 8 JSON brutos de results/)",
         operacao="contagem em codigo de sementes com skill_Q3(gatv2) - "
                  "skill_Q3(mlp) < 0; G4: ponto adotado e lote 1536 conferidos "
                  "contra auditar_unidade_quadrante.json pares.*.sinal_por_"
                  "quadrante.Q3; F1 contra auditar_f1_rampa.json por_quadrante_"
                  "pre_registrado.Q3.sinal_consistente; o D1 NAO tem contraparte "
                  "gravada em JSON de auditoria (so a contagem por codigo)")
    # n-weighted skill (text) and simple mean (figure)
    ga, ma = por_semente("r_adot_gatv2", uni), por_semente("r_adot_mlp", uni)
    if set(ga) != set(ma):
        raise RuntimeError("T10: sementes do ponto adotado nao pareiam")
    d_adot = {s: ga[s] - ma[s] for s in ga}
    media_adot = sum(d_adot.values()) / len(d_adot)
    s5r = s5["reserva"]["skill_por_braco_resumo"]
    igual(media_adot, s5r["v23_gatv2_b4s035 (receita adotada)"]["completa"]["skill"]["media"]
          - s5r["mlp_b4s035"]["completa"]["skill"]["media"], 1e-6,
          "T10 skill ponderada adotado (brutos) vs S5 (recalculada)")
    g1, m1 = por_semente("r_1536_gatv2", uni), por_semente("r_1536_mlp", uni)
    media_1536 = sum(g1[s] - m1[s] for s in g1) / len(g1)
    igual(media_1536, f1["delta_1536_reconferido"]["media"], 1e-12,
          "T10 skill ponderada 1536 (brutos) vs delta_1536_reconferido")
    fato("t10", "jstarse_t10_skill_ponderada_por_n_texto",
         {"ponto_adotado": {"media_gatv2_menos_mlp": media_adot, "n": len(d_adot),
                            "por_semente": {str(s): d_adot[s] for s in sorted(d_adot)}},
          "lote_1536": {"media_gatv2_menos_mlp": f1["delta_1536_reconferido"]["media"],
                        "ic95": f1["delta_1536_reconferido"]["ic95"],
                        "n": f1["delta_1536_reconferido"]["n"]}},
         "skill (adim.)", ["r_adot_gatv2", "r_adot_mlp", "f1_rampa"],
         "ponto adotado: results/{ofat_suavfina_b4s035,baseline_mlp_b4s035}.json "
         "<run>.reserva_uniao.skill_media_ponderada_por_n (media das diferencas "
         "pareadas por semente); lote 1536: auditar_f1_rampa.json "
         "delta_1536_reconferido.{media,ic95,n}",
         operacao="ponto adotado: media em codigo das diferencas por semente "
                  "(nao ha campo gravado); G4: == diferenca das medias de "
                  "auditar_s5_baseline_fisico.json reserva.skill_por_braco_resumo "
                  "(recalculadas) a 1e-6; lote 1536: G4 recomputo dos brutos == "
                  "campo a 1e-12")
    fato("t10", "jstarse_t10_skill_media_simples_figura",
         {par: sel(uq["pares"][par]["por_semente"], ["n", "media", "ic95", "unidade"])
          for par in ("lote_adotado", "lote_1536")},
         "skill (adim.)", "unid_quad",
         "pares.{lote_adotado,lote_1536}.por_semente.{n,media,ic95,unidade}",
         nota="media simples entre os 4 quadrantes por semente "
              "(auditar_unidade_quadrante.py l.161, d_sem = M.mean(axis=0)), "
              "base da Fig. 9")

    # ==================================================================
    # E8 -- S2 "28 of 30"
    # ==================================================================
    s2 = J["s2"]
    cpr = s2["criterio_pre_registrado"]
    vpc = s2["veredito_por_par_e_classe"]
    denom = cpr["n_pares"] * len(cpr["classes_do_criterio"])
    n_comb = sum(len(vpc[c]) for c in cpr["classes_do_criterio"])
    if n_comb != denom or len(cpr["pares"]) != cpr["n_pares"]:
        raise RuntimeError(f"G4: S2 denominador {denom} != combinacoes {n_comb}")
    nao_dep = sum(1 for c in cpr["classes_do_criterio"] for i in vpc[c].values()
                  if i["veredito"] == "nao depende")
    resumo_s2 = s2["resumo_vereditos"]
    if nao_dep != sum(resumo_s2[c]["nao depende"] for c in resumo_s2):
        raise RuntimeError("G4: S2 'nao depende' por busca != resumo_vereditos")
    exc = []
    for c in cpr["classes_do_criterio"]:
        for p, i in vpc[c].items():
            if i["veredito"] != "nao depende":
                ii = dict(i)
                ii["inverte_por_braco"] = {
                    k: (v if v is not None else "nao avaliavel (braco ii-a)")
                    for k, v in i["inverte_por_braco"].items()}
                exc.append({"classe": c, "par": p, **ii})
    fato("s2_28_de_30", "jstarse_s2_28_de_30",
         {"n_pares": cpr["n_pares"], "classes_do_criterio": cpr["classes_do_criterio"],
          "denominador": denom, "nao_depende": nao_dep,
          "excecoes": exc, "regra_nao_depende": cpr["regra_nao_depende"]},
         "pares x classes", "s2",
         "criterio_pre_registrado.{n_pares,pares,classes_do_criterio,"
         "regra_nao_depende} + veredito_por_par_e_classe.*.*.veredito",
         operacao="denominador = n_pares x len(classes_do_criterio); G4: == n de "
                  "combinacoes em veredito_por_par_e_classe; nao_depende por "
                  "busca == soma de resumo_vereditos; excecoes = veredito != "
                  "'nao depende'")

    # ==================================================================
    # source quality (read in code; nothing is corrected)
    # ==================================================================
    def estado_md(k):
        for ln in TXT[k].splitlines():
            if re.match(r"^\s*(STATUS|Estado)\s*:", ln) or re.search(
                    r"\|\s*Estado\s*:", ln):
                return ln.strip()
        return "linha de estado nao encontrada"

    qual = {}
    for k in ARQS:
        if k in ("folha_v22", "ca_s5", "ca_s8", "ca_dt3", "engia_t8_t10"):
            continue
        c = CARIMBO[k]
        q = {"sha256": c["sha256"], "script_origem": c.get("script_origem"),
             "sha256_script": c.get("sha256_script", "AUSENTE"),
             "retroativo": c.get("retroativo", False)}
        ca = J[k].get("contra_auditoria") if isinstance(J[k], dict) else None
        if ca is not None:
            q["campo_contra_auditoria_no_artefato"] = ca
        qual[ARQS[k].relative_to(RAIZ).as_posix()] = q
    for k, alvo in (("ca_s5", "s5"), ("ca_s8", "s8"), ("ca_dt3", "dt3")):
        qual[ARQS[alvo].name]["contra_auditoria_md"] = {
            "arquivo": ARQS[k].relative_to(RAIZ).as_posix(),
            "sha256_no_momento_da_leitura": sha0[k],
            "linha_de_estado": estado_md(k)}
    sem_script = sorted(n for n, q in qual.items()
                        if str(q["script_origem"]).startswith("AUSENTE")
                        or q["sha256_script"] == "AUSENTE")
    qualidade = {
        "fontes": qual,
        "sem_script_carimbado_ou_sem_sha256_do_script": sem_script,
        "parecer_citado_T8_T10": {
            "arquivo": ARQS["engia_t8_t10"].relative_to(RAIZ).as_posix(),
            "sha256": sha0["engia_t8_t10"],
            "linha_de_estado": estado_md("engia_t8_t10")},
        "notas": [
            "selecao_em_validacao.json e comparar_suavidade_fina_ba4.json estao "
            "na raiz do V23, nao em results/ (o pedido citava results/)",
            "comparar_suavidade_fina_ba4.json nao tem _proveniencia nem _fontes",
            "comparar_ponto_s035.json: _proveniencia.script = null, _fontes = [] "
            "(carimbo retroativo 2026-09-03)",
            "selecao_em_validacao.json: carimbo retroativo, sem sha256 do script",
            "auditar_unidade_quadrante.json e auditar_f1_rampa.json: carimbo "
            "anterior ao campo sha256_script (so commit)",
            "tabela_orcamento_P1 (auditar_rotulo_regua_v2.json) nao tem NMAD; "
            "dNMAD v23-mlp lido do JSON da Tab. IV",
            "Q3 do D1 (cap 4x, rampa x1) e skill ponderada do ponto adotado nao "
            "sao campos de JSON de auditoria: contados/derivados por codigo dos "
            "results/ brutos (fatos marcados derivado)",
            ("S5 and D-T3: the independent-check field of the artifact = PENDING; read "
             "the current status of that check at reading time"),
        ],
    }

    # ==================================================================
    # G3 (end): no source changed during the run
    # ==================================================================
    for k, p in ARQS.items():
        if PROV.sha256(p) != sha0[k]:
            raise RuntimeError(f"G3: {p.name} mudou durante a corrida")

    n_fatos_e = sum(len(v) for v in grupos.values())
    saida = {
        "gerado_para": v22["gerado_para"],
        "regra": v22["regra"],
        "protocolo": (("Design of checks S5, S7 and S8 (their JSON outputs enter fact sheet "
                       "v23); parts A-D = folha_de_fatos_jstars_v22.json (sha256 ")
                      + SHA_ESPERADO["folha_v22"][:8] + "...), identicas"),
        "protocolo_v22": v22["protocolo"],
        **copia,
        "parte_E": {
            "n_fatos": n_fatos_e,
            "grupos": {g: len(v) for g, v in grupos.items()},
            "qualidade_das_fontes": qualidade,
            "fatos": grupos,
        },
        "guardas": {
            "GI_partes_A_D_identicas_a_v22": True,
            "G1_ids_unicos": {"n_ids_v22": n_ids_v22, "n_ids_total": len(ids)},
            "G2_sem_nulos": True, "G3_sha256_conferidos": True,
            "G4_derivados_conferidos": True, "GF_guardas_das_fontes": True,
        },
    }
    saida_p = RAIZ / a.saida
    PROV.gravar(saida_p, saida, fontes_lidas=list(ARQS.values()), script=__file__)

    # GI (2): re-read from disk and compare for equality with v22
    relido = json.loads(saida_p.read_text(encoding="utf-8"))
    v22_disco = json.loads(ARQS["folha_v22"].read_text(encoding="utf-8"))
    for k in COPIADAS:
        if relido[k] != v22_disco[k]:
            raise RuntimeError(f"GI: {k} no JSON gravado difere da v22")
    log(f"GI ok: {', '.join(COPIADAS)} identicas a v22 (==, apos releitura)")
    log(f"parte_E: {n_fatos_e} fatos em {len(grupos)} grupos -> {a.saida}")
    for g, v in grupos.items():
        log(f"  {g}: {len(v)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
