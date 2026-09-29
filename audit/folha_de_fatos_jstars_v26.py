# -*- coding: utf-8 -*-
"""folha_de_fatos_jstars_v26 -- adds part H (R-O3: robustness against the
three-family ceiling) to the fact sheet for Article 1 of V23 (target: IEEE
JSTARS), applying a criterion fixed before the run.

Does not edit `folha_de_fatos_jstars_v25.py` or `folha_de_fatos_jstars_v25.json`.
Parts A ... G (and A_B_source) are read PROGRAMMATICALLY from
`folha_de_fatos_jstars_v25.json` (checked against its pinned sha256 before
copying) and reproduced IDENTICALLY, without exception. The generic helpers
(diff_trilhas, _sem_nulos, sel, igual) are imported from v24 (the same
source v25 imports from; the .py sha256 is checked before import), not
rewritten.

Part H -- every fact is read programmatically from
`auditar_robustez_o3.json`, with id `jstarsh_*`, value, unit, artifact,
field and provenance stamp. Nothing is typed by hand: whatever is
recomputed here only serves as a guard (G4) against the recorded value.

Guards (hard failures):
  G1  unique id across the whole A-H namespace.
  G2  no None/NaN in `valor`.
  G3  every source is listed in `_fontes`; pinned sha256 hashes (v25
      .json/.py, v24 .py, R-O3, S7) are checked before reading; the sources
      of this sheet that the stamped R-O3 artifact lists in
      `_fontes`/`conferencia_sha256_antes_de_calcular` have the same sha256
      today; verification records on disk carry the hash and the verdict;
      everything is re-checked at the end.
  G4  per condition >30 m: O3 ceiling == max of the per-pair ceilings and
      pair == argmax; ceilings, limits and verdicts == tabela_vereditos and
      == O3_primario_detalhe; O3 verdict recomputed (six limits > the
      transect O3 ceiling); verdict against the two networks recomputed
      (same rule, v23 x mlp ceiling) and == guard (b) of R-O3 (source);
      pre-declared reading recomputed (scope, failures, "covers"); C0 ==
      part G of v25 (jstarsg_o3_acima_30m).
  GI  parts A-G (and A_B_source) identical to v25 by equality, both in the
      assembled object and in the JSON re-read from disk (no divergent
      path).

Usage:  python folha_de_fatos_jstars_v26.py [--saida folha_de_fatos_jstars_v26.json]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import proveniencia as PROV          # noqa: E402

ART = RAIZ / "artigo_v23_2"
PAR4 = ART / "_pareceres_2026-09-28_rodada4"

ARQS = {
    "folha_v25": RAIZ / "folha_de_fatos_jstars_v25.json",
    "folha_v25_py": RAIZ / "folha_de_fatos_jstars_v25.py",
    "folha_v24_py": RAIZ / "folha_de_fatos_jstars_v24.py",
    "ro3": RAIZ / "auditar_robustez_o3.json",
    "s7": RAIZ / "auditar_s7_arvore_lidar.json",
    "desenho": PAR4 / "desenho_S5_S7_S8.md",
    "contra_ro3": PAR4 / "contra_auditoria_RO3.md",
    "contra_folha_v25": PAR4 / "contra_auditoria_folha_v25.md",
}

# Pinned sha256 hashes (previously verified); v24 .py = the one pinned by v25
SHA_ESPERADO = {
    "folha_v25": "aa3c5e9577dd0d9b476248cdc8e0e6bb5b78cf374f4e0d5bbe4a040a56c16b99",
    "folha_v25_py": "ed8b234ca16e6e8a880f87588af807b5d8b9f405a833a9396a8a2b82e7b159e7",
    "folha_v24_py": "9d24b51bcb2ef2d71a2cd9445d5c0a8473d63fcfe45773af7e1c1576a45ddd37",
    "ro3": "a49916313c23aa54ccfb70a6461b346cf601c5adb69e971a16632e8196e8d7e1",
    "s7": "09d3030b23bfe6d908cd9bc4b9832418b4872b36aa641198a6faf6007c9abf57",
}

CL = ">30m"
LIMS = ["lim_inf_A_media_m", "lim_inf_A_mediana_m", "lim_inf_reducao_mae_m"]
CHK = {"lim_inf_A_media_m": "acima_do_teto_media",
       "lim_inf_A_mediana_m": "acima_do_teto_mediana",
       "lim_inf_reducao_mae_m": "acima_do_teto_reducao"}
# R-O3 condition -> (id jstarsh_, label in words)
CONDICOES = {
    "C0_base_glo30_sem_mascara_ETH":
        ("jstarsh_ro3_c0_base_glo30_acima_30m",
         "base: GLO-30, sem mascara, classe de dossel ETH"),
    "C1_registro_ii_b_gedtm30":
        ("jstarsh_ro3_c1_registro_gedtm30_acima_30m",
         "registro vertical GEDTM30 (ii-b)"),
    "C2_registro_iii_sem_offset":
        ("jstarsh_ro3_c2_registro_sem_offset_acima_30m",
         "registro vertical sem offset (iii)"),
    "C3_mascara_hansen_2011_2023":
        ("jstarsh_ro3_c3_mascara_2011_2023_acima_30m",
         "mascara de perda florestal Hansen 2011-2023 (celulas e ancoras nao perturbadas)"),
    "C4_classe_lidar_desacoplada_Lnb":
        ("jstarsh_ro3_c4_classe_lidar_lnb_acima_30m",
         "classe de dossel do lidar desacoplada (Lnb)"),
    "C5_celulas_concordantes_C":
        ("jstarsh_ro3_c5_celulas_concordantes_acima_30m",
         "celulas concordantes (C = ETH e Lnb na mesma classe)"),
}
NOTA_LEITURA = ("em C a media e a reducao passam e so a mediana fica abaixo; Lnb ja "
                "nao era demonstrado contra as duas redes; o teto O3 e fixado pelo par "
                "arvore x pontual; transecto e pegada nao sao confirmacoes independentes")
REGRA_O = ("veredito = 'sustentado' se os limites inferiores de A (media e mediana) e "
           "da reducao de MAE, nas unidades transecto E pegada, ficam estritamente "
           "acima do teto do transecto; senao 'nao_demonstrado' (ARV2.veredito_O, "
           "operacionalizacao D3 do R-O3)")


def log(m=""):
    print(m, flush=True)


def rot(p: Path) -> str:
    try:
        return p.relative_to(RAIZ).as_posix()
    except ValueError:
        import os
        return Path(os.path.relpath(p, RAIZ)).as_posix()


def importar_v24():
    spec = importlib.util.spec_from_file_location(
        "folha_de_fatos_jstars_v24", ARQS["folha_v24_py"])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def veredito_regra(lims_por_unidade, teto):
    ok = all(v is not None and v > teto
             for u in ("transecto", "pegada") for v in
             (lims_por_unidade[u][k] for k in LIMS))
    return "sustentado" if ok else "nao_demonstrado"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default="folha_de_fatos_jstars_v26.json")
    a = ap.parse_args()

    for k, p in ARQS.items():
        if not p.exists():
            raise FileNotFoundError(f"fonte ausente: {k} -> {p}")

    # ------------------------------------------------------------------
    # G3: sha256 of every source BEFORE reading; pinned hashes checked
    # ------------------------------------------------------------------
    sha0 = {k: PROV.sha256(p) for k, p in ARQS.items()}
    for k, esp in SHA_ESPERADO.items():
        if sha0[k] != esp:
            raise RuntimeError(f"G3: {ARQS[k].name} mudou: sha256 atual="
                               f"{sha0[k]} esperado={esp}. Nao prossigo.")

    V24 = importar_v24()                      # generic helpers (READ-ONLY)
    diff_trilhas, _sem_nulos, igual = V24.diff_trilhas, V24._sem_nulos, V24.igual

    J = {k: json.loads(p.read_text(encoding="utf-8"))
         for k, p in ARQS.items() if p.suffix == ".json"}
    TXT = {k: p.read_text(encoding="utf-8") for k, p in ARQS.items()
           if p.suffix == ".md"}
    ro3, v25 = J["ro3"], J["folha_v25"]

    # G3: the stamped R-O3 artifact lists S7 with today's same sha256; the
    # R-O3 script itself is not re-read here (identity via sha256_script)
    n_cruz = 0
    fts = {f["caminho"]: f["sha256"] for f in ro3["_fontes"]}
    for k, p in ARQS.items():
        if rot(p) in fts:
            if fts[rot(p)] != sha0[k]:
                raise RuntimeError(f"G3: {rot(p)} != sha256 no _fontes do R-O3")
            n_cruz += 1
    for nome, sh in ro3["conferencia_sha256_antes_de_calcular"].items():
        k = next((kk for kk, p in ARQS.items() if p.name == nome), None)
        if k is not None:
            if sha0[k] != sh:
                raise RuntimeError(f"G3: {nome} != conferencia_sha256 do R-O3")
            n_cruz += 1
    if n_cruz < 2:
        raise RuntimeError("G3: o S7 nao foi cruzado com o _fontes/conferencia do R-O3")
    for k, sha_alvo in (("contra_ro3", SHA_ESPERADO["ro3"]),
                        ("contra_folha_v25", SHA_ESPERADO["folha_v25"])):
        if sha_alvo[:8] not in TXT[k] or "CONFIRMADO COM RESSALVA" not in TXT[k]:
            raise RuntimeError(f"G3: {ARQS[k].name} nao traz o sha256 {sha_alvo[:8]}"
                               " e o veredito CONFIRMADO COM RESSALVA")
    if "## R-O3" not in TXT["desenho"] or "pedido do dono, 29/09" not in TXT["desenho"]:
        raise RuntimeError("protocol: design lacks the 'R-O3' section or its marker")
    for gk, gv in ro3["guardas"].items():                 # R-O3 guards passed
        pilha = [gv]
        while pilha:
            x = pilha.pop()
            if isinstance(x, dict):
                if x.get("passou") is False:
                    raise RuntimeError(f"G3: guarda do R-O3 falhou: {gk}")
                pilha.extend(x.values())

    def carimbo_de(k):
        c = {"sha256": sha0[k]}
        if k in J and isinstance(J[k].get("_proveniencia"), dict):
            pv = J[k]["_proveniencia"]
            c["script_origem"] = pv.get("script") or "AUSENTE (null no _proveniencia)"
            c["sha256_script"] = pv.get("sha256_script") or "AUSENTE"
            c["carimbado_em"] = pv.get("em") or "AUSENTE"
            c["retroativo"] = bool(pv.get("retroativo", False))
        else:
            c["script_origem"] = "nao_se_aplica (codigo/texto lido direto)"
        return c
    CARIMBO = {k: carimbo_de(k) for k in ARQS}

    # ------------------------------------------------------------------
    # Parts A-G copied from v25 (GI: identical, without exception)
    # ------------------------------------------------------------------
    COPIADAS = ["parte_A", "parte_B", "parte_A_parte_B_fonte", "parte_C",
                "parte_D", "parte_E", "parte_F", "parte_G"]
    TOPO = {"gerado_para": "gerado_para", "regra": "regra",
            "protocolo_v25": "protocolo", "protocolo_v24": "protocolo_v24",
            "protocolo_v23": "protocolo_v23", "protocolo_v22": "protocolo_v22",
            "guardas_v25": "guardas", "guardas_v24": "guardas_v24",
            "guardas_v23": "guardas_v23"}
    faltam = set(v25) - set(COPIADAS) - set(TOPO.values()) - {"_proveniencia", "_fontes"}
    if faltam:
        raise RuntimeError(f"GI: chaves de topo da v25 sem destino na v26: {faltam}")
    copia = {k: json.loads(json.dumps(v25[k])) for k in COPIADAS}

    def conferir_GI(obj, rotulo):
        for k in COPIADAS:
            d = diff_trilhas(v25[k], obj[k])
            if d:
                raise RuntimeError(f"GI ({rotulo}): {k} difere da v25 em {d[:5]}")
        for novo, velho in TOPO.items():
            if diff_trilhas(v25[velho], obj[novo]):
                raise RuntimeError(f"GI ({rotulo}): {novo} != v25.{velho}")
    conferir_GI(copia | {n: v25[v] for n, v in TOPO.items()}, "objeto montado")

    ids = set()
    for f in copia["parte_A"]["fatos"]:
        if f["id"] in ids:
            raise RuntimeError(f"G1: id repetido na v25: {f['id']}")
        ids.add(f["id"])
    for parte in ("parte_B", "parte_C", "parte_D", "parte_E", "parte_F", "parte_G"):
        for gf in copia[parte]["fatos"].values():
            for f in gf:
                if f["id"] in ids:
                    raise RuntimeError(f"G1: id repetido na v25: {f['id']}")
                ids.add(f["id"])
    n_ids_v25 = len(ids)
    if n_ids_v25 != v25["guardas"]["G1_ids_unicos"]["n_ids_total"]:
        raise RuntimeError("G1: contagem de ids da v25 != guarda gravada")

    grupos: dict[str, list] = {}

    def fato(grupo, id_, valor, unidade, chaves, campo, nota=None,
             operacao=None, extra=None):
        if not id_.startswith("jstarsh_"):
            raise RuntimeError(f"id fora do prefixo jstarsh_: {id_}")
        if id_ in ids:                                           # G1
            raise RuntimeError(f"G1: id duplicado: {id_}")
        ids.add(id_)
        _sem_nulos(valor, id_)                                   # G2
        if isinstance(chaves, str):
            chaves = [chaves]
        linha = {"id": id_, "valor": valor, "unidade": unidade,
                 "artefato": " + ".join(rot(ARQS[c]) for c in chaves),
                 "campo": campo,
                 "carimbo": {rot(ARQS[c]): CARIMBO[c] for c in chaves}}
        if operacao:
            linha["derivado"] = True
            linha["operacao"] = operacao
        if extra:
            linha.update(extra)
        if nota:
            linha["nota"] = nota
        grupos.setdefault(grupo, []).append(linha)
        return linha

    # ==================================================================
    # H1 -- per condition, above 30 m
    # ==================================================================
    if list(ro3["condicoes"]) != list(CONDICOES):
        raise RuntimeError(f"R-O3: condicoes inesperadas {list(ro3['condicoes'])}")
    tab = {(r["condicao"], r["classe"]): r for r in ro3["tabela_vereditos"]}
    gb = ro3["guardas"]["b_teto_duas_redes_e_veredito_1e-6"]
    lpd = ro3["leitura_pre_declarada"]
    vered_O3, vered_2r = {}, {}
    for cond, (id_, rotulo) in CONDICOES.items():
        pc = ro3["condicoes"][cond]["por_classe"][CL]
        tr = tab[(cond, CL)]
        tpp = pc["tetos_por_par_transecto_m"]
        teto3 = pc["teto_O3_transecto_m"]
        par = pc["par_que_fixa_o_teto_transecto"]
        # G4: ceiling and pair recounted from the per-pair ceilings; == table and == detail
        igual(teto3, max(tpp.values()), 0.0, f"{cond} teto O3 == max tetos por par")
        if par != max(tpp, key=tpp.get):
            raise RuntimeError(f"G4: {cond} par que fixa != argmax")
        igual(tr["teto_O3_transecto_m"], teto3, 0.0, f"{cond} teto O3 tabela")
        if tr["par_que_fixa_transecto"] != par:
            raise RuntimeError(f"G4: {cond} par na tabela != por_classe")
        det = pc["O3_primario_detalhe"]
        igual(det["teto_abs_contrastes_lidar_m"], teto3, 0.0, f"{cond} teto detalhe")
        li = pc["limites_inferiores"]
        for u in ("transecto", "pegada"):
            for kl in LIMS:
                igual(li[u][kl], tr[f"{kl}_{u}"], 0.0, f"{cond} {kl} {u} tabela")
                igual(li[u][kl], det["checagens_por_unidade"][u][kl], 0.0,
                      f"{cond} {kl} {u} detalhe")
                if det["checagens_por_unidade"][u][CHK[kl]] != (li[u][kl] > teto3):
                    raise RuntimeError(f"G4: {cond} {CHK[kl]} {u} != limite > teto")
        igual(pc["menor_limite_inferior_m"],
              min(li[u][k] for u in ("transecto", "pegada") for k in LIMS), 0.0,
              f"{cond} menor limite")
        igual(pc["margem_menor_limite_menos_teto_O3_m"],
              pc["menor_limite_inferior_m"] - teto3, 1e-15, f"{cond} margem")
        v3 = pc["veredito_O3_primario"]
        if v3 != veredito_regra(li, teto3) or v3 != det["veredito"] \
                or v3 != tr["veredito_O3_primario"] or v3 != lpd["O3_maior30_por_condicao"][cond]:
            raise RuntimeError(f"G4: {cond} veredito O3 inconsistente")
        # against the two networks: rule recomputed with the v23 x mlp ceiling; == source
        t2 = pc["teto_duas_redes_v23_x_mlp_m"]["transecto"]
        igual(t2, tpp["v23 vs mlp"], 0.0, f"{cond} teto duas redes == par v23 x mlp")
        igual(t2, gb[cond][CL]["transecto"]["origem_m"], 0.0, f"{cond} teto duas redes origem")
        v2 = pc["veredito_O_duas_redes"]
        if v2 != veredito_regra(li, t2) or v2 != gb[cond][CL]["veredito_O_duas_redes"]["origem"] \
                or v2 != tr["veredito_O_duas_redes_hoje"]:
            raise RuntimeError(f"G4: {cond} veredito contra as duas redes inconsistente")
        vered_O3[cond], vered_2r[cond] = v3, v2

        abaixo = [f"{k} ({u})" for u in ("transecto", "pegada") for k in LIMS
                  if not li[u][k] > teto3]
        fato("ro3_por_condicao_acima_30m", id_,
             {"condicao": cond, "descricao": rotulo, "classe": CL,
              "teto_O3_transecto_m": teto3,
              "par_que_fixa_o_teto_O3": par,
              "tetos_por_par_transecto_m": tpp,
              "limites_inferiores_transecto": {k: li["transecto"][k] for k in LIMS},
              "limites_inferiores_pegada_leitura_secundaria": {k: li["pegada"][k] for k in LIMS},
              "margem_menor_limite_menos_teto_O3_m": pc["margem_menor_limite_menos_teto_O3_m"],
              "veredito_O3": v3,
              "teto_duas_redes_v23_x_mlp_transecto_m": t2,
              "veredito_contra_as_duas_redes": v2},
             "m", "ro3",
             f"condicoes['{cond}'].por_classe['{CL}'].{{teto_O3_transecto_m,"
             "par_que_fixa_o_teto_transecto,tetos_por_par_transecto_m,"
             "limites_inferiores.{transecto,pegada}.{lim_inf_A_media_m,lim_inf_A_mediana_m,"
             "lim_inf_reducao_mae_m},margem_menor_limite_menos_teto_O3_m,veredito_O3_primario,"
             "teto_duas_redes_v23_x_mlp_m.transecto,veredito_O_duas_redes}",
             nota=(REGRA_O + ". Limites que nao passam o teto O3: "
                   + (", ".join(abaixo) if abaixo else "nenhum")
                   + ". Bootstrap B = 10.000, semente 42; teto = max(|lo|,|hi|) do IC95 da "
                   "mediana de dMAE por unidade. G4: teto == max dos tetos por par, par == "
                   "argmax; limites e vereditos == tabela_vereditos e O3_primario_detalhe; "
                   "os dois vereditos recomputados pela regra; o contra as duas redes == "
                   "artefato de origem (guarda b do R-O3)."))

    # G4: C0 == O3 from part G of v25 (S7), same baseline
    g_o3 = {f["id"]: f for f in copia["parte_G"]["fatos"]["s7_O3"]}["jstarsg_o3_acima_30m"]
    pu = g_o3["valor"]["por_unidade"]["transecto"]
    pc0 = ro3["condicoes"]["C0_base_glo30_sem_mascara_ETH"]["por_classe"][CL]
    igual(pc0["teto_O3_transecto_m"], pu["teto_O3_m"], 1e-9, "C0 teto == v25 jstarsg_o3")
    if pc0["par_que_fixa_o_teto_transecto"] != pu["par_que_fixa_o_teto"]:
        raise RuntimeError("G4: C0 par != v25 jstarsg_o3")
    for kl in LIMS:
        igual(pc0["limites_inferiores"]["transecto"][kl], pu[kl], 1e-9, f"C0 {kl} == v25")

    # ==================================================================
    # H2 -- pre-declared reading
    # ==================================================================
    escopo = [c for c in CONDICOES if c != "C0_base_glo30_sem_mascara_ETH"
              and vered_2r[c] == "sustentado"]
    fora = [c for c in CONDICOES if c != "C0_base_glo30_sem_mascara_ETH"
            and vered_2r[c] != "sustentado"]
    falham = [c for c in escopo if vered_O3[c] != "sustentado"]
    cobre = "sim" if escopo and not falham else "nao"
    if escopo != lpd["condicoes_no_escopo_O_maior30_demonstrado_hoje"] \
            or fora != lpd["condicoes_fora_do_escopo_O_maior30_nao_demonstrado_hoje"] \
            or falham != lpd["condicoes_em_que_O3_falha"] \
            or cobre != lpd["a_robustez_cobre_as_tres_familias"]:
        raise RuntimeError("G4: leitura pre-declarada recomputada != gravada")
    if cobre != "nao" or falham != ["C5_celulas_concordantes_C"]:
        raise RuntimeError("leitura: o pedido espera 'nao' com falha em C; artefato diz outra coisa")
    fato("ro3_leitura_pre_declarada", "jstarsh_ro3_leitura_tres_familias",
         {"regra": lpd["regra_literal"],
          "a_robustez_cobre_as_tres_familias": lpd["a_robustez_cobre_as_tres_familias"],
          "condicao_que_falha": "C (celulas concordantes)",
          "condicoes_em_que_O3_falha": lpd["condicoes_em_que_O3_falha"],
          "condicoes_no_escopo": lpd["condicoes_no_escopo_O_maior30_demonstrado_hoje"],
          "condicoes_fora_do_escopo": lpd["condicoes_fora_do_escopo_O_maior30_nao_demonstrado_hoje"],
          "O3_acima_30m_por_condicao": lpd["O3_maior30_por_condicao"]},
         "rotulo", "ro3",
         "leitura_pre_declarada.{regra_literal,a_robustez_cobre_as_tres_familias,"
         "condicoes_em_que_O3_falha,condicoes_no_escopo_O_maior30_demonstrado_hoje,"
         "condicoes_fora_do_escopo_O_maior30_nao_demonstrado_hoje,O3_maior30_por_condicao}",
         nota=NOTA_LEITURA + ". G4: escopo (C1-C5 com veredito contra as duas redes "
              "'sustentado'), falhas e 'cobre' recomputados dos vereditos por condicao.",
         extra={"nota_rotulo": "'condicao_que_falha' e o nome em palavras de "
                               "C5_celulas_concordantes_C, a unica entrada de "
                               "condicoes_em_que_O3_falha (conferido por codigo)"})

    qualidade = {
        "fontes": {rot(ARQS[k]): {kk: vv for kk, vv in CARIMBO[k].items()}
                   for k in ("ro3", "s7")},
        "notas": [
            ("auditar_robustez_o3.json: the independent-check field is 'PENDENTE' in "
             "the JSON; the independent re-check was done later (CONFIRMED WITH "
             "CAVEAT, sha256 a4991631... verified by code)"),
            ("_proveniencia.arvore_suja = true (commit d35f050); script identity only "
             "via sha256_script"),
            ("Under O3, Lnb >30 m also does not pass (out of scope because it was "
             "already not demonstrated against both networks); against both networks "
             "Lnb failed only in the footprint unit"),
            "C5 >30 m ja era demonstrado contra as duas redes por margem estreita "
            "(limite inferior da mediana de A menos teto v23 x mlp; L2)",
            "transecto e pegada sao praticamente a mesma amostra acima de 30 m (L3)",
            "20-30 m e descritivo no R-O3 (leitura_pre_declarada.O3_20_30_por_condicao_"
            "descritivo) e nao entra nesta parte_H (L4)",
            "em todas as condicoes >30 m o par que fixa o teto O3 e tree x mlp (L8)",
        ],
    }

    # ==================================================================
    # G3 (final): no source changed during the run
    # ==================================================================
    for k, p in ARQS.items():
        if PROV.sha256(p) != sha0[k]:
            raise RuntimeError(f"G3: {p.name} mudou durante a corrida")

    n_fatos_h = sum(len(v) for v in grupos.values())
    saida = {
        "gerado_para": v25["gerado_para"],
        "regra": v25["regra"],
        "protocolo": (("Design of check R-O3; parte_H read from auditar_robustez_o3.json (sha256 "
                       "")
                      + SHA_ESPERADO["ro3"][:8] + ("..., independent check of R-O3); parts A-G = "
                                                   "folha_de_fatos_jstars_v25.json (sha256 ")
                      + SHA_ESPERADO["folha_v25"][:8] + "...), identicas"),
        "protocolo_v25": v25["protocolo"],
        "protocolo_v24": v25["protocolo_v24"],
        "protocolo_v23": v25["protocolo_v23"],
        "protocolo_v22": v25["protocolo_v22"],
        **copia,
        "parte_H": {
            "n_fatos": n_fatos_h,
            "grupos": {g: len(v) for g, v in grupos.items()},
            "qualidade_das_fontes": qualidade,
            "fatos": grupos,
        },
        "guardas": {
            "GI_partes_A_G_identicas_a_v25": {"trilhas_divergentes": [], "passou": True},
            "G1_ids_unicos": {"n_ids_v25": n_ids_v25, "n_ids_total": len(ids)},
            "G2_sem_nulos": True,
            "G3_sha256_conferidos": {"fixados": sorted(SHA_ESPERADO),
                                     "cruzamentos_com_o_R-O3": n_cruz, "passou": True},
            "G4_derivados_conferidos": True,
        },
        "guardas_v25": v25["guardas"],
        "guardas_v24": v25["guardas_v24"],
        "guardas_v23": v25["guardas_v23"],
    }
    saida_p = Path(a.saida)
    if not saida_p.is_absolute():
        saida_p = RAIZ / saida_p
    PROV.gravar(saida_p, saida, fontes_lidas=list(ARQS.values()), script=__file__)

    # GI (2): re-read from disk and compare with v25 re-read from disk
    relido = json.loads(saida_p.read_text(encoding="utf-8"))
    v25_disco = json.loads(ARQS["folha_v25"].read_text(encoding="utf-8"))
    if v25_disco != v25:
        raise RuntimeError("GI: v25 no disco mudou durante a corrida")
    conferir_GI(relido, "JSON relido do disco")
    if relido["parte_H"] != json.loads(json.dumps(saida["parte_H"])):
        raise RuntimeError("releitura: parte_H gravada != montada")
    log(f"GI ok: {', '.join(COPIADAS)} == v25 (0 trilhas; apos releitura)")
    log(f"G3: {len(SHA_ESPERADO)} sha256 fixados; {n_cruz} cruzamentos com o R-O3")
    log(f"ids: {n_ids_v25} da v25 + {len(ids) - n_ids_v25} novos = {len(ids)}")
    log(f"parte_H: {n_fatos_h} fatos em {len(grupos)} grupos -> {saida_p}")
    for g, v in grupos.items():
        log(f"  {g}: {', '.join(f['id'] for f in v)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
