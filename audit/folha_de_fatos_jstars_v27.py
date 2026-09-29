# -*- coding: utf-8 -*-
"""Fact sheet v27: append part I to the V23 JSTARS article fact sheet.
Part I holds the reanalysis of the R-F label filter, code definitions used in
the text, pre-specified contrasts, graph vs GEDTM30 against lidar, panel (b) of
the three-family figure, and the ANADEM MAE on the profile strip.

Parts A-H (and parte_A_parte_B_fonte) are read from folha_de_fatos_jstars_v26.json
(sha256 checked before reading) and copied unchanged; no existing file is edited.
Generic helpers (diff_trilhas, _sem_nulos, igual) come from the v24 script, whose
sha256 is checked before import.

Part I ids use the prefix `jstarsi_`. Each fact is {id, valor, unidade, artefato,
campo, sha256 stamp}; code definitions are read from the source (ast/regex) with
file:line, never typed by hand; values recomputed here serve only as guards.

Guards (raise RuntimeError on failure):
  G1  unique ids across parts A-I.
  G2  no None/NaN in `valor`.
  G3  every source in `_fontes` has a sha256; pinned hashes are checked before
      reading and again at the end; the three contrast artifacts cite the queue
      script with its current sha256; public-package copies match the originals.
  G4  derived values recomputed in code (R-F verdicts and population floor;
      non-passing minus passing difference; survivor ratio over 441; per-transect
      sums; x4 epoch caps == 4 x x1 caps; per-reserve sign counts; pointers to
      existing facts checked against the artifact).
  GI  parts A-H identical to v26, in the built object and in the re-read JSON.

Usage:  python folha_de_fatos_jstars_v27.py [--saida folha_de_fatos_jstars_v27.json]
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import proveniencia as PROV          # noqa: E402

ART = RAIZ / "artigo_v23_2"
PAR4 = ART / "_pareceres_2026-09-28_rodada4"
PAR5 = ART / "_pareceres_2026-09-29_r5h"
PKG = ART / "pacote_repositorio_jstars" / "audit"
QL = RAIZ / "_arquivo_2026-08-09" / "auditorias_pontuais"
LATD = Path("/trabalho/GNN_TOPO/SATELITES/laterais_nativa_diretas")

ARQS = {
    "folha_v26": RAIZ / "folha_de_fatos_jstars_v26.json",
    "folha_v24_py": RAIZ / "folha_de_fatos_jstars_v24.py",
    "desenho": PAR4 / "desenho_S5_S7_S8.md",
    "contra_rf": PAR5 / "contra_auditoria_RF.md",
    "contra_folha_v26": PAR4 / "contra_auditoria_folha_v26.md",
    # I1
    "rf_v1": RAIZ / "auditar_rf_filtro_gedi.json",
    "rf_v2": RAIZ / "auditar_rf_filtro_gedi_v2.json",
    # I2
    "juiz_py": RAIZ / "juiz_lidar_v4.py",
    "juiz_json": RAIZ / "juiz_grade_nativa_diretas.json",
    "laterais_prov": LATD / "_proveniencia.json",
    "b2_py": RAIZ / "b2_transferencia.py",
    "res_x1": RAIZ / "results" / "ofat_suavfina_b4s035.json",
    "res_x4_f1": RAIZ / "results" / "f1_rampa_gatv2_2_semobs.json",
    "res_x4_d1": RAIZ / "results" / "d1_orcamento_gatv2_2_semobs.json",
    "adm_py": RAIZ / "auditar_admissibilidade.py",
    "adm_json": RAIZ / "auditar_admissibilidade.json",
    "grade_py": RAIZ / "grade_v23.py",
    "s7_py": RAIZ / "auditar_s7_arvore_lidar.py",
    "regua_v2_py": RAIZ / "auditar_rotulo_regua_v2.py",
    "tabela3_py": RAIZ / "auditar_tabela3.py",
    "rotulos_py": RAIZ / "rotulos_lidar_v23.py",
    "ql4_json": QL / "qualidade_lidar_4biomas.json",
    "ql4_py": QL / "qualidade_lidar_4biomas.py",
    "qlz_json": QL / "qualidade_lidar_do_zero.json",
    "qlz_py": QL / "qualidade_lidar_do_zero.py",
    # I3
    "fila": RAIZ / "fila_2026-08-16_d8_f2_f1.sh",
    "d8": RAIZ / "auditar_d8_triangulo.json",
    "f2": RAIZ / "auditar_f2_celula.json",
    "f1": RAIZ / "auditar_f1_rampa.json",
    "pkg_d8": PKG / "auditar_d8_triangulo.json",
    "pkg_f2": PKG / "auditar_f2_celula.json",
    "pkg_f1": PKG / "auditar_f1_rampa.json",
    "tabela3": RAIZ / "auditar_tabela3.json",
    # I3/I4
    "nmad_anadem": RAIZ / "auditar_nmad_pareado_nativa_anadem.json",
    # I5
    "fig3_num": ART / "_propostas_2026-09-28" / "fig_r5" / "fig3_numeros.json",
    "fig3_py": ART / "fig" / "gerar_fig3.py",
    # I6
    "perfil": RAIZ / "auditar_perfil_anadem_faixa.json",
    "perfil_py": RAIZ / "auditar_perfil_anadem_faixa.py",
}

SHA_ESPERADO = {
    "folha_v26": "7f8701e91988198746172f4f97d900858621a843826be06880bd0c22f36b8842",
    "folha_v24_py": "9d24b51bcb2ef2d71a2cd9445d5c0a8473d63fcfe45773af7e1c1576a45ddd37",
    "rf_v1": "1334a3e80966e416bbe6822248a731afd065e84c381db425f9b0c5ac61827306",
    "rf_v2": "48b1698c371d9ed8c1700f842de9cf37b5e63ac08b092636cc1f5643bceaca91",
    "fila": "444263624df84c14d3748b3b73806c50a9f568cfde1266b5054759863c60c448",
    "juiz_py": "7768ba350a38cc94e9ab37a48cbac1b7b8917a902d3eb015ad430919c8b033c2",
    "perfil": "4325c5e3b63a339b25293b745580650fc458a41c28ea545e67816d16d069f405",
    "perfil_py": "ef6cd0fe52bfd3c232af33f73cfce3e982860f08531b56f442de01ce3e73a0b9",
    "fig3_num": "d61302f37bbbe3e970032154185a06764d4a1f531de25a90330fe4a654a25e32",
}

CLS = {">30m": "acima_30m", "20-30m": "20_30m"}
BRACOS = ["F0", "F1", "F2"]
LIMS = ["lim_inf_A_media_m", "lim_inf_A_mediana_m", "lim_inf_reducao_mae_m"]


def log(m=""):
    print(m, flush=True)


def rot(p: Path) -> str:
    try:
        return p.relative_to(RAIZ).as_posix()
    except ValueError:
        return p.as_posix()


def importar_v24():
    spec = importlib.util.spec_from_file_location(
        "folha_de_fatos_jstars_v24", ARQS["folha_v24_py"])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default="folha_de_fatos_jstars_v27.json")
    a = ap.parse_args()

    for k, p in ARQS.items():
        if not p.exists():
            raise FileNotFoundError(f"fonte ausente: {k} -> {p}")

    # G3: sha256 of every source before reading; pinned hashes must match.
    sha0 = {k: PROV.sha256(p) for k, p in ARQS.items()}
    for k, esp in SHA_ESPERADO.items():
        if sha0[k] != esp:
            raise RuntimeError(f"G3: {ARQS[k].name} mudou: sha256 atual={sha0[k]} "
                               f"esperado={esp}. Nao prossigo.")

    V24 = importar_v24()
    diff_trilhas, _sem_nulos, igual = V24.diff_trilhas, V24._sem_nulos, V24.igual

    J = {k: json.loads(p.read_text(encoding="utf-8"))
         for k, p in ARQS.items() if p.suffix == ".json"}
    TXT = {k: p.read_text(encoding="utf-8") for k, p in ARQS.items()
           if p.suffix in (".md", ".py", ".sh")}
    v26 = J["folha_v26"]

    for k, alvo, vered in (("contra_folha_v26", SHA_ESPERADO["folha_v26"], "CONFIRMADO COM RESSALVA"),
                           ("contra_rf", SHA_ESPERADO["rf_v1"], "CONFIRMADO COM RESSALVA")):
        if alvo[:8] not in TXT[k] or vered not in TXT[k]:
            raise RuntimeError(f"G3: {ARQS[k].name} sem sha256 {alvo[:8]} e veredito {vered}")
    if "## R-F" not in TXT["desenho"] or "folha de fatos v27" not in TXT["desenho"]:
        raise RuntimeError("protocolo: desenho sem a secao R-F que pede a folha v27")

    def carimbo_de(k):
        c = {"sha256": sha0[k]}
        if k in J and isinstance(J[k], dict) and isinstance(J[k].get("_proveniencia"), dict):
            pv = J[k]["_proveniencia"]
            c["script_origem"] = pv.get("script") or "AUSENTE (null no _proveniencia)"
            c["sha256_script"] = pv.get("sha256_script") or "AUSENTE"
            c["carimbado_em"] = pv.get("em") or "AUSENTE"
            c["retroativo"] = bool(pv.get("retroativo", False))
        elif k in J:
            c["script_origem"] = "AUSENTE (JSON sem _proveniencia)"
        else:
            c["script_origem"] = "nao_se_aplica (codigo/texto lido direto)"
        return c
    CARIMBO = {k: carimbo_de(k) for k in ARQS}

    # ---------------- source-code reading (ast/regex) ----------------
    def linha(k, padrao, unica=True):
        hits = [(i + 1, l.strip()) for i, l in enumerate(TXT[k].splitlines())
                if re.search(padrao, l)]
        if not hits or (unica and len(hits) != 1):
            raise RuntimeError(f"fonte {ARQS[k].name}: padrao {padrao!r} com {len(hits)} ocorrencias")
        return hits[0] if unica else hits

    def onde(k, padrao):
        n, txt = linha(k, padrao)
        return {"arquivo_linha": f"{rot(ARQS[k])}:{n}", "texto": txt}

    def const(k, nome):
        arv = ast.parse(TXT[k])
        for no in arv.body:
            if isinstance(no, ast.Assign):
                for t in no.targets:
                    if isinstance(t, ast.Name) and t.id == nome:
                        return ast.literal_eval(no.value), no.lineno
                    if isinstance(t, ast.Tuple) and isinstance(no.value, ast.Tuple):
                        for e, v in zip(t.elts, no.value.elts):
                            if isinstance(e, ast.Name) and e.id == nome:
                                return ast.literal_eval(v), no.lineno
        raise RuntimeError(f"constante {nome} nao achada em {ARQS[k].name}")

    def funcao(k, nome):
        arv = ast.parse(TXT[k])
        for no in ast.walk(arv):
            if isinstance(no, ast.FunctionDef) and no.name == nome:
                return ast.get_source_segment(TXT[k], no), no.lineno, no.end_lineno
        raise RuntimeError(f"funcao {nome} nao achada em {ARQS[k].name}")

    # ------------------------------------------------------------------
    # Parts A-H copied from v26 (guard GI)
    # ------------------------------------------------------------------
    COPIADAS = ["parte_A", "parte_B", "parte_A_parte_B_fonte", "parte_C",
                "parte_D", "parte_E", "parte_F", "parte_G", "parte_H"]
    TOPO = {"gerado_para": "gerado_para", "regra": "regra",
            "protocolo_v26": "protocolo", "protocolo_v25": "protocolo_v25",
            "protocolo_v24": "protocolo_v24", "protocolo_v23": "protocolo_v23",
            "protocolo_v22": "protocolo_v22",
            "guardas_v26": "guardas", "guardas_v25": "guardas_v25",
            "guardas_v24": "guardas_v24", "guardas_v23": "guardas_v23"}
    faltam = set(v26) - set(COPIADAS) - set(TOPO.values()) - {"_proveniencia", "_fontes"}
    if faltam:
        raise RuntimeError(f"GI: chaves de topo da v26 sem destino na v27: {faltam}")
    copia = {k: json.loads(json.dumps(v26[k])) for k in COPIADAS}

    def conferir_GI(obj, rotulo):
        for k in COPIADAS:
            d = diff_trilhas(v26[k], obj[k])
            if d:
                raise RuntimeError(f"GI ({rotulo}): {k} difere da v26 em {d[:5]}")
        for novo, velho in TOPO.items():
            if diff_trilhas(v26[velho], obj[novo]):
                raise RuntimeError(f"GI ({rotulo}): {novo} != v26.{velho}")
    conferir_GI(copia | {n: v26[v] for n, v in TOPO.items()}, "objeto montado")

    ids, por_id = set(), {}
    for f in copia["parte_A"]["fatos"]:
        if f["id"] in ids:
            raise RuntimeError(f"G1: id repetido na v26: {f['id']}")
        ids.add(f["id"]); por_id[f["id"]] = f
    for parte in ("parte_B", "parte_C", "parte_D", "parte_E", "parte_F", "parte_G", "parte_H"):
        for gf in copia[parte]["fatos"].values():
            for f in gf:
                if f["id"] in ids:
                    raise RuntimeError(f"G1: id repetido na v26: {f['id']}")
                ids.add(f["id"]); por_id[f["id"]] = f
    n_ids_v26 = len(ids)
    if n_ids_v26 != v26["guardas"]["G1_ids_unicos"]["n_ids_total"]:
        raise RuntimeError(f"G1: contagem de ids da v26 ({n_ids_v26}) != guarda gravada")

    grupos: dict[str, list] = {}

    def fato(grupo, id_, valor, unidade, chaves, campo, nota=None,
             operacao=None, extra=None):
        if not id_.startswith("jstarsi_"):
            raise RuntimeError(f"id fora do prefixo jstarsi_: {id_}")
        if id_ in ids:
            raise RuntimeError(f"G1: id duplicado: {id_}")
        ids.add(id_)
        _sem_nulos(valor, id_)
        if isinstance(chaves, str):
            chaves = [chaves]
        linha_ = {"id": id_, "valor": valor, "unidade": unidade,
                  "artefato": " + ".join(rot(ARQS[c]) for c in chaves),
                  "campo": campo,
                  "carimbo": {rot(ARQS[c]): CARIMBO[c] for c in chaves}}
        if operacao:
            linha_["derivado"] = True
            linha_["operacao"] = operacao
        if extra:
            linha_.update(extra)
        if nota:
            linha_["nota"] = nota
        grupos.setdefault(grupo, []).append(linha_)
        return linha_

    def ponteiro(id_existente, artefato_valor, chaves_cmp):
        """Check an existing fact (parts A-H) against the artifact; return the pointer."""
        if id_existente not in por_id:
            raise RuntimeError(f"G4: ponteiro para id inexistente {id_existente}")
        fv = por_id[id_existente]["valor"]
        for c in chaves_cmp:
            if diff_trilhas(fv[c], artefato_valor[c]):
                raise RuntimeError(f"G4: {id_existente}.{c} != artefato")
        return {"id_existente": id_existente, "conferido_contra_o_artefato": chaves_cmp}

    # ==================================================================
    # I1 -- label-filter reanalysis (R-F)
    # ==================================================================
    r1, r2 = J["rf_v1"], J["rf_v2"]
    # v2 is a superset of v1 on the R-F fields
    for c in ("bracos_definicao", "criterio", "por_braco", "remocao_de_celulas",
              "contagens_disparos", "vereditos_maior30m"):
        if diff_trilhas(r1[c], r2[c]):
            raise RuntimeError(f"G4: rf_v2.{c} != rf_v1.{c}")
    if not r2["guarda_G4_reproducao_v1"]["passou"] or \
            r2["guarda_G4_reproducao_v1"]["sha256_json_v1"] != SHA_ESPERADO["rf_v1"]:
        raise RuntimeError("G3: rf_v2 nao reproduz o rf_v1 fixado")
    for g in ("G1", "G2"):
        if not r1["guardas"][g]["passou"]:
            raise RuntimeError(f"R-F guarda {g} falhou")
    for cl in r1["guardas"]["G3"].values():
        if not cl["passou"]:
            raise RuntimeError("R-F guarda G3 falhou")
    piso = r1["criterio"]["piso_maior30m"]
    n_piso, n_protocolo_linha = linha("desenho", r"Piso de popula")
    fato("i1_filtro_rotulo", "jstarsi_rf_definicao_bracos_e_piso",
         {"bracos": r1["bracos_definicao"],
          "piso_acima_30m": {"celulas_min": piso["celulas_min"],
                             "transectos_min": piso["transectos_min"],
                             "texto_do_protocolo": n_protocolo_linha,
                             "protocolo_linha": f"{rot(ARQS['desenho'])}:{n_piso}"},
          "classe_20_30m": r1["criterio"]["classe_20_30m"],
          "regra": r1["criterio"]["regra"],
          "estimandos": r1["criterio"]["estimandos"],
          "bootstrap": r1["criterio"]["bootstrap"],
          "rotulo_por_celula": onde("desenho", r"R.tulo por c.lula")},
         "rotulo", ["rf_v1", "desenho"],
         "bracos_definicao; criterio.{piso_maior30m,classe_20_30m,regra,estimandos,bootstrap}",
         nota="definicoes literais do artefato; piso e regra do rotulo por celula citados do protocolo por regex")

    teto_cls = {}
    for cl, suf in CLS.items():
        for b in BRACOS:
            pb = r1["por_braco"][b][cl]
            li = pb["limites_inferiores"]
            teto = pb["teto_m"]
            teto_cls[cl] = teto
            if pb["n_transectos"] != len(pb["A"]["medias_por_transecto_m"]):
                raise RuntimeError(f"G4: {b}/{cl} n_transectos != len(medias_por_transecto)")
            igual(li["lim_inf_A_media_m"], pb["A"]["ic95_media_pond"][0], 0.0, f"{b}/{cl} lim media")
            igual(li["lim_inf_A_mediana_m"], pb["A"]["ic95_mediana"][0], 0.0, f"{b}/{cl} lim mediana")
            igual(li["lim_inf_reducao_mae_m"], pb["reducao_mae_equivalente"]["ic95"][0], 0.0,
                  f"{b}/{cl} lim reducao")
            acima = {k: li[k] > teto for k in LIMS}
            if acima != pb["acima_do_teto"]:
                raise RuntimeError(f"G4: {b}/{cl} acima_do_teto recomputado != gravado")
            if cl == ">30m":
                atinge = (pb["n_celulas"] >= piso["celulas_min"]
                          and pb["n_transectos"] >= piso["transectos_min"])
                if atinge != pb["atinge_piso"]:
                    raise RuntimeError(f"G4: {b}/{cl} atinge_piso recomputado != gravado")
                vr = ("nao avaliavel" if not atinge else
                      ("sustentado" if all(acima.values()) else "nao sustentado"))
                if vr != r1["vereditos_maior30m"][b]:
                    raise RuntimeError(f"G4: {b} veredito >30m != vereditos_maior30m")
                atinge_v = atinge
            else:
                if pb["atinge_piso"] is not None:
                    raise RuntimeError(f"{b}/{cl}: atinge_piso deveria ser null (piso nao definido)")
                vr = "sustentado" if all(acima.values()) else "nao sustentado"
                atinge_v = "nao_definido_pelo_protocolo (null no artefato)"
            if vr != pb["veredito"]:
                raise RuntimeError(f"G4: {b}/{cl} veredito recomputado {vr} != {pb['veredito']}")
            fato("i1_filtro_rotulo", f"jstarsi_rf_{b.lower()}_{suf}",
                 {"braco": b, "classe": cl, "n_celulas": pb["n_celulas"],
                  "n_transectos": pb["n_transectos"], "atinge_piso": atinge_v,
                  "A_media_pond_celula_m": pb["A"]["media_pond_celula_m"],
                  "A_ic95_media_pond_m": pb["A"]["ic95_media_pond"],
                  "A_mediana_m": pb["A"]["mediana_m"],
                  "A_ic95_mediana_m": pb["A"]["ic95_mediana"],
                  "A_medias_por_transecto_m": pb["A"]["medias_por_transecto_m"],
                  "reducao_mae_equivalente_m": pb["reducao_mae_equivalente"]["valor_m"],
                  "reducao_mae_ic95_m": pb["reducao_mae_equivalente"]["ic95"],
                  "limites_inferiores_m": li, "teto_O3_m": teto,
                  "limites_acima_do_teto": acima, "veredito": pb["veredito"],
                  "deslocamento_mediana_Fk_menos_F0": pb["deslocamento_mediana_Fk_menos_F0"]},
                 "m", "rf_v1", f"por_braco.{b}.'{cl}'",
                 nota=("leitura secundaria (20-30 m contra 0,500 m; piso nao definido)" if cl == "20-30m"
                       else "classe principal (>30 m contra 1,442 m); piso 110 celulas em 6 transectos")
                 + ". G4: limites == IC95 inferiores; acima_do_teto, atinge_piso e veredito recomputados.")

    pvn_val, dif_val, raz_val = {}, {}, {}
    n_f0 = r1["por_braco"]["F0"][">30m"]["n_celulas"]
    for b in ("F1", "F2"):
        pv = r2["passa_vs_nao_passa"][b][">30m"]
        pb = r1["por_braco"][b][">30m"]
        if pv["passam"]["n"] != pb["n_celulas"]:
            raise RuntimeError(f"G4: {b} passam.n != por_braco n")
        if sorted(pv["transectos_comuns"]) != sorted(pb["A"]["medias_por_transecto_m"]):
            raise RuntimeError(f"G4: {b} transectos comuns != transectos do braco")
        for g in ("passam", "nao_passam"):
            igual(sum(pv["por_transecto"][t][g]["n"] for t in pv["transectos_comuns"]),
                  pv[g]["n"], 0, f"{b} soma n {g} por transecto")
        dif_t = {t: pv["por_transecto"][t]["nao_passam"]["media_pond_celula_m"]
                 - pv["por_transecto"][t]["passam"]["media_pond_celula_m"]
                 for t in pv["transectos_comuns"]}
        dif = pv["nao_passam"]["media_pond_celula_m"] - pv["passam"]["media_pond_celula_m"]
        pvn_val[b] = {
            "transectos_comuns": pv["transectos_comuns"],
            "transectos_sem_celula_no_braco": pv["transectos_sem_celula_no_braco"],
            "passam": {"n": pv["passam"]["n"], "A_media_pond_celula_m": pv["passam"]["media_pond_celula_m"],
                       "A_mediana_m": pv["passam"]["mediana_m"]},
            "nao_passam": {"n": pv["nao_passam"]["n"],
                           "A_media_pond_celula_m": pv["nao_passam"]["media_pond_celula_m"],
                           "A_mediana_m": pv["nao_passam"]["mediana_m"]},
            "por_transecto": {t: {"passam_n": pv["por_transecto"][t]["passam"]["n"],
                                  "passam_A_media_m": pv["por_transecto"][t]["passam"]["media_pond_celula_m"],
                                  "nao_passam_n": pv["por_transecto"][t]["nao_passam"]["n"],
                                  "nao_passam_A_media_m": pv["por_transecto"][t]["nao_passam"]["media_pond_celula_m"]}
                              for t in pv["transectos_comuns"]}}
        dif_val[b] = {"total_m": dif, "por_transecto_m": dif_t,
                      "transectos_com_nao_passam_acima": sum(1 for x in dif_t.values() if x > 0)}
        raz_val[b] = {"n_sobreviventes": pb["n_celulas"], "n_F0": n_f0,
                      "razao": pb["n_celulas"] / n_f0}
        fato("i1_filtro_rotulo", f"jstarsi_rf_passa_vs_nao_passa_{b.lower()}_acima_30m",
             pvn_val[b], "m", "rf_v2", f"passa_vs_nao_passa.{b}.'>30m'",
             nota=(("A of both populations = adopted A (F0), only in the common transects; "
                    "for those that pass, A_Fk == A_F0 (one shot per cell: zero median "
                    "shift). New v2 block: awaiting an independent check, as the JSON "
                    "declares. G4: n of those that pass == n of the arm; per-transect sums == "
                    "totals.")))
    fato("i1_filtro_rotulo", "jstarsi_rf_dif_A_nao_passam_menos_passam_acima_30m", dif_val, "m", "rf_v2",
         "passa_vs_nao_passa.{F1,F2}.'>30m'.{passam,nao_passam,por_transecto}.media_pond_celula_m",
         operacao="nao_passam.media_pond_celula_m - passam.media_pond_celula_m (total e por transecto comum)")
    fato("i1_filtro_rotulo", "jstarsi_rf_razao_sobreviventes_acima_30m", raz_val, "razao", "rf_v1",
         "por_braco.{F1,F2,F0}.'>30m'.n_celulas", operacao="n_celulas(Fk) / n_celulas(F0)")

    dpc = r2["disparos_por_celula_ancora"][">30m"]
    if not (dpc["n_celulas"] == n_f0 == dpc["gedi_F0_por_celula"]["1"] == dpc["disparos_totais_F0_por_celula"]["1"]
            and dpc["atl08_F0_por_celula"]["0"] == n_f0 and dpc["celulas_com_atl08"] == 0):
        raise RuntimeError("G4: disparos_por_celula_ancora >30m nao e 441 x (1 GEDI, 0 ATL08)")
    fato("i1_filtro_rotulo", "jstarsi_rf_disparos_por_celula_ancora_acima_30m", dpc, "contagem", "rf_v2",
         "disparos_por_celula_ancora.'>30m'",
         nota="G4: as 441 celulas tem exatamente 1 disparo GEDI F0 e 0 ATL08; por isso o filtro remove "
              "a celula inteira ou a mantem com o mesmo rotulo")

    causa = {}
    for b in ("F1", "F2"):
        cr = r2["causa_da_remocao_por_transecto"][b][">30m"]
        rem = r1["remocao_de_celulas"][b][">30m"]
        zerados = sorted(t for t, v in cr.items() if v["n_passam"] == 0)
        if zerados != sorted(rem["transectos_removidos"]):
            raise RuntimeError(f"G4: {b} transectos com n_passam 0 != transectos_removidos")
        if b == "F1":
            cats = ["reprovados_so_degrade_flag", "reprovados_so_sensitivity",
                    "reprovados_degrade_e_sensitivity"]
        else:
            cats = ["reprovados_so_degrade_flag", "reprovados_so_sensitivity",
                    "reprovados_degrade_e_sensitivity", "reprovados_so_feixe", "reprovados_so_noite",
                    "reprovados_feixe_e_noite_so",
                    "reprovados_feixe_ou_noite_com_degrade_ou_sensitivity"]
        por_t = {}
        for t in zerados:
            v = cr[t]
            if sum(v[c] for c in cats) != v["n_reprovados"]:
                raise RuntimeError(f"G4: {b}/{t} particao das causas nao soma n_reprovados")
            dom = max(cats, key=lambda c: v[c])
            por_t[t] = {"n_gedi_F0": v["n_gedi_F0"], "n_reprovados": v["n_reprovados"],
                        "causa_dominante": dom, "n_causa_dominante": v[dom],
                        **{c: v[c] for c in cats},
                        **({"reprovados_por_noite_marginal": v["reprovados_por_noite"],
                            "reprovados_por_feixe_marginal": v["reprovados_por_feixe"]} if b == "F2" else {})}
        tot = {c: sum(cr[t][c] for t in cr) for c in cats}
        causa[b] = {"transectos_removidos": zerados, "celulas_removidas": rem["celulas_removidas"],
                    "por_transecto_removido": por_t, "totais_todos_transectos": tot}
    fato("i1_filtro_rotulo", "jstarsi_rf_causa_da_remocao_acima_30m", causa, "disparos", ["rf_v2", "rf_v1"],
         "causa_da_remocao_por_transecto.{F1,F2}.'>30m' + remocao_de_celulas.{F1,F2}.'>30m'",
         operacao="causa dominante = categoria da particao com maior contagem no transecto (argmax)",
         nota="G4: transectos com n_passam 0 == transectos_removidos; particao soma n_reprovados. "
              "Bloco novo do v2: aguarda conferencia.")

    # ==================================================================
    # I2 -- definitions used in the text
    # ==================================================================
    cst = {}
    for nome, unid in (("N_ALL_MIN", "retornos lidar por celula (n_all)"),
                       ("FRAC_CASCA_MIN", "fracao de n_all na casca inferior (adimensional)"),
                       ("DOSSEL_FLORESTA", "m (dossel_v2: altura de dossel do proprio lidar)"),
                       ("AGUA_LIMPO", "% de ocorrencia de agua (JRC Global Surface Water, 'occurrence')"),
                       ("N_CLAREIRA_MIN", "celulas de clareira por transecto"),
                       ("LIM_CORROBORACAO", "m")):
        v, ln = const("juiz_py", nome)
        cst[nome] = {"valor": v, "unidade": unid, "arquivo_linha": f"{rot(ARQS['juiz_py'])}:{ln}"}
    pj = J["perfil"]["populacao"]
    fato("i2_definicoes", "jstarsi_def_celula_floresta_julgada",
         {"constantes": cst,
          "guarda_de_qualidade": [onde("juiz_py", r'd\["n_all"\] >= N_ALL_MIN'),
                                  onde("juiz_py", r'd\["n_casca"\] / d\["n_all"\] >= FRAC_CASCA_MIN'),
                                  onde("juiz_py", r'np\.isfinite\(d\["z_solo_v2"\]\)\]')],
          "transecto_julgavel": [onde("juiz_py", r'v\["julgavel"\] = bool'),
                                 onde("juiz_py", r'd = d\[d\["transecto"\]\.isin\(julgaveis\)\]')],
          "expressao_flor": onde("juiz_py", r'^\s*flor = d\['),
          "camada_de_agua": {"leitura": onde("juiz_py", r'agua_jrc_amazonia_'),
                             "sentinela": onde("juiz_py", r'np\.where\(agua < 0'),
                             "laterais_usados_pelo_juiz": J["juiz_json"]["_proveniencia"]["nota"],
                             "laterais_prov_agua_jrc": J["laterais_prov"]["agua_jrc"]},
          "n_celulas_floresta_julgada": pj["n_floresta_julgavel_total"],
          "n_transectos_julgaveis": len(pj["julgaveis"])},
         "rotulo", ["juiz_py", "juiz_json", "laterais_prov", "perfil"],
         "constantes e linhas de juiz_lidar_v4.py (ast/regex); _proveniencia.nota do juiz; "
         "laterais_nativa_diretas/_proveniencia.json.agua_jrc; perfil.populacao",
         nota=("celula de floresta julgada = celula do parquet do lidar com n_all >= N_ALL_MIN, "
               "n_casca/n_all >= FRAC_CASCA_MIN e z_solo_v2 finito, num transecto julgavel (>= "
               "N_CLAREIRA_MIN clareiras e offset corroborado pelo topo a LIM_CORROBORACAO), deduplicada, "
               "com dossel_v2 >= DOSSEL_FLORESTA e agua < AGUA_LIMPO. Contagem da populacao lida do "
               "artefato do perfil (mesma populacao do juiz, guarda a 1e-6)."))

    x1, ln_x1 = const("b2_py", "ETAPAS_EPOCAS")

    def epocas_do_result(k):
        res = J[k]
        vals = [v["epocas"] for v in res.values() if isinstance(v, dict) and "epocas" in v]
        if not vals or any(v != vals[0] for v in vals):
            raise RuntimeError(f"{ARQS[k].name}: tetos de epoca ausentes ou diferentes entre corridas")
        return [vals[0][f"etapa{i+1}"] for i in range(len(vals[0]))], len(vals), res["config"].get("fator_epocas")
    e1, n1, f1_ = epocas_do_result("res_x1")
    e4a, n4a, f4a = epocas_do_result("res_x4_f1")
    e4b, n4b, f4b = epocas_do_result("res_x4_d1")
    if e1 != x1:
        raise RuntimeError(f"G4: tetos x1 no result {e1} != ETAPAS_EPOCAS {x1}")
    if not (f4a == f4b == 4) or e4a != [4 * t for t in x1] or e4b != e4a:
        raise RuntimeError(f"G4: tetos x4 {e4a}/{e4b} != 4 x {x1}")
    fato("i2_definicoes", "jstarsi_def_tetos_epocas_por_etapa",
         {"teto_1x_ETAPAS_EPOCAS": x1,
          "teto_1x_arquivo_linha": f"{rot(ARQS['b2_py'])}:{ln_x1}",
          "fator": onde("b2_py", r"ETAPAS_EPOCAS\[:\] = \[t \* a\.fator_epocas"),
          "argumento": onde("b2_py", r'add_argument\("--fator-epocas"'),
          "teto_1x_em_results": {"arquivo": rot(ARQS["res_x1"]), "tetos": e1, "n_corridas": n1,
                                 "fator_epocas_no_config": f1_ if f1_ is not None else "ausente (default 1)"},
          "teto_4x_em_results": {"arquivos": [rot(ARQS["res_x4_f1"]), rot(ARQS["res_x4_d1"])],
                                 "tetos": e4a, "n_corridas": [n4a, n4b], "fator_epocas_no_config": 4},
          "paciencia": onde("b2_py", r"return max\(6, teto // 3\)")},
         "epocas", ["b2_py", "res_x1", "res_x4_f1", "res_x4_d1"],
         "ETAPAS_EPOCAS (ast); linha do fator; results/*.json -> <corrida>.epocas e config.fator_epocas",
         nota="teto 4x = cada teto por etapa multiplicado por --fator-epocas (lista inteira, "
              "ETAPAS_EPOCAS[:] = [t * fator ...]); G4: tetos gravados nos results x4 == 4 x ETAPAS_EPOCAS")

    grid, _ = const("grade_py", "GRID")
    quad_n, ln_q = const("grade_py", "QUAD_N")
    n_nos_q = quad_n * quad_n

    def dossel_por_seed(k):
        out = {}
        for kk, v in J[k].items():
            if not (isinstance(v, dict) and v.get("G_raiox") and v.get("seed") is not None):
                continue
            g = v["G_raiox"]
            fr = {q: g[q]["F_dossel"]["frac_dossel_denso"] for q in ("Q1", "Q2", "Q3", "Q4")}
            out[str(v["seed"])] = {"frac_dossel_denso_por_quadrante": fr,
                                   "n_celulas_dossel_denso_aprox": int(round(sum(fr.values()) * n_nos_q)),
                                   "area_km2_por_quadrante": [g[q]["E_admissibilidade"]["area_km2"]
                                                              for q in ("Q1", "Q2", "Q3", "Q4")]}
        if not out:
            raise RuntimeError(f"{ARQS[k].name}: sem G_raiox")
        return out
    den_x1, den_x4 = dossel_por_seed("res_x1"), dossel_por_seed("res_x4_d1")
    fr_ref = next(iter(den_x1.values()))["frac_dossel_denso_por_quadrante"]
    for dd in (den_x1, den_x4):
        for s, v in dd.items():
            if v["frac_dossel_denso_por_quadrante"] != fr_ref:
                raise RuntimeError(f"G4: frac_dossel_denso difere entre corridas (seed {s})")
    celula_m = (2.0 / grid) * const("grade_py", "GRAU_M")[0]
    area_q = round((quad_n * celula_m / 1000.0) ** 2, 1)
    for dd in (den_x1, den_x4):
        for v in dd.values():
            if any(abs(a_ - area_q) > 0.05 for a_ in v["area_km2_por_quadrante"]):
                raise RuntimeError("G4: area_km2 gravada != (QUAD_N * CELULA_M)^2")
    n_den = next(iter(den_x1.values()))["n_celulas_dossel_denso_aprox"]
    fato("i2_definicoes", "jstarsi_def_replica_admissibilidade",
         {"replica": ("uma corrida = uma semente: o modelo do fim da cadeia (checkpoint '_ate<ultimo quadrante>') "
                      "infere Delta em TODOS os nos dos quatro quadrantes, e a contagem da corrida e a soma "
                      "das contagens dos quatro quadrantes; nao se restringe as reservas"),
          "agregacao": [onde("adm_py", r"agrega por corrida \(soma dos 4 quadrantes"),
                        onde("adm_py", r"viol \+= int\(round\(nv \* fr\)\)"),
                        onde("adm_py", r'^\s*s = v\.get\("seed"\)')],
          "inferencia_em_todos_os_nos": [onde("b2_py", r'f"_\{out\[.modo.\]\}_ate\{ult\}\.pt"'),
                                         onde("b2_py", r"delta = SP\.inferir_quadrante\(bioma, q, ck, log\)")],
          "QUAD_N": quad_n, "QUAD_N_arquivo_linha": f"{rot(ARQS['grade_py'])}:{ln_q}",
          "nos_por_quadrante": n_nos_q, "nos_por_replica": 4 * n_nos_q,
          "celulas_de_dossel_denso_por_replica_aprox": n_den,
          "frac_dossel_denso_por_quadrante": fr_ref,
          "sementes_conferidas": {"x1": sorted(den_x1), "x4_d1": sorted(den_x4)}},
         "celulas", ["adm_py", "b2_py", "grade_py", "res_x1", "res_x4_d1"],
         "auditar_admissibilidade.por_corrida (linhas); b2_transferencia._raiox_da_corrida (linhas); "
         "grade_v23.QUAD_N (ast); results -> <corrida>.G_raiox.Q*.F_dossel.frac_dossel_denso",
         operacao="nos_por_replica = 4 * QUAD_N^2; dossel denso ~= soma_Q frac_dossel_denso(Q) * QUAD_N^2",
         nota=("denominador de dossel denso e APROXIMADO: frac_dossel_denso e gravada arredondada a 5 casas "
               "(erro <= ~65 celulas por quadrante); a mascara e a mesma em todas as corridas conferidas "
               "(G4). A contagem 'violacoes > 1 m' e round(n_violacoes_dossel * frac_violacoes_acima_1m) "
               "por quadrante, tambem sobre fracao arredondada a 5 casas."))

    lim_ln, lim_txt = linha("b2_py", r"frac_violacoes_acima_1m=round\(float\(\(v > ")
    lim = float(re.search(r"v > ([0-9.]+)\)", lim_txt).group(1))
    fato("i2_definicoes", "jstarsi_def_violacao_admissibilidade",
         {"superficie_de_comparacao": "GLO-30 (DSM de entrada), laterais glo30_<bioma>_<Q>.npz chave DEM",
          "leitura_da_superficie": onde("b2_py", r'dsm = np\.load\(LATDIR_V23 / f"glo30_\{bioma\}_\{q\}\.npz"\)'),
          "solo_estimado": onde("b2_py", r"z = \(dsm - delta\)\.astype"),
          "mascara_dossel_denso": [onde("b2_py", r"m = pert >= 0\.5"),
                                   onde("b2_py", r'i = V\.NOMES_REGIME\.index\("dossel_denso"\)')],
          "sinal": [onde("b2_py", r"^\s*neg = delta < 0$"), onde("b2_py", r"^\s*nd = neg & m$"),
                    onde("b2_py", r"^\s*v = -delta\[nd\]$")],
          "limiar_m": lim, "limiar_arquivo_linha": f"{rot(ARQS['b2_py'])}:{lim_ln}",
          "arvore_mesma_funcao": onde("s7_py", r'"F_dossel": B2\._delta_sob_dossel\("amazonia", q, delta\)')},
         "m", ["b2_py", "s7_py"], "b2_transferencia._delta_sob_dossel e _raiox_da_corrida (linhas); "
         "auditar_s7_arvore_lidar.py (linha)",
         nota=("violacao > 1 m = no com pertinencia a 'dossel_denso' >= 0,5 (regime do vetor_v23, a mesma "
               "fonte que a penalidade usa) em que Delta_hat < -1 m, isto e, solo estimado z' = GLO-30 - "
               "Delta_hat mais de 1 m ACIMA do DSM GLO-30; nao e comparado a um produto de altura de dossel."))

    seg, l0, l1 = funcao("regua_v2_py", "reducao_de")
    fato("i2_definicoes", "jstarsi_def_reducao_mae_equivalente",
         {"funcao": f"{rot(ARQS['regua_v2_py'])}:{l0}-{l1}", "codigo": seg,
          "notacao": ("R = (1/N) sum_i |e_i| - (1/N) sum_i |e_i - A_bar|, A_bar = (1/N) sum_i A_i, "
                      "sobre as N celulas-ancora das unidades (transectos) da amostra, na classe"),
          "A": [onde("tabela3_py", r'anc\["vies_rotulo"\] = anc\["solo_anc"\] - anc\["z_ref"\]'),
                onde("regua_v2_py", r'ga = \{u: g\["termo_rotulo"\]')],
          "e": [onde("tabela3_py", r'anc\["e_modelo"\] = anc\["v23"\] - anc\["z_ref"\]'),
                onde("regua_v2_py", r'ge = \{u: g\["e_modelo_m"\]')],
          "docstring": onde("regua_v2_py", r"MAE\(\|e_modelo\|\) - MAE\(\|e_modelo - media_de_classe\(A\)\|\)")},
         "m", ["regua_v2_py", "tabela3_py"], "auditar_rotulo_regua_v2.mae_reducao_unidade.reducao_de (ast)",
         nota=("A = solo_anc - z_ref (termo de rotulo), e = v23 - z_ref (erro do grafo contra o lidar); "
               "reducao = MAE antes menos MAE depois de retirar de e a media de classe de A; no bootstrap "
               "as unidades sao reamostradas e A_bar recalculado na amostra"))

    q4 = J["ql4_json"]
    amz = [b for b in q4["biomas"] if b["bioma"] == "amazonia"]
    if len(amz) != 1:
        raise RuntimeError("ql4: bioma amazonia ausente/duplicado")
    qz = J["qlz_json"]
    fato("i2_definicoes", "jstarsi_def_regua_filtros_gedi",
         {"estado": (("artifacts on disk (_arquivo_2026-08-09/auditorias_pontuais), WITHOUT a "
                      "provenance stamp (no _proveniencia/_fontes; generated on Windows on "
                      "2026-08-05) and without an independent recomputation; descriptive, not "
                      "citable as a number of the article")),
          "comentario_no_codigo": onde("rotulos_py", r"solo exposto definido por cobertura arborea Landsat"),
          "comentario_sensitivity": onde("rotulos_py", r"sensitivity >= 0,95   INUTIL"),
          "comentario_degrade": onde("rotulos_py", r"degrade_flag == 0     NAO MONOTONICA"),
          "regua_4biomas": q4["regua"], "gerado_em_4biomas": q4["gerado_em"],
          "amazonia_4biomas": amz[0],
          "do_zero": {"gerado_em": qz["gerado_em"], "bioma": qz["bioma"], "quad": qz["quad"],
                      "treecover_max_pct": qz["treecover_max_pct"],
                      "disparos_em_solo_exposto": qz["disparos_em_solo_exposto"],
                      "erro_de_referencia": qz["erro_de_referencia"]}},
         "m", ["rotulos_py", "ql4_json", "ql4_py", "qlz_json", "qlz_py"],
         "rotulos_lidar_v23.py (linhas de comentario); qualidade_lidar_4biomas.json.{regua,gerado_em,biomas[amazonia]}; "
         "qualidade_lidar_do_zero.json.{gerado_em,bioma,quad,treecover_max_pct,disparos_em_solo_exposto,erro_de_referencia}",
         nota="regua = |DSM - solo GEDI| sobre solo exposto (cobertura arborea Landsat <= 10%); o comentario "
              "do codigo cita os dois scripts; os JSON existem mas nao sao carimbados")

    # ==================================================================
    # I3 -- pre-specified contrasts
    # ==================================================================
    fila_txt = TXT["fila"]

    def secao(ini, fim):
        i, j = fila_txt.index(ini), fila_txt.index(fim)
        return "\n".join(l.lstrip("# ").rstrip() for l in fila_txt[i:j].splitlines())
    sec = {"D8": secao("# ── D8", "# ── F2"), "F2": secao("# ── F2", "# ── F1"),
           "F1": secao("# ── F1", "# ORDEM:")}
    for k in ("d8", "f2", "f1"):
        f0 = J[k]["_fontes"][0]
        if Path(f0["caminho"]).name != ARQS["fila"].name or f0["sha256"] != sha0["fila"]:
            raise RuntimeError(f"G3: {ARQS[k].name} nao cita a fila com o sha256 de hoje")
    fato("i3_contrastes", "jstarsi_contrastes_arquivo_que_fixou",
         {"arquivo": rot(ARQS["fila"]), "tipo": "script bash de fila de execucao (.sh), cabecalho comentado",
          "sha256": sha0["fila"], "linha_cabecalho": onde("fila", r"FILA DE 2026-08-16"),
          "criterios_literais": sec,
          "artefatos_que_citam_com_sha256": [rot(ARQS[k]) for k in ("d8", "f2", "f1")]},
         "rotulo", ["fila", "d8", "f2", "f1"], "fila (cabecalho, secoes D8/F2/F1); <artefato>._fontes[0]",
         nota="G3: os tres artefatos listam a fila como primeira fonte com o sha256 de hoje")
    DESC = {"d8": ("D8", "arvore (XGBoost) contra grafo (GATv2) a fator de epocas 4, lambda_tv 0, boost 4, "
                   "pareado em 5 sementes", "delta_T_xgb_menos_gatv2_fator4"),
            "f2": ("F2", "fracao de ancoras de treino 0,25 contra 1,00 a fator de epocas 1 (supervisao x teto), "
                   "por braco", "diferenca_das_degradacoes_gatv2_menos_mlp"),
            "f1": ("F1", "rampa de fases escalada (--fator-fases 4) a fator de epocas 4: grafo menos pontual, "
                   "10 sementes", "delta_F1_gatv2_menos_mlp")}
    for k, (nome, varia, campo_res) in DESC.items():
        d = J[k]
        fato("i3_contrastes", f"jstarsi_contraste_{nome.lower()}",
             {"nome": nome, "o_que_varia": varia, "criterio_pre_declarado": d["criterio_pre_declarado"],
              "veredito": d["veredito"], "resultado": d[campo_res], "artefato_do_resultado": rot(ARQS[k])},
             "skill", k, f"criterio_pre_declarado; veredito; {campo_res}",
             nota="o_que_varia e resumo das secoes da fila (texto literal em jstarsi_contrastes_arquivo_que_fixou)")
    def folhas_nao_texto(o, t=""):
        if isinstance(o, dict):
            return [x for kk in sorted(o) for x in folhas_nao_texto(o[kk], f"{t}/{kk}")]
        if isinstance(o, list):
            return [x for i, vv in enumerate(o) for x in folhas_nao_texto(vv, f"{t}[{i}]")]
        return [] if isinstance(o, str) else [(t, o)]
    pres = {}
    for k in ("d8", "f2", "f1"):
        orig = {kk: vv for kk, vv in J[k].items() if kk != "_proveniencia"}
        pk = {kk: vv for kk, vv in J["pkg_" + k].items() if kk != "_proveniencia"}
        num_igual = folhas_nao_texto(orig) == folhas_nao_texto(pk)
        fila_pk = J["pkg_" + k]["_fontes"][0]["sha256"] == sha0["fila"]
        pres[rot(ARQS[k])] = {"no_pacote": rot(ARQS["pkg_" + k]),
                              "sha256_igual_ao_original": sha0["pkg_" + k] == sha0[k],
                              "folhas_numericas_e_booleanas_identicas": num_igual,
                              "pacote_cita_a_fila_com_o_mesmo_sha256": fila_pk}
        if not (num_igual and fila_pk):
            raise RuntimeError(f"G3: copia do pacote de {ARQS[k].name} difere em numero ou no sha da fila")
    fila_no_pacote = sorted(p.name for p in PKG.rglob("*fila*"))
    fato("i3_contrastes", "jstarsi_contrastes_no_repositorio_publico",
         {"artefatos": pres, "fila_no_pacote": bool(fila_no_pacote), "arquivos_fila_no_pacote": fila_no_pacote or ["nenhum"]},
         "rotulo", ["pkg_d8", "pkg_f2", "pkg_f1", "d8", "f2", "f1"],
         "presenca e sha256 em artigo_v23_2/pacote_repositorio_jstars/audit/; busca *fila* no pacote",
         nota=("the local package is the copy of the public repository; the copies are "
               "TRANSLATED (texts in English, internal note paths replaced by "
               "placeholders), so the sha256 differs from the original; all "
               "numeric/boolean leaves are identical (G3). The queue that fixed the "
               "criteria is NOT in the package (only its sha256, inside the _fontes of "
               "the three JSON)"))

    crit = J["nmad_anadem"]["criterio_pre_registrado"]
    tp = J["nmad_anadem"]["populacao_original_ETH"]["testes_principal_ETH_produtos_base"]
    ptr_nmad = {}
    for cl, suf in (("20-30m", "20_30m"), (">30m", "30m_mais")):
        art = tp["v23 vs fabdem"]["nmad"][cl]
        ptr_nmad[cl] = ponteiro(f"jstarsc_contraste_fabdem_nmad_{suf}", art,
                                ["mediana_m", "ic95_bootstrap_transecto_mediana", "n_transectos"])
        ptr_nmad[cl]["p_holm"] = art["p_holm"]
        ptr_nmad[cl]["vence_total"] = [art["vence"], art["total"]]
    t3 = J["tabela3"]["criterio_pre_registrado"]
    fato("i3_contrastes", "jstarsi_confirmatorios_referencia",
         {"nmad_grafo_menos_fabdem": {
             "fixado_em": rot(ARQS["nmad_anadem"]) + " -> criterio_pre_registrado (herdado de auditar_nmad_pareado.py)",
             "familia_confirmatoria": crit["familia_confirmatoria"], "metrica": crit["metrica_confirmatoria"],
             "correcao": crit["correcao"], "piso_celulas_por_transecto_faixa": crit["piso_celulas_por_transecto_faixa"],
             "min_transectos_por_faixa": crit["min_transectos_por_faixa"], "resultado": ptr_nmad},
          "A_acima_20m_piso_200": {
             "fixado_em": rot(ARQS["tabela3"]) + " -> criterio_pre_registrado",
             "piso_celulas_por_transecto_classe": t3["piso_celulas_por_transecto_classe"],
             "min_transectos_acima_do_piso": t3["min_transectos_acima_do_piso"], "aceite": t3["aceite"],
             "resultado": {"id_existente": "jstarsb_e4_confirmatorio",
                           "valor_existente": por_id["jstarsb_e4_confirmatorio"]["valor"]}}},
         "rotulo", ["nmad_anadem", "tabela3"],
         "criterio_pre_registrado dos dois artefatos; populacao_original_ETH.testes_principal_ETH_produtos_base."
         "'v23 vs fabdem'.nmad.{20-30m,>30m}",
         nota="estes dois nao estao na fila: o criterio vive no proprio artefato/script de auditoria. "
              "G4: os ponteiros para parte_C conferidos contra o artefato")

    # ==================================================================
    # I4 -- graph vs GEDTM30 against lidar
    # ==================================================================
    ptr_g = {}
    for cl, suf in (("20-30m", "20_30m"), (">30m", "30m_mais")):
        ptr_g[cl] = {}
        for met in ("mae", "nmad"):
            art = tp["v23 vs gedtm30"][met][cl]
            p = ponteiro(f"jstarsc_contraste_gedtm30_{met}_{suf}", art,
                         ["mediana_m", "ic95_bootstrap_transecto_mediana", "n_transectos", "media_m"])
            p["vence_total_grafo_menor"] = [art["vence"], art["total"]]
            ptr_g[cl][met] = p
    fato("i4_grafo_gedtm30", "jstarsi_grafo_gedtm30_contraste_lidar", ptr_g, "m", "nmad_anadem",
         "populacao_original_ETH.testes_principal_ETH_produtos_base.'v23 vs gedtm30'.{mae,nmad}.{20-30m,>30m}",
         nota="o contraste ja esta na parte_C (ids apontados); aqui so o ponteiro conferido e k/n "
              "(vence = transectos em que o grafo tem metrica menor). Exploratorio: p bruto, sem Holm")
    po = J["nmad_anadem"]["populacao_original_ETH"]
    base, ana = po["tabela_pool_ETH_produtos_base"], po["tabela_pool_ETH_anadem_so_onde_valido"]
    pool, conf_b = {}, {}
    for cl, suf in (("20-30m", "20_30m"), (">30m", "30m_mais")):
        pool[cl] = {nome: {"mae_m": base[p][cl]["mae"], "n": base[p][cl]["n"]}
                    for nome, p in (("grafo", "v23"), ("pontual", "mlp"), ("fabdem", "fabdem"),
                                    ("gedtm30", "gedtm30"))}
        a_ = ana[cl]
        pool[cl]["anadem"] = {"mae_m": a_["mae"], "n": a_["n"]}
        fb = por_id[f"jstarsb_tabIV_pool_{'20_30m' if cl == '20-30m' else '30m_mais'}"]["valor"]
        conf_b[cl] = all(abs(fb[p]["mae"] - pool[cl][nome]["mae_m"]) <= 1e-9
                         for nome, p in (("grafo", "v23"), ("pontual", "mlp"), ("fabdem", "fabdem"),
                                         ("gedtm30", "gedtm30"), ("anadem", "anadem")))
    fato("i4_grafo_gedtm30", "jstarsi_mae_agrupado_por_produto_20_30m_e_acima_30m",
         {"mae_agrupado": pool, "igual_a_jstarsb_tabIV_pool": conf_b},
         "m", "nmad_anadem",
         "populacao_original_ETH.{tabela_pool_ETH_produtos_base.<produto>,tabela_pool_ETH_anadem_so_onde_valido}"
         ".{20-30m,>30m}.{mae,n}",
         nota="populacao original (ANADEM so onde valido); igual_a_jstarsb_tabIV_pool compara com a "
              "parte_B (populacao comum aos produtos) a 1e-9")

    # ==================================================================
    # I5 -- panel (b) and panel (a) count of the three-family figure
    # ==================================================================
    fn = J["fig3_num"]
    fnum = fn["numeros"]
    pqd = J["d8"]["por_quadrante"]["xgb_menos_gatv2_fator4"]
    for f_ in fn["fontes"]:
        k = {"auditar_d8_triangulo.json": "d8"}.get(f_["caminho"])
        if k and not sha0[k].startswith(f_["sha256"]):
            raise RuntimeError(f"G3: fig3_numeros cita {f_['caminho']} com sha {f_['sha256']} != hoje")
    painel = {}
    for q in ("Q1", "Q2", "Q3", "Q4"):
        igual(fnum["b_por_quadrante"][q]["media"], pqd[q]["media"], 0.0, f"fig3 {q} media")
        if fnum["b_por_quadrante"][q]["sinal"] != pqd[q]["sinal_consistente"]:
            raise RuntimeError(f"G4: fig3 {q} sinal != d8")
        painel[q] = {"arvore_menos_grafo_skill": pqd[q]["media"], "sementes_com_o_mesmo_sinal": pqd[q]["sinal_consistente"]}
    n_arv = sum(1 for q in painel if painel[q]["arvore_menos_grafo_skill"] > 0)
    if n_arv != pqd["quadrantes_com_media_positiva"]:
        raise RuntimeError("G4: reservas favoraveis a arvore != quadrantes_com_media_positiva")
    fato("i5_fig3", "jstarsi_fig3_painel_b_por_reserva",
         {"por_reserva": painel, "reservas_que_favorecem_a_arvore": n_arv,
          "reservas_que_favorecem_o_grafo": 4 - n_arv, "margem_skill": fnum["b_margem"],
          "piso_p_teste_de_sinal": fnum["b_piso_teste_sinal_p"]},
         "skill", ["fig3_num", "d8", "fig3_py"],
         "fig3_numeros.numeros.b_por_quadrante; auditar_d8_triangulo.por_quadrante.xgb_menos_gatv2_fator4",
         operacao="favorece a arvore se media (arvore - grafo) > 0; contagem sobre as 4 reservas",
         nota="G4: valores da figura == artefato; contagem == quadrantes_com_media_positiva")
    adm_arv = por_id["jstarsg_adm_arvore_resumo"]["valor"]["violacoes_acima_1m_media"]
    igual(fnum["a_violacoes_acima_1m_media_arvore"], adm_arv, 0.0, "fig3 violacoes arvore == parte_G")
    fato("i5_fig3", "jstarsi_fig3_painel_a_violacoes",
         {"redes": fnum["a_violacoes_acima_1m_media"], "arvore": fnum["a_violacoes_acima_1m_media_arvore"],
          "arvore_id_existente": "jstarsg_adm_arvore_resumo"},
         "celulas por replica", ["fig3_num", "fig3_py"],
         "fig3_numeros.numeros.{a_violacoes_acima_1m_media,a_violacoes_acima_1m_media_arvore}",
         nota="G4: arvore == parte_G jstarsg_adm_arvore_resumo.violacoes_acima_1m_media")

    # ==================================================================
    # I6 -- ANADEM MAE on the profile strip
    # ==================================================================
    pf = J["perfil"]
    if pf["guarda_reproduz_juiz_1e-6"]["passou"] is not True:
        raise RuntimeError("I6: guarda do perfil nao passou; fato do ANADEM nao entra")
    rf_ = pf["resultado_faixa"]
    fato("i6_perfil_anadem", "jstarsi_perfil_mae_seis_produtos_np_t_0224",
         {"faixa": pf["faixa_do_perfil"], "mae_m": rf_["mae_m_seis_produtos"],
          "ordem_crescente": rf_["ordem_crescente_de_mae"],
          "n_celulas": rf_["n_celulas_floresta_julgada"], "n_anadem_finito": rf_["n_celulas_anadem_finito"],
          "guarda_max_abs_dif_m": pf["guarda_reproduz_juiz_1e-6"]["max_abs_dif_todos_transectos_m"]},
         "m", ["perfil", "perfil_py", "juiz_json"],
         "resultado_faixa; guarda_reproduz_juiz_1e-6",
         nota=pf["metodo"])
    fato("i6_perfil_anadem", "jstarsi_perfil_anadem_por_transecto_descritivo",
         {t: {"anadem_mae_m": v["anadem"], "n_celulas": v["n_celulas"]} for t, v in pf["por_transecto_mae_descritivo"].items()},
         "m", "perfil", "por_transecto_mae_descritivo.<T>.{anadem,n_celulas}",
         nota="descritivo, sem teste; mesma populacao e regua k0 dos outros cinco produtos")

    # ------------------------------------------------------------------
    qualidade = {"notas": [
        ("auditar_rf_filtro_gedi.json independently checked: CONFIRMED WITH "
         "CAVEAT; the new v2 blocks (disparos_por_celula_ancora, "
         "passa_vs_nao_passa, causa_da_remocao_por_transecto) await verification, "
         "as the JSON itself declares"),
        ("auditar_perfil_anadem_faixa.json was generated in this run and awaits an "
         "independent recomputation"),
        "qualidade_lidar_*.json: sem carimbo de proveniencia; descritivo",
        "denominador de dossel denso por replica e aproximado (fracao gravada a 5 casas)",
    ]}

    for k, p in ARQS.items():
        if PROV.sha256(p) != sha0[k]:
            raise RuntimeError(f"G3: {p.name} mudou durante a corrida")

    n_fatos_i = sum(len(v) for v in grupos.values())
    saida = {
        "gerado_para": v26["gerado_para"],
        "regra": v26["regra"],
        "protocolo": (("Check R-F, closed in fact sheet v27: parte_I; parts A-H = "
                       "folha_de_fatos_jstars_v26.json (sha256 ") + SHA_ESPERADO["folha_v26"][:8] + "...), identicas"),
        **{n: v26[v] for n, v in TOPO.items() if n.startswith("protocolo_")},
        **copia,
        "parte_I": {"n_fatos": n_fatos_i, "grupos": {g: len(v) for g, v in grupos.items()},
                    "qualidade_das_fontes": qualidade, "fatos": grupos},
        "guardas": {
            "GI_partes_A_H_identicas_a_v26": {"trilhas_divergentes": [], "passou": True},
            "G1_ids_unicos": {"n_ids_v26": n_ids_v26, "n_ids_total": len(ids)},
            "G2_sem_nulos": True,
            "G3_sha256_conferidos": {"fixados": sorted(SHA_ESPERADO), "passou": True},
            "G4_derivados_conferidos": True,
        },
        **{n: v26[v] for n, v in TOPO.items() if n.startswith("guardas_")},
    }
    saida_p = Path(a.saida)
    if not saida_p.is_absolute():
        saida_p = RAIZ / saida_p
    PROV.gravar(saida_p, saida, fontes_lidas=list(ARQS.values()), script=__file__)

    relido = json.loads(saida_p.read_text(encoding="utf-8"))
    if json.loads(ARQS["folha_v26"].read_text(encoding="utf-8")) != v26:
        raise RuntimeError("GI: v26 no disco mudou durante a corrida")
    conferir_GI(relido, "JSON relido do disco")
    if relido["parte_I"] != json.loads(json.dumps(saida["parte_I"])):
        raise RuntimeError("releitura: parte_I gravada != montada")
    log(f"GI ok: {', '.join(COPIADAS)} == v26 (apos releitura)")
    log(f"G3: {len(SHA_ESPERADO)} sha256 fixados; {len(ARQS)} fontes")
    log(f"ids: {n_ids_v26} da v26 + {len(ids) - n_ids_v26} novos = {len(ids)}")
    log(f"parte_I: {n_fatos_i} fatos em {len(grupos)} grupos -> {saida_p}")
    for g, v in grupos.items():
        log(f"  {g} ({len(v)}): {', '.join(f['id'] for f in v)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
