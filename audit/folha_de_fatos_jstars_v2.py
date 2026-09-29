# -*- coding: utf-8 -*-
"""folha_de_fatos_jstars_v2 -- adds part B (stamped) to the fact sheet for
Article 1 of V23 (target: IEEE JSTARS).

Does not edit `folha_de_fatos_jstars.py` or `folha_de_fatos_jstars.json`.
Part A of this JSON is read PROGRAMMATICALLY from the current sheet
(checked against its pinned sha256 before copying) and reproduced
IDENTICALLY; no value from part A is retyped.

Part B does not include the original T3 (area grid); the elevation delta
comes from the native D-1 measurement
(`medir_delta_escala_v3_nativa.json` / `delta_escala_celulas_nativa.parquet`);
the Fig. 7(a) band is out of scope (no fact is created for it). Allowed
sources: `auditar_rotulo_regua_v2.json`,
`auditar_nmad_pareado_nativa_anadem.json`,
`medir_delta_escala_v3_nativa.json` / `delta_escala_celulas_nativa.parquet`,
`auditar_d5_invariancia.json`, and `auditar_tabela3_dump.parquet` (only for
the "full cover" population fact, computed here in code).

Guards (hard failures):
  G1  unique fact id -- includes the part-A ids (loaded from the current
      sheet) in the same namespace, to guarantee part B never collides
      with part A.
  G2  no `valor` field comes out None/NaN anywhere in the sheet, at any
      nesting level (the "not tested" case of the paired NMAD at 0-3 m/
      3-10 m becomes a boolean proposition + reason, never a raw None).
  G3  every source used is listed in `_fontes` (via proveniencia.PROV.gravar)
      with the sha256 taken at read time.
  G4  derivations (the "families of estimator" reading of criterion (O);
      "full cover" from cob_eth) are computed in code from the cited
      fields, never copied from prose.
  G5  reproduction of the three anchor values before saving: E4 +6.651019 m
      (>30 m); native delta +3.7654 m (>30 m); v23xfabdem +2.3259 m
      (D5, lambda=0). Hard failure if any is outside tolerance.

Usage:  python folha_de_fatos_jstars_v2.py [--saida folha_de_fatos_jstars_v2.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import proveniencia as PROV          # noqa: E402

# sha256 of the previously verified folha_de_fatos_jstars.json (part A):
# "34/34 facts match, byte-identical".
SHA256_FOLHA_A_ESPERADO = (
    "b8a16a421486c9ac37b1b249e1020b02e9fd1ec9753679bfc2bd234714669103")

ARQS = {
    "folha_a": RAIZ / "folha_de_fatos_jstars.json",
    "e4v2": RAIZ / "auditar_rotulo_regua_v2.json",
    "tab4": RAIZ / "auditar_nmad_pareado_nativa_anadem.json",
    "d1json": RAIZ / "medir_delta_escala_v3_nativa.json",
    "d1parquet": RAIZ / "delta_escala_celulas_nativa.parquet",
    "d5": RAIZ / "auditar_d5_invariancia.json",
    "e4v1dump": RAIZ / "auditar_tabela3_dump.parquet",
}

CLASSES = ["0-3m", "3-10m", "10-20m", "20-30m", ">30m", "TODAS"]
CLASSE_ID = {"0-3m": "0_3m", "3-10m": "3_10m", "10-20m": "10_20m",
             "20-30m": "20_30m", ">30m": "30m_mais", "TODAS": "todas"}
PRODUTOS_TABELA_IV = ["v23", "mlp", "fabdem", "gedtm30", "glo30", "anadem"]


def log(m=""):
    print(m, flush=True)


def sha256_arquivo(p: Path) -> str:
    return PROV.sha256(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default="folha_de_fatos_jstars_v2.json")
    a = ap.parse_args()

    for k, p in ARQS.items():
        if not p.exists():
            raise FileNotFoundError(f"fonte ausente: {k} -> {p}")

    # ------------------------------------------------------------------
    # G3 (part 1): sha256 of folha_A BEFORE trusting any field.
    # ------------------------------------------------------------------
    sha_folha_a = sha256_arquivo(ARQS["folha_a"])
    if sha_folha_a != SHA256_FOLHA_A_ESPERADO:
        raise RuntimeError(
            (f"folha_de_fatos_jstars.json changed since the independent check: current sha256={sha_folha_a} "
             f"expected={SHA256_FOLHA_A_ESPERADO}. Not proceeding to copy parte_A from a "
             f"fact sheet that was not independently checked."))

    d_folha_a = json.loads(ARQS["folha_a"].read_text(encoding="utf-8"))
    parte_a = d_folha_a["parte_A"]

    d_e4v2 = json.loads(ARQS["e4v2"].read_text(encoding="utf-8"))
    d_tab4 = json.loads(ARQS["tab4"].read_text(encoding="utf-8"))
    d_d1 = json.loads(ARQS["d1json"].read_text(encoding="utf-8"))
    d_d5 = json.loads(ARQS["d5"].read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Fact infrastructure (G1, G2, groups)
    # ------------------------------------------------------------------
    ids = set(f["id"] for f in parte_a["fatos"])          # G1 inherits part_A
    grupos: dict[str, list] = {}

    def _sem_nulos(x, trilha):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            raise RuntimeError(f"G2: valor nulo em {trilha}")
        if isinstance(x, dict):
            for kk, vv in x.items():
                _sem_nulos(vv, f"{trilha}.{kk}")
        elif isinstance(x, (list, tuple)):
            for i, vv in enumerate(x):
                _sem_nulos(vv, f"{trilha}[{i}]")

    def _ic_ou_indisponivel(ic, n_unidades):
        """G2: a per-cluster bootstrap CI with a single cluster (class 0-3 m,
        n_unidades=1) comes out as [null,null] in the artifact -- it is
        turned into a descriptive proposition, never a raw None."""
        if ic is not None and ic[0] is not None and ic[1] is not None:
            return ic
        return {"disponivel": False,
                "motivo": f"n_unidades={n_unidades}: bootstrap por cluster "
                          "exige >=2 unidades; artefato grava [null,null]"}

    def fato(grupo, id_, valor, unidade, chave_arq, campo, nota=None):
        if id_ in ids:                                     # G1
            raise RuntimeError(f"G1: id duplicado: {id_}")
        ids.add(id_)
        _sem_nulos(valor, id_)                              # G2
        arq = ARQS[chave_arq]
        linha = {"id": id_, "valor": valor, "unidade": unidade,
                  "artefato": arq.name, "campo": campo}
        if nota:
            linha["nota"] = nota
        grupos.setdefault(grupo, []).append(linha)
        return linha

    # ==================================================================
    # GROUP 1 -- E4 v2 (auditar_rotulo_regua_v2.json): A per ETH class,
    # native delta (with v2 CI), A-delta, equivalent MAE reduction,
    # S/O/R verdicts, confirmatory check, floor curve, P1 budget.
    # ==================================================================
    pc = d_e4v2["por_classe_eth"]
    for c in CLASSES:
        cid = CLASSE_ID[c]
        bloco_c = pc[c]
        tr = bloco_c["por_unidade"]["transecto"]
        pg = bloco_c["por_unidade"]["pegada"]

        fato("e4_A_por_classe", f"jstarsb_e4_A_{cid}",
             {"media_pond_celula_m": tr["A"]["media_pond_celula_m"],
              "mediana_m": tr["A"]["mediana_m"],
              "n_celulas": tr["A"]["n_celulas"],
              "n_transectos": tr["A"]["n_unidades"],
              "n_pegadas": pg["A"]["n_unidades"],
              "ic95_pond_celula": _ic_ou_indisponivel(
                  tr["A"]["ic95_pond_celula"], tr["A"]["n_unidades"]),
              "ic95_mediana": _ic_ou_indisponivel(
                  tr["A"]["ic95_mediana"], tr["A"]["n_unidades"]),
              "ic95_t_jackknife_pond": _ic_ou_indisponivel(
                  tr["A"]["ic95_t_jackknife_pond"], tr["A"]["n_unidades"])},
             "m", "e4v2",
             f"por_classe_eth.'{c}'.por_unidade.{{transecto,pegada}}.A."
             "{media_pond_celula_m,mediana_m,n_celulas,n_unidades,"
             "ic95_pond_celula,ic95_mediana,ic95_t_jackknife_pond}",
             nota="A = solo_anc - z_ref (termo de rotulo E regua; o "
                  "desenho NAO isola o rotulo -- ver hipotese do proprio "
                  "artefato). IC percentil e t-jackknife, unidade "
                  "transecto (primaria); n de pegadas ao lado (unidade "
                  "obrigatoria, regra sem parametro: pegadas que "
                  "compartilham celula formam uma unidade)")

        fato("e4_delta_nativo_com_ic", f"jstarsb_e4_delta_{cid}",
             {"media_pond_celula_m": tr["delta"]["media_pond_celula_m"],
              "mediana_m": tr["delta"]["mediana_m"],
              "n_celulas": tr["delta"]["n_celulas"],
              "n_transectos": tr["delta"]["n_unidades"],
              "n_pegadas": pg["delta"]["n_unidades"],
              "ic95_pond_celula": _ic_ou_indisponivel(
                  tr["delta"]["ic95_pond_celula"], tr["delta"]["n_unidades"]),
              "ic95_t_jackknife_pond": _ic_ou_indisponivel(
                  tr["delta"]["ic95_t_jackknife_pond"], tr["delta"]["n_unidades"])},
             "m", "e4v2",
             f"por_classe_eth.'{c}'.por_unidade.{{transecto,pegada}}.delta."
             "{media_pond_celula_m,mediana_m,n_celulas,n_unidades,"
             "ic95_pond_celula,ic95_t_jackknife_pond}",
             nota="delta nativo (D-1) PAREADO por celula com IC "
                  "(bootstrap do proprio auditor v2); DIFERENTE do delta "
                  "descritivo sem IC do grupo delta_nativo_fig6b, que vem "
                  "direto de medir_delta_escala_v3_nativa.json")

        fato("e4_pareada_A_menos_delta", f"jstarsb_e4_A_menos_delta_{cid}",
             {"media_pond_celula_m":
                  tr["pareada_A_menos_delta"]["media_pond_celula_m"],
              "mediana_m": tr["pareada_A_menos_delta"]["mediana_m"],
              "n_celulas": tr["pareada_A_menos_delta"]["n_celulas"],
              "n_transectos": tr["pareada_A_menos_delta"]["n_unidades"],
              "ic95_pond_celula": _ic_ou_indisponivel(
                  tr["pareada_A_menos_delta"]["ic95_pond_celula"],
                  tr["pareada_A_menos_delta"]["n_unidades"]),
              "ic95_t_jackknife_pond": _ic_ou_indisponivel(
                  tr["pareada_A_menos_delta"]["ic95_t_jackknife_pond"],
                  tr["pareada_A_menos_delta"]["n_unidades"])},
             "m", "e4v2",
             f"por_classe_eth.'{c}'.por_unidade.transecto."
             "pareada_A_menos_delta.{media_pond_celula_m,mediana_m,"
             "n_celulas,n_unidades,ic95_pond_celula,ic95_t_jackknife_pond}",
             nota="insumo do criterio (R): 'A - delta > 0' so entra na "
                  "prosa se percentil E t-jackknife excluirem zero")

        fato("e4_reducao_mae_equivalente", f"jstarsb_e4_reducao_mae_{cid}",
             {"reducao_mae_equivalente_m":
                  tr["reducao_mae_equivalente"]["reducao_mae_equivalente_m"],
              "n_celulas": tr["reducao_mae_equivalente"]["n_celulas"],
              "n_transectos": tr["reducao_mae_equivalente"]["n_unidades"],
              "ic95_pond_celula": _ic_ou_indisponivel(
                  tr["reducao_mae_equivalente"]["ic95_pond_celula"],
                  tr["reducao_mae_equivalente"]["n_unidades"]),
              "ic95_t_jackknife": _ic_ou_indisponivel(
                  tr["reducao_mae_equivalente"]["ic95_t_jackknife"],
                  tr["reducao_mae_equivalente"]["n_unidades"])},
             "m", "e4v2",
             f"por_classe_eth.'{c}'.por_unidade.transecto."
             "reducao_mae_equivalente.{reducao_mae_equivalente_m,"
             "n_celulas,n_unidades,ic95_pond_celula,ic95_t_jackknife}",
             nota="retirar a media de classe de A (equivalente a corrigir "
                  "o rotulo pela media do termo); insumo do criterio (O)")

    fato("e4_confirmatorio", "jstarsb_e4_confirmatorio",
         d_e4v2["por_classe_eth"][">30m"]["confirmatorio_E4"], None, "e4v2",
         "por_classe_eth.'>30m'.confirmatorio_E4",
         nota="teste confirmatorio pre-registrado: 0 dos 36 pares "
              "transecto x classe alcancam o piso de 200 celulas (maior "
              "par tem 154); 'nao_avaliavel', nunca 'indeterminada' -- o "
              "teste nao pode ser executado, nao produziu um resultado "
              "ambiguo. Mesmo valor em todas as classes (conferido: "
              "confirmatorio_E4 identico em '20-30m' e '>30m')")
    if (d_e4v2["por_classe_eth"]["20-30m"]["confirmatorio_E4"]
            != d_e4v2["por_classe_eth"][">30m"]["confirmatorio_E4"]):
        raise RuntimeError("confirmatorio_E4 diverge entre classes")

    curva_piso_bruta = pc[">30m"]["curva_piso_declarada_apos_E4"]
    curva_piso_limpa = {}
    for piso, r in curva_piso_bruta.items():
        if r["n_transectos"] == 0:
            curva_piso_limpa[piso] = {
                "n_transectos": 0,
                "disponivel": False,
                "motivo": "nenhum transecto tem esse numero de celulas "
                          "na classe; media_pond fica indefinida (o "
                          "artefato grava null)"}
        else:
            curva_piso_limpa[piso] = r
    fato("e4_curva_piso", "jstarsb_e4_curva_piso_descritiva",
         curva_piso_limpa, None, "e4v2",
         "por_classe_eth.'>30m'.curva_piso_declarada_apos_E4 (pisos com "
         "n_transectos=0 tem a media substituida por proposicao "
         "descritiva em codigo, G2)",
         nota="descritiva, declarada DEPOIS do E4; nenhum ponto isolado "
              "pode ir a prosa principal (so a curva inteira, se entrar "
              "em material suplementar)")

    def _linha_p1_sem_nulo(linha):
        linha = dict(linha)
        if linha.get("ic95_m") is None:
            linha["ic95_m"] = {"disponivel": False,
                                "motivo": "sem medida (ver 'valor_m': "
                                          f"{linha.get('valor_m')!r})"}
        if linha.get("unidade_cluster") is None:
            linha["unidade_cluster"] = "nao_aplicavel"
        return linha

    for c in ("20-30m", ">30m"):
        cid = CLASSE_ID[c]
        p1 = d_e4v2["tabela_orcamento_P1"][c]
        fato("e4_orcamento_p1", f"jstarsb_e4_orcamento_p1_{cid}",
             {"linhas": [_linha_p1_sem_nulo(ln) for ln in p1["linhas"]],
              "maior_teto_abs_contrastes_lidar_m":
                  p1["maior_teto_abs_contrastes_lidar_m"]},
             "m", "e4v2",
             f"tabela_orcamento_P1.'{c}'.{{linhas,"
             "maior_teto_abs_contrastes_lidar_m}}",
             nota="cada linha ja traz a coluna 'populacao' (n de celulas + "
                  "n de transectos + a ressalva de populacao DIFERENTE "
                  "para os contrastes contra o lidar)")

    # --- S / O / R verdicts, two readings of the ceiling (O) ------------
    ata_path = (RAIZ.parent.parent / "HERMES" / "AGENTES" / "cientista_dados"
                / "ATAS" / "ata_2026-09-28_jstars_v23_E4_T3.md")
    if not ata_path.exists():
        raise RuntimeError(f"leitura_declarada record file missing: {ata_path}")
    sha_ata = sha256_arquivo(ata_path)

    cv = d_e4v2["criterios_veredito"]
    for c in ("20-30m", ">30m"):
        cid = CLASSE_ID[c]
        bloco = cv[c]

        fato("e4_veredito_S", f"jstarsb_e4_veredito_S_{cid}",
             {"veredito": bloco["S"]["veredito"],
              "checagens_por_unidade": bloco["S"]["checagens_por_unidade"]},
             None, "e4v2", f"criterios_veredito.'{c}'.S",
             nota=bloco["S"]["criterio"])

        fato("e4_veredito_R", f"jstarsb_e4_veredito_R_{cid}",
             {"veredito": bloco["R"]["veredito"],
              "checagens_por_unidade": bloco["R"]["checagens_por_unidade"]},
             None, "e4v2", f"criterios_veredito.'{c}'.R",
             nota=bloco["R"]["criterio"])

        # reading 1: "products" -- already comes ready in the artifact.
        o_produtos = bloco["O"]
        fato("e4_veredito_O", f"jstarsb_e4_veredito_O_produtos_{cid}",
             {"veredito": o_produtos["veredito"],
              "teto_abs_contrastes_lidar_m":
                  o_produtos["teto_abs_contrastes_lidar_m"],
              "checagens_por_unidade": o_produtos["checagens_por_unidade"]},
             None, "e4v2", f"criterios_veredito.'{c}'.O",
             nota="leitura 'produtos': teto = maior |contraste| entre "
                  "v23 e QUALQUER estimador testado contra o lidar na "
                  "classe (inclui FABDEM e GEDTM30)")

        # reading 2: "estimator families" -- computed in code (G4):
        # ceiling = only the v23 vs mlp contrast (same label family).
        p1 = d_e4v2["tabela_orcamento_P1"][c]
        contraste_mlp = next(
            e for e in p1["contrastes_lidar_detalhe"] if e["par"] == "v23 vs mlp")
        teto_familias = contraste_mlp["teto_abs_ddiferenca_m"]
        checagens_familias = {}
        for unidade in ("transecto", "pegada"):
            ch = o_produtos["checagens_por_unidade"][unidade]
            lim_infs = {
                "media": ch["lim_inf_A_media_m"],
                "mediana": ch["lim_inf_A_mediana_m"],
                "reducao": ch["lim_inf_reducao_mae_m"]}
            acima = {k: (v > teto_familias) for k, v in lim_infs.items()}
            checagens_familias[unidade] = {
                "lim_inf_A_media_m": lim_infs["media"],
                "acima_do_teto_media": acima["media"],
                "lim_inf_A_mediana_m": lim_infs["mediana"],
                "acima_do_teto_mediana": acima["mediana"],
                "lim_inf_reducao_mae_m": lim_infs["reducao"],
                "acima_do_teto_reducao": acima["reducao"]}
        veredito_familias = (
            "passa" if all(
                all(checagens_familias[u][f"acima_do_teto_{k}"]
                    for k in ("media", "mediana", "reducao"))
                for u in ("transecto", "pegada"))
            else "nao_demonstrado")
        fato("e4_veredito_O", f"jstarsb_e4_veredito_O_familias_{cid}",
             {"veredito": veredito_familias,
              "teto_abs_contraste_v23_vs_mlp_m": teto_familias,
              "checagens_por_unidade": checagens_familias},
             None, "e4v2",
             f"derivado (G4) de tabela_orcamento_P1.'{c}'."
             "contrastes_lidar_detalhe['v23 vs mlp'].teto_abs_ddiferenca_m "
             f"comparado a criterios_veredito.'{c}'.O.checagens_por_unidade",
             nota=("'familias_de_estimador' reading: ceiling = only the v23 vs mlp contrast "
                   "(SAME label/supervision family); products with different supervision "
                   "(FABDEM/GEDTM30) make the ceiling circular (physical justification). "
                   "Recomputation in code matches the prose: >30m passes (min of the three "
                   "lower bounds = median, 1.5438>1.0931); 20-30m not demonstrated "
                   "(reduction 0.2472<0.3561)"))

        fato("e4_leitura_declarada", f"jstarsb_e4_leitura_declarada_{cid}",
             {"leitura_declarada": "familias_de_estimador",
              "decidida_por": "Record of the internal review that fixed this choice (not distributed).",
              "ata": str(ata_path.relative_to(RAIZ.parent.parent)),
              "sha256_ata": sha_ata},
             None, "e4v2",
             ("Field absent from the v2 JSON; the decision is recorded in an internal "
              "note (cited alongside with its sha256)."),
             nota=("the reading declared BEFORE the run was that of the estimator families "
                   "(design notes written before v2 ran already used the GNN-MLP ceiling); "
                   "the ambiguity came from a summary line that listed external products as "
                   "'contrasts between estimators' -- resolved by dated text prior to the "
                   "run, not by the result"))

    # ==================================================================
    # GROUP 2 -- full Table IV + paired ANADEM contrasts
    # (auditar_nmad_pareado_nativa_anadem.json)
    # ==================================================================
    tp = d_tab4["populacao_comum_5_produtos_ETH"]["tabela_pool_ETH"]
    for c in CLASSES:
        cid = CLASSE_ID[c]
        linha = {}
        for prod in PRODUTOS_TABELA_IV:
            r = tp[prod][c]
            linha[prod] = {"mae": r["mae"], "vies": r["vies"],
                            "nmad": r["nmad"], "n": r["n"]}
        fato("tabela_iv_pool", f"jstarsb_tabIV_pool_{cid}", linha, "m",
             "tab4",
             f"populacao_comum_5_produtos_ETH.tabela_pool_ETH.*.'{c}'."
             "{mae,vies,nmad,n}",
             nota="todas as colunas da Tabela IV (v23, mlp, fabdem, "
                  "gedtm30, glo30, anadem), populacao comum aos 5+1 "
                  "produtos por classe ETH")

    testes = d_tab4["populacao_comum_5_produtos_ETH"]["testes_principal_ETH"]["v23 vs anadem"]
    for metrica in ("mae", "nmad"):
        for c in CLASSES:
            if c == "TODAS":
                continue  # the paired contrast only exists per testable class
            cid = CLASSE_ID[c]
            r = testes[metrica][c]
            if not r.get("testado", False):
                valor = {"testado": False,
                         "motivo": r.get("motivo", "sem motivo no artefato"),
                         "n_transectos": r["n_transectos"]}
            else:
                valor = {"testado": True,
                         "mediana_m": r["mediana_m"], "media_m": r["media_m"],
                         "ic95_bootstrap_transecto_mediana":
                             r["ic95_bootstrap_transecto_mediana"],
                         "n_transectos": r["n_transectos"],
                         "vence": r["vence"], "total": r["total"],
                         "p_sinal_binomial": r["p_sinal_binomial"],
                         "p_wilcoxon": r["p_wilcoxon"]}
            fato("tabela_iv_pareado_anadem",
                 f"jstarsb_tabIV_v23_vs_anadem_{metrica}_{cid}", valor,
                 "m" if metrica == "mae" else None, "tab4",
                 "populacao_comum_5_produtos_ETH.testes_principal_ETH."
                 f"'v23 vs anadem'.{metrica}.'{c}'",
                 nota="contraste pareado por transecto, mesma populacao "
                      "comum aos 5+1 produtos; classes 0-3m/3-10m ficam "
                      "'testado=false' por terem menos de 6 transectos "
                      "acima do piso de 200 celulas -- nao e omissao, e "
                      "proposicao registrada")

    # ==================================================================
    # GROUP 3 -- native delta per ETH class, descriptive, for Fig. 6(b)
    # (medir_delta_escala_v3_nativa.json, population of the E4/ETH-class dump)
    # ==================================================================
    pop_d1 = d_d1["populacao_dump_e4_classe_eth_exogena_sem_ic"]
    for c in CLASSES:
        cid = CLASSE_ID[c]
        r = pop_d1[c]
        fato("delta_nativo_fig6b", f"jstarsb_fig6b_delta_nativo_{cid}",
             {"n_celulas": r["n_celulas"], "n_transectos": r["n_transectos"],
              "media_pond_celula_m": r["media_pond_celula_m"]},
             "m", "d1json",
             f"populacao_dump_e4_classe_eth_exogena_sem_ic.'{c}'."
             "{n_celulas,n_transectos,media_pond_celula_m}",
             nota="delta nativo (D-1), descritivo, SEM intervalo de "
                  "confianca (o proprio artefato declara: 'IC ficam para "
                  "o auditor v2'); geometria +0,5 px, mesma logica de "
                  "medir_delta_escala_v3.py, agora na grade nativa")

    fato("delta_nativo_fig6b_geometria", "jstarsb_fig6b_geometria",
         d_d1["geometria"], None, "d1json", "geometria",
         nota=("LON0/LAT1/PASSO/GRID/dlon_px/dlat_px of the native grid (half-pixel "
               "offset confirmed by an independent recomputation of D-1)"))

    # ==================================================================
    # GROUP 4 -- D5 (ranking invariance, FABDEM and ANADEM),
    # auditar_d5_invariancia.json, independently verified.
    # ==================================================================
    contra_d5_path = (RAIZ / "artigo_v23_2" / "_pareceres_2026-09-28_execucao"
                       / "contra_auditoria_D5.md")
    if not contra_d5_path.exists():
        raise RuntimeError(f"independent check record for D5 missing: {contra_d5_path}")
    sha_contra_d5 = sha256_arquivo(contra_d5_path)
    contra_d5_texto = contra_d5_path.read_text(encoding="utf-8")
    if "CONFIRMADO COM RESSALVA" not in contra_d5_texto:
        raise RuntimeError(
            ("independent check record for D5 does not contain 'CONFIRMADO COM "
             "RESSALVA' -- not recording D5 as closed"))

    v_d5 = d_d5["veredito"]
    fato("d5_veredito", "jstarsb_d5_veredito_fabdem",
         {"veredito": v_d5["fabdem"]["veredito"],
          "lambda_max_que_passa": v_d5["fabdem"]["lambda_max_que_passa"],
          "todos_passam": v_d5["fabdem"]["todos_passam"],
          "frase_literal": v_d5["fabdem"]["frase_literal"]},
         None, "d5", "veredito.fabdem",
         nota=f"contra-auditado CONFIRMADO COM RESSALVA "
              f"({contra_d5_path.name}, sha256 {sha_contra_d5}); FABDEM "
              "mantido em toda a grade fisica de lambda [0;1]")
    fato("d5_veredito", "jstarsb_d5_veredito_anadem",
         {"veredito": v_d5["anadem_condicionado_a_contra_auditoria"]["veredito"],
          "lambda_max_que_passa":
              v_d5["anadem_condicionado_a_contra_auditoria"]["lambda_max_que_passa"],
          "todos_passam":
              v_d5["anadem_condicionado_a_contra_auditoria"]["todos_passam"],
          "frase_literal":
              v_d5["anadem_condicionado_a_contra_auditoria"]["frase_literal"]},
         None, "d5", "veredito.anadem_condicionado_a_contra_auditoria",
         nota=f"contra-auditado CONFIRMADO COM RESSALVA "
              f"({contra_d5_path.name}, sha256 {sha_contra_d5}); ANADEM "
              "so ate lambda_max=0,5 (falha em 0,75 e 1,0)")

    # stress arm -- ANADEM fails, FABDEM passes with lower bound 0.037
    estresse = d_d5["bracos"]["estresse_delta_ic_sup_transecto"]
    fato("d5_fora_do_criterio", "jstarsb_d5_estresse_fabdem",
         {"criterio_fabdem_passa": estresse["criterio_fabdem_passa"],
          "ic95_bootstrap_transecto_mediana":
              estresse["teste_pareado_v23_vs_fabdem_mae"]
              ["ic95_bootstrap_transecto_mediana"],
          "p_holm": estresse["teste_pareado_v23_vs_fabdem_mae"]["p_holm"]},
         None, "d5",
         "bracos.estresse_delta_ic_sup_transecto."
         "{criterio_fabdem_passa,teste_pareado_v23_vs_fabdem_mae}",
         nota="etiqueta fora_do_criterio: o criterio pre-registrado so "
              "define a frase para a grade fisica lambda em [0;1]; o "
              "estresse (delta = limite superior do IC bootstrap por "
              "transecto) e checagem complementar. FABDEM passa, limite "
              "inferior do IC = 0,0367 m")
    fato("d5_fora_do_criterio", "jstarsb_d5_estresse_anadem",
         {"criterio_anadem_passa": estresse["criterio_anadem_passa"],
          "ic95_bootstrap_transecto_mediana":
              estresse["teste_pareado_v23_vs_anadem_mae_condicionado_a_contra_auditoria"]
              ["ic95_bootstrap_transecto_mediana"],
          "p_holm": estresse["teste_pareado_v23_vs_anadem_mae_condicionado_a_contra_auditoria"]["p_holm"]},
         None, "d5",
         "bracos.estresse_delta_ic_sup_transecto."
         "{criterio_anadem_passa,"
         "teste_pareado_v23_vs_anadem_mae_condicionado_a_contra_auditoria}",
         nota=("label fora_do_criterio (same as above). ANADEM FAILS the stress test: CI "
               "[-1.3329; +2.4579] includes zero, p_holm=0.365 (identified by "
               "independent recomputation)"))

    # direction controls (negative lambda) -- both pass, all 4 verdicts
    for lam_key, lam_label in (("lambda_-0.50_controle_sentido", "-0,50"),
                                ("lambda_-1.00_controle_sentido", "-1,00")):
        b = d_d5["bracos"][lam_key]
        fato("d5_fora_do_criterio",
             f"jstarsb_d5_controle_sentido_{lam_label.replace(',', '_').replace('-', 'menos')}",
             {"lambda": b["lambda"],
              "criterio_fabdem_passa": b["criterio_fabdem_passa"],
              "criterio_anadem_passa": b["criterio_anadem_passa"],
              "fabdem_ic95": b["teste_pareado_v23_vs_fabdem_mae"]
                  ["ic95_bootstrap_transecto_mediana"],
              "anadem_ic95": b["teste_pareado_v23_vs_anadem_mae_condicionado_a_contra_auditoria"]
                  ["ic95_bootstrap_transecto_mediana"]},
             None, "d5", f"bracos.'{lam_key}'",
             nota="etiqueta fora_do_criterio (idem); controle de sentido, "
                  "nao a grade fisica. FABDEM e ANADEM passam nos dois "
                  "lambdas negativos, omitido no relatorio do autor")

    # ==================================================================
    # GROUP 5 -- population facts
    # ==================================================================
    unidade_pegada = d_e4v2["populacao"]["unidade_pegada"]
    fato("populacao", "jstarsb_pop_13_entregas_12_pegadas",
         {"n_transectos_julgaveis": d_e4v2["populacao"]["n_transectos_julgaveis"],
          "n_pegadas": unidade_pegada["n_pegadas"],
          "pares_com_celula_compartilhada":
              unidade_pegada["pares_com_celula_compartilhada"]},
         None, "e4v2",
         "populacao.{n_transectos_julgaveis,unidade_pegada.{n_pegadas,"
         "pares_com_celula_compartilhada}}",
         nota="13 entregas lidar = 12 pegadas: NP_T-0662 repete a pegada "
              "de NP_T-0357 (7.861 celulas compartilhadas)")

    import pandas as pd
    df_dump = pd.read_parquet(ARQS["e4v1dump"])
    if "cob_eth" not in df_dump.columns:
        raise RuntimeError("auditar_tabela3_dump.parquet sem coluna cob_eth")
    cob_min = float(df_dump["cob_eth"].min())
    cob_max = float(df_dump["cob_eth"].max())
    n_abaixo_0_8 = int((df_dump["cob_eth"] < 0.8).sum())
    fato("populacao", "jstarsb_pop_full_cover",
         {"cob_eth_min": cob_min, "cob_eth_max": cob_max,
          "n_linhas": int(len(df_dump)),
          "n_celulas_abaixo_de_0_8": n_abaixo_0_8,
          "filtro_cob_eth_maior_igual_0_8_e_vazio": n_abaixo_0_8 == 0},
         None, "e4v1dump", "coluna 'cob_eth' (computado em codigo: min, "
         "max, contagem abaixo de 0,8)",
         nota="'full cover': todas as 1.194 celulas-ancora tem "
              "cob_eth==1,0; o filtro cob_eth>=0,8 do E4 nao remove "
              "nenhuma celula (vazio por construcao, nao por acaso)")
    if not (cob_min == 1.0 and cob_max == 1.0 and n_abaixo_0_8 == 0):
        raise RuntimeError(
            f"G4: 'full cover' nao se sustenta -- cob_eth min={cob_min} "
            f"max={cob_max} n_abaixo_0.8={n_abaixo_0_8}")

    # ==================================================================
    # G5 -- reproduction of the three anchor values, fail-loud
    # ==================================================================
    anc_e4 = pc[">30m"]["por_unidade"]["transecto"]["A"]["media_pond_celula_m"]
    if round(anc_e4, 6) != round(6.651019, 6):
        raise RuntimeError(f"G5: ancora E4 nao reproduz: {anc_e4} != 6.651019")

    anc_delta_v2 = pc[">30m"]["por_unidade"]["transecto"]["delta"]["media_pond_celula_m"]
    anc_delta_d1 = pop_d1[">30m"]["media_pond_celula_m"]
    if round(anc_delta_v2, 4) != round(3.7654, 4):
        raise RuntimeError(f"G5: ancora delta (v2) nao reproduz: {anc_delta_v2} != 3.7654")
    if round(anc_delta_d1, 4) != round(3.7654, 4):
        raise RuntimeError(f"G5: ancora delta (D-1) nao reproduz: {anc_delta_d1} != 3.7654")
    if abs(anc_delta_v2 - anc_delta_d1) > 1e-6:
        raise RuntimeError(
            f"G5: delta do v2 e do D-1 divergem: {anc_delta_v2} vs {anc_delta_d1}")

    anc_fabdem = (d_d5["bracos"]["lambda_+0.00_sentido_fisico"]
                  ["teste_pareado_v23_vs_fabdem_mae"]["mediana_m"])
    if round(anc_fabdem, 4) != round(2.3259, 4):
        raise RuntimeError(f"G5: ancora v23xfabdem nao reproduz: {anc_fabdem} != 2.3259")

    log(f"G5 ok: E4={anc_e4:.6f} delta_nativo={anc_delta_v2:.4f} "
        f"v23xfabdem={anc_fabdem:.4f}")

    # ==================================================================
    n_fatos_b = sum(len(v) for v in grupos.values())
    saida = {
        "gerado_para": "Artigo 1 do V23 (alvo IEEE JSTARS, pacote_overleaf/main.tex)",
        "regra": ("todo numero citado no texto do paper tem linha nesta "
                  "folha; linha sem uso e permitida, numero sem linha NAO "
                  "EXISTE. Copiar da folha, conferir por diff contra o "
                  "artefato da linha."),
        "protocolo": (("Adjusted mapping: original T3 excluded, delta from the native D-1, band "
                       "of Fig. 7(a) excluded")),
        "parte_A": parte_a,
        "parte_A_fonte": {
            "arquivo": "folha_de_fatos_jstars.json",
            "sha256_conferido": sha_folha_a,
            "contra_auditoria": "_pareceres_2026-09-28_execucao/"
                                 "contra_auditoria_folha_A.md (CONFIRMADO)"},
        "parte_B": {
            "n_fatos": n_fatos_b,
            "grupos": {g: len(v) for g, v in grupos.items()},
            "fatos": grupos,
        },
    }
    fontes_lidas = list(ARQS.values()) + [ata_path, contra_d5_path]
    PROV.gravar(RAIZ / a.saida, saida, fontes_lidas=fontes_lidas, script=__file__)
    log(f"{len(parte_a['fatos'])} fatos em parte_A (copiados, identicos) + "
        f"{n_fatos_b} fatos em parte_B ({len(grupos)} grupos) gravados em "
        f"{a.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
