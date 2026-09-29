# -*- coding: utf-8 -*-
"""Build part C of the article's fact sheet (folha_de_fatos_jstars_v21.json).

Parts A and B are copied verbatim from folha_de_fatos_jstars_v2.json after its
sha256 is checked against a pinned value; no value is retyped. Part C reads
four groups, each fact tagged with artefact, field and sha256:
  C1  fidelity per ETH canopy-height class, with CIs over transect and
      footprint units (auditar_rotulo_regua_v2.json);
  C2  paired contrast v23 vs FABDEM/GEDTM30/MLP per ETH class, in MAE and
      NMAD: median, CI, p, Holm-adjusted p where defined, wins/ties/losses
      per transect, number of transects
      (auditar_nmad_pareado_nativa_anadem.json);
  C3  density of returns in the lower shell per canopy class
      (auditar_densidade_solo_nativa.json);
  C4  ANADEM vertical-datum check in canopy gaps: sanity guard
      (guarda_sanidade_anadem_vs_glo30.json) and an independent test
      (geo_produto_anadem.json), whose numbers are regex-extracted from text.

Checks (fail loudly): G1 unique fact ids across parts A-C; G2 no None/NaN
value at any nesting level; G3 every source recorded in _fontes with its
sha256 at read time; G4 derived quantities (losses = total - wins - ties;
guard vs independent test in C4) computed in code; G5 anchor value
v23 vs FABDEM MAE >30 m = +2.3259 m reproduced before writing.

Usage:  python folha_de_fatos_jstars_v21.py [--saida folha_de_fatos_jstars_v21.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import proveniencia as PROV          # noqa: E402

# Pinned sha256 of folha_de_fatos_jstars_v2.json (parts A and B).
SHA256_FOLHA_V2_ESPERADO = (
    "e32782c40b05d7b19cd7e7097097e06ea1fa1383fbb9581a73138ab4cbcfb80a")

ARQS = {
    "folha_v2": RAIZ / "folha_de_fatos_jstars_v2.json",
    "e4v2": RAIZ / "auditar_rotulo_regua_v2.json",
    "tab4": RAIZ / "auditar_nmad_pareado_nativa_anadem.json",
    "dens": RAIZ / "auditar_densidade_solo_nativa.json",
    "guarda_datum": (RAIZ / "artigo_v23_2" / "_pareceres_2026-09-28_execucao"
                      / "guarda_sanidade_anadem_vs_glo30.json"),
    "geo_datum": (RAIZ / "artigo_v23_2" / "_pareceres_2026-09-28_execucao"
                  / "geo_produto_anadem.json"),
}

CLASSES = ["0-3m", "3-10m", "10-20m", "20-30m", ">30m", "TODAS"]
CLASSE_ID = {"0-3m": "0_3m", "3-10m": "3_10m", "10-20m": "10_20m",
             "20-30m": "20_30m", ">30m": "30m_mais", "TODAS": "todas"}
CLASSES_PAREADO = ["0-3m", "3-10m", "10-20m", "20-30m", ">30m"]  # excludes TODAS (all classes)
PARES = {"fabdem": "v23 vs fabdem", "gedtm30": "v23 vs gedtm30",
         "mlp": "v23 vs mlp"}


def log(m=""):
    print(m, flush=True)


def sha256_arquivo(p: Path) -> str:
    return PROV.sha256(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default="folha_de_fatos_jstars_v21.json")
    a = ap.parse_args()

    for k, p in ARQS.items():
        if not p.exists():
            raise FileNotFoundError(f"fonte ausente: {k} -> {p}")

    # ------------------------------------------------------------------
    # G3 (part 1): verify the v2 sheet's sha256 before trusting any field.
    # ------------------------------------------------------------------
    sha_folha_v2 = sha256_arquivo(ARQS["folha_v2"])
    if sha_folha_v2 != SHA256_FOLHA_V2_ESPERADO:
        raise RuntimeError(
            (f"folha_de_fatos_jstars_v2.json changed since the independent check: current "
             f"sha256={sha_folha_v2} expected={SHA256_FOLHA_V2_ESPERADO}. Not proceeding "
             f"to copy parte_A/parte_B from a fact sheet that was not independently checked."))

    d_folha_v2 = json.loads(ARQS["folha_v2"].read_text(encoding="utf-8"))
    parte_a = d_folha_v2["parte_A"]
    parte_b = d_folha_v2["parte_B"]

    d_e4v2 = json.loads(ARQS["e4v2"].read_text(encoding="utf-8"))
    d_tab4 = json.loads(ARQS["tab4"].read_text(encoding="utf-8"))
    d_dens = json.loads(ARQS["dens"].read_text(encoding="utf-8"))
    d_guarda_datum = json.loads(ARQS["guarda_datum"].read_text(encoding="utf-8"))
    d_geo_datum = json.loads(ARQS["geo_datum"].read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Fact infrastructure (G1, G2, groups)
    # ------------------------------------------------------------------
    ids = set(f["id"] for f in parte_a["fatos"])            # G1: include part A ids
    for grupo_fatos in parte_b["fatos"].values():            # G1: include part B ids
        for f in grupo_fatos:
            if f["id"] in ids:
                raise RuntimeError(
                    f"G1: id de parte_B colide com parte_A: {f['id']}")
            ids.add(f["id"])
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
        """G2: a CI over <2 units is [null, null] in the artefact; return a
        descriptive marker instead of a raw None."""
        if ic is not None and ic[0] is not None and ic[1] is not None:
            return ic
        return {"disponivel": False,
                "motivo": f"n_unidades={n_unidades}: IC exige >=2 unidades; "
                          "artefato grava [null,null]"}

    def fato(grupo, id_, valor, unidade, chave_arq, campo, nota=None):
        if id_ in ids:                                       # G1
            raise RuntimeError(f"G1: id duplicado: {id_}")
        ids.add(id_)
        _sem_nulos(valor, id_)                                # G2
        arq = ARQS[chave_arq]
        linha = {"id": id_, "valor": valor, "unidade": unidade,
                  "artefato": str(arq.name), "campo": campo}
        if nota:
            linha["nota"] = nota
        grupos.setdefault(grupo, []).append(linha)
        return linha

    # ==================================================================
    # GROUP C1 -- fidelity per ETH class, with CIs over transect and
    # footprint units (auditar_rotulo_regua_v2.json)
    # ==================================================================
    pc = d_e4v2["por_classe_eth"]
    for c in CLASSES:
        cid = CLASSE_ID[c]
        bloco_c = pc[c]
        for unidade_nome in ("transecto", "pegada"):
            fid = bloco_c["por_unidade"][unidade_nome]["fidelidade"]
            fato("c1_fidelidade", f"jstarsc_fidelidade_{cid}_{unidade_nome}",
                 {"media_pond_celula_m": fid["media_pond_celula_m"],
                  "mediana_m": fid["mediana_m"],
                  "n_celulas": fid["n_celulas"],
                  "n_unidades": fid["n_unidades"],
                  "ic95_pond_celula": _ic_ou_indisponivel(
                      fid["ic95_pond_celula"], fid["n_unidades"]),
                  "ic95_mediana": _ic_ou_indisponivel(
                      fid["ic95_mediana"], fid["n_unidades"]),
                  "ic95_t_jackknife_pond": _ic_ou_indisponivel(
                      fid["ic95_t_jackknife_pond"], fid["n_unidades"])},
                 "m", "e4v2",
                 f"por_classe_eth.'{c}'.por_unidade.'{unidade_nome}'."
                 "fidelidade.{media_pond_celula_m,mediana_m,n_celulas,"
                 "n_unidades,ic95_pond_celula,ic95_mediana,"
                 "ic95_t_jackknife_pond}",
                 nota=f"fidelidade = |z_ref - solo_anc| ponderada por celula "
                      f"(definicao do proprio auditor v2), classe ETH '{c}', "
                      f"unidade '{unidade_nome}'; IC percentil, de mediana e "
                      "t-jackknife")

    # ==================================================================
    # GROUP C2 -- paired contrast v23 vs FABDEM/GEDTM30/MLP per ETH
    # class, in MAE and NMAD (auditar_nmad_pareado_nativa_anadem.json)
    # ==================================================================
    tb = d_tab4["populacao_original_ETH"]["testes_principal_ETH_produtos_base"]
    for par_id, par_chave in PARES.items():
        for metrica in ("mae", "nmad"):
            for c in CLASSES_PAREADO:
                cid = CLASSE_ID[c]
                r = tb[par_chave][metrica][c]
                if not r.get("testado", False):
                    valor = {"testado": False,
                             "motivo": r.get("motivo", "sem motivo no artefato"),
                             "n_transectos": r["n_transectos"]}
                else:
                    vitorias = r["vence"]
                    empates = r["empates"]
                    total = r["total"]
                    derrotas = total - vitorias - empates            # G4
                    if derrotas < 0:
                        raise RuntimeError(
                            f"G4: derrotas negativas em {par_chave}/{metrica}/"
                            f"{c}: vitorias={vitorias} empates={empates} "
                            f"total={total}")
                    valor = {"testado": True,
                             "mediana_m": r["mediana_m"],
                             "media_m": r["media_m"],
                             "ic95_bootstrap_transecto_mediana":
                                 r["ic95_bootstrap_transecto_mediana"],
                             "n_transectos": r["n_transectos"],
                             "vitorias": vitorias, "empates": empates,
                             "derrotas": derrotas, "total": total,
                             "p_sinal_binomial": r["p_sinal_binomial"],
                             "p_wilcoxon": r["p_wilcoxon"]}
                    if "p_holm" in r:
                        valor["p_holm"] = r["p_holm"]
                    else:
                        valor["p_holm"] = {
                            "disponivel": False,
                            "motivo": "par/classe fora da familia "
                                      "confirmatoria (criterio_pre_registrado."
                                      "familia_confirmatoria = so 'v23 vs "
                                      "fabdem' em 20-30m/>30m); demais pares/"
                                      "classes sao exploratorios, p bruto "
                                      "sem correcao Holm"}
                fato("c2_contraste_pareado",
                     f"jstarsc_contraste_{par_id}_{metrica}_{cid}", valor,
                     "m" if metrica == "mae" else None, "tab4",
                     "populacao_original_ETH.testes_principal_ETH_"
                     f"produtos_base.'{par_chave}'.{metrica}.'{c}'",
                     nota="contraste pareado por transecto; classes 0-3m/"
                          "3-10m sempre 'testado=false' (menos de 6 "
                          "transectos acima do piso de 200 celulas); "
                          "derrotas = total - vitorias - empates, calculado "
                          "em codigo (G4)")

    # ==================================================================
    # GROUP C3 -- density of returns in the lower shell per canopy
    # class (auditar_densidade_solo_nativa.json)
    # ==================================================================
    pf_eth = d_dens["por_faixa"]["ETH"]
    for c in CLASSES:
        cid = CLASSE_ID[c]
        r = pf_eth[c]
        fato("c3_densidade_casca", f"jstarsc_densidade_casca_{cid}",
             {"dens_n_casca": r["dens_n_casca"],
              "n_celulas": r["n_celulas"], "n_transectos": r["n_transectos"],
              "razao_casca_sobre_piso": r["razao_casca_sobre_piso"],
              "razao_das_medianas_casca_sobre_piso":
                  r["razao_das_medianas_casca_sobre_piso"]},
             "pt/m2", "dens",
             f"por_faixa.'ETH'.'{c}'.{{dens_n_casca,n_celulas,n_transectos,"
             "razao_casca_sobre_piso,razao_das_medianas_casca_sobre_piso}}",
             nota="dens_n_casca = densidade de retornos com z <= p2_all+1m "
                  "(populacao do estimador de chao), estratificacao "
                  "principal ETH; piso = 0,02*dens_n_all")

    fato("c3_densidade_definicoes", "jstarsc_densidade_definicoes",
         d_dens["definicoes"], None, "dens", "definicoes",
         nota="definicoes de area de celula, populacoes (n_all/n_cl2/"
              "n_casca), piso e limiares declarados do proprio auditor")

    fato("c3_densidade_contra_auditoria_anterior",
         "jstarsc_densidade_contra_auditoria_anterior",
         {"texto": d_dens["contra_auditoria"]}, None, "dens",
         "contra_auditoria",
         nota=("independent re-check already recorded in the artifact itself before this "
               "sheet; copied verbatim, without verifying the referenced file (outside "
               "the scope of this task)"))

    anc_dens_30m_mais = pf_eth[">30m"]["dens_n_casca"]["mediana"]
    fato("c3_densidade_texto_atual", "jstarsc_densidade_0_19_conferencia",
         {"mediana_dens_n_casca_30m_mais_pt_m2": anc_dens_30m_mais,
          "arredondado_2_casas": round(anc_dens_30m_mais, 2)},
         "pt/m2", "dens",
         "por_faixa.'ETH'.'>30m'.dens_n_casca.mediana (arredondado em codigo)",
         nota="conferencia do numero que o texto atual do artigo cita como "
              "'0,19 pontos/m2 acima de 30 m'")

    # ==================================================================
    # GROUP C4 -- ANADEM vertical-datum test in canopy gaps
    # (guarda_sanidade_anadem_vs_glo30.json + geo_produto_anadem.json)
    # ==================================================================
    clareira = d_guarda_datum["por_classe_dossel_eth"]["clareira (<2 m)"]
    fato("c4_datum_guarda", "jstarsc_datum_anadem_guarda_clareira",
         {"mediana_m": clareira["mediana_m"], "iqr_m": clareira["iqr_m"],
          "dp_m": clareira["dp_m"], "n": clareira["n"]},
         "m", "guarda_datum",
         "por_classe_dossel_eth.'clareira (<2 m)'.{mediana_m,iqr_m,dp_m,n}",
         nota="guarda de sanidade ANADEM vs GLO-30, classe de dossel ETH "
              "'clareira (<2 m)' (a menos afetada por penetracao de "
              "sinal); mediana ANADEM-GLO30 = 0,000 m")

    # The independent test's numbers exist only as free text
    # (achados id 'A3_datum_vertical'); extract them by regex.
    achado_a3 = next(a for a in d_geo_datum["achados"]
                      if a["id"] == "A3_datum_vertical")
    txt = achado_a3["descricao"]
    m_n = re.search(r"n=([\d.]+)\)", txt)
    m_med = re.search(r"mediana de ANADEM-GLO30 e ([+\-]?[\d,]+) m", txt)
    m_iqr = re.search(r"IQR ([\d,]+) m", txt)
    m_dp = re.search(r"dp ([\d,]+) m", txt)
    if not (m_n and m_med and m_iqr and m_dp):
        raise RuntimeError(
            "G4: nao consegui extrair os 4 numeros do teste independente "
            "do geo em achados[2].descricao (A3_datum_vertical) -- texto "
            "mudou de forma, conferir manualmente")
    n_geo = int(m_n.group(1).replace(".", ""))
    mediana_geo = float(m_med.group(1).replace(",", "."))
    iqr_geo = float(m_iqr.group(1).replace(",", "."))
    dp_geo = float(m_dp.group(1).replace(",", "."))

    diferenca_populacao_n = n_geo - clareira["n"]                    # G4
    fato("c4_datum_geo_independente",
         "jstarsc_datum_anadem_geo_independente",
         {"mediana_m": mediana_geo, "iqr_m": iqr_geo, "dp_m": dp_geo,
          "n": n_geo, "n_guarda_sanidade": clareira["n"],
          "diferenca_populacao_n": diferenca_populacao_n},
         "m", "geo_datum",
         "achados[id='A3_datum_vertical'].descricao (numeros extraidos por "
         "regex em codigo: 'n=...', 'mediana de ANADEM-GLO30 e ... m', "
         "'IQR ... m', 'dp ... m')",
         nota=("independent empirical test (run on the .npz files on the node grid, same "
               "clearing population ETH<2m); diferenca_populacao_n computed in code "
               "against the n of the sanity guard"))
    if diferenca_populacao_n != 0:
        log(f"AVISO C4: populacao do geo (n={n_geo}) difere da guarda "
            f"(n={clareira['n']}) em {diferenca_populacao_n} unidades -- "
            "registrado no fato, nao interpretado aqui")
    if round(mediana_geo, 3) != round(clareira["mediana_m"], 3):
        raise RuntimeError(
            f"G4: mediana do teste independente do geo ({mediana_geo}) nao "
            f"bate com a guarda ({clareira['mediana_m']}) dentro de 1e-3")

    # ==================================================================
    # G5 -- reproduce the anchor value, fail loudly
    # ==================================================================
    anc_fabdem = tb["v23 vs fabdem"]["mae"][">30m"]["mediana_m"]
    if round(anc_fabdem, 4) != round(2.3259, 4):
        raise RuntimeError(
            f"G5: ancora v23xfabdem nao reproduz: {anc_fabdem} != 2.3259")
    log(f"G5 ok: v23xfabdem_mae_30m_mais={anc_fabdem:.4f}")

    # ==================================================================
    n_fatos_c = sum(len(v) for v in grupos.values())
    saida = {
        "gerado_para": "Artigo 1 do V23 (alvo IEEE JSTARS, pacote_overleaf/main.tex)",
        "regra": ("todo numero citado no texto do paper tem linha nesta "
                  "folha; linha sem uso e permitida, numero sem linha NAO "
                  "EXISTE. Copiar da folha, conferir por diff contra o "
                  "artefato da linha."),
        "protocolo": (("Complements the V23 Article 1 fact sheet with missing facts (parte_C: "
                       "fidelity per ETH class, paired contrast v23 x FABDEM/GEDTM30/MLP in MAE "
                       "and NMAD, return density in the lower shell per canopy class, ANADEM "
                       "datum test in the clearings)")),
        "parte_A": parte_a,
        "parte_B": parte_b,
        "parte_A_parte_B_fonte": {
            "arquivo": "folha_de_fatos_jstars_v2.json",
            "sha256_conferido": sha_folha_v2,
            "contra_auditoria": "Independent check completed: CONFIRMED WITH RESERVATIONS"},
        "parte_C": {
            "n_fatos": n_fatos_c,
            "grupos": {g: len(v) for g, v in grupos.items()},
            "fatos": grupos,
        },
    }
    fontes_lidas = list(ARQS.values())
    PROV.gravar(RAIZ / a.saida, saida, fontes_lidas=fontes_lidas, script=__file__)
    log(f"{len(parte_a['fatos'])} fatos em parte_A + {parte_b['n_fatos']} "
        f"fatos em parte_B (copiados, identicos) + {n_fatos_c} fatos em "
        f"parte_C ({len(grupos)} grupos) gravados em {a.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
