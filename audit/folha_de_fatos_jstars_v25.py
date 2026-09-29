# -*- coding: utf-8 -*-
"""folha_de_fatos_jstars_v25 -- adds part G (S7: tree model vs. LiDAR) to the
fact sheet for Article 1 of V23 (target: IEEE JSTARS).

Does not edit `folha_de_fatos_jstars_v24.py` or `folha_de_fatos_jstars_v24.json`.
Parts A, B, A_B_source, C, D, E and F are read PROGRAMMATICALLY from
`folha_de_fatos_jstars_v24.json` (checked against its pinned sha256 before
copying) and reproduced IDENTICALLY, without exception. The generic helpers
(diff_trilhas, _sem_nulos, sel, igual) are imported from v24 (the .py sha256
is checked before import), not rewritten.

Part G -- every fact is read programmatically from
`auditar_s7_arvore_lidar.json` and from `treinar_arvore_persistir_v2.json`,
with id `jstarsg_*`, value, unit, artifact, field and provenance stamp. A
field computed here (never typed by hand) carries `derivado: true` and
`operacao`; whenever a recorded counterpart exists, the computation is
checked against it (guard G4).

Guards (hard failures):
  G1  unique id across the whole A-G namespace.
  G2  no None/NaN in `valor`.
  G3  every source is listed in `_fontes`; pinned sha256 hashes (v24
      .json/.py, S7, training v2) are checked before reading; sources listed
      in the stamped S7 artifact's `_fontes` must have the same sha256
      today; everything is re-checked at the end.
  G4  every reading is cross-checked against its recorded counterpart when
      one exists (g1' against training and f2; contrasts recounted from the
      per-unit differences; v23 x mlp against
      auditar_nmad_pareado_nativa_diretas.json; v23/mlp pool likewise; O3
      recomputed from the contrasts and from auditar_rotulo_regua_v2.json;
      network admissibility against auditar_admissibilidade.json; tree
      summary recomputed per seed; skill mean/sd recomputed).
  GI  parts A-F (and A_B_source) identical to v24 by equality, both in the
      assembled object and in the JSON re-read from disk (no divergent
      path).

Usage:  python folha_de_fatos_jstars_v25.py [--saida folha_de_fatos_jstars_v25.json]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import proveniencia as PROV          # noqa: E402

ART = RAIZ / "artigo_v23_2"
PAR4 = ART / "_pareceres_2026-09-28_rodada4"

ARQS = {
    "folha_v24": RAIZ / "folha_de_fatos_jstars_v24.json",
    "folha_v24_py": RAIZ / "folha_de_fatos_jstars_v24.py",
    "s7": RAIZ / "auditar_s7_arvore_lidar.json",
    "treino_v2": RAIZ / "treinar_arvore_persistir_v2.json",
    "treino_v1": RAIZ / "treinar_arvore_persistir.json",
    "f2_arvores": RAIZ / "results" / "f2_arvores.json",
    "nmad_par": RAIZ / "auditar_nmad_pareado_nativa_diretas.json",
    "regua_v2": RAIZ / "auditar_rotulo_regua_v2.json",
    "adm": RAIZ / "auditar_admissibilidade.json",
    "desenho": PAR4 / "desenho_S5_S7_S8.md",
    "contra_s7": PAR4 / "contra_auditoria_S7.md",
    "contra_folha_v24": PAR4 / "contra_auditoria_folha_v24.md",
}

# Pinned sha256 hashes (previously verified); training v2 = the sha256 that
# the stamped S7 artifact lists in _fontes
SHA_ESPERADO = {
    "folha_v24": "0c3d4c725d2d7de15feb4924096a3e9088af6cfcef4a50dbc6d43a9c405bdc1f",
    "folha_v24_py": "9d24b51bcb2ef2d71a2cd9445d5c0a8473d63fcfe45773af7e1c1576a45ddd37",
    "s7": "09d3030b23bfe6d908cd9bc4b9832418b4872b36aa641198a6faf6007c9abf57",
    "treino_v2": "5be410e2c01fd63a5969ce3af2e1c47efdb45fc2705f096c8083b25931911cd9",
}

SEMENTES = ["42", "123", "7", "2024", "31"]
CLASSES = ["20-30m", ">30m"]
UNIDADES = ["transecto", "pegada"]
PARES = ["tree vs v23", "tree vs mlp", "v23 vs mlp"]
METRICAS = ["mae", "nmad"]
CHAVE_REDES = ("redes_mesmo_ponto_b4s035 (auditar_admissibilidade.json, "
               "lote_adotado_epocas_x1)")

NOTA_G1 = (("per-seed tolerance relaxed after the failure of seed 31; deterministic "
            "retraining (difference 0.0)"))
NOTA_ADM = ("arvore no ponto de suavidade zero (contrato do braco), redes com "
            "lambda_TV = 0,35")
ORIENT = ("diferenca = metrica(a) - metrica(b) no par 'a vs b'; negativo = a com "
          "menor erro; 'vence' = k unidades com diferenca < 0 (a melhor), n = total")


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default="folha_de_fatos_jstars_v25.json")
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

    V24 = importar_v24()                      # generic helpers from v24 (READ-ONLY)
    diff_trilhas, _sem_nulos, sel, igual = (V24.diff_trilhas, V24._sem_nulos,
                                            V24.sel, V24.igual)

    J = {k: json.loads(p.read_text(encoding="utf-8"))
         for k, p in ARQS.items() if p.suffix == ".json"}
    TXT = {k: p.read_text(encoding="utf-8") for k, p in ARQS.items()
           if p.suffix == ".md"}

    # G3: everything the stamped S7 and training-v2 artifacts list in
    # _fontes has the same sha256 today (and every data source of this
    # sheet appears in one of the two)
    n_cruz = 0
    cruzadas = set()
    for ks in ("s7", "treino_v2"):
        fts = {f["caminho"]: f["sha256"] for f in J[ks]["_fontes"]}
        for k, p in ARQS.items():
            if rot(p) in fts:
                if fts[rot(p)] != sha0[k]:
                    raise RuntimeError(f"G3: {rot(p)} != sha256 listado em _fontes de {rot(ARQS[ks])}")
                n_cruz += 1
                cruzadas.add(k)
    for nome, sh in J["s7"]["conferencia_sha256"].items():
        k = next((kk for kk, p in ARQS.items() if rot(p) == nome), None)
        if k is not None and sha0[k] != sh:
            raise RuntimeError(f"G3: {nome} != conferencia_sha256 do S7")
    dados = {"treino_v2", "treino_v1", "f2_arvores", "nmad_par", "regua_v2", "adm"}
    if not dados <= cruzadas:
        raise RuntimeError(f"G3: fontes de dado sem cruzamento com os carimbados: {dados - cruzadas}")

    # verification records on disk, with hash and verdict (read, not assumed)
    for k, sha_alvo in (("contra_s7", SHA_ESPERADO["s7"]),
                        ("contra_folha_v24", SHA_ESPERADO["folha_v24"])):
        if sha_alvo[:8] not in TXT[k] or "CONFIRMADO COM RESSALVA" not in TXT[k]:
            raise RuntimeError(f"G3: {ARQS[k].name} nao traz o sha256 {sha_alvo[:8]}"
                               " e o veredito CONFIRMADO COM RESSALVA")
    if "S7 — retomada" not in TXT["desenho"] or "0,0092" not in TXT["desenho"]:
        raise RuntimeError("protocolo: desenho sem a secao 'S7 — retomada' / g1'")

    def carimbo_de(k):
        c = {"sha256": sha0[k]}
        if k in J and isinstance(J[k].get("_proveniencia"), dict):
            pv = J[k]["_proveniencia"]
            c["script_origem"] = pv.get("script") or "AUSENTE (null no _proveniencia)"
            s = pv.get("sha256_script") or pv.get("sha256_script_no_momento_do_carimbo")
            c["sha256_script"] = s or "AUSENTE"
            c["carimbado_em"] = pv.get("em") or "AUSENTE"
            c["retroativo"] = bool(pv.get("retroativo", False))
        elif k in J:
            c["script_origem"] = "AUSENTE (artefato sem _proveniencia)"
        else:
            c["script_origem"] = "nao_se_aplica (codigo/texto lido direto)"
        return c
    CARIMBO = {k: carimbo_de(k) for k in ARQS}

    # ------------------------------------------------------------------
    # Parts A-F copied from v24 (GI: identical, without exception)
    # ------------------------------------------------------------------
    v24 = J["folha_v24"]
    COPIADAS = ["parte_A", "parte_B", "parte_A_parte_B_fonte", "parte_C",
                "parte_D", "parte_E", "parte_F"]
    TOPO = {"gerado_para": "gerado_para", "regra": "regra",
            "protocolo_v24": "protocolo", "protocolo_v23": "protocolo_v23",
            "protocolo_v22": "protocolo_v22", "guardas_v24": "guardas",
            "guardas_v23": "guardas_v23"}
    faltam = set(v24) - set(COPIADAS) - set(TOPO.values()) - {"_proveniencia", "_fontes"}
    if faltam:
        raise RuntimeError(f"GI: chaves de topo da v24 sem destino na v25: {faltam}")
    copia = {k: json.loads(json.dumps(v24[k])) for k in COPIADAS}

    def conferir_GI(obj, rotulo):
        for k in COPIADAS:
            d = diff_trilhas(v24[k], obj[k])
            if d:
                raise RuntimeError(f"GI ({rotulo}): {k} difere da v24 em {d[:5]}")
        for novo, velho in TOPO.items():
            if diff_trilhas(v24[velho], obj[novo]):
                raise RuntimeError(f"GI ({rotulo}): {novo} != v24.{velho}")
    conferir_GI(copia | {n: v24[v] for n, v in TOPO.items()}, "objeto montado")

    ids = set()
    for f in copia["parte_A"]["fatos"]:
        if f["id"] in ids:
            raise RuntimeError(f"G1: id repetido na v24: {f['id']}")
        ids.add(f["id"])
    for parte in ("parte_B", "parte_C", "parte_D", "parte_E", "parte_F"):
        for gf in copia[parte]["fatos"].values():
            for f in gf:
                if f["id"] in ids:
                    raise RuntimeError(f"G1: id repetido na v24: {f['id']}")
                ids.add(f["id"])
    n_ids_v24 = len(ids)
    if n_ids_v24 != v24["guardas"]["G1_ids_unicos"]["n_ids_total"]:
        raise RuntimeError("G1: contagem de ids da v24 != guarda gravada")

    grupos: dict[str, list] = {}

    def fato(grupo, id_, valor, unidade, chaves, campo, nota=None,
             operacao=None, extra=None):
        if not id_.startswith("jstarsg_"):
            raise RuntimeError(f"id fora do prefixo jstarsg_: {id_}")
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

    s7, tv2, tv1, f2 = J["s7"], J["treino_v2"], J["treino_v1"], J["f2_arvores"]

    # ==================================================================
    # G1 -- guards g1' (mean of the five and per-seed) + original g1 and
    # retraining
    # ==================================================================
    ga = tv2["g1_linha_a"]
    if s7["g1_linha_treino"] != ga:
        raise RuntimeError("G4: auditar_s7.g1_linha_treino != treinar_v2.g1_linha_a")
    if tv2["sementes"] != [int(s) for s in SEMENTES] or sorted(f2["por_semente"]) != sorted(SEMENTES):
        raise RuntimeError("G4: sementes do treino v2 / f2 != 42 123 7 2024 31")
    sk5 = np.array([tv2["g1_linha"][s]["skill_reserva"] for s in SEMENTES], float)
    skf2 = np.array([f2["por_semente"][s]["reserva"]["skill_media_ponderada_por_n"]
                     for s in SEMENTES], float)
    igual(float(sk5.mean()), ga["skill_media_5"], 1e-15, "g1'(a) skill_media_5")
    igual(float(sk5.std(ddof=1)), ga["skill_dp_5"], 1e-15, "g1'(a) skill_dp_5 (ddof=1)")
    igual(float(skf2.mean()), ga["f2_skill_media"], 1e-15, "g1'(a) f2_skill_media")
    igual(float(skf2.mean()), f2["resumo"]["skill_reserva_media"], 1e-15, "f2 resumo media")
    igual(float(skf2.std(ddof=1)), f2["resumo"]["skill_reserva_dp"], 1e-15, "f2 resumo dp (ddof=1)")
    igual(float(skf2.std(ddof=1)), ga["f2_skill_dp"], 1e-15, "g1'(a) f2_skill_dp")
    igual(ga["skill_media_5"] - ga["f2_skill_media"], ga["dif_media"], 1e-15, "g1'(a) dif_media")
    if (abs(ga["dif_media"]) <= ga["tolerancia_a"]) != ga["passou_a"]:
        raise RuntimeError("G4: g1'(a) passou_a != |dif_media| <= tolerancia_a")
    # tolerances per the analysis protocol (0.002; 2 x f2 sd rounded to 4 places)
    igual(ga["tolerancia_a"], 0.002, 0.0, "tolerancia_a vs desenho (0,002)")
    igual(ga["tolerancia_b"], round(2 * round(ga["f2_skill_dp"], 4), 4), 1e-15,
          "tolerancia_b vs 2 x round(f2_dp, 4) (desenho: 2 x 0,0046 = 0,0092)")
    por_sem = {}
    for i, s in enumerate(SEMENTES):
        g = tv2["g1_linha"][s]
        igual(g["skill_reserva_f2"], skf2[i], 1e-15, f"g1' {s} skill_reserva_f2 vs f2")
        igual(g["skill_reserva"] - g["skill_reserva_f2"], g["dif_skill_reserva"], 1e-15,
              f"g1' {s} dif")
        if (abs(g["dif_skill_reserva"]) <= g["tolerancia_b"]) != g["passou_b"]:
            raise RuntimeError(f"G4: g1'(b) {s} passou_b")
        if g["tolerancia_b"] != ga["tolerancia_b"]:
            raise RuntimeError(f"G4: tolerancia_b {s} != g1_linha_a")
        por_sem[s] = {"origem": g["origem"], **sel(g, [
            "skill_reserva", "skill_reserva_f2", "dif_skill_reserva", "n_reserva",
            "arvores_finais", "tolerancia_b", "passou_b"])}
    maxd = float(np.max(np.abs(sk5 - skf2)))
    igual(maxd, ga["max_abs_dif_por_semente"], 1e-15, "g1'(b) max_abs_dif_por_semente")
    if all(v["passou_b"] for v in por_sem.values()) != ga["passou_b_todas"]:
        raise RuntimeError("G4: g1'(b) passou_b_todas")
    ext_g1 = {"no_protocolo": "desenho_S5_S7_S8.md, secao 'S7 — retomada': g1' (a) media "
                              "das cinco dentro de 0,002; (b) cada semente dentro de "
                              "2 x 0,0046 = 0,0092"}
    fato("s7_guarda_g1linha", "jstarsg_s7_g1linha_media5",
         {k: ga[k] for k in ("skill_media_5", "skill_dp_5", "f2_skill_media",
                             "f2_skill_dp", "dif_media", "tolerancia_a", "passou_a")},
         "skill (adim.)", ["treino_v2", "s7", "f2_arvores"],
         "treinar_arvore_persistir_v2.json g1_linha_a.{skill_media_5,skill_dp_5,"
         "f2_skill_media,f2_skill_dp,dif_media,tolerancia_a,passou_a} (== "
         "auditar_s7_arvore_lidar.json g1_linha_treino)",
         extra=ext_g1,
         nota=NOTA_G1 + (". G4: means and sd (ddof=1) recomputed from the 5 seeds of the v2 "
                         "training and from results/f2_arvores.json "
                         "por_semente[s]['reserva']['skill_media_ponderada_por_n'] to 1e-15; dif "
                         "and passou_a recomputed. The fields treinar_arvore_persistir_v2.json "
                         "'desenho' and auditar_s7_arvore_lidar.json diferencas_declaradas[1] "
                         "attribute the recalibration before the resumption to a process decision; "
                         "this sheet does NOT copy that attribution."))
    fato("s7_guarda_g1linha", "jstarsg_s7_g1linha_por_semente",
         {"por_semente": por_sem, "max_abs_dif_por_semente": ga["max_abs_dif_por_semente"],
          "tolerancia_b": ga["tolerancia_b"], "passou_b_todas": ga["passou_b_todas"]},
         "skill (adim.)", ["treino_v2", "f2_arvores"],
         "treinar_arvore_persistir_v2.json g1_linha[s].{origem,skill_reserva,"
         "skill_reserva_f2,dif_skill_reserva,n_reserva,arvores_finais,tolerancia_b,"
         "passou_b}; g1_linha_a.{max_abs_dif_por_semente,tolerancia_b,passou_b_todas}",
         extra=ext_g1,
         nota=NOTA_G1 + ". 42/123/7/2024 = boosters treinados em 28/09 17:53-18:02 e "
              "reaproveitados (reuso conferido no manifesto, max|dif| 0,0); 31 = "
              "retreinada na retomada. G4: skill_reserva_f2 == f2_arvores.json, dif, "
              "passou_b e o maximo recomputados.")
    # original g1 (failed) and determinism of the seed-31 retraining
    g31v1, g31v2 = tv1["g1"]["31"], tv2["g1_linha"]["31"]
    if tv1["status"] != "FALHOU_G1" or str(tv1["semente_que_falhou"]) != "31":
        raise RuntimeError("G4: treinar_arvore_persistir.json nao registra FALHOU_G1 na 31")
    if tv2["manifesto_da_parada"]["sha256"] != sha0["treino_v1"]:
        raise RuntimeError("G4: manifesto_da_parada.sha256 != sha256 de treinar_arvore_persistir.json")
    for s in SEMENTES:
        if tv1["g1"][s]["skill_reserva"] != tv2["g1_linha"][s]["skill_reserva"] or \
           tv1["g1"][s]["arvores_finais"] != tv2["g1_linha"][s]["arvores_finais"]:
            raise RuntimeError(f"G4: semente {s}: manifesto 18h04 != v2 (skill/arvores)")
    ret = g31v2["descritivo_vs_retreino_18h04"]
    igual(ret["skill_reserva_18h04"], g31v1["skill_reserva"], 0.0, "retreino 31 vs 18h04")
    igual(ret["dif"], g31v2["skill_reserva"] - g31v1["skill_reserva"], 0.0, "retreino 31 dif")
    falhas_v1 = [s for s in SEMENTES if not tv1["g1"][s]["passou"]]
    fato("s7_guarda_g1linha", "jstarsg_s7_g1_original_e_retreino31",
         {"g1_original": {"status": tv1["status"],
                          "tolerancia_por_semente": g31v1["tolerancia"],
                          "semente_que_falhou": tv1["semente_que_falhou"],
                          "sementes_reprovadas": falhas_v1,
                          "dif_skill_reserva_31": g31v1["dif_skill_reserva"],
                          "carimbado_em": tv1["_proveniencia"]["em"]},
          "retreino_31": {"skill_reserva": g31v2["skill_reserva"],
                          "skill_reserva_18h04": ret["skill_reserva_18h04"],
                          "dif": ret["dif"], "arvores": g31v2["arvores_finais"],
                          "arvores_18h04": ret["arvores_18h04"]}},
         "skill (adim.)", ["treino_v1", "treino_v2"],
         "treinar_arvore_persistir.json {status,semente_que_falhou,g1['31'].{tolerancia,"
         "dif_skill_reserva,passou}}; treinar_arvore_persistir_v2.json g1_linha['31']."
         "descritivo_vs_retreino_18h04",
         extra=ext_g1,
         nota=NOTA_G1 + (". The resumption drew nothing new: the same booster (skill and number of "
                         "trees identical to those of the 18:04 manifest, checked across the 5 "
                         "seeds) passed a wider guard, chosen with knowledge of the failure."))

    # ==================================================================
    # G2 -- contrasts against LiDAR (3 pairs x MAE/NMAD x 2 classes x 2 units)
    # ==================================================================
    CL = s7["contrastes_lidar"]
    tp_ref = J["nmad_par"]["testes_principal_ETH"]["v23 vs mlp"]
    n_wil = 0
    for par in PARES:
        for met in METRICAS:
            val, n_conf = {}, 0
            for c in CLASSES:
                val[c] = {}
                for u in UNIDADES:
                    no = CL[u][par][met][c]
                    if not no["testado"]:
                        raise RuntimeError(f"contraste nao testado: {u}/{par}/{met}/{c}")
                    d = np.array(list(no["diferencas_por_transecto"].values()), float)
                    if sorted(no["diferencas_por_transecto"]) != sorted(no["transectos"]) \
                            or len(d) != no["n_transectos"]:
                        raise RuntimeError(f"G4: unidades {u}/{par}/{met}/{c}")
                    k_neg = int((d < 0).sum())
                    emp = int((d == 0).sum())
                    igual(float(np.median(d)), no["mediana_m"], 1e-12, f"mediana {u}/{par}/{met}/{c}")
                    igual(float(d.mean()), no["media_m"], 1e-12, f"media {u}/{par}/{met}/{c}")
                    if k_neg != no["vence"] or emp != no["empates"] or \
                            len(d) - emp != no["total"]:
                        raise RuntimeError(f"G4: vence/empates/total {u}/{par}/{met}/{c}")
                    pb = float(stats.binomtest(no["vence"], no["total"], 0.5).pvalue)
                    igual(pb, no["p_sinal_binomial"], 1e-12, f"p_sinal {u}/{par}/{met}/{c}")
                    pw = float(stats.wilcoxon(d).pvalue)
                    igual(pw, no["p_wilcoxon"], 1e-12, f"p_wilcoxon {u}/{par}/{met}/{c}")
                    n_wil += 1
                    lo, hi = no["ic95_bootstrap_transecto_mediana"]
                    if not lo <= no["mediana_m"] <= hi:
                        raise RuntimeError(f"G4: mediana fora do IC {u}/{par}/{met}/{c}")
                    if par == "v23 vs mlp" and u == "transecto":   # source of guard g2
                        r = tp_ref[met][c]
                        for ch in ("mediana_m", "vence", "total", "p_sinal_binomial",
                                   "p_wilcoxon", "n_transectos"):
                            igual(no[ch], r[ch], 1e-9, f"v23 vs mlp {met}/{c} {ch} vs nmad_par")
                        for j in (0, 1):
                            igual(no["ic95_bootstrap_transecto_mediana"][j],
                                  r["ic95_bootstrap_transecto_mediana"][j], 1e-9,
                                  f"v23 vs mlp {met}/{c} ic95 vs nmad_par")
                        n_conf += 1
                    val[c][u] = {
                        "mediana_m": no["mediana_m"],
                        "ic95_bootstrap_mediana_m": [lo, hi],
                        "n_unidades": no["n_transectos"],
                        "unidades_excluidas_abaixo_do_piso": no["excluidos_abaixo_do_piso"],
                        "k_vence": no["vence"], "n_total": no["total"],
                        "k_n": f"{no['vence']}/{no['total']}",
                        "p_sinal_binomial": no["p_sinal_binomial"],
                        "p_wilcoxon": no["p_wilcoxon"],
                        "media_m": no["media_m"],
                        "ic95_exclui_zero": bool(lo > 0 or hi < 0),
                    }
            a_, b_ = par.split(" vs ")
            nota_c = ((f"{ORIENT}. Pair '{par}': negative = {a_} with lower {met.upper()} than {b_}. "
                       f"Bootstrap B = 10,000, seed 42, clustered by unit; transect and footprint "
                       f"use the SAME resampling stream (seed indexed by pair/metric/band, not by "
                       f"unit): the two CIs are not independent resamplings."))
            if par == "v23 vs mlp":
                nota_c += (f" G4: no transecto == auditar_nmad_pareado_nativa_diretas.json "
                           f"testes_principal_ETH['v23 vs mlp'].{met} (fonte da guarda g2) "
                           f"a 1e-9 ({n_conf} nos).")
            if met == "nmad" and par.startswith("tree"):
                piores = [f"{c}/{u}" for c in CLASSES for u in UNIDADES
                          if val[c][u]["ic95_exclui_zero"] and val[c][u]["mediana_m"] > 0]
                nota_c += ((f" Unfavorable to the tree (read, not interpreted): tree NMAD LARGER than that "
                            f"of {b_} with CI95 excluding zero in: {', '.join(piores) if piores else 'nenhuma celula'}."))
            fato("s7_contrastes_lidar",
                 f"jstarsg_lidar_{a_}_{b_}_{met}", val, "m (diferenca de metrica por unidade)",
                 ["s7"] + (["nmad_par"] if par == "v23 vs mlp" else []),
                 f"contrastes_lidar.{{transecto,pegada}}['{par}'].{met}.{{20-30m,>30m}}."
                 "{mediana_m,ic95_bootstrap_transecto_mediana,n_transectos,"
                 "excluidos_abaixo_do_piso,vence,total,p_sinal_binomial,p_wilcoxon,media_m}",
                 operacao="derivados SO os campos k_n (texto 'vence/total') e "
                          "ic95_exclui_zero (lo > 0 ou hi < 0); o resto e lido. G4: "
                          "mediana, media, vence, empates, total e p do sinal (binomial "
                          "bilateral) e p de Wilcoxon (scipy.stats.wilcoxon, padrao) "
                          "recontados de diferencas_por_transecto a 1e-12; mediana "
                          "dentro do IC.",
                 nota=nota_c)
    if n_wil != len(PARES) * len(METRICAS) * len(CLASSES) * len(UNIDADES):
        raise RuntimeError(f"G4: esperava 24 nos de contraste, conferi {n_wil}")

    # ==================================================================
    # G3 -- aggregate error (ETH pool) per family, >30 m and 20-30 m
    # ==================================================================
    pool = s7["tabela_pool_ETH_v23_mlp_tree"]
    pref = J["nmad_par"]["tabela_pool_ETH"]
    for fam in ("v23", "mlp"):
        for c in CLASSES:
            for ch in ("n", "mae", "nmad", "vies", "rmse"):
                igual(pool[fam][c][ch], pref[fam][c][ch], 1e-9, f"pool {fam}/{c}/{ch} vs nmad_par")
    for c in CLASSES:
        if len({pool[f][c]["n"] for f in ("v23", "mlp", "tree")}) != 1:
            raise RuntimeError(f"G4: pool {c}: n difere entre familias (populacao nao pareada)")
    for c, idc in ((">30m", "acima_30m"), ("20-30m", "20_30m")):
        fato("s7_pool_eth", f"jstarsg_pool_eth_{idc}",
             {"n_celulas": pool["tree"][c]["n"],
              **{fam: {"mae_m": pool[fam][c]["mae"], "nmad_m": pool[fam][c]["nmad"]}
                 for fam in ("tree", "v23", "mlp")}},
             "m", ["s7", "nmad_par"],
             f"tabela_pool_ETH_v23_mlp_tree.{{tree,v23,mlp}}['{c}'].{{n,mae,nmad}}",
             nota="erro agregado celula a celula contra o lidar, classe ETH exogena "
                  f"{c}, mesma populacao nas tres familias (n igual, conferido). G4: "
                  "v23 e mlp == auditar_nmad_pareado_nativa_diretas.json tabela_pool_ETH "
                  "a 1e-9 (n, mae, nmad, vies, rmse). tree = glo30 - media das 5 "
                  "sementes de Delta (produto_tree do S7).")

    # ==================================================================
    # G4 -- O3 per class (ceiling, pair, A limits, verdict)
    # ==================================================================
    O3 = s7["O3"]
    regua = J["regua_v2"]["por_classe_eth"]
    for c, idc in ((">30m", "acima_30m"), ("20-30m", "20_30m")):
        oc = O3[c]
        for ch in ("veredito_O3", "falha_ocorreria_ja_so_com_v23_x_mlp"):
            if ch not in oc:
                raise RuntimeError(f"O3 {c}: campo {ch} ausente (copiar os dois juntos)")
        val = {"por_unidade": {}}
        passa_u, so_v23 = {}, {}
        for u in UNIDADES:
            nu = oc["por_unidade"][u]
            if nu["pares_nao_testados"]:
                raise RuntimeError(f"O3 {c}/{u}: pares nao testados {nu['pares_nao_testados']}")
            for par in PARES:
                lo, hi = CL[u][par]["mae"][c]["ic95_bootstrap_transecto_mediana"]
                igual(nu["tetos_por_par_m"][par], max(abs(lo), abs(hi)), 0.0,
                      f"O3 {c}/{u} teto {par} vs contraste MAE")
            par_max = max(nu["tetos_por_par_m"], key=nu["tetos_por_par_m"].get)
            igual(nu["teto_O3_m"], nu["tetos_por_par_m"][par_max], 0.0, f"O3 {c}/{u} teto")
            if par_max != nu["par_que_fixa_o_teto"]:
                raise RuntimeError(f"G4: O3 {c}/{u} par que fixa o teto")
            ru = regua[c]["por_unidade"][u]
            lims = {"lim_inf_A_media_m": ru["A"]["ic95_pond_celula"][0],
                    "lim_inf_A_mediana_m": ru["A"]["ic95_mediana"][0],
                    "lim_inf_reducao_mae_m": ru["reducao_mae_equivalente"]["ic95_pond_celula"][0]}
            for ch, v in lims.items():
                igual(nu[ch], v, 0.0, f"O3 {c}/{u} {ch} vs auditar_rotulo_regua_v2")
            for ch, lch in (("acima_do_teto_media", "lim_inf_A_media_m"),
                            ("acima_do_teto_mediana", "lim_inf_A_mediana_m"),
                            ("acima_do_teto_reducao", "lim_inf_reducao_mae_m")):
                if (nu[lch] > nu["teto_O3_m"]) != nu[ch]:
                    raise RuntimeError(f"G4: O3 {c}/{u} {ch}")
            p_ = all(nu[lch] > nu["teto_O3_m"] for lch in lims)
            if p_ != nu["passa_na_unidade"]:
                raise RuntimeError(f"G4: O3 {c}/{u} passa_na_unidade")
            sv = all(nu[lch] > nu["tetos_por_par_m"]["v23 vs mlp"] for lch in lims)
            if sv != nu["passaria_so_com_v23_x_mlp_nesta_unidade"]:
                raise RuntimeError(f"G4: O3 {c}/{u} passaria_so_com_v23_x_mlp")
            passa_u[u], so_v23[u] = p_, sv
            folga = min(nu[lch] for lch in lims) - nu["teto_O3_m"]
            val["por_unidade"][u] = {
                **sel(nu, ["teto_O3_m", "par_que_fixa_o_teto", "tetos_por_par_m",
                           "lim_inf_A_media_m", "lim_inf_A_mediana_m",
                           "lim_inf_reducao_mae_m", "acima_do_teto_media",
                           "acima_do_teto_mediana", "acima_do_teto_reducao",
                           "passa_na_unidade",
                           "passaria_so_com_v23_x_mlp_nesta_unidade"]),
                "folga_min_m": folga}
        # v23 x mlp ceiling == recorded guard g2 (1.093 / 0.356)
        for u in UNIDADES:
            igual(oc["por_unidade"][u]["tetos_por_par_m"]["v23 vs mlp"],
                  s7["guardas"]["g2_ii"][c]["carimbado_m"], 1e-6, f"O3 {c}/{u} teto v23xmlp vs g2")
        ver = ("demonstrado para as tres familias" if all(passa_u.values())
               else "nao demonstrado contra a arvore")
        if ver != oc["veredito_O3"]:
            raise RuntimeError(f"G4: O3 {c} veredito recomputado '{ver}' != '{oc['veredito_O3']}'")
        falha_v23 = not all(so_v23.values())
        if falha_v23 != oc["falha_ocorreria_ja_so_com_v23_x_mlp"]:
            raise RuntimeError(f"G4: O3 {c} falha_ocorreria_ja_so_com_v23_x_mlp")
        val["veredito_O3"] = oc["veredito_O3"]
        val["falha_ocorreria_ja_so_com_v23_x_mlp"] = oc["falha_ocorreria_ja_so_com_v23_x_mlp"]
        nota_o = ("criterio O3 (desenho_S5_S7_S8.md, S7): teto = maior limite superior de "
                  "|dMAE| (max(|lo|,|hi|) do IC95 bootstrap da mediana) entre os tres pares "
                  "de familias, na classe e na unidade; demonstrado se os tres limites "
                  "inferiores (A media, A mediana, reducao de MAE equivalente; "
                  "auditar_rotulo_regua_v2.json) ficam acima do teto nas duas unidades. "
                  "G4: tetos recomputados dos contrastes MAE, limites == regua_v2, flags, "
                  "passa_na_unidade, veredito e falha_ocorreria_ja_so_com_v23_x_mlp "
                  "recomputados; teto v23 x mlp == guarda g2 carimbada.")
        if c == "20-30m":
            nota_o += (" READ TOGETHER veredito_O3 and falha_ocorreria_ja_so_com_v23_x_mlp: the label 'nao demonstrado contra a arvore' attributes to the tree a failure that already occurs with v23 x mlp alone (MAE reduction below the v23 x mlp ceiling in both units). The protocol defines the verdict only above 30 m.")
        fato("s7_O3", f"jstarsg_o3_{idc}", val, "m", ["s7", "regua_v2"],
             f"O3['{c}'].{{por_unidade.{{transecto,pegada}}.{{teto_O3_m,par_que_fixa_o_teto,"
             "tetos_por_par_m,lim_inf_*,acima_do_teto_*,passa_na_unidade,passaria_so_com_"
             "v23_x_mlp_nesta_unidade}},veredito_O3,falha_ocorreria_ja_so_com_v23_x_mlp}",
             operacao="derivado SO folga_min_m = min(tres limites inferiores) - teto_O3_m, "
                      "por unidade; o resto e lido.",
             nota=nota_o)

    # ==================================================================
    # G5 -- admissibility (ground > 1 m above the canopy)
    # ==================================================================
    ADM = s7["admissibilidade"]
    if ADM["nota"] != "as redes sao b4s035 (TV 0,35); a arvore e suavidade 0 (contrato do braco)" \
            or tv2["suavidade"] != 0.0:
        raise RuntimeError("G4: nota de suavidade do S7 / suavidade do treino mudou")
    aps, ars = ADM["arvore_por_semente"], ADM["arvore_resumo"]
    if sorted(aps) != sorted(SEMENTES):
        raise RuntimeError("G4: admissibilidade: sementes da arvore")
    v1m = np.array([aps[s]["violacoes_dossel_acima_1m"] for s in SEMENTES], float)
    igual(float(v1m.mean()), ars["violacoes_acima_1m_media"], 1e-9, "adm arvore media")
    igual(float(v1m.min()), ars["violacoes_acima_1m_min"], 0.0, "adm arvore min")
    igual(float(v1m.max()), ars["violacoes_acima_1m_max"], 0.0, "adm arvore max")
    igual(max(aps[s]["violacao_max_m"] for s in SEMENTES), ars["violacao_max_m_pior"], 0.0,
          "adm arvore pior violacao")
    if int((v1m == 0).sum()) != ars["sementes_com_zero_violacoes"]:
        raise RuntimeError("G4: adm arvore sementes_com_zero_violacoes")
    redes = ADM[CHAVE_REDES]
    for fam in ("gatv2", "mlp"):
        if redes[fam] != J["adm"]["tabela"]["lote_adotado_epocas_x1"][fam]:
            raise RuntimeError(f"G4: redes {fam} != auditar_admissibilidade.json lote_adotado_epocas_x1")
    CH_R = ["n_sementes", "sementes", "violacoes_acima_1m_min", "violacoes_acima_1m_media",
            "violacoes_acima_1m_max", "violacao_max_m_pior", "sementes_com_zero_violacoes"]
    fato("s7_admissibilidade", "jstarsg_adm_arvore_por_semente",
         {s: sel(aps[s], ["violacoes_dossel_acima_1m", "violacoes_dossel_acima_10cm",
                          "violacao_max_m", "quadrantes_com_medicao"]) for s in SEMENTES},
         "celulas / m", "s7",
         "admissibilidade.arvore_por_semente[s].{violacoes_dossel_acima_1m,"
         "violacoes_dossel_acima_10cm,violacao_max_m,quadrantes_com_medicao}",
         nota=NOTA_ADM + (". Dense-canopy cells with ground > 1 m above the canopy (Delta < -1 m), "
                          "summed over the 4 quadrants. Count reconstructed from a fraction stored "
                          "to 5 decimal places (round(n x frac)), the same used for the networks: "
                          "seed 31 = 1,065,411 here vs 1,065,414 in the direct count of the "
                          "independent check (immaterial)."))
    fato("s7_admissibilidade", "jstarsg_adm_arvore_resumo",
         sel(ars, CH_R), "celulas / m", "s7",
         "admissibilidade.arvore_resumo.{" + ",".join(CH_R) + "}",
         nota=NOTA_ADM + ". G4: min, media, max, pior violacao e sementes com zero "
              "recomputados de arvore_por_semente.")
    fato("s7_admissibilidade", "jstarsg_adm_redes_b4s035",
         {fam: sel(redes[fam], CH_R) for fam in ("gatv2", "mlp")},
         "celulas / m", ["s7", "adm"],
         f"admissibilidade['{CHAVE_REDES}'].{{gatv2,mlp}}.{{" + ",".join(CH_R) + "}",
         nota=NOTA_ADM + (". The network values used in the comparison; G4: == "
                          "auditar_admissibilidade.json tabela.lote_adotado_epocas_x1.{gatv2,mlp} "
                          "(equality of the whole block). The comparison mixes model family and "
                          "regularization."))

    # ==================================================================
    # G6 -- holdout skill: 5 retrained seeds and f2
    # ==================================================================
    rr = tv2["resumo_reserva"]
    igual(rr["skill_media"], float(sk5.mean()), 1e-15, "resumo_reserva skill_media")
    igual(rr["skill_dp"], float(sk5.std(ddof=1)), 1e-15, "resumo_reserva skill_dp")
    igual(rr["f2_skill_media"], float(skf2.mean()), 1e-15, "resumo_reserva f2 media")
    igual(rr["f2_skill_dp"], float(skf2.std(ddof=1)), 1e-15, "resumo_reserva f2 dp")
    for i, s in enumerate(SEMENTES):
        igual(tv2["g1b"][s]["skill_reserva_da_grade"], sk5[i], 0.0, f"g1b {s} skill da grade")
    fato("s7_skill_reserva", "jstarsg_arvore_skill_reserva_s7",
         {"por_semente": {s: float(sk5[i]) for i, s in enumerate(SEMENTES)},
          "media": rr["skill_media"], "dp": rr["skill_dp"], "n_sementes": len(SEMENTES),
          "n_reserva": tv2["g1_linha"]["31"]["n_reserva"]},
         "skill (adim.)", "treino_v2",
         "treinar_arvore_persistir_v2.json g1_linha[s].skill_reserva; "
         "resumo_reserva.{skill_media,skill_dp}",
         nota="skill da uniao das reservas ponderada por n de ancoras, arvore (boost 4, "
              "suavidade 0) das predicoes persistidas usadas no S7; dp amostral (ddof=1). "
              "G4: media/dp recomputados; == skill_reserva_da_grade (g1b) nas 5. " + NOTA_G1 + ".")
    fato("s7_skill_reserva", "jstarsg_arvore_skill_reserva_f2",
         {"por_semente": {s: float(skf2[i]) for i, s in enumerate(SEMENTES)},
          "media": f2["resumo"]["skill_reserva_media"], "dp": f2["resumo"]["skill_reserva_dp"],
          "n_sementes": f2["resumo"]["n_sementes"]},
         "skill (adim.)", ["f2_arvores", "treino_v2"],
         "results/f2_arvores.json por_semente[s].reserva.skill_media_ponderada_por_n; "
         "resumo.{skill_reserva_media,skill_reserva_dp,n_sementes}",
         nota="skill de reserva carimbada do braco de arvores (Tabela II); dp amostral "
              "(ddof=1). G4: == resumo_reserva.f2_* do treino v2 e recomputado a 1e-15. "
              "results/f2_arvores.json tem carimbo retroativo, sem sha256 do script.")

    # ==================================================================
    # source quality
    # ==================================================================
    qual = {}
    for k in ARQS:
        if k in ("folha_v24", "folha_v24_py"):
            continue
        c = CARIMBO[k]
        qual[rot(ARQS[k])] = {"sha256": c["sha256"], "script_origem": c.get("script_origem"),
                              "sha256_script": c.get("sha256_script", "nao_se_aplica"),
                              "retroativo": c.get("retroativo", False)}
    qualidade = {
        "fontes": qual,
        "notas": [
            ("auditar_s7_arvore_lidar.json: the independent-check field is 'PENDENTE' "
             "in the JSON; the independent re-check was done later (CONFIRMED WITH "
             "CAVEAT, sha256 09d3030b... verified by code)"),
            ("attribution of g1': the 'desenho' field of "
             "treinar_arvore_persistir_v2.json and diferencas_declaradas[1] of "
             "auditar_s7_arvore_lidar.json misattribute the recalibration; the g1' "
             "values were fixed after the failure. This sheet uses the correct note; "
             "the stamped JSON files were not modified"),
            "O3 20-30 m: veredito_O3 e falha_ocorreria_ja_so_com_v23_x_mlp gravados juntos "
            "(lacuna 2)",
            "contrastes: transecto e pegada com o mesmo fluxo de bootstrap (lacuna 5)",
            "admissibilidade: contagem reconstruida de fracao arredondada (lacuna 4); "
            "arvore em suavidade 0 x redes em lambda_TV 0,35 (lacuna 6)",
            "results/f2_arvores.json: carimbo retroativo, sem sha256 do script",
            "auditar_admissibilidade.json: carimbado em 16/08 em outro host, sem sha256 do script",
        ],
    }

    # ==================================================================
    # G3 (final): no source changed during the run
    # ==================================================================
    for k, p in ARQS.items():
        if PROV.sha256(p) != sha0[k]:
            raise RuntimeError(f"G3: {p.name} mudou durante a corrida")

    n_fatos_g = sum(len(v) for v in grupos.values())
    saida = {
        "gerado_para": v24["gerado_para"],
        "regra": v24["regra"],
        "protocolo": (("Design of check S7 (resumption); parte_G read from "
                       "auditar_s7_arvore_lidar.json (sha256 ") + SHA_ESPERADO["s7"][:8]
                      + ("..., independent check) and treinar_arvore_persistir_v2.json; parts A-F "
                         "= folha_de_fatos_jstars_v24.json (sha256 ")
                      + SHA_ESPERADO["folha_v24"][:8] + "...), identicas"),
        "protocolo_v24": v24["protocolo"],
        "protocolo_v23": v24["protocolo_v23"],
        "protocolo_v22": v24["protocolo_v22"],
        **copia,
        "parte_G": {
            "n_fatos": n_fatos_g,
            "grupos": {g: len(v) for g, v in grupos.items()},
            "qualidade_das_fontes": qualidade,
            "fatos": grupos,
        },
        "guardas": {
            "GI_partes_A_F_identicas_a_v24": {"trilhas_divergentes": [], "passou": True},
            "G1_ids_unicos": {"n_ids_v24": n_ids_v24, "n_ids_total": len(ids)},
            "G2_sem_nulos": True,
            "G3_sha256_conferidos": {"fixados": sorted(SHA_ESPERADO),
                                     "cruzamentos_com_fontes_do_S7_e_treino_v2": n_cruz, "passou": True},
            "G4_derivados_conferidos": True,
        },
        "guardas_v24": v24["guardas"],
        "guardas_v23": v24["guardas_v23"],
    }
    saida_p = Path(a.saida)
    if not saida_p.is_absolute():
        saida_p = RAIZ / saida_p
    PROV.gravar(saida_p, saida, fontes_lidas=list(ARQS.values()), script=__file__)

    # GI (2): re-read from disk and compare with v24 re-read from disk
    relido = json.loads(saida_p.read_text(encoding="utf-8"))
    v24_disco = json.loads(ARQS["folha_v24"].read_text(encoding="utf-8"))
    if v24_disco != v24:
        raise RuntimeError("GI: v24 no disco mudou durante a corrida")
    conferir_GI(relido, "JSON relido do disco")
    if relido["parte_G"] != json.loads(json.dumps(saida["parte_G"])):
        raise RuntimeError("releitura: parte_G gravada != montada")
    log(f"GI ok: {', '.join(COPIADAS)} == v24 (0 trilhas; apos releitura)")
    log(f"G3: {len(SHA_ESPERADO)} sha256 fixados; {n_cruz} cruzamentos com o _fontes do S7 e do treino v2")
    log(f"p_wilcoxon recomputado == gravado em {n_wil}/24 nos")
    log(f"ids: {n_ids_v24} da v24 + {len(ids) - n_ids_v24} novos = {len(ids)}")
    log(f"parte_G: {n_fatos_g} fatos em {len(grupos)} grupos -> {saida_p}")
    for g, v in grupos.items():
        log(f"  {g}: {', '.join(f['id'] for f in v)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
