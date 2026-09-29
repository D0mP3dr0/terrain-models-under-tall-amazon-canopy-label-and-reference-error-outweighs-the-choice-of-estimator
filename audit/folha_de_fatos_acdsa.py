# -*- coding: utf-8 -*-
"""folha_de_fatos_acdsa: consolidates EVERY citable number of the
ACDSA 2027 system paper into a single hash-stamped sheet, with an artifact
+ field per line. This is the mechanism that prevents citing a number from
prose/memory: the paper's rule is "every number in the text has a line
here; a line with no use is allowed, a number with no line does not
exist".

This script does not measure anything: it extracts by DIRECT INDEXING from
already hash-stamped artifacts and re-stamps the set.

GUARDS (fail loud):
  G1  unique fact id (duplicate = RuntimeError).
  G2  no VALUE (`valor` field) is None/NaN in the sheet, at ANY nesting
      level; `unidade: None` is legitimate and stays outside the guard.
      The only declared exception: G2_EXCECOES (today only
      receita_pesos_desligados, intentional Nones for a disabled
      constraint).
  G3  every source has _proveniencia with sha256_script OR retroativo =
      true (retroactive sources get a carimbo_retroativo on the line). No
      exceptions.
  G4  usage invariants: e5.papel contains "NAO citavel" (the paper may only
      use it as a guard); manifesto confere_checkpoint = true;
      contagem_arestas formula_confere_gerador = true; S5 has no d8_*/d1_*
      file (the fence).

FENCE: no D8/D1 as a claim, no label anatomy, no FABDEM/GEDTM30
comparison, no "continental". The lines in this sheet respect the fence by
construction (S5 already trims it; no d8_*/d1_* source enters here).

Usage:  python folha_de_fatos_acdsa.py [--saida folha_de_fatos_acdsa.json]
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

ARQS = {
    "arestas": RAIZ / "contagem_arestas.json",
    "celula": RAIZ / "geometria_celula.json",
    "manifesto": RAIZ / "manifesto_features_46.json",
    "ofat": RAIZ / "results" / "ofat_suavfina_b4s035.json",
    "s5": RAIZ / "auditar_tradeoff_suavidade_x1.json",
    "geom_res": RAIZ / "auditar_geometria_reserva.json",
    "e5": RAIZ / "auditar_e2_e5.json",
    "s1": RAIZ / "auditar_regime_reserva.json",
    "s3": RAIZ / "auditar_paridade_capacidade.json",
    "s6": RAIZ / "auditar_predicao_nula.json",
    "cmp_lambda": RAIZ / "comparar_ponto_s035.json",
    "s7": RAIZ / "auditar_recursos_s7.json",
    "s8": RAIZ / "auditar_hiperparametros_s8.json",
    "cmp_dossel": RAIZ / "comparar_ofat_dossel.json",
}


def log(m=""):
    print(m, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default="folha_de_fatos_acdsa.json")
    a = ap.parse_args()

    d = {}
    for k, p in ARQS.items():
        if not p.exists():
            raise FileNotFoundError(f"fonte ausente: {p}")
        d[k] = json.loads(p.read_text(encoding="utf-8"))

    fatos: list[dict] = []
    ids = set()
    G2_EXCECOES = {"receita_pesos_desligados"}   # intentional Nones (null config values copied as-is)

    def _sem_nulos(x, trilha):
        """Recursive G2: None/NaN at any nesting level raises RuntimeError."""
        if x is None or (isinstance(x, float) and math.isnan(x)):
            raise RuntimeError(f"G2: valor nulo em {trilha}")
        if isinstance(x, dict):
            for k, v in x.items():
                _sem_nulos(v, f"{trilha}.{k}")
        elif isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                _sem_nulos(v, f"{trilha}[{i}]")

    def fato(id_, valor, unidade, chave_arq, campo, nota=None):
        if id_ in ids:                                          # G1
            raise RuntimeError(f"G1: id duplicado: {id_}")
        ids.add(id_)
        if id_ not in G2_EXCECOES:                              # G2
            _sem_nulos(valor, id_)
        if "_proveniencia" not in d[chave_arq]:                 # fail loud
            raise RuntimeError(f"G3: {ARQS[chave_arq].name} sem "
                               f"_proveniencia")
        prov = d[chave_arq]["_proveniencia"]
        linha = {"id": id_, "valor": valor, "unidade": unidade,
                 "artefato": ARQS[chave_arq].name, "campo": campo}
        if prov.get("retroativo"):
            linha["carimbo_retroativo"] = True                  # G3
        elif not prov.get("sha256_script"):
            raise RuntimeError(f"G3: {ARQS[chave_arq].name} sem "
                               f"sha256_script e sem retroativo=true")  # G3
        if nota:
            linha["nota"] = nota
        fatos.append(linha)

    # --- system --------------------------------------------------------------
    if not d["arestas"]["formula_confere_gerador"]:             # G4
        raise RuntimeError("G4: contagem_arestas sem formula_confere_gerador")
    fato("nos_uniao", d["arestas"]["nos_uniao"], "nos", "arestas", "nos_uniao")
    fato("arestas_uniao", d["arestas"]["arestas_uniao"], "arestas",
         "arestas", "arestas_uniao")
    fato("grau_medio", d["arestas"]["grau_medio"], None, "arestas",
         "grau_medio")
    fato("area_km2", d["celula"]["area_km2"], "km2", "celula", "area_km2",
         nota="mosaico 2x2 graus, Amazonia central; 'continental' VETADO")
    fato("bbox", d["celula"]["bbox"], "graus", "celula", "bbox")
    fato("grade", d["celula"]["grade"], None, "celula", "grade")
    fato("lado_reserva_km",
         d["celula"]["conversoes_texto"]["reserva_1800_celulas_km"], "km",
         "celula", "conversoes_texto.reserva_1800_celulas_km",
         nota="arredonda para 55 km em prosa")
    if not d["manifesto"]["confere_checkpoint"]:                # G4
        raise RuntimeError("G4: manifesto sem confere_checkpoint")
    fato("d_in", d["manifesto"]["d_in_modelo"], "canais", "manifesto",
         "d_in_modelo",
         nota="46 = 44 colunas de fusao + 2 canais de observacao "
              "(observacao_b2.colunas)")
    fato("largura_fusao", d["manifesto"]["largura_X"], "colunas",
         "manifesto", "largura_X")
    # Config is read ONLY from the top level, failing loudly (a nested scan
    # could pick an arbitrary per-run config).
    cfg = d["ofat"]["config"]
    if "peso_suavidade" not in cfg:
        raise RuntimeError("ofat: config do topo sem peso_suavidade")
    for chave, unid in (("peso_suavidade", None), ("peso_declividade", None),
                        ("boost_atl08", None)):
        fato(f"receita_{chave}", cfg[chave], unid, "ofat", f"config.{chave}")
    fato("receita_pesos_desligados",
         {k: cfg[k] for k in ("peso_solo", "peso_agua", "peso_dossel")},
         None, "ofat", "config.peso_solo/peso_agua/peso_dossel",
         nota=("v2.3: null in the config does NOT disable the constraint — "
               "b2_transferencia.py (~3273-3282) resolves null to the measured "
               "PESOS_ALVO, and the penalties remain ACTIVE (see fisica_pesos_ativos). "
               "The id 'desligados' is historical; the values here are the config nulls, "
               "faithfully copied. The previous note ('null = constraint disabled') was "
               "an upstream reading, refuted by the physics records of the runs."))
    # EFFECTIVE penalty state, read from the 20 training records
    # (5 runs x 4 quadrants): from the artifact, not from the config.
    _fis = []
    for _k, _v in d["ofat"].items():
        if isinstance(_v, dict) and _v.get("layout_reservas"):
            for _q in ("Q1", "Q2", "Q3", "Q4"):
                _fis.append(_v["quadrantes"][_q]["treino"]["fisica"])
    if len(_fis) != 20:
        raise RuntimeError(
            f"fisica_pesos_ativos: esperados 20 registros, ha {len(_fis)}")
    _f0 = _fis[0]
    for _f in _fis:
        if (_f["ativa"] is not True or _f["peso_solo"] != _f0["peso_solo"]
                or _f["peso_agua"] != _f0["peso_agua"]):
            raise RuntimeError(f"fisica_pesos_ativos: registro divergente: {_f}")
        if "dossel" not in _f["ultimo_lote"]:
            raise RuntimeError(
                "fisica_pesos_ativos: testemunha do termo de dossel ausente")
    fato("fisica_pesos_ativos",
         {"ativa": True, "peso_solo": _f0["peso_solo"],
          "peso_agua": _f0["peso_agua"], "n_registros": len(_fis)},
         None, "ofat", "runs[*].quadrantes[*].treino.fisica",
         nota="as 3 penalidades de classe ATIVAS na receita adotada; termo "
              "de dossel presente em ultimo_lote nos 20 registros como "
              "testemunha. O ALVO do dossel (137.0) e constante de codigo "
              "(PESOS_ALVO) documentada no vault "
              "resultado_teste_1_peso_dossel_2026-08-09 e NAO entra como "
              "valor citavel desta folha por falta de JSON carimbado da era")
    # Number of loss terms, derived by enumeration rather than read from a
    # single field; the 3 per-class weights are the SAME fields as the fact
    # above (direct indexing already fails loudly if one is missing).
    _classes_perda = ("peso_solo", "peso_agua", "peso_dossel")
    for _k in _classes_perda:
        if _k not in cfg:                                    # fail-loud
            raise RuntimeError(f"n_termos_perda: campo {_k} sumiu do config")
    fato("n_termos_perda", 3 + len(_classes_perda), "termos", "ofat",
         "config (derivado: ancora + peso_suavidade + peso_declividade + "
         "peso_solo/peso_agua/peso_dossel)",
         nota="enumeracao: 1 ancora + 1 suavidade + 1 declividade + 3 "
              "penalidades por classe (solo/agua/dossel); mecanismos em "
              "b2_transferencia.py:perda_fisica (dossel hinge de sinal; "
              "agua planaridade em arestas; solo encolhimento de magnitude)")
    fato("sementes_receita", cfg["seeds"], None, "ofat", "config.seeds")

    # --- S5: x1 trade-off (central figure) -----------------------------------
    for p in d["s5"]["pontos"]:
        if p["teto"] != "x1" or "d8_" in p["arquivo"] or "d1_" in p["arquivo"]:
            raise RuntimeError(f"G4/cerca: ponto fora do recorte: {p}")  # G4
        base = f"s5_{p['familia']}_l{p['lambda']}"
        fato(f"{base}_skill", {"media": p["skill_media"],
                               "ic95": p["skill_ic95"]}, "skill", "s5",
             f"pontos[{p['familia']},{p['lambda']}].skill_media/skill_ic95")
        fato(f"{base}_viol_1m", p["violacoes_acima_1m_media"],
             "celulas/replica", "s5", "violacoes_acima_1m_media")
        fato(f"{base}_viol_10cm", p["violacoes_acima_10cm_media"],
             "celulas/replica", "s5", "violacoes_acima_10cm_media")
        fato(f"{base}_cobertura_testemunha", p["cobertura_testemunha"],
             "fracao", "s5", "cobertura_testemunha",
             nota="OBRIGATORIO ao lado de qualquer leitura de violacoes "
                  "(C15): cobertura < 1 significa ausencia de medicao, "
                  "nunca conformidade")
        fato(f"{base}_uso_orcamento", p["uso_do_orcamento_media"], "fracao",
             "s5", "uso_do_orcamento_media")
    fato("s5_limiar_leitura", d["s5"]["limiar_da_leitura_m"], "m", "s5",
         "limiar_da_leitura_m",
         nota="declarar o limiar; a 10 cm o grafo ja viola a x1")
    fato("s5_sha256_pontos", d["s5"]["sha256_pontos_recorte"], None, "s5",
         "sha256_pontos_recorte")

    # --- hold-out geometry ---------------------------------------------------
    fato("dist_reserva_mediana_km", d["geom_res"]["dist_reserva_km"]["mediana"],
         "km", "geom_res", "dist_reserva_km.mediana")
    fato("dist_em_dist_mediana_km",
         d["geom_res"]["dist_em_distribuicao_km"]["mediana"], "km",
         "geom_res", "dist_em_distribuicao_km.mediana",
         nota="citar as DUAS medianas, nunca o quociente "
              "(razao_em_metros_METRICAS_MISTAS e vetada pelo artefato)")
    fato("geom_n_combinacoes", d["geom_res"]["n_combinacoes"], None,
         "geom_res", "n_combinacoes")
    fato("buffer_celulas", d["geom_res"]["buffer_celulas"], "celulas",
         "geom_res", "buffer_celulas")
    fato("n_reserva_uniao", d["geom_res"]["n_reserva_uniao"], "ancoras",
         "geom_res", "n_reserva_uniao")

    # --- E5: protocol guard (NEVER a model result) ---------------------------
    e5 = d["e5"]["e5_guarda_g2"]
    if "NAO citavel" not in e5["papel"]:                        # G4
        raise RuntimeError("G4: e5.papel mudou — reavaliar uso")
    fato("e5_papel", e5["papel"], None, "e5", "e5_guarda_g2.papel")
    fato("e5_pareado_skill", {"media": e5["pareado_skill"]["media"],
                              "p": e5["pareado_skill"]["p"]}, "skill", "e5",
         "e5_guarda_g2.pareado_skill",
         nota="uso EXCLUSIVO como guarda do protocolo (secao de protocolo), "
              "nunca como ablacao/resultado")
    fato("e5_razao_dp", e5["razao_dp_entre_sementes"], None, "e5",
         "e5_guarda_g2.razao_dp_entre_sementes")

    # --- S1: decomposition by regime -----------------------------------------
    s1 = d["s1"]
    fato("q1_degenerado", s1["q1_degenerado"], None, "s1", "q1_degenerado")
    for nome in ("dossel_denso", "dossel_esparso"):
        b = s1["classes_diagnosticas"][nome]
        fato(f"s1_{nome}_n_uniao", b["n_nos_uniao"], "nos", "s1",
             f"classes_diagnosticas.{nome}.n_nos_uniao")
        for fam in ("gatv2", "mlp"):
            e = b[f"mae_uniao_{fam}"]
            fato(f"s1_{nome}_mae_{fam}", {"media": e["media"],
                                          "ic95": e["ic95"]}, "m", "s1",
                 f"classes_diagnosticas.{nome}.mae_uniao_{fam}")
        c = b["contraste_pareado_gatv2_menos_mlp_m"]
        fato(f"s1_{nome}_contraste", {"media": c["media"], "ic95": c["ic95"]},
             "m", "s1",
             f"classes_diagnosticas.{nome}."
             f"contraste_pareado_gatv2_menos_mlp_m",
             nota=s1["leitura"][nome]["frase"])
    fato("s1_fonte_nao_diagnosticas",
         s1["classes_nao_diagnosticas"]["fonte_citavel"], None, "s1",
         "classes_nao_diagnosticas.fonte_citavel")
    fato("s1_fracao_coberta_s6",
         s1["escopo_s6"]["fracao_da_uniao_coberta_por_classe"], "fracao",
         "s1", "escopo_s6.fracao_da_uniao_coberta_por_classe")

    # --- S3: capacity parity -------------------------------------------------
    s3 = d["s3"]
    for fam in ("gatv2_2_semobs", "mlp_semobs"):
        fato(f"s3_params_{fam}", s3["familias"][fam]["params_treinaveis"],
             "parametros", "s3", f"familias.{fam}.params_treinaveis")
    fato("s3_razao_mlp_gatv2", s3["leitura"]["razao_mlp_sobre_gatv2"], None,
         "s3", "leitura.razao_mlp_sobre_gatv2",
         nota="MLP e o controle MAIOR — fecha a saida 'faltou capacidade'")
    # Number of seeds in the double parameter count (the "ten seeds" of
    # Section V-B). len() of the field, with a fail-loud sanity check.
    _sg2 = s3["sementes_g2"]
    if not isinstance(_sg2, list) or len(_sg2) != len(set(_sg2)):
        raise RuntimeError("s3_n_sementes_paridade: sementes_g2 invalido "
                           "(nao-lista ou duplicata)")
    fato("s3_n_sementes_paridade", len(_sg2), "sementes", "s3",
         "sementes_g2 (len)",
         nota="contagem dupla checkpoint vs re-instanciacao conferida em "
              "cada uma; as 5 sementes da receita adotada sao subconjunto")

    # --- lambda = 0.35 criterion (comparar_ponto_s035.json) ------------------
    cmp_ = d["cmp_lambda"]
    fato("lambda_criterio_regra", cmp_["recomendacao"]["regra"], None,
         "cmp_lambda", "recomendacao.regra",
         nota="criterio pre-declarado antes da varredura do peso de "
              "suavidade")
    fato("lambda_guarda_skill", cmp_["tolerancia_skill"], "skill",
         "cmp_lambda", "tolerancia_skill")
    _p0 = cmp_["por_peso"]["0.0"]["criados_2m_soma"]
    _p35 = cmp_["por_peso"]["0.35"]["criados_2m_soma"]
    if not (_p0 > 0 and _p35 < _p0):                         # fail-loud
        raise RuntimeError(f"lambda_pocos_2m: valores implausiveis {_p0}/{_p35}")
    fato("lambda_pocos_2m",
         {"l0": _p0, "l035": _p35, "fracao_cortada": 1.0 - _p35 / _p0},
         "pocos", "cmp_lambda", "por_peso[0.0/0.35].criados_2m_soma",
         nota="pocos hidrologicos criados com profundidade >2m; "
              "fracao_cortada derivada aqui (1 - l035/l0)")
    fato("lambda_custo_skill",
         cmp_["por_peso"]["0.35"]["skill_delta_vs_p0"], "skill",
         "cmp_lambda", "por_peso.0.35.skill_delta_vs_p0",
         nota="excede a guarda (tolerancia_skill) em modulo; ver "
              "lambda_criterio_candidatos_vazio")
    fato("lambda_criterio_candidatos_vazio",
         cmp_["recomendacao"]["candidatos"] == [], None,
         "cmp_lambda", "recomendacao.candidatos",
         nota=("the rule did not select a candidate (guard exceeded); 0.35 was adopted "
               "by a formal decision, with an excess smaller than the between-seed "
               "standard deviation (skill_dp in the artifact itself)"))
    fato("s3_d_in_era_b1", s3["d_in_era_b1_derivado"]["d_in_era_b1"],
         "canais", "s3", "d_in_era_b1_derivado.d_in_era_b1",
         nota="derivado do artefato da era B1; d_in explica integralmente a "
              "diferenca vs era B1")

    # --- S6: null prediction (citable source of non-diagnosticity) -----------
    s6 = d["s6"]
    for nome in ("agua", "solo_exposto", "dossel_denso", "dossel_esparso"):
        v = s6["veredito"][nome]
        fato(f"s6_{nome}", {"veredito": v["veredito"],
                            "n_celulas_avaliadas": v["n_celulas_avaliadas"],
                            "unanime": v["unanime"]}, None, "s6",
             f"veredito.{nome}")
    # bounds measured on the evaluated cells (extracted, not re-measured)
    difs = {"agua": [], "solo_exposto": []}
    for quad, bq in s6["celulas"].items():
        for run, br in bq["runs"].items():
            for nome in difs:
                cel = br["classes"][nome]          # direct indexing
                if cel["avaliada"]:
                    difs[nome].append(
                        (cel["dif_rel"],
                         abs(cel["mae_modelo"] - cel["mae_nulo"]),
                         f"{quad}/{run}"))
    for nome, xs in difs.items():
        if not xs:
            raise RuntimeError(f"S6 sem celulas avaliadas para {nome}")
        # Both maxima, each with its own cell (dif_abs is not taken from the
        # cell with the largest dif_rel).
        pior_rel = max(xs, key=lambda t: t[0])
        pior_abs = max(xs, key=lambda t: t[1])
        fato(f"s6_{nome}_cota",
             {"dif_rel_max": pior_rel[0], "celula_dif_rel": pior_rel[2],
              "dif_abs_max_m": pior_abs[1], "celula_dif_abs": pior_abs[2]},
             None, "s6",
             f"celulas.*.runs.*.classes.{nome} (maximos independentes "
             f"sobre avaliadas)",
             nota="cota superior para a frase de nao-diagnosticidade; "
                  "MISTA reporta-se como MISTA + cota, nunca como NULA")

    # --- S7: computational cost of the adopted recipe ------------------------
    s7 = d["s7"]
    if s7["tag"] != "b4s035":                                    # fail-loud
        raise RuntimeError(f"s7: tag {s7['tag']} != b4s035")
    # Two-level chain of custody: the sources S7 declares it read
    # must exist on disk with the same sha256.
    import hashlib
    for chave in ("s7", "s8"):
        for fonte in d[chave]["_fontes"]:
            pf = RAIZ / fonte["caminho"]
            if not pf.exists():
                raise RuntimeError(f"D9: fonte do {chave} ausente: {pf}")
            if pf.name == "folha_de_fatos_acdsa.json":
                raise RuntimeError(f"D9: {chave} declara a propria folha como fonte (ciclo)")
            sha = hashlib.sha256(pf.read_bytes()).hexdigest()
            if sha != fonte["sha256"]:
                raise RuntimeError(f"D9: sha256 divergente em {pf.name} ({chave})")
    if s7["n_nos_uniao"] != d["arestas"]["nos_uniao"]:            # mirrors the contagem_arestas node count
        raise RuntimeError("s7: n_nos_uniao != contagem_arestas")
    for fam, nome in (("gatv2", "grafo"), ("mlp", "pontual")):
        f = s7["familias"][fam]
        if len(f["sementes"]) != 5 or f["sementes"] != sorted(cfg["seeds"]):
            raise RuntimeError(f"s7: sementes de {fam} != receita")
        fato(f"s7_{fam}_parede_b2_min", f["parede_b2_min"], "min",
             "s7", f"familias.{fam}.parede_b2_min",
             nota=f"{nome}: INICIO -> B2 CONCLUIDO; 5 sementes x 4 etapas, "
                  "carga, treino e predicao densa da fase B2; escopo igual nas "
                  "duas familias (D1)")
        fato(f"s7_{fam}_treino_puro_por_semente_min",
             f["treino_s_soma_media_min"], "min", "s7",
             f"familias.{fam}.treino_s_soma_media_min",
             nota="media das 5 sementes da soma de treino.treino_s das 4 etapas "
                  "(JSON da corrida); exclui carga, uniao e predicao (D2)")
        fato(f"s7_{fam}_vram_pico_gb", f["vram_pico_gb"], "GB", "s7",
             f"familias.{fam}.vram_pico_gb",
             nota=f"maximo alocado sobre sementes x etapas (JSON da corrida), em "
                  f"{f['vram_pico_onde']}; conferido contra o log (G3)")
        fato(f"s7_{fam}_rss_pico_gb", f["rss_pico_gb"], "GB", "s7",
             f"familias.{fam}.rss_pico_gb",
             nota="pico de RAM do processo (linhas [MEM] do log; G7 exige que "
                  "existam)")
    fato("s7_epocas_teto", s7["epocas_teto"], "epocas", "s7", "epocas_teto",
         nota="teto por etapa da janela crescente, igual nas duas familias (G4)")
    hw = s7["hardware_da_maquina_de_auditoria"]
    fato("s7_hardware_da_maquina_de_auditoria",
         {k: hw[k] for k in ("cpu", "ram_total_gb", "gpu", "vram_total_gb")},
         None, "s7", "hardware_da_maquina_de_auditoria",
         nota="DECLARACAO DO AUTOR (D3): hardware lido da maquina em que a "
              "auditoria rodou; nenhum artefato das corridas de 10-11/08 "
              "registra hardware (so 'dispositivo: cuda'); o host do carimbo "
              "dos JSONs e retroativo. Em prosa: 'the authors' workstation'")

    # --- S8: architecture, training, intervals and test ----------------------
    s8 = d["s8"]
    if s8["tag"] != "b4s035" or s8["d_in"] != d["manifesto"]["d_in_modelo"]:
        raise RuntimeError("s8: tag ou d_in divergem da receita")
    ag, am = s8["arquitetura"]["gatv2"], s8["arquitetura"]["mlp"]
    _p = {f["id"]: f["valor"] for f in fatos}
    if (ag["parametros_treinaveis"] != _p["s3_params_gatv2_2_semobs"]
            or am["parametros_treinaveis"] != _p["s3_params_mlp_semobs"]):   # mirrors the S3 parameter facts
        raise RuntimeError("s8: parametros das arquiteturas != fatos s3")
    fato("s8_gatv2_arquitetura",
         {k: ag[k] for k in ("camadas_gatv2conv", "cabecas", "canais_por_cabeca",
                              "largura_oculta", "dropout")},
         None, "s8", "arquitetura.gatv2",
         nota="instanciada de b1_kriging_indutivo.GATv2(46, 2); parametros "
              "treinaveis == s3_params_gatv2_2_semobs (G1)")
    fato("s8_mlp_arquitetura",
         {k: am[k] for k in ("blocos_lineares_ocultos", "largura_oculta",
                              "ativacao", "dropout")},
         None, "s8", "arquitetura.mlp",
         nota="instanciada de b1_kriging_indutivo.MLP(46); parametros == "
              "s3_params_mlp_semobs (G1)")
    tr = s8["treino"]
    fato("s8_fan_out", tr["fan_out_por_camada"], "vizinhos por camada", "s8",
         "treino.fan_out_por_camada",
         nota="lido do log do grafo (25 amostradores, todos [8, 8]); o MLP amostra "
              "sem vizinhos ([-1])")
    if set(tr["lote_testemunha_logs"].values()) != {tr["lote"]}:
        raise RuntimeError("s8: lote dos logs != BATCH_V23")
    fato("s8_otimizador",
         {"otimizador": tr["otimizador"], "weight_decay": tr["weight_decay_matrizes"],
          "weight_decay_escopo": tr["weight_decay_escopo"],
          "agendador": tr["agendador"], "lr_min": tr["lr_min"], "lote": tr["lote"]},
         None, "s8", "treino.{otimizador,weight_decay_matrizes,weight_decay_escopo,agendador,lr_min,lote}",
         nota="chamadas lidas de b2_transferencia.py (G5); lote = testemunha 'BATCH do vetor "
              "V23' dos dois logs, igual a BATCH_V23 (G8); decaimento em tensores com "
              "ndim > 1, so os vieses ficam fora")
    fato("s8_agendador_execucao", tr["agendador_execucao"], None, "s8",
         "treino.agendador_execucao",
         nota="cosseno configurado sobre o teto da etapa ate lr_min; a parada antecipada "
              "corta a curva, e so as etapa-corridas que esgotam o teto chegam ao piso. "
              "Em prosa: 'a cosine schedule over each stage's cap toward 1e-6, cut short "
              "when early stopping fires'")
    fato("s8_etapas",
         [{k: e[k] for k in ("etapa", "epocas_teto", "lr_inicial", "paciencia", "precisao")}
          for e in tr["etapas"]],
         None, "s8", "treino.etapas",
         nota="uniformes entre as 5 sementes e as 2 familias (G3); lr do log == JSON (G4)")
    fato("s8_ic95_metodo", s8["intervalos"]["metodo"], None, "s8", "intervalos.metodo",
         nota=f"reproduzido nos {s8['intervalos']['s5_pontos_reproduzidos']} pontos de S5 "
              "(G6); mesmo metodo em S1 (T_975_GL4, ddof=1)")
    fato("s8_e5_teste", s8["teste_e5"]["teste"], None, "s8", "teste_e5.teste",
         nota=f"p reproduzido = {s8['teste_e5']['p']:.6g} (G7) vs e5_pareado_skill")

    # --- canopy-penalty weight (comparar_ofat_dossel.json) -------------------
    cd_ = d["cmp_dossel"]
    rec = cd_["recomendacao"]
    if rec["peso"] != 137 or rec["peso"] != min(rec["candidatos"]):     # fail-loud
        raise RuntimeError(f"cmp_dossel: recomendacao {rec} nao aponta 137 como menor candidato")
    p0, p137 = cd_["por_peso"]["0"], cd_["por_peso"]["137"]
    sev0 = max(v["acima_1m"] for v in p0["severidade"].values())
    sev137 = max(v["acima_1m"] for v in p137["severidade"].values())
    if not (sev137 == 0.0 and sev0 > 0.1):                              # fail-loud
        raise RuntimeError(f"cmp_dossel: severidade implausivel {sev0}/{sev137}")
    if not p137["dentro_do_guarda_corpo"] or abs(p137["skill_delta_vs_p0"]) > cd_["tolerancia_skill"]:
        raise RuntimeError("cmp_dossel: 137 fora do guarda-corpo")
    fato("wc_criterio_regra", rec["regra"], None, "cmp_dossel", "recomendacao.regra",
         nota="criterio declarado antes da varredura OFAT de 2026-08-08 (vault "
              "resultado_teste_1_peso_dossel_2026-08-09); tolerancia_skill = "
              f"{cd_['tolerancia_skill']}")
    fato("wc_peso", rec["peso"], None, "cmp_dossel", "recomendacao.peso",
         nota=f"candidatos {rec['candidatos']}; 137 e o menor; igual a PESOS_ALVO['dossel'] "
              "em b2_transferencia.py (resolucao null -> alvo nas corridas adotadas)")
    fato("wc_custo_skill", p137["skill_delta_vs_p0"], "skill", "cmp_dossel",
         "por_peso.137.skill_delta_vs_p0",
         nota=f"skill {p137['skill']:.4f} vs {p0['skill']:.4f} no peso 0; dp entre sementes "
              f"{p0['skill_dp']:.4f} no peso 0")
    fato("wc_severidade_1m",
         {"peso0_max_quadrante": sev0, "peso137_max_quadrante": sev137,
          "peso0_por_quadrante": {q: v["acima_1m"] for q, v in p0["severidade"].items()}},
         "fracao das violacoes de dossel acima de 1 m", "cmp_dossel",
         "por_peso.{0,137}.severidade.*.acima_1m",
         nota="fracao das celulas de dossel denso com Delta < 0 cuja violacao passa de 1 m; "
              "zero nos 4 quadrantes com 137, 549 e 2196")

    # --- validation-split block and null-predictor tolerance -----------------
    _blocos = {d["ofat"][k]["quadrantes"][q]["particao"]["bloco_m"]
               for k in d["ofat"] if k.startswith("amazonia_")
               for q in ("Q1", "Q2", "Q3", "Q4")}
    if len(_blocos) != 1:                                        # fail-loud
        raise RuntimeError(f"split_bloco_m nao uniforme: {_blocos}")
    fato("split_bloco_m", _blocos.pop(), "m", "ofat",
         "amazonia_*/quadrantes/*/particao.bloco_m",
         nota="lado do bloco espacial do split treino/validacao/teste interno "
              "(52 celulas); em prosa 'about 1.6 km'")
    import re as _re
    _m = _re.search(r"<\s*([0-9.eE+-]+)\s*na celula", d["s6"]["criterio"]["NULA"])
    if not _m:
        raise RuntimeError("s6: tolerancia relativa nao encontrada em criterio.NULA")
    fato("s6_tolerancia_rel", float(_m.group(1)), None, "s6", "criterio.NULA",
         nota="|MAE(p,y) - MAE(0,y)| / MAE(0,y) < tolerancia na celula; "
              "pre-registrado em 2026-08-31; em prosa '10^-3'")

    saida = {
        "gerado_para": "paper de sistema ACDSA 2027 (acdsa_v23_sistema.tex)",
        "regra": ("todo numero citado no texto do paper tem linha nesta "
                  "folha; linha sem uso e permitida, numero sem linha NAO "
                  "EXISTE. Copiar da folha, conferir por diff contra o "
                  "artefato da linha."),
        "cerca": ("decisao_tema_2026-08-30.md: sem D8/D1 como claim, sem "
                  "anatomia do rotulo, sem FABDEM/GEDTM30 pareado, sem "
                  "'continental', sem autocitacao, E5 so como guarda"),
        "n_fatos": len(fatos),
        "fatos": fatos,
    }
    PROV.gravar(RAIZ / a.saida, saida, fontes_lidas=list(ARQS.values()),
                script=__file__)
    log(f"{len(fatos)} fatos gravados em {a.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
