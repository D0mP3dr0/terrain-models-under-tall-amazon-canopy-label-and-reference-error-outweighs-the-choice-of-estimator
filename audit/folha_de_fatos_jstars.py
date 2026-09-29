# -*- coding: utf-8 -*-
"""folha_de_fatos_jstars -- hash-stamped fact sheet for the manuscript,

Article 1 of V23 (target: IEEE JSTARS). Same pattern as
`folha_de_fatos_acdsa.py`: each fact = value + artifact + field + a
provenance stamp; no value typed by hand, everything read from an artifact
by code. Starts with section "parte_A". Section "parte_B" (E4/T3 audits,
not yet run) is reserved and empty -- the structure is ready to receive the
facts once E4/T3 exist; nothing is invented here.

VERSION 1. Authorized by the day's protocol.

GUARDS (fail loud, adapted from folha_de_fatos_acdsa.py):
  G1  unique fact id (duplicate = RuntimeError).
  G2  no `valor` field is None/NaN in the sheet, at any nesting level
      (exceptions only when the fact itself IS the absence of a datum, and
      then the value becomes a boolean/descriptive proposition, never a raw
      None).
  G3  every JSON source has _proveniencia with sha256_script OR a
      retroactive one. Two sources in this round (auditar_d8_triangulo.json,
      auditar_f2_celula.json, both predating the requirement that
      introduced sha256_script) have neither: this does NOT become a silent
      exception -- it is recorded per source in
      `parte_A.qualidade_das_fontes`, without blocking the read (today's
      protocol requests exactly these artifacts as the source of facts 4).
  G4  derivations (ratio, relative cut, excess over the guard) are computed
      in code from the cited fields, never copied from prose; checked
      against the artifact's pre-computed field when one exists (e.g.,
      razao_mlp_sobre_gatv2).

Usage:  python folha_de_fatos_jstars.py [--saida folha_de_fatos_jstars.json]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import proveniencia as PROV          # noqa: E402

ARQS = {
    "cmp_lambda": RAIZ / "comparar_ponto_s035.json",
    "acdsa": RAIZ / "folha_de_fatos_acdsa.json",
    "s3": RAIZ / "auditar_paridade_capacidade.json",
    "s7": RAIZ / "auditar_recursos_s7.json",
    "d8": RAIZ / "auditar_d8_triangulo.json",
    "f2cel": RAIZ / "auditar_f2_celula.json",
    "f2arv": RAIZ / "results" / "f2_arvores.json",
    "nmad_diretas": RAIZ / "auditar_nmad_pareado_nativa_diretas.json",
    "laterais_script": RAIZ / "preparar_laterais_nativa_v3_diretas.py",
    "main_tex": RAIZ / "artigo_v23_2" / "pacote_overleaf" / "main.tex",
}


def log(m=""):
    print(m, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default="folha_de_fatos_jstars.json")
    a = ap.parse_args()

    for k, p in ARQS.items():
        if not p.exists():
            raise FileNotFoundError(f"fonte ausente: {k} -> {p}")

    d = {}
    for k, p in ARQS.items():
        if p.suffix == ".json":
            d[k] = json.loads(p.read_text(encoding="utf-8"))

    fatos: list[dict] = []
    ids = set()
    qualidade_fontes: dict = {}

    def _sem_nulos(x, trilha):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            raise RuntimeError(f"G2: valor nulo em {trilha}")
        if isinstance(x, dict):
            for kk, vv in x.items():
                _sem_nulos(vv, f"{trilha}.{kk}")
        elif isinstance(x, (list, tuple)):
            for i, vv in enumerate(x):
                _sem_nulos(vv, f"{trilha}[{i}]")

    def fato(id_, valor, unidade, chave_arq, campo, nota=None):
        if id_ in ids:                                          # G1
            raise RuntimeError(f"G1: id duplicado: {id_}")
        ids.add(id_)
        _sem_nulos(valor, id_)                                   # G2
        arq = ARQS[chave_arq]
        linha = {"id": id_, "valor": valor, "unidade": unidade,
                 "artefato": str(arq.relative_to(RAIZ)) if arq.is_relative_to(RAIZ) else arq.name,
                 "campo": campo}
        if arq.suffix == ".json":
            prov = d[chave_arq].get("_proveniencia", {})
            if chave_arq not in qualidade_fontes:
                if prov.get("retroativo"):
                    estado = "retroativo"
                elif prov.get("sha256_script"):
                    estado = "sha256_script"
                elif prov.get("sha256_script_no_momento_do_carimbo"):
                    estado = "sha256_script_no_momento_do_carimbo"
                elif prov:
                    estado = "SEM sha256_script E SEM retroativo (anterior a 18/08)"
                else:
                    estado = "SEM _proveniencia"
                qualidade_fontes[chave_arq] = {
                    "arquivo": arq.name, "estado": estado,
                    "script_declarado": prov.get("script")}
            if prov.get("retroativo"):
                linha["carimbo_retroativo"] = True                # G3
            elif not (prov.get("sha256_script")
                      or prov.get("sha256_script_no_momento_do_carimbo")):
                linha["aviso_proveniencia"] = (
                    "fonte sem sha256_script e sem retroativo=true "
                    "(artefato de 16-17/08, anterior a demanda 18/08); "
                    "ver parte_A.qualidade_das_fontes")
        if nota:
            linha["nota"] = nota
        fatos.append(linha)

    # ======================================================================
    # FACT 1 -- lambda_tv = 0.35 (comparar_ponto_s035.json)
    # ======================================================================
    cmp_ = d["cmp_lambda"]
    tol = cmp_["tolerancia_skill"]
    p0 = cmp_["por_peso"]["0.0"]
    p035 = cmp_["por_peso"]["0.35"]

    fato("jstars_lambda_criterio_regra", cmp_["recomendacao"]["regra"], None,
         "cmp_lambda", "recomendacao.regra",
         nota="criterio pre-declarado antes da varredura do peso de suavidade fina")
    fato("jstars_lambda_guarda_skill", tol, "skill", "cmp_lambda",
         "tolerancia_skill")
    fato("jstars_lambda_peso_nao_eleito", cmp_["recomendacao"]["peso"] is None,
         None, "cmp_lambda", "recomendacao.peso",
         nota="recomendacao.peso = null na fonte (nenhum peso eleito pela "
              "regra); o valor bruto null nao entra na folha (guarda G2), "
              "fica a proposicao booleana derivada")
    fato("jstars_lambda_p0", {"skill": p0["skill"], "skill_dp": p0["skill_dp"],
                              "dentro_guarda_corpo": p0["dentro_guarda_corpo"]},
         "skill", "cmp_lambda",
         "por_peso.0.0.{skill,skill_dp,dentro_guarda_corpo}")
    fato("jstars_lambda_p035",
         {"skill": p035["skill"], "skill_dp": p035["skill_dp"],
          "skill_delta_vs_p0": p035["skill_delta_vs_p0"],
          "dentro_guarda_corpo": p035["dentro_guarda_corpo"]},
         "skill", "cmp_lambda",
         "por_peso.0.35.{skill,skill_dp,skill_delta_vs_p0,dentro_guarda_corpo}")
    fato("jstars_lambda_criados_2m",
         {"p0": p0["criados_2m_soma"], "p035": p035["criados_2m_soma"]},
         "pocos hidrologicos >2m", "cmp_lambda",
         "por_peso.{0.0,0.35}.criados_2m_soma")

    corte_relativo = 1.0 - p035["criados_2m_soma"] / p0["criados_2m_soma"]
    fato("jstars_lambda_corte_relativo_criados_2m", corte_relativo, "fracao",
         "cmp_lambda", "derivado: 1 - criados_2m_soma(0.35)/criados_2m_soma(0.0)",
         nota="calculado em codigo a partir dos dois campos citados acima")

    excesso = abs(p035["skill_delta_vs_p0"]) - tol
    fato("jstars_lambda_excesso_sobre_guarda", excesso, "skill", "cmp_lambda",
         "derivado: |por_peso.0.35.skill_delta_vs_p0| - tolerancia_skill",
         nota="calculado em codigo")

    fato("jstars_lambda_excesso_vs_skill_dp",
         {"excesso_sobre_guarda": excesso, "skill_dp_p035": p035["skill_dp"],
          "excesso_menor_que_skill_dp": excesso < p035["skill_dp"]},
         "skill", "cmp_lambda",
         "derivado: excesso_sobre_guarda comparado a por_peso.0.35.skill_dp",
         nota="calculado em codigo: o excesso sobre a guarda de skill e "
              "menor que a dispersao (dp) entre sementes no mesmo peso")

    acdsa_lambda = [f for f in d["acdsa"]["fatos"] if f["id"].startswith("lambda_")]
    fato("jstars_lambda_registro_acdsa",
         {"ids": [f["id"] for f in acdsa_lambda], "n": len(acdsa_lambda)},
         None, "acdsa", "fatos[*].id (prefixo 'lambda_'), filtrado em codigo",
         nota="o ACDSA (folha_de_fatos_acdsa.json) ja registra a mesma "
              "ressalva do 0.35 fora da guarda-corpo de skill para o mesmo "
              "criterio comparar_ponto_s035.json; ver os ids acima para os "
              "valores la carimbados (resposta ao tutor em "
              "ACDSA_latex/_planejamento/resposta_ao_tutor_2026-09-05.md §1, "
              "NAO portavel por conter bastidor — regra 6)")

    # ======================================================================
    # FACT 2 -- parameter count (auditar_paridade_capacidade.json)
    # ======================================================================
    s3 = d["s3"]
    p_gatv2 = s3["familias"]["gatv2_2_semobs"]["params_treinaveis"]
    p_mlp = s3["familias"]["mlp_semobs"]["params_treinaveis"]
    razao_artefato = s3["leitura"]["razao_mlp_sobre_gatv2"]
    razao_conferida = p_mlp / p_gatv2
    if abs(razao_artefato - razao_conferida) > 1e-9:
        raise RuntimeError(
            f"razao_mlp_sobre_gatv2 divergente: artefato={razao_artefato} "
            f"conferida={razao_conferida}")

    fato("jstars_params_gatv2", p_gatv2, "parametros", "s3",
         "familias.gatv2_2_semobs.params_treinaveis")
    fato("jstars_params_mlp", p_mlp, "parametros", "s3",
         "familias.mlp_semobs.params_treinaveis")
    fato("jstars_params_razao_mlp_sobre_gatv2", razao_artefato, None, "s3",
         "leitura.razao_mlp_sobre_gatv2",
         nota=f"conferida em codigo: {p_mlp}/{p_gatv2} = {razao_conferida:.10f}; "
              "MLP (controle) tem MAIS parametros que o GATv2")

    f2arv_cfg = d["f2arv"]["config"]
    tem_contagem_arvore = any(
        k in f2arv_cfg for k in
        ("n_estimators", "num_boost_round", "arvores_treinadas", "n_folhas"))
    fato("jstars_params_arvore_disponivel", tem_contagem_arvore, None, "f2arv",
         "config (chaves: " + ", ".join(sorted(f2arv_cfg.keys())) + ")",
         nota="nenhum artefato do projeto grava contagem de parametros ou "
              "folhas do XGBoost (results/f2_arvores.json.config so tem "
              "hiperparametros de treino: max_depth, learning_rate, "
              "min_child_weight, subsample, colsample_bytree); nao ha "
              "contagem da arvore para incluir")

    # ======================================================================
    # FACT 3 -- computational cost (auditar_recursos_s7.json)
    # ======================================================================
    s7 = d["s7"]
    fato("jstars_custo_estrategia", s7["estrategia"], None, "s7", "estrategia")
    hw = s7["hardware_da_maquina_de_auditoria"]
    fato("jstars_custo_hardware",
         {k: hw[k] for k in ("cpu", "ram_total_gb", "gpu", "vram_total_gb")},
         None, "s7", "hardware_da_maquina_de_auditoria.{cpu,ram_total_gb,gpu,vram_total_gb}",
         nota="declaracao do autor (D3): hardware da maquina em que ESTA "
              "auditoria rodou, nao medida das corridas originais de "
              "10-11/08 (que so registram 'dispositivo: cuda'); mesma "
              "ressalva usada no ACDSA para o mesmo artefato S7")
    for fam, nome in (("gatv2", "grafo"), ("mlp", "pontual")):
        f = s7["familias"][fam]
        fato(f"jstars_custo_{fam}_parede_b2_min", f["parede_b2_min"], "min",
             "s7", f"familias.{fam}.parede_b2_min",
             nota=f"{nome}: INICIO -> B2 CONCLUIDO, escopo igual nas duas familias")
        fato(f"jstars_custo_{fam}_treino_puro_min",
             f["treino_s_soma_media_min"], "min", "s7",
             f"familias.{fam}.treino_s_soma_media_min",
             nota="media das 5 sementes; exclui carga, uniao e predicao densa")
        fato(f"jstars_custo_{fam}_vram_pico_gb",
             {"valor": f["vram_pico_gb"], "onde": f["vram_pico_onde"]}, "GB",
             "s7", f"familias.{fam}.{{vram_pico_gb,vram_pico_onde}}")
        fato(f"jstars_custo_{fam}_rss_pico_gb", f["rss_pico_gb"], "GB", "s7",
             f"familias.{fam}.rss_pico_gb")
    fato("jstars_custo_epocas_teto", s7["epocas_teto"], "epocas", "s7",
         "epocas_teto", nota="teto por etapa da janela crescente, igual nas duas familias")

    acdsa_s7 = [f["id"] for f in d["acdsa"]["fatos"] if f["id"].startswith("s7_")]
    fato("jstars_custo_registro_acdsa", {"ids": acdsa_s7, "n": len(acdsa_s7)},
         None, "acdsa", "fatos[*].id (prefixo 's7_'), filtrado em codigo",
         nota="o ACDSA ja usa o mesmo auditar_recursos_s7.json para o "
              "paragrafo de custo computacional e a contagem de parametros; "
              "mesma fonte, mesma maquina de auditoria, mesmas ressalvas")

    # ======================================================================
    # FACT 4 -- physical admissibility (1,090 cells / 0.2; networks only)
    # ======================================================================
    tex_linhas = d["_main_tex_linhas"] if "_main_tex_linhas" in d else None
    main_tex_texto = ARQS["main_tex"].read_text(encoding="utf-8").splitlines()

    def _achar_trecho(alvo_linha_1based, padrao, janela=6):
        ini = max(0, alvo_linha_1based - 1 - janela)
        fim = min(len(main_tex_texto), alvo_linha_1based - 1 + janela)
        for i in range(ini, fim):
            if re.search(padrao, main_tex_texto[i]):
                return i + 1, main_tex_texto[i].strip()
        raise RuntimeError(
            f"main.tex: padrao {padrao!r} nao encontrado perto da linha {alvo_linha_1based}")

    ln60, tx60 = _achar_trecho(60, r"1,090 cells per replicate")
    ln153, tx153 = _achar_trecho(153, r"1,090 cells per")
    ln785, tx785 = _achar_trecho(785, r"1,090 cells per replicate")
    ln785b, tx785b = _achar_trecho(786, r"0\.2 cells")

    f2c = d["f2cel"]["admissibilidade"]
    x4_gatv2 = f2c["(100%, x4) gatv2"]["violacoes_acima_1m_media"]
    x1_gatv2 = f2c["(100%, x1) gatv2"]["violacoes_acima_1m_media"]

    fato("jstars_admissibilidade_x4_gatv2_1m", x4_gatv2, "celulas/replica",
         "f2cel", "admissibilidade.'(100%, x4) gatv2'.violacoes_acima_1m_media",
         nota=f"fonte numerica de 'about 1,090 cells per replicate': "
              f"main.tex:{ln60} \"{tx60}\"; main.tex:{ln153} \"{tx153}\"; "
              f"main.tex:{ln785} \"{tx785}\" (linhas localizadas por busca em codigo)")
    fato("jstars_admissibilidade_x1_gatv2_1m", x1_gatv2, "celulas/replica",
         "f2cel", "admissibilidade.'(100%, x1) gatv2'.violacoes_acima_1m_media",
         nota=f"fonte numerica de 'against 0.2 cells at the one-times cap': "
              f"main.tex:{ln785b} \"{tx785b}\" (linha localizada por busca em codigo)")

    xgb = d["d8"]["admissibilidade"]["xgb"]
    campos_none = [k for k, v in xgb.items() if v is None]
    campos_totais = [k for k in xgb if k != "nota"]
    if sorted(campos_none) != sorted(campos_totais):
        raise RuntimeError(
            f"admissibilidade.xgb nao esta totalmente nulo: presentes={campos_totais}, none={campos_none}")
    fato("jstars_admissibilidade_xgb_nulo", True, None, "d8",
         "admissibilidade.xgb (todos os campos numericos None, conferido em codigo)",
         nota=xgb.get("nota", "sem nota no artefato"))

    fato("jstars_admissibilidade_familias_medidas", ["gatv2", "mlp"], None,
         "f2cel",
         "derivado: chaves de admissibilidade com violacoes_acima_1m_media "
         "numerica em auditar_f2_celula.json, contra admissibilidade.xgb "
         "totalmente None em auditar_d8_triangulo.json",
         nota="a medida de admissibilidade fisica (violacoes acima de 1 m) "
              "so existe para as duas redes (grafo e pontual); a arvore "
              "(XGBoost) nao tem essa medida em nenhum artefato do projeto — "
              "f2_arvores.json nao grava o raster necessario (nota do "
              "proprio d8: 'E3, raster do XGBoost no eixo hidrologico, e o "
              "caminho')")

    # ======================================================================
    # FACT 5 -- GEDTM30: Table IV sampling method
    # ======================================================================
    script_txt = ARQS["laterais_script"].read_text(encoding="utf-8")
    m_metodo = re.search(
        r'def gedtm30_direto.*?return \{"metodo":\s*"([^"]+)"', script_txt, re.S)
    m_pixel = re.search(
        r'def gedtm30_direto.*?"pixel_nativo_graus":\s*([0-9.]+)', script_txt, re.S)
    if not (m_metodo and m_pixel):
        raise RuntimeError(
            "preparar_laterais_nativa_v3_diretas.py: nao achei metodo/pixel "
            "nativo em gedtm30_direto() por regex")
    metodo_gedtm30 = m_metodo.group(1)
    pixel_nativo = float(m_pixel.group(1))

    fato("jstars_gedtm30_metodo_amostragem", metodo_gedtm30, None,
         "laterais_script", "funcao gedtm30_direto(), campo de retorno 'metodo'",
         nota="extraido por regex do corpo da funcao que grava "
              "laterais_nativa_diretas/gedtm30_amazonia_*.npz, insumo direto "
              "de auditar_nmad_pareado_nativa_diretas.json (Tabela IV); "
              "em prosa: bilinear no reticulado (grade) de nos")
    fato("jstars_gedtm30_pixel_nativo_graus", pixel_nativo, "graus",
         "laterais_script", "funcao gedtm30_direto(), campo de retorno 'pixel_nativo_graus'")

    nmad_d = d["nmad_diretas"]
    laterais_usadas = nmad_d["entradas"]["laterais"]
    if "laterais_nativa_diretas" not in laterais_usadas:
        raise RuntimeError(
            f"auditar_nmad_pareado_nativa_diretas.json nao aponta para "
            f"laterais_nativa_diretas: {laterais_usadas}")
    pool_gedtm30_20_30 = nmad_d["tabela_pool_ETH"]["gedtm30"]["20-30m"]
    mae, vies, nmad = (pool_gedtm30_20_30["mae"], pool_gedtm30_20_30["vies"],
                       pool_gedtm30_20_30["nmad"])

    linha_tabela_iv = next(
        (ln for ln in main_tex_texto if "GEDTM30" in ln and "dagger" in ln), None)
    if not linha_tabela_iv:
        raise RuntimeError("main.tex: linha da Tabela IV com GEDTM30 nao encontrada")
    nums = re.findall(r'[-+]?\d+\.\d+', linha_tabela_iv.replace('$', ''))
    if len(nums) < 3:
        raise RuntimeError(f"main.tex: linha da Tabela IV sem numeros suficientes: {linha_tabela_iv!r}")
    tex_mae, tex_vies, tex_nmad = float(nums[0]), float(nums[1]), float(nums[2])
    confere = (round(mae, 2) == abs(tex_mae) and round(vies, 2) == abs(tex_vies)
               and round(nmad, 2) == abs(tex_nmad))

    fato("jstars_gedtm30_fonte_tabela_iv",
         {"artefato": "auditar_nmad_pareado_nativa_diretas.json",
          "entradas_laterais": laterais_usadas,
          "mae_20_30m": mae, "vies_20_30m": vies, "nmad_20_30m": nmad,
          "confere_com_main_tex_arredondado_2casas": confere},
         None, "nmad_diretas",
         "entradas.laterais + tabela_pool_ETH.gedtm30.'20-30m'.{mae,vies,nmad}",
         nota=f"conferencia em codigo contra a linha da Tabela IV em "
              f"main.tex ({linha_tabela_iv.strip()!r}): "
              f"mae {round(mae,2)} vs {abs(tex_mae)}, vies {round(vies,2)} "
              f"vs {abs(tex_vies)}, nmad {round(nmad,2)} vs {abs(tex_nmad)}")

    if not confere:
        raise RuntimeError(
            "jstars_gedtm30_fonte_tabela_iv: numeros do artefato nao "
            "conferem com a linha da Tabela IV em main.tex")

    # ======================================================================
    # FACT 6 -- ANADEM absent from the evaluated dataset
    # ======================================================================
    comando = ["find", "/trabalho/GNN_TOPO", "/arquivo", "-iname", "*anadem*"]
    r = subprocess.run(comando, capture_output=True, text=True, timeout=120)
    achados = [ln for ln in r.stdout.splitlines() if ln.strip()]
    ext_raster = (".tif", ".tiff", ".vrt", ".img", ".jp2", ".npz", ".parquet")
    achados_raster = [p for p in achados if p.lower().endswith(ext_raster)]

    ln95, tx95 = _achar_trecho(95, r"laipelt2024anadem")

    fato("jstars_anadem_busca",
         {"comando": " ".join(comando), "n_achados": len(achados),
          "achados_raster_ou_dado": achados_raster,
          "achados_bibliograficos": achados},
         None, "main_tex",
         "busca em disco (nao no manuscrito); citacao em main.tex:"
         f"{ln95} \"{tx95}\"",
         nota="nenhum arquivo de dado/raster com 'anadem' no nome em "
              "/trabalho/GNN_TOPO ou /arquivo; os achados sao todos material "
              "bibliografico (PDF/ficha/atalho .url) sobre o paper do ANADEM, "
              "nenhum e insumo do pipeline de treino ou avaliacao. Fato: "
              "ANADEM esta AUSENTE do conjunto avaliado (citado na Introducao, "
              "nao comparado na Tabela IV)")
    if achados_raster:
        raise RuntimeError(
            f"jstars_anadem_busca: achado(s) inesperado(s) de dado/raster: {achados_raster}")

    # ======================================================================
    saida = {
        "gerado_para": "Artigo 1 do V23 (alvo IEEE JSTARS, pacote_overleaf/main.tex)",
        "regra": ("todo numero citado no texto do paper tem linha nesta "
                  "folha; linha sem uso e permitida, numero sem linha NAO "
                  "EXISTE. Copiar da folha, conferir por diff contra o "
                  "artefato da linha."),
        "protocolo": (("Pre-declared protocol for this analysis (internal design note, not "
                       "distributed).")),
        "parte_A": {
            "n_fatos": len(fatos),
            "fatos": fatos,
            "qualidade_das_fontes": qualidade_fontes,
        },
        "parte_B": {
            "status": "pendente",
            "nota": ("reservado para as auditorias E4 (estimador e IC por "
                     "transecto do +6,65 m) e T3 (leitura do delta de escala "
                     "de 0,15 a 4,99 m sobre os 13 transectos), demandas 1-2 "
                     "do consolidado de 28/09. Nenhuma delas rodou ainda; "
                     "nenhum valor e inventado aqui. Outro passo acrescenta "
                     "esta secao quando os artefatos existirem."),
            "fatos": [],
        },
    }
    fontes_lidas = list(ARQS.values())
    PROV.gravar(RAIZ / a.saida, saida, fontes_lidas=fontes_lidas, script=__file__)
    log(f"{len(fatos)} fatos gravados em parte_A de {a.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
