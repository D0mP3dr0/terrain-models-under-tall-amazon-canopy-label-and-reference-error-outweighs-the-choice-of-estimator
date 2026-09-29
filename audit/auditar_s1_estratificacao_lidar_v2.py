"""S1 v2 -- fixes two methodological issues found in v1:

  (1) MISALIGNED (O) CEILING. v1 judged branch E's (O) criterion against
      a ceiling that includes external products (FABDEM/GEDTM30, read
      from auditar_rotulo_regua_v2.json -> 1.839 m at 20-30m, 2.957 m at
      >30m), while Lnb and C were judged against the two-network ceiling
      (v23 x mlp). The correct reading, declared BEFORE running, was that
      the ceiling should come from the families trained on the SAME
      label (v23 x mlp) -- products with different supervision make the
      ceiling circular. This script judges (O) for all THREE branches
      (E, Lnb, C) with the two-network ceiling recomputed IN EACH
      BRANCH'S OWN CLASS (E: 1.093 m at >30m and 0.356 m at 20-30m, from
      auditar_nmad_pareado_nativa_diretas.json,
      testes_principal_ETH["v23 vs mlp"].mae -- the SAME numbers v1 had
      already recomputed and stored to 1e-6, just not used to judge E).
      The ceiling with external products is kept as a descriptive field
      (`O_teto_produtos`), not deleted.

  (2) FALLBACK WITH NO BASIS IN THE DESIGN. v1's 20-30m H-S1 verdict came
      from a generic `else` ("no condition matched -> NAO_SE_SUSTENTA"),
      which ignores the design's own asymmetric rule: "a verdict that
      IMPROVES in Lnb or C (...) does not promote a new claim (...) it
      only enters as a sensitivity note" -- an improvement is not "does
      not hold". This script applies the criterion LITERALLY, with no
      fallback: (a) each criterion's (S)/(O)/(R) verdict in Lnb/C is
      compared to E's by ORDINAL RANKING (not_demonstrated < demonstrated;
      no_ordering < valid_ordering), not just equality -- an improvement
      does NOT break the "same verdict" condition of SE_SUSTENTA, but it
      is flagged and listed separately as sensitivity-only; a WORSENING
      does break it; (b) the NAO_SE_SUSTENTA trigger is ONLY what the
      design states (S not positive at >30m in Lnb or C, OR O at >30m not
      demonstrated in C) -- there is no equivalent trigger written for
      20-30m; (c) if nothing in the design's text matches, the verdict is
      "SEM_VEREDITO_LITERAL_NO_DESENHO" (honest -- it does not invent a
      category the design never wrote), NEVER an `else` defaulting to
      "does not hold".

WHAT DOES NOT CHANGE (reused from v1, imported, not rewritten): the sha256
check, the L/Lnb/C classification (same `montar_grade_vizinhos` /
`classificar_lnb_e_horn` functions), the v2 reproduction to 1e-6 (branch
E), the judgeable-forest population and the testes_principal/sensibilidade
guards to 1e-6, the recomputed (O) ceiling for Lnb and C (v23 x mlp), the
per-unit statistics (V2.estimando_unidade / V2.mae_reducao_unidade), and
the (S) and (R) criteria (they do not depend on the ceiling, so they do
not change). This script IMPORTS `auditar_s1_estratificacao_lidar.py`
(v1, ORIGINAL, UNCHANGED) and reuses those pieces; only E's (O) and the
H-S1 decision function are new.

NEW GUARD: the A, delta, and ceiling numbers for Lnb and C recomputed here
match the ones recorded in auditar_s1_estratificacao_lidar.json (v1) to
1e-9 -- fails loud if they don't match.

Usage:
  /trabalho/ambientes/s33_amb_virtual/.venv/bin/python auditar_s1_estratificacao_lidar_v2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import proveniencia as PROV                          # noqa: E402
import auditar_rotulo_regua_v2 as V2                  # noqa: E402 (reuso)
import auditar_nmad_pareado as NMD                    # noqa: E402 (reuso)
import auditar_s1_estratificacao_lidar as S1          # noqa: E402 (v1, INTACTO, so importado)

CONFIRMATORIAS = S1.CONFIRMATORIAS

V1_JSON = RAIZ / "auditar_s1_estratificacao_lidar.json"
SAIDA_JSON = RAIZ / "auditar_s1_estratificacao_lidar_v2.json"
SAIDA_DUMP = RAIZ / "auditar_s1_estratificacao_lidar_v2_dump.parquet"

# ordinal ranking of verdicts (for the asymmetric rule: improvement != degradation)
RANK_S = {"nao_sustentado": 0, "positivo": 1}
RANK_O = {"nao_demonstrado": 0, "sustentado": 1}
RANK_R = {"nenhuma_ordenacao": 0, "ordenacao_valida_A_menos_delta_maior_que_zero": 1}


def log(m: str = "") -> None:
    print(m, flush=True)


def comparar(mapa: dict, v_e: str, v_x: str) -> dict:
    """Compares one branch's (Lnb or C) verdict to E's by RANKING, not just
    equality: an improvement (higher rank) does not count as a divergence
    that breaks SE_SUSTENTA (the design's asymmetric rule), it is only
    flagged as sensitivity; a worsening (lower rank) does count."""
    if v_e not in mapa or v_x not in mapa:
        raise RuntimeError(f"veredito fora do ranking conhecido: E={v_e!r} X={v_x!r} mapa={mapa}")
    if v_x == v_e:
        return {"igual": True, "melhora": False, "piora": False}
    d = mapa[v_x] - mapa[v_e]
    return {"igual": False, "melhora": d > 0, "piora": d < 0}


def avaliar_hipotese_S1_v2(rot: str, E: dict, Lnb: dict, C: dict,
                           A_media_Lnb: float, A_ic95_E_transecto: list,
                           A_media_C: float) -> dict:
    """Applies the design's criterion LITERALLY, with no fallback. Returns
    the verdict, the reason (the design's own text), and the operands --
    never an `else` that returns NAO_SE_SUSTENTA by omission."""
    cmp_S_Lnb = comparar(RANK_S, E["S"]["veredito"], Lnb["S"]["veredito"])
    cmp_S_C = comparar(RANK_S, E["S"]["veredito"], C["S"]["veredito"])
    cmp_O_Lnb = comparar(RANK_O, E["O"]["veredito"], Lnb["O"]["veredito"])
    cmp_O_C = comparar(RANK_O, E["O"]["veredito"], C["O"]["veredito"])
    cmp_R_Lnb = comparar(RANK_R, E["R"]["veredito"], Lnb["R"]["veredito"])
    cmp_R_C = comparar(RANK_R, E["R"]["veredito"], C["R"]["veredito"])
    comparacoes = {"S_Lnb": cmp_S_Lnb, "S_C": cmp_S_C, "O_Lnb": cmp_O_Lnb,
                   "O_C": cmp_O_C, "R_Lnb": cmp_R_Lnb, "R_C": cmp_R_C}

    nenhuma_piora = not any(c["piora"] for c in comparacoes.values())
    sensibilidade_apenas = [nome for nome, c in comparacoes.items() if c["melhora"]]

    lo_E, hi_E = A_ic95_E_transecto
    dentro_ic_E = (lo_E is not None and hi_E is not None
                  and lo_E <= A_media_Lnb <= hi_E and lo_E <= A_media_C <= hi_E)

    # NAO_SE_SUSTENTA trigger: ONLY what the design states, and ONLY for
    # >30m ("the above-30m claim in the title falls"). There is no
    # equivalent trigger written for 20-30m -- none is invented here.
    if rot == ">30m":
        falha_S = Lnb["S"]["veredito"] != "positivo" or C["S"]["veredito"] != "positivo"
        falha_O_C = C["O"]["veredito"] == "nao_demonstrado"
        if falha_S or falha_O_C:
            return {"veredito": "NAO_SE_SUSTENTA",
                    "motivo": "(S) nao positivo em >30m em Lnb ou C, ou (O) nao demonstrado em C "
                              "em >30m -- regra literal do desenho: cai o above-30m do titulo",
                    "operandos": {"falha_S": falha_S, "falha_O_C": falha_O_C,
                                 "comparacoes": comparacoes}}

    if nenhuma_piora and dentro_ic_E:
        motivo = ("S/O/R iguais OU MELHORES que E em Lnb e C (nenhuma piora), e a media de A em "
                  "Lnb e em C cai no IC95 de E (transecto)")
        if sensibilidade_apenas:
            motivo += (f". Melhoras ({', '.join(sensibilidade_apenas)}) nao promovem claim novo "
                      "(regra assimetrica do desenho) -- entram so como sensibilidade, nao como "
                      "confirmatorio adicional")
        return {"veredito": "SE_SUSTENTA", "motivo": motivo,
                "operandos": {"comparacoes": comparacoes, "sensibilidade_apenas": sensibilidade_apenas,
                             "A_media_Lnb_m": A_media_Lnb, "A_media_C_m": A_media_C,
                             "ic95_E_transecto_m": A_ic95_E_transecto}}

    if nenhuma_piora and not dentro_ic_E:
        return {"veredito": "SE_SUSTENTA_COM_RESSALVA",
                "motivo": "nenhum criterio piora em Lnb ou C, mas a magnitude de A sai do IC95 de E "
                          "(transecto) -- a magnitude depende do estratificador",
                "operandos": {"comparacoes": comparacoes, "sensibilidade_apenas": sensibilidade_apenas,
                             "A_media_Lnb_m": A_media_Lnb, "A_media_C_m": A_media_C,
                             "ic95_E_transecto_m": A_ic95_E_transecto}}

    S_mantido = not cmp_S_Lnb["piora"] and not cmp_S_C["piora"]
    piora_Lnb = cmp_O_Lnb["piora"] or cmp_R_Lnb["piora"]
    piora_C = cmp_O_C["piora"] or cmp_R_C["piora"]
    so_um_piora = piora_Lnb != piora_C   # XOR: worsening in exactly one of the two decisive branches
    if S_mantido and so_um_piora:
        return {"veredito": "SE_SUSTENTA_COM_RESSALVA",
                "motivo": "(O) ou (R) piora em so um dos dois bracos decisivos (o outro mantem ou "
                          "melhora), com (S) mantido nos dois -- o claim fica restrito as celulas "
                          "concordantes",
                "operandos": {"comparacoes": comparacoes, "piora_Lnb": piora_Lnb, "piora_C": piora_C}}

    return {"veredito": "SEM_VEREDITO_LITERAL_NO_DESENHO",
            "motivo": ("none of the literal conditions of the design (SE_SUSTENTA / COM_RESSALVA "
                       "/ NAO_SE_SUSTENTA, the last one defined only for >30m) was satisfied -- "
                       "reported honestly instead of a fallback that would return 'nao se "
                       "sustenta' by default"),
            "operandos": {"comparacoes": comparacoes, "dentro_ic_E": dentro_ic_E,
                         "S_mantido": S_mantido, "piora_Lnb": piora_Lnb, "piora_C": piora_C}}


def main() -> int:
    log("=== conferencia previa de sha256 (ANTES de calcular) ===")
    shas = S1.conferir_sha256_entradas()
    shas[str(V1_JSON)] = PROV.sha256(V1_JSON)
    log(f"  sha256 {V1_JSON.name} = {shas[str(V1_JSON)][:16]}...")

    v1json = json.loads(V1_JSON.read_text(encoding="utf-8"))
    v2json_regua = json.loads(S1.V2_JSON.read_text(encoding="utf-8"))
    nmad_cache = json.loads(S1.NMAD_JSON.read_text(encoding="utf-8"))
    dump = pd.read_parquet(S1.V2_DUMP)
    nat = pd.read_parquet(S1.NATIVA_PARQUET)

    log("\n=== GUARDA E: reproduz auditar_rotulo_regua_v2.json a 1e-6 (reuso S1) ===")
    guarda_e = S1.guarda_E_reproduz_v2(dump, v2json_regua)

    log("\n=== reclassificacao L/Lnb/C (reuso S1.montar_grade_vizinhos/classificar_lnb_e_horn) ===")
    dump = dump.merge(
        nat[["quad", "idx_no", "transecto", "lin", "col", "z_p95_all", "z_solo_v2", "dossel_v2"]],
        on=["quad", "idx_no", "transecto"], how="left", validate="one_to_one")
    if int(dump["dossel_v2"].isna().sum()):
        raise RuntimeError("join dump x nativa: celulas sem geometria nativa")
    dump["classe_L"] = pd.cut(dump["dossel_v2"], S1.FAIXAS, labels=S1.ROT_F)
    viz = S1.montar_grade_vizinhos(nat)
    dump = S1.classificar_lnb_e_horn(dump, viz)
    dump["classe_C"] = np.where(dump["classe_eth"].astype(str) == dump["classe_lnb"].astype(str),
                                dump["classe_eth"].astype(str), np.nan)
    for c in ["classe_L", "classe_lnb"]:
        dump[c] = dump[c].astype(str).replace("nan", np.nan)

    log("\n=== populacao floresta julgavel + guardas testes_principal/sensibilidade (reuso S1) ===")
    flor, info, guarda_testes = S1.recomputar_populacao_e_guardas(shas)

    log("\n=== classificacao Lnb/C sobre a floresta julgavel + teto (O) recomputado (reuso S1) ===")
    viz_ids = viz[["quad", "idx_no", "transecto"] + [n for n, _, _ in S1.OFFSETS_HORN]]
    flor = flor.reset_index(drop=True)
    fmerge = flor[["quad", "idx_no", "transecto", "z_p95_all"]].merge(
        viz_ids, on=["quad", "idx_no", "transecto"], how="left", validate="one_to_one")
    zcols = [n for n, _, _ in S1.OFFSETS_HORN]
    z = fmerge[zcols].to_numpy(dtype=float)
    n_validos = np.sum(~np.isnan(z), axis=1)
    with np.errstate(invalid="ignore"):
        mediana_viz = np.nanmedian(z, axis=1)
    chao_nb = np.where(n_validos >= 3, mediana_viz, np.nan)
    dossel_lnb_flor = fmerge["z_p95_all"].to_numpy(dtype=float) - chao_nb
    flor["faixa_Lnb"] = pd.cut(dossel_lnb_flor, S1.FAIXAS, labels=S1.ROT_F)
    flor["faixa_C"] = np.where(flor["faixa_ex"].astype(str) == flor["faixa_Lnb"].astype(str),
                               flor["faixa_ex"].astype(str), np.nan)
    mat_ex = NMD.por_transecto_faixa(flor, "faixa_ex")
    mat_Lnb = NMD.por_transecto_faixa(flor, "faixa_Lnb")
    mat_C = NMD.por_transecto_faixa(flor, "faixa_C")
    teto_Lnb, _ = S1.teto_por_classe(mat_Lnb)
    teto_C, _ = S1.teto_por_classe(mat_C)

    log("\n=== teto (O) de E com o teto das duas redes (v23 x mlp), recomputado por classe ===")
    rerun_principal = NMD.testar(mat_ex, "v23", "mlp", None, "mae")
    teto_E_duas_redes = {}
    for rot in CONFIRMATORIAS:
        lo, hi = rerun_principal[rot]["ic95_bootstrap_transecto_mediana"]
        teto_E_duas_redes[rot] = max(abs(lo), abs(hi))
        lo_c, hi_c = nmad_cache["testes_principal_ETH"]["v23 vs mlp"]["mae"][rot]["ic95_bootstrap_transecto_mediana"]
        dif = max(abs(lo - lo_c), abs(hi - hi_c))
        if dif > 1e-6:
            raise RuntimeError(f"GUARDA teto_E_duas_redes FALHOU em {rot}: recomputado {[lo, hi]} "
                              f"!= cache {[lo_c, hi_c]} (diff {dif})")
        log(f"  {rot}: teto_E_duas_redes={teto_E_duas_redes[rot]:.6f} (cache: "
            f"{max(abs(lo_c), abs(hi_c)):.6f}, diff={dif:.3e})")

    log("\n=== estatisticas por braco (reuso S1.estatisticas_braco) ===")
    resultado_por_classe = {}
    guarda_v1 = {}
    for rot in CONFIRMATORIAS:
        est_Lnb = S1.estatisticas_braco(dump.assign(_classe_braco=dump["classe_lnb"]), rot)
        est_C = S1.estatisticas_braco(dump.assign(_classe_braco=dump["classe_C"]), rot)

        # --- GUARD: A/delta/ceilings match v1 ---
        alvo = v1json["resultado_por_classe"][rot]
        checagens_v1 = {}
        pares = [
            ("A_Lnb", est_Lnb["por_unidade"]["transecto"]["A"]["media_pond_celula_m"],
             alvo["estatisticas"]["Lnb"]["transecto"]["A"]["media_pond_celula_m"]),
            ("A_C", est_C["por_unidade"]["transecto"]["A"]["media_pond_celula_m"],
             alvo["estatisticas"]["C"]["transecto"]["A"]["media_pond_celula_m"]),
            ("delta_Lnb", est_Lnb["por_unidade"]["transecto"]["delta"]["media_pond_celula_m"],
             alvo["estatisticas"]["Lnb"]["transecto"]["delta"]["media_pond_celula_m"]),
            ("delta_C", est_C["por_unidade"]["transecto"]["delta"]["media_pond_celula_m"],
             alvo["estatisticas"]["C"]["transecto"]["delta"]["media_pond_celula_m"]),
            ("teto_Lnb", teto_Lnb[rot]["teto_m"], alvo["teto_O_m"]["Lnb_recomputado"]["teto_m"]),
            ("teto_C", teto_C[rot]["teto_m"], alvo["teto_O_m"]["C_recomputado"]["teto_m"]),
        ]
        ok_rot = True
        for nome, novo, v1_val in pares:
            dif = abs(novo - v1_val)
            ok = dif <= 1e-9
            ok_rot = ok_rot and ok
            checagens_v1[nome] = {"v2": novo, "v1": v1_val, "diferenca": dif, "passou": ok}
        if not ok_rot:
            raise RuntimeError(f"GUARDA numeros-batem-com-v1 FALHOU em {rot}: {checagens_v1}")
        guarda_v1[rot] = checagens_v1
        log(f"  guarda numeros=v1 {rot}: OK ({len(checagens_v1)} campos, tolerancia 1e-9)")

        S_Lnb = V2.veredito_S(est_Lnb["por_unidade"])
        R_Lnb = V2.veredito_R(est_Lnb["por_unidade"])
        O_Lnb = (V2.veredito_O(est_Lnb["por_unidade"], teto_Lnb[rot]["teto_m"])
                if teto_Lnb[rot]["testado"] else
                {"criterio": "(O) outweigh", "veredito": "nao_demonstrado",
                 "motivo": "teto Lnb nao testavel: " + str(teto_Lnb[rot].get("motivo"))})
        S_C = V2.veredito_S(est_C["por_unidade"])
        R_C = V2.veredito_R(est_C["por_unidade"])
        O_C = (V2.veredito_O(est_C["por_unidade"], teto_C[rot]["teto_m"])
              if teto_C[rot]["testado"] else
              {"criterio": "(O) outweigh", "veredito": "nao_demonstrado",
               "motivo": "teto C nao testavel: " + str(teto_C[rot].get("motivo"))})

        E_S = v2json_regua["criterios_veredito"][rot]["S"]
        E_R = v2json_regua["criterios_veredito"][rot]["R"]
        E_O_teto_produtos = v2json_regua["criterios_veredito"][rot]["O"]
        E_O_duas_redes = V2.veredito_O(v2json_regua["por_classe_eth"][rot]["por_unidade"],
                                       teto_E_duas_redes[rot])
        E_crit = {"S": E_S, "O": E_O_duas_redes, "R": E_R}

        A_ic95_E_transecto = v2json_regua["por_classe_eth"][rot]["por_unidade"]["transecto"]["A"]["ic95_pond_celula"]
        A_media_Lnb = est_Lnb["por_unidade"]["transecto"]["A"]["media_pond_celula_m"]
        A_media_C = est_C["por_unidade"]["transecto"]["A"]["media_pond_celula_m"]

        Lnb_crit = {"S": S_Lnb, "O": O_Lnb, "R": R_Lnb}
        C_crit = {"S": S_C, "O": O_C, "R": R_C}
        veredito_final = avaliar_hipotese_S1_v2(rot, E_crit, Lnb_crit, C_crit,
                                                A_media_Lnb, A_ic95_E_transecto, A_media_C)
        log(f"  {rot}: E(S={E_S['veredito']},O_duas_redes={E_O_duas_redes['veredito']},"
            f"O_produtos={E_O_teto_produtos['veredito']},R={E_R['veredito']}) "
            f"Lnb(S={S_Lnb['veredito']},O={O_Lnb['veredito']},R={R_Lnb['veredito']}) "
            f"C(S={S_C['veredito']},O={O_C['veredito']},R={R_C['veredito']}) "
            f"-> H-S1 v2: {veredito_final['veredito']}")

        resultado_por_classe[rot] = {
            "n_celulas": {"E": int((dump["classe_eth"] == rot).sum()),
                         "Lnb": est_Lnb["n_celulas"], "C": est_C["n_celulas"]},
            "teto_O_m": {"E_duas_redes_v23_mlp": teto_E_duas_redes[rot],
                        "E_teto_produtos_DESCRITIVO": E_O_teto_produtos.get("teto_abs_contrastes_lidar_m"),
                        "Lnb_recomputado": teto_Lnb[rot], "C_recomputado": teto_C[rot]},
            "criterios": {"E_com_teto_duas_redes": E_crit,
                         "E_com_teto_produtos_DESCRITIVO": {"S": E_S, "O": E_O_teto_produtos, "R": E_R},
                         "Lnb": Lnb_crit, "C": C_crit},
            "H_S1": veredito_final,
            "guarda_numeros_batem_com_v1": checagens_v1,
        }

    dump_saida = dump.copy()
    dump_saida.to_parquet(SAIDA_DUMP, index=False)
    log(f"\n  -> {SAIDA_DUMP.name} ({len(dump_saida):,} linhas)")

    fontes = [S1.V2_JSON, S1.V2_DUMP, S1.NATIVA_PARQUET, S1.NMAD_JSON, S1.JUIZ_JSON, V1_JSON,
             Path(__file__), RAIZ / "auditar_s1_estratificacao_lidar.py",
             RAIZ / "auditar_rotulo_regua_v2.py", RAIZ / "auditar_nmad_pareado.py",
             RAIZ / "juiz_lidar_v4.py"]

    PROV.gravar(SAIDA_JSON, {
        "hipotese": v1json["hipotese"],
        "correcao_desta_versao": {
            "defeito_1_teto_desalinhado": "v1 julgava (O) de E com o teto que inclui produtos "
                "(FABDEM/GEDTM30); Lnb e C usavam so v23xmlp. Este script julga os TRES bracos com "
                "o teto das duas redes (mesma familia de rotulo), por 'Decisao do teto (O)' em "
                "ata_2026-09-28_jstars_v23_E4_T3.md. O teto com produtos fica descritivo "
                "(O_teto_produtos), nao apagado.",
            "defeito_2_fallback_sem_base": "v1 tinha um else generico que devolvia NAO_SE_SUSTENTA "
                "quando nenhuma condicao literal batia (aconteceu em 20-30m, onde (O) MELHORA nos "
                "dois bracos decisivos). Este script compara cada criterio por RANKING ORDINAL "
                "(melhora != piora, regra assimetrica do desenho) e, se mesmo assim nenhuma condicao "
                "literal do desenho bater, devolve SEM_VEREDITO_LITERAL_NO_DESENHO em vez de inventar "
                "'nao se sustenta'.",
            "fonte_contra_auditoria": "_pareceres_2026-09-28_execucao/contra_auditoria_S1.md",
        },
        "nao_faz": "nao edita auditar_s1_estratificacao_lidar.py (v1, intacto), "
                  "auditar_rotulo_regua_v2.py, auditar_nmad_pareado.py nem juiz_lidar_v4.py (so "
                  "importa); nao interpreta o resultado para o artigo; nao audita o proprio "
                  "resultado; nao convoca ninguem",
        "conferencia_previa_sha256": shas,
        "guardas": {"E_reproduz_v2_1e-6": guarda_e,
                   "populacao_floresta_julgavel": {"n_recomputado": info["n_floresta_julgavel"]},
                   "testes_principal_e_sensibilidade_v23_vs_mlp_mae_1e-6": guarda_testes,
                   "teto_E_duas_redes_bate_com_cache_1e-6": {
                       rot: {"teto": teto_E_duas_redes[rot]} for rot in CONFIRMATORIAS},
                   "numeros_A_delta_teto_batem_com_v1_1e-9": guarda_v1},
        "resultado_por_classe": resultado_por_classe,
        "controle_negativo_permutacao_classe_lnb_por_transecto": v1json[
            "controle_negativo_permutacao_classe_lnb_por_transecto"],
        "classificacao_confusao_e_relevo_descritivos": v1json["classificacao"],
        "braco_L_descritivo": v1json["braco_L_descritivo"],
        "dump": {"arquivo": SAIDA_DUMP.name, "n_linhas": int(len(dump_saida)),
                "colunas": list(dump_saida.columns), "chave": ["quad", "idx_no", "transecto"]},
        "contra_auditoria": ("PENDING -- not yet checked by anyone other than the author of this "
                             "script (rule: no one audits their own result)"),
    }, fontes_lidas=fontes, script=__file__)
    log(f"\n  -> {SAIDA_JSON.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
