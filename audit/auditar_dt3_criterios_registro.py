"""D-T3 -- criteria (S), (O), (R) under alternative co-registrations.

QUESTION: do the (S), (O), and (R) verdicts of `auditar_rotulo_regua_v2.json`
-- computed only under GLO-30 registration -- change when the lidar
reference is co-registered to GEDTM30 (branch ii-b) or left unregistered
(branch iii), the two branches produced by
`auditar_s2_registro_referencia.py`?

WHAT CHANGES AND WHAT DOES NOT: dA/do_t = -1; fidelity and delta do not
depend on the registration. So, per anchor cell of transect t:
    A_arm      = A_i      - (off_arm[t] - off_i[t])
    e_modelo_arm = e_modelo_i - (off_arm[t] - off_i[t])      (e = A + fidelity)
    (A - delta)_arm = A_arm - delta
and the family ceiling (|dMAE| v23 vs mlp against the lidar) is RECOMPUTED
with z_ref = z_solo_v2 + off_arm[t] over the judgeable-forest population.

REUSE (modules READ, not edited; nothing reimplemented):
  auditar_rotulo_regua_v2 (ARV2): conferir_sha256_entradas, montar_unidade_pegada,
      estimando_unidade, mae_reducao_unidade, tabela_orcamento, contrastes_lidar,
      veredito_S, veredito_O, veredito_R
  auditar_nmad_pareado (REF): montar_populacao, guardas_populacao, testar
  auditar_nmad_pareado_nativa_anadem (ANA): por_transecto_faixa_generic
  auditar_s7_arvore_lidar (S7), ONLY if S7 has been produced with approved
      guards: conferir_sha, instalar_tree_em_montar, contrastes_por_unidade,
      criterio_O3

GUARDS (fail-loud; any failure -> RuntimeError, nothing is written):
  G0  sha256 of the inputs checked BEFORE computing anything (ARV2 + S2 +
      v2 fixed hashes).
  G0b off_t of branch (i) of S2 == the judge's offsets (13 judgeable
      transects, exact match).
  G1  GLO-30 registration: A, delta, A-delta, MAE reduction (all
      estimando/reducao keys), the P1 budget table, and the (S), (O), (R)
      verdicts in the 20-30m and >30m classes across the transect and
      footprint units reproduce auditar_rotulo_regua_v2.json to 1e-9; the
      v23-vs-mlp ceiling reproduces
      tabela_orcamento_P1[c].contrastes_lidar_detalhe["v23 vs mlp"] to 1e-9.
  G2  ii-b and iii: the weighted mean of A per class reproduces
      auditar_s2_registro_referencia.json efeito_sobre_A_por_classe[arm][c]
      .A_recomputado_m to 1e-6 (>30m: 9.898 / 6.774).
  G3  delta (mean, CI) is identical across the three registrations
      (invariant).
  GO3 (only with O3) under GLO-30, per-pair/per-unit ceilings and the O3
      verdict reproduce auditar_s7_arvore_lidar.json to 1e-9.

Usage:
  CUDA_VISIBLE_DEVICES="" /trabalho/ambientes/s33_amb_virtual/.venv/bin/python \
      auditar_dt3_criterios_registro.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import proveniencia as PROV                        # noqa: E402
import juiz_lidar_v4 as J                          # noqa: E402
import auditar_nmad_pareado as REF                 # noqa: E402  (read-only reference)
import auditar_nmad_pareado_nativa_anadem as ANA   # noqa: E402  (read-only reference)
import auditar_rotulo_regua_v2 as ARV2             # noqa: E402  (read-only reference)

# ------------------------------------------------------------------ entradas
PARQUET = Path("/trabalho/GNN_TOPO/SATELITES/lidar_eba/solo_lidar_30m_v3_nativa.parquet")
LAT_NATIVA = Path("/trabalho/GNN_TOPO/SATELITES/laterais_nativa_diretas")
LATDIR_TREINO = "/trabalho/GNN_TOPO/SATELITES/laterais"
JUIZ = RAIZ / "juiz_grade_nativa_diretas.json"
REGUA_JSON = RAIZ / "auditar_rotulo_regua_v2.json"
S2_JSON = RAIZ / "auditar_s2_registro_referencia.json"
S7_JSON = RAIZ / "auditar_s7_arvore_lidar.json"
S7_MANIFESTO = RAIZ / "treinar_arvore_persistir.json"
SAIDA = RAIZ / "auditar_dt3_criterios_registro.json"

SHA_FIXADO = {   # stamped hashes checked against the reference input files
    REGUA_JSON.name: "2470115aaf6eed5ff2ea7d878f2dd6a1ac3f7bf682240599102a33fb01129e9a",
    S2_JSON.name: "aa3f0f34b2965da9f879b060f2beccb39d994fb2c2ac7bbb8c5674e33742a7a2",
}

CLASSES = ["20-30m", ">30m"]
UNIDADES = ["transecto", "pegada"]
REGISTROS = {  # name -> key in S2.off_t_por_braco
    "i_glo30": "i_atual",
    "ii_b_gedtm30": "ii_b_gedtm30",
    "iii_sem_offset": "iii_sem_offset",
}
S2_ARM = {"ii_b_gedtm30": "ii-b", "iii_sem_offset": "iii"}
TOL_G1, TOL_G2 = 1e-9, 1e-6


def log(m=""):
    print(time.strftime("[%H:%M:%S] ") + str(m), flush=True)


# ------------------------------------------------------------- comparacao
def comparar(meu, alvo, tol: float, caminho: str, erros: list, cont: list) -> None:
    """Recursive comparison: numbers within `tol`, everything else by equality."""
    if isinstance(alvo, dict):
        if not isinstance(meu, dict):
            erros.append(f"{caminho}: tipo {type(meu).__name__} != dict"); return
        for k in alvo:
            if k not in meu:
                erros.append(f"{caminho}.{k}: ausente no recomputado"); continue
            comparar(meu[k], alvo[k], tol, f"{caminho}.{k}", erros, cont)
        for k in meu:
            if k not in alvo:
                erros.append(f"{caminho}.{k}: ausente no carimbado")
        return
    if isinstance(alvo, (list, tuple)):
        if not isinstance(meu, (list, tuple)) or len(meu) != len(alvo):
            erros.append(f"{caminho}: lista de tamanho diferente"); return
        for i, (a, b) in enumerate(zip(meu, alvo)):
            comparar(a, b, tol, f"{caminho}[{i}]", erros, cont)
        return
    if isinstance(alvo, bool) or alvo is None or isinstance(alvo, str):
        if meu != alvo:
            erros.append(f"{caminho}: {meu!r} != {alvo!r}")
        cont[0] += 1
        return
    if isinstance(alvo, (int, float)):
        fm, fa = float(meu), float(alvo)
        if np.isnan(fa):
            ok = np.isnan(fm)
        else:
            ok = abs(fm - fa) <= tol
        if not ok:
            erros.append(f"{caminho}: {fm!r} != {fa!r} (dif {fm - fa:.3e})")
        cont[0] += 1
        cont[1] = max(cont[1], 0.0 if np.isnan(fa) else abs(fm - fa))
        return
    erros.append(f"{caminho}: tipo nao previsto {type(alvo).__name__}")


def normal(x):
    """Round-trip through JSON to compare against the stamped value using the same type."""
    return json.loads(json.dumps(x, default=float))


# ------------------------------------------------------------- base (ARV2)
def carregar_base() -> dict:
    log("=== G0: sha256 conferido ANTES de calcular ===")
    shas = ARV2.conferir_sha256_entradas()
    for p in (REGUA_JSON, S2_JSON, JUIZ, PARQUET):
        h = PROV.sha256(p)
        if p.name in SHA_FIXADO and h != SHA_FIXADO[p.name]:
            raise RuntimeError(f"G0: {p.name} sha256 {h} != fixado {SHA_FIXADO[p.name]}. PARANDO.")
        shas[p.name] = h
    log(f"  G0: PASSOU ({len(shas)} entradas)")

    tabela3 = json.loads(ARV2.TABELA3_JSON.read_text(encoding="utf-8"))
    nmad = json.loads(ARV2.NMAD_JSON.read_text(encoding="utf-8"))
    d8 = json.loads(ARV2.D8_JSON.read_text(encoding="utf-8"))
    nativa_json = json.loads(ARV2.NATIVA_JSON.read_text(encoding="utf-8"))
    regua = json.loads(REGUA_JSON.read_text(encoding="utf-8"))
    s2 = json.loads(S2_JSON.read_text(encoding="utf-8"))
    juiz = json.loads(JUIZ.read_text(encoding="utf-8"))

    dump = pd.read_parquet(ARV2.DUMP_E4_PARQUET)
    nativa = pd.read_parquet(ARV2.NATIVA_PARQUET)

    # same guards (2) and (3) from v2, to guarantee the same base
    alvo_e4 = tabela3["por_classe_eth"][">30m"]["sem_piso"]["media_pond_celula_m"]
    recomp_e4 = float(dump.loc[dump["classe_eth"] == ">30m", "termo_rotulo"].mean())
    if abs(recomp_e4 - alvo_e4) > 1e-6:
        raise RuntimeError(f"guarda E4: {recomp_e4} != {alvo_e4}. PARANDO.")
    df = dump.merge(nativa[["transecto", "quad", "idx_no", "delta"]],
                    on=["transecto", "quad", "idx_no"], how="left", validate="one_to_one")
    alvo_delta = nativa_json["guarda_ii"]["valor"]
    recomp_delta = float(df.loc[df["classe_eth"] == ">30m", "delta"].mean())
    if abs(recomp_delta - alvo_delta) > 1e-4:
        raise RuntimeError(f"guarda delta nativo: {recomp_delta} != {alvo_delta}. PARANDO.")
    df["pareada_A_menos_delta"] = df["termo_rotulo"] - df["delta"]

    mapa_pegada, _pares = ARV2.montar_unidade_pegada(nativa)
    if mapa_pegada != regua["populacao"]["unidade_pegada"]["mapa"]:
        raise RuntimeError("mapa de pegada recomputado != auditar_rotulo_regua_v2.json. PARANDO.")
    df["pegada"] = df["transecto"].map(mapa_pegada)

    # G0b: off_t of branch (i) of S2 == the judge's offsets
    offs_juiz = {t: float(v["offset_m"]) for t, v in juiz["offsets_por_transecto"].items()
                 if v["julgavel"]}
    off_s2 = {k: {t: float(x) for t, x in s2["off_t_por_braco"][v].items()}
              for k, v in REGISTROS.items()}
    if off_s2["i_glo30"] != offs_juiz:
        raise RuntimeError("G0b: off_t (i) do S2 != offsets do juiz. PARANDO.")
    ts_dump = set(df["transecto"].unique())
    for k, o in off_s2.items():
        if not ts_dump <= set(o):
            raise RuntimeError(f"G0b: transectos do dump sem off_t no registro {k}: "
                               f"{sorted(ts_dump - set(o))}")
    log(f"  G0b: off_t (i) do S2 == juiz nos {len(offs_juiz)} julgaveis")
    return {"shas": shas, "tabela3": tabela3, "nmad": nmad, "d8": d8, "regua": regua,
            "s2": s2, "juiz": juiz, "df": df, "mapa_pegada": mapa_pegada, "off": off_s2}


# ------------------------------------------------------------- registro
def aplicar_registro(df: pd.DataFrame, off_i: dict, off_arm: dict) -> pd.DataFrame:
    """A and e_modelo shifted by -(off_arm - off_i) per transect; delta left intact."""
    d = df.copy()
    sh = d["transecto"].map({t: off_arm[t] - off_i[t] for t in off_i}).astype(np.float64)
    d["deslocamento_registro_m"] = sh
    d["termo_rotulo"] = d["termo_rotulo"] - sh
    d["e_modelo_m"] = d["e_modelo_m"] - sh
    d["pareada_A_menos_delta"] = d["termo_rotulo"] - d["delta"]
    return d


def estatisticas(df_arm: pd.DataFrame) -> dict:
    """Same calls and the same bootstrap tags as v2's main()."""
    out = {}
    for rot in CLASSES:
        s = df_arm[df_arm["classe_eth"] == rot]
        pu = {}
        for u in UNIDADES:
            pu[u] = {
                "A": ARV2.estimando_unidade(s, u, "termo_rotulo", f"{rot}|A|{u}"),
                "delta": ARV2.estimando_unidade(s, u, "delta", f"{rot}|delta|{u}"),
                "pareada_A_menos_delta": ARV2.estimando_unidade(
                    s, u, "pareada_A_menos_delta", f"{rot}|pareada|{u}"),
                "reducao_mae_equivalente": ARV2.mae_reducao_unidade(s, u, f"{rot}|reducao|{u}"),
            }
        out[rot] = pu
    return out


# ------------------------------------------------------------- teto familias
def z_ref_registro(flor: pd.DataFrame, off_arm: dict) -> pd.DataFrame:
    fl = flor.copy()
    fl["z_ref"] = fl["z_solo_v2"] + fl["transecto"].map(off_arm).astype(np.float64)
    return fl


def teto_familias(fl: pd.DataFrame, mapa_pegada: dict) -> dict:
    """Family ceiling: max(|lo|,|hi|) of the 95% CI of the median v23-mlp dMAE per
    unit (REF.testar; read via ARV2.contrastes_lidar), in the transect unit
    (primary) and the footprint unit (secondary)."""
    fl_p = fl.copy()
    fl_p["transecto"] = fl["transecto"].astype(str).map(mapa_pegada)
    if fl_p["transecto"].isna().any():
        raise RuntimeError("transecto sem pegada no mapa")
    out = {}
    for u, f in (("transecto", fl), ("pegada", fl_p)):
        mat = ANA.por_transecto_faixa_generic(f, "faixa_ex", ["v23", "mlp"])
        teste = REF.testar(mat, "v23", "mlp", None, "mae")
        out[u] = {}
        for c in CLASSES:
            linhas, teto = ARV2.contrastes_lidar({"testes_principal_ETH": {"v23 vs mlp": {"mae": teste}}}, c)
            det = next(r for r in linhas if r["par"] == "v23 vs mlp")
            no = teste[c]
            out[u][c] = {"teto_m": teto, "detalhe": det,
                         "diferencas_por_unidade_m": no.get("diferencas_por_transecto"),
                         "n_unidades": no.get("n_transectos"),
                         "excluidos_abaixo_do_piso": no.get("excluidos_abaixo_do_piso")}
    return out


# ------------------------------------------------------------- verdicts
def passa_unidade(crit: str, chk: dict) -> bool:
    if crit in ("S", "R"):
        return bool(chk["exclui_zero_percentil"] and chk["exclui_zero_t_jackknife"])
    return bool(chk["acima_do_teto_media"] and chk["acima_do_teto_mediana"]
                and chk["acima_do_teto_reducao"])


def rotulo_O(v: str) -> str:
    return {"sustentado": "demonstrado", "nao_demonstrado": "nao demonstrado"}.get(v, v)


def criterios_registro(pu_c: dict, teto: dict, c: str) -> dict:
    S = ARV2.veredito_S(pu_c)
    R = ARV2.veredito_R(pu_c)
    O_prim = ARV2.veredito_O(pu_c, teto["transecto"][c]["teto_m"])
    O_peg = ARV2.veredito_O(pu_c, teto["pegada"][c]["teto_m"])
    chk_sec = {"transecto": O_prim["checagens_por_unidade"]["transecto"],
               "pegada": O_peg["checagens_por_unidade"]["pegada"]}
    ok_sec = all(passa_unidade("O", chk_sec[u]) for u in UNIDADES)
    return {
        "S": S, "R": R,
        "O_familias_28_09": {**O_prim, "veredito_leitura": rotulo_O(O_prim["veredito"]),
                             "teto_de": "v23 x mlp, unidade transecto, aplicado as duas unidades "
                                        "(leitura decidida em 28/09)"},
        "O_familias_teto_por_unidade_secundario": {
            "teto_transecto_m": teto["transecto"][c]["teto_m"],
            "teto_pegada_m": teto["pegada"][c]["teto_m"],
            "checagens_por_unidade": chk_sec,
            "veredito_leitura": "demonstrado" if ok_sec else "nao demonstrado",
            "nota": "secundario, declarado antes de rodar; nao substitui o primario"},
    }


# ------------------------------------------------------------- O3 (S7)
def s7_aprovado() -> tuple[bool, str, dict | None]:
    if not S7_JSON.exists():
        return False, "O3 nao aplicado: S7 ausente (auditar_s7_arvore_lidar.json nao existe)", None
    s7 = json.loads(S7_JSON.read_text(encoding="utf-8"))
    g = s7.get("guardas", {})
    ok_i = str(g.get("g2_i", {}).get("status", "")).startswith("OK")
    ok_ii = all(g.get("g2_ii", {}).get(c, {}).get("passou") is True for c in CLASSES)
    man_ok = (S7_MANIFESTO.exists()
              and json.loads(S7_MANIFESTO.read_text(encoding="utf-8")).get("status") == "OK")
    if not (ok_i and ok_ii and man_ok):
        return False, (f"O3 nao aplicado: S7 sem guardas aprovadas (g2_i={ok_i}, g2_ii={ok_ii}, "
                       f"manifesto_treino_OK={man_ok})"), s7
    return True, "O3 aplicado: auditar_s7_arvore_lidar.json com guardas aprovadas", s7


# --------------------------------------------------------------------- main
def main() -> int:
    t0 = time.time()
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise SystemExit('CUDA_VISIBLE_DEVICES="" e obrigatorio (nada toca GPU)')
    base = carregar_base()
    df, regua, off = base["df"], base["regua"], base["off"]
    mapa_pegada = base["mapa_pegada"]

    usa_o3, motivo_o3, s7 = s7_aprovado()
    log(f"  {motivo_o3}")
    S7 = None
    if usa_o3:
        os.environ["V23_LATDIR"] = LATDIR_TREINO     # required by the S7 import (admissibility)
        import auditar_s7_arvore_lidar as S7         # noqa: E402  (read-only reference)
        base["shas"].update({f"s7:{k}": v for k, v in S7.conferir_sha(True).items()})
        base["shas"][S7_JSON.name] = PROV.sha256(S7_JSON)
        S7.instalar_tree_em_montar()

    log("=== populacao floresta-julgavel (teto) ===")
    J.LAT = LAT_NATIVA
    flor, info = REF.montar_populacao(PARQUET, base["juiz"])
    conf = REF.guardas_populacao(flor, base["juiz"])
    log(f"  {info['n_floresta_julgavel']} celulas, {info['n_julgaveis']} transectos; "
        f"tabela_regua_k0 reproduzida em {len(conf)} celulas")

    registros = {}
    guardas = {}
    for nome in REGISTROS:
        log(f"\n=== registro {nome} ===")
        d_arm = aplicar_registro(df, off["i_glo30"], off[nome])
        est = estatisticas(d_arm)
        fl = z_ref_registro(flor, off[nome])
        if nome == "i_glo30":
            dif_z = float(np.nanmax(np.abs(fl["z_ref"].to_numpy() - flor["z_ref"].to_numpy())))
            if dif_z != 0.0:
                raise RuntimeError(f"z_ref (i) recomputado difere do de REF em {dif_z}. PARANDO.")
        teto = teto_familias(fl, mapa_pegada)
        crit = {c: criterios_registro(est[c], teto, c) for c in CLASSES}
        reg = {"off_t_m": off[nome],
               "deslocamento_vs_glo30_por_transecto_m": {t: off[nome][t] - off["i_glo30"][t]
                                                         for t in off[nome]},
               "por_classe": {c: {"por_unidade": est[c],
                                  "teto_familias_v23_x_mlp": {u: teto[u][c] for u in UNIDADES},
                                  "criterios": crit[c]} for c in CLASSES}}
        for c in CLASSES:
            a_t = est[c]["transecto"]["A"]
            log(f"  {c}: A pond {a_t['media_pond_celula_m']:.4f} IC{[round(x, 3) for x in a_t['ic95_pond_celula']]} "
                f"| teto transecto {teto['transecto'][c]['teto_m']:.4f} pegada {teto['pegada'][c]['teto_m']:.4f} "
                f"| S={crit[c]['S']['veredito']} O={crit[c]['O_familias_28_09']['veredito_leitura']} "
                f"(sec {crit[c]['O_familias_teto_por_unidade_secundario']['veredito_leitura']}) "
                f"R={crit[c]['R']['veredito']}")

        if usa_o3:
            res, _mats = S7.contrastes_por_unidade(fl, mapa_pegada)
            pseudo_regua = {"por_classe_eth": {c: {"por_unidade": est[c]} for c in CLASSES}}
            o3 = S7.criterio_O3(res, pseudo_regua)
            reg["O3"] = {"contrastes_lidar_mae": {u: {p: res[u][p]["mae"] for p in res[u]}
                                                  for u in res},
                         "criterio": o3}
            for c in CLASSES:
                log(f"  O3 {c}: {o3[c]['veredito_O3']}")

        # ---------------- guards per registration
        if nome == "i_glo30":
            log("  --- G1: reproduz auditar_rotulo_regua_v2.json a 1e-9 ---")
            erros, cont = [], [0, 0.0]
            for c in CLASSES:
                for u in UNIDADES:
                    for k in ("A", "delta", "pareada_A_menos_delta", "reducao_mae_equivalente"):
                        comparar(normal(est[c][u][k]), regua["por_classe_eth"][c]["por_unidade"][u][k],
                                 TOL_G1, f"{c}.{u}.{k}", erros, cont)
                orc = ARV2.tabela_orcamento(c, est[c], base["nmad"], base["nmad"]["tabela_pool_ETH"],
                                            base["d8"])
                comparar(normal(orc), regua["tabela_orcamento_P1"][c], TOL_G1,
                         f"orcamento.{c}", erros, cont)
                crit_v2 = {"S": crit[c]["S"], "R": crit[c]["R"],
                           "O": ARV2.veredito_O(est[c], orc["maior_teto_abs_contrastes_lidar_m"])}
                comparar(normal(crit_v2), regua["criterios_veredito"][c], TOL_G1,
                         f"criterios.{c}", erros, cont)
                reg["por_classe"][c]["criterios_v2_teto_com_produtos_reproducao"] = crit_v2
                alvo_det = next(r for r in regua["tabela_orcamento_P1"][c]["contrastes_lidar_detalhe"]
                                if r["par"] == "v23 vs mlp")
                comparar(normal(teto["transecto"][c]["detalhe"]), alvo_det, TOL_G1,
                         f"teto_v23_mlp.{c}", erros, cont)
            if erros:
                raise RuntimeError("G1 FALHOU (registro GLO-30 nao reproduz o v2 a 1e-9): "
                                   + "; ".join(erros[:20]) + f" ... ({len(erros)} erros). PARANDO.")
            guardas["G1_glo30_reproduz_v2_1e-9"] = {"campos_conferidos": cont[0],
                                                    "max_abs_dif": cont[1], "passou": True}
            log(f"  G1: PASSOU ({cont[0]} campos, max |dif| {cont[1]:.2e})")
            if usa_o3:
                erros, cont = [], [0, 0.0]
                for c in CLASSES:
                    for u in UNIDADES:
                        for campo in ("tetos_por_par_m", "teto_O3_m", "par_que_fixa_o_teto",
                                      "passa_na_unidade"):
                            comparar(normal(o3[c]["por_unidade"][u][campo]),
                                     s7["O3"][c]["por_unidade"][u][campo], TOL_G1,
                                     f"O3.{c}.{u}.{campo}", erros, cont)
                    comparar(o3[c]["veredito_O3"], s7["O3"][c]["veredito_O3"], TOL_G1,
                             f"O3.{c}.veredito", erros, cont)
                if erros:
                    raise RuntimeError("GO3 FALHOU (O3 em GLO-30 nao reproduz o S7 a 1e-9): "
                                       + "; ".join(erros[:20]) + ". PARANDO.")
                guardas["GO3_glo30_reproduz_s7_1e-9"] = {"campos_conferidos": cont[0],
                                                         "max_abs_dif": cont[1], "passou": True}
                log(f"  GO3: PASSOU ({cont[0]} campos)")
        else:
            arm = S2_ARM[nome]
            g2 = {}
            for c in CLASSES:
                alvo = base["s2"]["efeito_sobre_A_por_classe"][arm][c]["A_recomputado_m"]
                n_alvo = base["s2"]["efeito_sobre_A_por_classe"][arm][c]["n"]
                meu = est[c]["transecto"]["A"]["media_pond_celula_m"]
                n_meu = est[c]["transecto"]["A"]["n_celulas"]
                dif = meu - alvo
                ok = abs(dif) <= TOL_G2 and n_meu == n_alvo
                g2[c] = {"recomputado_m": meu, "s2_m": alvo, "diferenca_m": dif, "n": n_meu,
                         "n_s2": n_alvo, "tolerancia_m": TOL_G2, "passou": bool(ok)}
                log(f"  G2 {arm} {c}: A {meu:.9f} vs S2 {alvo:.9f} dif {dif:.2e} n {n_meu}/{n_alvo} "
                    f"-> {'PASSOU' if ok else 'FALHOU'}")
                if not ok:
                    raise RuntimeError(f"G2 FALHOU em {arm}/{c}: A {meu} != S2 {alvo} "
                                       f"(dif {dif}, n {n_meu}/{n_alvo}). PARANDO.")
            guardas[f"G2_{nome}_reproduz_S2_A_1e-6"] = g2
        registros[nome] = reg

    # G3: delta invariante
    erros, cont = [], [0, 0.0]
    for nome in ("ii_b_gedtm30", "iii_sem_offset"):
        for c in CLASSES:
            for u in UNIDADES:
                comparar(normal(registros[nome]["por_classe"][c]["por_unidade"][u]["delta"]),
                         normal(registros["i_glo30"]["por_classe"][c]["por_unidade"][u]["delta"]),
                         0.0, f"delta.{nome}.{c}.{u}", erros, cont)
    if erros:
        raise RuntimeError("G3 FALHOU (delta mudou com o registro): " + "; ".join(erros[:10]))
    guardas["G3_delta_invariante_ao_registro"] = {"campos_conferidos": cont[0], "passou": True}
    log(f"\n  G3: delta identico nos tres registros ({cont[0]} campos)")

    # ---------------- verdict table and pre-registered reading
    tabela = []
    for nome in REGISTROS:
        for c in CLASSES:
            cr = registros[nome]["por_classe"][c]["criterios"]
            for u in UNIDADES:
                linha = {"registro": nome, "classe": c, "unidade": u,
                         "S_passa": passa_unidade("S", cr["S"]["checagens_por_unidade"][u]),
                         "O_familias_28_09_passa": passa_unidade(
                             "O", cr["O_familias_28_09"]["checagens_por_unidade"][u]),
                         "O_teto_por_unidade_passa": passa_unidade(
                             "O", cr["O_familias_teto_por_unidade_secundario"]["checagens_por_unidade"][u]),
                         "R_passa": passa_unidade("R", cr["R"]["checagens_por_unidade"][u])}
                if usa_o3:
                    linha["O3_passa"] = registros[nome]["O3"]["criterio"][c]["por_unidade"][u]["passa_na_unidade"]
                tabela.append(linha)

    def vered(nome, c, crit):
        cr = registros[nome]["por_classe"][c]["criterios"]
        if crit == "S":
            return cr["S"]["veredito"]
        if crit == "R":
            return cr["R"]["veredito"]
        if crit == "O":
            return cr["O_familias_28_09"]["veredito_leitura"]
        if crit == "O_sec":
            return cr["O_familias_teto_por_unidade_secundario"]["veredito_leitura"]
        if crit == "O3":
            return registros[nome]["O3"]["criterio"][c]["veredito_O3"]

    crits = ["S", "O", "O_sec", "R"] + (["O3"] if usa_o3 else [])
    estabilidade = {}
    for crit in crits:
        estabilidade[crit] = {}
        for c in CLASSES:
            v = {nome: vered(nome, c, crit) for nome in REGISTROS}
            ref_v = v["i_glo30"]
            muda = [n for n in REGISTROS if v[n] != ref_v]
            estabilidade[crit][c] = {"vereditos": v, "igual_nos_tres": not muda,
                                     "registros_que_mudam_o_veredito": muda}

    o_30 = {nome: vered(nome, ">30m", "O") for nome in REGISTROS}
    falham = [n for n, v in o_30.items() if v != "demonstrado"]
    leitura = {
        "regra_literal": "'O titulo nao depende do registro' so se o veredito (O) acima de 30 m for "
                         "'demonstrado' nos tres registros, nas duas unidades. Senao o texto nomeia o "
                         "registro sob o qual falha. Vale o mesmo para (S) e (R) por classe.",
        "O_maior30_por_registro": o_30,
        "titulo_nao_depende_do_registro": "sim" if not falham else "nao",
        "registros_em_que_falha": falham,
        "S_R_por_classe": {k: estabilidade[k] for k in ("S", "R")},
        "O_20_30_por_registro": estabilidade["O"]["20-30m"]["vereditos"],
        "secundario_teto_por_unidade_O_maior30": estabilidade["O_sec"][">30m"],
    }
    if usa_o3:
        o3_30 = {n: vered(n, ">30m", "O3") for n in REGISTROS}
        f3 = [n for n, v in o3_30.items() if v != "demonstrado para as tres familias"]
        leitura["O3_maior30_por_registro"] = o3_30
        leitura["O3_titulo_nao_depende_do_registro"] = "sim" if not f3 else "nao"
        leitura["O3_registros_em_que_falha"] = f3
    else:
        leitura["O3"] = motivo_o3

    for nome in REGISTROS:
        log(f"  {nome}: " + " | ".join(
            f"{c}: S={vered(nome, c, 'S')}, O={vered(nome, c, 'O')}, O_sec={vered(nome, c, 'O_sec')}, "
            f"R={vered(nome, c, 'R')}" + (f", O3={vered(nome, c, 'O3')}" if usa_o3 else "")
            for c in CLASSES))
    log(f"  LEITURA: titulo nao depende do registro = {leitura['titulo_nao_depende_do_registro']} "
        f"(falha em {falham})")

    fontes = [ARV2.TABELA3_JSON, ARV2.DUMP_E4_PARQUET, ARV2.NATIVA_JSON, ARV2.NATIVA_PARQUET,
              ARV2.NMAD_JSON, ARV2.D8_JSON, REGUA_JSON, S2_JSON, JUIZ, PARQUET,
              RAIZ / "auditar_rotulo_regua_v2.py", RAIZ / "auditar_s2_registro_referencia.py",
              RAIZ / "auditar_nmad_pareado.py", RAIZ / "auditar_nmad_pareado_nativa_anadem.py",
              RAIZ / "juiz_lidar_v4.py", RAIZ / "proveniencia.py", Path(__file__)]
    fontes += [J.SUP / f"delta_amazonia_ampliacao_b4s035{arm}_seed{sd}.npz"
               for arm in ("", "_mlp_semobs") for sd in J.SEEDS_PAR]
    fontes += [LAT_NATIVA / f"{n}_amazonia_{q}.npz" for q in J.QUADS
               for n in ("glo30", "fabdem", "gedtm30", "agua_jrc", "dossel_eth", "rotulos")]
    if usa_o3:
        fontes += [S7_JSON, S7_MANIFESTO, RAIZ / "auditar_s7_arvore_lidar.py"]
        fontes += [S7.npz_tree(sd) for sd in J.SEEDS_PAR]

    corpo = {
        "desenho": ("Pre-declared design note, section D-T3 (closing rule: runs once; the "
                    "result goes into the text as it comes out)"),
        "diario": "artigo_v23_2/_pareceres_2026-09-28_rodada4/oficina_DT3.md",
        "comando": " ".join([sys.executable] + sys.argv),
        "conferencia_sha256_antes_de_calcular": base["shas"],
        "operacionalizacao": {
            "registros": {k: f"auditar_s2_registro_referencia.json off_t_por_braco.{v}"
                          for k, v in REGISTROS.items()},
            "A_e_e_modelo": "deslocados por -(off_arm[t] - off_i[t]) por celula-ancora (dA/do=-1); "
                            "delta intacto; A-delta = A_arm - delta",
            "estatistica": "ARV2.estimando_unidade / mae_reducao_unidade com as MESMAS tags de "
                           "bootstrap do v2 em todos os registros (mesmos sorteios de unidades); "
                           "B=10000, semente 42, cluster por unidade",
            "teto_familias": "max(|lo|,|hi|) do IC95 bootstrap da mediana da dMAE v23-mlp por "
                             "unidade (REF.testar; piso 200 celulas, >=6 unidades), populacao "
                             "floresta-julgavel de REF.montar_populacao com z_ref = z_solo_v2 + "
                             "off_arm; primario = teto do transecto nas duas unidades (28/09); "
                             "secundario = teto recalculado na unidade pegada",
            "O3": motivo_o3,
        },
        "guardas": guardas,
        "populacao_teto": info,
        "registros": registros,
        "tabela_vereditos_registro_classe_unidade": tabela,
        "estabilidade_dos_vereditos": estabilidade,
        "leitura_pre_declarada": leitura,
        "nao_faz": "nao edita nenhum script/JSON existente; nao interpreta para o artigo; "
                   "nao audita o proprio resultado; nao abre teste novo",
        "contra_auditoria": "PENDENTE -- nao feita por quem escreveu este script",
        "tempo_total_s": round(time.time() - t0, 1),
    }
    PROV.gravar(SAIDA, corpo, fontes_lidas=fontes, script=__file__)
    log(f"  -> {SAIDA.name} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
