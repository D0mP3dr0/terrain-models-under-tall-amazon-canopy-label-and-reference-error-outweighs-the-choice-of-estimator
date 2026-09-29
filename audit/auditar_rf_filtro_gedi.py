"""Label-reference term A recomputed under the literature-standard GEDI shot filter.

Recomputes, for the 1,194 anchor cells (auditar_tabela3_dump.parquet),
the per-cell median orthometric ground of the approved GEDI + ATL08 shots under
three GEDI filters (ATL08 unchanged):
  F0  adopted: quality_flag == 1, num_modos >= 1, plus the biome plausibility cut
  F1  F0 and degrade_flag == 0 and sensitivity >= 0.95
  F2  F0 and degrade_flag == 0 and sensitivity >= 0.98 and power beam and night
and sets A_Fk(cell) = A_adopted(cell) + [median_Fk(ground) - median_F0(ground)].
Per arm and ETH class (20-30 m, >30 m) it reports n cells / transects, the
population floor, A (cell-weighted mean, median), the equivalent MAE reduction,
95% transect-cluster bootstrap CIs (B = 10,000, seed 42, functions imported from
auditar_rotulo_regua_v2), the verdict against the three-family estimator gap
(O3, read from auditar_s7_arvore_lidar.json, not recomputed) and the per-cell shift.

Guards (RuntimeError with the measured value; nothing is tuned and re-run):
  G1  F0 rebuild reproduces the stored label: |DSM - median_F0 - y_novo| <= 0.01 m
      in >= 99% of the 1,194 cells and no cell above 0.5 m.
  G2  F0, >30 m: A mean 6.651019321277299, median 3.699343681335449, 441 cells,
      10 transects, to 1e-6, by the same code.
  G3  reference gap is 1.442 m (>30 m) and 0.500 m (20-30 m), read, not recomputed.
Imported modules are only read; the script writes only its output JSON.

Usage
  python auditar_rf_filtro_gedi.py [--saida auditar_rf_filtro_gedi.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import proveniencia as PROV                 # noqa: E402
import rotulos_lidar_v23 as ROT             # noqa: E402  (preparar_gedi/atl08, celula, limites_do_bioma)
import auditar_rotulo_regua_v2 as RR        # noqa: E402  (estimando_unidade, mae_reducao_unidade, B/SEED)
from geoide_v23 import undulacao            # noqa: E402

BIOMA = "amazonia"
SAT = Path("/trabalho/GNN_TOPO/SATELITES")
DUMP_E4 = RAIZ / "auditar_tabela3_dump.parquet"
REGUA_JSON = RAIZ / "auditar_rotulo_regua_v2.json"
S7_JSON = RAIZ / "auditar_s7_arvore_lidar.json"

CLASSES = ["20-30m", ">30m"]
# ---- Constants fixed before the analysis (not tuned).
F1_SENS, F2_SENS = 0.95, 0.98
FEIXES_POTENCIA = ("BEAM0101", "BEAM0110", "BEAM1000", "BEAM1011")
G1_TOL_M, G1_FRAC_MIN, G1_MAX_M = 0.01, 0.99, 0.5
G2_ALVO = {"media_pond_celula_m": 6.651019321277299, "mediana_m": 3.699343681335449,
           "n_celulas": 441, "n_unidades": 10}
G2_TOL = 1e-6
G3_ALVO = {">30m": 1.442, "20-30m": 0.500}      # checked to 3 decimals
PISO_CEL, PISO_TRANS = 110, 6                   # 25% of 441 cells, >= 6 transects (>30 m class only)


def log(m: str = "") -> None:
    print(m, flush=True)


class _PainelMudo:
    def log(self, m=""):
        log("    [rotulos_lidar_v23] " + str(m))


# ------------------------------------------------------------------- G3
def guarda_g3(s7: dict) -> dict:
    out = {}
    for c in CLASSES:
        v = s7["O3"][c]["por_unidade"]["transecto"]["teto_O3_m"]
        ok = round(float(v), 3) == G3_ALVO[c]
        out[c] = {"teto_O3_m_lido": v, "alvo_3_casas": G3_ALVO[c], "passou": bool(ok)}
        if not ok:
            raise RuntimeError(f"G3 FALHOU: teto O3 {c} lido {v!r} != {G3_ALVO[c]} (3 casas)")
    log(f"  G3 PASSOU: teto >30m {out['>30m']['teto_O3_m_lido']!r}, "
        f"20-30m {out['20-30m']['teto_O3_m_lido']!r}")
    return out


# ---------------------------------------------------------------- shots
def montar_disparos(alvos: Path, gee: Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Same path as rotulos_lidar_v23.montar_bioma (l. 493-578) up to the
    plausibility cut, keeping the GEDI filter fields next to each shot."""
    import rasterio
    ROT.SAIDA = alvos
    painel = _PainelMudo()
    g = ROT.preparar_gedi(BIOMA, painel)
    a = ROT.preparar_atl08(BIOMA, painel)
    if g is None or a is None:
        raise RuntimeError("GEDI ou ATL08 ausente em " + str(alvos))

    # GEDI filter fields, aligned with preparar_gedi's `ok` mask (l. 286, 334)
    arq = alvos / f"gedi02_a_{BIOMA}_dedup.parquet"
    raw = pd.read_parquet(arq, columns=["lat", "lon", "elev_solo", "elev_topo",
                                        "degrade_flag", "sensitivity",
                                        "solar_elevation", "feixe"])
    n_gedi_bruto = len(raw)
    ok = (ROT._valido(raw["elev_solo"].to_numpy(np.float64))
          & ROT._valido(raw["elev_topo"].to_numpy(np.float64)))
    raw = raw[ok].reset_index(drop=True)
    if not (len(raw) == len(g)
            and np.array_equal(raw["lat"].to_numpy(np.float64), g["lat"].to_numpy())
            and np.array_equal(raw["lon"].to_numpy(np.float64), g["lon"].to_numpy())):
        raise RuntimeError("alinhamento GEDI bruto x preparar_gedi falhou "
                           f"({len(raw)} x {len(g)})")
    for col in ("degrade_flag", "sensitivity", "solar_elevation", "feixe"):
        g[col] = raw[col].to_numpy()
    del raw

    d = pd.concat([g, a], ignore_index=True)
    dsm_arq = gee / f"glo30_{BIOMA}.tif"
    with rasterio.open(dsm_arq) as src:
        if src.width != ROT.GRID or src.height != ROT.GRID:
            raise RuntimeError(f"{dsm_arq.name} {src.width}x{src.height} != {ROT.GRID}")
        dsm = src.read(1).astype(np.float32)

    bb = ROT.BIOMAS[BIOMA]
    r, c = ROT.celula(d["lat"].to_numpy(), d["lon"].to_numpy(), bb)
    d = d.assign(r=r, c=c)
    d = d[d.r >= 0].reset_index(drop=True)
    z = dsm[d.r.to_numpy(), d.c.to_numpy()]
    d["y_bruto"] = z.astype(np.float64) - d["solo"].to_numpy()
    d = d[np.isfinite(d.y_bruto.to_numpy())].reset_index(drop=True)
    N = undulacao(d.lon.to_numpy(), d.lat.to_numpy())
    d["n_geoide"] = N
    d["solo_orto"] = d["solo"].to_numpy() - N
    z2 = dsm[d.r.to_numpy(), d.c.to_numpy()].astype(np.float64)
    d["y_novo"] = z2 - d["solo_orto"].to_numpy()

    # plausibility cut, verbatim from montar_bioma l. 560-578
    lim = ROT.limites_do_bioma(d, (float(np.nanmin(dsm)), float(np.nanmax(dsm))), painel)
    impossivel = np.zeros(len(d), dtype=bool)
    for col in ("y_novo", "y_bruto"):
        v = d[col].to_numpy(np.float64)
        impossivel |= ~(np.isfinite(v) & (v >= lim["y_piso_m"]) & (v <= lim["y_teto_m"]))
    hd = d["h_dossel"].to_numpy(np.float64)
    impossivel |= np.isfinite(hd) & ((hd < ROT.DOSSEL_MIN_M) | (hd > lim["dossel_teto_m"]))
    for q in ROT.RH_GUARDAR:
        cq = f"rh{q}"
        if cq in d.columns:
            v = d[cq].to_numpy(np.float64)
            impossivel |= np.isfinite(v) & ((v < ROT.RH_MIN_M) | (v > lim["dossel_teto_m"]))
    so = d["solo_orto"].to_numpy(np.float64)
    impossivel |= ~(np.isfinite(so) & (so >= lim["solo_min_m"]) & (so <= lim["solo_max_m"]))
    n_imp = int((impossivel & d.bom.to_numpy().astype(bool)).sum())
    lim["reprovados"] = n_imp
    d.loc[impossivel, "bom"] = False
    info = {"n_gedi_bruto": int(n_gedi_bruto), "n_gedi_solo_topo_validos": int(len(g)),
            "n_atl08_terreno_valido": int(len(a)), "n_disparos_na_caixa_finitos": int(len(d)),
            "limites_do_bioma": lim, "reprovados_pelo_corte_de_plausibilidade": n_imp}
    return d, dsm, info


def mascaras_bracos(d: pd.DataFrame) -> dict[str, np.ndarray]:
    bom = d["bom"].to_numpy().astype(bool)
    gedi = d["fonte"].to_numpy() == 1
    deg = d["degrade_flag"].to_numpy(np.float64)
    sens = d["sensitivity"].to_numpy(np.float64)
    sol = d["solar_elevation"].to_numpy(np.float64)
    pot = d["feixe"].isin(FEIXES_POTENCIA).to_numpy()
    f1 = gedi & (deg == 0) & (sens >= F1_SENS)
    f2 = gedi & (deg == 0) & (sens >= F2_SENS) & pot & (sol < 0)
    return {"F0": bom, "F1": bom & (~gedi | f1), "F2": bom & (~gedi | f2)}


# ------------------------------------------------------------------ cells
def chaves_ancora(dump: pd.DataFrame) -> np.ndarray:
    qr = dump["quad"].map({q: v[0] for q, v in ROT.QUADS.items()}).to_numpy(np.int64)
    qc = dump["quad"].map({q: v[1] for q, v in ROT.QUADS.items()}).to_numpy(np.int64)
    idx = dump["idx_no"].to_numpy(np.int64)
    r = qr * ROT.QUAD + idx // ROT.QUAD
    c = qc * ROT.QUAD + idx % ROT.QUAD
    return r, c


def y_novo_gravado(dump: pd.DataFrame, latdir: Path) -> np.ndarray:
    """Stored label, read as in the evaluation script juiz_lidar_v4.py (l. 120-128)."""
    out = np.full(len(dump), np.nan)
    for q in ROT.QUADS:
        m = (dump["quad"] == q).to_numpy()
        if not m.any():
            continue
        z = np.load(latdir / f"rotulos_{BIOMA}_{q}.npz")
        bom = z["bom"] > 0.5
        s = (pd.DataFrame({"idx_no": z["idx_no"][bom].astype(np.int64),
                           "y_novo": z["y_novo"][bom]})
             .groupby("idx_no").first()["y_novo"])
        z.close()
        out[m] = dump.loc[m, "idx_no"].map(s).to_numpy(np.float64)
    return out


def estatisticas_classe(s: pd.DataFrame, rot: str, braco: str, teto: float) -> dict:
    n_cel = int(len(s))
    n_t = int(s["transecto"].nunique())
    A = RR.estimando_unidade(s, "transecto", "termo_rotulo", f"{rot}|A|transecto")
    red = RR.mae_reducao_unidade(s, "transecto", f"{rot}|reducao|transecto")
    desl = RR.estimando_unidade(s, "transecto", "deslocamento_m",
                                f"{rot}|deslocamento_{braco}|transecto")
    if rot == ">30m":
        piso = bool(n_cel >= PISO_CEL and n_t >= PISO_TRANS)
    else:
        piso = None      # population floor defined only for the >30 m class
    lo = {"lim_inf_A_media_m": (A.get("ic95_pond_celula") or [None])[0],
          "lim_inf_A_mediana_m": (A.get("ic95_mediana") or [None])[0],
          "lim_inf_reducao_mae_m": (red.get("ic95_pond_celula") or [None])[0]}
    if rot == ">30m" and not piso:
        veredito = "nao avaliavel"
    elif any(v is None for v in lo.values()):
        veredito = "nao avaliavel"
    else:
        veredito = ("sustentado" if all(v > teto for v in lo.values())
                    else "nao sustentado")
    return {
        "n_celulas": n_cel, "n_transectos": n_t, "atinge_piso": piso,
        "A": {"media_pond_celula_m": A.get("media_pond_celula_m"),
              "ic95_media_pond": A.get("ic95_pond_celula"),
              "mediana_m": A.get("mediana_m"), "ic95_mediana": A.get("ic95_mediana"),
              "medias_por_transecto_m": A.get("medias_por_unidade_m")},
        "reducao_mae_equivalente": {"valor_m": red.get("reducao_mae_equivalente_m"),
                                    "ic95": red.get("ic95_pond_celula")},
        "limites_inferiores": lo, "teto_m": teto,
        "acima_do_teto": {k: (None if v is None else bool(v > teto)) for k, v in lo.items()},
        "veredito": veredito,
        "deslocamento_mediana_Fk_menos_F0": {
            "n": desl.get("n_celulas"), "media_m": desl.get("media_pond_celula_m"),
            "mediana_m": desl.get("mediana_m"), "ic95_media": desl.get("ic95_pond_celula"),
            "ic95_mediana": desl.get("ic95_mediana")},
    }


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alvos", default=str(SAT / "alvos"))
    ap.add_argument("--gee", default=str(SAT / "gee"))
    ap.add_argument("--laterais", default=os.environ.get(
        "V23_LATDIR", str(SAT / "laterais_nativa_diretas")))
    ap.add_argument("--saida", default="auditar_rf_filtro_gedi.json")
    a = ap.parse_args()
    alvos, gee, latdir = Path(a.alvos), Path(a.gee), Path(a.laterais)
    saida = Path(a.saida)
    if not saida.is_absolute():
        saida = RAIZ / saida
    log(f"  alvos {alvos} | gee {gee} | laterais {latdir} | saida {saida}")

    s7 = json.loads(S7_JSON.read_text(encoding="utf-8"))
    regua = json.loads(REGUA_JSON.read_text(encoding="utf-8"))
    g3 = guarda_g3(s7)
    teto = {c: g3[c]["teto_O3_m_lido"] for c in CLASSES}

    # Population = anchor-cell dump; its sha256 must match the one recorded
    # by auditar_rotulo_regua_v2.
    sha_dump = PROV.sha256(DUMP_E4)
    sha_esp = regua["conferencia_previa_sha256"][DUMP_E4.name]
    if sha_dump != sha_esp:
        raise RuntimeError(f"dump E4 sha256 {sha_dump} != registrado em "
                           f"auditar_rotulo_regua_v2.json {sha_esp}")
    dump = pd.read_parquet(DUMP_E4)
    if len(dump) != 1194 or dump.duplicated(["quad", "idx_no"]).any():
        raise RuntimeError(f"dump E4 com {len(dump)} linhas ou chave duplicada")
    r_a, c_a = chaves_ancora(dump)
    dump["chave"] = r_a * ROT.GRID + c_a

    log("=== disparos (caminho de montar_bioma) ===")
    d, dsm, info = montar_disparos(alvos, gee)
    masks = mascaras_bracos(d)
    d["chave"] = d["r"].to_numpy(np.int64) * ROT.GRID + d["c"].to_numpy(np.int64)
    em_anc = d["chave"].isin(set(dump["chave"])).to_numpy()
    gedi = (d["fonte"] == 1).to_numpy()
    sens_gt1 = (d["sensitivity"].to_numpy(np.float64) > 1.0)

    contagens = {}
    for b, m in masks.items():
        contagens[b] = {
            "bioma": {"gedi_aprovados": int((m & gedi).sum()),
                      "atl08_aprovados": int((m & ~gedi).sum())},
            "celulas_ancora": {"gedi_aprovados": int((m & gedi & em_anc).sum()),
                               "atl08_aprovados": int((m & ~gedi & em_anc).sum()),
                               "gedi_aprovados_com_sensitivity_maior_que_1":
                                   int((m & gedi & em_anc & sens_gt1).sum())}}
        if b != "F0":
            for esc in ("bioma", "celulas_ancora"):
                contagens[b][esc]["gedi_removidos_vs_F0"] = (
                    contagens["F0"][esc]["gedi_aprovados"] - contagens[b][esc]["gedi_aprovados"])
        log(f"  {b}: {contagens[b]}")

    sub = d.loc[em_anc, ["chave", "solo_orto"]].copy()
    med = {}
    for b, m in masks.items():
        med[b] = (sub[m[em_anc]].groupby("chave")["solo_orto"].median())
    dump["med_F0"] = dump["chave"].map(med["F0"]).to_numpy(np.float64)

    # ---- G1
    y_grav = y_novo_gravado(dump, latdir)
    dsm_cel = dsm[r_a, c_a].astype(np.float64)
    dif = dsm_cel - dump["med_F0"].to_numpy() - y_grav
    adif = np.abs(dif)
    n_ok = int(np.sum(np.isfinite(adif) & (adif <= G1_TOL_M)))
    frac_ok = n_ok / len(dump)
    n_nan = int(np.sum(~np.isfinite(adif)))
    max_dif = float(np.nanmax(adif)) if np.isfinite(adif).any() else float("nan")
    n_acima_05 = int(np.sum(np.isfinite(adif) & (adif > G1_MAX_M)))
    g1 = {"n_celulas": int(len(dump)), "n_ate_0_01_m": n_ok, "fracao_ate_0_01_m": frac_ok,
          "n_sem_disparo_F0_ou_sem_rotulo": n_nan, "max_abs_dif_m": max_dif,
          "n_acima_0_5_m": n_acima_05,
          "quantis_abs_dif_m": {p: float(np.nanpercentile(adif, p)) for p in (50, 99, 100)},
          "passou": bool(frac_ok >= G1_FRAC_MIN and n_acima_05 == 0 and n_nan == 0)}
    log(f"  G1: {g1}")
    if not g1["passou"]:
        raise RuntimeError(f"G1 FALHOU: {g1}")

    # ---- arms
    bracos, remocao = {}, {}
    for b in masks:
        x = dump.copy()
        x["med_Fk"] = x["chave"].map(med[b]).to_numpy(np.float64)
        x = x[np.isfinite(x["med_Fk"].to_numpy())].copy()
        x["deslocamento_m"] = x["med_Fk"] - x["med_F0"]
        x["termo_rotulo"] = x["termo_rotulo"] + x["deslocamento_m"]   # A_Fk
        if b == "F0":
            alvo_g2 = regua["por_classe_eth"][">30m"]["por_unidade"]["transecto"]["A"]
            s = x[x["classe_eth"] == ">30m"]
            A0 = RR.estimando_unidade(s, "transecto", "termo_rotulo", ">30m|A|transecto")
            g2 = {"recomputado": {k: A0[k] for k in G2_ALVO},
                  "alvo_protocolo": G2_ALVO,
                  "alvo_regua_v2_json": {k: alvo_g2[k] for k in G2_ALVO},
                  "diferencas": {k: A0[k] - G2_ALVO[k] for k in G2_ALVO},
                  "ic95_media_pond_recomputado": A0["ic95_pond_celula"],
                  "ic95_media_pond_regua_v2_json": alvo_g2["ic95_pond_celula"],
                  "ic95_mediana_recomputado": A0["ic95_mediana"],
                  "ic95_mediana_regua_v2_json": alvo_g2["ic95_mediana"]}
            g2["passou"] = bool(all(abs(g2["diferencas"][k]) <= G2_TOL for k in G2_ALVO))
            log(f"  G2: {g2}")
            if not g2["passou"]:
                raise RuntimeError(f"G2 FALHOU: {g2}")
        bracos[b] = {c: estatisticas_classe(x[x["classe_eth"] == c], c, b, teto[c])
                     for c in CLASSES}
        remocao[b] = {}
        for c in CLASSES:
            t0 = dump[dump["classe_eth"] == c]
            t1 = x[x["classe_eth"] == c]
            remocao[b][c] = {
                "celulas_F0": int(len(t0)), "celulas_Fk": int(len(t1)),
                "celulas_removidas": int(len(t0) - len(t1)),
                "celulas_so_atl08_ou_sem_mudanca_de_mediana":
                    int((t1["deslocamento_m"] == 0).sum()),
                "transectos_removidos": sorted(set(t0["transecto"]) - set(t1["transecto"]))}
        for c in CLASSES:
            r_ = bracos[b][c]
            log(f"  {b} {c}: n={r_['n_celulas']} t={r_['n_transectos']} piso={r_['atinge_piso']} "
                f"lo={r_['limites_inferiores']} teto={r_['teto_m']:.4f} -> {r_['veredito']}")

    fontes = [alvos / f"gedi02_a_{BIOMA}_dedup.parquet", alvos / f"atl08_{BIOMA}.parquet",
              gee / f"glo30_{BIOMA}.tif", DUMP_E4, REGUA_JSON, S7_JSON,
              RAIZ / "rotulos_lidar_v23.py", RAIZ / "auditar_rotulo_regua_v2.py",
              RAIZ / "geoide_v23.py", RAIZ / "juiz_lidar_v4.py", Path(__file__)]
    fontes += [latdir / f"rotulos_{BIOMA}_{q}.npz" for q in ROT.QUADS]

    PROV.gravar(saida, {
        "pergunta": "R-F: acima de 30 m (classe ETH), A continua acima do gap entre "
                    "estimadores (O3) com o rotulo GEDI recalculado so com disparos "
                    "que passam no filtro padrao da literatura?",
        "protocolo": "Pre-declared design note for check R-F (not distributed).",
        "bracos_definicao": {
            "F0": "quality_flag == 1 e num_modos >= 1 + corte de plausibilidade do bioma",
            "F1": f"F0 e degrade_flag == 0 e sensitivity >= {F1_SENS}",
            "F2": f"F0 e degrade_flag == 0 e sensitivity >= {F2_SENS} e feixe em "
                  f"{list(FEIXES_POTENCIA)} e solar_elevation < 0",
            "atl08": "inalterado nos tres bracos (criterio adotado)"},
        "criterio": {"estimandos": ["lim_inf_A_media_m", "lim_inf_A_mediana_m",
                                    "lim_inf_reducao_mae_m"],
                     "regra": "sustentado se os tres limites inferiores > teto O3 da classe",
                     "bootstrap": {"B": RR.B, "semente": RR.SEED, "alpha": RR.ALPHA,
                                   "cluster": "transecto"},
                     "piso_maior30m": {"celulas_min": PISO_CEL, "transectos_min": PISO_TRANS},
                     "classe_20_30m": "leitura secundaria contra 0,500 m; piso nao "
                                      "definido pelo protocolo (atinge_piso = null)"},
        "decisoes_de_implementacao": [
            "populacao = auditar_tabela3_dump.parquet (sha256 conferido contra o registrado "
            "em auditar_rotulo_regua_v2.json); A_adotado = termo_rotulo do dump; "
            "montar_populacao nao e reexecutado",
            "limites de plausibilidade derivados uma vez, como no original (todos os "
            "disparos aprovados do bioma), e aplicados igualmente aos tres bracos; "
            "Fk = F0 e a condicao extra do braco",
            "passo de rugosidade de montar_bioma omitido: nao altera bom nem y_novo",
            "G1: 'nenhuma acima de 0,5 m sem explicacao registrada' implementado como "
            "zero celulas acima de 0,5 m e zero celulas sem disparo F0",
            "G3 conferido a 3 casas decimais (o protocolo fixa 1,442 e 0,500)",
            "bootstrap: mesmas etiquetas de fluxo de auditar_rotulo_regua_v2 para A e "
            "reducao em todos os bracos; deslocamento com etiqueta propria por braco",
            "classe 20-30 m: piso nao definido pelo protocolo (atinge_piso = null); "
            "veredito mecanico contra 0,500 m como leitura secundaria",
            "unidade: so transecto (o R-F nao pede pegada)",
            "sensitivity > 1 nao excluida (filtro literal do protocolo); contagem reportada",
        ],
        "guardas": {"G1": g1, "G2": g2, "G3": g3},
        "populacao": {"dump": DUMP_E4.name, "sha256_dump": sha_dump, "n_celulas": int(len(dump))},
        "disparos": info,
        "contagens_disparos": contagens,
        "remocao_de_celulas": remocao,
        "por_braco": bracos,
        "vereditos_maior30m": {b: bracos[b][">30m"]["veredito"] for b in bracos},
        "contra_auditoria": "PENDENTE -- nao feita por quem escreveu este script",
    }, fontes_lidas=fontes, script=__file__)
    log(f"  -> {saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
