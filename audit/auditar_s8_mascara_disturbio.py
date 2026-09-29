"""S8 -- forest disturbance mask (Hansen GFC).

QUESTION. Does forest loss between the acquisitions (GLO-30 2011-2015, ALS
2016-2018, labels 2022-2023) move A or the ordering of the products?
EVALUATION ONLY: nothing is retrained.

DATA. Hansen GFC `lossyear`, version and tiles from the manifest recorded
by `baixar_hansen_gfc.py` (SATELITES/hansen_gfc/manifesto_hansen_gfc.json).
A disturbed cell is any node-grid cell intersected, with positive area, by
a Hansen pixel with lossyear in [11, 23] (2011-2023). Implementation: a
binary indicator at native resolution -> explicit reprojection
(rasterio.warp.reproject, source CRS READ from the GeoTIFF, destination CRS
EPSG:4326 = the geographic node grid) with Resampling.max. Binary before
the max: the max of lossyear in a cell with loss in 2015 and 2024 would
give 24 and lose the 2015 one. Independent check (fail-loud): the same
indicator by EXACT area overlap, in integer arithmetic (unit 1/36000
degree, LCM of 1/3600 and 1/4000), across every evaluated cell.

REUSE (imported, not rewritten; READ-ONLY files):
  auditar_rotulo_regua_v2 (V2): sha256 of the inputs, building E4 + the
    native delta, footprint unit, estimando_unidade, mae_reducao_unidade,
    veredito_S, veredito_O.
  auditar_nmad_pareado_nativa_anadem (AN): admissible forest population
    with ANADEM, pool tables, per-transect x class matrix, reference
    reproduction guard, Holm correction, ANADEM sha256.
  auditar_nmad_pareado (REF): guardas_populacao, testar (same bootstrap
    B=10,000, seed 42, per-test stream).
  preparar_anadem_nativa (GRADE): node-grid constants.

GUARDS (fail-loud, WITHOUT the mask, before any masked number):
  (g0) sha256 of every input (V2 + ANADEM + Hansen against the manifest).
  (g1) A >30m (cell-weighted mean, E4) = auditar_tabela3.json (6.651019 m)
       to 1e-6.
  (g2) the (O) ceiling of the v23 x mlp families (MAE, bootstrap IC95 of
       the per-transect median) = 1.093 m (>30m) and 0.356 m (20-30m) from
       auditar_nmad_pareado_nativa_diretas.json, to 1e-6.
  (g3) the population (the judge's tabela_regua_k0) and every base pair of
       auditar_nmad_pareado_nativa_diretas.json to 1e-6 (AN.guarda_...).
  (g4) v23 x anadem against auditar_nmad_pareado_nativa_anadem.json to 1e-6.
  (g5) A, median, CIs and MAE reduction from auditar_rotulo_regua_v2.json
       to 1e-9, in both units, across every class.
  (g6) full Hansen coverage of the grid, and reproject-max == exact overlap
       in every evaluated cell.

PRE-DECLARED (fixed before running):
  the primary (O) ceiling = the definition above (v23 x mlp, MAE)
  RECOMPUTED on the undisturbed cells; the sensitivity check = the fixed
  ceiling 1.093/0.356. "Robust" criterion = the (S) and (O) verdicts in the
  20-30m and >30m classes match the unmasked run AND the sign of the
  median is unchanged in every Table IV contrast (v23 x {mlp, fabdem,
  gedtm30, glo30, anadem}; MAE and NMAD; 20-30m and >30m) whose unmasked
  IC95 excludes zero. Any gain/loss of zero-exclusion is listed as a
  descriptive change. O3 does not apply (S7 was not released yet).

Does not interpret for the paper; does not audit its own result.

USAGE
  /trabalho/ambientes/s33_amb_virtual/.venv/bin/python auditar_s8_mascara_disturbio.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import rasterio
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import proveniencia as PROV                            # noqa: E402
import auditar_rotulo_regua_v2 as V2                    # noqa: E402  (READ-ONLY, reuse)
import auditar_nmad_pareado_nativa_anadem as AN         # noqa: E402  (READ-ONLY, reuse)
import auditar_nmad_pareado as REF                      # noqa: E402  (READ-ONLY, reuse)
import juiz_lidar_v4 as J                               # noqa: E402  (READ-ONLY, reuse)
import preparar_anadem_nativa as GRADE                  # noqa: E402  (constants only)

# ---------------------------------------------------------------- entradas
HANSEN_DIR = Path("/trabalho/GNN_TOPO/SATELITES/hansen_gfc")
MANIFESTO = HANSEN_DIR / "manifesto_hansen_gfc.json"
PARQUET_LIDAR = Path("/trabalho/GNN_TOPO/SATELITES/lidar_eba/solo_lidar_30m_v3_nativa.parquet")
JUIZ_JSON = RAIZ / "juiz_grade_nativa_diretas.json"
LATERAIS = Path("/trabalho/GNN_TOPO/SATELITES/laterais_nativa_diretas")
RECORTE_ANADEM = Path("/trabalho/GNN_TOPO/SATELITES/anadem")
REF_DIRETAS = RAIZ / "auditar_nmad_pareado_nativa_diretas.json"
REF_ANADEM = RAIZ / "auditar_nmad_pareado_nativa_anadem.json"
REF_V2 = RAIZ / "auditar_rotulo_regua_v2.json"

SAIDA_JSON = RAIZ / "auditar_s8_mascara_disturbio.json"

ANO_MIN, ANO_MAX = 11, 23                    # lossyear 11..23 = 2011..2023
E4_ALVO_LITERAL = 6.651019                   # pre-declared reference value (m)
TETO_ALVO_LITERAL = {">30m": 1.093, "20-30m": 0.356}

ROT_F = V2.ROT_F
CLASSES_CONF = ["20-30m", ">30m"]
UNIDADES = ("transecto", "pegada")
PRODUTOS = AN.PRODUTOS_TODOS                  # v23 mlp fabdem gedtm30 glo30 anadem
PARES_TAB4 = [("v23", "mlp"), ("v23", "fabdem"), ("v23", "gedtm30"),
              ("v23", "glo30"), ("v23", "anadem")]
METRICAS = ("mae", "nmad")

# grid in integer units of 1/36000 degree (LCM of 1/3600 and 1/4000)
U = 36000
DST_CRS = CRS.from_epsg(4326)


def log(m: str = "") -> None:
    print(m, flush=True)


# ------------------------------------------------------------ mascara
def _conferir_grade_inteira() -> tuple[int, int]:
    """The node grid in integer units: west edge and north edge (u)."""
    P = GRADE.P
    if abs(P - 1.0 / 3600.0) > 1e-15 or GRADE.GRID != 7200 or GRADE.Q != 3600:
        raise RuntimeError(f"grade inesperada: P={P} GRID={GRADE.GRID} Q={GRADE.Q}")
    oeste_u = -60 * U + 5          # -60 + P/2
    norte_u = -2 * U - 5           # -2 - P/2
    if abs(oeste_u / U - GRADE.LON_O) > 1e-12 or abs(norte_u / U - GRADE.LAT_N) > 1e-12:
        raise RuntimeError("constantes da grade nao batem com a forma inteira")
    return oeste_u, norte_u


def construir_mascaras(manifesto: dict) -> tuple[dict, dict]:
    """Returns {name: GRID x GRID uint8 array} for the indicators (primary
    2011-2023; descriptive 2001-2010, 2024-2025, any) and the procedure
    record. Plus the coverage (1 = cell covered by a tile)."""
    G = GRADE.GRID
    faixas = {"perda_2011_2023": (ANO_MIN, ANO_MAX), "perda_2001_2010": (1, 10),
              "perda_2024_2025": (24, 25), "perda_qualquer_2001_2025": (1, 25)}
    saida = {k: np.zeros((G, G), np.uint8) for k in faixas}
    cobertura = np.zeros((G, G), np.uint8)
    tiles_info, brutos = [], []
    for reg in manifesto["tiles"]:
        p = Path(reg["arquivo"])
        with rasterio.open(p) as src:
            crs_src = src.crs
            t = src.transform
            res = (t.a, -t.e)
            if abs(res[0] - 0.00025) > 1e-15 or abs(res[1] - 0.00025) > 1e-15 or t.b or t.d:
                raise RuntimeError(f"{p.name}: transform inesperado {t}")
            # window covering the grid with a 4 px margin
            c0 = int(np.floor((GRADE.LON_O - t.c) / res[0])) - 4
            c1 = int(np.ceil((GRADE.LON_E - t.c) / res[0])) + 4
            r0 = int(np.floor((t.f - GRADE.LAT_N) / res[1])) - 4
            r1 = int(np.ceil((t.f - GRADE.LAT_S) / res[1])) + 4
            c0, r0 = max(c0, 0), max(r0, 0)
            c1, r1 = min(c1, src.width), min(r1, src.height)
            if c1 <= c0 or r1 <= r0:
                continue
            win = Window(c0, r0, c1 - c0, r1 - r0)
            a = src.read(1, window=win)
            t_win = rasterio.windows.transform(win, t)
        vmax = int(a.max())
        tiles_info.append({"tile": reg["tile"], "crs_lido": str(crs_src),
                           "janela_linhas_colunas": [r0, r1, c0, c1],
                           "lossyear_max_na_janela": vmax,
                           "contagem_por_codigo_na_janela": np.bincount(a.ravel(), minlength=26).tolist()})
        brutos.append({"tile": reg["tile"], "a": a, "r0": r0, "c0": c0,
                       "lat_topo_u": int(round(t.f * U)), "lon_esq_u": int(round(t.c * U))})
        for nome, (lo, hi) in faixas.items():
            ind = ((a >= lo) & (a <= hi)).astype(np.uint8)
            out = np.zeros((G, G), np.uint8)
            reproject(ind, out, src_transform=t_win, src_crs=crs_src,
                      dst_transform=GRADE.DESTINO, dst_crs=DST_CRS,
                      resampling=Resampling.max, src_nodata=None, dst_nodata=None)
            np.maximum(saida[nome], out, out=saida[nome])
        um = np.ones_like(a, dtype=np.uint8)
        cov = np.zeros((G, G), np.uint8)
        reproject(um, cov, src_transform=t_win, src_crs=crs_src,
                  dst_transform=GRADE.DESTINO, dst_crs=DST_CRS,
                  resampling=Resampling.max, src_nodata=None, dst_nodata=None)
        np.maximum(cobertura, cov, out=cobertura)
    if not cobertura.all():
        raise RuntimeError(f"GUARDA g6 FALHOU: {int((cobertura == 0).sum())} celulas da grade "
                           "sem cobertura Hansen -- tile faltando")
    proc = {"metodo": "indicador binario (lo<=lossyear<=hi) na resolucao nativa -> "
                      "rasterio.warp.reproject(Resampling.max), src_crs LIDO do GeoTIFF, "
                      "dst_crs EPSG:4326, dst_transform = grade de nos (preparar_anadem_nativa.DESTINO)",
            "dst_crs": str(DST_CRS), "dst_transform": list(GRADE.DESTINO)[:6],
            "faixas_lossyear": {k: list(v) for k, v in faixas.items()},
            "tiles": tiles_info, "cobertura_grade_completa": True,
            "rasterio": rasterio.__version__, "gdal": rasterio.__gdal_version__}
    return saida, {"proc": proc, "brutos": brutos}


def indicador_exato(r: np.ndarray, c: np.ndarray, brutos: list, lo: int, hi: int) -> np.ndarray:
    """Independent check: 1 if any Hansen pixel with lo<=lossyear<=hi
    intersects cell (r, c) of the GLOBAL grid with positive area. Integer
    arithmetic in 1/36000 degree: grid cell = 10 u, Hansen pixel = 9 u."""
    oeste_u, norte_u = _conferir_grade_inteira()
    res = np.zeros(len(r), dtype=bool)
    topo = norte_u - 10 * r            # latitude of the cell's top edge (u)
    base = topo - 10
    esq = oeste_u + 10 * c
    dir_ = esq + 10
    for b in brutos:
        a, R0, C0 = b["a"], b["r0"], b["c0"]
        # row i of the tile: [lat_top - 9(i+1), lat_top - 9i]; overlaps if
        # lat_top-9i-9 < topo and lat_top-9i > base
        A_ = b["lat_topo_u"] - topo
        B_ = b["lat_topo_u"] - base
        i0 = np.floor((A_ - 9) / 9).astype(np.int64) + 1
        i1 = np.ceil(B_ / 9).astype(np.int64) - 1
        Cc = esq - b["lon_esq_u"]
        Dd = dir_ - b["lon_esq_u"]
        j0 = np.floor((Cc - 9) / 9).astype(np.int64) + 1
        j1 = np.ceil(Dd / 9).astype(np.int64) - 1
        for di in range(3):
            for dj in range(3):
                ii, jj = i0 + di, j0 + dj
                ok = (ii <= i1) & (jj <= j1)
                li, lj = ii - R0, jj - C0
                dentro = ok & (li >= 0) & (li < a.shape[0]) & (lj >= 0) & (lj < a.shape[1])
                v = np.zeros(len(r), dtype=np.uint8)
                v[dentro] = a[li[dentro], lj[dentro]]
                res |= dentro & (v >= lo) & (v <= hi)
    return res


def rc_global(quad: pd.Series, idx: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    q0 = quad.map({q: rc[0] for q, rc in GRADE.QUADS.items()}).to_numpy(dtype=np.int64)
    c0 = quad.map({q: rc[1] for q, rc in GRADE.QUADS.items()}).to_numpy(dtype=np.int64)
    ix = idx.to_numpy(dtype=np.int64)
    return q0 + ix // GRADE.Q, c0 + ix % GRADE.Q


def marcar(df: pd.DataFrame, mascaras: dict, brutos: list, rotulo: str) -> tuple[pd.DataFrame, dict]:
    r, c = rc_global(df["quad"], df["idx_no"])
    df = df.copy()
    for nome, M in mascaras.items():
        df[nome] = M[r, c].astype(bool)
    exato = indicador_exato(r, c, brutos, ANO_MIN, ANO_MAX)
    diverg = int((exato != df["perda_2011_2023"].to_numpy()).sum())
    if diverg:
        raise RuntimeError(f"GUARDA g6 FALHOU ({rotulo}): reproject max diverge da sobreposicao "
                           f"exata em {diverg} celulas")
    return df, {"populacao": rotulo, "n_celulas": int(len(df)),
                "divergencias_reproject_vs_exato": diverg}


# ------------------------------------------------------- fracoes removidas
def fracoes(df: pd.DataFrame, col_classe: str) -> dict:
    def bloco(s):
        n = int(len(s))
        k = int(s["perda_2011_2023"].sum())
        return {"n": n, "removidas": k, "fracao_removida": (k / n) if n else None,
                "com_perda_2001_2010": int(s["perda_2001_2010"].sum()),
                "com_perda_2024_2025": int(s["perda_2024_2025"].sum()),
                "com_perda_qualquer_2001_2025": int(s["perda_qualquer_2001_2025"].sum())}
    cls = df[col_classe].astype(object).where(df[col_classe].notna(), "sem_classe")
    out = {"TODAS": bloco(df),
           "por_classe": {str(k): bloco(g) for k, g in df.groupby(cls, sort=False)},
           "por_transecto": {str(k): bloco(g) for k, g in df.groupby("transecto")},
           "por_transecto_classe_confirmatoria": {}}
    for rot in CLASSES_CONF:
        sub = df[cls == rot]
        out["por_transecto_classe_confirmatoria"][rot] = {
            str(k): bloco(g) for k, g in sub.groupby("transecto")}
    return out


# ------------------------------------------------------------- estatisticas A
def por_unidade_A(df: pd.DataFrame, rot: str) -> dict:
    s = df if rot == "TODAS" else df[df["classe_eth"] == rot]
    out = {}
    for u in UNIDADES:
        out[u] = {"A": V2.estimando_unidade(s, u, "termo_rotulo", f"{rot}|A|{u}"),
                  "reducao_mae_equivalente": V2.mae_reducao_unidade(s, u, f"{rot}|reducao|{u}")}
    return out


def _cmp(a, b, tol, rotulo):
    if a is None or b is None:
        if a is not b:
            raise RuntimeError(f"GUARDA g5 FALHOU {rotulo}: {a} != {b}")
        return 0
    if abs(float(a) - float(b)) > tol:
        raise RuntimeError(f"GUARDA g5 FALHOU {rotulo}: recomputado {a} != v2 {b} (tol {tol})")
    return 1


def guarda_v2(por_classe: dict, v2: dict) -> dict:
    n = 0
    for rot, pu in por_classe.items():
        ref = v2["por_classe_eth"][rot]["por_unidade"]
        for u in UNIDADES:
            a, b = pu[u]["A"], ref[u]["A"]
            if a["n_celulas"] != b["n_celulas"] or a["n_unidades"] != b["n_unidades"]:
                raise RuntimeError(f"GUARDA g5 FALHOU {rot}/{u}: n difere")
            for k in ("media_pond_celula_m", "mediana_m", "media_por_unidade_m"):
                n += _cmp(a[k], b[k], 1e-9, f"{rot}/{u}/A/{k}")
            for k in ("ic95_pond_celula", "ic95_mediana", "ic95_t_jackknife_pond"):
                for i in range(2):
                    n += _cmp(a[k][i], b[k][i], 1e-9, f"{rot}/{u}/A/{k}[{i}]")
            a, b = pu[u]["reducao_mae_equivalente"], ref[u]["reducao_mae_equivalente"]
            n += _cmp(a["reducao_mae_equivalente_m"], b["reducao_mae_equivalente_m"], 1e-9,
                      f"{rot}/{u}/reducao")
            for k in ("ic95_pond_celula", "ic95_t_jackknife"):
                for i in range(2):
                    n += _cmp(a[k][i], b[k][i], 1e-9, f"{rot}/{u}/reducao/{k}[{i}]")
    return {"status": "OK -- reproduz auditar_rotulo_regua_v2.json a 1e-9", "campos_conferidos": n}


# ---------------------------------------------------------- contrastes Tab IV
def contrastes(flor: pd.DataFrame) -> tuple[dict, dict, dict]:
    pool = AN.tabela_pool_generic(flor, "faixa_ex", PRODUTOS)
    mat = AN.por_transecto_faixa_generic(flor, "faixa_ex", PRODUTOS)
    testes = {}
    for a_, b_ in PARES_TAB4:
        testes[f"{a_} vs {b_}"] = {m: REF.testar(mat, a_, b_, None, m) for m in METRICAS}
    AN.aplicar_holm(testes["v23 vs fabdem"]["nmad"])
    AN.aplicar_holm(testes["v23 vs fabdem"]["mae"])
    return pool, mat, testes


def teto_familias(testes: dict) -> dict:
    out = {}
    for rot in ROT_F:
        r = testes["v23 vs mlp"]["mae"][rot]
        if not r.get("testado"):
            out[rot] = {"testado": False, "motivo": r.get("motivo"), "teto_m": None}
            continue
        lo, hi = r["ic95_bootstrap_transecto_mediana"]
        out[rot] = {"testado": True, "ic95_m": [lo, hi], "teto_m": max(abs(lo), abs(hi)),
                    "n_transectos": r["n_transectos"]}
    return out


def resumo_teste(r: dict) -> dict:
    if not r.get("testado"):
        return {"testado": False, "motivo": r.get("motivo"), "n_transectos": r.get("n_transectos")}
    lo, hi = r["ic95_bootstrap_transecto_mediana"]
    return {"testado": True, "mediana_m": r["mediana_m"], "media_m": r["media_m"],
            "ic95_m": [lo, hi], "ic_exclui_zero": bool(lo > 0 or hi < 0),
            "sinal_mediana": int(np.sign(r["mediana_m"])), "p_wilcoxon": r["p_wilcoxon"],
            "p_holm": r.get("p_holm"), "vence_a": r["vence"], "n_transectos": r["n_transectos"],
            "transectos": r["transectos"]}


def veredito_O_seguro(pu: dict, teto: dict) -> dict:
    if not teto["testado"]:
        return {"criterio": "(O)", "veredito": "nao_demonstrado",
                "motivo": "teto v23 x mlp nao testavel: " + str(teto.get("motivo"))}
    return V2.veredito_O(pu, teto["teto_m"])


# --------------------------------------------------------------------- main
def main() -> int:
    log("=== g0: sha256 das entradas (antes de calcular) ===")
    shas = V2.conferir_sha256_entradas()
    AN.conferir_sha256_anadem(LATERAIS, RECORTE_ANADEM)
    manifesto = json.loads(MANIFESTO.read_text(encoding="utf-8"))
    for reg in manifesto["tiles"]:
        h = PROV.sha256(Path(reg["arquivo"]))
        if h != reg["sha256"]:
            raise RuntimeError(f"GUARDA g0 FALHOU: {reg['arquivo']} sha256 {h} != manifesto {reg['sha256']}")
        shas[Path(reg["arquivo"]).name] = h
    for p in (REF_V2, REF_ANADEM, JUIZ_JSON, PARQUET_LIDAR, MANIFESTO):
        shas[p.name] = PROV.sha256(p)
    log(f"  Hansen {manifesto['versao']} tiles {[t['tile'] for t in manifesto['tiles']]}: sha256 confere")

    tabela3 = json.loads(V2.TABELA3_JSON.read_text(encoding="utf-8"))
    ref_diretas = json.loads(REF_DIRETAS.read_text(encoding="utf-8"))
    ref_anadem = json.loads(REF_ANADEM.read_text(encoding="utf-8"))
    v2json = json.loads(REF_V2.read_text(encoding="utf-8"))
    juiz = json.loads(JUIZ_JSON.read_text(encoding="utf-8"))

    # ------------------------------------------------ anchors (E4 + delta)
    dump = pd.read_parquet(V2.DUMP_E4_PARQUET)
    nativa = pd.read_parquet(V2.NATIVA_PARQUET)
    df = dump.merge(nativa[["transecto", "quad", "idx_no", "delta"]],
                    on=["transecto", "quad", "idx_no"], how="left", validate="one_to_one")
    mapa_pegada, _ = V2.montar_unidade_pegada(nativa)
    df["pegada"] = df["transecto"].map(mapa_pegada)

    log("\n=== g1: A >30m sem mascara = E4 (a 1e-6) ===")
    alvo_e4 = tabela3["por_classe_eth"][">30m"]["sem_piso"]["media_pond_celula_m"]
    recomp_e4 = float(df.loc[df["classe_eth"] == ">30m", "termo_rotulo"].mean())
    log(f"  recomputado {recomp_e4:.9f} alvo {alvo_e4:.9f} literal {E4_ALVO_LITERAL}")
    if abs(recomp_e4 - alvo_e4) > 1e-6 or abs(recomp_e4 - E4_ALVO_LITERAL) > 1e-6:
        raise RuntimeError(f"GUARDA g1 FALHOU: {recomp_e4} vs {alvo_e4} / {E4_ALVO_LITERAL}. PARANDO.")
    log("  g1 PASSOU")

    log("\n=== A sem mascara (todas as classes) e g5 contra o v2 (1e-9) ===")
    classes_A = ROT_F + ["TODAS"]
    A_sem = {rot: por_unidade_A(df, rot) for rot in classes_A}
    g5 = guarda_v2(A_sem, v2json)
    log(f"  g5: {g5}")

    # ------------------------------------------------ admissible forest
    log("\n=== populacao floresta julgavel (com ANADEM) e g3/g4 ===")
    J.LAT = LATERAIS
    flor, info = AN.montar_populacao_com_anadem(PARQUET_LIDAR, juiz, LATERAIS)
    conf_pop = REF.guardas_populacao(flor, juiz)
    pool_ex_b = AN.tabela_pool_generic(flor, "faixa_ex", AN.PRODUTOS_BASE)
    pool_v2_b = AN.tabela_pool_generic(flor, "faixa", AN.PRODUTOS_BASE)
    mat_ex_b = AN.por_transecto_faixa_generic(flor, "faixa_ex", AN.PRODUTOS_BASE)
    mat_v2_b = AN.por_transecto_faixa_generic(flor, "faixa", AN.PRODUTOS_BASE)
    principal_b, sens_b = AN.rodar_par(mat_ex_b, mat_v2_b, AN.PARES_BASE)
    g3 = AN.guarda_reproduz_referencia(pool_ex_b, pool_v2_b, principal_b, sens_b, ref_diretas)
    log(f"  g3: {g3}")

    pool_sem, mat_sem, testes_sem = contrastes(flor)
    ref_an = ref_anadem["populacao_original_ETH"]["testes_principal_ETH_v23_vs_anadem"]
    n_g4 = 0
    for m in METRICAS:
        for rot in ROT_F:
            a, b = testes_sem["v23 vs anadem"][m][rot], ref_an[m][rot]
            if bool(a.get("testado")) != bool(b.get("testado")):
                raise RuntimeError(f"GUARDA g4 FALHOU v23 vs anadem {m}/{rot}: testado difere")
            if not a.get("testado"):
                continue
            for k in ("mediana_m", "media_m", "p_wilcoxon", "p_sinal_binomial"):
                if not AN._prox(a[k], b[k]):
                    raise RuntimeError(f"GUARDA g4 FALHOU v23 vs anadem {m}/{rot}/{k}: {a[k]} != {b[k]}")
                n_g4 += 1
            for i in range(2):
                if not AN._prox(a["ic95_bootstrap_transecto_mediana"][i], b["ic95_bootstrap_transecto_mediana"][i]):
                    raise RuntimeError(f"GUARDA g4 FALHOU v23 vs anadem {m}/{rot}/ic95[{i}]")
                n_g4 += 1
    # the base pairs recomputed by this script's own function also match the reference
    for par in ("v23 vs mlp", "v23 vs fabdem", "v23 vs gedtm30"):
        for m in METRICAS:
            for rot in ROT_F:
                a, b = testes_sem[par][m][rot], ref_diretas["testes_principal_ETH"][par][m][rot]
                if bool(a.get("testado")) != bool(b.get("testado")):
                    raise RuntimeError(f"GUARDA g3b FALHOU {par} {m}/{rot}")
                if a.get("testado"):
                    for i in range(2):
                        if not AN._prox(a["ic95_bootstrap_transecto_mediana"][i],
                                        b["ic95_bootstrap_transecto_mediana"][i]):
                            raise RuntimeError(f"GUARDA g3b FALHOU {par} {m}/{rot}/ic95[{i}]")
                    if not AN._prox(a["mediana_m"], b["mediana_m"]):
                        raise RuntimeError(f"GUARDA g3b FALHOU {par} {m}/{rot}/mediana")
                    if "p_holm" in b and not AN._prox(a.get("p_holm"), b["p_holm"]):
                        raise RuntimeError(f"GUARDA g3b FALHOU {par} {m}/{rot}/p_holm")
                    n_g4 += 1
    log(f"  g4 (v23 x anadem + pares base pela funcao deste script): {n_g4} campos a 1e-6")

    log("\n=== g2: teto (O) das familias sem mascara (a 1e-6) ===")
    teto_sem = teto_familias(testes_sem)
    g2 = {}
    for rot in CLASSES_CONF:
        r = ref_diretas["testes_principal_ETH"]["v23 vs mlp"]["mae"][rot]
        alvo = max(abs(r["ic95_bootstrap_transecto_mediana"][0]), abs(r["ic95_bootstrap_transecto_mediana"][1]))
        rec = teto_sem[rot]["teto_m"]
        log(f"  {rot}: recomputado {rec:.9f} alvo {alvo:.9f} literal {TETO_ALVO_LITERAL[rot]}")
        if abs(rec - alvo) > 1e-6 or abs(rec - TETO_ALVO_LITERAL[rot]) > 5e-4:
            raise RuntimeError(f"GUARDA g2 FALHOU {rot}: {rec} vs {alvo}. PARANDO.")
        g2[rot] = {"recomputado_m": rec, "alvo_json_m": alvo, "diferenca_m": rec - alvo,
                   "literal_m": TETO_ALVO_LITERAL[rot], "tolerancia_m": 1e-6, "passou": True}
    log("  g2 PASSOU")

    # ------------------------------------------------ mask
    log("\n=== mascara Hansen (reproject max + conferencia exata, g6) ===")
    mascaras, extra = construir_mascaras(manifesto)
    df, g6a = marcar(df, mascaras, extra["brutos"], "ancoras_E4")
    flor, g6b = marcar(flor, mascaras, extra["brutos"], "floresta_julgavel")
    grade_total = {k: {"celulas": int(M.sum()), "fracao_da_grade": float(M.mean())}
                   for k, M in mascaras.items()}
    log(f"  g6: {g6a} | {g6b}")
    log(f"  grade 2x2 graus: {grade_total['perda_2011_2023']}")

    fr_anc = fracoes(df, "classe_eth")
    fr_cel = fracoes(flor, "faixa_ex")
    log(f"  ancoras removidas: {fr_anc['TODAS']['removidas']}/{fr_anc['TODAS']['n']}; "
        f"celulas removidas: {fr_cel['TODAS']['removidas']}/{fr_cel['TODAS']['n']}")

    df_m = df[~df["perda_2011_2023"]].copy()
    flor_m = flor[~flor["perda_2011_2023"]].copy()

    # ------------------------------------------------ A and masked criteria
    log("\n=== A, (S), (O) nas celulas NAO perturbadas ===")
    A_com = {rot: por_unidade_A(df_m, rot) for rot in classes_A}
    pool_com, _, testes_com = contrastes(flor_m)
    teto_com = teto_familias(testes_com)

    criterios = {}
    for rot in CLASSES_CONF:
        teto_fixo = {"testado": True, "teto_m": g2[rot]["alvo_json_m"]}
        criterios[rot] = {
            "sem_mascara": {"S": V2.veredito_S(A_sem[rot]),
                            "O_teto_familias": veredito_O_seguro(A_sem[rot], teto_sem[rot])},
            "com_mascara": {"S": V2.veredito_S(A_com[rot]),
                            "O_teto_recomputado_primario": veredito_O_seguro(A_com[rot], teto_com[rot]),
                            "O_teto_fixo_28_09_sensibilidade": veredito_O_seguro(A_com[rot], teto_fixo)},
            "teto_sem_mascara": teto_sem[rot], "teto_com_mascara": teto_com[rot],
        }
        c = criterios[rot]
        log(f"  {rot}: S {c['sem_mascara']['S']['veredito']} -> {c['com_mascara']['S']['veredito']}; "
            f"O {c['sem_mascara']['O_teto_familias']['veredito']} -> "
            f"{c['com_mascara']['O_teto_recomputado_primario']['veredito']} (primario, teto "
            f"{teto_com[rot]['teto_m']}) / {c['com_mascara']['O_teto_fixo_28_09_sensibilidade']['veredito']} (fixo)")

    # ------------------------------------------------ Table IV
    tab4 = {}
    for par in testes_sem:
        tab4[par] = {}
        for m in METRICAS:
            tab4[par][m] = {rot: {"sem_mascara": resumo_teste(testes_sem[par][m][rot]),
                                  "com_mascara": resumo_teste(testes_com[par][m][rot])}
                            for rot in ROT_F}

    # ------------------------------------------------ robustness criterion
    mudancas_criterio, mudancas_descritivas = [], []
    for rot in CLASSES_CONF:
        c = criterios[rot]
        for nome, v0, v1 in (("S", c["sem_mascara"]["S"]["veredito"], c["com_mascara"]["S"]["veredito"]),
                             ("O", c["sem_mascara"]["O_teto_familias"]["veredito"],
                              c["com_mascara"]["O_teto_recomputado_primario"]["veredito"])):
            if v0 != v1:
                mudancas_criterio.append(f"({nome}) {rot}: {v0} -> {v1}")
        v_fixo = c["com_mascara"]["O_teto_fixo_28_09_sensibilidade"]["veredito"]
        if v_fixo != c["sem_mascara"]["O_teto_familias"]["veredito"]:
            mudancas_descritivas.append(f"(O) {rot} com teto fixo 28/09 (sensibilidade): "
                                        f"{c['sem_mascara']['O_teto_familias']['veredito']} -> {v_fixo}")
    contrastes_avaliados = []
    for par in tab4:
        for m in METRICAS:
            for rot in ROT_F:
                s0, s1 = tab4[par][m][rot]["sem_mascara"], tab4[par][m][rot]["com_mascara"]
                dentro = rot in CLASSES_CONF
                if not s0.get("testado"):
                    if s1.get("testado"):
                        mudancas_descritivas.append(f"{par} {m} {rot}: passou a testavel com mascara")
                    continue
                if dentro and s0["ic_exclui_zero"]:
                    reg = {"par": par, "metrica": m, "classe": rot,
                           "sinal_sem": s0["sinal_mediana"], "ic_sem": s0["ic95_m"],
                           "sinal_com": s1.get("sinal_mediana"), "ic_com": s1.get("ic95_m"),
                           "testado_com": s1.get("testado")}
                    reg["sinal_inalterado"] = bool(s1.get("testado") and s1["sinal_mediana"] == s0["sinal_mediana"])
                    contrastes_avaliados.append(reg)
                    if not reg["sinal_inalterado"]:
                        mudancas_criterio.append(
                            f"Tabela IV {par} {m} {rot}: sinal {s0['sinal_mediana']:+d} -> "
                            f"{s1.get('sinal_mediana') if s1.get('testado') else 'nao testavel'}")
                if not s1.get("testado"):
                    mudancas_descritivas.append(f"{par} {m} {rot}: deixou de ser testavel com mascara "
                                                f"({s1.get('motivo')})")
                    continue
                if s0["ic_exclui_zero"] != s1["ic_exclui_zero"]:
                    mudancas_descritivas.append(
                        f"{par} {m} {rot}{'' if dentro else ' (classe fora da Tabela IV)'}: IC95 "
                        f"{'deixou de excluir' if s0['ic_exclui_zero'] else 'passou a excluir'} zero "
                        f"({s0['ic95_m'][0]:+.3f},{s0['ic95_m'][1]:+.3f}) -> "
                        f"({s1['ic95_m'][0]:+.3f},{s1['ic95_m'][1]:+.3f})")
                elif s0["sinal_mediana"] != s1["sinal_mediana"]:
                    mudancas_descritivas.append(
                        f"{par} {m} {rot}{'' if dentro else ' (classe fora da Tabela IV)'}: sinal da "
                        f"mediana {s0['sinal_mediana']:+d} -> {s1['sinal_mediana']:+d} (IC inclui zero nos dois)")
    robusto = not mudancas_criterio
    veredito = {
        "criterio": "robusto se os vereditos (S) e (O) nas duas classes (20-30m, >30m) e o sinal dos "
                    "contrastes da Tabela IV cujo IC exclui zero nao mudam (desenho S8); teto (O) "
                    "primario = v23 x mlp recomputado nas celulas nao perturbadas",
        "veredito": "ROBUSTO" if robusto else "NAO_ROBUSTO",
        "o_que_mudou_no_criterio": mudancas_criterio,
        "mudancas_descritivas_fora_do_criterio": mudancas_descritivas,
        "contrastes_tab4_com_ic_excluindo_zero_sem_mascara": contrastes_avaliados,
        "O3": "not applied: S7 (tree model against the lidar) was not produced",
    }
    log(f"\n=== VEREDITO S8: {veredito['veredito']} ===")
    for x in mudancas_criterio:
        log(f"  mudou: {x}")
    for x in mudancas_descritivas:
        log(f"  descritivo: {x}")

    # lists of removed cells (auditability; no extra file)
    anc_rem = df.loc[df["perda_2011_2023"], ["quad", "idx_no", "transecto", "classe_eth"]]
    cel_rem = flor.loc[flor["perda_2011_2023"], ["quad", "idx_no"]]

    fontes = [V2.TABELA3_JSON, V2.DUMP_E4_PARQUET, V2.NATIVA_JSON, V2.NATIVA_PARQUET, V2.NMAD_JSON,
              V2.D8_JSON, REF_V2, REF_DIRETAS, REF_ANADEM, JUIZ_JSON, PARQUET_LIDAR, MANIFESTO,
              RAIZ / "auditar_rotulo_regua_v2.py", RAIZ / "auditar_nmad_pareado_nativa_anadem.py",
              RAIZ / "auditar_nmad_pareado.py", RAIZ / "juiz_lidar_v4.py",
              RAIZ / "preparar_anadem_nativa.py", RAIZ / "baixar_hansen_gfc.py", Path(__file__)]
    fontes += [Path(t["arquivo"]) for t in manifesto["tiles"]]
    fontes += [LATERAIS / f"{n}_amazonia_{q}.npz" for q in J.QUADS
               for n in ("glo30", "fabdem", "gedtm30", "anadem", "agua_jrc", "dossel_eth", "rotulos")]
    fontes += [J.SUP / f"delta_amazonia_ampliacao_b4s035{arm}_seed{sd}.npz"
               for arm in ("", "_mlp_semobs") for sd in J.SEEDS_PAR]

    PROV.gravar(SAIDA_JSON, {
        "pergunta": "a perda florestal entre as aquisicoes (GLO-30 2011-2015, ALS 2016-2018, rotulos "
                    "2022-2023) move A ou a ordem dos produtos? (desenho S8; so avaliacao, nada retreinado)",
        "protocolo": ("Pre-declared design note for this check (S8 and closing rule; not "
                      "distributed)."),
        "declaracoes_antes_de_rodar": "artigo_v23_2/_pareceres_2026-09-28_rodada4/oficina_S8.md, Bloco 1",
        "nao_faz": "nao retreina; nao edita nenhum script/JSON existente; nao interpreta para o artigo; "
                   "nao audita o proprio resultado",
        "hansen": {"versao": manifesto["versao"], "tiles": manifesto["tiles"],
                   "definicao_perturbada": f"qualquer pixel com lossyear em [{ANO_MIN},{ANO_MAX}] "
                                           "(2011-2023) intersectando a celula com area positiva",
                   "procedimento": extra["proc"], "grade_2x2_graus": grade_total},
        "conferencia_previa_sha256": shas,
        "guardas": {
            "g1_A_maior30m_E4": {"recomputado_m": recomp_e4, "alvo_json_m": alvo_e4,
                                 "diferenca_m": recomp_e4 - alvo_e4, "literal_m": E4_ALVO_LITERAL,
                                 "tolerancia_m": 1e-6, "passou": True},
            "g2_teto_familias": g2,
            "g3_populacao_e_pares_base_diretas": g3,
            "g3_tabela_regua_k0_celulas": len(conf_pop),
            "g4_v23_anadem_e_pares_base_campos": n_g4,
            "g5_v2": g5,
            "g6_mascara": [g6a, g6b],
        },
        "populacao": {"ancoras_E4": int(len(df)), "ancoras_nao_perturbadas": int(len(df_m)),
                      "floresta_julgavel": int(len(flor)), "floresta_nao_perturbada": int(len(flor_m)),
                      "info_juiz": info, "unidade_pegada": mapa_pegada},
        "fracao_removida": {"ancoras_E4_por_classe_eth": fr_anc,
                            "celulas_floresta_julgavel_por_classe_eth": fr_cel},
        "celulas_removidas": {
            "ancoras": anc_rem.astype({"idx_no": int}).to_dict(orient="records"),
            "floresta_julgavel_por_quad": {q: sorted(int(i) for i in g["idx_no"])
                                           for q, g in cel_rem.groupby("quad")}},
        "A_por_classe_eth": {"sem_mascara": A_sem, "com_mascara_nao_perturbadas": A_com},
        "criterios_S_O": criterios,
        "tabela_IV": {"pool_ETH_sem_mascara": pool_sem, "pool_ETH_com_mascara": pool_com,
                      "contrastes_pareados_v23_menos_produto": tab4,
                      "nota": "diferenca = metrica(v23) - metrica(produto) por transecto; IC95 bootstrap "
                              "da mediana por transecto (B=10.000, semente 42, REF.testar); Holm so no "
                              "v23 vs fabdem (familia confirmatoria original)"},
        "veredito_S8": veredito,
        "contra_auditoria": "PENDENTE -- nao feita por quem escreveu este script",
    }, fontes_lidas=fontes, script=__file__)
    log(f"\n  -> {SAIDA_JSON.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
