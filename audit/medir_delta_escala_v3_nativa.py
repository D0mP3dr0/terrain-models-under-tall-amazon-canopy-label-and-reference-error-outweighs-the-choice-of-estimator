"""D-1 -- scale delta on the NATIVE GEOMETRY, for ALL cells of the 13
judgeable transects (not just the 1,194 anchors).

WHY THIS EXISTS. An earlier scale-delta script grids on `LON0=-60,
LAT1=-2` -- the AREA grid anchored on the integer degree -- while the
parquet used by the E4/T3 pipeline grids on
`LON0=-60+0.5*STEP, LAT1=-2-0.5*STEP` -- the NATIVE grid, centered on the
Copernicus node. The two grids use the SAME integer pair (row, col) for
DIFFERENT cells: native cell (i,j) covers 1/4 of each of the four
neighboring area cells. A join on (transect, row, col) between the area
delta table and the E4 dump therefore measures the delta of the area cell
of the same index, with 1/4-area overlap, never "the same cells". This
script closes that gap: same logic as the area-grid script (lower-shell
estimator, interpolated anchor quantile p2, Richardson extrapolation over
4 scales, per-transect cluster bootstrap), but with cells defined on the
NATIVE geometry.

WHAT CHANGES relative to the area-grid script: ONLY the grid origin (LON0,
LAT1, parameters --dlon-px/--dlat-px, default 0.5/0.5 -- matching the
native gridding script) and the list of transects (the 13 JUDGEABLE ones,
read from `juiz_grade_nativa_diretas.json.offsets_por_transecto`, not all
19). The shell estimator (`casca_por_grupo`), the 4 scales
(30.7/15.4/10.2/6.1 m), the cluster bootstrap (B=10,000, seed 42) and the
least-squares extrapolation are COPIED VERBATIM from the area-grid script
-- no new logic.

GUARDS (fail-loud; nothing is adjusted to make them pass):
  (i) Reproduction: with --dlon-px 0 --dlat-px 0 over the 19 transects (the
      original AREA grid), the 0-3 m and >30 m points must match the
      area-grid reference to 1e-6 m. Runs BEFORE the native computation
      (saves CPU if the logic has diverged) and aborts with RuntimeError if
      it does not match -- proof that only the geometry changed, not the
      method.
  (ii) Native delta on a known population: on the cells of the E4 dump
      (join by quad+idx_no+transect, ETH class ">30m"), the mean native
      delta must fall within [+3.571; +3.760] m -- the range measured on
      the four neighboring area cells. Runs AFTER the native computation,
      on the real population. If it falls outside that range, the script
      raises RuntimeError with the numbers; the failure is recorded as-is,
      without recalibrating the range or the code to fit.

SOURCE CHECK BEFORE READING. Before re-reading any .laz file, its sha256 is
compared against the `_fontes` record of the previous run that read it in
full; if a file changed, the script aborts BEFORE spending CPU.

OUTPUT
  delta_escala_celulas_nativa.parquet  -- per cell: transect, quad,
      idx_no, row, col, z30, z0, delta, resid_max, canopy, n30 (same keys
      as the native-grid ground parquet: quad + idx_no).
  medir_delta_escala_v3_nativa.json    -- stamped (_proveniencia/_fontes),
      declared geometry, guards (i) and (ii), per_class results (13
      transects, native grid), area_grid guard (19 transects, area grid,
      informative).

USAGE
  python medir_delta_escala_v3_nativa.py
      # guard (i) [area, 19T] -> native [13T judgeable] -> guard (ii)
  python medir_delta_escala_v3_nativa.py --sem-guarda-i    # skips (i), debug only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
from pyproj import Transformer

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import proveniencia as PROV          # noqa: E402

# The area-grid script recorded a Windows source path in its `_fontes`
# block. On this host the same 19 .laz files (same sha256) are at
# /trabalho/GNN_TOPO/SATELITES/lidar_eba.
DIR = Path("/trabalho/GNN_TOPO/SATELITES/lidar_eba")
GRID = 7200
PASSO = 2.0 / GRID
CASCA_M = 1.0
EPSG = {"NP_T-0390": 31980, "NP_T-0391": 31980}
CHUNK = 5_000_000
SUB = [1, 2, 3, 5]                    # identical to the area-grid script
ESCALAS = [30.7 / s for s in SUB]
FAIXAS = [-1, 3, 10, 20, 30, 500]
ROT_F = ["0-3m", "3-10m", "10-20m", "20-30m", ">30m"]
MIN_PTS_SUB = 60
B = 10_000
FAIXA_GUARDA_II = (3.571, 3.760)       # measured range from the four neighboring area cells

# Guard (ii) was originally mis-specified: it assumed the native delta
# would fall inside the convex hull of the four neighboring area deltas,
# without any such guarantee (delta is a NON-LINEAR function of the point
# cloud). The result is NOT adjusted, the range is NOT widened. sha256 of
# the parquet that produced the original guard (ii) result (checked BEFORE
# use in the --gravar-de-parquet mode, which does NOT re-read any .laz):
SHA256_CELULAS_NATIVA_CONFERIDO = "110d75ca56626f268f32ece2acb2d0719ff7a19ced3e8f2b18fa3c122b2a14c2"
DECISAO_GUARDA_II_CHEFE = (
    ("guard misspecified (convex hull without guarantee); result accepted "
     "conditional on an independent recomputation"))
# Guard (i): BIT-EXACT reproduction (0.000e+00 difference) of the area-grid
# reference in both target classes. Reference values = the area-grid
# script's per-class output.
GUARDA_I_REGISTRADA_11_41 = {
    "0-3m": {"delta_m_recomputado": 0.14633569924401027,
             "delta_m_referencia": 0.14633569924401027,
             "diferenca_m": 0.0, "bate": True},
    ">30m": {"delta_m_recomputado": 4.9925998123264606,
             "delta_m_referencia": 4.9925998123264606,
             "diferenca_m": 0.0, "bate": True},
    "n_celulas": 152567, "n_transectos": 19, "tolerancia_m": 1e-6,
    "nota": ("Recorded from the run log of 2026-09-28 11:09-11:41; NOT recomputed in "
             "this mode, which does not reread any .laz file")}


def log(m=""):
    print(m, flush=True)


def sha256(p: Path) -> str:
    return PROV.sha256(p)


def casca_por_grupo(cid: np.ndarray, z: np.ndarray, min_pts: int):
    """IDENTICAL to the area-grid script: median of the lower shell
    (z <= p2 + CASCA_M) per group, with p2 = the interpolated 0.02 quantile."""
    ini = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]])
    fim = np.r_[ini[1:], len(cid)]
    n = fim - ini
    out = np.full(len(ini), np.nan, np.float64)
    for g, (a, b) in enumerate(zip(ini, fim)):
        if n[g] < min_pts:
            continue
        v = z[a:b]
        p2 = np.quantile(v, 0.02)
        out[g] = np.median(v[v <= p2 + CASCA_M])
    return cid[ini], out, n


def conferir_fontes_previamente(transectos: list[str], regua_ref: dict) -> list[dict]:
    """Compares the sha256 of each .laz file, BEFORE re-reading it, against
    the `_fontes` record of the area-grid script's reference run. Aborts if
    any file changed since that run. Returns the list of sources (for the
    `_fontes` block of the output JSON)."""
    ref_por_nome = {}
    for f in regua_ref.get("_fontes", []):
        nome = Path(f["caminho"]).stem
        ref_por_nome[nome] = f["sha256"]
    conferidas = []
    for t in transectos:
        arq = DIR / f"{t}.laz"
        if not arq.exists():
            raise RuntimeError(f"conferencia previa: {arq} nao existe")
        h = sha256(arq)
        ref = ref_por_nome.get(t)
        if ref is not None and h != ref:
            raise RuntimeError(
                f"conferencia previa de fontes: {t}.laz mudou desde a corrida "
                f"de delta_escala_regua_v3.json (sha256 atual {h[:12]}... != "
                f"referencia {ref[:12]}...). PARANDO antes de gastar CPU "
                "(regra do protocolo D-1: sha256 conferido ANTES de calcular).")
        conferidas.append({"transecto": t, "sha256": h, "sha256_referencia": ref,
                           "conferido_contra": "delta_escala_regua_v3.json" if ref else None})
        log(f"  conferencia previa OK: {t}.laz sha256={h[:12]}...")
    return conferidas


def processar_transectos(transectos: list[str], lon0: float, lat1: float) -> pd.DataFrame:
    """Reads the .laz files for `transectos`, grids on (lon0, lat1), computes
    z30 and the 3 sub-scales, extrapolates (Richardson) and returns the
    per-cell DataFrame with quad/idx_no (same formula as the native
    gridding script: quad from (row<3600, col<3600); idx_no =
    (row%3600)*3600 + (col%3600)). Literal copy of the area-grid script's
    main-loop logic; only the grid origin and the transect list change."""
    t0 = time.time()
    linhas = []
    for nome in transectos:
        arq = DIR / f"{nome}.laz"
        tr = Transformer.from_crs(EPSG.get(nome, 31981), 4326, always_xy=True)
        lons, lats, zs = [], [], []
        with laspy.open(arq) as las:
            for ch in las.chunk_iterator(CHUNK):
                z = np.asarray(ch.z, dtype=np.float32)
                lon, lat = tr.transform(np.asarray(ch.x, dtype=np.float64),
                                        np.asarray(ch.y, dtype=np.float64))
                ok = (z > -5.0) & (z < 400.0)
                lons.append(lon[ok]); lats.append(lat[ok]); zs.append(z[ok])
        lon = np.concatenate(lons); lat = np.concatenate(lats)
        z = np.concatenate(zs)
        del lons, lats, zs

        col30 = np.floor((lon - lon0) / PASSO).astype(np.int64)
        lin30 = np.floor((lat1 - lat) / PASSO).astype(np.int64)
        dentro = (col30 >= 0) & (col30 < GRID) & (lin30 >= 0) & (lin30 < GRID)
        lon, lat, z = lon[dentro], lat[dentro], z[dentro]
        col30, lin30 = col30[dentro], lin30[dentro]
        cid30 = lin30 * GRID + col30

        o = np.lexsort((z, cid30))
        c0, z0v = cid30[o], z[o]
        u30, med30, n30 = casca_por_grupo(c0, z0v, MIN_PTS_SUB)
        ini = np.flatnonzero(np.r_[True, c0[1:] != c0[:-1]])
        fim = np.r_[ini[1:], len(c0)]
        p95 = z0v[np.minimum(ini + ((fim - ini) * 95) // 100, fim - 1)]
        lin_u, col_u = u30 // GRID, u30 % GRID
        quad = np.where(lin_u < 3600, np.where(col_u < 3600, "Q1", "Q2"),
                        np.where(col_u < 3600, "Q3", "Q4"))
        idx_no = (lin_u % 3600) * 3600 + (col_u % 3600)
        base = pd.DataFrame({"cid": u30, "z30": med30, "n30": n30,
                             "dossel": p95 - med30, "lin": lin_u, "col": col_u,
                             "quad": quad, "idx_no": idx_no.astype(np.int64)})

        for s in SUB[1:]:
            fx = np.floor(((lon - lon0) / PASSO % 1.0) * s).astype(np.int64)
            fy = np.floor(((lat1 - lat) / PASSO % 1.0) * s).astype(np.int64)
            sub_id = cid30 * (s * s) + fy * s + fx
            o = np.lexsort((z, sub_id))
            us, meds, ns = casca_por_grupo(sub_id[o], z[o], MIN_PTS_SUB)
            df = pd.DataFrame({"cid": us // (s * s), "zs": meds})
            agg = df.dropna().groupby("cid")["zs"].mean()
            base[f"z{s}"] = base["cid"].map(agg)

        base["transecto"] = nome
        linhas.append(base)
        log(f"  {nome}: {len(base):,} celulas | {time.time()-t0:.0f} s")

    d = pd.concat(linhas, ignore_index=True).dropna(subset=["z30"] + [f"z{s}" for s in SUB[1:]])
    d["faixa"] = pd.cut(d["dossel"], FAIXAS, labels=ROT_F)

    L = np.array(ESCALAS)
    cols = ["z30"] + [f"z{s}" for s in SUB[1:]]
    Z = d[cols].to_numpy()
    A = np.vstack([np.ones_like(L), L]).T
    coef, *_ = np.linalg.lstsq(A, Z.T, rcond=None)
    d["z0"] = coef[0]
    d["delta"] = d["z0"] - d["z30"]
    resid = Z.T - A @ coef
    d["resid_max"] = np.abs(resid).max(axis=0)
    return d


def estatisticas_por_faixa(d: pd.DataFrame, seed: int = 42) -> dict:
    """IDENTICAL to the cluster bootstrap of the area-grid script's main loop."""
    rng = np.random.default_rng(seed)
    trs = d["transecto"].unique()
    out = {"n_celulas": int(len(d)), "transectos": len(trs), "por_faixa": {}}
    for rot in ROT_F + ["TODAS"]:
        s = d if rot == "TODAS" else d[d["faixa"] == rot]
        if len(s) < 50:
            continue
        g = s.groupby("transecto")["delta"].agg(["sum", "count"])
        g = g[g["count"] > 0]
        soma = g["sum"].to_numpy(); cnt = g["count"].to_numpy().astype(float)
        T = len(g)
        J = rng.integers(0, T, size=(B, T))
        bs_pond = soma[J].sum(axis=1) / cnt[J].sum(axis=1)
        bs_nao = (soma[J] / cnt[J]).mean(axis=1)
        ic = (float(np.percentile(bs_pond, 2.5)), float(np.percentile(bs_pond, 97.5)))
        ponto = float(s["delta"].mean())
        assert ic[0] <= ponto <= ic[1], f"{rot}: estimando do ponto difere do bootstrap"
        out["por_faixa"][rot] = {
            "n": int(len(s)), "n_transectos": int(T), "z30_medio": float(s["z30"].mean()),
            "estimando": "media de delta ponderada por celula (cluster = transecto)",
            "delta_m": ponto, "ic95": list(ic), "B": int(B),
            "delta_m_media_de_transectos": float((soma / cnt).mean()),
            "ic95_media_de_transectos": [float(np.percentile(bs_nao, 2.5)), float(np.percentile(bs_nao, 97.5))],
            "resid_mediano_m": float(s["resid_max"].median())}
        log(f"  {rot:8s} n={len(s):>7,d} delta={s['delta'].mean():+.4f} m  IC[{ic[0]:+.4f},{ic[1]:+.4f}]")
    return out


def rodar_guarda_i(regua_ref: dict, fontes_conferidas_todas19: list[str]) -> dict:
    """(i) AREA grid (dlon=dlat=0), 19 transects: must reproduce the
    area-grid reference output to 1e-6 m at 0-3m and >30m. Fail-loud."""
    log("\n=== GUARDA (i): grade de area, 19 transectos (reproducao) ===")
    d_area = processar_transectos(fontes_conferidas_todas19, lon0=-60.0, lat1=-2.0)
    stats = estatisticas_por_faixa(d_area, seed=42)
    alvo = regua_ref["por_faixa"]
    tol = 1e-6
    resultado = {}
    for rot in ("0-3m", ">30m"):
        recomputado = stats["por_faixa"][rot]["delta_m"]
        referencia = alvo[rot]["delta_m"]
        dif = recomputado - referencia
        resultado[rot] = {"delta_m_recomputado": recomputado,
                          "delta_m_referencia": referencia,
                          "diferenca_m": dif, "tolerancia_m": tol,
                          "bate": abs(dif) <= tol}
        log(f"  {rot}: recomputado={recomputado:.10f} referencia={referencia:.10f} "
            f"diff={dif:.3e} bate={abs(dif) <= tol}")
        if abs(dif) > tol:
            raise RuntimeError(
                f"GUARDA (i) FALHOU na classe {rot}: delta recomputado "
                f"{recomputado:.10f} != delta_escala_regua_v3.json "
                f"{referencia:.10f} (diferenca {dif:.6f} m > tolerancia {tol} m). "
                "A logica de medir_delta_escala_v3_nativa.py diverge de "
                "medir_delta_escala_v3.py na configuracao de grade de area -- "
                "PARANDO sem ajustar nada para caber (regra do protocolo D-1).")
    resultado["n_celulas"] = stats["n_celulas"]
    resultado["n_transectos"] = stats["transectos"]
    resultado["n_celulas_referencia"] = regua_ref.get("n_celulas")
    resultado["n_transectos_referencia"] = regua_ref.get("transectos")
    log("  GUARDA (i): PASSOU (0-3m e >30m batem a 1e-6 m com a grade de area)")
    return resultado


def rodar_guarda_ii(d_nativa: pd.DataFrame, dump_path: Path) -> tuple[dict, list]:
    """(ii) on the cells of the E4 dump (auditar_tabela3_dump.parquet),
    ETH class >30 m: the mean native delta must fall within
    [+3.571; +3.760] m. Fail-loud, without adjusting the range or the code."""
    log("\n=== GUARDA (ii): delta nativo nas celulas do dump do E4 (classe >30m) ===")
    dump = pd.read_parquet(dump_path)
    faltam = [c for c in ("quad", "idx_no", "transecto", "classe_eth") if c not in dump.columns]
    if faltam:
        raise RuntimeError(f"auditar_tabela3_dump.parquet sem colunas esperadas: {faltam}")
    alvo = dump[dump["classe_eth"] == ">30m"][["quad", "idx_no", "transecto"]].copy()
    n_dump_alvo = len(alvo)
    cel = d_nativa[["quad", "idx_no", "transecto", "delta"]].copy()
    j = alvo.merge(cel, on=["quad", "idx_no", "transecto"], how="left", validate="one_to_one")
    n_match = int(j["delta"].notna().sum())
    n_falta = int(j["delta"].isna().sum())
    media = float(j["delta"].mean(skipna=True))
    lo, hi = FAIXA_GUARDA_II
    dentro = lo <= media <= hi
    resultado = {
        "n_celulas_dump_maior30m": n_dump_alvo,
        "n_celulas_com_delta_nativo": n_match,
        "n_celulas_sem_par_no_delta_nativo": n_falta,
        "media_delta_nativo_m": media,
        "faixa_esperada_m": list(FAIXA_GUARDA_II),
        "fonte_faixa_esperada": "Expected range taken from the T3 half-pixel alignment check output",
        "dentro_da_faixa": dentro,
    }
    log(f"  dump >30m: {n_dump_alvo} celulas | casadas com delta nativo: {n_match} | "
        f"sem par: {n_falta}")
    log(f"  media do delta nativo (celulas casadas) = {media:+.4f} m | "
        f"faixa esperada [{lo:+.3f}, {hi:+.3f}] m | dentro={dentro}")
    if not dentro:
        raise RuntimeError(
            (f"GUARD (ii) FAILED: mean native delta in class >30m ({media:+.4f} m, n={n_match}) "
             f"outside [{lo:+.3f}, {hi:+.3f}] m (range of the 4 area cells of the T3 half-pixel "
             f"alignment check). STOPPING without adjusting the range or the code to fit "
             f"(protocol rule) -- see the run log for the explanation."))
    log("  GUARDA (ii): PASSOU")
    if n_falta:
        log(f"  AVISO: {n_falta} celulas do dump >30m sem par no delta nativo "
            "(fora do escopo da guarda: reportado, nao investigado a causa)")
    return resultado, j[j["delta"].isna()][["quad", "idx_no", "transecto"]].to_dict("records")


def por_faixa_sem_ic(d: pd.DataFrame, col_faixa: str) -> dict:
    """Descriptive statistics WITHOUT a confidence interval (the CI comes
    from a separate auditor, not from this recording step): per class,
    n_celulas, n_transectos, cell-weighted mean delta (= simple pool mean,
    since each row is one cell)."""
    out = {}
    for rot in ROT_F + ["TODAS"]:
        s = d if rot == "TODAS" else d[d[col_faixa] == rot]
        if len(s) == 0:
            continue
        out[rot] = {
            "n_celulas": int(len(s)),
            "n_transectos": int(s["transecto"].nunique()),
            "media_pond_celula_m": float(s["delta"].mean()),
        }
    return out


def gravar_json_de_parquet(a) -> int:
    """`--gravar-de-parquet` mode: the original guard (ii) was
    mis-specified (convex hull without a guarantee; delta is a NON-LINEAR
    function of the point cloud). The result is NOT adjusted, the range is
    NOT widened. This mode does NOT re-read any .laz: it only checks the
    sha256 of the already-recorded parquet (BEFORE use) and recomputes
    descriptive statistics (without a CI) from it and from the E4 dump."""
    t0 = time.time()
    celulas_p = Path(a.celulas)
    h = sha256(celulas_p)
    log(f"  conferencia previa: {celulas_p.name} sha256={h}")
    if h != SHA256_CELULAS_NATIVA_CONFERIDO:
        raise RuntimeError(
            (f"pre-check: sha256 of {celulas_p.name} ({h}) != the previously verified hash "
             f"({SHA256_CELULAS_NATIVA_CONFERIDO}). STOPPING before use -- not recording "
             f"statistics on a parquet different from the one that produced the original "
             f"guard (ii)."))
    log("  sha256 confere com o parquet da corrida de 11:09-11:41 (guarda ii original)")

    d = pd.read_parquet(celulas_p)
    d["faixa"] = pd.cut(d["dossel"], FAIXAS, labels=ROT_F)
    julgaveis = sorted(d["transecto"].unique())
    log(f"  {len(d):,} celulas, {len(julgaveis)} transectos (lido do parquet, SEM reler .laz)")

    populacao_13T_sem_ic = por_faixa_sem_ic(d, "faixa")
    log("  populacao inteira (13T, classe pelo dossel do proprio delta), sem IC:")
    for rot, v in populacao_13T_sem_ic.items():
        log(f"    {rot:8s} n={v['n_celulas']:>7,d} n_t={v['n_transectos']:2d} "
            f"delta={v['media_pond_celula_m']:+.4f} m")

    dump_p = Path(a.dump_e4)
    h_dump = sha256(dump_p)
    log(f"  {dump_p.name} sha256={h_dump} (registrado, sem referencia previa fixa)")
    dump = pd.read_parquet(dump_p)
    j = dump[["quad", "idx_no", "transecto", "classe_eth"]].merge(
        d[["quad", "idx_no", "transecto", "delta"]],
        on=["quad", "idx_no", "transecto"], how="left", validate="one_to_one")
    dump_sem_ic = por_faixa_sem_ic(j.rename(columns={"classe_eth": "faixa"}), "faixa")
    log("  celulas do dump do E4 (classe ETH exogena), sem IC:")
    for rot, v in dump_sem_ic.items():
        log(f"    {rot:8s} n={v['n_celulas']:>7,d} n_t={v['n_transectos']:2d} "
            f"delta={v['media_pond_celula_m']:+.4f} m")

    alvo = j[j["classe_eth"] == ">30m"]
    valor = float(alvo["delta"].mean())
    lo, hi = FAIXA_GUARDA_II
    passou = bool(lo <= valor <= hi)
    guarda_ii = {
        "valor": valor,
        "faixa": [lo, hi],
        "passou": passou,
        "n_celulas_dump_maior30m": int(len(alvo)),
        "n_celulas_com_delta_nativo": int(alvo["delta"].notna().sum()),
        "n_celulas_sem_par": int(alvo["delta"].isna().sum()),
        "decisao": DECISAO_GUARDA_II_CHEFE,
    }
    log((f"  GUARD (ii): value={valor:+.6f} m range=[{lo},{hi}] passed={passou} -- NOT "
         f"adjusted, NOT widened"))

    lon0 = -60.0 + 0.5 * PASSO
    lat1 = -2.0 - 0.5 * PASSO
    out = {
        "hipotese": "delta de escala (Richardson, casca inferior) na GEOMETRIA "
                    "NATIVA (+0,5 px), mesma logica de medir_delta_escala_v3.py, "
                    "para todas as celulas dos 13 transectos julgaveis",
        "modo": "gravar-de-parquet: NAO rele .laz; estatisticas recalculadas a "
                "partir de delta_escala_celulas_nativa.parquet ja gravado "
                "(corrida de 2026-09-28 11:09-11:41) e de auditar_tabela3_dump.parquet",
        "geometria": {"LON0": lon0, "LAT1": lat1, "PASSO": PASSO, "GRID": GRID,
                      "dlon_px": 0.5, "dlat_px": 0.5,
                      "convencao": "identica a gridar_lidar_eba_v3_nativa.py"},
        "escalas_m": [float(x) for x in ESCALAS],
        "transectos_julgaveis": julgaveis,
        "n_transectos_julgaveis": len(julgaveis),
        "n_celulas": int(len(d)),
        "guarda_i_reproducao_grade_area_19T": GUARDA_I_REGISTRADA_11_41,
        "guarda_ii": guarda_ii,
        "populacao_13T_classe_pelo_dossel_proprio_sem_ic": populacao_13T_sem_ic,
        "populacao_dump_e4_classe_eth_exogena_sem_ic": dump_sem_ic,
        "celulas_parquet": str(celulas_p),
        "sha256_celulas_parquet": h,
        "versao": ("v2 (D-1): guard (ii) recorded as it came out (the original guard was "
                   "ill-specified -- convex hull without guarantee; no adjustment, no "
                   "widening of the range); per-class statistics WITHOUT CI (CIs are left to "
                   "the v2 audit script)"),
        "cpu_total_s": round(time.time() - t0, 1),
    }
    PROV.gravar(Path(a.saida), out,
               fontes_lidas=[celulas_p, dump_p, Path(__file__)], script=__file__)
    log(f"\n  -> {Path(a.saida).name} | CPU total {out['cpu_total_s']:.1f} s (sem reler .laz)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dlon-px", type=float, default=0.5)
    ap.add_argument("--dlat-px", type=float, default=0.5)
    ap.add_argument("--regua-ref", default=str(RAIZ / "delta_escala_regua_v3.json"))
    ap.add_argument("--juiz", default=str(RAIZ / "juiz_grade_nativa_diretas.json"))
    ap.add_argument("--dump-e4", default=str(RAIZ / "auditar_tabela3_dump.parquet"))
    ap.add_argument("--saida", default=str(RAIZ / "medir_delta_escala_v3_nativa.json"))
    ap.add_argument("--celulas", default=str(RAIZ / "delta_escala_celulas_nativa.parquet"))
    ap.add_argument("--sem-guarda-i", action="store_true",
                    help="pula a guarda (i) [DEBUG SOMENTE; nao usar na corrida citavel]")
    ap.add_argument("--gravar-de-parquet", action="store_true",
                    help=("Does NOT re-read the .laz files: writes the JSON from the already "
                          "computed parquet, recording guard (ii) as it came out (the original "
                          "guard was poorly specified)"))
    a = ap.parse_args()

    if a.gravar_de_parquet:
        return gravar_json_de_parquet(a)

    t0 = time.time()
    regua_ref = json.loads(Path(a.regua_ref).read_text(encoding="utf-8"))
    juiz = json.loads(Path(a.juiz).read_text(encoding="utf-8"))
    julgaveis = sorted(t for t, v in juiz["offsets_por_transecto"].items() if v["julgavel"])
    todos19 = sorted(p.stem for p in DIR.glob("NP_T-*.laz"))
    log(f"  transectos julgaveis (fonte: juiz_grade_nativa_diretas.json): "
        f"{len(julgaveis)} -> {julgaveis}")
    log(f"  todos os transectos em disco: {len(todos19)}")
    faltam = set(julgaveis) - set(todos19)
    if faltam:
        raise RuntimeError(f".laz ausentes para transectos julgaveis: {sorted(faltam)}")

    log("\n=== STEP 0: conferencia previa de fontes (sha256 ANTES de calcular) ===")
    fontes_19 = conferir_fontes_previamente(todos19, regua_ref)
    fontes_13 = [f for f in fontes_19 if f["transecto"] in set(julgaveis)]

    guarda_i = None
    if not a.sem_guarda_i:
        guarda_i = rodar_guarda_i(regua_ref, todos19)
    else:
        log("\n=== GUARDA (i) PULADA (--sem-guarda-i; NAO usar para o numero citavel) ===")

    lon0 = -60.0 + a.dlon_px * PASSO
    lat1 = -2.0 - a.dlat_px * PASSO
    log(f"\n=== STEP 2: calculo NATIVO -- {len(julgaveis)} transectos julgaveis, "
        f"LON0={lon0:.9f} LAT1={lat1:.9f} (dlon {a.dlon_px:+.2f} px, dlat {a.dlat_px:+.2f} px) ===")
    d_nativa = processar_transectos(julgaveis, lon0=lon0, lat1=lat1)
    log(f"\n  {len(d_nativa):,} celulas nativas com as {len(SUB)} escalas, "
        f"{d_nativa['transecto'].nunique()} transectos")
    stats_nativa = estatisticas_por_faixa(d_nativa, seed=42)

    cols_dump = ["transecto", "quad", "idx_no", "lin", "col", "z30", "z0",
                "delta", "resid_max", "dossel", "n30"]
    celulas_out = Path(a.celulas)
    d_nativa[cols_dump].to_parquet(celulas_out, index=False)
    log(f"\n  -> {celulas_out.name} ({len(d_nativa):,} linhas, chave quad+idx_no)")

    guarda_ii, sem_par = rodar_guarda_ii(d_nativa, Path(a.dump_e4))

    out = {
        "hipotese": "delta de escala (Richardson, casca inferior) recomputado na "
                    "GEOMETRIA NATIVA (+0,5 px), mesma logica de medir_delta_escala_v3.py, "
                    "para todas as celulas dos 13 transectos julgaveis",
        "geometria": {"LON0": lon0, "LAT1": lat1, "PASSO": PASSO, "GRID": GRID,
                      "dlon_px": a.dlon_px, "dlat_px": a.dlat_px,
                      "convencao": "identica a gridar_lidar_eba_v3_nativa.py: celula (i,j) "
                                  "cobre lon [LON0+j*PASSO, LON0+(j+1)*PASSO], "
                                  "lat [LAT1-(i+1)*PASSO, LAT1-i*PASSO]; quad/idx_no da "
                                  "grade do projeto (GRID=7200, quadrantes de 3600)"},
        "escalas_m": [float(x) for x in ESCALAS],
        "transectos_julgaveis": julgaveis,
        "n_transectos_julgaveis": len(julgaveis),
        "fonte_transectos_julgaveis": "juiz_grade_nativa_diretas.json.offsets_por_transecto[*].julgavel",
        "conferencia_previa_fontes": {"criterio": "sha256 de cada .laz comparado com "
                                      "_fontes de delta_escala_regua_v3.json ANTES de reler; "
                                      "aborta (RuntimeError) se divergir",
                                      "fontes_13_julgaveis": fontes_13},
        "guarda_i_reproducao_grade_area_19T": guarda_i,
        "n_celulas": stats_nativa["n_celulas"],
        "n_transectos": stats_nativa["transectos"],
        "por_faixa": stats_nativa["por_faixa"],
        "guarda_ii_delta_nativo_dump_e4_maior30m": guarda_ii,
        "celulas_sem_par_guarda_ii": sem_par,
        "celulas_parquet": str(celulas_out),
        "versao": "v1 (D-1, rigor 28/09): geometria nativa (+0,5 px), 13 transectos "
                  "julgaveis, todas as celulas (nao so ancoras); logica identica a "
                  "medir_delta_escala_v3.py (D-5 quantil interpolado, D-2b dump por celula)",
        "cpu_total_s": round(time.time() - t0, 1),
    }

    fontes_todas = [DIR / f"{t}.laz" for t in julgaveis]
    fontes_todas += [Path(a.regua_ref), Path(a.juiz), Path(a.dump_e4), Path(__file__)]
    solo_nativo_sidecar = Path("/trabalho/GNN_TOPO/SATELITES/lidar_eba/solo_lidar_30m_v3_nativa.json")
    if solo_nativo_sidecar.exists():
        fontes_todas.append(solo_nativo_sidecar)
    PROV.gravar(Path(a.saida), out, fontes_lidas=fontes_todas, script=__file__)
    log(f"\n  -> {Path(a.saida).name} | CPU total {out['cpu_total_s']:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
