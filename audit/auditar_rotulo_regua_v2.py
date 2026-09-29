"""D-2/P1/P3 -- v2 auditor of the label-vs-ruler term, with the native
delta, the equivalent MAE reduction, the transect (primary) and footprint
(required) units, a 5/10/20 km site grouping (descriptive), and the P1
budget table with the (S)/(O)/(R) criteria.

WHY IT EXISTS. This design closes after E4 (`auditar_tabela3.py`) and D-1
(`medir_delta_escala_v3_nativa.py`): the term measured by E4 is
A = solo_anc - z_ref (label AND ruler), and that design does not isolate
the label alone. This script does not edit E4 or D-1 -- it only joins
them and adds: a genuinely paired native delta (per cell, joined by
transect+quad+idx_no), the equivalent MAE reduction (removing the class
mean of A), a footprint unit (parameter-free rule: footprints sharing a
cell form one unit), a 5/10/20 km site grouping (descriptive), a
percentile CI per unit (B=10,000, seed 42), t-jackknife, the sign test,
LOO, a floor curve (declared after E4, no point selected), and the P1
budget table with a population column.

GUARDS (fail loud, checked BEFORE any new statistic):
  (1) sha256 of ALL inputs checked before computing; the
      delta_escala_celulas_nativa.parquet file must match the sha256
      independently checked for D-1 (110d75ca...).
  (2) reproduces auditar_tabela3.json (E4): cell-weighted mean of A in
      the >30m class = +6.651019 m, to 1e-6.
  (3) reproduces the mean native delta over E4's cells (>30m class) =
      +3.76544 m (the same population guard (ii) of D-1 measured), to
      1e-4.
If any guard fails, the script stops with a RuntimeError -- it never
adjusts anything to make a number fit.

PRE-REGISTERED CRITERIA (verbatim, not reinterpreted here):
  (S) Sign: positive in class c only if the lower bound of the
      cell-weighted mean of A is > 0 in BOTH the percentile and the
      t-jackknife CIs, in BOTH the transect AND footprint units.
  (O) Outweigh: holds in class c if the lower bound of A (mean AND
      median) and of the equivalent MAE reduction sit above the largest
      upper bound of |dMAE| among estimators measured against the lidar
      in the same class, in BOTH the transect AND footprint units.
  (R) Label alone: the conditional statement "A - delta > 0" is only
      made if the percentile AND the t-jackknife CIs exclude zero in
      BOTH the transect AND footprint units.
  Floor: no point on the curve is singled out.

Usage:
  python auditar_rotulo_regua_v2.py
"""
from __future__ import annotations

import hashlib
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import proveniencia as PROV          # noqa: E402

# ---------------------------------------------------------------- entradas
TABELA3_JSON = RAIZ / "auditar_tabela3.json"
DUMP_E4_PARQUET = RAIZ / "auditar_tabela3_dump.parquet"
NATIVA_JSON = RAIZ / "medir_delta_escala_v3_nativa.json"
NATIVA_PARQUET = RAIZ / "delta_escala_celulas_nativa.parquet"
NMAD_JSON = RAIZ / "auditar_nmad_pareado_nativa_diretas.json"
D8_JSON = RAIZ / "auditar_d8_triangulo.json"

SHA256_NATIVA_PARQUET_ESPERADO = (
    "110d75ca56626f268f32ece2acb2d0719ff7a19ced3e8f2b18fa3c122b2a14c2")
SAIDA_JSON = RAIZ / "auditar_rotulo_regua_v2.json"
SAIDA_DUMP = RAIZ / "auditar_rotulo_regua_v2_dump.parquet"

ROT_F = ["0-3m", "3-10m", "10-20m", "20-30m", ">30m"]
CONFIRMATORIAS = ["20-30m", ">30m"]
DESCRITIVAS = ["0-3m", "3-10m", "10-20m", "TODAS"]
TODAS_CLASSES = ROT_F + ["TODAS"]

B, SEED, ALPHA = 10_000, 42, 0.05
PISO_GRID = [1, 5, 10, 25, 50, 100, 200]
LIMIARES_SITIO_KM = [5, 10, 20]
LAT_REF_SITIO = -3.0
M_LAT_ARCSEC = 30.87
M_LON_ARCSEC = 30.87 * float(np.cos(np.radians(LAT_REF_SITIO)))


def log(m: str = "") -> None:
    print(m, flush=True)


# --------------------------------------------------------- sha256 previo
def sha256(p: Path) -> str:
    return PROV.sha256(p)


def conferir_sha256_entradas() -> dict:
    """Upfront check (fail loud): sha256 of ALL inputs BEFORE computing.
    delta_escala_celulas_nativa.parquet must match the value independently
    checked for D-1."""
    arquivos = [TABELA3_JSON, DUMP_E4_PARQUET, NATIVA_JSON, NATIVA_PARQUET,
                NMAD_JSON, D8_JSON]
    out = {}
    for p in arquivos:
        if not p.exists():
            raise RuntimeError(f"conferencia previa: entrada ausente: {p}")
        h = sha256(p)
        out[p.name] = h
        log(f"  sha256 {p.name} = {h[:16]}...")
    h_nativa = out[NATIVA_PARQUET.name]
    if h_nativa != SHA256_NATIVA_PARQUET_ESPERADO:
        raise RuntimeError(
            f"GUARDA sha256 FALHOU: {NATIVA_PARQUET.name} tem sha256 "
            f"{h_nativa} != o conferido pelo contra-auditor do D-1 "
            f"({SHA256_NATIVA_PARQUET_ESPERADO}). PARANDO antes de calcular "
            "(regra do protocolo: sha256 conferido ANTES de calcular).")
    log("  GUARDA sha256: delta_escala_celulas_nativa.parquet confere com "
        "o contra-auditado (110d75ca...)")
    return out


# --------------------------------------------------------------- rng util
def rng_stream(*partes: str) -> np.random.Generator:
    """One bootstrap stream per (class, unit, estimand) combination --
    determinism independent of processing order (same practice as
    auditar_tabela3.py:_rng_classe)."""
    chave = "|".join(str(p) for p in partes)
    h = int.from_bytes(hashlib.sha256(chave.encode()).digest()[:4], "big")
    return np.random.default_rng(np.random.SeedSequence([SEED, h]))


# ------------------------------------------------------- unidades: pegada
def montar_unidade_pegada(nativa: pd.DataFrame) -> tuple[dict, list]:
    """PARAMETER-FREE rule: footprints sharing a cell (quad, idx_no) form
    one unit. Recomputed here directly from the native geometry of the
    13 judgeable transects."""
    celulas = {t: set(zip(g["quad"], g["idx_no"]))
               for t, g in nativa.groupby("transecto")}
    ts = sorted(celulas)
    pai = {t: t for t in ts}

    def achar(x):
        while pai[x] != x:
            pai[x] = pai[pai[x]]
            x = pai[x]
        return x

    def unir(a, b):
        ra, rb = achar(a), achar(b)
        if ra != rb:
            pai[ra] = rb

    pares = []
    for a, b in itertools.combinations(ts, 2):
        k = len(celulas[a] & celulas[b])
        if k > 0:
            unir(a, b)
            pares.append({"a": a, "b": b, "celulas_compartilhadas": k})

    grupos: dict[str, list] = {}
    for t in ts:
        grupos.setdefault(achar(t), []).append(t)
    mapa = {}
    for membros in grupos.values():
        nome = "PEGADA_" + "_".join(sorted(membros))
        for m in membros:
            mapa[m] = nome
    log(f"  unidade pegada: {len(ts)} transectos -> {len(grupos)} pegadas "
        f"(pares com celula compartilhada: {pares})")
    return mapa, pares


# -------------------------------------------------------- unidades: sitio
def montar_unidades_sitio(nativa: pd.DataFrame, limiares_km: list[int]
                          ) -> tuple[dict, dict]:
    """Descriptive sensitivity check (not part of the criteria): site
    grouped by the MINIMUM distance between each transect's cell cloud,
    converted to meters (~30.87 m/arcsec, reference latitude -3.0
    degrees), clustered by single-linkage at the {5, 10, 20} km
    thresholds. No point is singled out."""
    ts = sorted(nativa["transecto"].unique())
    pts = {t: np.c_[g["lin"].to_numpy(dtype=float) * M_LAT_ARCSEC,
                    g["col"].to_numpy(dtype=float) * M_LON_ARCSEC]
           for t, g in nativa.groupby("transecto")}
    dist_m = {}
    for a, b in itertools.combinations(ts, 2):
        d, _ = cKDTree(pts[b]).query(pts[a], k=1)
        dist_m[(a, b)] = float(d.min())

    unidades = {}
    for lim_km in limiares_km:
        lim_m = lim_km * 1000.0
        pai = {t: t for t in ts}

        def achar(x):
            while pai[x] != x:
                pai[x] = pai[pai[x]]
                x = pai[x]
            return x

        def unir(a, b):
            ra, rb = achar(a), achar(b)
            if ra != rb:
                pai[ra] = rb

        for (a, b), d in dist_m.items():
            if d <= lim_m:
                unir(a, b)
        grupos: dict[str, list] = {}
        for t in ts:
            grupos.setdefault(achar(t), []).append(t)
        mapa = {}
        for membros in grupos.values():
            nome = "SITIO_" + "_".join(sorted(membros))
            for m in membros:
                mapa[m] = nome
        unidades[lim_km] = mapa
        log(f"  unidade sitio {lim_km} km: {len(grupos)} grupos")
    dist_km = {f"{a}|{b}": round(d / 1000.0, 3) for (a, b), d in dist_m.items()}
    return unidades, dist_km


# --------------------------------------------------- estimando por unidade
def estimando_unidade(df: pd.DataFrame, unit_col: str, val_col: str,
                      tag: str) -> dict:
    """Cell-weighted mean, median, per-unit mean; 95% percentile CI per
    unit (cluster bootstrap, B=10,000, seed 42); t-jackknife CI
    (delete-one, Tukey variance); sign test between units;
    leave-one-unit-out (LOO) range."""
    s = df.dropna(subset=[unit_col, val_col])
    grupos = {u: g[val_col].to_numpy(dtype=float)
              for u, g in s.groupby(unit_col) if len(g)}
    unidades = sorted(grupos)
    n_u = len(unidades)
    n_cel = int(sum(len(v) for v in grupos.values()))
    if n_u == 0:
        return {"n_celulas": 0, "n_unidades": 0}

    pool_tudo = np.concatenate([grupos[u] for u in unidades])
    pond = float(pool_tudo.mean())
    mediana = float(np.median(pool_tudo))
    medias_u = {u: float(np.mean(grupos[u])) for u in unidades}
    media_unidade = float(np.mean(list(medias_u.values())))

    if n_u >= 2:
        rng = rng_stream(tag, unit_col, val_col)
        arr = np.array(unidades, dtype=object)
        idx = rng.integers(0, n_u, size=(B, n_u))
        bs_pond = np.empty(B)
        bs_mediana = np.empty(B)
        bs_unidade = np.empty(B)
        for i in range(B):
            escolhidos = arr[idx[i]]
            pool = np.concatenate([grupos[u] for u in escolhidos])
            bs_pond[i] = pool.mean()
            bs_mediana[i] = np.median(pool)
            bs_unidade[i] = np.mean([medias_u[u] for u in escolhidos])

        def ic(v):
            return [float(np.percentile(v, 100 * ALPHA / 2)),
                    float(np.percentile(v, 100 * (1 - ALPHA / 2)))]

        ic_pond, ic_mediana, ic_unidade = ic(bs_pond), ic(bs_mediana), ic(bs_unidade)

        theta_i = np.array([
            float(np.concatenate([grupos[uu] for uu in unidades if uu != u]).mean())
            for u in unidades])
        theta_bar = float(theta_i.mean())
        var_jk = (n_u - 1) / n_u * float(np.sum((theta_i - theta_bar) ** 2))
        se_jk = float(np.sqrt(var_jk))
        tcrit = float(stats.t.ppf(1 - ALPHA / 2, df=n_u - 1))
        ic_t_jack = [pond - tcrit * se_jk, pond + tcrit * se_jk]
        loo_min, loo_max = float(theta_i.min()), float(theta_i.max())
    else:
        ic_pond = ic_mediana = ic_unidade = [None, None]
        ic_t_jack = [None, None]
        loo_min = loo_max = None

    positivos = sum(1 for v in medias_u.values() if v > 0)
    p_bilateral = (float(stats.binomtest(positivos, n_u, 0.5).pvalue)
                  if n_u >= 1 else None)

    return {
        "n_celulas": n_cel, "n_unidades": n_u, "unidades": unidades,
        "media_pond_celula_m": pond, "mediana_m": mediana,
        "media_por_unidade_m": media_unidade,
        "medias_por_unidade_m": medias_u,
        "ic95_pond_celula": ic_pond, "ic95_mediana": ic_mediana,
        "ic95_media_por_unidade": ic_unidade,
        "ic95_t_jackknife_pond": ic_t_jack,
        "loo_pond_min_m": loo_min, "loo_pond_max_m": loo_max,
        "teste_sinal": {"positivos": positivos, "n_unidades": n_u,
                        "p_bilateral": p_bilateral},
    }


# ----------------------------------------------- reducao de MAE equivalente
def mae_reducao_unidade(df: pd.DataFrame, unit_col: str, tag: str) -> dict:
    """Equivalent MAE reduction: MAE(|e_modelo|) -
    MAE(|e_modelo - media_de_classe(A)|). Same unit (m) as the
    estimator-vs-lidar contrasts. Cluster bootstrap per unit and
    delete-one-unit jackknife."""
    s = df.dropna(subset=[unit_col, "termo_rotulo", "e_modelo_m"])
    ga = {u: g["termo_rotulo"].to_numpy(dtype=float)
          for u, g in s.groupby(unit_col) if len(g)}
    ge = {u: g["e_modelo_m"].to_numpy(dtype=float)
          for u, g in s.groupby(unit_col) if len(g)}
    unidades = sorted(ga)
    n_u = len(unidades)
    n_cel = int(sum(len(v) for v in ga.values()))
    if n_u == 0:
        return {"n_celulas": 0, "n_unidades": 0}

    def reducao_de(us: list) -> float:
        A = np.concatenate([ga[u] for u in us])
        E = np.concatenate([ge[u] for u in us])
        media_a = A.mean()
        mae_antes = float(np.abs(E).mean())
        mae_depois = float(np.abs(E - media_a).mean())
        return mae_antes - mae_depois

    ponto = reducao_de(unidades)
    if n_u >= 2:
        rng = rng_stream(tag, "reducao_mae", unit_col)
        arr = np.array(unidades, dtype=object)
        idx = rng.integers(0, n_u, size=(B, n_u))
        bs = np.empty(B)
        for i in range(B):
            bs[i] = reducao_de(list(arr[idx[i]]))
        ic_pond = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

        theta_i = np.array([reducao_de([u for u in unidades if u != uu])
                            for uu in unidades])
        theta_bar = float(theta_i.mean())
        var_jk = (n_u - 1) / n_u * float(np.sum((theta_i - theta_bar) ** 2))
        se_jk = float(np.sqrt(var_jk))
        tcrit = float(stats.t.ppf(1 - ALPHA / 2, df=n_u - 1))
        ic_t_jack = [ponto - tcrit * se_jk, ponto + tcrit * se_jk]
        loo_min, loo_max = float(theta_i.min()), float(theta_i.max())
    else:
        ic_pond = [None, None]
        ic_t_jack = [None, None]
        loo_min = loo_max = None

    return {"n_celulas": n_cel, "n_unidades": n_u,
            "reducao_mae_equivalente_m": float(ponto),
            "ic95_pond_celula": ic_pond, "ic95_t_jackknife": ic_t_jack,
            "loo_min_m": loo_min, "loo_max_m": loo_max}


# ------------------------------------------------------------- curva de piso
def curva_de_piso(df_classe: pd.DataFrame) -> dict:
    """Descriptive, labeled 'declared after E4'. The floor restricts the
    input CELLS (only cells from transects with >= floor cells in the
    class), transect unit (primary). No point is chosen."""
    cel_por_t = df_classe.groupby("transecto").size().to_dict()
    out = {}
    for piso in PISO_GRID:
        ts_ok = sorted(t for t, n in cel_por_t.items() if n >= piso)
        sub = df_classe[df_classe["transecto"].isin(ts_ok)]
        sub_delta = sub.dropna(subset=["delta"])
        out[str(piso)] = {
            "n_transectos": len(ts_ok),
            "n_celulas_A": int(len(sub)),
            "media_pond_A_m": float(sub["termo_rotulo"].mean()) if len(sub) else None,
            "n_celulas_delta": int(len(sub_delta)),
            "media_pond_delta_m": float(sub_delta["delta"].mean()) if len(sub_delta) else None,
        }
    return out


# ------------------------------------------------------- contrastes (P1)
def contrastes_lidar(nmad: dict, rot: str) -> tuple[list, float | None]:
    pares = ["v23 vs mlp", "v23 vs fabdem", "v23 vs gedtm30"]
    linhas = []
    for par in pares:
        node = nmad["testes_principal_ETH"].get(par, {}).get("mae", {}).get(rot)
        if not node or not node.get("testado"):
            linhas.append({"par": par, "testado": False,
                           "motivo": (node or {}).get("motivo", "classe ausente")})
            continue
        lo, hi = node["ic95_bootstrap_transecto_mediana"]
        teto = max(abs(lo), abs(hi))
        linhas.append({
            "par": par, "testado": True, "media_m": node["media_m"],
            "mediana_m": node["mediana_m"],
            "ic95_bootstrap_transecto_mediana_m": [lo, hi],
            "teto_abs_ddiferenca_m": teto, "n_transectos": node["n_transectos"],
        })
    testados = [r for r in linhas if r["testado"]]
    teto_geral = max(r["teto_abs_ddiferenca_m"] for r in testados) if testados else None
    return linhas, teto_geral


def tabela_orcamento(rot: str, por_unidade: dict, nmad: dict, tabela_pool: dict,
                     d8: dict) -> dict:
    """P1: 'outweigh' budget table, with a population column. A:
    cell-weighted mean (441/460 anchor cells); contrasts against the
    lidar measured over the entire judgeable forest (a much larger n) --
    DIFFERENT populations, declared in the column."""
    tr = por_unidade["transecto"]
    A = tr["A"]
    red = tr["reducao_mae_equivalente"]
    delta = tr["delta"]
    pareada = tr["pareada_A_menos_delta"]
    linhas = [
        {"linha": "A (media ponderada por celula)", "valor_m": A["media_pond_celula_m"],
         "ic95_m": A["ic95_pond_celula"], "unidade_cluster": "transecto",
         "populacao": f"{A['n_celulas']} celulas-ancora ({A['n_unidades']} transectos)",
         "fonte": "auditar_tabela3_dump.parquet (E4)"},
        {"linha": "A (mediana)", "valor_m": A["mediana_m"],
         "ic95_m": A["ic95_mediana"], "unidade_cluster": "transecto",
         "populacao": f"{A['n_celulas']} celulas-ancora ({A['n_unidades']} transectos)",
         "fonte": "auditar_tabela3_dump.parquet (E4)"},
        {"linha": "reducao de MAE equivalente (retirar media de classe de A)",
         "valor_m": red["reducao_mae_equivalente_m"], "ic95_m": red["ic95_pond_celula"],
         "unidade_cluster": "transecto",
         "populacao": f"{red['n_celulas']} celulas-ancora ({red['n_unidades']} transectos)",
         "fonte": "auditar_rotulo_regua_v2 (este script)"},
        {"linha": "delta nativo (D-1)", "valor_m": delta["media_pond_celula_m"],
         "ic95_m": delta["ic95_pond_celula"], "unidade_cluster": "transecto",
         "populacao": f"{delta['n_celulas']} celulas-ancora com par no delta nativo "
                      f"({delta['n_unidades']} transectos)",
         "fonte": "delta_escala_celulas_nativa.parquet (D-1)"},
        {"linha": "A - delta (pareada por celula)", "valor_m": pareada["media_pond_celula_m"],
         "ic95_m": pareada["ic95_pond_celula"], "unidade_cluster": "transecto",
         "populacao": f"{pareada['n_celulas']} celulas-ancora com par no delta nativo "
                      f"({pareada['n_unidades']} transectos)",
         "fonte": "auditar_rotulo_regua_v2 (este script)"},
    ]
    contrastes, teto_geral = contrastes_lidar(nmad, rot)
    n_pool = tabela_pool.get("v23", {}).get(rot, {}).get("n")
    for c in contrastes:
        if not c["testado"]:
            linhas.append({"linha": f"contraste {c['par']} (contra o lidar)",
                           "valor_m": None, "ic95_m": None, "unidade_cluster": "transecto",
                           "populacao": "nao testado: " + c["motivo"],
                           "fonte": "auditar_nmad_pareado_nativa_diretas.json"})
            continue
        linhas.append({
            "linha": f"contraste {c['par']} (contra o lidar)",
            "valor_m": c["media_m"], "mediana_m": c["mediana_m"],
            "ic95_m": c["ic95_bootstrap_transecto_mediana_m"],
            "unidade_cluster": "transecto",
            "populacao": f"{n_pool} celulas (floresta julgavel; POPULACAO DIFERENTE "
                         f"de A -- nao e a mesma amostra) ({c['n_transectos']} transectos)",
            "fonte": "auditar_nmad_pareado_nativa_diretas.json.testes_principal_ETH",
        })
    d8_delta = d8["delta_T_em_metros_mae"]
    linhas.append({
        "linha": "arvore - grafo (reserva D8, fator4; NAO E CONTRA O LIDAR)",
        "valor_m": d8_delta["media"], "ic95_m": d8_delta["ic95"],
        "unidade_cluster": "semente",
        "populacao": f"{d8_delta['n']} sementes, reserva sintetica (nao e o lidar)",
        "fonte": "auditar_d8_triangulo.json.delta_T_em_metros_mae",
    })
    linhas.append({
        "linha": "arvore contra o lidar", "valor_m": "not measured", "ic95_m": None,
        "unidade_cluster": None, "populacao": "not measured",
        "fonte": "P2 not run; never reported as zero",
    })
    return {"linhas": linhas, "maior_teto_abs_contrastes_lidar_m": teto_geral,
            "contrastes_lidar_detalhe": contrastes}


# ------------------------------------------------------------- criterios
def veredito_S(por_unidade: dict) -> dict:
    checagens = {}
    ok_geral = True
    for u in ("transecto", "pegada"):
        A = por_unidade[u]["A"]
        ic_p, ic_t = A["ic95_pond_celula"], A["ic95_t_jackknife_pond"]
        ok_p = ic_p[0] is not None and ic_p[0] > 0
        ok_t = ic_t[0] is not None and ic_t[0] > 0
        checagens[u] = {"ic95_pond_celula": ic_p, "exclui_zero_percentil": ok_p,
                        "ic95_t_jackknife": ic_t, "exclui_zero_t_jackknife": ok_t}
        ok_geral = ok_geral and ok_p and ok_t
    veredito = "positivo" if ok_geral else "nao_sustentado"
    motivo = None
    if not ok_geral:
        falhas = [u for u in ("transecto", "pegada")
                  if not (checagens[u]["exclui_zero_percentil"]
                          and checagens[u]["exclui_zero_t_jackknife"])]
        motivo = f"sensibilidade que falhou: unidade(s) {falhas}"
    return {"criterio": "(S) limite inferior de A > 0 no percentil E no "
                        "t-jackknife, nas unidades transecto E pegada",
            "checagens_por_unidade": checagens, "veredito": veredito, "motivo": motivo}


def veredito_O(por_unidade: dict, teto_geral: float | None) -> dict:
    if teto_geral is None:
        return {"criterio": "(O) outweigh", "veredito": "nao_demonstrado",
                "motivo": "nenhum contraste testado nesta classe (sem teto de referencia)"}
    checagens = {}
    ok_geral = True
    for u in ("transecto", "pegada"):
        A = por_unidade[u]["A"]
        red = por_unidade[u]["reducao_mae_equivalente"]
        lo_media = A["ic95_pond_celula"][0]
        lo_mediana = A["ic95_mediana"][0]
        lo_red = red["ic95_pond_celula"][0]
        ok_media = lo_media is not None and lo_media > teto_geral
        ok_mediana = lo_mediana is not None and lo_mediana > teto_geral
        ok_red = lo_red is not None and lo_red > teto_geral
        checagens[u] = {
            "lim_inf_A_media_m": lo_media, "acima_do_teto_media": ok_media,
            "lim_inf_A_mediana_m": lo_mediana, "acima_do_teto_mediana": ok_mediana,
            "lim_inf_reducao_mae_m": lo_red, "acima_do_teto_reducao": ok_red,
        }
        ok_geral = ok_geral and ok_media and ok_mediana and ok_red
    veredito = "sustentado" if ok_geral else "nao_demonstrado"
    return {"criterio": "(O) limite inferior de A (media E mediana) e da reducao de "
                        "MAE equivalente acima do maior teto de |contraste| entre "
                        "estimadores contra o lidar, nas unidades transecto E pegada",
            "teto_abs_contrastes_lidar_m": teto_geral,
            "checagens_por_unidade": checagens, "veredito": veredito}


def veredito_R(por_unidade: dict) -> dict:
    checagens = {}
    ok_geral = True
    for u in ("transecto", "pegada"):
        p = por_unidade[u]["pareada_A_menos_delta"]
        ic_p, ic_t = p["ic95_pond_celula"], p["ic95_t_jackknife_pond"]
        ok_p = ic_p[0] is not None and (ic_p[0] > 0 or ic_p[1] < 0)
        ok_t = ic_t[0] is not None and (ic_t[0] > 0 or ic_t[1] < 0)
        checagens[u] = {"ic95_pond_celula": ic_p, "exclui_zero_percentil": ok_p,
                        "ic95_t_jackknife": ic_t, "exclui_zero_t_jackknife": ok_t}
        ok_geral = ok_geral and ok_p and ok_t
    veredito = ("ordenacao_valida_A_menos_delta_maior_que_zero" if ok_geral
               else "nenhuma_ordenacao")
    return {"criterio": "(R) frase condicional 'A - delta > 0' so entra se percentil "
                        "E t-jackknife excluirem zero nas unidades transecto E pegada",
            "checagens_por_unidade": checagens, "veredito": veredito}


# --------------------------------------------------------------------- main
def main() -> int:
    log("=== conferencia previa de sha256 (ANTES de calcular) ===")
    shas = conferir_sha256_entradas()

    import json
    tabela3 = json.loads(TABELA3_JSON.read_text(encoding="utf-8"))
    nmad = json.loads(NMAD_JSON.read_text(encoding="utf-8"))
    d8 = json.loads(D8_JSON.read_text(encoding="utf-8"))

    dump = pd.read_parquet(DUMP_E4_PARQUET)
    nativa = pd.read_parquet(NATIVA_PARQUET)
    log(f"  dump E4: {len(dump):,} celulas | nativa D-1: {len(nativa):,} "
        f"celulas, {nativa['transecto'].nunique()} transectos")

    log("\n=== GUARDA (2): reproduz +6,651019 m do E4 (classe >30m, a 1e-6) ===")
    alvo_e4 = tabela3["por_classe_eth"][">30m"]["sem_piso"]["media_pond_celula_m"]
    recomp_e4 = float(dump.loc[dump["classe_eth"] == ">30m", "termo_rotulo"].mean())
    dif_e4 = recomp_e4 - alvo_e4
    log(f"  recomputado={recomp_e4:.9f} alvo={alvo_e4:.9f} diff={dif_e4:.3e}")
    if abs(dif_e4) > 1e-6:
        raise RuntimeError(
            f"GUARDA (2) FALHOU: media ponderada de A em >30m recomputada "
            f"{recomp_e4:.9f} != auditar_tabela3.json {alvo_e4:.9f} "
            f"(diferenca {dif_e4:.9f} m > 1e-6). PARANDO.")
    log("  GUARDA (2): PASSOU")

    df = dump.merge(
        nativa[["transecto", "quad", "idx_no", "delta"]],
        on=["transecto", "quad", "idx_no"], how="left", validate="one_to_one")
    n_sem_par = int(df["delta"].isna().sum())
    log(f"\n  join E4 x delta nativo por (transecto,quad,idx_no): "
        f"{len(df):,} celulas E4, {n_sem_par} sem par no delta nativo")

    log("\n=== GUARDA (3): reproduz +3,76544 m do delta nativo (classe >30m, a 1e-4) ===")
    alvo_delta = nativa_json = json.loads(NATIVA_JSON.read_text(encoding="utf-8"))
    alvo_delta = alvo_delta["guarda_ii"]["valor"]
    recomp_delta = float(df.loc[df["classe_eth"] == ">30m", "delta"].mean())
    dif_delta = recomp_delta - alvo_delta
    log(f"  recomputado={recomp_delta:.6f} alvo={alvo_delta:.6f} diff={dif_delta:.3e}")
    if abs(dif_delta) > 1e-4:
        raise RuntimeError(
            f"GUARDA (3) FALHOU: media do delta nativo em >30m recomputada "
            f"{recomp_delta:.6f} != medir_delta_escala_v3_nativa.json.guarda_ii.valor "
            f"{alvo_delta:.6f} (diferenca {dif_delta:.6f} m > 1e-4). PARANDO.")
    log("  GUARDA (3): PASSOU")

    df["pareada_A_menos_delta"] = df["termo_rotulo"] - df["delta"]

    log("\n=== unidades ===")
    mapa_pegada, pares_pegada = montar_unidade_pegada(nativa)
    mapas_sitio, dist_km = montar_unidades_sitio(nativa, LIMIARES_SITIO_KM)
    df["pegada"] = df["transecto"].map(mapa_pegada)
    for lim in LIMIARES_SITIO_KM:
        df[f"sitio_{lim}km"] = df["transecto"].map(mapas_sitio[lim])

    unit_cols = {"transecto": "transecto", "pegada": "pegada",
                "sitio_5km": "sitio_5km", "sitio_10km": "sitio_10km",
                "sitio_20km": "sitio_20km"}

    por_classe = {}
    for rot in TODAS_CLASSES:
        log(f"\n=== classe {rot} ===")
        s = df if rot == "TODAS" else df[df["classe_eth"] == rot]
        por_unidade = {}
        for nome_u, col_u in unit_cols.items():
            por_unidade[nome_u] = {
                "A": estimando_unidade(s, col_u, "termo_rotulo", f"{rot}|A|{nome_u}"),
                "delta": estimando_unidade(s, col_u, "delta", f"{rot}|delta|{nome_u}"),
                "fidelidade": estimando_unidade(s, col_u, "fidelidade_m",
                                                f"{rot}|fidelidade|{nome_u}"),
                "pareada_A_menos_delta": estimando_unidade(
                    s, col_u, "pareada_A_menos_delta", f"{rot}|pareada|{nome_u}"),
                "reducao_mae_equivalente": mae_reducao_unidade(
                    s, col_u, f"{rot}|reducao|{nome_u}"),
            }
        piso = curva_de_piso(s) if rot != "TODAS" else None
        veredito_e4_original = (tabela3["por_classe_eth"].get(rot, {}).get("veredito")
                                if rot in tabela3["por_classe_eth"] else None)
        por_classe[rot] = {
            "confirmatoria": rot in CONFIRMATORIAS,
            "n_celulas": int(len(s)),
            "por_unidade": por_unidade,
            "curva_piso_declarada_apos_E4": piso,
            "confirmatorio_E4_original_bruto": veredito_e4_original,
            "confirmatorio_E4": "nao_avaliavel",
        }
        A_t = por_unidade["transecto"]["A"]
        log(f"  A (transecto) pond={A_t.get('media_pond_celula_m')} "
            f"IC{A_t.get('ic95_pond_celula')} n_cel={A_t.get('n_celulas')}")

    log("\n=== tabela de orcamento P1 e criterios (S)/(O)/(R) ===")
    criterios = {}
    orcamento = {}
    tabela_pool = nmad["tabela_pool_ETH"]
    for rot in CONFIRMATORIAS:
        orc = tabela_orcamento(rot, por_classe[rot]["por_unidade"], nmad, tabela_pool, d8)
        orcamento[rot] = orc
        crit = {
            "S": veredito_S(por_classe[rot]["por_unidade"]),
            "O": veredito_O(por_classe[rot]["por_unidade"], orc["maior_teto_abs_contrastes_lidar_m"]),
            "R": veredito_R(por_classe[rot]["por_unidade"]),
        }
        criterios[rot] = crit
        log(f"  {rot}: S={crit['S']['veredito']} O={crit['O']['veredito']} "
            f"R={crit['R']['veredito']}")

    dump_saida = df.copy()
    dump_saida.to_parquet(SAIDA_DUMP, index=False)
    log(f"\n  -> {SAIDA_DUMP.name} ({len(dump_saida):,} linhas)")

    fontes = [TABELA3_JSON, DUMP_E4_PARQUET, NATIVA_JSON, NATIVA_PARQUET,
             NMAD_JSON, D8_JSON, Path(__file__)]

    PROV.gravar(SAIDA_JSON, {
        "hipotese": "o termo A (rotulo+regua) supera o termo de escolha do "
                    "estimador acima de 20 m de dossel ETH; o desenho NAO "
                    "isola o rotulo (consolidado 2026-09-28)",
        "nao_faz": ("does not edit auditar_tabela3.py (E4) or medir_delta_escala_v3_nativa.py "
                    "(D-1); does not interpret the result for the paper; does not audit its "
                    "own result"),
        "conferencia_previa_sha256": shas,
        "guardas": {
            "reproducao_E4_maior30m": {"recomputado_m": recomp_e4, "alvo_m": alvo_e4,
                                       "diferenca_m": dif_e4, "tolerancia_m": 1e-6,
                                       "passou": True},
            "reproducao_delta_nativo_E4_maior30m": {
                "recomputado_m": recomp_delta, "alvo_m": alvo_delta,
                "diferenca_m": dif_delta, "tolerancia_m": 1e-4, "passou": True,
                "nota": ("target = medir_delta_escala_v3_nativa.json.guarda_ii.valor "
                         "(3.7654373275574855 m); that original guard (ii) came out 'passou=False' "
                         "against the band [3.571;3.760], but the band was judged ill-specified "
                         "(convex hull without guarantee) and the number was accepted, conditional "
                         "on an independent check -- which has already come out CONFIRMED WITH "
                         "RESERVATIONS. This guard (3) checks that THIS script reproduces the SAME "
                         "number, not that it falls within the band [3.571;3.760].")},
        },
        "populacao": {
            "n_celulas_E4": int(len(df)), "n_celulas_com_par_delta_nativo": int(len(df) - n_sem_par),
            "n_celulas_sem_par_delta_nativo": n_sem_par,
            "transectos_julgaveis": sorted(nativa["transecto"].unique().tolist()),
            "n_transectos_julgaveis": int(nativa["transecto"].nunique()),
            "unidade_pegada": {"mapa": mapa_pegada, "n_pegadas": len(set(mapa_pegada.values())),
                              "pares_com_celula_compartilhada": pares_pegada},
            "unidade_sitio": {str(lim): {"mapa": mapas_sitio[lim],
                                        "n_sitios": len(set(mapas_sitio[lim].values()))}
                             for lim in LIMIARES_SITIO_KM},
            "distancias_min_km_entre_transectos": dist_km,
        },
        "criterios_pre_declarados": {
            "S": "positivo na classe c so se o limite inferior da media ponderada "
                "de A for > 0 no percentil E no t-jackknife, nas unidades "
                "transecto E pegada",
            "O": "outweigh vale na classe c se o limite inferior de A (media E "
                "mediana) e o da reducao de MAE equivalente ficarem acima do "
                "maior limite superior de |ΔMAE| entre estimadores medido "
                "contra o lidar na mesma classe, nas unidades transecto E pegada",
            "R": "a frase condicional 'A - delta > 0' so entra se o percentil e "
                "o t-jackknife excluirem zero na unidade transecto E pegada",
            "piso": "curva {1,5,10,25,50,100,200} descritiva, declarada apos o "
                   "E4; nenhum ponto e escolhido",
            "bootstrap": {"B": B, "semente": SEED, "alpha": ALPHA, "cluster": "unidade"},
        },
        "por_classe_eth": por_classe,
        "tabela_orcamento_P1": orcamento,
        "criterios_veredito": criterios,
        "dump": {"arquivo": SAIDA_DUMP.name, "n_linhas": int(len(dump_saida)),
                "colunas": list(dump_saida.columns),
                "chave": ["quad", "idx_no", "transecto"]},
        "contra_auditoria": ("PENDING -- not yet checked by anyone other than the author of this "
                             "script (rule: no one audits their own result)"),
    }, fontes_lidas=fontes, script=__file__)
    log(f"\n  -> {SAIDA_JSON.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
