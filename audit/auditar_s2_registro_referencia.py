"""S2 -- does the Table IV result DEPEND on registering the ruler to GLO-30?

The design and criterion below are FIXED beforehand (not edited by this
script; only read from the reused modules).

Does NOT edit juiz_lidar_v4.py, auditar_nmad_pareado.py or
auditar_nmad_pareado_nativa_anadem.py: it imports all three as modules and
REUSES their pure functions (J.montar, J.metricas, REF.nmad, REF.holm,
REF.rng_do_teste, REF.guardas_populacao, REFA.montar_populacao_com_anadem,
REFA.tabela_pool_generic, REFA.por_transecto_faixa_generic, REFA.rodar_par,
REFA.conferir_sha256_anadem, REFA.guarda_reproduz_referencia) instead of
duplicating logic. It only adds what S2 requires: recomputing `off_t` (and
only `off_t`) per arm, over the SAME fixed population (same cells, same
mask, same dedup) that REFA already builds.

ARMS (same population; only off_t per transect changes):
  (i)    current -- off_t = juiz_grade_nativa_diretas.offsets_por_transecto[t].offset_m
         (median of GLO-30 - z_solo_v2 in clearings)
  (ii-a) INDEPENDENT orbital registration -- off_t = median(raw GEDI/ATL08
         anchor approved ONLY by that mission's own hard quality criterion
         - z_solo_v2) in clearings. The raw anchor is gridded on the SAME
         native grid (rotulos_lidar_v23.celula + QUADS partition) and only
         passes through preparar_gedi/preparar_atl08 (hard criterion),
         WITHOUT the physical-plausibility cut that uses the GLO-30 DSM
         (that cut only happens in montar_bioma, not called here).
         Evaluable only if >=6 transects have >=10 approved shots in a
         clearing (N_CLAREIRA_MIN); a transect below the minimum drops out
         and is listed.
  (ii-b) counterfactual registration to the other family -- off_t =
         median(GEDTM30 - z_solo_v2) in clearings (same code as arm (i),
         GLO-30 swapped for GEDTM30, juiz_lidar_v4.py:167-183).
  (ii-c) perturbation by the CI -- 1000 independent per-transect bootstrap
         draws of the median of (GLO-30 - z_solo_v2) in clearings (same
         estimator as the judge's l.176), a declared seed (one RNG stream
         per transect, via REF.rng_do_teste, so it does not depend on
         order).
  (iii)  no offset -- off_t = 0 (a stress test, not a candidate).

FAIL-LOUD GUARDS (abort with RuntimeError if violated):
  (i) reproduces auditar_nmad_pareado_nativa_anadem.json.
      populacao_comum_5_produtos_ETH.{tabela_pool_ETH,testes_principal_ETH}
      to 1e-6 (the SAME guard REFA already runs against nativa_diretas, now
      also against its own ANADEM file).
  recomputed n_clareira == juiz_grade_nativa_diretas.offsets_por_transecto[t].n_clareira
      for the 13 admissible transects (confirms that the clearing
      population used here matches the judge's).
  invariance (i): NMAD per transect-class (ETH stratification) is IDENTICAL
      across ALL arms, cell by cell of (transect, class, product), to 1e-6
      -- off_t is a per-transect additive constant, and NMAD is
      shift-invariant.
  invariance (ii): the pool bias difference between any two products
      (pred_p - pred_q does not contain z_ref) is IDENTICAL across arms, to
      1e-9.
  A (vies_rotulo_m): the ANALYTICAL account (A(i) - weighted sum of
      Delta-off_t per transect) matches the RECOMPUTED account
      (mean(solo_anc - z_ref)) to 1e-6, per arm and per class.
  ANADEM sha256 checked BEFORE any computation
      (REFA.conferir_sha256_anadem, hardcoded).

Does not interpret the result for the paper (no claim prose), does not
audit itself: only the hash-stamped number.

USAGE
  /trabalho/ambientes/s33_amb_virtual/.venv/bin/python auditar_s2_registro_referencia.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import proveniencia as PROV                        # noqa: E402
import juiz_lidar_v4 as J                           # noqa: E402
import auditar_nmad_pareado as REF                  # noqa: E402  (READ, not edited)
import auditar_nmad_pareado_nativa_anadem as REFA   # noqa: E402  (READ, not edited)
import rotulos_lidar_v23 as ROT                     # noqa: E402  (only preparar_gedi/atl08 + celula)
from geoide_v23 import undulacao                    # noqa: E402

PRODUTOS_TODOS = list(REFA.PRODUTOS_TODOS)           # ["v23","mlp","fabdem","gedtm30","glo30","anadem"]
PARES15 = [(PRODUTOS_TODOS[i], PRODUTOS_TODOS[j])
           for i in range(len(PRODUTOS_TODOS)) for j in range(i + 1, len(PRODUTOS_TODOS))]
CLASSES_CRITERIO = ["20-30m", ">30m"]
FAM_GLO30 = ["glo30", "fabdem", "anadem"]           # anadem and fabdem are derived from GLO-30
FAIXAS, ROT_F = J.FAIXAS, J.ROT_F
DOSSEL_FLORESTA, AGUA_CLAREIRA = J.DOSSEL_FLORESTA, 10.0
N_CLAREIRA_MIN = 10                                 # same minimum as the judge's (J.N_CLAREIRA_MIN)
MIN_TRANSECTOS_IIA = 6
import os as _os
B_IIC = int(_os.environ.get("S2_B_IIC", "1000"))    # override for smoke testing only (never in the final run)
SEED = 42


def log(m=""):
    print(m, flush=True)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


class _PainelMudo:
    def log(self, m=""):
        log("    [rotulos_lidar_v23] " + str(m))


# --------------------------------------------------------------- clareira
def montar_clareira(parquet: Path, julgaveis: list[str]) -> pd.DataFrame:
    """Same clearing population as the judge's (J.montar + quality guards +
    dossel_v2<DOSSEL_FLORESTA & agua<10), restricted to admissible transects."""
    J.SOLO = parquet
    d = J.montar()
    d = d[(d["n_all"] >= J.N_ALL_MIN)
          & (d["n_casca"] / d["n_all"] >= J.FRAC_CASCA_MIN)
          & np.isfinite(d["z_solo_v2"])]
    cl = d[(d["dossel_v2"] < DOSSEL_FLORESTA) & (d["agua"] < AGUA_CLAREIRA)
           & d["transecto"].isin(julgaveis)].copy()
    return cl


# --------------------------------------------------------- arm (ii-a)
def montar_ancoras_independentes(bioma: str = "amazonia") -> pd.DataFrame:
    """Raw GEDI/ATL08 shots approved ONLY by the hard quality criterion
    (preparar_gedi/preparar_atl08), WITHOUT the physical-plausibility cut
    that uses the GLO-30 DSM (that cut only happens in ROT.montar_bioma,
    NOT called here). Gridded on the same 7200x7200 native grid and
    converted to orthometric height by the same geoid undulation."""
    ROT.SAIDA = Path("/trabalho/GNN_TOPO/SATELITES/alvos")
    painel = _PainelMudo()
    partes = []
    for prep, fonte in ((ROT.preparar_gedi, 1), (ROT.preparar_atl08, 2)):
        r = prep(bioma, painel)
        if r is None:
            continue
        r = r[r["bom"].to_numpy().astype(bool)].copy()
        r["fonte_dura"] = fonte
        partes.append(r[["lat", "lon", "solo", "fonte_dura"]])
    d = pd.concat(partes, ignore_index=True)
    N = undulacao(d["lon"].to_numpy(), d["lat"].to_numpy())
    d["solo_orto"] = d["solo"].to_numpy() - N
    bb = ROT.BIOMAS[bioma]
    r_g, c_g = ROT.celula(d["lat"].to_numpy(), d["lon"].to_numpy(), bb)
    d["r"], d["c"] = r_g, c_g
    d = d[d["r"] >= 0].reset_index(drop=True)
    quad_col = np.full(len(d), "", dtype=object)
    idx_col = np.full(len(d), -1, dtype=np.int64)
    for q, (qr, qc) in ROT.QUADS.items():
        m = ((d["r"] >= qr * ROT.QUAD) & (d["r"] < (qr + 1) * ROT.QUAD)
             & (d["c"] >= qc * ROT.QUAD) & (d["c"] < (qc + 1) * ROT.QUAD)).to_numpy()
        quad_col[m] = q
        idx_col[m] = ((d.loc[m, "r"].to_numpy() - qr * ROT.QUAD) * ROT.QUAD
                      + (d.loc[m, "c"].to_numpy() - qc * ROT.QUAD))
    d["quad"], d["idx_no"] = quad_col, idx_col
    d = d[d["quad"] != ""].reset_index(drop=True)
    return d[["quad", "idx_no", "fonte_dura", "solo_orto"]]


def braco_iia(cl: pd.DataFrame, julgaveis: list[str], ancoras: pd.DataFrame) -> dict:
    """off_t per transect under independent orbital registration; availability
    and the exclusion reason for each transect below the minimum."""
    j = cl.merge(ancoras, on=["quad", "idx_no"], how="inner")
    out = {"off_t": {}, "n_disparos_por_transecto": {}, "avaliavel_por_transecto": {},
           "excluidos": {}}
    for t in julgaveis:
        s = j[j["transecto"] == t]
        n = int(len(s))
        out["n_disparos_por_transecto"][t] = n
        if n >= N_CLAREIRA_MIN:
            out["off_t"][t] = float(np.median((s["solo_orto"] - s["z_solo_v2"]).to_numpy()))
            out["avaliavel_por_transecto"][t] = True
        else:
            out["avaliavel_por_transecto"][t] = False
            out["excluidos"][t] = f"{n} disparos independentes em clareira < minimo {N_CLAREIRA_MIN}"
    n_avaliaveis = sum(out["avaliavel_por_transecto"].values())
    out["n_transectos_avaliaveis"] = n_avaliaveis
    out["avaliavel"] = bool(n_avaliaveis >= MIN_TRANSECTOS_IIA)
    out["motivo"] = (None if out["avaliavel"] else
                     f"{n_avaliaveis} transectos com >= {N_CLAREIRA_MIN} disparos independentes "
                     f"em clareira, abaixo do minimo de {MIN_TRANSECTOS_IIA} exigido pelo desenho")
    return out


# --------------------------------------------------------------- z_ref per arm
def z_ref_do_braco(flor: pd.DataFrame, off_t: dict[str, float]) -> np.ndarray:
    off = flor["transecto"].map(off_t).to_numpy(dtype=np.float64)
    return flor["z_solo_v2"].to_numpy(dtype=np.float64) + off


def recomputar(flor: pd.DataFrame, off_t: dict[str, float]) -> tuple[pd.DataFrame, dict, dict]:
    """flor with the arm's z_ref; tabela_pool (ETH) and por_transecto_faixa
    (ETH), reusing REFA.tabela_pool_generic/por_transecto_faixa_generic."""
    fl = flor.copy()
    fl["z_ref"] = z_ref_do_braco(fl, off_t)
    pool = REFA.tabela_pool_generic(fl, "faixa_ex", PRODUTOS_TODOS)
    mat = REFA.por_transecto_faixa_generic(fl, "faixa_ex", PRODUTOS_TODOS)
    return fl, pool, mat


# --------------------------------------------------------------- guardas
def _prox(a, b, tol=1e-6) -> bool:
    if a is None or b is None:
        return a is None and b is None
    fa, fb = float(a), float(b)
    if np.isnan(fa) or np.isnan(fb):
        return np.isnan(fa) and np.isnan(fb)
    return abs(fa - fb) <= tol


def guarda_invariancia_nmad(mat_i: dict, mat_arm: dict, nome_arm: str) -> int:
    n = 0
    for rot in ROT_F:
        chaves = set(mat_i[rot]) | set(mat_arm[rot])
        for t in chaves:
            ri, ra = mat_i[rot].get(t, {}), mat_arm[rot].get(t, {})
            if ri.get("n_cel") != ra.get("n_cel"):
                raise RuntimeError(f"guarda invariancia NMAD [{nome_arm}]: {rot}/{t}/n_cel "
                                   f"{ra.get('n_cel')} != braco (i) {ri.get('n_cel')}")
            for p in PRODUTOS_TODOS:
                if p not in ri or p not in ra:
                    continue
                if not _prox(ri[p]["nmad"], ra[p]["nmad"], 1e-6):
                    raise RuntimeError(f"guarda invariancia NMAD [{nome_arm}]: {rot}/{t}/{p} "
                                       f"recomputado {ra[p]['nmad']} != braco (i) {ri[p]['nmad']} "
                                       f"(deveria ser EXATAMENTE invariante a off_t por transecto)")
                n += 1
    return n


def guarda_invariancia_vies(pool_i: dict, pool_arm: dict, nome_arm: str) -> int:
    n = 0
    for a_, b_ in PARES15:
        for rot in ROT_F + ["TODAS"]:
            vi_a, vi_b = pool_i[a_][rot].get("vies"), pool_i[b_][rot].get("vies")
            va_a, va_b = pool_arm[a_][rot].get("vies"), pool_arm[b_][rot].get("vies")
            if vi_a is None or vi_b is None:
                continue
            d_i, d_a = vi_a - vi_b, va_a - va_b
            if not _prox(d_i, d_a, 1e-9):
                raise RuntimeError(f"guarda invariancia vies [{nome_arm}]: {a_} vs {b_}/{rot} "
                                   f"diferenca recomputada {d_a} != braco (i) {d_i} "
                                   f"(deveria ser EXATAMENTE invariante a qualquer offset)")
            n += 1
    return n


# --------------------------------------------------------------- termo A
def termo_a(flor_i: pd.DataFrame, off_i: dict, off_arm: dict, nome_arm: str) -> dict:
    """A = solo_anc - z_solo_v2 - off_t, per ETH class, only on cells with an
    approved anchor (solo_anc not null). ANALYTICAL account (from A(i) and
    the per-transect shift) checked against the RECOMPUTED account --
    fail-loud guard."""
    anc = flor_i[flor_i["solo_anc"].notna()].copy()
    out = {}
    for rot in ROT_F + ["TODAS"]:
        s = anc if rot == "TODAS" else anc[anc["faixa_ex"] == rot]
        if len(s) < 5:
            continue
        z_ref_i = s["z_solo_v2"] + s["transecto"].map(off_i)
        a_i = float((s["solo_anc"] - z_ref_i).mean())
        z_ref_arm = s["z_solo_v2"] + s["transecto"].map(
            lambda t: off_arm.get(t, off_i[t]))
        a_recomp = float((s["solo_anc"] - z_ref_arm).mean())
        pesos = s["transecto"].value_counts(normalize=True)
        delta_analitico = -sum(pesos.get(t, 0.0) * (off_arm.get(t, off_i[t]) - off_i[t])
                               for t in pesos.index)
        a_analitico = a_i + delta_analitico
        bate = _prox(a_analitico, a_recomp, 1e-6)
        if not bate:
            raise RuntimeError(f"guarda termo A [{nome_arm}]: {rot} analitico {a_analitico} "
                               f"!= recomputado {a_recomp} (tol 1e-6)")
        out[rot] = {"n": int(len(s)), "A_braco_i_m": a_i, "A_analitico_m": a_analitico,
                    "A_recomputado_m": a_recomp, "guarda_bate": bate}
    return out


# --------------------------------------------------------------- ordering/verdict
def ordenacoes(pool: dict) -> dict:
    out = {}
    for rot in CLASSES_CRITERIO:
        out[rot] = {}
        for a_, b_ in PARES15:
            ma, mb = pool[a_][rot].get("mae"), pool[b_][rot].get("mae")
            na, nb = pool[a_][rot].get("nmad"), pool[b_][rot].get("nmad")
            out[rot][f"{a_}_vs_{b_}"] = {
                "sinal_mae": None if ma is None or mb is None else float(np.sign(ma - mb)),
                "sinal_nmad": None if na is None or nb is None else float(np.sign(na - nb)),
                "dif_mae_m": None if ma is None or mb is None else float(ma - mb),
            }
    return out


def veredito_par(par: str, rot: str, ord_i: dict, tau: dict, resultados_bracos: dict,
                 fracao_iic_mantida: dict) -> dict:
    inverte = {}
    for arm in ("ii-a", "ii-b", "iii"):
        ord_arm = resultados_bracos.get(arm)
        if ord_arm is None:
            inverte[arm] = None
            continue
        si, sa = ord_i[rot][par]["sinal_mae"], ord_arm[rot][par]["sinal_mae"]
        ni, na = ord_i[rot][par]["sinal_nmad"], ord_arm[rot][par]["sinal_nmad"]
        inv_mae = (si is not None and sa is not None and si != 0 and sa != 0 and si != sa)
        inv_nmad = (ni is not None and na is not None and ni != 0 and na != 0 and ni != na)
        inverte[arm] = bool(inv_mae or inv_nmad)
    frac = fracao_iic_mantida[rot][par]
    dif_mae_i = abs(ord_i[rot][par]["dif_mae_m"] or 0.0)
    disponiveis = [a for a in ("ii-a", "ii-b", "iii") if inverte.get(a) is not None]
    # "depends" rule (pre-declared criterion): a LARGE inversion
    # (|dMAE(i)| >= that arm's tau) in ii-a or ii-b; OR the ii-c draws
    # invert more than 5% of the time.
    algum_inverte_grande = any(inverte.get(a) and dif_mae_i >= tau.get(a, float("inf"))
                               for a in ("ii-a", "ii-b"))
    algum_inverte_pequeno = any(inverte.get(a) for a in ("ii-a", "ii-b", "iii"))
    if algum_inverte_grande or (np.isfinite(frac) and frac < 0.95):
        veredito = "depende"
    elif not algum_inverte_pequeno and (np.isfinite(frac) and frac >= 0.95):
        veredito = "nao depende"
    else:
        # an inversion exists (in ii-a/ii-b/iii) but none is "large", or the
        # ii-c fraction is not evaluable (pair with a null sign in (i)):
        # stays in the cautious "with caveat" bucket.
        veredito = "depende com ressalva"
    return {"veredito": veredito, "inverte_por_braco": inverte,
           "fracao_ii_c_mantida": frac, "dif_mae_i_m": ord_i[rot][par]["dif_mae_m"],
           "tau_ii_b_m": tau.get("ii-b"), "disponiveis": disponiveis}


# --------------------------------------------------------------- main
def main() -> int:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="/trabalho/GNN_TOPO/SATELITES/lidar_eba/solo_lidar_30m_v3_nativa.parquet")
    ap.add_argument("--juiz", default=str(RAIZ / "juiz_grade_nativa_diretas.json"))
    ap.add_argument("--laterais", default="/trabalho/GNN_TOPO/SATELITES/laterais_nativa_diretas")
    ap.add_argument("--referencia", default=str(RAIZ / "auditar_nmad_pareado_nativa_anadem.json"))
    ap.add_argument("--recorte-dir", default="/trabalho/GNN_TOPO/SATELITES/anadem")
    ap.add_argument("--saida", default="auditar_s2_registro_referencia.json")
    a = ap.parse_args()
    parquet, juiz_p, laterais = Path(a.parquet), Path(a.juiz), Path(a.laterais)
    ref_p, recorte_dir, saida = Path(a.referencia), Path(a.recorte_dir), RAIZ / a.saida
    J.LAT = laterais

    log("== S2: dependencia do registro a referencia (GLO-30) ==")

    # ---- pre-flight: sha256 BEFORE any computation
    REFA.conferir_sha256_anadem(laterais, recorte_dir)
    fontes_criticas = [parquet, juiz_p, ref_p, RAIZ / "juiz_lidar_v4.py",
                       RAIZ / "auditar_nmad_pareado.py",
                       RAIZ / "auditar_nmad_pareado_nativa_anadem.py",
                       RAIZ / "rotulos_lidar_v23.py", RAIZ / "geoide_v23.py"]
    sha_antes = {str(p): (_sha256(p) if p.exists() else "AUSENTE") for p in fontes_criticas}
    log(f"  sha256 conferido ANTES de calcular, {len(sha_antes)} fontes criticas")

    juiz = json.loads(juiz_p.read_text(encoding="utf-8"))
    ref = json.loads(ref_p.read_text(encoding="utf-8"))
    offs = juiz["offsets_por_transecto"]
    julgaveis = sorted([t for t, v in offs.items() if v["julgavel"]])
    off_i = {t: float(offs[t]["offset_m"]) for t in julgaveis}
    log(f"  julgaveis ({len(julgaveis)}): {julgaveis}")

    # ---- fixed population (6 products, cells with valid ANADEM) -- REUSED from REFA
    flor_full, info = REFA.montar_populacao_com_anadem(parquet, juiz, laterais)
    conf_regua = REF.guardas_populacao(flor_full, juiz)
    log(f"  populacao reproduz tabela_regua_k0 do juiz (n exato, nmad 1e-6) em {len(conf_regua)} celulas")
    flor = flor_full[flor_full["anadem"].notna()].copy()
    log(f"  populacao FIXA do S2 (celulas com ANADEM valido): {len(flor):,} "
        f"({len(flor_full) - len(flor):,} perdidas por ANADEM ausente)")

    # ---- guard (i): reproduces auditar_nmad_pareado_nativa_anadem.json
    pool_i = REFA.tabela_pool_generic(flor, "faixa_ex", PRODUTOS_TODOS)
    mat_i = REFA.por_transecto_faixa_generic(flor, "faixa_ex", PRODUTOS_TODOS)
    principal_i, _ = REFA.rodar_par(mat_i, mat_i, REFA.PARES_TODOS)
    n_pool_conf, n_teste_conf = 0, 0
    for p in PRODUTOS_TODOS:
        for rot in ROT_F + ["TODAS"]:
            b = ref["populacao_comum_5_produtos_ETH"]["tabela_pool_ETH"][p][rot]
            a_ = pool_i[p][rot]
            for k in ("n", "mae", "vies", "nmad", "rmse"):
                if k not in b:
                    continue
                if k == "n":
                    if a_.get(k) != b[k]:
                        raise RuntimeError(f"guarda (i): {p}/{rot}/n {a_.get(k)} != referencia {b[k]}")
                elif not _prox(a_.get(k), b[k]):
                    raise RuntimeError(f"guarda (i): {p}/{rot}/{k} {a_.get(k)} != referencia {b[k]}")
                n_pool_conf += 1
    for k_ref, meu in (("v23 vs fabdem", principal_i.get("v23 vs fabdem")),
                       ("v23 vs gedtm30", principal_i.get("v23 vs gedtm30")),
                       ("v23 vs mlp", principal_i.get("v23 vs mlp"))):
        if meu is None or k_ref not in ref["populacao_comum_5_produtos_ETH"]["testes_principal_ETH"]:
            continue
        b = ref["populacao_comum_5_produtos_ETH"]["testes_principal_ETH"][k_ref]["nmad"]
        for rot in ROT_F:
            if not b[rot].get("testado"):
                continue
            for campo in ("mediana_m", "p_wilcoxon"):
                if not _prox(meu["nmad"][rot].get(campo), b[rot].get(campo)):
                    raise RuntimeError(f"guarda (i) testes: {k_ref}/{rot}/{campo} nao reproduz referencia")
                n_teste_conf += 1
    log(f"  guarda (i): reproduz auditar_nmad_pareado_nativa_anadem.json a 1e-6 "
        f"({n_pool_conf} campos de tabela + {n_teste_conf} campos de teste)")

    # ---- clearing (same definition as the judge's) and n_clareira check
    cl = montar_clareira(parquet, julgaveis)
    n_cl_recomp = {t: int(len(cl[cl["transecto"] == t])) for t in julgaveis}
    for t in julgaveis:
        if n_cl_recomp[t] != offs[t]["n_clareira"]:
            raise RuntimeError(f"guarda n_clareira: {t} recomputado {n_cl_recomp[t]} "
                               f"!= juiz {offs[t]['n_clareira']}")
    log(f"  guarda n_clareira: {len(julgaveis)} transectos batem com o juiz")

    # ---- arm (ii-b): registration to GEDTM30
    off_iib = {t: float(np.median((cl.loc[cl.transecto == t, "gedtm30"]
                                   - cl.loc[cl.transecto == t, "z_solo_v2"]).to_numpy()))
              for t in julgaveis}

    # ---- arm (iii): no offset
    off_iii = {t: 0.0 for t in julgaveis}

    # ---- arm (ii-a): independent orbital registration
    ancoras = montar_ancoras_independentes("amazonia")
    r_iia = braco_iia(cl, julgaveis, ancoras)
    log(f"  braco (ii-a): {r_iia['n_transectos_avaliaveis']}/{len(julgaveis)} transectos com "
        f">= {N_CLAREIRA_MIN} disparos independentes em clareira; avaliavel={r_iia['avaliavel']}")

    # ---- arm (ii-c): perturbation by the CI, 1000 draws/transect
    off_iic_draws = {}
    for t in julgaveis:
        res = (cl.loc[cl.transecto == t, "glo30"] - cl.loc[cl.transecto == t, "z_solo_v2"]).to_numpy()
        rng = REF.rng_do_teste(t, "off_t", "mediana", "bootstrap_iic")
        off_iic_draws[t] = np.array([float(np.median(rng.choice(res, len(res), True)))
                                     for _ in range(B_IIC)])

    # ---- recompute tabela_pool/mat per arm (FIXED population, only off_t changes)
    _, pool_iib, mat_iib = recomputar(flor, off_iib)
    _, pool_iii, mat_iii = recomputar(flor, off_iii)
    resultado_iia = None
    pool_iia, mat_iia = None, None
    if r_iia["avaliavel"]:
        off_iia_completo = dict(off_i)
        off_iia_completo.update(r_iia["off_t"])
        _, pool_iia, mat_iia = recomputar(flor, off_iia_completo)

    # ---- invariance guards (fail-loud)
    n_inv_nmad = 0
    n_inv_nmad += guarda_invariancia_nmad(mat_i, mat_iib, "ii-b")
    n_inv_nmad += guarda_invariancia_nmad(mat_i, mat_iii, "iii")
    if mat_iia is not None:
        n_inv_nmad += guarda_invariancia_nmad(mat_i, mat_iia, "ii-a")
    log(f"  guarda invariancia NMAD por transecto-faixa: {n_inv_nmad} celulas conferidas, 0 violacoes")

    n_inv_vies = 0
    n_inv_vies += guarda_invariancia_vies(pool_i, pool_iib, "ii-b")
    n_inv_vies += guarda_invariancia_vies(pool_i, pool_iii, "iii")
    if pool_iia is not None:
        n_inv_vies += guarda_invariancia_vies(pool_i, pool_iia, "ii-a")
    log(f"  guarda invariancia diferenca de vies (pool): {n_inv_vies} pares conferidos, 0 violacoes")

    # ---- term A
    a_iib = termo_a(flor, off_i, off_iib, "ii-b")
    a_iii = termo_a(flor, off_i, off_iii, "iii")
    a_iia = termo_a(flor, off_i, {**off_i, **r_iia["off_t"]}, "ii-a") if r_iia["avaliavel"] else None
    log("  guarda termo A: analitico == recomputado (1e-6) em ii-b, iii"
        + (", ii-a" if a_iia is not None else ""))

    # ---- rankings and the pre-declared criterion
    ord_i = ordenacoes(pool_i)
    ord_iib = ordenacoes(pool_iib)
    ord_iii = ordenacoes(pool_iii)
    ord_iia = ordenacoes(pool_iia) if pool_iia is not None else None

    tau = {"ii-b": {rot: float(np.median([abs(off_iib[t] - off_i[t]) for t in julgaveis]))
                   for rot in CLASSES_CRITERIO}}
    tau["ii-b"] = float(np.median([abs(off_iib[t] - off_i[t]) for t in julgaveis]))
    tau["iii"] = float(np.median([abs(off_iii[t] - off_i[t]) for t in julgaveis]))
    tau["ii-a"] = (float(np.median([abs(r_iia["off_t"].get(t, off_i[t]) - off_i[t]) for t in julgaveis]))
                  if r_iia["avaliavel"] else float("nan"))

    log(f"  braco (ii-c): {B_IIC} sorteios/transecto — recomputando MAE pool por classe/produto")
    mae_draws = mae_pool_todos_os_sorteios(flor, off_iic_draws, julgaveis)  # [rot][produto] -> array(B_IIC)
    fracao_mantida = {}
    for rot in CLASSES_CRITERIO:
        fracao_mantida[rot] = {}
        for a_, b_ in PARES15:
            par = f"{a_}_vs_{b_}"
            si = ord_i[rot][par]["sinal_mae"]
            ma_k, mb_k = mae_draws[rot].get(a_), mae_draws[rot].get(b_)
            if si is None or si == 0 or ma_k is None or mb_k is None:
                fracao_mantida[rot][par] = float("nan")
                continue
            valido = np.isfinite(ma_k) & np.isfinite(mb_k)
            sinais = np.sign(ma_k[valido] - mb_k[valido])
            fracao_mantida[rot][par] = (float((sinais == si).mean())
                                        if valido.sum() else float("nan"))

    resultados_bracos = {"ii-a": ord_iia, "ii-b": ord_iib, "iii": ord_iii}
    veredito_pares = {}
    for rot in CLASSES_CRITERIO:
        veredito_pares[rot] = {}
        for a_, b_ in PARES15:
            par = f"{a_}_vs_{b_}"
            veredito_pares[rot][par] = veredito_par(par, rot, ord_i, tau, resultados_bracos,
                                                    fracao_mantida)

    # ---- decisive test: GLO-30 family (FABDEM, ANADEM) vs GEDTM30 above 30m under (ii-b)
    decisivo = {}
    for p in ("fabdem", "anadem"):
        par = f"{p}_vs_gedtm30" if (p, "gedtm30") in PARES15 else f"gedtm30_vs_{p}"
        invertido_sinal = None
        mi = pool_i[p][">30m"].get("mae"); mg = pool_i["gedtm30"][">30m"].get("mae")
        ma = pool_iib[p][">30m"].get("mae"); mgb = pool_iib["gedtm30"][">30m"].get("mae")
        if None not in (mi, mg, ma, mgb):
            sinal_i = np.sign(mi - mg)   # negative = p wins (lower MAE) against GEDTM30
            sinal_b = np.sign(ma - mgb)
            invertido_sinal = bool(sinal_i != sinal_b and sinal_i != 0 and sinal_b != 0)
        decisivo[p] = {
            "mae_i": mi, "mae_gedtm30_i": mg, "mae_ii_b": ma, "mae_gedtm30_ii_b": mgb,
            "vantagem_i_m": (None if mi is None or mg is None else mi - mg),
            "vantagem_ii_b_m": (None if ma is None or mgb is None else ma - mgb),
            "troca_de_sinal_sob_ii_b": invertido_sinal,
            "veredito": veredito_pares[">30m"].get(f"{p}_vs_gedtm30",
                                                    veredito_pares[">30m"].get(f"gedtm30_vs_{p}")),
        }

    # ---- effect on A per class (summary)
    efeito_a = {"ii-b": a_iib, "iii": a_iii, "ii-a": a_iia}

    # ---- build the output
    criterio_declarado = {
        "unidade": "transecto julgavel fixo (13 de juiz_grade_nativa_diretas.json)",
        "populacao_fixa": "celulas floresta-julgavel com ANADEM valido (mesma de "
                          "populacao_comum_5_produtos_ETH de auditar_nmad_pareado_nativa_anadem.json)",
        "pares": [f"{a_} vs {b_}" for a_, b_ in PARES15], "n_pares": len(PARES15),
        "classes_do_criterio": CLASSES_CRITERIO,
        "tau": "mediana de |off_alternativo - off_atual| entre os 13 transectos, por braco",
        "regra_nao_depende": "ordenacao por MAE e por NMAD identica a (i) em (ii-a se avaliavel), "
                             "(ii-b) e (iii); em (ii-c) mantida em >=95% dos sorteios",
        "regra_depende_com_ressalva": "mantida em ii-a/ii-b e >=95% de ii-c mas muda em iii; OU so um "
                                      "par se inverte com |dMAE(i)| < tau",
        "regra_depende": "em ii-a ou ii-b inverte par com |dMAE(i)| >= tau; OU par se inverte em "
                         ">5% dos sorteios de ii-c; OU sob ii-b troca de sinal a vantagem de MAE "
                         "do FABDEM/ANADEM sobre o GEDTM30 em >30m",
        "teste_decisivo": "familia GLO-30 (FABDEM, ANADEM) vs GEDTM30 em >30m sob (ii-b)",
        "previsao_registrada_antes_de_rodar": "em 20-30m, ANADEM/GEDTM30/FABDEM dentro de 0,17 m de "
                                              "MAE — os tres pares desse trio devem cair em 'com "
                                              "ressalva' em quase qualquer braco",
    }

    fontes = list(fontes_criticas) + [
        laterais / f"{n}_amazonia_{q}.npz" for q in J.QUADS
        for n in ("glo30", "fabdem", "gedtm30", "anadem", "agua_jrc", "dossel_eth", "rotulos")
    ] + [laterais / "_proveniencia_anadem.json", laterais / "_proveniencia.json",
        recorte_dir / "anadem_v1_21M_recorte_amazonia_2x2.tif",
        recorte_dir / "anadem_v1_21M_recorte_amazonia_2x2.proveniencia.json",
        Path("/trabalho/GNN_TOPO/SATELITES/alvos/gedi02_a_amazonia_dedup.parquet"),
        Path("/trabalho/GNN_TOPO/SATELITES/alvos/gedi02_a_amazonia.parquet"),
        Path("/trabalho/GNN_TOPO/SATELITES/alvos/atl08_amazonia.parquet"),
        Path(__file__)]
    fontes += [J.SUP / f"delta_amazonia_ampliacao_b4s035{arm}_seed{sd}.npz"
              for arm in ("", "_mlp_semobs") for sd in J.SEEDS_PAR]

    corpo = {
        "sha256_conferido_antes_de_calcular": sha_antes,
        "criterio_pre_registrado": criterio_declarado,
        "entradas": {"parquet": str(parquet), "juiz": str(juiz_p), "laterais": str(laterais),
                    "referencia_guarda_i": str(ref_p)},
        "julgaveis": julgaveis,
        "guarda_i_reproduz_referencia_1e-6": {"campos_tabela": n_pool_conf, "campos_teste": n_teste_conf},
        "guarda_n_clareira": {t: n_cl_recomp[t] for t in julgaveis},
        "guarda_invariancia_nmad_transecto_faixa": {"celulas_conferidas": n_inv_nmad, "violacoes": 0},
        "guarda_invariancia_diferenca_vies_pool": {"pares_conferidos": n_inv_vies, "violacoes": 0},
        "off_t_por_braco": {"i_atual": off_i, "ii_a_independente": r_iia["off_t"],
                            "ii_b_gedtm30": off_iib, "iii_sem_offset": off_iii},
        "braco_ii_a_disponibilidade": {k: v for k, v in r_iia.items() if k != "off_t"},
        "tau_m": tau,
        "tabela_pool_ETH": {"i_atual": pool_i, "ii_b_gedtm30": pool_iib, "iii_sem_offset": pool_iii,
                            "ii_a_independente": pool_iia},
        "ordenacoes_por_braco": {"i_atual": ord_i, "ii_a_independente": ord_iia,
                                 "ii_b_gedtm30": ord_iib, "iii_sem_offset": ord_iii},
        "fracao_ii_c_mantida_ordenacao": fracao_mantida,
        "veredito_por_par_e_classe": veredito_pares,
        "teste_decisivo_familia_glo30_vs_gedtm30_30m": decisivo,
        "efeito_sobre_A_por_classe": efeito_a,
        "resumo_vereditos": {
            rot: {v: sum(1 for par in veredito_pares[rot] if veredito_pares[rot][par]["veredito"] == v)
                 for v in ("nao depende", "depende com ressalva", "depende")}
            for rot in CLASSES_CRITERIO},
        "tempo_total_s": round(time.time() - t0, 1),
    }
    PROV.gravar(saida, corpo, fontes_lidas=fontes, script=__file__)
    log(f"  -> {saida.name} ({time.time()-t0:.1f} s)")
    return 0


def mae_pool_todos_os_sorteios(flor: pd.DataFrame, off_iic_draws: dict,
                               julgaveis: list[str]) -> dict:
    """Pool MAE per class/product for the B_IIC draws of arm (ii-c),
    vectorized: an (transect x draw) matrix of off_t, expanded per row
    once; z_ref is only recomputed per draw (not per pair), reused across
    the 6 products."""
    trans_idx, cats = pd.factorize(flor["transecto"].to_numpy())
    off_mat = np.stack([off_iic_draws[t] for t in cats])         # (n_transectos, B)
    z_solo = flor["z_solo_v2"].to_numpy(dtype=np.float64)
    faixa = flor["faixa_ex"].to_numpy()
    out = {rot: {p: np.full(len(off_iic_draws[cats[0]]), np.nan) for p in PRODUTOS_TODOS}
          for rot in CLASSES_CRITERIO}
    preds = {p: flor[p].to_numpy(dtype=np.float64) for p in PRODUTOS_TODOS}
    masks = {rot: (faixa == rot) for rot in CLASSES_CRITERIO}
    B = off_mat.shape[1]
    for k in range(B):
        off_col = off_mat[trans_idx, k]
        z_ref = z_solo + off_col
        for rot in CLASSES_CRITERIO:
            m = masks[rot]
            for p in PRODUTOS_TODOS:
                e = preds[p][m] - z_ref[m]
                e = e[np.isfinite(e)]
                out[rot][p][k] = float(np.abs(e).mean()) if len(e) else np.nan
    return out


if __name__ == "__main__":
    raise SystemExit(main())
