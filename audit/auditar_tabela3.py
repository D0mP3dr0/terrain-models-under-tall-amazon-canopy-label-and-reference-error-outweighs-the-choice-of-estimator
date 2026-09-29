"""E4 -- Table III auditor (label term by ETH class), with a bootstrap CI
per transect and a pre-declared cell floor.

WHY. Table III of the manuscript (`main.tex` tab:label, l. 829-843) reports
only the cell-weighted mean of the label term, with no CI, no transect
count, and is sensitive to a single 1-cell transect (NP_T-0639, +26.45 m in
the >30 m class). This script reconstructs
`juiz_grade_nativa_diretas.json[decomposicao_por_dossel_exogeno_eth]` from
the SAME sources (guard: it reproduces `vies_rotulo_m` and `n` per class to
1e-6/exact -- it STOPS if they do not match) and adds what is missing:
transect count, a per-transect cell floor (inherited from
`auditar_nmad_pareado.py:63`), median, mean of per-transect means, and a
bootstrap IC95 of 10,000 per-transect resamples, WITH and WITHOUT the floor,
to record the sensitivity to that single transect.

PRE-REGISTERED CRITERION (constants below):
  PISO_CEL = 200      minimum cells per transect-strip to enter the
                       "with floor" estimand
  MIN_TRANSECTOS = 6  transects above the floor; a class with fewer becomes
                       "indeterminate" (no CI on the floored estimands)
  B = 10_000          bootstrap resamples, cluster = transect
  SEED = 42
  ALPHA = 0.05         two-sided IC95

DESIGN:
  H: the label term above 20 m of canopy is positive, with a transect-
     clustered CI excluding zero.
  Per ETH class: transect count, cell count, cell-weighted mean, median,
  mean of per-transect means (transects with >= PISO_CEL cells in the class
  only), and a bootstrap IC95 of 10,000 PER-TRANSECT resamples for each
  estimand.
  Adopted reading where the design is ambiguous: the floor restricts the
  input CELLS (only cells from transects with >= PISO_CEL cells in the
  class); the "two floored estimands" that decide acceptance are (a) the
  cell-weighted mean WITH the floor and (b) the mean of per-transect means
  WITH the floor, both with a per-transect bootstrap IC95. The median is
  reported with and without the floor as a third, informative estimand, but
  it does NOT enter the formal acceptance criterion. The unfloored version
  of each estimand is also recorded in full (sensitivity to NP_T-0639).
  Acceptance: a class's signal is supported if the CI (with floor) excludes
  zero in both estimands (a) and (b); otherwise the class is reported as
  "indeterminate".
  Data: `juiz_grade_nativa_diretas.json` and the sources that produced it
  (the same population that `auditar_nmad_pareado.py` reconstructs, via the
  same path: `J.montar()` + guards + the judge's admissible offsets +
  dedup).
  Output: hash-stamped JSON + parquet keyed by (quad, idx_no, transect,
  eth_class, label_term).

USAGE
  python auditar_tabela3.py
  python auditar_tabela3.py --parquet <solo_v3_nativa.parquet> \
         --juiz juiz_grade_nativa_diretas.json --laterais <laterais_nativa_diretas> \
         --saida auditar_tabela3.json --dump auditar_tabela3_dump.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import proveniencia as PROV          # noqa: E402
import juiz_lidar_v4 as J             # noqa: E402

FAIXAS, ROT_F = J.FAIXAS, J.ROT_F
# Pre-declared constants, inherited from auditar_nmad_pareado.py:63
# (PISO_CEL, MIN_TRANSECTOS, B, SEED, ALPHA = 200, 6, 10_000, 42, 0.05).
PISO_CEL, MIN_TRANSECTOS, B, SEED, ALPHA = 200, 6, 10_000, 42, 0.05


def log(m: str = "") -> None:
    print(m, flush=True)


def montar_populacao(parquet: Path, juiz: dict) -> tuple[pd.DataFrame, dict]:
    """Reproduces the judge's population (guards, admissible transects,
    offsets, dedup, forest, anchors with exogenous ETH canopy). Same path
    as `auditar_nmad_pareado.py:montar_populacao`, up to the ruler-free
    decomposition that only exists in `juiz_lidar_v4.py` (l. 287-331)."""
    J.SOLO = parquet
    d = J.montar()
    n0 = len(d)
    d = d[(d["n_all"] >= J.N_ALL_MIN)
          & (d["n_casca"] / d["n_all"] >= J.FRAC_CASCA_MIN)
          & np.isfinite(d["z_solo_v2"])]
    offs = juiz["offsets_por_transecto"]              # KeyError se faltar
    trans_parquet = set(d["transecto"].unique())
    if trans_parquet != set(offs.keys()):
        raise RuntimeError(
            f"transectos do parquet {sorted(trans_parquet)} != chaves de "
            f"offsets {sorted(offs)}")
    julgaveis = [t for t, v in offs.items() if v["julgavel"]]
    d = d[d["transecto"].isin(julgaveis)].copy()
    d["off_t"] = d["transecto"].map({t: offs[t]["offset_m"] for t in julgaveis})
    d["z_ref"] = d["z_solo_v2"] + d["off_t"]
    n_cl = {t: offs[t]["n_clareira"] for t in julgaveis}
    if len(set(n_cl.values())) != len(n_cl):
        raise RuntimeError(
            f"empate de n_clareira entre julgaveis {n_cl} — o dedup deixa de "
            "ser determinado pela regra e passa a depender da ordem das linhas")
    d["_rank"] = d["transecto"].map(n_cl)
    d = (d.sort_values("_rank", ascending=False)
           .drop_duplicates(subset=["quad", "idx_no"], keep="first"))
    d["faixa"] = pd.cut(d["dossel_v2"], FAIXAS, labels=ROT_F)
    flor = d[(d["dossel_v2"] >= J.DOSSEL_FLORESTA) & (d["agua"] < J.AGUA_LIMPO)].copy()
    if "h_eth" not in flor.columns:
        raise RuntimeError("parquet/laterais sem h_eth — a estratificacao "
                           "ETH da Tabela III nao pode ser feita")

    # ── ruler-free decomposition (juiz_lidar_v4.py l. 287-291): only
    # on cells with a quality anchor (solo_anc not null).
    anc = flor[flor["solo_anc"].notna()].copy()
    anc["e_modelo"] = anc["v23"] - anc["z_ref"]
    anc["fidelidade"] = anc["v23"] - anc["solo_anc"]
    anc["vies_rotulo"] = anc["solo_anc"] - anc["z_ref"]     # label term

    # ── stratification by EXOGENOUS ETH canopy (juiz_lidar_v4.py l. 315-317):
    # ETH coverage >= 80% and classified by the SAME bins as dossel_v2.
    ax = anc[anc["h_eth"].notna() & (anc["cob_eth"] >= 0.8)].copy()
    ax["faixa_ex"] = pd.cut(ax["h_eth"], FAIXAS, labels=ROT_F)

    info = {"n_linhas_parquet": int(n0), "n_floresta_julgavel": int(len(flor)),
            "n_ancoras_dossel_exogeno": int(len(ax)),
            "julgaveis": julgaveis, "n_julgaveis": len(julgaveis)}
    return ax, info


def guarda_reproducao(ax: pd.DataFrame, juiz: dict) -> dict:
    """Guard (hard fail): reproduces decomposicao_por_dossel_exogeno_eth from
    the judge -- exact n and vies_rotulo_m to 1e-6 -- for EVERY class plus
    the TODAS pool. This is the check required before adding anything else:
    confirm the script reproduces the current +6.65 m value first."""
    alvo = juiz["decomposicao_por_dossel_exogeno_eth"]
    conf = {}
    for rot in ROT_F + ["TODAS"]:
        s = ax if rot == "TODAS" else ax[ax["faixa_ex"] == rot]
        n_r, vr_r = len(s), float(s["vies_rotulo"].mean()) if len(s) else float("nan")
        if rot not in alvo:
            # the judge only stores classes with n>=5 (juiz_lidar_v4.py l. 298);
            # classes below that (typically 0-3 m, with almost no anchor
            # under forest cover) have no target to check against --
            # recorded, not a guard failure.
            conf[rot] = {"n": n_r, "vies_rotulo_m_recomputado": vr_r,
                        "vies_rotulo_m_juiz": None, "diferenca_m": None,
                        "nota": "classe ausente do juiz (n<5 no artefato original)"}
            continue
        n_j, vr_j = alvo[rot]["n"], alvo[rot]["vies_rotulo_m"]
        if n_r != n_j:
            raise RuntimeError(f"guarda n: classe {rot} recomputado {n_r} != juiz {n_j}")
        if abs(vr_r - vr_j) > 1e-6:
            raise RuntimeError(
                f"guarda vies_rotulo_m: classe {rot} recomputado {vr_r!r} != "
                f"juiz {vr_j!r} (diferenca {vr_r - vr_j:.9f} m) — NAO reproduz "
                "o numero do manuscrito; PARAR aqui (regra do protocolo E4)")
        conf[rot] = {"n": n_r, "vies_rotulo_m_recomputado": vr_r,
                    "vies_rotulo_m_juiz": vr_j, "diferenca_m": vr_r - vr_j}
    return conf


def _rng_classe(rot: str, tag: str) -> np.random.Generator:
    """A separate bootstrap stream per (class, estimand) -- the CI does not
    depend on the order in which classes are processed (same practice as
    auditar_nmad_pareado.py:rng_do_teste)."""
    h = int.from_bytes(
        __import__("hashlib").sha256(f"{rot}|{tag}".encode()).digest()[:4], "big")
    return np.random.default_rng(np.random.SeedSequence([SEED, h]))


def _bootstrap_cluster(s: pd.DataFrame, transectos: list[str], rot: str, tag: str
                        ) -> dict:
    """Per-transect (cluster) bootstrap, B resamples: for each resample,
    transects are redrawn WITH replacement and the three estimands are
    recomputed over the cells of the drawn transects (cell-weighted pool
    mean; pool median; mean of per-drawn-transect means). B=10_000, a fixed
    seed per class+estimand."""
    n_t = len(transectos)
    if n_t == 0:
        return {"n_transectos": 0}
    grupos = {t: s.loc[s["transecto"] == t, "vies_rotulo"].to_numpy()
              for t in transectos}
    medias_t = {t: float(np.mean(v)) for t, v in grupos.items() if len(v)}
    rng = _rng_classe(rot, tag)
    idx = rng.integers(0, n_t, size=(B, n_t))
    ts_arr = np.array(transectos)
    bs_pond, bs_mediana, bs_transecto = np.empty(B), np.empty(B), np.empty(B)
    for i in range(B):
        escolhidos = ts_arr[idx[i]]
        pool = np.concatenate([grupos[t] for t in escolhidos])
        bs_pond[i] = pool.mean()
        bs_mediana[i] = np.median(pool)
        bs_transecto[i] = np.mean([medias_t[t] for t in escolhidos])
    def ic(v: np.ndarray) -> list[float]:
        return [float(np.percentile(v, 100 * ALPHA / 2)),
                float(np.percentile(v, 100 * (1 - ALPHA / 2)))]
    return {
        "n_transectos": n_t,
        "ic95_pond_celula": ic(bs_pond),
        "ic95_mediana": ic(bs_mediana),
        "ic95_media_por_transecto": ic(bs_transecto),
    }


def estatisticas_por_classe(ax: pd.DataFrame) -> dict:
    """For each ETH class (plus TODAS): cell count, transect count, the
    three estimands WITHOUT and WITH the floor (floor = only cells from
    transects with >= PISO_CEL cells IN THE CLASS), a per-transect
    bootstrap IC95 for each, and the acceptance verdict (two floored
    estimands excluding zero)."""
    out = {}
    for rot in ROT_F + ["TODAS"]:
        s = ax if rot == "TODAS" else ax[ax["faixa_ex"] == rot]
        n_cel = len(s)
        cel_por_t = s.groupby("transecto").size().to_dict()
        transectos_todos = sorted(cel_por_t)
        transectos_piso = sorted(t for t, n in cel_por_t.items() if n >= PISO_CEL)
        excluidos_piso = sorted(t for t, n in cel_por_t.items() if n < PISO_CEL)

        def ponto(sub: pd.DataFrame, ts: list[str]) -> dict:
            if len(ts) == 0:
                return {"n_celulas": 0, "n_transectos": 0,
                       "media_pond_celula_m": None, "mediana_m": None,
                       "media_por_transecto_m": None}
            sub2 = sub[sub["transecto"].isin(ts)]
            medias_t = sub2.groupby("transecto")["vies_rotulo"].mean()
            return {"n_celulas": int(len(sub2)), "n_transectos": len(ts),
                    "media_pond_celula_m": float(sub2["vies_rotulo"].mean()),
                    "mediana_m": float(sub2["vies_rotulo"].median()),
                    "media_por_transecto_m": float(medias_t.mean()),
                    "celulas_por_transecto": {t: int(n) for t, n in cel_por_t.items() if t in ts}}

        sem_piso = ponto(s, transectos_todos)
        com_piso = ponto(s, transectos_piso)

        indeterminada = len(transectos_piso) < MIN_TRANSECTOS
        ic_sem_piso = (_bootstrap_cluster(s, transectos_todos, rot, "sem_piso")
                       if transectos_todos else {})
        ic_com_piso = ({} if indeterminada else
                       _bootstrap_cluster(s, transectos_piso, rot, "com_piso"))

        veredito = "indeterminada"
        motivo = None
        if indeterminada:
            motivo = (f"menos de {MIN_TRANSECTOS} transectos acima do piso de "
                      f"{PISO_CEL} celulas ({len(transectos_piso)} disponiveis)")
        else:
            ic_pond = ic_com_piso["ic95_pond_celula"]
            ic_trans = ic_com_piso["ic95_media_por_transecto"]
            exclui_zero_pond = ic_pond[0] > 0 or ic_pond[1] < 0
            exclui_zero_trans = ic_trans[0] > 0 or ic_trans[1] < 0
            if exclui_zero_pond and exclui_zero_trans:
                sinal = "+" if com_piso["media_pond_celula_m"] > 0 else "-"
                veredito = f"sustenta_sinal_{sinal}"
            else:
                veredito = "indeterminada"
                motivo = ("IC95 (com piso) nao exclui zero em pelo menos um dos "
                          "dois estimandos de aceite (media ponderada por celula, "
                          "media por transecto)")

        out[rot] = {
            "n_celulas": n_cel,
            "n_transectos": len(transectos_todos),
            "n_transectos_acima_do_piso": len(transectos_piso),
            "transectos": transectos_todos,
            "transectos_acima_do_piso": transectos_piso,
            "transectos_excluidos_pelo_piso": excluidos_piso,
            "celulas_por_transecto": {t: int(n) for t, n in cel_por_t.items()},
            "sem_piso": {**sem_piso, **ic_sem_piso},
            "com_piso": {**com_piso, **ic_com_piso},
            "veredito": veredito, "motivo_indeterminada": motivo,
        }
    return out


def sensibilidade_np_t_0639(ax: pd.DataFrame) -> dict:
    """Specifically records the effect of the single-cell transect
    NP_T-0639 on the classes where it appears: point estimate with and
    without that transect, for both the pool and per-transect estimands."""
    alvo = "NP_T-0639"
    out = {}
    for rot in ROT_F + ["TODAS"]:
        s = ax if rot == "TODAS" else ax[ax["faixa_ex"] == rot]
        if alvo not in set(s["transecto"]):
            continue
        com = s
        sem = s[s["transecto"] != alvo]
        medias_com = com.groupby("transecto")["vies_rotulo"].mean()
        medias_sem = sem.groupby("transecto")["vies_rotulo"].mean()
        out[rot] = {
            "n_celulas_no_transecto": int((s["transecto"] == alvo).sum()),
            "vies_rotulo_no_transecto_m": float(
                s.loc[s["transecto"] == alvo, "vies_rotulo"].mean()),
            "com_transecto": {
                "n_transectos": int(com["transecto"].nunique()),
                "media_pond_celula_m": float(com["vies_rotulo"].mean()),
                "media_por_transecto_m": float(medias_com.mean())},
            "sem_transecto": {
                "n_transectos": int(sem["transecto"].nunique()),
                "media_pond_celula_m": float(sem["vies_rotulo"].mean()) if len(sem) else None,
                "media_por_transecto_m": float(medias_sem.mean()) if len(medias_sem) else None},
        }
    return out


def montar_dump(ax: pd.DataFrame) -> pd.DataFrame:
    """Dump keyed by (quad, idx_no, transecto, classe_eth, termo_rotulo) --
    Table III joins on this, without re-reading .laz."""
    cols = {"quad": "quad", "idx_no": "idx_no", "transecto": "transecto",
            "faixa_ex": "classe_eth", "vies_rotulo": "termo_rotulo",
            "h_eth": "h_eth", "cob_eth": "cob_eth",
            "e_modelo": "e_modelo_m", "fidelidade": "fidelidade_m"}
    faltantes = [c for c in cols if c not in ax.columns]
    if faltantes:
        raise RuntimeError(f"colunas faltando no dump: {faltantes}")
    dump = ax[list(cols)].rename(columns=cols).copy()
    dump["classe_eth"] = dump["classe_eth"].astype(str)
    piso_por_classe_transecto = (
        dump.groupby(["classe_eth", "transecto"]).size().rename("n_cel_classe_transecto"))
    dump = dump.merge(piso_por_classe_transecto, on=["classe_eth", "transecto"], how="left")
    dump["acima_do_piso"] = dump["n_cel_classe_transecto"] >= PISO_CEL
    return dump.reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet",
                    default=str(Path("/trabalho/GNN_TOPO/SATELITES/lidar_eba"
                                     "/solo_lidar_30m_v3_nativa.parquet")))
    ap.add_argument("--juiz", default="juiz_grade_nativa_diretas.json")
    ap.add_argument("--laterais",
                    default=str(Path("/trabalho/GNN_TOPO/SATELITES/laterais_nativa_diretas")))
    ap.add_argument("--saida", default="auditar_tabela3.json")
    ap.add_argument("--dump", default="auditar_tabela3_dump.parquet")
    a = ap.parse_args()
    parquet, juiz_p = Path(a.parquet), Path(a.juiz)
    if not juiz_p.is_absolute():
        juiz_p = RAIZ / juiz_p
    saida, dump_p = RAIZ / a.saida, RAIZ / a.dump
    J.LAT = Path(a.laterais)

    log(f"  parquet {parquet.name} | juiz {juiz_p.name} | laterais {J.LAT.name}")
    juiz = json.loads(juiz_p.read_text(encoding="utf-8"))
    ax, info = montar_populacao(parquet, juiz)
    log(f"  populacao: {info['n_linhas_parquet']:,} linhas -> "
        f"{info['n_floresta_julgavel']:,} floresta julgavel -> "
        f"{info['n_ancoras_dossel_exogeno']:,} com ancora e dossel ETH "
        f"({info['n_julgaveis']} transectos julgaveis)")

    conf = guarda_reproducao(ax, juiz)
    resumo_guarda = [f"{r}: {conf[r]['vies_rotulo_m_recomputado']:+.4f}"
                     for r in conf if conf[r]["vies_rotulo_m_recomputado"] == conf[r]["vies_rotulo_m_recomputado"]]
    log("  GUARDA: reproduz decomposicao_por_dossel_exogeno_eth (n exato, "
        "vies_rotulo_m a 1e-6) em todas as classes — " + str(resumo_guarda))
    log(f"  >>> classe >30m recomputada = {conf['>30m']['vies_rotulo_m_recomputado']:+.4f} m "
        f"(juiz = {conf['>30m']['vies_rotulo_m_juiz']:+.4f} m) <<<")

    por_classe = estatisticas_por_classe(ax)
    log("\n  === Tabela III recomputada, por classe ETH (com piso) ===")
    for rot in ROT_F + ["TODAS"]:
        r = por_classe[rot]
        cp = r["com_piso"]
        if r["veredito"] == "indeterminada" and "ic95_pond_celula" not in cp:
            log(f"  {rot:8s} n_cel={r['n_celulas']:5d} n_t={r['n_transectos']:2d} "
                f"n_t_piso={r['n_transectos_acima_do_piso']:2d} -> INDETERMINADA "
                f"({r['motivo_indeterminada']})")
            continue
        log(f"  {rot:8s} n_cel={r['n_celulas']:5d} n_t_piso={r['n_transectos_acima_do_piso']:2d} "
            f"pond={cp['media_pond_celula_m']:+.2f} IC{cp['ic95_pond_celula']} "
            f"mediana={cp['mediana_m']:+.2f} "
            f"p/transecto={cp['media_por_transecto_m']:+.2f} IC{cp['ic95_media_por_transecto']} "
            f"-> {r['veredito']}")

    sens_0639 = sensibilidade_np_t_0639(ax)
    log(f"\n  sensibilidade NP_T-0639 (com/sem transecto): "
        f"{ {k: (v['com_transecto']['media_pond_celula_m'], v['sem_transecto']['media_pond_celula_m']) for k, v in sens_0639.items()} }")

    dump = montar_dump(ax)
    dump.to_parquet(dump_p, index=False)
    log(f"  -> {dump_p.name} ({len(dump):,} linhas)")

    fontes = [parquet, juiz_p, RAIZ / "juiz_lidar_v4.py", Path(__file__),
             J.LAT / "_proveniencia.json"]
    fontes += [J.LAT / f"{n}_amazonia_{q}.npz" for q in J.QUADS
              for n in ("glo30", "fabdem", "gedtm30", "agua_jrc", "dossel_eth", "rotulos")]
    fontes += [J.SUP / f"delta_amazonia_ampliacao_b4s035{arm}_seed{sd}.npz"
              for arm in ("", "_mlp_semobs") for sd in J.SEEDS_PAR]

    PROV.gravar(saida, {
        "hipotese": "o termo de rotulo acima de 20 m de dossel ETH e positivo, "
                    "com IC agrupado por transecto excluindo zero",
        "criterio_pre_registrado": {
            "piso_celulas_por_transecto_classe": PISO_CEL,
            "min_transectos_acima_do_piso": MIN_TRANSECTOS,
            "bootstrap": {"B": B, "semente": SEED, "cluster": "transecto"},
            "alpha": ALPHA,
            "estimandos": ["media_ponderada_por_celula", "mediana",
                          "media_das_medias_por_transecto"],
            "aceite": "IC95 (com piso) exclui zero nos dois estimandos "
                     "media_ponderada_por_celula e media_das_medias_por_transecto; "
                     "senao a classe e indeterminada",
            "fonte_das_constantes": ("inherited from auditar_nmad_pareado.py:63; no other value was fixed for "
                                     "this run"),
            "leitura_de_ambiguidade": ("the floor restricts the input CELLS (only cells from transects with >= "
                                       "floor cells in the class)"),
        },
        "entradas": {"parquet": str(parquet), "juiz": str(juiz_p), "laterais": str(J.LAT)},
        "populacao": info,
        "guarda_reproducao_decomposicao_por_dossel_exogeno_eth": conf,
        "por_classe_eth": por_classe,
        "sensibilidade_transecto_1_celula_NP_T_0639": sens_0639,
        "dump": {"arquivo": dump_p.name, "n_linhas": int(len(dump)),
                "colunas": list(dump.columns), "chave": ["quad", "idx_no"]},
        "contra_auditoria": ("PENDING — not yet checked by anyone other than the author of this script "
                             "(rule: no one audits their own result)"),
    }, fontes_lidas=fontes, script=__file__)
    log(f"  -> {saida.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
