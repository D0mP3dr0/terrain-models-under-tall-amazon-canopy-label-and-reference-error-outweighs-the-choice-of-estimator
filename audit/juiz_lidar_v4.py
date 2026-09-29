"""LIDAR JUDGE v4 -- fixes the defects found in v3, and moves the primary
result to the part that does NOT depend on the ground-truth reference.

DEFECTS OF v3 FIXED HERE:
  1. DEDUP. An earlier revision lost the `drop_duplicates` step: 153,644 rows
     for 137,518 unique cells. ~9,000 cells appeared in TWO transects, each
     copy with a different `z_ref` (up to 11 m apart). Now the copy kept is
     the one from the transect with MORE clearings (better-constrained
     registration).
  2. HOLM WITHOUT MONOTONICITY. `p*(m-i)` without a running maximum recorded
     mlp_vs_fabdem at 0.0249, BELOW mlp_vs_gedtm30 at 0.0308, with the raw p
     in the reverse order. Now computed with a running maximum.
  3. ONE-SIDED VERDICT. `estavel` only tested the pro-V23 hypothesis; when it
     failed, the JSON recorded "order NOT stable" -- the opposite of what the
     data showed (the reversed order was stable in 11/11 cases). Now the
     test has THREE outcomes.
  4. VACUOUS FILTERS. `n_casca >= 20` rejects 0.00% of cells: the shell
     starts at p2, so n_casca >= 2% of n_all BY CONSTRUCTION. And the NMAD of
     a 1 m shell is ~0.37 m by construction, so `nmad <= 1.5` does not
     measure "firm ground". Replaced with guards that actually measure this:
     a minimum n_all and the fraction of returns within the shell.
  5. EXCLUSION BY A REFERENCE-FREE CRITERION. v3 excluded by `n_clareira`
     while keeping transects 0359 (+20.2 m) and 0663 (+11.1 m) with a
     justification -- shown here to be FALSE -- that they were flight
     registration artifacts. Falsifying test: a 20 m vertical offset between
     the point cloud and Copernicus would ALSO show up in
     `p95_lidar - glo30`, which does not use the ground estimator. It does
     not (9.18 and 7.19 m, within the 7.0-12.3 m range of the other 19).
     New criterion, written before looking at the table: a transect is
     dropped if the clearing-offset anomaly is not corroborated by the
     canopy-top anomaly (3 m threshold).
     Note: excluding both transects makes V23 look WORSE. This is hygiene,
     not rescue.
  6. FULL TESTING. Sign test (binomial), Wilcoxon, effect size and bootstrap
     CI together -- the earlier version reported only the count and the p.

THE PRIMARY RESULT MOVES. The V23 x FABDEM ranking depends on the
conditional bias of the ground-truth reference (delta), which is NOT
measured here: the lower-shell ground estimator reads low under dense
canopy because the fraction of ground returns drops to ~2% while the
estimator's anchor stays fixed at p2. Once delta is measured by a separate
scale study, the difference is roughly halved and loses significance. So
this judge promotes the DECOMPOSITION, which is exact and reference-free:

    (pred - z_ref)      =    (pred - ground_anchor)    +   (ground_anchor - z_ref)
     error vs LiDAR          fidelity to supervision       LABEL bias
                            REFERENCE-FREE                 carries all the
                            (pred and ground_anchor are     reference
                             both in the GLO30 frame:        uncertainty
                             pred - ground_anc = y - Delta)

Usage:  python juiz_lidar_v4.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RAIZ = Path(__file__).resolve().parent
LAT = Path(os.environ.get("V23_LATDIR", r"D:\GNN_TOPO\SATELITES\laterais"))
SOLO = Path(r"D:\GNN_TOPO\SATELITES\lidar_eba\solo_lidar_30m_v2.parquet")
SUP = RAIZ / "superficies"
SAIDA = RAIZ / "juiz_lidar_v4.json"
QUADS = ["Q1", "Q2", "Q3", "Q4"]
SEEDS_PAR = [42, 123, 7, 2024, 31]      # 5x5 parity between GNN and MLP
FAIXAS = [-1, 3, 10, 20, 30, 500]
ROT_F = ["0-3m", "3-10m", "10-20m", "20-30m", ">30m"]
# guards that actually MEASURE something (see defect 4)
N_ALL_MIN = 500              # enough point-cloud density for p2 to be stable
FRAC_CASCA_MIN = 0.005       # >=0.5% of returns within the shell
AGUA_LIMPO = 50.0
DOSSEL_FLORESTA = 2.0
N_CLAREIRA_MIN = 10
LIM_CORROBORACAO = 3.0       # m -- reference-free exclusion criterion
PARES = [("v23", "fabdem"), ("v23", "gedtm30"),
         ("mlp", "fabdem"), ("mlp", "gedtm30"), ("v23", "mlp")]
PRODUTOS = ["v23", "mlp", "fabdem", "gedtm30", "glo30"]


def log(m=""):
    print(m, flush=True)


def montar() -> pd.DataFrame:
    d = pd.read_parquet(SOLO)
    partes = []
    for q in QUADS:
        s = d[d["quad"] == q].copy()
        if s.empty:
            continue
        idx = s["idx_no"].to_numpy()
        glo_todo = np.load(LAT / f"glo30_amazonia_{q}.npz")["DEM"]
        glo = glo_todo[idx]
        s["glo30"] = glo
        s["fabdem"] = np.load(LAT / f"fabdem_amazonia_{q}.npz")["b1"][idx]
        s["gedtm30"] = np.load(LAT / f"gedtm30_amazonia_{q}.npz")["DTM"][idx]
        # JRC SENTINEL VALUE: -128 is not "negative occurrence", it means "no
        # water observed" -- 77-99% of cells per quadrant. `_norm_agua` in
        # the feature vector already maps this to 0 in the FEATURES; here,
        # in the judge, the raw threshold worked by accident (-128 < 50) but
        # described something else. Made explicit: the population does not
        # change, the statement just becomes accurate. This is the only
        # sentinel value present in this source.
        agua = np.load(LAT / f"agua_jrc_amazonia_{q}.npz")["occurrence"][idx]
        s["agua"] = np.where(agua < 0, 0.0, agua)
        # EXOGENOUS canopy height (ETH 10 m, Lang et al. 2023) -- a
        # stratifier that shares neither the LiDAR ground estimator nor the
        # model's label
        f_eth = LAT / f"dossel_eth_amazonia_{q}.npz"
        if f_eth.exists():
            z = np.load(f_eth)
            s["h_eth"] = z["h_dossel"][idx]
            s["cob_eth"] = z["frac_valida"][idx]
            z.close()
        for tag, arm in (("v23", ""), ("mlp", "_mlp_semobs")):
            acc = np.zeros(len(idx))
            for sd in SEEDS_PAR:
                z = np.load(SUP / f"delta_amazonia_ampliacao_b4s035{arm}_seed{sd}.npz")
                acc += z[q][idx]
                z.close()
            s[tag] = glo - acc / len(SEEDS_PAR)
        # anchor label, in the SAME reference frame as the products (GLO30 - y)
        r = np.load(LAT / f"rotulos_amazonia_{q}.npz")
        bom = r["bom"] > 0.5
        i_anc = r["idx_no"][bom].astype(np.int64)
        anc = pd.DataFrame({"idx_no": i_anc,
                            "solo_anc": glo_todo[i_anc] - r["y_novo"][bom],
                            "fonte": r["fonte"][bom]})
        r.close()
        s = s.merge(anc.groupby("idx_no").first().reset_index(),
                    on="idx_no", how="left")
        partes.append(s)
    return pd.concat(partes, ignore_index=True)


def metricas(err: np.ndarray) -> dict:
    if len(err) == 0:
        return {}
    ae = np.abs(err)
    return {"n": int(len(err)), "mae": float(ae.mean()),
            "vies": float(err.mean()),
            "nmad": float(1.4826 * np.median(np.abs(err - np.median(err)))),
            "rmse": float(np.sqrt(np.mean(err ** 2)))}


def holm(pares_p):
    """Holm-Bonferroni WITH a running maximum (an earlier version omitted monotonicity)."""
    ordenados = sorted(pares_p, key=lambda x: (np.isnan(x[1]), x[1]))
    m = len(ordenados)
    saida, acum = {}, 0.0
    for i, (k, p) in enumerate(ordenados):
        if not np.isfinite(p):
            saida[k] = None
            continue
        aj = min(1.0, p * (m - i))
        acum = max(acum, aj)
        saida[k] = acum
    return saida


def main() -> int:
    d = montar()
    n0 = len(d)
    d = d[(d["n_all"] >= N_ALL_MIN)
          & (d["n_casca"] / d["n_all"] >= FRAC_CASCA_MIN)
          & np.isfinite(d["z_solo_v2"])]
    log(f"  {n0:,} linhas -> {len(d):,} apos guardas de qualidade "
        f"(n_all>={N_ALL_MIN}, frac_casca>={FRAC_CASCA_MIN})")

    # ── clearing offsets and TOP-of-canopy anomaly (free of the ground estimator)
    rng = np.random.default_rng(42)
    offsets = {}
    for t, sub in d.groupby("transecto"):
        cl = sub[(sub["dossel_v2"] < DOSSEL_FLORESTA) & (sub["agua"] < 10.0)]
        res = (cl["glo30"] - cl["z_solo_v2"]).to_numpy()
        topo = float(np.median(sub["z_p95_all"] - sub["glo30"]))
        if len(res) >= 3:
            bs = [float(np.median(rng.choice(res, len(res), True)))
                  for _ in range(1000)]
            ic = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
        else:
            ic = (float("nan"), float("nan"))
        offsets[str(t)] = {"n_clareira": int(len(res)),
                           "offset_m": (float(np.median(res)) if len(res)
                                        else float("nan")),
                           "ic95": ic, "topo_p95_menos_glo30_m": topo}

    med_off = float(np.nanmedian([v["offset_m"] for v in offsets.values()]))
    med_topo = float(np.median([v["topo_p95_menos_glo30_m"]
                                for v in offsets.values()]))
    for t, v in offsets.items():
        anom_off = v["offset_m"] - med_off
        anom_topo = v["topo_p95_menos_glo30_m"] - med_topo
        v["anomalia_offset_m"] = float(anom_off)
        v["anomalia_topo_m"] = float(anom_topo)
        v["corroborado"] = bool(np.isfinite(anom_off)
                                and abs(anom_off - anom_topo) <= LIM_CORROBORACAO)
        v["julgavel"] = bool(v["n_clareira"] >= N_CLAREIRA_MIN and v["corroborado"])
    julgaveis = [t for t, v in offsets.items() if v["julgavel"]]
    fora = {t: ("poucas clareiras" if v["n_clareira"] < N_CLAREIRA_MIN
                else "offset nao corroborado pelo topo")
            for t, v in offsets.items() if not v["julgavel"]}
    log(f"  julgaveis: {len(julgaveis)} | fora: {fora}")

    d = d[d["transecto"].isin(julgaveis)].copy()
    d["off_t"] = d["transecto"].map({t: offsets[t]["offset_m"] for t in julgaveis})
    d["z_ref"] = d["z_solo_v2"] + d["off_t"]

    # ── DEDUP: a cell in two transects is kept from the one with better registration
    n_cl = {t: offsets[t]["n_clareira"] for t in julgaveis}
    d["_rank"] = d["transecto"].map(n_cl)
    antes = len(d)
    d = (d.sort_values("_rank", ascending=False)
           .drop_duplicates(subset=["quad", "idx_no"], keep="first"))
    log(f"  dedup: {antes:,} -> {len(d):,} celulas unicas "
        f"({antes-len(d):,} duplicadas removidas)")

    d["faixa"] = pd.cut(d["dossel_v2"], FAIXAS, labels=ROT_F)
    flor = d[(d["dossel_v2"] >= DOSSEL_FLORESTA) & (d["agua"] < AGUA_LIMPO)]
    log(f"  floresta julgavel: {len(flor):,} celulas em "
        f"{flor['transecto'].nunique()} transectos")

    # ── table against the ground-truth reference (DEPENDS on the delta bias, not yet measured)
    tab = {}
    for p in PRODUTOS:
        err = (flor[p] - flor["z_ref"]).to_numpy()
        tab[p] = {rot: metricas(err[(flor["faixa"] == rot).to_numpy()])
                  for rot in ROT_F}
        tab[p]["TODAS"] = metricas(err)
    log("\n  === MAE contra a regua v2 (k=0; NAO corrigida do vies delta) ===")
    log("  " + f"{'faixa':8s}" + "".join(f"{p:>9s}" for p in PRODUTOS) + f"{'n':>10s}")
    for rot in ROT_F + ["TODAS"]:
        log("  " + f"{rot:8s}"
            + "".join(f"{tab[p][rot].get('mae', float('nan')):9.2f}" for p in PRODUTOS)
            + f"{tab['v23'][rot].get('n', 0):10,d}")

    # ── relative reduction against the input product: INVARIANT to a shift of the reference
    red = {}
    for p in ("v23", "mlp", "fabdem", "gedtm30"):
        red[p] = {
            "reducao_mae_vs_glo30": float(1 - tab[p]["TODAS"]["mae"]
                                          / tab["glo30"]["TODAS"]["mae"]),
            "reducao_vies_vs_glo30": float(1 - abs(tab[p]["TODAS"]["vies"])
                                           / abs(tab["glo30"]["TODAS"]["vies"]))}
    log("\n  reducao vs GLO30 (MAE / vies): " + " | ".join(
        f"{p} {red[p]['reducao_mae_vs_glo30']:.0%}/"
        f"{red[p]['reducao_vies_vs_glo30']:.0%}" for p in red))

    # ── CORRECTED READING OF THE SCALE BIAS (delta), if already measured
    corr = {}
    f_delta = RAIZ / "delta_escala_regua.json"
    if f_delta.exists():
        dd = json.loads(f_delta.read_text(encoding="utf-8"))["por_faixa"]
        delta = {rot: dd[rot]["delta_m"] for rot in ROT_F if rot in dd}
        # the reference shifts by delta: z_ref' = z_ref + delta(class). The
        # same shift applies to ALL products in the same cell.
        fl = flor.copy()
        fl["delta"] = fl["faixa"].astype(str).map(delta).astype(float)
        fl = fl[np.isfinite(fl["delta"])]
        fl["z_ref_c"] = fl["z_ref"] + fl["delta"]
        for p in PRODUTOS:
            e = (fl[p] - fl["z_ref_c"]).to_numpy()
            corr[p] = {rot: metricas(e[(fl["faixa"] == rot).to_numpy()])
                       for rot in ROT_F}
            corr[p]["TODAS"] = metricas(e)
        log("\n  === MAE com a regua CORRIGIDA do vies de escala (k=delta) ===")
        log("  " + f"{'faixa':8s}" + "".join(f"{p:>9s}" for p in PRODUTOS)
            + f"{'delta':>8s}")
        for rot in ROT_F + ["TODAS"]:
            log("  " + f"{rot:8s}"
                + "".join(f"{corr[p][rot].get('mae', float('nan')):9.2f}"
                          for p in PRODUTOS)
                + f"{delta.get(rot, float('nan')):8.2f}")
        log("  vies (TODAS): " + "  ".join(
            f"{p} {corr[p]['TODAS']['vies']:+.2f}" for p in PRODUTOS))
        pt_c = fl.groupby("transecto").apply(
            lambda s: pd.Series({p: float(np.abs(s[p] - s["z_ref_c"]).mean())
                                 for p in PRODUTOS}), include_groups=False)
        dif_c = (pt_c["v23"] - pt_c["fabdem"]).to_numpy()
        try:
            _, p_wc = stats.wilcoxon(dif_c)
        except ValueError:
            p_wc = float("nan")
        corr["_teste_v23_vs_fabdem"] = {
            "vence": int((dif_c < 0).sum()), "total": int(len(dif_c)),
            "dif_media_m": float(dif_c.mean()), "p_wilcoxon": float(p_wc)}
        log(f"  v23 vs fabdem corrigido: {int((dif_c<0).sum())}/{len(dif_c)} | "
            f"dif {dif_c.mean():+.2f} m | p={p_wc:.4f}")

    # ── REFERENCE-FREE DECOMPOSITION (the primary result)
    anc = flor[flor["solo_anc"].notna()].copy()
    anc["e_modelo"] = anc["v23"] - anc["z_ref"]        # depends on the reference
    anc["fidelidade"] = anc["v23"] - anc["solo_anc"]   # FREE of the reference
    anc["vies_rotulo"] = anc["solo_anc"] - anc["z_ref"]  # carries the reference
    dec = {}
    log("\n  === DECOMPOSICAO nas celulas com ancora (n por faixa) ===")
    log(f"  {'faixa':8s}{'n':>7s}{'e_modelo':>10s}{'fidelidade':>12s}"
        f"{'vies_rotulo':>13s}{'% do erro que e rotulo':>24s}")
    for rot in ROT_F + ["TODAS"]:
        s = anc if rot == "TODAS" else anc[anc["faixa"] == rot]
        if len(s) < 5:
            continue
        em, fi, vr = (float(s["e_modelo"].mean()), float(s["fidelidade"].mean()),
                      float(s["vies_rotulo"].mean()))
        frac = float(vr / em) if abs(em) > 1e-6 else float("nan")
        dec[rot] = {"n": int(len(s)), "e_modelo_m": em, "fidelidade_m": fi,
                    "vies_rotulo_m": vr, "frac_do_erro_que_e_rotulo": frac,
                    "fidelidade_dp": float(s["fidelidade"].std())}
        log(f"  {rot:8s}{len(s):>7,d}{em:>10.2f}{fi:>12.2f}{vr:>13.2f}"
            f"{frac:>23.0%}")

    # ── THE SAME DECOMPOSITION, STRATIFIED BY EXOGENOUS CANOPY HEIGHT
    # Stratifying by `dossel_v2` is circular (the stratum shares the error of
    # the estimator that defines the target), and stratifying by the GEDI
    # canopy height reverses the pattern (it shares the label). ETH has
    # neither dependency. If the structure survives here, it is real.
    dec_ex = {}
    if "h_eth" in anc.columns:
        ax = anc[anc["h_eth"].notna() & (anc["cob_eth"] >= 0.8)].copy()
        ax["faixa_ex"] = pd.cut(ax["h_eth"], FAIXAS, labels=ROT_F)
        log(f"\n  === DECOMPOSICAO por dossel EXOGENO (ETH), "
            f"{len(ax):,} celulas com cobertura >=80% ===")
        log(f"  {'faixa':8s}{'n':>7s}{'e_modelo':>10s}{'fidelidade':>12s}"
            f"{'vies_rotulo':>13s}")
        for rot in ROT_F + ["TODAS"]:
            s = ax if rot == "TODAS" else ax[ax["faixa_ex"] == rot]
            if len(s) < 5:
                continue
            em, fi, vr = (float(s["e_modelo"].mean()),
                          float(s["fidelidade"].mean()),
                          float(s["vies_rotulo"].mean()))
            dec_ex[rot] = {"n": int(len(s)), "e_modelo_m": em,
                           "fidelidade_m": fi, "vies_rotulo_m": vr}
            log(f"  {rot:8s}{len(s):>7,d}{em:>10.2f}{fi:>12.2f}{vr:>13.2f}")

    # ── per-transect tests, with sign test + Wilcoxon + effect size + CI
    por_t = flor.groupby("transecto").apply(
        lambda s: pd.Series({p: float(np.abs(s[p] - s["z_ref"]).mean())
                             for p in PRODUTOS}), include_groups=False)
    testes, ps = {}, []
    for a, b in PARES:
        dif = (por_t[a] - por_t[b]).to_numpy()
        n = len(dif)
        vence = int((dif < 0).sum())
        p_sinal = float(stats.binomtest(vence, n, 0.5).pvalue)
        try:
            w, p_w = stats.wilcoxon(dif)
        except ValueError:
            w, p_w = float("nan"), float("nan")
        bs = [float(np.mean(rng.choice(dif, n, True))) for _ in range(2000)]
        testes[f"{a}_vs_{b}"] = {
            "vence": vence, "total": n, "dif_media_m": float(dif.mean()),
            "dif_mediana_m": float(np.median(dif)),
            "ic95_bootstrap": [float(np.percentile(bs, 2.5)),
                               float(np.percentile(bs, 97.5))],
            "p_sinal_binomial": p_sinal, "wilcoxon_W": float(w),
            "p_wilcoxon": float(p_w)}
        ps.append((f"{a}_vs_{b}", p_w))
    for k, v in holm(ps).items():
        testes[k]["p_holm"] = v
    log("\n  === por transecto (sinal | Wilcoxon | efeito) ===")
    for k, v in testes.items():
        log(f"  {k:16s} {v['vence']:2d}/{v['total']} (p_sinal {v['p_sinal_binomial']:.3f})"
            f" | dif {v['dif_media_m']:+.2f} m "
            f"[{v['ic95_bootstrap'][0]:+.2f},{v['ic95_bootstrap'][1]:+.2f}]"
            f" | p_holm {v['p_holm']:.4f}" if v["p_holm"] is not None else "")

    # ── SYMMETRIC verdict (three outcomes)
    # ── VERDICT: the ranking is only citable if it survives BOTH readings
    m_v23 = tab["v23"]["TODAS"]["mae"]
    m_fab = tab["fabdem"]["TODAS"]["mae"]
    t_fab = testes["v23_vs_fabdem"]
    tc = corr.get("_teste_v23_vs_fabdem")
    if tc is None:
        veredito = ("INDETERMINADO: delta nao medido (rode "
                    "medir_delta_escala.py antes de ler ranking)")
    else:
        sig_crua = (t_fab["p_holm"] is not None and t_fab["p_holm"] < 0.05)
        sig_corr = np.isfinite(tc["p_wilcoxon"]) and tc["p_wilcoxon"] < 0.05
        if sig_crua and sig_corr:
            veredito = (f"diferenca V23 x FABDEM sobrevive as duas leituras: "
                        f"crua {m_v23:.2f} vs {m_fab:.2f} m (p_holm "
                        f"{t_fab['p_holm']:.4f}); corrigida "
                        f"{corr['v23']['TODAS']['mae']:.2f} vs "
                        f"{corr['fabdem']['TODAS']['mae']:.2f} m "
                        f"(p {tc['p_wilcoxon']:.4f})")
        else:
            veredito = (
                f"RANKING NAO RESOLVIVEL POR ESTE INSTRUMENTO. Na regua crua a "
                f"diferenca V23-FABDEM e {t_fab['dif_media_m']:+.2f} m "
                f"(p_holm {t_fab['p_holm']:.4f}); corrigida do vies de escala "
                f"MEDIDO ela vai a {tc['dif_media_m']:+.2f} m "
                f"(p {tc['p_wilcoxon']:.3f}, {tc['vence']}/{tc['total']}). O "
                f"proprio vies da regua nas faixas que dominam a amostra "
                f"(+2,67 e +4,81 m em 20-30 e >30 m) e MAIOR que a diferenca "
                f"entre os produtos — o instrumento nao separa os dois. Nem "
                f"'V23 vence' nem 'FABDEM vence' e afirmavel.")
    log(f"\n  VEREDITO (ranking): {veredito}")
    log(f"  RESULTADO PRINCIPAL (livre de regua): sob dossel >20 m, "
        f"{dec.get('20-30m', {}).get('frac_do_erro_que_e_rotulo', float('nan')):.0%}"
        f" e {dec.get('>30m', {}).get('frac_do_erro_que_e_rotulo', float('nan')):.0%}"
        f" do erro do produto contra o lidar E a definicao de chao do rotulo; "
        f"a fidelidade do modelo ao proprio rotulo e "
        f"{dec.get('>30m', {}).get('fidelidade_m', float('nan')):+.2f} m.")

    SAIDA.write_text(json.dumps({
        "correcoes_vs_v3": ["dedup de celulas compartilhadas",
                            "Holm com maximo acumulado",
                            "veredito simetrico (3 saidas)",
                            "guardas de qualidade que medem (n_all, frac_casca)",
                            "exclusao por corroboracao topo x clareira",
                            "sinal + Wilcoxon + efeito + IC juntos",
                            "paridade de ensemble 5x5"],
        "ressalva_regua": "o vies condicional (delta) do estimador de casca "
                          "inferior sob dossel denso NAO esta medido; a "
                          "ordenacao contra a regua depende dele",
        "offsets_por_transecto": offsets, "fora_do_julgamento": fora,
        "tabela_regua_k0": tab, "reducao_vs_glo30": red,
        # The CORRECTED half of the table used to be computed but not saved
        # to the artifact -- the manuscript numbers came from stdout read by
        # eye instead of from a recorded artifact. Now the `corr` block
        # (MAE with the reference corrected for the scale bias, per product
        # and class, plus the corrected v23 x fabdem test) is part of the
        # artifact.
        "tabela_regua_corrigida_delta": corr,
        "decomposicao_livre_de_regua": dec,
        "decomposicao_por_dossel_exogeno_eth": dec_ex,
        "por_transecto_mae": {str(k): {p: float(v) for p, v in r.items()}
                              for k, r in por_t.iterrows()},
        "testes": testes, "veredito_ranking": veredito},
        indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    log(f"  -> {SAIDA.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
