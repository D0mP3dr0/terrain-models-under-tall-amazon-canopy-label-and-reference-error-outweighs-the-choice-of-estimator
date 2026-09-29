"""Fig. (profile) — What the products look like on the ground: profile along
one airborne-lidar strip.

Declared strip-selection rule (to avoid cherry-picking): the strip retained by the
judge with the largest median canopy height (lidar p95 - lidar ground), the regime
that motivates the article. This selects NP_T-0224 (Q1).
(a) elevation along the strip's principal axis (90 m bins, median of the cells in each
    bin): lidar ground, lidar top (p95, forest filled between the two), GLO-30, FABDEM,
    GEDTM30, ANADEM and the two products of this study (GLO-30 minus the mean of the 5
    seeds of Delta, as in the judge).
(b) product minus lidar ground, same bins.
Per-strip MAE values in the annotation come from juiz_grade_nativa_diretas.json
(por_transecto_mae, k0 yardstick, no offset correction), NOT from the profile; the
profile is an illustration, not a number. Node geometry: reference solo_lidar_30m_v3_nativa;
covariates from laterais_nativa_diretas (GEDTM30 read directly from the COG, ETH from the 10 m product).
Raw inputs (recorded with sha256): solo_lidar_30m_v3_nativa.parquet,
laterais_nativa_diretas/*.npz,
superficies/delta_amazonia_ampliacao_b4s035*_seed*.npz.

ANADEM uses laterais_nativa_diretas/anadem_amazonia_Q*.npz, key "DTM", on the same node
grid as GEDTM30. It has no per-strip MAE in juiz_grade_nativa_diretas.json, so it appears
only as a profile series and is marked "not computed per strip" in the annotation.
Satellite inputs are resolved from E.RAIZ.parent / "SATELITES".
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import estilo_jstars as E

E.aplicar()
SAT = E.RAIZ.parent / "SATELITES"
LAT = SAT / "laterais_nativa_diretas"
SOLO = SAT / "lidar_eba" / "solo_lidar_30m_v3_nativa.parquet"
SUP = E.RAIZ / "superficies"
J = E.RAIZ / "juiz_grade_nativa_diretas.json"
SEEDS = [42, 123, 7, 2024, 31]
BIN_M = 90.0

j = json.loads(J.read_text(encoding="utf-8"))
julg = {t for t, v in j["offsets_por_transecto"].items() if v["julgavel"]}
d = pd.read_parquet(SOLO)
d["h_lidar"] = d["z_p95_all"] - d["z_solo_v2"]
med = d[d["transecto"].isin(julg)].groupby("transecto")["h_lidar"].median()
T = med.idxmax()                              # declared selection rule
s = d[d["transecto"] == T].copy(); q = s["quad"].iloc[0]; idx = s["idx_no"].to_numpy()
glo = np.load(LAT / f"glo30_amazonia_{q}.npz")["DEM"][idx]
s["glo30"] = glo
s["fabdem"] = np.load(LAT / f"fabdem_amazonia_{q}.npz")["b1"][idx]
s["gedtm30"] = np.load(LAT / f"gedtm30_amazonia_{q}.npz")["DTM"][idx]
s["anadem"] = np.load(LAT / f"anadem_amazonia_{q}.npz")["DTM"][idx]
fontes = [J]
for tag, arm in (("v23", ""), ("mlp", "_mlp_semobs")):
    acc = np.zeros(len(idx))
    for sd in SEEDS:
        f = SUP / f"delta_amazonia_ampliacao_b4s035{arm}_seed{sd}.npz"
        z = np.load(f); acc += z[q][idx]; z.close(); fontes.append(f)
    s[tag] = glo - acc / len(SEEDS)
# principal axis (metres) and bins
xy = np.column_stack([s["col"].to_numpy() * 30.0, s["lin"].to_numpy() * 30.0]); xy -= xy.mean(0)
w_, v_ = np.linalg.eigh(np.cov(xy.T)); ax1 = v_[:, 1]
s["s_m"] = xy @ ax1; s["s_m"] -= s["s_m"].min()
s["bin"] = (s["s_m"] // BIN_M).astype(int)
cols = ["z_solo_v2", "z_p95_all", "glo30", "fabdem", "gedtm30", "anadem", "v23", "mlp"]
g = s.groupby("bin").agg(**{c: (c, "median") for c in cols}, n=("lin", "size"), s_km=("s_m", "median"))
g = g[g["n"] >= 5]; g["s_km"] = g["s_km"] / 1000.0
pm = j["por_transecto_mae"][T]

fig, (a, b) = plt.subplots(2, 1, figsize=(E.COL2, 3.6), sharex=True,
                           gridspec_kw=dict(height_ratios=[2.2, 1.0], hspace=0.08))
x = g["s_km"].to_numpy()
a.fill_between(x, g["z_solo_v2"], g["z_p95_all"], color="#cfe8c4", lw=0, label="Forest (lidar ground to lidar canopy top, 95th percentile)")
a.plot(x, g["z_p95_all"], color="#4f9d4a", lw=0.6)
a.plot(x, g["z_solo_v2"], color="k", lw=1.2, label="Airborne-lidar ground")
# the three external products in grey, with distinct line styles
SER = [("glo30", E.PROD["glo30"], "-"), ("fabdem", E.PROD["fabdem"], ":"), ("gedtm30", E.PROD["gedtm30"], "--"),
       ("anadem", E.PROD["anadem"], "-."), ("mlp", E.PROD["mlp"], "-"), ("v23", E.PROD["v23"], "-")]
for c, P, ls in SER:
    a.plot(x, g[c], color=P["cor"], lw=1.1 if c in ("v23", "mlp") else 0.9, ls=ls,
           label=P["nome"])
    b.plot(x, g[c] - g["z_solo_v2"], color=P["cor"], lw=1.1 if c in ("v23", "mlp") else 0.9, ls=ls)
b.axhline(0, color="k", lw=0.8)
b.set_ylabel("Product − lidar\nground (m)")
a.set_ylabel("Elevation (m)")
b.set_xlabel("Distance along the strip (km), 90 m bins (median of the cells in each bin)")
a.legend(loc="upper left", fontsize=7.5, ncol=4, columnspacing=0.8, handlelength=1.8)
a.text(0.0, 1.02, "MAE against the lidar ground on this strip (m): GLO-30 %.2f · FABDEM %.2f · GEDTM30 %.2f ·\n"
       "pointwise %.2f · graph %.2f · ANADEM not computed per strip" % (pm["glo30"], pm["fabdem"], pm["gedtm30"], pm["mlp"], pm["v23"]),
       transform=a.transAxes, ha="left", va="bottom", fontsize=7.5, color="k")   # placed above the curves
ymin = min(g["z_solo_v2"].min(), g[["fabdem", "gedtm30", "anadem", "v23", "mlp"]].min().min()) - 3
a.set_ylim(ymin - 4, g["z_p95_all"].max() + 30)   # headroom keeps the legend clear of the canopy-top curve
b.set_ylim(min(-2, (g[["fabdem", "gedtm30", "anadem", "v23", "mlp"]].min(axis=1) - g["z_solo_v2"]).min() - 1),
           (g["glo30"] - g["z_solo_v2"]).max() + 2)
E.rotular_paineis((a, b), x=-0.075)
fig.subplots_adjust(left=0.085, right=0.995, top=0.93, bottom=0.12)
numeros = {
    "regra_de_escolha": "faixa mantida com maior altura mediana de dossel (p95 - solo do lidar)",
    "faixa": T, "quadrante": q, "n_celulas": int(len(s)), "n_bins": int(len(g)), "bin_m": BIN_M,
    "altura_mediana_dossel_lidar_m": float(med[T]),
    "mae_por_produto_nesta_faixa_m_juiz_k0": pm,
    "mae_fonte": "juiz_grade_nativa_diretas.json -> por_transecto_mae." + T,
    "candidatas_altura_mediana_m": {t: float(v) for t, v in med.sort_values(ascending=False).items()},
    "produto_deste_estudo": "GLO-30 menos media de 5 sementes de Delta (b4s035; mlp_semobs), como em juiz_lidar_v4_nativa.py",
    "geometria": "no (Copernicus): solo_lidar_30m_v3_nativa + laterais_nativa_diretas",
    "nota": "perfil e ilustracao (medianas por bin de 90 m); numeros citaveis sao os do juiz",
    "anadem_2026-09-28c": (("product added to panels (a)/(b); "
                            "laterais_nativa_diretas/anadem_amazonia_{quadrante}.npz, key DTM, same "
                            "node grid as GEDTM30; color/marker from estilo_jstars.PROD['anadem'] "
                            "(#808080, X), same code as in Fig. 7")),
    "r4_2026-09-28": (("MAE annotation moved above panel (a), clear of the curves, with 'ANADEM "
                       "not computed per strip'; distinct dashes for the three external products "
                       "in gray (FABDEM ':', ANADEM '-.', GEDTM30 '--'); legend 'lidar canopy "
                       "top (95th percentile)'")),
    "anadem_mae_nesta_faixa": ("nao incluido no texto de MAE do painel (a): "
                               "juiz_grade_nativa_diretas.json.por_transecto_mae nao tem entrada "
                               "'anadem' para nenhum transecto; sem numero carimbado por transecto, "
                               "so a serie visual do perfil"),
}
E.salvar(fig, "fig8", numeros, fontes)
