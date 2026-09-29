"""Fig. 1 — Study cell, reserves, airborne-lidar transects and canopy regime.

Single map (double column): the 2x2 degree cell (bbox from
geometria_celula.json), background = canopy height from the exogenous
product (ETH, dossel_eth_amazonia_Q*.npz), the four corner reserves
(layout_reservas.caixas_globais from the run JSONs) with the 67-cell buffer,
the 19 airborne EBA/INPE lidar transects (solo_lidar_30m_v2.parquet: global
row/col + transect), 13 kept and 6 excluded by the transect criterion
(juiz_lidar_v4.json), and the density of orbital anchors
(rotulos_amazonia_Q*.npz: local idx_no per quadrant). The canopy background
is the rationale for the cell choice (a canopy regime, not a biome).
Raw data sources (not hash-stamped) are recorded in fig1_numeros.json with
sha256; the caption numbers (area, separation, buffer, n anchors) come from
hash-stamped artifacts.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, BoundaryNorm
import estilo_jstars as E

E.aplicar()
from matplotlib import patheffects as PE
SAT = E.RAIZ.parent / "SATELITES"
G = E.RAIZ / "geometria_celula.json"
J = E.RAIZ / "juiz_lidar_v4.json"
R = E.RAIZ / "results" / "ofat_suavfina_b4s035.json"
SOLO = SAT / "lidar_eba" / "solo_lidar_30m_v2.parquet"
g = json.loads(G.read_text(encoding="utf-8"))
j = json.loads(J.read_text(encoding="utf-8"))
r = json.loads(R.read_text(encoding="utf-8"))
esc = next(v for k, v in r.items() if isinstance(v, dict) and v.get("layout_reservas"))
lay = esc["layout_reservas"]; caixas = lay["caixas_globais"]
buf = esc["quadrantes"]["Q1"]["particao"]["buffer_celulas"]
lon0, lat_s, lon1, lat_n = g["bbox"]           # [-60, -4, -58, -2]
N = 7200; Q = 3600
OFF = {"Q1": (0, 0), "Q2": (0, Q), "Q3": (Q, 0), "Q4": (Q, Q)}   # (row0, col0)

# ── background: ETH canopy height, mosaicked at 7200x7200 and downsampled by 8
S = 8
canopy = np.full((N // S, N // S), np.nan, dtype=np.float32)
n_anc = {}
for q, (r0, c0) in OFF.items():
    h = np.load(SAT / "laterais" / f"dossel_eth_amazonia_{q}.npz")["h_dossel"].reshape(Q, Q)
    hs = h[::S, ::S]
    canopy[r0 // S:(r0 + Q) // S, c0 // S:(c0 + Q) // S] = hs
    z = np.load(SAT / "laterais" / f"rotulos_amazonia_{q}.npz")
    n_anc[q] = int(len(z["idx_no"]))
# ── transectos (lin/col globais)
solo = pd.read_parquet(SOLO, columns=["lin", "col", "transecto"])
julg = {t for t, v in j["offsets_por_transecto"].items() if v["julgavel"]}
fora = set(j["fora_do_julgamento"].keys())

def cx(col): return lon0 + (col + 0.5) / N * (lon1 - lon0)
def cy(lin): return lat_n - (lin + 0.5) / N * (lat_n - lat_s)

fig = plt.figure(figsize=(E.COL2, 4.3))
gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.62, left=0.06, right=0.985, top=0.985, bottom=0.20)
ax = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[0, 1])
cmap = ListedColormap(["#f7f7f7", "#dbe9c8", "#a8cf8a", "#5aa554", "#1e6f3f"])
norm = BoundaryNorm([0, 3, 10, 20, 30, 60], cmap.N)      # >30 m: classe unica (r4)
cmap.set_over("#1e6f3f")                                  # canopy > 60 m falls in the ">30" class
im = ax.imshow(canopy, cmap=cmap, norm=norm, extent=[lon0, lon1, lat_s, lat_n],
               origin="upper", interpolation="nearest", zorder=1)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, ticks=[1.5, 6.5, 15, 25, 45])
cb.ax.set_yticklabels(["0–3", "3–10", "10–20", "20–30", ">30"])
cb.set_label("Canopy height class (m)", fontsize=7.5)
cb.ax.tick_params(labelsize=7.5)

# quadrants and held-out blocks
for q, (r0, c0) in OFF.items():
    ax.add_patch(Rectangle((cx(c0) - 0.5 / N * 2, cy(r0 + Q)), (lon1 - lon0) / 2, (lat_n - lat_s) / 2,
                           fill=False, ec="k", lw=0.5, ls=":", zorder=3))
    # quadrant label at the INNER corner (near the cell center), outside the reserve
    ri = r0 + (Q - 260 if r0 == 0 else 260); ci = c0 + (Q - 260 if c0 == 0 else 260)
    ax.text(cx(ci), cy(ri), q, ha="center", va="center", fontsize=9, color="k", alpha=0.7,
            fontweight="bold", zorder=4, bbox=dict(fc="white", ec="none", alpha=0.6, pad=1))
for q, (rr0, rr1, cc0, cc1) in caixas.items():
    # buffer
    ax.add_patch(Rectangle((cx(cc0 - buf), cy(rr1 + buf)), (cc1 - cc0 + 2 * buf) / N * (lon1 - lon0),
                           (rr1 - rr0 + 2 * buf) / N * (lat_n - lat_s), fill=False, ec=E.VERMELHO,
                           lw=0.6, ls="--", zorder=5))
    ax.add_patch(Rectangle((cx(cc0), cy(rr1)), (cc1 - cc0) / N * (lon1 - lon0), (rr1 - rr0) / N * (lat_n - lat_s),
                           fill=False, ec=E.VERMELHO, lw=1.2, zorder=6))
    ax.text(cx((cc0 + cc1) // 2), cy((rr0 + rr1) // 2), f"reserve\n{q}", ha="center", va="center",
            fontsize=7, color=E.VERMELHO, fontweight="bold", zorder=7,
            path_effects=[PE.withStroke(linewidth=1.8, foreground="white")])

# transects: STRIPS. The EBA airborne lidar footprint is a continuous strip
# of ~12 km x ~0.7 km per transect, ~8 km2 = ~8,000 cells of 30 m; at this
# map scale it would render as a thin line. The convex hull of each
# transect's cells is drawn as a filled polygon, and the measured dimensions
# are written to numeros.json.
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
COR_KEPT, COR_OUT = E.AMARELO, "#ffffff"      # solid Okabe-Ito yellow, thicker outline
dims = {}
for t, sub in solo.groupby("transecto"):
    kept = t in julg
    pts = np.column_stack([cx(sub["col"].values), cy(sub["lin"].values)])
    hull = ConvexHull(pts); poly = pts[hull.vertices]
    if kept:   # narrow strip (~0.7 km) -> thick yellow outline with a black halo, so the yellow stays visible
        ax.add_patch(Polygon(poly, closed=True, fc=COR_KEPT, ec=COR_KEPT, lw=1.6, zorder=8,
                             path_effects=[PE.withStroke(linewidth=2.8, foreground="k")]))
    else:
        ax.add_patch(Polygon(poly, closed=True, fc=COR_OUT, ec="k", lw=0.6, ls=(0, (2, 1.2)), zorder=8))
    # oriented dimensions (PCA in meters): length x width
    xy = np.column_stack([sub["col"].values * 30.0, sub["lin"].values * 30.0])
    xy = xy - xy.mean(0); w_, v_ = np.linalg.eigh(np.cov(xy.T)); pr = xy @ v_
    L = (pr[:, 1].max() - pr[:, 1].min() + 30) / 1000; W = (pr[:, 0].max() - pr[:, 0].min() + 30) / 1000
    dims[t] = {"mantido": bool(kept), "n_celulas": int(len(sub)), "area_km2": round(len(sub) * 0.0009, 2),
               "comprimento_km": round(L, 2), "largura_km": round(W, 2)}
L_med = float(np.median([d["comprimento_km"] for d in dims.values()]))
W_med = float(np.median([d["largura_km"] for d in dims.values()]))
A_med = float(np.median([d["area_km2"] for d in dims.values()]))
# manual legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
h = [Line2D([], [], color=E.VERMELHO, lw=1.2, label="Spatial reserve, 1 800 × 1 800 cells"),
     Line2D([], [], color=E.VERMELHO, lw=0.6, ls="--", label=f"Exclusion buffer, {buf} cells (~2.07 km)"),
     Patch(fc=COR_KEPT, ec="k", lw=0.9, label=f"Airborne lidar strip, kept by the transect criterion (n={len(julg)})"),
     Patch(fc=COR_OUT, ec="k", lw=0.6, ls=(0, (2, 1.2)), label=f"Airborne lidar strip, excluded (n={len(fora)})"),
     Line2D([], [], color="k", lw=0.5, ls=":", label="Quadrant boundary (Q1–Q4)")]
ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.66, -0.10), ncol=2, fontsize=7.5, frameon=False, columnspacing=0.8,
          handlelength=1.8)
ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
ax.set_xlim(lon0, lon1); ax.set_ylim(lat_s, lat_n)
ax.set_xticks(np.arange(lon0, lon1 + 0.01, 0.5)); ax.set_yticks(np.arange(lat_s, lat_n + 0.01, 0.5))
ax.tick_params(labelsize=7.5)
ax.set_aspect("equal")
sep_km = lay["separacao_minima_m"] / 1000
RODAPE_A = (
        f"2°×2° cell, {g['area_km2']:,.0f} km², 30 m posting; minimum reserve separation {sep_km:.1f} km ({lay['separacao_minima_celulas']} cells)\n"
        f"orbital anchors: {sum(n_anc.values()):,} candidate cells (GEDI L2A and ICESat-2 ATL08); "
        f"lidar strips: median {L_med:.1f} km × {W_med:.2f} km ({A_med:.1f} km² each)")
# this caption text is not rendered on the figure itself; it is exported to
# numeros.json for the LaTeX caption

# ── panel (b): distribution of canopy height classes over four populations
# (whole cell, union of reserves, kept lidar strips, candidate anchors).
# Derived from the SAME background rasters (not hash-stamped): values are
# written to numeros.json as "regime_por_populacao" and flagged as derived.
BINS = [0, 3, 10, 20, 30, 1e9]                      # single ">30" bin (previously split into 30-40 and >40)
CL_ROT = ["0–3", "3–10", "10–20", "20–30", ">30"]
CORES = ["#f7f7f7", "#dbe9c8", "#a8cf8a", "#5aa554", "#1e6f3f"]
def share(vals):
    vals = np.asarray(vals, float); vals = vals[np.isfinite(vals)]
    hst, _ = np.histogram(np.clip(vals, 0, 1e8), bins=BINS)
    return (hst / max(hst.sum(), 1)).tolist(), int(hst.sum())
pop = {}
full = np.full((N, N), np.nan, dtype=np.float32)
anc_vals = []
for q, (r0, c0) in OFF.items():
    hq = np.load(SAT / "laterais" / f"dossel_eth_amazonia_{q}.npz")["h_dossel"].reshape(Q, Q)
    full[r0:r0 + Q, c0:c0 + Q] = hq
    idx = np.load(SAT / "laterais" / f"rotulos_amazonia_{q}.npz")["idx_no"].astype(np.int64)
    anc_vals.append(hq.ravel()[idx])
pop["whole cell"] = share(full.ravel())
res_vals = np.concatenate([full[rr0:rr1, cc0:cc1].ravel() for (rr0, rr1, cc0, cc1) in caixas.values()])
pop["spatial reserves"] = share(res_vals)
sk = solo[solo["transecto"].isin(julg)]
pop["lidar strips (kept)"] = share(full[sk["lin"].values, sk["col"].values])
pop["orbital anchors"] = share(np.concatenate(anc_vals))
del full
nomes = list(pop.keys())
left = np.zeros(len(nomes))
for k in range(len(BINS) - 1):
    vals = np.array([pop[n][0][k] for n in nomes])
    axb.barh(np.arange(len(nomes)), vals, left=left, color=CORES[k], edgecolor="k", linewidth=0.4,
             height=0.62, label=CL_ROT[k])
    for yi, (v, l0) in enumerate(zip(vals, left)):
        if v >= 0.08:
            axb.text(l0 + v / 2, yi, f"{100*v:.0f}%", ha="center", va="center", fontsize=7.5,
                     color="white" if k >= 3 else "k")
    left += vals
axb.set_yticks(np.arange(len(nomes)))
axb.set_yticklabels([f"{n}\n(n = {pop[n][1]:,})" for n in nomes], fontsize=7.5)
axb.invert_yaxis()
axb.set_xlim(0, 1); axb.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); axb.set_xticklabels(["0", "25", "50", "75", "100"])
# n = cells with a canopy value (share() discards non-finite values)
axb.set_xlabel("Share by canopy height class (%)\nn = cells with a canopy value")
RODAPE_B = "colours as in the colour bar of (a)"   # not rendered on the figure; exported for the caption
axb.set_ylabel(""); axb.spines["left"].set_visible(False); axb.tick_params(axis="y", length=0)
# align panel (b) height with the map (equal aspect shrinks the map)
fig.canvas.draw()
pa, pb = ax.get_position(), axb.get_position()
axb.set_position([pb.x0, pa.y0, pb.width, pa.height])
E.rotular_paineis((ax, axb), x=-0.10)

numeros = {
    "regime_por_populacao": {n: {"share_por_classe": dict(zip(CL_ROT, pop[n][0])), "n_celulas": pop[n][1]} for n in nomes},
    "regime_nota": "derivado dos rasters de dossel ETH (nao carimbado); classes " + str(BINS[:-1]) + "+; anchors = candidate cells",
    "bbox": g["bbox"], "area_km2": g["area_km2"], "grade": g["grade"],
    "posting_centro_m": g["posting_centro"],
    "reservas_caixas_globais": caixas, "buffer_celulas": buf,
    "separacao_minima_celulas": lay["separacao_minima_celulas"], "separacao_minima_m": lay["separacao_minima_m"],
    "transectos_total": int(solo["transecto"].nunique()), "transectos_mantidos": len(julg),
    "transectos_dimensoes": dims, "faixa_mediana_km": {"comprimento": L_med, "largura": W_med, "area_km2": A_med},
    "transectos_excluidos": sorted(fora), "n_ancoras_candidatas_por_quadrante": n_anc,
    "fonte": {"geometria": "geometria_celula.json", "reservas": "results/ofat_suavfina_b4s035.json -> layout_reservas, quadrantes.Q1.particao.buffer_celulas",
              "juiz": "juiz_lidar_v4.json -> offsets_por_transecto[*].julgavel, fora_do_julgamento",
              "dossel": "SATELITES/laterais/dossel_eth_amazonia_Q*.npz (h_dossel, reduzido por 8 para o desenho)",
              "transectos": "SATELITES/lidar_eba/solo_lidar_30m_v2.parquet (lin, col, transecto)",
              "ancoras": ("SATELITES/laterais/rotulos_amazonia_Q*.npz (count of idx_no; candidates "
                          "before the quality filter)")},
    "rodape_removido_para_legenda": {"a": RODAPE_A, "b": RODAPE_B},
    "nota": ("the anchor count here is that of candidate cells in the label rasters; "
             "the article's citable number (157,847 in the union of reserves; 95.1% "
             "GEDI) comes from F8/F9"),
}
E.salvar(fig, "fig1", numeros, [G, J, R])
