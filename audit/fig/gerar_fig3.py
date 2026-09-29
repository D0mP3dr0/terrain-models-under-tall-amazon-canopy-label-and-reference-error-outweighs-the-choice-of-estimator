"""Fig. 4 — Three estimator families under the same cap on training epochs,
with physical admissibility beside every error figure.

(a) skill per seed (5 connected points) for the graph and pointwise networks at both
    epoch caps (x1 -> x4) at the tree-ensemble setting (ICESat-2 weight 4, smoothness 0),
    and the tree ensemble at the same setting; mean +- 95% CI (t, 4 df); the pre-specified
    contrast tree - graph at x4 is annotated (indeterminate). Canopy violations > 1 m per
    replicate are printed beneath the axis (tree value: mean over seeds, read from
    folha_de_fatos_jstars_v25.json, fact jstarsg_adm_arvore_resumo).
(b) per-quadrant reading of the tree - graph contrast at x4 (Q1..Q4), drawn as an
    amplitude, not a test (4 quadrants: sign-test floor p = 0.125).
Single source: auditar_d8_triangulo.json (por_semente, resumo_por_braco,
delta_T_xgb_menos_gatv2_fator4, por_quadrante, admissibilidade).
No "winner" arrow. Axis weights, the skill margin and the sign-test floor are read from
the artifact (config_d8, criterio_pre_declarado.margem, n of por_quadrante), not typed in.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import estilo_jstars as E

E.aplicar()
A = E.RAIZ / "auditar_d8_triangulo.json"
d = json.loads(A.read_text(encoding="utf-8"))
ps = d["por_semente"]; sementes = sorted(ps, key=int)
res = d["resumo_por_braco"]; adm = d["admissibilidade"]
dT = d["delta_T_xgb_menos_gatv2_fator4"]; pq = d["por_quadrante"]["xgb_menos_gatv2_fator4"]
cfg = d["guardas_verificadas"]["config_d8"]
# tree-ensemble admissibility, read from the fact sheet
FOLHA = E.RAIZ / "folha_de_fatos_jstars_v25.json"
_fatos = json.loads(FOLHA.read_text(encoding="utf-8"))["parte_G"]["fatos"]["s7_admissibilidade"]
adm_arv = next(f for f in _fatos if f["id"] == "jstarsg_adm_arvore_resumo")["valor"]
viol_arv = adm_arv["violacoes_acima_1m_media"]
boost = cfg["boost_atl08"]; suav = cfg["peso_suavidade"]
margem = d["criterio_pre_declarado"]["margem"]
n_res = sum(1 for k in pq if k.startswith("Q")); p_piso = 2 * 0.5 ** n_res            # two-sided sign test, all signs equal

def ic95(v):
    v = np.asarray(v, float); m = v.mean(); se = v.std(ddof=1) / np.sqrt(len(v))
    t = stats.t.ppf(0.975, len(v) - 1); return m, m - t*se, m + t*se

fig, (a, b) = plt.subplots(1, 2, figsize=(E.COL2, 2.9),
                           gridspec_kw=dict(width_ratios=[1.55, 1], wspace=0.35))

# (a) x positions: 0 = cap x1, 1 = cap x4; tree ensemble at x = 1.7
xs = {"x1": 0.0, "x4": 1.0, "xgb": 1.7}
series = {
    "gatv2": {"x1": [ps[s]["gatv2_e2"] for s in sementes], "x4": [ps[s]["gatv2_d8"] for s in sementes]},
    "mlp":   {"x1": [ps[s]["mlp_e2"] for s in sementes],   "x4": [ps[s]["mlp_d8"] for s in sementes]},
}
xgb = [ps[s]["xgb"] for s in sementes]
viol = {"gatv2": {"x1": adm["gatv2_e2_x1"]["violacoes_acima_1m_media"], "x4": adm["gatv2_d8_x4"]["violacoes_acima_1m_media"]},
        "mlp":   {"x1": adm["mlp_e2_x1"]["violacoes_acima_1m_media"],   "x4": adm["mlp_d8_x4"]["violacoes_acima_1m_media"]}}
def ms_viol(v):   # fixed size: admissibility is printed on its own line beneath the axis
    return 6

for fam in ("gatv2", "mlp"):
    F = E.FAM[fam]; dx = -0.045 if fam == "gatv2" else 0.045
    for s_i in range(len(sementes)):
        a.plot([xs["x1"] + dx, xs["x4"] + dx], [series[fam]["x1"][s_i], series[fam]["x4"][s_i]],
               color=F["cor"], alpha=0.35, lw=0.7, ls=F["ls"])
        a.plot([xs["x1"] + dx, xs["x4"] + dx], [series[fam]["x1"][s_i], series[fam]["x4"][s_i]],
               F["m"], color=F["cor"], alpha=0.35, ms=2.5, mfc="none")
    for cap in ("x1", "x4"):
        m, lo, hi = ic95(series[fam][cap])
        a.errorbar(xs[cap] + dx, m, yerr=[[m - lo], [hi - m]], fmt=F["m"], color=F["cor"],
                   ms=ms_viol(viol[fam][cap]), mec="k", mew=0.5, capsize=2.5, elinewidth=0.9,
                   label=F["nome"] if cap == "x1" else None, zorder=5)
# tree ensemble
m, lo, hi = ic95(xgb); F = E.FAM["xgb"]
a.plot([xs["xgb"]]*len(xgb), xgb, F["m"], color=F["cor"], alpha=0.35, ms=2.5, mfc="none")
a.errorbar(xs["xgb"], m, yerr=[[m - lo], [hi - m]], fmt=F["m"], color=F["cor"], ms=6, mec="k",
           mew=0.5, capsize=2.5, elinewidth=0.9, label=F["nome"], zorder=5)
# pre-specified contrast
a.annotate("", xy=(xs["xgb"] - 0.08, res["xgb"]["skill"]["media"]),
           xytext=(xs["x4"] + 0.12, res["gatv2_d8_x4"]["skill"]["media"]),
           arrowprops=dict(arrowstyle="<->", lw=0.7, color="k"))
a.text(0.80, 0.2995,   # placed clear of the tree-ensemble marker
       f"tree − graph at cap ×4: difference = {E.menos(format(dT['media'], '+.4f'))}, 95% CI [{E.menos(format(dT['ic95'][0], '+.4f'))}, {E.menos(format(dT['ic95'][1], '+.4f'))}]\n"
       "pre-specified rule: indeterminate (no ordering)",
       fontsize=7.5, ha="center", va="top")
# admissibility on its own line beneath the axis, next to every error figure
a.text(-0.78, -0.20, "canopy violations\n> 1 m per replicate:", transform=a.get_xaxis_transform(),
       fontsize=7.5, ha="left", va="top", color=E.CINZA)
for cap in ("x1", "x4"):
    a.text(xs[cap] - 0.05, -0.20, f"{viol['gatv2'][cap]:.0f}", transform=a.get_xaxis_transform(),
           fontsize=7.5, ha="right", va="top", color=E.FAM["gatv2"]["cor"], fontweight="bold")
    a.text(xs[cap] + 0.05, -0.20, f"{viol['mlp'][cap]:.0f}", transform=a.get_xaxis_transform(),
           fontsize=7.5, ha="left", va="top", color=E.FAM["mlp"]["cor"], fontweight="bold")
a.text(xs["xgb"], -0.20, f"{viol_arv:,.0f}", transform=a.get_xaxis_transform(),
       fontsize=7.5, ha="center", va="top", color=E.FAM["xgb"]["cor"], fontweight="bold")
a.set_xticks([xs["x1"], xs["x4"], xs["xgb"]])
a.set_xticklabels(["cap ×1", "cap ×4", "tree ensemble\n(own schedule)"])
a.set_xlim(-0.80, 2.05)
a.set_ylim(0.266, 0.300)
a.set_ylabel("Skill on the spatial reserves (5 seeds)")
# the tree ensemble's own schedule is not labelled a "cap" on the axis
a.set_xlabel(f"Configuration (ICESat-2 weight ×{boost:g}, smoothness weight {suav:g})", labelpad=31)
a.legend(loc="upper left", fontsize=7.5, bbox_to_anchor=(0.0, 0.90))

# (b) per quadrant
qs = ["Q1", "Q2", "Q3", "Q4"]
vals = [pq[q]["media"] for q in qs]
sig = [pq[q]["sinal_consistente"] for q in qs]
cols = [E.VERDE if v > 0 else E.AZUL for v in vals]
b.bar(np.arange(4), vals, 0.6, color=cols, edgecolor="k", linewidth=0.5)
b.axhline(0, color="k", lw=0.6)
b.axhspan(-margem, margem, color=E.CINZA, alpha=0.18, lw=0, label=f"±{margem:.3f} skill margin")
for i, (v, s) in enumerate(zip(vals, sig)):
    b.text(i, (v + 0.0015) if v >= 0 else 0.0015, E.menos(f"{v:+.4f}") + f"\n{s}/5", ha="center",
           va="bottom", fontsize=7.5)
b.set_xticks(np.arange(4)); b.set_xticklabels(qs)
b.set_ylabel("Tree ensemble minus graph network (skill), cap ×4")
b.set_xlabel("Spatial reserve")
b.set_ylim(-0.022, 0.047)   # keeps the margin legend clear of the Q3 label
b.legend(loc="upper left", fontsize=7.5)
b.text(0.98, 0.03, f"amplitude, not a test\n({n_res} reserves: sign-test floor p = {p_piso:.3f})",
       transform=b.transAxes, fontsize=7.5, ha="right", va="bottom", color=E.CINZA)
fig.subplots_adjust(left=0.075, right=0.995, top=0.95, bottom=0.26)
E.rotular_paineis((a, b), x=-0.13)

numeros = {
    "a_skill_por_semente": ps,
    "a_medias": {k: v["skill"]["media"] for k, v in res.items()},
    "a_violacoes_acima_1m_media": {k: adm[k]["violacoes_acima_1m_media"] for k in
                                   ("gatv2_e2_x1", "mlp_e2_x1", "gatv2_d8_x4", "mlp_d8_x4")},
    "a_violacoes_acima_1m_media_arvore": viol_arv,
    "a_admissibilidade_arvore_resumo": adm_arv,
    "a_delta_T": {k: dT[k] for k in ("media", "ic95", "p", "sinal_consistente")},
    "a_veredito": d["veredito"],
    "a_receita_eixo": {"boost_atl08": boost, "peso_suavidade": suav},
    "b_margem": margem, "b_piso_teste_sinal_p": p_piso, "b_n_reservas": n_res,
    "b_por_quadrante": {q: {"media": pq[q]["media"], "sinal": pq[q]["sinal_consistente"]} for q in qs},
    "fonte": "auditar_d8_triangulo.json -> por_semente, resumo_por_braco, admissibilidade, "
             "delta_T_xgb_menos_gatv2_fator4, por_quadrante.xgb_menos_gatv2_fator4; "
             "folha_de_fatos_jstars_v25.json -> parte_G.fatos.s7_admissibilidade[jstarsg_adm_arvore_resumo]",
}
E.salvar(fig, "fig3", numeros, [A, FOLHA])
