# -*- coding: utf-8 -*-
"""Per-transect MAE of six terrain products (glo30, fabdem, gedtm30, mlp, v23 and
ANADEM) against the lidar ground, over the same population and the same reference
(k0) as the evaluation script that produced `juiz_grade_nativa_diretas.json`
(`por_transecto_mae`). It supplies the ANADEM value for the profile figure
(strip NP_T-0224, `artigo_v23_2/fig/gerar_fig8.py`).

Population logic is reused by import (no new code):
  * `auditar_nmad_pareado_nativa_anadem.montar_populacao_com_anadem` -- mirror
    of `juiz_lidar_v4.main` (guards n_all/frac_casca, judgeable transects and
    clearing offsets as in that script, z_ref = z_solo_v2 + off_t, dedup by
    n_clareira, forest = dossel_v2 >= DOSSEL_FLORESTA and agua < AGUA_LIMPO),
    with the "anadem" column (laterais_nativa_diretas/anadem_amazonia_Q*.npz,
    key "DTM", as in gerar_fig8.py:60) added BEFORE the filters;
  * `auditar_nmad_pareado.guardas_populacao` -- the population reproduces
    tabela_regua_k0 of the evaluation script (exact n, NMAD to 1e-6);
  * `auditar_nmad_pareado_nativa_anadem.conferir_sha256_anadem`.
Per-transect MAE is the expression of `juiz_lidar_v4.py` l. 334-336 applied to
the product list plus ANADEM: float(np.abs(s[p] - s["z_ref"]).mean())
per transect group (pandas skips NaN; ANADEM's n is its count of finite cells).

Guard: the five MAE values stored in
`juiz_grade_nativa_diretas.json` -> por_transecto_mae["NP_T-0224"] are
reproduced to 1e-6 (as an extra check, so are those of every transect).
On failure the measured values are written with `passou = false` and the exit code is 2.

Usage:  python auditar_perfil_anadem_faixa.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import proveniencia as PROV                          # noqa: E402
import juiz_lidar_v4 as J                            # noqa: E402
import auditar_nmad_pareado as REF                   # noqa: E402
import auditar_nmad_pareado_nativa_anadem as ANA     # noqa: E402

SAT = Path("/trabalho/GNN_TOPO/SATELITES")
PARQUET = SAT / "lidar_eba" / "solo_lidar_30m_v3_nativa.parquet"
LATERAIS = SAT / "laterais_nativa_diretas"
RECORTE = SAT / "anadem"
JUIZ = RAIZ / "juiz_grade_nativa_diretas.json"
FIG8 = RAIZ / "artigo_v23_2" / "fig" / "gerar_fig8.py"
SAIDA = RAIZ / "auditar_perfil_anadem_faixa.json"
FAIXA = "NP_T-0224"
PROD5 = list(J.PRODUTOS)                  # ["v23","mlp","fabdem","gedtm30","glo30"]
PROD6 = PROD5 + ["anadem"]
TOL = 1e-6


def log(m=""):
    print(m, flush=True)


def main() -> int:
    for p in (PARQUET, LATERAIS / "_proveniencia.json", JUIZ, FIG8):
        if not p.exists():
            raise FileNotFoundError(p)
    J.LAT = LATERAIS
    ANA.conferir_sha256_anadem(LATERAIS, RECORTE)
    juiz = json.loads(JUIZ.read_text(encoding="utf-8"))

    flor, info = ANA.montar_populacao_com_anadem(PARQUET, juiz, LATERAIS)
    conf = REF.guardas_populacao(flor, juiz)
    log(f"populacao: {info['n_floresta_julgavel']} celulas de floresta julgada em "
        f"{info['n_julgaveis']} transectos; tabela_regua_k0 reproduzida em {len(conf)} "
        "celulas produto x faixa")

    # juiz_lidar_v4.py l. 334-336, with ANADEM added to the product list
    por_t = flor.groupby("transecto").apply(
        lambda s: pd.Series({p: float(np.abs(s[p] - s["z_ref"]).mean())
                             for p in PROD6}), include_groups=False)
    n_t = flor.groupby("transecto").agg(
        n_celulas=("z_ref", "size"),
        n_anadem_finito=("anadem", lambda x: int(np.isfinite(x).sum())))

    # Guard: the five stored MAEs reproduced to 1e-6 (profile strip and all transects)
    pm = juiz["por_transecto_mae"]
    difs = {}
    for t, linha in pm.items():
        if t not in por_t.index:
            raise RuntimeError(f"transecto {t} do juiz ausente da populacao")
        difs[t] = {p: abs(float(por_t.loc[t, p]) - float(linha[p])) for p in PROD5}
    if set(por_t.index) != set(pm):
        raise RuntimeError("transectos recomputados != chaves de por_transecto_mae")
    max_faixa = max(difs[FAIXA].values())
    max_todos = max(max(v.values()) for v in difs.values())
    guarda = bool(max_faixa <= TOL and max_todos <= TOL)
    log(f"guarda {FAIXA}: max |dif| = {max_faixa:.3e}; todos os transectos: "
        f"{max_todos:.3e} -> {'PASSOU' if guarda else 'FALHOU'}")

    tabela = {str(t): {**{p: float(por_t.loc[t, p]) for p in PROD6},
                       "n_celulas": int(n_t.loc[t, "n_celulas"]),
                       "n_celulas_anadem_finito": int(n_t.loc[t, "n_anadem_finito"])}
              for t in por_t.index}
    fx = tabela[FAIXA]
    ordem = sorted(PROD6, key=lambda p: fx[p])
    saida = {
        "tarefa": (("MAE of ANADEM on the profile strip (NP_T-0224) over the same population "
                    "and reference as the judge")),
        "faixa_do_perfil": FAIXA,
        "guarda_reproduz_juiz_1e-6": {
            "passou": guarda, "tolerancia_m": TOL,
            "max_abs_dif_faixa_m": max_faixa, "max_abs_dif_todos_transectos_m": max_todos,
            "dif_por_produto_faixa_m": difs[FAIXA],
            "fonte": "juiz_grade_nativa_diretas.json -> por_transecto_mae"},
        "resultado_faixa": {
            "anadem_mae_m": fx["anadem"],
            "mae_m_seis_produtos": {p: fx[p] for p in PROD6},
            "ordem_crescente_de_mae": ordem,
            "n_celulas_floresta_julgada": fx["n_celulas"],
            "n_celulas_anadem_finito": fx["n_celulas_anadem_finito"]},
        "por_transecto_mae_descritivo": tabela,
        "n_transectos_julgaveis": int(len(tabela)),
        "populacao": {
            "n_floresta_julgavel_total": info["n_floresta_julgavel"],
            "n_anadem_finito_total": int(np.isfinite(flor["anadem"]).sum()),
            "julgaveis": info["julgaveis"],
            "reproduz_tabela_regua_k0_celulas_produto_x_faixa": len(conf)},
        "metodo": ("mesma populacao e mesma regua (k0, sem correcao delta) dos outros cinco "
                   "produtos: auditar_nmad_pareado_nativa_anadem.montar_populacao_com_anadem "
                   "(espelho de juiz_lidar_v4.main; anadem acrescentado antes dos filtros) e "
                   "a expressao de juiz_lidar_v4.py l. 334-336 (MAE = media de |produto - "
                   "z_ref| por transecto, z_ref = z_solo_v2 + offset de clareira do "
                   "transecto); ANADEM de laterais_nativa_diretas/anadem_amazonia_Q*.npz, "
                   "chave DTM (gerar_fig8.py l. 60); celulas com ANADEM nao finito ficam fora "
                   "so da media do ANADEM"),
        "nota": ("descritivo por transecto: sem teste; o numero da faixa e o que a anotacao "
                 "do painel (a) da Fig. de perfil pode trazer ao lado dos outros cinco"),
    }
    fontes = [PARQUET, PARQUET.with_suffix(".json"), LATERAIS / "_proveniencia.json",
              *(LATERAIS / f"anadem_amazonia_{q}.npz" for q in J.QUADS), JUIZ, FIG8,
              RAIZ / "juiz_lidar_v4.py", RAIZ / "auditar_nmad_pareado.py",
              RAIZ / "auditar_nmad_pareado_nativa_anadem.py"]
    PROV.gravar(SAIDA, saida, fontes_lidas=fontes, script=__file__)
    log(f"{FAIXA}: " + " | ".join(f"{p} {fx[p]:.4f}" for p in PROD6)
        + f" | n {fx['n_celulas']} (anadem finito {fx['n_celulas_anadem_finito']})")
    log(f"-> {SAIDA.name}")
    return 0 if guarda else 2


if __name__ == "__main__":
    raise SystemExit(main())
