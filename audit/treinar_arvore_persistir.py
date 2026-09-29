"""S7 -- tree-model arm with PERSISTED PREDICTION over the whole grid.

WHY THIS EXISTS. `braco_arvores.py` only evaluates the tree model at the
anchors (validation, test, holdout) and never saved the Delta field.
Without that field, the tree model cannot enter the LiDAR comparison, and
the article's ceiling test (O) was left with only v23 x mlp. This script
produces the persisted prediction "in the same format and with the same
per-seed aggregation as the network predictions".

WHAT IT DOES (and nothing beyond that):
  1. Reads the config ALREADY CHOSEN in results/f2_arvores.json["config"] (no
     search) and the contract's boost setting; checks that the config
     matches f2_arvores_config.json.
  2. For each seed (42 123 7 2024 31) calls, by IMPORT and without editing,
     braco_arvores.carregar_tudo / treinar / avaliar.
  3. GUARD g1 (fail-loud): per-seed holdout skill reproduces
     f2_arvores.json["por_semente"][s]["reserva"]["skill_media_ponderada_por_n"]
     within |diff| <= 0.002 and the same holdout n. On failure -> STOPS
     (exit 2), no grid is produced.
  4. Predicts Delta at ALL nodes of every quadrant (dd["X"] from the SAME
     B2.preparar_peca, with the same carregar_tudo arguments) and writes
     delta_amazonia_arvore_f2_s7_seed{s}.npz (keys Q1..Q4, float32, Delta in
     meters; z' = glo30 - Delta) -- the same format used elsewhere for
     persisted surfaces -- with a .json sidecar (npz sha256 + per-quadrant
     diagnostics).
  5. GUARD g1b (fail-loud): the grid prediction at the holdout nodes matches
     the training-time prediction (max |diff| <= 1e-5 m), and the skill
     recomputed from the grid matches the evaluation-time skill to 1e-9.

WHAT IT DOES NOT DO: no hyperparameter search, no GPU use (requires
CUDA_VISIBLE_DEVICES=""), no edits to braco_arvores.py / b2_transferencia.py
/ results/*, no interpretation of the results.

USAGE
  CUDA_VISIBLE_DEVICES="" V23_LATDIR=/trabalho/GNN_TOPO/SATELITES/laterais \
  OMP_NUM_THREADS=12 /trabalho/ambientes/s33_amb_virtual/.venv/bin/python \
      treinar_arvore_persistir.py
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

LATDIR_ESPERADO = "/trabalho/GNN_TOPO/SATELITES/laterais"
if os.environ.get("V23_LATDIR") != LATDIR_ESPERADO:
    raise SystemExit(f"V23_LATDIR tem de ser {LATDIR_ESPERADO} (o diretorio "
                     f"de laterais do treino original); veio "
                     f"{os.environ.get('V23_LATDIR')!r}")
if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
    raise SystemExit('CUDA_VISIBLE_DEVICES="" e obrigatorio: este script nao '
                     "toca GPU de experimento")

import proveniencia as PROV          # noqa: E402
import braco_arvores as BA           # noqa: E402  (LIDO, NAO editado)
import b2_transferencia as B2        # noqa: E402  (LIDO, NAO editado)

F2_JSON = RAIZ / "results" / "f2_arvores.json"
F2_CFG = RAIZ / "f2_arvores_config.json"
SEEDS = [42, 123, 7, 2024, 31]
QUADS = BA.QUADS
TOL_G1 = 0.002
TOL_G1B_PRED = 1e-5
TOL_G1B_SKILL = 1e-9
PREFIXO = "delta_amazonia_arvore_f2_s7"
MANIFESTO = RAIZ / "treinar_arvore_persistir.json"


def log(m=""):
    print(time.strftime("[%H:%M:%S] ") + str(m), flush=True)


def caminho_npz(s: int) -> Path:
    return RAIZ / f"{PREFIXO}_seed{s}.npz"


def caminho_modelo(s: int) -> Path:
    return RAIZ / f"arvore_f2_s7_seed{s}.ubj"


def skill_de(preds: dict, ys: dict, labs: dict) -> float:
    """Same arithmetic as BA.avaliar(rot=3): per-quadrant skill against the
    quadrant's TRAINING label mean as the trivial baseline, n-weighted average."""
    maes, ns, skills = [], [], []
    for q in QUADS:
        lab, y = labs[q], ys[q]
        m = lab == 3
        if not m.any():
            continue
        p = preds[q]
        yy = y[m]
        c_triv = float(y[lab == 0].mean())
        mae = float(np.abs(p - yy).mean())
        triv = float(np.abs(yy - c_triv).mean())
        maes.append(mae); ns.append(int(m.sum()))
        skills.append(1.0 - mae / max(triv, 1e-9))
    return float(np.average(skills, weights=ns))


def fontes_laterais() -> list:
    lat = Path(LATDIR_ESPERADO)
    return sorted(p for p in lat.glob("*_amazonia_Q*.npz"))


def gravar_manifesto(corpo: dict, fontes: list) -> None:
    prov = PROV.bloco(__file__)
    prov["comando"] = " ".join([sys.executable] + sys.argv)
    prov["ambiente"] = {k: os.environ.get(k) for k in
                        ("V23_LATDIR", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")}
    saida = {"_proveniencia": prov, "_fontes": PROV.fontes(fontes), **corpo}
    MANIFESTO.write_text(json.dumps(saida, indent=2, ensure_ascii=False,
                                    default=float), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    a = ap.parse_args()
    import xgboost as xgb
    t0 = time.time()

    f2 = json.loads(F2_JSON.read_text(encoding="utf-8"))
    cfg = f2["config"]
    cfg_arq = json.loads(F2_CFG.read_text(encoding="utf-8"))["config"]
    if cfg != cfg_arq:
        raise RuntimeError(f"config de f2_arvores.json {cfg} != f2_arvores_config.json {cfg_arq}")
    boost = float(f2["contrato"]["boost"])
    log(f"config (sem busca) {cfg} | boost {boost:g} | nthread {BA.NTHREAD} | "
        f"tetos {BA.ARVORES_ETAPA}")

    fontes = [F2_JSON, F2_CFG, RAIZ / "braco_arvores.py", RAIZ / "b2_transferencia.py",
              RAIZ / "vetor_v23.py", RAIZ / "topo_v23.py", RAIZ / "grade_v23.py",
              RAIZ / "regime.py", Path(__file__)] + fontes_laterais()

    modelos, labs, ys, pred_res, g1 = {}, {}, {}, {}, {}
    # ── stage 1: per-seed training + guard g1
    for s in a.seeds:
        ts = time.time()
        pecas = BA.carregar_tudo(s, boost)
        m = BA.treinar(pecas, cfg, seed=s)
        r = {"validacao": BA.avaliar(m, pecas, 1),
             "teste": BA.avaliar(m, pecas, 2),
             "reserva": BA.avaliar(m, pecas, 3)}
        ref = f2["por_semente"][str(s)]
        dif = {k: r[k]["skill_media_ponderada_por_n"] - ref[k]["skill_media_ponderada_por_n"]
               for k in ("validacao", "teste", "reserva")}
        dif_mae = {k: r[k]["mae_media_ponderada_por_n_m"] - ref[k]["mae_media_ponderada_por_n_m"]
                   for k in ("validacao", "teste", "reserva")}
        n_ok = r["reserva"]["n"] == ref["reserva"]["n"]
        passou = bool(abs(dif["reserva"]) <= TOL_G1 and n_ok)
        g1[str(s)] = {"skill_reserva": r["reserva"]["skill_media_ponderada_por_n"],
                      "skill_reserva_f2": ref["reserva"]["skill_media_ponderada_por_n"],
                      "dif_skill_reserva": dif["reserva"],
                      "n_reserva": r["reserva"]["n"], "n_reserva_f2": ref["reserva"]["n"],
                      "dif_skill_validacao": dif["validacao"], "dif_skill_teste": dif["teste"],
                      "dif_mae_m": dif_mae, "arvores_finais": int(m.num_boosted_rounds()),
                      "avaliacao": r, "tolerancia": TOL_G1, "passou": passou,
                      "segundos": round(time.time() - ts, 1)}
        log(f"seed {s}: skill reserva {r['reserva']['skill_media_ponderada_por_n']:+.6f} "
            f"(f2 {ref['reserva']['skill_media_ponderada_por_n']:+.6f}, dif {dif['reserva']:+.2e}) "
            f"n {r['reserva']['n']} | arvores {m.num_boosted_rounds()} | "
            f"g1 {'PASSOU' if passou else 'FALHOU'} | {time.time()-ts:.0f} s")
        if not passou:
            gravar_manifesto({"status": "FALHOU_G1", "semente_que_falhou": s,
                              "config": cfg, "boost": boost, "g1": g1,
                              "nota": "guarda g1 falhou: nenhuma grade foi predita nem gravada"},
                             fontes)
            log(f"GUARDA g1 FALHOU na semente {s}: PARANDO (manifesto gravado)")
            return 2
        m.save_model(str(caminho_modelo(s)))
        modelos[s] = m
        labs[s] = {q: pecas[q]["lab"].copy() for q in QUADS}
        ys[s] = {q: pecas[q]["y"].copy() for q in QUADS}
        pred_res[s] = {q: m.predict(xgb.DMatrix(pecas[q]["X"][pecas[q]["lab"] == 3]))
                       for q in QUADS}
        del pecas
        gc.collect()

    # ── stage 2: prediction over the whole grid, quadrant outer loop, seed inner loop
    campos = {s: {} for s in a.seeds}
    diag = {s: {} for s in a.seeds}
    g1b = {str(s): {"max_abs_dif_pred_reserva_m": 0.0} for s in a.seeds}
    preds_grade_res = {s: {} for s in a.seeds}
    for q in QUADS:
        tq = time.time()
        dd, _lab, prep, _ = B2.preparar_peca(
            "amazonia", q, a.seeds[0], B2.BLOCO, "v23",
            True, True, False, False, False, True)      # same arguments as BA.carregar_tudo
        del prep, _lab
        gc.collect()
        X, idx = dd["X"], np.asarray(dd["idx"])
        n = int(dd["n_nodes"])
        if X.shape[0] != n:
            raise RuntimeError(f"{q}: X tem {X.shape[0]} linhas, n_nodes {n}")
        dsm = np.load(Path(LATDIR_ESPERADO) / f"glo30_amazonia_{q}.npz")["DEM"].astype(np.float32)
        dm = xgb.DMatrix(X)
        log(f"{q}: X {X.shape} carregado em {time.time()-tq:.0f} s")
        for s in a.seeds:
            if len(labs[s][q]) != len(idx):
                raise RuntimeError(f"{q}/seed {s}: lab {len(labs[s][q])} != idx {len(idx)}")
            tp = time.time()
            out = modelos[s].predict(dm).astype(np.float32)
            if out.shape != (n,):
                raise RuntimeError(f"{q}/seed {s}: predicao {out.shape} != ({n},)")
            pr = out[idx][labs[s][q] == 3]
            dmax = float(np.max(np.abs(pr - pred_res[s][q]))) if pr.size else 0.0
            g1b[str(s)]["max_abs_dif_pred_reserva_m"] = max(
                g1b[str(s)]["max_abs_dif_pred_reserva_m"], dmax)
            if dmax > TOL_G1B_PRED:
                raise RuntimeError(f"GUARDA g1b FALHOU {q}/seed {s}: predicao da grade nos nos "
                                   f"de reserva difere da do treino em {dmax} m (> {TOL_G1B_PRED})")
            preds_grade_res[s][q] = pr
            campos[s][q] = out
            z = dsm - out
            diag[s][q] = {
                "delta_mediano_m": round(float(np.median(out)), 4),
                "delta_p5_m": round(float(np.percentile(out, 5)), 4),
                "delta_p95_m": round(float(np.percentile(out, 95)), 4),
                "frac_delta_negativo_todas_celulas": round(float((out < 0).mean()), 6),
                "z_min_m": round(float(z.min()), 3),
                "z_max_m": round(float(z.max()), 3)}
            del z
            log(f"  {q} seed {s}: {time.time()-tp:.0f} s | Delta mediano {np.median(out):.3f} m "
                f"| max|dif| reserva {dmax:.2e}")
        del dd, X, dm, dsm
        gc.collect()

    for s in a.seeds:
        sk = skill_de(preds_grade_res[s], ys[s], labs[s])
        d = sk - g1[str(s)]["skill_reserva"]
        g1b[str(s)].update({"skill_reserva_da_grade": sk, "dif_vs_avaliar": d,
                            "passou": bool(abs(d) <= TOL_G1B_SKILL)})
        if abs(d) > TOL_G1B_SKILL:
            raise RuntimeError(f"GUARDA g1b FALHOU seed {s}: skill da grade {sk} != avaliar "
                               f"{g1[str(s)]['skill_reserva']} (dif {d})")
    log("guarda g1b: PASSOU nas 5 sementes")

    # ── stage 3: write npz + sidecar with sha256
    artefatos = {}
    for s in a.seeds:
        p = caminho_npz(s)
        np.savez(p, **{q: campos[s][q] for q in QUADS})
        h = PROV.sha256(p)
        side = {"bioma": "amazonia", "modo": "arvore_f2_s7", "seed": s, "arm": "xgboost_f2",
                "conteudo": "Delta em metros por quadrante; z' = glo30 - Delta",
                "modelo": caminho_modelo(s).name,
                "sha256_modelo": PROV.sha256(caminho_modelo(s)),
                "config": cfg, "boost": boost, "suavidade": 0.0,
                "npz": p.name, "sha256_npz": h, "bytes_npz": p.stat().st_size,
                "quadrantes": diag[s],
                "gerado_por": Path(__file__).name,
                "sha256_script": PROV.sha256(Path(__file__))}
        p.with_suffix(".json").write_text(json.dumps(side, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
        artefatos[str(s)] = {"npz": p.name, "sha256_npz": h, "sidecar": p.with_suffix(".json").name,
                             "modelo": caminho_modelo(s).name,
                             "sha256_modelo": side["sha256_modelo"]}
        log(f"-> {p.name} sha256 {h[:16]}... ({p.stat().st_size/2**20:.0f} MB)")

    sk = [g1[str(s)]["skill_reserva"] for s in a.seeds]
    gravar_manifesto({
        "status": "OK",
        "desenho": "artigo_v23_2/_pareceres_2026-09-28_rodada4/desenho_S5_S7_S8.md, secao S7",
        "config": cfg, "boost": boost, "suavidade": 0.0, "sementes": a.seeds,
        "nthread_xgboost": BA.NTHREAD,
        "g1": g1, "g1b": g1b,
        "resumo_reserva": {"skill_media": float(np.mean(sk)),
                           "skill_dp": float(np.std(sk, ddof=1)) if len(sk) > 1 else None,
                           "f2_skill_media": f2["resumo"]["skill_reserva_media"],
                           "f2_skill_dp": f2["resumo"]["skill_reserva_dp"]},
        "artefatos": artefatos,
        "formato": "igual a superficies/delta_amazonia_ampliacao_b4s035*_seed*.npz: chaves Q1..Q4, "
                   "float32, 12.960.000 nos (3600x3600), Delta em metros; agregacao entre sementes "
                   "e feita pelo auditor (media de 5, como juiz_lidar_v4.montar)",
        "segundos_total": round(time.time() - t0, 1),
    }, fontes)
    log(f"-> {MANIFESTO.name} ({time.time()-t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
