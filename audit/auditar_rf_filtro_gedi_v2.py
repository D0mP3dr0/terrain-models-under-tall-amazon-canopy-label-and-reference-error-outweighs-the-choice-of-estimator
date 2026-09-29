"""Descriptive complement to auditar_rf_filtro_gedi.py (v1); adds no new test.

Imports v1 and re-runs its full computation (V1.main) with in-memory hooks only:
the shot table and arm masks are captured, `granule` is attached to the GEDI
shots, and v1's PROV.gravar is replaced by a capture (v1 writes nothing). The
v2 JSON carries every v1 field unchanged plus three descriptive blocks:
  disparos_por_celula_ancora       F0-approved shots per anchor cell (GEDI, ATL08)
  passa_vs_nao_passa               F1/F2, common transects: A of cells that pass
                                   vs cells that do not, total and per transect
  causa_da_remocao_por_transecto   per transect: F0 GEDI shots, passing shots,
                                   rejections by degrade / sensitivity / both
                                   (F2: beam, night) and number of granules
Guard G4 (RuntimeError): every v1 JSON field except the provenance and status
fields (IGNORAR_G4) is reproduced identically; the sha256 of the v1 JSON is logged.
No criterion, arm, floor or verdict changes; v1 files are only read.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import proveniencia as PROV                  # noqa: E402
import auditar_rf_filtro_gedi as V1          # noqa: E402  (READ-ONLY, reuse by import)

V1_JSON = RAIZ / "auditar_rf_filtro_gedi.json"
V1_PY = RAIZ / "auditar_rf_filtro_gedi.py"
SAIDA = Path(os.environ.get("RF_V2_SAIDA", str(RAIZ / "auditar_rf_filtro_gedi_v2.json")))  # env only for dev runs
IGNORAR_G4 = {"_proveniencia", "_fontes", "contra_auditoria"}
log = V1.log

# ------------------------------------------------------------ hooks (memory only)
CAP: dict = {}
_orig_montar, _orig_mascaras = V1.montar_disparos, V1.mascaras_bracos
_orig_prep_gedi = V1.ROT.preparar_gedi


def _prep_gedi_com_granule(bioma, painel):
    g = _orig_prep_gedi(bioma, painel)
    raw = pd.read_parquet(V1.ROT.SAIDA / f"gedi02_a_{bioma}_dedup.parquet",
                          columns=["lat", "lon", "elev_solo", "elev_topo", "granule"])
    ok = (V1.ROT._valido(raw["elev_solo"].to_numpy(np.float64))
          & V1.ROT._valido(raw["elev_topo"].to_numpy(np.float64)))
    raw = raw[ok].reset_index(drop=True)
    if not (len(raw) == len(g)
            and np.array_equal(raw["lat"].to_numpy(np.float64), g["lat"].to_numpy())
            and np.array_equal(raw["lon"].to_numpy(np.float64), g["lon"].to_numpy())):
        raise RuntimeError("alinhamento granule x preparar_gedi falhou")
    g["granule"] = raw["granule"].to_numpy()
    return g


def _montar(*a, **k):
    d, dsm, info = _orig_montar(*a, **k)
    CAP["d"] = d
    return d, dsm, info


def _mascaras(d):
    m = _orig_mascaras(d)
    CAP["masks"] = m
    return m


class _ProvCaptura:
    sha256 = staticmethod(PROV.sha256)

    @staticmethod
    def gravar(saida, dados, fontes_lidas=(), script=None):
        CAP["dados"], CAP["fontes"] = dados, list(fontes_lidas)
        return Path(saida)


# ------------------------------------------------------------------- G4
def _igual(a, b, caminho, difs):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                difs.append(f"{caminho}/{k}: ausente em {'v2' if k not in a else 'v1'}")
            else:
                _igual(a[k], b[k], f"{caminho}/{k}", difs)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            difs.append(f"{caminho}: comprimento {len(a)} != {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                _igual(x, y, f"{caminho}[{i}]", difs)
    elif isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return
    elif a != b or type(a) is not type(b) and not (
            isinstance(a, (int, float)) and isinstance(b, (int, float))
            and not isinstance(a, bool) and not isinstance(b, bool)):
        difs.append(f"{caminho}: {a!r} != {b!r}")


def guarda_g4(dados: dict) -> dict:
    sha_v1 = PROV.sha256(V1_JSON)
    v1 = json.loads(V1_JSON.read_text(encoding="utf-8"))
    v2 = json.loads(json.dumps(dados, ensure_ascii=False, default=float))
    a = {k: v for k, v in v2.items() if k not in IGNORAR_G4}
    b = {k: v for k, v in v1.items() if k not in IGNORAR_G4}
    difs: list = []
    _igual(a, b, "", difs)
    g4 = {"sha256_json_v1": sha_v1, "campos_comparados": sorted(b),
          "ignorados": sorted(IGNORAR_G4), "n_diferencas": len(difs),
          "diferencas": difs[:50], "passou": not difs}
    log(f"  G4: sha256 v1 {sha_v1} | campos {len(b)} | diferencas {len(difs)}")
    if difs:
        raise RuntimeError(f"G4 FALHOU: {len(difs)} diferencas, primeiras {difs[:10]}")
    return g4


# ---------------------------------------------------- new descriptive blocks
def _dist(v: np.ndarray) -> dict:
    return {"0": int((v == 0).sum()), "1": int((v == 1).sum()),
            "2": int((v == 2).sum()), "3+": int((v >= 3).sum())}


def _a_stats(x: pd.DataFrame, col: str) -> dict:
    if len(x) == 0:
        return {"n": 0, "media_pond_celula_m": None, "mediana_m": None}
    return {"n": int(len(x)), "media_pond_celula_m": float(x[col].mean()),
            "mediana_m": float(x[col].median())}


def blocos(d: pd.DataFrame, masks: dict, dump: pd.DataFrame) -> dict:
    gedi = (d["fonte"] == 1).to_numpy()
    em = d["chave"].isin(set(dump["chave"])).to_numpy()
    info_cel = dump.set_index("chave")[["transecto", "classe_eth"]]
    sh = d.loc[em, ["chave", "solo_orto", "degrade_flag", "sensitivity",
                    "solar_elevation", "feixe", "granule"]].copy()
    sh["gedi"] = gedi[em]
    for b, m in masks.items():
        sh[b] = m[em]
    sh = sh.join(info_cel, on="chave")

    # medians per arm (same rule as v1: median of approved solo_orto)
    med = {b: sh[sh[b]].groupby("chave")["solo_orto"].median() for b in masks}
    cel = dump.copy()
    for b in masks:
        cel[f"med_{b}"] = cel["chave"].map(med[b]).to_numpy(np.float64)
    ng = sh[sh["F0"] & sh["gedi"]].groupby("chave").size()
    na = sh[sh["F0"] & ~sh["gedi"]].groupby("chave").size()
    cel["n_gedi_F0"] = cel["chave"].map(ng).fillna(0).astype(int)
    cel["n_atl08_F0"] = cel["chave"].map(na).fillna(0).astype(int)

    out1, out2, out3 = {}, {}, {}
    for c in V1.CLASSES:
        k = cel[cel["classe_eth"] == c]
        tot = (k["n_gedi_F0"] + k["n_atl08_F0"]).to_numpy()
        out1[c] = {"n_celulas": int(len(k)),
                   "disparos_totais_F0_por_celula": _dist(tot),
                   "gedi_F0_por_celula": _dist(k["n_gedi_F0"].to_numpy()),
                   "atl08_F0_por_celula": _dist(k["n_atl08_F0"].to_numpy()),
                   "celulas_com_atl08": int((k["n_atl08_F0"] > 0).sum()),
                   "celulas_so_atl08": int(((k["n_atl08_F0"] > 0) & (k["n_gedi_F0"] == 0)).sum())}

    for b in ("F1", "F2"):
        out2[b], out3[b] = {}, {}
        for c in V1.CLASSES:
            k = cel[cel["classe_eth"] == c].copy()
            k["passa"] = np.isfinite(k[f"med_{b}"].to_numpy())
            k["A_Fk"] = k["termo_rotulo"] + (k[f"med_{b}"] - k["med_F0"])
            comuns = sorted(k.loc[k["passa"], "transecto"].unique())
            kc = k[k["transecto"].isin(comuns)]
            p, n = kc[kc["passa"]], kc[~kc["passa"]]
            por_t = {}
            for t in comuns:
                pt, nt = p[p["transecto"] == t], n[n["transecto"] == t]
                por_t[t] = {"passam": {**_a_stats(pt, "termo_rotulo"),
                                       "A_Fk": _a_stats(pt, "A_Fk")},
                            "nao_passam": _a_stats(nt, "termo_rotulo")}
            out2[b][c] = {"transectos_comuns": comuns,
                          "transectos_sem_celula_no_braco": sorted(
                              set(k["transecto"]) - set(comuns)),
                          "passam": {**_a_stats(p, "termo_rotulo"), "A_Fk": _a_stats(p, "A_Fk")},
                          "nao_passam": _a_stats(n, "termo_rotulo"),
                          "por_transecto": por_t}

            s = sh[sh["F0"] & sh["gedi"] & (sh["classe_eth"] == c)]
            deg_ruim = s["degrade_flag"].to_numpy(np.float64) != 0
            thr = V1.F1_SENS if b == "F1" else V1.F2_SENS
            sen_ruim = ~(s["sensitivity"].to_numpy(np.float64) >= thr)
            if b == "F2":
                fei_ruim = ~s["feixe"].isin(V1.FEIXES_POTENCIA).to_numpy()
                noi_ruim = ~(s["solar_elevation"].to_numpy(np.float64) < 0)
            else:
                fei_ruim = noi_ruim = np.zeros(len(s), dtype=bool)
            s = s.assign(_deg=deg_ruim, _sen=sen_ruim, _fei=fei_ruim, _noi=noi_ruim)
            por_t = {}
            for t, g in s.groupby("transecto"):
                dg, sn, fe, no = (g[x].to_numpy() for x in ("_deg", "_sen", "_fei", "_noi"))
                outros_ok = ~fe & ~no
                r = {"n_gedi_F0": int(len(g)), "n_passam": int(g[b].sum()),
                     "reprovados_so_degrade_flag": int((dg & ~sn & outros_ok).sum()),
                     "reprovados_so_sensitivity": int((~dg & sn & outros_ok).sum()),
                     "reprovados_degrade_e_sensitivity": int((dg & sn & outros_ok).sum()),
                     "n_granules_F0": int(g["granule"].nunique()),
                     "n_granules_passam": int(g.loc[g[b], "granule"].nunique())}
                if b == "F2":
                    r.update({"reprovados_por_feixe": int(fe.sum()),
                              "reprovados_por_noite": int(no.sum()),
                              "reprovados_feixe_ou_noite_com_degrade_ou_sensitivity":
                                  int(((fe | no) & (dg | sn)).sum()),
                              "reprovados_so_feixe": int((fe & ~no & ~dg & ~sn).sum()),
                              "reprovados_so_noite": int((no & ~fe & ~dg & ~sn).sum()),
                              "reprovados_feixe_e_noite_so": int((fe & no & ~dg & ~sn).sum())})
                chk = r["n_gedi_F0"] - r["n_passam"]
                n_rep = int((dg | sn | fe | no).sum())
                if chk != n_rep:
                    raise RuntimeError(f"causa {b} {c} {t}: reprovados {chk} != condicoes {n_rep}")
                r["n_reprovados"] = chk
                soma = sum(v for kk, v in r.items() if kk.startswith("reprovados_")
                           and kk not in ("reprovados_por_feixe", "reprovados_por_noite"))
                if soma != chk:
                    raise RuntimeError(f"particao {b} {c} {t}: soma {soma} != {chk}")
                r["particao"] = ("as categorias reprovados_so_* , reprovados_degrade_e_sensitivity, "
                                 "reprovados_feixe_e_noite_so e reprovados_feixe_ou_noite_com_"
                                 "degrade_ou_sensitivity somam n_reprovados; reprovados_por_feixe "
                                 "e reprovados_por_noite sao marginais (sobrepoem-se)"
                                 if b == "F2" else
                                 "so_degrade + so_sensitivity + degrade_e_sensitivity = n_reprovados")
                por_t[t] = r
            out3[b][c] = por_t
    return {"disparos_por_celula_ancora": out1, "passa_vs_nao_passa": out2,
            "causa_da_remocao_por_transecto": out3}


def main() -> int:
    V1.montar_disparos, V1.mascaras_bracos = _montar, _mascaras
    V1.ROT.preparar_gedi = _prep_gedi_com_granule
    V1.PROV = _ProvCaptura
    sys.argv = [str(V1_PY), "--saida", str(SAIDA)]
    log("=== v1.main() (captura em memoria; o v1 nao grava nada) ===")
    rc = V1.main()
    if rc != 0 or "dados" not in CAP:
        raise RuntimeError(f"v1.main rc={rc}, captura {sorted(CAP)}")
    dados = CAP["dados"]
    log("=== G4 ===")
    g4 = guarda_g4(dados)

    dump = pd.read_parquet(V1.DUMP_E4)
    r_a, c_a = V1.chaves_ancora(dump)
    dump["chave"] = r_a * V1.ROT.GRID + c_a
    novos = blocos(CAP["d"], CAP["masks"], dump)
    for c in V1.CLASSES:
        log(f"  bloco1 {c}: {novos['disparos_por_celula_ancora'][c]}")
    for b in ("F1", "F2"):
        for c in V1.CLASSES:
            x = novos["passa_vs_nao_passa"][b][c]
            log(f"  bloco2 {b} {c}: passam {x['passam']['n']} "
                f"{x['passam']['media_pond_celula_m']} | nao passam {x['nao_passam']['n']} "
                f"{x['nao_passam']['media_pond_celula_m']} | comuns {x['transectos_comuns']}")
            for t, v in x["por_transecto"].items():
                log(f"      {t}: passam {v['passam']['n']} {v['passam']['media_pond_celula_m']} "
                    f"| nao passam {v['nao_passam']['n']} {v['nao_passam']['media_pond_celula_m']}")

    saida = {k: v for k, v in dados.items() if k != "contra_auditoria"}
    saida.update(novos)
    saida["guarda_G4_reproducao_v1"] = g4
    saida["v2_nota"] = (("descriptive complement requested by the independent check (its caveats "
                         "1, 3 and 4); no criterion, arm, floor or verdict changes; in the "
                         "passa_vs_nao_passa block the A of both groups is the adopted A (F0); for "
                         "those that pass, A_Fk is also given"))
    saida["contra_auditoria"] = (("v1 (auditar_rf_filtro_gedi.json) independently checked: CONFIRMED WITH "
                                  "CAVEAT. The new blocks (disparos_por_celula_ancora, passa_vs_nao_passa, "
                                  "causa_da_remocao_por_transecto) await verification."))
    fontes = CAP["fontes"] + [V1_JSON, V1_PY, Path(__file__)]
    PROV.gravar(SAIDA, saida, fontes_lidas=fontes, script=__file__)
    log(f"  -> {SAIDA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
