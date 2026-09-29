# -*- coding: utf-8 -*-
"""S7 — auditar_recursos_s7: computational cost of the adopted recipe
(b4s035), supporting the ACDSA paper's single-workstation claim (wall
time, resource consumption, and training strategy).

Field definitions:
  D1  wall time is scoped as `parede_b2_min` = START -> "===== B2
      CONCLUIDO =====" (a line present in both logs; training + dense
      prediction of the B2 stage). The later block of the graph log
      (materialization + poco_herdado, which also processes the other
      branch) is reported separately as `pos_b2_min`, with `tem_pos_b2`
      per family; guard G8 checks that both fields are recorded in both
      families.
  D2  `treino_s_soma_min` is read from quadrantes[*].treino.treino_s (the
      primary source for pure training time), per seed; the interval from
      the log header to post-scale-up becomes `semente_ponta_a_ponta_min`
      (includes graph loading and union); `carga_uniao_min` = header ->
      "[MEM] uniao fechada".
  D3  hardware is labeled `hardware_da_maquina_de_auditoria`, with an
      explicit caveat: there is no artifact from the original runs that
      attests which machine was used; the logs only record `dispositivo:
      cuda`. The host recorded in the JSON timestamps is retroactive and
      is reported as such, not as proof.
  D4  access to `_proveniencia` fails loud; required in both sources.
  D5  G7: a missing "[MEM] RSS=" line or "VRAM: pico" line -> abort.
  D6  per-seed fields are dictionaries {seed: value}, not lists.
  D7  `n_nos_uniao` is read from the "uniao: 4 quadrantes, N nos, M
      arestas" log line, guarded against equality with
      contagem_arestas.json.
  D8  the MLP date (no START/END in the log) is reported as
      `data_estimada_por_mtime` and is NOT treated as a citable fact.

GUARDS (fail loud)
  G1  'b4s035' tag in the header of both logs
  G2  5 seed headers in each log, matching the JSON's seeds
  G3  |vram_pico(JSON) - vram_pico(log)| <= 0.05 GB per family
  G4  epocas_teto identical across seeds and across families
  G5  parede_b2 > sum of the per-seed end-to-end times
  G6  timestamps never go backward (the parser does not cross midnight)
  G7  RSS= and "VRAM: pico" lines exist (absence is never treated as zero)
  G8  parede_b2 and pos_b2 recorded in both families
  G9  n_nos_uniao from the log == contagem_arestas.json:nos_uniao (all 5 lines)

Usage:  python auditar_recursos_s7.py [--saida auditar_recursos_s7.json]
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import socket
import sys
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import proveniencia as PROV          # noqa: E402

FAMILIAS = {
    "gatv2": {"log": RAIZ / "logs" / "ponto_s035.log",
              "json": RAIZ / "results" / "ofat_suavfina_b4s035.json",
              "braco": "gatv2_2_semobs"},
    "mlp":   {"log": RAIZ / "logs" / "baseline_mlp.log",
              "json": RAIZ / "results" / "baseline_mlp_b4s035.json",
              "braco": "mlp_semobs"},
}
ARESTAS = RAIZ / "contagem_arestas.json"
TAG = "b4s035"
ETAPAS = ("etapa1", "etapa2", "etapa3", "etapa4")
QUADS = ("Q1", "Q2", "Q3", "Q4")

RE_TS = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\]")
RE_INICIO = re.compile(r"INICIO (\d{2})/(\d{2})/(\d{4})\s+(\d{1,2}):(\d{2}):(\d{2})")
RE_FIM = re.compile(r"FIM (\d{2})/(\d{2})/(\d{4})\s+(\d{1,2}):(\d{2}):(\d{2})")
RE_SEED = re.compile(r"===== amazonia \| seed (\d+) \| (\S+) \|")
RE_RSS = re.compile(r"RSS=([\d.]+) GB")
RE_VRAM = re.compile(r"VRAM: pico ([\d.]+) GB alocado, ([\d.]+) GB reservado")
RE_UNIAO = re.compile(r"uniao: 4 quadrantes, ([\d,]+) nos, ([\d,]+) arestas")
B2_FIM = "===== B2 CONCLUIDO ====="


def log(m=""):
    print(m, flush=True)


def _seg(h, m, s):
    return int(h) * 3600 + int(m) * 60 + int(s)


def ler_log(p: Path, braco: str, nos_esperados: int) -> dict:
    linhas = p.read_text(encoding="utf-8", errors="replace").splitlines()
    if f"'{TAG}'" not in "\n".join(linhas[:5]):                    # G1
        raise RuntimeError(f"G1: {p.name} sem etiqueta '{TAG}' no cabecalho")
    inicio = fim = b2 = None
    inicio_origem = "linha INICIO"
    data = None
    data_origem = None
    seed_aberta = None            # (seed, header_ts)
    ponta_a_ponta: dict[int, float] = {}
    carga_uniao: dict[int, float] = {}
    rss_max = vram_max = vram_res_max = 0.0
    n_rss = n_vram = 0
    primeiro_ts = ultimo_ts = None
    nos_uniao_linhas = []
    for ln in linhas:
        m = RE_INICIO.search(ln)
        if m:
            data = f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
            data_origem = "linha INICIO do log"
            inicio = _seg(m.group(4), m.group(5), m.group(6))
            continue
        m = RE_FIM.search(ln)
        if m:
            fim = _seg(m.group(4), m.group(5), m.group(6))
            continue
        m = RE_TS.match(ln)
        if not m:
            continue
        ts = _seg(*m.groups())
        if ultimo_ts is not None and ts < ultimo_ts:                # G6
            raise RuntimeError(f"G6: {p.name}: timestamp retrocede ({ln[:12]})")
        if primeiro_ts is None:
            primeiro_ts = ts
        ultimo_ts = ts
        if B2_FIM in ln:
            b2 = ts
        ms = RE_SEED.search(ln)
        if ms:
            if ms.group(2) != braco:
                raise RuntimeError(f"G2: {p.name}: braco {ms.group(2)} != {braco}")
            seed_aberta = (int(ms.group(1)), ts)
            continue
        mu = RE_UNIAO.search(ln)
        if mu:
            nos_uniao_linhas.append(int(mu.group(1).replace(",", "")))
        if "uniao fechada" in ln and seed_aberta is not None:
            carga_uniao[seed_aberta[0]] = round((ts - seed_aberta[1]) / 60.0, 2)
        if "pos-ampliacao" in ln and seed_aberta is not None:
            ponta_a_ponta[seed_aberta[0]] = round((ts - seed_aberta[1]) / 60.0, 2)
            seed_aberta = None
        mr = RE_RSS.search(ln)
        if mr:
            n_rss += 1
            rss_max = max(rss_max, float(mr.group(1)))
        mv = RE_VRAM.search(ln)
        if mv:
            n_vram += 1
            vram_max = max(vram_max, float(mv.group(1)))
            vram_res_max = max(vram_res_max, float(mv.group(2)))
    if inicio is None:
        inicio, inicio_origem = primeiro_ts, "primeiro timestamp do log (sem linha INICIO)"
        mt = datetime.fromtimestamp(p.stat().st_mtime)
        data, data_origem = mt.strftime("%Y-%m-%d"), "mtime do arquivo (D8: estimativa, nao citavel)"
    if b2 is None:
        raise RuntimeError(f"{p.name}: sem linha '{B2_FIM}'")
    if n_rss == 0 or n_vram == 0:                                   # G7
        raise RuntimeError(f"G7: {p.name}: linhas RSS={n_rss}, VRAM pico={n_vram}")
    if len(ponta_a_ponta) != 5 or len(carga_uniao) != 5:            # G2
        raise RuntimeError(f"G2: {p.name}: {len(ponta_a_ponta)} sementes fechadas, "
                           f"{len(carga_uniao)} unioes (esperado 5)")
    if len(nos_uniao_linhas) != 5 or any(n != nos_esperados for n in nos_uniao_linhas):  # G9
        raise RuntimeError(f"G9: {p.name}: nos da uniao {nos_uniao_linhas} != {nos_esperados}")
    parede_b2 = (b2 - inicio) / 60.0
    if parede_b2 <= sum(ponta_a_ponta.values()):                    # G5
        raise RuntimeError(f"G5: {p.name}: parede_b2 {parede_b2:.1f} <= soma ponta-a-ponta")
    fim_ef = fim if fim is not None else ultimo_ts
    out = {"log": p.name,
           "parede_b2_min": round(parede_b2, 1),
           "parede_b2_escopo": "INICIO -> B2 CONCLUIDO: 5 sementes x 4 etapas, "
                               "carga, treino e predicao densa da fase B2",
           "inicio_origem": inicio_origem,
           "tem_pos_b2": fim is not None and fim_ef > b2,
           "pos_b2_min": round((fim_ef - b2) / 60.0, 1),
           "pos_b2_escopo": "B2 CONCLUIDO -> FIM: materializacao e poco_herdado "
                            "(processa tambem outro braco); NAO comparavel entre familias",
           "semente_ponta_a_ponta_min": ponta_a_ponta,
           "semente_ponta_a_ponta_escopo": "cabecalho da semente -> [MEM] pos-ampliacao "
                                           "(inclui carga e uniao do grafo)",
           "carga_uniao_min": carga_uniao,
           "sementes_no_log": sorted(ponta_a_ponta),
           "rss_pico_gb": rss_max, "n_linhas_rss": n_rss,
           "vram_pico_log_gb": vram_max, "vram_reservada_log_gb": vram_res_max,
           "n_nos_uniao_log": nos_uniao_linhas[0]}
    if data_origem == "linha INICIO do log":
        out["data"] = data
    else:
        out["data_estimada_por_mtime"] = data
    out["data_origem"] = data_origem
    return out


def ler_json(p: Path, braco: str) -> dict:
    d = json.loads(p.read_text(encoding="utf-8"))
    prov = d["_proveniencia"]                                       # D4: fails loud
    runs = {k: v for k, v in d.items()
            if k.startswith("amazonia_") and braco in k and TAG in k}
    if len(runs) != 5:
        raise RuntimeError(f"{p.name}: {len(runs)} corridas {braco}/{TAG}, esperado 5")
    epocas = None
    vram = {}
    vram_res = []
    usadas = {e: [] for e in ETAPAS}
    treino_s = {}
    for k, r in runs.items():
        seed = int(re.search(r"seed(\d+)", k).group(1))
        e = dict(r["epocas"])
        if epocas is None:
            epocas = e
        elif e != epocas:                                           # G4
            raise RuntimeError(f"G4: {p.name}: epocas divergem em {k}")
        soma = 0.0
        for i, q in enumerate(QUADS):
            qq = r["quadrantes"][q]
            vram[(seed, q)] = float(qq["vram_pico_gb"])
            vram_res.append(float(qq["vram_reservada_gb"]))
            usadas[ETAPAS[i]].append(int(qq["treino"]["epocas_usadas"]))
            soma += float(qq["treino"]["treino_s"])
        treino_s[seed] = round(soma / 60.0, 2)
    (seed_max, q_max), vmax = max(vram.items(), key=lambda kv: kv[1])
    tr = list(treino_s.values())
    return {"json": p.name, "sementes": sorted(treino_s),
            "epocas_teto": epocas,
            "epocas_usadas_media": {e: round(sum(v) / len(v), 1) for e, v in usadas.items()},
            "vram_pico_gb": vmax, "vram_pico_onde": f"seed {seed_max}, etapa {q_max}",
            "vram_reservada_max_gb": max(vram_res),
            "treino_s_soma_min": treino_s,
            "treino_s_soma_media_min": round(sum(tr) / len(tr), 2),
            "treino_s_escopo": "soma de quadrantes[*].treino.treino_s das 4 etapas, "
                               "por semente (treino puro, sem carga nem predicao)",
            "host_carimbo": prov["host"],
            "carimbo_retroativo": bool(prov.get("retroativo", False))}


def hardware_da_maquina_de_auditoria() -> dict:
    hw = {"host": socket.gethostname(), "cpu": platform.processor(),
          "os": platform.platform(), "lido_em": datetime.now().strftime("%Y-%m-%dT%H:%M")}
    try:
        import psutil
        hw["ram_total_gb"] = round(psutil.virtual_memory().total / 2**30)
    except Exception as e:                                       # noqa: BLE001
        raise RuntimeError(f"psutil indisponivel: {e}")
    import torch
    hw["gpu"] = torch.cuda.get_device_name(0)
    hw["vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    hw["torch"] = torch.__version__
    try:
        import subprocess
        nome = subprocess.run(["powershell", "-NoProfile", "-Command",
                               "(Get-CimInstance Win32_Processor).Name"],
                              capture_output=True, text=True, timeout=30).stdout.strip()
        if nome:
            hw["cpu"] = nome
    except Exception:                                            # noqa: BLE001
        pass
    hw["nota"] = ("D3: maquina em que ESTA auditoria rodou. Os logs das corridas "
                  "de 10-11/08 nao registram hardware (so 'dispositivo: cuda'); o "
                  "host do carimbo dos JSONs e retroativo (02/09). Declaracao do "
                  "autor, nao medida de artefato de corrida.")
    return hw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default="auditar_recursos_s7.json")
    a = ap.parse_args()

    arestas = json.loads(ARESTAS.read_text(encoding="utf-8"))
    nos_uniao = int(arestas["nos_uniao"])
    out = {"tag": TAG, "familias": {}, "n_nos_uniao": nos_uniao}
    epocas_ref = None
    hosts = {}
    for fam, cfg in FAMILIAS.items():
        for k in ("log", "json"):
            if not cfg[k].exists():
                raise FileNotFoundError(cfg[k])
        L = ler_log(cfg["log"], cfg["braco"], nos_uniao)
        J = ler_json(cfg["json"], cfg["braco"])
        if L["sementes_no_log"] != J["sementes"]:                    # G2
            raise RuntimeError(f"G2: {fam}: sementes log != json")
        if abs(J["vram_pico_gb"] - L["vram_pico_log_gb"]) > 0.05:    # G3
            raise RuntimeError(f"G3: {fam}: vram json {J['vram_pico_gb']} vs log "
                               f"{L['vram_pico_log_gb']}")
        if epocas_ref is None:
            epocas_ref = J["epocas_teto"]
        elif J["epocas_teto"] != epocas_ref:                         # G4
            raise RuntimeError("G4: epocas_teto diferem entre familias")
        hosts[fam] = {"host": J["host_carimbo"], "retroativo": J["carimbo_retroativo"]}
        out["familias"][fam] = {"braco": cfg["braco"], **L, **J}
        log(f"  {fam}: parede_b2 {L['parede_b2_min']} min | pos_b2 {L['pos_b2_min']} min "
            f"| treino puro/semente {J['treino_s_soma_media_min']} min | ponta-a-ponta "
            f"{round(sum(L['semente_ponta_a_ponta_min'].values())/5, 1)} min | VRAM pico "
            f"{J['vram_pico_gb']} GB ({J['vram_pico_onde']}) | RSS pico {L['rss_pico_gb']} GB")
    for fam, f in out["familias"].items():                          # G8
        if "parede_b2_min" not in f or "pos_b2_min" not in f:
            raise RuntimeError(f"G8: {fam} sem parede_b2/pos_b2")
    out["epocas_teto"] = epocas_ref
    out["host_carimbo_dos_jsons"] = hosts
    out["estrategia"] = ("janela crescente como uniao disjunta dos quadrantes num "
                         "unico grafo; amostragem por vizinhanca em mini-lotes "
                         "(NeighborLoader); arestas 8-conexas geradas da malha; "
                         "fila com guarda de RAM e orcamento de epocas travado por "
                         "etapa (b2_transferencia.py)")
    out["hardware_da_maquina_de_auditoria"] = hardware_da_maquina_de_auditoria()
    fontes = [cfg[k] for cfg in FAMILIAS.values() for k in ("log", "json")] + [ARESTAS]
    PROV.gravar(RAIZ / a.saida, out, fontes_lidas=fontes, script=__file__)
    log(f"gravado {a.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
