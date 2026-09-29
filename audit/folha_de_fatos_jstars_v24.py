# -*- coding: utf-8 -*-
"""folha_de_fatos_jstars_v24 -- fact sheet for the IEEE JSTARS article (V23
model): copies parts A-E from the v23 sheet, corrects one note, and adds part F.

Parts A-E are read in code from `folha_de_fatos_jstars_v23.json` (sha256
checked before copying) and reproduced unchanged, with ONE declared
exception: the `nota` of `jstarse_t10_skill_media_simples_figura` (value,
field, artifact and stamp untouched). The correction is recorded in
`parte_F.correcoes_sobre_v23`.

Part F: every fact is read in code, with id `jstarsf_*`, value, unit,
artifact, field and stamp (sha256 of the artifact at read time plus its own
_proveniencia when present). Facts computed from fields carry
`derivado: true` and `operacao` and are checked against a stored
counterpart when one exists (G4).

Guards (fail loudly):
  G1  unique ids across parts A-F.
  G2  no None/NaN in `valor`.
  G3  every source listed in `_fontes`; the v23 sha256 is checked against
      the pinned value before reading, and all sources are re-hashed at the
      end (none changed mid-run).
  G4  every computed value is checked against the stored field when present
      (mean/se/t/p/CI recomputed; TOST formula against the stored tost_p_*;
      counts against fig1_numeros.json; epoch caps against per-run records).
  GI  parts A-E identical to v23 except the corrected note (the only
      differing path), in the built object and in the JSON re-read from disk.

Usage:  python folha_de_fatos_jstars_v24.py [--saida folha_de_fatos_jstars_v24.json]
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import proveniencia as PROV          # noqa: E402

ART = RAIZ / "artigo_v23_2"
PAR4 = ART / "_pareceres_2026-09-28_rodada4"
FIGR4B = ART / "_propostas_2026-09-28" / "fig_r4b"
SAT = RAIZ.parent / "SATELITES" / "laterais"      # what gerar_fig1.py reads (E.RAIZ.parent/"SATELITES")
QS = ["Q1", "Q2", "Q3", "Q4"]

ARQS = {
    "folha_v23": RAIZ / "folha_de_fatos_jstars_v23.json",
    # T10 / Fig. 9 sources
    "f2_celula": RAIZ / "auditar_f2_celula.json",
    "unid_quad": RAIZ / "auditar_unidade_quadrante.json",
    "fig9_num": FIGR4B / "fig9_numeros.json",
    "main_r4b": ART / "manuscrito_r4b" / "main.tex",
    "fig9_pdf_r4b": ART / "manuscrito_r4b" / "fig9.pdf",
    "fig1_pdf_r4b": ART / "manuscrito_r4b" / "fig1.pdf",
    "manifesto_fig_r4b": FIGR4B / "SHA256_2026-09-28r4b.txt",
    "coer_r4b": PAR4 / "coerencia_r4b.md",
    # Fig. 1(b) sources
    "gerar_fig1": ART / "fig" / "gerar_fig1.py",
    "fig1_num": FIGR4B / "fig1_numeros.json",
    **{f"dossel_{q}": SAT / f"dossel_eth_amazonia_{q}.npz" for q in QS},
    **{f"rotulos_{q}": SAT / f"rotulos_amazonia_{q}.npz" for q in QS},
    # trees per stage / B2 epoch caps
    "braco_arvores": RAIZ / "braco_arvores.py",
    "f2_arvores": RAIZ / "results" / "f2_arvores.json",
    "b2": RAIZ / "b2_transferencia.py",
    "r_adot_gatv2": RAIZ / "results" / "ofat_suavfina_b4s035.json",
    "r_adot_mlp": RAIZ / "results" / "baseline_mlp_b4s035.json",
}

# pinned sha256 of the v23 fact sheet
SHA_ESPERADO = {
    "folha_v23": "1dbce07c1ec76ff8812bc1c28b737ae9886bae36a933f374fadd312ad8f9e768",
}

ID_CORRIGIDO = "jstarse_t10_skill_media_simples_figura"
NOTA_V24 = (
    "media simples entre os 4 quadrantes por semente (auditar_unidade_quadrante.py "
    "l.161, d_sem = M.mean(axis=0)). CORRECAO v24 (coerencia_r4b.md F1-i): NAO e a "
    "base da Fig. 9. Na Fig. 9 a unidade semente desenha a skill da uniao das "
    "reservas ponderada por n de ancoras (ponto adotado: auditar_f2_celula.json "
    "exploratorios_2x2_sem_pre_registro.ordenacao_intracelula_gatv2_menos_mlp."
    "'(100%, x1)', fato jstarsf_t10_skill_ponderada_adotado_ic95; lote 1536: "
    "auditar_f1_rampa.json delta_1536_reconferido). A media simples so aparece "
    "como o PONTO da unidade reserva: ela e igual a pares.*.por_quadrante.media "
    "(mesma grande media da matriz quadrante x semente; conferido G4), mas o IC "
    "desenhado na unidade reserva e o por quadrante (fatos jstarsf_t10_por_"
    "quadrante_*), nao o ic95 por semente deste fato. fig_r4b/fig9_numeros.json "
    "grava este bloco como por_semente_media_simples_NAO_desenhada.")


def log(m=""):
    print(m, flush=True)


def rot(p: Path) -> str:
    try:
        return p.relative_to(RAIZ).as_posix()
    except ValueError:
        return Path(os.path.relpath(p, RAIZ)).as_posix()


def _sem_nulos(x, trilha):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        raise RuntimeError(f"G2: valor nulo/NaN em {trilha}")
    if isinstance(x, dict):
        for kk, vv in x.items():
            _sem_nulos(vv, f"{trilha}.{kk}")
    elif isinstance(x, (list, tuple)):
        for i, vv in enumerate(x):
            _sem_nulos(vv, f"{trilha}[{i}]")


def sel(d, chaves):
    return {k: d[k] for k in chaves if k in d}


def igual(a, b, tol, rotulo):
    if abs(a - b) > tol:
        raise RuntimeError(f"G4: {rotulo}: {a} != {b} (tol {tol})")


def diff_trilhas(a, b, t=""):
    """List of paths where a and b differ (recursive; lists compared by index)."""
    if type(a) is not type(b):
        return [t or "<raiz>"]
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            tk = f"{t}.{k}" if t else str(k)
            if k not in a or k not in b:
                out.append(tk)
            else:
                out += diff_trilhas(a[k], b[k], tk)
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [t]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff_trilhas(x, y, f"{t}[{i}]")
        return out
    return [] if a == b else [t]


def tem_chave(o, pad):
    if isinstance(o, dict):
        return any(pad in str(k).lower() or tem_chave(v, pad) for k, v in o.items())
    if isinstance(o, list):
        return any(tem_chave(v, pad) for v in o)
    return False


def const_py(p: Path, nome: str):
    """Literal value of a module-level constant, read via AST (without importing)."""
    arv = ast.parse(p.read_text(encoding="utf-8"))
    for no in arv.body:
        if isinstance(no, ast.Assign):
            for alvo in no.targets:
                if isinstance(alvo, ast.Name) and alvo.id == nome:
                    return ast.literal_eval(no.value), no.lineno
    raise RuntimeError(f"constante {nome} nao encontrada em {p.name}")


def tost_casa(media, se, n, margem, alfa):
    """Paired TOST as in auditar_lote1536.py (l.123-126) and
    auditar_unidade_quadrante.py (l.122-123): two one-sided t tests, n-1 df, against +-margin."""
    gl = n - 1
    t_inf = (media + margem) / se
    t_sup = (media - margem) / se
    p_inf = float(1 - stats.t.cdf(t_inf, gl))
    p_sup = float(stats.t.cdf(t_sup, gl))
    return {"gl": gl, "t_inf": float(t_inf), "p_inf": p_inf,
            "t_sup": float(t_sup), "p_sup": p_sup,
            "equivalentes": bool(p_inf < alfa and p_sup < alfa)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default="folha_de_fatos_jstars_v24.json")
    a = ap.parse_args()

    for k, p in ARQS.items():
        if not p.exists():
            raise FileNotFoundError(f"fonte ausente: {k} -> {p}")

    # ------------------------------------------------------------------
    # G3: sha256 of every source BEFORE reading; pinned value checked
    # ------------------------------------------------------------------
    sha0 = {k: PROV.sha256(p) for k, p in ARQS.items()}
    for k, esp in SHA_ESPERADO.items():
        if sha0[k] != esp:
            raise RuntimeError(f"G3: {ARQS[k].name} mudou: sha256 atual="
                               f"{sha0[k]} esperado={esp}. Nao prossigo.")

    J = {k: json.loads(p.read_text(encoding="utf-8"))
         for k, p in ARQS.items() if p.suffix == ".json"}

    def carimbo_de(k):
        c = {"sha256": sha0[k]}
        if k in J and isinstance(J[k], dict) and isinstance(
                J[k].get("_proveniencia"), dict):
            pv = J[k]["_proveniencia"]
            c["script_origem"] = pv.get("script") or "AUSENTE (null no _proveniencia)"
            s = pv.get("sha256_script") or pv.get(
                "sha256_script_no_momento_do_carimbo")
            c["sha256_script"] = s or "AUSENTE"
            c["carimbado_em"] = pv.get("em") or "AUSENTE"
            c["retroativo"] = bool(pv.get("retroativo", False))
        elif k in J:
            c["script_origem"] = "AUSENTE (artefato sem _proveniencia)"
        elif ARQS[k].suffix == ".npz":
            c["script_origem"] = "AUSENTE (raster bruto .npz, nao carimbado)"
        else:
            c["script_origem"] = "nao_se_aplica (codigo/texto lido direto)"
        return c
    CARIMBO = {k: carimbo_de(k) for k in ARQS}

    # ------------------------------------------------------------------
    # Parts A-E copied from v23
    # ------------------------------------------------------------------
    v23 = J["folha_v23"]
    COPIADAS = ["parte_A", "parte_B", "parte_A_parte_B_fonte", "parte_C",
                "parte_D", "parte_E"]
    copia = {k: json.loads(json.dumps(v23[k])) for k in COPIADAS}
    for k in COPIADAS:                                           # GI (1a)
        if copia[k] != v23[k]:
            raise RuntimeError(f"GI: {k} copiada difere da v23")

    # correction: only the note changes
    alvo = [(i, f) for i, f in enumerate(copia["parte_E"]["fatos"]["t10"])
            if f["id"] == ID_CORRIGIDO]
    if len(alvo) != 1:
        raise RuntimeError(f"GI: {ID_CORRIGIDO} nao encontrado 1 vez em parte_E.t10")
    i_alvo, f_alvo = alvo[0]
    nota_v23 = f_alvo["nota"]
    f_alvo["nota"] = NOTA_V24
    TRILHA_NOTA = f"fatos.t10[{i_alvo}].nota"

    def conferir_GI(obj, rotulo):
        for k in COPIADAS:
            d = diff_trilhas(v23[k], obj[k])
            esperado = [TRILHA_NOTA] if k == "parte_E" else []
            if d != esperado:
                raise RuntimeError(f"GI ({rotulo}): {k} difere da v23 em {d} "
                                   f"(esperado {esperado})")
        # with the note reverted, parte_E must equal v23 again
        pe = json.loads(json.dumps(obj["parte_E"]))
        pe["fatos"]["t10"][i_alvo]["nota"] = nota_v23
        if pe != v23["parte_E"]:
            raise RuntimeError(f"GI ({rotulo}): parte_E sem a nota != v23")
    conferir_GI(copia, "objeto montado")                         # GI (1b)

    ids = set()
    for f in copia["parte_A"]["fatos"]:
        if f["id"] in ids:
            raise RuntimeError(f"G1: id repetido na v23: {f['id']}")
        ids.add(f["id"])
    for parte in ("parte_B", "parte_C", "parte_D", "parte_E"):
        for gf in copia[parte]["fatos"].values():
            for f in gf:
                if f["id"] in ids:
                    raise RuntimeError(f"G1: id repetido na v23: {f['id']}")
                ids.add(f["id"])
    n_ids_v23 = len(ids)
    if n_ids_v23 != v23["guardas"]["G1_ids_unicos"]["n_ids_total"]:
        raise RuntimeError("G1: contagem de ids da v23 != guarda gravada")

    grupos: dict[str, list] = {}

    def fato(grupo, id_, valor, unidade, chaves, campo, nota=None,
             operacao=None, extra=None):
        if not id_.startswith("jstarsf_"):
            raise RuntimeError(f"id fora do prefixo jstarsf_: {id_}")
        if id_ in ids:                                           # G1
            raise RuntimeError(f"G1: id duplicado: {id_}")
        ids.add(id_)
        _sem_nulos(valor, id_)                                   # G2
        if isinstance(chaves, str):
            chaves = [chaves]
        linha = {"id": id_, "valor": valor, "unidade": unidade,
                 "artefato": " + ".join(rot(ARQS[c]) for c in chaves),
                 "campo": campo,
                 "carimbo": {rot(ARQS[c]): CARIMBO[c] for c in chaves}}
        if operacao:
            linha["derivado"] = True
            linha["operacao"] = operacao
        if extra:
            linha.update(extra)
        if nota:
            linha["nota"] = nota
        grupos.setdefault(grupo, []).append(linha)
        return linha

    fatos_v23 = {f["id"]: f for gf in v23["parte_E"]["fatos"].values() for f in gf}
    TEX = ARQS["main_r4b"].read_text(encoding="utf-8").splitlines()

    def trecho(padroes, janela=8):
        """First window of main.tex lines that contains ALL the patterns."""
        for i in range(len(TEX)):
            bloco = "\n".join(TEX[i:i + janela])
            if all(p in bloco for p in padroes):
                j0 = next(j for j in range(i, i + janela) if padroes[0] in TEX[j])
                j1 = max(j for j in range(i, min(i + janela, len(TEX)))
                         if any(p in TEX[j] for p in padroes))
                return {"arquivo": rot(ARQS["main_r4b"]),
                        "sha256": sha0["main_r4b"],
                        "linhas": [j0 + 1, j1 + 1],
                        "texto": " ".join(s.strip() for s in TEX[j0:j1 + 1])}
        raise RuntimeError(f"trecho com {padroes} nao encontrado no main.tex")

    # ==================================================================
    # Manuscript <-> figure link (fig_r4b manifest)
    # ==================================================================
    man = {}
    for ln in ARQS["manifesto_fig_r4b"].read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([0-9a-f]{64})\s+(\S+)$", ln.strip())
        if m:
            man[m.group(2)] = m.group(1)
    for nome, k in (("fig9.pdf", "fig9_pdf_r4b"), ("fig1.pdf", "fig1_pdf_r4b"),
                    ("fig9_numeros.json", "fig9_num"), ("fig1_numeros.json", "fig1_num")):
        if man.get(nome) != sha0[k]:
            raise RuntimeError(f"G4: {nome} do manifesto fig_r4b != {rot(ARQS[k])}")

    # ==================================================================
    # F1 -- T10: n-weighted skill at the adopted point with 95% CI + derived TOST
    # ==================================================================
    f2 = J["f2_celula"]
    uq = J["unid_quad"]
    CEL = "(100%, x1)"
    pa = f2["exploratorios_2x2_sem_pre_registro"][
        "ordenacao_intracelula_gatv2_menos_mlp"][CEL]
    d = np.array([pa["por_semente"][str(s)] for s in pa["sementes"]], float)
    n = len(d)
    if n != pa["n"]:
        raise RuntimeError("G4: n != len(por_semente) em f2 (100%, x1)")
    md, dp = float(d.mean()), float(d.std(ddof=1))
    se = dp / math.sqrt(n)
    tc = stats.t.ppf(0.975, n - 1)
    t_ = md / se
    p_ = float(2 * stats.t.sf(abs(t_), n - 1))
    for rotulo, x, y in (("media", md, pa["media"]), ("dp", dp, pa["dp"]),
                         ("se", se, pa["se"]), ("t", t_, pa["t"]), ("p", p_, pa["p"]),
                         ("ic95_inf", md - tc * se, pa["ic95"][0]),
                         ("ic95_sup", md + tc * se, pa["ic95"][1])):
        igual(x, y, 1e-12, f"f2 (100%, x1) {rotulo} recomputado (t, n-1 gl)")
    # same weighted skill that v23 already records (point only) and that Fig. 9 plots
    pv23 = fatos_v23["jstarse_t10_skill_ponderada_por_n_texto"]["valor"]["ponto_adotado"]
    igual(pa["media"], pv23["media_gatv2_menos_mlp"], 1e-15,
          "f2 (100%, x1) media vs v23 jstarse_t10_skill_ponderada_por_n_texto")
    if {str(s): pa["por_semente"][str(s)] for s in pa["sementes"]} != pv23["por_semente"]:
        raise RuntimeError("G4: por_semente f2 (100%, x1) != v23 ponto_adotado.por_semente")
    f9 = J["fig9_num"]["numeros"]
    f9p = f9["pares"]["lote_adotado"]["por_semente_ponderada_por_n"]
    if f9p != sel(pa, ["n", "sementes", "media", "ic95", "sinal_consistente"]):
        raise RuntimeError("G4: fig9_numeros por_semente_ponderada_por_n != f2 (100%, x1)")
    if sorted(pa["sementes"]) != sorted(uq["pares"]["lote_adotado"]["sementes"]):
        raise RuntimeError("G4: sementes f2 (100%, x1) != unid_quad lote_adotado")

    tr_t = trecho(["+0.0010", "0.010", "passes"], janela=3)
    num_t = [float(x) for x in re.findall(r"[-+]?\d+\.\d+", tr_t["texto"].replace("$", ""))]
    if round(pa["media"], 4) not in num_t or 0.010 not in num_t:
        raise RuntimeError(f"G4: ponto ponderado/margem nao estao no trecho {tr_t}")
    fato("t10_ponderada_adotado", "jstarsf_t10_skill_ponderada_adotado_ic95",
         sel(pa, ["n", "sementes", "media", "dp", "se", "t", "p", "ic95",
                  "sinal_consistente", "por_semente"]),
         "skill (adim.)", "f2_celula",
         "exploratorios_2x2_sem_pre_registro.ordenacao_intracelula_gatv2_menos_mlp."
         "'(100%, x1)'.{n,sementes,media,dp,se,t,p,ic95,sinal_consistente,por_semente}",
         extra={"no_manuscrito": tr_t},
         nota="skill da uniao das reservas ponderada por n de ancoras, gatv2 - mlp, "
              "ponto adotado, unidade = semente (5); e o ponto e o IC desenhados na "
              "coluna 'unit = seed' da Fig. 9 (fig_r4b/fig9_numeros.json "
              "pares.lote_adotado.por_semente_ponderada_por_n, conferido ==). G4: "
              "media, dp, se, t, p e ic95 recomputados de por_semente (t, n-1 gl) a "
              "1e-12; media e por_semente == v23 jstarse_t10_skill_ponderada_por_n_"
              "texto.ponto_adotado.")

    if tem_chave(f2, "tost") or tem_chave(f2, "equival"):
        raise RuntimeError("f2 passou a ter TOST gravado: ler o campo, nao derivar")
    margem, alfa = uq["margem_skill"], uq["alfa"]
    # G4 on the formula: reproduces the tost_p_* stored in auditar_unidade_quadrante
    n_conf = 0
    for par, pp in uq["pares"].items():
        for un in ("por_semente", "por_quadrante"):
            r = pp[un]
            tt = tost_casa(r["media"], r["se"], r["n"], margem, alfa)
            igual(tt["p_inf"], r["tost_p_inf"], 1e-12, f"TOST p_inf {par}.{un}")
            igual(tt["p_sup"], r["tost_p_sup"], 1e-12, f"TOST p_sup {par}.{un}")
            if tt["equivalentes"] != r["equivalentes"]:
                raise RuntimeError(f"G4: TOST equivalentes {par}.{un}")
            n_conf += 1
    tost = tost_casa(pa["media"], pa["se"], pa["n"], margem, alfa)
    tc90 = stats.t.ppf(1 - alfa, n - 1)
    tost["ic90"] = [pa["media"] - tc90 * pa["se"], pa["media"] + tc90 * pa["se"]]
    tost["ic90_cabe_na_margem"] = bool(tost["ic90"][0] > -margem
                                       and tost["ic90"][1] < margem)
    if tost["ic90_cabe_na_margem"] != tost["equivalentes"]:
        raise RuntimeError("G4: IC90 dentro da margem != veredito TOST")
    tost = {"margem": margem, "alfa": alfa, "media": pa["media"],
            "se": pa["se"], "n": pa["n"], **tost}
    fato("t10_ponderada_adotado", "jstarsf_t10_tost_ponderado_adotado", tost,
         "skill (adim.) / p", ["f2_celula", "unid_quad"],
         "auditar_f2_celula.json ...'(100%, x1)'.{media,se,n}; margem e alfa: "
         "auditar_unidade_quadrante.json {margem_skill, alfa}",
         extra={"no_manuscrito": tr_t},
         operacao="TOST NAO gravado em auditar_f2_celula.json (nenhuma chave "
                  "'tost'/'equival' no JSON; conferido por codigo): derivado com a "
                  "formula da casa (auditar_lote1536.py l.123-126): t_inf = (media + "
                  "margem)/se, p_inf = 1 - T_{n-1}(t_inf); t_sup = (media - margem)/se, "
                  "p_sup = T_{n-1}(t_sup); equivalentes = p_inf < alfa e p_sup < alfa; "
                  "IC90 = media +- t_{1-alfa, n-1} se. G4: a mesma funcao reproduz os "
                  f"{n_conf} pares tost_p_inf/tost_p_sup/equivalentes gravados em "
                  "auditar_unidade_quadrante.json pares.*.{por_semente,por_quadrante} "
                  "a 1e-12; IC90 dentro da margem == veredito TOST.",
         nota="margem 0,010 = margem_skill de auditar_unidade_quadrante.json (a mesma "
              "de auditar_lote1536.py e auditoria_equivalencia_10s.py); a 'margem' de "
              "auditar_f2_celula.criterio_pre_declarado e de outro criterio "
              "(degradacao Delta_frac) e NAO foi usada. fig_r4b/fig9_numeros.json "
              "registra tost_ponderado_carimbado.lote_adotado = nao carimbado.")

    # ==================================================================
    # F2 -- T10: per quadrant (text: -0.032 to +0.039; +0.0034; p = 0.23)
    # ==================================================================
    tr_q = trecho(["0.0034", "0.032", "0.039", "0.23"])
    CH_Q = ["unidade", "n", "media", "dp", "se", "ic95", "t", "p", "tost_p_inf",
            "tost_p_sup", "equivalentes", "ic_cabe_na_margem"]
    for par in ("lote_adotado", "lote_1536"):
        pq = uq["pares"][par]["por_quadrante"]
        if f9["pares"][par]["por_quadrante"] != pq:
            raise RuntimeError(f"G4: fig9_numeros {par}.por_quadrante != unid_quad")
        # the simple mean (corrected fact) == point of the hold-out unit
        igual(uq["pares"][par]["por_semente"]["media"], pq["media"], 1e-15,
              f"{par}: media simples por semente vs por_quadrante.media")
        igual(fatos_v23[ID_CORRIGIDO]["valor"][par]["media"], pq["media"], 1e-15,
              f"{par}: v23 {ID_CORRIGIDO} media vs por_quadrante.media")
    qa = uq["pares"]["lote_adotado"]["por_quadrante"]
    q15 = uq["pares"]["lote_1536"]["por_quadrante"]
    # G4 field identification: rounded as in the text
    num_txt = [float(x) for x in re.findall(r"[-+]?\d+\.\d+", tr_q["texto"].replace("$", ""))]
    for rotulo, v, casas in (("ic95 inf lote_adotado", qa["ic95"][0], 3),
                             ("ic95 sup lote_adotado", qa["ic95"][1], 3),
                             ("media lote_adotado", qa["media"], 4),
                             ("p lote_1536", q15["p"], 2)):
        if round(v, casas) not in num_txt:
            raise RuntimeError(f"G4: {rotulo} = {v} arredondado a {casas} casas "
                               f"nao esta no trecho {tr_q['linhas']}: {num_txt}")
    fato("t10_por_quadrante", "jstarsf_t10_por_quadrante_lote_adotado",
         sel(qa, CH_Q), "skill (adim.) / p", "unid_quad",
         "pares.lote_adotado.por_quadrante.{" + ",".join(CH_Q) + "}",
         extra={"no_manuscrito": tr_q},
         nota="unidade = reserva espacial (quadrante, n = 4), media simples entre as 4 "
              "reservas das medias por semente. No texto r4b: '+0.0034' = media e "
              "'-0.032 to +0.039' = ic95 (t, 3 gl). Mesmo bloco que a Fig. 9 desenha na "
              "coluna 'unit = spatial reserve' (fig_r4b/fig9_numeros.json, conferido ==).")
    fato("t10_por_quadrante", "jstarsf_t10_por_quadrante_lote_1536",
         sel(q15, CH_Q), "skill (adim.) / p", "unid_quad",
         "pares.lote_1536.por_quadrante.{" + ",".join(CH_Q) + "}",
         extra={"no_manuscrito": tr_q},
         nota="unidade = reserva espacial (quadrante, n = 4), lote 1536 (10 sementes). "
              "No texto r4b: 'p = 0.23 by quadrant' = p (teste t bilateral de media "
              "zero, 3 gl). Mesmo bloco da Fig. 9 (conferido ==).")
    fato("t10_por_quadrante", "jstarsf_t10_delta_por_quadrante",
         {par: {"delta_por_quadrante": uq["pares"][par]["delta_por_quadrante"],
                "sinal_por_quadrante": uq["pares"][par]["sinal_por_quadrante"]}
          for par in ("lote_adotado", "lote_1536")},
         "skill (adim.)", "unid_quad",
         "pares.{lote_adotado,lote_1536}.{delta_por_quadrante,sinal_por_quadrante}",
         nota="valores por reserva desenhados ao lado na Fig. 9 ('its individual values "
              "are shown beside it'); base de 'the sign reverses between reserves'.")

    # ==================================================================
    # F3 -- Fig. 1(b): number of cells with an exogenous canopy value
    # ==================================================================
    N, Q = 7200, 3600
    OFF = {"Q1": (0, 0), "Q2": (0, Q), "Q3": (Q, 0), "Q4": (Q, Q)}
    BINS = [0, 3, 10, 20, 30, 1e9]                   # as in gerar_fig1.py

    def n_share(vals):                               # = share() of gerar_fig1.py
        vals = np.asarray(vals, float)
        vals = vals[np.isfinite(vals)]
        hst, _ = np.histogram(np.clip(vals, 0, 1e8), bins=BINS)
        return int(hst.sum())
    n_cel = {"total": 0, "finitos": 0, "n_share": 0}
    n_anc = {"candidatas": 0, "finitos": 0, "n_share": 0}
    por_q = {}
    for q in QS:
        h = np.load(ARQS[f"dossel_{q}"])["h_dossel"].reshape(Q, Q)
        idx = np.load(ARQS[f"rotulos_{q}"])["idx_no"].astype(np.int64)
        hv = h.ravel()
        av = hv[idx]
        pq_ = {"celulas_total": int(hv.size), "celulas_com_dossel": int(np.isfinite(hv).sum()),
               "ancoras_candidatas": int(idx.size),
               "ancoras_com_dossel": int(np.isfinite(av).sum())}
        por_q[q] = pq_
        n_cel["total"] += pq_["celulas_total"]
        n_cel["finitos"] += pq_["celulas_com_dossel"]
        n_cel["n_share"] += n_share(hv)
        n_anc["candidatas"] += pq_["ancoras_candidatas"]
        n_anc["finitos"] += pq_["ancoras_com_dossel"]
        n_anc["n_share"] += n_share(av)
        del h, hv, av, idx
    if n_cel["total"] != N * N:
        raise RuntimeError("G4: soma de celulas dos 4 quadrantes != 7200x7200")
    f1 = J["fig1_num"]["numeros"]
    rp = f1["regime_por_populacao"]
    for rotulo, x, y in (("celula inteira finitos vs share()", n_cel["finitos"], n_cel["n_share"]),
                         ("ancoras finitos vs share()", n_anc["finitos"], n_anc["n_share"]),
                         ("celula inteira vs fig1_numeros", n_cel["finitos"], rp["whole cell"]["n_celulas"]),
                         ("ancoras vs fig1_numeros", n_anc["finitos"], rp["orbital anchors"]["n_celulas"]),
                         ("candidatas vs fig1_numeros",
                          n_anc["candidatas"], sum(f1["n_ancoras_candidatas_por_quadrante"].values()))):
        if x != y:
            raise RuntimeError(f"G4: Fig. 1b {rotulo}: {x} != {y}")
    for q in QS:
        if por_q[q]["ancoras_candidatas"] != f1["n_ancoras_candidatas_por_quadrante"][q]:
            raise RuntimeError(f"G4: Fig. 1b candidates {q} != fig1_numeros")
    ch_d = [f"dossel_{q}" for q in QS]
    ch_r = [f"rotulos_{q}" for q in QS]
    op_n = ("contagem em codigo dos rasters que gerar_fig1.py le (l.158-172): n = "
            "celulas com h_dossel finito (= soma do histograma de share(), que descarta "
            "nao-finitos; conferido ==); G4: == fig_r4b/fig1_numeros.json "
            "regime_por_populacao.<pop>.n_celulas; fig1_numeros.json e fig1.pdf do "
            "manuscrito_r4b conferidos contra o manifesto SHA256_2026-09-28r4b.txt")
    nota_n = ("fig_r4b/fig1_numeros.json nao carimba os rasters (fontes = geometria, "
              "juiz, results; regime_nota: 'derivado dos rasters de dossel ETH (nao "
              "carimbado)'); por isso o n foi contado dos .npz, com sha256 de cada um.")
    fato("fig1b_n", "jstarsf_fig1b_n_celula_inteira",
         {"n_celulas_com_dossel": n_cel["finitos"], "celulas_total": n_cel["total"],
          "celulas_sem_dossel": n_cel["total"] - n_cel["finitos"],
          "por_quadrante": {q: sel(por_q[q], ["celulas_total", "celulas_com_dossel"]) for q in QS}},
         "celulas", ch_d + ["gerar_fig1", "fig1_num", "manifesto_fig_r4b", "fig1_pdf_r4b"],
         "SATELITES/laterais/dossel_eth_amazonia_Q*.npz h_dossel (3600x3600 por quadrante)",
         operacao=op_n, nota="populacao 'whole cell' do painel (b) da Fig. 1. " + nota_n)
    fato("fig1b_n", "jstarsf_fig1b_n_ancoras",
         {"n_ancoras_com_dossel": n_anc["finitos"], "ancoras_candidatas": n_anc["candidatas"],
          "ancoras_sem_dossel": n_anc["candidatas"] - n_anc["finitos"],
          "por_quadrante": {q: sel(por_q[q], ["ancoras_candidatas", "ancoras_com_dossel"]) for q in QS}},
         "celulas", ch_d + ch_r + ["gerar_fig1", "fig1_num", "manifesto_fig_r4b", "fig1_pdf_r4b"],
         ("SATELITES/laterais/rotulos_amazonia_Q*.npz idx_no (candidate anchors) "
          "indexing dossel_eth_amazonia_Q*.npz h_dossel"),
         operacao=op_n,
         nota=("population 'orbital anchors' of panel (b) of Fig. 1: CANDIDATE cells "
               "(before the quality filter) with a canopy value; this is not the citable "
               "anchor n of the article (gerar_fig1.py note: 157,847 in the union of the "
               "reserves comes from F8/F9). ") + nota_n)

    # ==================================================================
    # F4 -- trees per stage and B2 epoch caps
    # ==================================================================
    arv, l_arv = const_py(ARQS["braco_arvores"], "ARVORES_ETAPA")
    ep, l_ep = const_py(ARQS["b2"], "ETAPAS_EPOCAS")
    cur = J["f2_arvores"]["contrato"]["curriculo"]
    m = re.search(r"\[([\d,\s]+)\]", cur)
    arv_json = [int(x) for x in m.group(1).split(",")] if m else None
    if arv_json != arv:
        raise RuntimeError(f"G4: f2_arvores contrato.curriculo {arv_json} != ARVORES_ETAPA {arv}")
    runs_ep = {}
    for k in ("r_adot_gatv2", "r_adot_mlp"):
        for run, r in J[k].items():
            if isinstance(r, dict) and "epocas" in r and "etapas" in r:
                reg = [r["epocas"][f"etapa{i + 1}"] for i in range(len(r["epocas"]))]
                tetos = [e["epocas_teto"] for e in r["etapas"]]
                if reg != ep or tetos != ep:
                    raise RuntimeError(f"G4: {run} epocas {reg}/{tetos} != ETAPAS_EPOCAS {ep}")
                runs_ep[f"{rot(ARQS[k])}:{run}"] = reg
    if len(runs_ep) < 10:
        raise RuntimeError(f"G4: esperava 10 corridas (5 gatv2 + 5 mlp), achei {len(runs_ep)}")
    razao = [x / y for x, y in zip(arv, ep)]
    fato("arvores_epocas", "jstarsf_b2_tetos_epocas_por_etapa",
         {"tetos_epocas": ep, "etapas": len(ep), "corridas_conferidas": len(runs_ep)},
         "epocas", ["b2", "r_adot_gatv2", "r_adot_mlp"],
         f"b2_transferencia.py ETAPAS_EPOCAS (l.{l_ep}, lido por AST); results/"
         "{ofat_suavfina_b4s035,baseline_mlp_b4s035}.json <run>.epocas.etapa1..4 e "
         "<run>.etapas[*].epocas_teto",
         nota="tetos por etapa do curriculo do B2 (a parada antecipada decide as epocas "
              "usadas). G4: == o registrado em cada corrida do ponto adotado (5 gatv2 + "
              "5 mlp). Com --fator-epocas (x4) os tetos sao multiplicados; o ponto "
              "adotado e x1.")
    fato("arvores_epocas", "jstarsf_arvores_esquema_por_etapa",
         {"arvores_por_etapa": arv, "tetos_epocas_b2": ep, "razao_arvores_por_epoca": razao},
         "arvores", ["braco_arvores", "f2_arvores", "b2"],
         f"braco_arvores.py ARVORES_ETAPA (l.{l_arv}, lido por AST) == results/"
         "f2_arvores.json contrato.curriculo (lista extraida do texto 'continuacao do "
         "ensemble, tetos [...]')",
         operacao="razao_arvores_por_epoca = ARVORES_ETAPA / ETAPAS_EPOCAS, elemento a "
                  "elemento; G4: lista do .py == lista do contrato gravado",
         nota="comentario em braco_arvores.py l.68 e docstring l.29-30: 'tetos de arvores "
              "por etapa, na proporcao dos tetos de epoca do B2 (45/25/18/14)'. "
              "results/f2_arvores.json tem carimbo retroativo e sem sha256 do script; o "
              "sha256 de braco_arvores.py e o do momento desta leitura.")

    # ==================================================================
    # recorded correction + source quality
    # ==================================================================
    correcoes = [{
        "id": ID_CORRIGIDO, "parte": "parte_E", "trilha": f"parte_E.{TRILHA_NOTA}",
        "campo_alterado": "nota (somente; valor, campo, artefato e carimbo intactos)",
        "nota_v23": nota_v23, "nota_v24": NOTA_V24,
        "origem": {"arquivo": rot(ARQS["coer_r4b"]), "sha256": sha0["coer_r4b"],
                   "item": "F1 (i)"},
        "conferido_por_codigo": [
            "fig_r4b/fig9_numeros.json pares.lote_adotado.por_semente_ponderada_por_n == "
            "auditar_f2_celula.json '(100%, x1)' (unidade semente da Fig. 9)",
            "pares.{lote_adotado,lote_1536}.por_semente.media == por_quadrante.media "
            "(1e-15), e == valor.<par>.media deste fato",
            "fig_r4b/fig9_numeros.json pares.*.por_quadrante == auditar_unidade_quadrante",
            "manuscrito_r4b/fig9.pdf e fig9_numeros.json == manifesto fig_r4b",
        ],
    }]
    qual = {}
    for k in ARQS:
        if k == "folha_v23":
            continue
        c = CARIMBO[k]
        qual[rot(ARQS[k])] = {"sha256": c["sha256"], "script_origem": c.get("script_origem"),
                              "sha256_script": c.get("sha256_script", "nao_se_aplica"),
                              "retroativo": c.get("retroativo", False)}
    qualidade = {
        "fontes": qual,
        "notas": [
            "auditar_f2_celula.json nao tem TOST: jstarsf_t10_tost_ponderado_adotado e "
            "derivado (formula da casa conferida contra os tost_p_* de "
            "auditar_unidade_quadrante.json)",
            "auditar_unidade_quadrante.json: carimbo anterior ao campo sha256_script "
            "(so commit)",
            "n da Fig. 1(b): rasters SATELITES/laterais/*.npz nao carimbados (brutos); "
            "fig1_numeros.json nao os lista nas fontes; contagem feita aqui com sha256",
            "results/f2_arvores.json: carimbo retroativo, sem sha256 do script",
            "tetos de epoca lidos do codigo (b2_transferencia.py) e conferidos contra o "
            "registro por corrida dos results do ponto adotado",
        ],
    }

    # ==================================================================
    # G3 (end): no source changed during the run
    # ==================================================================
    for k, p in ARQS.items():
        if PROV.sha256(p) != sha0[k]:
            raise RuntimeError(f"G3: {p.name} mudou durante a corrida")

    n_fatos_f = sum(len(v) for v in grupos.values())
    saida = {
        "gerado_para": v23["gerado_para"],
        "regra": v23["regra"],
        "protocolo": (("Design of checks S5, S7 and S8; parts A-E = "
                       "folha_de_fatos_jstars_v23.json (sha256 ")
                      + SHA_ESPERADO["folha_v23"][:8] + "...), identicas exceto a "
                      "nota de " + ID_CORRIGIDO),
        "protocolo_v23": v23["protocolo"],
        "protocolo_v22": v23["protocolo_v22"],
        **copia,
        "parte_F": {
            "n_fatos": n_fatos_f,
            "grupos": {g: len(v) for g, v in grupos.items()},
            "correcoes_sobre_v23": correcoes,
            "qualidade_das_fontes": qualidade,
            "fatos": grupos,
        },
        "guardas": {
            "GI_partes_A_E_identicas_a_v23_exceto_nota": {
                "unica_trilha_divergente": f"parte_E.{TRILHA_NOTA}", "passou": True},
            "G1_ids_unicos": {"n_ids_v23": n_ids_v23, "n_ids_total": len(ids)},
            "G2_sem_nulos": True, "G3_sha256_conferidos": True,
            "G4_derivados_conferidos": True,
        },
        "guardas_v23": v23["guardas"],
    }
    saida_p = RAIZ / a.saida
    PROV.gravar(saida_p, saida, fontes_lidas=list(ARQS.values()), script=__file__)

    # GI (2): re-read from disk and compare with v23 re-read from disk
    relido = json.loads(saida_p.read_text(encoding="utf-8"))
    v23_disco = json.loads(ARQS["folha_v23"].read_text(encoding="utf-8"))
    if v23_disco != v23:
        raise RuntimeError("GI: v23 no disco mudou durante a corrida")
    conferir_GI(relido, "JSON relido do disco")
    log(f"GI ok: {', '.join(COPIADAS)} == v23 exceto parte_E.{TRILHA_NOTA} (apos releitura)")
    log(f"ids: {n_ids_v23} da v23 + {len(ids) - n_ids_v23} novos = {len(ids)}")
    log(f"parte_F: {n_fatos_f} fatos em {len(grupos)} grupos -> {a.saida}")
    for g, v in grupos.items():
        log(f"  {g}: {', '.join(f['id'] for f in v)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
