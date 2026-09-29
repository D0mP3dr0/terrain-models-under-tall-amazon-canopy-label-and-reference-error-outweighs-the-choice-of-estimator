# -*- coding: utf-8 -*-
"""S3 — auditar_paridade_capacidade: trainable-parameter count for the
adopted-recipe (b4s035) families, for the ACDSA paper's capacity-parity
table.

d_in for the historical B1 era is DERIVED from the B1 era's own artifact:
feat_nomes recorded per block (e.g. [topo8, comp10, regime4]) = 22
columns, plus n_obs observation channels (n_obs = d_in_modelo -
largura_X of the current manifest = 2, the same relation holds in both
eras), giving d_in_b1 = 24. Guard G4 checks that instantiating the model
at d_in_b1 REPRODUCES both B1-era params_milhoes figures within the
4-decimal rounding window (+-50 parameters), otherwise it aborts. Also
handled: the absence of optimizer state in the checkpoints; the real
justification for excluding sage2 from the reference block (a SAGE class
exists in b1_kriging_indutivo.py:221, but no sage2 checkpoint exists for
the b4s035 recipe); the ratio against B1 recorded with the precision of
its denominator.

The B1-era reference values are read from the JSONs at runtime (mlp from
results/b1_referencia.json; gatv2_2 from
results/b1_gatv2_bloco_bioma.json), with uniqueness checked across all
occurrences; both JSONs and manifesto_features_46.json are listed in
fontes_lidas.

TWO independent measurements, which MUST agree:
  M1  the REAL state_dict from the recipe's checkpoint (primary source),
      summing numel() only for tensors whose keys are TRAINABLE
      parameters in the instantiated class.
  M2  instantiating the b1_kriging_indutivo classes (MLP, GATv2) at
      d_in=46 (manifesto_features_46.json: d_in_modelo=46,
      confere_checkpoint=true — verified at runtime), summing p.numel()
      over parameters with requires_grad.
GUARD G1: M1 == M2 per family (trainable vs trainable), else fail loud.
GUARD G2: all 10 seeds on disk give the SAME M1 count.
GUARD G3: any state_dict key that matches neither a parameter nor a
      buffer of the instantiated class -> fail loud (the class does not
      match the checkpoint).

No GPU, no training: torch.load(map_location='cpu') plus module
construction.

Usage:  python auditar_paridade_capacidade.py [--saida auditar_paridade_capacidade.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import proveniencia as PROV          # noqa: E402

import torch                          # noqa: E402
from b1_kriging_indutivo import MLP, GATv2   # noqa: E402

CKPT = RAIZ / "checkpoints_b2"
SEEDS = (7, 31, 42, 123, 2024, 100, 200, 300, 400, 500)   # C17
SEEDS_RECEITA = (7, 31, 42, 123, 2024)
D_IN = 46
# d_in for the B1 era is DERIVED from the B1 era's own artifact at
# runtime: sum of the feat_nomes widths + n_obs; see derivar_d_in_era_b1()
JANELA_ARREDONDAMENTO = 50   # params_milhoes for the B1 era has 4 decimal places -> +-50
B1_REF = RAIZ / "results" / "b1_referencia.json"
B1_GATV2 = RAIZ / "results" / "b1_gatv2_bloco_bioma.json"
MANIFESTO = RAIZ / "manifesto_features_46.json"
FAMILIAS = {
    "gatv2_2_semobs": {
        "arquivo": "amazonia_seed{seed}_gatv2_2_semobs_v23_ampliacao_b4s035_ateQ4.pt",
        "constroi": lambda d_in: GATv2(d_in, 2),
        "era_b1": ("gatv2_2", B1_GATV2),
    },
    "mlp_semobs": {
        "arquivo": "amazonia_seed{seed}_mlp_semobs_v23_ampliacao_b4s035_ateQ4.pt",
        "constroi": lambda d_in: MLP(d_in),
        "era_b1": ("mlp", B1_REF),
    },
}


def log(m=""):
    print(m, flush=True)


def state_dict_de(caminho: Path) -> dict:
    obj = torch.load(caminho, map_location="cpu", weights_only=False)
    sd = obj.get("model") if isinstance(obj, dict) and "model" in obj else obj
    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
    if not hasattr(sd, "items"):
        raise RuntimeError(f"{caminho.name}: formato inesperado ({type(obj)})")
    sd = {k: v for k, v in sd.items() if torch.is_tensor(v)}
    if not sd:
        raise RuntimeError(f"{caminho.name}: nenhum tensor no state_dict")
    return sd


def derivar_d_in_era_b1(n_obs: int) -> tuple[int, dict]:
    """d_in for the B1 era comes from the B1 era's own artifact. feat_nomes
    recorded per block (e.g. ['topo8','comp10','regime4']) encodes the width
    in its numeric suffix; d_in = sum + n_obs (observation channels of the
    inductive kriging model, the same d_in = width + n_obs relation holds
    in both eras). Fails loud if feat_nomes diverges between blocks or
    does not end in a digit."""
    import re
    d = json.loads(B1_REF.read_text(encoding="utf-8"))
    listas = {k: tuple(v["feat_nomes"]) for k, v in d.items()
              if isinstance(v, dict) and "feat_nomes" in v}
    if not listas:
        raise RuntimeError(f"{B1_REF.name}: nenhum feat_nomes — nao da para "
                           f"derivar o d_in da era B1")
    if len(set(listas.values())) != 1:
        raise RuntimeError(f"{B1_REF.name}: feat_nomes divergente entre "
                           f"blocos: {sorted(set(listas.values()))}")
    nomes = next(iter(set(listas.values())))
    larguras = {}
    for nome in nomes:
        m = re.search(r"(\d+)$", nome)
        if not m:
            raise RuntimeError(f"{B1_REF.name}: feat_nomes contem {nome!r} "
                               f"sem sufixo numerico — derivacao impossivel")
        larguras[nome] = int(m.group(1))
    total = sum(larguras.values())
    return total + n_obs, {"feat_nomes": list(nomes), "larguras": larguras,
                           "colunas": total, "n_obs": n_obs,
                           "d_in_era_b1": total + n_obs}


def params_era_b1(braco: str, arq: Path) -> float:
    """The B1-era value is read from the JSON at runtime, with uniqueness
    checked across ALL occurrences of F_custo.params_milhoes for the
    branch."""
    d = json.loads(arq.read_text(encoding="utf-8"))
    achados = []

    def rec(x, p):
        if isinstance(x, dict):
            if braco in x and isinstance(x[braco], dict) \
                    and isinstance(x[braco].get("F_custo"), dict) \
                    and "params_milhoes" in x[braco]["F_custo"]:
                achados.append((p + "/" + braco,
                                float(x[braco]["F_custo"]["params_milhoes"])))
            for k, v in x.items():
                rec(v, p + "/" + k)
    rec(d, "")
    if not achados:
        raise RuntimeError(f"{arq.name}: nenhuma ocorrencia de "
                           f"{braco}/F_custo.params_milhoes")
    vals = {v for _, v in achados}
    if len(vals) != 1:
        raise RuntimeError(f"{arq.name}: {braco} com params_milhoes "
                           f"divergentes entre ocorrencias: {sorted(vals)}")
    return vals.pop()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default="auditar_paridade_capacidade.json")
    a = ap.parse_args()

    # manifest checked at runtime
    man = json.loads(MANIFESTO.read_text(encoding="utf-8"))
    if man.get("d_in_modelo") != D_IN or not man.get("confere_checkpoint"):
        raise RuntimeError(
            f"{MANIFESTO.name}: d_in_modelo={man.get('d_in_modelo')} / "
            f"confere_checkpoint={man.get('confere_checkpoint')} — esperado "
            f"{D_IN}/true; o manifesto nao sustenta a instanciacao")
    n_obs = int(man["d_in_modelo"]) - int(man["largura_X"])
    if n_obs <= 0:
        raise RuntimeError(f"{MANIFESTO.name}: n_obs={n_obs} — relacao "
                           f"d_in = largura + n_obs quebrada")
    d_in_b1, derivacao_b1 = derivar_d_in_era_b1(n_obs)

    fontes = [MANIFESTO, B1_REF, B1_GATV2]
    saida = {"d_in": D_IN, "receita": "v23_ampliacao_b4s035 (adotada)",
             "sementes_g2": list(SEEDS),
             "sementes_da_receita": list(SEEDS_RECEITA),
             "d_in_era_b1_derivado": derivacao_b1,          # v3/B1
             "familias": {},
             "nota_sage2": ("sage2 fora do bloco de referencia (C7; "
                            "justificativa corrigida na v3/R3): a classe "
                            "SAGE existe em b1_kriging_indutivo.py, mas NAO "
                            "existe checkpoint sage2 na receita b4s035 — sem "
                            "M1 nao ha dupla medicao; valor historico "
                            "permanece nos JSONs da era B1"),
             "nota_checkpoints": ("v3/R2: tensores_totais == treinaveis em "
                                  "todas as sementes porque os checkpoints "
                                  "b4s035 nao guardam estado de otimizador "
                                  "nem buffers — so pesos do modelo")}

    for nome, spec in FAMILIAS.items():
        # reference instantiation (M2) and trainable/buffer partition
        modelo = spec["constroi"](D_IN)
        nomes_treinaveis = {n for n, p in modelo.named_parameters()
                            if p.requires_grad}
        nomes_buffers = {n for n, _ in modelo.named_buffers()}
        m2 = sum(p.numel() for p in modelo.parameters() if p.requires_grad)

        # M1 across the 10 seeds on disk (G2), trainable only
        contagens, totais = {}, {}
        for seed in SEEDS:
            arq = CKPT / spec["arquivo"].format(seed=seed)
            if not arq.exists():
                raise FileNotFoundError(f"checkpoint ausente: {arq.name}")
            fontes.append(arq)
            sd = state_dict_de(arq)
            desconhecidas = set(sd) - nomes_treinaveis - nomes_buffers
            if desconhecidas:                                       # G3
                raise RuntimeError(
                    f"G3: {arq.name}: chaves do state_dict sem par na "
                    f"instanciacao: {sorted(desconhecidas)[:6]} — a classe "
                    f"instanciada NAO e a do checkpoint")
            faltantes = nomes_treinaveis - set(sd)
            if faltantes:                                           # G3
                raise RuntimeError(
                    f"G3: {arq.name}: parametros treinaveis ausentes do "
                    f"state_dict: {sorted(faltantes)[:6]}")
            contagens[seed] = sum(v.numel() for k, v in sd.items()
                                  if k in nomes_treinaveis)
            totais[seed] = sum(v.numel() for v in sd.values())
        if len(set(contagens.values())) != 1:                       # G2
            raise RuntimeError(f"G2: {nome} com contagens distintas por "
                               f"semente: {contagens}")
        m1 = next(iter(contagens.values()))
        if m1 != m2:                                                # G1
            raise RuntimeError(
                f"G1: {nome}: state_dict tem {m1:,} parametros treinaveis, "
                f"instanciacao em d_in={D_IN} tem {m2:,} — a classe "
                f"instanciada NAO e a do checkpoint; nada gravado")

        # divergence vs the B1 era, with the d_in delta MEASURED via M2 and
        # the B1-era d_in DERIVED from its artifact
        braco_b1, arq_b1 = spec["era_b1"]
        p_b1 = params_era_b1(braco_b1, arq_b1)          # in millions
        m2_b1din = sum(p.numel() for p in
                       spec["constroi"](d_in_b1).parameters()
                       if p.requires_grad)
        # G4: the instantiation at d_in_b1 must REPRODUCE the B1-era record
        # within the 4-decimal rounding window
        residuo = m2_b1din - int(round(p_b1 * 1e6))
        if abs(residuo) > JANELA_ARREDONDAMENTO:
            raise RuntimeError(
                f"G4: {nome}: instanciacao em d_in={d_in_b1} da {m2_b1din:,} "
                f"parametros, era B1 registrou {p_b1} M "
                f"({int(round(p_b1*1e6)):,}) — residuo {residuo:+,} fora da "
                f"janela +-{JANELA_ARREDONDAMENTO}; o d_in derivado nao "
                f"reproduz a era B1, a arquitetura pode ter mudado; nada "
                f"gravado")
        delta_din = m2 - m2_b1din
        divergencia = {
            "era_b1_params_milhoes": p_b1,
            "era_b1_braco": braco_b1,
            "era_b1_arquivo": arq_b1.name,
            # ratio with the precision of the denominator (4 decimal places)
            "razao_atual_sobre_b1": round(m1 / (p_b1 * 1e6), 4),
            "nota_razao": ("arredondada a 4 casas — o denominador da era B1 "
                           "foi gravado com 4 casas decimais"),
            "delta_por_d_in": int(delta_din),
            "nota_delta": (f"medido por instanciacao: M2(d_in={D_IN}) - "
                           f"M2(d_in={d_in_b1}); d_in da era B1 derivado do "
                           f"artefato (feat_nomes), nao de constante"),
            "residuo_vs_registro_b1": int(residuo),
            "origem_da_divergencia": (
                f"d_in explica integralmente: M2(d_in={d_in_b1}) reproduz o "
                f"registro da era B1 dentro do arredondamento de 4 casas "
                f"(residuo {residuo:+,}, janela +-{JANELA_ARREDONDAMENTO}); "
                f"a arquitetura nao mudou, so a largura de entrada"),
        }

        saida["familias"][nome] = {
            "params_treinaveis": m1, "params_milhoes": m1 / 1e6,
            "m1_treinaveis_por_semente": {str(s): c
                                          for s, c in contagens.items()},
            "tensores_totais_no_checkpoint_por_semente": {
                str(s): t for s, t in totais.items()},
            "m2_instanciacao": m2, "g1_m1_igual_m2": True,
            "buffers_na_instanciacao": sorted(nomes_buffers),
            "divergencia_vs_era_b1": divergencia,
        }
        log(f"{nome}: {m1:,} treinaveis ({m1/1e6:.4f} M) — M1==M2, "
            f"{len(SEEDS)} sementes identicas | vs B1 {p_b1} M: razao "
            f"{divergencia['razao_atual_sobre_b1']:.4f}, delta d_in "
            f"{delta_din:+,}, residuo {residuo:+,}")

    ps = {k: v["params_treinaveis"] for k, v in saida["familias"].items()}
    saida["leitura"] = {
        "razao_mlp_sobre_gatv2": ps["mlp_semobs"] / ps["gatv2_2_semobs"],
        "nota": ("razao > 1 significa que o baseline MLP tem MAIS parametros "
                 "que o GATv2 — favoravel ao claim de comparacao justa por "
                 "capacidade; numero descritivo, nao teste")}
    log(f"razao MLP/GATv2: {saida['leitura']['razao_mlp_sobre_gatv2']:.3f}")

    PROV.gravar(RAIZ / a.saida, saida, fontes_lidas=fontes, script=__file__)
    log(f"gravado: {a.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
