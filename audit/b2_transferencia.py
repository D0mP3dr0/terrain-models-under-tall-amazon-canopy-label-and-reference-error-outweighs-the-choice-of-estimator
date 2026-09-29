"""V23-B2: chained transfer Q1->Q4 with a per-quadrant reserve never seen during training.

PURPOSE
-------
Instead of training each quadrant in isolation (b1_kriging_indutivo.py), the next
quadrant INHERITS the model state of the previous one, so that by Q4 the model carries
the morphology of four regions instead of memorizing one.

Each quadrant also holds out a reserve that the model never sees at any point during
training.

WHAT "NEVER SEEN" REQUIRES UNDER THIS PROTOCOL
-----------------------------------------
In inductive kriging the label of the observed anchors enters as an INPUT CHANNEL,
not only as a target. Excluding the reserve only from the target would still leave it
visible through the input. Four simultaneous guarantees are required, each checked by
an assertion:

  G1  the reserve is never a training target
  G2  the reserve's label never enters obs_val/obs_flg (the observation channel)
  G3  the reserve is never used for early stopping or model selection
  G4  the scale of the observation channel (`esc`) is measured ONLY on the training
      observations

On G4: `elev_std` comes from the edge_attr/delta-x ratio of the DEM, a FEATURE
statistic, so it does not leak; `esc` is the mean of |y|, a label statistic, and would
leak if computed over the whole set.

What is NOT guaranteed, and should not be: the reserve nodes' FEATURES are visible
through message passing. The protocol is transductive in features and inductive in
label -- which is the correct definition of inductive kriging. What the buffer
guarantees is that no training node reaches a reserve LABEL: with 2 layers the
receptive field is 2 hops (~62 m), against a ~2,072 m buffer: a ~33x margin.

RESERVE GEOMETRY
--------------------
Layout on each 3600x3600-cell quadrant (30.922 m cells):

  - unit: a contiguous 1800x1800-cell sub-quadrant (~55.7 x 55.7 km) = 1/4 of the area
  - LITERAL layout in code, not randomized:  Q1->NW  Q2->NE  Q3->SW  Q4->SE
  - a 67-cell (~2,072 m) buffer eroded on the TRAINING side; those nodes are discarded

The corner COINCIDES with the quadrant's position within the biome (Q1=NW, Q2=NE,
Q3=SW, Q4=SE, verified identical across all four biomes). This puts the four reserves
at the four extremes of the 7200x7200 grid, with a separation of 3,600 cells (~111 km)
across all six pairs: four independent geographic contexts. verificar_layout() checks
this at runtime against the quadrant offsets.

The buffer also exceeds the correlation range measured by variogram (Amazonia 1,550 m,
Pantanal 1,100 m, Cerrado and Mata Atlantica 775 m), i.e., training and reserve are
spatially independent by the geostatistical criterion.

TRANSFER
-------------
Transfer settings:

  - weights always carry over from the previous quadrant
  - optimizer state carries over by default (`--otimizador continuar`)
  - per-quadrant lr warm restart: CosineAnnealingLR with T_max = that quadrant's
    epochs, reinstantiated at each quadrant (one cosine cycle per quadrant)
  - lr and epoch schedule: Q1 longer with a high lr, Q2..Q4 shorter with a low lr
  - no layer freezing.

B2 predicts ONE target (the DEM-LiDAR residual). The five-phase curriculum below
(FASES) phases in physical constraints, not targets.

FORGETTING MEASURE
----------------------
Q1's reserve is never seen, but it comes from a quadrant trained at the start.
Evaluating each reserve (a) right after training its own quadrant and (b) with the
final model gives the forgetting matrix for free. A positive delta = catastrophic
forgetting.

MANDATORY BASELINE
--------------------
`--sem-transferencia` trains each quadrant from scratch, with the same reserves and
the same seeds. Without this arm there is no possible claim about a transfer gain.

Usage:
    python b2_transferencia.py --biomes amazonia --arms gatv2_2 --seeds 42
    python b2_transferencia.py --biomes amazonia --arms gatv2_2 --seeds 42 --sem-transferencia
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import metricas_v23 as MET  # noqa: E402
from grade_v23 import CELULA_M  # noqa: E402

# Where the V23 vector lives. v22 still reads `QDIR`; the two paths coexist because
# comparison with the frozen baseline requires the old one to stay reproducible.
# Paths can be overridden by environment variables (V23_LATDIR, V23_RAIZ). Patching the
# module constants does not work: the script runs as `__main__` with its own globals.
LATDIR_V23 = Path(os.environ.get("V23_LATDIR", r"D:\GNN_TOPO\SATELITES\laterais"))
from b1_kriging_indutivo import (  # noqa: E402
    BATCH, BLOCO, BLOCO_POR_BIOMA, DEV, GRID, HIDDEN, MC_PASSES, QDIR, RESULTS,
    carregar, construir, log, mem_guard,
)

PREDS_B2 = Path(os.environ.get("V23_RAIZ", r"C:\GNN_TOPO\V23")) / "predicoes_b2"
CKPT_B2 = Path(os.environ.get("V23_RAIZ", r"C:\GNN_TOPO\V23")) / "checkpoints_b2"

# ── reserve geometry ──
MEIO = GRID // 2                  # 1800 cells (~55.7 km)
BUFFER_CEL = 67                   # ~2,072 m at 30.922 m/cell; measured correlation range: 1,550 m

# ── MIXED PRECISION (bfloat16).
# `bfloat16` rather than `float16`: both nearly halve activation memory, but bf16 has
# the same exponent as float32 and loses only mantissa, so it does not overflow and
# needs no `GradScaler`.
#
# Caveat: the directional-derivative constraint SUBTRACTS two neighbouring predictions,
# the worst case for a short mantissa. Compare `--bf16 off` against `--bf16 on` on
# `decl_diag.ultimo_lote.erro_m`, not only on the altimetric error.
MAX_NORM = 1.0                    # gradient-norm clipping
USA_AMP = False                   # set in `main` according to the feature vector and device
USA_CURRICULO = False             # likewise; see `FASES` and `pesos_da_epoca`
SEM_ESTADO_OPT = False            # `--sem-estado-otimizador`: weights carry over, Adam state does not
COM_RAIOX = True                  # full-coverage diagnostic families; see `_raiox_da_corrida`
PAINEL = True                     # per-epoch panel; disable with `--sem-painel`
AMP_DTYPE = torch.bfloat16

# ── BATCH SIZE (target nodes per batch), measured on a 15.9 GB RTX 5070 Ti with the
# physical constraints active:
#
#   BATCH    peak      reserved   headroom   reserved/peak
#   14,336   9.08 GB   23.08 GB   -45%       2.54x
#   10,240   6.69      10.73      +33%       1.60x
#    8,192   5.45      11.60      +27%       2.13x   <- lower peak, LARGER reserve
#    6,144   4.17       6.95      +56%       1.67x   <- chosen
#    4,096   2.86       4.35      +73%       1.52x
#
# The criterion is the reserved/peak ratio, not the peak: it measures allocator
# fragmentation, which decides whether memory spills to host RAM over PCIe (about 5x
# slower). The water constraint traverses the whole subgraph's `edge_index` and allocates
# blocks the caching allocator does not reuse; `expandable_segments` is not supported on
# Windows. The 56% headroom at 6,144 is kept for MC Dropout; 4,096 only adds steps per
# epoch.
BATCH_V23 = 6144
# LITERAL layout, not randomized. (r0, r1, c0, c1) in cells, half-open intervals.
# Corner labels are Portuguese abbreviations: NO = NW, SO = SW.
#
# Each reserve corner COINCIDES with the quadrant's fixed position in the biome (Q1=NW,
# Q2=NE, Q3=SW, Q4=SE, identical in all four biomes), placing the four reserves at the
# four corners of the 7200x7200 grid, 3,600 cells (~111 km) apart in all six pairs.
# Rotating the corner inside each quadrant instead would make the Q3 and Q4 reserves
# touch at global column 3600. Checked at runtime by verificar_layout().
RESERVA_POR_QUAD = {
    "Q1": ("NO", (0, MEIO, 0, MEIO)),
    "Q2": ("NE", (0, MEIO, MEIO, GRID)),
    "Q3": ("SO", (MEIO, GRID, 0, MEIO)),
    "Q4": ("SE", (MEIO, GRID, MEIO, GRID)),
}
SEP_MIN_RESERVAS_CEL = 1800     # ~55.7 km; the current layout gives 3,600
ORDEM_QUAD = ["Q1", "Q2", "Q3", "Q4"]

# ─────────────── GROWING WINDOW OVER THE QUADRANTS ───────────────
#
# The sequential chain Q1->Q2->Q3->Q4 trains each quadrant to convergence and passes the
# weights on. In a pilot run, Q1's reserve error grew by ~0.9 m by the end of the chain
# (catastrophic forgetting), because Q1 LEAVES THE LOSS as training moves on. The growing
# window removes that cause: a quadrant that enters never leaves the loss.
#
#   stage 1       stage 2       stage 3       stage 4
#   +---+---+     +---+---+     +---+---+     +---+---+
#   |Q1 |   |     |Q1 |Q2 |     |Q1 |Q2 |     |Q1 |Q2 |
#   +---+---+     +---+---+     +---+---+     +---+---+
#   |   |   |     |   |   |     |Q3 |   |     |Q3 |Q4 |
#   +---+---+     +---+---+     +---+---+     +---+---+
#
# The training region is always simply connected and grows outward, so at no stage does
# the model hold two disconnected pieces of terrain. The only constraint is that the
# SECOND quadrant cannot be diagonal to the first (Q1 then Q4 would touch only at a
# corner); `verificar_contiguidade` enforces this on every prefix of the order.
#
# Stage k passes over k quadrants, so the cost per epoch grows 1:2:3:4 while the epoch
# caps fall. Each stage is ended by early stopping on the validation anchors of the
# window, not by a fixed schedule.
#
# BATCH COMPOSITION: MIXED OR BLOCKED. Both modes exist because the choice has a cost
# either way:
#
#   mixed     `NeighborLoader` draws seeds from ALL quadrants of the window in the same
#             batch. Gradient decorrelated across regions.
#   blocked   each batch comes from ONE quadrant; quadrants alternate per `CICLO_ESPACIAL`.
#             Gradient correlated by region, as in the chain.
#
# In a pilot run (Amazonia, seed 42), the mixed window gave the best anchor skill and no
# forgetting, but removed 3-4x FEWER pits than the chain and had a higher p99.9 slope in
# all four quadrants: its `Delta` field is rougher.
#
# Hypothesis tested by the blocked arm: pits are a second-difference property of the
# field. Mixing quadrants in a batch decorrelates the gradient (which prevents
# forgetting) but also lets neighbouring cells be updated by gradients from unrelated
# regions. If forgetting comes from a quadrant LEAVING THE LOSS and roughness from MIXING
# THE BATCH, a growing window with blocked batches keeps both the anchor gain and the
# chain's pit removal. This arm separates the two effects.
#
# Neighbour context is preserved in both modes: the neighbourhood is sampled PER SEED and
# travels with the node, not with the batch; only the gradient statistics change. In both
# cases sampling is deterministic given (biome, seed, block) and the partition is fixed.
#
# CICLO_ESPACIAL is the (non-random) rotation order of batches in blocked mode:
#
#        Q1 (NW) -- Q2 (NE)          Q1 -> Q2 : share the northern vertical border
#         |            |             Q2 -> Q4 : share the eastern horizontal border
#        Q3 (SW) -- Q4 (SE)          Q4 -> Q3 : share the southern vertical border
#                                    Q3 -> Q1 : share the western horizontal border
#
# A Hamiltonian cycle on the 2x2 grid adjacency: EVERY batch transition crosses a shared
# border, including the wrap-around. No diagonal jumps.
CICLO_ESPACIAL = ["Q1", "Q2", "Q4", "Q3"]
LOTE_BLOCADO = False              # enabled by --lotes blocado; see `main`
FRAC_ANCORAS_TREINO = 1.0         # < 1 via --frac-ancoras-treino; see `particao_com_reserva`

# ── RUN TAG, appended to `modo` in EVERY artifact name.
#
# Checkpoint names follow `{bioma}_seed{s}_{arm}_{vetor}_{modo}_ate{quad}.pt`. In a
# one-factor-at-a-time (OFAT) sweep, `arm`, `vetor` and `modo` are identical across
# points (only a loss weight changes), so without a tag all points would write to the
# SAME file, each overwriting the previous one.
TAG = ""


def com_tag(modo: str) -> str:
    return f"{modo}_{TAG}" if TAG else modo
#
# DISJOINT UNION, NO STITCHING. The quadrants become a single `Data` by index offset,
# and NO edge crosses the boundary between them. Stitching the borders would be more
# faithful to the terrain, but a stitching edge would create a message path between a
# training node of one quadrant and a reserve node of its neighbour, and
# `verificar_garantias` does not check leakage through neighbourhood hops.
# Stage caps must exceed the curriculum: the five phases total 30 epochs (10+5+5+5+5)
# and early stopping cannot act before the curriculum ends plus 5 epochs, so a cap of 30
# would make early stopping unreachable. A cap of 45 leaves 15 free epochs after the
# ramp, with stopping possible from epoch 35.
ETAPAS_EPOCAS = [45, 25, 18, 14]        # per-stage caps; early stopping decides
ETAPAS_LR = [1e-3, 5e-4, 3e-4, 2e-4]    # stage 1 starts cold, later stages are already warm

# Patience proportional to the cap. `PACIENCIA = 12` suits the chain caps (80 and 40);
# with caps of 25 to 14 it would never trigger, and the cap rather than early stopping
# would decide.
def paciencia_da_etapa(teto: int) -> int:
    return max(6, teto // 3)

# Position of each quadrant in the biome box, as (row, col) of 2x2 blocks. Same table as
# `verificar_layout`; here it is used for adjacency, there for separation.
OFFSET_QUAD = {"Q1": (0, 0), "Q2": (0, 1), "Q3": (1, 0), "Q4": (1, 1)}


def verificar_contiguidade(ordem: list) -> dict:
    """Every prefix of the growing-window order must be connected by an EDGE, not a corner.

    Without this guard, `--quads Q1 Q4 ...` would train stage 2 on two pieces of terrain
    touching only at a vertex, the discontinuity the growing window exists to avoid. The
    default order (Q1 Q2 Q3 Q4) passes; the guard is for overrides.
    """
    prefixos = {}
    for k in range(1, len(ordem) + 1):
        pref = ordem[:k]
        vistos, pilha = {pref[0]}, [pref[0]]
        while pilha:
            q = pilha.pop()
            r, c = OFFSET_QUAD[q]
            for outro in pref:
                if outro in vistos:
                    continue
                r2, c2 = OFFSET_QUAD[outro]
                if abs(r - r2) + abs(c - c2) == 1:      # Manhattan distance 1 = shared border
                    vistos.add(outro)
                    pilha.append(outro)
        prefixos[k] = {"janela": list(pref), "conexo": len(vistos) == len(pref)}
        if len(vistos) != len(pref):
            soltos = [q for q in pref if q not in vistos]
            raise ValueError(
                f"ordem de ampliacao invalida: na etapa {k} a janela {pref} nao e conexa "
                f"por aresta — {soltos} so toca(m) o resto num canto. Reordene para que "
                f"cada quadrante que entra divida uma borda com os que ja estao.")
    return {"ordem": list(ordem), "prefixos": prefixos,
            "criterio": "adjacencia de aresta (Manhattan 1) na grade 2x2 de quadrantes"}

# ─────────────── cross-sensor structure injection ───────────────
# A single linear projection over concatenated columns computes a weighted SUM, not a
# PRODUCT, while the sensor-feature mechanisms used here are multiplicative or
# conditional. Two levels address this: named interaction products (level 2) and a
# regime-gated mixture of per-sensor experts (level 3).
#
# Column layout of dd["X"] for the v22 vector with SAR (the regime requires SAR):
#     0:8    topo8    elevation slope aspect_cos aspect_sin curvature tpi tri roughness
#     8:13   optico5  ndvi ndwi bsi bright shadow
#    13:18   sar5     vv vh rvi sar_diff sar_sum
#    18:22   regime4  agua solo_exposto dossel_denso dossel_esparso
# The observation channels (value, flag) are appended later, in preparar_tensores.
COL_TOPO, COL_OPTICO, COL_SAR, COL_REGIME = (0, 8), (8, 13), (13, 18), (18, 22)


def _cols(dd: dict) -> dict:
    """Column slices per family. The V23 vector carries them in `dd`; v22 uses fixed ones.

    This lets the new vector, which inserts columns WITHIN each family rather than at the
    end, reuse `zerar_blocos`, `adicionar_interacoes` and `layout_grupos` unchanged.
    """
    return dd.get("cols") or {"topo": COL_TOPO, "optico": COL_OPTICO,
                              "sar": COL_SAR, "regime": COL_REGIME}


def _nomes(dd: dict) -> list:
    return dd.get("nomes_col") or (NOMES_TOPO + NOMES_OPTICO + NOMES_SAR + NOMES_REGIME)
# `amplitude3x3` (formerly `tri`): the graph computes `max3x3 - min3x3`, a local range,
# not Riley's Terrain Ruggedness Index, hence the rename.
NOMES_TOPO = ["elevation", "slope", "aspect_cos", "aspect_sin",
              "curvature", "tpi", "amplitude3x3", "roughness"]
NOMES_OPTICO = ["ndvi", "ndwi", "bsi", "bright", "shadow"]
NOMES_SAR = ["vv", "vh", "rvi", "sar_diff", "sar_sum"]
NOMES_REGIME = ["agua", "solo_exposto", "dossel_denso", "dossel_esparso"]

# LEVEL 2: named interaction products. Each pair is a named physical mechanism, not an
# exhaustive combination: all pairwise products would give 22*21/2 = 231 noisy columns.
INTERACOES = [
    ("roughness", "rvi", "rugosidade que pode ser dossel"),
    ("roughness", "um_menos_rvi", "rugosidade que sobra como terreno"),
    ("tri", "rvi", "amplitude local que pode ser dossel"),
    ("tpi", "ndwi", "depressao que acumula agua"),
    ("curvature", "ndwi", "concavidade com agua"),
    ("slope", "shadow", "geometria contra iluminacao solar"),
    ("vv", "ndwi", "espelhamento especular em lamina de agua"),
    ("ndvi", "rvi", "concordancia optico-radar de dossel"),
]

# ─────────────────────────────────────────────────────────────────────────────
# V23 VECTOR LIST: it REPLACES the list above rather than adding to it.
#
# The list above serves the v22 vector (the frozen baseline), whose B3 ablation requires
# the eight products to be bit-identical. For V23 this list is used INSTEAD: columns
# `vv`, `rvi`, `tri`, `dem_desvio` and `d_fabdem` no longer exist, and five of the eight
# old products needed one of them.
#
# SEASONALITY: mechanisms are not duplicated per season. In Amazonia only 29.9% of pixels
# change by more than 1 dB between seasons, and the two seasons correlate at +0.850 per
# pixel, so wet/dry pairs would be correlated noise. The rule follows the mechanism:
#
#   STRUCTURE mechanism (canopy, volume, roughness)  ->  MEAN of the two seasons
#   WATER mechanism (flooding, moisture)              ->  AMPLITUDE between them
#
# Means and amplitudes are ALIASES computed in the assembler (like `um_menos_rvi`); no
# new column enters the vector.
#
# Two mechanisms were dropped with the DEM branch, without substitute:
#
#   `d_fabdem x l_rvi_asc`      relied on a competing DEM trained on our own reference.
#
#   `l_hv_asc x dem_desvio`     the equivalent (canopy volume where C and L band
#                               disagree) is a DIFFERENCE, not a product, and a linear
#                               layer forms differences on its own. A product is
#                               justified only when the mechanism is multiplicative.
APELIDOS_V23 = {
    # seasonal mean: structure is not seasonal
    "rvi_m":          (("c_rvi_u", "c_rvi_s"), lambda a, b: 0.5 * (a + b)),
    "vv_m":           (("c_vv_u", "c_vv_s"), lambda a, b: 0.5 * (a + b)),
    "sd_vv_m":        (("c_sdvv_u", "c_sdvv_s"), lambda a, b: 0.5 * (a + b)),
    "um_menos_rvi_m": (("rvi_m",), lambda a: 1.0 - a),
    # optical amplitude each pixel actually experienced, independent of a defined season
    "ndwi_amp":       (("o_ndwi_p90", "o_ndwi_p10"), lambda a, b: a - b),
}

INTERACOES_V23 = [
    # ── the original eight, re-pointed: same physics, V23 column names.
    ("roughness", "rvi_m", "rugosidade que pode ser dossel"),
    ("roughness", "um_menos_rvi_m", "rugosidade que sobra como terreno"),
    ("amplitude3x3", "rvi_m", "amplitude local que pode ser dossel"),
    ("tpi", "ndwi", "depressao que acumula agua"),
    ("curvature", "ndwi", "concavidade com agua"),
    ("slope", "shadow", "geometria contra iluminacao solar"),
    ("vv_m", "ndwi", "espelhamento especular em lamina de agua"),
    ("ndvi", "rvi_m", "concordancia optico-radar de dossel"),
    # ── L band: the SAME mechanism where C band saturates and L band does not. The
    # difference between the two pairs measures how much saturation affected the C-band term.
    ("roughness", "l_rvi_asc", "rugosidade que pode ser dossel, sem saturacao de banda C"),
    ("amplitude3x3", "l_rvi_asc", "amplitude local contra volume em banda L"),
    ("ndvi", "l_rvi_asc", "concordancia optico-radar onde o optico satura e a L nao"),
    ("agua_ocorrencia", "l_hh_asc", "lamina de agua contra espelhamento em banda L"),
    # ── two products that exist only because of the seasonal composite.
    #
    # FLOODING. High amplitude in radar AND in optical: their agreement is much stronger
    # evidence than either alone, and flooding is where the DSM is most wrong. `sd_VV` is the
    # only SAR band keeping its sign in all four biomes (-0.16 to -0.22 against the target)
    # and correlates with elevation at -0.353 in Amazonia (low terrain varies more);
    # `o_ndwi_p10` is the strongest optical column of the vector, at -0.275.
    ("sd_vv_m", "ndwi_amp", "alaga: amplitude no radar concordando com amplitude no optico"),
    # VIEWING GEOMETRY. Backscatter depends on the angle at which the wave meets the
    # TERRAIN, and the term is multiplicative, which a linear layer cannot form.
    #
    # No circularity with layer 2 of `raiox_geomorfologico`: here `slope` is from the INPUT
    # GLO-30, whereas there the local angle comes from the OUTPUT `z'`. Valid in the three
    # biomes with two tracks; in the Pantanal the angle correlates -0.944 with the raster
    # column, so the product becomes `slope x position`.
    ("c_ang_u", "slope", "angulo de visada contra inclinacao do terreno"),
]

# LEVEL 3: physical prior of the gate. Rows = regime, columns = sensor branch.
# Based on the measured residual: -0.11 m on bare soil versus +12.06 m under canopy
# (Amazonia).
#   agua           DSM over water is often noise/void; optical discriminates
#   solo_exposto   DSM = DTM, the topography itself is reliable
#   dossel_denso   DSM is canopy; SAR is the only sensor that sees beneath it
#   dossel_esparso intermediate regime
#                    topo  optico   sar   inter
PRIOR_PORTA = np.array([[0.3,   1.0,   0.6,   0.8],      # agua
                        [1.0,   0.4,   0.4,   0.6],      # solo_exposto
                        [0.4,   0.6,   1.0,   1.0],      # dossel_denso
                        [0.7,   0.6,   0.8,   0.9]],     # dossel_esparso
                       dtype=np.float32)
RAMOS_PRIOR = ["topo", "optico", "sar", "inter"]         # columns of PRIOR_PORTA

# ONE HEAD PER DATA SOURCE.
#
# Each SOURCE has its own encoder: a shared linear projection would have to learn, with
# the same weights, the scale of a sigma0 in dB and of a dimensionless height difference.
#
# THE PRIOR OF A NEW BRANCH IS INHERITED. The matrix above has four columns (four
# families). Each new branch takes the column of the PHYSICAL family it belongs to, a
# verifiable statement rather than an invented number. `A` remains trainable, so the
# data can disagree with the inherited prior, and the deviation is reported.
HERANCA_PRIOR = {
    "topo": "topo",          # 8 topographic columns, from GLO-30
    "optico": "optico",      # NDVI, NDWI, BSI, brightness, shadow
    "agua": "optico",        # JRC occurrence and seasonality, mapped to optical
    "alos_l": "sar",         # ALOS-1, L band, 2006-2011
    "palsar2": "sar",        # PALSAR-2, L band, annual mosaic
    # V23 branches: DIRECT inheritance (same physical family, different source/window).
    "s1_umida": "sar",       # Sentinel-1 C band, wet season 2021-2023
    "s1_seca": "sar",        # Sentinel-1 C band, dry season 2021-2023
    "s2_amplitude": "optico",  # NDVI and NDWI percentiles and composite depth
    "inter": "inter",        # cross products
    "sar": "sar",            # old-layout name, kept for the v22 vector
    # `dem_ind` and `sar_c` are deliberately absent: an orphan prior accepted silently would
    # let a branch enter the gate with arbitrary weight (see `prior_do_ramo`).
}


def prior_do_ramo(nome: str) -> np.ndarray:
    fam = HERANCA_PRIOR.get(nome)
    if fam is None:
        raise KeyError(f"ramo `{nome}` sem prior declarado em HERANCA_PRIOR. Um ramo "
                       f"sem prior entraria na porta com peso arbitrario.")
    return PRIOR_PORTA[:, RAMOS_PRIOR.index(fam)].copy()

# ─────────────────────── FIVE-PHASE CURRICULUM ───────────────────────
#
# V23 has a SINGLE target, which guarantees that predicted slope is the gradient of
# predicted elevation. The curriculum therefore phases in THREE PHYSICAL CONSTRAINTS,
# not targets.
#
# The ORDER follows the measured reliability of each piece of evidence:
#
#   1  label only          Without a reasonable field, the constraints pull a random
#                          surface toward a physical consistency unrelated to terrain.
#   2  + bare soil         The most EXTERNAL evidence: where tree cover is low,
#                          `|DSM - ground|` IS the error, measured without LiDAR
#                          (median error 2.37 m, bias -0.42 m).
#   3  + water             A water surface is equipotential and JRC occurrence is
#                          reliable: a strong constraint on a well-delimited class.
#   4  + slope             LAST: measured to be the LEAST reliable of the three
#                          (median angular error 4.1 degrees; 71% of measurements
#                          with confidence below 0.5).
#   5  consolidation       Everything active, lr at the cosine minimum.
#
# LINEAR RAMP within each phase, from zero to target. Switching a term on at an epoch
# boundary creates a step in the loss that costs epochs to recover from.
#
# Only on a COLD start (Q1 in the chain, stage 1 in the growing window). Later stages
# inherit trained weights and have few epochs, so they run in phase-5 mode, with every
# constraint active from the first epoch.
# `dossel` enters TOGETHER with `solo` in phase 2: both constrain `Delta` directly and are
# the two sign constraints, one per cover regime. Separating them would give one of them
# a five-epoch head start with no physical reason.
FASES = [
    ("rotulo",       10, ()),
    ("solo_dossel",   5, ("solo", "dossel")),
    ("agua",          5, ("solo", "dossel", "agua")),
    ("declividade",   5, ("solo", "dossel", "agua", "decl")),
    ("consolidacao",  5, ("solo", "dossel", "agua", "decl")),
]

# Target weights come from MEASURED RATIOS, not guesses. At the end of phase 1 (all
# weights zero), each weight is set so that its constraint contributes 10% of the label
# loss, i.e. peso = 0.10 * L_label / L_constraint (three seeds; PESOS_ALVO uses ~means):
#
#   seed   label    soil -> weight   water -> weight   slope -> weight
#   42     6.666    2.283   0.292    1.558   0.428     7.989   0.083
#   123    6.616    2.411   0.274    1.750   0.378     7.932   0.083
#   7      6.677    2.465   0.271    2.061   0.324     8.055   0.083
#
# A default weight of 1.0 would make some constraints negligible and let others dominate.
# These are STARTING POINTS: the OFAT sweep explores 0.5x, 1x, 2x, 4x around them,
# judged by each constraint's own residual and by the geomorphometric diagnostics, with
# MAE as a guardrail, not an objective (MAE is measured at the anchors while the physics
# acts away from them, so minimizing MAE would always select zero).
#
# `dossel` has no phase-1 ratio. Its weight (137) was chosen by an OFAT sweep (4 weights
# x 5 seeds): it removes severe violations (>1 m, from 20-27% of cells to 0%) at a cost of
# 0.0033 in skill. A weight of 0.28 contributed only 0.005% of the loss.
PESOS_ALVO = {"solo": 0.28, "agua": 0.38, "dossel": 137.0, "decl": 0.083}
PESO_DOSSEL = 137.0
# Smoothness of the `Delta` field over edges: `mean(|p_i - p_j|)`. Zero by default. The
# high-frequency energy of `Delta` predicts the pit count (Spearman +0.52 to +0.68), and
# the main cost of the best regime lies in the tail of pits deeper than 2 m. L1 (total
# variation) rather than L2 preserves legitimate steps at forest edges (canopy ->
# clearing is a REAL discontinuity of `Delta`) while suppressing cell-to-cell
# oscillation, which is what creates pits.
PESO_SUAVE = 0.0
# Reweighting by label SOURCE. Zero = off. With K > 0, ATL08 anchors (source 2/3) weigh K
# in the loss and GEDI anchors weigh 1. Rationale: GEDI places the ground 1.5-6.5 m above
# ICESat-2 as canopy height increases, and with ~95% GEDI labels (219k GEDI vs 2k ATL08)
# the model reproduces the GEDI bias under 20-30 m canopy. The boost tests whether giving
# weight to the minority instrument moves the model there, judged on the ATL08 anchors
# of the RESERVE (never trained on, so the evaluation is not circular).
BOOST_ATL08 = 0.0
VIZ_EXATA = False             # exact k-hop neighbourhood (lattice); see fan_out
TETO_VRAM_GB = 0.0            # >0: abort if peak VRAM exceeds this cap (fail fast
                              # instead of silently spilling to host RAM)
SUAVIZAR_X = 0                # radius (cells) of the spatial mean of X; 0 = raw
VAZIO_KM = 0.0                # radius of the synthetic-void mask (km)
# Centroids (lon, lat) of the 19 EBA lidar transects inside the box, derived from the
# L3A footprints (Zenodo 4890398). The void mask is purely geometric: identical across
# arms and seeds by construction.
CENTROIDES_EBA = [
    (-59.051, -2.348), (-58.551, -2.772), (-58.001, -3.230), (-59.948, -2.952),
    (-59.931, -2.943), (-59.940, -2.944), (-58.427, -3.953), (-60.000, -2.538),
    (-59.999, -2.540), (-58.871, -2.866), (-58.707, -2.893), (-58.054, -3.227),
    (-59.241, -3.505), (-59.277, -3.512), (-59.948, -2.949), (-59.193, -3.822),
    (-59.940, -2.946), (-58.558, -3.496), (-58.684, -3.410)]


def pesos_da_epoca(ep: int, com_curriculo: bool, alvos: dict) -> dict:
    """Weight of each constraint at epoch `ep`, ramped within the phase that introduces it.

    The switch is `com_curriculo`, not the quadrant name: the curriculum is for a COLD
    start, which is Q1 in the chain and stage 1 in the growing window.
    """
    if not com_curriculo:
        return dict(alvos)                      # phase-5 regime from the first epoch
    ini = 0
    for nome, n_ep, ativas in FASES:
        if ep < ini + n_ep:
            k = (ep - ini + 1) / float(n_ep)    # 1/n ... 1 within the phase
            # Constraints DEBUTING in this phase: active here but not in the previous one.
            # Several can debut together (`solo` and `dossel`), so each gets its own ramp.
            antes = set()
            for nm, ne, at in FASES:
                if nm == nome:
                    break
                antes |= set(at)
            novas = set(ativas) - antes
            fora = {}
            for c, alvo in alvos.items():
                if c not in ativas:
                    fora[c] = 0.0
                elif c in novas and nome != "consolidacao":
                    fora[c] = alvo * k          # debuting in this phase, ramped
                else:
                    fora[c] = alvo              # already active, at full value
            fora["_fase"] = nome
            return fora
        ini += n_ep
    return dict(alvos)                          # epochs past the curriculum: everything at full value


# Epoch/lr ladder: Q1 learns from scratch, later quadrants fine-tune.
#
# These are CAPS, not fixed counts: early stopping ends each quadrant, so an easy quadrant
# stops early and a hard one uses the cap. Epochs actually used are recorded per
# quadrant, and their differences indicate which quadrant is harder.
EPOCAS_ESCADA = {"Q1": 80, "Q2": 40, "Q3": 40, "Q4": 40}

# Epochs without validation improvement before stopping.
PACIENCIA = 12

# Early stopping cannot act INSIDE the curriculum: each phase ADDS a term to the loss, so
# validation worsens at phase boundaries by construction, not for lack of learning. The
# floor is the end of the curriculum; in later quadrants, with no curriculum, it is a
# short minimum for the inherited weights to settle.
MIN_EPOCAS_APOS_CURRICULO = 5
LR_ESCADA = {"Q1": 1e-3, "Q2": 2e-4, "Q3": 2e-4, "Q4": 2e-4}
LR_MIN = 1e-6                     # eta_min of CosineAnnealingLR
DECAIMENTO = 0.01                 # matrices only; see the parameter groups in treinar_quadrante


# ─────────────────────── spatial partition ───────────────────────
def dist_chebyshev_ao_retangulo(row: np.ndarray, col: np.ndarray,
                                caixa: tuple) -> np.ndarray:
    """Chebyshev distance, in cells, from each anchor to the reserve rectangle.

    Zero inside the rectangle. Chebyshev (not Euclidean) because the graph is 8-connected:
    it is the metric in which one hop costs 1 in any direction, i.e. the actual
    structural reach (equivalent to cKDTree(..., p=np.inf)).
    """
    r0, r1, c0, c1 = caixa
    dr = np.maximum(np.maximum(r0 - row, row - (r1 - 1)), 0)
    dc = np.maximum(np.maximum(c0 - col, col - (c1 - 1)), 0)
    return np.maximum(dr, dc)


def zerar_blocos(dd: dict, sem_sar: bool, sem_regime: bool) -> dict:
    """Zero out a sensor's columns while keeping the number of columns.

    As in `_semobs`: columns stay and their content is zeroed, so `d_in` is unchanged and
    capacity is matched. The measured difference is attributable to the sensor's
    INFORMATION, not to network size.

    Two variants because the regime is DERIVED from SAR (the prototypes use vv and rvi):
      sem_sar          zeroes the SAR columns; the regime still carries SAR indirectly
      sem_sar+regime   also zeroes the regime; measures the TOTAL SAR contribution
    """
    X, c = dd["X"], _cols(dd)
    if X.shape[1] < c["regime"][1]:
        raise ValueError("ablacao de sensor exige o layout com SAR e regime")
    if sem_sar:
        # ALL radar, not only the block named `sar`. With one head per source, radar is split
        # across sar_c, alos_l and palsar2; zeroing only one would measure "without Sentinel-1"
        # and be reported as "without SAR".
        radar = [k for k in c if k != "regime" and HERANCA_PRIOR.get(k) == "sar"]
        if not radar:
            raise ValueError("nenhum bloco de radar no layout; `--sem-sar` nao se aplica")
        for k in radar:
            X[:, c[k][0]:c[k][1]] = 0.0
        log(f"    zerados os blocos de radar: {', '.join(radar)}")
    if sem_regime:
        X[:, c["regime"][0]:c["regime"][1]] = 0.0
    dd["X"] = X
    log(f"    zerado: {'SAR(5) ' if sem_sar else ''}{'regime(4)' if sem_regime else ''}")
    return dd


def adicionar_interacoes(dd: dict) -> dict:
    """LEVEL 2: append the named products. No architecture change.

    Pure feature engineering, so that the ablation is clean: does the model improve when
    given `roughness x rvi` for free, or had it already discovered it? Both answers are
    informative, and neither is obtainable without the control arm.
    """
    X, c = dd["X"], _cols(dd)
    if X.shape[1] < c["regime"][1]:
        raise ValueError(
            f"interacoes exigem o bloco de regime (esperado >={c['regime'][1]} colunas, "
            f"veio {X.shape[1]}). Sem SAR nao ha regime e a ablacao nao se aplica.")
    nomes = _nomes(dd)
    col = {n: X[:, i] for i, n in enumerate(nomes[:X.shape[1]])}

    # Which list applies. They are NOT combined: the first is the frozen baseline (v22
    # names); the second is V23, where five of those columns no longer exist. The test is
    # whether the vector declares its own branches, which only V23 does.
    v23 = bool(dd.get("cols")) and "s1_umida" in dd["cols"]
    if v23:
        for nome, (fontes, fn) in APELIDOS_V23.items():
            if all(f in col for f in fontes):
                col[nome] = fn(*[col[f] for f in fontes])
        lista = INTERACOES_V23
    else:
        col["um_menos_rvi"] = 1.0 - col["rvi"]
        lista = INTERACOES

    faltam = sorted({n for par in lista for n in par[:2] if n not in col})
    if faltam:
        raise ValueError(
            f"interacoes do vetor {'V23' if v23 else 'v22'} pedem colunas que ele nao "
            f"tem: {faltam}. Um produto que se omite em silencio some da ablacao sem "
            f"deixar rastro — por isso isto falha em vez de pular.")

    usadas = list(lista)
    novas = np.stack([col[a] * col[b] for a, b, _ in usadas], axis=1)
    novas = np.nan_to_num(novas, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    dd["X"] = np.concatenate([X, novas], axis=1)
    dd["col_interacoes"] = (X.shape[1], dd["X"].shape[1])
    dd["n_interacoes"] = len(usadas)
    dd["nomes_interacoes"] = [f"{a}x{b}" for a, b, _ in usadas]
    if dd.get("nomes_col"):
        dd["nomes_col"] = list(dd["nomes_col"]) + dd["nomes_interacoes"]
    dd["feat_nomes"] = dd.get("feat_nomes", []) + [f"inter{len(usadas)}"]
    log(f"    interacoes: +{novas.shape[1]} colunas "
        f"({' | '.join(dd['nomes_interacoes'])})")
    return dd


class PortaRegime(nn.Module):
    """LEVEL 3: mixture of experts gated by the PHYSICAL REGIME, not by a learned gate.

    One encoder per sensor family; soft regime membership weights the branches. The
    mixing matrix starts from the physical prior and is trainable, so the data can
    disagree with the physics; its deviation from the prior is a reportable result,
    which is why `A` is logged at the end of training.

    The observation channels are NOT weighted: they carry the propagated label and must
    pass through in every regime.
    """

    def __init__(self, grupos: dict, col_regime: tuple, h: int = HIDDEN):
        super().__init__()
        self.grupos = {k: v for k, v in grupos.items() if k != "obs"}
        self.col_obs = grupos["obs"]
        self.col_regime = col_regime
        self.enc = nn.ModuleDict({k: nn.Linear(v[1] - v[0], h)
                                  for k, v in self.grupos.items()})
        self.enc_obs = nn.Linear(grupos["obs"][1] - grupos["obs"][0], h)
        self.A = nn.Parameter(torch.tensor(
            np.stack([prior_do_ramo(k) for k in self.grupos], axis=1)))
        self.ordem_ramos = list(self.grupos)

    def forward(self, x):
        m = x[:, self.col_regime[0]:self.col_regime[1]]          # [N,4] membership
        w = torch.relu(m @ self.A)                               # [N, n_ramos]
        h = self.enc_obs(x[:, self.col_obs[0]:self.col_obs[1]])
        for j, k in enumerate(self.ordem_ramos):
            a, b = self.grupos[k]
            h = h + w[:, j:j + 1] * self.enc[k](x[:, a:b])
        return h


class GATv2Portao(nn.Module):
    """GATv2 preceded by the regime gate. The convolutions are identical to the plain arm,
    so that the measured difference is attributable to THE GATE alone."""

    def __init__(self, grupos: dict, col_regime: tuple, camadas: int,
                 h: int = HIDDEN, heads: int = 8):
        super().__init__()
        from torch_geometric.nn import GATv2Conv
        assert h % heads == 0
        self.porta = PortaRegime(grupos, col_regime, h)
        self.convs = nn.ModuleList(
            [GATv2Conv(h, h // heads, heads=heads, concat=True) for _ in range(camadas)])
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(h, 1)

    def forward(self, x, edge_index):
        x = self.drop(torch.relu(self.porta(x)))
        for conv in self.convs:
            x = self.drop(torch.relu(conv(x, edge_index)))
        return self.head(x).squeeze(-1)


def layout_grupos(d_in: int, com_interacoes: bool,
                  cols: dict | None = None,
                  n_inter: int | None = None) -> tuple[dict, tuple]:
    """Column slices per sensor family, including the 2 observation channels at the end."""
    c = cols or {"topo": COL_TOPO, "optico": COL_OPTICO,
                 "sar": COL_SAR, "regime": COL_REGIME}
    # ONE BRANCH PER declared BLOCK, in column order: three with the old layout, seven with
    # the V23 vector, each source with its own encoder.
    g = {k: v for k, v in sorted(c.items(), key=lambda x: x[1][0]) if k != "regime"}
    fim = c["regime"][1]
    if com_interacoes:
        # Count the products actually appended, not the base list: the V23 vector adds
        # L-band products, and the layout must follow or the `obs` slice lands in the
        # wrong place and the assertion below fails.
        g["inter"] = (fim, fim + (n_inter if n_inter is not None else len(INTERACOES)))
        fim = g["inter"][1]
    g["obs"] = (fim, fim + 2)
    if g["obs"][1] != d_in:
        raise ValueError(f"layout de colunas inconsistente: esperado d_in={g['obs'][1]}, "
                         f"veio {d_in}. Grupos: {g}")
    return g, c["regime"]


def verificar_layout(bioma: str, quads: list) -> dict:
    """Compose the reserves in GLOBAL biome coordinates and require a minimum separation.

    A geometric claim about the reserves (e.g. that they are mutually non-contiguous) is
    only valid when checked against the quadrants' actual offsets in the biome grid.
    """
    # Quadrant positions are FIXED and deterministic on the 7200-cell grid: Q1 NW, Q2 NE,
    # Q3 SW, Q4 SE (the original Q1 graph records
    # `{'name': 'Q1', 'r0': 0, 'c0': 0, 'full_grid': [7200, 7200]}`).
    OFFSET = {"Q1": (0, 0), "Q2": (0, 1), "Q3": (1, 0), "Q4": (1, 1)}
    caixas = {}
    for q in quads:
        if q not in OFFSET or q not in RESERVA_POR_QUAD:
            continue
        qr, qc = OFFSET[q]
        r0, c0 = qr * (GRID // 2), qc * (GRID // 2)
        _, (a, b, c, e) = RESERVA_POR_QUAD[q]
        caixas[q] = (r0 + a, r0 + b, c0 + c, c0 + e)

    pares, pior, pior_par = {}, None, None
    ks = list(caixas)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            A, B = caixas[ks[i]], caixas[ks[j]]
            gap = max(max(A[0] - B[1], B[0] - A[1]), max(A[2] - B[3], B[2] - A[3]))
            pares[f"{ks[i]}x{ks[j]}"] = int(gap)
            if pior is None or gap < pior:
                pior, pior_par = int(gap), f"{ks[i]}x{ks[j]}"

    diag = {"caixas_globais": {k: list(v) for k, v in caixas.items()},
            "gap_por_par_celulas": pares,
            "separacao_minima_celulas": pior,
            "separacao_minima_m": (pior * CELULA_M) if pior is not None else None,
            "par_mais_proximo": pior_par,
            "exigido_celulas": SEP_MIN_RESERVAS_CEL}
    if pior is not None and len(caixas) > 1 and pior < SEP_MIN_RESERVAS_CEL:
        raise AssertionError(
            f"{bioma}: reservas proximas demais — {pior_par} separadas por {pior} "
            f"celulas ({pior*30} m), minimo exigido {SEP_MIN_RESERVAS_CEL} "
            f"({SEP_MIN_RESERVAS_CEL*30} m). Caixas globais: {caixas}")
    log(f"  layout das reservas: separacao minima {pior} celulas "
        f"({(pior or 0)*30/1000:.0f} km) no par {pior_par}")
    return diag


def particao_com_reserva(dd: dict, quad: str, seed: int,
                         bloco_cel: int) -> tuple[np.ndarray, dict]:
    """Return a label per anchor and the partition diagnostics.

    Labels:   0 observed (train)  1 val (masked, train)
              2 in-distribution test (masked, train)  3 RESERVE  -1 discarded
    """
    idx = dd["idx"]
    row, col = idx // GRID, idx % GRID
    nome, caixa = RESERVA_POR_QUAD[quad]
    d = dist_chebyshev_ao_retangulo(row, col, caixa)

    lab = np.full(len(idx), -1, dtype=np.int8)
    lab[d == 0] = 3                                   # reserve
    treino_ok = d >= BUFFER_CEL                       # 0 < d < BUFFER -> discarded

    # Inside the training region, keep B1's masked-anchor protocol: split by spatial
    # BLOCK (never by node), 60/20/20.
    rng = np.random.default_rng(seed)
    bl = dd["bloco"]
    ub = np.unique(bl[treino_ok])
    r = rng.random(len(ub))
    mp = dict(zip(ub.tolist(),
                  np.where(r < 0.60, 0, np.where(r < 0.80, 1, 2)).tolist()))
    for i in np.nonzero(treino_ok)[0]:
        lab[i] = mp[bl[i]]

    # SAMPLE-EFFICIENCY CURVE. Subsample only the OBSERVED training anchors
    # (label 0). Val, test and reserve are untouched, so the evaluation population
    # is identical across fractions and the curves are comparable.
    #
    # NESTED FRACTIONS. One uniform value per anchor is drawn once from a DEDICATED
    # stream (`seed + 90000`) and thresholded, so the 25% set is a subset of the 50%
    # set, which is a subset of the 100% set. Independent draws per fraction would
    # confound density with which anchors were dropped.
    #
    # Dropped anchors become -1, the buffer label: they leave training and all
    # evaluation without a new label or any change to consumers of `lab`.
    n_antes = int((lab == 0).sum())
    if FRAC_ANCORAS_TREINO < 1.0:
        obs = np.nonzero(lab == 0)[0]
        u = np.random.default_rng(seed + 90000).random(len(obs))
        lab[obs[u >= FRAC_ANCORAS_TREINO]] = -1

    diag = {
        "quadrante": quad, "reserva_quina": nome, "reserva_caixa_celulas": list(caixa),
        "buffer_celulas": BUFFER_CEL, "buffer_m": round(BUFFER_CEL * CELULA_M),
        "bloco_celulas": int(bloco_cel), "bloco_m": round(bloco_cel * CELULA_M),
        "n_observada": int((lab == 0).sum()), "n_val": int((lab == 1).sum()),
        "n_teste_em_dist": int((lab == 2).sum()), "n_reserva": int((lab == 3).sum()),
        "n_descartada_no_buffer": int((lab == -1).sum()),
        # The reserve is 1/4 of the AREA by construction, but LiDAR transects are not
        # uniform on the grid, so the fraction of ANCHORS differs (0.205 in a test run
        # on Amazonia). Both numbers are reported to avoid confusing the two.
        "frac_area_reserva": 0.25,
        "frac_ancoras_na_reserva": float((lab == 3).mean()),
        # Requested and realized subsampling fractions, recorded side by side.
        "frac_ancoras_treino_pedida": float(FRAC_ANCORAS_TREINO),
        "n_observada_antes_da_subamostragem": n_antes,
        "frac_ancoras_treino_realizada": (
            float((lab == 0).sum()) / n_antes if n_antes else None),
    }
    return lab, diag


def verificar_garantias(lab: np.ndarray, obs_val: np.ndarray, obs_flg: np.ndarray,
                        idx: np.ndarray, dd: dict, quad: str) -> dict:
    """The four guarantees, checked on data rather than asserted in comments.

    A "never seen" claim that is not verified by assertion is worthless.
    """
    m_res = lab == 3
    n_res = int(m_res.sum())
    if n_res == 0:
        raise ValueError(f"{quad}: reserva vazia; a geometria esta errada")

    # G2: observation channel zeroed across the whole reserve
    ir = idx[m_res]
    vaz_val = int((obs_val[ir] != 0).sum())
    vaz_flg = int((obs_flg[ir] != 0).sum())
    if vaz_val or vaz_flg:
        raise AssertionError(
            f"{quad}: G2 VIOLADA — {vaz_val} valores e {vaz_flg} flags nao nulos no "
            f"canal de observacao sobre nos da reserva")

    # Effective spatial separation: smallest distance between a reserve and a training node.
    # The GUARD is in CELLS, because the partition separates in cells; only the conversion
    # to metres uses the true cell size (30.922 m, not 30 m).
    from scipy.spatial import cKDTree
    pos = dd["pos_anc"]
    m_tr = np.isin(lab, (0, 1, 2))
    dmin_cel = float(cKDTree(pos[m_tr]).query(pos[m_res], k=1, p=np.inf)[0].min())
    if dmin_cel < BUFFER_CEL - 1e-6:
        raise AssertionError(
            f"{quad}: buffer violado — reserva a {dmin_cel:.0f} celulas "
            f"({dmin_cel * CELULA_M:.0f} m) do treino, minimo exigido "
            f"{BUFFER_CEL} celulas ({BUFFER_CEL * CELULA_M:.0f} m)")

    return {"n_reserva": n_res, "canal_obs_zerado_na_reserva": True,
            "dist_chebyshev_min_reserva_treino_celulas": dmin_cel,
            "dist_chebyshev_min_reserva_treino_m": dmin_cel * CELULA_M,
            "buffer_exigido_m": BUFFER_CEL * CELULA_M}


# ─────────────────────── training one quadrant ───────────────────────
def preparar_tensores(dd: dict, lab: np.ndarray, quad: str,
                      sem_obs: bool = False) -> dict:
    """Build X with the observation channels and check the guarantees before training.

    `sem_obs=True` keeps both columns but ZEROES them: the decisive test of the
    observation channel. The error-versus-distance-to-anchor curve is FLAT (ratio 1.011
    between 0-100 m and >1000 m). If the model propagated labels, error would grow with
    distance; even within 100 m, reachable in 2 hops, there is no gain. Zeroing the
    channel and checking whether anything changes decides whether it contributes.

    The columns remain so that `d_in` is unchanged: capacity is matched, and the measured
    difference is attributable to INFORMATION rather than network size.
    """
    N, idx, y = dd["n_nodes"], dd["idx"], dd["y"]
    m_obs = lab == 0

    obs_val = np.zeros(N, dtype=np.float32)
    obs_flg = np.zeros(N, dtype=np.float32)
    # G4: scale measured ONLY on the observed training anchors. Mean |y| is a LABEL
    # statistic; computed over the whole set it would leak the reserve via normalization.
    esc = float(np.abs(y[m_obs]).mean()) or 1.0
    if not sem_obs:
        obs_val[idx[m_obs]] = y[m_obs] / esc
        obs_flg[idx[m_obs]] = 1.0

    garantias = verificar_garantias(lab, obs_val, obs_flg, idx, dd, quad)
    garantias["escala_obs_de_treino"] = esc

    X = torch.from_numpy(np.concatenate(
        [dd["X"], obs_val[:, None], obs_flg[:, None]], axis=1))
    yfull = torch.zeros(N, dtype=torch.float32)
    yfull[idx] = torch.from_numpy(y)

    # PER-ANCHOR WEIGHT: label uncertainty carried into the loss.
    #
    # `dd["qualidade"]` comes from `rotulos_lidar_v23.py` as `1/(1 + dispersion/4)`: it
    # halves for every 4 m of disagreement among the six GEDI ground estimates, a scale
    # taken from the measured error growth over bare soil. Rejected anchors were already
    # removed from `lab` by `aplicar_qualidade`, so every live weight is positive.
    #
    # Rationale: label error is not random but concentrated. GEDI fails to find the ground
    # under closed canopy, exactly where DEM error is largest; with equal weights the model
    # becomes biased there, not merely noisier. The median weight is 0.53 in Amazonia versus
    # 0.97 in the Cerrado.
    #
    # Outside the anchors the weight is zero: those nodes are not in the label loss.
    wfull = torch.zeros(N, dtype=torch.float32)
    qual = dd.get("qualidade")
    if qual is not None:
        wfull[idx] = torch.from_numpy(np.asarray(qual, dtype=np.float32))
        garantias["peso_rotulo"] = {
            "origem": "dd['qualidade'] — 1/(1 + dispersao_solo/4)",
            "mediana_nas_ancoras": round(float(np.median(qual[qual > 0])), 4)
            if (qual > 0).any() else None,
            "min": round(float(qual[qual > 0].min()), 5) if (qual > 0).any() else None,
        }
    else:
        # the v22 vector has no quality field: weight 1 at every anchor reproduces the plain L1
        wfull[idx] = 1.0
        garantias["peso_rotulo"] = {"origem": "ausente — todas as ancoras com peso 1"}

    # The SOURCE BOOST overrides the quality weight: GEDI=1, ATL08=K. Mixing both schemes
    # would make the effect unattributable. `l1_ponderada` acts only when PONDERAR is on;
    # `main` turns it on when the boost is requested.
    if BOOST_ATL08 > 0 and dd.get("fonte_anc") is not None:
        f = np.asarray(dd["fonte_anc"])
        wfull[idx] = torch.from_numpy(
            np.where(f >= 2, float(BOOST_ATL08), 1.0).astype(np.float32))
        garantias["peso_rotulo"] = {
            "origem": f"boost por fonte — ATL08 x{BOOST_ATL08:g}, GEDI x1",
            "n_atl08": int((f >= 2).sum()), "n_gedi": int((f == 1).sum())}

    from torch_geometric.data import Data
    # The v22 vector's `edge_index` came from the `.pt` as a tensor; V23's comes from
    # `grade_v23.arestas` as NumPy on purpose (the grid module does not import torch, so it
    # can be used without CUDA). The conversion happens here; without it `NeighborLoader`
    # fails inside pyg_lib's `index_sort` with an uninformative error.
    ei = dd["edge_index"]
    if not torch.is_tensor(ei):
        ei = torch.from_numpy(np.ascontiguousarray(ei)).long()
    data = Data(x=X, edge_index=ei.contiguous(), y=yfull, w=wfull)
    del obs_val, obs_flg
    return {"data": data, "d_in": X.shape[1], "garantias": garantias}


def mascara(N: int, idx: np.ndarray, sel: np.ndarray) -> torch.Tensor:
    m = torch.zeros(N, dtype=torch.bool)
    m[idx[sel]] = True
    return m


def fan_out(camadas: int) -> list:
    # VIZ_EXATA: on an 8-connected LATTICE the k-hop ball is bounded by (2k+1)^2
    # nodes, so neighbour sampling is unnecessary. The EXACT neighbourhood gives the
    # same number of nodes per batch in 7x less time at k=2, and removes the
    # sampling-budget vs. reach confound in the depth study. At k=8 with batch 6144
    # the peak is 10.63 GB (fits in 15.9 GB).
    if VIZ_EXATA:
        return [-1] * max(camadas, 1)
    return [-1] if camadas == 0 else (
        [8] * camadas if camadas <= 2 else [8, 8] + [4] * (camadas - 2))


# ── SHARED NEIGHBOUR SAMPLER ──
#
# `NeighborLoader` builds a CSC index of the graph at EVERY instantiation. A single
# quadrant has 117 million edges; the union of four has 467 million, and a growing-window
# run creates about 25 loaders (two per stage for training, one per evaluation). The
# graph does not change between them, only `input_nodes`, so the sampler is built once.
# Verified against the default path: same seeds, same `n_id` and same `x` in the first
# batch; it is the same object the loader would build itself.
#
# ONE-SLOT CACHE holding a STRONG reference to `Data`. A cache keyed by `id(data)` without
# a reference would be a silent bug: CPython reuses addresses after GC, so Q2's `Data`
# could land at the freed address of Q1's and receive Q1's sampler (wrong neighbourhoods,
# no error). Holding the object prevents address reuse; one slot bounds memory, since the
# chain switches graph per quadrant and the growing window has a single graph.
_AMOSTRADOR: dict = {"data": None, "viz": None, "am": None, "indisponivel": False}


def carregador(data, viz: list, input_nodes, shuffle: bool):
    """`NeighborLoader` over the shared sampler, falling back to the default path.

    The fallback exists because `neighbor_sampler` is a PyG version detail. If it
    disappears in an update, the run is slower but still CORRECT, and the warning is
    emitted once, not per batch.
    """
    from torch_geometric.loader import NeighborLoader
    comum = dict(num_neighbors=viz, input_nodes=input_nodes, batch_size=BATCH,
                 shuffle=shuffle, num_workers=0)
    if _AMOSTRADOR["indisponivel"]:
        return NeighborLoader(data, **comum)
    if _AMOSTRADOR["data"] is not data or _AMOSTRADOR["viz"] != list(viz):
        _AMOSTRADOR.update(data=None, viz=None, am=None)     # release the previous graph
        try:
            from torch_geometric.sampler import NeighborSampler
            t0 = time.time()
            am = NeighborSampler(data, num_neighbors=viz)
            log(f"    amostrador montado em {time.time()-t0:.1f} s "
                f"({data.edge_index.shape[1]:,} arestas, fan_out {viz})")
            _AMOSTRADOR.update(data=data, viz=list(viz), am=am)
        except Exception as e:                       # noqa: BLE001 — ver docstring
            _AMOSTRADOR["indisponivel"] = True
            log(f"    amostrador compartilhado indisponivel ({type(e).__name__}: "
                f"{str(e)[:80]}); cada loader monta o seu")
            return NeighborLoader(data, **comum)
    return NeighborLoader(data, neighbor_sampler=_AMOSTRADOR["am"], **comum)


def limpar_amostradores() -> None:
    """Release the reference to the graph; otherwise the 4-quadrant union stays alive."""
    _AMOSTRADOR.update(data=None, viz=None, am=None)


def inferir(modelo, loader, n_alvo: int) -> np.ndarray:
    modelo.eval()
    out = []
    with torch.no_grad():
        for bt in loader:
            bt = bt.to(DEV)
            out.append(modelo(bt.x, bt.edge_index)[:bt.batch_size].cpu().numpy())
    return np.concatenate(out)[:n_alvo]


# --------------------------------------------------------------- physical constraints
#
# Water and bare soil are LOSSES that backpropagate to the same `Delta`, not only
# post-training metrics (as in `metricas_v23.py`).
#
#   bare soil   no canopy, DSM = DTM, so the correction must be ZERO there.
#   water       a water surface is equipotential, so `z'` must be CONSTANT there.
#
# Measured support: median residual on bare soil of -0.11 m in Amazonia and -0.18 m in
# the Pantanal, with none of this given to the estimator.
#
# Height difference rather than slope: `Data` is built without `edge_attr`, so edge
# length does not reach the loop. Between adjacent DEM nodes it is ~30 m orthogonal and
# ~42 m diagonal; the factor is absorbed in `peso_agua`, and the target is zero anyway.
# Imposing `|z'_i - z'_j| -> 0` on water edges imposes zero slope up to that constant.
#
# Membership enters as a WEIGHT, not a threshold: `regime.py` yields soft membership and
# thresholding would discard its gradation.
# Constraint weights live at module level because `rodar_bioma` does not receive `args`
# (same pattern as `EPOCAS_ESCADA` and `LR_ESCADA`). Zero by default: without
# `--peso-solo`/`--peso-agua` training is bit-identical to the unconstrained run.
PESO_SOLO = 0.0
PESO_AGUA = 0.0
# Weight the loss by label uncertainty. ON by default: without it, the relaxed label
# criterion (which tripled the number of labels) would trade precision for quantity.
# `--sem-ponderacao` turns it off for the ablation arm.
PONDERAR = True
# Weight of the track-slope constraint. Both terms are in METRES and normalized by their
# own counts, so 1.0 means "same importance as the altimetric target". A starting
# point, not a calibration.
PESO_DECL = 1.0


def l1_ponderada(p, y, w, lossf):
    """L1 with per-anchor weights, normalized by the sum of the weights.

    Normalizing by `w.sum()` rather than `n` keeps the scale comparable to the plain mean
    when weights are near 1; otherwise weighting would lower the mean gradient, acting as
    a learning-rate cut and mixing two effects.

    With `w` None it falls back to the plain L1, so the unweighted arm is bit-identical.
    """
    if w is None:
        return lossf(p, y)
    s = w.sum()
    if s <= 0:
        return lossf(p, y)
    return (w * (p - y).abs()).sum() / s


def perda_fisica(p_todos, bt, elev_std: float, col_reg: tuple,
                 peso_solo: float, peso_agua: float, peso_dossel: float = 0.0):
    """Water and bare-soil terms. Returns `(termo, diagnostico)`.

    `p_todos` is the model output at ALL subgraph nodes, not only the seeds, which lets
    the physics act away from the anchors, where it matters.

    THE TERMS ARE ALWAYS COMPUTED, EVEN WITH ZERO WEIGHT.
    ----------------------------------------------------
    The raw value of each constraint is needed for calibration. On bare soil the
    residual is ~0.1 m while the label loss is several metres, so a weight of 1.0 would
    make the constraint negligible. Measuring the raw value in the zero-weight phase
    gives the initial weight as a RATIO, `peso = fracao_alvo * L_rotulo / L_restricao`,
    around which the OFAT sweep explores.
    With zero weight the term is computed under `no_grad` and does not enter the loss: it
    costs one extra pass over the batch and does not touch the gradient.
    """
    a, _ = col_reg
    m_agua = bt.x[:, a].clamp(0.0, 1.0)          # NOMES_REGIME[0]
    m_solo = bt.x[:, a + 1].clamp(0.0, 1.0)      # NOMES_REGIME[1]
    termo = p_todos.new_zeros(())
    diag = {}
    ativo = False

    den = m_solo.sum()
    if den > 0:
        if peso_solo > 0:
            l_solo = (m_solo * p_todos.abs()).sum() / den
            termo = termo + peso_solo * l_solo
            ativo = True
        else:
            with torch.no_grad():
                l_solo = (m_solo * p_todos.detach().abs()).sum() / den
        diag["solo"] = float(l_solo.detach())
        diag["solo_cobertura"] = float(den.detach() / max(m_solo.numel(), 1))

    # ── `Delta >= 0` UNDER CANOPY.
    #
    # `Delta = DSM - ground`. The TanDEM-X DSM phase centre lies between the ground and the
    # canopy top, so under vegetation the ground is BELOW the surface and `Delta` is
    # positive by physics. `Delta < 0` there means the model put the ground ABOVE what the
    # radar saw, which is impossible.
    #
    # This adds the SIGN constraint under canopy, where ~90% of nodes lie. In a pilot run
    # `Delta` was negative in 24.4% of Q3 cells and 21.2% of Q4, versus 1.1% in Q1, and the
    # worst quadrants had the BEST anchor skill: anchor metrics alone would not reveal it.
    #
    # HINGE, not quadratic: only the wrong side is penalized. A large positive `Delta` is
    # legitimate (tall canopy); penalizing it would teach the model to underestimate canopy.
    m_dossel = bt.x[:, a + 2].clamp(0.0, 1.0)    # NOMES_REGIME[2] = dossel_denso
    den = m_dossel.sum()
    if den > 0:
        if peso_dossel > 0:
            l_dossel = (m_dossel * torch.relu(-p_todos)).sum() / den
            termo = termo + peso_dossel * l_dossel
            ativo = True
        else:
            with torch.no_grad():
                l_dossel = (m_dossel * torch.relu(-p_todos.detach())).sum() / den
        diag["dossel"] = float(l_dossel.detach())
        diag["dossel_frac_negativo"] = float(
            ((p_todos.detach() < 0).float() * m_dossel).sum() / den)

    i, j = bt.edge_index
    w = m_agua[i] * m_agua[j]                    # an edge counts only if BOTH ends are water
    den = w.sum()
    if den > 0:
        # z' = DSM - Delta, with DSM in metres: x[:,0] is the normalized elevation.
        dz_dsm = (bt.x[i, 0] - bt.x[j, 0]) * elev_std
        if peso_agua > 0:
            dz_corr = dz_dsm - (p_todos[i] - p_todos[j])
            l_agua = (w * dz_corr.abs()).sum() / den
            termo = termo + peso_agua * l_agua
            ativo = True
        else:
            with torch.no_grad():
                pd = p_todos.detach()
                l_agua = (w * (dz_dsm - (pd[i] - pd[j])).abs()).sum() / den
        diag["agua"] = float(l_agua.detach())
        diag["agua_cobertura"] = float(den.detach() / max(w.numel(), 1))

    return (termo if ativo else None), diag


def perda_suavidade(p_todos, bt):
    """Total variation of `Delta` over subgraph edges: `mean(|p_i - p_j|)`.

    Acts on ALL subgraph nodes, like the physics terms: it matters where there are no
    anchors. Edges are duplicated (both directions); the mean absorbs the factor 2.
    Always computed, even with zero weight, so the raw value is measured and the weight
    for 10% of the label loss can be derived, as for `dossel`.
    """
    src, dst = bt.edge_index[0], bt.edge_index[1]
    return (p_todos[src] - p_todos[dst]).abs().mean()


def perda_declividade(p_todos, bt, decl: dict, mapa: torch.Tensor):
    """Directional-derivative constraint, at the scale at which slope was measured.

    Track slope is the component along ONE direction, over a ~114 m baseline. As a scalar
    target it would teach the model to underestimate; with the direction it becomes an
    exact constraint on the difference of `Delta` between the two nodes the track crosses:

        | (p[i+] - p[i-]) - alvo_dp |     with alvo_dp precomputed in `vetor_v23`

    `mapa` maps ORIGINAL node index to position in the batch. It is filled per batch and
    reset to -1 afterwards; allocating one per batch would cost ~104 MB of churn.

    Not every pair fits in the batch: the two nodes are 1-2 cells from the centre at the
    median, but the p95 is farther and neighbour sampling reaches two hops. Pairs outside
    are ignored and the term is normalized by those included, so its value does not
    depend on how many fit.
    """
    n_id = bt.n_id
    mapa[n_id] = torch.arange(n_id.numel(), device=n_id.device)
    try:
        pos_mais = mapa[decl["i_mais"]]
        pos_menos = mapa[decl["i_menos"]]
        val = (pos_mais >= 0) & (pos_menos >= 0)
        n = int(val.sum())
        if n == 0:
            return None, 0
        dp = p_todos[pos_mais[val]] - p_todos[pos_menos[val]]
        return (dp - decl["alvo_dp"][val]).abs().mean(), n
    finally:
        mapa[n_id] = -1


def treinar_quadrante(modelo, prep: dict, dd: dict, lab: np.ndarray, quad: str,
                      estado_opt: dict | None, epocas: int, lr: float,
                      col_reg: tuple | None = None,
                      peso_solo: float = 0.0, peso_agua: float = 0.0,
                      com_curriculo: bool = False,
                      paciencia: int | None = None,
                      grupos: list | None = None) -> dict:
    """Train one STAGE. Returns the optimizer state for the next one (resume mode).

    Early stopping uses ONLY label 1 (validation within the training region). G3: the
    reserve does not enter here in any form.

    `quad` is only the stage LABEL in the log and JSON: the quadrant name in the chain,
    the whole window in the growing window ("stage 2: Q1+Q2"). The trained set is
    defined by `lab`; nothing here reads geography, so one function serves both regimes.

    `com_curriculo` ENABLES the 5-phase ramp. The caller decides: a COLD start justifies
    it, not any particular quadrant.

    `grupos` is a list of `(name, boolean_mask)` that BLOCKS the batches: each group gets
    its own loader and batches are drawn round-robin, one per group per round, in list
    order. Without it there is a single loader over `lab == 0` and batches are MIXED.
    The trained population is the same in both cases and each anchor is seen once per
    epoch; only the grouping changes, and with it the spatial correlation of the
    gradient. See BATCH COMPOSITION at the top of the file.
    """
    data, N, idx = prep["data"], dd["n_nodes"], dd["idx"]
    viz = fan_out(modelo._camadas_b2)

    def loader(sel, shuffle):
        return carregador(data, viz, mascara(N, idx, sel), shuffle)

    # ── WEIGHT DECAY ON MATRICES ONLY.
    #
    # `AdamW(params, lr=lr)` applies `weight_decay = 0.01` by DEFAULT to ALL parameters.
    # Some parameters should not decay, as the fidelity diagnostics (`C_fidelidade`) show:
    #
    #   HEAD BIAS. The target is in METRES (median 13.8 m) and the features are near unit
    #   scale, so the output bias must reach ~13; decay pulls it toward zero at every step.
    #   Measured: `vies_mediano` of -1.38 m, a systematic underestimate.
    #
    #   HEAD WEIGHTS. Decayed, they compress the output range. Measured:
    #   `razao_dispersao` 0.438 and `inclinacao_alvo_sobre_pred` 0.804 (the target varies
    #   1.25 m per predicted metre), the signature of regression to the mean.
    #
    #   GATE MATRIX `A`. Decayed toward zero, it ATTENUATES every branch: gated arms had a
    #   dispersion ratio of 0.07 versus 0.32 for the plain arm; the gate was switching
    #   branches off rather than selecting.
    #
    # Decay is meant to regularize transformation matrices, not to shrink offsets or
    # physical priors; separating the parameter groups is standard practice.
    sem_decaimento, com_decaimento = [], []
    for nome, par in modelo.named_parameters():
        if not par.requires_grad:
            continue
        (sem_decaimento if (par.ndim <= 1 or nome.endswith(".A") or nome == "porta.A")
         else com_decaimento).append(par)
    opt = torch.optim.AdamW(
        [{"params": com_decaimento, "weight_decay": DECAIMENTO},
         {"params": sem_decaimento, "weight_decay": 0.0}], lr=lr)
    if estado_opt is not None:
        try:
            opt.load_state_dict(estado_opt)
            for g in opt.param_groups:      # lr warm restart, momentum preserved
                g["lr"] = lr
            herdou_opt = True
        except ValueError:
            herdou_opt = False              # incompatible shape; continue with a fresh optimizer
    else:
        herdou_opt = False
    # CosineAnnealingLR reinstantiated per quadrant = one cycle per quadrant
    # (warm restart).
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epocas, eta_min=LR_MIN)
    lossf = nn.L1Loss()

    # ── TRAINING BATCHES: one loader (mixed) or one per group (blocked, round-robin).
    ld_tr = None if grupos else loader(lab == 0, True)
    lds_gr = [(nome, loader(sel, True)) for nome, sel in grupos] if grupos else []

    def lotes_treino():
        """Batches for one epoch. In blocked mode, one per group per round.

        Groups differ in size (60 to 76 thousand anchors per quadrant), so loaders run
        out at different rounds. An exhausted group leaves the rotation and the others go
        on, preserving "each anchor seen once per epoch". Truncating at the smallest group
        would drop training anchors and give the blocked arm less data than the mixed
        one, invalidating the comparison.
        """
        if not grupos:
            yield from ld_tr
            return
        its = [iter(ld) for _, ld in lds_gr]
        while its:
            vivos = []
            for it in its:
                try:
                    lote = next(it)
                except StopIteration:
                    continue
                yield lote
                vivos.append(it)
            its = vivos

    ld_va = loader(lab == 1, False)
    if grupos:
        log(f"    lotes BLOCADOS em rodizio: "
            + " -> ".join(f"{n}({int(s.sum()):,})" for n, s in grupos))
    melhor, melhor_est, curva = float("inf"), None, []
    t0 = time.time()
    fis_diag = {}
    usa_fisica = col_reg is not None and (peso_solo > 0 or peso_agua > 0)
    if com_curriculo:
        log(f"    curriculo de {len(FASES)} fases: "
            + " -> ".join(f"{n}({e})" for n, e, _ in FASES))

    # Slope constraint: tensors moved to the device once, plus the node-index-to-batch
    # position map, reused at every iteration.
    decl = None
    mapa_no = None
    decl_diag = {}
    r = dd.get("decl_trilha")
    # Built whenever the data exist, REGARDLESS of the weight, so the raw value is
    # measured in the zero-weight phase, which provides the sweep's starting point.
    # With zero weight it is computed but not added to the loss.
    if r is not None and r.get("n", 0) > 0:
        decl = {"i_mais": torch.as_tensor(r["i_mais"], dtype=torch.long, device=DEV),
                "i_menos": torch.as_tensor(r["i_menos"], dtype=torch.long, device=DEV),
                "alvo_dp": torch.as_tensor(r["alvo_dp_m"], dtype=torch.float32,
                                           device=DEV)}
        mapa_no = torch.full((dd["n_nodes"],), -1, dtype=torch.long, device=DEV)
        decl_diag = {"restricoes": int(r["n"]), "sep_mediana_m": r.get("sep_mediana_m"),
                     "frac_negativa": r.get("frac_negativa"), "peso": PESO_DECL}
    # ── WEIGHT CALIBRATION: per-epoch mean of each term's RAW value.
    #
    # The UNWEIGHTED value of each constraint is recorded next to the label loss at every
    # epoch, including zero-weight epochs. The initial weight of each constraint is
    # derived from it as a ratio:
    #
    #     peso = fracao_alvo * L_rotulo / L_restricao
    #
    # Both terms are then in the same unit (metres), so it is clear whether weight 1 is
    # negligible or dominant.
    #
    # Weights are not selected by MAE: MAE is measured at the ANCHORS, where the LiDAR
    # label is already the best evidence; the constraints help where there are NO
    # anchors (97% of the box), which MAE cannot see. Minimizing MAE always selects zero.
    # The selection criterion is in `raiox_geomorfologico.py`; MAE is a guardrail with a
    # declared tolerance, not the objective.
    calib: list = []
    n_pulados = 0
    # MELHORA_MINIMA keeps validation noise from counting as progress: 1 cm on a target
    # whose median is ~13 m.
    MELHORA_MINIMA = 0.01
    # Patience follows the stage CAP (see `paciencia_da_etapa`): with caps of 14 to 25
    # epochs, the chain's patience of 12 would never trigger.
    pac = PACIENCIA if paciencia is None else int(paciencia)
    sem_melhora, melhor_ep, epocas_usadas = 0, 0, epocas
    melhor_abs = float("inf")     # best ABSOLUTE validation; see the selection block
    for ep in range(epocas):
        modelo.train()
        acc = {"rotulo": 0.0, "solo": 0.0, "agua": 0.0, "dossel": 0.0, "decl": 0.0,
               "suave": 0.0}
        cnt = {"rotulo": 0, "solo": 0, "agua": 0, "dossel": 0, "decl": 0, "suave": 0}
        # Curriculum: this epoch's weights, ramped within the phase introducing each one.
        # The values passed to `treinar_quadrante` are TARGETS; these are what act.
        # `suave` is in no FASES phase, so under the curriculum it stays zero until the
        # curriculum ends and then enters at full value: smoothness only makes sense once
        # the field has a shape.
        alvos_ep = {"solo": peso_solo, "agua": peso_agua, "dossel": PESO_DOSSEL,
                    "decl": PESO_DECL, "suave": PESO_SUAVE}
        w_ep = (pesos_da_epoca(ep, True, alvos_ep) if com_curriculo
                else {**alvos_ep,
                      "_fase": "pos_curriculo" if USA_CURRICULO else "sem_curriculo"})
        p_solo_ep, p_agua_ep = w_ep["solo"], w_ep["agua"]
        p_dossel_ep, p_decl_ep = w_ep["dossel"], w_ep["decl"]
        p_suave_ep = w_ep.get("suave", PESO_SUAVE)
        n_lote_ep = 0
        for bt in lotes_treino():
            bt = bt.to(DEV)
            # SPILL GUARD: on Windows the driver does not raise OOM; it silently spills to
            # host RAM and the run slows down sharply (measured: 0.47 s/step became 3.37 s).
            # Checking only at the end of the stage is too late; abort on the first batch
            # that exceeds the cap.
            n_lote_ep += 1
            if TETO_VRAM_GB > 0 and n_lote_ep % 10 == 0:
                pico_ag = torch.cuda.max_memory_allocated() / 2 ** 30
                if pico_ag > TETO_VRAM_GB:
                    raise RuntimeError(
                        f"VRAM de pico {pico_ag:.2f} GB passou do teto "
                        f"{TETO_VRAM_GB:g} GB no lote {n_lote_ep} da epoca {ep}. "
                        f"Reduza --batch (receita medida: k=8 -> 3072) ou a "
                        f"profundidade.")
            opt.zero_grad()
            # Output at ALL subgraph nodes. The LiDAR L1 uses only the seeds (anchors); the
            # physics uses the whole subgraph, where water and bare soil occur with no anchor
            # at all, exactly the case it is meant to cover.
            with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=USA_AMP):
                p_todos = modelo(bt.x, bt.edge_index)
                perda = l1_ponderada(p_todos[:bt.batch_size], bt.y[:bt.batch_size],
                                     bt.w[:bt.batch_size] if PONDERAR else None, lossf)
                acc["rotulo"] += float(perda.detach()); cnt["rotulo"] += 1
                # Always called, even with both weights at zero: this call MEASURES.
                if col_reg is not None:
                    extra, d = perda_fisica(p_todos, bt, dd["elev_std"], col_reg,
                                            p_solo_ep, p_agua_ep, p_dossel_ep)
                    if extra is not None:
                        perda = perda + extra
                        fis_diag = d
                    for k in ("solo", "agua", "dossel"):
                        if k in d:
                            acc[k] += d[k]; cnt[k] += 1
                # Smoothness: always measured; enters the loss only with positive weight.
                if p_suave_ep > 0:
                    ts = perda_suavidade(p_todos, bt)
                    perda = perda + p_suave_ep * ts
                else:
                    with torch.no_grad():
                        ts = perda_suavidade(p_todos.detach(), bt)
                acc["suave"] += float(ts.detach()); cnt["suave"] += 1
                if decl is not None:
                    td, nd = perda_declividade(p_todos, bt, decl, mapa_no)
                    if td is not None:
                        acc["decl"] += float(td.detach()); cnt["decl"] += 1
                        if p_decl_ep > 0:
                            perda = perda + p_decl_ep * td
                        decl_diag["ultimo_lote"] = {
                            "pares": nd, "erro_m": round(float(td.detach()), 4)}
            # A BATCH WITH NaN OR Inf IS SKIPPED, NOT PROPAGATED. bfloat16's short mantissa
            # makes this occasional, and a single bad batch poisons the weights: a NaN
            # gradient contaminates every parameter it touches.
            if not torch.isfinite(perda):
                n_pulados += 1
                if n_pulados <= 3:
                    log(f"    AVISO epoca {ep}: perda nao finita, lote pulado "
                        f"({n_pulados}o)")
                continue
            perda.backward()
            # Clip the gradient norm before the step: in bf16 a large gradient keeps
            # its float32 exponent but reaches the optimizer with a poor mantissa;
            # clipping limits the damage without changing the direction.
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), MAX_NORM)
            opt.step()
        sched.step()

        med = {k: (acc[k] / cnt[k]) if cnt[k] else None for k in acc}
        # Precomputed ratio: the weight at which the constraint would equal 10% of the
        # label loss. A starting point for the sweep (it sets the order of magnitude),
        # not the answer.
        razao = {k: (round(0.10 * med["rotulo"] / med[k], 3)
                     if med.get(k) and med[k] > 1e-9 and med.get("rotulo") else None)
                 for k in ("solo", "agua", "dossel", "decl", "suave")}
        calib.append({"epoca": ep, "fase": w_ep.get("_fase"),
                      "pesos": {**{k: round(w_ep[k], 4)
                                   for k in ("solo", "agua", "dossel", "decl")},
                                "suave": round(p_suave_ep, 4)},
                      **{k: (round(v, 5) if v is not None else None)
                         for k, v in med.items()},
                      "peso_para_10pct": razao})
        if com_curriculo and (ep == 0 or w_ep.get("_fase") !=
                              calib[-2].get("fase") if len(calib) > 1 else True):
            log(f"    fase `{w_ep.get('_fase')}` a partir da epoca {ep} | "
                f"solo {w_ep['solo']:.2f} agua {w_ep['agua']:.2f} "
                f"decl {w_ep['decl']:.3f}")
        modelo.eval()
        # Two validations. The WEIGHTED one selects the best epoch, so that selection
        # optimizes what training minimizes. The UNWEIGHTED MAE goes to the learning
        # curve, since it is the one comparable across runs.
        tot = n = tot_w = som_w = 0.0
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=AMP_DTYPE,
                                                 enabled=USA_AMP):
            for bt in ld_va:
                bt = bt.to(DEV)
                p = modelo(bt.x, bt.edge_index)[:bt.batch_size].float()
                e = (p - bt.y[:bt.batch_size]).abs()
                tot += float(e.sum())
                n += bt.batch_size
                if PONDERAR:
                    w = bt.w[:bt.batch_size]
                    tot_w += float((w * e).sum())
                    som_w += float(w.sum())
        va = tot / max(n, 1)
        va_sel = (tot_w / som_w) if (PONDERAR and som_w > 0) else va
        curva.append(round(va, 4))
        # SELECTION and PATIENCE use DIFFERENT criteria:
        #   selection -> ANY improvement saves the weights (a better model is never
        #                discarded because the gain is small);
        #   patience  -> only an improvement ABOVE the threshold resets the counter,
        #                so validation noise does not keep the run alive.
        if va_sel < melhor_abs:
            melhor_abs = va_sel
            melhor_ep = ep
            melhor_est = {k: v.detach().cpu().clone()
                          for k, v in modelo.state_dict().items()}
        if va_sel < melhor - MELHORA_MINIMA:
            melhor = va_sel
            sem_melhora = 0
        else:
            sem_melhora += 1

        # ── EPOCH PANEL: header with phase and epoch, an aligned table and the lr.
        # With a single target, each row is one TERM of the loss.
        #
        # Terms are ALWAYS shown, even at zero weight, with the weight alongside: a
        # constraint that is measured but not applied is still informative.
        if PAINEL:
            fase = w_ep.get("_fase", "-")
            marca = "  <- melhor" if sem_melhora == 0 else (
                f"  ({sem_melhora}/{pac} sem melhora)" if sem_melhora else "")
            log(f"")
            log(f"  [{quad} | {fase}] Ep {ep+1}/{epocas} | val {va:.4f} m | "
                f"melhor {melhor:.4f} m{marca}")
            log(f"  {'termo':<16} {'valor':>9} {'peso':>8} {'contrib':>9} "
                f"{'p/ 10%':>8}")
            log("  " + "-" * 54)
            pesos_ep = {"rotulo": 1.0, "solo": p_solo_ep, "agua": p_agua_ep,
                        "dossel": p_dossel_ep, "decl": p_decl_ep,
                        "suave": p_suave_ep}
            for termo in ("rotulo", "solo", "agua", "dossel", "decl", "suave"):
                v = med.get(termo)
                if v is None:
                    log(f"  {termo:<16} {'---':>9} {'---':>8} {'---':>9} {'---':>8}")
                    continue
                w = pesos_ep[termo]
                contrib = w * v
                alvo10 = razao.get(termo)
                log(f"  {termo:<16} {v:>9.4f} {w:>8.3f} {contrib:>9.4f} "
                    f"{(f'{alvo10:.3f}' if alvo10 else '---'):>8}")
            log("  " + "-" * 54)
            log(f"  LR: {sched.get_last_lr()[0]:.2e} | lotes pulados: {n_pulados} | "
                f"{'bf16' if USA_AMP else 'fp32'}")


        # ── EARLY STOPPING. The floor is the end of the curriculum plus a margin: each
        # phase adds a loss term and validation worsens at the phase boundary BY
        # CONSTRUCTION, so counting that as a lack of progress would stop the run
        # exactly when the physics enters.
        piso = ((sum(e for _, e, _ in FASES) if com_curriculo else 0)
                + MIN_EPOCAS_APOS_CURRICULO)
        if ep + 1 >= piso and sem_melhora >= pac:
            log(f"    parada antecipada na epoca {ep+1}/{epocas}: {pac} epocas "
                f"sem melhorar (melhor foi a {melhor_ep+1}, val {melhor:.4f})")
            epocas_usadas = ep + 1
            break
    else:
        epocas_usadas = epocas
    if melhor_est:
        modelo.load_state_dict(melhor_est)

    est_opt = {k: v for k, v in opt.state_dict().items()}
    hist = {"quadrante": quad, "epocas_teto": epocas,
            "epocas_usadas": epocas_usadas, "melhor_epoca": melhor_ep + 1,
            "parou_antes": epocas_usadas < epocas, "paciencia": pac,
            "lr_inicial": lr,
            "lr_final": float(sched.get_last_lr()[0]),
            "val_mae_melhor": melhor_abs,
            "val_mae_limiar_paciencia": melhor,
            "val_mae_e_ponderado": bool(PONDERAR),
            "curva_val_mae": curva,
            "otimizador_herdado": herdou_opt, "treino_s": round(time.time() - t0, 1),
            "fisica": {"ativa": usa_fisica, "peso_solo": peso_solo,
                       "peso_agua": peso_agua, "ultimo_lote": fis_diag},
            "declividade_trilha": decl_diag or {"ativa": False},
            # Per-epoch series of the RAW value of each term (m) and the weight each
            # constraint would need to equal 10% of the label loss, stored in the JSON so
            # that the OFAT sweep starts from a measured ratio.
            "calibragem_por_epoca": calib,
            "lotes_pulados_nao_finitos": n_pulados,
            "composicao_do_lote": ("blocado" if grupos else "misto"),
            "grupos_do_lote": ([n for n, _ in grupos] if grupos else None),
            "precisao": "bfloat16" if USA_AMP else "float32"}
    del ld_tr, ld_va, lds_gr, melhor_est
    return {"estado_opt": est_opt, "hist": hist}


def avaliar(modelo, prep: dict, dd: dict, lab: np.ndarray, rotulo: int,
            escopo: str, com_sigma: bool, pred: np.ndarray | None = None) -> dict:
    """Full metric suite on a subset of anchors.

    Also returns `skill` = 1 - MAE / MAE_trivial, where the TRIVIAL predictor is a
    constant equal to the mean of the observed TRAINING labels, evaluated on the same
    subset (the same reference as test T5 of b1a_anti_leakage). Raw MAE between two
    geographic subsets compares two different difficulties: the held-out corner can
    show a lower MAE than the in-distribution test while lying much farther from any
    observed anchor, simply because its target standard deviation is smaller.

    `test_mae` is comparable between arms only on the SAME subset; across subsets,
    use `skill`.
    """
    from scipy.spatial import cKDTree
    data, N, idx, y = prep["data"], dd["n_nodes"], dd["idx"], dd["y"]
    sel = lab == rotulo
    n = int(sel.sum())
    if n == 0:
        return {"erro": "subconjunto vazio"}
    ld = None
    if pred is None:
        viz = fan_out(modelo._camadas_b2)
        ld = carregador(data, viz, mascara(N, idx, sel), False)
        t0 = time.time()
        p = inferir(modelo, ld, n)
        t_inf = time.time() - t0
    else:
        p, t_inf = pred, 0.0     # external prediction (IDW): no neural inference

    pm = dd["pos_anc_m"]                       # metres, from the actual grid geometry
    dist = cKDTree(pm[lab == 0]).query(pm[sel], k=1)[0]
    ctx = {k: (v[sel] if isinstance(v, np.ndarray) else v)
           for k, v in dd["ctx_anc"].items() if v is not None}
    ctx["dist_ancora_m"] = dist

    sigma = None
    if com_sigma and MC_PASSES > 0 and ld is not None:
        modelo.train()                     # dropout active: MC Dropout
        acum = np.zeros((MC_PASSES, n), dtype=np.float32)
        with torch.no_grad():
            for t in range(MC_PASSES):
                acum[t] = inferir(modelo, ld, n)
        sigma = acum.std(axis=0)
        modelo.eval()
        del acum

    m = MET.suite_completa(y[sel], p, ctx, sigma=sigma, ancora=None,
                           custo={"inferencia_s": round(t_inf, 2),
                                  "us_por_no_inferencia": round(1e6 * t_inf / n, 2)})
    m["test_mae"] = m["A_erro"]["mae"]
    m["n"] = n
    m["dist_mediana_m"] = float(np.median(dist))

    # Trivial reference: constant = mean of the OBSERVED TRAINING labels. Using the
    # subset's own mean (or its standard deviation) would leak, so the constant comes
    # from outside the subset.
    c_triv = float(y[lab == 0].mean())
    mae_triv = float(np.abs(y[sel] - c_triv).mean())
    m["comparabilidade"] = {
        "constante_trivial_de_treino_m": c_triv,
        "mae_trivial_m": mae_triv,
        "razao_contra_trivial": m["test_mae"] / max(mae_triv, 1e-9),
        "skill": 1.0 - m["test_mae"] / max(mae_triv, 1e-9),
        "std_alvo_no_subconjunto_m": float(y[sel].std()),
        "nota": "test_mae so e comparavel entre bracos sobre o MESMO subconjunto; "
                "entre subconjuntos de geografia distinta use skill",
    }
    PREDS_B2.mkdir(parents=True, exist_ok=True)
    # The saved `idx` is ALWAYS LOCAL TO THE QUADRANT. In the growing window `dd` is
    # the disjoint union and `dd["idx"]` is offset by `quadrant_position x
    # n_per_quadrant`; saving it would mix two index conventions in the same outputs
    # (Q1 hides the conflict because its offset is zero) and every per-node join
    # would silently return an empty set. Consumers need the node WITHIN the quadrant,
    # which matches the side tables, `dim_ancora` and the grid. `dd["idx_local"]`,
    # when present, is that translation; otherwise `idx` is already local.
    idx_saida = dd.get("idx_local")
    idx_saida = idx[sel] if idx_saida is None else np.asarray(idx_saida)[sel]
    np.savez_compressed(PREDS_B2 / f"{escopo}.npz",
                        p=p.astype(np.float32), y=y[sel].astype(np.float32),
                        dist=dist.astype(np.float32), idx=idx_saida,
                        idx_grafo=idx[sel],
                        sigma=(sigma if sigma is not None else np.zeros(0, np.float32)))
    del ld
    return m


# ─────────────────────── orchestration ───────────────────────
def _carregar_vetor(vetor: str, bioma: str, quad: str, usar_sar: bool,
                    bloco_por_bioma: bool) -> dict:
    """Single switch between the baseline (v22) and the V23 feature vectors."""
    if vetor == "v22":
        return carregar(bioma, quad, usar_sar=usar_sar,
                        bloco_por_bioma=bloco_por_bioma)
    if vetor == "v23":
        from vetor_v23 import carregar_v23
        return carregar_v23(bioma, quad, usar_sar=usar_sar,
                            bloco_por_bioma=bloco_por_bioma)
    raise ValueError(f"vetor desconhecido: {vetor} (use v22 ou v23)")


def _mascarar_vazio(lab: np.ndarray, dd: dict, quad: str) -> np.ndarray:
    """E1: an anchor (any source, classes 0-3) within VAZIO_KM km of an EBA
    transect centroid becomes CLASS 4 -- it leaves train/val/test/reserve
    and becomes the `vazio` (void) evaluation population. Approximate
    metric distance (30.72 m N-S x 30.87 m E-W at the box center),
    declared beforehand."""
    passo = 2.0 / 7200.0
    l0 = 0 if quad in ("Q1", "Q2") else 3600
    c0 = 0 if quad in ("Q1", "Q3") else 3600
    lin = dd["idx"] // 3600 + l0
    col = dd["idx"] % 3600 + c0
    perto = np.zeros(len(lab), dtype=bool)
    for lon, lat in CENTROIDES_EBA:
        rc = (-2.0 - lat) / passo
        cc = (lon + 60.0) / passo
        d2 = ((lin - rc) * 30.72) ** 2 + ((col - cc) * 30.87) ** 2
        perto |= d2 <= (VAZIO_KM * 1000.0) ** 2
    alvo = perto & (lab >= 0) & (lab <= 3)
    novo = lab.copy()
    novo[alvo] = 4
    log(f"    vazio sintetico R={VAZIO_KM:g} km: {int(alvo.sum()):,} ancoras "
        f"mascaradas -> classe 4 (de {int((lab >= 0).sum()):,} vivas)")
    return novo


def preparar_peca(bioma: str, quad: str, seed: int, bloco_cel: int, vetor: str,
                  usar_sar: bool, bloco_por_bioma: bool, sem_sar: bool, sem_reg: bool,
                  com_inter: bool, sem_obs: bool) -> tuple:
    """Load -> split -> quality filter -> tensors, for ONE quadrant.

    Shared by the chain and the growing window so both use exactly the same path.

    THE ORDER PRESERVES THE PAIRING. The split is drawn on the FULL anchor set, with
    the same seed as the baseline; only then does the quality filter remove anchors
    from training and validation. Test and reserve keep the same population, which
    allows paired comparison.
    """
    dd = _carregar_vetor(vetor, bioma, quad, usar_sar, bloco_por_bioma)
    if SUAVIZAR_X > 0:
        # T1: spatial mean (2r+1)x(2r+1) per column of X. Adjudicates the
        # mechanism "is the graph's advantage just input smoothing?" -- a
        # separate arm, not comparable to raw runs.
        from scipy.ndimage import uniform_filter
        t = 2 * SUAVIZAR_X + 1
        Xs = dd["X"]
        for c in range(Xs.shape[1]):
            Xs[:, c] = uniform_filter(Xs[:, c].reshape(3600, 3600), size=t,
                                      mode="nearest").ravel()
        log(f"    features SUAVIZADAS: janela {t}x{t} (~±{31 * SUAVIZAR_X} m) — "
            f"outro braco")
    if sem_sar or sem_reg:
        dd = zerar_blocos(dd, sem_sar=True, sem_regime=sem_reg)
    if com_inter:
        dd = adicionar_interacoes(dd)
    lab, diag = particao_com_reserva(dd, quad, seed, bloco_cel)
    if "qualidade" in dd:
        from vetor_v23 import aplicar_qualidade
        lab, d_qual = aplicar_qualidade(lab, dd["qualidade"])
        diag["qualidade_rotulo"] = {**d_qual, **dd.get("diag_rotulo", {})}
        # `aplicar_qualidade` returns NESTED keys: `{removidas: {...}, restam: {...}}`.
        rm, rs = d_qual["removidas"], d_qual["restam"]
        log(f"    qualidade: -{rm['treino']:,} treino, -{rm['validacao']:,} val, "
            f"-{rm['teste']:,} teste, -{rm['reserva']:,} reserva | restam "
            f"{rs['treino']:,}/{rs['validacao']:,}/{rs['teste']:,}/{rs['reserva']:,}")
    if VAZIO_KM > 0:
        lab = _mascarar_vazio(lab, dd, quad)
    # Per-anchor source, only when the boost needs it: map node idx -> source from
    # the label side table, aligned with dd["idx"] (still local to the quadrant here,
    # even on the union path, because the piece is built before the offset).
    if BOOST_ATL08 > 0:
        z = np.load(LATDIR_V23 / f"rotulos_{bioma}_{quad}.npz")
        fmap = np.zeros(3600 * 3600, dtype=np.int8)
        fmap[z["idx_no"].astype(np.int64)] = z["fonte"].astype(np.int8)
        z.close()
        dd["fonte_anc"] = fmap[dd["idx"]]
    prep = preparar_tensores(dd, lab, quad, sem_obs=sem_obs)
    return dd, lab, prep, diag


def falta_lateral(bioma: str, quad: str, vetor: str) -> list:
    """What prevents this quadrant from running. Empty list = ready.

    The requirements depend on the VECTOR: v22 reads the `.pt`, whereas V23 builds
    from side tables and derives the grid arithmetically, so requiring the `.pt` for
    V23 would silently skip every quadrant.
    """
    if vetor == "v22":
        return ([] if (QDIR / f"{bioma}_v18.5_{quad}_gpu.pt").exists()
                else ["grafo .pt do v22"])
    return [c for c in ("glo30", "rotulos", "s2", "s1_umida", "s1_seca")
            if not (LATDIR_V23 / f"{c}_{bioma}_{quad}.npz").exists()]


def rodar_bioma(bioma: str, seed: int, arm: str, transferir: bool,
                usar_sar: bool, bloco_por_bioma: bool, vetor: str = "v22") -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    bloco_cel = BLOCO_POR_BIOMA.get(bioma, BLOCO) if bloco_por_bioma else BLOCO
    modo = com_tag("transferencia" if transferir else "isolado")
    # Ablation suffixes: `_inter` = level 2 (explicit products), `_gate` = level 3
    # (gate by physical regime). Combinable. The base name is kept for `construir()`.
    com_inter = "_inter" in arm
    com_porta = "_gate" in arm
    sem_obs = "_semobs" in arm
    sem_sar = "_semsar" in arm
    sem_reg = "_semsarreg" in arm
    arm_base = (arm.replace("_inter", "").replace("_gate", "").replace("_semobs", "")
                .replace("_semsarreg", "").replace("_semsar", ""))
    if com_porta and not arm_base.startswith("gatv2_"):
        raise ValueError(f"porta de regime so implementada sobre gatv2_N; veio {arm}")
    log(f"===== {bioma} | seed {seed} | {arm} | {modo} "
        f"{'| interacoes' if com_inter else ''}{'| porta de regime' if com_porta else ''}"
        f"{'| SEM canal de observacao' if sem_obs else ''}"
        f" =====")

    out = {"bioma": bioma, "seed": seed, "braco": arm, "modo": modo, "vetor": vetor,
           "ordem_quadrantes": ORDEM_QUAD, "epocas": EPOCAS_ESCADA,
           "lr": LR_ESCADA, "quadrantes": {}, "esquecimento": {},
           "layout_reservas": verificar_layout(bioma, ORDEM_QUAD),
           "ablacao": {"interacoes": com_inter, "porta_regime": com_porta,
                       "sem_canal_observacao": sem_obs,
                       "sem_sar": sem_sar, "sem_regime": sem_reg}}

    estado_pesos: dict | None = None
    estado_opt: dict | None = None
    d_in_ref: int | None = None
    labs: dict = {}          # labels per quadrant, to reproduce in the final evaluation

    for quad in ORDEM_QUAD:
        falta = falta_lateral(bioma, quad, vetor)
        if falta:
            log(f"  {quad}: {falta} ausente, pulando")
            continue
        mem_guard(f"pre-carga {bioma}_{quad}")
        dd, lab, prep, diag = preparar_peca(bioma, quad, seed, bloco_cel, vetor,
                                            usar_sar, bloco_por_bioma, sem_sar, sem_reg,
                                            com_inter, sem_obs)
        labs[quad] = lab

        # Input-dimension consistency check: otherwise transfer fails at the first
        # layer and the error only surfaces at Q2.
        if d_in_ref is None:
            d_in_ref = prep["d_in"]
        elif prep["d_in"] != d_in_ref:
            raise ValueError(
                f"{bioma}_{quad}: d_in={prep['d_in']} difere do Q1 ({d_in_ref}). "
                f"Transferencia impossivel — provavel ausencia de SAR em um quadrante.")

        if com_porta:
            grupos, col_reg = layout_grupos(prep["d_in"], com_inter,
                                            dd.get("cols"), dd.get("n_interacoes"))
            camadas = int(arm_base.split("_")[1])
            modelo = GATv2Portao(grupos, col_reg, camadas)
        else:
            modelo, camadas = construir(arm_base, prep["d_in"])
        modelo._camadas_b2 = camadas
        modelo = modelo.to(DEV)
        herdou_pesos = False
        if transferir and estado_pesos is not None:
            modelo.load_state_dict(estado_pesos, strict=False)
            herdou_pesos = True
            log(f"  {quad}: TRANSFER — pesos herdados do quadrante anterior")

        r = treinar_quadrante(modelo, prep, dd, lab, quad,
                              (estado_opt if (transferir and not SEM_ESTADO_OPT)
                               else None),
                              EPOCAS_ESCADA[quad], LR_ESCADA[quad],
                              col_reg=_cols(dd)["regime"],
                              peso_solo=PESO_SOLO, peso_agua=PESO_AGUA,
                              com_curriculo=(USA_CURRICULO and quad == "Q1"))
        r["hist"]["pesos_herdados"] = herdou_pesos

        # ── IDW reference on the same subsets ──
        # On the reserve the median distance to an observed anchor is ~17.8 km, against
        # ~0.5 km on the in-distribution test: a Euclidean interpolator has nothing to
        # propagate at that range, whereas graph message passing does. IDW does not
        # train, so it does not depend on the mode, but it is recomputed per quadrant
        # because the observed anchors change with the seed.
        from b1_kriging_indutivo import idw as _idw
        pos, yv = dd["pos_anc"], dd["y"]
        m_obs_a = lab == 0
        # The vector suffix goes into the prediction FILE NAME, BETWEEN arm and mode,
        # so the `PADRAO` pattern of auditoria_v23.py still separates mode and arm (and
        # `gatv2_2_v23` appears as its own arm). Without it, a --vetor v23 run would
        # overwrite the frozen baseline .npz files, from which all metrics are recomputed.
        suf = "" if vetor == "v22" else f"_{vetor}"
        esc_base = f"{bioma}_{quad}_seed{seed}_{arm}{suf}_{modo}"
        ref_idw = {}
        for rot, nome_sub in ((3, "reserva"), (2, "teste_em_dist"), (4, "vazio")):
            sel_r = lab == rot
            if sel_r.sum() == 0:
                continue
            p_idw = _idw(pos[m_obs_a], yv[m_obs_a], pos[sel_r])
            ref_idw[nome_sub] = avaliar(None, prep, dd, lab, rot,
                                        f"{esc_base}_idw_{nome_sub}",
                                        com_sigma=False, pred=p_idw)
        log(f"  {quad}: IDW reserva MAE={ref_idw['reserva']['test_mae']:.3f} m "
            f"(skill {ref_idw['reserva']['comparabilidade']['skill']:+.3f}) | "
            f"IDW em-dist MAE={ref_idw['teste_em_dist']['test_mae']:.3f} m "
            f"(skill {ref_idw['teste_em_dist']['comparabilidade']['skill']:+.3f})")

        # IMMEDIATE evaluation: this quadrant's reserve + in-distribution test
        res_imediata = avaliar(modelo, prep, dd, lab, 3,
                               f"{esc_base}_reserva_imediata", com_sigma=True)
        res_em_dist = avaliar(modelo, prep, dd, lab, 2,
                              f"{esc_base}_teste_em_dist", com_sigma=False)

        out["quadrantes"][quad] = {
            "particao": diag, "garantias": prep["garantias"], "treino": r["hist"],
            "reserva_imediata": res_imediata, "teste_em_distribuicao": res_em_dist,
            "referencia_idw": ref_idw,
        }
        if com_porta:
            A = modelo.porta.A.detach().cpu().numpy()
            ramos = modelo.porta.ordem_ramos
            prior = np.stack([prior_do_ramo(k) for k in ramos], axis=1)
            out["quadrantes"][quad]["porta_regime"] = {
                "ramos": ramos, "regimes": NOMES_REGIME,
                "prior_herdado_de": {k: HERANCA_PRIOR[k] for k in ramos},
                "aprendida": A.tolist(), "prior_fisico": prior.tolist(),
                "desvio_do_prior": (A - prior).tolist(),
                "desvio_abs_medio": float(np.abs(A - prior).mean()),
                "nota": "quanto a porta se afasta do prior fisico e resultado: mede se "
                        "o dado concorda com a fisica de regime medida em 2026-07-30",
            }
            log("    porta aprendida (linha=regime, coluna=" + "/".join(ramos) + "):")
            for i, rn in enumerate(NOMES_REGIME):
                log(f"      {rn:15s} " + "  ".join(
                    f"{A[i, j]:+.3f}({A[i, j]-prior[i, j]:+.3f})"
                    for j in range(len(ramos))))
        log(f"  {quad}: reserva MAE={res_imediata['test_mae']:.3f} m | "
            f"em-dist MAE={res_em_dist['test_mae']:.3f} m | "
            f"val={r['hist']['val_mae_melhor']:.4f} | {r['hist']['treino_s']:.0f}s")

        estado_pesos = {k: v.detach().cpu().clone()
                        for k, v in modelo.state_dict().items()}
        estado_opt = r["estado_opt"]

        # Persist the weights, so the corrected raster can be generated without
        # retraining. `lab` is reproducible from (bioma, quad, seed, bloco), so these
        # four are enough to rebuild the same observed set at inference.
        CKPT_B2.mkdir(parents=True, exist_ok=True)
        # The vector suffix also goes into the checkpoint name; without it a --vetor
        # v23 run with --arms gatv2_2_semobs would overwrite the frozen baseline
        # weights `{bioma}_seed{n}_gatv2_2_semobs_transferencia_ateQ4.pt`.
        torch.save({"state_dict": estado_pesos, "arm": arm, "camadas": camadas,
                    "vetor": vetor,
                    "d_in": prep["d_in"], "bioma": bioma, "quadrante": quad,
                    "seed": seed, "modo": modo, "bloco_celulas": bloco_cel,
                    "usar_sar": usar_sar, "bloco_por_bioma": bloco_por_bioma,
                    "escala_obs": prep["garantias"]["escala_obs_de_treino"],
                    "val_mae": r["hist"]["val_mae_melhor"],
                    "reserva_mae": res_imediata["test_mae"],
                    "ordem_quadrantes": ORDEM_QUAD,
                    "reserva_por_quad": {k: v[0] for k, v in RESERVA_POR_QUAD.items()}},
                   CKPT_B2 / f"{bioma}_seed{seed}_{arm}{suf}_{modo}_ate{quad}.pt")

        # PEAK VRAM PER QUADRANT: the peak, not the mean, sizes `BATCH` and `fan_out`,
        # and it is only visible in-process (external `nvidia-smi` sampling misses the
        # transient). Reset per quadrant so the value belongs to THAT quadrant.
        if torch.cuda.is_available():
            pico = torch.cuda.max_memory_allocated() / 2 ** 30
            reserv = torch.cuda.max_memory_reserved() / 2 ** 30
            livre, tot = (x / 2 ** 30 for x in torch.cuda.mem_get_info())
            out["quadrantes"][quad]["vram_pico_gb"] = round(float(pico), 3)
            out["quadrantes"][quad]["vram_reservada_gb"] = round(float(reserv), 3)
            log(f"    VRAM: pico {pico:.2f} GB alocado, {reserv:.2f} GB reservado | "
                f"livre {livre:.1f} de {tot:.1f} GB | folga {100*(1-reserv/tot):.0f}%")
            torch.cuda.reset_peak_memory_stats()

        del modelo, prep, r
        for k in list(dd.keys()):
            dd[k] = None
        del dd
        mem_guard(f"pos-{bioma}_{quad}")

    if not out["quadrantes"]:
        return {"erro": "nenhum quadrante disponivel"}

    # ── final pass: end-of-chain model against ALL reserves ──
    # Reloads one graph at a time, the only way to measure forgetting without
    # holding four quadrants in RAM at once.
    log("  --- passe final: modelo do fim da cadeia em todas as reservas ---")
    suf_f = "" if vetor == "v22" else f"_{vetor}"
    for quad in [q for q in ORDEM_QUAD if q in out["quadrantes"]]:
        mem_guard(f"pre-final {bioma}_{quad}")
        dd = _carregar_vetor(vetor, bioma, quad, usar_sar, bloco_por_bioma)
        if sem_sar or sem_reg:
            dd = zerar_blocos(dd, sem_sar=True, sem_regime=sem_reg)
        if com_inter:
            dd = adicionar_interacoes(dd)
        lab = labs[quad]
        prep = preparar_tensores(dd, lab, quad, sem_obs=sem_obs)
        if com_porta:
            grupos, col_reg = layout_grupos(prep["d_in"], com_inter,
                                            dd.get("cols"), dd.get("n_interacoes"))
            modelo = GATv2Portao(grupos, col_reg, int(arm_base.split("_")[1]))
            camadas = int(arm_base.split("_")[1])
        else:
            modelo, camadas = construir(arm_base, prep["d_in"])
        modelo._camadas_b2 = camadas
        modelo = modelo.to(DEV)
        modelo.load_state_dict(estado_pesos, strict=False)
        fin = avaliar(modelo, prep, dd, lab, 3,
                      f"{bioma}_{quad}_seed{seed}_{arm}{suf_f}_{modo}_reserva_final",
                      com_sigma=False)
        r_ime = out["quadrantes"][quad]["reserva_imediata"]
        ime = r_ime["test_mae"]
        out["quadrantes"][quad]["reserva_final"] = fin
        out["esquecimento"][quad] = {
            "mae_imediato_m": ime, "mae_final_m": fin["test_mae"],
            "delta_m": fin["test_mae"] - ime,
            "delta_pct": 100.0 * (fin["test_mae"] - ime) / max(ime, 1e-9),
            # here the subset is the SAME in both measurements, so the MAE delta is
            # valid; skill is reported alongside for consistency
            "skill_imediato": r_ime["comparabilidade"]["skill"],
            "skill_final": fin["comparabilidade"]["skill"],
        }
        log(f"  {quad}: reserva final MAE={fin['test_mae']:.3f} m "
            f"(imediato {ime:.3f} m, delta {fin['test_mae'] - ime:+.3f} m)")
        del modelo, prep
        for k in list(dd.keys()):
            dd[k] = None
        del dd
        mem_guard(f"pos-final {bioma}_{quad}")

    qs = list(out["esquecimento"].keys())
    maes = [out["esquecimento"][q]["mae_final_m"] for q in qs]
    skills = [out["quadrantes"][q]["reserva_final"]["comparabilidade"]["skill"]
              for q in qs]
    ns = [out["quadrantes"][q]["reserva_final"]["n"] for q in qs]
    out["reserva_uniao"] = {
        "mae_media_simples_m": float(np.mean(maes)),
        "mae_media_ponderada_por_n_m": float(np.average(maes, weights=ns)),
        "skill_media_ponderada_por_n": float(np.average(skills, weights=ns)),
        "n_total": int(sum(ns)),
        "nota": "media de MAEs por quadrante; a ponderada por n e a comparavel a um "
                "MAE calculado sobre a uniao dos nos. As quatro reservas tem "
                "variabilidade de alvo distinta, portanto a media de skill e a "
                "agregacao defensavel e a de MAE e apenas descritiva.",
    }

    limpar_amostradores()      # the last quadrant's graph need not survive

    # ── GEOMORPHOLOGICAL X-RAY: the two families that do NOT fit on anchors.
    #
    # Everything above is measured on ~23 thousand anchors out of 12.96 million nodes
    # per quadrant. Two families need the WHOLE terrain:
    #
    #   ADMISSIBILITY  pits without outflow and slopes beyond the angle of repose are
    #                  properties of the terrain, which only exists as a whole.
    #   LOCAL ANGLE    sigma0 depends on the geometry of every cell the radar imaged.
    #                  INDEPENDENT evidence: SAR was not used in the label and the
    #                  label was not used in SAR.
    #
    # They distinguish "hit the target where there is a target" from "produced a
    # plausible terrain".
    if COM_RAIOX:
        try:
            out["G_raiox"] = _raiox_da_corrida(bioma, out, log)
        except Exception as e:
            log(f"  raio-X falhou: {type(e).__name__}: {str(e)[:140]}")
            out["G_raiox"] = {"erro": f"{type(e).__name__}: {str(e)[:200]}"}
    return out


class Uniao:
    """DISJOINT union of the quadrants in a single `Data`, built one at a time.

    WHY INCREMENTAL. Stacking the four `Data` and concatenating at the end would keep
    both copies alive at the peak (17 GB union + 17 GB pieces). Filling preallocated
    buffers and releasing each piece when done keeps the peak at union + ONE quadrant,
    about 22 GB (the `mem_guard` RAM floor is 16 GB free, on a 128 GB machine).

    PREALLOCATION USES THE FIRST QUADRANT'S COUNTS. They are deterministic: the grid
    is 3600x3600 in every quadrant and `grade_v23.arestas` visits the same eight
    neighbours with the same border rule (116,596,804 edges over the four Amazonia
    quadrants). A mismatch is a bug, not variation, so the guard raises instead of
    reallocating.

    NO EDGE CROSSES A BOUNDARY. The index offset separates the blocks and each
    quadrant's `edge_index` is shifted as a whole, so a quadrant's subgraph inside
    the union is ISOMORPHIC to its standalone graph with the same features: inferring
    on the union is inferring on the quadrant, and `superficie_v23`, which loads one
    quadrant at a time, remains valid unchanged.
    """

    def __init__(self, n_quads: int):
        self.n_quads = n_quads
        self.X = self.EI = self.Y = self.W = None
        self.n_por = self.e_por = 0
        self.k_no = self.k_ar = 0
        self.elev_std = None
        self.d_in = None
        self.ordem: list = []
        self.idx: list = []
        self.lab: list = []
        self.quad_de: list = []
        self.pos_m: list = []
        self.y: list = []
        self.ctx: dict = {}
        self.ctx_escalar: dict = {}
        self.dm, self.dn, self.dalvo = [], [], []
        self.garantias: dict = {}
        self.particao: dict = {}
        self.cols = None
        self.n_inter = None

    def adicionar(self, quad: str, dd: dict, lab: np.ndarray, prep: dict,
                  diag: dict) -> None:
        d = prep["data"]
        n, e = int(dd["n_nodes"]), int(d.edge_index.shape[1])
        if self.X is None:
            self.n_por, self.e_por, self.d_in = n, e, int(prep["d_in"])
            self.elev_std = float(dd["elev_std"])
            self.cols, self.n_inter = dd.get("cols"), dd.get("n_interacoes")
            self.X = torch.empty((n * self.n_quads, self.d_in), dtype=d.x.dtype)
            self.EI = torch.empty((2, e * self.n_quads), dtype=torch.long)
            self.Y = torch.zeros(n * self.n_quads, dtype=torch.float32)
            self.W = torch.zeros(n * self.n_quads, dtype=torch.float32)
        else:
            if (n, e) != (self.n_por, self.e_por):
                raise ValueError(
                    f"{quad}: {n:,} nos e {e:,} arestas contra {self.n_por:,} e "
                    f"{self.e_por:,} do primeiro quadrante. A grade e aritmetica e "
                    f"deveria ser identica; divergencia aqui e defeito de geometria.")
            if int(prep["d_in"]) != self.d_in:
                raise ValueError(
                    f"{quad}: d_in={prep['d_in']} contra {self.d_in} do primeiro. "
                    f"Unir vetores de largura diferente nao e uniao, e mistura.")
            # `elev_std` is the global frozen scale (38.1346 in Amazonia). The union's
            # physics uses ONE value; if quadrants disagreed, the ground term would be in
            # different units in different parts of the same batch.
            if abs(float(dd["elev_std"]) - self.elev_std) > 1e-6:
                raise ValueError(
                    f"{quad}: elev_std={dd['elev_std']} contra {self.elev_std}. As "
                    f"escalas deveriam estar congeladas por bioma — veja escalas_v23.json.")

        off = self.k_no
        self.X[off:off + n] = d.x
        self.Y[off:off + n] = d.y
        self.W[off:off + n] = d.w
        torch.add(d.edge_index, off, out=self.EI[:, self.k_ar:self.k_ar + e])
        self.k_no += n
        self.k_ar += e

        self.ordem.append(quad)
        self.idx.append(np.asarray(dd["idx"], dtype=np.int64) + off)
        self.lab.append(np.asarray(lab))
        self.quad_de.append(np.full(len(lab), quad, dtype=object))
        self.pos_m.append(dd["pos_anc_m"])
        self.y.append(np.asarray(dd["y"], dtype=np.float32))
        for k, v in (dd.get("ctx_anc") or {}).items():
            if isinstance(v, np.ndarray):
                self.ctx.setdefault(k, []).append(v)
            elif k not in self.ctx_escalar:
                self.ctx_escalar[k] = v
        r = dd.get("decl_trilha")
        if r is not None and r.get("n", 0) > 0:
            self.dm.append(np.asarray(r["i_mais"], dtype=np.int64) + off)
            self.dn.append(np.asarray(r["i_menos"], dtype=np.int64) + off)
            self.dalvo.append(np.asarray(r["alvo_dp_m"], dtype=np.float32))
        self.garantias[quad] = prep["garantias"]
        self.particao[quad] = diag

    def fechar(self) -> tuple:
        """Return `(ddu, prepu, lab_u, quad_de)`, which the rest of the file consumes."""
        from torch_geometric.data import Data
        if self.X is None:
            raise ValueError("uniao vazia: nenhum quadrante foi adicionado")
        if self.k_no != self.X.shape[0] or self.k_ar != self.EI.shape[1]:
            raise ValueError(
                f"uniao incompleta: {self.k_no:,}/{self.X.shape[0]:,} nos e "
                f"{self.k_ar:,}/{self.EI.shape[1]:,} arestas preenchidos")
        lab_u = np.concatenate(self.lab)
        quad_de = np.concatenate(self.quad_de)
        ddu = {
            "n_nodes": int(self.k_no),
            "idx": np.concatenate(self.idx),
            # The SAME node, indexed within the QUADRANT. `idx` above is the union index
            # needed by `NeighborLoader`; this one matches the side tables. See the note
            # in `avaliar`.
            "idx_local": np.concatenate(self.idx) % self.n_por,
            "y": np.concatenate(self.y),
            "pos_anc_m": np.concatenate(self.pos_m, axis=0),
            "ctx_anc": {**self.ctx_escalar,
                        **{k: np.concatenate(v) for k, v in self.ctx.items()}},
            "elev_std": self.elev_std,
            "cols": self.cols,
            "n_interacoes": self.n_inter,
            "decl_trilha": None,
        }
        if self.dm:
            ddu["decl_trilha"] = {"n": int(sum(len(a) for a in self.dm)),
                                  "i_mais": np.concatenate(self.dm),
                                  "i_menos": np.concatenate(self.dn),
                                  "alvo_dp_m": np.concatenate(self.dalvo)}
        prepu = {"data": Data(x=self.X, edge_index=self.EI.contiguous(),
                              y=self.Y, w=self.W),
                 "d_in": self.d_in,
                 "particao": self.particao,
                 "garantias": {"por_quadrante": self.garantias,
                               **self.conferir_isolamento(lab_u, ddu["idx"], quad_de)}}
        return ddu, prepu, lab_u, quad_de

    def conferir_isolamento(self, lab_u, idx_u, quad_de) -> dict:
        """G5, specific to the union: no edge links quadrants, no reserve is a seed.

        The four guarantees of `verificar_garantias` are PER QUADRANT and still hold
        (`preparar_peca` checked them before the piece entered). The union adds one
        failure mode: with a wrong index offset, an edge would cross the boundary and a
        training node of one quadrant would reach the neighbour's reserve through
        message passing, unnoticed by earlier guards. Checked ARITHMETICALLY on
        `edge_index`.
        """
        bloco_o = self.EI[0] // self.n_por
        bloco_d = self.EI[1] // self.n_por
        cruzam = int((bloco_o != bloco_d).sum())
        if cruzam:
            raise AssertionError(
                f"G5 VIOLADA — {cruzam:,} arestas atravessam a fronteira entre "
                f"quadrantes. A uniao e disjunta por construcao; a costura de bordas e "
                f"mudanca separada, com auditoria de vazamento propria.")
        fora = int(((self.EI < 0) | (self.EI >= self.k_no)).sum())
        if fora:
            raise AssertionError(f"G5 VIOLADA — {fora:,} indices de aresta fora da uniao")
        n_res = {q: int(((quad_de == q) & (lab_u == 3)).sum()) for q in self.ordem}
        vazias = [q for q, v in n_res.items() if v == 0]
        if vazias:
            raise AssertionError(f"reserva vazia na uniao: {vazias}")
        return {"G5_arestas_entre_quadrantes": cruzam,
                "G5_nos_por_quadrante": self.n_por,
                "G5_arestas_por_quadrante": self.e_por,
                "n_reserva_por_quadrante": n_res,
                "nota": "uniao disjunta: o subgrafo de cada quadrante e isomorfo ao "
                        "grafo dele sozinho, logo as garantias por quadrante bastam"}


def rodar_bioma_ampliacao(bioma: str, seed: int, arm: str, usar_sar: bool,
                          bloco_por_bioma: bool, vetor: str = "v23") -> dict:
    """Growing window: a quadrant that enters the loss never leaves it.

    See the growing-window notes at the top of the file for the rationale. Steps:

      1  build the disjoint union of the quadrants (once, not per stage)
      2  at stage k, train with `lab` masked outside the window `ordem[:k]`
      3  evaluate the reserve of EVERY quadrant seen so far, at every stage

    Step 2 makes each stage cheap: the union is not rebuilt, only the SEED nodes
    change. Step 3 tracks the Q1 reserve over the four stages, showing whether the
    growing window resolves forgetting rather than comparing only the endpoints.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    bloco_cel = BLOCO_POR_BIOMA.get(bioma, BLOCO) if bloco_por_bioma else BLOCO
    modo = com_tag("ampliacao_blocado" if LOTE_BLOCADO else "ampliacao")
    com_inter = "_inter" in arm
    com_porta = "_gate" in arm
    sem_obs = "_semobs" in arm
    sem_sar = "_semsar" in arm
    sem_reg = "_semsarreg" in arm
    arm_base = (arm.replace("_inter", "").replace("_gate", "").replace("_semobs", "")
                .replace("_semsarreg", "").replace("_semsar", ""))
    if com_porta and not arm_base.startswith("gatv2_"):
        raise ValueError(f"porta de regime so implementada sobre gatv2_N; veio {arm}")
    suf = "" if vetor == "v22" else f"_{vetor}"

    ordem = [q for q in ORDEM_QUAD if not falta_lateral(bioma, q, vetor)]
    pulados = [q for q in ORDEM_QUAD if q not in ordem]
    if not ordem:
        return {"erro": "nenhum quadrante disponivel"}
    if pulados:
        log(f"  quadrantes sem lateral, fora da janela: {pulados}")
    contig = verificar_contiguidade(ordem)

    log(f"===== {bioma} | seed {seed} | {arm} | JANELA CRESCENTE "
        f"{' -> '.join('+'.join(ordem[:k+1]) for k in range(len(ordem)))} =====")

    out = {"bioma": bioma, "seed": seed, "braco": arm, "modo": modo, "vetor": vetor,
           "ordem_quadrantes": list(ordem), "quadrantes": {}, "esquecimento": {},
           "etapas": [], "trajetoria_reserva": {},
           "composicao_do_lote": "blocado" if LOTE_BLOCADO else "misto",
           "ciclo_espacial": (CICLO_ESPACIAL if LOTE_BLOCADO else None),
           "epocas": {f"etapa{k+1}": ETAPAS_EPOCAS[k] for k in range(len(ordem))},
           "lr": {f"etapa{k+1}": ETAPAS_LR[k] for k in range(len(ordem))},
           "layout_reservas": verificar_layout(bioma, ordem),
           "contiguidade": contig,
           "ablacao": {"interacoes": com_inter, "porta_regime": com_porta,
                       "sem_canal_observacao": sem_obs,
                       "sem_sar": sem_sar, "sem_regime": sem_reg}}

    # ── 1. union ──
    u = Uniao(len(ordem))
    ref_idw: dict = {}
    for quad in ordem:
        mem_guard(f"pre-carga {bioma}_{quad}")
        dd, lab, prep, diag = preparar_peca(bioma, quad, seed, bloco_cel, vetor,
                                            usar_sar, bloco_por_bioma, sem_sar, sem_reg,
                                            com_inter, sem_obs)
        # IDW BEFORE the union, with the quadrant's own `dd`. It depends on neither the
        # model nor the training regime, so it equals the chain value and stays paired.
        # After the union, `pos_anc_m` of different quadrants shares the same local
        # coordinate space and the Euclidean neighbourhood would lose meaning.
        from b1_kriging_indutivo import idw as _idw
        pos, yv = dd["pos_anc_m"], dd["y"]
        m_obs_a = lab == 0
        esc_base = f"{bioma}_{quad}_seed{seed}_{arm}{suf}_{modo}"
        ref_idw[quad] = {}
        for rot, nome_sub in ((3, "reserva"), (2, "teste_em_dist"), (4, "vazio")):
            sel_r = lab == rot
            if sel_r.sum() == 0:
                continue
            p_idw = _idw(pos[m_obs_a], yv[m_obs_a], pos[sel_r])
            ref_idw[quad][nome_sub] = avaliar(None, prep, dd, lab, rot,
                                              f"{esc_base}_idw_{nome_sub}",
                                              com_sigma=False, pred=p_idw)
        u.adicionar(quad, dd, lab, prep, diag)
        del prep
        for k in list(dd.keys()):
            dd[k] = None
        del dd, lab
        mem_guard(f"uniao apos {quad}")

    ddu, prepu, lab_u, quad_de = u.fechar()
    n_uni, e_uni = ddu["n_nodes"], prepu["data"].edge_index.shape[1]
    log(f"  uniao: {len(ordem)} quadrantes, {n_uni:,} nos, {e_uni:,} arestas, "
        f"{len(lab_u):,} ancoras | G5 conferida: 0 arestas entre quadrantes")
    del u
    mem_guard("uniao fechada")

    # ── 2. model, ONE for all four stages ──
    if com_porta:
        grupos, col_reg_p = layout_grupos(prepu["d_in"], com_inter, ddu.get("cols"),
                                          ddu.get("n_interacoes"))
        camadas = int(arm_base.split("_")[1])
        modelo = GATv2Portao(grupos, col_reg_p, camadas)
    else:
        modelo, camadas = construir(arm_base, prepu["d_in"])
    modelo._camadas_b2 = camadas
    modelo = modelo.to(DEV)
    col_reg = _cols(ddu)["regime"]

    def avaliar_quad(q: str, rotulo: int, escopo: str, com_sigma: bool) -> dict:
        """Evaluate one quadrant of the union with EVERYTHING restricted to it.

        Restricting label 0 matters too: `avaliar` derives the trivial constant and the
        distance to observed anchors from it. With the whole window, the constant would
        be the window mean and `skill` would no longer pair with the chain, which uses
        the quadrant's own mean. And `pos_anc_m` is LOCAL to the quadrant: mixing
        quadrants would place two distant anchors at the same coordinate.
        """
        lab_q = np.where(quad_de == q, lab_u, np.int8(-1))
        return avaliar(modelo, prepu, ddu, lab_q, rotulo, escopo, com_sigma=com_sigma)

    # ── 3. stages ──
    estado_opt: dict | None = None
    for k, quad in enumerate(ordem):
        janela = ordem[:k + 1]
        rotulo_etapa = f"etapa {k+1}: {'+'.join(janela)}"
        # THE MASK IS THE WINDOW. Outside it the label becomes -1, so `lab == 0` and
        # `lab == 1` inside `treinar_quadrante` are already restricted to the quadrants
        # seen so far: training and validation grow together, and early stopping at
        # each stage uses the union of the validation sets seen so far.
        lab_janela = np.where(np.isin(quad_de, janela), lab_u, np.int8(-1))
        n_tr = int((lab_janela == 0).sum())
        n_va = int((lab_janela == 1).sum())
        # ROTATION ORDER: `CICLO_ESPACIAL` restricted to the window. It is the
        # Hamiltonian cycle of the 2x2 grid, so consecutive batches always cross a
        # shared edge. A window quadrant outside the cycle would enter last, and the
        # guard below ensures none is silently lost.
        grupos = None
        if LOTE_BLOCADO:
            ciclo = ([q for q in CICLO_ESPACIAL if q in janela]
                     + [q for q in janela if q not in CICLO_ESPACIAL])
            if sorted(ciclo) != sorted(janela):
                raise ValueError(f"rodizio {ciclo} nao cobre a janela {janela}")
            grupos = [(q, (lab_janela == 0) & (quad_de == q)) for q in ciclo]
        log(f"  --- {rotulo_etapa} | {n_tr:,} ancoras de treino, {n_va:,} de validacao "
            f"| teto {ETAPAS_EPOCAS[k]} epocas, paciencia "
            f"{paciencia_da_etapa(ETAPAS_EPOCAS[k])}, lr {ETAPAS_LR[k]:.0e} ---")

        r = treinar_quadrante(modelo, prepu, ddu, lab_janela, rotulo_etapa,
                              (estado_opt if not SEM_ESTADO_OPT else None),
                              ETAPAS_EPOCAS[k], ETAPAS_LR[k], col_reg=col_reg,
                              peso_solo=PESO_SOLO, peso_agua=PESO_AGUA,
                              com_curriculo=(USA_CURRICULO and k == 0),
                              paciencia=paciencia_da_etapa(ETAPAS_EPOCAS[k]),
                              grupos=grupos)
        estado_opt = r["estado_opt"]
        r["hist"]["janela"] = list(janela)
        r["hist"]["etapa"] = k + 1
        r["hist"]["n_ancoras_treino"] = n_tr
        r["hist"]["n_ancoras_validacao"] = n_va

        # The quadrant that JUST entered: immediate reserve and in-distribution test,
        # with the same artifact names as the chain, for direct pairing.
        esc_base = f"{bioma}_{quad}_seed{seed}_{arm}{suf}_{modo}"
        res_ime = avaliar_quad(quad, 3, f"{esc_base}_reserva_imediata", True)
        res_dist = avaliar_quad(quad, 2, f"{esc_base}_teste_em_dist", False)
        out["quadrantes"][quad] = {
            "particao": prepu["particao"][quad],
            "garantias": prepu["garantias"]["por_quadrante"][quad],
            "treino": r["hist"], "reserva_imediata": res_ime,
            "teste_em_distribuicao": res_dist,
            "referencia_idw": ref_idw.get(quad, {}),
        }

        # TRAJECTORY: the reserve of EACH quadrant seen so far, at every stage. It
        # measures forgetting continuously rather than at the endpoints; if the growing
        # window works, the Q1 line stays flat from stage 1 to 4.
        traj = {}
        for q in janela:
            if q == quad:
                traj[q] = {"mae_m": res_ime["test_mae"],
                           "skill": res_ime["comparabilidade"]["skill"]}
                continue
            # The NAME carries the EVALUATED quadrant, not the one that entered, and uses
            # `trajetoria` instead of `reserva` on purpose: the `PADRAO` of
            # `auditoria_v23.py` only recognizes the five canonical sets, so these
            # per-stage diagnostics stay out of the comparison table.
            esc_t = (f"{bioma}_{q}_seed{seed}_{arm}{suf}_{modo}"
                     f"_trajetoria_etapa{k+1}")
            m = avaliar_quad(q, 3, esc_t, False)
            traj[q] = {"mae_m": m["test_mae"], "skill": m["comparabilidade"]["skill"]}
        out["trajetoria_reserva"][f"etapa{k+1}"] = traj
        log("    reservas apos a etapa: " + " | ".join(
            f"{q} {v['mae_m']:.3f} m (skill {v['skill']:+.3f})" for q, v in traj.items()))

        out["etapas"].append({
            "etapa": k + 1, "entrou": quad, "janela": list(janela),
            "epocas_teto": ETAPAS_EPOCAS[k], "epocas_usadas": r["hist"]["epocas_usadas"],
            "parou_antes": r["hist"]["parou_antes"], "lr": ETAPAS_LR[k],
            "val_mae_melhor": r["hist"]["val_mae_melhor"],
            "n_ancoras_treino": n_tr, "n_ancoras_validacao": n_va,
            "treino_s": r["hist"]["treino_s"]})

        if com_porta:
            A = modelo.porta.A.detach().cpu().numpy()
            ramos = modelo.porta.ordem_ramos
            prior = np.stack([prior_do_ramo(kk) for kk in ramos], axis=1)
            out["quadrantes"][quad]["porta_regime"] = {
                "ramos": ramos, "regimes": NOMES_REGIME,
                "prior_herdado_de": {kk: HERANCA_PRIOR[kk] for kk in ramos},
                "aprendida": A.tolist(), "prior_fisico": prior.tolist(),
                "desvio_do_prior": (A - prior).tolist(),
                "desvio_abs_medio": float(np.abs(A - prior).mean())}

        # Checkpoint with the SAME name as the chain (`_ate{quad}`), only with
        # `modo=ampliacao`, so `_raiox_da_corrida` and `superficie_v23` work unchanged:
        # they build the path from (bioma, seed, arm, mode, last quadrant).
        CKPT_B2.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": {kk: v.detach().cpu().clone()
                                   for kk, v in modelo.state_dict().items()},
                    "arm": arm, "camadas": camadas, "vetor": vetor,
                    "d_in": prepu["d_in"], "bioma": bioma, "quadrante": quad,
                    "seed": seed, "modo": modo, "bloco_celulas": bloco_cel,
                    "usar_sar": usar_sar, "bloco_por_bioma": bloco_por_bioma,
                    "escala_obs": prepu["garantias"]["por_quadrante"][quad][
                        "escala_obs_de_treino"],
                    "val_mae": r["hist"]["val_mae_melhor"],
                    "reserva_mae": res_ime["test_mae"],
                    "etapa": k + 1, "janela": list(janela),
                    "ordem_quadrantes": list(ordem),
                    "reserva_por_quad": {kk: v[0] for kk, v in RESERVA_POR_QUAD.items()}},
                   CKPT_B2 / f"{bioma}_seed{seed}_{arm}{suf}_{modo}_ate{quad}.pt")

        if torch.cuda.is_available():
            pico = torch.cuda.max_memory_allocated() / 2 ** 30
            reserv = torch.cuda.max_memory_reserved() / 2 ** 30
            out["quadrantes"][quad]["vram_pico_gb"] = round(float(pico), 3)
            out["quadrantes"][quad]["vram_reservada_gb"] = round(float(reserv), 3)
            log(f"    VRAM: pico {pico:.2f} GB alocado, {reserv:.2f} GB reservado")
            if TETO_VRAM_GB > 0 and pico > TETO_VRAM_GB:
                raise RuntimeError(
                    f"VRAM de pico {pico:.2f} GB passou do teto {TETO_VRAM_GB:g} GB. "
                    f"No Windows o driver NAO da OOM: ele derrama para a RAM e a "
                    f"corrida fica ~30x mais lenta em silencio (medido em "
                    f"2026-08-12). Reduza --batch ou a profundidade.")
            torch.cuda.reset_peak_memory_stats()
        del r
        mem_guard(f"pos-etapa{k+1}")

    # ── 4. final pass and forgetting ──
    # In the chain this requires reloading one graph at a time; here the union is
    # already in memory, so the final pass is one more evaluation per quadrant.
    log("  --- passe final: modelo da ultima etapa em todas as reservas ---")
    # `delta_m` IS NOT FORGETTING IN THIS REGIME. In the chain, `imediato` is the
    # quadrant trained to convergence and `final` is what remains after the others,
    # so a positive delta IS forgetting. Here `imediato` is the state at the end of
    # the STAGE in which the quadrant entered (a model that will still see every
    # later stage), so a negative delta is expected and only says how much later
    # stages HELPED.
    #
    # The paired comparison between regimes is `reserva_final` (same subset, same
    # trivial constant, same skill) and `trajetoria_reserva`, which shows whether any
    # quadrant worsens at ANY stage.
    out["nota_esquecimento"] = (
        "delta_m = final - imediato. Na AMPLIACAO `imediato` e o fim da etapa de entrada, "
        "nao um modelo convergido naquele quadrante: delta negativo e o esperado e mede "
        "ajuda das etapas seguintes, NAO ausencia de esquecimento. Para comparar com a "
        "cadeia use `reserva_final` e `trajetoria_reserva`.")
    for quad in ordem:
        esc = f"{bioma}_{quad}_seed{seed}_{arm}{suf}_{modo}_reserva_final"
        fin = avaliar_quad(quad, 3, esc, False)
        ime = out["quadrantes"][quad]["reserva_imediata"]["test_mae"]
        out["quadrantes"][quad]["reserva_final"] = fin
        out["esquecimento"][quad] = {
            "mae_imediato_m": ime, "mae_final_m": fin["test_mae"],
            "delta_m": fin["test_mae"] - ime,
            "delta_pct": 100.0 * (fin["test_mae"] - ime) / max(ime, 1e-9),
            "skill_imediato": (out["quadrantes"][quad]["reserva_imediata"]
                               ["comparabilidade"]["skill"]),
            "skill_final": fin["comparabilidade"]["skill"]}
        log(f"  {quad}: reserva final MAE={fin['test_mae']:.3f} m "
            f"(imediato {ime:.3f} m, delta {fin['test_mae'] - ime:+.3f} m)")

    # E1: synthetic void population (class 4), evaluated per quadrant at the end,
    # with the same restricted scope as avaliar_quad (trivial constant and distance
    # come from the quadrant itself), so it pairs across arms.
    if VAZIO_KM > 0:
        out["vazio"] = {}
        for quad in ordem:
            if int(((quad_de == quad) & (lab_u == 4)).sum()) == 0:
                continue
            v = avaliar_quad(quad, 4,
                             f"{bioma}_{quad}_seed{seed}_{arm}{suf}_{modo}_vazio",
                             False)
            out["vazio"][quad] = v
            log(f"  {quad}: VAZIO (R={VAZIO_KM:g} km) MAE={v['test_mae']:.3f} m "
                f"(n={v['n']:,})")

    qs = list(out["esquecimento"].keys())
    maes = [out["esquecimento"][q]["mae_final_m"] for q in qs]
    skills = [out["quadrantes"][q]["reserva_final"]["comparabilidade"]["skill"]
              for q in qs]
    ns = [out["quadrantes"][q]["reserva_final"]["n"] for q in qs]
    out["reserva_uniao"] = {
        "mae_media_simples_m": float(np.mean(maes)),
        "mae_media_ponderada_por_n_m": float(np.average(maes, weights=ns)),
        "skill_media_ponderada_por_n": float(np.average(skills, weights=ns)),
        "n_total": int(sum(ns)),
        "nota": "media de MAEs por quadrante; a ponderada por n e a comparavel a um MAE "
                "calculado sobre a uniao dos nos. As quatro reservas tem variabilidade de "
                "alvo distinta, portanto a media de skill e a agregacao defensavel e a de "
                "MAE e apenas descritiva."}

    del modelo, prepu, ddu
    limpar_amostradores()
    mem_guard("pos-ampliacao")

    if COM_RAIOX:
        try:
            out["G_raiox"] = _raiox_da_corrida(bioma, out, log)
        except Exception as e:
            log(f"  raio-X falhou: {type(e).__name__}: {str(e)[:140]}")
            out["G_raiox"] = {"erro": f"{type(e).__name__}: {str(e)[:200]}"}
    return out


_REGIME_CACHE: dict = {}


def _pertinencia_dossel(bioma: str, quad: str, n_nodes: int):
    """Membership in `dossel_denso`, from the SAME source the loss uses.

    Cached per quadrant: in a sweep the five seeds run in the same process, and
    recomputing would cost ~20 s per quadrant per run (~208 MB per quadrant).
    """
    ch = (bioma, quad)
    if ch not in _REGIME_CACHE:
        import vetor_v23 as V
        i = V.NOMES_REGIME.index("dossel_denso")
        _, reg, _ = V.optico_e_regime(bioma, quad, n_nodes)
        _REGIME_CACHE[ch] = np.clip(reg[:, i].astype(np.float32), 0.0, 1.0)
        del reg
    return _REGIME_CACHE[ch]


def _delta_sob_dossel(bioma: str, quad: str, delta: np.ndarray) -> dict:
    """`Delta < 0` restricted to dense canopy, WITH magnitude.

    `frac_delta_negativo` is not enough: it covers ALL cells, while the constraint
    penalizes `relu(-Delta)` weighted by canopy membership. Over bare soil the target
    is `Delta = 0` and a tiny negative is noise, not a violation.

    A SIGN COUNT IS NOT SEVERITY: where `Delta` is negative it may be so by
    millimetres, so a large fraction of "violating cells" can sit a few mm from zero.
    Hence the violation percentiles and the fractions beyond 10 cm and 1 m are
    reported together.
    """
    try:
        pert = _pertinencia_dossel(bioma, quad, delta.size)
    except Exception as e:                                    # noqa: BLE001
        return {"erro": f"{type(e).__name__}: {str(e)[:120]}"}
    m = pert >= 0.5
    neg = delta < 0
    r = {"frac_dossel_denso": round(float(m.mean()), 5),
         "frac_neg_todas_celulas": round(float(neg.mean()), 6),
         "frac_neg_no_dossel": round(float(neg[m].mean()), 6) if m.any() else None,
         # What the loss actually integrates, in its own unit.
         "termo_da_perda_m": round(float((pert * np.maximum(-delta, 0.0)).sum()
                                         / max(pert.sum(), 1e-9)), 8)}
    nd = neg & m
    if nd.any():
        v = -delta[nd]
        r.update(n_violacoes_dossel=int(nd.sum()),
                 violacao_mediana_m=round(float(np.median(v)), 6),
                 violacao_p95_m=round(float(np.percentile(v, 95)), 5),
                 violacao_max_m=round(float(v.max()), 4),
                 frac_violacoes_acima_10cm=round(float((v > 0.1).mean()), 5),
                 frac_violacoes_acima_1m=round(float((v > 1.0).mean()), 5))
    return r


def _raiox_da_corrida(bioma: str, out: dict, log) -> dict:
    """Generate `z'` on all nodes and run the full-coverage families.

    Uses the END-OF-CHAIN checkpoint, the same model whose final pass produced
    `reserva_final`, so the X-ray and the anchor suite describe the same state.
    """
    import raiox_geomorfologico as RX
    import superficie_v23 as SP

    qs = [q for q in ORDEM_QUAD if q in out["quadrantes"]]
    if not qs:
        return {"pulado": "nenhum quadrante"}
    ult = qs[-1]
    ck_path = (CKPT_B2 / f"{bioma}_seed{out['seed']}_{out['braco']}"
               f"{'' if out['vetor'] == 'v22' else '_' + out['vetor']}"
               f"_{out['modo']}_ate{ult}.pt")
    if not ck_path.exists():
        return {"pulado": f"checkpoint ausente: {ck_path.name}"}
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    log(f"  raio-X: {ck_path.name}")

    fam = {}
    for q in qs:
        delta = SP.inferir_quadrante(bioma, q, ck, log)
        dsm = np.load(LATDIR_V23 / f"glo30_{bioma}_{q}.npz")["DEM"].astype(np.float32)
        z = (dsm - delta).astype(np.float32)
        slope, aspect = RX.derivadas(bioma, q, z)
        fam[q] = {
            "E_admissibilidade": RX.c1_admissibilidade(bioma, q, z, slope),
            "E_angulo_local": RX.c2_angulo_local(bioma, q, slope, aspect),
            "D_derivadas_vs_lidar": RX.c3_verdade_lidar(bioma, q, z),
            # A negative Delta under canopy is impossible: the ground cannot lie above
            # the surface the radar saw. Counted here because it needs the full field.
            "frac_delta_negativo": round(float((delta < 0).mean()), 6),
            "delta_mediano_m": round(float(np.median(delta)), 4),
            # The family that judges the canopy constraint by what it actually penalizes
            # (`frac_delta_negativo` mixes canopy with bare soil).
            "F_dossel": _delta_sob_dossel(bioma, q, delta),
        }
        del delta, dsm, z, slope, aspect
        gc.collect()
    return fam


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--biomes", nargs="+", default=["amazonia"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--arms", nargs="+", default=["gatv2_2"])
    # ── THE THREE QUADRANT PRESENTATION REGIMES.
    #
    # They differ ONLY in the order in which quadrants enter the loss. Geography,
    # split, reserves, buffer, seed and block are identical, so any difference can
    # be attributed to the regime and not to the cropping.
    #
    #   cadeia      Q1 to convergence, weights carried forward, then Q2, ... A quadrant
    #               LEAVES the loss as training moves on. This is the CONTROL.
    #   ampliacao   growing window: Q1, Q1+Q2, Q1+Q2+Q3, all. A quadrant that enters
    #               never leaves. See the growing-window notes at the top of the file.
    #   isolado     each quadrant from scratch. Does not measure forgetting; measures
    #               how much transfer (of any kind) is worth over no transfer.
    #
    # The DEFAULT remains `cadeia`, so existing run commands keep their behaviour.
    ap.add_argument("--regime", choices=["cadeia", "ampliacao", "isolado"], default=None,
                    help="como os quadrantes entram na perda. Padrao: cadeia (ou isolado "
                         "com --sem-transferencia). Os tres compartilham particao, "
                         "reservas e seeds.")
    ap.add_argument("--sem-transferencia", action="store_true",
                    help="apelido de --regime isolado, mantido para as filas ja escritas: "
                         "cada quadrante treina do zero, mesmas reservas e mesmas seeds.")
    ap.add_argument("--tag", default="",
                    help="etiqueta anexada ao `modo` em todo nome de artefato "
                         "(checkpoint, predicao, escopo do JSON). OBRIGATORIA em "
                         "varredura: sem ela, dois pontos que so diferem num peso da "
                         "perda gravam no mesmo checkpoint e o ultimo apaga os demais — "
                         "inclusive os de experimentos anteriores no mesmo regime.")
    ap.add_argument("--lotes", choices=["misto", "blocado"], default="misto",
                    help="so vale com --regime ampliacao. `misto`: o lote sorteia "
                         "sementes de todos os quadrantes da janela. `blocado`: cada lote "
                         "vem de um quadrante, em rodizio pelo ciclo espacial. Separa o "
                         "que hoje anda junto — o quadrante ficar na perda (que impede o "
                         "esquecimento) e o lote cruzar quadrantes (suspeito de deixar o "
                         "campo rugoso: 3 a 4x menos pocos removidos que a cadeia).")
    ap.add_argument("--sem-sar", action="store_true")
    ap.add_argument("--bloco-por-bioma", action="store_true", default=True)
    ap.add_argument("--quads", nargs="+", default=None,
                    help="subconjunto da cadeia, em ordem (padrao Q1 Q2 Q3 Q4)")
    ap.add_argument("--smoke", action="store_true",
                    help="2 epocas por quadrante: valida o caminho, nao o resultado")
    ap.add_argument("--retomar", action="store_true",
                    help="pula (bioma,seed,braco) ja presentes no JSON de saida. "
                         "Necessario porque uma etapa morta por teto de tempo perde "
                         "tudo o que ja tinha rodado se a fila reiniciar do zero.")
    ap.add_argument("--peso-declividade", type=float, default=None,
                    help="peso da restricao de derivada direcional da trilha LiDAR. "
                         "0 desliga, para o braco de ablacao")
    ap.add_argument("--sem-ponderacao", action="store_true",
                    help="braco de ablacao: perda L1 simples, sem o peso de incerteza do "
                         "rotulo. Reproduz bit a bit o comportamento anterior a 2026-08-05")
    ap.add_argument("--peso-solo", type=float, default=None,
                    help="peso da restricao `Delta = 0` em solo exposto. 0 desliga e o "
                         "treino fica identico ao anterior")
    ap.add_argument("--peso-dossel", type=float, default=None,
                    help="peso da restricao `Delta >= 0` sob dossel denso. O chao nao "
                         "pode estar acima da superficie que o radar viu; medido no POC "
                         "de 2026-08-07, o modelo violava isso em 24%% das celulas do Q3.")
    ap.add_argument("--peso-agua", type=float, default=None,
                    help="peso da restricao `z' constante` em lamina de agua. 0 desliga")
    ap.add_argument("--boost-atl08", type=float, default=None,
                    help="peso K das ancoras ATL08 na perda (GEDI fica em 1). Liga a "
                         "ponderacao. Testa se dar voz ao instrumento minoritario move "
                         "o modelo na faixa 20-30 m onde ele reproduz o vies do GEDI; "
                         "juiz = ancoras ATL08 da reserva, nunca treinadas.")
    ap.add_argument("--suavizar-features", type=int, default=None,
                    help=("T1: radius in cells of the spatial mean of X (e.g., 2 = 5x5 window ~ ±62 "
                          "m); separate arm"))
    ap.add_argument("--vizinhanca-exata", action="store_true",
                    help="usa a vizinhanca COMPLETA por salto ([-1]*k) em vez do "
                         "fan_out amostrado — correto no reticulado e sem "
                         "confundir alcance com orcamento de amostragem")
    ap.add_argument("--teto-vram", type=float, default=None,
                    help="aborta se o pico de VRAM passar deste valor em GB "
                         "(evita transbordo silencioso para RAM no Windows)")
    ap.add_argument("--vazio-km", type=float, default=None,
                    help=("E1: masks anchors within R km of the EBA transect centroids (class 4, "
                          "population `vazio`)"))
    ap.add_argument("--peso-suavidade", type=float, default=None,
                    help="peso da variacao total de Delta sobre as arestas "
                         "(`media(|p_i - p_j|)`). 0 (padrao) desliga e apenas mede. E o "
                         "termo do quarto teste: a energia de alta frequencia de Delta "
                         "prediz poço (rho +0,52 a +0,68) e a cauda >2 m dos poços "
                         "criados e o custo real do regime vencedor.")
    # ── TRANSFER-LADDER controls (lr, epochs and optimizer state for Q2-Q4): they
    # separate catastrophic forgetting from a badly sized ladder by varying the ladder.
    ap.add_argument("--lr-seguintes", type=float, default=None,
                    help="lr de Q2, Q3 e Q4. Padrao 2e-4, herdado do GNN_RF. O Q1 nao "
                         "muda: ele aprende do zero e e outro regime.")
    ap.add_argument("--epocas-seguintes", type=int, default=None,
                    help="epocas de Q2, Q3 e Q4. Padrao 15.")
    ap.add_argument("--sem-estado-otimizador", action="store_true",
                    help="os PESOS atravessam, o estado do Adam NAO. Isola quanto do "
                         "efeito de transferencia vem do momento acumulado e quanto vem "
                         "da representacao aprendida — sao coisas diferentes e hoje "
                         "atravessam juntas.")
    ap.add_argument("--sem-painel", action="store_true",
                    help="desliga o painel por epoca e volta a uma linha por quadrante. "
                         "Util em varredura, onde 80 painéis por corrida poluem o log.")
    ap.add_argument("--sem-raiox", action="store_true",
                    help="pula o raio-X geomorfologico. Ele custa ~20 s por quadrante "
                         "de inferencia sobre 12,96 milhoes de nos, e e a UNICA "
                         "avaliacao que nao depende de ancora — desligar so faz sentido "
                         "em varredura de hiperparametro.")
    ap.add_argument("--curriculo", choices=["auto", "on", "off"], default="auto",
                    help="curriculo de 5 fases no Q1: rotulo -> solo -> agua -> "
                         "declividade -> consolidacao, cada peso em rampa dentro da fase "
                         "que o introduz. `auto` = LIGADO no v23, desligado no v22. "
                         "Desligado, os pesos valem cheios desde a primeira epoca — que e "
                         "o braco de ablacao do proprio curriculo.")
    ap.add_argument("--bf16", choices=["auto", "on", "off"], default="auto",
                    help="precisao mista em bfloat16. `auto` = LIGADA para o vetor v23 "
                         "com CUDA, DESLIGADA para o v22 — o baseline congelado tem de "
                         "continuar reproduzivel bit a bit, e mudar a numerica dele "
                         "junto com o vetor tornaria impossivel atribuir a diferenca.")
    ap.add_argument("--batch", type=int, default=None,
                    help="nos-alvo por lote do NeighborLoader. Nao muda o resultado em "
                         "expectativa — a perda e media, entao o gradiente de um lote "
                         "menor tem a mesma esperanca —, muda VRAM de pico e tempo. "
                         "Padrao: o BATCH de b1_kriging_indutivo.")
    ap.add_argument("--fator-epocas", type=int, default=1,
                    help="multiplica os tetos de ETAPAS_EPOCAS. Existe para "
                         "casar PASSOS DE GRADIENTE sem mexer no lote: com o "
                         "lote /4 e as epocas fixas, o orcamento de otimizacao "
                         "quadruplica sem que nada compense. "
                         "`--batch 6144 --fator-epocas 4` tem aproximadamente "
                         "os passos de `--batch 1536`. Serve tambem a curva de "
                         "eficiencia amostral: com 1/f das ancoras de treino, "
                         "`--fator-epocas f` devolve o mesmo numero de passos.")
    ap.add_argument("--fator-fases", type=int, default=1,
                    help=("multiplies the epochs of each phase of the physics curriculum (FASES) by "
                          "the factor. It exists for check F1: D1 scaled only the caps and the ramp "
                          "stayed at 30 epochs, changing the fraction of stage 1 under full physics "
                          "from 2/3 to 1/6 — a second difference in treatment. `--fator-epocas 4 "
                          "--fator-fases 4` restores the fractions of the x1 budget and attributes "
                          "the D1-1536 excess."))
    ap.add_argument("--frac-ancoras-treino", type=float, default=1.0,
                    help="subamostra as ancoras OBSERVADAS de treino (rotulo 0) "
                         "para esta fracao, por semente. Val, teste e reserva "
                         "ficam INTACTOS, entao a populacao de avaliacao nao "
                         "muda e as curvas sao comparaveis. Existe para medir "
                         "se a agregacao de vizinhanca SUBSTITUI dado: se a "
                         "vantagem do grafo cresce quando o dado encolhe, a "
                         "contribuicao do grafo esta demonstrada. Combinar com "
                         "`--fator-epocas` para casar passos entre fracoes.")
    ap.add_argument("--saida", default="b2_transferencia")
    ap.add_argument("--vetor", choices=["v22", "v23"], default="v22",
                    help="v22 = as 22 colunas do baseline congelado (padrao, nao muda "
                         "nada do que ja rodou). v23 = vetor novo (banda L do ALOS, "
                         "DEMs independentes, agua) com rotulo filtrado por qualidade.")
    a = ap.parse_args()

    # OPTIMIZATION BUDGET, SEPARATE FROM THE BATCH. Everything governing training
    # here is counted in EPOCHS: the `ETAPAS_EPOCAS` ceilings, `CosineAnnealingLR`
    # (`T_max = epochs`, with `sched.step()` outside the batch loop) and patience
    # (`paciencia_da_etapa`). Only `opt.step()` is per batch. Measured consequence:
    # `--batch 1536` is not an OFAT of the batch size -- it also quadruples the
    # GRADIENT STEPS (3.885x for GATv2, 4.054x for MLP), and the epoch ceiling only
    # binds the graph (best epoch 13.6 of 14, 0/10 seeds stopping on their own,
    # against 7.8 of 14 and 8/10 for the MLP).
    #
    # This flag exists to match steps WITHOUT touching the batch size:
    #   --batch 6144 --fator-epocas 4   ~=  steps of  --batch 1536
    # It also serves the sample-efficiency curve: with 1/f of the training
    # anchors, `--fator-epocas f` returns the same number of steps.
    #
    # WHAT IT DELIBERATELY DOES NOT DO. (1) It does not touch `FASES`: the
    # physics ramp stays at the same 30 epochs, so with factor 4 it occupies
    # 1/6 of stage 1 instead of 2/3 -- a real difference against the 1536 run
    # that must be declared by whoever consumes the result. (2) It does not
    # scale the lr: the cosine schedule already uses `T_max = epochs`, so it
    # stretches along with it and the lr integral over steps matches on its
    # own. (3) It does not touch patience, which is already
    # `max(6, teto // 3)` and tracks the ceiling.
    if a.fator_epocas != 1:
        if a.fator_epocas < 1:
            raise SystemExit("--fator-epocas tem de ser >= 1")
        ETAPAS_EPOCAS[:] = [t * a.fator_epocas for t in ETAPAS_EPOCAS]
        log(f"TETOS DE EPOCA x{a.fator_epocas}: {ETAPAS_EPOCAS} | paciencia "
            f"{[paciencia_da_etapa(t) for t in ETAPAS_EPOCAS]} "
            f"(orcamento de otimizacao, NAO lote)")

    # Scales each phase's epochs (`FASES`) by the same factor as the ceilings.
    # Scaling only the ceilings changes the fraction of stage 1 under full physics
    # (2/3 -> 1/6) and, since `suave` belongs to no phase and enters only at epoch
    # 31, lets a longer stage 1 train under the TV term that the x1 budget never
    # reaches. `--fator-epocas 4 --fator-fases 4` restores the x1 phase fractions.
    if a.fator_fases != 1:
        if a.fator_fases < 1:
            raise SystemExit("--fator-fases tem de ser >= 1")
        FASES[:] = [(nome, n_ep * a.fator_fases, ativas)
                    for nome, n_ep, ativas in FASES]
        log(f"RAMPA DE FASES x{a.fator_fases}: "
            + " -> ".join(f"{n}({e})" for n, e, _ in FASES)
            + f" | soma {sum(e for _, e, _ in FASES)} epocas "
            f"(o termo `suave` continua entrando so DEPOIS da rampa)")

    if a.frac_ancoras_treino != 1.0:
        if not 0.0 < a.frac_ancoras_treino <= 1.0:
            raise SystemExit("--frac-ancoras-treino tem de estar em (0, 1]")
        global FRAC_ANCORAS_TREINO
        FRAC_ANCORAS_TREINO = float(a.frac_ancoras_treino)
        log(f"SUBAMOSTRAGEM DE ANCORAS DE TREINO: {FRAC_ANCORAS_TREINO:.0%} "
            f"(val, teste e reserva INTACTOS; fracoes aninhadas)")

    if a.quads:
        faltam = [q for q in a.quads if q not in RESERVA_POR_QUAD]
        if faltam:
            raise SystemExit(f"quadrante sem reserva definida: {faltam}")
        ORDEM_QUAD[:] = a.quads

    regime = a.regime or ("isolado" if a.sem_transferencia else "cadeia")
    if a.regime and a.sem_transferencia and a.regime != "isolado":
        raise SystemExit(
            f"--regime {a.regime} contradiz --sem-transferencia, que e apelido de "
            f"--regime isolado. Passe um dos dois.")
    global TAG
    TAG = a.tag.strip().strip("_")
    if TAG:
        if not TAG.replace("_", "").isalnum():
            raise SystemExit(f"--tag deve ser alfanumerica; veio {a.tag!r}")
        log(f"etiqueta de corrida: '{TAG}' — entra em checkpoint, predicao e escopo")

    global LOTE_BLOCADO
    LOTE_BLOCADO = a.lotes == "blocado"
    if LOTE_BLOCADO and regime != "ampliacao":
        raise SystemExit(
            f"--lotes blocado so tem sentido com --regime ampliacao; veio {regime}. "
            f"Na cadeia e no isolado o lote ja e de um quadrante so, por construcao.")

    if regime == "ampliacao":
        # Fail HERE, before loading 20 GB, if the order contains a diagonal jump.
        verificar_contiguidade(ORDEM_QUAD)
        if len(ETAPAS_EPOCAS) < len(ORDEM_QUAD):
            raise SystemExit(
                f"ETAPAS_EPOCAS tem {len(ETAPAS_EPOCAS)} entradas para "
                f"{len(ORDEM_QUAD)} quadrantes")
        log(f"regime: JANELA CRESCENTE — "
            + " -> ".join("+".join(ORDEM_QUAD[:k + 1])
                          for k in range(len(ORDEM_QUAD))))
        if LOTE_BLOCADO:
            log(f"  lote BLOCADO, rodizio em {[q for q in CICLO_ESPACIAL if q in ORDEM_QUAD]}"
                f" (ciclo hamiltoniano: toda transicao cruza borda compartilhada)")
        else:
            log("  lote MISTO: sementes de todos os quadrantes da janela no mesmo lote")
        tetos = ETAPAS_EPOCAS[:len(ORDEM_QUAD)]
        log(f"  tetos por etapa {tetos} | paciencia "
            f"{[paciencia_da_etapa(t) for t in tetos]} | "
            f"lr {ETAPAS_LR[:len(ORDEM_QUAD)]}")
        log(f"  custo maximo {sum((k + 1) * t for k, t in enumerate(tetos))} "
            f"quadrante-epocas (a cadeia usa "
            f"{sum(EPOCAS_ESCADA[q] for q in ORDEM_QUAD)}); a parada antecipada reduz")
    else:
        log(f"regime: {regime}")

    if a.smoke:
        for q in EPOCAS_ESCADA:
            EPOCAS_ESCADA[q] = 2
        ETAPAS_EPOCAS[:] = [2] * len(ETAPAS_EPOCAS)
        log("MODO SMOKE: 2 epocas por etapa — numeros NAO citaveis")

    if a.lr_seguintes is not None:
        for q in ("Q2", "Q3", "Q4"):
            LR_ESCADA[q] = a.lr_seguintes
        log(f"escada de lr: Q1 {LR_ESCADA['Q1']:.0e} | Q2-Q4 {a.lr_seguintes:.0e}")
    if a.epocas_seguintes is not None:
        for q in ("Q2", "Q3", "Q4"):
            EPOCAS_ESCADA[q] = a.epocas_seguintes
        log(f"escada de epocas: Q1 {EPOCAS_ESCADA['Q1']} | Q2-Q4 {a.epocas_seguintes}")

    global SEM_ESTADO_OPT
    SEM_ESTADO_OPT = a.sem_estado_otimizador
    if SEM_ESTADO_OPT:
        log("estado do otimizador NAO atravessa entre quadrantes; so os pesos")

    global COM_RAIOX, PAINEL
    COM_RAIOX = not a.sem_raiox
    PAINEL = not a.sem_painel

    global USA_CURRICULO
    USA_CURRICULO = (a.vetor == "v23") if a.curriculo == "auto" else (a.curriculo == "on")

    global USA_AMP
    if a.bf16 == "auto":
        USA_AMP = torch.cuda.is_available() and a.vetor == "v23"
    else:
        USA_AMP = (a.bf16 == "on") and torch.cuda.is_available()
    log(f"precisao: {'bfloat16 (autocast, sem GradScaler)' if USA_AMP else 'float32'} "
        f"| recorte de gradiente {MAX_NORM}")

    global BATCH
    if a.batch:
        BATCH = a.batch
        log(f"BATCH sobrescrito para {BATCH} nos-alvo por lote")
    elif a.vetor == "v23":
        # v22 keeps the BATCH inherited from `b1_kriging_indutivo`, so the frozen
        # baseline stays reproducible; only V23 uses the sized value (`BATCH_V23`).
        BATCH = BATCH_V23
        log(f"BATCH do vetor V23: {BATCH} nos-alvo por lote (dimensionado por varredura)")

    global PESO_SOLO, PESO_AGUA, PONDERAR, PESO_DECL
    # Without an explicit value, V23 uses the MEASURED target weights (`PESOS_ALVO`)
    # and v22 keeps what it always had -- physics off and slope at 1.0 -- so that
    # the frozen baseline stays reproducible.
    pad = (PESOS_ALVO if a.vetor == "v23"
           else {"solo": 0.0, "agua": 0.0, "dossel": 0.0, "decl": 1.0})
    PESO_SOLO = pad["solo"] if a.peso_solo is None else a.peso_solo
    PESO_AGUA = pad["agua"] if a.peso_agua is None else a.peso_agua
    global PESO_DOSSEL
    PESO_DOSSEL = pad["dossel"] if a.peso_dossel is None else a.peso_dossel
    global PESO_SUAVE
    # DEFAULT 0.35, chosen from the curve {0, 0.25, 0.35, 0.5, 1, 4, 16}: 0.35
    # cuts 50.3% of the >2 m pits created (55,968 -> 27,821) at a cost of 0.0111
    # skill, exceeding the 0.010 guard by less than the between-seed sd
    # (comparar_ponto_s035.json). Runs before this default used --peso-suavidade 0.
    PESO_SUAVE = 0.35 if a.peso_suavidade is None else a.peso_suavidade
    if PESO_SUAVE:
        log(f"suavidade de Delta ATIVA: peso {PESO_SUAVE} — outro braco, nao comparavel "
            f"a corridas sem ela")
    a.peso_declividade = (pad["decl"] if a.peso_declividade is None
                          else a.peso_declividade)
    # OFF BY DEFAULT IN V23, based on measurement.
    #
    # The bare-soil ruler showed that, WITHIN the approved population, the spread
    # among the six soil estimates barely predicts error, and what little it does
    # predict is out of order: the 2-5 m bin gets weight 0.567 -- almost half -- and
    # has the SMALLEST error in the table, 2.26 m, while the 0.5-2 m bin gets 0.713
    # with a larger error, 2.84 m. That is 8,071 shots, 20% of the sample, penalized
    # in the wrong direction. Recalibrating `DISP_ESCALA` to track the ruler would
    # produce an almost-flat weight, which is the same as turning it off, with one
    # more parameter to justify.
    #
    # There is also a comparison reason: the frozen baseline had NO weighting.
    # Without it, the difference between v22 and V23 stays in data and vector; with
    # it, the difference would sit in data, vector AND loss function, and the gain
    # could not be attributed to just one of the three.
    #
    # Caveat: the ruler only covers bare soil, 9.8% of the Amazonia shots
    # and, by construction, the easiest terrain. Under closed canopy, where 90% of
    # the anchors are, the spread could well predict error -- and that is where the
    # second ruler, GEDI against ICESat-2 on the shared cells, would answer. The
    # `peso` field stays recorded in the labels: turning it back on is a flag, not
    # rework.
    PONDERAR = (not a.sem_ponderacao) and a.vetor != "v23"
    global VIZ_EXATA, TETO_VRAM_GB
    VIZ_EXATA = bool(a.vizinhanca_exata)
    TETO_VRAM_GB = 0.0 if a.teto_vram is None else a.teto_vram
    if VIZ_EXATA:
        log("vizinhanca EXATA por salto ([-1]*k): reticulado 8-conexo, bola de k "
            "saltos limitada por (2k+1)^2 — sem amostragem, sem confundimento")
    if TETO_VRAM_GB > 0:
        log(f"teto de VRAM: {TETO_VRAM_GB:g} GB (aborta em vez de transbordar)")
    global SUAVIZAR_X, VAZIO_KM
    SUAVIZAR_X = 0 if a.suavizar_features is None else a.suavizar_features
    VAZIO_KM = 0.0 if a.vazio_km is None else a.vazio_km
    if VAZIO_KM > 0:
        log(f"VAZIO SINTETICO E1: ancoras a ate {VAZIO_KM:g} km dos transectos EBA "
            f"saem do treino/val/teste/reserva e viram populacao `vazio`")
    global BOOST_ATL08
    # DEFAULT 4.0. Clean OFAT (pd137 = simple L1; boost = GEDI weight 1, ATL08
    # weight K) improves BOTH rulers on the reserve (ATL08 -1.57 m, GEDI -0.55 m),
    # with a plateau from K=2 to 16, a global spatial effect and no effect on pits
    # (juiz_boost_completo.json, anatomia_boost.json). K=4 is the plateau point where
    # the full validation battery ran (10 seeds). Earlier runs used --boost-atl08 0.
    BOOST_ATL08 = 4.0 if a.boost_atl08 is None else a.boost_atl08
    if BOOST_ATL08 > 0:
        PONDERAR = True          # without this the per-source weight never reaches the loss
        log(f"BOOST ATL08 x{BOOST_ATL08:g}: ancora ICESat-2 pesa {BOOST_ATL08:g}, GEDI "
            f"pesa 1 — outro braco, ponderacao LIGADA")
    PESO_DECL = a.peso_declividade
    log(f"perda de rotulo: {'L1 PONDERADO pela incerteza' if PONDERAR else 'L1 simples'}")
    log(f"restricao de declividade de trilha: "
        f"{'peso ' + str(PESO_DECL) if PESO_DECL > 0 else 'DESLIGADA'}")
    if PESO_SOLO or PESO_AGUA:
        log(f"RESTRICOES FISICAS ATIVAS: solo={PESO_SOLO} agua={PESO_AGUA} — o resultado "
            f"NAO e comparavel a corrida sem elas; e outro braco")

    RESULTS.mkdir(parents=True, exist_ok=True)
    alvo = RESULTS / f"{a.saida}.json"
    feito: dict = {}
    if a.retomar and alvo.exists():
        try:
            feito = json.loads(alvo.read_text(encoding="utf-8"))
            prontos = [k for k, v in feito.items()
                       if isinstance(v, dict) and "quadrantes" in v and v["quadrantes"]]
            log(f"retomada: {len(prontos)} escopos ja concluidos serao pulados")
        except Exception as e:
            log(f"retomada: nao consegui ler {alvo.name} ({e}); comecando do zero")
            feito = {}
    # `out` starts by PRESERVING everything already in the file under --retomar.
    # Otherwise a completion run with a reduced `--arms` would rewrite the JSON with
    # only that run's arms and silently erase the others.
    preservados = {k: v for k, v in feito.items()
                   if k not in ("config", "geometria_reserva")}
    if preservados:
        log(f"retomada: {len(preservados)} escopos do arquivo anterior serao "
            f"preservados mesmo fora deste --arms")
    out: dict = dict(preservados)
    out["config"] = vars(a)
    # `config` describes only THIS run. A file built by successive completion runs
    # holds arms absent from the last `--arms`; without the history, a check that
    # compares "keys present" against "config.arms" would report false missing
    # data, or miss data that really disappeared.
    out["config_historico"] = (feito.get("config_historico", [])
                               + [{"em": time.strftime("%Y-%m-%dT%H:%M"), **vars(a)}])
    out["geometria_reserva"] = {
        "unidade": "sub-quadrante contiguo de 1800x1800 celulas (54x54 km)",
        "fracao": 0.25, "layout": {k: v[0] for k, v in RESERVA_POR_QUAD.items()},
        "buffer_celulas": BUFFER_CEL, "buffer_m": round(BUFFER_CEL * CELULA_M),
        "determinismo": "layout literal em codigo, nao sorteado",
        "referencia": "SplitManifest de train_mining_stgnn_v20.py:709-779"}
    for b in a.biomes:
        for s in a.seeds:
            for arm in a.arms:
                # THE REGIME IS PART OF THE KEY. Otherwise the three regimes would write
                # to the same JSON scope and the last to run would erase the others.
                ck = f"{b}_seed{s}_{arm}_" + com_tag({
                    "cadeia": "transf", "isolado": "isolado",
                    "ampliacao": "ampliacao_blocado" if LOTE_BLOCADO else "ampliacao",
                }[regime])
                if a.vetor != "v22":
                    ck += f"_{a.vetor}"
                anterior = feito.get(ck)
                if (a.retomar and isinstance(anterior, dict)
                        and anterior.get("quadrantes")):
                    log(f"{ck}: ja concluido, pulando")
                    out[ck] = anterior
                    alvo.write_text(json.dumps(out, indent=2, default=float),
                                    encoding="utf-8")
                    continue
                try:
                    if regime == "ampliacao":
                        out[ck] = rodar_bioma_ampliacao(
                            b, s, arm, usar_sar=not a.sem_sar,
                            bloco_por_bioma=a.bloco_por_bioma, vetor=a.vetor)
                    else:
                        out[ck] = rodar_bioma(b, s, arm, regime == "cadeia",
                                              usar_sar=not a.sem_sar,
                                              bloco_por_bioma=a.bloco_por_bioma,
                                              vetor=a.vetor)
                except (MemoryError, ValueError, AssertionError) as e:
                    log(f"{ck}: ABORTADO — {e}")
                    out[ck] = {"erro": str(e), "tipo": type(e).__name__}
                alvo.write_text(json.dumps(out, indent=2, default=float),
                                encoding="utf-8")
                gc.collect()
    log("===== B2 CONCLUIDO =====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
