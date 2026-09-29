"""Reconstructs the grid topology -- what is left of the retired graph cache.

WHY THIS FILE EXISTS
  The 15.5 GB-per-quadrant `.pt` files were retired after verifying they were the cache
  of a deterministic computation, not run data. What training still took from them was
  the topology: `edge_index` for the dem-dem edges and `edge_attr` with distance and
  elevation difference.

  This is reconstructible from arithmetic. The grid is regular, 3600x3600 over one
  degree, and the eight nearest neighbors of a cell are the eight surrounding cells --
  there is no kNN to run nor approximation to make. Elevation comes from the `glo30`
  raster, which is already on the grid.

  Reconstructing also FIXES an inherited defect. The old graph computed
  `dist = sqrt(drow^2 + dcol^2) * resolution`, i.e. in pixel space times a constant,
  treating a degree of longitude as if it were worth the same as a degree of latitude.
  In EPSG:4326 the east-west degree shortens with `cos(latitude)`: about 0.1% in the
  Amazon and 5.2% in the Atlantic Forest. Here the distance comes out correct from the
  source, instead of being patched downstream.

GENERATED, NOT STORED
  There are ~103 million edges per quadrant. Storing them would cost about 2.5 GB per
  quadrant, 40 GB in total, to reproduce in seconds a deterministic computation -- the
  same mistake the `.pt` files made. `--cache` exists for whoever wants to trade disk
  for time.

USAGE
  python grade_v23.py --validar
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np

# Path parametrized by environment variable: the default keeps the usual
# machine unchanged. The variable exists because running the same code on
# another machine (Colab/VM) via monkeypatch does NOT work: the script runs
# as `__main__`, with its own globals, and constants patched on the
# imported module never reach it.
LATDIR = Path(os.environ.get("V23_LATDIR", r"D:\GNN_TOPO\SATELITES\laterais"))
GRID, QUAD_N = 7200, 3600
GRAU_M = 111_320.0
# North-south cell step, in meters. 2 degrees divided by 7200 gives 2.7778e-4 degree,
# which at 111,320 m per degree gives 30.922 m -- NOT the flat 30.0 m the old generator
# assumed. The east-west axis is this value times `cos(latitude)`.
CELULA_M = (2.0 / GRID) * GRAU_M
BBOX = {
    "amazonia": [-60.0, -4.0, -58.0, -2.0],
    "cerrado": [-48.0, -16.0, -46.0, -14.0],
    "mata_atlantica": [-42.0, -20.0, -40.0, -18.0],
    "pantanal": [-58.0, -20.0, -56.0, -18.0],
}
# The eight neighbors, in (drow, dcol) order. Row grows towards the SOUTH.
VIZINHOS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def metros_por_celula(bioma: str, quad: str):
    """Cell size in meters, per ROW -- the east-west axis shortens with latitude."""
    lon_min, lat_min, lon_max, lat_max = BBOX[bioma]
    passo = (lat_max - lat_min) / GRID
    linha0 = 0 if quad in ("Q1", "Q2") else QUAD_N
    lat = lat_max - (linha0 + np.arange(QUAD_N) + 0.5) * passo
    my = passo * GRAU_M                                  # north-south: constant
    mx = my * np.cos(np.radians(lat))                    # east-west: per row
    return mx.astype(np.float64), float(my), lat


def escala_metrica(bioma: str, quad: str) -> tuple[float, float]:
    """(mx, my) CONSTANT for the quadrant -- the affine map used to measure distance
    between nodes.

    Different from `metros_por_celula`, which returns `mx` per row. The distinction
    matters and is not cosmetic:

      - A grid edge is LOCAL: it links immediate neighbors, so the `mx` of its own row
        is correct, and that is what `arestas` uses.
      - The distance between any two anchors crosses the whole quadrant. Scaling each
        point by the `mx` of its own row is not a map: two points could share the same
        `x` coordinate and have different longitudes. The correct choice is a single
        scale, and the mean over the quadrant minimizes the maximum error.

    Residual error of the affine map within a quadrant (1 degree of latitude): +-0.45%
    in the Atlantic Forest, less elsewhere. Against the 3.1% of the flat `30.0` it
    replaces.
    """
    mx, my, _ = metros_por_celula(bioma, quad)
    return float(mx.mean()), float(my)


def posicao_metros(bioma: str, quad: str, idx: np.ndarray) -> np.ndarray:
    """Metric coordinate (north-south, east-west) of each node, in meters.

    `idx` is the linear index of the node in the quadrant's 3600x3600 grid, in raster
    order. Returns `(N, 2)` float32, in the same order as `idx`.

    REPLACES `pos_celulas * 30.0`, which was the inherited defect: the cell does not
    measure 30 m, it measures `2/7200` of a degree, which gives 30.922 m on the
    north-south axis and less on the east-west axis. Every distance derived from that
    came out 3.1% short -- including the error-vs-distance-to-anchor curve, which is a
    published result.
    """
    mx, my = escala_metrica(bioma, quad)
    idx = np.asarray(idx, dtype=np.int64)
    lin = idx // QUAD_N
    col = idx % QUAD_N
    return np.stack([lin * my, col * mx], axis=1).astype(np.float32)


def arestas(bioma: str, quad: str, z: np.ndarray | None = None):
    """Returns `(edge_index, edge_attr)` for the quadrant grid.

    `edge_index` is (2, E) with source and destination; `edge_attr` is (E, 2) with
    distance in meters and elevation difference in meters. Both directions of each
    pair are present, because every node generates an edge to each of its eight
    neighbors.

    `z` is the per-node elevation in meters. If not given, it is read from the
    `glo30` raster.
    """
    if z is None:
        z = np.load(LATDIR / f"glo30_{bioma}_{quad}.npz")["DEM"].astype(np.float64)
    if z.size != QUAD_N * QUAD_N:
        raise ValueError(f"{bioma}/{quad}: elevacao com {z.size} valores, "
                         f"esperado {QUAD_N * QUAD_N}")

    mx, my, _ = metros_por_celula(bioma, quad)
    lin = np.repeat(np.arange(QUAD_N, dtype=np.int32), QUAD_N)
    col = np.tile(np.arange(QUAD_N, dtype=np.int32), QUAD_N)

    src_l, dst_l, dist_l = [], [], []
    for dl, dc in VIZINHOS:
        l2, c2 = lin + dl, col + dc
        dentro = (l2 >= 0) & (l2 < QUAD_N) & (c2 >= 0) & (c2 < QUAD_N)
        s = (lin[dentro].astype(np.int64) * QUAD_N + col[dentro])
        d = (l2[dentro].astype(np.int64) * QUAD_N + c2[dentro])
        # Distance using the `mx` of the edge's MIDPOINT row: for `dl` from -1 to 1 the
        # difference between using the source, the destination or the mean stays in
        # the fifth decimal place, but the mean is the one that does not favor one
        # direction over the other.
        mxm = 0.5 * (mx[lin[dentro]] + mx[l2[dentro]])
        dist = np.hypot(dc * mxm, dl * my)
        src_l.append(s); dst_l.append(d); dist_l.append(dist)
        del l2, c2, dentro, s, d, mxm, dist

    src = np.concatenate(src_l); dst = np.concatenate(dst_l)
    dist = np.concatenate(dist_l)
    del src_l, dst_l, dist_l

    edge_index = np.stack([src, dst]).astype(np.int64)
    edge_attr = np.stack([dist, z[dst] - z[src]], axis=1).astype(np.float32)
    return edge_index, edge_attr


def validar(bioma: str, quad: str, ref: Path) -> dict:
    """Compares the reconstructed grid against the preserved reference quadrant."""
    import torch
    d = torch.load(ref, map_location="cpu", weights_only=False, mmap=True)
    et = ("dem", "adjacent_to", "dem")
    ei_ref = d[et].edge_index.numpy()
    ea_ref = d[et].edge_attr.numpy()
    n_ref = int(d["dem"].x.shape[0])
    del d

    ei, ea = arestas(bioma, quad)
    # Edge sets as a unique key source*N + destination
    N = QUAD_N * QUAD_N
    k_ref = ei_ref[0].astype(np.int64) * N + ei_ref[1]
    k_new = ei[0] * N + ei[1]
    sr, sn = set(k_ref.tolist()), set(k_new.tolist())

    # Distance: compare only on the common edges, in the same order
    ordem_ref = {k: i for i, k in enumerate(k_ref.tolist())}
    comuns = np.array(sorted(sr & sn), dtype=np.int64)
    idx_ref = np.array([ordem_ref[k] for k in comuns.tolist()], dtype=np.int64)
    pos_new = {k: i for i, k in enumerate(k_new.tolist())}
    idx_new = np.array([pos_new[k] for k in comuns.tolist()], dtype=np.int64)

    d_ref = ea_ref[idx_ref, 0].astype(np.float64)
    d_new = ea[idx_new, 0].astype(np.float64)
    razao = d_new / np.maximum(d_ref, 1e-9)

    return {
        "nos_referencia": n_ref, "nos_reconstruidos": N,
        "arestas_referencia": int(ei_ref.shape[1]),
        "arestas_reconstruidas": int(ei.shape[1]),
        "so_na_referencia": len(sr - sn), "so_na_reconstruida": len(sn - sr),
        "comuns": int(comuns.size),
        "dist_razao_mediana": round(float(np.median(razao)), 6),
        "dist_razao_min": round(float(razao.min()), 6),
        "dist_razao_max": round(float(razao.max()), 6),
    }


def main() -> int:
    args = sys.argv[1:]
    if "--validar" in args:
        ref = Path(r"D:\GNN_TOPO\graph_referencia\amazonia_v18.5_Q1_gpu.pt")
        if not ref.exists():
            print(f"  referencia ausente: {ref}")
            return 1
        print("  validando contra o quadrante de referencia preservado", flush=True)
        r = validar("amazonia", "Q1", ref)
        for k, v in r.items():
            print(f"    {k:26s} {v}")
        return 0

    for b in BBOX:
        for q in ("Q1", "Q2", "Q3", "Q4"):
            ei, ea = arestas(b, q)
            print(f"  {b}/{q}: {ei.shape[1]:,} arestas | dist mediana "
                  f"{np.median(ea[:, 0]):.2f} m | dz mediano "
                  f"{np.median(np.abs(ea[:, 1])):.3f} m", flush=True)
            del ei, ea
    return 0


if __name__ == "__main__":
    sys.exit(main())
