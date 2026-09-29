"""The eight topographic features, computed directly from elevation -- without the
graph in the path.

WHY THIS FILE EXISTS
---------------------------
Columns 0 to 7 of the feature vector used to come from `dem.x`, computed during graph
construction. After the retired graph cache was dropped, the downstream correction
routine became a patch: it received columns already computed with the wrong geometry
and recomputed four of them on top. A patch on a patch -- elevation arrived normalized,
was denormalized back to meters, differentiated, and normalized again.

Here all eight are computed at once from the `glo30` raster, the same surface the
target uses.

THE GEOMETRY, WHICH WAS THE DEFECT
------------------------------
The cell measures `2/7200` of a degree. On the north-south axis this gives 30.922 m --
not the flat 30.0 m the old graph builder assumed, a 3.1% error on every distance. On
the east-west axis it shortens with `cos(latitude)`, which used to be ignored: 0.1% in
the Amazon and 5.2% in the Atlantic Forest. The two errors compounded, with different
signs per biome -- which is poison for cross-biome comparison, the central axis of the
article. See `grade_v23`.

`px` is per ROW, because `cos` varies within the quadrant.

THE EIGHT FEATURES, AND WHAT EACH ONE ACTUALLY IS
------------------------------------
    elevation      (z - global median) / elev_std
    slope          degrees / 90
    aspect_cos     cosine of slope orientation
    aspect_sin     sine
    curvature      Laplacian, times cell area, divided by elev_std
    tpi            z minus the 3x3 mean, divided by elev_std
    amplitude3x3   3x3 max minus 3x3 min, divided by elev_std. This is NOT Riley's TRI,
                   which sums the squared differences to the eight neighbors. The old
                   name described a different measure, and a table labeling this
                   column "TRI" would be incorrect.
    roughness      3x3 standard deviation, divided by elev_std

ALL SCALES ARE GLOBAL
----------------------------
`elev_std` and the global median, from `escalas_v23.json`, apply to all sixteen
quadrants. Normalizing by the quadrant's own statistics erases the difference between
biomes and would prevent both chained transfer learning and a single model fit across
the four biomes.
"""
from __future__ import annotations

import numpy as np

import grade_v23 as G

QUAD_N = G.QUAD_N
SLOPE_MAX = 90.0
NOMES = ["elevation", "slope", "aspect_cos", "aspect_sin",
         "curvature", "tpi", "amplitude3x3", "roughness"]


def _janela3(z: np.ndarray, fn) -> np.ndarray:
    """Reduction over the 3x3 window, with replicated border."""
    p = np.pad(z, 1, mode="edge")
    pil = np.stack([p[i:i + z.shape[0], j:j + z.shape[1]]
                    for i in range(3) for j in range(3)])
    return fn(pil, axis=0)


def metros(bioma: str, quad: str):
    """(px per row, py) in meters -- the correct cell geometry."""
    mx, my, lat = G.metros_por_celula(bioma, quad)
    return mx, my, lat


def montar(bioma: str, quad: str, z: np.ndarray, elev_std: float,
           elev_mediana: float) -> tuple:
    """Returns `(matrix (N, 8), names)` from the elevation in METERS.

    `z` comes in the natural node order of the quadrant, which is raster order: row by
    row, 3600 by 3600.
    """
    if z.size != QUAD_N * QUAD_N:
        raise ValueError(f"{bioma}/{quad}: {z.size} valores de elevacao, esperado "
                         f"{QUAD_N * QUAD_N}")
    zz = z.astype(np.float64).reshape(QUAD_N, QUAD_N)
    px, py, _ = metros(bioma, quad)
    es = max(float(elev_std), 1e-9)

    gy = np.gradient(zz, py, axis=0)
    gx = np.gradient(zz, axis=1) / px[:, None]
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    aspect = np.remainder(np.arctan2(-gx, gy) + 2 * np.pi, 2 * np.pi)
    gxx = np.gradient(gx, axis=1) / px[:, None]
    gyy = np.gradient(gy, py, axis=0)
    # Curvature normalized by a PHYSICAL scale -- cell area over `elev_std` -- not by
    # a quantile of the quadrant itself. A quantile would give a per-quadrant scale
    # and would corrupt weight transfer, which is the defect this file avoids.
    area = py * float(px.mean())
    curv = (gxx + gyy) * area / es
    del gx, gy, gxx, gyy

    media3 = _janela3(zz, np.mean)
    cols = [
        (zz - elev_mediana) / es,
        slope / SLOPE_MAX,
        np.cos(aspect),
        np.sin(aspect),
        curv,
        (zz - media3) / es,
        (_janela3(zz, np.max) - _janela3(zz, np.min)) / es,
        _janela3(zz, np.std) / es,
    ]
    X = np.stack([c.reshape(-1) for c in cols], axis=1).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), list(NOMES)
