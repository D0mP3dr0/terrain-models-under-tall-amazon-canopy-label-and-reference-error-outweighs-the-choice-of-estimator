"""V23 -- spectral/SAR composites and surface-regime estimator.

Nothing here is persisted in the data artifact: the composites and the regime are
derived at training time from (a) dem.x of the frozen graph and (b) the SAR side
raster of nodes_full.

dem.x column convention (v18.5 graph):
    0..7   topographic (elev_norm, slope/90, aspect_cos, aspect_sin,
           curvature, tpi/std, tri/std, roughness/std)
    8..13  B02, B03, B04, B08, NDVI, NDWI
    14     has_lidar          <- NEVER enters as a feature
    15     normalized z_lidar <- NEVER enters as a feature (it is the target)
    16     reserved
"""
from __future__ import annotations

import numpy as np

COL_B02, COL_B03, COL_B04, COL_B08 = 8, 9, 10, 11
COL_NDVI, COL_NDWI = 12, 13
COL_HAS_LIDAR, COL_Z_LIDAR = 14, 15

CLASSES = ("agua", "solo_exposto", "dossel_denso", "dossel_esparso")
EPS = 1e-6


# ────────────────────────── composites ──────────────────────────
def composicoes(x: np.ndarray, vv: np.ndarray, vh: np.ndarray) -> dict:
    """Ten spectral/SAR composites. x = dem.x [N,17]; vv/vh in dB [N]."""
    b02, b03 = x[:, COL_B02], x[:, COL_B03]
    b04, b08 = x[:, COL_B04], x[:, COL_B08]

    bsi = ((b04 + b02) - b08) / ((b04 + b02) + b08 + EPS)
    bright = (b02 + b03 + b04 + b08) / 4.0
    shadow = 1.0 - np.sqrt(np.clip((b02**2 + b03**2 + b04**2) / 3.0, 0.0, None))

    # dB -> linear power for the RVI (the physical definition requires power)
    p_vv = np.power(10.0, vv / 10.0)
    p_vh = np.power(10.0, vh / 10.0)
    rvi = np.clip(4.0 * p_vh / (p_vv + p_vh + EPS), 0.0, 2.0)

    return {
        "ndvi": x[:, COL_NDVI],
        "ndwi": x[:, COL_NDWI],
        "bsi": np.clip(bsi, -1.0, 1.0),
        "bright": bright,
        "shadow": np.clip(shadow, 0.0, 1.0),
        "vv": vv,
        "vh": vh,
        "rvi": rvi,
        "sar_diff": vh - vv,
        "sar_sum": vv + vh,
    }


# ────────────────────────── regime ──────────────────────────
# Prototypes in ABSOLUTE units. Standardizing per biome (a discarded alternative)
# erases the physical meaning: in a predominantly vegetated biome the NDWI median is
# ~-0.5, so "relatively high NDWI" is still vegetation, and the water class ended up
# capturing 20%+ of the nodes. The water criterion is NDWI > 0 and strongly negative
# VV, not "above the local median".
_ORDEM = ("ndvi", "ndwi", "bsi", "vv", "rvi")

PROTOTIPOS = {
    #                 ndvi  ndwi   bsi     vv    rvi
    "agua":         (-0.05, +0.30, -0.30, -17.0, 0.35),
    "solo_exposto": (+0.15, -0.35, +0.15,  -8.0, 0.45),
    "dossel_denso": (+0.80, -0.70, -0.55, -11.0, 0.95),
    "dossel_esparso": (+0.45, -0.50, -0.15, -9.0, 0.70),
}
# Characteristic scale of each variable (physical unit, not a data statistic).
ESCALAS = {"ndvi": 0.20, "ndwi": 0.20, "bsi": 0.18, "vv": 3.5, "rvi": 0.25}


def regime(comp: dict, T: float = 1.0) -> np.ndarray:
    """Soft membership [N,4] summing to 1, by distance to the physical prototype.

    T = temperature (hyperparameter, used in the sensitivity analysis).
    Low T -> near-hard assignment; high T -> diffuse membership at transitions.
    """
    V = np.stack([comp[k].astype(np.float32) for k in _ORDEM], axis=1)      # [N,5]
    esc = np.array([ESCALAS[k] for k in _ORDEM], dtype=np.float32)
    d2 = np.empty((V.shape[0], len(CLASSES)), dtype=np.float32)
    for k, nome in enumerate(CLASSES):
        p = np.array(PROTOTIPOS[nome], dtype=np.float32)
        d2[:, k] = (((V - p) / esc) ** 2).sum(axis=1)
    escore = -d2 / (2.0 * max(T, EPS))
    escore -= escore.max(axis=1, keepdims=True)
    e = np.exp(escore)
    return e / (e.sum(axis=1, keepdims=True) + EPS)


def diagnostico(r: np.ndarray, comp: dict) -> dict:
    """Internal separability: dominant fraction per class and mean signature."""
    dom = r.argmax(axis=1)
    out = {"fracao_por_classe": {}, "assinatura_media": {}, "confianca_media": float(r.max(axis=1).mean())}
    for k, nome in enumerate(CLASSES):
        m = dom == k
        out["fracao_por_classe"][nome] = float(m.mean())
        if m.any():
            out["assinatura_media"][nome] = {
                v: float(np.median(comp[v][m])) for v in ("ndvi", "ndwi", "bsi", "vv", "rvi")
            }
    return out
