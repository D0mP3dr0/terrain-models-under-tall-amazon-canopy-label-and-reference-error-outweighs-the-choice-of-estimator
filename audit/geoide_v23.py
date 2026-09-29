"""EGM2008 geoid undulation on the project grid, point by point.

GEDI and ATL08 heights refer to the WGS84 ellipsoid; Copernicus GLO-30 heights are
orthometric (EGM2008 geoid). The target is their difference, so the geoid undulation
would enter the label in full unless removed.

A single per-biome offset is not enough. Measured undulation N within each 2-degree box:

    biome            N from          to         range within the box
    amazonia          -17.09 m     -8.10 m           9.00 m
    cerrado           -16.57 m    -12.34 m           4.23 m
    mata_atlantica    -11.34 m     -5.63 m           5.71 m
    pantanal           +7.06 m    +15.69 m           8.62 m

A constant per biome removes the median but leaves a ramp of up to ~4.5 m: a smooth,
large-scale field unrelated to canopy or DEM error. A graph network would learn it
easily, and the model would appear to correct the DEM while partly reproducing the
geoid shape.

Method: pyproj transforms EPSG:4979 (WGS84 3D ellipsoidal) to EPSG:3855 (EGM2008
orthometric height) with the official NGA grid, fetched on demand by PROJ from
cdn.proj.org. No assumption about Copernicus enters this path (unlike a bare-soil
offset estimate, which assumes Copernicus is correct over exposed ground).

With the geoid applied, the residual (DSM - ground) over bare soil is no longer a
correction to subtract but a measurement of the actual vertical bias of Copernicus over
exposed ground; a value near zero confirms the whole chain.

    python geoide_v23.py                    # writes geoide_{bioma}.tif
    python geoide_v23.py --conferir         # only measures the interpolation error
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

from fila_aquisicao_alvos import BIOMAS, GRID, Painel, gravar_atomico  # noqa: E402

GEEDIR = Path(r"D:\GNN_TOPO\SATELITES\gee")

# Sampling grid before interpolation. The public EGM2008 grid has a 2.5 arc-minute step
# (~4.6 km); 241 points over 2 degrees give ~0.92 km, five times finer than the source.
# Interpolating to 7200x7200 adds no detail absent from the source, and the check below
# measures the interpolation error instead of assuming it is small.
N_AMOSTRA = 241
ERRO_MAX_M = 0.01          # 1 cm; maximum tolerated interpolation error
LOTE = 2_000_000


def _transformador():
    import pyproj
    pyproj.network.set_network_enabled(True)
    if not pyproj.network.is_network_enabled():
        raise RuntimeError(
            "a rede do PROJ esta desligada e a grade do EGM2008 nao esta local. Sem ela "
            "a transformacao devolve o valor de entrada SEM AVISO, o que produziria "
            "rotulos silenciosamente errados.")
    return pyproj.Transformer.from_crs("EPSG:4979", "EPSG:3855", always_xy=True)


def undulacao(lon, lat, tr=None) -> np.ndarray:
    """N = h_ellipsoidal - H_orthometric, in metres. Positive = geoid above the ellipsoid.

    Single source of this quantity: both the raster and the labels call this function,
    so a datum offset cannot silently diverge between label and product.
    """
    tr = tr or _transformador()
    lon = np.asarray(lon, dtype=np.float64).ravel()
    lat = np.asarray(lat, dtype=np.float64).ravel()
    saida = np.empty(lon.size, dtype=np.float64)
    for i in range(0, lon.size, LOTE):
        f = slice(i, i + LOTE)
        _, _, H = tr.transform(lon[f], lat[f], np.zeros(lon[f].size))
        saida[f] = -np.asarray(H)           # H at h=0 equals -N by definition
    if not np.isfinite(saida).all():
        raise RuntimeError("a transformacao devolveu valor nao finito — grade do "
                           "EGM2008 indisponivel para parte dos pontos")
    return saida


def ortometrica(lon, lat, h_elipsoidal, tr=None) -> np.ndarray:
    """Ellipsoidal lidar height -> orthometric height, comparable to GLO-30."""
    return np.asarray(h_elipsoidal, dtype=np.float64).ravel() - undulacao(lon, lat, tr)


def grade_do_bioma(bioma: str, tr=None) -> tuple:
    """N on the 7200x7200 biome grid by bilinear interpolation. Returns (grid, measured error)."""
    w, s, e, n = BIOMAS[bioma]
    tr = tr or _transformador()

    # coarse sample spanning the whole box, edges included
    lo = np.linspace(w, e, N_AMOSTRA)
    la = np.linspace(n, s, N_AMOSTRA)              # north -> south, like raster rows
    L, A = np.meshgrid(lo, la)
    Ng = undulacao(L, A, tr).reshape(N_AMOSTRA, N_AMOSTRA)

    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((la[::-1], lo), Ng[::-1], method="linear",
                                     bounds_error=False, fill_value=None)

    # Check: 20,000 random points, interpolated value vs. direct transformation
    rng = np.random.default_rng(20260802)
    plon = rng.uniform(w, e, 20_000)
    plat = rng.uniform(s, n, 20_000)
    erro = np.abs(interp(np.stack([plat, plon], axis=1)) - undulacao(plon, plat, tr))
    medido = {"erro_max_m": float(erro.max()), "erro_p99_m": float(np.percentile(erro, 99)),
              "n_pontos_conferidos": int(erro.size)}
    if erro.max() > ERRO_MAX_M:
        raise RuntimeError(
            f"{bioma}: interpolacao do geoide erra ate {erro.max()*100:.1f} cm, acima do "
            f"limite de {ERRO_MAX_M*100:.0f} cm. Aumente N_AMOSTRA.")

    # evaluate on the full grid row by row, to avoid materialising 51.8 M coordinate pairs
    passo_lon = (e - w) / GRID
    passo_lat = (n - s) / GRID
    col = w + (np.arange(GRID) + 0.5) * passo_lon
    saida = np.empty((GRID, GRID), dtype=np.float32)
    for r in range(GRID):
        lat_r = n - (r + 0.5) * passo_lat
        saida[r] = interp(np.stack([np.full(GRID, lat_r), col], axis=1)).astype(np.float32)
    return saida, medido


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--biomas", nargs="+", default=list(BIOMAS))
    ap.add_argument("--conferir", action="store_true",
                    help="mede o erro da interpolacao e nao grava nada")
    a = ap.parse_args()

    import rasterio
    from rasterio.transform import from_bounds

    tr = _transformador()
    painel = Painel([f"geoide:{b}" for b in a.biomas])
    saida = {}
    GEEDIR.mkdir(parents=True, exist_ok=True)
    for b in a.biomas:
        painel.comecar(f"geoide:{b}")
        t0 = time.time()
        N, medido = grade_do_bioma(b, tr)
        medido.update({"n_min_m": float(N.min()), "n_max_m": float(N.max()),
                       "n_medio_m": float(N.mean()),
                       "amplitude_na_caixa_m": float(N.max() - N.min()),
                       "segundos": round(time.time() - t0, 1)})
        painel.log(f"  {b}: N de {N.min():+.2f} a {N.max():+.2f} m "
                   f"(amplitude {N.max()-N.min():.2f} m); interpolacao erra ate "
                   f"{medido['erro_max_m']*100:.2f} cm")
        if not a.conferir:
            alvo = GEEDIR / f"geoide_{b}.tif"
            tmp = alvo.with_suffix(".tif.tmp")
            with rasterio.open(
                    tmp, "w", driver="GTiff", height=GRID, width=GRID, count=1,
                    dtype="float32", crs="EPSG:4326",
                    transform=from_bounds(*BIOMAS[b], GRID, GRID),
                    compress="deflate", predictor=2, tiled=True,
                    blockxsize=512, blockysize=512, BIGTIFF="YES") as dst:
                dst.write(N, 1)
                dst.set_band_description(1, "n_egm2008")
            tmp.replace(alvo)
            medido["arquivo"] = str(alvo)
            painel.log(f"  {b}: {alvo.name} gravado")
        saida[b] = medido
        painel.terminar(f"geoide:{b}", True)
        del N

    (RAIZ / "results").mkdir(exist_ok=True)
    gravar_atomico(RAIZ / "results" / "geoide_v23.json",
                   json.dumps(saida, indent=2, ensure_ascii=False, default=str))
    print(f"\n  {len(saida)} bioma(s) -> {GEEDIR}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
