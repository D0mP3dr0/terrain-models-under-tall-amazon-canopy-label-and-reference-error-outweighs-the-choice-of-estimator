"""Node-geometry side rasters produced by DIRECT READING from the sources.

Replaces the provisional version (mean of the 4 area-grid neighbors) for the
two fields that matter to the judging pipeline:
  - dossel_eth: resampled DIRECTLY from the native 10 m tif (24000x24000,
    same bounding box) onto the NODE grid by area averaging (rasterio
    Resampling.average, two sum/weight passes). This is the PRIMARY
    stratifier of the paired test.
  - gedtm30: window from the official COG (native pixel 0.00025 degrees --
    neither the area grid nor the node grid coincides with it; any grid
    involves interpolation) reprojected bilinearly DIRECTLY onto the node
    grid. This removes the double resampling (bilinear->area->mean of 4)
    that was corrupting Table IV.
  - agua_jrc: no local native source (the water-occurrence tif is already
    on the project's area grid). Keeps the node-averaged (mean of 4)
    version, hardlinked from the earlier pipeline stage -- declared in the
    sidecar. Role in the judging pipeline: exclusion mask; low sensitivity.

NODE GRID (same as the native gridding script, dlon=dlat=+0.5 px):
  cell (i,j) covers lon [LON0 + j*P, LON0 + (j+1)*P], LON0 = -60 + P/2
                     lat [LAT1 - (i+1)*P, LAT1 - i*P], LAT1 = -2  - P/2
  P = 2/7200 degrees. East/south borders: the ETH source ends at -58/-4, so
  the last column/row only has half a source cell. Caveat: frac_valida does
  NOT measure geometric coverage -- Resampling.average on the mask only
  averages over the covered part, so a border cell with half a source cell
  that is 100% valid comes out with frac_valida = 1.0, and the judging
  script's cob_eth >= 0.8 filter does NOT exclude it; its mean value only
  represents half a cell. sd_interno is a dead field (zeros), inherited from
  the original layout.

WRITING: new destination directory; glo30/fabdem/labels are hardlinked from
the ORIGINALS (read-only); any write goes to a temp file + os.replace and
fails if the destination has nlink > 1 (a hardlink must never be opened for
writing).

This version fixes three defects of an earlier one:
(i) an earlier version's fractional COG window was rounded down by read()
while windows.transform used the fractional offset -> a 0.1111 px (3.09 m)
residual GEDTM30 offset; the window is now anchored on INTEGER offsets
(floor/ceil) with an assertion, so t_jan is exact; (ii) the border-coverage
docstring corrected (frac_valida is not geometric coverage); (iii) the
destination transform and the ETH sum/weight step were CONFIRMED by an
independent recomputation (1.8e-6 m over 400 cells). GEDTM30 numbers from
before this fix have an incorrect second decimal place.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")

import rasterio                                      # noqa: E402
from rasterio.transform import from_bounds as tr_from_bounds  # noqa: E402
from rasterio.warp import Resampling, reproject      # noqa: E402
from rasterio.windows import from_bounds as win_from_bounds   # noqa: E402

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import proveniencia as PROV  # noqa: E402

SRC_LATERAIS = Path(r"D:\GNN_TOPO\SATELITES\laterais")
SRC_NATIVA_V2B = Path(r"D:\GNN_TOPO\SATELITES\laterais_nativa")
DST = Path(r"D:\GNN_TOPO\SATELITES\laterais_nativa_diretas")
ETH_TIF = Path(r"D:\GNN_TOPO\SATELITES\dossel_exogeno"
               r"\ETH_CanopyHeight_2020_10m_lon-60_-58_lat-4_-2.tif")
COG_GEDTM30 = ("/vsicurl/https://s3.opengeohub.org/global/edtm/"
               "gedtm_rf_m_30m_s_20060101_20151231_go_epsg.4326.3855_v20250611.tif")
ESCALA_GEDTM = 10.0
NODATA_ETH = 255

GRID = 7200
Q = 3600
P = 2.0 / GRID
# node-grid bounds (dlon = dlat = +0.5 px)
LON_O, LON_E = -60.0 + P / 2, -58.0 + P / 2
LAT_S, LAT_N = -4.0 - P / 2, -2.0 - P / 2
DESTINO = tr_from_bounds(LON_O, LAT_S, LON_E, LAT_N, GRID, GRID)
QUADS = {"Q1": (0, 0), "Q2": (0, Q), "Q3": (Q, 0), "Q4": (Q, Q)}
LINKAR_ORIGINAIS = ["glo30", "fabdem", "rotulos"]


def log(m=""):
    print(m, flush=True)


def gravar_npz(alvo: Path, **campos):
    if alvo.exists() and os.stat(alvo).st_nlink > 1:
        alvo.unlink()                        # nunca escrever por cima de hardlink
    tmp = alvo.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **campos)
    os.replace(tmp, alvo)
    if os.stat(alvo).st_nlink != 1:
        raise RuntimeError(f"{alvo.name}: nlink != 1 apos gravar")


def por_quadrante(M: np.ndarray) -> dict:
    return {q: M[r0:r0 + Q, c0:c0 + Q].reshape(-1) for q, (r0, c0) in QUADS.items()}


def eth_direto() -> dict:
    t0 = time.time()
    with rasterio.open(ETH_TIF) as src:
        a = src.read(1)
        valido = (a != NODATA_ETH).astype(np.float32)
        soma = np.zeros((GRID, GRID), np.float32)
        peso = np.zeros((GRID, GRID), np.float32)
        reproject(np.where(a == NODATA_ETH, 0, a).astype(np.float32), soma,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=DESTINO, dst_crs=src.crs,
                  resampling=Resampling.average)
        reproject(valido, peso,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=DESTINO, dst_crs=src.crs,
                  resampling=Resampling.average)
    with np.errstate(invalid="ignore", divide="ignore"):
        media = np.where(peso > 0, soma / peso, np.nan).astype(np.float32)
    sd = np.zeros_like(media)                # within-cell sd: not used by the lidar judge
    hq, fq, sq = por_quadrante(media), por_quadrante(peso), por_quadrante(sd)
    for q in QUADS:
        gravar_npz(DST / f"dossel_eth_amazonia_{q}.npz",
                   h_dossel=hq[q].astype(np.float32),
                   sd_interno=sq[q].astype(np.float32),
                   frac_valida=fq[q].astype(np.float32))
    ok = np.isfinite(media)
    log(f"  dossel_eth DIRETO 10m->no: com dado {ok.mean():.1%} | mediana "
        f"{np.nanmedian(media):.2f} m | frac_valida<1 na borda leste/sul | {time.time()-t0:.0f} s")
    return {"metodo": "Resampling.average do tif 10 m nativo direto na grade de nos (duas passagens soma/peso)",
            "frac_com_dado": float(ok.mean()), "h_mediana": float(np.nanmedian(media))}


def gedtm30_direto() -> dict:
    t0 = time.time()
    margem = 4 * P
    with rasterio.open(COG_GEDTM30) as src:
        # A fractional window used to be rounded down by read() while
        # windows.transform used the fractional offset -> a 0.1111 px
        # (3.09 m) residual shift. Anchor on INTEGER offsets before reading
        # and before deriving t_jan.
        jan = win_from_bounds(LON_O - margem, LAT_S - margem, LON_E + margem, LAT_N + margem, src.transform)
        jan = jan.round_offsets(op="floor").round_lengths(op="ceil")
        if not (float(jan.col_off).is_integer() and float(jan.row_off).is_integer()):
            raise RuntimeError(f"janela do COG com offset fracionario: {jan}")
        bruto = src.read(1, window=jan)
        nodata = src.nodata
        t_jan = rasterio.windows.transform(jan, src.transform)
        if bruto.shape != (int(jan.height), int(jan.width)):
            raise RuntimeError(f"read() devolveu {bruto.shape} para janela {jan}")
    a = bruto.astype(np.float32)
    a[bruto == nodata] = np.nan
    a /= ESCALA_GEDTM
    saida = np.full((GRID, GRID), np.nan, np.float32)
    reproject(a, saida, src_transform=t_jan, src_crs="EPSG:4326",
              dst_transform=DESTINO, dst_crs="EPSG:4326",
              src_nodata=np.nan, dst_nodata=np.nan,
              resampling=Resampling.bilinear)
    dq = por_quadrante(saida)
    for q in QUADS:
        gravar_npz(DST / f"gedtm30_amazonia_{q}.npz", DTM=dq[q].astype(np.float32))
    v = saida[np.isfinite(saida)]
    log(f"  gedtm30 DIRETO COG->no (bilinear): {len(v):,} validos | mediana "
        f"{np.median(v):.2f} m | janela {bruto.shape} | {time.time()-t0:.0f} s")
    return {"metodo": "janela do COG oficial reprojetada bilinear direto na grade de nos",
            "cog": COG_GEDTM30, "pixel_nativo_graus": 0.00025,
            "n_validos": int(len(v)), "mediana_m": float(np.median(v))}


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    fontes = []
    # 1) hardlinks to the ORIGINALS (read-only)
    for q in QUADS:
        for base in LINKAR_ORIGINAIS:
            s = SRC_LATERAIS / f"{base}_amazonia_{q}.npz"
            d = DST / s.name
            if not s.exists():
                raise SystemExit(f"falta {s}")
            if d.exists() or d.is_symlink():
                d.unlink()
            os.link(s, d)
            fontes.append(s)
    log(f"  hardlink: {LINKAR_ORIGINAIS} (originais de laterais/)")
    # 2) water: keeps the node-averaged (mean of 4) version -- hardlinked from laterais_nativa
    for q in QUADS:
        s = SRC_NATIVA_V2B / f"agua_jrc_amazonia_{q}.npz"
        d = DST / s.name
        if not s.exists():
            raise SystemExit(f"falta {s} (rodar preparar_laterais_nativa.py v2b antes)")
        if d.exists() or d.is_symlink():
            d.unlink()
        os.link(s, d)
        fontes.append(s)
    log("  agua_jrc: v2b (media de 4) por hardlink — SEM fonte nativa local; declarado")
    # 3) direct reads
    rel_eth = eth_direto()
    fontes.append(ETH_TIF)
    rel_ged = gedtm30_direto()
    PROV.gravar(DST / "_proveniencia.json", {
        "versao": ("v3 direct, script v2 (after the independent check): ETH directly from "
                   "the 10 m product; GEDTM30 directly from the COG with a window at integer "
                   "offsets; water = v2b (mean of 4, no local native source); "
                   "glo30/fabdem/labels via hardlink of the originals"),
        "grade_de_nos": {"LON_O": LON_O, "LAT_N": LAT_N, "P": P, "GRID": GRID,
                         "convencao": "identica a gridar_lidar_eba_v3_nativa.py (dlon=dlat=+0,5 px)"},
        "dossel_eth": rel_eth, "gedtm30": rel_ged,
        "agua_jrc": {"metodo": "media dos 4 vizinhos da grade de area (v2b), por hardlink",
                     "motivo": "gee/agua_jrc_amazonia.tif ja e a grade de area do projeto; nao ha fonte nativa local"},
        "status": ("script v2 re-read: transform and ETH CONFIRMED, GEDTM30 corrected "
                   "(integer window) after v1 was REFUTED"),
        "ressalva_frac_valida": "frac_valida nao mede cobertura geometrica na borda leste/sul (media so sobre a parte coberta)",
        "destino": str(DST),
    }, fontes_lidas=fontes, script=__file__)
    log(f"  -> {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
