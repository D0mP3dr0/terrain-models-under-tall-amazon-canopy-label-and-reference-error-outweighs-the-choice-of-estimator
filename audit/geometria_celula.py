"""Cell geometry of the study grid in WGS84 -- the source of the area,
posting and range figures cited in the manuscript.

Rationale: three different cell-size figures were in circulation for this
grid: a flat 30.0 m, a spherical 30.922 m from `grade_v23.GRAU_M = 111.320`
(which overestimates the NORTH-SOUTH step by ~0.7%: 111.320 m/degree is the
length of a degree of LONGITUDE at the equator, while a degree of LATITUDE
there is ~110.574 m), and the ellipsoidal 30.72/30.87 m used by the B2 void
mask (correct). Note the sign of the anisotropy: near the equator the E-W
posting is LARGER than the N-S posting, not smaller ("E-W scaled by cos
latitude" is easy to misread as the opposite). This script fixes the
citable numbers on the WGS84 ellipsoid and measures the error of the
approximations.

Bounding box: Amazon BBOX used in grade_v23 (lon -60..-58, lat -4..-2),
7200^2 grid.

Usage:  python geometria_celula.py
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
SAIDA = RAIZ / "geometria_celula.json"

A = 6_378_137.0                    # WGS84 semi-major axis (m)
F = 1 / 298.257223563
E2 = F * (2 - F)
GRID = 7200
LON0, LAT0, LON1, LAT1 = -60.0, -4.0, -58.0, -2.0
ARCO = (LAT1 - LAT0) / GRID        # 1 arc-second in degrees (2/7200)


def M(lat):                        # meridional radius of curvature
    s2 = math.sin(math.radians(lat)) ** 2
    return A * (1 - E2) / (1 - E2 * s2) ** 1.5


def N(lat):                        # prime-vertical radius of curvature
    s2 = math.sin(math.radians(lat)) ** 2
    return A / math.sqrt(1 - E2 * s2)


def area_faixa(lat_a, lat_b):
    """Area between two parallels, width LON1-LON0, on the ellipsoid (m^2)."""
    def q(lat):
        s = math.sin(math.radians(lat))
        e = math.sqrt(E2)
        return (A ** 2 * (1 - E2) / 2) * (
            s / (1 - E2 * s * s) + math.atanh(e * s) / e)
    dlon = math.radians(LON1 - LON0)
    return dlon * (q(lat_b) - q(lat_a))


def main() -> int:
    rad = math.radians(ARCO)
    postings = {}
    for lat in (-4.0, -3.0, -2.0):
        ns = M(lat) * rad
        ew = N(lat) * math.cos(math.radians(lat)) * rad
        postings[str(lat)] = {"ns_m": ns, "ew_m": ew,
                              "diagonal_m": math.hypot(ns, ew)}
    lat_c = (LAT0 + LAT1) / 2
    ns_c = M(lat_c) * rad
    ew_c = N(lat_c) * math.cos(math.radians(lat_c)) * rad

    area = area_faixa(LAT0, LAT1)
    area_30m = (GRID * 30.0) ** 2
    area_esf = (GRID * 30.9222) ** 2 * math.cos(math.radians(lat_c))

    out = {
        "gerado_em": time.strftime("%Y-%m-%d %H:%M"),
        "elipsoide": "WGS84", "bbox": [LON0, LAT0, LON1, LAT1],
        "grade": f"{GRID}x{GRID} (1 arcsec)",
        "posting_por_latitude": postings,
        "posting_centro": {"ns_m": ns_c, "ew_m": ew_c},
        "area_km2": area / 1e6,
        "aproximacoes_superadas": {
            "30m_por_celula_km2": area_30m / 1e6,
            "esferica_grau111320_km2": area_esf / 1e6,
            "nota": "30 m subestima a area em ~5,5%; a esferica de "
                    "grade_v23 superestima o passo N-S em ~0,7%"},
        "alcances_gnn": {
            "k2_m": 2 * math.hypot(ns_c, ew_c),
            "k4_m": 4 * math.hypot(ns_c, ew_c),
            "k8_m": 8 * math.hypot(ns_c, ew_c),
            "nota": "bola de k saltos no reticulado 8-conexo: k celulas em "
                    "Chebyshev; alcance maximo = k x diagonal"},
        "conversoes_texto": {
            "buffer_67_celulas_km": 67 * ns_c / 1000,
            "reserva_1800_celulas_km": 1800 * ns_c / 1000,
            "separacao_3600_celulas_km": 3600 * ns_c / 1000,
            "espaco_130m_em_celulas": 130.0 / ns_c},
    }
    print(f"  posting no centro (lat {lat_c}): N-S {ns_c:.3f} m | "
          f"E-O {ew_c:.3f} m | diagonal {math.hypot(ns_c, ew_c):.3f} m")
    print(f"  area WGS84: {area/1e6:,.1f} km^2  "
          f"(30m: {area_30m/1e6:,.1f} | esferica: {area_esf/1e6:,.1f})")
    print(f"  alcance k=2: {out['alcances_gnn']['k2_m']:.1f} m | "
          f"k=4: {out['alcances_gnn']['k4_m']:.1f} m | "
          f"k=8: {out['alcances_gnn']['k8_m']:.1f} m")
    print(f"  buffer 67 cel: {out['conversoes_texto']['buffer_67_celulas_km']:.2f} km | "
          f"reservas 3600 cel: {out['conversoes_texto']['separacao_3600_celulas_km']:.1f} km")
    SAIDA.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                     encoding="utf-8")
    print(f"  -> {SAIDA.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
