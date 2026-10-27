"""The study-area grid: one origin shared by the 10 m, 90 m and Wald grids.

Pure Python, so it can be checked without Earth Engine. The square is snapped
to whole Wald blocks (90 m × wald_factor) in UTM, which moves it by at most
half a block.
"""
from __future__ import annotations

import math


def utm_forward(lon: float, lat: float, zone: int, north: bool = True) -> tuple[float, float]:
    """WGS84 longitude/latitude to UTM easting/northing (Krüger series, sub-millimetre)."""
    a, f = 6378137.0, 1 / 298.257223563
    k0, e0 = 0.9996, 500000.0
    n = f / (2 - f)
    A = a / (1 + n) * (1 + n**2 / 4 + n**4 / 64)
    alpha = [n / 2 - 2 * n**2 / 3 + 5 * n**3 / 16, 13 * n**2 / 48 - 3 * n**3 / 5, 61 * n**3 / 240]
    lon0 = math.radians(zone * 6 - 183)
    phi, lam = math.radians(lat), math.radians(lon) - lon0
    e = math.sqrt(f * (2 - f))
    t = math.sinh(math.atanh(math.sin(phi)) - e * math.atanh(e * math.sin(phi)))
    xi = math.atan2(t, math.cos(lam))
    eta = math.atanh(math.sin(lam) / math.sqrt(1 + t * t))
    x = eta + sum(al * math.cos(2 * j * xi) * math.sinh(2 * j * eta) for j, al in enumerate(alpha, 1))
    y = xi + sum(al * math.sin(2 * j * xi) * math.cosh(2 * j * eta) for j, al in enumerate(alpha, 1))
    return e0 + k0 * A * x, (0.0 if north else 10000000.0) + k0 * A * y


def make_grid(job: dict) -> dict:
    crs = job['crs']
    zone, north = int(crs[-2:]), crs.startswith('EPSG:326')
    cx, cy = utm_forward(job['area']['center_lon'], job['area']['center_lat'], zone, north)
    g = job['grid']
    fine, coarse, wf = g['fine_m'], g['coarse_m'], g['wald_factor']
    block = coarse * wf
    n_blocks = max(1, round(job['area']['side_km'] * 1000 / block))
    side = n_blocks * block
    x0 = round((cx - side / 2) / block) * block
    y0 = round((cy - side / 2) / block) * block
    x1, y1 = x0 + side, y0 + side
    xm = x0 + (n_blocks // 2) * block
    return {
        'crs': crs, 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1, 'xm': xm, 'side_m': side,
        'fine_m': fine, 'coarse_m': coarse, 'wald_m': block, 'wald_factor': wf,
        'fine_t': [fine, 0, x0, 0, -fine, y1],
        'coarse_t': [coarse, 0, x0, 0, -coarse, y1],
        'wald_t': [block, 0, x0, 0, -block, y1],
        'check_t': [coarse * 10, 0, x0, 0, -coarse * 10, y1],   # 900 m, for cheap inventory sums
        'n_fine': side // fine, 'n_coarse': side // coarse, 'n_wald': n_blocks,
        'center_utm': [round(cx, 2), round(cy, 2)],
    }


def utm_point(job: dict, lon: float, lat: float) -> tuple[float, float]:
    crs = job['crs']
    return utm_forward(lon, lat, int(crs[-2:]), crs.startswith('EPSG:326'))
