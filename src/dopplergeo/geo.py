"""Geodesy helpers: local East-North-Up frames, slant distances, grid cells.

Everything works on numpy arrays and broadcasts. Latitudes and longitudes are
in degrees, distances in metres, times in seconds.

The ENU frame is a local tangent-plane (equirectangular) approximation on a
sphere. Over the <100 km spans this project uses, the error is well under the
200 m candidate-grid spacing.
"""

from __future__ import annotations

import numpy as np

from .constants import EARTH_RADIUS_M


def enu_from_latlon(
    lat: np.ndarray | float,
    lon: np.ndarray | float,
    lat0: float,
    lon0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project (lat, lon) to local east/north metres about (lat0, lon0)."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    dlon = np.deg2rad(lon - lon0)
    # Wrap the longitude difference into [-pi, pi).
    dlon = (dlon + np.pi) % (2 * np.pi) - np.pi
    dlat = np.deg2rad(lat - lat0)
    east = EARTH_RADIUS_M * np.cos(np.deg2rad(lat0)) * dlon
    north = EARTH_RADIUS_M * dlat
    return east, north


def latlon_from_enu(
    east: np.ndarray | float,
    north: np.ndarray | float,
    lat0: float,
    lon0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`enu_from_latlon`."""
    east = np.asarray(east, dtype=float)
    north = np.asarray(north, dtype=float)
    lat = lat0 + np.rad2deg(north / EARTH_RADIUS_M)
    lon = lon0 + np.rad2deg(east / (EARTH_RADIUS_M * np.cos(np.deg2rad(lat0))))
    lon = (lon + 180.0) % 360.0 - 180.0
    return lat, lon


def haversine_m(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray:
    """Great-circle ground distance in metres (broadcasts)."""
    phi1 = np.deg2rad(np.asarray(lat1, dtype=float))
    phi2 = np.deg2rad(np.asarray(lat2, dtype=float))
    dphi = phi2 - phi1
    dlam = np.deg2rad(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def slant_distance_m(
    obs_lat: np.ndarray | float,
    obs_lon: np.ndarray | float,
    obs_alt_m: np.ndarray | float,
    src_lat: np.ndarray | float,
    src_lon: np.ndarray | float,
    src_alt_m: np.ndarray | float,
) -> np.ndarray:
    """3-D straight-line distance between observer(s) and source(s).

    Altitudes are height above mean sea level for both the observer (ground
    elevation from a DEM) and the aircraft (ADS-B altitude), so the vertical
    leg is simply their difference.
    """
    ground = haversine_m(obs_lat, obs_lon, src_lat, src_lon)
    dz = np.asarray(src_alt_m, dtype=float) - np.asarray(obs_alt_m, dtype=float)
    return np.sqrt(ground**2 + dz**2)


def metres_per_degree(lat_deg: float) -> tuple[float, float]:
    """(metres per degree latitude, metres per degree longitude) at a latitude."""
    m_lat = np.deg2rad(1.0) * EARTH_RADIUS_M
    m_lon = m_lat * float(np.cos(np.deg2rad(lat_deg)))
    return float(m_lat), m_lon


def grid_cell(
    lat: np.ndarray | float,
    lon: np.ndarray | float,
    spacing_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map points to integer cells of a global grid with ~``spacing_m`` spacing.

    Rows are bands of constant latitude height; the column width inside a row
    is computed from the row's own latitude, so two flights that both touch
    the same place land in the same cell regardless of where they came from.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    m_lat = np.deg2rad(1.0) * EARTH_RADIUS_M
    dlat = spacing_m / m_lat
    row = np.floor(lat / dlat).astype(np.int64)
    row_lat = (row + 0.5) * dlat
    dlon = spacing_m / (m_lat * np.cos(np.deg2rad(row_lat)))
    col = np.floor(((lon + 180.0) % 360.0 - 180.0) / dlon).astype(np.int64)
    return row, col


def grid_cell_center(
    row: np.ndarray, col: np.ndarray, spacing_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Centre (lat, lon) of the cells produced by :func:`grid_cell`."""
    m_lat = np.deg2rad(1.0) * EARTH_RADIUS_M
    dlat = spacing_m / m_lat
    row_lat = (np.asarray(row, dtype=float) + 0.5) * dlat
    dlon = spacing_m / (m_lat * np.cos(np.deg2rad(row_lat)))
    lon = (np.asarray(col, dtype=float) + 0.5) * dlon
    lon = (lon + 180.0) % 360.0 - 180.0
    return row_lat, lon


def grid_points(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    spacing_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """All cell centres of the global grid inside a bounding box.

    Returns ``(row, col, lat, lon)`` flattened, so cell identity is shared
    with :func:`grid_cell`.
    """
    m_lat = np.deg2rad(1.0) * EARTH_RADIUS_M
    dlat = spacing_m / m_lat
    r0 = int(np.floor(lat_min / dlat))
    r1 = int(np.floor(lat_max / dlat))
    rows = []
    cols = []
    for r in range(r0, r1 + 1):
        row_lat = (r + 0.5) * dlat
        dlon = spacing_m / (m_lat * np.cos(np.deg2rad(row_lat)))
        c0 = int(np.floor(lon_min / dlon))
        c1 = int(np.floor(lon_max / dlon))
        c = np.arange(c0, c1 + 1, dtype=np.int64)
        rows.append(np.full(c.shape, r, dtype=np.int64))
        cols.append(c)
    if not rows:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0), np.zeros(0)
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    lat, lon = grid_cell_center(row, col, spacing_m)
    return row, col, lat, lon
