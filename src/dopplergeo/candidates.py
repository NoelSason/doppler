"""Candidate observer locations from flight tracks.

Given a measured event (an aircraft speed range and a CPA slant-distance
range from a Doppler curve fit), :func:`generate_candidates` lays the global
grid over every flight and keeps the cells whose closest point of approach
to that flight matches. :func:`cpas_for_points` is the inverse question:
for a fixed set of cells, which flights pass within a given range, and when.

Both work in a local ENU frame about each flight's own centroid with the
cell elevation as the "up" coordinate, so the slant distance is the true
3-D distance (see ``docs/DESIGN.md``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .acoustics import Trajectory
from .adsb.schema import flight_bounds, trajectory_from_rows
from .constants import SPEED_OF_SOUND_MPS
from .dem import ElevationProvider
from .geo import enu_from_latlon, grid_points, latlon_from_enu, metres_per_degree

CANDIDATE_COLUMNS: tuple[str, ...] = (
    "row",
    "col",
    "lat",
    "lon",
    "elev_m",
    "flight_id",
    "t_cpa_tx",
    "t_cpa_rx",
    "cpa_slant_m",
    "speed_mps",
)
CPA_COLUMNS: tuple[str, ...] = (
    "row",
    "col",
    "flight_id",
    "t_cpa_tx",
    "t_cpa_rx",
    "cpa_slant_m",
    "speed_mps",
)

SAMPLE_STEP_S: float = 1.0
MIN_INTERVAL_S: float = 10.0
GRID_CHUNK: int = 4096


@dataclass(frozen=True)
class EventSpec:
    """What a Doppler-curve fit says about the aircraft that caused an event."""

    speed_range: tuple[float, float]
    cpa_slant_range: tuple[float, float]

    def __post_init__(self) -> None:
        lo, hi = self.speed_range
        if not lo <= hi:
            raise ValueError("speed_range must be (low, high)")
        lo, hi = self.cpa_slant_range
        if not lo <= hi:
            raise ValueError("cpa_slant_range must be (low, high)")


# --------------------------------------------------------------------------
# Core geometry
# --------------------------------------------------------------------------


def _sample_times(traj: Trajectory, step_s: float = SAMPLE_STEP_S) -> np.ndarray:
    """Uniform samples over the trajectory span with spacing <= ``step_s``."""
    span = traj.t_end - traj.t_start
    n = int(np.ceil(span / step_s)) + 1
    return np.linspace(traj.t_start, traj.t_end, max(n, 2))


def _closest_samples(
    grid_enu: np.ndarray, track_enu: np.ndarray, chunk: int = GRID_CHUNK
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For every grid point, the index of the nearest track sample and the
    squared distances at that sample and its two neighbours (clamped).

    Works in blocks of ``chunk`` grid points so the ``N_grid x N_track``
    distance matrix never exists in full.
    """
    m = grid_enu.shape[0]
    k_track = track_enu.shape[0]
    k_best = np.empty(m, dtype=np.int64)
    d2_prev = np.empty(m)
    d2_best = np.empty(m)
    d2_next = np.empty(m)
    pp = np.einsum("ij,ij->i", track_enu, track_enu)
    for start in range(0, m, chunk):
        g = grid_enu[start : start + chunk]
        gg = np.einsum("ij,ij->i", g, g)
        d2 = gg[:, None] + pp[None, :] - 2.0 * (g @ track_enu.T)
        np.maximum(d2, 0.0, out=d2)
        k = np.argmin(d2, axis=1)
        idx = np.arange(g.shape[0])
        k_best[start : start + chunk] = k
        d2_best[start : start + chunk] = d2[idx, k]
        d2_prev[start : start + chunk] = d2[idx, np.maximum(k - 1, 0)]
        d2_next[start : start + chunk] = d2[idx, np.minimum(k + 1, k_track - 1)]
    return k_best, d2_prev, d2_best, d2_next


def _refine_cpa(
    ts: np.ndarray,
    k: np.ndarray,
    d2_prev: np.ndarray,
    d2_best: np.ndarray,
    d2_next: np.ndarray,
) -> np.ndarray:
    """Sub-sample CPA time from a parabola through the three squared
    distances around the argmin. For straight constant-speed flight
    ``d^2(t)`` is exactly quadratic, so this is exact there and a good
    approximation otherwise. Endpoint argmins are returned unrefined."""
    dt = ts[1] - ts[0] if ts.size > 1 else 0.0
    interior = (k > 0) & (k < ts.size - 1)
    denom = d2_prev - 2.0 * d2_best + d2_next
    with np.errstate(divide="ignore", invalid="ignore"):
        offset = 0.5 * (d2_prev - d2_next) / denom
    offset = np.where(interior & (denom > 0) & np.isfinite(offset), offset, 0.0)
    offset = np.clip(offset, -1.0, 1.0)
    return ts[k] + offset * dt


def _cpa_to_grid(
    traj: Trajectory,
    ts: np.ndarray,
    track_enu: np.ndarray,
    grid_enu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(k, t_cpa_tx, slant_m, speed_mps)`` for every grid point, with the
    refined time re-evaluated on the spline."""
    k, d2p, d2b, d2n = _closest_samples(grid_enu, track_enu)
    t_cpa = _refine_cpa(ts, k, d2p, d2b, d2n)
    p = traj.position(t_cpa)
    slant = np.linalg.norm(p - grid_enu, axis=-1)
    speed = traj.speed(t_cpa)
    return k, t_cpa, slant, speed


def _in_range_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of True as inclusive ``(i0, i1)`` index pairs."""
    if mask.size == 0 or not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _grid_over_track(
    track_enu: np.ndarray,
    lat0: float,
    lon0: float,
    pad_m: float,
    grid_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Global grid cells covering the track's horizontal bbox plus ``pad_m``."""
    e_min, n_min = track_enu[:, 0].min() - pad_m, track_enu[:, 1].min() - pad_m
    e_max, n_max = track_enu[:, 0].max() + pad_m, track_enu[:, 1].max() + pad_m
    lat_min, lon_min = latlon_from_enu(e_min, n_min, lat0, lon0)
    lat_max, lon_max = latlon_from_enu(e_max, n_max, lat0, lon0)
    return grid_points(float(lat_min), float(lat_max), float(lon_min), float(lon_max), grid_m)


def _empty_frame(columns: tuple[str, ...]) -> pd.DataFrame:
    dtypes = {
        "row": "int64",
        "col": "int64",
        "flight_id": "string",
    }
    return pd.DataFrame({c: pd.Series(dtype=dtypes.get(c, "float64")) for c in columns})


def _flight_groups(tracks: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [(str(fid), rows) for fid, rows in tracks.groupby("flight_id", sort=False) if len(rows) >= 2]


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def generate_candidates(
    tracks: pd.DataFrame,
    spec: EventSpec,
    dem: ElevationProvider,
    *,
    grid_m: float = 200.0,
    c: float = SPEED_OF_SOUND_MPS,
    speed_margin_mps: float = 0.0,
) -> pd.DataFrame:
    """Grid cells whose CPA to some flight matches ``spec``.

    One row per (cell, flight, in-range interval). The CPA must be interior
    to the interval (an argmin at either end is a truncated approach, not a
    closest approach) and its slant distance inside ``spec.cpa_slant_range``.
    Speed is the 3-D spline speed; the interval filter uses
    ``speed_range`` widened by ``speed_margin_mps`` on both sides.
    """
    speed_lo, speed_hi = spec.speed_range
    slant_lo, slant_hi = spec.cpa_slant_range
    speed_lo -= speed_margin_mps
    speed_hi += speed_margin_mps
    frames: list[pd.DataFrame] = []

    for fid, rows in _flight_groups(tracks):
        lat0 = float(rows["lat"].mean())
        lon0 = float(rows["lon"].mean())
        traj = trajectory_from_rows(rows, lat0, lon0)
        ts_all = _sample_times(traj)
        pos_all = traj.position(ts_all)
        spd_all = traj.speed(ts_all)
        in_range = (spd_all >= speed_lo) & (spd_all <= speed_hi)

        for i0, i1 in _in_range_intervals(in_range):
            ts = ts_all[i0 : i1 + 1]
            if ts[-1] - ts[0] < MIN_INTERVAL_S:
                continue
            track_enu = pos_all[i0 : i1 + 1]
            row, col, glat, glon = _grid_over_track(track_enu, lat0, lon0, slant_hi, grid_m)
            if row.size == 0:
                continue
            elev = np.asarray(dem.elevation_m(glat, glon), dtype=float)
            ge, gn = enu_from_latlon(glat, glon, lat0, lon0)
            grid_enu = np.column_stack([ge, gn, elev])

            k, t_cpa, slant, speed = _cpa_to_grid(traj, ts, track_enu, grid_enu)
            keep = (k > 0) & (k < ts.size - 1) & (slant >= slant_lo) & (slant <= slant_hi)
            if not keep.any():
                continue
            frames.append(
                pd.DataFrame(
                    {
                        "row": row[keep],
                        "col": col[keep],
                        "lat": glat[keep],
                        "lon": glon[keep],
                        "elev_m": elev[keep],
                        "flight_id": pd.Series(fid, index=np.arange(int(keep.sum())), dtype="string"),
                        "t_cpa_tx": t_cpa[keep],
                        "t_cpa_rx": t_cpa[keep] + slant[keep] / c,
                        "cpa_slant_m": slant[keep],
                        "speed_mps": speed[keep],
                    }
                )
            )

    if not frames:
        return _empty_frame(CANDIDATE_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    return out[list(CANDIDATE_COLUMNS)]


def cpas_for_points(
    tracks: pd.DataFrame,
    points: pd.DataFrame,
    dem: ElevationProvider,
    *,
    max_slant_m: float = 10_000.0,
    c: float = SPEED_OF_SOUND_MPS,
) -> pd.DataFrame:
    """Every (point, flight) closest approach within ``max_slant_m``.

    ``points`` needs ``row, col, lat, lon``; ``elev_m`` is filled from the
    DEM when missing. Unlike :func:`generate_candidates`, a CPA at the first
    or last sample of a track counts: for the silence filter a passing
    aircraft matters even when its track is cut short.
    """
    required = ("row", "col", "lat", "lon")
    missing = [col for col in required if col not in points.columns]
    if missing:
        raise ValueError(f"points is missing columns: {missing}")
    if len(points) == 0 or len(tracks) == 0:
        return _empty_frame(CPA_COLUMNS)

    pts = points.reset_index(drop=True)
    p_lat = pts["lat"].to_numpy(dtype=float)
    p_lon = pts["lon"].to_numpy(dtype=float)
    if "elev_m" in pts.columns:
        p_elev = pts["elev_m"].to_numpy(dtype=float)
        bad = ~np.isfinite(p_elev)
        if bad.any():
            p_elev = p_elev.copy()
            p_elev[bad] = np.asarray(dem.elevation_m(p_lat[bad], p_lon[bad]), dtype=float)
    else:
        p_elev = np.asarray(dem.elevation_m(p_lat, p_lon), dtype=float)
    p_row = pts["row"].to_numpy(dtype=np.int64)
    p_col = pts["col"].to_numpy(dtype=np.int64)

    bounds = flight_bounds(tracks)
    frames: list[pd.DataFrame] = []
    for fid, rows in _flight_groups(tracks):
        b = bounds.loc[fid]
        m_lat, m_lon = metres_per_degree(float(0.5 * (b["lat_min"] + b["lat_max"])))
        d_lat = max_slant_m / m_lat
        d_lon = max_slant_m / m_lon
        sel = (
            (p_lat >= b["lat_min"] - d_lat)
            & (p_lat <= b["lat_max"] + d_lat)
            & (p_lon >= b["lon_min"] - d_lon)
            & (p_lon <= b["lon_max"] + d_lon)
        )
        if not sel.any():
            continue
        idx = np.flatnonzero(sel)
        lat0 = float(rows["lat"].mean())
        lon0 = float(rows["lon"].mean())
        traj = trajectory_from_rows(rows, lat0, lon0)
        ts = _sample_times(traj)
        track_enu = traj.position(ts)
        ge, gn = enu_from_latlon(p_lat[idx], p_lon[idx], lat0, lon0)
        grid_enu = np.column_stack([ge, gn, p_elev[idx]])
        _, t_cpa, slant, speed = _cpa_to_grid(traj, ts, track_enu, grid_enu)
        keep = slant <= max_slant_m
        if not keep.any():
            continue
        sub = idx[keep]
        frames.append(
            pd.DataFrame(
                {
                    "row": p_row[sub],
                    "col": p_col[sub],
                    "flight_id": pd.Series(fid, index=np.arange(sub.size), dtype="string"),
                    "t_cpa_tx": t_cpa[keep],
                    "t_cpa_rx": t_cpa[keep] + slant[keep] / c,
                    "cpa_slant_m": slant[keep],
                    "speed_mps": speed[keep],
                }
            )
        )

    if not frames:
        return _empty_frame(CPA_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    return out[list(CPA_COLUMNS)]
