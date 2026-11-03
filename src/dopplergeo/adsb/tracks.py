"""Track preparation: segmenting, resampling, 3-D speed, clipping.

All functions take and return the canonical table from :mod:`schema` and
never mutate their input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..geo import haversine_m
from .schema import COLUMNS, STRING_COLUMNS, normalize_tracks

# Faster than any aircraft this pipeline cares about; a consecutive pair of
# reports implying more than this is treated as a bad fix or a reused icao24.
MAX_PLAUSIBLE_SPEED_MPS: float = 400.0


def _same_flight_as_prev(df: pd.DataFrame) -> np.ndarray:
    """Boolean per row: same ``flight_id`` as the row before (False on row 0).

    Done in numpy because comparing a pandas string column with its
    ``shift()`` yields ``<NA>`` on the first row.
    """
    fid = df["flight_id"].to_numpy(dtype=object)
    same = np.zeros(len(fid), dtype=bool)
    if len(fid) > 1:
        same[1:] = fid[1:] == fid[:-1]
    return same


def _step_distance_m(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """3-D distance and time step between each row and the previous row.

    Both are NaN on the first row of each flight. ``df`` must be sorted by
    ``(flight_id, time)``.
    """
    same = _same_flight_as_prev(df)
    lat = df["lat"].to_numpy(dtype=float)
    lon = df["lon"].to_numpy(dtype=float)
    alt = df["alt_m"].to_numpy(dtype=float)
    t = df["time"].to_numpy(dtype=float)
    d = np.full(len(df), np.nan)
    dt = np.full(len(df), np.nan)
    if len(df) > 1:
        ground = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
        dz = alt[1:] - alt[:-1]
        d[1:] = np.hypot(ground, dz)
        dt[1:] = t[1:] - t[:-1]
    d[~same] = np.nan
    dt[~same] = np.nan
    return d, dt


def split_on_gaps(
    tracks: pd.DataFrame,
    max_gap_s: float = 60.0,
    *,
    max_speed_mps: float = MAX_PLAUSIBLE_SPEED_MPS,
) -> pd.DataFrame:
    """Cut each flight into segments at time gaps and impossible jumps.

    A new segment starts when the time since the previous report exceeds
    ``max_gap_s`` or when the implied 3-D speed between consecutive reports
    exceeds ``max_speed_mps``. When a flight splits, segment ``k`` (0-based)
    of flight ``f`` gets ``flight_id = f"{f}_{k}"``; a flight that stays
    whole keeps its id unchanged. ``icao24`` is untouched either way.
    """
    df = normalize_tracks(tracks)
    if len(df) == 0:
        return df
    d, dt = _step_distance_m(df)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = d / dt
    first = np.isnan(dt)  # first row of each flight
    cut = first | (dt > max_gap_s) | (speed > max_speed_mps)
    by_flight = pd.Series(cut.astype(np.int64), index=df.index).groupby(df["flight_id"], sort=False)
    seg = by_flight.cumsum() - 1
    n_seg = by_flight.transform("sum")
    split = (n_seg > 1).to_numpy(dtype=bool)
    renamed = (df["flight_id"].astype("string") + "_" + seg.astype("string")).astype("string")
    df["flight_id"] = df["flight_id"].astype("string").where(~split, renamed)
    return normalize_tracks(df)


def _mean_of_available(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise mean of the finite inputs; NaN where neither is finite."""
    fa = np.isfinite(a)
    fb = np.isfinite(b)
    out = np.where(fa & fb, (a + b) / 2.0, np.where(fa, a, b))
    return np.where(fa | fb, out, np.nan)


def _interp_valid(tg: np.ndarray, t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``np.interp`` over the finite samples only; all-NaN if none exist."""
    ok = np.isfinite(v)
    if not ok.any():
        return np.full(tg.shape, np.nan)
    return np.interp(tg, t[ok], v[ok])


def _interp_angle(tg: np.ndarray, t: np.ndarray, deg: np.ndarray) -> np.ndarray:
    ok = np.isfinite(deg)
    if not ok.any():
        return np.full(tg.shape, np.nan)
    unwrapped = np.unwrap(deg[ok], period=360.0)
    return np.mod(np.interp(tg, t[ok], unwrapped), 360.0)


def resample_tracks(tracks: pd.DataFrame, step_s: float = 1.0) -> pd.DataFrame:
    """Linearly interpolate every flight onto a regular ``step_s`` grid.

    The grid of a flight starts at its first report and ends at the last
    grid point that is <= its last report. ``lat``, ``alt_m``, ``gs_mps`` and
    ``vrate_mps`` are interpolated linearly (NaN samples are skipped; ends
    are held constant); ``lon`` and ``track_deg`` are unwrapped first so
    crossings of the antimeridian / north are handled. String columns are
    copied from the flight's first row. Run :func:`split_on_gaps` first so no
    interpolation bridges a real gap.
    """
    if step_s <= 0:
        raise ValueError("step_s must be positive")
    df = normalize_tracks(tracks)
    if len(df) == 0:
        return df
    pieces: list[pd.DataFrame] = []
    for fid, g in df.groupby("flight_id", sort=False):
        t = g["time"].to_numpy(dtype=float)
        n = int(np.floor((t[-1] - t[0]) / step_s + 1e-9)) + 1
        tg = t[0] + step_s * np.arange(n, dtype=float)
        piece = {
            c: pd.Series([g[c].iloc[0]] * n, dtype="string") for c in STRING_COLUMNS
        }
        piece["flight_id"] = pd.Series([fid] * n, dtype="string")
        piece["time"] = tg
        piece["lat"] = np.interp(tg, t, g["lat"].to_numpy(dtype=float))
        lon_unwrapped = np.unwrap(g["lon"].to_numpy(dtype=float), period=360.0)
        lon = np.interp(tg, t, lon_unwrapped)
        piece["lon"] = (lon + 180.0) % 360.0 - 180.0
        piece["alt_m"] = np.interp(tg, t, g["alt_m"].to_numpy(dtype=float))
        piece["gs_mps"] = _interp_valid(tg, t, g["gs_mps"].to_numpy(dtype=float))
        piece["track_deg"] = _interp_angle(tg, t, g["track_deg"].to_numpy(dtype=float))
        piece["vrate_mps"] = _interp_valid(tg, t, g["vrate_mps"].to_numpy(dtype=float))
        pieces.append(pd.DataFrame(piece)[list(COLUMNS)])
    return normalize_tracks(pd.concat(pieces, ignore_index=True))


def add_speed_3d(tracks: pd.DataFrame) -> pd.DataFrame:
    """Add ``speed_mps = hypot(ground speed, vertical rate)``.

    Where ``gs_mps`` is NaN the ground speed is estimated from consecutive
    positions (:func:`geo.haversine_m`, central difference where both
    neighbours exist); where ``vrate_mps`` is NaN the climb rate is
    estimated the same way from ``alt_m``. A lone report with no reported
    speed stays NaN. The ``gs_mps``/``vrate_mps`` columns are not changed.
    """
    df = normalize_tracks(tracks)
    if len(df) == 0:
        out = df.copy()
        out["speed_mps"] = pd.Series(dtype="float64")
        return out
    same_prev = _same_flight_as_prev(df)
    same_next = np.append(same_prev[1:], False)
    lat = df["lat"].to_numpy(dtype=float)
    lon = df["lon"].to_numpy(dtype=float)
    alt = df["alt_m"].to_numpy(dtype=float)
    t = df["time"].to_numpy(dtype=float)
    n = len(df)

    ground_prev = np.full(n, np.nan)
    dz_prev = np.full(n, np.nan)
    dt_prev = np.full(n, np.nan)
    if n > 1:
        ground_prev[1:] = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
        dz_prev[1:] = alt[1:] - alt[:-1]
        dt_prev[1:] = t[1:] - t[:-1]
    ground_prev[~same_prev] = np.nan
    dz_prev[~same_prev] = np.nan
    dt_prev[~same_prev] = np.nan
    ground_next = np.append(ground_prev[1:], np.nan)
    dz_next = np.append(dz_prev[1:], np.nan)
    dt_next = np.append(dt_prev[1:], np.nan)
    ground_next[~same_next] = np.nan
    dz_next[~same_next] = np.nan
    dt_next[~same_next] = np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        gs_b = ground_prev / dt_prev
        gs_f = ground_next / dt_next
        vr_b = dz_prev / dt_prev
        vr_f = dz_next / dt_next
    gs_est = _mean_of_available(gs_b, gs_f)
    vr_est = _mean_of_available(vr_b, vr_f)

    gs = df["gs_mps"].to_numpy(dtype=float)
    vr = df["vrate_mps"].to_numpy(dtype=float)
    gs = np.where(np.isfinite(gs), gs, gs_est)
    vr = np.where(np.isfinite(vr), vr, vr_est)
    vr = np.where(np.isfinite(vr), vr, 0.0)  # no climb info at all: treat as level
    out = df.copy()
    out["speed_mps"] = np.hypot(gs, vr)
    return out


def clip_bbox(
    tracks: pd.DataFrame,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> pd.DataFrame:
    """Keep rows inside the box. Flight ids are unchanged, so a flight that
    leaves and re-enters keeps one id; run :func:`split_on_gaps` afterwards
    if that matters."""
    m = (
        (tracks["lat"] >= lat_min)
        & (tracks["lat"] <= lat_max)
        & (tracks["lon"] >= lon_min)
        & (tracks["lon"] <= lon_max)
    )
    return tracks.loc[m].reset_index(drop=True)


def clip_time(tracks: pd.DataFrame, t0: float, t1: float) -> pd.DataFrame:
    """Keep rows with ``t0 <= time <= t1`` (unix seconds)."""
    m = (tracks["time"] >= t0) & (tracks["time"] <= t1)
    return tracks.loc[m].reset_index(drop=True)


__all__ = [
    "MAX_PLAUSIBLE_SPEED_MPS",
    "add_speed_3d",
    "clip_bbox",
    "clip_time",
    "resample_tracks",
    "split_on_gaps",
]
