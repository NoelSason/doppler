"""Canonical ADS-B track table used by every stage after ingestion.

One row per position report. Columns (all required unless noted):

======  =========  ===================================================
column  dtype      meaning
======  =========  ===================================================
flight_id  str     unique per continuous flight segment (icao24 + segment index)
icao24     str     24-bit transponder address, lower-case hex
callsign   str     may be empty
typecode   str     ICAO aircraft type designator (e.g. "C172", "B738"); may be empty
time       float   unix seconds (UTC)
lat        float   degrees
lon        float   degrees
alt_m      float   altitude above mean sea level, metres (geometric if available,
                   else barometric; see ``alt_source``)
alt_source str     "geom" or "baro"
gs_mps     float   ground speed, m/s (NaN if unknown)
track_deg  float   ground track, degrees clockwise from north (NaN if unknown)
vrate_mps  float   vertical rate, m/s, positive up (NaN if unknown)
======  =========  ===================================================

Rows are sorted by ``(flight_id, time)``; times within a flight are strictly
increasing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..acoustics import Trajectory
from ..geo import enu_from_latlon

COLUMNS: tuple[str, ...] = (
    "flight_id",
    "icao24",
    "callsign",
    "typecode",
    "time",
    "lat",
    "lon",
    "alt_m",
    "alt_source",
    "gs_mps",
    "track_deg",
    "vrate_mps",
)

STRING_COLUMNS = ("flight_id", "icao24", "callsign", "typecode", "alt_source")
FLOAT_COLUMNS = ("time", "lat", "lon", "alt_m", "gs_mps", "track_deg", "vrate_mps")


def empty_tracks() -> pd.DataFrame:
    data: dict[str, object] = {c: pd.Series(dtype="string") for c in STRING_COLUMNS}
    data.update({c: pd.Series(dtype="float64") for c in FLOAT_COLUMNS})
    return pd.DataFrame(data)[list(COLUMNS)]


def normalize_tracks(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce dtypes, fill optional columns, sort, and validate."""
    out = df.copy()
    for c in STRING_COLUMNS:
        if c not in out:
            out[c] = ""
        out[c] = out[c].fillna("").astype("string")
    for c in FLOAT_COLUMNS:
        if c not in out:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    out = out[list(COLUMNS)]
    out = out.dropna(subset=["time", "lat", "lon", "alt_m"])
    out = out.sort_values(["flight_id", "time"], kind="stable")
    # Drop exact duplicate timestamps inside a flight; keep the first report.
    out = out[~out.duplicated(subset=["flight_id", "time"], keep="first")]
    return out.reset_index(drop=True)


def validate_tracks(df: pd.DataFrame) -> None:
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"track table missing columns: {missing}")
    if len(df) == 0:
        return
    bad_lat = (df["lat"].abs() > 90).any()
    bad_lon = (df["lon"].abs() > 180).any()
    if bad_lat or bad_lon:
        raise ValueError("track table has out-of-range coordinates")
    dt = df.groupby("flight_id", sort=False)["time"].diff().dropna()
    if (dt <= 0).any():
        raise ValueError("times within a flight must be strictly increasing")


def write_tracks(df: pd.DataFrame, path: str | Path) -> None:
    validate_tracks(df)
    df.to_parquet(Path(path), index=False)


def read_tracks(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(Path(path))
    return normalize_tracks(df)


def flight_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Per-flight bounding box and time span, indexed by ``flight_id``."""
    g = df.groupby("flight_id", sort=False)
    return pd.DataFrame(
        {
            "lat_min": g["lat"].min(),
            "lat_max": g["lat"].max(),
            "lon_min": g["lon"].min(),
            "lon_max": g["lon"].max(),
            "t_min": g["time"].min(),
            "t_max": g["time"].max(),
            "n": g.size(),
        }
    )


def trajectory_from_rows(
    rows: pd.DataFrame, lat0: float, lon0: float
) -> Trajectory:
    """Build a :class:`Trajectory` (ENU about ``lat0, lon0``, up = alt_m)
    from the rows of a single flight."""
    rows = rows.sort_values("time")
    east, north = enu_from_latlon(rows["lat"].to_numpy(), rows["lon"].to_numpy(), lat0, lon0)
    return Trajectory(
        t=rows["time"].to_numpy(dtype=float),
        east=east,
        north=north,
        up=rows["alt_m"].to_numpy(dtype=float),
    )
