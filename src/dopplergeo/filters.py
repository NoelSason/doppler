"""Combine and prune candidate tables.

* :func:`pair_by_offset` joins two candidate tables on grid cell and keeps
  the pairs whose heard-CPA times differ by a measured offset.
* :func:`silence_filter` drops pairs where a third aircraft would have been
  audible between the two events, which the recording says did not happen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .candidates import cpas_for_points
from .constants import SPEED_OF_SOUND_MPS
from .dem import ElevationProvider

PAIR_COLUMNS: tuple[str, ...] = (
    "row",
    "col",
    "lat",
    "lon",
    "elev_m",
    "flight_id_1",
    "t_cpa_tx_1",
    "t_cpa_rx_1",
    "cpa_slant_1",
    "speed_1",
    "flight_id_2",
    "t_cpa_tx_2",
    "t_cpa_rx_2",
    "cpa_slant_2",
    "speed_2",
    "offset_s",
)

_RENAME = {
    "flight_id": "flight_id_{k}",
    "t_cpa_tx": "t_cpa_tx_{k}",
    "t_cpa_rx": "t_cpa_rx_{k}",
    "cpa_slant_m": "cpa_slant_{k}",
    "speed_mps": "speed_{k}",
}


def _side(cand: pd.DataFrame, k: int, keep_location: bool) -> pd.DataFrame:
    cols = ["row", "col"] + (["lat", "lon", "elev_m"] if keep_location else []) + list(_RENAME)
    out = cand[cols].rename(columns={a: b.format(k=k) for a, b in _RENAME.items()})
    return out


def _empty_pairs() -> pd.DataFrame:
    dtypes = {"row": "int64", "col": "int64", "flight_id_1": "string", "flight_id_2": "string"}
    return pd.DataFrame({c: pd.Series(dtype=dtypes.get(c, "float64")) for c in PAIR_COLUMNS})


def pair_by_offset(
    cand1: pd.DataFrame, cand2: pd.DataFrame, offset_s: float, tol_s: float
) -> pd.DataFrame:
    """Cells that hear event 1 and then event 2 ``offset_s +/- tol_s`` later.

    Joins on ``(row, col)``; the location columns come from ``cand1``.
    ``offset_s`` in the output is the actual ``t_cpa_rx_2 - t_cpa_rx_1``.
    """
    if len(cand1) == 0 or len(cand2) == 0:
        return _empty_pairs()
    merged = _side(cand1, 1, True).merge(_side(cand2, 2, False), on=["row", "col"], how="inner")
    if len(merged) == 0:
        return _empty_pairs()
    diff = merged["t_cpa_rx_2"].to_numpy(dtype=float) - merged["t_cpa_rx_1"].to_numpy(dtype=float)
    keep = (np.abs(diff - offset_s) <= tol_s) & (
        merged["flight_id_1"].to_numpy(dtype=object) != merged["flight_id_2"].to_numpy(dtype=object)
    )
    out = merged.loc[keep].copy()
    out["offset_s"] = diff[keep]
    out = out[list(PAIR_COLUMNS)].reset_index(drop=True)
    return out


def silence_filter(
    pairs: pd.DataFrame,
    all_tracks: pd.DataFrame,
    dem: ElevationProvider,
    *,
    radius_m: float = 4000.0,
    margin_s: float = 30.0,
    c: float = SPEED_OF_SOUND_MPS,
) -> pd.DataFrame:
    """Drop pairs with an intruding third aircraft.

    A pair is dropped when any flight other than its two has a CPA to the
    cell within ``radius_m`` whose heard time falls in the open interval
    ``(t_cpa_rx_1 + margin_s, t_cpa_rx_2 - margin_s)``.
    """
    if len(pairs) == 0:
        return pairs.copy()
    cells = pairs[["row", "col", "lat", "lon", "elev_m"]].drop_duplicates(["row", "col"])
    cpas = cpas_for_points(all_tracks, cells, dem, max_slant_m=radius_m, c=c)
    if len(cpas) == 0:
        return pairs.reset_index(drop=True)

    work = pairs.reset_index(drop=True)
    work["_pair"] = np.arange(len(work))
    joined = work[
        ["_pair", "row", "col", "flight_id_1", "flight_id_2", "t_cpa_rx_1", "t_cpa_rx_2"]
    ].merge(cpas[["row", "col", "flight_id", "t_cpa_rx"]], on=["row", "col"], how="inner")
    if len(joined) == 0:
        return work.drop(columns="_pair")

    fid = joined["flight_id"].to_numpy(dtype=object)
    other = (fid != joined["flight_id_1"].to_numpy(dtype=object)) & (
        fid != joined["flight_id_2"].to_numpy(dtype=object)
    )
    t = joined["t_cpa_rx"].to_numpy(dtype=float)
    lo = joined["t_cpa_rx_1"].to_numpy(dtype=float) + margin_s
    hi = joined["t_cpa_rx_2"].to_numpy(dtype=float) - margin_s
    intruding = other & (t > lo) & (t < hi)
    bad = np.unique(joined["_pair"].to_numpy()[intruding])
    keep = ~np.isin(work["_pair"].to_numpy(), bad)
    return work.loc[keep].drop(columns="_pair").reset_index(drop=True)
