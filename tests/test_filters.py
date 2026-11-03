"""pair_by_offset and silence_filter on a built scenario.

Flight A flies east past an observer; flight B flies north past the same
observer so that B's CPA is *heard* 920 s after A's. The observer is placed
exactly on a grid-cell centre so the timings are exact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dopplergeo.acoustics import closest_point_of_approach
from dopplergeo.adsb.schema import trajectory_from_rows
from dopplergeo.candidates import EventSpec, generate_candidates
from dopplergeo.dem import FlatElevation
from dopplergeo.filters import PAIR_COLUMNS, pair_by_offset, silence_filter
from dopplergeo.geo import enu_from_latlon, grid_cell, grid_cell_center, latlon_from_enu
from dopplergeo.synth import straight_track

C = 343.0
GRID_M = 200.0
ALT = 1000.0
SPEED = 60.0
LEG = 300.0  # seconds from a leg's start to its CPA
SPEC = EventSpec(speed_range=(40.0, 80.0), cpa_slant_range=(500.0, 5000.0))


def observer_cell() -> tuple[int, int, float, float]:
    lat, lon = latlon_from_enu(18_000.0, 1_500.0, 37.0, -122.0)
    row, col = grid_cell(lat, lon, GRID_M)
    clat, clon = grid_cell_center(row, col, GRID_M)
    return int(row), int(col), float(clat), float(clon)


ROW, COL, OLAT, OLON = observer_cell()


def heard_cpa(rows: pd.DataFrame) -> tuple[float, float, float]:
    """``(t_cpa_tx, slant, t_cpa_rx)`` of the observer cell for one flight."""
    lat0, lon0 = float(rows["lat"].mean()), float(rows["lon"].mean())
    traj = trajectory_from_rows(rows, lat0, lon0)
    e, n = enu_from_latlon(OLAT, OLON, lat0, lon0)
    t_tx, d = closest_point_of_approach(traj, np.array([float(e), float(n), 0.0]), oversample_s=0.02)
    return t_tx, d, t_tx + d / C


def passing_flight(fid: str, heading_deg: float, offset_m: float, t_rx_wanted: float) -> pd.DataFrame:
    """A straight flight at SPEED whose CPA to the observer is ``offset_m``
    (horizontal) to the right of its path and is heard at ``t_rx_wanted``.
    The CPA happens LEG seconds after the first sample."""
    h = np.deg2rad(heading_deg)
    # Unit vectors along the heading and to its left.
    along = np.array([np.sin(h), np.cos(h)])
    left = np.array([-np.cos(h), np.sin(h)])
    cpa_point = -offset_m * left  # observer is `offset_m` to the right of the path
    start = cpa_point - along * SPEED * LEG
    lat_s, lon_s = latlon_from_enu(float(start[0]), float(start[1]), OLAT, OLON)
    slant = float(np.hypot(offset_m, ALT))
    t0 = t_rx_wanted - slant / C - LEG
    return straight_track(fid, t0, 2 * LEG, float(lat_s), float(lon_s), ALT, SPEED, heading_deg)


@pytest.fixture(scope="module")
def scenario() -> dict[str, object]:
    a = straight_track("A", 1000.0, 600.0, 37.0, -122.0, ALT, SPEED, 90.0)
    _, _, t_rx_a = heard_cpa(a)
    b = passing_flight("B", 0.0, 1_500.0, t_rx_a + 920.0)
    _, _, t_rx_b = heard_cpa(b)
    assert abs((t_rx_b - t_rx_a) - 920.0) < 1.0
    cands = generate_candidates(pd.concat([a, b], ignore_index=True), SPEC, FlatElevation(0.0), grid_m=GRID_M, c=C)
    return {"a": a, "b": b, "t_rx_a": t_rx_a, "t_rx_b": t_rx_b, "cands": cands}


def cell_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    return pairs[(pairs["row"] == ROW) & (pairs["col"] == COL)]


def test_pair_by_offset_finds_920_and_rejects_950(scenario: dict[str, object]) -> None:
    cands = scenario["cands"]
    assert isinstance(cands, pd.DataFrame)
    pairs = pair_by_offset(cands, cands, 920.0, 10.0)
    assert tuple(pairs.columns) == PAIR_COLUMNS
    hit = cell_pairs(pairs)
    assert len(hit) == 1
    h = hit.iloc[0]
    assert h["flight_id_1"] == "A" and h["flight_id_2"] == "B"
    assert abs(h["offset_s"] - 920.0) < 10.0
    assert h["offset_s"] == pytest.approx(h["t_cpa_rx_2"] - h["t_cpa_rx_1"])
    assert h["t_cpa_rx_1"] == pytest.approx(scenario["t_rx_a"], abs=0.5)
    assert h["t_cpa_rx_2"] == pytest.approx(scenario["t_rx_b"], abs=0.5)
    assert h["lat"] == pytest.approx(OLAT) and h["lon"] == pytest.approx(OLON)
    # Never pairs a flight with itself, and the offset sign matters.
    assert (pairs["flight_id_1"] != pairs["flight_id_2"]).all()
    assert ((pairs["t_cpa_rx_2"] - pairs["t_cpa_rx_1"]) > 0).all()
    assert len(cell_pairs(pair_by_offset(cands, cands, 950.0, 10.0))) == 0
    assert len(cell_pairs(pair_by_offset(cands, cands, -920.0, 10.0))) == 1  # B then A


def test_pair_by_offset_empty_inputs(scenario: dict[str, object]) -> None:
    cands = scenario["cands"]
    assert isinstance(cands, pd.DataFrame)
    empty = cands.iloc[:0]
    out = pair_by_offset(empty, cands, 920.0, 10.0)
    assert tuple(out.columns) == PAIR_COLUMNS and len(out) == 0
    assert len(pair_by_offset(cands, cands, 5000.0, 1.0)) == 0


def _pairs(scenario: dict[str, object]) -> pd.DataFrame:
    cands = scenario["cands"]
    assert isinstance(cands, pd.DataFrame)
    pairs = cell_pairs(pair_by_offset(cands, cands, 920.0, 10.0)).reset_index(drop=True)
    assert len(pairs) == 1
    return pairs


def _tracks(scenario: dict[str, object], *extra: pd.DataFrame) -> pd.DataFrame:
    a, b = scenario["a"], scenario["b"]
    assert isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame)
    return pd.concat([a, b, *extra], ignore_index=True)


def test_silence_filter_drops_intruder_two_km_away_mid_interval(scenario: dict[str, object]) -> None:
    t_rx_a = scenario["t_rx_a"]
    assert isinstance(t_rx_a, float)
    c3 = passing_flight("C", 90.0, 2_000.0, t_rx_a + 460.0)
    kept = silence_filter(_pairs(scenario), _tracks(scenario, c3), FlatElevation(0.0), radius_m=4000.0, margin_s=30.0, c=C)
    assert len(kept) == 0
    # Too small a radius and the same intruder is inaudible.
    kept2 = silence_filter(_pairs(scenario), _tracks(scenario, c3), FlatElevation(0.0), radius_m=1500.0, margin_s=30.0, c=C)
    assert len(kept2) == 1


def test_silence_filter_keeps_intruder_eight_km_away(scenario: dict[str, object]) -> None:
    t_rx_a = scenario["t_rx_a"]
    assert isinstance(t_rx_a, float)
    c8 = passing_flight("C", 90.0, 8_000.0, t_rx_a + 460.0)
    kept = silence_filter(_pairs(scenario), _tracks(scenario, c8), FlatElevation(0.0), radius_m=4000.0, margin_s=30.0, c=C)
    assert len(kept) == 1
    assert tuple(kept.columns) == PAIR_COLUMNS


def test_silence_filter_keeps_intruder_outside_interval_or_within_margin(scenario: dict[str, object]) -> None:
    t_rx_a, t_rx_b = scenario["t_rx_a"], scenario["t_rx_b"]
    assert isinstance(t_rx_a, float) and isinstance(t_rx_b, float)
    before = passing_flight("C", 90.0, 2_000.0, t_rx_a - 200.0)
    after = passing_flight("D", 90.0, 2_000.0, t_rx_b + 200.0)
    in_margin_1 = passing_flight("E", 90.0, 2_000.0, t_rx_a + 10.0)
    in_margin_2 = passing_flight("F", 90.0, 2_000.0, t_rx_b - 10.0)
    tracks = _tracks(scenario, before, after, in_margin_1, in_margin_2)
    kept = silence_filter(_pairs(scenario), tracks, FlatElevation(0.0), radius_m=4000.0, margin_s=30.0, c=C)
    assert len(kept) == 1
    # Shrink the margin and the two near-margin flights become intruders.
    kept2 = silence_filter(_pairs(scenario), tracks, FlatElevation(0.0), radius_m=4000.0, margin_s=5.0, c=C)
    assert len(kept2) == 0


def test_silence_filter_ignores_the_pair_flights_and_empty_input(scenario: dict[str, object]) -> None:
    pairs = _pairs(scenario)
    kept = silence_filter(pairs, _tracks(scenario), FlatElevation(0.0), c=C)
    assert len(kept) == 1
    empty = silence_filter(pairs.iloc[:0], _tracks(scenario), FlatElevation(0.0), c=C)
    assert len(empty) == 0 and tuple(empty.columns) == PAIR_COLUMNS
