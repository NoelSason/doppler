"""generate_candidates / cpas_for_points against synthetic straight flights.

The truth for a cell is its *own* CPA (cell centre vs. the spline), computed
with :func:`acoustics.closest_point_of_approach` in the same frame the
candidate generator uses (the flight's centroid).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from dopplergeo.acoustics import closest_point_of_approach
from dopplergeo.adsb.schema import normalize_tracks, trajectory_from_rows
from dopplergeo.candidates import (
    CANDIDATE_COLUMNS,
    CPA_COLUMNS,
    EventSpec,
    cpas_for_points,
    generate_candidates,
)
from dopplergeo.dem import ArrayElevation, FlatElevation
from dopplergeo.geo import enu_from_latlon, grid_cell, grid_cell_center, latlon_from_enu, metres_per_degree
from dopplergeo.synth import straight_track

GRID_M = 200.0
C = 343.0
LAT_S, LON_S = 37.0, -122.0


def flight_a() -> pd.DataFrame:
    """East at 60 m/s, 1000 m MSL, from (37, -122) at t=1000 for 600 s (36 km)."""
    return straight_track("A", 1000.0, 600.0, LAT_S, LON_S, 1000.0, 60.0, 90.0)


def cell_at(east_m: float, north_m: float) -> tuple[int, int, float, float]:
    """Grid cell containing the point ``east_m, north_m`` from the flight A origin."""
    lat, lon = latlon_from_enu(east_m, north_m, LAT_S, LON_S)
    row, col = grid_cell(lat, lon, GRID_M)
    clat, clon = grid_cell_center(row, col, GRID_M)
    return int(row), int(col), float(clat), float(clon)


def truth(rows: pd.DataFrame, clat: float, clon: float, elev_m: float) -> tuple[float, float]:
    lat0, lon0 = float(rows["lat"].mean()), float(rows["lon"].mean())
    traj = trajectory_from_rows(rows, lat0, lon0)
    e, n = enu_from_latlon(clat, clon, lat0, lon0)
    return closest_point_of_approach(traj, np.array([float(e), float(n), elev_m]), oversample_s=0.02)


def rows_for(cands: pd.DataFrame, row: int, col: int, fid: str) -> pd.DataFrame:
    return cands[(cands["row"] == row) & (cands["col"] == col) & (cands["flight_id"] == fid)]


SPEC = EventSpec(speed_range=(40.0, 80.0), cpa_slant_range=(500.0, 5000.0))


def test_columns_and_empty_input() -> None:
    out = generate_candidates(normalize_tracks(flight_a().iloc[:0]), SPEC, FlatElevation(0.0))
    assert tuple(out.columns) == CANDIDATE_COLUMNS
    assert len(out) == 0
    pts = pd.DataFrame({"row": [], "col": [], "lat": [], "lon": []})
    out2 = cpas_for_points(flight_a(), pts, FlatElevation(0.0))
    assert tuple(out2.columns) == CPA_COLUMNS and len(out2) == 0


def test_true_cpa_cell_is_found_with_subsecond_time() -> None:
    tr = flight_a()
    row, col, clat, clon = cell_at(18_000.0, 1_500.0)
    t_true, d_true = truth(tr, clat, clon, 0.0)
    cands = generate_candidates(tr, SPEC, FlatElevation(0.0), grid_m=GRID_M, c=C)
    assert tuple(cands.columns) == CANDIDATE_COLUMNS
    hit = rows_for(cands, row, col, "A")
    assert len(hit) == 1, "exactly one row per (cell, flight, interval)"
    h = hit.iloc[0]
    assert abs(h["t_cpa_tx"] - t_true) < 0.5
    assert abs(h["cpa_slant_m"] - d_true) < 20.0
    assert h["t_cpa_rx"] == pytest.approx(h["t_cpa_tx"] + h["cpa_slant_m"] / C)
    assert h["speed_mps"] == pytest.approx(60.0, abs=0.5)
    assert h["elev_m"] == 0.0
    assert h["lat"] == pytest.approx(clat) and h["lon"] == pytest.approx(clon)
    # Refinement does better than the 1 s sample grid: the truth here is
    # oversampled at 0.02 s so this bound is meaningful.
    assert abs(h["t_cpa_tx"] - t_true) < 0.05
    # Everything returned respects the spec.
    assert cands["cpa_slant_m"].between(500.0, 5000.0).all()
    assert cands["speed_mps"].between(40.0 - 1e-6, 80.0 + 1e-6).all()
    assert cands["flight_id"].dtype == "string"
    assert not cands.duplicated(["row", "col", "flight_id"]).any()


def test_endpoint_cpas_are_excluded() -> None:
    tr = flight_a()
    cands = generate_candidates(tr, SPEC, FlatElevation(0.0))
    # Abeam before the start and after the end: argmin lands on an endpoint.
    for east in (-3_000.0, 39_000.0):
        row, col, _, _ = cell_at(east, 1_500.0)
        assert len(rows_for(cands, row, col, "A")) == 0
    # Sanity: the interior neighbour is present.
    row, col, _, _ = cell_at(3_000.0, 1_500.0)
    assert len(rows_for(cands, row, col, "A")) == 1


def test_speed_filter_excludes_out_of_range_flight() -> None:
    fast = straight_track("B", 1000.0, 600.0, LAT_S, LON_S, 1000.0, 120.0, 90.0)
    tracks = pd.concat([flight_a(), fast], ignore_index=True)
    cands = generate_candidates(tracks, SPEC, FlatElevation(0.0))
    assert (cands["flight_id"] == "A").any()
    assert not (cands["flight_id"] == "B").any()
    wide = generate_candidates(tracks, SPEC, FlatElevation(0.0), speed_margin_mps=50.0)
    assert (wide["flight_id"] == "B").any()
    assert wide.loc[wide["flight_id"] == "B", "speed_mps"].between(119.0, 121.0).all()


def test_slant_filter() -> None:
    tr = flight_a()
    row, col, clat, clon = cell_at(18_000.0, 1_500.0)
    _, d_true = truth(tr, clat, clon, 0.0)  # ~1.8 km
    narrow = generate_candidates(tr, EventSpec((40.0, 80.0), (500.0, d_true - 100.0)), FlatElevation(0.0))
    assert len(rows_for(narrow, row, col, "A")) == 0
    around = generate_candidates(tr, EventSpec((40.0, 80.0), (d_true - 100.0, d_true + 100.0)), FlatElevation(0.0))
    assert len(rows_for(around, row, col, "A")) == 1


def test_speed_intervals_split_an_accelerating_flight() -> None:
    # 50 -> 130 m/s over 600 s; in [40, 80] only for the first ~225 s.
    accel = 80.0 / 600.0
    tr = straight_track("ACC", 1000.0, 600.0, LAT_S, LON_S, 1000.0, 50.0, 90.0, accel_mps2=accel)
    cands = generate_candidates(tr, SPEC, FlatElevation(0.0))

    def east_at(t_rel: float) -> float:
        return 50.0 * t_rel + 0.5 * accel * t_rel**2

    row, col, _, _ = cell_at(east_at(100.0), 1_500.0)
    hit = rows_for(cands, row, col, "ACC")
    assert len(hit) == 1
    assert hit.iloc[0]["speed_mps"] == pytest.approx(50.0 + 100.0 * accel, abs=1.0)
    assert 1090.0 < hit.iloc[0]["t_cpa_tx"] < 1110.0
    row, col, _, _ = cell_at(east_at(400.0), 1_500.0)
    assert len(rows_for(cands, row, col, "ACC")) == 0
    assert cands["speed_mps"].max() <= 80.0 + 1e-6


def test_sloped_terrain_enters_slant_distance() -> None:
    tr = flight_a()
    row, col, clat, clon = cell_at(18_000.0, 1_500.0)
    # Terrain ramps 10% with latitude and is ~500 m at the observer's latitude.
    m_lat, _ = metres_per_degree(LAT_S)
    lat_axis = np.linspace(36.9, 37.1, 41)
    lon_axis = np.linspace(-122.3, -121.5, 9)
    grid = 500.0 + 0.1 * (lat_axis[:, None] - clat) * m_lat + 0.0 * lon_axis[None, :]
    dem = ArrayElevation(lat_axis, lon_axis, grid)
    elev = float(dem.elevation_m(clat, clon))
    assert abs(elev - 500.0) < 1e-6

    t_slope, d_slope = truth(tr, clat, clon, elev)
    _, d_flat = truth(tr, clat, clon, 0.0)
    assert abs(d_flat - d_slope) > 100.0

    cands = generate_candidates(tr, SPEC, dem)
    hit = rows_for(cands, row, col, "A")
    assert len(hit) == 1
    h = hit.iloc[0]
    assert h["elev_m"] == pytest.approx(elev)
    assert abs(h["cpa_slant_m"] - d_slope) < 20.0
    assert abs(h["t_cpa_tx"] - t_slope) < 0.5
    # Every candidate's elevation is what the DEM says for its own centre.
    np.testing.assert_allclose(
        cands["elev_m"].to_numpy(), dem.elevation_m(cands["lat"].to_numpy(), cands["lon"].to_numpy()), atol=1e-6
    )


def test_two_in_range_intervals_give_two_rows_for_one_cell() -> None:
    """Out at 60 m/s, a fast 180-degree turn at 150 m/s, back at 60 m/s.

    The observer sits between the two legs, so the same cell gets one CPA
    per in-range interval.
    """
    v_turn, rate = 150.0, 3.0
    radius = v_turn / np.deg2rad(rate)
    leg1 = straight_track("OB", 1000.0, 300.0, LAT_S, LON_S, 1000.0, 60.0, 90.0)
    end1 = leg1.iloc[-1]
    turn = straight_track(
        "OB", float(end1["time"]) + 1.0, 60.0, float(end1["lat"]), float(end1["lon"]),
        1000.0, v_turn, 90.0, turn_rate_dps=-rate,
    )
    end2 = turn.iloc[-1]
    leg3 = straight_track("OB", float(end2["time"]) + 1.0, 300.0, float(end2["lat"]), float(end2["lon"]), 1000.0, 60.0, 270.0)
    tr = normalize_tracks(pd.concat([leg1, turn, leg3], ignore_index=True))

    row, col, _, _ = cell_at(9_000.0, radius)  # midway between the legs
    cands = generate_candidates(tr, EventSpec((40.0, 80.0), (500.0, 5000.0)), FlatElevation(0.0))
    hit = rows_for(cands, row, col, "OB").sort_values("t_cpa_tx")
    assert len(hit) == 2
    assert hit.iloc[0]["t_cpa_tx"] < 1300.0 < 1361.0 < hit.iloc[1]["t_cpa_tx"]
    assert hit["speed_mps"].between(55.0, 65.0).all()
    assert hit["cpa_slant_m"].between(2500.0, 3600.0).all()


def test_cpas_for_points_allows_endpoints_and_fills_elevation() -> None:
    tr = flight_a()
    far = straight_track("FAR", 1000.0, 600.0, 38.5, -120.0, 1000.0, 60.0, 90.0)  # >100 km away
    tracks = pd.concat([tr, far], ignore_index=True)
    r1, c1, lat1, lon1 = cell_at(18_000.0, 1_500.0)
    r2, c2, lat2, lon2 = cell_at(-3_000.0, 1_500.0)  # CPA at the first sample
    r3, c3, lat3, lon3 = cell_at(18_000.0, 20_000.0)  # beyond max_slant
    pts = pd.DataFrame({"row": [r1, r2, r3], "col": [c1, c2, c3], "lat": [lat1, lat2, lat3], "lon": [lon1, lon2, lon3]})
    dem = FlatElevation(250.0)
    out = cpas_for_points(tracks, pts, dem, max_slant_m=10_000.0, c=C)
    assert tuple(out.columns) == CPA_COLUMNS
    assert set(out["flight_id"]) == {"A"}
    assert set(zip(out["row"], out["col"])) == {(r1, c1), (r2, c2)}
    t_true, d_true = truth(tr, lat1, lon1, 250.0)
    h1 = out[(out["row"] == r1) & (out["col"] == c1)].iloc[0]
    assert abs(h1["t_cpa_tx"] - t_true) < 0.05
    assert abs(h1["cpa_slant_m"] - d_true) < 1.0
    assert h1["t_cpa_rx"] == pytest.approx(h1["t_cpa_tx"] + h1["cpa_slant_m"] / C)
    h2 = out[(out["row"] == r2) & (out["col"] == c2)].iloc[0]
    assert h2["t_cpa_tx"] == pytest.approx(1000.0)  # the track's first sample
    # An elev_m column with NaN gets filled; a given value is respected.
    pts2 = pts.assign(elev_m=[np.nan, 0.0, np.nan])
    out2 = cpas_for_points(tracks, pts2, dem, max_slant_m=10_000.0, c=C)
    h1b = out2[(out2["row"] == r1) & (out2["col"] == c1)].iloc[0]
    assert h1b["cpa_slant_m"] == pytest.approx(h1["cpa_slant_m"])
    h2b = out2[(out2["row"] == r2) & (out2["col"] == c2)].iloc[0]
    assert h2b["cpa_slant_m"] > h2["cpa_slant_m"]  # observer at 0 m is further from a 1000 m aircraft than one at 250 m
    with pytest.raises(ValueError):
        cpas_for_points(tracks, pts.drop(columns="lat"), dem)


def test_performance_twenty_flights_ten_minutes() -> None:
    rng = np.random.default_rng(7)
    flights = [
        straight_track(
            f"F{i:02d}", 1000.0 + rng.uniform(0, 600), 600.0,
            LAT_S + rng.uniform(-0.2, 0.2), LON_S + rng.uniform(-0.3, 0.3),
            rng.uniform(600.0, 2000.0), rng.uniform(50.0, 70.0), rng.uniform(0.0, 360.0),
        )
        for i in range(20)
    ]
    tracks = pd.concat(flights, ignore_index=True)
    spec = EventSpec(speed_range=(40.0, 80.0), cpa_slant_range=(300.0, 10_000.0))
    t0 = time.perf_counter()
    cands = generate_candidates(tracks, spec, FlatElevation(0.0))
    elapsed = time.perf_counter() - t0
    assert elapsed < 60.0, f"generate_candidates took {elapsed:.1f} s"
    assert cands["flight_id"].nunique() == 20
    assert cands["cpa_slant_m"].between(300.0, 10_000.0).all()
