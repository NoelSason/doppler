"""Tests for the ADS-B ingestion modules (aircraft_types, sources, tracks)."""

from __future__ import annotations

import gzip
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from dopplergeo.adsb import aircraft_types, sources, tracks
from dopplergeo.adsb.schema import COLUMNS, validate_tracks
from dopplergeo.constants import FEET_TO_M, FPM_TO_MPS, KNOTS_TO_MPS
from dopplergeo.geo import metres_per_degree

T0 = 1_757_300_000.0  # arbitrary unix time


def fake_trace(icao: str = "A1B2C3", typecode: str = "C172", lat0: float = 37.0) -> dict[str, Any]:
    """A readsb trace_full object with the awkward cases in it."""
    return {
        "icao": icao,
        "r": "N123AB",
        "t": typecode,
        "timestamp": T0,
        "trace": [
            # on the ground -> dropped
            [0.0, lat0, -122.0, "ground", 10.0, 90.0, 0, 0,
             {"flight": "N123AB  ", "t": "C172"}, "adsb_icao", None, None, None, None],
            # geometric altitude and rate present
            [5.0, lat0 + 0.001, -122.001, 1000, 100.0, 90.0, 0, 500,
             None, "adsb_icao", 1050, 480, 95, 0.0],
            # alt_geom missing -> baro
            [10.0, lat0 + 0.002, -122.002, 1100, 100.0, 91.0, 0, 500,
             None, "adsb_icao", None, None, None, None],
            # trailing short entry (no aircraft/source/geom fields)
            [15.0, lat0 + 0.003, -122.003, 1200, 100.0, 92.0, 0, 500],
            # no altitude at all -> dropped
            [20.0, lat0 + 0.004, -122.004, None, 100.0, 93.0, 0, 0],
            # stale position flag -> dropped by default
            [25.0, lat0 + 0.004, -122.004, 1300, 100.0, 93.0, 1, 0],
        ],
    }


# ----------------------------------------------------------------------
# parse_readsb_trace
# ----------------------------------------------------------------------


def test_parse_readsb_trace_conversion() -> None:
    df = sources.parse_readsb_trace(fake_trace())
    validate_tracks(df)
    assert list(df.columns) == list(COLUMNS)
    assert len(df) == 3
    assert df["time"].tolist() == [T0 + 5, T0 + 10, T0 + 15]
    assert (df["icao24"] == "a1b2c3").all()
    assert (df["flight_id"] == "a1b2c3").all()
    assert (df["callsign"] == "N123AB").all()
    assert (df["typecode"] == "C172").all()
    assert df["alt_source"].tolist() == ["geom", "baro", "baro"]
    np.testing.assert_allclose(df["alt_m"].to_numpy(), [1050 * FEET_TO_M, 1100 * FEET_TO_M, 1200 * FEET_TO_M])
    np.testing.assert_allclose(df["gs_mps"].to_numpy(), 100.0 * KNOTS_TO_MPS)
    np.testing.assert_allclose(df["vrate_mps"].to_numpy(), [480 * FPM_TO_MPS, 500 * FPM_TO_MPS, 500 * FPM_TO_MPS])
    assert df["track_deg"].tolist() == [90.0, 91.0, 92.0]


def test_parse_readsb_trace_keeps_stale_when_asked() -> None:
    df = sources.parse_readsb_trace(fake_trace(), drop_stale=False)
    assert len(df) == 4
    assert df["time"].iloc[-1] == T0 + 25


def test_parse_readsb_trace_typecode_fallback_and_empty() -> None:
    obj = fake_trace(typecode="")
    df = sources.parse_readsb_trace(obj)
    assert (df["typecode"] == "C172").all()  # from the aircraft dict
    assert len(sources.parse_readsb_trace({"icao": "abc", "timestamp": T0, "trace": []})) == 0


# ----------------------------------------------------------------------
# load_readsb_history
# ----------------------------------------------------------------------


def _write_history(root: Path) -> None:
    a = root / "traces" / "c3" / "trace_full_a1b2c3.json"
    a.parent.mkdir(parents=True)
    a.write_text(json.dumps(fake_trace("A1B2C3", "C172", lat0=37.0)))
    b = root / "traces" / "ee" / "trace_full_ddddee.json.gz"
    b.parent.mkdir(parents=True)
    with gzip.open(b, "wt", encoding="utf-8") as fh:
        json.dump(fake_trace("DDDDEE", "B738", lat0=40.0), fh)


def test_load_readsb_history_with_filters(tmp_path: Path) -> None:
    _write_history(tmp_path)
    all_rows = sources.load_readsb_history(tmp_path)
    validate_tracks(all_rows)
    assert sorted(all_rows["icao24"].unique()) == ["a1b2c3", "ddddee"]
    assert len(all_rows) == 6

    in_box = sources.load_readsb_history(tmp_path, bbox=(36.5, 37.5, -123.0, -121.0))
    assert in_box["icao24"].unique().tolist() == ["a1b2c3"]
    assert len(in_box) == 3

    in_time = sources.load_readsb_history(tmp_path, time_range=(T0 + 8, T0 + 12))
    assert len(in_time) == 2  # one row per file at T0 + 10
    assert (in_time["time"] == T0 + 10).all()

    nothing = sources.load_readsb_history(tmp_path, time_range=(T0 + 1000, T0 + 2000))
    assert len(nothing) == 0
    assert list(nothing.columns) == list(COLUMNS)


# ----------------------------------------------------------------------
# adsb.lol download helpers (offline parts)
# ----------------------------------------------------------------------


def test_adsblol_release_tag() -> None:
    assert sources.adsblol_release_tag("2026-09-08") == "v2026.09.08-planes-readsb-prod-0"
    with pytest.raises(ValueError):
        sources.adsblol_release_tag("2026-9-8")


def test_extract_split_tar(tmp_path: Path) -> None:
    payload = json.dumps(fake_trace()).encode()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo("traces/c3/trace_full_a1b2c3.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    data = buf.getvalue()
    cut = len(data) // 3
    parts = [tmp_path / "x.tar.aa", tmp_path / "x.tar.ab", tmp_path / "x.tar.ac"]
    parts[0].write_bytes(data[:cut])
    parts[1].write_bytes(data[cut : 2 * cut])
    parts[2].write_bytes(data[2 * cut :])
    out = sources.extract_split_tar(parts, tmp_path / "out")
    df = sources.load_readsb_history(out)
    assert len(df) == 3


# ----------------------------------------------------------------------
# OpenSky CSV
# ----------------------------------------------------------------------


def test_load_opensky_csv(tmp_path: Path) -> None:
    csv = tmp_path / "os.csv"
    pd.DataFrame(
        {
            "time": [T0, T0 + 10, T0 + 20],
            "icao24": ["ABC123", "abc123", "abc123"],
            "lat": [37.0, 37.01, 37.02],
            "lon": [-122.0, -122.0, -122.0],
            "velocity": [60.0, 61.0, np.nan],
            "heading": [10.0, 11.0, 12.0],
            "vertrate": [1.0, 1.0, 1.0],
            "callsign": ["N1  ", "N1", "N1"],
            "baroaltitude": [300.0, 310.0, 320.0],
            "geoaltitude": [np.nan, 330.0, 340.0],
            "onground": [False, False, True],
            "typecode": ["c172", "C172", "C172"],
        }
    ).to_csv(csv, index=False)
    df = sources.load_opensky_csv(csv)
    validate_tracks(df)
    assert len(df) == 2  # on-ground row dropped
    assert df["alt_source"].tolist() == ["baro", "geom"]
    assert df["alt_m"].tolist() == [300.0, 330.0]
    assert (df["icao24"] == "abc123").all()
    assert (df["callsign"] == "N1").all()
    assert (df["typecode"] == "C172").all()


# ----------------------------------------------------------------------
# tracks
# ----------------------------------------------------------------------


def _straight(n: int, *, fid: str = "abc", t0: float = 0.0, step: float = 10.0, dlat: float = 0.001) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "flight_id": fid,
            "icao24": fid,
            "callsign": "X",
            "typecode": "C172",
            "time": t0 + step * np.arange(n),
            "lat": 37.0 + dlat * np.arange(n),
            "lon": -122.0,
            "alt_m": 500.0,
            "alt_source": "baro",
            "gs_mps": 50.0,
            "track_deg": 0.0,
            "vrate_mps": 0.0,
        }
    )


def test_split_on_gaps_time_and_speed() -> None:
    a = _straight(5)  # t = 0..40
    b = _straight(5, t0=400.0)  # gap of 360 s -> new segment
    c = _straight(3, t0=800.0)
    c.loc[1, "lat"] = 38.0  # ~111 km in 10 s -> impossible -> new segment
    whole = _straight(4, fid="xyz")  # no gap: keeps its id
    df = pd.concat([a, b, c, whole], ignore_index=True)
    out = tracks.split_on_gaps(df, max_gap_s=60.0)
    validate_tracks(out)
    abc = out.loc[out["icao24"] == "abc", "flight_id"].tolist()
    assert abc[:5] == ["abc_0"] * 5
    assert abc[5:10] == ["abc_1"] * 5
    # c: first row joins after a gap (segment 2), the jump starts segment 3,
    # and the return jump starts segment 4.
    assert abc[10:] == ["abc_2", "abc_3", "abc_4"]
    assert out.loc[out["icao24"] == "xyz", "flight_id"].tolist() == ["xyz"] * 4
    assert out["flight_id"].nunique() == 6


def test_resample_tracks() -> None:
    df = _straight(3, step=10.0)  # t = 0, 10, 20
    df.loc[:, "track_deg"] = [350.0, 10.0, 30.0]
    df.loc[1, "gs_mps"] = np.nan
    out = tracks.resample_tracks(df, step_s=1.0)
    validate_tracks(out)
    assert len(out) == 21
    t = out["time"].to_numpy()
    assert np.all(np.diff(t) > 0)
    np.testing.assert_allclose(t, np.arange(21.0))
    np.testing.assert_allclose(out.loc[5, "lat"], 37.0 + 0.0005)
    np.testing.assert_allclose(out.loc[15, "lat"], 37.0 + 0.0015)
    # 350 -> 10 through north: midpoint is 0, not 180
    np.testing.assert_allclose(out.loc[5, "track_deg"], 0.0, atol=1e-9)
    np.testing.assert_allclose(out.loc[15, "track_deg"], 20.0)
    # NaN gs sample skipped, so the line is flat at 50
    np.testing.assert_allclose(out["gs_mps"].to_numpy(), 50.0)
    assert (out["callsign"] == "X").all()
    assert (out["flight_id"] == "abc").all()


def test_resample_tracks_antimeridian() -> None:
    df = _straight(2, step=10.0)
    df["lon"] = [179.9, -179.9]
    out = tracks.resample_tracks(df, step_s=5.0)
    assert out["lon"].tolist()[1] == pytest.approx(180.0) or out["lon"].tolist()[1] == pytest.approx(-180.0)


def test_add_speed_3d_fallback() -> None:
    m_lat, _ = metres_per_degree(37.0)
    df = _straight(5, step=1.0, dlat=100.0 / m_lat)  # 100 m per second north
    df["gs_mps"] = np.nan
    df["vrate_mps"] = np.nan
    df["alt_m"] = 500.0 + 10.0 * np.arange(5)  # 10 m/s climb
    out = tracks.add_speed_3d(df)
    assert "speed_mps" in out.columns
    np.testing.assert_allclose(out["speed_mps"].to_numpy(), np.hypot(100.0, 10.0), rtol=1e-3)
    # reported values win when present
    df2 = _straight(3)
    df2["gs_mps"] = 30.0
    df2["vrate_mps"] = 4.0
    out2 = tracks.add_speed_3d(df2)
    np.testing.assert_allclose(out2["speed_mps"].to_numpy(), 5.0 * np.hypot(6.0, 0.8))


def test_clip_bbox_and_time() -> None:
    df = _straight(5)
    box = tracks.clip_bbox(df, 37.0015, 37.0035, -123.0, -121.0)
    assert box["lat"].tolist() == [37.002, 37.003]
    assert (box["flight_id"] == "abc").all()
    tt = tracks.clip_time(df, 10.0, 20.0)
    assert tt["time"].tolist() == [10.0, 20.0]


# ----------------------------------------------------------------------
# aircraft types
# ----------------------------------------------------------------------


def test_engine_class_examples() -> None:
    assert aircraft_types.engine_class("C172") == "piston"
    assert aircraft_types.engine_class("B738") == "jet"
    assert aircraft_types.engine_class("AT72") == "turboprop"
    assert aircraft_types.engine_class("PAY2") == "turboprop"
    assert aircraft_types.engine_class("JS41") == "turboprop"
    assert aircraft_types.engine_class("PA46") == "piston"
    assert aircraft_types.engine_class("") == "unknown"
    assert aircraft_types.engine_class("ZZZZ") == "unknown"
    assert aircraft_types.engine_class(None) == "unknown"
    assert aircraft_types.engine_class(float("nan")) == "unknown"
    assert aircraft_types.engine_class(" b738 ") == "jet"
    assert aircraft_types.is_known_jet("A320")
    assert not aircraft_types.is_known_jet("")


def test_filter_out_jets_keeps_unknown() -> None:
    df = pd.DataFrame(
        {
            "flight_id": list("abcde"),
            "typecode": ["B738", "C172", "", "ZZZZ", "AT72"],
            "time": np.arange(5.0),
        }
    )
    out = aircraft_types.filter_out_jets(df)
    assert out["flight_id"].tolist() == ["b", "c", "d", "e"]
    assert len(aircraft_types.filter_out_jets(df.drop(columns=["typecode"]))) == 5


def test_load_type_table_opensky_and_generic(tmp_path: Path) -> None:
    aircraft_types.reset_type_table()
    try:
        opensky = tmp_path / "aircraftDatabase.csv"
        pd.DataFrame(
            {
                "icao24": ["1", "2", "3", "4", "5"],
                "typecode": ["QQ01", "QQ01", "QQ02", "QQ03", "QQ03"],
                "icaoaircrafttype": ["L2J", "L2J", "L1P", "L2J", "L2T"],  # QQ03 contested
            }
        ).to_csv(opensky, index=False)
        aircraft_types.load_type_table(opensky)
        assert aircraft_types.engine_class("QQ01") == "jet"
        assert aircraft_types.engine_class("QQ02") == "piston"
        assert aircraft_types.engine_class("QQ03") == "unknown"

        generic = tmp_path / "types.csv"
        generic.write_text("typecode,engine_type\nQQ04,jet\nQQ05,T\nQQ06,L1E\nC172,jet\n")
        aircraft_types.load_type_table(generic)
        assert aircraft_types.engine_class("QQ04") == "jet"
        assert aircraft_types.engine_class("QQ05") == "turboprop"
        assert aircraft_types.engine_class("QQ06") == "electric"
        assert aircraft_types.engine_class("C172") == "piston"  # curated entry not overridden
        aircraft_types.load_type_table(generic, override=True)
        assert aircraft_types.engine_class("C172") == "jet"
    finally:
        aircraft_types.reset_type_table()
    assert aircraft_types.engine_class("C172") == "piston"
    assert aircraft_types.engine_class("QQ01") == "unknown"
