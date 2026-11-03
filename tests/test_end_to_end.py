"""Full pipeline on synthetic audio: the search must find the microphone.

The audio is rendered by the waveform synthesiser (phase = f0 * emission
time), not by the frequency model the search uses, so this is an independent
check of the physics as well as of the plumbing.
"""

from pathlib import Path

import pytest

from dopplergeo.demo import run_demo


@pytest.fixture(scope="module")
def report(tmp_path_factory) -> dict:
    out = Path(tmp_path_factory.mktemp("demo"))
    return run_demo(out, seed=0, figures=False, log=lambda *_: None)


def test_measurements_bracket_the_truth(report):
    for key in ("event1", "event2"):
        m = report["measurements"][key]
        t = report["truth"]["evt1" if key == "event1" else "evt2"]
        assert m["speed_range"][0] <= t["speed"] <= m["speed_range"][1]
        assert m["cpa_slant_range"][0] <= t["slant_m"] <= m["cpa_slant_range"][1]
        assert m["t_cpa_range"][0] <= t["t_cpa_rx_audio"] <= m["t_cpa_range"][1]
        assert abs(m["f0_hz"] - t["f0"]) < 1.0


def test_truth_cell_survives_the_filters(report):
    assert report["two_event"]["truth_cell_survived_filters"]


def test_two_event_search_finds_the_microphone(report):
    best = report["two_event"]["best"]
    assert best["flight_id_1"] == "evt1" and best["flight_id_2"] == "evt2"
    assert report["two_event"]["error_m"] < 150.0
    top = report["two_event"]["top_pairs"]
    if len(top) > 1:
        assert top[0]["total"] > 1.3 * top[1]["total"]


def test_single_event_search_with_known_time(report):
    best = report["single_event"]["best"]
    assert best["flight_id"] == "evt2"
    assert report["single_event"]["error_m"] < 300.0
