import json

import numpy as np
import pytest

from dopplergeo.acoustics import closest_point_of_approach, ideal_curve
from dopplergeo.adsb.schema import trajectory_from_rows
from dopplergeo.constants import SPEED_OF_SOUND_MPS as C
from dopplergeo.curve_fit import CurveMeasurement, fit_ideal_curve
from dopplergeo.spectrogram import Ridge, compute_spectrogram, extract_ridge
from dopplergeo.synth import Source, render_audio, straight_track

SR = 8000
LAT0, LON0 = 37.0, -122.0
T_START = 1100.0
DURATION = 100.0
F0 = 120.0


def _scenario(speed_mps: float = 60.0, noise_rms: float = 0.05, **track_kwargs):
    """Flyover with the observer 400 m abeam of the straight-line CPA at
    t = 1150 (geometry clock), 1000 m below the aircraft."""
    df = straight_track(
        "A", 1000.0, 300.0, LAT0, LON0, 1000.0, speed_mps, 90.0, **track_kwargs
    )
    traj = trajectory_from_rows(df, LAT0, LON0)
    obs = np.array([speed_mps * 150.0, 400.0, 0.0])
    t_cpa_tx, d = closest_point_of_approach(traj, obs)
    t_cpa_rx = t_cpa_tx + d / C - T_START  # audio clock
    audio = render_audio([Source(traj, F0)], obs, T_START, DURATION, SR, noise_rms=noise_rms)
    return traj, obs, audio, {"speed": speed_mps, "slant": d, "t_cpa_rx": t_cpa_rx, "f0": F0}


def _fit(audio, f_lo: float = 90.0, f_hi: float = 170.0, max_slope: float = 3.0):
    spec = compute_spectrogram(audio, SR)
    ridge = extract_ridge(spec, 5.0, 95.0, f_lo, f_hi, max_slope_hz_per_s=max_slope)
    return fit_ideal_curve(ridge), ridge


def _assert_close(m: CurveMeasurement, truth: dict) -> None:
    assert abs(m.speed_mps - truth["speed"]) < 3.0
    assert abs(m.cpa_slant_m - truth["slant"]) < 0.10 * truth["slant"]
    assert abs(m.f0_hz - truth["f0"]) < 0.5
    assert abs(m.t_cpa_rx - truth["t_cpa_rx"]) < 0.5


def _assert_ranges_contain(m: CurveMeasurement, truth: dict) -> None:
    assert m.speed_range[0] <= truth["speed"] <= m.speed_range[1]
    assert m.cpa_slant_range[0] <= truth["slant"] <= m.cpa_slant_range[1]
    assert m.t_cpa_range[0] <= truth["t_cpa_rx"] <= m.t_cpa_range[1]
    assert m.f0_range[0] <= truth["f0"] <= m.f0_range[1]


def test_fit_straight_track_clean():
    _, _, audio, truth = _scenario()
    m, _ = _fit(audio)
    _assert_close(m, truth)
    _assert_ranges_contain(m, truth)
    assert m.rmse_hz < 0.3


def test_fit_straight_track_noisy():
    _, _, audio, truth = _scenario(noise_rms=5.0)
    m, ridge = _fit(audio)
    assert np.median(ridge.strength) < 5.0  # genuinely weak tone
    _assert_close(m, truth)
    _assert_ranges_contain(m, truth)
    assert m.rmse_hz < 0.5


def test_fit_fast_aircraft():
    _, _, audio, truth = _scenario(speed_mps=120.0)
    # (c + v) / (c - v) = 2.08: the tone sweeps from ~173 Hz down to ~83 Hz,
    # and the slope at CPA, f0 v^2 / (c d) = 4.7 Hz/s, exceeds the default
    # 3 Hz/s ridge limit, so the extractor needs a looser one here.
    slope_at_cpa = F0 * truth["speed"] ** 2 / (C * truth["slant"])
    assert slope_at_cpa > 3.0
    m, _ = _fit(audio, f_lo=75.0, f_hi=185.0, max_slope=6.0)
    _assert_close(m, truth)
    _assert_ranges_contain(m, truth)


def test_fit_turning_track_degrades_gracefully():
    _, _, audio, truth = _scenario(turn_rate_dps=0.3)
    m, _ = _fit(audio)
    # Not a straight line, so the point estimate drifts, but not far ...
    assert abs(m.speed_mps - truth["speed"]) < 10.0
    assert abs(m.cpa_slant_m - truth["slant"]) < 0.10 * truth["slant"]
    assert abs(m.t_cpa_rx - truth["t_cpa_rx"]) < 1.0
    assert abs(m.f0_hz - truth["f0"]) < 0.5
    # ... and the ranges still contain the truth.
    _assert_ranges_contain(m, truth)


def test_fit_on_exact_ideal_curve_recovers_parameters():
    t = np.arange(0.0, 90.0, 0.25)
    truth = {"f0": 110.0, "speed": 75.0, "slant": 800.0, "t_cpa_rx": 47.3}
    f = ideal_curve(t, truth["f0"], truth["speed"], truth["slant"], truth["t_cpa_rx"])
    ridge = Ridge(t, f, np.full(t.size, 6.0))
    m = fit_ideal_curve(ridge)
    assert abs(m.speed_mps - 75.0) < 0.1
    assert abs(m.cpa_slant_m - 800.0) < 2.0
    assert abs(m.f0_hz - 110.0) < 0.01
    assert abs(m.t_cpa_rx - 47.3) < 0.01
    assert m.rmse_hz < 1e-3
    # Ranges never shrink below the minimum halfwidths.
    assert m.speed_range == pytest.approx((60.0, 90.0))
    assert m.cpa_slant_range == pytest.approx((500.0, 1100.0))
    assert m.t_cpa_range == pytest.approx((44.3, 50.3))
    assert m.f0_range == pytest.approx((108.0, 112.0))


def test_fit_weighting_ignores_faded_segments():
    t = np.arange(0.0, 90.0, 0.25)
    f = ideal_curve(t, 110.0, 75.0, 800.0, 47.3)
    rng = np.random.default_rng(3)
    # The last third of the ridge is garbage with zero strength.
    bad = t > 60.0
    f_bad = f.copy()
    f_bad[bad] = rng.uniform(100.0, 130.0, bad.sum())
    strength = np.where(bad, 0.0, 6.0)
    m = fit_ideal_curve(Ridge(t, f_bad, strength))
    assert abs(m.speed_mps - 75.0) < 0.5
    assert abs(m.cpa_slant_m - 800.0) < 10.0
    # Unweighted, the garbage pulls the fit away.
    m_u = fit_ideal_curve(Ridge(t, f_bad, strength), weight_by_strength=False)
    assert m_u.rmse_hz > m.rmse_hz


def test_fit_bounds_are_respected():
    t = np.arange(0.0, 90.0, 0.25)
    f = ideal_curve(t, 110.0, 75.0, 800.0, 47.3)
    ridge = Ridge(t, f, np.full(t.size, 6.0))
    m = fit_ideal_curve(ridge, speed_bounds=(15.0, 50.0), slant_bounds=(50.0, 10_000.0))
    assert 15.0 <= m.speed_mps <= 50.0
    assert m.rmse_hz > 0.5  # the true curve is outside the box


def test_fit_rejects_tiny_ridge():
    t = np.arange(0.0, 1.0, 0.25)
    with pytest.raises(ValueError):
        fit_ideal_curve(Ridge(t, np.full(t.size, 100.0), np.ones(t.size)))


def test_fit_handles_flat_ridge_with_minimum_halfwidths():
    # A constant tone carries no Doppler information: the fit must still
    # return finite numbers and ranges no narrower than the minimums.
    t = np.arange(0.0, 60.0, 0.25)
    m = fit_ideal_curve(Ridge(t, np.full(t.size, 100.0), np.ones(t.size)))
    for v in (m.f0_hz, m.speed_mps, m.cpa_slant_m, m.t_cpa_rx, m.rmse_hz):
        assert np.isfinite(v)
    assert m.speed_range[1] - m.speed_range[0] >= 2 * 15.0 - 1e-9
    assert m.cpa_slant_range[1] - m.cpa_slant_range[0] >= 2 * 300.0 - 1e-9
    assert m.t_cpa_range[1] - m.t_cpa_range[0] >= 2 * 3.0 - 1e-9
    assert m.f0_range[1] - m.f0_range[0] >= 2 * 2.0 - 1e-9


def test_measurement_json_roundtrip():
    m = CurveMeasurement(
        f0_hz=120.0, speed_mps=60.0, cpa_slant_m=1077.0, t_cpa_rx=53.14, rmse_hz=0.02,
        speed_range=(45.0, 75.0), cpa_slant_range=(777.0, 1377.0),
        t_cpa_range=(50.1, 56.1), f0_range=(118.0, 122.0),
    )
    d = m.to_dict()
    text = json.dumps(d)
    back = CurveMeasurement.from_dict(json.loads(text))
    assert back == m
    assert isinstance(back.speed_range, tuple)
