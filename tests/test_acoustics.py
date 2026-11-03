"""Physics core: closed-form curve, real-track model, synthesiser, geodesy."""

import numpy as np
from scipy.signal import hilbert

from dopplergeo.acoustics import (
    Trajectory,
    closest_point_of_approach,
    doppler_frequency,
    ideal_curve,
    ideal_emission_offset,
    received_frequency,
    solve_emission_times,
)
from dopplergeo.adsb.schema import trajectory_from_rows
from dopplergeo.constants import SPEED_OF_SOUND_MPS as C
from dopplergeo.geo import (
    enu_from_latlon,
    grid_cell,
    grid_cell_center,
    grid_points,
    haversine_m,
    latlon_from_enu,
    slant_distance_m,
)
from dopplergeo.synth import Source, render_audio, straight_track


def test_doppler_sign_convention():
    assert doppler_frequency(100.0, -50.0) > 100.0  # approaching -> higher
    assert doppler_frequency(100.0, 50.0) < 100.0  # receding -> lower
    assert doppler_frequency(100.0, 0.0) == 100.0


def test_ideal_emission_offset_is_causal_and_consistent():
    v, d = 70.0, 1500.0
    t_rel = np.linspace(-60, 60, 241)
    tau = ideal_emission_offset(t_rel, v, d)
    r = np.sqrt(d * d + v * v * tau * tau)
    np.testing.assert_allclose(tau + r / C, t_rel, atol=1e-9)
    assert np.all(tau < t_rel)


def test_ideal_curve_crosses_f0_at_t_cpa_rx():
    f = ideal_curve(np.array([80.0, 100.0, 120.0]), 130.0, 80.0, 900.0, 100.0)
    assert f[0] > 130.0 > f[2]
    assert abs(f[1] - 130.0) < 1e-9
    # total ratio fixes the speed independent of f0 and d
    far = ideal_curve(np.array([-1e5, 1e5]), 130.0, 80.0, 900.0, 0.0)
    np.testing.assert_allclose(far[0] / far[1], (C + 80.0) / (C - 80.0), rtol=1e-6)


def _straight():
    df = straight_track("A", 1000.0, 300.0, 37.0, -122.0, 1000.0, 60.0, 90.0)
    traj = trajectory_from_rows(df, 37.0, -122.0)
    obs = np.array([60 * 150.0, 400.0, 0.0])  # CPA at t=1150, 400 m north, 1000 m below
    return traj, obs


def test_track_model_matches_closed_form_for_straight_flight():
    traj, obs = _straight()
    t_cpa, d = closest_point_of_approach(traj, obs)
    assert abs(t_cpa - 1150.0) < 0.3
    assert abs(d - np.hypot(400.0, 1000.0)) < 1.0
    t_rx = np.linspace(1100, 1200, 101)
    fm, valid = received_frequency(traj, obs, t_rx, 120.0)
    fi = ideal_curve(t_rx, 120.0, 60.0, d, t_cpa + d / C)
    assert valid.all()
    np.testing.assert_allclose(fm, fi, atol=1e-4)


def test_emission_time_solver_satisfies_delay_equation():
    traj, obs = _straight()
    t_rx = np.linspace(1050, 1250, 50)
    t_tx, valid = solve_emission_times(traj, obs, t_rx)
    r = np.linalg.norm(traj.position(t_tx) - obs, axis=-1)
    np.testing.assert_allclose(t_tx + r / C, t_rx, atol=1e-3)
    assert valid.all()


def test_naive_model_differs_from_retarded_model():
    """Using the radial velocity at receive time is not just a delayed curve."""
    traj, obs = _straight()
    t = np.linspace(1100, 1200, 401)
    fm, _ = received_frequency(traj, obs, t, 120.0)
    rel = traj.position(t) - obs
    vr = np.einsum("ij,ij->i", rel, traj.velocity(t)) / np.linalg.norm(rel, axis=1)
    f_naive = doppler_frequency(120.0, vr)
    # Best time shift of the naive curve still leaves a visible residual.
    best = min(
        np.max(np.abs(np.interp(t, t + s, f_naive) - fm)[40:-40])
        for s in np.arange(0, 6, 0.25)
    )
    assert best > 0.3


def test_synth_audio_instantaneous_frequency_matches_model():
    traj, obs = _straight()
    t_cpa, d = closest_point_of_approach(traj, obs)
    sr = 8000
    audio = render_audio([Source(traj, 120.0, harmonics=(1.0,))], obs, 1100.0, 100.0, sr, noise_rms=0.0)
    inst = np.diff(np.unwrap(np.angle(hilbert(audio.astype(float))))) * sr / (2 * np.pi)
    for t_audio in (10.0, 50.0, 90.0):
        i = int(t_audio * sr)
        measured = inst[i : i + sr // 10].mean()
        expected = ideal_curve(1100.0 + t_audio + 0.05, 120.0, 60.0, d, t_cpa + d / C)
        assert abs(measured - expected) < 0.05


def test_trajectory_accepts_two_points():
    tr = Trajectory(t=[0.0, 10.0], east=[0.0, 100.0], north=[0.0, 0.0], up=[0.0, 0.0])
    np.testing.assert_allclose(tr.position(5.0), [50.0, 0.0, 0.0])
    np.testing.assert_allclose(tr.speed(5.0), 10.0)
    assert tr.speed(11.0) == 0.0


def test_geo_roundtrip_and_distances():
    lat, lon = 37.1234, -122.4321
    e, n = enu_from_latlon(lat + 0.01, lon + 0.01, lat, lon)
    la, lo = latlon_from_enu(e, n, lat, lon)
    assert abs(la - (lat + 0.01)) < 1e-9 and abs(lo - (lon + 0.01)) < 1e-9
    assert abs(haversine_m(lat, lon, lat + 0.01, lon) - 1111.95) < 2.0
    assert abs(slant_distance_m(lat, lon, 0.0, lat, lon, 3000.0) - 3000.0) < 1e-6


def test_grid_cells_are_shared_and_near_200m():
    r, c = grid_cell(37.123, -122.456, 200.0)
    la, lo = grid_cell_center(r, c, 200.0)
    e, n = enu_from_latlon(la, lo, 37.123, -122.456)
    assert abs(e) <= 100.5 and abs(n) <= 100.5
    r2, c2 = grid_cell(la, lo, 200.0)
    assert r2 == r and c2 == c
    rr, cc, lats, lons = grid_points(37.0, 37.02, -122.02, -122.0, 200.0)
    assert 90 < len(rr) < 140
    assert lats.min() >= 36.999 and lats.max() <= 37.021
    rr2, cc2 = grid_cell(lats, lons, 200.0)
    assert np.array_equal(rr, rr2) and np.array_equal(cc, cc2)
