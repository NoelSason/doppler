"""Tests for dopplergeo.model and dopplergeo.scoring.

The spectrogram is a test double (no import of dopplergeo.spectrogram): a
(n_freq, n_time) brightness array with a vectorised bilinear ``sample``,
onto which the exact curves from ``acoustics.received_frequency`` are painted
for two flights heard at a known audio/geometry clock offset.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from dopplergeo.acoustics import Trajectory
from dopplergeo.adsb.schema import trajectory_from_rows
from dopplergeo.geo import latlon_from_enu
from dopplergeo.model import curve_times, model_curve, observer_enu_for, t_cpa_rx_for
from dopplergeo.scoring import (
    EventScore,
    PairScore,
    ScoreConfig,
    score_candidates,
    score_event,
    score_pair,
    score_single,
)
from dopplergeo.synth import straight_track

LAT0, LON0 = 37.0, -122.0
T0 = 1_700_000_000.0  # geometry clock (unix s)
T_OFFSET = T0 + 10.0  # audio = geometry - T_OFFSET
F0_1, F0_2 = 100.0, 120.0
EVENT_GAP_S = 920.0


# --------------------------------------------------------------------------
# Spectrogram test double
# --------------------------------------------------------------------------


@dataclass
class FakeSpectrogram:
    times: np.ndarray
    freqs: np.ndarray
    values: np.ndarray  # (n_freq, n_time), whitened brightness >= 0

    def sample(self, t: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = np.asarray(t, dtype=float)
        f = np.asarray(f, dtype=float)
        dt = self.times[1] - self.times[0]
        df = self.freqs[1] - self.freqs[0]
        inb = (t >= self.times[0]) & (t <= self.times[-1]) & (f >= self.freqs[0]) & (f <= self.freqs[-1])
        ti = np.clip((t - self.times[0]) / dt, 0.0, self.times.size - 1.0)
        fi = np.clip((f - self.freqs[0]) / df, 0.0, self.freqs.size - 1.0)
        i0 = np.minimum(np.floor(ti).astype(int), self.times.size - 2)
        j0 = np.minimum(np.floor(fi).astype(int), self.freqs.size - 2)
        wt = ti - i0
        wf = fi - j0
        v = (
            self.values[j0, i0] * (1 - wf) * (1 - wt)
            + self.values[j0 + 1, i0] * wf * (1 - wt)
            + self.values[j0, i0 + 1] * (1 - wf) * wt
            + self.values[j0 + 1, i0 + 1] * wf * wt
        )
        return np.where(inb, v, 0.0), inb


def paint_curve(
    spec: FakeSpectrogram,
    traj: Trajectory,
    obs: np.ndarray,
    f0: float,
    t_cpa_rx_geom: float,
    *,
    harmonics: tuple[float, ...] = (1.0,),
    half_window_s: float = 90.0,
    sigma_hz: float = 0.5,
    amplitude: float = 6.0,
) -> None:
    """Add a Gaussian ridge along the exact modelled curve (audio clock)."""
    t_geom = curve_times(t_cpa_rx_geom, half_window_s, spec.times[1] - spec.times[0])
    freq, valid = model_curve(traj, obs, f0, t_geom)
    t_audio = t_geom - T_OFFSET
    cols = np.rint((t_audio - spec.times[0]) / (spec.times[1] - spec.times[0])).astype(int)
    ok = valid & (cols >= 0) & (cols < spec.times.size)
    for k, w in enumerate(harmonics, start=1):
        ridge = amplitude * w * np.exp(-((spec.freqs[:, None] - k * freq[ok][None, :]) ** 2) / (2 * sigma_hz**2))
        spec.values[:, cols[ok]] += ridge


# --------------------------------------------------------------------------
# Scene: two flights, one observer, one clock offset
# --------------------------------------------------------------------------


@dataclass
class Scene:
    spec: FakeSpectrogram
    spec_harm: FakeSpectrogram
    traj1: Trajectory
    traj2: Trajectory
    rows1: pd.DataFrame
    rows2: pd.DataFrame
    obs_lat: float
    obs_lon: float
    obs_elev: float
    obs: np.ndarray
    t_cpa_rx_1: float
    t_cpa_rx_2: float
    slant_1: float
    slant_2: float
    m1: SimpleNamespace
    m2: SimpleNamespace


def _track_from_enu(flight_id: str, t0: float, east0: float, north0: float, **kw: float | str) -> pd.DataFrame:
    lat, lon = latlon_from_enu(east0, north0, LAT0, LON0)
    return straight_track(flight_id, t0, 240.0, float(lat), float(lon), **kw)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def scene() -> Scene:
    obs_lat, obs_lon = latlon_from_enu(120.0, -80.0, LAT0, LON0)
    obs_elev = 50.0
    obs = observer_enu_for(float(obs_lat), float(obs_lon), obs_elev, LAT0, LON0)

    # Flight 1: straight, eastbound, passes ~500 m north of the observer at 350 m AMSL.
    rows1 = _track_from_enu("F1", T0, -7200.0 + 120.0, 500.0 - 80.0, alt_m=350.0, speed_mps=60.0, heading_deg=90.0)
    # Flight 2: 920 s later, northbound and turning, slightly faster, climbing.
    rows2 = _track_from_enu(
        "F2",
        T0 + EVENT_GAP_S,
        -1300.0 + 120.0,
        -7200.0 - 80.0,
        alt_m=400.0,
        speed_mps=55.0,
        heading_deg=0.0,
        turn_rate_dps=0.3,
        climb_mps=1.0,
    )
    traj1 = trajectory_from_rows(rows1, LAT0, LON0)
    traj2 = trajectory_from_rows(rows2, LAT0, LON0)
    _, slant_1, t_cpa_rx_1 = t_cpa_rx_for(traj1, obs)
    _, slant_2, t_cpa_rx_2 = t_cpa_rx_for(traj2, obs)

    times = np.arange(0.0, 1200.0 + 1e-9, 0.25)
    freqs = np.arange(0.0, 300.0 + 1e-9, 0.25)
    spec = FakeSpectrogram(times, freqs, np.zeros((freqs.size, times.size)))
    paint_curve(spec, traj1, obs, F0_1, t_cpa_rx_1)
    paint_curve(spec, traj2, obs, F0_2, t_cpa_rx_2)
    spec_harm = FakeSpectrogram(times, freqs, np.zeros((freqs.size, times.size)))
    paint_curve(spec_harm, traj1, obs, F0_1, t_cpa_rx_1, harmonics=(1.0, 0.5, 0.3))
    paint_curve(spec_harm, traj2, obs, F0_2, t_cpa_rx_2, harmonics=(1.0, 0.5, 0.3))

    # Measurements deliberately off the truth: f0 by +0.5 Hz / -0.5 Hz and
    # event 1's time by +1.0 s, so the sweeps have something to recover.
    m1 = SimpleNamespace(f0_hz=F0_1 + 0.5, t_cpa_rx=t_cpa_rx_1 - T_OFFSET + 1.0)
    m2 = SimpleNamespace(f0_hz=F0_2 - 0.5, t_cpa_rx=t_cpa_rx_2 - T_OFFSET)
    return Scene(
        spec, spec_harm, traj1, traj2, rows1, rows2,
        float(obs_lat), float(obs_lon), obs_elev, obs,
        t_cpa_rx_1, t_cpa_rx_2, slant_1, slant_2, m1, m2,
    )


@pytest.fixture(scope="module")
def cfg() -> ScoreConfig:
    return ScoreConfig()


# --------------------------------------------------------------------------
# model.py
# --------------------------------------------------------------------------


def test_curve_times_symmetric_and_contains_center() -> None:
    t = curve_times(10.0, 2.0, 0.25)
    assert t.size == 17
    assert t[0] == pytest.approx(8.0) and t[-1] == pytest.approx(12.0)
    assert np.any(np.isclose(t, 10.0))


def test_observer_enu_for_matches_frame() -> None:
    lat, lon = latlon_from_enu(300.0, -200.0, LAT0, LON0)
    enu = observer_enu_for(float(lat), float(lon), 77.0, LAT0, LON0)
    assert enu.shape == (3,)
    np.testing.assert_allclose(enu, [300.0, -200.0, 77.0], atol=1e-3)


def test_t_cpa_rx_for_is_tx_plus_travel(scene: Scene) -> None:
    t_tx, slant, t_rx = t_cpa_rx_for(scene.traj1, scene.obs)
    assert t_rx == pytest.approx(t_tx + slant / 343.0)
    assert 500.0 < slant < 700.0  # 500 m horizontal, 300 m vertical
    assert 200.0 < scene.slant_2 < 3000.0
    assert scene.traj2.t_start < t_tx + EVENT_GAP_S < scene.traj2.t_end


def test_model_curve_crosses_f0_at_cpa(scene: Scene) -> None:
    t = curve_times(scene.t_cpa_rx_1, 30.0, 0.5)
    f, valid = model_curve(scene.traj1, scene.obs, F0_1, t)
    assert valid.all()
    k = np.argmin(np.abs(t - scene.t_cpa_rx_1))
    assert f[k] == pytest.approx(F0_1, abs=0.05)
    assert f[0] > F0_1 + 10 and f[-1] < F0_1 - 10


# --------------------------------------------------------------------------
# scoring.py
# --------------------------------------------------------------------------


def _pair_at(scene: Scene, obs: np.ndarray, cfg: ScoreConfig, spec: FakeSpectrogram | None = None) -> PairScore:
    _, _, t1 = t_cpa_rx_for(scene.traj1, obs)
    _, _, t2 = t_cpa_rx_for(scene.traj2, obs)
    return score_pair(
        spec or scene.spec,
        scene.traj1,
        scene.traj2,
        obs,
        m1=scene.m1,
        m2=scene.m2,
        t_cpa_rx_geom_1=t1,
        t_cpa_rx_geom_2=t2,
        cfg=cfg,
    )


def test_true_observer_recovers_dt_and_f0(scene: Scene, cfg: ScoreConfig) -> None:
    ps = _pair_at(scene, scene.obs, cfg)
    # Painted ridge peaks at 6.0; a mean along the curve near that is "high".
    assert ps.score_1 > 4.5
    assert ps.score_2 > 4.5
    assert ps.total == pytest.approx(ps.score_1 + ps.score_2)
    assert ps.dt_s == pytest.approx(-1.0, abs=cfg.time_step_s + 1e-9)
    assert ps.f0_1 == pytest.approx(F0_1, abs=cfg.f0_step_hz + 1e-9)
    assert ps.f0_2 == pytest.approx(F0_2, abs=cfg.f0_step_hz + 1e-9)


def test_observer_300m_away_scores_lower(scene: Scene, cfg: ScoreConfig) -> None:
    true = _pair_at(scene, scene.obs, cfg)
    # Across-track moves change the curve's transition shape for both flights.
    for d in ((0.0, 300.0), (-212.0, 212.0), (0.0, -300.0), (300.0, 300.0)):
        far = _pair_at(scene, scene.obs + np.array([d[0], d[1], 0.0]), cfg)
        assert far.total < 0.65 * true.total, (d, far, true)
        assert far.score_1 < 0.75 * true.score_1 and far.score_2 < 0.75 * true.score_2
    # An along-track move for the eastbound flight 1 is only a time shift,
    # which the row's own CPA time re-anchors away. The northbound flight 2,
    # placed with event 1's dt, is what exposes it: that is the pair's job.
    along = _pair_at(scene, scene.obs + np.array([300.0, 0.0, 0.0]), cfg)
    assert along.score_1 == pytest.approx(true.score_1, rel=0.02)
    assert along.score_2 < 0.65 * true.score_2
    assert along.total < 0.85 * true.total


def test_lock_penalises_wrong_inter_event_offset(scene: Scene, cfg: ScoreConfig) -> None:
    # Same geometry for event 2, but the flight passes 15 s later than the
    # one that was painted (a candidate pair with the wrong time gap).
    rows2_late = scene.rows2.copy()
    rows2_late["time"] = rows2_late["time"] + 15.0
    traj2_late = trajectory_from_rows(rows2_late, LAT0, LON0)
    _, _, t1 = t_cpa_rx_for(scene.traj1, scene.obs)
    _, _, t2_late = t_cpa_rx_for(traj2_late, scene.obs)
    assert t2_late == pytest.approx(scene.t_cpa_rx_2 + 15.0, abs=0.3)

    good = _pair_at(scene, scene.obs, cfg)
    bad = score_pair(
        scene.spec, scene.traj1, traj2_late, scene.obs,
        m1=scene.m1, m2=scene.m2, t_cpa_rx_geom_1=t1, t_cpa_rx_geom_2=t2_late, cfg=cfg,
    )
    assert bad.score_1 == pytest.approx(good.score_1)
    assert bad.dt_s == pytest.approx(good.dt_s)
    # 15 s off at ~1.5 Hz/s through the CPA: the ridge is missed everywhere
    # except the flat asymptotes of the +-60 s window.
    assert bad.score_2 < 0.5 * good.score_2

    # Anchoring event 2 on its own measured time would hide the error: that
    # is exactly what the lock prevents.
    free = score_event(
        scene.spec, traj2_late, scene.obs,
        f0=scene.m2.f0_hz, t_cpa_rx_audio=scene.m2.t_cpa_rx, t_cpa_rx_geom=t2_late, cfg=cfg,
    )
    assert free.score > 4.5


def test_lock_dt_fixes_alignment(scene: Scene, cfg: ScoreConfig) -> None:
    _, _, t1 = t_cpa_rx_for(scene.traj1, scene.obs)
    locked = score_event(
        scene.spec, scene.traj1, scene.obs,
        f0=scene.m1.f0_hz, t_cpa_rx_audio=scene.m1.t_cpa_rx, t_cpa_rx_geom=t1, cfg=cfg, lock_dt=2.0,
    )
    assert isinstance(locked, EventScore)
    assert locked.dt_s == 2.0
    best = score_event(
        scene.spec, scene.traj1, scene.obs,
        f0=scene.m1.f0_hz, t_cpa_rx_audio=scene.m1.t_cpa_rx, t_cpa_rx_geom=t1, cfg=cfg,
    )
    # 3 s off the true alignment loses the ~20 s transition through the CPA;
    # the asymptotes of the +-60 s window still match, so the drop is partial.
    assert locked.score < 0.85 * best.score


def test_harmonic_weights_raise_score_when_harmonics_painted(scene: Scene) -> None:
    base = _pair_at(scene, scene.obs, ScoreConfig(harmonics=(1.0,)), spec=scene.spec_harm)
    harm = _pair_at(scene, scene.obs, ScoreConfig(harmonics=(1.0, 0.5, 0.3)), spec=scene.spec_harm)
    assert harm.score_1 > 1.2 * base.score_1
    assert harm.score_2 > 1.2 * base.score_2
    # Without harmonics painted, extra weights add nothing but background (0).
    plain = _pair_at(scene, scene.obs, ScoreConfig(harmonics=(1.0, 0.5, 0.3)))
    assert plain.score_1 == pytest.approx(base.score_1, rel=0.02)


def test_curve_outside_spectrogram_scores_zero(scene: Scene, cfg: ScoreConfig) -> None:
    _, _, t1 = t_cpa_rx_for(scene.traj1, scene.obs)
    es = score_event(
        scene.spec, scene.traj1, scene.obs,
        f0=scene.m1.f0_hz, t_cpa_rx_audio=5000.0, t_cpa_rx_geom=t1, cfg=cfg,
    )
    assert es.score == 0.0
    # Model invalid: curve centred far outside the track's span.
    es = score_event(
        scene.spec, scene.traj1, scene.obs,
        f0=scene.m1.f0_hz, t_cpa_rx_audio=scene.m1.t_cpa_rx, t_cpa_rx_geom=t1 + 2000.0, cfg=cfg,
    )
    assert es.score == 0.0


def test_score_candidates_table(scene: Scene, cfg: ScoreConfig) -> None:
    far_lat, far_lon = latlon_from_enu(120.0 + 300.0, -80.0, LAT0, LON0)
    rows = []
    for lat, lon in ((scene.obs_lat, scene.obs_lon), (float(far_lat), float(far_lon))):
        obs = observer_enu_for(lat, lon, scene.obs_elev, LAT0, LON0)
        tx1, s1, rx1 = t_cpa_rx_for(scene.traj1, obs)
        tx2, s2, rx2 = t_cpa_rx_for(scene.traj2, obs)
        rows.append(
            dict(
                row=0, col=0, lat=lat, lon=lon, elev_m=scene.obs_elev,
                flight_id_1="F1", t_cpa_tx_1=tx1, t_cpa_rx_1=rx1, cpa_slant_1=s1, speed_1=60.0,
                flight_id_2="F2", t_cpa_tx_2=tx2, t_cpa_rx_2=rx2, cpa_slant_2=s2, speed_2=55.0,
                offset_s=rx2 - rx1,
            )
        )
    pairs = pd.DataFrame(rows)
    out = score_candidates(
        scene.spec, pairs, {"F1": scene.traj1, "F2": scene.traj2}, LAT0, LON0, scene.m1, scene.m2, cfg,
    )
    for c in ("score_1", "score_2", "dt_s", "f0_1", "f0_2", "total"):
        assert c in out.columns
    assert len(out) == 2 and list(out.columns[: len(pairs.columns)]) == list(pairs.columns)
    assert out["total"].iloc[0] > out["total"].iloc[1]
    direct = _pair_at(scene, scene.obs, cfg)
    assert out["total"].iloc[0] == pytest.approx(direct.total)
    assert out["dt_s"].iloc[0] == pytest.approx(direct.dt_s)

    empty = score_candidates(
        scene.spec, pairs.iloc[:0], {"F1": scene.traj1, "F2": scene.traj2}, LAT0, LON0, scene.m1, scene.m2, cfg,
    )
    assert len(empty) == 0 and "total" in empty.columns


def _single_table(scene: Scene, offsets: list[tuple[float, float]], traj: Trajectory, fid: str) -> pd.DataFrame:
    rows = []
    for de, dn in offsets:
        lat, lon = latlon_from_enu(120.0 + de, -80.0 + dn, LAT0, LON0)
        obs = observer_enu_for(float(lat), float(lon), scene.obs_elev, LAT0, LON0)
        tx, s, rx = t_cpa_rx_for(traj, obs)
        rows.append(
            dict(row=0, col=0, lat=float(lat), lon=float(lon), elev_m=scene.obs_elev,
                 flight_id=fid, t_cpa_tx=tx, t_cpa_rx=rx, cpa_slant_m=s, speed_mps=60.0)
        )
    return pd.DataFrame(rows)


def test_score_single_default_anchor(scene: Scene, cfg: ScoreConfig) -> None:
    cands = _single_table(scene, [(0.0, 0.0), (300.0, 0.0)], scene.traj1, "F1")
    out = score_single(scene.spec, cands, {"F1": scene.traj1}, LAT0, LON0, scene.m1, cfg)
    for c in ("score", "dt_s", "f0", "total"):
        assert c in out.columns
    assert out["score"].iloc[0] > 4.5
    assert out["score"].iloc[0] > out["score"].iloc[1]
    assert out["dt_s"].iloc[0] == pytest.approx(-1.0, abs=cfg.time_step_s + 1e-9)
    assert out["f0"].iloc[0] == pytest.approx(F0_1, abs=cfg.f0_step_hz + 1e-9)


def test_score_single_known_time_gates_rows(scene: Scene, cfg: ScoreConfig) -> None:
    # Two candidate rows along the track direction: the true observer, and
    # one 6 km further east. The far one has a near-identical curve shape but
    # its geometry CPA is 100 s later, which the known time rules out.
    cands = _single_table(scene, [(0.0, 0.0), (6000.0, 0.0)], scene.traj1, "F1")
    assert cands["t_cpa_rx"].iloc[1] - cands["t_cpa_rx"].iloc[0] > 50.0
    # The known time is the true CPA time, with the measurement's +1.0 s bias
    # (m1.t_cpa_rx is 1 s late) expressed through the row's dt.
    known = scene.t_cpa_rx_1 + 1.0
    out = score_single(
        scene.spec, cands, {"F1": scene.traj1}, LAT0, LON0, scene.m1, cfg, t_known_rx_geom=known,
    )
    assert out["score"].iloc[0] > 4.5
    assert out["dt_s"].iloc[0] == pytest.approx(-1.0, abs=1e-6)
    assert out["score"].iloc[1] == 0.0
    assert abs(out["dt_s"].iloc[1]) > cfg.time_sweep_s

    # Without the known time, the far row re-anchors on its own CPA and scores
    # well on shape alone: the gate is what separates them.
    free = score_single(scene.spec, cands, {"F1": scene.traj1}, LAT0, LON0, scene.m1, cfg)
    assert free["score"].iloc[1] > 2.0


def test_score_pair_timing(scene: Scene, cfg: ScoreConfig) -> None:
    _pair_at(scene, scene.obs, cfg)  # warm up
    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        _pair_at(scene, scene.obs, cfg)
    per_call_ms = (time.perf_counter() - t0) / n * 1e3
    print(f"\nscore_pair: {per_call_ms:.1f} ms per candidate (cfg defaults, 2 events)")
    assert per_call_ms < 200.0
