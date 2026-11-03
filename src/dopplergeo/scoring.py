"""Score candidate observer positions against a whitened spectrogram.

A candidate is scored by laying the *modelled* Doppler curve of a flight
(from its ADS-B trajectory and the candidate's position) onto the whitened
spectrogram and averaging the brightness under it. The score is maximised
over two nuisance parameters:

* ``dt`` – the residual audio/geometry clock alignment, swept ± ``time_sweep_s``;
* ``f0`` – the emitted fundamental, swept ± ``f0_sweep_hz`` around the fitted value.

Both sweeps are exact re-uses of one modelled curve: the Doppler ratio
``f / f0 = c / (c + v_radial)`` does not depend on ``f0``, so a new ``f0``
scales the frequency array; and ``dt`` shifts the audio times. The curve is
therefore modelled once per (candidate, event), and the spectrogram is
sampled once on an ``(n_harmonics, n_dt, n_f0, n_curve)`` grid, which is the
hot path.

Clock mapping used throughout::

    audio_time = (t_rx_geom - t_cpa_rx_geom) + t_cpa_rx_audio + dt

For a flight pair the second event re-uses the first event's ``dt`` and is
anchored on the *first* event's measured time, so a candidate whose two CPAs
are the wrong distance apart in time is penalised instead of silently
re-aligned.

Any object with ``.times``, ``.freqs`` and
``.sample(t, f) -> (values, inbounds)`` (bilinear, 0 outside) is accepted as
the spectrogram, and any object with ``.f0_hz`` and ``.t_cpa_rx`` as a
measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from .acoustics import Trajectory
from .constants import SPEED_OF_SOUND_MPS
from .model import curve_times, model_curve, observer_enu_for

# Fraction of the curve samples that must be usable (model valid and
# fundamental inside the spectrogram) for a cell to score at all.
MIN_INBOUNDS_FRACTION: float = 0.25


class SpectrogramLike(Protocol):
    times: np.ndarray
    freqs: np.ndarray

    def sample(self, t: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


class MeasurementLike(Protocol):
    f0_hz: float
    t_cpa_rx: float


@dataclass
class ScoreConfig:
    half_window_s: float = 60.0  # curve extent each side of CPA
    curve_step_s: float = 0.25
    time_sweep_s: float = 3.0  # ± sweep of the audio/geometry alignment
    time_step_s: float = 0.25
    f0_sweep_hz: float = 1.5  # ± sweep of the emitted frequency
    f0_step_hz: float = 0.25
    harmonics: tuple[float, ...] = (1.0, 0.5)  # weights for k = 1..K multiples of f0; propeller tones carry harmonics, and the second doubles the frequency resolution of a mismatch
    # Brightness is clipped at this many whitened sigmas before averaging, so
    # the score measures how much of the curve is lit rather than how bright
    # its brightest half is. A wrong flight with the right speed matches one
    # asymptote of the S and can be very bright there; clipping keeps that from
    # outscoring a curve that is faint but lit end to end. None disables it.
    clip_sigma: float | None = None


@dataclass
class EventScore:
    score: float
    dt_s: float
    f0_hz: float


@dataclass
class PairScore:
    score_1: float
    score_2: float
    dt_s: float
    f0_1: float
    f0_2: float
    total: float


def _sweep(center: float, half: float, step: float) -> np.ndarray:
    """Symmetric grid ``center ± half`` that always contains ``center``."""
    if half <= 0 or step <= 0:
        return np.array([float(center)], dtype=float)
    n = int(round(half / step))
    return float(center) + np.arange(-n, n + 1, dtype=float) * step


def score_event(
    spec_w: SpectrogramLike,
    traj: Trajectory,
    observer_enu: np.ndarray,
    *,
    f0: float,
    t_cpa_rx_audio: float,
    t_cpa_rx_geom: float,
    cfg: ScoreConfig,
    lock_dt: float | None = None,
    c: float = SPEED_OF_SOUND_MPS,
) -> EventScore:
    """Score one flyover of one flight at one observer position.

    Parameters
    ----------
    spec_w
        Whitened spectrogram (brightness >= 0, 0 outside its bounds).
    f0
        Measured fundamental (Hz); the sweep is centred here.
    t_cpa_rx_audio
        Audio-clock time the CPA was heard (from the curve fit).
    t_cpa_rx_geom
        Geometry-clock time the CPA is heard at this candidate.
    lock_dt
        If given, the alignment is fixed at this value and only ``f0`` is
        swept.

    Returns the best cell of the ``(dt, f0)`` grid. A cell scores 0 when
    fewer than :data:`MIN_INBOUNDS_FRACTION` of its curve samples are usable.
    """
    weights = np.asarray(cfg.harmonics, dtype=float)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("cfg.harmonics must be a non-empty tuple of weights")
    t_rel = curve_times(0.0, cfg.half_window_s, cfg.curve_step_s)
    n_curve = t_rel.size

    freq, valid = model_curve(traj, observer_enu, f0, t_cpa_rx_geom + t_rel, c=c)
    dts = np.array([float(lock_dt)]) if lock_dt is not None else _sweep(0.0, cfg.time_sweep_s, cfg.time_step_s)
    f0s = _sweep(f0, cfg.f0_sweep_hz, cfg.f0_step_hz)

    if valid.sum() < MIN_INBOUNDS_FRACTION * n_curve:
        return EventScore(score=0.0, dt_s=float(dts[0]), f0_hz=float(f0))

    # Grid axes: (harmonic k, dt i, f0 j, curve sample l).
    ks = np.arange(1, weights.size + 1, dtype=float)
    t_grid = t_rel[None, None, None, :] + t_cpa_rx_audio + dts[None, :, None, None]
    f_grid = (freq / f0)[None, None, None, :] * f0s[None, None, :, None] * ks[:, None, None, None]
    shape = np.broadcast_shapes(t_grid.shape, f_grid.shape)
    t_flat = np.broadcast_to(t_grid, shape).ravel()
    f_flat = np.broadcast_to(f_grid, shape).ravel()

    values, inbounds = spec_w.sample(t_flat, f_flat)
    values = np.asarray(values, dtype=float).reshape(shape)
    if cfg.clip_sigma is not None:
        values = np.minimum(values, cfg.clip_sigma)
    inbounds = np.asarray(inbounds, dtype=bool).reshape(shape)

    # Count a curve sample only where the model is valid and the fundamental
    # is inside the spectrogram; harmonics that fall outside contribute 0.
    usable = inbounds[0] & valid[None, None, :]  # (n_dt, n_f0, n_curve)
    count = usable.sum(axis=-1)  # (n_dt, n_f0)
    summed = np.einsum("k,kijl->ij", weights, values * usable[None])
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.where(count > 0, summed / np.maximum(count, 1), 0.0)
    score = np.where(count >= MIN_INBOUNDS_FRACTION * n_curve, score, 0.0)

    i, j = np.unravel_index(int(np.argmax(score)), score.shape)
    return EventScore(score=float(score[i, j]), dt_s=float(dts[i]), f0_hz=float(f0s[j]))


def score_pair(
    spec_w: SpectrogramLike,
    traj1: Trajectory,
    traj2: Trajectory,
    observer_enu: np.ndarray,
    *,
    m1: MeasurementLike,
    m2: MeasurementLike,
    t_cpa_rx_geom_1: float,
    t_cpa_rx_geom_2: float,
    cfg: ScoreConfig,
    c: float = SPEED_OF_SOUND_MPS,
) -> PairScore:
    """Score a candidate for two flyovers.

    Event 1 picks ``dt`` (and its own ``f0``). Event 2 is placed with that
    same ``dt`` and anchored on event 1's measured time::

        audio_time_2 = (t_rx_geom_2 - t_cpa_rx_geom_1) + m1.t_cpa_rx + dt

    ``m2.t_cpa_rx`` is deliberately unused, so a candidate whose two CPAs are
    the wrong time apart cannot re-align event 2 on its own measurement.
    """
    e1 = score_event(
        spec_w,
        traj1,
        observer_enu,
        f0=m1.f0_hz,
        t_cpa_rx_audio=m1.t_cpa_rx,
        t_cpa_rx_geom=t_cpa_rx_geom_1,
        cfg=cfg,
        c=c,
    )
    # With t_cpa_rx_audio set this way, score_event's mapping
    # (t_rx_geom - t_cpa_rx_geom_2) + t_cpa_rx_audio + dt reduces to the
    # event-1-anchored expression above.
    e2 = score_event(
        spec_w,
        traj2,
        observer_enu,
        f0=m2.f0_hz,
        t_cpa_rx_audio=(t_cpa_rx_geom_2 - t_cpa_rx_geom_1) + m1.t_cpa_rx,
        t_cpa_rx_geom=t_cpa_rx_geom_2,
        cfg=cfg,
        lock_dt=e1.dt_s,
        c=c,
    )
    return PairScore(
        score_1=e1.score,
        score_2=e2.score,
        dt_s=e1.dt_s,
        f0_1=e1.f0_hz,
        f0_2=e2.f0_hz,
        total=e1.score + e2.score,
    )


PAIR_SCORE_COLUMNS: tuple[str, ...] = ("score_1", "score_2", "dt_s", "f0_1", "f0_2", "total")
SINGLE_SCORE_COLUMNS: tuple[str, ...] = ("score", "dt_s", "f0", "total")


def score_candidates(
    spec_w: SpectrogramLike,
    pairs: pd.DataFrame,
    trajectories: dict[str, Trajectory],
    lat0: float,
    lon0: float,
    m1: MeasurementLike,
    m2: MeasurementLike,
    cfg: ScoreConfig,
    c: float = SPEED_OF_SOUND_MPS,
) -> pd.DataFrame:
    """Score every row of a ``pair_by_offset`` table.

    ``trajectories`` must be keyed by ``flight_id`` and built in the
    ``(lat0, lon0)`` ENU frame. Uses the table's ``t_cpa_rx_1``/``t_cpa_rx_2``
    as the geometry-clock CPA times. Returns ``pairs`` plus
    ``score_1, score_2, dt_s, f0_1, f0_2, total``.
    """
    n = len(pairs)
    cols: dict[str, np.ndarray] = {name: np.zeros(n, dtype=float) for name in PAIR_SCORE_COLUMNS}
    lat = pairs["lat"].to_numpy(dtype=float)
    lon = pairs["lon"].to_numpy(dtype=float)
    elev = pairs["elev_m"].to_numpy(dtype=float)
    fid1 = pairs["flight_id_1"].to_numpy()
    fid2 = pairs["flight_id_2"].to_numpy()
    t1 = pairs["t_cpa_rx_1"].to_numpy(dtype=float)
    t2 = pairs["t_cpa_rx_2"].to_numpy(dtype=float)
    for i in range(n):
        obs = observer_enu_for(lat[i], lon[i], elev[i], lat0, lon0)
        ps = score_pair(
            spec_w,
            trajectories[str(fid1[i])],
            trajectories[str(fid2[i])],
            obs,
            m1=m1,
            m2=m2,
            t_cpa_rx_geom_1=t1[i],
            t_cpa_rx_geom_2=t2[i],
            cfg=cfg,
            c=c,
        )
        cols["score_1"][i] = ps.score_1
        cols["score_2"][i] = ps.score_2
        cols["dt_s"][i] = ps.dt_s
        cols["f0_1"][i] = ps.f0_1
        cols["f0_2"][i] = ps.f0_2
        cols["total"][i] = ps.total
    out = pairs.copy()
    for name in PAIR_SCORE_COLUMNS:
        out[name] = cols[name]
    return out


def score_single(
    spec_w: SpectrogramLike,
    cands: pd.DataFrame,
    trajectories: dict[str, Trajectory],
    lat0: float,
    lon0: float,
    m: MeasurementLike,
    cfg: ScoreConfig,
    *,
    t_known_rx_geom: float | None = None,
    c: float = SPEED_OF_SOUND_MPS,
) -> pd.DataFrame:
    """Score every row of a ``generate_candidates`` table for one flyover.

    By default each row is anchored on its own CPA (``t_cpa_rx`` ↔
    ``m.t_cpa_rx``) and ``dt`` is swept. If ``t_known_rx_geom`` (the known
    geometry-clock time the CPA was heard, with ``cfg.time_sweep_s`` covering
    its uncertainty) is given, the anchor becomes ``t_known_rx_geom`` ↔
    ``m.t_cpa_rx`` instead: ``dt`` is then fixed at
    ``row.t_cpa_rx - t_known_rx_geom`` and rows outside the sweep score 0, so
    only candidates whose geometry CPA lands near the known time can score.

    Returns ``cands`` plus ``score, dt_s, f0, total`` (``total`` equals
    ``score``; it is there so single- and pair-mode tables rank on one name).
    """
    n = len(cands)
    cols: dict[str, np.ndarray] = {name: np.zeros(n, dtype=float) for name in SINGLE_SCORE_COLUMNS}
    lat = cands["lat"].to_numpy(dtype=float)
    lon = cands["lon"].to_numpy(dtype=float)
    elev = cands["elev_m"].to_numpy(dtype=float)
    fid = cands["flight_id"].to_numpy()
    t_cpa = cands["t_cpa_rx"].to_numpy(dtype=float)
    for i in range(n):
        lock: float | None = None
        if t_known_rx_geom is not None:
            lock = float(t_cpa[i] - t_known_rx_geom)
            if abs(lock) > cfg.time_sweep_s:
                cols["dt_s"][i] = lock
                cols["f0"][i] = float(m.f0_hz)
                continue
        obs = observer_enu_for(lat[i], lon[i], elev[i], lat0, lon0)
        es = score_event(
            spec_w,
            trajectories[str(fid[i])],
            obs,
            f0=m.f0_hz,
            t_cpa_rx_audio=m.t_cpa_rx,
            t_cpa_rx_geom=t_cpa[i],
            cfg=cfg,
            lock_dt=lock,
            c=c,
        )
        cols["score"][i] = es.score
        cols["dt_s"][i] = es.dt_s
        cols["f0"][i] = es.f0_hz
        cols["total"][i] = es.score
    out = cands.copy()
    for name in SINGLE_SCORE_COLUMNS:
        out[name] = cols[name]
    return out
