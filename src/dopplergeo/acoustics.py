"""Doppler physics: the Doppler equation, the ideal straight-line curve, and
the retarded-time ("when was this sound emitted?") solver for a real track.

Sign convention used everywhere in this package:

* ``v_radial`` is the component of the source velocity along the line from the
  source to the observer, **positive when the source is receding**.
* A stationary observer then hears ``f = f0 * c / (c + v_radial)``: higher
  than ``f0`` while the aircraft approaches, equal to ``f0`` at the closest
  point of approach (CPA), lower afterwards.

Times are seconds; ``t_rx`` is the time a sound is *received* and ``t_tx``
the time it was *emitted*. They differ by the slant range over the speed of
sound, which for a 3 km range is about 9 s, and that delay is *not* constant
over a flyover. Modelling the curve at receive time with the emission-time
geometry is the whole point of this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from .constants import SPEED_OF_SOUND_MPS


def doppler_frequency(
    f0: float | np.ndarray,
    v_radial: np.ndarray | float,
    c: float = SPEED_OF_SOUND_MPS,
) -> np.ndarray:
    """Frequency heard by a stationary observer from a source emitting ``f0``."""
    return np.asarray(f0, dtype=float) * c / (c + np.asarray(v_radial, dtype=float))


# --------------------------------------------------------------------------
# Ideal straight-line, constant-speed curve (closed form)
# --------------------------------------------------------------------------


def ideal_emission_offset(
    t_rel: np.ndarray | float, v: float, d: float, c: float = SPEED_OF_SOUND_MPS
) -> np.ndarray:
    """Emission time (relative to the geometric CPA) for each receive time.

    ``t_rel`` is receive time minus the *emission* time of the CPA. With
    ``tau`` the emission offset, ``t_rel = tau + sqrt(d^2 + v^2 tau^2) / c``;
    that is a quadratic in ``tau`` whose causal root (``tau < t_rel``) is
    returned.
    """
    t_rel = np.asarray(t_rel, dtype=float)
    c2 = c * c
    v2 = v * v
    a = c2 - v2
    if a <= 0:
        raise ValueError("source speed must be below the speed of sound")
    disc = d * d * a + v2 * c2 * t_rel * t_rel
    return (c2 * t_rel - np.sqrt(disc)) / a


def ideal_curve(
    t_rx: np.ndarray | float,
    f0: float,
    v: float,
    d: float,
    t_cpa_rx: float,
    c: float = SPEED_OF_SOUND_MPS,
) -> np.ndarray:
    """Received frequency vs receive time for straight, constant-speed flight.

    Parameters
    ----------
    t_rx
        Receive times (s), in the same clock as ``t_cpa_rx``.
    f0
        Emitted frequency (Hz).
    v
        Aircraft speed (m/s), relative to the air the sound travels in.
    d
        Slant distance at the closest point of approach (m).
    t_cpa_rx
        Receive time at which the CPA is *heard*, i.e. when ``f == f0``.
        This is the geometric CPA time plus ``d / c``.
    """
    t_rel = np.asarray(t_rx, dtype=float) - (t_cpa_rx - d / c)
    tau = ideal_emission_offset(t_rel, v, d, c)
    r = np.sqrt(d * d + v * v * tau * tau)
    v_radial = v * v * tau / r
    return doppler_frequency(f0, v_radial, c)


# --------------------------------------------------------------------------
# Arbitrary track
# --------------------------------------------------------------------------


@dataclass
class Trajectory:
    """A smooth 3-D path ``p(t)`` in a local ENU frame (metres, seconds).

    Built from sampled points; positions and velocities come from a cubic
    spline so the modelled frequency is smooth. ``t`` must be strictly
    increasing. The frame origin is arbitrary; observers are expressed in the
    same frame.
    """

    t: np.ndarray
    east: np.ndarray
    north: np.ndarray
    up: np.ndarray

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float)
        self.east = np.asarray(self.east, dtype=float)
        self.north = np.asarray(self.north, dtype=float)
        self.up = np.asarray(self.up, dtype=float)
        if self.t.ndim != 1 or self.t.size < 2:
            raise ValueError("a trajectory needs at least two samples")
        if not np.all(np.diff(self.t) > 0):
            raise ValueError("trajectory times must be strictly increasing")
        pts = np.column_stack([self.east, self.north, self.up])
        # A natural cubic spline degrades gracefully to a straight line for
        # two samples, so one code path covers every track length.
        self._spline = CubicSpline(self.t, pts, axis=0, bc_type="natural")
        self._pts = pts

    @property
    def t_start(self) -> float:
        return float(self.t[0])

    @property
    def t_end(self) -> float:
        return float(self.t[-1])

    def position(self, t: np.ndarray | float) -> np.ndarray:
        """Shape (..., 3) ENU position; clamps to the ends outside the span."""
        tt = np.clip(np.asarray(t, dtype=float), self.t[0], self.t[-1])
        return self._spline(tt)

    def velocity(self, t: np.ndarray | float) -> np.ndarray:
        """Shape (..., 3) ENU velocity (m/s); zero outside the span."""
        tt = np.asarray(t, dtype=float)
        inside = (tt >= self.t[0]) & (tt <= self.t[-1])
        tc = np.clip(tt, self.t[0], self.t[-1])
        v = self._spline(tc, 1)
        return np.where(inside[..., None], v, 0.0)

    def speed(self, t: np.ndarray | float) -> np.ndarray:
        return np.linalg.norm(self.velocity(t), axis=-1)


def solve_emission_times(
    traj: Trajectory,
    observer_enu: np.ndarray,
    t_rx: np.ndarray,
    c: float = SPEED_OF_SOUND_MPS,
    max_iter: int = 50,
    tol_s: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """For each receive time find the emission time ``t_tx`` with
    ``t_tx + |p(t_tx) - observer| / c == t_rx``.

    Fixed-point iteration ``t_tx <- t_rx - r(t_tx)/c`` converges because the
    range changes slower than the speed of sound (contraction factor
    ``|v_radial| / c < 1``).

    Returns ``(t_tx, valid)`` where ``valid`` marks receive times whose
    emission time falls inside the trajectory's time span.
    """
    obs = np.asarray(observer_enu, dtype=float).reshape(3)
    t_rx = np.asarray(t_rx, dtype=float)
    t_tx = t_rx - np.linalg.norm(traj.position(t_rx) - obs, axis=-1) / c
    for _ in range(max_iter):
        r = np.linalg.norm(traj.position(t_tx) - obs, axis=-1)
        t_new = t_rx - r / c
        if np.max(np.abs(t_new - t_tx)) < tol_s:
            t_tx = t_new
            break
        t_tx = t_new
    valid = (t_tx >= traj.t_start) & (t_tx <= traj.t_end)
    return t_tx, valid


def radial_velocity(
    traj: Trajectory, observer_enu: np.ndarray, t_tx: np.ndarray
) -> np.ndarray:
    """Radial velocity at emission times (positive = receding from observer)."""
    obs = np.asarray(observer_enu, dtype=float).reshape(3)
    p = traj.position(t_tx)
    v = traj.velocity(t_tx)
    rel = p - obs
    r = np.linalg.norm(rel, axis=-1)
    r = np.where(r == 0, 1e-9, r)
    return np.einsum("...k,...k->...", rel, v) / r


def received_frequency(
    traj: Trajectory,
    observer_enu: np.ndarray,
    t_rx: np.ndarray,
    f0: float,
    c: float = SPEED_OF_SOUND_MPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Modelled received frequency at receive times for a real track.

    This is the correct pipeline: for each *receive* time, find the emission
    time, take the radial velocity *then*, and run it through the Doppler
    equation. Returns ``(freq_hz, valid)``.
    """
    t_tx, valid = solve_emission_times(traj, observer_enu, t_rx, c=c)
    vr = radial_velocity(traj, observer_enu, t_tx)
    return doppler_frequency(f0, vr, c), valid


def closest_point_of_approach(
    traj: Trajectory,
    observer_enu: np.ndarray,
    oversample_s: float = 0.25,
) -> tuple[float, float]:
    """``(t_cpa_tx, slant_m)``: emission time and range of closest approach.

    Scans the spline at ``oversample_s`` resolution; accurate to a fraction of
    a metre for aircraft speeds.
    """
    obs = np.asarray(observer_enu, dtype=float).reshape(3)
    n = int(np.ceil((traj.t_end - traj.t_start) / oversample_s)) + 1
    ts = np.linspace(traj.t_start, traj.t_end, max(n, 2))
    r = np.linalg.norm(traj.position(ts) - obs, axis=-1)
    k = int(np.argmin(r))
    return float(ts[k]), float(r[k])
