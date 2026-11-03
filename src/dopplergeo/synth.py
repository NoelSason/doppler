"""Synthetic flights and audio for tests and demos.

The audio renderer is deliberately *not* built on the frequency model in
:mod:`dopplergeo.acoustics`. It renders the received waveform directly as
``s(tau(t))`` where ``tau(t)`` is the emission time of the sound arriving at
``t`` and ``s`` is the engine tone. The instantaneous received frequency of
that waveform is ``f0 * d tau / d t = f0 * c / (c + v_radial)``, so a search
that recovers the right location from this audio has been checked against an
independent formulation of the same physics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .acoustics import Trajectory, solve_emission_times
from .adsb.schema import normalize_tracks
from .constants import SPEED_OF_SOUND_MPS
from .geo import latlon_from_enu


def straight_track(
    flight_id: str,
    t0: float,
    duration_s: float,
    lat_start: float,
    lon_start: float,
    alt_m: float,
    speed_mps: float,
    heading_deg: float,
    *,
    step_s: float = 1.0,
    turn_rate_dps: float = 0.0,
    accel_mps2: float = 0.0,
    climb_mps: float = 0.0,
    icao24: str | None = None,
    typecode: str = "C172",
    callsign: str = "",
) -> pd.DataFrame:
    """A flight that may turn, accelerate and climb at constant rates.

    Returns rows in the canonical track schema, ``step_s`` apart.
    """
    n = int(np.floor(duration_s / step_s)) + 1
    t = t0 + np.arange(n) * step_s
    rel = t - t0
    speed = speed_mps + accel_mps2 * rel
    heading = np.deg2rad(heading_deg + turn_rate_dps * rel)
    # Integrate the ground path with the trapezoid rule.
    ve = speed * np.sin(heading)
    vn = speed * np.cos(heading)
    east = np.concatenate([[0.0], np.cumsum(0.5 * (ve[1:] + ve[:-1]) * step_s)])
    north = np.concatenate([[0.0], np.cumsum(0.5 * (vn[1:] + vn[:-1]) * step_s)])
    lat, lon = latlon_from_enu(east, north, lat_start, lon_start)
    return normalize_tracks(
        pd.DataFrame(
            {
                "flight_id": flight_id,
                "icao24": icao24 or flight_id.lower()[:6].ljust(6, "0"),
                "callsign": callsign,
                "typecode": typecode,
                "time": t,
                "lat": lat,
                "lon": lon,
                "alt_m": alt_m + climb_mps * rel,
                "alt_source": "geom",
                "gs_mps": speed,
                "track_deg": np.rad2deg(heading) % 360.0,
                "vrate_mps": climb_mps,
            }
        )
    )


@dataclass
class Source:
    """An aircraft tone to render: a trajectory plus its engine fundamental."""

    trajectory: Trajectory
    f0_hz: float
    amplitude: float = 1.0
    harmonics: tuple[float, ...] = field(default_factory=lambda: (1.0, 0.5, 0.3, 0.2))
    reference_range_m: float = 1000.0


def render_audio(
    sources: list[Source],
    observer_enu: np.ndarray,
    t_start: float,
    duration_s: float,
    sample_rate: int = 8000,
    noise_rms: float = 0.05,
    c: float = SPEED_OF_SOUND_MPS,
    seed: int = 0,
    coarse_step_s: float = 0.01,
) -> np.ndarray:
    """Render what a microphone at ``observer_enu`` records.

    Emission times are solved on a ``coarse_step_s`` grid and interpolated to
    every sample (``tau(t)`` is smooth, so the interpolation error is far
    below one cycle). Amplitude follows ``1/r`` relative to
    ``reference_range_m``; white Gaussian noise of ``noise_rms`` is added.
    """
    obs = np.asarray(observer_enu, dtype=float).reshape(3)
    n = int(round(duration_s * sample_rate))
    t = t_start + np.arange(n) / sample_rate
    out = np.zeros(n, dtype=np.float64)
    n_coarse = int(np.ceil(duration_s / coarse_step_s)) + 1
    t_coarse = t_start + np.arange(n_coarse) * coarse_step_s
    for src in sources:
        tau_c, valid_c = solve_emission_times(src.trajectory, obs, t_coarse, c=c)
        tau = np.interp(t, t_coarse, tau_c)
        valid = np.interp(t, t_coarse, valid_c.astype(float)) >= 0.999
        if not valid.any():
            continue
        r = np.linalg.norm(src.trajectory.position(tau) - obs, axis=-1)
        gain = src.amplitude * src.reference_range_m / np.maximum(r, 1.0)
        phase = 2 * np.pi * src.f0_hz * tau
        tone = np.zeros(n)
        for k, a in enumerate(src.harmonics, start=1):
            tone += a * np.sin(k * phase)
        out += np.where(valid, gain * tone, 0.0)
    rng = np.random.default_rng(seed)
    out += rng.normal(0.0, noise_rms, size=n)
    return out.astype(np.float32)
