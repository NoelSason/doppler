"""Thin geometry-to-curve layer between the ADS-B trajectories and the scorer.

Everything here is a small, named wrapper around :mod:`dopplergeo.acoustics`
and :mod:`dopplergeo.geo` so that the scorer reads as "build the observer,
find the CPA, model the curve" without repeating frame conventions.

Clocks: trajectories and ``t_rx`` are in the geometry clock (unix seconds).
Mapping to the audio clock happens in :mod:`dopplergeo.scoring`, not here.
"""

from __future__ import annotations

import numpy as np

from .acoustics import Trajectory, closest_point_of_approach, received_frequency
from .constants import SPEED_OF_SOUND_MPS
from .geo import enu_from_latlon


def curve_times(t_center: float, half_window_s: float, step_s: float) -> np.ndarray:
    """Receive times ``t_center ± half_window_s`` on a ``step_s`` grid.

    The grid is symmetric and always contains ``t_center`` itself, so the
    sample at the CPA is never lost to floating-point rounding of the span.
    """
    if step_s <= 0:
        raise ValueError("step_s must be positive")
    n = int(round(half_window_s / step_s))
    return float(t_center) + np.arange(-n, n + 1, dtype=float) * step_s


def observer_enu_for(
    lat: float, lon: float, elev_m: float, lat0: float, lon0: float
) -> np.ndarray:
    """ENU position of an observer at ground elevation ``elev_m`` (m AMSL).

    The frame must be the same ``(lat0, lon0)`` the trajectories were built
    in with :func:`dopplergeo.adsb.schema.trajectory_from_rows`; ``up`` is
    altitude above mean sea level in both, so the slant leg is their
    difference.
    """
    east, north = enu_from_latlon(lat, lon, lat0, lon0)
    return np.array([float(east), float(north), float(elev_m)], dtype=float)


def model_curve(
    traj: Trajectory,
    observer_enu: np.ndarray,
    f0: float,
    t_rx: np.ndarray,
    c: float = SPEED_OF_SOUND_MPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Modelled received frequency at receive times ``t_rx`` (geometry clock).

    Returns ``(freq_hz, valid)`` where ``valid`` marks receive times whose
    emission time lies inside the track's span.
    """
    return received_frequency(traj, observer_enu, np.asarray(t_rx, dtype=float), f0, c=c)


def t_cpa_rx_for(
    traj: Trajectory,
    observer_enu: np.ndarray,
    c: float = SPEED_OF_SOUND_MPS,
) -> tuple[float, float, float]:
    """``(t_cpa_tx, slant_m, t_cpa_rx)`` for one trajectory and observer.

    ``t_cpa_rx = t_cpa_tx + slant_m / c`` is when the observer hears the
    closest point of approach, i.e. when the received frequency crosses
    ``f0``.
    """
    t_cpa_tx, slant = closest_point_of_approach(traj, observer_enu)
    return t_cpa_tx, slant, t_cpa_tx + slant / c
