"""Fit the ideal straight-line Doppler curve to an extracted ridge.

The fit recovers four numbers from a tone's frequency track: the emitted
frequency ``f0``, the speed ``v``, the slant distance at closest approach
``d`` and the audio-clock time ``t_cpa_rx`` at which the CPA is heard.

Why a multi-start converges: the total frequency ratio ``f_max / f_min``
equals ``(c + v) / (c - v)``, which pins ``v`` independently of ``f0``; given
``v``, the steepness of the drop through ``f0`` pins ``d``. So the surface
has one broad basin once the start is on the right side of the CPA, and a
handful of starts in speed and distance find it.

Ranges are deliberately generous: ``fit +- max(min halfwidth, sigma_mult *
std)``, with the std taken from the Jacobian covariance of the least-squares
solution. A wrong range silently drops the true answer downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.optimize import least_squares

from .acoustics import ideal_curve
from .constants import SPEED_OF_SOUND_MPS
from .spectrogram import Ridge

_MIN_POINTS = 5
_START_SPEEDS_MPS = (30.0, 55.0, 85.0, 130.0, 200.0)
_START_SLANTS_M = (300.0, 1000.0, 3000.0)
_SINGULAR_COND = 1e12


@dataclass
class CurveMeasurement:
    """Result of :func:`fit_ideal_curve`; all times are the audio clock."""

    f0_hz: float
    speed_mps: float
    cpa_slant_m: float
    t_cpa_rx: float
    rmse_hz: float
    speed_range: tuple[float, float]
    cpa_slant_range: tuple[float, float]
    t_cpa_range: tuple[float, float]
    f0_range: tuple[float, float]

    def to_dict(self) -> dict:
        return {
            "f0_hz": float(self.f0_hz),
            "speed_mps": float(self.speed_mps),
            "cpa_slant_m": float(self.cpa_slant_m),
            "t_cpa_rx": float(self.t_cpa_rx),
            "rmse_hz": float(self.rmse_hz),
            "speed_range": [float(x) for x in self.speed_range],
            "cpa_slant_range": [float(x) for x in self.cpa_slant_range],
            "t_cpa_range": [float(x) for x in self.t_cpa_range],
            "f0_range": [float(x) for x in self.f0_range],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CurveMeasurement":
        def pair(key: str) -> tuple[float, float]:
            lo, hi = d[key]
            return (float(lo), float(hi))

        return cls(
            f0_hz=float(d["f0_hz"]),
            speed_mps=float(d["speed_mps"]),
            cpa_slant_m=float(d["cpa_slant_m"]),
            t_cpa_rx=float(d["t_cpa_rx"]),
            rmse_hz=float(d["rmse_hz"]),
            speed_range=pair("speed_range"),
            cpa_slant_range=pair("cpa_slant_range"),
            t_cpa_range=pair("t_cpa_range"),
            f0_range=pair("f0_range"),
        )


def _smooth(y: np.ndarray, width: int) -> np.ndarray:
    """Centred moving average with edge padding."""
    width = max(1, min(width, y.size))
    if width == 1:
        return y.copy()
    pad = width // 2
    yp = np.pad(y, pad, mode="edge")
    kernel = np.ones(width) / width
    return np.convolve(yp, kernel, mode="valid")[: y.size]


def _t_cpa_candidates(t: np.ndarray, f: np.ndarray, f0_guess: float) -> list[float]:
    """Starting times for the heard CPA.

    One is where the smoothed ridge crosses the median frequency going
    down (the curve passes through ``f0`` at the CPA); the other is the time
    of steepest descent, which is the same moment for the ideal curve. Both
    are kept because a ridge that covers only one side of the flyover may
    lack a crossing.
    """
    dt = float(np.median(np.diff(t))) if t.size > 1 else 1.0
    width = int(round(2.0 / dt)) | 1  # about 2 s, odd
    fs = _smooth(f, width)
    cands: list[float] = []
    above = fs > f0_guess
    cross = np.flatnonzero(above[:-1] & ~above[1:])
    if cross.size:
        k = int(cross[np.argmin(np.abs(cross - f.size / 2))])
        # Linear interpolation of the crossing between samples k and k + 1.
        y0, y1 = fs[k] - f0_guess, fs[k + 1] - f0_guess
        frac = y0 / (y0 - y1) if y0 != y1 else 0.5
        cands.append(float(t[k] + frac * (t[k + 1] - t[k])))
    if t.size > 2:
        slope = np.gradient(fs, t)
        cands.append(float(t[int(np.argmin(slope))]))
    if not cands:
        cands.append(float(np.mean(t)))
    return cands


def _speed_from_ratio(f: np.ndarray, c: float) -> float:
    """Speed implied by the ridge's frequency extremes; a lower bound when the
    ridge does not reach both asymptotes."""
    hi, lo = np.percentile(f, [97.0, 3.0])
    if lo <= 0 or hi <= lo:
        return 0.0
    r = hi / lo
    return float(c * (r - 1.0) / (r + 1.0))


def fit_ideal_curve(
    ridge: Ridge,
    *,
    c: float = SPEED_OF_SOUND_MPS,
    speed_bounds: tuple[float, float] = (15.0, 250.0),
    slant_bounds: tuple[float, float] = (50.0, 10_000.0),
    min_speed_halfwidth: float = 15.0,
    min_slant_halfwidth: float = 300.0,
    min_time_halfwidth: float = 3.0,
    min_f0_halfwidth: float = 2.0,
    sigma_mult: float = 3.0,
    weight_by_strength: bool = True,
) -> CurveMeasurement:
    """Multi-start least-squares fit of :func:`acoustics.ideal_curve`.

    Parameters are ``(f0, v, d, t_cpa_rx)``. Starts are the product of
    several speeds and slant distances inside the bounds (plus the speed
    implied by the ridge's frequency ratio) with ``f0`` at the median ridge
    frequency and ``t_cpa_rx`` at the median crossing or the steepest
    descent. With ``weight_by_strength`` each residual is scaled by the
    square root of the ridge's whitened brightness, so frames where the tone
    has faded into the background count for little. ``rmse_hz`` is the
    weighted RMS residual with those weights normalised to unit mean square.
    """
    t = np.asarray(ridge.times, dtype=np.float64)
    f = np.asarray(ridge.freqs, dtype=np.float64)
    s = np.asarray(ridge.strength, dtype=np.float64)
    ok = np.isfinite(t) & np.isfinite(f) & np.isfinite(s)
    t, f, s = t[ok], f[ok], s[ok]
    if t.size < _MIN_POINTS:
        raise ValueError(f"ridge has {t.size} usable points; need at least {_MIN_POINTS}")

    if weight_by_strength:
        w = np.sqrt(np.clip(s, 0.0, None))
        if not np.any(w > 0):
            w = np.ones_like(f)
        w = w / np.sqrt(np.mean(w * w))
    else:
        w = np.ones_like(f)

    f0_guess = float(np.median(f))
    f_lo, f_hi = float(f.min()), float(f.max())
    f_span = max(f_hi - f_lo, 1.0)
    t_lo, t_hi = float(t.min()), float(t.max())
    t_span = max(t_hi - t_lo, 1.0)

    v_lo, v_hi = float(speed_bounds[0]), float(min(speed_bounds[1], 0.95 * c))
    d_lo, d_hi = float(slant_bounds[0]), float(slant_bounds[1])
    lower = np.array([max(f_lo - f_span, 0.5 * f_lo), v_lo, d_lo, t_lo - t_span])
    upper = np.array([f_hi + f_span, v_hi, d_hi, t_hi + t_span])

    def residuals(p: np.ndarray) -> np.ndarray:
        f0, v, d, tc = p
        return w * (ideal_curve(t, f0, v, d, tc, c) - f)

    speeds = sorted({float(np.clip(v, v_lo, v_hi)) for v in _START_SPEEDS_MPS}
                    | {float(np.clip(_speed_from_ratio(f, c), v_lo, v_hi))})
    slants = sorted({float(np.clip(d, d_lo, d_hi)) for d in _START_SLANTS_M})
    t_starts = _t_cpa_candidates(t, f, f0_guess)
    f0_start = float(np.clip(f0_guess, lower[0], upper[0]))

    best = None
    for v0, d0, tc0 in product(speeds, slants, t_starts):
        x0 = np.array([f0_start, v0, d0, float(np.clip(tc0, lower[3], upper[3]))])
        try:
            res = least_squares(
                residuals, x0, bounds=(lower, upper), method="trf",
                x_scale=np.array([1.0, 10.0, 100.0, 1.0]), max_nfev=400,
            )
        except (ValueError, FloatingPointError):
            continue
        if not np.all(np.isfinite(res.fun)):
            continue
        if best is None or res.cost < best.cost:
            best = res
    if best is None:
        raise RuntimeError("every start of the curve fit failed")

    f0, v, d, tc = (float(x) for x in best.x)
    n, n_par = best.fun.size, best.x.size
    dof = max(n - n_par, 1)
    rmse = float(np.sqrt(2.0 * best.cost / n))
    s2 = 2.0 * best.cost / dof

    std = np.full(n_par, np.inf)
    jtj = best.jac.T @ best.jac
    if np.all(np.isfinite(jtj)) and np.linalg.cond(jtj) < _SINGULAR_COND:
        cov = np.linalg.inv(jtj) * s2
        var = np.diag(cov)
        std = np.where(var > 0, np.sqrt(np.abs(var)), 0.0)
    halfwidths = np.maximum(
        [min_f0_halfwidth, min_speed_halfwidth, min_slant_halfwidth, min_time_halfwidth],
        sigma_mult * std,
    )
    # A singular or non-finite covariance falls back to the minimum halfwidths.
    halfwidths = np.where(np.isfinite(halfwidths), halfwidths,
                          [min_f0_halfwidth, min_speed_halfwidth, min_slant_halfwidth,
                           min_time_halfwidth])
    hf, hv, hd, ht = (float(h) for h in halfwidths)

    return CurveMeasurement(
        f0_hz=f0,
        speed_mps=v,
        cpa_slant_m=d,
        t_cpa_rx=tc,
        rmse_hz=rmse,
        speed_range=(max(0.0, v - hv), v + hv),
        cpa_slant_range=(max(0.0, d - hd), d + hd),
        t_cpa_range=(tc - ht, tc + ht),
        f0_range=(max(0.0, f0 - hf), f0 + hf),
    )
