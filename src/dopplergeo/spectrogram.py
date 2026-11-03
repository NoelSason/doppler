"""Audio loading, a narrow-band spectrogram, and ridge (tone) extraction.

The spectrogram is tuned for propeller tones below a few hundred Hz: a long
Hann window (2 s at the default 0.5 Hz resolution) zero-padded so that the
bin spacing is at most ``freq_resolution_hz``. ``Spectrogram.sample`` is the
inner loop of the scoring stage, so it is plain numpy fancy indexing over a
uniform (time, frequency) grid and nothing else.

Times are the audio clock: seconds from the first sample of the file.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import resample_poly
from scipy.signal.windows import hann

_POWER_EPS = 1e-12
_SCALE_EPS = 1e-6
_FRAMES_PER_CHUNK = 256
_REFINE_BINS = 2


def load_audio(path: str | Path, target_sr: int = 8000) -> tuple[np.ndarray, int]:
    """Read any soundfile-supported file as mono float64 at ``target_sr``.

    Channels are averaged; resampling uses a polyphase filter.
    """
    data, sr = sf.read(str(path), dtype="float64", always_2d=True)
    mono = data.mean(axis=1)
    sr = int(sr)
    if sr != target_sr:
        g = gcd(sr, target_sr)
        mono = resample_poly(mono, target_sr // g, sr // g)
    return np.ascontiguousarray(mono, dtype=np.float64), int(target_sr)


@dataclass
class Spectrogram:
    """Log-power spectrogram on a uniform time/frequency grid.

    ``power_db`` has shape ``(n_freq, n_time)``; ``times`` are frame centres
    in seconds from the start of the audio; ``freqs`` are bin centres in Hz.
    Both axes must be uniformly spaced (``compute_spectrogram`` guarantees it).
    """

    times: np.ndarray
    freqs: np.ndarray
    power_db: np.ndarray

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=np.float64)
        self.freqs = np.asarray(self.freqs, dtype=np.float64)
        self.power_db = np.asarray(self.power_db, dtype=np.float64)
        if self.power_db.shape != (self.freqs.size, self.times.size):
            raise ValueError(
                f"power_db shape {self.power_db.shape} does not match "
                f"(n_freq={self.freqs.size}, n_time={self.times.size})"
            )

    @property
    def dt(self) -> float:
        """Frame spacing (s)."""
        return float(self.times[1] - self.times[0]) if self.times.size > 1 else 1.0

    @property
    def df(self) -> float:
        """Bin spacing (Hz)."""
        return float(self.freqs[1] - self.freqs[0]) if self.freqs.size > 1 else 1.0

    def whitened(self) -> "Spectrogram":
        """Per-bin robust standardisation, clipped at zero.

        Each frequency bin is centred on its median over time and scaled by
        ``1.4826 * MAD`` (the standard deviation of a Gaussian with that MAD).
        Values below the background are clipped to 0 so a curve only earns
        credit from energy above the noise floor.
        """
        p = self.power_db
        med = np.median(p, axis=1, keepdims=True)
        mad = np.median(np.abs(p - med), axis=1, keepdims=True)
        scale = np.maximum(1.4826 * mad, _SCALE_EPS)
        z = (p - med) / scale
        np.maximum(z, 0.0, out=z)
        return Spectrogram(self.times.copy(), self.freqs.copy(), z)

    def sample(
        self, t: np.ndarray | float, f: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bilinear interpolation of ``power_db`` at arbitrary ``(t, f)``.

        ``t`` and ``f`` are broadcast against each other. Returns
        ``(values, inbounds)`` with the same broadcast shape; ``values`` is 0
        wherever the query falls outside the grid.
        """
        t_arr, f_arr = np.broadcast_arrays(
            np.asarray(t, dtype=np.float64), np.asarray(f, dtype=np.float64)
        )
        nf, nt = self.power_db.shape
        it = (t_arr - self.times[0]) / self.dt
        jf = (f_arr - self.freqs[0]) / self.df
        inb = (it >= 0.0) & (it <= nt - 1) & (jf >= 0.0) & (jf <= nf - 1)
        # Park out-of-bounds queries on cell (0, 0); their values are zeroed below.
        it = np.where(inb, it, 0.0)
        jf = np.where(inb, jf, 0.0)
        i0 = it.astype(np.intp)  # floor: it >= 0
        j0 = jf.astype(np.intp)
        i1 = np.minimum(i0 + 1, nt - 1)
        j1 = np.minimum(j0 + 1, nf - 1)
        wt = it - i0  # 0 at the last frame, where i1 == i0
        wf = jf - j0
        p = self.power_db
        vals = (
            p[j0, i0] * ((1.0 - wt) * (1.0 - wf))
            + p[j0, i1] * (wt * (1.0 - wf))
            + p[j1, i0] * ((1.0 - wt) * wf)
            + p[j1, i1] * (wt * wf)
        )
        vals *= inb
        return vals, inb


def compute_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    *,
    f_max: float = 600.0,
    freq_resolution_hz: float = 0.5,
    hop_s: float = 0.25,
) -> Spectrogram:
    """Hann-windowed, zero-padded STFT of ``audio`` in dB.

    The window is ``1 / freq_resolution_hz`` seconds long (2 s by default)
    so a bin that narrow carries real information; ``nfft`` is the next power
    of two at or above ``sample_rate / freq_resolution_hz`` so the bin spacing
    is at most ``freq_resolution_hz``. Frames are centred on
    ``k * hop`` and the audio is zero-padded at both ends, so ``times[0] == 0``.
    Magnitudes are scaled so a full-scale sine reads 0 dB.
    """
    x = np.asarray(audio, dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError("audio is empty")
    sr = int(sample_rate)
    nperseg = max(2, int(round(sr / freq_resolution_hz)))
    nfft = 1 << int(np.ceil(np.log2(max(nperseg, sr / freq_resolution_hz))))
    hop = max(1, int(round(hop_s * sr)))

    window = hann(nperseg, sym=False)
    scale = 2.0 / window.sum()
    freqs_all = np.fft.rfftfreq(nfft, d=1.0 / sr)
    n_keep = int(np.searchsorted(freqs_all, f_max, side="right"))
    n_keep = max(n_keep, 1)
    freqs = freqs_all[:n_keep].copy()

    n_frames = x.size // hop + 1
    pad_front = nperseg // 2
    needed = (n_frames - 1) * hop + nperseg
    pad_back = max(0, needed - pad_front - x.size)
    xp = np.pad(x, (pad_front, pad_back))
    frames = sliding_window_view(xp, nperseg)[::hop][:n_frames]

    power_db = np.empty((n_keep, n_frames), dtype=np.float64)
    for k0 in range(0, n_frames, _FRAMES_PER_CHUNK):
        chunk = frames[k0 : k0 + _FRAMES_PER_CHUNK] * window
        spec = np.fft.rfft(chunk, n=nfft, axis=1)[:, :n_keep] * scale
        power_db[:, k0 : k0 + chunk.shape[0]] = 10.0 * np.log10(
            spec.real**2 + spec.imag**2 + _POWER_EPS
        ).T

    times = np.arange(n_frames, dtype=np.float64) * (hop / sr)
    return Spectrogram(times, freqs, power_db)


@dataclass
class Ridge:
    """A tone's frequency track: one sample per spectrogram frame.

    ``strength`` is the whitened brightness along the path (0 = background).
    """

    times: np.ndarray
    freqs: np.ndarray
    strength: np.ndarray


def extract_ridge(
    spec: Spectrogram,
    t_start: float,
    t_end: float,
    f_min: float,
    f_max: float,
    *,
    max_slope_hz_per_s: float = 3.0,
    already_whitened: bool = False,
) -> Ridge:
    """Brightest smooth frequency path through a (time, frequency) box.

    Viterbi over the whitened power: the path maximises the sum of brightness
    along it, minus a quadratic penalty that reaches one whitened sigma at
    the maximum allowed jump, ``max_slope_hz_per_s * hop``. Jumps beyond that
    are not allowed. The peak is refined to sub-bin precision with a
    power-weighted centroid over the neighbouring bins.

    Pass ``already_whitened=True`` to skip the whitening when ``spec`` came
    from :meth:`Spectrogram.whitened`.
    """
    spec_w = spec if already_whitened else spec.whitened()
    i0 = int(np.searchsorted(spec_w.times, t_start, side="left"))
    i1 = int(np.searchsorted(spec_w.times, t_end, side="right"))
    j0 = int(np.searchsorted(spec_w.freqs, f_min, side="left"))
    j1 = int(np.searchsorted(spec_w.freqs, f_max, side="right"))
    if i1 <= i0:
        raise ValueError(f"no spectrogram frames in [{t_start}, {t_end}] s")
    if j1 <= j0:
        raise ValueError(f"no spectrogram bins in [{f_min}, {f_max}] Hz")
    W = spec_w.power_db[j0:j1, i0:i1]
    nf, nt = W.shape
    df = spec_w.df
    hop = spec_w.dt

    k_max = max(1, int(np.ceil(max_slope_hz_per_s * hop / df)))
    k_max = min(k_max, nf - 1) if nf > 1 else 0
    shifts = np.arange(-k_max, k_max + 1)
    penalty = (shifts / max(k_max, 1)) ** 2
    bins = np.arange(nf)

    score = W[:, 0].copy()
    back = np.zeros((nf, nt), dtype=np.intp)
    best = np.empty(nf)
    arg = np.empty(nf, dtype=np.intp)
    for t in range(1, nt):
        best.fill(-np.inf)
        arg.fill(0)
        for s, pen in zip(shifts, penalty):
            # Candidate predecessor for bin j is bin j - s.
            if s >= 0:
                cand = score[: nf - s] - pen
                sl = slice(s, nf)
            else:
                cand = score[-s:] - pen
                sl = slice(0, nf + s)
            better = cand > best[sl]
            best[sl] = np.where(better, cand, best[sl])
            arg[sl] = np.where(better, bins[sl] - s, arg[sl])
        score = best + W[:, t]
        back[:, t] = arg

    path = np.empty(nt, dtype=np.intp)
    path[-1] = int(np.argmax(score))
    for t in range(nt - 1, 0, -1):
        path[t - 1] = back[path[t], t]

    frames = np.arange(nt)
    centre = W[path, frames]

    # Sub-bin refinement: power-weighted centroid over +-_REFINE_BINS bins.
    # A 2 s window smears the chirp near CPA into a plateau several bins
    # wide, where a parabola through three bins is unreliable; the centroid
    # of a symmetric smear is unbiased. Linear power from the raw spectrogram
    # is best (whitening rescales each bin differently); whitened brightness
    # is the fallback when that is all we have.
    offs = np.arange(-_REFINE_BINS, _REFINE_BINS + 1)
    jj = np.clip((j0 + path)[:, None] + offs[None, :], 0, spec_w.freqs.size - 1)
    tt = (i0 + frames)[:, None]
    if already_whitened:
        wgt = spec_w.power_db[jj, tt]
    else:
        lin = np.power(10.0, spec.power_db[jj, tt] / 10.0)
        wgt = lin - lin.min(axis=1, keepdims=True)
    total = wgt.sum(axis=1)
    delta = np.where(total > 0, (wgt * offs).sum(axis=1) / np.where(total > 0, total, 1.0), 0.0)
    delta = np.clip(delta, -_REFINE_BINS, _REFINE_BINS)

    freqs = spec_w.freqs[j0 + path] + delta * df
    return Ridge(spec_w.times[i0:i1].copy(), freqs, centre.copy())
