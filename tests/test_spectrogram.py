import numpy as np
import pytest
import soundfile as sf

from dopplergeo.acoustics import received_frequency
from dopplergeo.adsb.schema import trajectory_from_rows
from dopplergeo.spectrogram import (
    Spectrogram,
    compute_spectrogram,
    extract_ridge,
    load_audio,
)
from dopplergeo.synth import Source, render_audio, straight_track

SR = 8000
LAT0, LON0 = 37.0, -122.0
T_START = 1100.0
DURATION = 100.0
F0 = 120.0


def _flyover(noise_rms: float, **track_kwargs):
    df = straight_track("A", 1000.0, 300.0, LAT0, LON0, 1000.0, 60.0, 90.0, **track_kwargs)
    traj = trajectory_from_rows(df, LAT0, LON0)
    obs = np.array([60 * 150.0, 400.0, 0.0])
    audio = render_audio([Source(traj, F0)], obs, T_START, DURATION, SR, noise_rms=noise_rms)
    return traj, obs, audio


def test_compute_spectrogram_axes_and_tone_peak():
    t = np.arange(int(10 * SR)) / SR
    audio = np.sin(2 * np.pi * 100.0 * t)
    spec = compute_spectrogram(audio, SR)

    assert spec.power_db.shape == (spec.freqs.size, spec.times.size)
    assert spec.times[0] == 0.0
    np.testing.assert_allclose(np.diff(spec.times), 0.25)
    assert spec.df <= 0.5
    assert spec.freqs[-1] <= 600.0
    assert spec.times[-1] <= 10.0 < spec.times[-1] + 0.25

    mid = spec.times.size // 2
    k = int(np.argmax(spec.power_db[:, mid]))
    assert abs(spec.freqs[k] - 100.0) <= spec.df
    # Unit-amplitude sine reads about 0 dB (Hann scalloping costs < 1.5 dB).
    assert -1.5 < spec.power_db[k, mid] <= 0.1
    # Far from the tone the floor is far down.
    far = int(np.searchsorted(spec.freqs, 300.0))
    assert spec.power_db[far, mid] < -60.0


def test_whitened_is_nonnegative_and_flags_tone():
    rng = np.random.default_rng(1)
    t = np.arange(int(20 * SR)) / SR
    audio = 0.3 * np.sin(2 * np.pi * 150.0 * t) + rng.normal(0, 0.5, t.size)
    spec = compute_spectrogram(audio, SR)
    w = spec.whitened()

    assert w.power_db.shape == spec.power_db.shape
    assert w.power_db.min() >= 0.0
    np.testing.assert_array_equal(w.times, spec.times)
    np.testing.assert_array_equal(w.freqs, spec.freqs)
    k = int(np.argmin(np.abs(spec.freqs - 150.0)))
    # A bin the tone occupies the whole time has no contrast against its own
    # median, so look at the bin's neighbours' background instead: the tone
    # sits many sigma above the floor of the bins beside it.
    assert np.median(w.power_db[k, :]) == 0.0
    background = w.power_db[k - 10 : k - 4, :]
    assert np.percentile(background, 95) < 3.0
    # Whitening the ridge of a changing tone (a chirp) gives large values.
    t2 = np.arange(int(20 * SR)) / SR
    chirp = 0.3 * np.sin(2 * np.pi * (150.0 * t2 - 0.5 * t2**2)) + rng.normal(0, 0.5, t2.size)
    w2 = compute_spectrogram(chirp, SR).whitened()
    mid = w2.times.size // 2
    assert w2.power_db[:, mid].max() > 5.0


def test_sample_bilinear_exact_on_bilinear_field():
    times = np.arange(0.0, 10.0, 0.25)
    freqs = np.arange(0.0, 50.0, 0.5)
    field = 2.0 * times[None, :] + 3.0 * freqs[:, None] + 1.0
    spec = Spectrogram(times, freqs, field)

    rng = np.random.default_rng(0)
    t = rng.uniform(0.0, times[-1], 1000)
    f = rng.uniform(0.0, freqs[-1], 1000)
    vals, inb = spec.sample(t, f)
    assert inb.all()
    np.testing.assert_allclose(vals, 2.0 * t + 3.0 * f + 1.0, rtol=0, atol=1e-9)

    # Grid points come back exactly; corners included.
    vals, inb = spec.sample(times[-1], freqs[-1])
    assert inb and vals == pytest.approx(field[-1, -1])
    vals, inb = spec.sample(0.0, 0.0)
    assert vals == pytest.approx(field[0, 0])

    # Out of bounds: value 0, mask False.
    vals, inb = spec.sample(np.array([-0.1, 5.0, 9.75, 5.0]), np.array([10.0, -1.0, 10.0, 60.0]))
    np.testing.assert_array_equal(inb, [False, False, True, False])
    assert vals[0] == 0.0 and vals[1] == 0.0 and vals[3] == 0.0

    # Broadcasting: a column of times against a row of frequencies.
    vals, inb = spec.sample(t[:5, None], f[None, :7])
    assert vals.shape == (5, 7) and inb.shape == (5, 7)


def test_sample_is_fast():
    spec = compute_spectrogram(np.random.default_rng(0).normal(size=SR * 60), SR)
    rng = np.random.default_rng(1)
    t = rng.uniform(0, 60, 1_000_000)
    f = rng.uniform(0, 600, 1_000_000)
    import time

    t0 = time.perf_counter()
    vals, inb = spec.sample(t, f)
    elapsed = time.perf_counter() - t0
    assert vals.shape == (1_000_000,)
    assert elapsed < 2.0


def test_extract_ridge_follows_linear_chirp():
    rng = np.random.default_rng(2)
    t = np.arange(int(60 * SR)) / SR
    f_inst = 140.0 - 0.5 * t  # Hz; 140 -> 110 over 60 s
    phase = 2 * np.pi * np.cumsum(f_inst) / SR
    audio = np.sin(phase) + rng.normal(0, 1.0, t.size)
    spec = compute_spectrogram(audio, SR)
    ridge = extract_ridge(spec, 2.0, 58.0, 100.0, 150.0)

    assert ridge.times[0] >= 2.0 and ridge.times[-1] <= 58.0
    assert ridge.times.size == ridge.freqs.size == ridge.strength.size
    truth = 140.0 - 0.5 * ridge.times
    err = ridge.freqs - truth
    assert np.sqrt(np.mean(err**2)) < 0.25
    assert np.abs(err).max() < 0.6
    assert np.median(ridge.strength) > 3.0


def test_extract_ridge_on_doppler_flyover_near_noise_floor():
    # noise_rms=8 leaves the tone only a few dB above the floor per bin.
    traj, obs, audio = _flyover(noise_rms=8.0)
    spec = compute_spectrogram(audio, SR)
    w = spec.whitened()
    ridge = extract_ridge(spec, 5.0, 95.0, 95.0, 150.0)

    # Confirm the regime: the ridge is weak in whitened units ...
    assert 1.0 < np.median(ridge.strength) < 4.0
    # ... and only a few dB above the box's background in raw power.
    box = spec.power_db[
        np.searchsorted(spec.freqs, 95.0) : np.searchsorted(spec.freqs, 150.0),
        np.searchsorted(spec.times, 5.0) : np.searchsorted(spec.times, 95.0),
    ]
    on_ridge, _ = spec.sample(ridge.times, ridge.freqs)
    contrast_db = np.median(on_ridge) - np.median(box)
    assert 2.0 < contrast_db < 12.0

    truth, valid = received_frequency(traj, obs, ridge.times + T_START, F0)
    assert valid.all()
    err = ridge.freqs - truth
    assert np.sqrt(np.mean(err**2)) < 0.4
    assert np.percentile(np.abs(err), 90) < 0.5
    assert np.abs(err).max() < 1.5

    # Passing the whitened spectrogram with the flag gives the same path.
    ridge_w = extract_ridge(w, 5.0, 95.0, 95.0, 150.0, already_whitened=True)
    np.testing.assert_array_equal(ridge_w.strength, ridge.strength)
    assert np.abs(ridge_w.freqs - truth).max() < 1.5


def test_extract_ridge_clean_flyover_is_accurate():
    traj, obs, audio = _flyover(noise_rms=0.05)
    spec = compute_spectrogram(audio, SR)
    ridge = extract_ridge(spec, 5.0, 95.0, 95.0, 150.0)
    truth, _ = received_frequency(traj, obs, ridge.times + T_START, F0)
    err = ridge.freqs - truth
    assert np.sqrt(np.mean(err**2)) < 0.1
    assert np.abs(err).max() < 0.3


def test_extract_ridge_respects_slope_limit():
    traj, obs, audio = _flyover(noise_rms=0.05)
    spec = compute_spectrogram(audio, SR)
    ridge = extract_ridge(spec, 5.0, 95.0, 95.0, 150.0, max_slope_hz_per_s=3.0)
    slopes = np.abs(np.diff(ridge.freqs)) / np.diff(ridge.times)
    # Hard limit on the bin path plus up to two bins of sub-bin refinement.
    assert slopes.max() <= 3.0 + 2 * spec.df / spec.dt + 1e-9


def test_extract_ridge_rejects_empty_box():
    spec = compute_spectrogram(np.zeros(SR * 5), SR)
    with pytest.raises(ValueError):
        extract_ridge(spec, 10.0, 20.0, 100.0, 200.0)
    with pytest.raises(ValueError):
        extract_ridge(spec, 0.0, 5.0, 700.0, 800.0)


def test_load_audio_mixes_and_resamples(tmp_path):
    sr_in = 44100
    t = np.arange(int(3 * sr_in)) / sr_in
    left = np.sin(2 * np.pi * 200.0 * t)
    right = np.zeros_like(left)
    path = tmp_path / "stereo.wav"
    sf.write(path, np.column_stack([left, right]), sr_in)

    audio, sr = load_audio(path, target_sr=SR)
    assert sr == SR
    assert audio.dtype == np.float64 and audio.ndim == 1
    assert abs(audio.size - 3 * SR) <= 2
    # Mono mix halves the amplitude; the tone survives the resample.
    assert np.std(audio) == pytest.approx(0.5 / np.sqrt(2), rel=0.05)
    spec = compute_spectrogram(audio, SR)
    mid = spec.times.size // 2
    assert abs(spec.freqs[np.argmax(spec.power_db[:, mid])] - 200.0) <= spec.df

    same, sr2 = load_audio(path, target_sr=sr_in)
    assert sr2 == sr_in and same.size == t.size
