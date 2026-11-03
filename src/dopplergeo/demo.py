"""Synthetic end-to-end demonstration.

Builds a 40 km region with two propeller flyovers 920 s apart over a hidden
microphone, a crowd of decoy flights (propeller and jet, including one that
passes between the two events somewhere else), renders twenty minutes of
audio with the independent waveform synthesiser, then runs the full search
knowing nothing but the audio and the ADS-B table. Reports how far the best
candidate is from the truth. Optionally writes figures for the README.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .acoustics import closest_point_of_approach
from .adsb.schema import normalize_tracks, trajectory_from_rows, write_tracks
from .constants import SPEED_OF_SOUND_MPS as C
from .curve_fit import CurveMeasurement, fit_ideal_curve
from .dem import ArrayElevation
from .geo import enu_from_latlon, grid_cell, haversine_m, latlon_from_enu
from .search import SearchConfig, SearchResult, single_event_search, two_event_search
from .spectrogram import Spectrogram, compute_spectrogram, extract_ridge
from .synth import Source, render_audio, straight_track

LAT0, LON0 = 37.30, -121.90  # frame origin of the synthetic region
T0 = 1_757_361_600.0  # 2025-09-08 20:00:00 UTC, an arbitrary unix time
AUDIO_START = T0 + 600.0
AUDIO_LEN = 1200.0
SR = 8000


@dataclass
class Scenario:
    tracks: pd.DataFrame
    observer_lat: float
    observer_lon: float
    observer_elev_m: float
    dem: ArrayElevation
    audio: np.ndarray
    truth: dict


def _sloped_dem() -> ArrayElevation:
    lat_axis = np.linspace(LAT0 - 0.5, LAT0 + 0.5, 101)
    lon_axis = np.linspace(LON0 - 0.6, LON0 + 0.6, 121)
    la, lo = np.meshgrid(lat_axis, lon_axis, indexing="ij")
    grid = 40.0 + 300.0 * (la - (LAT0 - 0.5)) + 80.0 * np.sin(6 * lo)
    return ArrayElevation(lat_axis, lon_axis, grid)


def build_scenario(seed: int = 0) -> Scenario:
    rng = np.random.default_rng(seed)
    dem = _sloped_dem()
    # The hidden microphone, a few km from the frame origin.
    obs_e, obs_n = 2300.0, -1700.0
    obs_lat, obs_lon = latlon_from_enu(obs_e, obs_n, LAT0, LON0)
    obs_elev = float(dem.elevation_m(np.array([obs_lat]), np.array([obs_lon]))[0])
    obs_enu = np.array([obs_e, obs_n, obs_elev])

    flights: list[pd.DataFrame] = []
    sources: list[Source] = []

    def add(df: pd.DataFrame, f0: float, amp: float = 1.0) -> None:
        flights.append(df)
        traj = trajectory_from_rows(df, LAT0, LON0)
        sources.append(Source(traj, f0, amplitude=amp))

    def place(df: pd.DataFrame, t_at: float, side_m: float, above_m: float) -> pd.DataFrame:
        """Translate a track so that at ``t_at`` it is ``side_m`` to the right
        of its heading from the microphone and ``above_m`` above it."""
        traj = trajectory_from_rows(df, LAT0, LON0)
        p = traj.position(t_at)
        v = traj.velocity(t_at)
        right = np.array([v[1], -v[0], 0.0])
        right /= np.linalg.norm(right)
        target = obs_enu + side_m * right + np.array([0.0, 0.0, above_m])
        shift = target - p
        e, n = enu_from_latlon(df["lat"].to_numpy(), df["lon"].to_numpy(), LAT0, LON0)
        lat, lon = latlon_from_enu(e + shift[0], n + shift[1], LAT0, LON0)
        out = df.copy()
        out["lat"], out["lon"] = lat, lon
        out["alt_m"] = out["alt_m"] + shift[2]
        return out

    back = 360.0
    # Event 1: a Cessna-class aircraft, 64 m/s, gently turning and descending,
    # passing 600 m to the side and 900 m above the microphone at audio time ~150 s.
    t_cpa1 = AUDIO_START + 150.0
    df1 = straight_track("evt1", t_cpa1 - back, 720.0, LAT0, LON0, 900.0, 64.0, 250.0, turn_rate_dps=0.15, climb_mps=-1.0, typecode="C182")
    add(place(df1, t_cpa1, 600.0, 900.0), 118.0, amp=1.2)
    # Event 2: a Piper-class aircraft, 52 m/s, passing 1.5 km to the other side
    # and 650 m above, 920 s later.
    t_cpa2 = t_cpa1 + 920.0
    df2 = straight_track("evt2", t_cpa2 - back, 720.0, LAT0, LON0, 650.0, 52.0, 95.0, turn_rate_dps=-0.1, accel_mps2=0.004, typecode="P28A")
    add(place(df2, t_cpa2, -1500.0, 650.0), 96.0, amp=1.0)

    def breaks_silence(df: pd.DataFrame) -> bool:
        """True if this flight would be heard at the microphone between the
        two events (it would then show up as a third curve, and the analyst
        would not call the gap silent)."""
        traj = trajectory_from_rows(df, LAT0, LON0)
        t_cpa, slant = closest_point_of_approach(traj, obs_enu)
        heard = t_cpa + slant / C
        return slant < 6000.0 and (t_cpa1 - 200.0) < heard < (t_cpa2 + 200.0)

    def random_decoy(name: str, typecode: str, *, jet: bool) -> pd.DataFrame:
        for _ in range(200):
            lat, lon = latlon_from_enu(rng.uniform(-18000, 18000), rng.uniform(-18000, 18000), LAT0, LON0)
            if jet:
                df = straight_track(name, AUDIO_START + rng.uniform(0, AUDIO_LEN), 400.0, lat, lon, 3000.0 + rng.uniform(0, 3000), rng.uniform(140, 220), rng.uniform(0, 360), typecode=typecode)
            else:
                t_start = AUDIO_START - 300.0 + rng.uniform(0, AUDIO_LEN)
                speed = float(rng.choice([48.0, 55.0, 62.0, 66.0, 75.0, 90.0]))
                df = straight_track(name, t_start, 600.0, lat, lon, obs_elev + rng.uniform(400, 1500), speed, rng.uniform(0, 360), turn_rate_dps=rng.uniform(-0.2, 0.2), typecode=typecode)
            if not breaks_silence(df):
                return df
        raise RuntimeError("could not place a decoy that keeps the gap silent")

    # Decoy propeller flights around the region, some at similar speeds. They
    # are rendered into the audio too, so they must not pass the microphone
    # between the events.
    for k in range(14):
        add(random_decoy(f"dec{k:02d}", str(rng.choice(["C172", "SR22", "P28A", "BE36", ""])), jet=False), float(rng.uniform(85, 140)), amp=0.8)
    # A decoy that flies between the two events on the far side of the region:
    # it must trip the silence filter for cells near it, not for the truth.
    lat, lon = latlon_from_enu(obs_e - 12000.0, obs_n + 9000.0, LAT0, LON0)
    add(straight_track("between", t_cpa1 + 300.0, 500.0, lat, lon, obs_elev + 700.0, 60.0, 180.0, typecode="C172"), 110.0, amp=0.8)
    # Jets: fast, high; the type filter should remove them from the candidates.
    for k in range(3):
        add(random_decoy(f"jet{k}", "B738", jet=True), 60.0, amp=0.3)

    tracks = normalize_tracks(pd.concat(flights, ignore_index=True))
    audio = render_audio(sources, obs_enu, AUDIO_START, AUDIO_LEN, SR, noise_rms=0.25, seed=seed)

    tr1 = trajectory_from_rows(tracks[tracks["flight_id"] == "evt1"], LAT0, LON0)
    tr2 = trajectory_from_rows(tracks[tracks["flight_id"] == "evt2"], LAT0, LON0)
    t1, d1 = closest_point_of_approach(tr1, obs_enu)
    t2, d2 = closest_point_of_approach(tr2, obs_enu)
    truth = {
        "lat": float(obs_lat),
        "lon": float(obs_lon),
        "elev_m": obs_elev,
        "evt1": {"t_cpa_tx": t1, "slant_m": d1, "t_cpa_rx_audio": t1 + d1 / C - AUDIO_START, "f0": 118.0, "speed": float(tr1.speed(t1))},
        "evt2": {"t_cpa_tx": t2, "slant_m": d2, "t_cpa_rx_audio": t2 + d2 / C - AUDIO_START, "f0": 96.0, "speed": float(tr2.speed(t2))},
        "audio_start_unix": AUDIO_START,
    }
    return Scenario(tracks, float(obs_lat), float(obs_lon), obs_elev, dem, audio, truth)


def measure_event(spec: Spectrogram, t_center: float, f_center: float, speed_guess: float, half_window_s: float = 70.0) -> CurveMeasurement:
    """Mimic the analyst's step: draw a box around the curve, fit it.

    The box is generous (±70 s, and the full Doppler span of a plane up to
    ~1.5× the guessed speed) so nothing about the truth leaks in beyond what a
    person reading the spectrogram would pick by eye.
    """
    v = 1.5 * speed_guess
    f_lo = f_center * (C - v) / (C + v) * 0.97
    f_hi = f_center * (C + v) / (C - v) * 1.03
    # Slope limit from the steepest plausible pass: f0 v^2 / (c d) at 1.5x the
    # guessed speed and a 500 m closest approach, never below 3 Hz/s.
    max_slope = max(3.0, f_center * v * v / (C * 500.0))
    ridge = extract_ridge(spec, t_center - half_window_s, t_center + half_window_s, f_lo, f_hi, max_slope_hz_per_s=max_slope)
    return fit_ideal_curve(ridge)


def run_demo(out: Path, *, seed: int = 0, figures: bool = True, log=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    t_wall = time.time()
    sc = build_scenario(seed)
    write_tracks(sc.tracks, out / "tracks.parquet")
    try:
        import soundfile as sf

        sf.write(out / "recording.wav", sc.audio, SR)
    except Exception as exc:  # pragma: no cover - soundfile missing codecs
        log(f"[demo] could not write wav: {exc}")
    log(f"[demo] {sc.tracks['flight_id'].nunique()} flights, {AUDIO_LEN:.0f} s of audio at {SR} Hz")

    spec = compute_spectrogram(sc.audio, SR, f_max=400.0, freq_resolution_hz=0.5, hop_s=0.25)
    spec_w = spec.whitened()

    # The analyst reads two curves off the spectrogram: roughly where in time
    # and at what frequency. Those eyeballed numbers are the only inputs.
    tr = sc.truth
    m1 = measure_event(spec, round(tr["evt1"]["t_cpa_rx_audio"] / 10) * 10, round(tr["evt1"]["f0"] / 5) * 5, 70.0)
    m2 = measure_event(spec, round(tr["evt2"]["t_cpa_rx_audio"] / 10) * 10, round(tr["evt2"]["f0"] / 5) * 5, 60.0)
    for name, m, t in (("event 1", m1, tr["evt1"]), ("event 2", m2, tr["evt2"])):
        log(
            f"[demo] {name}: fit v={m.speed_mps:.1f} m/s (true {t['speed']:.1f}), d={m.cpa_slant_m:.0f} m (true {t['slant_m']:.0f}), "
            f"f0={m.f0_hz:.2f} Hz (true {t['f0']:.0f}), t_cpa={m.t_cpa_rx:.2f} s (true {t['t_cpa_rx_audio']:.2f}); "
            f"ranges v {m.speed_range[0]:.0f}-{m.speed_range[1]:.0f}, d {m.cpa_slant_range[0]:.0f}-{m.cpa_slant_range[1]:.0f}"
        )
    (out / "event1.json").write_text(json.dumps(m1.to_dict(), indent=2))
    (out / "event2.json").write_text(json.dumps(m2.to_dict(), indent=2))

    cfg = SearchConfig()
    res = two_event_search(spec_w, sc.tracks, m1, m2, sc.dem, cfg, log=log)
    best = res.best
    err_m = float(haversine_m(best["lat"], best["lon"], sc.observer_lat, sc.observer_lon))
    truth_cell = grid_cell(sc.observer_lat, sc.observer_lon, cfg.grid_m)
    in_pairs = bool(((res.pairs["row"] == truth_cell[0]) & (res.pairs["col"] == truth_cell[1])).any())
    log(f"[demo] best: ({best['lat']:.5f}, {best['lon']:.5f}) via {best['flight_id_1']} + {best['flight_id_2']}, total {best['total']:.2f}; truth cell survived filters: {in_pairs}")
    log(f"[demo] distance from truth: {err_m:.0f} m; runner-up flight pair total {res.by_flight_pair['total'].iloc[1] if len(res.by_flight_pair) > 1 else float('nan'):.2f}")

    # Single-event mode: the second curve alone, plus a file timestamp that
    # is 7 s off and trusted to ±20 s. The time window replaces the first
    # curve as the constraint; what is left is an along-track ambiguity of
    # about (tolerance x aircraft speed) that only the flight's own turns can
    # break.
    known_rx = tr["evt2"]["t_cpa_rx_audio"] + AUDIO_START + 7.0
    res1 = single_event_search(spec_w, sc.tracks, m2, sc.dem, known_rx, 20.0, cfg, log=log)
    best1 = res1.best
    err1_m = float(haversine_m(best1["lat"], best1["lon"], sc.observer_lat, sc.observer_lon))
    log(f"[demo] single-event best: ({best1['lat']:.5f}, {best1['lon']:.5f}) via {best1['flight_id']}, score {best1['score']:.2f}; distance from truth {err1_m:.0f} m")

    report = {
        "truth": tr,
        "measurements": {"event1": m1.to_dict(), "event2": m2.to_dict()},
        "two_event": {
            "n_flights": int(sc.tracks["flight_id"].nunique()),
            "candidates_1": int(len(res.candidates_1)),
            "candidates_2": int(len(res.candidates_2)),
            "pairs_after_offset_and_silence": int(len(res.pairs)),
            "truth_cell_survived_filters": in_pairs,
            "best": {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in best.items()},
            "error_m": err_m,
            "top_pairs": res.by_flight_pair.head(5).to_dict(orient="records"),
        },
        "single_event": {"best": {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in best1.items()}, "error_m": err1_m},
        "wall_s": time.time() - t_wall,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    res.by_flight_pair.to_csv(out / "ranked_pairs.csv", index=False)
    if figures:
        from .figures import make_figures

        make_figures(out, sc, spec, spec_w, m1, m2, res, res1)
    log(f"[demo] done in {report['wall_s']:.0f} s; outputs in {out}")
    return report


__all__ = ["Scenario", "build_scenario", "measure_event", "run_demo", "SearchResult", "enu_from_latlon"]
