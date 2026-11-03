"""Command-line interface.

    dopplergeo measure  audio.wav --t0 410 --t1 540 --fmin 90 --fmax 160 -o event1.json
    dopplergeo search   audio.wav tracks.parquet event1.json event2.json -o results.csv
    dopplergeo search-single audio.wav tracks.parquet event.json --time 2026-09-08T20:37:00Z --tol 60
    dopplergeo demo     --out demo/        # synthetic end-to-end run with figures
    dopplergeo fetch-adsb 2026-09-08 --out data/adsb
    dopplergeo load-adsb data/adsb/extracted --bbox 36 38 -123 -121 -o tracks.parquet
    dopplergeo fetch-dem --bbox 36 38 -123 -121 -o dem.tif
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _parse_time(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()


def _dem_from_args(args) -> "object":
    from .dem import FlatElevation

    if getattr(args, "dem", None):
        from .dem import RasterElevation

        return RasterElevation(args.dem)
    if getattr(args, "terrain_tiles", False):
        from .dem import TerrainTilesElevation

        return TerrainTilesElevation()
    return FlatElevation(float(getattr(args, "flat_elevation", 0.0)))


def cmd_measure(args) -> int:
    from .curve_fit import fit_ideal_curve
    from .spectrogram import compute_spectrogram, extract_ridge, load_audio

    audio, sr = load_audio(args.audio, target_sr=args.sample_rate)
    spec = compute_spectrogram(audio, sr, f_max=max(args.fmax * 1.2, args.spec_fmax), freq_resolution_hz=args.resolution, hop_s=args.hop)
    ridge = extract_ridge(spec, args.t0, args.t1, args.fmin, args.fmax, max_slope_hz_per_s=args.max_slope)
    m = fit_ideal_curve(ridge, c=args.speed_of_sound)
    d = m.to_dict()
    print(json.dumps(d, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(d, indent=2))
    return 0


def _load_measurement(path: str):
    from .curve_fit import CurveMeasurement

    return CurveMeasurement.from_dict(json.loads(Path(path).read_text()))


def _spectrogram_for(args):
    from .spectrogram import compute_spectrogram, load_audio

    audio, sr = load_audio(args.audio, target_sr=args.sample_rate)
    spec = compute_spectrogram(audio, sr, f_max=args.spec_fmax, freq_resolution_hz=args.resolution, hop_s=args.hop)
    return spec.whitened()


def _search_config(args):
    from .scoring import ScoreConfig
    from .search import SearchConfig

    return SearchConfig(
        grid_m=args.grid,
        fine_grid_m=args.fine_grid,
        offset_tol_s=args.offset_tol,
        silence_radius_m=args.silence_radius,
        silence_margin_s=args.silence_margin,
        exclude_jets=not args.keep_jets,
        refine_top_k=args.refine_top,
        c=args.speed_of_sound,
        score=ScoreConfig(harmonics=tuple(args.harmonics)),
    )


def cmd_search(args) -> int:
    from .adsb.schema import read_tracks
    from .search import two_event_search

    spec_w = _spectrogram_for(args)
    tracks = read_tracks(args.tracks)
    m1 = _load_measurement(args.event1)
    m2 = _load_measurement(args.event2)
    res = two_event_search(spec_w, tracks, m1, m2, _dem_from_args(args), _search_config(args))
    print(res.summary(args.top))
    if args.out:
        res.by_flight_pair.to_csv(args.out, index=False)
        print(f"wrote {args.out}")
    return 0


def cmd_search_single(args) -> int:
    from .adsb.schema import read_tracks
    from .search import single_event_search

    spec_w = _spectrogram_for(args)
    tracks = read_tracks(args.tracks)
    m = _load_measurement(args.event)
    res = single_event_search(spec_w, tracks, m, _dem_from_args(args), _parse_time(args.time), args.tol, _search_config(args))
    print(res.summary(args.top))
    if args.out:
        res.by_flight_pair.to_csv(args.out, index=False)
        print(f"wrote {args.out}")
    return 0


def cmd_demo(args) -> int:
    from .demo import run_demo

    run_demo(Path(args.out), seed=args.seed, figures=not args.no_figures)
    return 0


def cmd_fetch_adsb(args) -> int:
    from .adsb.sources import download_adsblol_day

    path = download_adsblol_day(args.date, args.out, extract=not args.no_extract)
    print(path)
    return 0


def cmd_load_adsb(args) -> int:
    from .adsb.schema import write_tracks
    from .adsb.sources import load_opensky_csv, load_readsb_history

    bbox = tuple(args.bbox) if args.bbox else None
    trange = (_parse_time(args.t0), _parse_time(args.t1)) if args.t0 and args.t1 else None
    src = Path(args.src)
    if src.is_dir():
        df = load_readsb_history(src, bbox=bbox, time_range=trange)
    else:
        df = load_opensky_csv(src)
    write_tracks(df, args.out)
    print(f"{df['flight_id'].nunique()} flights, {len(df)} rows -> {args.out}")
    return 0


def cmd_fetch_dem(args) -> int:
    from .dem import fetch_dem

    lat_min, lat_max, lon_min, lon_max = args.bbox
    path = fetch_dem(lat_min, lat_max, lon_min, lon_max, args.out, zoom=args.zoom)
    print(path)
    return 0


def _add_audio_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--sample-rate", type=int, default=8000)
    p.add_argument("--resolution", type=float, default=0.5, help="spectrogram bin spacing, Hz")
    p.add_argument("--hop", type=float, default=0.25, help="spectrogram hop, s")
    p.add_argument("--spec-fmax", type=float, default=600.0, help="top of the spectrogram, Hz")
    p.add_argument("--speed-of-sound", type=float, default=343.0)


def _add_search_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dem", help="GeoTIFF elevation raster (needs rasterio)")
    p.add_argument("--terrain-tiles", action="store_true", help="fetch AWS terrain tiles on demand (needs rasterio + network)")
    p.add_argument("--flat-elevation", type=float, default=0.0, help="constant ground elevation (m) when no DEM is given")
    p.add_argument("--grid", type=float, default=200.0)
    p.add_argument("--fine-grid", type=float, default=100.0)
    p.add_argument("--offset-tol", type=float, default=5.0)
    p.add_argument("--silence-radius", type=float, default=4000.0)
    p.add_argument("--silence-margin", type=float, default=30.0)
    p.add_argument("--keep-jets", action="store_true")
    p.add_argument("--refine-top", type=int, default=50)
    p.add_argument("--harmonics", type=float, nargs="+", default=[1.0, 0.5])
    p.add_argument("--top", type=int, default=10)
    p.add_argument("-o", "--out")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dopplergeo", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("measure", help="fit a Doppler curve in a time/frequency box of a recording")
    m.add_argument("audio")
    m.add_argument("--t0", type=float, required=True)
    m.add_argument("--t1", type=float, required=True)
    m.add_argument("--fmin", type=float, required=True)
    m.add_argument("--fmax", type=float, required=True)
    m.add_argument("--max-slope", type=float, default=6.0, help="ridge tracker slope limit, Hz/s (about f0*v^2/(c*d) at the closest point)")
    m.add_argument("-o", "--out")
    _add_audio_args(m)
    m.set_defaults(func=cmd_measure)

    s = sub.add_parser("search", help="two-event search, unknown recording time")
    s.add_argument("audio")
    s.add_argument("tracks")
    s.add_argument("event1")
    s.add_argument("event2")
    _add_audio_args(s)
    _add_search_args(s)
    s.set_defaults(func=cmd_search)

    s1 = sub.add_parser("search-single", help="one-event search with a known recording time")
    s1.add_argument("audio")
    s1.add_argument("tracks")
    s1.add_argument("event")
    s1.add_argument("--time", required=True, help="unix seconds or ISO-8601 UTC of the event's CPA")
    s1.add_argument("--tol", type=float, default=60.0, help="± seconds")
    _add_audio_args(s1)
    _add_search_args(s1)
    s1.set_defaults(func=cmd_search_single)

    d = sub.add_parser("demo", help="synthetic end-to-end run")
    d.add_argument("--out", default="demo")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--no-figures", action="store_true")
    d.set_defaults(func=cmd_demo)

    f = sub.add_parser("fetch-adsb", help="download one day of adsb.lol global history")
    f.add_argument("date", help="YYYY-MM-DD")
    f.add_argument("--out", default="data/adsb")
    f.add_argument("--no-extract", action="store_true")
    f.set_defaults(func=cmd_fetch_adsb)

    l = sub.add_parser("load-adsb", help="convert readsb history or an OpenSky CSV to the track table")
    l.add_argument("src")
    l.add_argument("-o", "--out", required=True)
    l.add_argument("--bbox", type=float, nargs=4, metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"))
    l.add_argument("--t0")
    l.add_argument("--t1")
    l.set_defaults(func=cmd_load_adsb)

    g = sub.add_parser("fetch-dem", help="download and merge terrain tiles into a GeoTIFF")
    g.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"))
    g.add_argument("-o", "--out", required=True)
    g.add_argument("--zoom", type=int, default=11)
    g.set_defaults(func=cmd_fetch_dem)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    np.set_printoptions(precision=3, suppress=True)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
