"""ADS-B data sources -> the canonical track table.

* readsb / tar1090 ``trace_full_<hex>.json`` files, as published daily by
  adsb.lol (``adsblol/globe_history_<year>`` GitHub releases) and by
  adsbexchange's globe_history.
* OpenSky state-vector exports (CSV or parquet) and, optionally, live
  ``pyopensky`` Trino queries.

Everything returns a frame that passed :func:`schema.normalize_tracks`, with
``flight_id == icao24``; :func:`dopplergeo.adsb.tracks.split_on_gaps` turns
that into per-segment ids later.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import tarfile
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..constants import FEET_TO_M, FPM_TO_MPS, KNOTS_TO_MPS
from .schema import COLUMNS, empty_tracks, normalize_tracks

BBox = tuple[float, float, float, float]  # lat_min, lat_max, lon_min, lon_max
TimeRange = tuple[float, float]  # t0, t1 unix seconds

# readsb trace entry layout (index -> meaning). Entries may be shorter than
# this; every index is read defensively.
_I_DT, _I_LAT, _I_LON, _I_ALT_BARO, _I_GS, _I_TRACK, _I_FLAGS, _I_VRATE = range(8)
_I_AIRCRAFT, _I_SOURCE, _I_ALT_GEOM, _I_GEOM_RATE, _I_IAS, _I_ROLL = range(8, 14)

# readsb trace flags bitfield.
FLAG_STALE = 1  # position is stale (re-emitted, not a fresh fix)
FLAG_NEW_LEG = 2
FLAG_VRATE_GEOM = 4  # entry[7] is a geometric rate
FLAG_ALT_GEOM = 8  # entry[3] is a geometric altitude


# --------------------------------------------------------------------------
# readsb traces
# --------------------------------------------------------------------------


def _get(entry: Sequence[Any], i: int) -> Any:
    return entry[i] if i < len(entry) else None


def _num(x: Any) -> float:
    """float(x) for ints/floats/numeric strings; NaN otherwise."""
    if x is None or isinstance(x, bool):
        return float("nan")
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _read_json(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _icao_from_name(path: Path) -> str:
    name = path.name
    for suffix in (".json.gz", ".json"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    if name.startswith("trace_full_"):
        name = name[len("trace_full_") :]
    return name.lower()


def parse_readsb_trace(
    obj_or_path: dict[str, Any] | str | Path,
    *,
    drop_stale: bool = True,
) -> pd.DataFrame:
    """Convert one readsb/tar1090 ``trace_full_<hex>.json`` to schema rows.

    Top-level keys used: ``icao`` (hex), ``t`` (type code), ``timestamp``
    (unix seconds), ``trace`` (list of entries). Each entry is
    ``[dt, lat, lon, alt_baro, gs_kt, track, flags, vrate_fpm, aircraft,
    source, alt_geom_ft, geom_rate_fpm, ias, roll]``; trailing fields may be
    absent.

    * ``time = timestamp + dt``
    * ``alt_m`` prefers ``alt_geom`` ("geom"); else ``alt_baro`` ("baro",
      or "geom" when the flags say the baro slot holds a geometric value).
      Rows on the ground (``alt_baro == "ground"``) or without any altitude
      are dropped.
    * ``vrate_mps`` prefers ``geom_rate``; else the baro vertical rate.
    * ``callsign`` is the most recent ``aircraft["flight"]`` seen so far
      (rows before the first one take the first one seen).
    * ``flight_id`` is the icao24; segments are assigned later.

    ``drop_stale`` removes entries flagged stale (bit 0), which re-emit the
    last known position under a later timestamp.
    """
    if isinstance(obj_or_path, (str, Path)):
        path = Path(obj_or_path)
        obj = _read_json(path)
        fallback_icao = _icao_from_name(path)
    else:
        obj = obj_or_path
        fallback_icao = ""

    icao = str(obj.get("icao") or fallback_icao).strip().lower()
    typecode = str(obj.get("t") or "").strip().upper()
    t_base = _num(obj.get("timestamp"))
    trace: list[Sequence[Any]] = obj.get("trace") or []
    if not trace or not np.isfinite(t_base):
        return empty_tracks()

    n = len(trace)
    time = np.full(n, np.nan)
    lat = np.full(n, np.nan)
    lon = np.full(n, np.nan)
    alt = np.full(n, np.nan)
    gs = np.full(n, np.nan)
    trk = np.full(n, np.nan)
    vr = np.full(n, np.nan)
    alt_source: list[str] = [""] * n
    callsign: list[str] = [""] * n
    keep = np.ones(n, dtype=bool)

    current_callsign = ""
    first_callsign: str | None = None
    for k, e in enumerate(trace):
        if not isinstance(e, (list, tuple)):
            keep[k] = False
            continue
        ac = _get(e, _I_AIRCRAFT)
        if isinstance(ac, dict):
            fl = ac.get("flight")
            if isinstance(fl, str) and fl.strip():
                current_callsign = fl.strip()
                if first_callsign is None:
                    first_callsign = current_callsign
            if not typecode:
                t_ac = ac.get("t")
                if isinstance(t_ac, str) and t_ac.strip():
                    typecode = t_ac.strip().upper()
        callsign[k] = current_callsign

        flags = _get(e, _I_FLAGS)
        flags_i = int(flags) if isinstance(flags, (int, float)) and not isinstance(flags, bool) else 0
        if drop_stale and flags_i & FLAG_STALE:
            keep[k] = False
            continue

        alt_baro = _get(e, _I_ALT_BARO)
        if isinstance(alt_baro, str) and alt_baro.strip().lower() == "ground":
            keep[k] = False
            continue

        alt_geom = _num(_get(e, _I_ALT_GEOM))
        if np.isfinite(alt_geom):
            alt[k] = alt_geom * FEET_TO_M
            alt_source[k] = "geom"
        else:
            ab = _num(alt_baro)
            if not np.isfinite(ab):
                keep[k] = False
                continue
            alt[k] = ab * FEET_TO_M
            alt_source[k] = "geom" if flags_i & FLAG_ALT_GEOM else "baro"

        time[k] = t_base + _num(_get(e, _I_DT))
        lat[k] = _num(_get(e, _I_LAT))
        lon[k] = _num(_get(e, _I_LON))
        gs[k] = _num(_get(e, _I_GS)) * KNOTS_TO_MPS
        trk[k] = _num(_get(e, _I_TRACK))
        geom_rate = _num(_get(e, _I_GEOM_RATE))
        if np.isfinite(geom_rate):
            vr[k] = geom_rate * FPM_TO_MPS
        else:
            vr[k] = _num(_get(e, _I_VRATE)) * FPM_TO_MPS

    if first_callsign is not None:
        callsign = [c or first_callsign for c in callsign]

    df = pd.DataFrame(
        {
            "flight_id": icao,
            "icao24": icao,
            "callsign": pd.Series(callsign, dtype="string"),
            "typecode": typecode,
            "time": time,
            "lat": lat,
            "lon": lon,
            "alt_m": alt,
            "alt_source": pd.Series(alt_source, dtype="string"),
            "gs_mps": gs,
            "track_deg": trk,
            "vrate_mps": vr,
        }
    )
    df = df.loc[keep]
    return normalize_tracks(df)


def _mask_bbox(df: pd.DataFrame, bbox: BBox) -> pd.Series:
    lat_min, lat_max, lon_min, lon_max = bbox
    return (
        (df["lat"] >= lat_min)
        & (df["lat"] <= lat_max)
        & (df["lon"] >= lon_min)
        & (df["lon"] <= lon_max)
    )


def _mask_time(df: pd.DataFrame, time_range: TimeRange) -> pd.Series:
    t0, t1 = time_range
    return (df["time"] >= t0) & (df["time"] <= t1)


def iter_trace_files(root: str | Path) -> Iterator[Path]:
    """Yield every ``trace_full_*.json`` / ``.json.gz`` under ``root``, sorted."""
    root = Path(root)
    if root.is_file():
        yield root
        return
    files = [p for p in root.rglob("trace_full_*.json*") if p.is_file()]
    files = [p for p in files if p.name.endswith((".json", ".json.gz"))]
    yield from sorted(files)


def load_readsb_history(
    root: str | Path,
    *,
    bbox: BBox | None = None,
    time_range: TimeRange | None = None,
    drop_stale: bool = True,
) -> pd.DataFrame:
    """Load every readsb trace under a directory and concatenate.

    ``root`` is an extracted adsb.lol / adsbexchange globe_history archive
    (``traces/<last two hex chars>/trace_full_<hex>.json``, plain or
    gzipped; the directory layout does not matter, the walk is recursive).
    Archives are not opened here: extract first, e.g.::

        cat v2026.09.08-planes-readsb-prod-0.tar.* | tar -x -C out_dir

    or use :func:`download_adsblol_day`.

    ``bbox = (lat_min, lat_max, lon_min, lon_max)`` and
    ``time_range = (t0, t1)`` are applied per file, so memory stays bounded
    by the rows you actually keep.
    """
    frames: list[pd.DataFrame] = []
    for path in iter_trace_files(root):
        df = parse_readsb_trace(path, drop_stale=drop_stale)
        if len(df) == 0:
            continue
        if time_range is not None:
            # Cheap whole-file rejection before the row mask.
            if df["time"].iloc[-1] < time_range[0] or df["time"].iloc[0] > time_range[1]:
                continue
            df = df.loc[_mask_time(df, time_range)]
        if bbox is not None:
            df = df.loc[_mask_bbox(df, bbox)]
        if len(df):
            frames.append(df)
    if not frames:
        return empty_tracks()
    return normalize_tracks(pd.concat(frames, ignore_index=True))


# --------------------------------------------------------------------------
# adsb.lol daily archives
# --------------------------------------------------------------------------

_GITHUB_API = "https://api.github.com"
_USER_AGENT = "dopplergeo/0.1 (+https://github.com/noelsason)"


def adsblol_release_tag(date: str, variant: str = "prod-0") -> str:
    """``"2026-09-08"`` -> ``"v2026.09.08-planes-readsb-prod-0"``.

    Observed variants on ``adsblol/globe_history_2026`` (Nov 2026):
    ``prod-0`` (the full feed), ``staging-0``, ``mlatonly-0``.
    """
    y, m, d = date.split("-")
    if len(y) != 4 or len(m) != 2 or len(d) != 2:
        raise ValueError(f"date must be YYYY-MM-DD, got {date!r}")
    return f"v{y}.{m}.{d}-planes-readsb-{variant}"


def _github_headers() -> dict[str, str]:
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _github_json(url: str) -> Any:
    req = urllib.request.Request(url, headers=_github_headers())
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def list_adsblol_assets(date: str, variant: str = "prod-0") -> list[dict[str, Any]]:
    """Assets of the release for one day: ``[{"name", "size", "url"}, ...]``.

    Verified against the GitHub API on 2026-11-03: the release
    ``v2026.09.08-planes-readsb-prod-0`` carries
    ``v2026.09.08-planes-readsb-prod-0.tar.aa`` (2,000,000,000 bytes) and
    ``...tar.ab`` (1,874,165,248 bytes); the ``mlatonly-0`` variant is a
    single ``.tar``. Parts are the output of ``split -b 2000000000``, so
    they concatenate back into one tar.
    """
    year = date[:4]
    tag = adsblol_release_tag(date, variant)
    url = f"{_GITHUB_API}/repos/adsblol/globe_history_{year}/releases/tags/{tag}"
    try:
        rel = _github_json(url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise FileNotFoundError(f"no adsb.lol release {tag} (HTTP 404 at {url})") from exc
        raise
    assets = [
        {"name": a["name"], "size": int(a["size"]), "url": a["browser_download_url"]}
        for a in rel.get("assets", [])
    ]
    assets.sort(key=lambda a: a["name"])
    return assets


def _download_asset(url: str, dest: Path, expected_size: int, chunk: int = 1 << 20) -> Path:
    """Stream ``url`` to ``dest``; skip if complete, resume if partial."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if have == expected_size:
        return dest
    if have > expected_size:
        dest.unlink()
        have = 0
    headers = {"User-Agent": _USER_AGENT}
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp:
        resuming = have > 0 and resp.status == 206
        mode = "ab" if resuming else "wb"
        with open(dest, mode) as fh:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                fh.write(buf)
    got = dest.stat().st_size
    if got != expected_size:
        raise OSError(f"{dest.name}: downloaded {got} bytes, expected {expected_size}")
    return dest


class _ConcatReader(io.RawIOBase):
    """Read-only stream over the concatenation of several files.

    Lets :mod:`tarfile` extract a split tar without first writing the joined
    archive to disk.
    """

    def __init__(self, parts: Sequence[Path]) -> None:
        super().__init__()
        self._parts = list(parts)
        self._idx = 0
        self._fh: io.BufferedReader | None = None

    def readable(self) -> bool:
        return True

    def _open_next(self) -> bool:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        if self._idx >= len(self._parts):
            return False
        self._fh = open(self._parts[self._idx], "rb")
        self._idx += 1
        return True

    def readinto(self, b: bytearray | memoryview) -> int:  # type: ignore[override]
        view = memoryview(b)
        while True:
            if self._fh is None and not self._open_next():
                return 0
            assert self._fh is not None
            n = self._fh.readinto(view)
            if n:
                return n
            self._fh.close()
            self._fh = None

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        super().close()


def extract_split_tar(parts: Sequence[Path], out_dir: str | Path) -> Path:
    """Extract a tar given as ordered parts (``.tar.aa``, ``.tar.ab``, ...)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with _ConcatReader(parts) as raw:
        stream = io.BufferedReader(raw, buffer_size=1 << 20)
        with tarfile.open(fileobj=stream, mode="r|*") as tar:
            tar.extractall(out, filter="data")
    return out


def download_adsblol_day(
    date: str,
    out_dir: str | Path,
    *,
    extract: bool = True,
    variant: str = "prod-0",
) -> Path:
    """Download one day of adsb.lol global history and (optionally) extract it.

    adsb.lol publishes each day as a GitHub release on
    ``adsblol/globe_history_<YYYY>`` tagged ``v<YYYY.MM.DD>-planes-readsb-prod-0``
    whose assets are a split tar: ``<tag>.tar.aa``, ``<tag>.tar.ab``, ...
    (2 GB parts; a full day is roughly 3.5 to 4 GB, more for busy days).

    Parts land in ``out_dir/<tag>/parts/``. Existing complete parts are
    skipped and partial ones are resumed with a Range request. With
    ``extract=True`` the parts are streamed straight into
    ``out_dir/<tag>/extracted/`` (no joined tar is written) and that
    directory is returned, ready for :func:`load_readsb_history`. With
    ``extract=False`` the parts directory is returned; extract by hand with::

        cat <tag>.tar.* | tar -x -C extracted/
    """
    tag = adsblol_release_tag(date, variant)
    base = Path(out_dir) / tag
    parts_dir = base / "parts"
    assets = list_adsblol_assets(date, variant)
    if not assets:
        raise FileNotFoundError(f"release {tag} has no assets")
    parts = [_download_asset(a["url"], parts_dir / a["name"], a["size"]) for a in assets]
    if not extract:
        return parts_dir
    return extract_split_tar(parts, base / "extracted")


# --------------------------------------------------------------------------
# OpenSky
# --------------------------------------------------------------------------

_OPENSKY_REQUIRED = ("time", "icao24", "lat", "lon")


def _opensky_frame_to_tracks(df: pd.DataFrame) -> pd.DataFrame:
    """OpenSky state-vector columns -> schema.

    Expects ``time, icao24, lat, lon, velocity (m/s), heading, vertrate
    (m/s), callsign, baroaltitude (m), geoaltitude (m)`` and optionally
    ``typecode`` and ``onground``.
    """
    cols = {c.lower(): c for c in df.columns}
    missing = [c for c in _OPENSKY_REQUIRED if c not in cols]
    if missing:
        raise ValueError(f"OpenSky frame missing columns: {missing}")

    def col(name: str) -> pd.Series | None:
        c = cols.get(name)
        return df[c] if c is not None else None

    n = len(df)
    time_col = df[cols["time"]]
    if pd.api.types.is_datetime64_any_dtype(time_col):
        ts = pd.to_datetime(time_col, utc=True)
        # Resolution-independent (pandas 3 may store us, not ns).
        time = (ts - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(seconds=1)
    else:
        time = pd.to_numeric(time_col, errors="coerce")

    geo = col("geoaltitude")
    baro = col("baroaltitude")
    geo_v = pd.to_numeric(geo, errors="coerce") if geo is not None else pd.Series(np.nan, index=df.index)
    baro_v = pd.to_numeric(baro, errors="coerce") if baro is not None else pd.Series(np.nan, index=df.index)
    use_geo = geo_v.notna()
    alt_m = geo_v.where(use_geo, baro_v)
    alt_source = pd.Series(np.where(use_geo, "geom", "baro"), index=df.index, dtype="string")

    icao = col("icao24")
    icao_s = icao.astype("string").str.strip().str.lower() if icao is not None else pd.Series([""] * n)
    cs = col("callsign")
    tc = col("typecode")

    out = pd.DataFrame(
        {
            "flight_id": icao_s,
            "icao24": icao_s,
            "callsign": cs.astype("string").str.strip() if cs is not None else "",
            "typecode": tc.astype("string").str.strip().str.upper() if tc is not None else "",
            "time": time,
            "lat": pd.to_numeric(col("lat"), errors="coerce"),
            "lon": pd.to_numeric(col("lon"), errors="coerce"),
            "alt_m": alt_m,
            "alt_source": alt_source,
            "gs_mps": pd.to_numeric(col("velocity"), errors="coerce") if col("velocity") is not None else np.nan,
            "track_deg": pd.to_numeric(col("heading"), errors="coerce") if col("heading") is not None else np.nan,
            "vrate_mps": pd.to_numeric(col("vertrate"), errors="coerce") if col("vertrate") is not None else np.nan,
        }
    )
    og = col("onground")
    if og is not None:
        on_ground = og.astype("string").str.strip().str.lower().isin(["true", "1", "t", "yes"])
        out = out.loc[~on_ground.to_numpy()]
    return normalize_tracks(out[list(COLUMNS)])


def load_opensky_csv(path: str | Path) -> pd.DataFrame:
    """OpenSky state-vector export (``.csv``, ``.csv.gz`` or ``.parquet``) -> schema."""
    p = Path(path)
    if p.suffix in (".parquet", ".pq"):
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    return _opensky_frame_to_tracks(df)


def opensky_history(start: Any, stop: Any, bbox: BBox) -> pd.DataFrame:
    """Query OpenSky's Trino history through ``pyopensky`` (optional extra).

    ``bbox = (lat_min, lat_max, lon_min, lon_max)``; ``start``/``stop`` are
    anything ``pyopensky`` accepts (unix seconds, ISO strings, datetimes).
    Credentials come from pyopensky's own config file.
    """
    try:
        from pyopensky.trino import Trino
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "opensky_history needs the optional 'pyopensky' package: "
            "install with `uv sync --extra opensky` (or `pip install pyopensky`) "
            "and configure your OpenSky credentials."
        ) from exc
    lat_min, lat_max, lon_min, lon_max = bbox
    df = Trino().history(start, stop, bounds=(lon_min, lat_min, lon_max, lat_max))
    if df is None or len(df) == 0:
        return empty_tracks()
    return _opensky_frame_to_tracks(df)
