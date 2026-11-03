"""Search orchestration: from measured curves and ADS-B tracks to a ranked
list of candidate recording locations.

Two modes:

* :func:`two_event_search` – two Doppler curves in the same recording, an
  unknown recording time. The inter-event offset, the per-event speed and
  slant-distance ranges, the silence between them, and finally the full
  modelled curve shape narrow the search.
* :func:`single_event_search` – one curve and a known recording time (within
  some tolerance).

Both end with the same physics-based scoring and a 100 m refinement pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .acoustics import Trajectory
from .adsb.aircraft_types import filter_out_jets
from .adsb.schema import normalize_tracks, trajectory_from_rows
from .adsb.tracks import split_on_gaps
from .candidates import EventSpec, generate_candidates
from .constants import COARSE_GRID_M, FINE_GRID_M, SPEED_OF_SOUND_MPS
from .curve_fit import CurveMeasurement
from .dem import ElevationProvider
from .filters import pair_by_offset, silence_filter
from .geo import grid_cell, grid_cell_center, grid_points
from .model import observer_enu_for, t_cpa_rx_for
from .scoring import ScoreConfig, score_candidates, score_single


@dataclass
class SearchConfig:
    grid_m: float = COARSE_GRID_M
    fine_grid_m: float = FINE_GRID_M
    # Extra tolerance on the inter-event offset beyond what the curve fits
    # report; measuring "the centre" of a two-minute curve is a few seconds
    # of slop either way.
    offset_tol_s: float = 5.0
    silence_radius_m: float = 4000.0
    silence_margin_s: float = 30.0
    exclude_jets: bool = True
    speed_margin_mps: float = 0.0
    # How many coarse (cell, flight pair) rows to refine to the fine grid.
    refine_top_k: int = 50
    # Fine refinement covers the coarse cell plus one coarse cell of margin.
    refine_margin_cells: int = 1
    c: float = SPEED_OF_SOUND_MPS
    score: ScoreConfig = field(default_factory=ScoreConfig)


@dataclass
class SearchResult:
    """Everything a search produced, coarsest to finest."""

    candidates_1: pd.DataFrame
    candidates_2: pd.DataFrame | None
    pairs: pd.DataFrame  # after the offset + silence filters, before scoring
    scored: pd.DataFrame  # coarse grid, scored
    refined: pd.DataFrame  # fine grid around the best coarse rows, scored
    by_flight_pair: pd.DataFrame  # best row per flight pair, descending

    @property
    def best(self) -> pd.Series:
        if len(self.by_flight_pair) == 0:
            raise LookupError(
                "no candidate survived the filters; widen the measurement ranges, "
                "the offset tolerance, or the silence radius"
            )
        return self.by_flight_pair.iloc[0]

    def summary(self, n: int = 5) -> str:
        cols = [c for c in ("lat", "lon", "flight_id_1", "flight_id_2", "flight_id", "score_1", "score_2", "total", "score") if c in self.by_flight_pair.columns]
        return self.by_flight_pair[cols].head(n).to_string(index=False)


# ---------------------------------------------------------------------------
# Track preparation
# ---------------------------------------------------------------------------


def prepare_tracks(tracks: pd.DataFrame, *, exclude_jets: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(audible_sources, all_flights)``.

    ``all_flights`` keeps everything (for the silence filter: a jet is still a
    sound). ``audible_sources`` drops known jets when requested, since the
    measured curves come from propeller aircraft. Both are gap-split.
    """
    all_tracks = split_on_gaps(normalize_tracks(tracks))
    sources = filter_out_jets(all_tracks) if exclude_jets else all_tracks
    return sources, all_tracks


def flight_frames(tracks: pd.DataFrame, flight_ids: list[str]) -> dict[str, tuple[Trajectory, float, float]]:
    """Trajectory per flight, each in an ENU frame about its own centroid.

    Returns ``flight_id -> (trajectory, lat0, lon0)``.
    """
    out: dict[str, tuple[Trajectory, float, float]] = {}
    sub = tracks[tracks["flight_id"].isin(flight_ids)]
    for fid, rows in sub.groupby("flight_id", sort=False):
        lat0 = float(rows["lat"].mean())
        lon0 = float(rows["lon"].mean())
        out[str(fid)] = (trajectory_from_rows(rows, lat0, lon0), lat0, lon0)
    return out


# ---------------------------------------------------------------------------
# Two-event search
# ---------------------------------------------------------------------------


def _event_spec(m: CurveMeasurement, margin: float) -> EventSpec:
    lo, hi = m.speed_range
    return EventSpec(speed_range=(max(1.0, lo - margin), hi + margin), cpa_slant_range=tuple(m.cpa_slant_range))


def _offset_tolerance(m1: CurveMeasurement, m2: CurveMeasurement, cfg: SearchConfig) -> float:
    h1 = 0.5 * (m1.t_cpa_range[1] - m1.t_cpa_range[0])
    h2 = 0.5 * (m2.t_cpa_range[1] - m2.t_cpa_range[0])
    return float(max(cfg.offset_tol_s, h1 + h2))


def _score_pairs_grouped(spec_w, pairs: pd.DataFrame, tracks: pd.DataFrame, m1, m2, cfg: SearchConfig) -> pd.DataFrame:
    """Score every row of ``pairs``, grouping by flight pair so each group
    uses the local ENU frame of its first flight."""
    if len(pairs) == 0:
        return pairs.assign(score_1=[], score_2=[], dt_s=[], f0_1=[], f0_2=[], total=[])
    fids = sorted(set(pairs["flight_id_1"]) | set(pairs["flight_id_2"]))
    frames = flight_frames(tracks, fids)
    out = []
    for (f1, f2), grp in pairs.groupby(["flight_id_1", "flight_id_2"], sort=False):
        traj1, lat0, lon0 = frames[str(f1)]
        traj2 = trajectory_from_rows(tracks[tracks["flight_id"] == f2], lat0, lon0)
        trajs = {str(f1): traj1, str(f2): traj2}
        out.append(score_candidates(spec_w, grp, trajs, lat0, lon0, m1, m2, cfg.score, c=cfg.c))
    return pd.concat(out, ignore_index=True)


def _fine_points_around(rows: pd.DataFrame, cfg: SearchConfig, dem: ElevationProvider) -> pd.DataFrame:
    """Fine-grid cells covering each coarse cell in ``rows`` plus a margin.

    Returns one row per (fine cell, parent row index) with lat/lon/elev_m and
    the parent's flight columns copied over.
    """
    m_lat = np.deg2rad(1.0) * 6_371_008.8
    dlat = cfg.grid_m / m_lat
    out = []
    for idx, r in rows.iterrows():
        row_lat = (r["row"] + 0.5) * dlat
        dlon = cfg.grid_m / (m_lat * np.cos(np.deg2rad(row_lat)))
        k = cfg.refine_margin_cells + 0.5
        lat_min = (r["row"] + 0.5 - k) * dlat + 1e-9
        lat_max = (r["row"] + 0.5 + k) * dlat - 1e-9
        lon_c = (r["col"] + 0.5) * dlon
        lon_c = (lon_c + 180.0) % 360.0 - 180.0
        lon_min = lon_c - k * dlon + 1e-9
        lon_max = lon_c + k * dlon - 1e-9
        frow, fcol, flat, flon = grid_points(lat_min, lat_max, lon_min, lon_max, cfg.fine_grid_m)
        df = pd.DataFrame({"row": frow, "col": fcol, "lat": flat, "lon": flon})
        df["elev_m"] = np.asarray(dem.elevation_m(flat, flon), dtype=float)
        df["parent"] = idx
        for c in rows.columns:
            if c.startswith("flight_id") or c in ("cpa_slant_1", "cpa_slant_2", "speed_1", "speed_2", "cpa_slant_m", "speed_mps"):
                df[c] = r[c]
        out.append(df)
    if not out:
        return pd.DataFrame(columns=["row", "col", "lat", "lon", "elev_m", "parent"])
    fine = pd.concat(out, ignore_index=True)
    # The margins of neighbouring parents overlap; keep one copy per fine cell
    # and flight combination.
    key = ["row", "col"] + [c for c in fine.columns if c.startswith("flight_id")]
    return fine.drop_duplicates(subset=key).reset_index(drop=True)


def _recompute_cpas(fine: pd.DataFrame, tracks: pd.DataFrame, cfg: SearchConfig, two_events: bool) -> pd.DataFrame:
    """Fill t_cpa_tx/t_cpa_rx/slant/speed for fine points from the geometry."""
    fine = fine.copy()
    id_cols = ["flight_id_1", "flight_id_2"] if two_events else ["flight_id"]
    fids = sorted({str(v) for c in id_cols for v in fine[c]})
    frames = flight_frames(tracks, fids)
    suffixes = ["_1", "_2"] if two_events else [""]
    for col, suf in zip(id_cols, suffixes):
        t_tx = np.zeros(len(fine))
        t_rx = np.zeros(len(fine))
        slant = np.zeros(len(fine))
        speed = np.zeros(len(fine))
        for i, (fid, lat, lon, elev) in enumerate(zip(fine[col], fine["lat"], fine["lon"], fine["elev_m"])):
            traj, lat0, lon0 = frames[str(fid)]
            obs = observer_enu_for(lat, lon, elev, lat0, lon0)
            t_tx[i], slant[i], t_rx[i] = t_cpa_rx_for(traj, obs, cfg.c)
            speed[i] = float(traj.speed(t_tx[i]))
        fine[f"t_cpa_tx{suf}"] = t_tx
        fine[f"t_cpa_rx{suf}"] = t_rx
        fine["cpa_slant" + (suf if two_events else "_m")] = slant
        fine["speed" + (suf if two_events else "_mps")] = speed
    if two_events:
        fine["offset_s"] = fine["t_cpa_rx_2"] - fine["t_cpa_rx_1"]
    return fine


def _best_per_flight_pair(df: pd.DataFrame, keys: list[str], score_col: str) -> pd.DataFrame:
    if len(df) == 0:
        return df
    idx = df.groupby(keys, sort=False)[score_col].idxmax()
    return df.loc[idx].sort_values(score_col, ascending=False).reset_index(drop=True)


def two_event_search(
    spec_w,
    tracks: pd.DataFrame,
    m1: CurveMeasurement,
    m2: CurveMeasurement,
    dem: ElevationProvider,
    cfg: SearchConfig | None = None,
    *,
    log=print,
) -> SearchResult:
    """Locate a recording containing two Doppler curves.

    ``spec_w`` is the whitened spectrogram of the recording; ``m1``/``m2`` the
    fitted measurements of the two curves (audio clock, ``m1`` first).
    """
    cfg = cfg or SearchConfig()
    sources, all_tracks = prepare_tracks(tracks, exclude_jets=cfg.exclude_jets)
    offset = m2.t_cpa_rx - m1.t_cpa_rx
    tol = _offset_tolerance(m1, m2, cfg)
    log(f"[search] {sources['flight_id'].nunique()} candidate flights, {all_tracks['flight_id'].nunique()} total; offset {offset:.1f} s ± {tol:.1f} s")

    cand1 = generate_candidates(sources, _event_spec(m1, cfg.speed_margin_mps), dem, grid_m=cfg.grid_m, c=cfg.c)
    cand2 = generate_candidates(sources, _event_spec(m2, cfg.speed_margin_mps), dem, grid_m=cfg.grid_m, c=cfg.c)
    log(f"[search] event 1: {len(cand1)} (cell, flight) candidates; event 2: {len(cand2)}")

    pairs = pair_by_offset(cand1, cand2, offset, tol)
    log(f"[search] offset filter: {len(pairs)} pairs in {pairs[['row', 'col']].drop_duplicates().shape[0] if len(pairs) else 0} cells")
    pairs = silence_filter(pairs, all_tracks, dem, radius_m=cfg.silence_radius_m, margin_s=cfg.silence_margin_s, c=cfg.c)
    log(f"[search] silence filter: {len(pairs)} pairs")

    scored = _score_pairs_grouped(spec_w, pairs, sources, m1, m2, cfg)
    scored = scored.sort_values("total", ascending=False).reset_index(drop=True)
    log(f"[search] scored {len(scored)} coarse candidates")

    top = scored.head(cfg.refine_top_k)
    fine = _fine_points_around(top, cfg, dem)
    if len(fine):
        fine = _recompute_cpas(fine, sources, cfg, two_events=True)
        refined = _score_pairs_grouped(spec_w, fine, sources, m1, m2, cfg)
        refined = refined.sort_values("total", ascending=False).reset_index(drop=True)
    else:
        refined = fine
    log(f"[search] refined {len(refined)} fine candidates")

    merged = pd.concat([refined, scored], ignore_index=True) if len(refined) else scored
    by_pair = _best_per_flight_pair(merged, ["flight_id_1", "flight_id_2"], "total")
    return SearchResult(cand1, cand2, pairs, scored, refined, by_pair)


# ---------------------------------------------------------------------------
# Single-event search with a known time
# ---------------------------------------------------------------------------


def single_event_search(
    spec_w,
    tracks: pd.DataFrame,
    m: CurveMeasurement,
    dem: ElevationProvider,
    t_known_rx_geom: float,
    tol_s: float,
    cfg: SearchConfig | None = None,
    *,
    log=print,
) -> SearchResult:
    """Locate a recording from one curve whose CPA was heard at
    ``t_known_rx_geom`` (unix seconds) give or take ``tol_s``."""
    cfg = cfg or SearchConfig()
    sources, _all = prepare_tracks(tracks, exclude_jets=cfg.exclude_jets)
    window = tol_s + cfg.score.half_window_s + 60.0
    near = sources[(sources["time"] >= t_known_rx_geom - window) & (sources["time"] <= t_known_rx_geom + window)]
    keep = near["flight_id"].unique()
    sources = sources[sources["flight_id"].isin(keep)]
    log(f"[search] {len(keep)} flights near {t_known_rx_geom:.0f} ± {tol_s:.0f} s")

    cand = generate_candidates(sources, _event_spec(m, cfg.speed_margin_mps), dem, grid_m=cfg.grid_m, c=cfg.c)
    cand = cand[np.abs(cand["t_cpa_rx"] - t_known_rx_geom) <= tol_s].reset_index(drop=True)
    log(f"[search] {len(cand)} (cell, flight) candidates in the time window")

    # The known time only gates which (cell, flight) rows are considered; the
    # curve is still aligned by its own CPA with the usual small sweep, since
    # "about 20:37" is not a clock the spectrogram can be pinned to.
    score_cfg = cfg.score

    def _score(df: pd.DataFrame) -> pd.DataFrame:
        if len(df) == 0:
            return df.assign(score=[], dt_s=[], f0=[])
        frames = flight_frames(sources, sorted(set(df["flight_id"])))
        parts = []
        for fid, grp in df.groupby("flight_id", sort=False):
            traj, lat0, lon0 = frames[str(fid)]
            parts.append(score_single(spec_w, grp, {str(fid): traj}, lat0, lon0, m, score_cfg, c=cfg.c))
        return pd.concat(parts, ignore_index=True)

    scored = _score(cand).sort_values("score", ascending=False).reset_index(drop=True)
    top = scored.head(cfg.refine_top_k)
    fine = _fine_points_around(top, cfg, dem)
    if len(fine):
        fine = _recompute_cpas(fine, sources, cfg, two_events=False)
        refined = _score(fine).sort_values("score", ascending=False).reset_index(drop=True)
    else:
        refined = fine
    merged = pd.concat([refined, scored], ignore_index=True) if len(refined) else scored
    by_flight = _best_per_flight_pair(merged, ["flight_id"], "score")
    return SearchResult(cand, None, cand, scored, refined, by_flight)


__all__ = [
    "SearchConfig",
    "SearchResult",
    "prepare_tracks",
    "flight_frames",
    "two_event_search",
    "single_event_search",
    "grid_cell",
    "grid_cell_center",
]
