"""README figures from a demo run (needs matplotlib, the ``plots`` extra)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .adsb.aircraft_types import is_known_jet
from .adsb.schema import trajectory_from_rows
from .geo import enu_from_latlon
from .model import curve_times, model_curve, observer_enu_for
from .search import SearchResult

BG = "#0f1117"
FG = "#e6e6e6"
ACCENT = "#ffb347"
ACCENT2 = "#5ec8ff"
TRUTH = "#ff5a5a"


def _style():
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.edgecolor": "#444",
            "axes.labelcolor": FG,
            "xtick.color": FG,
            "ytick.color": FG,
            "text.color": FG,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "legend.facecolor": "#1b1e27",
            "legend.edgecolor": "#444",
        }
    )


def _curve_in_audio_clock(sc, row: pd.Series, flight_col: str, suffix: str, m1, cfg_half: float = 75.0):
    """Modelled curve for a scored row, in (audio seconds, Hz)."""
    fid = str(row[flight_col])
    rows = sc.tracks[sc.tracks["flight_id"] == fid]
    lat0, lon0 = float(rows["lat"].mean()), float(rows["lon"].mean())
    traj = trajectory_from_rows(rows, lat0, lon0)
    obs = observer_enu_for(row["lat"], row["lon"], row["elev_m"], lat0, lon0)
    t_geom = row[f"t_cpa_rx{suffix}"] + curve_times(0.0, cfg_half, 0.25)
    f0 = row["f0_1"] if suffix == "_1" else row["f0_2"]
    f, valid = model_curve(traj, obs, f0, t_geom)
    t_audio = (t_geom - row["t_cpa_rx_1"]) + m1.t_cpa_rx + row["dt_s"]
    return t_audio[valid], f[valid]


def fig_spectrograms(out: Path, sc, spec, m1, m2, res: SearchResult) -> None:
    import matplotlib.pyplot as plt

    best = res.by_flight_pair.iloc[0]
    second = res.by_flight_pair.iloc[1] if len(res.by_flight_pair) > 1 else None
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, m, suffix, title in ((axes[0], m1, "_1", "Event 1"), (axes[1], m2, "_2", "Event 2")):
        t0, t1 = m.t_cpa_rx - 75, m.t_cpa_rx + 75
        v = 1.6 * m.speed_mps
        f_lo, f_hi = m.f0_hz * (343 - v) / (343 + v) - 5, m.f0_hz * (343 + v) / (343 - v) + 5
        ti = (spec.times >= t0) & (spec.times <= t1)
        fi = (spec.freqs >= f_lo) & (spec.freqs <= f_hi)
        img = spec.power_db[np.ix_(fi, ti)]
        vmin, vmax = np.percentile(img, [40, 99.5])
        ax.imshow(img, aspect="auto", origin="lower", extent=[spec.times[ti][0], spec.times[ti][-1], spec.freqs[fi][0], spec.freqs[fi][-1]], cmap="magma", vmin=vmin, vmax=vmax)
        if second is not None:
            t, f = _curve_in_audio_clock(sc, second, f"flight_id{suffix}", suffix, m1)
            ax.plot(t, f, color=ACCENT2, lw=1.2, ls=":", label=f"runner-up ({second['flight_id_1']}+{second['flight_id_2']}), total {second['total']:.1f}")
        t, f = _curve_in_audio_clock(sc, best, f"flight_id{suffix}", suffix, m1)
        ax.plot(t, f, color="white", lw=1.0, ls="--", label=f"best ({best['flight_id_1']}+{best['flight_id_2']}), total {best['total']:.1f}")
        ax.set_xlim(t0, t1)
        ax.set_ylim(f_lo, f_hi)
        ax.set_xlabel("recording time (s)")
        ax.set_ylabel("frequency (Hz)")
        ax.set_title(f"{title}: fitted v = {m.speed_mps:.0f} m/s, CPA {m.cpa_slant_m:.0f} m")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Modelled curves of the best and runner-up candidates over the recording's spectrogram", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "spectrogram_fit.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_map(out: Path, sc, res: SearchResult, cfg_grid: float = 200.0) -> None:
    import matplotlib.pyplot as plt

    lat0, lon0 = sc.observer_lat, sc.observer_lon
    fig, ax = plt.subplots(figsize=(8.5, 8))
    for fid, rows in sc.tracks.groupby("flight_id"):
        e, n = enu_from_latlon(rows["lat"].to_numpy(), rows["lon"].to_numpy(), lat0, lon0)
        jet = is_known_jet(str(rows["typecode"].iloc[0]))
        ax.plot(e / 1000, n / 1000, color="#777" if jet else "#aaa", lw=0.8, ls="--" if jet else "-", alpha=0.7)
        ax.annotate(fid, (e[-1] / 1000, n[-1] / 1000), fontsize=6, color="#999")
    c1 = res.candidates_1
    e, n = enu_from_latlon(c1["lat"].to_numpy(), c1["lon"].to_numpy(), lat0, lon0)
    ax.scatter(e / 1000, n / 1000, s=1, color="#3a4b7a", alpha=0.5, label=f"event-1 speed and distance ring ({len(c1):,} cells)")
    p = res.pairs
    e, n = enu_from_latlon(p["lat"].to_numpy(), p["lon"].to_numpy(), lat0, lon0)
    ax.scatter(e / 1000, n / 1000, s=6, color=ACCENT2, alpha=0.9, label=f"heard both, 920 s apart, silence between ({len(p):,})")
    sc_df = res.scored
    e, n = enu_from_latlon(sc_df["lat"].to_numpy(), sc_df["lon"].to_numpy(), lat0, lon0)
    top = sc_df["total"] >= np.percentile(sc_df["total"], 95) if len(sc_df) else np.zeros(0, bool)
    ax.scatter(e[top] / 1000, n[top] / 1000, s=14, color=ACCENT, alpha=0.9, label="top 5% by curve-shape score")
    b = res.best
    e, n = enu_from_latlon(b["lat"], b["lon"], lat0, lon0)
    ax.scatter([e / 1000], [n / 1000], marker="*", s=260, color="white", edgecolor="black", zorder=5, label="best candidate")
    ax.scatter([0], [0], marker="x", s=90, color=TRUTH, zorder=6, label="true microphone")
    ax.set_xlabel("east of the true microphone (km)")
    ax.set_ylabel("north (km)")
    ax.set_aspect("equal")
    ax.set_xlim(-22, 22)
    ax.set_ylim(-22, 22)
    ax.legend(loc="lower left", fontsize=8, markerscale=1.5)
    ax.set_title("Flight tracks and surviving candidate cells")
    fig.tight_layout()
    fig.savefig(out / "map.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    # Zoomed inset as its own figure.
    fig, ax = plt.subplots(figsize=(6, 5.5))
    r = res.refined if len(res.refined) else res.scored
    e, n = enu_from_latlon(r["lat"].to_numpy(), r["lon"].to_numpy(), lat0, lon0)
    sca = ax.scatter(e, n, c=r["total"], s=38, cmap="viridis", marker="s")
    plt.colorbar(sca, ax=ax, label="score (event 1 + event 2)")
    e, n = enu_from_latlon(b["lat"], b["lon"], lat0, lon0)
    ax.scatter([e], [n], marker="*", s=300, color="white", edgecolor="black", zorder=5, label="best candidate")
    ax.scatter([0], [0], marker="x", s=110, color=TRUTH, zorder=6, label="true microphone")
    ax.set_xlim(-1500, 1500)
    ax.set_ylim(-1500, 1500)
    ax.set_aspect("equal")
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("100 m refinement grid around the best coarse cells")
    fig.tight_layout()
    fig.savefig(out / "refinement.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_scores(out: Path, res: SearchResult) -> None:
    import matplotlib.pyplot as plt

    s = res.scored
    if len(s) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    ax.scatter(s["score_1"], s["score_2"], s=6, color="#8899cc", alpha=0.5, label="every surviving cell")
    bp = res.by_flight_pair
    ax.scatter(bp["score_1"], bp["score_2"], s=30, color=ACCENT, label="best cell per flight pair")
    b = bp.iloc[0]
    ax.scatter([b["score_1"]], [b["score_2"]], marker="*", s=260, color="white", edgecolor="black", zorder=5, label=f"winner: {b['flight_id_1']} + {b['flight_id_2']}")
    ax.set_xlabel("event 1 score")
    ax.set_ylabel("event 2 score (offset locked to event 1)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Curve-shape scores")
    ax = axes[1]
    top = bp.head(12).iloc[::-1]
    labels = [f"{a} + {c}" for a, c in zip(top["flight_id_1"], top["flight_id_2"])]
    ax.barh(labels, top["total"], color=[ACCENT if i == len(top) - 1 else "#5a6a99" for i in range(len(top))])
    ax.set_xlabel("total score")
    ax.set_title("Best cell per flight pair")
    fig.tight_layout()
    fig.savefig(out / "scores.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_hero(out: Path, spec, m1) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_axes([0, 0, 1, 1])
    t0, t1 = m1.t_cpa_rx - 90, m1.t_cpa_rx + 90
    v = 1.6 * m1.speed_mps
    f_lo, f_hi = m1.f0_hz * (343 - v) / (343 + v) - 8, m1.f0_hz * (343 + v) / (343 - v) + 8
    ti = (spec.times >= t0) & (spec.times <= t1)
    fi = (spec.freqs >= f_lo) & (spec.freqs <= f_hi)
    img = spec.power_db[np.ix_(fi, ti)]
    vmin, vmax = np.percentile(img, [35, 99.7])
    ax.imshow(img, aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax, alpha=0.9)
    ax.set_axis_off()
    fig.text(0.05, 0.62, "dopplergeo", fontsize=64, fontweight="bold", color="white", ha="left", va="center")
    fig.text(0.05, 0.40, "Where was this recorded? Ask the planes that flew over.", fontsize=24, color="#f3f3f3", ha="left", va="center")
    fig.text(0.05, 0.28, "Doppler curves in the audio, matched against ADS-B flight tracks.", fontsize=17, color="#d0d0d0", ha="left", va="center")
    fig.savefig(out / "hero.png", dpi=100)
    plt.close(fig)


def fig_delay_matters(out: Path, sc, m1, res: SearchResult) -> None:
    """The naive (receive-time geometry) curve versus the retarded-time curve."""
    import matplotlib.pyplot as plt

    from .acoustics import doppler_frequency

    best = res.by_flight_pair.iloc[0]
    fid = str(best["flight_id_1"])
    rows = sc.tracks[sc.tracks["flight_id"] == fid]
    lat0, lon0 = float(rows["lat"].mean()), float(rows["lon"].mean())
    traj = trajectory_from_rows(rows, lat0, lon0)
    obs = observer_enu_for(best["lat"], best["lon"], best["elev_m"], lat0, lon0)
    t_geom = best["t_cpa_rx_1"] + curve_times(0.0, 75.0, 0.25)
    f_right, valid = model_curve(traj, obs, best["f0_1"], t_geom)
    rel = traj.position(t_geom) - obs
    vr = np.einsum("ij,ij->i", rel, traj.velocity(t_geom)) / np.linalg.norm(rel, axis=1)
    f_naive = doppler_frequency(best["f0_1"], vr)
    t_audio = (t_geom - best["t_cpa_rx_1"]) + m1.t_cpa_rx + best["dt_s"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_audio[valid], f_naive[valid], color=ACCENT2, lw=1.6, ls=":", label="radial velocity at receive time (wrong)")
    ax.plot(t_audio[valid], f_right[valid], color="white", lw=1.6, label="radial velocity at emission time (right)")
    ax.set_xlabel("recording time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.legend(fontsize=9)
    ax.set_title("Sound takes seconds to arrive, and the delay changes during the pass")
    fig.tight_layout()
    fig.savefig(out / "delay.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def make_figures(out: Path, sc, spec, spec_w, m1, m2, res: SearchResult, res_single: SearchResult | None = None) -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("[demo] matplotlib not installed; skipping figures (uv sync --extra plots)")
        return
    _style()
    fig_hero(out, spec, m1)
    fig_spectrograms(out, sc, spec, m1, m2, res)
    fig_map(out, sc, res)
    fig_scores(out, res)
    fig_delay_matters(out, sc, m1, res)
