# dopplergeo design

Pipeline for locating a recording from the Doppler curves of propeller
aircraft heard in it, matched against ADS-B flight tracks.

## Conventions (binding for every module)

* Python 3.12, numpy/scipy/pandas/pyarrow/soundfile only. `rasterio`,
  `matplotlib`, `pyopensky` are optional extras and must be imported lazily.
* Units: metres, seconds, Hz, degrees. Times are unix seconds (UTC) for
  ADS-B ("geometry clock") and seconds from the start of the file for audio
  ("audio clock"). Never mix them without an explicit offset.
* Altitudes are metres above mean sea level for aircraft **and** for
  observers (ground elevation from a DEM). The slant distance is the 3-D
  distance between the two.
* `v_radial` is positive when the source recedes; `f = f0 c / (c + v_radial)`.
* `t_cpa_tx` = emission time of the closest point of approach.
  `t_cpa_rx = t_cpa_tx + d_cpa / c` = when the observer *hears* the CPA
  (the moment the received frequency crosses `f0`).
* Candidate points live on the global grid from `geo.grid_cell` /
  `geo.grid_points`, identified by integer `(row, col)` at a given spacing.
  Two flights that cover the same place produce the same `(row, col)`.
* Dataframes use the column names given below exactly.

## Shared foundation (done)

* `constants.py` – `SPEED_OF_SOUND_MPS`, grid spacings, unit factors.
* `geo.py` – `enu_from_latlon`, `latlon_from_enu`, `haversine_m`,
  `slant_distance_m`, `grid_cell`, `grid_cell_center`, `grid_points`.
* `acoustics.py` – `doppler_frequency`, `ideal_curve(t_rx, f0, v, d, t_cpa_rx)`,
  `Trajectory` (cubic-spline ENU path with `.position(t)`, `.velocity(t)`,
  `.speed(t)`), `solve_emission_times`, `radial_velocity`,
  `received_frequency(traj, obs_enu, t_rx, f0) -> (freq, valid)`,
  `closest_point_of_approach(traj, obs_enu) -> (t_cpa_tx, slant_m)`.
* `adsb/schema.py` – canonical track table (`COLUMNS`), `normalize_tracks`,
  `validate_tracks`, `read_tracks`/`write_tracks` (parquet), `flight_bounds`,
  `trajectory_from_rows(rows, lat0, lon0)`.
* `synth.py` – `straight_track(...)` (can turn/accelerate/climb) and
  `render_audio(sources, observer_enu, t_start, duration_s, sample_rate, ...)`.

## Module contracts

### `spectrogram.py`

```python
def load_audio(path, target_sr: int = 8000) -> tuple[np.ndarray, int]  # mono float64

@dataclass
class Spectrogram:
    times: np.ndarray      # frame centres, audio clock (s)
    freqs: np.ndarray      # Hz
    power_db: np.ndarray   # shape (n_freq, n_time), log magnitude
    def whitened(self) -> "Spectrogram"   # per-frequency-bin robust z-score, clipped at >= 0
    def sample(self, t, f) -> tuple[np.ndarray, np.ndarray]  # bilinear values + in-bounds mask (0 outside)

def compute_spectrogram(audio, sample_rate, *, f_max=600.0, freq_resolution_hz=0.5,
                        hop_s=0.25) -> Spectrogram

@dataclass
class Ridge:
    times: np.ndarray; freqs: np.ndarray; strength: np.ndarray

def extract_ridge(spec: Spectrogram, t_start, t_end, f_min, f_max,
                  *, max_slope_hz_per_s=3.0, jump_penalty=4.0,
                  already_whitened=False) -> Ridge   # Viterbi over whitened power
```
The jump penalty is quadratic in the per-frame frequency step and reaches
``jump_penalty`` whitened sigmas at the slope limit; set the limit from the
expected pass (``f0 v^2 / (c d)`` at the closest point, about 1–7 Hz/s).

### `curve_fit.py`

```python
@dataclass
class CurveMeasurement:
    f0_hz: float; speed_mps: float; cpa_slant_m: float; t_cpa_rx: float   # audio clock
    rmse_hz: float
    speed_range: tuple[float, float]; cpa_slant_range: tuple[float, float]
    t_cpa_range: tuple[float, float]; f0_range: tuple[float, float]
    def to_dict(self) -> dict; @classmethod from_dict(cls, d)

def fit_ideal_curve(ridge: Ridge, *, c=343.0, speed_bounds=(15.0, 250.0),
                    slant_bounds=(50.0, 10_000.0), min_speed_halfwidth=15.0,
                    min_slant_halfwidth=300.0, min_time_halfwidth=3.0,
                    min_f0_halfwidth=2.0, sigma_mult=3.0) -> CurveMeasurement
```
Multi-start `scipy.optimize.least_squares` of `acoustics.ideal_curve` against
the ridge. Robust by default (`robust=True, robust_scale_hz=1.0`): a Cauchy
loss on the whole ridge and on three sub-windows, the fit with the lowest
truncated-quadratic cost over the whole ridge wins, then a plain least
squares on its inliers gives the values and the covariance. This survives a
ridge that wandered onto a neighbouring tone for the start or the end of the
box. Ranges = fit ± max(min halfwidth, sigma_mult × std from the Jacobian
covariance). Generous on purpose: a wrong range silently drops the true
answer.

### `adsb/aircraft_types.py`, `adsb/sources.py`, `adsb/tracks.py`

```python
def engine_class(typecode: str) -> str   # "jet" | "turboprop" | "piston" | "electric" | "unknown"
def is_known_jet(typecode: str) -> bool
def filter_out_jets(tracks: pd.DataFrame) -> pd.DataFrame
def load_type_table(csv_path) -> None     # OpenSky aircraftDatabase.csv or ICAO-8643-style CSV

def parse_readsb_trace(obj_or_path) -> pd.DataFrame        # one readsb trace_full json -> schema rows
def load_readsb_history(root, *, bbox=None, time_range=None) -> pd.DataFrame
def download_adsblol_day(date: str, out_dir, *, extract=True) -> Path
def load_opensky_csv(path) -> pd.DataFrame
def opensky_history(start, stop, bbox) -> pd.DataFrame      # optional pyopensky

def split_on_gaps(tracks, max_gap_s=60.0) -> pd.DataFrame   # new flight_id "<icao24>_<k>"
def resample_tracks(tracks, step_s=1.0) -> pd.DataFrame
def add_speed_3d(tracks) -> pd.DataFrame                    # column "speed_mps"
def clip_bbox(tracks, lat_min, lat_max, lon_min, lon_max) -> pd.DataFrame
def clip_time(tracks, t0, t1) -> pd.DataFrame
```

### `dem.py`, `candidates.py`, `filters.py`

```python
class ElevationProvider(Protocol):
    def elevation_m(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray

class FlatElevation(z_m)                       # tests
class ArrayElevation(lat_axis, lon_axis, grid) # bilinear
class RasterElevation(path)                    # rasterio, lazy import
class TerrainTilesElevation(zoom=11, cache_dir=...)  # AWS terrain tiles geotiff, rasterio + network

@dataclass
class EventSpec:
    speed_range: tuple[float, float]
    cpa_slant_range: tuple[float, float]

def generate_candidates(tracks, spec: EventSpec, dem, *, grid_m=200.0, c=343.0,
                        speed_margin_mps=0.0) -> pd.DataFrame
# columns: row, col, lat, lon, elev_m, flight_id, t_cpa_tx, t_cpa_rx, cpa_slant_m, speed_mps
# One row per (grid cell, flight) whose true CPA (interior to the track, not an endpoint)
# has slant distance inside cpa_slant_range and 3-D speed inside speed_range.

def cpas_for_points(tracks, points, dem, *, max_slant_m=10_000.0, c=343.0) -> pd.DataFrame
# points: DataFrame with row, col, lat, lon (elev_m optional) -> every (point, flight) CPA within max_slant_m
# columns: row, col, flight_id, t_cpa_tx, t_cpa_rx, cpa_slant_m, speed_mps

def pair_by_offset(cand1, cand2, offset_s, tol_s) -> pd.DataFrame
# join on (row, col); keep t_cpa_rx_2 - t_cpa_rx_1 within offset_s ± tol_s; flight_id_1 != flight_id_2
# columns: row, col, lat, lon, elev_m, flight_id_1, t_cpa_tx_1, t_cpa_rx_1, cpa_slant_1, speed_1,
#          flight_id_2, t_cpa_tx_2, t_cpa_rx_2, cpa_slant_2, speed_2, offset_s

def silence_filter(pairs, all_tracks, dem, *, radius_m=4000.0, margin_s=30.0, c=343.0) -> pd.DataFrame
# drop pairs where any flight other than flight_id_1/flight_id_2 has a CPA within radius_m
# whose t_cpa_rx lies in (t_cpa_rx_1 + margin_s, t_cpa_rx_2 - margin_s)
```

### `model.py`, `scoring.py`

```python
def model_curve(traj, observer_enu, f0, t_rx) -> tuple[np.ndarray, np.ndarray]   # received_frequency

@dataclass
class ScoreConfig:
    half_window_s: float = 60.0      # curve extent each side of CPA
    curve_step_s: float = 0.25
    time_sweep_s: float = 3.0        # ± sweep of the audio/geometry alignment
    time_step_s: float = 0.25
    f0_sweep_hz: float = 1.5         # ± sweep of the emitted frequency
    f0_step_hz: float = 0.25
    harmonics: tuple[float, ...] = (1.0, 0.5)   # weights for k = 1..K multiples of f0
    clip_sigma: float | None = None  # optional brightness clip before averaging

@dataclass
class EventScore:
    score: float; dt_s: float; f0_hz: float

def score_event(spec_w, traj, observer_enu, *, f0, t_cpa_rx_audio, t_cpa_rx_geom,
                cfg, lock_dt: float | None = None) -> EventScore
# audio_time = (t_rx_geom - t_cpa_rx_geom) + t_cpa_rx_audio + dt
# score = mean whitened brightness along the modelled curve (harmonic-weighted),
# maximised over dt (unless lock_dt is given) and f0.

def score_pair(spec_w, traj1, traj2, observer_enu, *, m1, m2, t_cpa_rx_geom_1, t_cpa_rx_geom_2,
               cfg) -> PairScore(score_1, score_2, dt_s, f0_1, f0_2, total)
# Event 1 picks dt. Event 2 is placed with the SAME dt (locked): its audio time is
# (t_rx_geom_2 - t_cpa_rx_geom_1) + m1.t_cpa_rx + dt, with only its own f0 swept.
```

### `search.py` (integration), `cli.py`, `demo.py`

* `two_event_search(spec_w, tracks, m1, m2, dem, cfg)` → `SearchResult` with
  the candidate tables of each stage, the coarse scores, the 100 m refinement
  and the best row per flight pair.
* `single_event_search(spec_w, tracks, m, dem, t_known_rx_geom, tol_s, cfg)`:
  the known time only gates which (cell, flight) rows are considered; each
  curve is still aligned by its own CPA with the usual small sweep. What is
  left is an along-track ambiguity of about `tol_s × speed` that only the
  flight's own turns can break.
* Each flight pair is scored in the ENU frame of its first flight, so a
  nationwide search never relies on one tangent plane.
* `demo.py` builds a synthetic region, renders audio with `synth.py`, and
  runs both searches; `figures.py` draws the README figures from the result.

### Environment note

Python 3.12+ skips `.pth` files that carry the macOS hidden flag, and
dotfolders under `~/Desktop` on the author's machine keep acquiring that
flag, which silently breaks the editable install. The `justfile` therefore
puts the virtualenv in `~/.venvs/dopplergeo` (`UV_PROJECT_ENVIRONMENT`).
