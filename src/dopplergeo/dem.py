"""Ground elevation providers.

Observers sit on the ground, and the slant distance to an aircraft is the 3-D
distance between the two, so every candidate grid point needs its elevation
above mean sea level. All providers answer ``elevation_m(lat, lon)`` with
numpy broadcasting and never raise for points outside their coverage: they
clamp to the edge of what they have.

* :class:`FlatElevation` – a constant, for tests and flat sites.
* :class:`ArrayElevation` – a lat/lon grid in memory, bilinear.
* :class:`RasterElevation` – any GeoTIFF through ``rasterio`` (optional extra).
* :class:`TerrainTilesElevation` – AWS Terrain Tiles, downloaded lazily and
  cached on disk (``rasterio`` + network).

``rasterio`` is imported lazily inside the classes that need it.
"""

from __future__ import annotations

import math
import os
import tempfile
import urllib.error
import urllib.request
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from scipy.interpolate import RegularGridInterpolator

TERRAIN_TILES_URL = "https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{z}/{x}/{y}.tif"
TERRAIN_TILE_PX = 512
DEFAULT_TERRAIN_CACHE = "~/.cache/dopplergeo/terrain"

# Web Mercator (EPSG:3857) sphere radius and world extent in metres.
_WEB_MERCATOR_R = 6378137.0
_WEB_MERCATOR_EXTENT = 2.0 * math.pi * _WEB_MERCATOR_R
_MAX_MERCATOR_LAT = 85.05112878


@runtime_checkable
class ElevationProvider(Protocol):
    def elevation_m(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Ground elevation above mean sea level (m); broadcasts over inputs."""
        ...


def _broadcast(lat: Any, lon: Any) -> tuple[np.ndarray, np.ndarray]:
    lat_a, lon_a = np.broadcast_arrays(np.asarray(lat, dtype=float), np.asarray(lon, dtype=float))
    return lat_a, lon_a


def _bilinear(arr: np.ndarray, ri: np.ndarray, ci: np.ndarray) -> np.ndarray:
    """Sample ``arr[row, col]`` at fractional indices with edge clamping."""
    h, w = arr.shape
    ri = np.clip(ri, 0.0, h - 1.0)
    ci = np.clip(ci, 0.0, w - 1.0)
    r0 = np.floor(ri).astype(np.int64)
    c0 = np.floor(ci).astype(np.int64)
    r1 = np.minimum(r0 + 1, h - 1)
    c1 = np.minimum(c0 + 1, w - 1)
    fr = ri - r0
    fc = ci - c0
    top = arr[r0, c0] * (1.0 - fc) + arr[r0, c1] * fc
    bot = arr[r1, c0] * (1.0 - fc) + arr[r1, c1] * fc
    return top * (1.0 - fr) + bot * fr


def _nearest(arr: np.ndarray, ri: np.ndarray, ci: np.ndarray) -> np.ndarray:
    h, w = arr.shape
    r = np.clip(np.rint(ri), 0, h - 1).astype(np.int64)
    c = np.clip(np.rint(ci), 0, w - 1).astype(np.int64)
    return arr[r, c]


def _fill_nodata(arr: np.ndarray, nodata: float | None, what: str) -> np.ndarray:
    """Replace nodata (and NaN) with 0 m, warning once per call."""
    out = np.asarray(arr, dtype=np.float64)
    mask = ~np.isfinite(out)
    if nodata is not None and np.isfinite(nodata):
        mask |= out == nodata
    n_bad = int(mask.sum())
    if n_bad:
        warnings.warn(
            f"{what}: {n_bad} nodata pixel(s) replaced with 0 m",
            RuntimeWarning,
            stacklevel=3,
        )
        out = np.where(mask, 0.0, out)
    return out


# --------------------------------------------------------------------------
# In-memory providers
# --------------------------------------------------------------------------


class FlatElevation:
    """Constant elevation everywhere."""

    def __init__(self, z_m: float = 0.0) -> None:
        self.z_m = float(z_m)

    def elevation_m(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        lat_a, _ = _broadcast(lat, lon)
        return np.full(lat_a.shape, self.z_m, dtype=np.float64)


class ArrayElevation:
    """Bilinear interpolation over a regular lat/lon grid.

    ``grid`` has shape ``(len(lat_axis), len(lon_axis))``. Axes may be
    ascending or descending; queries outside the grid clamp to its edge.
    """

    def __init__(self, lat_axis: np.ndarray, lon_axis: np.ndarray, grid: np.ndarray) -> None:
        lat_axis = np.asarray(lat_axis, dtype=np.float64)
        lon_axis = np.asarray(lon_axis, dtype=np.float64)
        grid = np.asarray(grid, dtype=np.float64)
        if grid.shape != (lat_axis.size, lon_axis.size):
            raise ValueError(
                f"grid shape {grid.shape} does not match axes "
                f"({lat_axis.size}, {lon_axis.size})"
            )
        if lat_axis.size < 2 or lon_axis.size < 2:
            raise ValueError("each axis needs at least two samples")
        if lat_axis[0] > lat_axis[-1]:
            lat_axis = lat_axis[::-1]
            grid = grid[::-1, :]
        if lon_axis[0] > lon_axis[-1]:
            lon_axis = lon_axis[::-1]
            grid = grid[:, ::-1]
        if not (np.all(np.diff(lat_axis) > 0) and np.all(np.diff(lon_axis) > 0)):
            raise ValueError("axes must be strictly monotonic")
        self.lat_axis = lat_axis
        self.lon_axis = lon_axis
        self.grid = np.ascontiguousarray(grid)
        self._interp = RegularGridInterpolator(
            (lat_axis, lon_axis), self.grid, method="linear", bounds_error=False, fill_value=None
        )

    def elevation_m(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        lat_a, lon_a = _broadcast(lat, lon)
        lat_c = np.clip(lat_a, self.lat_axis[0], self.lat_axis[-1])
        lon_c = np.clip(lon_a, self.lon_axis[0], self.lon_axis[-1])
        pts = np.column_stack([lat_c.ravel(), lon_c.ravel()])
        return np.asarray(self._interp(pts), dtype=np.float64).reshape(lat_a.shape)


# --------------------------------------------------------------------------
# rasterio-backed providers
# --------------------------------------------------------------------------


def _require_rasterio() -> Any:
    try:
        import rasterio  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "rasterio is required for raster DEMs; install with `uv sync --extra dem`"
        ) from exc
    return rasterio


def _pixel_index(transform: Any, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fractional (row, col) *centre* indices of dataset coordinates."""
    inv = ~transform
    col = inv.a * xs + inv.b * ys + inv.c
    row = inv.d * xs + inv.e * ys + inv.f
    # GDAL transforms map pixel corners; pixel (i, j) is centred at i + 0.5.
    return row - 0.5, col - 0.5


class RasterElevation:
    """Elevation from a GeoTIFF (or anything rasterio opens).

    Small rasters are read into memory once; larger ones are sampled from
    disk per query (a window read around the queried points, falling back to
    ``dataset.sample`` for very spread-out queries). Nodata pixels become 0 m
    with a warning. Any CRS works: lat/lon queries are transformed into the
    raster's CRS with ``rasterio.warp.transform``.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        max_pixels_in_memory: int = 16_000_000,
        method: str = "bilinear",
    ) -> None:
        rasterio = _require_rasterio()
        if method not in ("bilinear", "nearest"):
            raise ValueError("method must be 'bilinear' or 'nearest'")
        self.path = Path(path)
        self.method = method
        self._max_pixels = int(max_pixels_in_memory)
        self._ds = rasterio.open(self.path)
        self.crs = self._ds.crs
        self.transform = self._ds.transform
        self.nodata = self._ds.nodata
        self.width = int(self._ds.width)
        self.height = int(self._ds.height)
        self._array: np.ndarray | None = None
        if self.width * self.height <= self._max_pixels:
            self._array = _fill_nodata(self._ds.read(1), self.nodata, str(self.path))

    def close(self) -> None:
        if self._ds is not None:
            self._ds.close()
            self._ds = None

    def __enter__(self) -> "RasterElevation":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _to_dataset_xy(self, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.crs is None or self.crs.is_geographic:
            return lon.astype(np.float64), lat.astype(np.float64)
        from rasterio.warp import transform as warp_transform

        xs, ys = warp_transform("EPSG:4326", self.crs, lon.ravel().tolist(), lat.ravel().tolist())
        return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)

    def _sample(self, arr: np.ndarray, ri: np.ndarray, ci: np.ndarray) -> np.ndarray:
        if self.method == "nearest":
            return _nearest(arr, ri, ci)
        return _bilinear(arr, ri, ci)

    def elevation_m(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        lat_a, lon_a = _broadcast(lat, lon)
        if lat_a.size == 0:
            return np.zeros(lat_a.shape, dtype=np.float64)
        xs, ys = self._to_dataset_xy(lat_a.ravel(), lon_a.ravel())
        ri, ci = _pixel_index(self.transform, xs, ys)
        if self._array is not None:
            return self._sample(self._array, ri, ci).reshape(lat_a.shape)
        return self._sample_from_disk(ri, ci, xs, ys).reshape(lat_a.shape)

    def _sample_from_disk(
        self, ri: np.ndarray, ci: np.ndarray, xs: np.ndarray, ys: np.ndarray
    ) -> np.ndarray:
        from rasterio.windows import Window

        r0 = int(np.clip(np.floor(ri.min()) - 1, 0, self.height - 1))
        r1 = int(np.clip(np.ceil(ri.max()) + 2, 1, self.height))
        c0 = int(np.clip(np.floor(ci.min()) - 1, 0, self.width - 1))
        c1 = int(np.clip(np.ceil(ci.max()) + 2, 1, self.width))
        if (r1 - r0) * (c1 - c0) <= self._max_pixels:
            win = Window(c0, r0, c1 - c0, r1 - r0)
            block = _fill_nodata(self._ds.read(1, window=win), self.nodata, str(self.path))
            # Indices relative to the window; clamp against the full raster first.
            ri_w = np.clip(ri, 0.0, self.height - 1.0) - r0
            ci_w = np.clip(ci, 0.0, self.width - 1.0) - c0
            return self._sample(block, ri_w, ci_w)
        # Queries spread over more of the raster than we will hold: nearest
        # pixel through rasterio's own sampler.
        vals = np.fromiter(
            (v[0] for v in self._ds.sample(zip(xs.tolist(), ys.tolist()))),
            dtype=np.float64,
            count=xs.size,
        )
        return _fill_nodata(vals, self.nodata, str(self.path))


# --------------------------------------------------------------------------
# AWS Terrain Tiles
# --------------------------------------------------------------------------


def _mercator_xy(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """EPSG:3857 metres from degrees (latitude clamped to the Mercator limit)."""
    lat_c = np.clip(lat, -_MAX_MERCATOR_LAT, _MAX_MERCATOR_LAT)
    x = _WEB_MERCATOR_R * np.deg2rad(lon)
    y = _WEB_MERCATOR_R * np.log(np.tan(np.pi / 4.0 + np.deg2rad(lat_c) / 2.0))
    return x, y


def tile_coords(lat: np.ndarray | float, lon: np.ndarray | float, zoom: int) -> tuple[np.ndarray, np.ndarray]:
    """Fractional XYZ tile coordinates ``(x, y)`` at ``zoom`` (broadcasts)."""
    lat_a, lon_a = _broadcast(lat, lon)
    n = float(2**zoom)
    lon_w = (lon_a + 180.0) % 360.0 - 180.0
    x = (lon_w + 180.0) / 360.0 * n
    lat_r = np.deg2rad(np.clip(lat_a, -_MAX_MERCATOR_LAT, _MAX_MERCATOR_LAT))
    y = (1.0 - np.log(np.tan(lat_r) + 1.0 / np.cos(lat_r)) / np.pi) / 2.0 * n
    return x, y


def tile_url(zoom: int, x: int, y: int) -> str:
    return TERRAIN_TILES_URL.format(z=zoom, x=x, y=y)


def _download(url: str, dest: Path, timeout_s: float) -> None:
    """Fetch ``url`` to ``dest`` atomically (temp file + rename)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=dest.name, suffix=".part", dir=dest.parent)
    tmp = Path(tmp_name)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "dopplergeo/0.1"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp, os.fdopen(fd, "wb") as fh:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                fh.write(chunk)
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class TerrainTilesElevation:
    """Elevation from the public AWS Terrain Tiles GeoTIFFs.

    Tiles are the standard Web Mercator XYZ scheme, 512 px square. Each tile
    is downloaded on first use into ``cache_dir`` and kept open in memory
    (bounded LRU). Queries are answered by bilinear (default) or nearest
    sampling in global pixel space, so points near tile edges blend across
    tiles correctly. Nodata (ocean in some tiles) becomes 0 m.

    Zoom 11 is ~38 m/px at the equator and ~30 m/px at 37 degrees north,
    well under the 200 m candidate grid.
    """

    def __init__(
        self,
        zoom: int = 11,
        cache_dir: str | os.PathLike[str] = DEFAULT_TERRAIN_CACHE,
        *,
        method: str = "bilinear",
        max_tiles_in_memory: int = 64,
        timeout_s: float = 60.0,
    ) -> None:
        _require_rasterio()
        if not 0 <= zoom <= 15:
            raise ValueError("terrain tiles exist for zoom 0..15")
        if method not in ("bilinear", "nearest"):
            raise ValueError("method must be 'bilinear' or 'nearest'")
        self.zoom = int(zoom)
        self.cache_dir = Path(os.path.expanduser(str(cache_dir)))
        self.method = method
        self.timeout_s = float(timeout_s)
        self._max_tiles = int(max_tiles_in_memory)
        self._tiles: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
        self._n = 2**self.zoom

    # -- tile files ---------------------------------------------------------

    def tile_path(self, x: int, y: int) -> Path:
        return self.cache_dir / str(self.zoom) / str(x) / f"{y}.tif"

    def ensure_tile(self, x: int, y: int) -> Path:
        """Return the cached tile path, downloading it if needed."""
        path = self.tile_path(x, y)
        if not path.exists():
            _download(tile_url(self.zoom, x, y), path, self.timeout_s)
        return path

    def _tile_array(self, x: int, y: int) -> np.ndarray:
        key = (x, y)
        arr = self._tiles.get(key)
        if arr is not None:
            self._tiles.move_to_end(key)
            return arr
        rasterio = _require_rasterio()
        try:
            path = self.ensure_tile(x, y)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                warnings.warn(
                    f"terrain tile {self.zoom}/{x}/{y} not found; treating as 0 m",
                    RuntimeWarning,
                    stacklevel=4,
                )
                arr = np.zeros((TERRAIN_TILE_PX, TERRAIN_TILE_PX), dtype=np.float64)
            else:
                raise
        else:
            with rasterio.open(path) as ds:
                arr = _fill_nodata(ds.read(1), ds.nodata, f"terrain tile {self.zoom}/{x}/{y}")
        self._tiles[key] = arr
        while len(self._tiles) > self._max_tiles:
            self._tiles.popitem(last=False)
        return arr

    # -- sampling -----------------------------------------------------------

    def _pixel_values(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """Values at integer global pixel coordinates, grouped by tile."""
        world = self._n * TERRAIN_TILE_PX
        px = np.clip(px, 0, world - 1)
        py = np.clip(py, 0, world - 1)
        tx = px // TERRAIN_TILE_PX
        ty = py // TERRAIN_TILE_PX
        out = np.empty(px.shape, dtype=np.float64)
        keys = tx * self._n + ty
        for key in np.unique(keys):
            sel = keys == key
            x = int(key // self._n)
            y = int(key % self._n)
            arr = self._tile_array(x, y)
            out[sel] = arr[py[sel] - y * TERRAIN_TILE_PX, px[sel] - x * TERRAIN_TILE_PX]
        return out

    def elevation_m(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        lat_a, lon_a = _broadcast(lat, lon)
        if lat_a.size == 0:
            return np.zeros(lat_a.shape, dtype=np.float64)
        x, y = tile_coords(lat_a.ravel(), lon_a.ravel(), self.zoom)
        # Continuous pixel-centre index in global pixel space.
        gx = x * TERRAIN_TILE_PX - 0.5
        gy = y * TERRAIN_TILE_PX - 0.5
        if self.method == "nearest":
            vals = self._pixel_values(np.rint(gx).astype(np.int64), np.rint(gy).astype(np.int64))
            return vals.reshape(lat_a.shape)
        x0 = np.floor(gx).astype(np.int64)
        y0 = np.floor(gy).astype(np.int64)
        fx = gx - x0
        fy = gy - y0
        v00 = self._pixel_values(x0, y0)
        v10 = self._pixel_values(x0 + 1, y0)
        v01 = self._pixel_values(x0, y0 + 1)
        v11 = self._pixel_values(x0 + 1, y0 + 1)
        top = v00 * (1.0 - fx) + v10 * fx
        bot = v01 * (1.0 - fx) + v11 * fx
        return (top * (1.0 - fy) + bot * fy).reshape(lat_a.shape)


def covering_tiles(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float, zoom: int
) -> list[tuple[int, int]]:
    """XYZ tile indices whose extent intersects the bounding box."""
    x_lo, y_lo = tile_coords(lat_max, lon_min, zoom)  # north-west corner
    x_hi, y_hi = tile_coords(lat_min, lon_max, zoom)  # south-east corner
    n = 2**zoom
    xs = range(int(np.floor(x_lo)), min(int(np.floor(x_hi)), n - 1) + 1)
    ys = range(int(np.floor(y_lo)), min(int(np.floor(y_hi)), n - 1) + 1)
    return [(x, y) for y in ys for x in xs]


def fetch_dem(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    out_path: str | os.PathLike[str],
    zoom: int = 11,
    *,
    cache_dir: str | os.PathLike[str] = DEFAULT_TERRAIN_CACHE,
    timeout_s: float = 60.0,
) -> Path:
    """Download the terrain tiles covering a bounding box and merge them
    into one GeoTIFF (EPSG:3857) at ``out_path``. Returns the path.

    Open the result with :class:`RasterElevation`.
    """
    rasterio = _require_rasterio()
    from rasterio.merge import merge as rio_merge

    if lat_min > lat_max or lon_min > lon_max:
        raise ValueError("bounding box has min > max")
    provider = TerrainTilesElevation(zoom=zoom, cache_dir=cache_dir, timeout_s=timeout_s)
    paths = [provider.ensure_tile(x, y) for x, y in covering_tiles(lat_min, lat_max, lon_min, lon_max, zoom)]
    datasets = [rasterio.open(p) for p in paths]
    try:
        mosaic, transform = rio_merge(datasets)
        profile = datasets[0].profile.copy()
    finally:
        for ds in datasets:
            ds.close()
    profile.update(
        driver="GTiff",
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=transform,
        count=1,
        compress="deflate",
    )
    # Tiled layout keys from the source tiles can conflict with the mosaic size.
    for key in ("blockxsize", "blockysize", "tiled"):
        profile.pop(key, None)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(mosaic[0], 1)
    return out
