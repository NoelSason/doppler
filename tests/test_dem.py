"""Elevation providers: in-memory ones always, rasterio-backed ones only when
the ``dem`` extra is installed, network ones only when the tile bucket answers."""

from __future__ import annotations

import socket
import urllib.error
from pathlib import Path

import numpy as np
import pytest

from dopplergeo.dem import (
    TERRAIN_TILE_PX,
    ArrayElevation,
    ElevationProvider,
    FlatElevation,
    covering_tiles,
    tile_coords,
)

try:
    import rasterio

    HAVE_RASTERIO = True
except ImportError:  # pragma: no cover
    HAVE_RASTERIO = False

needs_rasterio = pytest.mark.skipif(not HAVE_RASTERIO, reason="rasterio extra not installed")


# --------------------------------------------------------------------------
# FlatElevation / ArrayElevation
# --------------------------------------------------------------------------


def test_flat_broadcasts_and_is_a_provider() -> None:
    dem = FlatElevation(12.5)
    assert isinstance(dem, ElevationProvider)
    assert dem.elevation_m(37.0, -122.0).shape == ()
    out = dem.elevation_m(np.linspace(37, 38, 5), -122.0)
    assert out.shape == (5,)
    assert np.all(out == 12.5)
    out2 = dem.elevation_m(np.zeros((2, 3)), np.zeros((2, 3)))
    assert out2.shape == (2, 3)


def _plane_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_axis = np.linspace(37.0, 38.0, 11)
    lon_axis = np.linspace(-123.0, -121.0, 21)
    grid = 100.0 + 1000.0 * (lat_axis[:, None] - 37.0) + 500.0 * (lon_axis[None, :] + 123.0)
    return lat_axis, lon_axis, grid


def test_array_bilinear_is_exact_on_a_plane() -> None:
    lat_axis, lon_axis, grid = _plane_grid()
    dem = ArrayElevation(lat_axis, lon_axis, grid)
    lat = np.array([37.123, 37.5, 37.987])
    lon = np.array([-122.456, -121.0, -122.999])
    want = 100.0 + 1000.0 * (lat - 37.0) + 500.0 * (lon + 123.0)
    np.testing.assert_allclose(dem.elevation_m(lat, lon), want, atol=1e-9)
    assert isinstance(dem, ElevationProvider)


def test_array_clamps_outside_bounds_and_accepts_descending_axes() -> None:
    lat_axis, lon_axis, grid = _plane_grid()
    dem = ArrayElevation(lat_axis, lon_axis, grid)
    # Far outside: clamps to the nearest edge instead of extrapolating.
    assert dem.elevation_m(50.0, -122.0) == pytest.approx(dem.elevation_m(38.0, -122.0))
    assert dem.elevation_m(37.5, -100.0) == pytest.approx(dem.elevation_m(37.5, -121.0))
    # Descending axes (north-up rasters) give the same answers.
    dem_desc = ArrayElevation(lat_axis[::-1], lon_axis, grid[::-1, :])
    q_lat = np.array([37.2, 37.9])
    q_lon = np.array([-122.7, -121.3])
    np.testing.assert_allclose(dem_desc.elevation_m(q_lat, q_lon), dem.elevation_m(q_lat, q_lon))


def test_array_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        ArrayElevation(np.arange(3.0), np.arange(4.0), np.zeros((4, 3)))


# --------------------------------------------------------------------------
# Tile arithmetic (pure numpy, no network)
# --------------------------------------------------------------------------


def test_tile_coords_match_known_tile() -> None:
    x, y = tile_coords(37.8719, -122.2585, 11)
    assert int(np.floor(x)) == 328
    assert int(np.floor(y)) == 790
    # Tile 328/790 spans ~37.857..37.996 N, -122.344..-122.168 E.
    assert covering_tiles(37.86, 37.95, -122.30, -122.20, 11) == [(328, 790)]
    assert covering_tiles(37.85, 37.95, -122.30, -122.20, 11) == [(328, 790), (328, 791)]
    tiles = covering_tiles(37.0, 38.0, -122.5, -121.5, 11)
    assert (328, 790) in tiles
    assert len(tiles) >= 4


# --------------------------------------------------------------------------
# RasterElevation on a synthetic GeoTIFF
# --------------------------------------------------------------------------


def _write_plane_tif(path: Path, nodata: float | None = None, crs: str = "EPSG:4326") -> tuple[np.ndarray, np.ndarray]:
    from rasterio.transform import from_origin

    n_lat, n_lon = 40, 60
    d = 0.01
    lat_top, lon_left = 38.0, -123.0
    # Pixel centres.
    lats = lat_top - (np.arange(n_lat) + 0.5) * d
    lons = lon_left + (np.arange(n_lon) + 0.5) * d
    grid = 100.0 + 1000.0 * (lats[:, None] - 37.0) + 500.0 * (lons[None, :] + 123.0)
    grid = grid.astype(np.float32)
    if nodata is not None:
        grid[0, 0] = nodata
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=n_lat,
        width=n_lon,
        count=1,
        dtype="float32",
        crs=crs,
        transform=from_origin(lon_left, lat_top, d, d),
        nodata=nodata,
    ) as dst:
        dst.write(grid, 1)
    return lats, lons


@needs_rasterio
def test_raster_bilinear_matches_plane(tmp_path: Path) -> None:
    from dopplergeo.dem import RasterElevation

    p = tmp_path / "plane.tif"
    _write_plane_tif(p)
    with RasterElevation(p) as dem:
        assert isinstance(dem, ElevationProvider)
        lat = np.array([37.75, 37.6123, 37.9])
        lon = np.array([-122.5, -122.8765, -122.6])
        want = 100.0 + 1000.0 * (lat - 37.0) + 500.0 * (lon + 123.0)
        np.testing.assert_allclose(dem.elevation_m(lat, lon), want, atol=1e-2)
        # Pixel centre exactly, nearest == bilinear.
        assert dem.elevation_m(37.995, -122.995) == pytest.approx(100.0 + 1000.0 * 0.995 + 500.0 * 0.005, abs=1e-2)
    with RasterElevation(p, method="nearest") as dem_n:
        assert dem_n.elevation_m(37.995, -122.995) == pytest.approx(100.0 + 1000.0 * 0.995 + 500.0 * 0.005, abs=1e-2)


@needs_rasterio
def test_raster_disk_path_agrees_with_memory_path(tmp_path: Path) -> None:
    from dopplergeo.dem import RasterElevation

    p = tmp_path / "plane.tif"
    _write_plane_tif(p)
    lat = np.random.default_rng(0).uniform(37.65, 37.95, 50)
    lon = np.random.default_rng(1).uniform(-122.9, -122.5, 50)
    # 40 x 60 px raster: a 2000 px budget forces the per-query window read.
    with RasterElevation(p) as mem, RasterElevation(p, max_pixels_in_memory=2000) as disk:
        assert mem._array is not None and disk._array is None
        np.testing.assert_allclose(disk.elevation_m(lat, lon), mem.elevation_m(lat, lon), atol=1e-3)
    # Window too big for the budget -> dataset.sample (nearest) fallback.
    with RasterElevation(p, max_pixels_in_memory=0, method="nearest") as tiny, RasterElevation(p, method="nearest") as mem_n:
        np.testing.assert_allclose(tiny.elevation_m(lat, lon), mem_n.elevation_m(lat, lon), atol=1e-3)


@needs_rasterio
def test_raster_nodata_becomes_zero_with_warning(tmp_path: Path) -> None:
    from dopplergeo.dem import RasterElevation

    p = tmp_path / "nodata.tif"
    lats, lons = _write_plane_tif(p, nodata=-9999.0)
    with pytest.warns(RuntimeWarning, match="nodata"):
        dem = RasterElevation(p, method="nearest")
    # Pixel (0, 0) is the north-west corner.
    assert dem.elevation_m(lats[0], lons[0]) == 0.0
    assert dem.elevation_m(lats[1], lons[1]) != 0.0
    dem.close()


@needs_rasterio
def test_raster_projected_crs_transforms_queries(tmp_path: Path) -> None:
    """A raster in EPSG:3857 is queried in lat/lon."""
    from rasterio.transform import from_origin

    from dopplergeo.dem import RasterElevation, _mercator_xy

    p = tmp_path / "merc.tif"
    x0, y0 = _mercator_xy(np.array(38.0), np.array(-123.0))
    n = 64
    px = 500.0
    xs = float(x0) + (np.arange(n) + 0.5) * px
    ys = float(y0) - (np.arange(n) + 0.5) * px
    grid = (0.001 * (xs[None, :] - float(x0)) + 0.002 * (float(y0) - ys[:, None])).astype(np.float32)
    with rasterio.open(
        p, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32", crs="EPSG:3857",
        transform=from_origin(float(x0), float(y0), px, px),
    ) as dst:
        dst.write(grid, 1)
    lat = np.array([37.9, 37.8])
    lon = np.array([-122.9, -122.8])
    qx, qy = _mercator_xy(lat, lon)
    want = 0.001 * (qx - float(x0)) + 0.002 * (float(y0) - qy)
    with RasterElevation(p) as dem:
        np.testing.assert_allclose(dem.elevation_m(lat, lon), want, rtol=1e-4, atol=0.5)


# --------------------------------------------------------------------------
# TerrainTilesElevation (network)
# --------------------------------------------------------------------------


def _terrain(tmp_path: Path, **kw: object):
    from dopplergeo.dem import TerrainTilesElevation

    dem = TerrainTilesElevation(zoom=11, cache_dir=tmp_path / "terrain", timeout_s=20.0, **kw)  # type: ignore[arg-type]
    try:
        dem.ensure_tile(328, 790)
    except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as exc:
        pytest.skip(f"terrain tiles unreachable: {exc}")
    return dem


@needs_rasterio
def test_terrain_tiles_known_elevations(tmp_path: Path) -> None:
    dem = _terrain(tmp_path)
    assert dem.tile_path(328, 790).exists()
    with rasterio.open(dem.tile_path(328, 790)) as ds:
        assert ds.width == TERRAIN_TILE_PX and ds.height == TERRAIN_TILE_PX
        # The XYZ arithmetic must agree with the tile's own georeferencing.
        from dopplergeo.dem import _WEB_MERCATOR_EXTENT

        exp_x0 = -_WEB_MERCATOR_EXTENT / 2 + 328 * _WEB_MERCATOR_EXTENT / 2048
        exp_y0 = _WEB_MERCATOR_EXTENT / 2 - 790 * _WEB_MERCATOR_EXTENT / 2048
        assert ds.transform.c == pytest.approx(exp_x0, abs=1e-3)
        assert ds.transform.f == pytest.approx(exp_y0, abs=1e-3)
    berkeley = float(dem.elevation_m(37.8719, -122.2585))
    assert 50.0 < berkeley < 120.0
    denver = float(dem.elevation_m(39.7392, -104.9903))
    assert 1500.0 < denver < 1700.0
    diablo = float(dem.elevation_m(37.8816, -121.9143))
    assert 1100.0 < diablo < 1200.0
    # Nearest and bilinear agree to within one pixel of relief.
    dem_n = _terrain(tmp_path, method="nearest")
    assert abs(float(dem_n.elevation_m(37.8719, -122.2585)) - berkeley) < 15.0
    # Broadcasting over an array of points, straddling a tile edge.
    lat = np.linspace(37.70, 37.95, 25)
    lon = np.linspace(-122.35, -122.10, 25)
    out = dem.elevation_m(lat[:, None], lon[None, :])
    assert out.shape == (25, 25)
    assert np.all(np.isfinite(out))
    assert isinstance(dem, ElevationProvider)


@needs_rasterio
def test_fetch_dem_merges_tiles_and_agrees_with_tiles(tmp_path: Path) -> None:
    from dopplergeo.dem import RasterElevation, fetch_dem

    dem = _terrain(tmp_path)
    out = fetch_dem(37.80, 37.92, -122.32, -122.18, tmp_path / "merged.tif", zoom=11, cache_dir=tmp_path / "terrain")
    assert out.exists()
    with rasterio.open(out) as ds:
        assert ds.crs.to_epsg() == 3857
        assert ds.width >= TERRAIN_TILE_PX and ds.height >= TERRAIN_TILE_PX
    lat = np.array([37.8719, 37.85, 37.90])
    lon = np.array([-122.2585, -122.30, -122.20])
    with RasterElevation(out) as merged:
        np.testing.assert_allclose(merged.elevation_m(lat, lon), dem.elevation_m(lat, lon), atol=0.5)
