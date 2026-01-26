import numpy as np
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import LineString

from infrawatch.scoring.traction_risk import sample_ndvi_for_line


def test_sample_ndvi_for_line_returns_stats():
    data = np.ones((10, 10), dtype=np.float32)
    transform = from_origin(0, 10, 1, 1)

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            transform=transform,
            crs="EPSG:32633",
            nodata=-9999,
        ) as dataset:
            dataset.write(data, 1)

        line = LineString([(2, 8), (8, 2)])
        stats = sample_ndvi_for_line(
            memfile.name,
            line,
            buffer_m=0.5,
            line_crs="EPSG:32633",
        )

    assert stats["data_status"] == "OK"
    assert stats["count"] > 0
    assert np.isclose(stats["mean_ndvi"], 1.0)
    assert np.isclose(stats["p90_ndvi"], 1.0)
    assert np.isclose(stats["pct_above_0_6"], 1.0)


def test_sample_ndvi_for_line_no_overlap():
    data = np.ones((10, 10), dtype=np.float32)
    transform = from_origin(0, 10, 1, 1)

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            transform=transform,
            crs="EPSG:32633",
            nodata=-9999,
        ) as dataset:
            dataset.write(data, 1)

        line = LineString([(20, 20), (30, 30)])
        stats = sample_ndvi_for_line(
            memfile.name,
            line,
            buffer_m=1.0,
            line_crs="EPSG:32633",
        )

    assert stats["data_status"] == "NO_DATA"
    assert stats["count"] == 0


def test_sample_ndvi_for_line_clipped_window():
    data = np.ones((10, 10), dtype=np.float32)
    transform = from_origin(0, 10, 1, 1)

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            transform=transform,
            crs="EPSG:32633",
            nodata=-9999,
        ) as dataset:
            dataset.write(data, 1)

        line = LineString([(-5, 6), (-5, 5)])
        stats = sample_ndvi_for_line(
            memfile.name,
            line,
            buffer_m=10.0,
            line_crs="EPSG:32633",
        )

    assert stats["data_status"] == "OK"
    assert stats["count"] > 0
