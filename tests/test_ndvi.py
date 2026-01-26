import numpy as np

from infrawatch.analytics.ndvi import compute_ndvi


def test_compute_ndvi_values():
    red = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    nir = np.array([[3.0, 6.0], [9.0, 12.0]], dtype=np.float32)

    ndvi = compute_ndvi(red, nir)

    expected = (nir - red) / (nir + red)
    assert np.allclose(ndvi, expected)


def test_compute_ndvi_handles_zero_denominator():
    red = np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32)
    nir = np.array([[-1.0, 1.0], [-2.0, 2.0]], dtype=np.float32)

    ndvi = compute_ndvi(red, nir)

    assert np.isnan(ndvi).all()
