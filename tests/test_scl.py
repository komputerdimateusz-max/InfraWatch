import numpy as np

from infrawatch.analytics.ndvi import compute_ndvi
from infrawatch.analytics.scl import scl_valid_mask


def test_scl_valid_mask_excludes_clouds_and_optional_water():
    scl = np.array([[4, 3], [6, 11]], dtype=np.uint8)

    mask_without_water = scl_valid_mask(scl, exclude_water=True)
    assert mask_without_water.tolist() == [[True, False], [False, False]]

    mask_with_water = scl_valid_mask(scl, exclude_water=False)
    assert mask_with_water.tolist() == [[True, False], [True, False]]


def test_ndvi_masking_with_scl():
    red = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    nir = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    scl = np.array([[4, 3], [6, 8]], dtype=np.uint8)

    ndvi = compute_ndvi(red, nir)
    valid = scl_valid_mask(scl, exclude_water=True)
    masked = ndvi.copy()
    masked[~valid] = np.nan

    assert np.isfinite(masked[0, 0])
    assert np.isnan(masked[0, 1])
    assert np.isnan(masked[1, 0])
    assert np.isnan(masked[1, 1])
