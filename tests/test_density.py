import numpy as np
import pytest

from lib.density import (
    enclosed_thresholds,
    estimate_density,
    make_epanechnikov_kernel,
    make_gaussian_kernel,
    scalar_epanechnikov_kernel,
    scalar_gaussian_kernel,
)

_trapz = getattr(np, "trapezoid", None) or np.trapz


def test_epanechnikov_1d_is_normalized():
    # Regression: the normalizer multiplied by pi**(d/2) instead of dividing.
    # d-dimensional kernels reduce the last axis, so 1-D points are (n, 1).
    kernel = make_epanechnikov_kernel(1)
    assert np.isclose(kernel(np.array([[0.0]]))[0], 0.75)

    xs = np.linspace(-1.0, 1.0, 20001)
    assert np.isclose(_trapz(kernel(xs[:, None]), xs), 1.0, atol=1e-3)


def test_scalar_gaussian_matches_factory():
    # Regression: scalar_gaussian_kernel was off by a factor of 2*pi.
    x = np.linspace(-3.0, 3.0, 11)
    assert np.allclose(scalar_gaussian_kernel(x), make_gaussian_kernel(1)(x[:, None]))


def test_scalar_epanechnikov_matches_factory():
    x = np.linspace(-1.5, 1.5, 13)
    assert np.allclose(
        scalar_epanechnikov_kernel(x), make_epanechnikov_kernel(1)(x[:, None])
    )


def test_kernel_factories_are_cached():
    # Regression: each call rebuilt (and recompiled) the numba kernel.
    assert make_gaussian_kernel(2) is make_gaussian_kernel(2)
    assert make_epanechnikov_kernel(2) is make_epanechnikov_kernel(2)


def test_kernel_factories_reject_nonpositive_dim():
    with pytest.raises(ValueError):
        make_gaussian_kernel(0)
    with pytest.raises(ValueError):
        make_epanechnikov_kernel(0)


def test_flux_functions_are_renamed():
    from lib.density import kernel_flux, mesh_kernel_flux  # noqa: F401


def test_kernel_flux_rejects_short_sequence():
    from lib.density import kernel_flux

    points = np.zeros((3, 2))
    sequence = np.zeros((1, 2))
    with pytest.raises(ValueError):
        kernel_flux(points, sequence, kernel_fn=make_gaussian_kernel(2))


def test_estimate_density_grid_integrates_to_one(rng):
    x = rng.normal(size=500)
    y = rng.normal(size=500)
    _, _, Z, cell_area = estimate_density(x, y, gridsize=200)
    assert np.isclose(Z.sum() * cell_area, 1.0, atol=0.02)


def test_enclosed_thresholds_rejects_out_of_range_levels():
    density = np.ones((10, 10))
    with pytest.raises(ValueError):
        enclosed_thresholds(density, 1.0, [0.0, 1.0])


def test_enclosed_thresholds_ordering():
    # Larger tail mass -> smaller enclosed mass -> higher density threshold.
    Z = np.linspace(0, 1, 100).reshape(10, 10)
    thresholds = enclosed_thresholds(Z, 0.01, [0.25, 0.75])
    assert thresholds[1] > thresholds[0]
