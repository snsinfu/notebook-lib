import numpy as np
import pytest

from lib.density import (
    BandwidthMethod,
    bw_method,
    enclosed_thresholds,
    estimate_conditional_density_mesh,
    estimate_density_mesh,
    make_epanechnikov_kernel,
    make_gaussian_kernel,
    resolve_bandwidth,
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


def test_estimate_density_mesh_integrates_to_one(rng):
    x = rng.normal(size=500)
    y = rng.normal(size=500)
    _, _, Z, cell_area = estimate_density_mesh(x, y, gridsize=200)
    assert np.isclose(Z.sum() * cell_area, 1.0, atol=0.02)


def test_estimate_density_mesh_kernel_integrates_to_one(rng):
    x = rng.normal(size=500)
    y = rng.normal(size=500)
    _, _, Z, cell_area = estimate_density_mesh(
        x, y, gridsize=200, kernel="epanechnikov", bandwidth=0.5
    )
    assert np.isclose(Z.sum() * cell_area, 1.0, atol=0.05)


def test_bw_method_scott_matches_default(rng):
    data = np.vstack([rng.normal(size=100), rng.normal(size=100)])
    assert np.allclose(
        resolve_bandwidth(None, data), resolve_bandwidth(bw_method("scott"), data)
    )
    assert isinstance(bw_method("silverman"), BandwidthMethod)


def test_bw_method_rejects_bad_values():
    with pytest.raises(ValueError):
        bw_method("chu")
    with pytest.raises(ValueError):
        bw_method(-1.0)


def test_resolve_bandwidth_absolute_is_independent_of_data(rng):
    data = np.vstack([rng.normal(size=100), rng.normal(size=100)]) * 10
    assert np.allclose(resolve_bandwidth(0.3, data), 0.3)


def test_resolve_bandwidth_rejects_bad_values():
    data = np.zeros((2, 5))
    with pytest.raises(ValueError):
        resolve_bandwidth(0.0, data)
    with pytest.raises(TypeError):
        resolve_bandwidth("scott", data)  # must be wrapped in bw_method(...)


def test_custom_bandwidth_method(rng):
    class Fixed(BandwidthMethod):
        def estimate_bandwidth(self, samples, *, weights=None):
            return np.full(samples.shape[0], 0.25)

    data = np.vstack([rng.normal(size=50), rng.normal(size=50)])
    assert np.allclose(resolve_bandwidth(Fixed(), data), 0.25)


def test_enclosed_thresholds_rejects_out_of_range_levels():
    density = np.ones((10, 10))
    with pytest.raises(ValueError):
        enclosed_thresholds(density, 1.0, [0.0, 1.0])


def test_enclosed_thresholds_ordering():
    # Larger tail mass -> smaller enclosed mass -> higher density threshold.
    Z = np.linspace(0, 1, 100).reshape(10, 10)
    thresholds = enclosed_thresholds(Z, 0.01, [0.25, 0.75])
    assert thresholds[1] > thresholds[0]


@pytest.mark.parametrize("given", ["x", "y"])
def test_estimate_conditional_density_mesh_slices_integrate_to_one(rng, given):
    # f(x|y) integrates to one over x (and f(y|x) over y) when the marginal
    # shares the joint's bandwidth. Check the central slices away from the grid
    # padding where the truncated window still captures the mass.
    y = rng.normal(size=1000)
    x = 0.8 * y + rng.normal(size=1000)
    X, Y, Z, _ = estimate_conditional_density_mesh(x, y, given=given, gridsize=200)
    if given == "y":
        integrals = Z.sum(axis=1) * (X[0, 1] - X[0, 0])
    else:
        integrals = Z.sum(axis=0) * (Y[1, 0] - Y[0, 0])
    core = integrals[50:150]
    assert np.allclose(core, 1.0, atol=0.05)


@pytest.mark.parametrize("given", ["x", "y"])
def test_estimate_conditional_density_mesh_compact_kernel_is_finite(rng, given):
    # Epanechnikov has bounded support; the zero-marginal region must not
    # produce nan/inf in the ratio.
    y = rng.normal(size=1000)
    x = 0.8 * y + rng.normal(size=1000)
    _, _, Z, _ = estimate_conditional_density_mesh(
        x, y, given=given, kernel="epanechnikov", bandwidth=0.5
    )
    assert np.all(np.isfinite(Z))
    assert np.all(Z >= 0)


def test_estimate_conditional_density_mesh_rejects_callable_kernel(rng):
    x = rng.normal(size=50)
    y = rng.normal(size=50)
    with pytest.raises(TypeError):
        estimate_conditional_density_mesh(x, y, kernel=lambda points: points)


def test_estimate_conditional_density_mesh_rejects_bad_given(rng):
    x = rng.normal(size=50)
    y = rng.normal(size=50)
    with pytest.raises(ValueError):
        estimate_conditional_density_mesh(x, y, given="z")
