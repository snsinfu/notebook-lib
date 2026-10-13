import pathlib
import subprocess
import sys

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.stats

from lib.density import estimate_density
from lib.plot import (
    attach_colorbar,
    bw_method,
    get_x_grid_line_at,
    get_y_grid_line_at,
    kde_at,
    kde_contour,
    register_colormaps,
    set_axis,
    slice_cmap,
    smooth_matrix,
    texsci,
    use_log_ticks,
)
from lib.plot.density import _density_contour

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_texsci_positive_exponent():
    assert texsci(1.23e5, 3) == r"1.23 \times 10^{5}"


def test_texsci_negative_exponent():
    assert texsci(1.23e-5, 3) == r"1.23 \times 10^{-5}"


def test_texsci_plain_value():
    assert texsci(1.23, 3) == "1.23"


def test_scipy_density_matches_gaussian_kde(rng):
    x = rng.normal(size=200)
    y = rng.normal(size=200)
    got = kde_at(x, y, kernel="scipy")
    expected = scipy.stats.gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))
    assert np.allclose(got, expected)


def test_scipy_weighted_density_matches_gaussian_kde(rng):
    x = rng.normal(size=200)
    y = rng.normal(size=200)
    weights = rng.uniform(0.1, 2.0, size=200)
    got = kde_at(x, y, weights=weights, kernel="scipy")
    expected = scipy.stats.gaussian_kde(np.vstack([x, y]), weights=weights)(
        np.vstack([x, y])
    )
    assert np.allclose(got, expected)


def test_kde_at_weights_change_result(rng):
    x = rng.normal(size=200)
    y = rng.normal(size=200)
    weights = rng.uniform(0.1, 2.0, size=200)
    unweighted = kde_at(x, y, kernel="scipy")
    weighted = kde_at(x, y, weights=weights, kernel="scipy")
    assert not np.allclose(unweighted, weighted)


def test_kde_at_default_evaluates_at_samples(rng):
    x = rng.normal(size=100)
    y = rng.normal(size=100)
    assert np.allclose(kde_at(x, y), kde_at(x, y, at=(x, y)))


def test_kernel_density_integrates_to_one(rng):
    # Regression: the kernel path returned sum(K) without 1/n or 1/prod(h).
    x = rng.normal(size=500)
    y = rng.normal(size=500)
    grid = np.linspace(-5, 5, 121)
    X, Y = np.meshgrid(grid, grid)
    density = kde_at(
        x, y, at=(X.ravel(), Y.ravel()), kernel="gaussian", bandwidth=0.5
    ).reshape(X.shape)
    cell_area = (grid[1] - grid[0]) ** 2
    assert np.isclose(density.sum() * cell_area, 1.0, atol=0.05)


def test_kernel_weighted_density_integrates_to_one(rng):
    x = rng.normal(size=500)
    y = rng.normal(size=500)
    weights = rng.uniform(0.1, 2.0, size=500)
    grid = np.linspace(-5, 5, 121)
    X, Y = np.meshgrid(grid, grid)
    density = kde_at(
        x,
        y,
        at=(X.ravel(), Y.ravel()),
        weights=weights,
        kernel="gaussian",
        bandwidth=0.5,
    ).reshape(X.shape)
    cell_area = (grid[1] - grid[0]) ** 2
    assert np.isclose(density.sum() * cell_area, 1.0, atol=0.05)


def test_kde_at_rejects_negative_bandwidth(rng):
    x = rng.normal(size=50)
    y = rng.normal(size=50)
    with pytest.raises(ValueError):
        kde_at(x, y, kernel="gaussian", bandwidth=-1.0)


def test_kde_at_rejects_unknown_kernel(rng):
    x = rng.normal(size=50)
    y = rng.normal(size=50)
    with pytest.raises(ValueError):
        kde_at(x, y, kernel="histogram")


def test_scipy_rejects_absolute_bandwidth(rng):
    # scipy's gaussian_kde only takes a relative factor, never a data-unit width.
    x = rng.normal(size=50)
    y = rng.normal(size=50)
    with pytest.raises(ValueError):
        kde_at(x, y, kernel="scipy", bandwidth=0.5)


def test_default_scipy_bandwidth_has_effect(rng):
    # Regression: bandwidth used to be silently ignored by the default backend.
    x = rng.normal(size=200)
    narrow = kde_at(x, bandwidth=bw_method(0.2))
    wide = kde_at(x, bandwidth=bw_method(1.0))
    assert not np.allclose(narrow, wide)


def test_gaussian_absolute_bandwidth_has_effect(rng):
    x = rng.normal(size=200)
    narrow = kde_at(x, kernel="gaussian", bandwidth=0.05)
    wide = kde_at(x, kernel="gaussian", bandwidth=5.0)
    assert not np.allclose(narrow, wide)


def test_scipy_bw_method_matches_gaussian_kde(rng):
    x = rng.normal(size=200)
    y = rng.normal(size=200)
    got = kde_at(x, y, kernel="scipy", bandwidth=bw_method(0.5))
    expected = scipy.stats.gaussian_kde(np.vstack([x, y]), bw_method=0.5)(
        np.vstack([x, y])
    )
    assert np.allclose(got, expected)


def test_scott_bw_method_matches_default(rng):
    x = rng.normal(size=200)
    y = rng.normal(size=200)
    assert np.allclose(kde_at(x, y), kde_at(x, y, bandwidth=bw_method("scott")))


@pytest.mark.parametrize("kernel", ["scipy", "gaussian", "epanechnikov"])
def test_silverman_bw_method_accepted(rng, kernel):
    x = rng.normal(size=200)
    y = rng.normal(size=200)
    density = kde_at(x, y, kernel=kernel, bandwidth=bw_method("silverman"))
    assert np.all(np.isfinite(density))
    assert np.all(density > 0)


def test_gaussian_absolute_bandwidth_scales_with_data(rng):
    # Absolute widths are homogeneous: scaling the samples by s and the width
    # by s leaves the density shape, with mass scaling by s**(-dim).
    x = rng.normal(size=200)
    y = rng.normal(size=200)
    grid = np.linspace(-5, 5, 51)
    X, Y = np.meshgrid(grid, grid)
    base = kde_at(x, y, at=(X.ravel(), Y.ravel()), kernel="gaussian", bandwidth=0.5)
    scaled = kde_at(
        10 * x,
        10 * y,
        at=(10 * X.ravel(), 10 * Y.ravel()),
        kernel="gaussian",
        bandwidth=5.0,
    )
    assert np.allclose(scaled, base / 100.0)


def test_kde_at_custom_bandwidth_method(rng):
    from lib.density import BandwidthMethod

    class Fixed(BandwidthMethod):
        def estimate_bandwidth(self, samples, *, weights=None):
            return np.full(samples.shape[0], 0.25)

    x = rng.normal(size=200)
    y = rng.normal(size=200)
    assert np.allclose(
        kde_at(x, y, kernel="gaussian", bandwidth=Fixed()),
        kde_at(x, y, kernel="gaussian", bandwidth=0.25),
    )
    with pytest.raises(TypeError):
        kde_at(x, y, kernel="scipy", bandwidth=Fixed())


@pytest.mark.parametrize(
    "weights",
    [
        np.array([1.0, 2.0]),  # wrong length
        np.array([1.0, -1.0, 1.0]),  # negative
        np.zeros(3),  # sums to zero
    ],
)
def test_kde_at_rejects_bad_weights(weights):
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        kde_at(x, y, weights=weights)


def test_scipy_1d_density_matches_gaussian_kde(rng):
    x = rng.normal(size=200)
    got = kde_at(x, kernel="scipy")
    expected = scipy.stats.gaussian_kde(np.vstack([x]))(np.vstack([x]))
    assert np.allclose(got, expected)


def test_scipy_1d_at_query_matches_gaussian_kde(rng):
    x = rng.normal(size=200)
    qx = rng.normal(size=50)
    got = kde_at(x, at=qx, kernel="scipy")
    expected = scipy.stats.gaussian_kde(np.vstack([x]))(qx)
    assert np.allclose(got, expected)


def test_scipy_1d_weighted_density_matches_gaussian_kde(rng):
    x = rng.normal(size=200)
    weights = rng.uniform(0.1, 2.0, size=200)
    got = kde_at(x, weights=weights, kernel="scipy")
    expected = scipy.stats.gaussian_kde(np.vstack([x]), weights=weights)(np.vstack([x]))
    assert np.allclose(got, expected)


def test_kde_at_1d_default_evaluates_at_samples(rng):
    x = rng.normal(size=100)
    assert np.allclose(kde_at(x), kde_at(x, at=x))


def test_kernel_1d_density_integrates_to_one(rng):
    x = rng.normal(size=500)
    grid = np.linspace(-5, 5, 121)
    density = kde_at(x, at=grid, kernel="gaussian", bandwidth=0.5)
    assert np.isclose(density.sum() * (grid[1] - grid[0]), 1.0, atol=0.05)


def test_kde_at_1d_rejects_2d_at(rng):
    x = rng.normal(size=50)
    with pytest.raises(ValueError):
        kde_at(x, at=(x, x))


def test_kde_at_2d_rejects_flat_at(rng):
    x = rng.normal(size=50)
    y = rng.normal(size=50)
    with pytest.raises(ValueError):
        kde_at(x, y, at=x)


def test_kde_at_at_is_keyword_only(rng):
    x = rng.normal(size=50)
    y = rng.normal(size=50)
    with pytest.raises(TypeError):
        kde_at(x, y, x)


def test_smooth_matrix_preserves_shape():
    matrix = np.zeros((20, 20))
    matrix[10, 10] = 1.0
    smoothed = smooth_matrix(matrix, 1.0)
    assert smoothed.shape == matrix.shape
    assert smoothed.max() <= 1.0


def test_smooth_matrix_rejects_negative_sigma():
    with pytest.raises(ValueError):
        smooth_matrix(np.ones((5, 5)), -1.0)


def test_set_axis_runs():
    _, ax = plt.subplots()
    set_axis(ax, spines="left bottom", ticks="left bottom")
    plt.close(ax.figure)


def test_set_axis_resets_axis_inversion():
    # Regression: set_axis only ever inverted axes and never reset them, so it
    # was not idempotent.
    _, ax = plt.subplots()
    set_axis(ax, spines="left bottom", ticks="left top")
    assert ax.yaxis.get_inverted()
    set_axis(ax, spines="left bottom", ticks="left bottom")
    assert not ax.yaxis.get_inverted()
    plt.close(ax.figure)


def test_set_axis_rejects_unknown_side():
    _, ax = plt.subplots()
    with pytest.raises(ValueError):
        set_axis(ax, spines="topleft")
    plt.close(ax.figure)


def test_attach_colorbar_horizontal_honors_length():
    # Regression: the horizontal branch hard-coded width 0.5 and ignored length.
    sm = plt.cm.ScalarMappable(cmap="viridis")
    sm.set_array([0.0, 1.0])
    widths = []
    for length in (0.3, 0.9):
        _, ax = plt.subplots()
        cbar = attach_colorbar(ax, sm, orientation="horizontal", length=length)
        ax.figure.canvas.draw()
        widths.append(cbar.ax.get_position().width)
        plt.close(ax.figure)
    assert widths[1] > widths[0]


def test_grid_line_lookup_finds_tick():
    _, ax = plt.subplots()
    ax.set_xticks([0.0, 1.0, 2.0])
    ax.set_yticks([0.0, 1.0])
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True)
    assert get_x_grid_line_at(ax, 1.0) is not None
    assert get_y_grid_line_at(ax, 1.0) is not None
    plt.close(ax.figure)


def test_use_log_ticks_runs():
    _, ax = plt.subplots()
    use_log_ticks(ax.xaxis)
    plt.close(ax.figure)


def test_use_log_ticks_base_10_positions():
    _, ax = plt.subplots()
    ax.set_xlim(0.0, 3.0)
    use_log_ticks(ax.xaxis, base=10.0)
    ticks = np.asarray(ax.xaxis.get_major_locator()())
    inside = ticks[(ticks >= 0.0) & (ticks <= 3.0)]
    assert np.allclose(inside, [0.0, 1.0, 2.0, 3.0])
    plt.close(ax.figure)


def test_use_log_ticks_derives_transforms_from_base():
    # Regression: log/exp and base were independent arguments, so they could
    # silently disagree (e.g. natural log with base 10).
    _, ax = plt.subplots()
    use_log_ticks(ax.xaxis, base=np.e)
    locator = ax.xaxis.get_major_locator()
    assert np.isclose(locator._fwd(np.e), 1.0)
    assert np.isclose(locator._inv(1.0), np.e)

    use_log_ticks(ax.xaxis, base=10.0)
    locator = ax.xaxis.get_major_locator()
    assert np.isclose(locator._fwd(10.0), 1.0)
    assert np.isclose(locator._inv(1.0), 10.0)
    plt.close(ax.figure)


def test_import_does_not_register_colormaps():
    # Regression: importing lib.plot mutated global matplotlib state.
    code = "import matplotlib; import lib.plot; print('wrr' in matplotlib.colormaps)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"


def test_register_colormaps_registers_names():
    register_colormaps()
    assert "wrr" in matplotlib.colormaps
    assert "bkr" in matplotlib.colormaps


def test_slice_cmap_trims_linear_colormap():
    # `viridis` is a ListedColormap in matplotlib 3.11; use a segmented map.
    cmap = matplotlib.colormaps["bwr"]
    sliced = slice_cmap(cmap, 0.1, 0.9)
    assert isinstance(sliced, matplotlib.colors.LinearSegmentedColormap)
    assert sliced.name == "bwr_sliced"
    assert np.allclose(sliced(0.0), cmap(0.1))
    assert np.allclose(sliced(1.0), cmap(0.9))
    assert not np.allclose(sliced(0.0), cmap(0.0))


def test_slice_cmap_preserves_listed_type():
    colors = [
        "red",
        "orange",
        "yellow",
        "green",
        "blue",
        "purple",
        "brown",
        "pink",
        "gray",
        "black",
    ]
    cmap = matplotlib.colors.ListedColormap(colors)
    sliced = slice_cmap(cmap, 0.1, 0.9)
    assert isinstance(sliced, matplotlib.colors.ListedColormap)
    assert sliced.N == 8
    assert list(sliced.colors) == list(cmap.colors)[1:9]


def test_slice_cmap_listed_rounds_trim_counts():
    # 7 colors: round(0.1 * 7) == round(0.7) == 1 trimmed from each end.
    cmap = matplotlib.colors.ListedColormap(
        ["red", "orange", "yellow", "green", "blue", "purple", "black"]
    )
    sliced = slice_cmap(cmap, 0.1, 0.9)
    assert sliced.N == 5
    assert list(sliced.colors) == list(cmap.colors)[1:6]


def test_slice_cmap_custom_name():
    sliced = slice_cmap(matplotlib.colormaps["viridis"], 0.0, 1.0, name="mine")
    assert sliced.name == "mine"


@pytest.mark.parametrize("lo,hi", [(0.9, 0.1), (-0.1, 0.5), (0.5, 1.1), (0.5, 0.5)])
def test_slice_cmap_rejects_invalid_range(lo, hi):
    with pytest.raises(ValueError):
        slice_cmap(matplotlib.colormaps["viridis"], lo, hi)


def test_kde_contour_returns_contourset(rng):
    x = rng.normal(size=300)
    y = rng.normal(size=300)
    _, ax = plt.subplots()
    contour_set = kde_contour(x, y, levels=[0.5], ax=ax)
    assert contour_set is not None
    plt.close(ax.figure)


def test_kde_contour_accepts_kernel(rng):
    x = rng.normal(size=300)
    y = rng.normal(size=300)
    _, ax = plt.subplots()
    contour_set = kde_contour(
        x, y, levels=[0.5], ax=ax, kernel="epanechnikov", bandwidth=1.0
    )
    assert contour_set is not None
    plt.close(ax.figure)


def test_kde_contour_conditional_rejects_callable_kernel(rng):
    x = rng.normal(size=50)
    y = rng.normal(size=50)
    _, ax = plt.subplots()
    with pytest.raises(TypeError):
        kde_contour(x, y, ax=ax, conditional_on="y", kernel=lambda points: points)
    plt.close(ax.figure)


def test_density_contour_returns_contourset(rng):
    x = rng.normal(size=300)
    y = rng.normal(size=300)
    X, Y, Z, cell_area = estimate_density(x, y)
    _, ax = plt.subplots()
    contour_set = _density_contour(X, Y, Z, levels=[0.5], ax=ax, cell_area=cell_area)
    assert contour_set is not None
    assert np.all(np.isfinite(np.asarray(contour_set.levels)))
    plt.close(ax.figure)


def test_density_contour_rejects_mismatched_shapes():
    X = np.zeros((5, 5))
    Y = np.zeros((5, 5))
    Z = np.zeros((4, 5))
    with pytest.raises(ValueError):
        _density_contour(X, Y, Z, levels=[0.5], cell_area=1.0)


@pytest.mark.parametrize("conditional_on", ["x", "y"])
def test_kde_contour_conditional_returns_contourset(rng, conditional_on):
    x = rng.normal(size=300)
    y = 0.8 * x + rng.normal(size=300)
    _, ax = plt.subplots()
    contour_set = kde_contour(x, y, levels=[0.5], ax=ax, conditional_on=conditional_on)
    assert contour_set is not None
    assert np.all(np.isfinite(np.asarray(contour_set.levels)))
    plt.close(ax.figure)
