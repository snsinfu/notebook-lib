import pathlib
import subprocess
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.stats

from lib.plot import (
    attach_colorbar,
    get_x_grid_line_at,
    get_y_grid_line_at,
    kde_at,
    kde_contour,
    register_colormaps,
    set_axis,
    smooth_matrix,
    texsci,
    use_log_ticks,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_texsci_positive_exponent():
    assert texsci(1.23e5, 3) == r"1.23 \times 10^{5}"


def test_texsci_negative_exponent():
    assert texsci(1.23e-5, 3) == r"1.23 \times 10^{-5}"


def test_texsci_plain_value():
    assert texsci(1.23, 3) == "1.23"


def test_scipy_density_matches_gaussian_kde(rng):
    data = rng.normal(size=(200, 1))
    at = data[:5]
    got = kde_at(data, at, method="scipy")
    expected = scipy.stats.gaussian_kde(data.T)(at.T)
    assert np.allclose(got, expected)


def test_kernel_density_integrates_to_one(rng):
    # Regression: the kernel path returned sum(K) without 1/n or 1/prod(h).
    data = rng.normal(size=(500, 2))
    grid = np.linspace(-5, 5, 121)
    X, Y = np.meshgrid(grid, grid)
    at = np.column_stack([X.ravel(), Y.ravel()])
    density = kde_at(data, at, method="kernel", bandwidth=0.5).reshape(X.shape)
    cell_area = (grid[1] - grid[0]) ** 2
    assert np.isclose(density.sum() * cell_area, 1.0, atol=0.05)


def test_kde_at_rejects_negative_bandwidth(rng):
    data = rng.normal(size=(50, 2))
    with pytest.raises(ValueError):
        kde_at(data, method="kernel", bandwidth=-1.0)


def test_kde_at_rejects_unknown_method(rng):
    data = rng.normal(size=(50, 2))
    with pytest.raises(ValueError):
        kde_at(data, method="histogram")


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


def test_kde_contour_returns_contourset(rng):
    x = rng.normal(size=300)
    y = rng.normal(size=300)
    _, ax = plt.subplots()
    contour_set = kde_contour(x, y, levels=[0.5], ax=ax)
    assert contour_set is not None
    plt.close(ax.figure)
