"""
Kernel density estimation on grids and probability-mass math.

These are the numerical density helpers. Point-wise estimation and contour
drawing, which are intended for plotting, live in `lib.plot.density`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import scipy.stats


def estimate_density(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    gridsize: int = 200,
    cut: float = 3.0,
    bw_method: str | float | Callable[..., float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Evaluate a Gaussian KDE of `(x, y)` on a regular grid.

    Parameters:
    - `x`, `y`: sample coordinates (same length).
    - `gridsize`: number of grid points along each dimension.
    - `cut`: grid padding, in bandwidths, beyond the data range.
    - `bw_method`: passed to `scipy.stats.gaussian_kde`.

    Returns `(X, Y, Z, cell_area)`: meshgrid coordinates, the density at each
    grid point, and the area of one grid cell for integrating the density.
    """
    data = np.vstack([np.asarray(x, float).ravel(), np.asarray(y, float).ravel()])
    if data.shape[1] < 2:
        raise ValueError("need at least two samples to fit a KDE")
    if gridsize < 2:
        raise ValueError("gridsize must be at least 2")

    kde = scipy.stats.gaussian_kde(data, bw_method=bw_method)

    # Per-dimension kernel bandwidth (std) = sqrt of the KDE covariance diag.
    bandwidth = np.sqrt(np.diag(kde.covariance))
    lo = data.min(axis=1) - cut * bandwidth
    hi = data.max(axis=1) + cut * bandwidth

    gx = np.linspace(lo[0], hi[0], gridsize)
    gy = np.linspace(lo[1], hi[1], gridsize)
    X, Y = np.meshgrid(gx, gy)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    cell_area = (gx[1] - gx[0]) * (gy[1] - gy[0])
    return X, Y, Z, float(cell_area)


def enclosed_thresholds(
    Z: npt.ArrayLike,
    cell_area: float,
    levels: npt.ArrayLike,
) -> np.ndarray:
    """
    Density thresholds for contours at the given tail-mass levels.

    `level` is the fraction of probability mass lying *outside* the contour, so
    the enclosed region `{density >= threshold}` contains `1 - level`. The
    density is sorted once, so the cost is independent of how many levels are
    requested.

    Returns an array aligned with `levels`.
    """
    levels = np.atleast_1d(np.asarray(levels, float))
    if np.any((levels <= 0) | (levels >= 1)):
        raise ValueError("levels must be fractions strictly between 0 and 1")

    flat = np.asarray(Z, float).ravel()
    sorted_density = flat[np.argsort(flat)[::-1]]
    cum_mass = np.cumsum(sorted_density) * cell_area
    total_mass = cum_mass[-1]

    idx = np.searchsorted(cum_mass, (1.0 - levels) * total_mass)
    idx = np.clip(idx, 0, sorted_density.size - 1)
    return sorted_density[idx]
