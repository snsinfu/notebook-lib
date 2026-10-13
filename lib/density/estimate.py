"""
Kernel density estimation on grids and probability-mass math.

These are the numerical density helpers. Point-wise estimation and contour
drawing, which are intended for plotting, live in `lib.plot.density`.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.stats

from .bandwidth import BandwidthMethod, resolve_bandwidth, scipy_bandwidth
from .kernels import (
    Kernel,
    resolve_kernel,
    scalar_epanechnikov_kernel,
    scalar_gaussian_kernel,
)
from .regression import kernel_regression

_SCALAR_KERNELS = {
    "gaussian": scalar_gaussian_kernel,
    "epanechnikov": scalar_epanechnikov_kernel,
}


def _check_weights(weights: npt.ArrayLike | None, n: int) -> np.ndarray | None:
    """
    Validate per-sample weights against `n` samples and return them as a
    `(n,)` float array, or `None`.
    """
    if weights is None:
        return None
    values = np.asarray(weights, dtype=float).ravel()
    if values.shape != (n,):
        raise ValueError("weights must have the same length as the samples")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("weights must be finite and non-negative")
    if values.sum() <= 0:
        raise ValueError("weights must not sum to zero")
    return values


def estimate_density_at(
    data: npt.ArrayLike,
    query: npt.ArrayLike,
    *,
    weights: npt.ArrayLike | None = None,
    kernel: str | Kernel = "scipy",
    bandwidth: float | npt.ArrayLike | BandwidthMethod | None = None,
) -> np.ndarray:
    """
    Evaluate a 1-D or 2-D KDE at arbitrary query points.

    `data` and `query` have shape `(dim, n)` and `(dim, m)`. `kernel` is
    `"scipy"`, an in-library kernel name, or a callable; `bandwidth` is
    absolute (see `lib.density.bandwidth`).

    Returns density values of shape `(m,)`.
    """
    data = np.asarray(data, dtype=float)
    query = np.asarray(query, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must be a (dim, n) array")
    if query.ndim != 2 or query.shape[0] != data.shape[0]:
        raise ValueError("query must be a (dim, m) array matching data")

    n = data.shape[1]
    weight_values = _check_weights(weights, n)

    if kernel == "scipy":
        kde = scipy.stats.gaussian_kde(
            data, bw_method=scipy_bandwidth(bandwidth), weights=weight_values
        )
        return kde(query)

    kernel_fn = resolve_kernel(kernel, data.shape[0])
    collected = np.ones(n) if weight_values is None else weight_values
    widths = resolve_bandwidth(bandwidth, data, collected)
    # Weighted density: (1 / (sum_i w_i * prod(h))) * sum_i w_i K((q - x_i) / h).
    density = kernel_regression(
        query.T,
        data.T,
        collected,
        kernel_fn=kernel_fn,
        scale=1 / widths,
        average=False,
    )
    return density / (collected.sum() * np.prod(widths))


def _joint_density(
    data: np.ndarray,
    *,
    gridsize: int,
    cut: float,
    weights: npt.ArrayLike | None,
    kernel: str | Kernel,
    bandwidth: float | npt.ArrayLike | BandwidthMethod | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray | None]:
    """
    Evaluate a 2-D KDE on a regular grid of `data` of shape `(2, n)`.

    Shared by `estimate_density` and `estimate_conditional_density`. Returns
    `(X, Y, Z, cell_area, width, weight_values)`: the meshgrid coordinates, the
    density, the area of one grid cell, the per-axis bandwidth, and the
    validated weights. `width` is resolved from the 2-D data, so heuristics use
    the 2-D Scott/Silverman factor.
    """
    if data.shape[1] < 2:
        raise ValueError("need at least two samples to fit a KDE")
    if gridsize < 2:
        raise ValueError("gridsize must be at least 2")
    weight_values = _check_weights(weights, data.shape[1])

    if kernel == "scipy":
        kde = scipy.stats.gaussian_kde(
            data, bw_method=scipy_bandwidth(bandwidth), weights=weight_values
        )
        # Per-dimension kernel bandwidth (std) = sqrt of the KDE covariance diag.
        width = np.sqrt(np.diag(kde.covariance))
    else:
        collected = np.ones(data.shape[1]) if weight_values is None else weight_values
        width = resolve_bandwidth(bandwidth, data, collected)

    lo = data.min(axis=1) - cut * width
    hi = data.max(axis=1) + cut * width

    gx = np.linspace(lo[0], hi[0], gridsize)
    gy = np.linspace(lo[1], hi[1], gridsize)
    X, Y = np.meshgrid(gx, gy)
    Z = estimate_density_at(
        data,
        np.vstack([X.ravel(), Y.ravel()]),
        weights=weights,
        kernel=kernel,
        bandwidth=bandwidth,
    ).reshape(X.shape)
    cell_area = (gx[1] - gx[0]) * (gy[1] - gy[0])
    return X, Y, Z, float(cell_area), width, weight_values


def estimate_density(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    gridsize: int = 200,
    cut: float = 3.0,
    weights: npt.ArrayLike | None = None,
    kernel: str | Kernel = "scipy",
    bandwidth: float | npt.ArrayLike | BandwidthMethod | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Evaluate a 2-D KDE of `(x, y)` on a regular grid.

    Parameters:
    - `x`, `y`: sample coordinates (same length).
    - `gridsize`: number of grid points along each dimension.
    - `cut`: grid padding, in bandwidths, beyond the data range.
    - `weights`: optional per-sample weights.
    - `kernel`: `"scipy"`, an in-library kernel name, or a callable.
    - `bandwidth`: absolute width, `None` (Scott's rule), or `bw_method(...)`.

    Returns `(X, Y, Z, cell_area)`: meshgrid coordinates, the density at each
    grid point, and the area of one grid cell for integrating the density.
    """
    data = np.vstack([np.asarray(x, float).ravel(), np.asarray(y, float).ravel()])
    X, Y, Z, cell_area, _, _ = _joint_density(
        data,
        gridsize=gridsize,
        cut=cut,
        weights=weights,
        kernel=kernel,
        bandwidth=bandwidth,
    )
    return X, Y, Z, cell_area


def _marginal_density(
    samples: np.ndarray,
    at: np.ndarray,
    *,
    kernel: str,
    width: float,
    weights: np.ndarray | None,
) -> np.ndarray:
    """
    Evaluate a 1-D KDE of `samples` at `at` with an absolute `width`.

    `kernel` is a name (`"scipy"` uses the Gaussian kernel). The result uses
    the same kernel and width as the parent joint estimate, so it is the exact
    marginal of a product-kernel estimate (or of scipy's Gaussian mixture).
    """
    scalar = scalar_gaussian_kernel if kernel == "scipy" else _SCALAR_KERNELS[kernel]
    if weights is None:
        weight_values = np.ones(samples.size)
    else:
        weight_values = weights
    scaled = (np.asarray(at, float)[:, None] - samples[None, :]) / width
    return (scalar(scaled) * weight_values[None, :]).sum(1) / (
        weight_values.sum() * width
    )


def estimate_conditional_density(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    given: str = "y",
    gridsize: int = 200,
    cut: float = 3.0,
    weights: npt.ArrayLike | None = None,
    kernel: str = "scipy",
    bandwidth: float | npt.ArrayLike | BandwidthMethod | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Evaluate the conditional density f(x|y) or f(y|x) on a regular grid.

    `given="y"` returns f(x|y) = f(x,y)/f(y); `given="x"` returns f(y|x). The
    marginal is evaluated with the same kernel and bandwidth as the joint, so
    each slice integrates to one over the varying axis (up to grid truncation).

    The result is the raw ratio, not renormalized; `lib.plot.density` draws its
    levels as tail mass of the plotted field. Only the named kernels are
    supported: a general callable has no defined 1-D product marginal.

    Parameters:
    - `x`, `y`: sample coordinates (same length).
    - `given`: `"y"` for f(x|y), `"x"` for f(y|x).
    - `gridsize`, `cut`: as in `estimate_density`.
    - `weights`: optional per-sample weights.
    - `kernel`: `"scipy"`, `"gaussian"`, or `"epanechnikov"`.
    - `bandwidth`: absolute width, `None` (Scott's rule), or `bw_method(...)`.

    Returns `(X, Y, Z, cell_area)`.
    """
    x_values = np.asarray(x, float).ravel()
    y_values = np.asarray(y, float).ravel()
    if x_values.shape != y_values.shape:
        raise ValueError("x and y must have the same length")
    if given not in {"x", "y"}:
        raise ValueError("given must be 'x' or 'y'")
    if kernel not in {"scipy", "gaussian", "epanechnikov"}:
        raise TypeError(
            "conditional density requires kernel='scipy', 'gaussian', or 'epanechnikov'"
        )

    data = np.vstack([x_values, y_values])
    X, Y, Z, cell_area, width, weight_values = _joint_density(
        data,
        gridsize=gridsize,
        cut=cut,
        weights=weights,
        kernel=kernel,
        bandwidth=bandwidth,
    )

    gx, gy = X[0, :], Y[:, 0]
    if given == "y":
        denom = _marginal_density(
            y_values, gy, kernel=kernel, width=width[1], weights=weight_values
        )[:, None]
    else:
        denom = _marginal_density(
            x_values, gx, kernel=kernel, width=width[0], weights=weight_values
        )[None, :]

    # Compact-support kernels (Epanechnikov) have a zero marginal outside the
    # sample support; Z/inf -> 0 there instead of 0/0 -> nan.
    Z = Z / np.where(denom > 0, denom, np.inf)
    return X, Y, Z, cell_area


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
