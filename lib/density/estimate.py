"""
Kernel density estimation on grids and probability-mass math.

These are the numerical density helpers. Point-wise estimation and contour
drawing, which are intended for plotting, live in `lib.plot.density`.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import scipy.stats
from scipy.signal import find_peaks

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


class DensityMeshAux(NamedTuple):
    """
    Grid metadata returned alongside a density mesh.

    `cell_area` is the area of one grid cell, used to integrate the field.
    `joint` is the joint density `p(x, y)` on the mesh: for a joint estimate it
    is the returned field itself, and for a conditional estimate it is the
    numerator `f(x, y)` before the division by the marginal.
    """

    cell_area: float
    joint: np.ndarray


class ConditionalRidgeAux(NamedTuple):
    """
    Per-row diagnostics returned alongside a conditional ridge.

    All arrays are aligned with the conditioning axis of the mesh (length
    `gridsize`).

    - `support`: the joint density `p(x, y)` at the conditioning value is above
      the `min_density` floor.
    - `ambiguous`: the slice has more than one mode (`n_modes >= 2`).
    - `n_modes`: number of modes resolved in the slice.
    - `contrast`: `(max - median) / max` of the slice density.
    """

    support: np.ndarray
    ambiguous: np.ndarray
    n_modes: np.ndarray
    contrast: np.ndarray


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

    Shared by `estimate_density_mesh` and `estimate_conditional_density_mesh`.
    Returns `(X, Y, Z, cell_area, width, weight_values)`: the meshgrid
    coordinates, the density, the area of one grid cell, the per-axis bandwidth,
    and the validated weights. `width` is resolved from the 2-D data, so
    heuristics use the 2-D Scott/Silverman factor.
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


def estimate_density_mesh(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    gridsize: int = 200,
    cut: float = 3.0,
    weights: npt.ArrayLike | None = None,
    kernel: str | Kernel = "scipy",
    bandwidth: float | npt.ArrayLike | BandwidthMethod | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, DensityMeshAux]:
    """
    Evaluate a 2-D KDE of `(x, y)` on a regular grid.

    Parameters:
    - `x`, `y`: sample coordinates (same length).
    - `gridsize`: number of grid points along each dimension.
    - `cut`: grid padding, in bandwidths, beyond the data range.
    - `weights`: optional per-sample weights.
    - `kernel`: `"scipy"`, an in-library kernel name, or a callable.
    - `bandwidth`: absolute width, `None` (Scott's rule), or `bw_method(...)`.

    Returns `(X, Y, Z, aux)`: meshgrid coordinates, the density at each grid
    point, and a `DensityMeshAux` whose `cell_area` integrates the density and
    whose `joint` is the density field itself.
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
    return X, Y, Z, DensityMeshAux(float(cell_area), Z)


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
    scaled = (at[:, None] - samples[None, :]) / width
    return (scalar(scaled) * weight_values[None, :]).sum(1) / (
        weight_values.sum() * width
    )


def estimate_conditional_density_mesh(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    given: str = "y",
    gridsize: int = 200,
    cut: float = 3.0,
    weights: npt.ArrayLike | None = None,
    kernel: str = "scipy",
    bandwidth: float | npt.ArrayLike | BandwidthMethod | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, DensityMeshAux]:
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
    - `gridsize`, `cut`: as in `estimate_density_mesh`.
    - `weights`: optional per-sample weights.
    - `kernel`: `"scipy"`, `"gaussian"`, or `"epanechnikov"`.
    - `bandwidth`: absolute width, `None` (Scott's rule), or `bw_method(...)`.

    Returns `(X, Y, Z, aux)`: meshgrid coordinates, the conditional density at
    each grid point, and a `DensityMeshAux` whose `cell_area` integrates the
    field and whose `joint` is the numerator `f(x, y)` before division.
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
    joint = Z
    Z = Z / np.where(denom > 0, denom, np.inf)
    return X, Y, Z, DensityMeshAux(float(cell_area), joint)


def _slice_modes(
    z: np.ndarray,
    axis_coord: np.ndarray,
    bandwidth: float,
    *,
    min_prominence: float,
    min_separation: float,
) -> tuple[int, float]:
    """
    Count the modes of one conditional slice.

    `z` is the density along `axis_coord` (the axis being maximized). A peak is
    counted when its prominence and height exceed `min_prominence` (a fraction
    of the slice maximum), and a secondary peak is counted only when it is
    farther than `min_separation` bandwidths from the dominant one. Returns
    `(n_modes, contrast)`, where `contrast` is `(max - median) / max`.
    """
    zmax = float(np.max(z))
    if zmax <= 0 or not np.isfinite(zmax):
        return 0, 0.0

    contrast = (zmax - float(np.median(z))) / zmax

    peaks, props = find_peaks(z, prominence=0.0)
    if peaks.size == 0:
        return 0, contrast

    kept = (props["prominences"] > min_prominence * zmax) & (
        z[peaks] > min_prominence * zmax
    )
    modes = peaks[kept]
    if modes.size == 0:
        return 0, contrast

    modes = modes[np.argsort(z[modes])[::-1]]
    main = int(modes[0])
    separation = min_separation * bandwidth
    secondary = [
        int(mode)
        for mode in modes[1:]
        if abs(axis_coord[mode] - axis_coord[main]) > separation
    ]
    return 1 + len(secondary), contrast


def _trace_ridge(field: np.ndarray, jump_penalty: float) -> np.ndarray:
    """
    Trace one continuous ridge through the rows of `field`.

    `field` has shape `(n_cond, n_max)`. Each row is normalized by its maximum
    and a Viterbi pass maximizes the summed density minus a quadratic jump cost
    `jump_penalty * (column - previous column) ** 2` between adjacent rows.
    Returns the maximizing column index for each row.
    """
    n_cond, n_max = field.shape
    normalized = field / np.maximum(
        field.max(axis=1, keepdims=True), np.finfo(float).tiny
    )
    columns = np.arange(n_max)
    penalty = jump_penalty * (columns[:, None] - columns[None, :]) ** 2

    score = np.empty((n_cond, n_max))
    back = np.zeros((n_cond, n_max), dtype=int)
    score[0] = normalized[0]
    for i in range(1, n_cond):
        candidates = score[i - 1][:, None] - penalty
        back[i] = np.argmax(candidates, axis=0)
        score[i] = normalized[i] + candidates[back[i], columns]

    path = np.empty(n_cond, dtype=int)
    path[-1] = int(np.argmax(score[-1]))
    for i in range(n_cond - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def estimate_conditional_ridge(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    given: str = "y",
    gridsize: int = 200,
    cut: float = 3.0,
    weights: npt.ArrayLike | None = None,
    kernel: str | Kernel = "scipy",
    bandwidth: float | npt.ArrayLike | BandwidthMethod | None = None,
    min_density: float = 0.0,
    min_prominence: float = 0.25,
    min_separation: float = 2.0,
    jump_penalty: float = 0.02,
    on_ambiguous: str = "dominant",
) -> tuple[np.ndarray, np.ndarray, ConditionalRidgeAux]:
    """
    Estimate the ridge (mode locus) of a conditional density.

    `given="y"` returns the mode of `p(x|y)` at each `y`, as `x` against `y`;
    `given="x"` returns the mode of `p(y|x)` at each `x`. For a fixed
    conditioning value the marginal is constant, so the mode is a property of
    the joint density `p(x, y)`; the estimate is built from the joint mesh. A
    Viterbi pass keeps the curve continuous and selects one branch at
    crossings.

    Rows where the joint density is below `min_density * max(p(x, y))` are set
    to NaN (disabled by default). Other rows are returned as traced, with their
    per-row diagnostics in `ConditionalRidgeAux`.

    Parameters:
    - `x`, `y`: sample coordinates (same length).
    - `given`: `"y"` returns `x` as a function of `y`; `"x"` returns `y` as a
      function of `x`.
    - `gridsize`, `cut`: as in `estimate_density_mesh`.
    - `weights`: optional per-sample weights.
    - `kernel`: `"scipy"`, an in-library kernel name, or a callable.
    - `bandwidth`: absolute width, `None` (Scott's rule), or `bw_method(...)`.
    - `min_density`: joint-density floor, a fraction of the peak `p(x, y)` in
      `[0, 1)`; rows below it are set to NaN. `0` (default) disables the mask.
    - `min_prominence`: prominence and height floor for a peak to count as a
      mode, as a fraction of the slice maximum.
    - `min_separation`: minimum distance between two modes, in bandwidths.
    - `jump_penalty`: Viterbi jump cost, in squared grid columns.
    - `on_ambiguous`: action where `aux.ambiguous` is True: `"dominant"`
      (default) keeps the dominant branch, `"nan"` sets those rows to NaN, and
      `"raise"` raises `ValueError`.

    Returns `(ridge_x, ridge_y, aux)`: the traced ridge and its diagnostics.
    """
    if given not in {"x", "y"}:
        raise ValueError("given must be 'x' or 'y'")
    if on_ambiguous not in {"dominant", "nan", "raise"}:
        raise ValueError("on_ambiguous must be 'dominant', 'nan', or 'raise'")
    if not np.isfinite(min_density) or min_density < 0 or min_density >= 1:
        raise ValueError("min_density must be a fraction in [0, 1)")
    if not np.isfinite(min_prominence) or not 0 <= min_prominence <= 1:
        raise ValueError("min_prominence must be a fraction in [0, 1]")
    if not np.isfinite(min_separation) or min_separation < 0:
        raise ValueError("min_separation must be non-negative")
    if not np.isfinite(jump_penalty) or jump_penalty < 0:
        raise ValueError("jump_penalty must be non-negative")

    x_values = np.asarray(x, float).ravel()
    y_values = np.asarray(y, float).ravel()
    if x_values.shape != y_values.shape:
        raise ValueError("x and y must have the same length")

    data = np.vstack([x_values, y_values])
    X, Y, Z, _, width, _ = _joint_density(
        data,
        gridsize=gridsize,
        cut=cut,
        weights=weights,
        kernel=kernel,
        bandwidth=bandwidth,
    )

    if given == "y":
        field = Z
        max_coord = X[0, :]
        cond_coord = Y[:, 0]
        max_h = float(width[0])
        support = Z.max(axis=1) >= min_density * float(Z.max())
    else:
        field = Z.T
        max_coord = Y[:, 0]
        cond_coord = X[0, :]
        max_h = float(width[1])
        support = Z.max(axis=0) >= min_density * float(Z.max())

    n_cond = field.shape[0]
    n_modes = np.zeros(n_cond, dtype=int)
    contrast = np.zeros(n_cond)
    for i in range(n_cond):
        n_modes[i], contrast[i] = _slice_modes(
            field[i],
            max_coord,
            max_h,
            min_prominence=min_prominence,
            min_separation=min_separation,
        )

    path = _trace_ridge(field, jump_penalty)
    ridge_max = max_coord[path]

    ambiguous = n_modes >= 2
    ridge_max = np.where(support, ridge_max, np.nan)
    if on_ambiguous == "nan":
        ridge_max = np.where(ambiguous, np.nan, ridge_max)
    elif on_ambiguous == "raise" and np.any(ambiguous):
        raise ValueError(
            "ridge is ambiguous (multimodal) for some conditioning values; "
            "use on_ambiguous='dominant' or 'nan'"
        )

    aux = ConditionalRidgeAux(support, ambiguous, n_modes, contrast)
    if given == "y":
        return ridge_max, cond_coord, aux
    return cond_coord, ridge_max, aux


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
