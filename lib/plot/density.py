"""
Plot-facing kernel density estimation and contour helpers.

These functions are meant for plotting (for example density-colored scatter or
contour lines) even though they do not draw on their own. Pure numerical
estimation lives in `lib.density`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ..density.estimate import enclosed_thresholds, estimate_density
from ..density.kernels import Kernel, make_epanechnikov_kernel, make_gaussian_kernel
from ..density.regression import kernel_regression

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.contour


def _resolve_kernel(kernel: str | Kernel, dim: int) -> Kernel:
    """
    Return a compiled `dim`-dimensional kernel for a name or callable.
    """
    if not isinstance(kernel, str):
        return kernel
    factories = {
        "gaussian": make_gaussian_kernel,
        "epanechnikov": make_epanechnikov_kernel,
    }
    if kernel not in factories:
        raise ValueError(
            f"unrecognized kernel {kernel!r}; expected one of {sorted(factories)}"
        )
    return factories[kernel](dim)


def kde_at(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    at: npt.ArrayLike | None = None,
    *,
    weights: npt.ArrayLike | None = None,
    method: str = "scipy",
    bw_method: str | float | Callable[..., float] | None = None,
    kernel: str | Kernel = "gaussian",
    bandwidth: float | npt.ArrayLike = 1.0,
) -> np.ndarray:
    """
    Evaluate a KDE of the samples `(x, y)`.

    Intended for plotting (e.g. density-colored scatter); no axes are drawn. By
    default the density is evaluated at the sample points, so the result can be
    used directly as a color array.

    Parameters:
    - `x`, `y`: sample coordinates (same length).
    - `at`: optional `(2, m)` query array, or a pair `(qx, qy)` of equal-length
      arrays; defaults to the samples.
    - `weights`: optional per-sample weights, same length as `x`. The density is
      normalized by `sum(weights)`, so it integrates to one; equal weights
      reproduce the unweighted estimate.
    - `method`: `"scipy"` uses `scipy.stats.gaussian_kde`; `"kernel"` uses this
      package's kernels and the `bandwidth` argument.
    - `bw_method`: passed to `scipy.stats.gaussian_kde` (`"scipy"` only).
    - `kernel`: kernel name or function (`"kernel"` only).
    - `bandwidth`: scalar or per-axis bandwidth (`"kernel"` only).

    Returns density values of shape `(m,)`.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length")
    data = np.vstack([x, y])  # (2, n)

    if weights is None:
        weight_values = np.ones(data.shape[1])
    else:
        weight_values = np.asarray(weights, dtype=float).ravel()
        if weight_values.shape != x.shape:
            raise ValueError("weights must have the same length as x and y")
        if not np.all(np.isfinite(weight_values)) or np.any(weight_values < 0):
            raise ValueError("weights must be finite and non-negative")
        if weight_values.sum() <= 0:
            raise ValueError("weights must not sum to zero")

    query = data if at is None else np.asarray(at, dtype=float)
    if query.ndim != 2 or query.shape[0] != 2:
        raise ValueError("at must be a (2, m) array or a pair of coordinate arrays")

    if method not in {"scipy", "kernel"}:
        raise ValueError(f"unknown method {method!r}; expected 'scipy' or 'kernel'")

    if method == "scipy":
        import scipy.stats

        kde = scipy.stats.gaussian_kde(data, bw_method=bw_method, weights=weight_values)
        return kde(query)

    kernel_fn = _resolve_kernel(kernel, 2)
    bandwidth = np.asarray(bandwidth, dtype=float)
    if np.any(bandwidth <= 0):
        raise ValueError("bandwidth must be positive")
    bandwidth = np.broadcast_to(bandwidth, (2,))
    scale = 1 / bandwidth
    # Weighted density: (1 / (sum_i w_i * prod(h))) * sum_i w_i K((q - x_i) / h).
    density = kernel_regression(
        query.T,
        data.T,
        weight_values,
        kernel_fn=kernel_fn,
        scale=scale,
        average=False,
    )
    return density / (weight_values.sum() * np.prod(bandwidth))


def kde_contour(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    levels: float | npt.ArrayLike = (0.5,),
    ax: matplotlib.axes.Axes | None = None,
    *,
    gridsize: int = 200,
    cut: float = 3.0,
    bw_method: str | float | Callable[..., float] | None = None,
    fill: bool = False,
    **contour_kwargs: Any,
) -> matplotlib.contour.ContourSet:
    """
    Draw KDE contour(s) enclosing the requested probability mass.

    Parameters:
    - `x`, `y`: sample coordinates.
    - `levels`: tail probability mass for each contour, in `(0, 1)`: the
      fraction of mass lying outside the contour. The enclosed mass is
      `1 - level`, so `0.5` gives the median contour.
    - `ax`: axes to draw into; the current axes is used if omitted.
    - `gridsize`, `cut`, `bw_method`: see `lib.density.estimate_density`.
    - `fill`: draw filled contours (`contourf`) instead of lines.
    - `contour_kwargs`: forwarded to `Axes.contour` / `Axes.contourf`.

    Returns the drawn contour set; its `.levels` holds the density thresholds.
    """
    if ax is None:
        import matplotlib.pyplot as plt

        ax = plt.gca()

    X, Y, Z, cell_area = estimate_density(
        x, y, gridsize=gridsize, cut=cut, bw_method=bw_method
    )
    thresholds = np.unique(enclosed_thresholds(Z, cell_area, levels))

    draw = ax.contourf if fill else ax.contour
    return draw(X, Y, Z, levels=np.sort(thresholds), **contour_kwargs)
