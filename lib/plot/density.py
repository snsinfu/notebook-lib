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


def _as_points(x: npt.ArrayLike) -> np.ndarray:
    """
    Return `x` as an `(n, d)` float array, promoting 1-D input.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    return x


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
    data: npt.ArrayLike,
    at: npt.ArrayLike | None = None,
    *,
    method: str = "scipy",
    bw_method: str | float | Callable[..., float] | None = None,
    kernel: str | Kernel = "gaussian",
    bandwidth: float | npt.ArrayLike = 1.0,
) -> np.ndarray:
    """
    Evaluate a KDE of `data` at the query points `at`.

    Intended for plotting (e.g. density-colored scatter); no axes are drawn.

    Parameters:
    - `data`: `(n, d)` sample coordinates (a 1-D array is treated as `d = 1`).
    - `at`: `(m, d)` query points; defaults to `data`.
    - `method`: `"scipy"` uses `scipy.stats.gaussian_kde`; `"kernel"` uses this
      package's kernels and the `bandwidth` argument.
    - `bw_method`: passed to `scipy.stats.gaussian_kde` (`"scipy"` only).
    - `kernel`: kernel name or function (`"kernel"` only).
    - `bandwidth`: scalar or per-axis bandwidth (`"kernel"` only).

    Returns density values of shape `(m,)`.
    """
    data = _as_points(data)
    at = data if at is None else _as_points(at)
    dim = data.shape[1]

    if method not in {"scipy", "kernel"}:
        raise ValueError(f"unknown method {method!r}; expected 'scipy' or 'kernel'")

    if method == "scipy":
        import scipy.stats

        kde = scipy.stats.gaussian_kde(data.T, bw_method=bw_method)
        return kde(at.T)

    if at.shape[1] != dim:
        raise ValueError("at must have the same number of columns as data")
    kernel_fn = _resolve_kernel(kernel, dim)
    bandwidth = np.asarray(bandwidth, dtype=float)
    if np.any(bandwidth <= 0):
        raise ValueError("bandwidth must be positive")
    bandwidth = np.broadcast_to(bandwidth, (dim,))
    scale = 1 / bandwidth
    # Raw sum of kernel weights, then the KDE normalization constants:
    # (1 / (n * prod(h))) * sum_i K((x - x_i) / h).
    density = kernel_regression(
        at, data, np.array(1.0), kernel_fn=kernel_fn, scale=scale, average=False
    )
    return density / (data.shape[0] * np.prod(bandwidth))


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
