"""
Plot-facing kernel density estimation and contour helpers.

These functions are meant for plotting (for example density-colored scatter or
contour lines) even though they do not draw on their own. Pure numerical
estimation lives in `lib.density`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ..density.bandwidth import BandwidthMethod
from ..density.estimate import (
    enclosed_thresholds,
    estimate_density,
    estimate_density_at,
)
from ..density.kernels import Kernel

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.contour


def kde_at(
    x: npt.ArrayLike,
    y: npt.ArrayLike | None = None,
    *,
    at: npt.ArrayLike | None = None,
    weights: npt.ArrayLike | None = None,
    kernel: str | Kernel = "scipy",
    bandwidth: float | npt.ArrayLike | BandwidthMethod | None = None,
) -> np.ndarray:
    """
    Evaluate a 1-D or 2-D KDE.

    Intended for plotting (e.g. density-colored scatter or line); no axes are
    drawn. By default the density is evaluated at the sample points, so the
    result can be used directly as a color array.

    Parameters:
    - `x`: sample coordinates for the first (only) axis.
    - `y`: optional sample coordinates for the second axis; omit it (or pass
      `None`) for a 1-D estimate.
    - `at`: optional query points; defaults to the samples. For a 1-D estimate a
      flat array of query coordinates; for a 2-D estimate a `(2, m)` array or a
      pair `(qx, qy)` of equal-length arrays. Keyword-only because its meaning
      depends on whether `y` is given.
    - `weights`: optional per-sample weights, same length as `x`. The density is
      normalized by `sum(weights)`, so it integrates to one; equal weights
      reproduce the unweighted estimate.
    - `kernel`: `"scipy"` (default) uses `scipy.stats.gaussian_kde`;
      `"gaussian"` or `"epanechnikov"` use this package's kernels; a callable
      is used as a custom kernel.
    - `bandwidth`: absolute kernel width in data units, as a positive scalar or
      per-axis array, or `None` for Scott's rule. For a scipy-convention
      relative width use `bw_method("scott")`, `bw_method("silverman")`, or
      `bw_method(factor)`. With `kernel="scipy"` only the relative forms (and
      `None`) are accepted.

    Returns density values of shape `(m,)`.
    """
    x_values = np.asarray(x, dtype=float).ravel()
    if y is None:
        dim = 1
        data = x_values[None, :]  # (1, n)
    else:
        y_values = np.asarray(y, dtype=float).ravel()
        if x_values.shape != y_values.shape:
            raise ValueError("x and y must have the same length")
        dim = 2
        data = np.vstack([x_values, y_values])  # (2, n)

    if at is None:
        query = data
    else:
        query = np.asarray(at, dtype=float)
        if dim == 1:
            if query.ndim != 1:
                raise ValueError(
                    "at must be a 1-D array of query points when y is omitted"
                )
            query = query[None, :]  # (1, m)
        elif query.ndim != 2 or query.shape[0] != dim:
            raise ValueError("at must be a (2, m) array or a pair of coordinate arrays")

    return estimate_density_at(
        data, query, weights=weights, kernel=kernel, bandwidth=bandwidth
    )


def kde_contour(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    levels: float | npt.ArrayLike = (0.5,),
    ax: matplotlib.axes.Axes | None = None,
    *,
    gridsize: int = 200,
    cut: float = 3.0,
    weights: npt.ArrayLike | None = None,
    kernel: str | Kernel = "scipy",
    bandwidth: float | npt.ArrayLike | BandwidthMethod | None = None,
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
    - `gridsize`, `cut`: see `lib.density.estimate_density`.
    - `weights`: optional per-sample weights.
    - `kernel`: `"scipy"`, an in-library kernel name, or a callable.
    - `bandwidth`: absolute width, `None` (Scott's rule), or `bw_method(...)`.
    - `fill`: draw filled contours (`contourf`) instead of lines.
    - `contour_kwargs`: forwarded to `Axes.contour` / `Axes.contourf`.

    Returns the drawn contour set; its `.levels` holds the density thresholds.
    """
    if ax is None:
        import matplotlib.pyplot as plt

        ax = plt.gca()

    X, Y, Z, cell_area = estimate_density(
        x,
        y,
        gridsize=gridsize,
        cut=cut,
        weights=weights,
        kernel=kernel,
        bandwidth=bandwidth,
    )
    thresholds = np.unique(enclosed_thresholds(Z, cell_area, levels))

    draw = ax.contourf if fill else ax.contour
    return draw(X, Y, Z, levels=np.sort(thresholds), **contour_kwargs)
