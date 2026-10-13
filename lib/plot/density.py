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
    DensityMeshAux,
    enclosed_thresholds,
    estimate_conditional_density_mesh,
    estimate_density_at,
    estimate_density_mesh,
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


def _check_min_density(value: float) -> float:
    """
    Validate a relative `min_density` and return it as a float.

    The value is the fraction of the peak joint density below which cells are
    masked, so it must lie in `[0, 1)`.
    """
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError("min_density must be a number")
    value = float(value)
    if not np.isfinite(value) or value < 0 or value >= 1:
        raise ValueError("min_density must be a fraction in [0, 1)")
    return value


def _suppress_low_density(
    Z: np.ndarray, joint: np.ndarray, min_density: float
) -> np.ndarray:
    """
    Set `Z` to `NaN` where the joint density is below `min_density * max(joint)`.

    `min_density` is relative to the peak joint density, `0` disables the mask.
    The result is the display field; callers compute contour thresholds from `Z`
    *before* masking so the levels do not move.
    """
    if min_density <= 0:
        return Z
    floor = min_density * float(np.max(joint))
    return np.where(joint < floor, np.nan, Z)


def _density_contour(
    X: npt.ArrayLike,
    Y: npt.ArrayLike,
    Z: npt.ArrayLike,
    levels: float | npt.ArrayLike = (0.5,),
    ax: matplotlib.axes.Axes | None = None,
    *,
    aux: DensityMeshAux,
    min_density: float = 0.0,
    fill: bool = False,
    **contour_kwargs: Any,
) -> matplotlib.contour.ContourSet:
    """
    Draw contours of a precomputed density field.

    Private low-level counterpart of `kde_contour`: it assumes `Z` is already
    evaluated on the mesh `X`, `Y` (as produced by
    `lib.density.estimate_density_mesh`), so any custom or modified density can be
    drawn in the same style. `levels` is the tail probability mass in `(0, 1)`:
    the fraction of mass lying outside each contour, so the enclosed region
    holds `1 - level`. The geometry is invariant to the overall scale of `Z`,
    because `lib.density.enclosed_thresholds` normalizes by
    `sum(Z) * aux.cell_area`.

    Parameters:
    - `X`, `Y`, `Z`: mesh coordinates and the field value at each grid point.
    - `levels`: tail probability mass for each contour, in `(0, 1)`.
    - `ax`: axes to draw into; the current axes is used if omitted.
    - `aux`: grid metadata from a mesh estimator; `aux.cell_area` integrates
      `Z` and `aux.joint` is the joint density `p(x, y)` on the same mesh.
    - `min_density`: fraction of the peak joint density below which cells are set
      to `NaN` (a support/reliability filter for sparse regions); `0` (default)
      disables it. Thresholds are computed before masking, so `levels` is
      unchanged.
    - `fill`: draw filled contours (`contourf`) instead of lines.
    - `contour_kwargs`: forwarded to `Axes.contour` / `Axes.contourf`.

    Returns the drawn contour set; its `.levels` holds the density thresholds.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    if Z.shape != X.shape or Z.shape != Y.shape:
        raise ValueError("X, Y and Z must have the same shape")
    if aux.joint.shape != Z.shape:
        raise ValueError("aux.joint must have the same shape as Z")
    _check_min_density(min_density)

    if ax is None:
        import matplotlib.pyplot as plt

        ax = plt.gca()

    thresholds = np.unique(enclosed_thresholds(Z, aux.cell_area, levels))
    Z = _suppress_low_density(Z, aux.joint, min_density)
    draw = ax.contourf if fill else ax.contour
    return draw(X, Y, Z, levels=np.sort(thresholds), **contour_kwargs)


def kde_contour(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    levels: float | npt.ArrayLike = (0.5,),
    ax: matplotlib.axes.Axes | None = None,
    *,
    conditional_on: str | None = None,
    gridsize: int | tuple[int, int] = 200,
    x_extent: tuple[float, float] | None = None,
    y_extent: tuple[float, float] | None = None,
    pad: float = 3.0,
    weights: npt.ArrayLike | None = None,
    kernel: str | Kernel = "scipy",
    bandwidth: float | npt.ArrayLike | BandwidthMethod | None = None,
    min_density: float = 0.0,
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
    - `conditional_on`: `None` (default) draws the joint density; `"y"` draws
      the conditional density of `x` given `y` (`f(x|y) = f(x,y)/f(y)`) and
      `"x"` draws `f(y|x)`. Not available with a callable `kernel`.
    - `gridsize`, `x_extent`, `y_extent`, `pad`: see
      `lib.density.estimate_density_mesh`. The extent sets the evaluation
      window; it does not change the axes limits.
    - `weights`: optional per-sample weights.
    - `kernel`: `"scipy"`, an in-library kernel name, or a callable.
    - `bandwidth`: absolute width, `None` (Scott's rule), or `bw_method(...)`.
    - `min_density`: fraction of the peak joint density `p(x,y)` below which the
      field is set to `NaN`. This is a support/reliability filter for sparse
      regions (for example, spurious conditional peaks where the conditioning
      variable has little data); it applies to both modes. `0` (default)
      disables it, and it does not change the meaning of `levels`.
    - `fill`: draw filled contours (`contourf`) instead of lines.
    - `contour_kwargs`: forwarded to `Axes.contour` / `Axes.contourf`.

    Returns the drawn contour set; its `.levels` holds the density thresholds.
    """
    if conditional_on is None:
        X, Y, Z, aux = estimate_density_mesh(
            x,
            y,
            gridsize=gridsize,
            x_extent=x_extent,
            y_extent=y_extent,
            pad=pad,
            weights=weights,
            kernel=kernel,
            bandwidth=bandwidth,
        )
    elif isinstance(kernel, str):
        X, Y, Z, aux = estimate_conditional_density_mesh(
            x,
            y,
            given=conditional_on,
            gridsize=gridsize,
            x_extent=x_extent,
            y_extent=y_extent,
            pad=pad,
            weights=weights,
            kernel=kernel,
            bandwidth=bandwidth,
        )
    else:
        raise TypeError("conditional_on cannot be used with a callable kernel")
    return _density_contour(
        X,
        Y,
        Z,
        levels=levels,
        ax=ax,
        aux=aux,
        min_density=min_density,
        fill=fill,
        **contour_kwargs,
    )
