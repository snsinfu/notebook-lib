"""
Colormaps and named-cycle color helpers.
"""

from __future__ import annotations

import typing
from collections.abc import Iterable, Sequence
from typing import Any

import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np


def register_cmap(cm: matplotlib.colors.Colormap) -> matplotlib.colors.Colormap:
    """
    Register `cm` with matplotlib, accounting for the API change.
    """
    register_legacy = getattr(matplotlib.cm, "register_cmap", None)
    if register_legacy is not None:
        register_legacy(cmap=cm)
    else:
        matplotlib.colormaps.register(cm)
    return cm


def lerp_palette(
    nodes: dict[float, Any],
    ndiv: int = 20,
) -> np.ndarray:
    """
    Linearly interpolate color `nodes` into `ndiv` RGBA colors.
    """
    points = np.array(sorted(nodes.keys()))
    colors = np.array([matplotlib.colors.to_rgba(nodes[key]) for key in points])
    x = np.linspace(points[0], points[-1], num=ndiv)
    return np.transpose(
        [np.interp(x, points, colors[:, i]) for i in range(colors.shape[1])]
    )


def _slice_linear_cmap(
    cmap: matplotlib.colors.Colormap,
    lo: float,
    hi: float,
    name: str,
) -> matplotlib.colors.LinearSegmentedColormap:
    """
    Resample `cmap` over `[lo, hi]` into a new linear colormap named `name`.
    """
    n = cmap.N
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        name,
        cmap(np.linspace(lo, hi, n)),
        N=n,
    )


def _slice_listed_cmap(
    cmap: matplotlib.colors.ListedColormap,
    lo: float,
    hi: float,
    name: str,
) -> matplotlib.colors.ListedColormap:
    """
    Trim whole colors from the ends of `cmap` and rename the result `name`.
    """
    n = cmap.N
    colors = np.asarray(cmap.colors)[round(lo * n) : n - round((1.0 - hi) * n)]
    if len(colors) == 0:
        raise ValueError("slice removes all colors from the colormap")
    return matplotlib.colors.ListedColormap(colors, name=name)


def slice_cmap(
    cmap: matplotlib.colors.Colormap,
    lo: float,
    hi: float,
    name: str | None = None,
) -> matplotlib.colors.Colormap:
    """
    Return a copy of `cmap` restricted to the fractional range `[lo, hi]`.

    `lo` and `hi` are fractions in `[0, 1]`, so `slice_cmap(cmap, 0.1, 0.9)`
    trims the first and last 10%. A `ListedColormap` loses `round(lo * N)`
    colors from the front and `round((1 - hi) * N)` from the back and stays
    listed; any other colormap is resampled over `[lo, hi]` into a
    `LinearSegmentedColormap`.
    """
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError(f"require 0 <= lo < hi <= 1, got lo={lo}, hi={hi}")
    cmap_name = name if name is not None else f"{cmap.name}_sliced"
    if isinstance(cmap, matplotlib.colors.ListedColormap):
        return _slice_listed_cmap(cmap, lo, hi, cmap_name)
    return _slice_linear_cmap(cmap, lo, hi, cmap_name)


# "fall" colormap with nicer extrema, nice for showing contact heat map.
cm_wrr = matplotlib.colors.LinearSegmentedColormap.from_list(
    "wrr",
    lerp_palette(
        {
            0.00: (1.00, 1.00, 1.00),
            0.01: (1.00, 0.95, 0.90),
            0.50: (1.00, 0.60, 0.00),
            1.00: (0.55, 0.00, 0.00),
        }
    ),
)

# Blue-red colormap akin to bwr but with black center.
cm_bkr = matplotlib.colors.LinearSegmentedColormap.from_list(
    "bkr", [(0, 0, 1), (0, 0, 0), (1, 0, 0)]
)


def register_colormaps() -> None:
    """
    Register `cm_wrr` and `cm_bkr` with matplotlib under their names.
    """
    register_cmap(cm_wrr)
    register_cmap(cm_bkr)


class _OkLAB(typing.NamedTuple):
    l: float
    a: float
    b: float


def _oklab(r: float, g: float, b: float) -> _OkLAB:
    """
    Convert an sRGB triple to OkLab.
    """
    # https://bottosson.github.io/posts/oklab/
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

    l_ = l ** (1 / 3)
    m_ = m ** (1 / 3)
    s_ = s ** (1 / 3)

    return _OkLAB(
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


def _oklab_distance(x: _OkLAB, y: _OkLAB) -> float:
    """
    Euclidean distance between two OkLab colors.
    """
    lx, ax, bx = x
    ly, ay, by = y
    return ((lx - ly) ** 2 + (ax - ay) ** 2 + (bx - by) ** 2) ** (1 / 2)


def _find_nearest_color(
    colors: Sequence[Any],
    reference_color: Any,
) -> Any:
    """
    Return the color in `colors` nearest to `reference_color` in OkLab.
    """
    reference_lab = _oklab(*matplotlib.colors.to_rgb(reference_color))
    evaluations = [
        (
            color,
            _oklab_distance(_oklab(*matplotlib.colors.to_rgb(color)), reference_lab),
        )
        for color in colors
    ]
    match min(evaluations, default=None, key=lambda entry: entry[1]):
        case (color, _):
            return color
        case None:
            return None


def register_cycle_palette() -> None:
    """
    Register `"C:<name>"` named colors for the current color cycle.
    """
    matplotlib.colors.get_named_colors_mapping().update(
        {
            f"C:{name}": color
            for name, color in _make_cycle_palette(
                plt.rcParams["axes.prop_cycle"]
            ).items()
        }
    )


def _make_cycle_palette(
    prop_cycle: Iterable[dict[str, object]],
    base_colors: list[str] | None = None,
) -> dict[str, Any]:
    """
    Map base color names to the nearest colors of a property cycle.
    """
    if base_colors is None:
        base_colors = list("rgbcmykw")
    prop_colors = [prop["color"] for prop in prop_cycle]
    return {name: _find_nearest_color(prop_colors, name) for name in base_colors}
