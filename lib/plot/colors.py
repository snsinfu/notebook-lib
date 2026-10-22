"""
Colormaps and named-cycle color helpers.
"""

from __future__ import annotations

import typing
from collections.abc import Iterable, Sequence
from typing import Any, Literal

import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate


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
    nodes: dict[float, Any] | Sequence[Any],
    ndiv: int = 20,
    *,
    colorspace: Literal["rgb", "oklab"] = "rgb",
) -> np.ndarray:
    """
    Interpolate color `nodes` into `ndiv` RGBA colors.

    `nodes` is a mapping from positions to colors, or a sequence of colors at
    uniformly spaced positions in `[0, 1]`. A color is an `(r, g, b)` or
    `(r, g, b, a)` triple, a color name, or an `OkLAB` object (alpha 1).
    `colorspace="oklab"` blends in OkLab and clips to the sRGB gamut;
    `"rgb"` blends the gamma-encoded RGBA channels linearly (matplotlib's
    convention, no linearization). A single node is constant.
    """
    if colorspace not in ("rgb", "oklab"):
        raise ValueError(f"unknown colorspace {colorspace!r}")

    if isinstance(nodes, dict):
        keys = sorted(nodes)
        points = np.array(keys, dtype=float)
        colors = [nodes[key] for key in keys]
    else:
        colors = list(nodes)
        points = np.linspace(0.0, 1.0, len(colors))

    if len(points) == 0:
        raise ValueError("require at least one color node")

    if colorspace == "oklab":
        parsed = [_to_oklab(color) for color in colors]
        values = np.array([(c.l, c.a, c.b, alpha) for c, alpha in parsed])
    else:
        values = np.array([_to_rgba(color) for color in colors])

    x = np.linspace(points[0], points[-1], num=ndiv)
    if len(points) == 1:
        return np.repeat(values, ndiv, axis=0)

    interpolated = scipy.interpolate.interp1d(points, values, axis=0, kind="linear")(x)

    if colorspace == "oklab":
        rgb = np.array([oklab_to_rgb(*row[:3]) for row in interpolated])
        interpolated = np.column_stack([np.clip(rgb, 0.0, 1.0), interpolated[:, 3]])
    return interpolated


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


class OkLAB(typing.NamedTuple):
    l: float
    a: float
    b: float


def _srgb_to_linear(c: float) -> float:
    """
    Decode an sRGB channel value to linear light.
    """
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(c: float) -> float:
    """
    Encode a linear-light channel value as sRGB.
    """
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * c ** (1 / 2.4) - 0.055


def rgb_to_oklab(r: float, g: float, b: float) -> OkLAB:
    """
    Convert an sRGB triple to OkLab.
    """
    # https://bottosson.github.io/posts/oklab/ (matrices are for linear sRGB).
    r = _srgb_to_linear(r)
    g = _srgb_to_linear(g)
    b = _srgb_to_linear(b)
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l, m, s = l ** (1 / 3), m ** (1 / 3), s ** (1 / 3)
    return OkLAB(
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    )


def oklab_to_rgb(L: float, a: float, b: float) -> tuple[float, float, float]:
    """
    Convert an OkLab triple to sRGB.
    """
    l = L + 0.3963377774 * a + 0.2158037573 * b
    m = L - 0.1055613458 * a - 0.0638541728 * b
    s = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l**3, m**3, s**3
    return (
        _linear_to_srgb(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s),
        _linear_to_srgb(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
        _linear_to_srgb(-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s),
    )


def _to_oklab(color: Any) -> tuple[OkLAB, float]:
    """
    Convert a color specification to OkLab and alpha.
    """
    if isinstance(color, OkLAB):
        return color, 1.0
    r, g, b, alpha = matplotlib.colors.to_rgba(color)
    return rgb_to_oklab(r, g, b), alpha


def _to_rgba(color: Any) -> tuple[float, float, float, float]:
    """
    Convert a color specification to RGBA, accepting OkLab inputs.
    """
    if isinstance(color, OkLAB):
        return (*oklab_to_rgb(*color), 1.0)
    return matplotlib.colors.to_rgba(color)


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

# White to light gray to salmon colormap, interpolated in OkLab.
cm_wjr = matplotlib.colors.LinearSegmentedColormap.from_list(
    "wjr",
    lerp_palette(
        [(1.0, 1.0, 1.0), (0.9, 0.9, 0.9), (1.0, 0.63, 0.58)],
        ndiv=256,
        colorspace="oklab",
    ),
)


def register_colormaps() -> None:
    """
    Register `cm_wrr`, `cm_bkr` and `cm_wjr` with matplotlib under their names.
    """
    register_cmap(cm_wrr)
    register_cmap(cm_bkr)
    register_cmap(cm_wjr)


def _oklab_distance(x: OkLAB, y: OkLAB) -> float:
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
    reference_lab = rgb_to_oklab(*matplotlib.colors.to_rgb(reference_color))
    evaluations = [
        (
            color,
            _oklab_distance(
                rgb_to_oklab(*matplotlib.colors.to_rgb(color)), reference_lab
            ),
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
