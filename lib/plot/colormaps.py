"""
Bundled colormaps and their matplotlib registration.
"""

from __future__ import annotations

import matplotlib.colors

from .colors import lerp_palette, register_cmap

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
