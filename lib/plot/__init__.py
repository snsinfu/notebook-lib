"""
Matplotlib helpers: axes, ticks, colormaps, patches, smoothing, KDE plots.

Contains functions intended for plotting, including pure-computation helpers
that exist only to feed plots (`kde_at`, `texsci`).
"""

from __future__ import annotations

from ..density.bandwidth import bw_method
from .axes import (
    get_x_grid_line_at,
    get_y_grid_line_at,
    make_scaled_formatter,
    set_axis,
    update_plot,
)
from .colors import (
    OkLAB,
    cm_bkr,
    cm_wjr,
    cm_wrr,
    lerp_palette,
    oklab_to_rgb,
    register_cmap,
    register_colormaps,
    register_cycle_palette,
    rgb_to_oklab,
    slice_cmap,
)
from .density import kde_at, kde_contour
from .logticks import (
    LogTicksFormatter,
    LogTicksLocator,
    use_log_ticks,
)
from .patches import attach_colorbar, plot_arc, plot_rect
from .smoothing import smooth_matrix
from .text import texsci

__all__ = [
    "LogTicksFormatter",
    "LogTicksLocator",
    "OkLAB",
    "attach_colorbar",
    "bw_method",
    "cm_bkr",
    "cm_wjr",
    "cm_wrr",
    "get_x_grid_line_at",
    "get_y_grid_line_at",
    "kde_at",
    "kde_contour",
    "lerp_palette",
    "make_scaled_formatter",
    "oklab_to_rgb",
    "plot_arc",
    "plot_rect",
    "register_cmap",
    "register_colormaps",
    "register_cycle_palette",
    "rgb_to_oklab",
    "set_axis",
    "slice_cmap",
    "smooth_matrix",
    "texsci",
    "update_plot",
    "use_log_ticks",
]
