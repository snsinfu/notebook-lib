"""
Matplotlib helpers: axes, ticks, colormaps, patches, matrix helpers, KDE plots.

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
    set_table_axis,
    update_plot,
)
from .colormaps import cm_bkr, cm_wjr, cm_wrr, register_colormaps
from .colors import (
    OkLAB,
    lerp_palette,
    oklab_to_rgb,
    register_cmap,
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
from .matrix import (
    make_triu_mesh,
    matrix_cell_coords,
    smooth_matrix,
    transform_triu_coords,
)
from .patches import attach_colorbar, plot_arc, plot_rect
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
    "make_triu_mesh",
    "matrix_cell_coords",
    "oklab_to_rgb",
    "plot_arc",
    "plot_rect",
    "register_cmap",
    "register_colormaps",
    "register_cycle_palette",
    "rgb_to_oklab",
    "set_axis",
    "set_table_axis",
    "slice_cmap",
    "smooth_matrix",
    "texsci",
    "transform_triu_coords",
    "update_plot",
    "use_log_ticks",
]
