"""
Axes, tick, and grid-line helpers.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal, cast

import matplotlib.axes
import matplotlib.figure
import matplotlib.lines
import matplotlib.ticker
import numpy as np


def update_plot(obj: matplotlib.axes.Axes | matplotlib.figure.Figure) -> None:
    """
    Redraw `obj` (an Axes or Figure) so its artists are up to date.
    """
    figure = obj.get_figure() if isinstance(obj, matplotlib.axes.Axes) else obj
    if figure is not None:
        figure.canvas.draw()


def get_x_grid_line_at(
    ax: matplotlib.axes.Axes, x: float
) -> matplotlib.lines.Line2D | None:
    """
    Return the vertical grid line at coordinate `x`, or None.
    """
    for pos, line in _get_x_grid_lines(ax):
        if np.isclose(pos, x):
            return line
    return None


def _get_x_grid_lines(
    ax: matplotlib.axes.Axes,
) -> Iterator[tuple[float, matplotlib.lines.Line2D]]:
    """
    Yield `(x, line)` for each vertical grid line of `ax`.
    """
    update_plot(ax)
    for line in ax.get_xgridlines():
        xdata = np.asarray(line.get_xdata(), dtype=float)
        if xdata.size and np.allclose(xdata, xdata[0]):
            yield float(xdata[0]), line


def get_y_grid_line_at(
    ax: matplotlib.axes.Axes, y: float
) -> matplotlib.lines.Line2D | None:
    """
    Return the horizontal grid line at coordinate `y`, or None.
    """
    for pos, line in _get_y_grid_lines(ax):
        if np.isclose(pos, y):
            return line
    return None


def _get_y_grid_lines(
    ax: matplotlib.axes.Axes,
) -> Iterator[tuple[float, matplotlib.lines.Line2D]]:
    """
    Yield `(y, line)` for each horizontal grid line of `ax`.
    """
    update_plot(ax)
    for line in ax.get_ygridlines():
        ydata = np.asarray(line.get_ydata(), dtype=float)
        if ydata.size and np.allclose(ydata, ydata[0]):
            yield float(ydata[0]), line


def set_axis(
    ax: matplotlib.axes.Axes,
    spines: str = "none",
    ticks: str | None = None,
    direction: str | None = None,
) -> None:
    """
    Configure spines, ticks, and labels of `ax` in one call.

    `spines` and `ticks` are space-separated side names among
    `top right bottom left` or the special values `all` / `none`. `ticks`
    defaults to `spines`. Tick labels move to the shown side; the orthogonal
    axis is inverted so its ticks still increase away from the labeled side.
    """
    all_sides = ["top", "right", "bottom", "left"]
    valid_keys = {"all", "none", *all_sides}

    def map_side_key(key: str) -> list[str]:
        if key not in valid_keys:
            raise ValueError(
                f"unknown side {key!r}; expected one of {sorted(valid_keys)}"
            )
        if key == "all":
            return all_sides
        if key == "none":
            return []
        return [key]

    def make_side_set(sides: str) -> set[str]:
        tokens = sides.split()
        return {side for token in tokens for side in map_side_key(token)}

    if ticks is None:
        ticks = spines

    tick_options: dict[str, Any] = {}
    if direction is not None:
        tick_options["direction"] = direction

    # Hide everything first.
    for side in all_sides:
        hide_params: dict[str, Any] = {side: False, f"label{side}": False}
        ax.spines[side].set_visible(False)
        ax.tick_params(which="both", **hide_params)

    # Enable specified sides.
    spines_set = make_side_set(spines)
    for side in spines_set:
        ax.spines[side].set_visible(True)

    ticks_set = make_side_set(ticks)
    for side in ticks_set:
        show_params: dict[str, Any] = {side: True, f"label{side}": True}
        ax.tick_params(which="both", **show_params, **tick_options)

    # Move labels when ticks are shown only on one side.
    def specialize(spec: set[str], special: str, default: str) -> str:
        if special in spec and default not in spec:
            return special
        return default

    x_label_pos = specialize(ticks_set, "top", "bottom")
    ax.xaxis.set_label_position(cast("Literal['top', 'bottom']", x_label_pos))
    y_label_pos = specialize(ticks_set, "right", "left")
    ax.yaxis.set_label_position(cast("Literal['left', 'right']", y_label_pos))

    # Ticks should conventionally increase from the labeled side. Set the
    # inversion explicitly so repeated calls are idempotent.
    ax.yaxis.set_inverted(x_label_pos == "top")
    ax.xaxis.set_inverted(y_label_pos == "right")


def make_scaled_formatter(
    scale: float,
    places: int = 1,
    postfix: str = "",
) -> matplotlib.ticker.Formatter:
    """
    Return a tick formatter that multiplies values by `scale`.
    """

    @matplotlib.ticker.FuncFormatter
    def formatter(x: float, n: int) -> str:
        return "{0:.{1}f}{2}".format(x * scale, places, postfix)

    return formatter
