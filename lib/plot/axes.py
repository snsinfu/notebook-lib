"""
Axes, tick, and grid-line helpers.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
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
    labels: str | None = None,
) -> None:
    """
    Configure spines, ticks, and labels of `ax` in one call.

    `spines` and `ticks` are space-separated side names among
    `top right bottom left` or the special values `all` / `none`. `ticks`
    defaults to `spines`. `labels` selects where tick labels are shown and
    defaults to `ticks`; pass `ticks="none", labels="top left"` to show
    tickless labels. Labels move to the shown side; the orthogonal axis is
    inverted so its ticks still increase away from the labeled side. When
    `labels` is empty, the tick sides determine label position and inversion.
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
    if labels is None:
        labels = ticks

    tick_options: dict[str, Any] = {}
    if direction is not None:
        tick_options["direction"] = direction

    # Hide everything first.
    for side in all_sides:
        hide_params: dict[str, Any] = {side: False, f"label{side}": False}
        ax.spines[side].set_visible(False)
        ax.tick_params(which="both", **hide_params)

    # Enable specified spines.
    spines_set = make_side_set(spines)
    for side in spines_set:
        ax.spines[side].set_visible(True)

    # Enable tick marks and labels independently.
    ticks_set = make_side_set(ticks)
    labels_set = make_side_set(labels)
    for side in all_sides:
        show_params: dict[str, Any] = {
            side: side in ticks_set,
            f"label{side}": side in labels_set,
        }
        ax.tick_params(which="both", **show_params, **tick_options)

    # Move labels when they are shown only on one side. If no labels are
    # requested, keep the tick-side convention so inversion stays predictable.
    effective_set = labels_set if labels_set else ticks_set

    def specialize(spec: set[str], special: str, default: str) -> str:
        if special in spec and default not in spec:
            return special
        return default

    x_label_pos = specialize(effective_set, "top", "bottom")
    ax.xaxis.set_label_position(cast("Literal['top', 'bottom']", x_label_pos))
    y_label_pos = specialize(effective_set, "right", "left")
    ax.yaxis.set_label_position(cast("Literal['left', 'right']", y_label_pos))

    # Ticks should conventionally increase from the labeled side. Set the
    # inversion explicitly so repeated calls are idempotent.
    ax.yaxis.set_inverted(x_label_pos == "top")
    ax.xaxis.set_inverted(y_label_pos == "right")


def set_table_axis(
    ax: matplotlib.axes.Axes,
    rows: Sequence[str],
    cols: Sequence[str],
    *,
    labels: str = "top left",
    spines: str = "none",
) -> None:
    """
    Configure `ax` for a heat-map table of cells.

    `rows` and `cols` are the row and column labels; their lengths define the
    table shape. Major ticks sit at cell centers, minor ticks at the internal
    cell boundaries, and labels are shown without tick marks. The helper does
    not draw the grid; the caller decides whether to add it, for example
    `ax.grid(True, which="minor", lw=0.5, ls="-", color="w")`.

    The default top/left labels assume row 0 is at the top (`Axes.matshow` or
    `origin="upper"`). Use `labels="bottom left"` with `origin="lower"`. The
    helper does not set axis limits.
    """
    row_labels = list(rows)
    col_labels = list(cols)
    if not row_labels or not col_labels:
        raise ValueError("rows and cols must be non-empty")
    nrows = len(row_labels)
    ncols = len(col_labels)

    set_axis(ax, spines=spines, ticks="none", labels=labels)

    ax.set_xticks(np.arange(ncols))
    ax.set_yticks(np.arange(nrows))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.set_xticks(np.arange(ncols - 1) + 0.5, minor=True)
    ax.set_yticks(np.arange(nrows - 1) + 0.5, minor=True)

    ax.tick_params(which="major", pad=0)
    ax.tick_params(
        which="minor",
        length=0,
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False,
    )


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
