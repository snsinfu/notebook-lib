"""
Patch and colorbar drawing helpers.
"""

from __future__ import annotations

from typing import Any

import matplotlib.axes
import matplotlib.cm
import matplotlib.colorbar
import matplotlib.patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_rect(
    ax: matplotlib.axes.Axes,
    lx: float,
    ly: float,
    tx: float,
    ty: float,
    **kwargs: Any,
) -> matplotlib.patches.Rectangle:
    """
    Add an axis-aligned rectangle spanning `(lx, ly)` to `(tx, ty)`.
    """
    rectangle = matplotlib.patches.Rectangle((lx, ly), tx - lx, ty - ly, **kwargs)
    ax.add_patch(rectangle)
    return rectangle


def plot_arc(
    ax: matplotlib.axes.Axes,
    cx: float,
    cy: float,
    width: float,
    height: float,
    theta1: float,
    theta2: float,
    angle: float = 0,
    **kwargs: Any,
) -> matplotlib.patches.Arc:
    """
    Plot an arc of an ellipse centered at `(cx, cy)`.

    `theta1`/`theta2` are start/end angles in degrees.
    """
    arc = matplotlib.patches.Arc(
        (cx, cy),
        width,
        height,
        theta1=theta1,
        theta2=theta2,
        angle=angle,
        **kwargs,
    )
    ax.add_patch(arc)
    return arc


def attach_colorbar(
    ax: matplotlib.axes.Axes,
    sm: matplotlib.cm.ScalarMappable,
    size: float = 0.03,
    length: float = 0.5,
    margin: float | None = None,
    orientation: str = "vertical",
    extend: str | None = None,
    outline: bool = True,
) -> matplotlib.colorbar.Colorbar:
    """
    Attach a colorbar to a side of an axes.

    `size` is the colorbar thickness and `length` its long dimension (both as
    axes fractions); `margin` is the gap to the axes (defaults to `size`).
    `orientation` is `"vertical"` (right) or `"horizontal"` (bottom).
    """
    if margin is None:
        margin = size

    match orientation:
        case "vertical":
            bbox = (1 + margin, 0, size, length)  # right
        case "horizontal":
            bbox = (0.5 - length / 2, -(size + margin), length, size)  # bottom
        case _:
            raise ValueError(
                "orientation must be 'vertical' or 'horizontal',"
                + f" not '{orientation}'"
            )

    cbar_ax = inset_axes(
        ax,
        width="100%",
        height="100%",
        bbox_to_anchor=bbox,
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    fig = ax.get_figure()
    if fig is None:
        raise RuntimeError("axes is not attached to a figure")
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation=orientation, extend=extend)

    if not outline:
        cbar.outline.set_edgecolor("none")

    return cbar
