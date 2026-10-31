"""
Tree layout data for plot drawing.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import NamedTuple


class TreeLayoutNode[Node](NamedTuple):
    """
    One node of a computed tree layout.

    `x` is the horizontal center of the subtree, `depth` is the root-relative
    depth (root is 0), and `width` is the full horizontal span of the subtree
    (the leaf width for a leaf). `children` preserves the input order. Vertical
    placement is not included; derive it from `depth` in the caller.
    """

    node: Node
    x: float
    depth: int
    width: float
    children: list[TreeLayoutNode[Node]]


class _BuildNode[Node](NamedTuple):
    node: Node
    width: float
    margin: float
    children: list[_BuildNode[Node]]


def compute_tree_layout[Node](
    root: Node,
    *,
    children: Callable[[Node], Iterable[Node]],
    leaf_width: float | Callable[[Node], float] = 1.0,
    branch_margin: float | Callable[[Node], float] = 0.0,
) -> TreeLayoutNode[Node]:
    """
    Compute a horizontal tree layout without overlapping subtrees.

    `x` is the subtree center, `width` its full horizontal span, and `depth`
    the root-relative depth. Vertical placement is left to the caller. The
    root is at `x = 0`. `leaf_width` is a scalar or a callable evaluated for
    leaves only, and `branch_margin` is a scalar or a callable evaluated for
    internal nodes only.
    """
    leaf_width_of = _resolve_scalar(leaf_width)
    margin_of = _resolve_scalar(branch_margin)

    def build(node: Node) -> _BuildNode[Node]:
        child_nodes = list(children(node))
        if not child_nodes:
            width = _check_nonnegative(leaf_width_of(node), "leaf_width")
            return _BuildNode(node, width, 0.0, [])

        margin = _check_nonnegative(margin_of(node), "branch_margin")
        built = [build(child) for child in child_nodes]
        width = sum(child.width for child in built) + margin * (len(built) - 1)
        return _BuildNode(node, width, margin, built)

    def place(node: _BuildNode[Node], x: float, depth: int) -> TreeLayoutNode[Node]:
        cursor = x - node.width / 2
        placed: list[TreeLayoutNode[Node]] = []
        for child in node.children:
            placed.append(place(child, cursor + child.width / 2, depth + 1))
            cursor += child.width + node.margin
        return TreeLayoutNode(node.node, x, depth, node.width, placed)

    return place(build(root), 0.0, 0)


def _resolve_scalar[Node](
    value: float | Callable[[Node], float],
) -> Callable[[Node], float]:
    if callable(value):
        return value
    constant = float(value)
    return lambda _node: constant


def _check_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value
