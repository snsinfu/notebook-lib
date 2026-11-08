"""
Tree layout data and traversal for plot drawing.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator
from typing import Literal, NamedTuple


class TreeLayoutNode[Node](NamedTuple):
    """
    One node of a computed tree layout.

    `x` is the horizontal center of the subtree, `y` is the vertical center,
    `depth` is the root-relative depth (root is 0), and `width` is the full
    horizontal span of the subtree (the leaf width for a leaf). `children`
    preserves the input order. `depth` is kept for callers that need discrete
    levels; `y` already encodes the level position. Use `iter_nodes` for a
    stateless pre-order traversal.
    """

    node: Node
    x: float
    y: float
    depth: int
    width: float
    children: list[TreeLayoutNode[Node]]

    def iter_nodes(self) -> Iterator[TreeLayoutNode[Node]]:
        """
        Yield this node and its descendants in pre-order.

        Yields layout nodes, not the wrapped `node` values. Children keep the
        stored order. This is the stateless counterpart to `walk`.
        """
        yield self
        for child in self.children:
            yield from child.iter_nodes()


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
    branch_height: float = 1.0,
    direction: Literal["down", "up"] = "down",
    x: float = 0.0,
    y: float = 0.0,
) -> TreeLayoutNode[Node]:
    """
    Compute a tree layout without overlapping subtrees.

    `x` is the subtree center and `width` its full horizontal span.
    `branch_height` is the vertical distance between a node and its children:
    `direction="down"` decreases `y` with depth and `"up"` increases it. This
    is a data direction, so inverting the y axis flips the drawn tree. The
    root is at `(x, y)` (default the origin), and `depth` is root-relative.
    `leaf_width` is a scalar or a callable evaluated for leaves only, and
    `branch_margin` is a scalar or a callable evaluated for internal nodes
    only.
    """
    leaf_width_of = _resolve_scalar(leaf_width)
    margin_of = _resolve_scalar(branch_margin)
    height = _check_nonnegative(branch_height, "branch_height")
    match direction:
        case "down":
            step = -height
        case "up":
            step = height
        case _:
            raise ValueError(f"direction must be 'down' or 'up', not {direction!r}")
    origin_x = _check_finite(x, "x")
    origin_y = _check_finite(y, "y")

    def build(node: Node) -> _BuildNode[Node]:
        child_nodes = list(children(node))
        if not child_nodes:
            width = _check_nonnegative(leaf_width_of(node), "leaf_width")
            return _BuildNode(node, width, 0.0, [])

        margin = _check_nonnegative(margin_of(node), "branch_margin")
        built = [build(child) for child in child_nodes]
        width = sum(child.width for child in built) + margin * (len(built) - 1)
        return _BuildNode(node, width, margin, built)

    def place(
        node: _BuildNode[Node], center_x: float, center_y: float, depth: int
    ) -> TreeLayoutNode[Node]:
        cursor = center_x - node.width / 2
        child_y = center_y + step
        placed: list[TreeLayoutNode[Node]] = []
        for child in node.children:
            placed.append(place(child, cursor + child.width / 2, child_y, depth + 1))
            cursor += child.width + node.margin
        return TreeLayoutNode(node.node, center_x, center_y, depth, node.width, placed)

    return place(build(root), origin_x, origin_y, 0)


def _resolve_scalar[Node](
    value: float | Callable[[Node], float],
) -> Callable[[Node], float]:
    if callable(value):
        return value
    constant = float(value)
    return lambda _node: constant


def _check_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _check_nonnegative(value: float, name: str) -> float:
    value = _check_finite(value, name)
    if value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value
