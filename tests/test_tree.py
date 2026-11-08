import itertools
import math

import pytest

from lib.plot import TreeLayoutNode, compute_tree_layout


class Node:
    def __init__(self, name, width=1.0, children=()):
        self.name = name
        self.width = width
        self.children = list(children)


def layout_tree(
    root,
    margin=0.0,
    leaf_width=lambda node: node.width,
    branch_height=1.0,
    direction="down",
    x=0.0,
    y=0.0,
):
    return compute_tree_layout(
        root,
        children=lambda node: node.children,
        leaf_width=leaf_width,
        branch_margin=margin,
        branch_height=branch_height,
        direction=direction,
        x=x,
        y=y,
    )


def leaf_intervals(node):
    if not node.children:
        return [(node.x - node.width / 2, node.x + node.width / 2)]
    intervals = []
    for child in node.children:
        intervals.extend(leaf_intervals(child))
    return intervals


def random_tree(rng, depth):
    if depth == 0 or rng.random() < 0.35:
        return Node("leaf", width=float(rng.uniform(0.1, 3.0)))
    n_children = int(rng.integers(1, 4))
    return Node(
        "node", children=[random_tree(rng, depth - 1) for _ in range(n_children)]
    )


def test_single_leaf():
    leaf = Node("a", width=2.0)
    result = layout_tree(leaf)
    assert result.node is leaf
    assert result.x == 0.0
    assert result.y == 0.0
    assert result.depth == 0
    assert result.width == 2.0
    assert result.children == []


def test_two_leaves_with_margin():
    root = Node("root", children=[Node("a", width=2.0), Node("b", width=4.0)])
    result = layout_tree(root, margin=1.0)
    assert result.width == 7.0
    assert result.x == 0.0
    assert [child.x for child in result.children] == [-2.5, 1.5]
    assert [child.y for child in result.children] == [-1.0, -1.0]
    assert [child.depth for child in result.children] == [1, 1]


def test_parents_centered_over_children():
    left = Node("left", children=[Node("a", width=1.0), Node("b", width=3.0)])
    right = Node("right", children=[Node("c", width=2.0)])
    root = Node("root", children=[left, right])
    result = layout_tree(root, margin=0.5)
    left_layout, right_layout = result.children
    assert result.width == 7.0
    assert left_layout.width == 4.5
    assert left_layout.x == -1.25
    assert right_layout.x == 2.5
    assert [child.x for child in left_layout.children] == [-3.0, -0.5]


def test_chain_preserves_x_and_increments_depth():
    grandchild = Node("c", width=2.0)
    child = Node("b", children=[grandchild])
    root = Node("a", children=[child])
    result = layout_tree(root)
    assert result.x == 0.0
    assert result.y == 0.0
    assert result.width == 2.0
    assert result.depth == 0
    assert result.children[0].x == 0.0
    assert result.children[0].y == -1.0
    assert result.children[0].depth == 1
    assert result.children[0].children[0].x == 0.0
    assert result.children[0].children[0].y == -2.0
    assert result.children[0].children[0].depth == 2


def test_branch_height_scales_y():
    grandchild = Node("c")
    child = Node("b", children=[grandchild])
    root = Node("a", children=[child])
    result = layout_tree(root, branch_height=2.5)
    assert [result.y, result.children[0].y, result.children[0].children[0].y] == [
        0.0,
        -2.5,
        -5.0,
    ]


def test_direction_up_grows_y_positive():
    grandchild = Node("c")
    child = Node("b", children=[grandchild])
    root = Node("a", children=[child])
    result = layout_tree(root, branch_height=2.5, direction="up")
    assert [result.y, result.children[0].y, result.children[0].children[0].y] == [
        0.0,
        2.5,
        5.0,
    ]


@pytest.mark.parametrize("direction", ["down", "up"])
def test_branch_height_zero_flattens_levels(direction):
    grandchild = Node("c")
    child = Node("b", children=[grandchild])
    root = Node("a", children=[child])
    result = layout_tree(root, branch_height=0.0, direction=direction)
    assert [result.y, result.children[0].y, result.children[0].children[0].y] == [
        0.0,
        0.0,
        0.0,
    ]


def test_root_origin_offsets_layout():
    root = Node("root", children=[Node("a", width=2.0), Node("b", width=4.0)])
    result = layout_tree(root, margin=1.0, x=10.0, y=5.0, branch_height=2.0)
    assert result.x == 10.0
    assert result.y == 5.0
    assert result.width == 7.0
    assert [child.x for child in result.children] == [7.5, 11.5]
    assert [child.y for child in result.children] == [3.0, 3.0]
    assert [child.depth for child in result.children] == [1, 1]


def test_root_origin_accepts_negative_values():
    grandchild = Node("c")
    child = Node("b", children=[grandchild])
    root = Node("a", children=[child])
    result = layout_tree(root, x=-3.0, y=-4.0, branch_height=2.0, direction="up")
    assert result.x == -3.0
    assert result.y == -4.0
    assert result.children[0].x == -3.0
    assert result.children[0].y == -2.0
    assert result.children[0].children[0].y == 0.0


def test_leaf_intervals_do_not_overlap(rng):
    for _ in range(25):
        root = random_tree(rng, 4)
        result = layout_tree(root, margin=0.5)
        intervals = sorted(leaf_intervals(result))
        for (_, right), (left, _) in itertools.pairwise(intervals):
            assert right <= left + 1e-12


def test_branch_margin_callable():
    left = Node("left", children=[Node("a"), Node("b")])
    right = Node("c")
    root = Node("root", children=[left, right])
    result = compute_tree_layout(
        root,
        children=lambda node: node.children,
        leaf_width=lambda node: node.width,
        branch_margin=lambda node: 2.0 if node.name == "root" else 0.5,
    )
    left_layout, right_layout = result.children
    assert result.width == 5.5
    assert left_layout.width == 2.5
    assert left_layout.x == -1.5
    assert [child.x for child in left_layout.children] == [-2.25, -0.75]
    assert right_layout.x == 2.25


def test_scalar_leaf_width():
    root = Node("root", children=[Node("a", width=10.0), Node("b", width=20.0)])
    result = layout_tree(root, margin=1.0, leaf_width=2.0)
    assert result.width == 5.0
    assert [child.width for child in result.children] == [2.0, 2.0]
    assert [child.x for child in result.children] == [-1.5, 1.5]


def test_default_leaf_width():
    root = Node("root", children=[Node("a", width=99.0), Node("b", width=99.0)])
    result = compute_tree_layout(root, children=lambda node: node.children)
    assert result.width == 2.0
    assert [child.width for child in result.children] == [1.0, 1.0]


def test_repeated_node_object_gets_two_positions():
    leaf = Node("dup", width=1.0)
    root = Node("root", children=[leaf, leaf])
    result = layout_tree(root, margin=1.0)
    assert result.width == 3.0
    assert len(result.children) == 2
    assert all(child.node is leaf for child in result.children)
    assert [child.x for child in result.children] == [-1.0, 1.0]


def test_leaf_width_not_called_for_internal_nodes():
    def leaf_width(node):
        assert not node.children
        return 1.0

    root = Node("root", children=[Node("a"), Node("b")])
    layout_tree(root, leaf_width=leaf_width)


def test_branch_margin_not_called_for_leaves():
    def branch_margin(node):
        assert node.children
        return 0.5

    root = Node("root", children=[Node("a"), Node("b")])
    layout_tree(root, margin=branch_margin)


def test_invalid_direction():
    with pytest.raises(ValueError, match="direction"):
        layout_tree(Node("leaf"), direction="sideways")


@pytest.mark.parametrize("height", [-1.0, math.nan, math.inf])
def test_invalid_branch_height(height):
    with pytest.raises(ValueError, match="branch_height"):
        layout_tree(Node("leaf"), branch_height=height)


@pytest.mark.parametrize("name", ["x", "y"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_invalid_root_origin(name, value):
    with pytest.raises(ValueError, match=f"^{name} must be finite"):
        layout_tree(Node("leaf"), **{name: value})


@pytest.mark.parametrize("width", [-1.0, math.nan, math.inf])
def test_invalid_scalar_leaf_width(width):
    with pytest.raises(ValueError, match="leaf_width"):
        layout_tree(Node("leaf"), leaf_width=width)


@pytest.mark.parametrize("width", [-1.0, math.nan, math.inf])
def test_invalid_callable_leaf_width(width):
    with pytest.raises(ValueError, match="leaf_width"):
        layout_tree(Node("leaf"), leaf_width=lambda node: width)


@pytest.mark.parametrize("margin", [-1.0, math.nan, math.inf])
def test_invalid_scalar_branch_margin(margin):
    root = Node("root", children=[Node("a"), Node("b")])
    with pytest.raises(ValueError, match="branch_margin"):
        layout_tree(root, margin=margin)


@pytest.mark.parametrize("margin", [-1.0, math.nan, math.inf])
def test_invalid_callable_branch_margin(margin):
    root = Node("root", children=[Node("a"), Node("b")])
    with pytest.raises(ValueError, match="branch_margin"):
        layout_tree(root, margin=lambda node: margin)


def test_iter_nodes_preorder():
    root = Node(
        "root",
        children=[
            Node("a", children=[Node("a1"), Node("a2")]),
            Node("b"),
        ],
    )
    result = layout_tree(root)
    assert [node.node.name for node in result.iter_nodes()] == [
        "root",
        "a",
        "a1",
        "a2",
        "b",
    ]


def test_iter_nodes_single_leaf():
    leaf = Node("leaf")
    result = layout_tree(leaf)
    assert list(result.iter_nodes()) == [result]


def test_iter_nodes_yields_layout_nodes():
    root = Node("root", children=[Node("a"), Node("b")])
    result = layout_tree(root)
    nodes = list(result.iter_nodes())
    assert all(isinstance(node, TreeLayoutNode) for node in nodes)
    assert [node.node for node in nodes] == [root, root.children[0], root.children[1]]


def test_iter_nodes_subtree():
    root = Node("root", children=[Node("a", children=[Node("a1")]), Node("b")])
    result = layout_tree(root)
    assert [node.node.name for node in result.children[0].iter_nodes()] == [
        "a",
        "a1",
    ]


def test_iter_nodes_repeated_node():
    leaf = Node("dup")
    root = Node("root", children=[leaf, leaf])
    result = layout_tree(root)
    nodes = list(result.iter_nodes())
    assert len(nodes) == 3
    assert nodes[1] is not nodes[2]
    assert nodes[1].node is leaf
    assert nodes[2].node is leaf
    assert nodes[1].x != nodes[2].x
