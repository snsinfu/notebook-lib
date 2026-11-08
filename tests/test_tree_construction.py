import pytest

from lib.tree import build_forest


class Node:
    def __init__(self, children, value):
        self.children = children
        self.value = value


def make_node(children, value):
    return Node(children, value)


def test_build_forest_structure():
    records = [
        ("root", ("a", "b"), "root"),
        ("a", (), "a"),
        ("b", (), "b"),
    ]
    trees = build_forest(records, make_node=make_node)
    assert len(trees) == 1
    root = trees[0]
    assert root.value == "root"
    assert isinstance(root.children, list)
    assert [child.value for child in root.children] == ["a", "b"]
    assert [child.children for child in root.children] == [[], []]


def test_build_forest_preserves_child_order_regardless_of_record_order():
    records = [
        ("a", (), "a"),
        ("b", (), "b"),
        ("root", ("b", "a"), "root"),
    ]
    [root] = build_forest(records, make_node=make_node)
    assert [child.value for child in root.children] == ["b", "a"]


def test_build_forest_single_leaf():
    [root] = build_forest([("only", (), 1)], make_node=make_node)
    assert root.value == 1
    assert root.children == []


def test_build_forest_empty_records():
    assert build_forest([], make_node=make_node) == []


def test_build_forest_multiple_roots():
    records = [
        ("left_root", ("left_child",), "left_root"),
        ("left_child", (), "left_child"),
        ("right_root", ("right_child",), "right_root"),
        ("right_child", (), "right_child"),
    ]
    trees = build_forest(records, make_node=make_node)
    assert [tree.value for tree in trees] == ["left_root", "right_root"]
    assert [child.value for child in trees[0].children] == ["left_child"]
    assert [child.value for child in trees[1].children] == ["right_child"]


def test_build_forest_preserves_root_record_order():
    records = [("second", (), 2), ("first", (), 1)]
    trees = build_forest(records, make_node=make_node)
    assert [tree.value for tree in trees] == [2, 1]


def test_build_forest_builds_children_before_parent():
    calls = []

    def record_node(children, value):
        calls.append(value)
        return Node(children, value)

    records = [
        ("root", ("a",), "root"),
        ("a", ("b",), "a"),
        ("b", (), "b"),
    ]
    build_forest(records, make_node=record_node)
    assert calls == ["b", "a", "root"]


def test_build_forest_value_can_carry_node_id():
    value = {"id": 10, "name": "root"}
    [root] = build_forest([(10, (), value)], make_node=make_node)
    assert root.value is value
    assert root.value["id"] == 10


def test_build_forest_accepts_generator_records():
    records = (
        (node_id, children, node_id)
        for node_id, children in [("root", ("a",)), ("a", ())]
    )
    [root] = build_forest(records, make_node=make_node)
    assert root.value == "root"
    assert [child.value for child in root.children] == ["a"]


def test_build_forest_accepts_generator_child_ids():
    records = [
        ("root", (child for child in ("a", "b")), "root"),
        ("a", (), "a"),
        ("b", (), "b"),
    ]
    [root] = build_forest(records, make_node=make_node)
    assert [child.value for child in root.children] == ["a", "b"]


def test_build_forest_rejects_duplicate_node_id():
    records = [("a", (), 1), ("a", (), 2)]
    with pytest.raises(ValueError, match="duplicate node id"):
        build_forest(records, make_node=make_node)


def test_build_forest_rejects_unknown_child():
    records = [("root", ("missing",), 0)]
    with pytest.raises(ValueError, match="unknown child"):
        build_forest(records, make_node=make_node)


def test_build_forest_rejects_self_child():
    records = [("a", ("a",), 0)]
    with pytest.raises(ValueError, match="own child"):
        build_forest(records, make_node=make_node)


def test_build_forest_rejects_duplicate_child():
    records = [("root", ("a", "a"), 0), ("a", (), 1)]
    with pytest.raises(ValueError, match="more than once"):
        build_forest(records, make_node=make_node)


def test_build_forest_rejects_multiple_parents():
    records = [
        ("root", ("a", "b"), 0),
        ("a", ("c",), 1),
        ("b", ("c",), 2),
        ("c", (), 3),
    ]
    with pytest.raises(ValueError, match="multiple parents"):
        build_forest(records, make_node=make_node)


def test_build_forest_rejects_pure_cycle():
    records = [("a", ("b",), 1), ("b", ("a",), 2)]
    with pytest.raises(ValueError, match="cycle"):
        build_forest(records, make_node=make_node)


def test_build_forest_rejects_disconnected_cycle():
    records = [("root", (), 0), ("a", ("b",), 1), ("b", ("a",), 2)]
    with pytest.raises(ValueError, match="cycle"):
        build_forest(records, make_node=make_node)
