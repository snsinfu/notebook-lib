"""
Tree and forest data-structure helpers.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable


def build_forest[Id: Hashable, T, Node](
    records: Iterable[tuple[Id, Iterable[Id], T]],
    *,
    make_node: Callable[[list[Node], T], Node],
) -> list[Node]:
    """
    Build a forest bottom-up from `(node_id, child_ids, value)` records.

    `make_node` receives the built children in `child_ids` order and the
    record's value. The node id resolves the topology and is not passed to
    `make_node`; include it in `value` if the constructed node needs it.
    Records may be in any order. Returns one root per tree in record order;
    empty records produce an empty list.
    """
    values: dict[Id, T] = {}
    children_by_id: dict[Id, list[Id]] = {}

    for node_id, child_ids, value in records:
        if node_id in values:
            raise ValueError(f"duplicate node id: {node_id!r}")
        values[node_id] = value
        children_by_id[node_id] = list(child_ids)

    parent_of: dict[Id, Id] = {}
    for node_id, child_ids in children_by_id.items():
        seen_children: set[Id] = set()
        for child_id in child_ids:
            if child_id == node_id:
                raise ValueError(f"node {node_id!r} cannot be its own child")
            if child_id not in values:
                raise ValueError(f"node {node_id!r} has unknown child id {child_id!r}")
            if child_id in seen_children:
                raise ValueError(
                    f"node {node_id!r} lists child {child_id!r} more than once"
                )
            seen_children.add(child_id)
            if child_id in parent_of:
                raise ValueError(f"node {child_id!r} has multiple parents")
            parent_of[child_id] = node_id

    roots = [node_id for node_id in values if node_id not in parent_of]

    built: set[Id] = set()

    def build(node_id: Id) -> Node:
        built.add(node_id)
        children = [build(child_id) for child_id in children_by_id[node_id]]
        return make_node(children, values[node_id])

    trees = [build(root_id) for root_id in roots]
    if len(built) != len(values):
        raise ValueError("records contain a cycle")
    return trees
