"""Topological sorting of capabilities based on ordering constraints."""

from __future__ import annotations

import heapq
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from pydantic_ai.exceptions import UserError

from .abstract import AbstractCapability, CapabilityOrdering, CapabilityPosition

_AddEdge = Callable[[int, int], None]


def sort_capabilities(
    capabilities: Sequence[AbstractCapability[Any]],
) -> list[AbstractCapability[Any]]:
    """Sort capabilities to satisfy ordering constraints.

    Preserves the original order as a tiebreaker when constraints allow.
    Raises ``UserError`` on conflicts (duplicate positions, missing requirements, cycles).
    """
    caps = list(capabilities)
    n = len(caps)
    if n <= 1:
        return caps

    orderings: list[CapabilityOrdering | None] = [_effective_ordering(cap) for cap in caps]
    leaf_types: list[set[type]] = [_collect_leaf_types(cap) for cap in caps]

    _validate_constraints(caps, orderings, leaf_types)

    edges, in_degree = _build_dag(caps, orderings, leaf_types)

    return _topo_sort(caps, edges, in_degree)


def _validate_constraints(
    caps: list[AbstractCapability[Any]],
    orderings: list[CapabilityOrdering | None],
    leaf_types: list[set[type]],
) -> None:
    """Validate position uniqueness and required dependencies."""
    outermost = [i for i, o in enumerate(orderings) if o and o.position == 'outermost']
    innermost = [i for i, o in enumerate(orderings) if o and o.position == 'innermost']

    if len(outermost) > 1:
        names = [type(caps[i]).__name__ for i in outermost]
        raise UserError(f"Multiple capabilities declare position 'outermost': {', '.join(names)}")
    if len(innermost) > 1:
        names = [type(caps[i]).__name__ for i in innermost]
        raise UserError(f"Multiple capabilities declare position 'innermost': {', '.join(names)}")

    all_leaf_types: set[type] = set[type]().union(*leaf_types)
    for i, ordering in enumerate(orderings):
        if ordering and ordering.requires:
            for req_type in ordering.requires:
                if not any(issubclass(t, req_type) for t in all_leaf_types):
                    raise UserError(
                        f'`{type(caps[i]).__name__}` requires `{req_type.__name__}` '
                        f'but it was not found among the capabilities.'
                    )


def _build_dag(
    caps: list[AbstractCapability[Any]],
    orderings: list[CapabilityOrdering | None],
    leaf_types: list[set[type]],
) -> tuple[dict[int, set[int]], dict[int, int]]:
    """Build a DAG from position and before/after constraints."""
    n = len(caps)
    edges: dict[int, set[int]] = {i: set() for i in range(n)}
    in_degree: dict[int, int] = {i: 0 for i in range(n)}

    def add_edge(before: int, after: int) -> None:
        if after not in edges[before]:
            edges[before].add(after)
            in_degree[after] += 1

    _add_position_edges(n, orderings, add_edge)
    _add_relative_edges(n, orderings, leaf_types, add_edge)

    return edges, in_degree


def _add_position_edges(
    n: int,
    orderings: list[CapabilityOrdering | None],
    add_edge: _AddEdge,
) -> None:
    outermost = [i for i, o in enumerate(orderings) if o and o.position == 'outermost']
    innermost = [i for i, o in enumerate(orderings) if o and o.position == 'innermost']

    if outermost:
        oi = outermost[0]
        for j in range(n):
            if j != oi:
                add_edge(oi, j)
    if innermost:
        ii = innermost[0]
        for j in range(n):
            if j != ii:
                add_edge(j, ii)


def _add_relative_edges(
    n: int,
    orderings: list[CapabilityOrdering | None],
    leaf_types: list[set[type]],
    add_edge: _AddEdge,
) -> None:
    for i, ordering in enumerate(orderings):
        if not ordering:
            continue
        for before_type in ordering.before:
            for j in range(n):
                if i != j and any(issubclass(t, before_type) for t in leaf_types[j]):
                    add_edge(i, j)
        for after_type in ordering.after:
            for j in range(n):
                if i != j and any(issubclass(t, after_type) for t in leaf_types[j]):
                    add_edge(j, i)


def _topo_sort(
    caps: list[AbstractCapability[Any]],
    edges: dict[int, set[int]],
    in_degree: dict[int, int],
) -> list[AbstractCapability[Any]]:
    """Kahn's algorithm with original-index tiebreaking for stability."""
    n = len(caps)
    queue: list[int] = []
    for i in range(n):
        if in_degree[i] == 0:
            heapq.heappush(queue, i)

    result: list[AbstractCapability[Any]] = []
    while queue:
        i = heapq.heappop(queue)
        result.append(caps[i])
        for j in edges[i]:
            in_degree[j] -= 1
            if in_degree[j] == 0:
                heapq.heappush(queue, j)

    if len(result) != n:
        remaining = [type(caps[i]).__name__ for i in range(n) if in_degree[i] > 0]
        raise UserError(f'Circular ordering constraints among capabilities: {", ".join(remaining)}')

    return result


def _effective_ordering(cap: AbstractCapability[Any]) -> CapabilityOrdering | None:
    """Get the effective ordering for a capability, merging from leaves for nested groups."""
    from .combined import CombinedCapability

    if not isinstance(cap, CombinedCapability):
        return type(cap).get_ordering()

    merged_position: CapabilityPosition | None = None
    merged_before: list[type[AbstractCapability[Any]]] = []
    merged_after: list[type[AbstractCapability[Any]]] = []
    merged_requires: list[type[AbstractCapability[Any]]] = []
    has_any = False

    for leaf in iter_leaves(cap):
        ordering = type(leaf).get_ordering()
        if ordering is None:
            continue
        has_any = True
        if ordering.position is not None:
            merged_position = ordering.position
        merged_before.extend(ordering.before)
        merged_after.extend(ordering.after)
        merged_requires.extend(ordering.requires)

    if not has_any:
        return None
    return CapabilityOrdering(
        position=merged_position,
        before=merged_before,
        after=merged_after,
        requires=merged_requires,
    )


def iter_leaves(cap: AbstractCapability[Any]) -> Iterator[AbstractCapability[Any]]:
    """Recursively yield all leaf capabilities."""
    from .combined import CombinedCapability

    if isinstance(cap, CombinedCapability):
        for child in cap.capabilities:
            yield from iter_leaves(child)
    else:
        yield cap


def _collect_leaf_types(cap: AbstractCapability[Any]) -> set[type]:
    return {type(leaf) for leaf in iter_leaves(cap)}
