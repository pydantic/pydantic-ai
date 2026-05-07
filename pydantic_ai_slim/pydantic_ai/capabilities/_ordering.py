"""Topological sorting of capabilities based on ordering constraints."""

from __future__ import annotations

from collections.abc import Sequence
from graphlib import CycleError, TopologicalSorter
from typing import TYPE_CHECKING, Any

from pydantic_ai.exceptions import UserError

from .abstract import AbstractCapability, CapabilityOrdering, CapabilityRef

if TYPE_CHECKING:
    from .abstract import CapabilityPosition


def sort_capabilities(
    capabilities: Sequence[AbstractCapability[Any]],
) -> list[AbstractCapability[Any]]:
    """Sort capabilities to satisfy ordering constraints.

    Preserves the original order as a tiebreaker when constraints allow.
    Raises `UserError` on conflicts (missing requirements, cycles).
    """
    caps = list(capabilities)
    n = len(caps)
    if n <= 1:
        return caps

    cap_leaves: list[list[AbstractCapability[Any]]] = [collect_leaves(cap) for cap in caps]
    orderings: list[CapabilityOrdering | None] = [_effective_ordering(leaves) for leaves in cap_leaves]
    leaf_types: list[set[type]] = [{type(leaf) for leaf in leaves} for leaves in cap_leaves]

    _validate_requires(caps, orderings, leaf_types)

    return _topo_sort(caps, orderings, leaf_types, cap_leaves)


def _validate_requires(
    caps: list[AbstractCapability[Any]],
    orderings: list[CapabilityOrdering | None],
    leaf_types: list[set[type]],
) -> None:
    """Validate required dependencies."""
    all_leaf_types: set[type] = set[type]().union(*leaf_types)
    for i, ordering in enumerate(orderings):
        if ordering and ordering.requires:
            for req_type in ordering.requires:
                if not any(issubclass(t, req_type) for t in all_leaf_types):
                    raise UserError(
                        f'`{type(caps[i]).__name__}` requires `{req_type.__name__}` '
                        f'but it was not found among the capabilities.'
                    )


def _topo_sort(
    caps: list[AbstractCapability[Any]],
    orderings: list[CapabilityOrdering | None],
    leaf_types: list[set[type]],
    cap_leaves: list[list[AbstractCapability[Any]]],
) -> list[AbstractCapability[Any]]:
    """Topological sort using graphlib.TopologicalSorter.

    Edges go from outer (earlier) to inner (later). TopologicalSorter
    preserves insertion order as tiebreaker for unconstrained nodes.
    """
    n = len(caps)
    ts: TopologicalSorter[int] = TopologicalSorter()

    # Add all nodes in original order (establishes tiebreaker)
    for i in range(n):
        ts.add(i)

    _add_position_edges(ts, n, orderings)
    _add_relative_edges(ts, n, orderings, leaf_types, cap_leaves)

    try:
        sorted_indices = list(ts.static_order())
    except CycleError:
        raise UserError('Circular ordering constraints among capabilities')

    return [caps[i] for i in sorted_indices]


def _add_position_edges(
    ts: TopologicalSorter[int],
    n: int,
    orderings: list[CapabilityOrdering | None],
) -> None:
    outermost = {i for i, o in enumerate(orderings) if o and o.position == 'outermost'}
    innermost = {i for i, o in enumerate(orderings) if o and o.position == 'innermost'}

    # Outermost tier: each member must come before all non-members.
    for oi in outermost:
        for j in range(n):
            if j != oi and j not in outermost:
                ts.add(j, oi)  # j depends on oi (oi comes first)

    # Innermost tier: each member must come after all non-members.
    for ii in innermost:
        for j in range(n):
            if j != ii and j not in innermost:
                ts.add(ii, j)  # ii depends on j (j comes first)


def _add_relative_edges(
    ts: TopologicalSorter[int],
    n: int,
    orderings: list[CapabilityOrdering | None],
    leaf_types: list[set[type]],
    cap_leaves: list[list[AbstractCapability[Any]]],
) -> None:
    for i, ordering in enumerate(orderings):
        if not ordering:
            continue
        # wraps=[X] → I come before X
        for ref in ordering.wraps:
            for j in range(n):
                if i != j and _ref_matches(ref, leaf_types[j], cap_leaves[j]):
                    ts.add(j, i)  # j depends on i (i comes first)
        # wrapped_by=[X] → X comes before me
        for ref in ordering.wrapped_by:
            for j in range(n):
                if i != j and _ref_matches(ref, leaf_types[j], cap_leaves[j]):
                    ts.add(i, j)  # i depends on j (j comes first)


def _ref_matches(
    ref: CapabilityRef,
    leaf_types: set[type],
    leaves: list[AbstractCapability[Any]],
) -> bool:
    """Check if a capability ref matches any leaf in a capability group.

    Type refs match via `issubclass`; instance refs match via `is` identity.
    """
    if isinstance(ref, type):
        return any(issubclass(t, ref) for t in leaf_types)
    return any(leaf is ref for leaf in leaves)


def _effective_ordering(leaves: list[AbstractCapability[Any]]) -> CapabilityOrdering | None:
    """Get the effective ordering for a capability, merging from all its leaves.

    For plain capabilities (single leaf), returns `get_ordering()` directly.
    For containers (`CombinedCapability`, `WrapperCapability`), merges
    constraints from all leaves.
    """
    merged_position: CapabilityPosition | None = None
    merged_wraps: list[CapabilityRef] = []
    merged_wrapped_by: list[CapabilityRef] = []
    merged_requires: list[type[AbstractCapability[Any]]] = []
    has_any = False

    for leaf in leaves:
        ordering = leaf.get_ordering()
        if ordering is None:
            continue
        has_any = True
        if ordering.position is not None:
            if merged_position is not None and merged_position != ordering.position:
                raise UserError(
                    f'Conflicting positions in nested CombinedCapability: {merged_position!r} and {ordering.position!r}'
                )
            merged_position = ordering.position
        merged_wraps.extend(ordering.wraps)
        merged_wrapped_by.extend(ordering.wrapped_by)
        merged_requires.extend(ordering.requires)

    if not has_any:
        return None
    return CapabilityOrdering(
        position=merged_position,
        wraps=merged_wraps,
        wrapped_by=merged_wrapped_by,
        requires=merged_requires,
    )


def collect_leaves(cap: AbstractCapability[Any]) -> list[AbstractCapability[Any]]:
    """Collect all leaf capabilities using the `apply` visitor pattern."""
    leaves: list[AbstractCapability[Any]] = []
    cap.apply(leaves.append)
    return leaves


def has_capability_type(
    capabilities: Sequence[AbstractCapability[Any]],
    cap_type: type[AbstractCapability[Any]],
) -> bool:
    """Check whether any leaf in a capability list/tree is an instance of the given type."""
    return any(isinstance(leaf, cap_type) for cap in capabilities for leaf in collect_leaves(cap))
