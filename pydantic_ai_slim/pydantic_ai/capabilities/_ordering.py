"""Topological sorting of capabilities based on ordering constraints."""

from __future__ import annotations

from collections.abc import Sequence
from graphlib import CycleError, TopologicalSorter
from typing import TYPE_CHECKING, Any, TypeVar

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

    exclusive = [i for i in range(n) if _exclusive_leaf(caps[i]) is not None]

    _validate_requires(caps, orderings, leaf_types)
    _validate_exclusive_execution(caps, exclusive)

    return _topo_sort(caps, orderings, leaf_types, cap_leaves, exclusive)


_EXECUTION_HOOKS = ('wrap_tool_execute', 'get_wrapper_toolset')
"""The two ways a capability can take part in executing a tool.

A capability reaching execution by wrapping the toolset is nested inside another just as surely as
one reaching it through the hook, so a rule about what may nest inside what has to see both.
"""


def participates_in_tool_execution(cap: AbstractCapability[Any]) -> bool:
    """Whether any leaf of `cap` takes part in executing a tool, rather than only observing it."""
    return any(
        getattr(type(leaf), name) is not getattr(AbstractCapability, name)
        for leaf in claimed_leaves(cap)
        for name in _EXECUTION_HOOKS
    )


def claimed_leaves(cap: AbstractCapability[Any]) -> list[AbstractCapability[Any]]:
    """Every capability inside `cap`, wrappers included and seen through.

    `collect_leaves` stops at a `WrapperCapability`, which is right for ordering -- a wrapper is one
    node in the chain, and its own `get_ordering` is what places it. It is wrong for a rule about
    what a capability *claims*: wrapping a durability capability to prefix its tools does not stop
    it being the engine, and the claim has to be visible through the wrapper for the pair to be
    seen at all.
    """
    from .wrapper import WrapperCapability

    found: list[AbstractCapability[Any]] = []
    for leaf in collect_leaves(cap):
        found.append(leaf)
        if isinstance(leaf, WrapperCapability):
            found.extend(claimed_leaves(leaf.wrapped))
    return found


def _safe_at_runtime(cap: AbstractCapability[Any]) -> bool:
    """Whether everything in `cap` is declared safe to add after the agent was built."""
    return all(leaf._safe_at_runtime for leaf in claimed_leaves(cap))  # pyright: ignore[reportPrivateUsage]


def _exclusive_leaf(cap: AbstractCapability[Any]) -> AbstractCapability[Any] | None:
    """The leaf that declared `exclusive_execution`, so errors name it rather than its container."""
    for leaf in claimed_leaves(cap):
        ordering = leaf.get_ordering()
        if ordering and ordering.exclusive_execution:
            return leaf
    return None


def _validate_exclusive_execution(caps: list[AbstractCapability[Any]], exclusive: list[int]) -> None:
    """Refuse two capabilities that each claim to be the innermost participant in execution.

    Checked before sorting rather than after: the edges that place such a capability last would
    make a second one a cycle, and `Circular ordering constraints` says nothing about what is
    actually wrong.
    """
    if len(exclusive) < 2:
        return
    type_names = sorted({type(_exclusive_leaf(caps[i])).__name__ for i in exclusive})
    # A repeat of one capability reads as a count; distinct ones read as a list.
    subject = (
        f'{len(exclusive)} `{type_names[0]}` capabilities'
        if len(type_names) == 1
        else ' and '.join(f'`{name}`' for name in type_names)
    )
    raise UserError(
        f'{subject} each require that nothing nests inside them when a tool executes, and only one '
        'capability can be innermost. Whichever ran outside the other would be wrapping it rather '
        'than the tool, so what it observed or recorded would be about the wrong thing. Attach one.'
    )


def reject_nested_execution(
    existing: Sequence[AbstractCapability[Any]],
    added: Sequence[AbstractCapability[Any]],
) -> None:
    """Refuse capabilities added after construction that would nest inside an exclusive one.

    A capability added for a run is composed *inside* the agent's own, so one that takes part in
    executing a tool lands within any capability that declared `exclusive_execution`. Asked here,
    before the additions are bound, rather than left to the sort: binding is where a durability
    capability registers its durable units, and a configuration that is going to be refused should
    be refused before it registers anything.
    """
    exclusive = next((leaf for cap in existing if (leaf := _exclusive_leaf(cap)) is not None), None)
    if exclusive is None:
        return
    # `_safe_at_runtime` already says "adding this per-run is fine even under a capability that has
    # taken over execution" -- `Instrumentation` wraps a tool call to time it and changes nothing
    # about what runs. Reusing it keeps one answer to that question rather than two.
    intruders = [cap for cap in added if participates_in_tool_execution(cap) and not _safe_at_runtime(cap)]
    if not intruders:
        return
    names = ', '.join(sorted(f'`{type(cap).__name__}`' for cap in intruders))
    raise UserError(
        f'`{type(exclusive).__name__}` requires that nothing nests inside it when a tool executes, '
        f'but {names} would, having been added for this run rather than to the agent. A capability '
        'that takes part in executing a tool has to be there when the agent is built, so the two '
        'can be ordered against each other.'
    )


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
    exclusive: list[int],
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

    _add_position_edges(ts, n, orderings, exclusive)
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
    exclusive: list[int],
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

    # `innermost` is a tier, so listed order decides who is last within it. A capability that owns
    # execution needs to *be* last, so it comes after everything -- including its own tier, which
    # the edges above leave unordered among themselves.
    for ei in exclusive:
        for j in range(n):
            if j != ei:
                ts.add(ei, j)  # ei depends on j (j comes first)


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
                    f'Conflicting positions among nested leaves: {merged_position!r} and {ordering.position!r}. '
                    f'Wrap each tier in its own capability or expose the leaves as siblings.'
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


def is_innermost(cap: AbstractCapability[Any]) -> bool:
    """Whether a capability (merging the orderings of its nested leaves) is in the `innermost` tier."""
    ordering = _effective_ordering(collect_leaves(cap))
    return ordering is not None and ordering.position == 'innermost'


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


CapabilityT = TypeVar('CapabilityT', bound=AbstractCapability[Any])


def find_capability(
    capabilities: Sequence[AbstractCapability[Any]],
    cap_type: type[CapabilityT],
) -> CapabilityT | None:
    """Return the first leaf in a capability list/tree that is an instance of `cap_type`, else `None`."""
    for cap in capabilities:
        for leaf in collect_leaves(cap):
            if isinstance(leaf, cap_type):
                return leaf
    return None
