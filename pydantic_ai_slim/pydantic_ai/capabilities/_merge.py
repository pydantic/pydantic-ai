"""Merging two capabilities that resolved to the same `id` into the one a run uses.

Private to the package: this is what `AbstractCapability.combine` does by default, so a capability
that wants it gets it by declaring a default `id` and writing no `combine` at all. One that wants
something else overrides `combine` and writes the merge it needs, rather than parameterizing this.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence, Set as AbstractSet
from typing import TYPE_CHECKING, Any, cast

from pydantic_ai._utils import replace_no_init
from pydantic_ai.tools import AgentDepsT

if TYPE_CHECKING:
    from .abstract import AbstractCapability


def merge_capability_fields(
    capabilities: Sequence[AbstractCapability[AgentDepsT]],
) -> AbstractCapability[AgentDepsT]:
    """Merge same-class capabilities field by field, the way overlapping dicts merge.

    The `combine` implementation for a capability an agent has one of, where two of them are one
    configuration stated twice and nothing either of them states should be dropped:

    | both sides | result |
    |---|---|
    | neither states a value | unset |
    | only one states a value | that value survives |
    | both state a mapping, sequence, or set | the union, later entries winning a shared key |
    | anything else | the later value |

    `None` reads as "not stated", which is what makes a restriction survive being composed with a
    capability that doesn't mention it: `WebSearch(allowed_domains=[...])` beside a plain
    `WebSearch()` keeps the allow-list, rather than the plain one silently widening it.

    Merging starts from the last capability so the fields that can't be reconciled — and the
    subclass state `replace_no_init` carries over — are the last one's, matching the "later wins"
    fallback.

    Fields declared `compare=False` are per-run bookkeeping rather than configuration, so they are
    left as the last capability's rather than merged.

    Two limits worth knowing, both from configuration that is resolved before `combine` sees it:
    a field a `__post_init__` has already materialized (`NativeOrLocalTool` turns `native=True` into
    a tool instance and `local=None` into its default) can no longer be told apart from one the user
    stated, so those take the later value; and two objects that carry no meaningful equality (two
    stores, two clients) likewise take the later. A capability that needs either reconciled — or
    needs a numeric budget to take the *smaller* value rather than the later one — overrides
    `combine` itself.
    """
    merged = capabilities[-1]
    changes: dict[str, Any] = {}
    for field in dataclasses.fields(merged):
        if not field.compare:
            continue
        current = getattr(merged, field.name)
        value = merge_field_values([getattr(capability, field.name) for capability in capabilities])
        if value is not current:
            changes[field.name] = value
    return replace_no_init(merged, **changes) if changes else merged


def merge_field_values(values: Sequence[Any]) -> Any:
    """Merge one field's values across the capabilities sharing an id, in application order."""
    stated = [value for value in values if value is not None]
    if not stated:
        return None
    first, *rest = stated
    if all(_same_value(first, other) for other in rest):
        return first
    if all(isinstance(value, Mapping) for value in stated):
        merged_mapping: dict[Any, Any] = {}
        for value in cast('list[Mapping[Any, Any]]', stated):
            merged_mapping.update(value)
        return merged_mapping
    if all(isinstance(value, AbstractSet) for value in stated):
        return set[Any]().union(*cast('list[AbstractSet[Any]]', stated))
    if all(isinstance(value, Sequence) and not isinstance(value, (str, bytes)) for value in stated):
        # Ordered union: a shared entry keeps the position its first mention gave it.
        merged_sequence: list[Any] = []
        for value in cast('list[Sequence[Any]]', stated):
            merged_sequence.extend(
                entry for entry in value if not any(_same_value(entry, kept) for kept in merged_sequence)
            )
        return merged_sequence
    return stated[-1]


def _same_value(left: Any, right: Any) -> bool:
    """Whether two field values are interchangeable, treating an unusable `__eq__` as "no"."""
    if left is right:
        return True
    try:
        return bool(left == right)
    except Exception:  # pragma: no cover
        # A field whose `__eq__` raises (e.g. an array-like) is not something we can merge.
        return False
