"""Merging two capabilities that resolved to the same `id` into the one a run uses.

Private to the package: this is what `AbstractCapability.combine` does by default, so a capability
that wants it gets it by declaring a default `id` and writing no `combine` at all. One that wants
something else overrides `combine` and writes the merge it needs, rather than parameterizing this.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence, Set as AbstractSet
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

from pydantic_ai._utils import replace_no_init
from pydantic_ai.exceptions import UserError
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

    An attribute no field declares is a harder limit, and is refused. The table above is what the
    merge promises, and its first row is the load-bearing one: a value only one side states
    survives. That cannot hold for an attribute the merge cannot enumerate -- `replace_no_init`
    copies the last instance, so such an attribute silently takes the last value even where an
    earlier instance was the only one to state it. Refusing says so instead.

    A leading underscore does not exempt an attribute from that: it does not distinguish
    configuration from derived state, and configuration is exactly what must not be dropped. Two
    kinds are exempt, and they are exempt for different reasons. State the class can produce again
    -- a `cached_property` -- is *dropped* from the copy, so it is recomputed from the merged fields
    rather than carried over stale. `__orig_class__` is not the capability's state at all, so it is
    left alone and simply not counted against it. Nothing here re-runs `__post_init__`
    (`replace_no_init` exists precisely to skip it), so state a `__post_init__` derives is declared
    as a field and recomputed in the capability's own `combine`, the way `NativeOrLocalTool`
    rebuilds its native tool.
    """
    merged = capabilities[-1]
    field_names = {field.name for field in dataclasses.fields(merged)}
    # A subclass may turn an inherited `cached_property`'s name into a real field. It is then
    # configuration, merged like any other, and dropping it would leave the class default showing.
    rebuildable = _rebuildable_attributes(type(merged)) - field_names
    for capability in capabilities:
        # Not "cannot be merged" -- `replace_no_init` copies the last instance, so an undeclared
        # attribute does get *a* value. What it cannot get is the promise above: a value only an
        # earlier instance stated would be dropped rather than survive. Refuse rather than let that
        # differ silently from every declared field.
        undeclared = set(vars(capability)) - field_names - rebuildable - _NOT_CAPABILITY_STATE
        if undeclared:
            cls_name = type(merged).__name__
            names = ', '.join(sorted(undeclared))
            raise UserError(
                f'Capability id {merged.id!r} is used by multiple {cls_name} capabilities, but {cls_name} '
                f'sets {names} outside its dataclass fields. Merging keeps a value only one of them states, '
                'and cannot do that for an attribute it cannot enumerate -- the last instance would win '
                f'silently. Declare {names} as dataclass fields to have them merged; or, for state derived '
                'from other fields, make it a `cached_property` so merging can recompute it, or declare it '
                'as a field and override `combine` to recompute it after merging, since this does not '
                're-run `__post_init__`.'
            )
    changes: dict[str, Any] = {}
    for field in dataclasses.fields(merged):
        if not field.compare:
            continue
        current = getattr(merged, field.name)
        value = merge_field_values(
            [getattr(capability, field.name) for capability in capabilities], field_name=field.name
        )
        if value is not current:
            changes[field.name] = value
    result = replace_no_init(merged, **changes) if changes else merged
    if changes:
        # The copy carries the last instance's cached values, which were derived from *its* fields.
        # Dropping them is what makes the next read recompute against the merged ones -- the answer
        # `__post_init__` cannot give here, since `replace_no_init` deliberately does not run it.
        instance_dict = cast('dict[str, Any]', result.__dict__)
        for name in rebuildable & set(instance_dict):
            del instance_dict[name]
    return result


_NOT_CAPABILITY_STATE = frozenset({'__orig_class__'})
"""Instance attributes that are not the capability's own state, so the merge has no say over them.

`typing` attaches `__orig_class__` when a generic class is instantiated through a subscription. It
records what the annotation said, which `copy.copy` carries over correctly and nothing can
recompute -- so it is neither counted against the class nor dropped, only left where it is.
"""


def _rebuildable_attributes(cls: type[AbstractCapability[Any]]) -> frozenset[str]:
    """Attribute names the class can produce again on demand, so merging drops rather than keeps them.

    A `cached_property` is the supported way to derive state from fields. It is declared on the
    class, so this can find it without a list of exceptions to maintain, and discarding the cached
    value is what makes the merged capability recompute against the merged fields instead of
    reporting what the last instance happened to have cached.

    Only names that can actually be recomputed belong here, since everything here is deleted from
    the merged copy -- an attribute that is merely uninteresting to the merge goes in
    `_NOT_CAPABILITY_STATE` instead, where it is exempt from the check without being destroyed.
    """
    names: set[str] = set()
    for klass in cls.__mro__:
        names.update(name for name, value in vars(klass).items() if isinstance(value, cached_property))
    return frozenset(names)


def merge_field_values(values: Sequence[Any], *, field_name: str) -> Any:
    """Merge one field's values across the capabilities sharing an id, in application order.

    `field_name` names the field in the error raised when a declared collection type cannot be
    rebuilt; the merge itself does not depend on it. Required rather than defaulted, so no caller
    can produce an error that leaves its reader guessing which field to look at.
    """
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
        return _as_declared(first, merged_mapping, field_name)
    if all(isinstance(value, AbstractSet) for value in stated):
        merged_set = set[Any]().union(*cast('list[AbstractSet[Any]]', stated))
        return _as_declared(first, merged_set, field_name)
    if all(isinstance(value, Sequence) and not _is_record(value) for value in stated):
        # Ordered union: a shared entry keeps the position its first mention gave it.
        merged_sequence: list[Any] = []
        for value in cast('list[Sequence[Any]]', stated):
            merged_sequence.extend(
                entry for entry in value if not any(_same_value(entry, kept) for kept in merged_sequence)
            )
        return _as_declared(first, merged_sequence, field_name)
    return stated[-1]


def _is_record(value: Any) -> bool:
    """Whether a value is a record that merely satisfies `Sequence`, rather than a collection.

    A `str` is a sequence of characters and a `NamedTuple` a sequence of its fields, but neither is
    something two capabilities are contributing *entries* to: unioning them would splice one
    value's characters or columns into the other's. They take the later value, which is what the
    table gives everything the merge cannot combine.
    """
    if isinstance(value, (str, bytes)):
        return True
    # `_fields` is what `typing.NamedTuple` and `collections.namedtuple` both leave behind; there is
    # no type to `isinstance` against.
    return isinstance(value, tuple) and hasattr(cast('Any', value), '_fields')


def _as_declared(first: Any, merged: Any, field_name: str) -> Any:
    """Rebuild `merged` as the collection type the field already held, or refuse.

    The union is computed in a plain `dict`/`set`/`list`, which would hand a field annotated
    `tuple[str, ...]` or `frozenset[str]` a value of the wrong type -- and `replace_no_init` skips
    the `__post_init__` that might otherwise have caught it.

    A collection type that cannot be rebuilt from its contents leaves no answer worth guessing.
    Handing back the plain union gives the field a value its own annotation says it cannot hold;
    handing back one instance's value drops entries the other stated, which is the one thing the
    merge promises not to do. Records that only look like collections are already gone by here
    (see `_is_record`), so what is left is a real collection with no way to rebuild it -- and
    saying so is what the capability's author needs to hear.
    """
    if type(first) is type(merged):
        return merged
    try:
        return type(first)(merged)
    except (TypeError, ValueError) as exc:
        raise UserError(
            f'Merging capabilities under one `id` unioned field {field_name!r} into a '
            f'{type(merged).__name__}, but its declared type {type(first).__name__} cannot be rebuilt '
            f'from that. Keeping the union would '
            f'give the field a value {type(first).__name__} does not describe, and keeping one instance '
            f'would drop what the other stated. Give {type(first).__name__} a constructor that takes its '
            'contents, declare the field as a plain collection type, or override `combine` to merge this '
            'field itself.'
        ) from exc


def _same_value(left: Any, right: Any) -> bool:
    """Whether two field values are interchangeable, treating an unusable `__eq__` as "no"."""
    if left is right:
        return True
    try:
        return bool(left == right)
    except Exception:
        # A field whose `__eq__` raises (e.g. an array-like) is not something we can merge.
        return False
