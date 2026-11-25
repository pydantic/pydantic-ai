"""Private helpers for capabilities that combine by conjunction, so that adding one only ever narrows.

`FileSystem` and `Shell` declare a default `id`, so two on one agent are one capability. Their
restrictions must all hold, whatever order they were listed in: an allowlist intersects, a denylist
unions, a limit takes the smaller value.

Only what an instance *states* takes part, so that `Shell(allowed_commands=['pytest'])` beside `Coder()`
narrows the commands without also cutting `Coder`'s timeout to `Shell`'s default. A field states its
value unless the value is one of the marker types below: a default that is equal to the plain value and
behaves like it, told apart only by type. A default of `None`, `[]`, `False` or `True` needs no marker
where it is also the identity of the field's conjunction (no limit, nothing denied); the few that are not
(`root_dir=None`, `allow_interactive=False`) conjoin as the value they mean, which can only narrow.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import fields, replace
from typing import Any, TypeVar

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError

T = TypeVar('T')
N = TypeVar('N', int, float)
C = TypeVar('C', bound=AbstractCapability[Any])
K = TypeVar('K', bound=Hashable)
V = TypeVar('V')


class DefaultInt(int):
    """An `int` default that no one stated."""

    __slots__ = ()


class DefaultFloat(float):
    """A `float` default that no one stated."""

    __slots__ = ()


class DefaultStrs(tuple[str, ...]):
    """A `tuple[str, ...]` default that no one stated."""

    __slots__ = ()


def is_stated(value: object) -> bool:
    """Whether `value` was stated, rather than being a default marker."""
    return not isinstance(value, (DefaultInt, DefaultFloat, DefaultStrs))


def state_defaults(capability: C) -> C:
    """`capability` with every default marker replaced by the plain value it equals, so all of it is stated.

    For a packaged harness's own instance: its defaults are part of what it promises, so an instance added
    beside it must not drop them by leaving the same fields unstated.
    """
    changes: dict[str, object] = {}
    for field in fields(capability):
        value = getattr(capability, field.name)
        if field.init and not is_stated(value):
            changes[field.name] = (
                tuple(value) if isinstance(value, tuple) else float(value) if isinstance(value, float) else int(value)
            )
    return replace(capability, **changes)


def stated(values: Iterable[T]) -> list[T]:
    """The values that were stated, in order."""
    return [value for value in values if is_stated(value)]


def smallest(values: Sequence[N | None], *, default: N | None) -> N | None:
    """The smallest stated value, `None` meaning unbounded; `default` when none is stated."""
    given = stated(values)
    bounded = [value for value in given if value is not None]
    if bounded:
        return min(bounded)
    return None if given else default


def union(lists: Iterable[Sequence[str]]) -> list[str]:
    """Every entry of every list, in first-seen order."""
    return list(dict.fromkeys(entry for entries in lists for entry in entries))


def intersection(
    lists: Sequence[Sequence[str]], *, owner: str, field: str, empty_means_all: bool = True
) -> list[str] | None:
    """The entries every list has, in the first one's order; `None` when there is nothing to intersect.

    With `empty_means_all`, an empty list states no restriction and drops out. Raises when the lists that
    take part share no entry, since an empty result would read as "no restriction" (or register nothing)
    rather than as the contradiction it is.
    """
    taking_part = [list(entries) for entries in lists if entries or not empty_means_all]
    if not taking_part:
        return None
    common = [entry for entry in taking_part[0] if all(entry in entries for entries in taking_part[1:])]
    if not common and any(taking_part):
        raise UserError(
            f'These `{owner}` capabilities share an id, so every one of their `{field}` lists applies, but no entry '
            f'is in all of them: {", ".join(map(repr, taking_part))}. State one list, or give each `{owner}` its '
            'own `id` to keep them apart.'
        )
    return common


def merged_mapping(
    mappings: Iterable[Mapping[K, V] | None], *, owner: str, field: str, show: Callable[[K], Any] = repr
) -> dict[K, V] | None:
    """The union of the stated mappings; raises when two give one key different values."""
    merged: dict[K, V] | None = None
    for mapping in mappings:
        if mapping is None:
            continue
        merged = {} if merged is None else merged
        for key, value in mapping.items():
            if key in merged and merged[key] != value:
                raise UserError(
                    f'These `{owner}` capabilities share an id, but give `{field}` {show(key)} different values. '
                    f'State it once, or give each `{owner}` its own `id` to keep them apart.'
                )
            merged[key] = value
    return merged


def agreed(values: Sequence[T | None], *, owner: str, field: str) -> T | None:
    """The one stated non-`None` value; raises when two differ."""
    given = list(dict.fromkeys(value for value in values if value is not None))
    if len(given) > 1:
        raise UserError(
            f'These `{owner}` capabilities share an id, but disagree on `{field}` ({", ".join(map(repr, given))}). '
            f'State it once, or give each `{owner}` its own `id` to keep them apart.'
        )
    return given[0] if given else None
