"""Private helpers for capabilities that combine by conjunction, so that adding one only ever narrows.

`FileSystem` and `Shell` declare a default `id`, so two on one agent are one capability. Their
restrictions must all hold, whatever order they were listed in: an allowlist intersects, a denylist
unions, a limit takes the smaller value.

Only what an instance *states* takes part, so that `Shell(allowed_commands=['pytest'])` beside `Coder()`
narrows the commands without also cutting `Coder`'s timeout to `Shell`'s default. A field is stated when
the constructor was passed it, whatever its value, as `RecordsStated` records. A default of `None`, `[]`,
`False` or `True` that is also the identity of the field's conjunction (no limit, nothing denied) needs no
such test; the few that are not (`root_dir=None`, `allow_interactive=False`) conjoin as the value they mean,
which can only narrow.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import Field
from typing import Any, ClassVar, Self, TypeVar

from pydantic_ai.exceptions import UserError

T = TypeVar('T')
K = TypeVar('K', bound=Hashable)
V = TypeVar('V')


class RecordsStated:
    """A mixin for a dataclass capability that records which fields its constructor was passed.

    Recorded in `__new__`, which receives the constructor's arguments, so the dataclass `__init__` and its
    signature stay as they are. The record is a plain attribute, not a field, so it is not in `repr`,
    equality, the agent spec or its schema. `copy` and `deepcopy` keep it. `dataclasses.replace()` and
    `from_spec` call the constructor, so they record what they pass: `from_spec` the spec's own arguments,
    and `replace()` every field, so a `replace()`d copy counts every field as stated, on every Python version.
    """

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]
    _stated: frozenset[str]

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        instance = super().__new__(cls)
        positional = [field.name for field in cls.__dataclass_fields__.values() if field.init and not field.kw_only]
        instance._stated = frozenset(positional[: len(args)]) | frozenset(kwargs)
        return instance

    @classmethod
    def _stated_values(cls, capabilities: Iterable[RecordsStated], name: str) -> list[Any]:
        """The values of field `name` that `capabilities` were passed, in order."""
        return [getattr(capability, name) for capability in capabilities if name in capability._stated]

    @classmethod
    def _smallest(cls, capabilities: Sequence[RecordsStated], name: str) -> Any:
        """The smallest stated value of field `name`, `None` meaning unbounded; the class default when none is."""
        given = cls._stated_values(capabilities, name)
        bounded = [value for value in given if value is not None]
        if bounded:
            return min(bounded)
        return None if given else cls.__dataclass_fields__[name].default


C = TypeVar('C', bound=RecordsStated)


def state_all(capability: C) -> C:
    """Count every field of `capability` as stated, defaults included, and return it.

    For a packaged harness's own instance: its defaults are part of what it promises, so an instance added
    beside it must not drop them by leaving the same fields unset.
    """
    capability._stated = frozenset(capability.__dataclass_fields__)  # pyright: ignore[reportPrivateUsage]
    return capability


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
