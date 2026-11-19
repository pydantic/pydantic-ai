"""Check a checked-in classification table against the members it classifies.

A table classifies the members of something that grows: the fields of a message class, the values a provider SDK
returns, the parameters of an entry point. A test enumerates those members and passes them to `assert_classified`
with the table. A new member fails until someone classifies it, and an entry whose member is gone fails as stale,
so the table can't drift from what it describes. Every entry carries a reason, so a reviewer can check the
classification rather than take it on trust.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol, TypeVar

from pydantic_ai._utils import get_union_args

KeyT = TypeVar('KeyT', bound=Hashable)


class Classified(Protocol):
    """A table entry: how a member is handled, and why."""

    @property
    def reason(self) -> str: ...


@dataclass(frozen=True)
class Exempt:
    """A member a check leaves out on purpose."""

    reason: str


def assert_classified(
    enumerated: Iterable[KeyT],
    table: Mapping[KeyT, Classified],
    *,
    complete_walk: bool,
    table_name: str,
) -> None:
    """Fail unless `table` classifies every member of `enumerated`, each with a reason.

    Also fail on a stale entry, one that names no member, but only when `complete_walk` is true: an enumeration
    that skipped members, such as modules whose optional dependencies aren't installed, would report their entries
    as stale when they aren't.
    """
    members = set(enumerated)
    problems: list[str] = []
    if unclassified := members - table.keys():
        problems.append(f'`{table_name}` has no entry for {_names(unclassified)}: add one with its reason.')
    if unexplained := [key for key, entry in table.items() if not entry.reason.strip()]:
        problems.append(f'`{table_name}` entries {_names(unexplained)} have no reason.')
    if complete_walk and (stale := table.keys() - members):
        problems.append(f'`{table_name}` entries {_names(stale)} no longer name a member: remove them.')
    assert not problems, '\n'.join(problems)


def union_members(union: object) -> set[type]:
    """The classes a union admits, with nested unions flattened and `Annotated` wrappers removed."""
    members: set[type] = set()
    for member in get_union_args(union):
        members |= union_members(member) if get_union_args(member) else {member}
    return members


def _names(keys: Iterable[Hashable]) -> list[str]:
    return sorted(_name(key) for key in keys)


def _name(key: object) -> str:
    if isinstance(key, type):
        return key.__qualname__
    if isinstance(key, tuple):
        parts: tuple[object, ...] = key  # pyright: ignore[reportUnknownVariableType]
        return '.'.join(_name(part) for part in parts)
    return str(key)
