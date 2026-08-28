#!/usr/bin/env python3
"""Typed boundaries for the triage automation scripts.

Data from outside enters script logic through the models here instead of
hand-walked JSON. The two boundaries have opposite postures: agent output and
snapshot files are validated strictly, because a model wrote them and a flood
or malformed entry must fail the run loudly; GitHub payloads are read
leniently, because GitHub omits fields per event kind and each call site
chooses its own failure direction when data is missing.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, TypeVar, cast

from pydantic import BaseModel, BeforeValidator, Field


def parse_time(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace('Z', '+00:00'))


def item_labels(item: Mapping[str, Any]) -> set[str]:
    """The item's exact label names, skipping malformed entries.

    The router's casefolding search-query guard is deliberately separate.
    """
    values: set[str] = set()
    for entry in item.get('labels', []):
        if isinstance(entry, Mapping):
            name = cast('Mapping[str, object]', entry).get('name')
            if isinstance(name, str):
                values.add(name)
    return values


def _decimal_string(value: object) -> int:
    """Agents must write item numbers as positive decimal strings."""
    if not isinstance(value, str) or re.fullmatch(r'[1-9][0-9]*', value) is None:
        raise ValueError('must be a positive decimal string')
    return int(value)


ItemNumber = Annotated[int, BeforeValidator(_decimal_string)]


class AgentItem(BaseModel):
    """One entry of the agent's output; subclasses add the judgment fields."""

    item_number: ItemNumber


TItem = TypeVar('TItem', bound=AgentItem)


def agent_items(path: str, item_model: type[TItem], *, tag: str, limit: int) -> list[TItem]:
    """Parse the agent's output entries of one `type` tag, ignoring the rest.

    Matching entries validate strictly, and duplicate or too many item numbers
    fail the run: the agent must not be able to act on an item twice or flood
    the batch past what the snapshot allowed.
    """
    loaded: object = json.loads(Path(path).read_text(encoding='utf-8'))
    entries = cast('dict[str, object]', loaded).get('items') if isinstance(loaded, dict) else None
    if not isinstance(entries, list):
        raise ValueError('Agent output must contain an items list')
    items = [
        item_model.model_validate(value)
        for value in cast('list[object]', entries)
        if isinstance(value, dict) and cast('dict[str, object]', value).get('type') == tag
    ]
    numbers = [item.item_number for item in items]
    if len(numbers) > limit or len(numbers) != len(set(numbers)):
        raise ValueError('Agent output contains too many or duplicate items')
    return items


class SnapshotCandidate(BaseModel):
    number: int = Field(ge=1)
    updated_at: str


class _Snapshot(BaseModel):
    candidates: list[SnapshotCandidate]


def snapshot_candidates(path: str, *, limit: int) -> dict[int, str]:
    """Return the trusted candidate map (number -> snapshot updated_at)."""
    snapshot = _Snapshot.model_validate_json(Path(path).read_text(encoding='utf-8'))
    candidates = {candidate.number: candidate.updated_at for candidate in snapshot.candidates}
    if len(candidates) != len(snapshot.candidates):
        raise ValueError('Snapshot candidates must have unique numbers')
    if len(candidates) > limit:
        raise ValueError('Snapshot exceeds the candidate limit')
    return candidates


def _account(value: object) -> object:
    # GitHub sends `null` for deleted accounts; read it as an empty account so
    # call sites compare logins without None-guards.
    return cast('dict[str, object]', value) if isinstance(value, dict) else {}


class Account(BaseModel):
    login: str = ''
    type: str = ''


class LabelStub(BaseModel):
    name: str = ''


class IssueEvent(BaseModel):
    """One REST issue or timeline event; GitHub omits fields per event kind."""

    event: str = ''
    id: Annotated[int | str | None, BeforeValidator(lambda value: value if isinstance(value, (int, str)) else None)] = None
    created_at: str = ''
    actor: Annotated[Account, BeforeValidator(_account)] = Field(default_factory=Account)
    # On `assigned`/`unassigned` events `actor` mirrors the assignee; the
    # performer is `assigner`. Verified against the live API.
    assignee: Annotated[Account, BeforeValidator(_account)] = Field(default_factory=Account)
    assigner: Annotated[Account, BeforeValidator(_account)] = Field(default_factory=Account)
    label: Annotated[LabelStub, BeforeValidator(_account)] = Field(default_factory=LabelStub)
