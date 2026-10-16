#!/usr/bin/env python3
"""Typed boundaries for the triage automation scripts.

Data from outside enters script logic through the models here instead of
hand-walked JSON. The two boundaries have opposite postures: agent output and
snapshot files are validated strictly, because a model wrote them and a flood
or malformed entry must fail the run loudly; GitHub payloads are read
leniently — a null or missing field reads as its default, because GitHub omits
fields per event kind — while a wrong-typed field still fails loudly, because
that means corruption rather than an absent value.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, TypeVar, cast

from pydantic import BaseModel, BeforeValidator, Field, model_validator


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


class _AgentOutput(BaseModel):
    items: list[dict[str, Any]]


def agent_items(path: str, item_model: type[TItem], *, tag: str, limit: int) -> list[TItem]:
    """Parse the agent's output entries of one `type` tag, ignoring the rest.

    Matching entries validate strictly, and duplicate or too many item numbers
    fail the run: the agent must not be able to act on an item twice or flood
    the batch past what the snapshot allowed.
    """
    entries = _AgentOutput.model_validate_json(Path(path).read_text(encoding='utf-8')).items
    items = [item_model.model_validate(entry) for entry in entries if entry.get('type') == tag]
    numbers = [item.item_number for item in items]
    if len(numbers) > limit or len(numbers) != len(set(numbers)):
        raise ValueError('Agent output contains too many or duplicate items')
    return items


class SnapshotCandidate(BaseModel):
    number: int = Field(ge=1, strict=True)
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


class _GitHubObject(BaseModel):
    """Lenient base for GitHub payloads: null and absent both read as defaults."""

    @model_validator(mode='before')
    @classmethod
    def _nulls_as_missing(cls, value: object) -> object:
        # GitHub sends explicit `null` for deleted accounts and absent
        # performers; a non-object payload reads as an entirely absent one.
        if isinstance(value, dict):
            return {key: item for key, item in cast('dict[str, object]', value).items() if item is not None}
        return {}


class Account(_GitHubObject):
    login: str = ''
    type: str = ''


class LabelStub(_GitHubObject):
    name: str = ''


class IssueEvent(_GitHubObject):
    """One REST issue or timeline event; GitHub omits fields per event kind."""

    event: str = ''
    # The census dedup key; a non-scalar id reads as unknown.
    id: Annotated[int | str | None, BeforeValidator(lambda value: value if type(value) in (int, str) else None)] = None
    created_at: str = ''
    actor: Account = Field(default_factory=Account)
    # On `assigned`/`unassigned` events `actor` mirrors the assignee; the
    # performer is `assigner`. Verified against the live API.
    assignee: Account = Field(default_factory=Account)
    assigner: Account = Field(default_factory=Account)
    label: LabelStub = Field(default_factory=LabelStub)
