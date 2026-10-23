"""Combine policies shared by capabilities that declare a default `id`."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields
from typing import Any, TypeVar

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError

CapabilityT = TypeVar('CapabilityT', bound=AbstractCapability[Any])


def one_per_id(capabilities: Sequence[CapabilityT]) -> CapabilityT:
    """Resolve capabilities that share an `id`: the same configuration stated twice is one, and two that disagree raise.

    For a capability whose configuration is an access boundary or a connection to one account, a
    field-by-field merge could widen what the agent reaches or send one account's credential to
    another's server, so this narrows rather than unions (see "Deciding What Two Of It Mean" in
    `agent_docs/capability-authoring.md`). The error names the fields that disagree but not their values,
    which can be secrets.
    """
    first = capabilities[0]
    for other in capabilities[1:]:
        disagree = [
            field.name
            for field in fields(first)
            if field.compare and field.name != 'id' and getattr(first, field.name) != getattr(other, field.name)
        ]
        if disagree:
            names = ', '.join(repr(name) for name in disagree)
            raise UserError(
                f'Capability id {first.id!r} is used by multiple {type(first).__name__} capabilities that disagree '
                f'on {names}. Give each its own `id` and wrap them in `PrefixTools`, or make them agree.'
            )
    return first
