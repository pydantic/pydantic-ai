"""Shared helpers for capabilities that connect to hosted MCP servers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields
from os import environ
from typing import Any, TypeVar

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import ToolDefinition

CapabilityT = TypeVar('CapabilityT', bound=AbstractCapability[Any])


def credential(auth: str | None, *, env: str | None, service: str) -> str:
    """The API key or token to connect with: `auth`, else the `env` variable. An empty string counts as unset."""
    if not auth and env is not None:
        auth = environ.get(env)
    if not auth:
        raise UserError(
            f'Set `{env}` or pass `auth` to connect to {service}.' if env else f'Pass `auth` to connect to {service}.'
        )
    return auth


def is_read_only(tool: ToolDefinition) -> bool:
    """Whether the server explicitly marks a tool read-only."""
    match (tool.metadata or {}).get('annotations'):
        case {'readOnlyHint': True}:
            return True
        case _:
            return False


def one_connection(capabilities: Sequence[CapabilityT]) -> CapabilityT:
    """Resolve hosted MCP capabilities that share an `id`, the way `SubAgents.combine` refuses what it cannot merge.

    One capability is one connection to one account, and its settings are an access boundary, so this narrows
    rather than unions (see "Deciding What Two Of It Mean" in `agent_docs/capability-authoring.md`): the same
    configuration stated twice is that one connection; two that disagree raise, naming the fields but not their
    values, since `auth` is a secret.
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
