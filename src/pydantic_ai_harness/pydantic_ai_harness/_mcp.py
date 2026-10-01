"""Shared helpers for capabilities that connect to hosted MCP servers."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import fields
from os import environ
from typing import Any, TypeVar

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset

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


class MCPReadOnlyNoToolsWarning(UserWarning):
    """`read_only=True` removed every tool from a hosted MCP server, so the agent gets none of its tools.

    The filter keeps only tools the server annotates with `readOnlyHint: true`, and some servers
    publish no annotations at all. Disable `read_only`, or filter this category when an empty
    toolset is intended.
    """


def read_only_toolset(toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
    """Keep tools marked as read-only and warn when the filter removes every tool."""

    def prepare_read_only(_ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        read_only_tools = [tool for tool in tool_defs if is_read_only(tool)]
        if tool_defs and not read_only_tools:
            warnings.warn(
                f'`read_only=True` removed every tool from {toolset.label} because none was marked with '
                '`readOnlyHint: true`. '
                'Disable `read_only` or configure the server to publish read-only annotations, or filter '
                '`MCPReadOnlyNoToolsWarning` if this is intended.',
                MCPReadOnlyNoToolsWarning,
                stacklevel=2,
            )
        return read_only_tools

    return toolset.prepared(prepare_read_only)


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
