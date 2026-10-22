"""Shared helpers for capabilities that connect to hosted MCP servers."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from os import environ

from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, PreparedToolset
from pydantic_ai_harness._combine import one_per_id
from pydantic_ai_harness._warn import MCPReadOnlyNoToolsWarning

one_connection = one_per_id
"""Resolve hosted MCP capabilities that share an `id`: one connection stated twice is one, two that disagree raise."""


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


@dataclass
class _IdentifiedPreparedToolset(PreparedToolset[AgentDepsT]):
    """A `PreparedToolset` that keeps the wrapped toolset's `id`, so durable execution can still find it by that id."""

    @property
    def id(self) -> str | None:
        return self.wrapped.id


def read_only_toolset(toolset: AbstractToolset[AgentDepsT], *, keep_id: bool = False) -> AbstractToolset[AgentDepsT]:
    """Keep tools marked as read-only and warn when the filter removes every tool.

    `keep_id=True` gives the filtered toolset the wrapped toolset's `id`; otherwise it has none, like any wrapper.
    """

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

    if keep_id:
        return _IdentifiedPreparedToolset(toolset, prepare_read_only)
    return toolset.prepared(prepare_read_only)
