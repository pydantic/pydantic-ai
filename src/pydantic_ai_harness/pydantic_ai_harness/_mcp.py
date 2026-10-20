"""Shared helpers for capabilities that connect to hosted MCP servers."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from os import environ

from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, DynamicToolset
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


def connect_with_auth(
    auth: str | Callable[[RunContext[AgentDepsT]], str | None] | None,
    connect: Callable[[str | None], AbstractToolset[AgentDepsT]],
    *,
    id: str | None,
    expected: str = 'an API key or token',
) -> AbstractToolset[AgentDepsT]:
    """Connect with `auth` now, or, when it is a function, once per run with the token it returns.

    A function that returns `None` or `''` gives that run no tools. It must not return `'oauth'`, which FastMCP
    reads as a browser sign-in that would hang a server run; `expected` names what it should return instead.
    """
    if not callable(auth):
        return connect(auth)

    def connect_for_run(ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        token = auth(ctx)
        if token == 'oauth':
            raise UserError(f"The `auth` function must return {expected}, not 'oauth'.")
        return connect(token) if token else None

    # Registered once under a fixed `id`, as durable execution requires; filled per run.
    return DynamicToolset(connect_for_run, per_run_step=False, id=id)


def is_read_only(tool: ToolDefinition) -> bool:
    """Whether the server explicitly marks a tool read-only."""
    match (tool.metadata or {}).get('annotations'):
        case {'readOnlyHint': True}:
            return True
        case _:
            return False


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
