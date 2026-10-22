"""Shared helpers for capabilities that connect to hosted MCP servers."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from os import environ

from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, DynamicToolset
from pydantic_ai_harness._warn import MCPReadOnlyNoToolsWarning


def credential(auth: str | None, *, env: str | None, service: str) -> str:
    """The API key or token to connect with: `auth`, else the `env` variable. An empty string counts as unset."""
    if not auth and env is not None:
        auth = environ.get(env)
    if not auth:
        raise UserError(
            f'Set `{env}` or pass `auth` to connect to {service}.' if env else f'Pass `auth` to connect to {service}.'
        )
    return auth


def per_run_connection(
    auth: Callable[[RunContext[AgentDepsT]], str | None],
    connect: Callable[[str], AbstractToolset[AgentDepsT]],
    *,
    id: str | None,
    expects: str = 'an API key or token',
) -> DynamicToolset[AgentDepsT]:
    """Call `auth` at the start of each run and `connect` with the token it returns; `None` or `''` means no tools.

    `auth` is bound when the toolset is built, as a string `auth` is; per-run state comes from its `ctx`.
    Registered once under a fixed `id`, as durable execution requires. `expects` names the token in the error raised
    when `auth` returns `'oauth'`.
    """

    def connect_for_run(ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        token = auth(ctx)
        if token == 'oauth':
            # FastMCP reads 'oauth' as "log in through a browser", which would hang a server run.
            raise UserError(f"The `auth` function must return {expects}, not 'oauth'.")
        return connect(token) if token else None

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
