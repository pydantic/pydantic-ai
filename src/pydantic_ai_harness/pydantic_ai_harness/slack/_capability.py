"""Slack hosted MCP capability."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset, DynamicToolset

from pydantic_ai_harness._mcp import credential, is_read_only, one_connection

try:
    from pydantic_ai.mcp import MCPToolset, MCPToolsetClient
except ImportError as exc:  # pragma: no cover
    raise ImportError('Install Slack support with: uv add "pydantic-ai-harness[slack]"') from exc


_ID = 'slack'


@dataclass(kw_only=True)
class Slack(AbstractCapability[AgentDepsT]):
    """Give an agent Slack's hosted tools, acting as the connected user."""

    id: str | None = _ID
    """Stable capability and toolset ID, so `defer_loading=True` needs none.

    One `Slack` is one connection to one account, like `StackOne`'s linked account. Two sharing this `id` are
    one connection stated twice when they agree, and an error when they differ; give each its own `id` to keep both.
    """
    description: str | None = 'Use Slack messages, channels, and canvases.'
    auth: str | Callable[[RunContext[AgentDepsT]], str | None] | None = field(default=None, repr=False)
    """A Slack user token or a function of the run context that returns one.

    Unset, it uses `SLACK_USER_TOKEN`. A function never does: if it returns `None` or `''`, that run has no Slack tools.
    """
    read_only: bool = False
    """Expose only tools the server marks read-only; unmarked tools are omitted."""
    include_instructions: bool = True
    """Forward the server's instructions to the agent."""
    client: MCPToolsetClient | None = field(default=None, repr=False)
    """Your own MCP client or transport, for full control of the connection. It cannot be combined with `auth`."""

    def __post_init__(self) -> None:
        if self.client is not None and self.auth is not None:
            raise UserError('`client` owns the connection, so it cannot be combined with `auth`.')

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Two under one `id` are one connection stated twice; two that disagree raise rather than merge."""
        return one_connection(capabilities)

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Build the Slack connection and optional read-only selection."""
        id = self.id or _ID
        if self.client is not None:
            toolset: AbstractToolset[AgentDepsT] = MCPToolset(
                self.client, id=id, include_instructions=self.include_instructions
            )
        elif callable(self.auth):
            # Registered once under a fixed `id`, as durable execution requires; filled per run.
            toolset = DynamicToolset(self._connect_for_run, per_run_step=False, id=id)
        else:
            toolset = self._connect(self.auth)
        if self.read_only:
            return toolset.filtered(lambda _ctx, tool: is_read_only(tool))
        return toolset

    def _connect_for_run(self, ctx: RunContext[AgentDepsT]) -> MCPToolset[AgentDepsT] | None:
        auth = self.auth(ctx) if callable(self.auth) else self.auth
        if auth == 'oauth':
            # FastMCP reads 'oauth' as "log in through a browser", which would hang a server run.
            raise UserError("The `auth` function must return an API key or token, not 'oauth'.")
        return self._connect(auth) if auth else None

    def _connect(self, auth: str | None) -> MCPToolset[AgentDepsT]:
        return MCPToolset(
            'https://mcp.slack.com/mcp',
            id=self.id or _ID,
            auth=credential(auth, env='SLACK_USER_TOKEN', service='Slack'),
            headers=None,
            include_instructions=self.include_instructions,
        )
