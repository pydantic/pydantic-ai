"""Linear hosted MCP capability."""

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
    raise ImportError('Install Linear support with: uv add "pydantic-ai-harness[linear]"') from exc


_ID = 'linear'


@dataclass(kw_only=True)
class Linear(AbstractCapability[AgentDepsT]):
    """Give an agent the tools of Linear's hosted MCP server, with the permissions of the connected user."""

    id: str | None = _ID
    """Stable capability and toolset ID, so `defer_loading=True` needs none.

    One `Linear` is one connection to one account, like `StackOne`'s linked account. Two sharing this `id` are
    one connection stated twice when they agree, and an error when they differ; give each its own `id` to keep both.
    """
    description: str | None = 'Use Linear issues, projects, and teams.'
    auth: str | Callable[[RunContext[AgentDepsT]], str | None] | None = field(default=None, repr=False)
    """A Linear API key or token, or a function of the run context that returns one.

    `'oauth'` signs in through the browser on your own machine.

    Unset, it uses `LINEAR_ACCESS_TOKEN`. A function never does: if it returns `None` or `''`, that run has no
    Linear tools.
    """
    read_only: bool = False
    """Use Linear's read-only endpoint. With `client`, keep only the tools the server marks as read-only."""
    include_instructions: bool = True
    """Pass the server's own instructions to the agent."""
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
        """Return the Linear tools."""
        id = self.id or _ID
        if self.client is not None:
            toolset: AbstractToolset[AgentDepsT] = MCPToolset(
                self.client, id=id, include_instructions=self.include_instructions
            )
            if self.read_only:
                return toolset.filtered(lambda _ctx, tool: is_read_only(tool))
            return toolset
        if callable(self.auth):
            return DynamicToolset(self._connect_for_run, per_run_step=False, id=id)
        return self._connect(self.auth)

    def _connect_for_run(self, ctx: RunContext[AgentDepsT]) -> MCPToolset[AgentDepsT] | None:
        auth = self.auth(ctx) if callable(self.auth) else self.auth
        if auth == 'oauth':
            # FastMCP reads 'oauth' as "log in through a browser", which would hang a server run.
            raise UserError("The `auth` function must return an API key or token, not 'oauth'.")
        return self._connect(auth) if auth else None

    def _connect(self, auth: str | None) -> MCPToolset[AgentDepsT]:
        return MCPToolset(
            'https://mcp.linear.app/mcp/readonly' if self.read_only else 'https://mcp.linear.app/mcp',
            id=self.id or _ID,
            auth=credential(auth, env='LINEAR_ACCESS_TOKEN', service='Linear'),
            headers=None,
            include_instructions=self.include_instructions,
        )
