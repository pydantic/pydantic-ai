"""Linear hosted MCP capability."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai_harness._mcp import credential, one_connection, per_run_connection, prefixed_toolset, read_only_toolset

try:
    from pydantic_ai.mcp import MCPToolset, MCPToolsetClient
except ImportError as exc:  # pragma: no cover
    raise ImportError('Install Linear support with: uv add "pydantic-ai-harness[linear]"') from exc


_ID = 'linear'


@dataclass(kw_only=True)
class Linear(AbstractCapability[AgentDepsT]):
    """Give an agent the tools of Linear's hosted MCP server, with the permissions of the connected user."""

    id: str | None = _ID
    """Stable capability ID, also used for the Linear MCP toolset, so `defer_loading=True` needs none.

    Each `Linear` is one connection to one Linear account. Two with the same `id` and settings are one connection, and
    two with the same `id` but different settings raise. Give each its own `id` to keep both. With `None` or `''`
    the toolset still uses `linear`, but the capability is not combined with others.
    """
    description: str | None = 'Use Linear issues, projects, and teams.'
    tool_prefix: str | None = 'linear'
    """Prepended with `_` to the server's tool names, as in `linear_<tool>`, so they don't collide with other tools.

    Give two instances distinct prefixes to use both. `None` keeps the names the server gives its tools.
    """
    auth: str | Callable[[RunContext[AgentDepsT]], str | None | Awaitable[str | None]] | None = field(
        default=None, repr=False
    )
    """A Linear API key or token, or a sync or async function of the run context that returns one.

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
        if self.client is not None and self.auth:
            raise UserError('`client` owns the connection, so it cannot be combined with `auth`.')

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Two under one `id` are one connection stated twice; two that disagree raise rather than merge."""
        return one_connection(capabilities)

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Return the Linear tools."""
        if self.client is not None:
            toolset: AbstractToolset[AgentDepsT] = MCPToolset(
                self.client, id=self.id or _ID, include_instructions=self.include_instructions
            )
            if self.read_only:
                toolset = read_only_toolset(toolset)
        elif callable(self.auth):
            toolset = per_run_connection(self.auth, self._connect, id=self.id or _ID)
        else:
            toolset = self._connect(self.auth)
        return prefixed_toolset(toolset, self.tool_prefix)

    def _connect(self, auth: str | None) -> MCPToolset[AgentDepsT]:
        return MCPToolset(
            'https://mcp.linear.app/mcp/readonly' if self.read_only else 'https://mcp.linear.app/mcp',
            id=self.id or _ID,
            auth=credential(auth, env='LINEAR_ACCESS_TOKEN', service='Linear'),
            include_instructions=self.include_instructions,
        )
