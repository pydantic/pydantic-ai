"""LogfireMCP hosted MCP capability."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset, DynamicToolset

from pydantic_ai_harness._mcp import credential, is_read_only, one_connection

try:
    from pydantic_ai.mcp import MCPToolset, MCPToolsetClient
except ImportError as exc:  # pragma: no cover
    raise ImportError('Install LogfireMCP support with: uv add "pydantic-ai-harness[logfire-mcp]"') from exc

from pydantic_ai.agent.abstract import AgentInstructions
from pydantic_ai.messages import ModelRequest
from pydantic_ai.tools import RunContext

LOGFIRE_US_MCP_URL = 'https://logfire-us.pydantic.dev/mcp'
LOGFIRE_EU_MCP_URL = 'https://logfire-eu.pydantic.dev/mcp'

_INSTRUCTIONS = (
    'Timestamps in tool schemas and examples, and project creation timestamps, are examples or metadata rather than '
    'the current time. Query transport bounds apply in addition to SQL time predicates and default to a short '
    'window, so widen them explicitly when needed. Create a Logfire link only when the user asks for one.'
)


_ID = 'logfire-mcp'


@dataclass(kw_only=True)
class LogfireMCP(AbstractCapability[AgentDepsT]):
    """Query Logfire telemetry and manage observability resources through its hosted tools."""

    id: str | None = _ID
    """Stable capability and toolset ID, so `defer_loading=True` needs none.

    One `LogfireMCP` is one connection to one account, like `StackOne`'s linked account. Two sharing this `id` are
    one connection stated twice when they agree, and an error when they differ; give each its own `id` to keep both.
    """
    description: str | None = 'Query Logfire telemetry and manage observability resources.'
    auth: str | Callable[[RunContext[AgentDepsT]], str | None] | None = field(default=None, repr=False)
    """A Logfire API key, `'oauth'` to sign in through the browser locally, or a function of the run context that returns a key.

    Unset, it uses `LOGFIRE_API_KEY`. A function never does: if it returns `None` or `''`, that run has no Logfire tools.
    """
    read_only: bool = False
    """Expose only tools the server marks read-only; unmarked tools are omitted."""
    include_instructions: bool = True
    """Include server instructions, query guidance, and the current UTC time."""
    client: MCPToolsetClient | None = field(default=None, repr=False)
    """Your own MCP client or transport, for full control of the connection. It cannot be combined with `auth` or `url`."""
    url: str = LOGFIRE_US_MCP_URL
    """Hosted US, hosted EU, or self-hosted MCP endpoint."""

    def __post_init__(self) -> None:
        if self.client is not None and (self.auth is not None or self.url != LOGFIRE_US_MCP_URL):
            raise UserError('`client` owns the connection, so it cannot be combined with `auth` or `url`.')

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Two under one `id` are one connection stated twice; two that disagree raise rather than merge."""
        return one_connection(capabilities)

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Build the LogfireMCP connection and optional read-only selection."""
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
            self.url,
            id=self.id or _ID,
            auth=credential(auth, env='LOGFIRE_API_KEY', service='Logfire'),
            headers=None,
            include_instructions=self.include_instructions,
        )

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Return query guidance and the current UTC time."""
        if not self.include_instructions:
            return None
        return [_INSTRUCTIONS, self._current_utc]

    def _current_utc(self, ctx: RunContext[AgentDepsT]) -> str | None:
        # The run stamps the request it is about to send, which is the last one, so this needs no clock read
        # of its own (Temporal's workflow sandbox rejects those). Older requests may come from saved history.
        for message in reversed(ctx.messages):
            if isinstance(message, ModelRequest) and message.timestamp:
                return f'Current UTC time is `{message.timestamp.isoformat(timespec="seconds")}`.'
        return None
