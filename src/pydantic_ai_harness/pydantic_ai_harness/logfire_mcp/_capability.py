"""LogfireMCP hosted MCP capability."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from pydantic_ai._instructions import SourcedInstruction, append_on_change_by_default
from pydantic_ai.agent.abstract import AgentInstructions
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import InstructionPart, ModelRequest
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset, DynamicToolset
from pydantic_ai_harness._mcp import credential, one_connection, read_only_toolset

try:
    from pydantic_ai.mcp import MCPToolset, MCPToolsetClient
except ImportError as exc:  # pragma: no cover
    raise ImportError('Install LogfireMCP support with: uv add "pydantic-ai-harness[logfire-mcp]"') from exc

LOGFIRE_US_MCP_URL = 'https://logfire-us.pydantic.dev/mcp'
LOGFIRE_EU_MCP_URL = 'https://logfire-eu.pydantic.dev/mcp'

_INSTRUCTIONS = (
    'Timestamps in tool schemas and examples, and project creation timestamps, are examples or metadata rather than '
    'the current time. Query transport bounds apply in addition to SQL time predicates and default to a short '
    'window, so widen them explicitly when needed. Create a Logfire link only when the user asks for one. '
    "Logfire tool names here start with `logfire_`, which the server's own text leaves out: `query_run` is "
    '`logfire_query_run`.'
)

_TOOL_PREFIX = 'logfire'
"""Every tool is named `logfire_<server tool name>`, so Logfire's tools never collide with another toolset's."""


_ID = 'logfire-mcp'


@dataclass(kw_only=True)
class LogfireMCP(AbstractCapability[AgentDepsT]):
    """Query Logfire telemetry and manage observability resources through its hosted tools."""

    id: str | None = _ID
    """Stable capability and toolset ID, so `defer_loading=True` needs none.

    Each `LogfireMCP` is one connection to one Logfire account. Two with the same `id` and settings are one
    connection, and two with the same `id` but different settings raise. Give each its own `id` to keep both.
    `None` or `''` means the default, `logfire-mcp`.
    """
    description: str | None = 'Query Logfire telemetry and manage observability resources.'
    auth: str | Callable[[RunContext[AgentDepsT]], str | None] | None = field(default=None, repr=False)
    """A Logfire API key, `'oauth'` to sign in through the browser locally, or a function of the run context that returns a key.

    Unset, it uses `LOGFIRE_API_KEY`. A function never does: if it returns `None` or `''`, that run has no Logfire tools.
    """
    read_only: bool = False
    """Expose only tools the server marks read-only; unmarked tools are omitted."""
    include_instructions: bool = True
    """Include server instructions, query guidance, and the current UTC hour."""
    client: MCPToolsetClient | None = field(default=None, repr=False)
    """Your own MCP client or transport, for full control of the connection.

    It cannot be combined with a non-empty `auth` or a `url` other than the default.
    """
    url: str = LOGFIRE_US_MCP_URL
    """Hosted US, hosted EU, or self-hosted MCP endpoint. The API key is sent to it as a bearer token."""

    def __post_init__(self) -> None:
        # The toolset always needs an id, and `''` auth means unset, so normalize both here and let
        # `combine` compare what the connection uses.
        self.id = self.id or _ID
        if self.auth == '':
            self.auth = None
        if self.client is not None and (self.auth is not None or self.url != LOGFIRE_US_MCP_URL):
            raise UserError(
                '`client` owns the connection, so it cannot be combined with `auth` or a `url` other than the default.'
            )

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Two under one `id` are one connection stated twice; two that disagree raise rather than merge."""
        return one_connection(capabilities)

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Build the LogfireMCP connection and optional read-only selection, with tool names prefixed `logfire_`."""
        if self.client is not None:
            toolset: AbstractToolset[AgentDepsT] = MCPToolset(
                self.client, id=self.id, include_instructions=self.include_instructions
            )
        elif callable(self.auth):
            # Registered once under a fixed `id`, as durable execution requires; filled per run.
            toolset = DynamicToolset(self._connect_for_run, per_run_step=False, id=self.id)
        else:
            toolset = self._connect(self.auth)
        if self.read_only:
            toolset = read_only_toolset(toolset)
        return toolset.prefixed(_TOOL_PREFIX)

    def _connect_for_run(self, ctx: RunContext[AgentDepsT]) -> MCPToolset[AgentDepsT] | None:
        auth = self.auth(ctx) if callable(self.auth) else self.auth
        if auth == 'oauth':
            # FastMCP reads 'oauth' as "log in through a browser", which would hang a server run.
            raise UserError("The `auth` function must return an API key or token, not 'oauth'.")
        return self._connect(auth) if auth else None

    def _connect(self, auth: str | None) -> MCPToolset[AgentDepsT]:
        return MCPToolset(
            self.url,
            id=self.id,
            auth=credential(auth, env='LOGFIRE_API_KEY', service='Logfire'),
            include_instructions=self.include_instructions,
        )

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Return query guidance and the current UTC hour."""
        if not self.include_instructions:
            return None
        return [InstructionPart(_INSTRUCTIONS, name='guidance'), self._current_utc]

    def _collect_instructions(self) -> list[SourcedInstruction[AgentDepsT]]:
        # The hour moves on during a conversation, so each new one is appended to the history rather than
        # rewriting the instructions, which would miss the prompt cache for everything after them.
        return append_on_change_by_default(super()._collect_instructions(), key='logfire_mcp')

    def _current_utc(self, ctx: RunContext[AgentDepsT]) -> str | None:
        # The run stamps the request it is about to send, which is the last one, so this needs no clock read
        # of its own (Temporal's workflow sandbox rejects those). Older requests may come from saved history.
        # Each new value is appended to the history, so a finer stamp would add one on every request.
        for message in reversed(ctx.messages):
            if isinstance(message, ModelRequest) and message.timestamp:
                hour = message.timestamp.replace(minute=0, second=0, microsecond=0)
                return f'The current UTC time is within the hour starting `{hour.isoformat(timespec="minutes")}`.'
        return None
