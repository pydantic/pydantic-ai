"""Day AI hosted MCP capability.

Provider contract, verified 2026-09-24:

- `https://day.ai/api/mcp` is the Streamable HTTP endpoint; there is one host, with no regional variants.
- Authentication is a Day AI OAuth access token, sent as a bearer token. There is no API key: an unauthenticated
  request gets a 401 whose `WWW-Authenticate` header points at the protected-resource metadata, and the
  authorization server at `https://day.ai` supports dynamic client registration with PKCE.
- Using the server needs a paid Day AI Agent tier; the tier's tools and limits apply.
- The server does not mark tools read-only (it advertises MCP protocol `2024-11-05`, which predates tool
  annotations), so there is no `read_only` option.

Sources: https://day.ai/mcp, https://github.com/day-ai/day-ai-sdk, and the endpoint itself
(`https://day.ai/.well-known/oauth-protected-resource/api/mcp`, `https://day.ai/.well-known/oauth-authorization-server`).
Re-check by fetching both metadata documents and sending an unauthenticated `initialize` POST to the endpoint.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset, DynamicToolset

from pydantic_ai_harness._mcp import credential, one_connection

try:
    from pydantic_ai.mcp import MCPToolset, MCPToolsetClient
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'MCP support is required for the DayAI capability. Install it with: uv add "pydantic-ai-harness[day-ai]"'
    ) from _import_error

_DAY_AI_MCP_URL = 'https://day.ai/api/mcp'
_ID = 'day_ai'
_DEFAULT_DESCRIPTION = (
    'Work inside a Day AI workspace: search and update CRM records, read meeting context, and draft emails.'
)


@dataclass(kw_only=True)
class DayAI(AbstractCapability[AgentDepsT]):
    """Let an agent search and update the Day AI CRM.

    Set `DAY_AI_ACCESS_TOKEN` or pass a Day AI access token as `auth`. The agent can then use every tool the
    user's Day AI Agent tier and workspace role allow.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import DayAI

    agent = Agent('openai:gpt-5.6-sol', capabilities=[DayAI()])
    ```
    """

    id: str | None = _ID
    """Stable capability and toolset ID, so `defer_loading=True` needs none.

    One `DayAI` is one connection to one account, like `StackOne`'s linked account. Two sharing this `id` are
    one connection stated twice when they agree, and an error when they differ; give each its own `id` to keep both.
    """

    description: str | None = _DEFAULT_DESCRIPTION
    """Routing description used when the capability is loaded on demand."""

    auth: str | Callable[[RunContext[AgentDepsT]], str | None] | None = field(default=None, repr=False)
    """A Day AI access token, `'oauth'` to sign in through the browser locally, or a function of the run context that returns a token.

    Unset, it uses `DAY_AI_ACCESS_TOKEN`. A function never does: if it returns `None` or `''`, that run has no Day AI tools.
    """

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
        """Return the Day AI MCP tools."""
        id = self.id or _ID
        if self.client is not None:
            return MCPToolset(self.client, id=id, include_instructions=self.include_instructions)
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
            _DAY_AI_MCP_URL,
            id=self.id or _ID,
            auth=credential(auth, env='DAY_AI_ACCESS_TOKEN', service='Day AI'),
            include_instructions=self.include_instructions,
        )
