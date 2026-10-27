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
(`https://day.ai/.well-known/oauth-protected-resource/api/mcp`,
`https://day.ai/.well-known/oauth-authorization-server`).
Re-check by fetching both metadata documents and sending an unauthenticated `initialize` POST to the endpoint.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai_harness._mcp import credential, one_connection, per_run_connection, prefixed_toolset

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

    Each `id` is one connection to one Day AI account. Two `DayAI` with the same `id` and the same settings are
    merged into one, and two that differ raise an error; give each account its own `id`.
    """

    description: str | None = _DEFAULT_DESCRIPTION
    """Routing description used when the capability is loaded on demand."""

    tool_prefix: str | None = 'day_ai'
    """Prepended with `_` to the server's tool names, as in `day_ai_<tool>`, so they don't collide with other tools.

    Give two instances distinct prefixes to use both. `None` keeps the names the server gives its tools.
    """

    auth: str | Callable[[RunContext[AgentDepsT]], str | None | Awaitable[str | None]] | None = field(
        default=None, repr=False
    )
    """A Day AI access token, `'oauth'` to sign in through the browser locally, or a sync or async function
    of the run context that returns a token.

    Unset, it uses `DAY_AI_ACCESS_TOKEN`. A function never does: if it returns `None` or `''`, that run has no
    Day AI tools.
    """

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
        """Return the Day AI MCP tools."""
        id = self.id or _ID
        if self.client is not None:
            toolset: AbstractToolset[AgentDepsT] = MCPToolset(
                self.client, id=id, include_instructions=self.include_instructions
            )
        elif callable(self.auth):
            toolset = per_run_connection(self.auth, self._connect, id=id, expects='an access token')
        else:
            toolset = self._connect(self.auth)
        return prefixed_toolset(toolset, self.tool_prefix)

    def _connect(self, auth: str | None) -> MCPToolset[AgentDepsT]:
        return MCPToolset(
            _DAY_AI_MCP_URL,
            id=self.id or _ID,
            auth=credential(auth, env='DAY_AI_ACCESS_TOKEN', service='Day AI'),
            include_instructions=self.include_instructions,
        )
