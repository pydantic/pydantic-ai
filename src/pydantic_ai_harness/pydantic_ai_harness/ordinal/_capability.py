"""Ordinal hosted MCP capability.

Provider contract, verified 2026-09-11:

- `https://app.tryordinal.com/mcp` is the Streamable HTTP endpoint.
- Authentication is an Ordinal OAuth access token, sent as a bearer token. The previous API-key server at
  `https://app.tryordinal.com/api/mcp` is deprecated and is not used here.

Source: https://docs.tryordinal.com/mcp/introduction
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
        'MCP support is required for the Ordinal capability. Install it with: uv add "pydantic-ai-harness[ordinal]"'
    ) from _import_error

_ORDINAL_MCP_URL = 'https://app.tryordinal.com/mcp'
_ID = 'ordinal'
_DEFAULT_DESCRIPTION = 'Work inside an Ordinal workspace: draft, schedule, and analyze social posts.'


@dataclass(kw_only=True)
class Ordinal(AbstractCapability[AgentDepsT]):
    """Let an agent draft, schedule, and analyze social posts in Ordinal.

    Set `ORDINAL_ACCESS_TOKEN` or pass an Ordinal access token as `auth`. The agent can then reach every
    workspace the user belongs to.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import Ordinal

    agent = Agent('openai:gpt-5.6-sol', capabilities=[Ordinal()])
    ```
    """

    id: str | None = _ID
    """Stable capability and toolset ID, so `defer_loading=True` needs none.

    One `Ordinal` is one connection to one account, like `StackOne`'s linked account. Two sharing this `id` are
    one connection stated twice when they agree, and an error when they differ; give each its own `id` to keep both.
    """

    description: str | None = _DEFAULT_DESCRIPTION
    """Routing description used when the capability is loaded on demand."""

    auth: str | Callable[[RunContext[AgentDepsT]], str | None] | None = field(default=None, repr=False)
    """An Ordinal access token, `'oauth'` to sign in through the browser locally, or a function of the run context that returns a token.

    Unset, it uses `ORDINAL_ACCESS_TOKEN`. A function never does: if it returns `None` or `''`, that run has no Ordinal tools.
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
        """Return the Ordinal MCP tools."""
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
            _ORDINAL_MCP_URL,
            id=self.id or _ID,
            auth=credential(auth, env='ORDINAL_ACCESS_TOKEN', service='Ordinal'),
            include_instructions=self.include_instructions,
        )
