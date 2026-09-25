"""Grain hosted MCP capability.

Provider contract, verified 2026-09-24:

- `https://api.grain.com/_/mcp` is the Streamable HTTP endpoint; there is one host, with no regional variants.
  An unauthenticated POST answers 401 with `WWW-Authenticate: Bearer resource_metadata=...`, and an unknown bearer
  token answers 401 `invalid_token`.
- Authentication is a Grain OAuth access token, sent as a bearer token. The protected-resource metadata names
  `https://api.grain.com` as the authorization server, which supports PKCE and dynamic client registration.
- The server has tools that change data: Grain's 2026-05-13 release added creating clips and tagging meetings.
  `read_only` keeps only the tools the server marks read-only.

Sources: https://grain.com/release-note/06-18-2025, https://developers.grain.com/mcp/claude-setup.html,
https://developers.grain.com/mcp/tools-and-usage.html,
https://grain.com/release-note/grain-mcp-and-api-updates, and https://grain.com/.well-known/oauth-protected-resource.
Re-check by sending an unauthenticated POST to the endpoint.
"""

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
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'MCP support is required for the Grain capability. Install it with: uv add "pydantic-ai-harness[grain]"'
    ) from _import_error

_GRAIN_MCP_URL = 'https://api.grain.com/_/mcp'
_ID = 'grain'
_DEFAULT_DESCRIPTION = 'Search and read Grain meetings: transcripts, notes, people, companies, and deals.'


@dataclass(kw_only=True)
class Grain(AbstractCapability[AgentDepsT]):
    """Let an agent search and read Grain meetings, transcripts, and notes.

    Set `GRAIN_ACCESS_TOKEN` or pass a Grain access token as `auth`. The agent can then read every meeting the
    user can see in Grain.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import Grain

    agent = Agent('openai:gpt-5.6-sol', capabilities=[Grain()])
    ```
    """

    id: str | None = _ID
    """Stable capability and toolset ID, so `defer_loading=True` needs none.

    One `Grain` is one connection to one account, like `StackOne`'s linked account. Two sharing this `id` are
    one connection stated twice when they agree, and an error when they differ; give each its own `id` to keep both.
    """

    description: str | None = _DEFAULT_DESCRIPTION
    """Routing description used when the capability is loaded on demand."""

    auth: str | Callable[[RunContext[AgentDepsT]], str | None] | None = field(default=None, repr=False)
    """A Grain access token, `'oauth'` to sign in through the browser locally, or a function of the run context that returns a token.

    Unset, it uses `GRAIN_ACCESS_TOKEN`. A function never does: if it returns `None` or `''`, that run has no Grain tools.
    """

    read_only: bool = False
    """Give the agent only the tools the server labels as read-only. A tool without that label is left out."""

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
        """Return the Grain MCP tools."""
        id = self.id or _ID
        if self.client is not None:
            toolset: AbstractToolset[AgentDepsT] = MCPToolset(
                self.client, id=id, include_instructions=self.include_instructions
            )
        elif callable(self.auth):
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
            _GRAIN_MCP_URL,
            id=self.id or _ID,
            auth=credential(auth, env='GRAIN_ACCESS_TOKEN', service='Grain'),
            include_instructions=self.include_instructions,
        )
