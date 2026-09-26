"""Pylon hosted MCP capability.

Provider contract, verified 2026-09-24:

- `https://mcp.usepylon.com` is the Streamable HTTP endpoint (stateless, no server-side sessions). There is one
  host; `/mcp` returns 404.
- Authentication is a Pylon OAuth access token, sent as a bearer token. Pylon REST API tokens are not accepted:
  "The Pylon MCP server only supports OAuth authentication. API key authentication is not currently available."
- The server's protected-resource metadata names `https://o.auth.usepylon.com` as its authorization server, which
  supports dynamic client registration, PKCE (S256), and refresh tokens (`offline_access`).
- Only Member and Admin users with the `MCP Access` role can sign in; Viewer and Integration users cannot.
- The server has tools that change issues and accounts. `read_only` keeps only the tools the server marks read-only;
  its annotations could not be inspected without a Pylon account.

Sources: https://docs.usepylon.com/pylon-docs/integrations/pylon-mcp,
https://support.usepylon.com/articles/2407390554-connecting-to-the-pylon-mcp-server

Re-check with an unauthenticated `POST https://mcp.usepylon.com` (401 with a `resource_metadata` challenge) and
`GET https://mcp.usepylon.com/.well-known/oauth-protected-resource` and `/.well-known/oauth-authorization-server`.
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
        'MCP support is required for the Pylon capability. Install it with: uv add "pydantic-ai-harness[pylon]"'
    ) from _import_error

_PYLON_MCP_URL = 'https://mcp.usepylon.com'
_ID = 'pylon'
_DEFAULT_DESCRIPTION = (
    'Work in Pylon: search, read, create, and update support issues, look up and update accounts, and look up contacts.'
)


@dataclass(kw_only=True)
class Pylon(AbstractCapability[AgentDepsT]):
    """Let an agent manage support issues, look up and update accounts, and look up contacts in Pylon.

    Set `PYLON_ACCESS_TOKEN` or pass a Pylon access token as `auth`. The agent then acts as that user and
    sees the same data they see in the Pylon dashboard.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import Pylon

    agent = Agent('openai:gpt-5.6-sol', capabilities=[Pylon()])
    ```
    """

    id: str | None = _ID
    """Stable capability and toolset ID, so `defer_loading=True` needs none.

    One `Pylon` is one connection to one account, like `StackOne`'s linked account. Two sharing this `id` are
    one connection stated twice when they agree, and an error when they differ; give each its own `id` to keep both.
    """

    description: str | None = _DEFAULT_DESCRIPTION
    """Routing description used when the capability is loaded on demand."""

    auth: str | Callable[[RunContext[AgentDepsT]], str | None] | None = field(default=None, repr=False)
    """A Pylon access token, `'oauth'` to sign in through the browser locally, or a function of the run context that returns a token.

    Unset, it uses `PYLON_ACCESS_TOKEN`. A function never does: if it returns `None` or `''`, that run has no Pylon tools.
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
        """Return the Pylon MCP tools."""
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
            _PYLON_MCP_URL,
            id=self.id or _ID,
            auth=credential(auth, env='PYLON_ACCESS_TOKEN', service='Pylon'),
            include_instructions=self.include_instructions,
        )
