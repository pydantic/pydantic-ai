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

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai_harness._mcp import credential, one_connection, per_run_connection, prefixed_toolset, read_only_toolset

try:
    from pydantic_ai.mcp import MCPToolset, MCPToolsetClient
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'MCP support is required for the Grain capability. Install it with: uv add "pydantic-ai-harness[grain]"'
    ) from _import_error

_GRAIN_MCP_URL = 'https://api.grain.com/_/mcp'
_ID = 'grain'
_DEFAULT_DESCRIPTION = 'Use Grain meetings, transcripts, notes, clips, people, companies, and deals.'


@dataclass(kw_only=True)
class Grain(AbstractCapability[AgentDepsT]):
    """Let an agent search and read Grain meetings, transcripts, and notes, and create clips and tag meetings.

    Set `GRAIN_ACCESS_TOKEN` or pass a Grain access token as `auth`. The agent can then reach every meeting the
    user can see in Grain. Pass `read_only=True` to keep only the tools Grain marks read-only.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import Grain

    agent = Agent('openai:gpt-5.6-sol', capabilities=[Grain()])
    ```
    """

    id: str | None = _ID
    """Stable capability and toolset ID, so `defer_loading=True` needs none.

    Each `Grain` is one connection to one Grain account. Two with the same `id` and settings are one connection, and
    two with the same `id` but different settings raise. Give each its own `id` to keep both. `None` or `''` means
    the default, `grain`.
    """

    description: str | None = _DEFAULT_DESCRIPTION
    """Routing description used when the capability is loaded on demand."""

    tool_prefix: str | None = 'grain'
    """Prepended with `_` to the server's tool names, as in `grain_<tool>`, so they don't collide with other tools.

    Give two instances distinct prefixes to use both. `None` keeps the names the server gives its tools.
    """

    auth: str | Callable[[RunContext[AgentDepsT]], str | None | Awaitable[str | None]] | None = field(
        default=None, repr=False
    )
    """A Grain access token, `'oauth'` to sign in through the browser locally, or a sync or async function
    of the run context that returns a token.

    Unset, it uses `GRAIN_ACCESS_TOKEN`. A function never does: if it returns `None` or `''`, that run has no Grain tools.
    """

    read_only: bool = False
    """Give the agent only the tools the server labels as read-only. A tool without that label is left out."""

    include_instructions: bool = True
    """Pass the server's own instructions to the agent."""

    client: MCPToolsetClient | None = field(default=None, repr=False)
    """Your own MCP client or transport, for full control of the connection. It cannot be combined with `auth`."""

    def __post_init__(self) -> None:
        # The toolset always needs an id, so normalize here and let `combine` see the id the toolset uses.
        self.id = self.id or _ID
        if self.client is not None and self.auth:
            raise UserError('`client` owns the connection, so it cannot be combined with `auth`.')

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Two under one `id` are one connection stated twice; two that disagree raise rather than merge."""
        return one_connection(capabilities)

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Return the Grain MCP tools."""
        if self.client is not None:
            toolset: AbstractToolset[AgentDepsT] = MCPToolset(
                self.client, id=self.id, include_instructions=self.include_instructions
            )
        elif callable(self.auth):
            toolset = per_run_connection(self.auth, self._connect, id=self.id)
        else:
            toolset = self._connect(self.auth)
        if self.read_only:
            toolset = read_only_toolset(toolset)
        return prefixed_toolset(toolset, self.tool_prefix)

    def _connect(self, auth: str | None) -> MCPToolset[AgentDepsT]:
        return MCPToolset(
            _GRAIN_MCP_URL,
            id=self.id,
            auth=credential(auth, env='GRAIN_ACCESS_TOKEN', service='Grain'),
            include_instructions=self.include_instructions,
        )
