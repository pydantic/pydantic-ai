"""PostHog hosted MCP capability.

Provider contract, verified 2026-09-24:

- `https://mcp.posthog.com/mcp` is the Streamable HTTP endpoint for both US and EU Cloud. Its edge worker resolves
  the region from the token, so one URL serves every account; an unauthenticated `initialize` returns 401 with
  OAuth protected-resource metadata.
- Authentication is a PostHog personal API key (the "MCP Server" preset) or an OAuth access token, sent as a bearer
  token. The authorization server at `https://oauth.posthog.com` supports dynamic client registration.
- `?features=a,b` limits the server to those feature groups; an empty value means every group.
  `x-posthog-read-only: true` makes the server drop every tool it does not annotate `readOnlyHint`.
- Without an explicit mode the server wraps its tools behind one `posthog` tool driven by commands; this capability
  keeps that default.

Sources: https://posthog.com/docs/model-context-protocol,
https://github.com/PostHog/posthog/blob/master/services/mcp/README.md, and
`services/mcp/src/lib/request-properties.ts` and `services/mcp/src/proxy.ts` in the same repository.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from urllib.parse import urlencode

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset, DynamicToolset

from pydantic_ai_harness._mcp import credential, is_read_only, one_connection

try:
    from pydantic_ai.mcp import MCPToolset, MCPToolsetClient
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'MCP support is required for the PostHog capability. Install it with: uv add "pydantic-ai-harness[posthog]"'
    ) from _import_error

_POSTHOG_MCP_URL = 'https://mcp.posthog.com/mcp'
_ID = 'posthog'
_DEFAULT_DESCRIPTION = 'Query PostHog analytics and manage feature flags, experiments, and dashboards.'


@dataclass(kw_only=True)
class PostHog(AbstractCapability[AgentDepsT]):
    """Let an agent query PostHog product analytics and manage its resources.

    Set `POSTHOG_PERSONAL_API_KEY` or pass a PostHog personal API key as `auth`. The agent can then do whatever
    the key's scopes allow.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import PostHog

    agent = Agent('openai:gpt-5.6-sol', capabilities=[PostHog()])
    ```
    """

    id: str | None = _ID
    """Stable capability and toolset ID, so `defer_loading=True` needs none.

    One `PostHog` is one connection to one account, like `StackOne`'s linked account. Two sharing this `id` are
    one connection stated twice when they agree, and an error when they differ; give each its own `id` to keep both.
    """

    description: str | None = _DEFAULT_DESCRIPTION
    """Routing description used when the capability is loaded on demand."""

    auth: str | Callable[[RunContext[AgentDepsT]], str | None] | None = field(default=None, repr=False)
    """A PostHog personal API key or OAuth access token, `'oauth'` to sign in through the browser locally, or a function of the run context that returns one.

    Unset, it uses `POSTHOG_PERSONAL_API_KEY`. A function never does: if it returns `None` or `''`, that run has no PostHog tools.
    """

    read_only: bool = False
    """Ask the server for read-only tools. With a custom `client`, keep only the tools the server marks as read-only.

    PostHog's default mode serves one `posthog` tool that is not marked read-only, so with a custom `client` this
    leaves no PostHog tools; send the `x-posthog-read-only: true` header from the client instead.
    """

    features: str | Sequence[str] | None = None
    """PostHog feature groups to offer, such as `'flags'` or `['flags', 'insights']`. `None` offers every group."""

    include_instructions: bool = True
    """Pass the server's own instructions to the agent."""

    client: MCPToolsetClient | None = field(default=None, repr=False)
    """Your own MCP client or transport, for full control of the connection.

    It cannot be combined with `auth` or `features`.
    """

    def __post_init__(self) -> None:
        if self.client is not None and (self.auth is not None or self.features is not None):
            raise UserError('`client` owns the connection, so it cannot be combined with `auth` or `features`.')
        if self.features is None:
            return
        # A snapshot, so the groups validated here are the groups sent, whatever happens to the caller's list.
        self.features = (self.features,) if isinstance(self.features, str) else tuple(self.features)
        # PostHog reads an empty features list as every group, which includes write tools.
        if not self.features:
            raise UserError('`features` must name at least one feature group; use `None` for every group.')
        # The server splits the query parameter on commas without trimming, so a comma would enable several groups and
        # padding would match none.
        if any(not group or group != group.strip() or ',' in group for group in self.features):
            raise UserError('Each `features` entry must name one feature group, such as `flags`.')

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Two under one `id` are one connection stated twice; two that disagree raise rather than merge."""
        return one_connection(capabilities)

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Return the PostHog MCP tools."""
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
        url = _POSTHOG_MCP_URL
        if self.features is not None:
            url += '?' + urlencode({'features': ','.join(self.features)})
        return MCPToolset(
            url,
            id=self.id or _ID,
            auth=credential(auth, env='POSTHOG_PERSONAL_API_KEY', service='PostHog'),
            headers={'x-posthog-read-only': 'true'} if self.read_only else {},
            include_instructions=self.include_instructions,
        )
