"""GitHub hosted MCP capability."""

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
except ImportError as exc:  # pragma: no cover
    raise ImportError('Install GitHub support with: uv add "pydantic-ai-harness[github]"') from exc


GITHUB_MCP_URL = 'https://api.githubcopilot.com/mcp/'
_ID = 'github'


@dataclass(kw_only=True)
class GitHub(AbstractCapability[AgentDepsT]):
    """Use GitHub's hosted tools with the permissions of the connected credential."""

    id: str | None = _ID
    """Stable capability and toolset ID, so `defer_loading=True` needs none.

    One `GitHub` is one connection to one account, like `StackOne`'s linked account. Two sharing this `id` are
    one connection stated twice when they agree, and an error when they differ; give each its own `id` to keep both.
    """
    description: str | None = 'Read and change GitHub resources.'
    auth: str | Callable[[RunContext[AgentDepsT]], str | None] | None = field(default=None, repr=False)
    """A GitHub token or a function of the run context that returns one.

    Unset, it uses `GITHUB_TOKEN`. A function never does: if it returns `None` or `''`, that run has no GitHub tools.
    """
    read_only: bool = False
    """Offer only read tools. With a custom `client`, keep only the tools the server marks as read-only."""
    include_instructions: bool = True
    """Forward the server's instructions to the agent."""
    client: MCPToolsetClient | None = field(default=None, repr=False)
    """Your own MCP client or transport, for full control of the connection.

    It cannot be combined with `auth`, `url`, or `toolsets`.
    """
    url: str = GITHUB_MCP_URL
    """The MCP server URL, for example a GitHub Enterprise Cloud endpoint."""
    toolsets: list[str] | None = None
    """GitHub tool groups to offer, such as `'repos'`. `None` keeps the server's defaults."""

    def __post_init__(self) -> None:
        if self.client is not None and (
            self.auth is not None or self.url != GITHUB_MCP_URL or self.toolsets is not None
        ):
            raise UserError('`client` owns the connection, so it cannot be combined with `auth`, `url`, or `toolsets`.')
        # GitHub reads an empty toolsets header as its defaults, which include write tools.
        if self.toolsets == []:
            raise UserError('`toolsets` must name at least one tool group; use `None` for the defaults.')
        # The header joins groups with commas, so one entry holding a comma would enable several groups.
        if self.toolsets is not None and any(not group.strip() or ',' in group for group in self.toolsets):
            raise UserError('Each `toolsets` entry must name one tool group, such as `repos`.')

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Two under one `id` are one connection stated twice; two that disagree raise rather than merge."""
        return one_connection(capabilities)

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Return the GitHub tools."""
        id = self.id or _ID
        if self.client is not None:
            toolset: AbstractToolset[AgentDepsT] = MCPToolset(
                self.client, id=id, include_instructions=self.include_instructions
            )
            if self.read_only:
                return toolset.filtered(lambda _ctx, tool: is_read_only(tool))
            return toolset
        if callable(self.auth):
            # Registered once under a fixed `id`, as durable execution requires; filled per run.
            return DynamicToolset(self._connect_for_run, per_run_step=False, id=id)
        return self._connect(self.auth)

    def _connect_for_run(self, ctx: RunContext[AgentDepsT]) -> MCPToolset[AgentDepsT] | None:
        auth = self.auth(ctx) if callable(self.auth) else self.auth
        if auth == 'oauth':
            # FastMCP reads 'oauth' as "log in through a browser", which would hang a server run.
            raise UserError("The `auth` function must return an API key or token, not 'oauth'.")
        return self._connect(auth) if auth else None

    def _connect(self, auth: str | None) -> MCPToolset[AgentDepsT]:
        headers = {'X-MCP-Readonly': 'true'} if self.read_only else {}
        if self.toolsets is not None:
            headers['X-MCP-Toolsets'] = ','.join(self.toolsets)
        return MCPToolset(
            self.url,
            id=self.id or _ID,
            auth=credential(auth, env='GITHUB_TOKEN', service='GitHub'),
            headers=headers,
            include_instructions=self.include_instructions,
        )
