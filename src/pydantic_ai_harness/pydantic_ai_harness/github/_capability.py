"""GitHub hosted MCP capability."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset, DynamicToolset
from pydantic_ai_harness._mcp import credential, one_connection, read_only_toolset

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

    Each `GitHub` is one connection to one GitHub account. Two with the same `id` and settings are one connection, and
    two with the same `id` but different settings raise. Give each its own `id` to keep both. `None` or `''` means
    the default, `github`.
    """
    description: str | None = 'Use GitHub repositories, issues, and pull requests.'
    auth: str | Callable[[RunContext[AgentDepsT]], str | None] | None = field(default=None, repr=False)
    """A GitHub token or a function of the run context that returns one.

    Unset, it uses `GITHUB_TOKEN`. A function never does: if it returns `None` or `''`, that run has no GitHub tools.
    `'oauth'` raises an error, since GitHub does not support automatic OAuth client registration.
    """
    read_only: bool = False
    """Offer only the tools the server marks as read-only. The hosted connection also asks for GitHub's read-only mode."""
    lockdown: bool = False
    """Ask GitHub to withhold public-repository content written by users without push access.

    GitHub describes this as a best-effort content filter, not a security boundary: other tools can still return it.
    """
    include_instructions: bool = True
    """Forward the server's instructions to the agent."""
    client: MCPToolsetClient | None = field(default=None, repr=False)
    """Your own MCP client or transport, for full control of the connection.

    It cannot be combined with `auth`, `url`, `toolsets`, or `lockdown`.
    """
    url: str = GITHUB_MCP_URL
    """The `https://` MCP server URL, for example a GitHub Enterprise Cloud endpoint."""
    toolsets: list[str] | None = None
    """GitHub tool groups to offer, such as `'repos'`. `None` keeps the server's defaults."""

    def __post_init__(self) -> None:
        # The toolset always needs an id, so normalize here and let `combine` see the id the toolset uses.
        self.id = self.id or _ID
        if self.client is not None and (
            self.auth is not None or self.url != GITHUB_MCP_URL or self.toolsets is not None or self.lockdown
        ):
            raise UserError(
                '`client` owns the connection, so it cannot be combined with `auth`, `url`, `toolsets`, or `lockdown`.'
            )
        parts = urlsplit(self.url)
        if parts.scheme != 'https' or not parts.hostname:
            # The token travels in a header, so plain `http://` would send it unencrypted, even to a loopback host
            # when an `HTTP_PROXY` is set.
            raise UserError(f'`url` must be an `https://` URL; got {self.url!r}. Pass `client` to connect another way.')
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
        toolset: AbstractToolset[AgentDepsT]
        if self.client is not None:
            toolset = MCPToolset(self.client, id=self.id, include_instructions=self.include_instructions)
        elif callable(auth := self.auth):

            def connect_for_run(ctx: RunContext[AgentDepsT]) -> MCPToolset[AgentDepsT] | None:
                token = auth(ctx)
                return self._connect(token) if token else None

            # Registered once under a fixed `id`, as durable execution requires; filled per run.
            toolset = DynamicToolset(connect_for_run, per_run_step=False, id=self.id)
        else:
            toolset = self._connect(auth)
        # The hosted connection also sends `X-MCP-Readonly`; filtering here keeps write tools out when the header is
        # dropped or ignored on the way.
        return read_only_toolset(toolset) if self.read_only else toolset

    def _connect(self, auth: str | None) -> MCPToolset[AgentDepsT]:
        token = credential(auth, env='GITHUB_TOKEN', service='GitHub')
        if token == 'oauth':
            # FastMCP reads 'oauth' as a browser sign-in that registers its own client, which GitHub does not allow.
            raise UserError(
                "GitHub needs a token, not 'oauth': GitHub does not support automatic OAuth client registration."
            )
        headers = {'X-MCP-Readonly': 'true'} if self.read_only else {}
        if self.lockdown:
            headers['X-MCP-Lockdown'] = 'true'
        if self.toolsets is not None:
            headers['X-MCP-Toolsets'] = ','.join(self.toolsets)
        return MCPToolset(
            self.url,
            id=self.id,
            auth=token,
            headers=headers,
            include_instructions=self.include_instructions,
        )
