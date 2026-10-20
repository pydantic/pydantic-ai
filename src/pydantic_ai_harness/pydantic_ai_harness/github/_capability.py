"""GitHub hosted MCP capability."""

from __future__ import annotations

import re
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
# One tool group name, such as `repos` or `code_security`: no commas, spaces, or control characters.
_GROUP = re.compile(r'[\w-]+', re.ASCII)


@dataclass(kw_only=True)
class GitHub(AbstractCapability[AgentDepsT]):
    """Use GitHub's hosted tools with the permissions of the connected credential."""

    id: str | None = _ID
    """Stable capability and toolset ID, so `defer_loading=True` needs none.

    Each `GitHub` is one connection to one GitHub account. Two with the same `id` and settings are one connection, and
    two with the same `id` but different settings raise. Give each its own `id` to keep both. `None` or `''` means
    the default, `github`.
    """
    description: str | None = None
    """Describes the capability when the agent loads it on demand. Unset, it says whether the tools can make changes."""
    auth: str | Callable[[RunContext[AgentDepsT]], str | None] | None = field(default=None, repr=False)
    """A GitHub token or a function of the run context that returns one.

    Unset, it uses `GITHUB_TOKEN`. A function never does: if it returns `None` or `''`, that run has no GitHub tools.
    `'oauth'` raises an error, since GitHub does not support automatic OAuth client registration.
    """
    read_only: bool = False
    """Offer only the tools the server marks as read-only. The hosted connection also asks for GitHub's read-only mode."""
    include_instructions: bool = True
    """Forward the server's instructions to the agent."""
    client: MCPToolsetClient | None = field(default=None, repr=False)
    """Your own MCP client or transport, for full control of the connection.

    It cannot be combined with `auth`, `url`, or `toolsets`.
    """
    url: str = GITHUB_MCP_URL
    """The `https://` MCP server URL, for example a GitHub Enterprise Cloud endpoint."""
    toolsets: list[str] | None = None
    """GitHub tool groups to offer, such as `'repos'`. `None` keeps the server's defaults."""

    def __post_init__(self) -> None:
        # The toolset always needs an id, so normalize here and let `combine` see the id the toolset uses.
        self.id = self.id or _ID
        if self.client is not None and (
            self.auth is not None or self.url != GITHUB_MCP_URL or self.toolsets is not None
        ):
            raise UserError('`client` owns the connection, so it cannot be combined with `auth`, `url`, or `toolsets`.')
        parts = urlsplit(self.url)
        if parts.scheme != 'https' or not parts.hostname:
            # The token travels in a header, so a plain `http://` URL would send it unencrypted.
            raise UserError('`url` must be an `https://` URL; pass `client` to connect another way.')
        if self.toolsets is not None:
            # A copy, so later changes to the caller's list cannot bypass the checks below.
            self.toolsets = list(self.toolsets)
            # GitHub reads an empty toolsets header as its defaults, which include write tools.
            if not self.toolsets:
                raise UserError('`toolsets` must name at least one tool group; use `None` for the defaults.')
            # The header joins groups with commas, so one entry holding a comma would enable several groups.
            if not all(_GROUP.fullmatch(group) for group in self.toolsets):
                raise UserError('Each `toolsets` entry must name one tool group, such as `repos`.')

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Two under one `id` are one connection stated twice; two that disagree raise rather than merge."""
        return one_connection(capabilities)

    def get_description(self) -> str:
        """Return `description`, or one that says whether the tools can make changes when it is unset."""
        if self.description is not None:
            return self.description
        return 'Read GitHub resources.' if self.read_only else 'Read and change GitHub resources.'

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
        if self.toolsets is not None:
            headers['X-MCP-Toolsets'] = ','.join(self.toolsets)
        return MCPToolset(
            self.url,
            id=self.id,
            auth=token,
            headers=headers,
            include_instructions=self.include_instructions,
        )
