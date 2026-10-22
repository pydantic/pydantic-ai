"""GitHub hosted MCP capability."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai_harness._combine import one_per_id
from pydantic_ai_harness._mcp import credential, per_run_connection, read_only_toolset
from pydantic_ai_harness._warn import HarnessDeprecationWarning

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
    """The `https://` MCP server URL, for example a GitHub Enterprise Cloud endpoint. `http://` is deprecated."""
    toolsets: list[str] | None = None
    """GitHub tool groups to offer, such as `'repos'`. `None` keeps the server's defaults."""

    def __post_init__(self) -> None:
        if self.client is not None and (
            self.auth is not None or self.url != GITHUB_MCP_URL or self.toolsets is not None or self.lockdown
        ):
            raise UserError(
                '`client` owns the connection, so it cannot be combined with `auth`, `url`, `toolsets`, or `lockdown`.'
            )
        parts = urlsplit(self.url)
        if parts.scheme not in ('https', 'http') or not parts.hostname:
            raise UserError(f'`url` must be an `https://` URL; got {self.url!r}. Pass `client` to connect another way.')
        if parts.scheme == 'http':
            # The token travels in a header, so plain `http://` sends it unencrypted, even to a loopback host when an
            # `HTTP_PROXY` is set. It worked before, so it warns now and is to raise in a future release.
            warnings.warn(
                f'`GitHub(url={self.url!r})` sends the GitHub token unencrypted. Use an `https://` URL, or pass '
                '`client` to connect another way; a plain `http://` URL will raise in a future release.',
                HarnessDeprecationWarning,
                stacklevel=3,
            )
        # GitHub reads an empty toolsets header as its defaults, which include write tools.
        if self.toolsets == []:
            raise UserError('`toolsets` must name at least one tool group; use `None` for the defaults.')
        # The header joins groups with commas, so one entry holding a comma would enable several groups.
        if self.toolsets is not None and any(not group.strip() or ',' in group for group in self.toolsets):
            raise UserError('Each `toolsets` entry must name one tool group, such as `repos`.')

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Two under one `id` are one connection stated twice; two that disagree raise rather than merge."""
        return one_per_id(capabilities)

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Return the GitHub tools."""
        toolset: AbstractToolset[AgentDepsT]
        if self.client is not None:
            toolset = MCPToolset(self.client, id=self.id or _ID, include_instructions=self.include_instructions)
        elif callable(self.auth):
            toolset = per_run_connection(self.auth, self._connect, id=self.id or _ID, expects='a GitHub token')
        else:
            toolset = self._connect(self.auth)
        # The hosted connection also sends `X-MCP-Readonly`; filtering here keeps write tools out when the header is
        # dropped or ignored on the way.
        # The hosted connection kept its `id` before this filter was added, so the filtered one keeps it too.
        return read_only_toolset(toolset, keep_id=self.client is None) if self.read_only else toolset

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
            id=self.id or _ID,
            auth=token,
            headers=headers,
            include_instructions=self.include_instructions,
        )
