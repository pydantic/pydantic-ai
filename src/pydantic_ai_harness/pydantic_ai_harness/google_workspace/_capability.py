"""Google Workspace hosted MCP capability."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Literal

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset, CombinedToolset
from pydantic_ai_harness._mcp import credential, per_run_connection, read_only_toolset

try:
    from pydantic_ai.mcp import MCPToolset
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'MCP support is required for the Google Workspace capability. '
        'Install it with: uv add "pydantic-ai-harness[google-workspace]"'
    ) from _import_error

GoogleWorkspaceService = Literal['gmail', 'drive', 'docs', 'sheets', 'slides', 'calendar', 'chat', 'people']
"""A Google Workspace product Google serves over MCP."""

_MCP_URLS: dict[str, str] = {
    'gmail': 'https://gmailmcp.googleapis.com/mcp/v1',
    'drive': 'https://drivemcp.googleapis.com/mcp/v1',
    'docs': 'https://docsmcp.googleapis.com/mcp/v1',
    'sheets': 'https://sheetsmcp.googleapis.com/mcp/v1',
    'slides': 'https://slidesmcp.googleapis.com/mcp/v1',
    'calendar': 'https://calendarmcp.googleapis.com/mcp/v1',
    'chat': 'https://chatmcp.googleapis.com/mcp/v1',
    'people': 'https://people.googleapis.com/mcp/v1',
}

_DEFAULT_DESCRIPTION = 'Use Gmail, Calendar, Drive, and the other Google Workspace products.'


@dataclass
class GoogleWorkspace(AbstractCapability[AgentDepsT]):
    """Give an agent the tools of Google's hosted Workspace MCP servers for the selected products.

    This includes tools that send, change, and delete; the token's scopes decide what they can reach.
    """

    services: GoogleWorkspaceService | Sequence[GoogleWorkspaceService]
    """Workspace products to expose, such as `'gmail'` or `['gmail', 'calendar']`."""

    _: KW_ONLY

    id: str | None = None
    """Stable capability and toolset ID, `google-workspace-<products>` with the products sorted when not given.

    Instances whose products overlap register the same tool names, so wrap one in `PrefixTools`. When their products
    are the same, also give it a distinct `id`, since their default ones collide.

    The id is derived once, at construction: `dataclasses.replace` with different `services` keeps the old one unless
    you also pass `id=None`.
    """

    description: str | None = _DEFAULT_DESCRIPTION
    """Describes the capability when the agent loads it on demand. Left at its default, it names the selected products."""

    auth: str | Callable[[RunContext[AgentDepsT]], str | None | Awaitable[str | None]] | None = field(
        default=None, repr=False
    )
    """A Google access token or a sync or async function of the run context that returns one.

    Unset, it uses `GOOGLE_ACCESS_TOKEN`. A function never does: if it returns `None` or `''`, that run has no Google
    Workspace tools. `'oauth'` raises an error, since Google does not support automatic OAuth client registration.
    """

    read_only: bool = False
    """Keep only the tools Google marks as read-only."""

    include_instructions: bool = True
    """Pass the servers' own instructions to the agent."""

    def __post_init__(self) -> None:
        """Normalize `services` to a tuple of products that have an endpoint, and derive the `id` from them."""
        self.services = (self.services,) if isinstance(self.services, str) else tuple(dict.fromkeys(self.services))
        if not self.services:
            raise UserError('Google Workspace needs at least one service.')
        for service in self.services:
            if service not in _MCP_URLS:
                raise UserError(f'Unknown Google Workspace service {service!r}; expected one of {sorted(_MCP_URLS)}.')
        if self.id is None:
            self.id = f'google-workspace-{"-".join(sorted(self.services))}'

    def get_description(self) -> str | None:
        """Return `description`, or one naming the selected products while it is left at its default."""
        if self.description != _DEFAULT_DESCRIPTION:
            return self.description
        return f'Use these Google Workspace products: {", ".join(service.title() for service in self.services)}.'

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Return the tools for the selected products, with names prefixed by product."""
        toolset = (
            per_run_connection(self.auth, self._connect, id=self.id, expects='an access token')
            if callable(self.auth)
            else self._connect(self.auth)
        )
        return read_only_toolset(toolset) if self.read_only else toolset

    def _connect(self, auth: str | None) -> AbstractToolset[AgentDepsT]:
        auth = credential(auth, env='GOOGLE_ACCESS_TOKEN', service='Google Workspace')
        if auth == 'oauth':
            # FastMCP reads 'oauth' as a browser sign-in that registers its own client, which Google does not allow.
            raise UserError(
                "Google Workspace needs an access token, not 'oauth': "
                'Google does not support automatic OAuth client registration.'
            )
        return CombinedToolset(
            [
                MCPToolset[AgentDepsT](
                    _MCP_URLS[service],
                    id=f'{self.id}-{service}',
                    auth=auth,
                    include_instructions=self.include_instructions,
                ).prefixed(service)
                for service in self.services
            ]
        )
