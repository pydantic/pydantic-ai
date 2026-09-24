"""Google Workspace hosted MCP capability."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Literal

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset, CombinedToolset, DynamicToolset

from pydantic_ai_harness._mcp import credential, is_read_only

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
    """Stable capability and toolset ID, derived from `services` when not given.

    The products are what decide this capability's tools, so they are what identify it -- the same way `StackOne`
    is identified by its linked account. Deriving it rather than fixing it to `'google-workspace'` is what lets one
    agent reach two different sets of products: their ids differ, so they stay two capabilities. Two for the same
    products are a mistake, and collide.
    """

    description: str | None = _DEFAULT_DESCRIPTION
    """Describes the capability when the agent loads it on demand."""

    auth: str | Callable[[RunContext[AgentDepsT]], str | None] | None = field(default=None, repr=False)
    """A Google access token or a function of the run context that returns one.

    Unset, it uses `GOOGLE_ACCESS_TOKEN`. A function never does: if it returns `None` or `''`, that run has no Google
    Workspace tools.
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
        self.id = self._derived_id()

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Return the tools for the selected products, with names prefixed by product."""
        toolset = (
            DynamicToolset(self._connect_for_run, per_run_step=False, id=self._derived_id())
            if callable(self.auth)
            else self._connect(self.auth)
        )
        return toolset.filtered(lambda _ctx, tool_def: is_read_only(tool_def)) if self.read_only else toolset

    def _derived_id(self) -> str:
        """This capability's `id`, falling back to the one the products name."""
        return self.id if self.id is not None else f'google-workspace-{"-".join(sorted(self.services))}'

    def _connect_for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        auth = self.auth(ctx) if callable(self.auth) else self.auth
        if auth == 'oauth':
            # FastMCP reads 'oauth' as "log in through a browser", which would hang a server run.
            raise UserError("The `auth` function must return an API key or token, not 'oauth'.")
        return self._connect(auth) if auth else None

    def _connect(self, auth: str | None) -> AbstractToolset[AgentDepsT]:
        auth = credential(auth, env='GOOGLE_ACCESS_TOKEN', service='Google Workspace')
        prefix = self._derived_id()
        return CombinedToolset(
            [
                MCPToolset[AgentDepsT](
                    _MCP_URLS[service],
                    id=f'{prefix}-{service}',
                    auth=auth,
                    include_instructions=self.include_instructions,
                ).prefixed(service)
                for service in self.services
            ]
        )
