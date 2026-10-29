"""Notion hosted MCP capability."""

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
except ImportError as exc:  # pragma: no cover
    raise ImportError('Install Notion support with: uv add "pydantic-ai-harness[notion]"') from exc


_ID = 'notion'


@dataclass(kw_only=True)
class Notion(AbstractCapability[AgentDepsT]):
    """Give the agent the tools of Notion's hosted MCP server, with the permissions of the connected user."""

    id: str | None = _ID
    """Stable capability and toolset ID, so `defer_loading=True` needs none.

    One `Notion` is one connection to one account, like `StackOne`'s linked account. Two sharing this `id` are
    one connection stated twice when they agree, and an error when they differ; give each its own `id` to keep both.
    """
    description: str | None = 'Search and change Notion workspace content.'
    tool_prefix: str | None = None
    """Prepended with `_` to the server's tool names, as in `work_notion-search`.

    Unset, the server's names are kept, since they already start with the service, as in `notion-search`.
    Give two instances distinct prefixes to use both.
    """
    auth: str | Callable[[RunContext[AgentDepsT]], str | None | Awaitable[str | None]] | None = field(
        default=None, repr=False
    )
    """A Notion OAuth access token, `'oauth'` to sign in through the browser locally, or a sync or async function
    of the run context that returns a token.

    Unset, it uses `NOTION_ACCESS_TOKEN`. A function never does: if it returns `None` or `''`, that run has no
    Notion tools.
    """
    read_only: bool = False
    """Keep only the tools the server marks as read-only."""
    include_instructions: bool = True
    """Forward the server's instructions to the agent."""
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
        """Return the Notion MCP toolset."""
        id = self.id or _ID
        if self.client is not None:
            toolset: AbstractToolset[AgentDepsT] = MCPToolset(
                self.client, id=id, include_instructions=self.include_instructions
            )
        elif callable(self.auth):
            toolset = per_run_connection(self.auth, self._connect, id=id)
        else:
            toolset = self._connect(self.auth)
        if self.read_only:
            toolset = read_only_toolset(toolset)
        return prefixed_toolset(toolset, self.tool_prefix)

    def _connect(self, auth: str | None) -> MCPToolset[AgentDepsT]:
        return MCPToolset(
            'https://mcp.notion.com/mcp',
            id=self.id or _ID,
            auth=credential(auth, env='NOTION_ACCESS_TOKEN', service='Notion'),
            headers=None,
            include_instructions=self.include_instructions,
        )
