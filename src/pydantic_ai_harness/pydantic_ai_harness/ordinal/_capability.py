"""Ordinal hosted MCP capability.

Provider contract, verified 2026-09-11:

- `https://app.tryordinal.com/mcp` is the Streamable HTTP endpoint.
- Authentication is an Ordinal OAuth access token, sent as a bearer token. The previous API-key server at
  `https://app.tryordinal.com/api/mcp` is deprecated and is not used here.

Source: https://docs.tryordinal.com/mcp/introduction
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai_harness._mcp import credential, one_connection, per_run_connection, prefixed_toolset

try:
    from pydantic_ai.mcp import MCPToolset, MCPToolsetClient
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'MCP support is required for the Ordinal capability. Install it with: uv add "pydantic-ai-harness[ordinal]"'
    ) from _import_error

_ORDINAL_MCP_URL = 'https://app.tryordinal.com/mcp'
_ID = 'ordinal'
_DEFAULT_DESCRIPTION = 'Work inside an Ordinal workspace: draft, schedule, and analyze social posts.'


@dataclass(kw_only=True)
class Ordinal(AbstractCapability[AgentDepsT]):
    """Let an agent draft, schedule, and analyze social posts in Ordinal.

    Set `ORDINAL_ACCESS_TOKEN` or pass an Ordinal access token as `auth`. The agent can then reach every
    workspace the user belongs to.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import Ordinal

    agent = Agent('openai:gpt-5.6-sol', capabilities=[Ordinal()])
    ```
    """

    id: str | None = _ID
    """Stable capability and toolset ID, so `defer_loading=True` needs none.

    One `Ordinal` is one connection to one account, like `StackOne`'s linked account. Two sharing this `id` are
    one connection stated twice when they agree, and an error when they differ; give each its own `id` to keep both.
    """

    description: str | None = _DEFAULT_DESCRIPTION
    """Routing description used when the capability is loaded on demand."""

    tool_prefix: str | None = None
    """Prepended with `_` to the server's tool names, as in `work_ordinal_get_workspace_context`.

    Unset, the server's names are kept, since they already start with the service, as in
    `ordinal_get_workspace_context`. Give two instances distinct prefixes to use both.
    """

    auth: str | Callable[[RunContext[AgentDepsT]], str | None | Awaitable[str | None]] | None = field(
        default=None, repr=False
    )
    """An Ordinal access token, `'oauth'` to sign in through the browser locally, or a sync or async function
    of the run context that returns a token.

    Unset, it uses `ORDINAL_ACCESS_TOKEN`. A function never does: if it returns `None` or `''`, that run has no Ordinal tools.
    """

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
        """Return the Ordinal MCP tools."""
        id = self.id or _ID
        if self.client is not None:
            toolset: AbstractToolset[AgentDepsT] = MCPToolset(
                self.client, id=id, include_instructions=self.include_instructions
            )
        elif callable(self.auth):
            toolset = per_run_connection(self.auth, self._connect, id=id)
        else:
            toolset = self._connect(self.auth)
        return prefixed_toolset(toolset, self.tool_prefix)

    def _connect(self, auth: str | None) -> MCPToolset[AgentDepsT]:
        return MCPToolset(
            _ORDINAL_MCP_URL,
            id=self.id or _ID,
            auth=credential(auth, env='ORDINAL_ACCESS_TOKEN', service='Ordinal'),
            include_instructions=self.include_instructions,
        )
