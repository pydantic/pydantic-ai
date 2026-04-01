from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

from pydantic_ai.builtin_tools import MCPServerTool
from pydantic_ai.tools import AgentDepsT, RunContext, Tool
from pydantic_ai.toolsets import AbstractToolset

from .builtin_or_local import BuiltinOrLocalTool

try:
    from pydantic_ai.mcp import MCPServer
    from pydantic_ai.toolsets.fastmcp import FastMCPToolset
except ImportError:  # pragma: lax no cover
    if not TYPE_CHECKING:
        MCPServer = Any  # type: ignore[assignment,misc]
        FastMCPToolset = Any  # type: ignore[assignment,misc]


@dataclass(init=False)
class MCP(BuiltinOrLocalTool[AgentDepsT]):
    """MCP server capability.

    Uses the model's builtin MCP server support when available, connecting
    directly via HTTP when it isn't.
    """

    url: str
    """The URL of the MCP server."""

    id: str | None
    """Unique identifier for the MCP server. Defaults to a slug derived from the URL."""

    authorization_token: str | None
    """Authorization header value for MCP server requests. Passed to both builtin and local."""

    headers: dict[str, str] | None
    """HTTP headers for MCP server requests. Passed to both builtin and local."""

    allowed_tools: list[str] | None
    """Filter to only these tools. Applied to both builtin and local."""

    description: str | None
    """Description of the MCP server. Builtin-only; ignored by local tools."""

    def __init__(
        self,
        url: str,
        *,
        builtin: MCPServerTool
        | Callable[[RunContext[AgentDepsT]], Awaitable[MCPServerTool | None] | MCPServerTool | None]
        | bool = True,
        local: MCPServer | FastMCPToolset[AgentDepsT] | Callable[..., Any] | Literal[False] | None = None,
        id: str | None = None,
        authorization_token: str | None = None,
        headers: dict[str, str] | None = None,
        allowed_tools: list[str] | None = None,
        description: str | None = None,
    ) -> None:
        self.url = url
        self.builtin = builtin
        self.local = local
        self.id = id
        self.authorization_token = authorization_token
        self.headers = headers
        self.allowed_tools = allowed_tools
        self.description = description
        self.__post_init__()

    @cached_property
    def _resolved_id(self) -> str:
        if self.id:
            return self.id
        # Include hostname to avoid collisions (e.g. two /sse URLs on different hosts)
        parsed = urlparse(self.url)
        path = parsed.path.rstrip('/')
        slug = path.split('/')[-1] if path else ''
        host = parsed.hostname or ''
        return f'{host}-{slug}' if slug else host or self.url

    def _default_builtin(self) -> MCPServerTool:
        return MCPServerTool(
            id=self._resolved_id,
            url=self.url,
            authorization_token=self.authorization_token,
            headers=self.headers,
            allowed_tools=self.allowed_tools,
            description=self.description,
        )

    def _builtin_unique_id(self) -> str:
        return f'mcp_server:{self._resolved_id}'

    def _default_local(self) -> Tool[AgentDepsT] | AbstractToolset[AgentDepsT] | None:
        # Merge authorization_token into headers for local connection
        local_headers = dict(self.headers or {})
        if self.authorization_token:
            local_headers['Authorization'] = self.authorization_token

        # Transport detection matching _mcp_server_discriminator() in pydantic_ai.mcp
        if self.url.endswith('/sse'):
            from pydantic_ai.mcp import MCPServerSSE

            return MCPServerSSE(self.url, headers=local_headers or None, include_instructions=True)

        from pydantic_ai.mcp import MCPServerStreamableHTTP

        return MCPServerStreamableHTTP(self.url, headers=local_headers or None, include_instructions=True)

    def get_toolset(self) -> AbstractToolset[AgentDepsT] | None:
        toolset = super().get_toolset()
        if toolset is not None and self.allowed_tools is not None:
            allowed = set(self.allowed_tools)
            return toolset.filtered(lambda _ctx, tool_def: tool_def.name in allowed)
        return toolset
