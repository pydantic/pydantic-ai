from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

from pydantic_ai.builtin_tools import MCPServerTool
from pydantic_ai.tools import AgentBuiltinTool, AgentDepsT, Tool
from pydantic_ai.toolsets import AbstractToolset

from .builtin_or_local import BuiltinOrLocalTool

if TYPE_CHECKING:
    from pydantic_ai.mcp import MCPServer
    from pydantic_ai.toolsets.fastmcp import FastMCPToolset


def resolve_mcp_server_id(explicit_id: str | None, url: str | None) -> str | None:
    """Resolve an MCP server ID from an explicit ID or URL.

    Returns the explicit ID if set, otherwise derives a slug from the URL's
    hostname and path. Returns None if neither is available.
    """
    if explicit_id:
        return explicit_id
    if url:
        # Include hostname to avoid collisions (e.g. two /sse URLs on different hosts)
        parsed = urlparse(url)
        path = parsed.path.rstrip('/')
        slug = path.split('/')[-1] if path else ''
        host = parsed.hostname or ''
        return f'{host}-{slug}' if slug else host or url
    return None


def filter_allowed_tools(
    toolset: AbstractToolset[AgentDepsT] | None, allowed_tools: list[str] | None
) -> AbstractToolset[AgentDepsT] | None:
    """Wrap a toolset in a filter for `allowed_tools`, if set."""
    if toolset is not None and allowed_tools is not None:
        allowed = set(allowed_tools)
        return toolset.filtered(lambda _ctx, tool_def: tool_def.name in allowed)
    return toolset


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
        *,
        url: str,
        builtin: MCPServerTool | AgentBuiltinTool[AgentDepsT] | bool = True,
        local: MCPServer | FastMCPToolset[Any] | Callable[..., Any] | Literal[False] | None = None,
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
        return resolve_mcp_server_id(self.id, self.url) or self.url

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

    def _default_local(self) -> Tool[Any] | AbstractToolset[Any] | None:
        # Merge authorization_token into headers for local connection
        local_headers = dict(self.headers or {})
        if self.authorization_token:
            local_headers['Authorization'] = self.authorization_token

        # Transport detection matching _mcp_server_discriminator() in pydantic_ai.mcp
        if self.url.endswith('/sse'):
            from pydantic_ai.mcp import MCPServerSSE

            return MCPServerSSE(self.url, headers=local_headers or None)

        from pydantic_ai.mcp import MCPServerStreamableHTTP

        return MCPServerStreamableHTTP(self.url, headers=local_headers or None)

    def get_toolset(self) -> AbstractToolset[AgentDepsT] | None:
        return filter_allowed_tools(super().get_toolset(), self.allowed_tools)
