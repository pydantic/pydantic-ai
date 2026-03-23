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
    from pydantic_ai.toolsets.fastmcp import FastMCPToolset


@dataclass(init=False)
class FastMCP(BuiltinOrLocalTool[AgentDepsT]):
    """FastMCP server capability.

    Uses the model's builtin MCP server support when a URL is available,
    connecting via FastMCP locally when it isn't.

    Accepts the same broad range of server inputs as
    [`FastMCPToolset`][pydantic_ai.toolsets.fastmcp.FastMCPToolset] — a FastMCP
    `Client`, transport, `FastMCP` server, URL string, filesystem `Path`,
    `MCPConfig` dict, etc.

    When `server` is an HTTP(S) URL (or `url` is provided explicitly), the
    builtin [`MCPServerTool`][pydantic_ai.builtin_tools.MCPServerTool] is
    registered so that models with native MCP support can connect directly.
    For non-URL servers (in-process, stdio, etc.) only the local
    `FastMCPToolset` is used.
    """

    server: Any
    """The MCP server to connect to via FastMCP.

    Accepts any input that [`FastMCPToolset`][pydantic_ai.toolsets.fastmcp.FastMCPToolset]
    supports: a FastMCP `Client`, `ClientTransport`, `FastMCP` server,
    `FastMCP1Server`, `AnyUrl`, `Path`, `MCPConfig`, `dict`, or `str` URL.
    """

    url: str | None
    """URL for the builtin MCPServerTool. Auto-derived from `server` when it's an HTTP(S) URL string."""

    id: str | None
    """Unique identifier. Defaults to a slug derived from the URL when available."""

    authorization_token: str | None
    """Authorization header value for MCP server requests.

    Passed to the builtin MCPServerTool and merged into headers for the local
    FastMCPToolset when `server` is a URL.
    """

    headers: dict[str, str] | None
    """HTTP headers for MCP server requests.

    Passed to the builtin MCPServerTool and to the local FastMCPToolset
    transport when `server` is a URL.
    """

    allowed_tools: list[str] | None
    """Filter to only these tools. Applied to both builtin and local."""

    description: str | None
    """Description of the MCP server. Builtin-only; ignored by local tools."""

    def __init__(
        self,
        server: Any,
        *,
        builtin: MCPServerTool | AgentBuiltinTool[AgentDepsT] | bool = True,
        local: FastMCPToolset[Any]
        | Tool[Any]
        | AbstractToolset[Any]
        | Callable[..., Any]
        | Literal[False]
        | None = None,
        url: str | None = None,
        id: str | None = None,
        authorization_token: str | None = None,
        headers: dict[str, str] | None = None,
        allowed_tools: list[str] | None = None,
        description: str | None = None,
    ) -> None:
        self.server = server
        # Auto-derive URL from server when it's an HTTP(S) URL string
        if url is None and isinstance(server, str) and server.startswith(('http://', 'https://')):
            url = server
        self.url = url
        # Disable builtin when no URL is available (can't create MCPServerTool)
        if builtin is True and self.url is None:
            builtin = False
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
        if self.url:
            # Include hostname to avoid collisions (e.g. two /sse URLs on different hosts)
            parsed = urlparse(self.url)
            path = parsed.path.rstrip('/')
            slug = path.split('/')[-1] if path else ''
            host = parsed.hostname or ''
            return f'{host}-{slug}' if slug else host or self.url
        return 'fastmcp'

    def _default_builtin(self) -> MCPServerTool | None:
        if self.url is None:
            return None
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

    def _default_local(self) -> AbstractToolset[Any] | None:
        from pydantic_ai.toolsets.fastmcp import FastMCPToolset

        server = self.server

        # When server is a URL and we have auth/headers, construct transport with headers
        if (
            isinstance(server, str)
            and server.startswith(('http://', 'https://'))
            and (self.authorization_token or self.headers)
        ):
            local_headers = dict(self.headers or {})
            if self.authorization_token:
                local_headers['Authorization'] = self.authorization_token

            if server.endswith('/sse'):
                from fastmcp.client import SSETransport

                server = SSETransport(server, headers=local_headers)
            else:
                from fastmcp.client.transports import StreamableHttpTransport

                server = StreamableHttpTransport(server, headers=local_headers)

        return FastMCPToolset(server)

    def get_toolset(self) -> AbstractToolset[AgentDepsT] | None:
        toolset = super().get_toolset()
        if toolset is not None and self.allowed_tools is not None:
            allowed = set(self.allowed_tools)
            return toolset.filtered(lambda _ctx, tool_def: tool_def.name in allowed)
        return toolset
