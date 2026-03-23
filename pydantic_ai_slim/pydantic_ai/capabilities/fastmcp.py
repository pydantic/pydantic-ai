from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai.builtin_tools import MCPServerTool
from pydantic_ai.tools import AgentBuiltinTool, AgentDepsT, Tool
from pydantic_ai.toolsets import AbstractToolset

from .builtin_or_local import BuiltinOrLocalTool
from .mcp import filter_allowed_tools, resolve_mcp_server_id

if TYPE_CHECKING:
    from fastmcp.client import Client as FastMCPClient
    from fastmcp.client.transports import ClientTransport
    from fastmcp.mcp_config import MCPConfig
    from fastmcp.server import FastMCP as FastMCPServer
    from mcp.server.fastmcp import FastMCP as FastMCP1Server
    from pydantic import AnyUrl

    from pydantic_ai.toolsets.fastmcp import FastMCPToolset

    FastMCPServerInput = (
        FastMCPClient[Any]
        | ClientTransport
        | FastMCPServer
        | FastMCP1Server
        | AnyUrl
        | Path
        | MCPConfig
        | dict[str, Any]
        | str
    )


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
    For non-URL servers (in-process, stdio, etc.) the builtin is automatically
    disabled and only the local `FastMCPToolset` is used.
    """

    server: FastMCPServerInput
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
        server: FastMCPServerInput,
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
        return resolve_mcp_server_id(self.id, self.url) or 'fastmcp'

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

        server: Any = self.server

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

                server = SSETransport(server, headers=local_headers or None)
            else:
                from fastmcp.client.transports import StreamableHttpTransport

                server = StreamableHttpTransport(server, headers=local_headers or None)

        return FastMCPToolset(server)

    def get_toolset(self) -> AbstractToolset[AgentDepsT] | None:
        return filter_allowed_tools(super().get_toolset(), self.allowed_tools)
