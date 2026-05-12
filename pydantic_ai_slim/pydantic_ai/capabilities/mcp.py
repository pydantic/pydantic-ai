from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

from pydantic_ai._utils import install_deprecated_kwarg_alias
from pydantic_ai.native_tools import MCPServerTool
from pydantic_ai.tools import AgentDepsT, RunContext, Tool
from pydantic_ai.toolsets import AbstractToolset

from .native_or_local import NativeOrLocalTool

try:
    from pydantic_ai.mcp import MCPServer, MCPToolset, MCPToolsetClient
    from pydantic_ai.toolsets.fastmcp import FastMCPToolset  # pyright: ignore[reportDeprecated]
except ImportError:  # pragma: lax no cover
    if not TYPE_CHECKING:
        MCPServer = Any  # type: ignore[assignment,misc]
        MCPToolset = Any  # type: ignore[assignment,misc]
        MCPToolsetClient = Any  # type: ignore[assignment,misc]
        FastMCPToolset = Any  # type: ignore[assignment,misc]


@dataclass(init=False)
class MCP(NativeOrLocalTool[AgentDepsT]):
    """MCP server capability.

    Uses the model's native MCP server support when available, connecting
    directly via HTTP when it isn't.
    """

    url: str
    """The URL of the MCP server."""

    id: str | None
    """Unique identifier for the MCP server. Defaults to a slug derived from the URL."""

    authorization_token: str | None
    """Authorization header value for MCP server requests. Passed to both native and local."""

    headers: dict[str, str] | None
    """HTTP headers for MCP server requests. Passed to both native and local."""

    allowed_tools: list[str] | None
    """Filter to only these tools. Applied to both native and local."""

    description: str | None
    """Description of the MCP server. Native-only; ignored by local tools."""

    def __init__(
        self,
        url: str,
        *,
        native: MCPServerTool
        | Callable[[RunContext[AgentDepsT]], Awaitable[MCPServerTool | None] | MCPServerTool | None]
        | bool = True,
        local: MCPToolsetClient
        | MCPToolset[AgentDepsT]
        | MCPServer
        | FastMCPToolset[AgentDepsT]  # pyright: ignore[reportDeprecated]
        | Callable[..., Any]
        | Literal[False]
        | None = None,
        id: str | None = None,
        authorization_token: str | None = None,
        headers: dict[str, str] | None = None,
        allowed_tools: list[str] | None = None,
        description: str | None = None,
    ) -> None:
        self.url = url
        self.native = native
        # Wide `local=` inputs (URL, Path, transport, FastMCP server, pre-built `fastmcp.Client`,
        # etc.) are wrapped into an `MCPToolset` here. Pre-built toolsets, callables, `False`, and
        # `None` pass through to `NativeOrLocalTool` unchanged.
        if local is not None and local is not False and not isinstance(local, AbstractToolset) and not callable(local):
            from pydantic_ai.mcp import MCPToolset as _MCPToolset

            local = _MCPToolset(local, include_instructions=True)
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

    def _default_native(self) -> MCPServerTool:
        return MCPServerTool(
            id=self._resolved_id,
            url=self.url,
            authorization_token=self.authorization_token,
            headers=self.headers,
            allowed_tools=self.allowed_tools,
            description=self.description,
        )

    def _native_unique_id(self) -> str:
        return f'mcp_server:{self._resolved_id}'

    def _default_local(self) -> Tool[AgentDepsT] | AbstractToolset[AgentDepsT] | None:
        # Merge authorization_token into headers for local connection.
        local_headers = dict(self.headers or {})
        if self.authorization_token:
            local_headers['Authorization'] = self.authorization_token

        # `MCPToolset` infers SSE vs Streamable HTTP from the URL.
        from pydantic_ai.mcp import MCPToolset

        return MCPToolset(self.url, headers=local_headers or None, include_instructions=True)

    def get_toolset(self) -> AbstractToolset[AgentDepsT] | None:
        toolset = super().get_toolset()
        if toolset is not None and self.allowed_tools is not None:
            allowed = set(self.allowed_tools)
            return toolset.filtered(lambda _ctx, tool_def: tool_def.name in allowed)
        return toolset

    @classmethod
    def from_spec(
        cls,
        url: str,
        *,
        native: MCPServerTool | bool = True,
        local: str | Literal[False] | None = None,
        id: str | None = None,
        authorization_token: str | None = None,
        headers: dict[str, str] | None = None,
        allowed_tools: list[str] | None = None,
        description: str | None = None,
    ) -> MCP[AgentDepsT]:
        """Construct an `MCP` capability from spec-serializable args.

        Restricts the runtime-wide `local=` union to the JSON/YAML-serializable subset
        (`str | False | None`) so `AgentSpec` schema generation works. Non-serializable runtime
        values like `fastmcp.Client`, `ClientTransport`, or pre-built `MCPToolset` instances can
        still be passed to `MCP(...)` directly — they just can't roundtrip through a spec file.
        """
        return cls(
            url,
            native=native,
            local=local,
            id=id,
            authorization_token=authorization_token,
            headers=headers,
            allowed_tools=allowed_tools,
            description=description,
        )


install_deprecated_kwarg_alias(MCP, old='builtin', new='native')
