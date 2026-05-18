from __future__ import annotations

import warnings
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from pydantic_ai._utils import install_deprecated_kwarg_alias
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.exceptions import UserError
from pydantic_ai.native_tools import MCPServerTool
from pydantic_ai.tools import AgentDepsT, RunContext, Tool
from pydantic_ai.toolsets import AbstractToolset

from .native_or_local import NativeOrLocalTool

try:
    from pydantic_ai.mcp import MCPServer
    from pydantic_ai.toolsets.fastmcp import FastMCPToolset
except ImportError:  # pragma: lax no cover
    if not TYPE_CHECKING:
        MCPServer = Any  # type: ignore[assignment,misc]
        FastMCPToolset = Any  # type: ignore[assignment,misc]


def _mcp_id_from_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip('/')
    slug = path.split('/')[-1] if path else ''

    if parsed.hostname:
        return f'{parsed.hostname}-{slug}' if slug else parsed.hostname

    if parsed.scheme and slug:
        return f'{parsed.scheme}-{slug}'

    return url


@dataclass(init=False)
class MCP(NativeOrLocalTool[AgentDepsT]):
    """MCP server capability.

    Uses the model's native MCP server support when available, connecting
    directly via HTTP when it isn't.
    """

    url: str
    """The URL of the MCP server."""

    authorization_token: str | None
    """Authorization header value for MCP server requests. Passed to both native and local."""

    headers: dict[str, str] | None
    """HTTP headers for MCP server requests. Passed to both native and local."""

    allowed_tools: list[str] | None
    """Filter to only these tools. Applied to both native and local."""

    description: str | None = None
    """Description of the MCP server. Native-only; ignored by local tools."""

    def __init__(
        self,
        url: str,
        *,
        native: MCPServerTool
        | Callable[[RunContext[AgentDepsT]], Awaitable[MCPServerTool | None] | MCPServerTool | None]
        | bool
        | None = None,
        local: MCPServer | FastMCPToolset[AgentDepsT] | Callable[..., Any] | bool | None = None,
        id: str | None = None,
        authorization_token: str | None = None,
        headers: dict[str, str] | None = None,
        allowed_tools: list[str] | None = None,
        description: str | None = None,
        defer_loading: bool | None = None,
    ) -> None:
        # In v2, MCP's `native` default flips from True to False. Warn whenever the user is
        # relying on the default — passing only `local=False` today gives native-only behavior,
        # but in v2 that combo will raise "both can't be False" without an explicit `native=True`.
        if native is None:
            warnings.warn(
                'MCP() defaults will change in v2: it will run the MCP server locally instead of '
                "preferring the model's native MCP support. To keep the current native-preferred "
                'behavior (with local as a fallback), pass `native=True`. To adopt the new '
                'local-first behavior now, install the MCP extra (`pip install '
                '"pydantic-ai-slim[mcp]"`) and pass `native=False`. For native-only (no local '
                'fallback), pass `native=True, local=False`.',
                PydanticAIDeprecationWarning,
                stacklevel=3,
            )
            native = True

        self.url = url
        self.native = native
        self.local = local
        # `AbstractCapability.__new__` seeds `id` with a UUID for subclasses that don't call
        # `super().__init__()`. MCP needs a stable semantic id so provider server labels and
        # local fallback `unless_native` markers refer to the same server across runs.
        if id:
            self.id = id
        elif isinstance(native, MCPServerTool):
            self.id = native.id
        else:
            self.id = _mcp_id_from_url(url)
        self.authorization_token = authorization_token
        self.headers = headers
        self.allowed_tools = allowed_tools
        self.description = description
        self.defer_loading = defer_loading
        self.__post_init__()

    def _default_native(self) -> MCPServerTool:
        return MCPServerTool(
            id=self.id,
            url=self.url,
            authorization_token=self.authorization_token,
            headers=self.headers,
            allowed_tools=self.allowed_tools,
            description=self.description,
        )

    def _native_unique_id(self) -> str:
        if isinstance(self.native, MCPServerTool):
            return self.native.unique_id
        return f'mcp_server:{self.id}'

    def _default_local(self) -> Tool[AgentDepsT] | AbstractToolset[AgentDepsT] | None:
        # The MCP extra may not be installed, in which case the capability still constructs cleanly
        # — the model just has to support MCP natively (or the user has to opt into `native=True`).
        try:
            return self._build_local()
        except ImportError:
            return None

    def _resolve_local_strategy(self, name: str | bool) -> Tool[AgentDepsT] | AbstractToolset[AgentDepsT]:
        if name is True:
            try:
                return self._build_local()
            except ImportError as e:
                raise UserError(
                    'MCP(local=True) requires the MCP extra — `pip install "pydantic-ai-slim[mcp]"`.'
                ) from e
        raise UserError(
            f'MCP(local={name!r}) is not a known strategy. '
            'Pass `local=True` for the default local MCP transport, or a Tool/callable directly.'
        )

    def _build_local(self) -> Tool[AgentDepsT] | AbstractToolset[AgentDepsT]:
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


install_deprecated_kwarg_alias(MCP, old='builtin', new='native')
