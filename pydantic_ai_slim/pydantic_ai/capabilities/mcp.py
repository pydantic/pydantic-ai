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

if TYPE_CHECKING:
    from pydantic_ai.mcp import MCPServer, MCPToolset, MCPToolsetClient
    from pydantic_ai.toolsets.fastmcp import FastMCPToolset  # pyright: ignore[reportDeprecated]
else:
    try:
        from pydantic_ai.mcp import MCPServer, MCPToolset, MCPToolsetClient
        from pydantic_ai.toolsets.fastmcp import FastMCPToolset
    except ImportError:  # pragma: lax no cover
        MCPServer = Any
        MCPToolset = Any
        MCPToolsetClient = Any
        FastMCPToolset = Any


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
        local: MCPToolsetClient
        | MCPToolset[AgentDepsT]
        | MCPServer
        | FastMCPToolset[AgentDepsT]  # pyright: ignore[reportDeprecated]
        | Callable[..., Any]
        | bool
        | None = None,
        id: str | None = None,
        authorization_token: str | None = None,
        headers: dict[str, str] | None = None,
        allowed_tools: list[str] | None = None,
        description: str | None = None,
        defer_loading: bool = False,
    ) -> None:
        self.description = description
        self.defer_loading = defer_loading
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
        # Non-string runtime `local=` inputs the base class doesn't recognize (Path, transport,
        # FastMCP server, pre-built `fastmcp.Client`, `AnyUrl`, etc.) are wrapped into an
        # `MCPToolset` here. Strings flow through `_resolve_local_strategy` below; pre-built
        # toolsets, callables, bools, and `None` pass through to `NativeOrLocalTool` unchanged.
        # Reaching this branch implies a fastmcp-typed object, which can only exist when the `mcp`
        # extra (and hence fastmcp) is installed; the module-level `MCPToolset` is the real class.
        if (
            local is not None
            and not isinstance(local, (bool, str))
            and not isinstance(local, AbstractToolset)
            and not callable(local)
        ):
            local = MCPToolset(local, include_instructions=True)
        self.local = local
        if id is not None:
            self.id = id
        elif isinstance(native, MCPServerTool):
            self.id = native.id
        else:
            self.id = _mcp_id_from_url(url)
        self.authorization_token = authorization_token
        self.headers = headers
        self.allowed_tools = allowed_tools
        self.__post_init__()

    def _default_native(self) -> MCPServerTool:
        mcp_id = self.id or _mcp_id_from_url(self.url)
        return MCPServerTool(
            id=mcp_id,
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
            return self._build_local(self.url)
        except ImportError:
            return None

    def _resolve_local_strategy(self, name: str | bool) -> Tool[AgentDepsT] | AbstractToolset[AgentDepsT]:
        # MCP has no named string strategies. `local=True` uses the URL from `MCP(url=...)`; a
        # string is treated as an override URL and validated to match — we only accept actual URLs
        # here so the same value can roundtrip through `from_spec`/`AgentSpec` and be served as a
        # native MCP tool by models that support it. Local-only inputs that aren't URLs (script
        # paths, `fastmcp.Client` instances, etc.) must be passed as `local=MCPToolset(...)` instead.
        try:
            if isinstance(name, str):
                _require_url(name)
                return self._build_local(name)
            return self._build_local(self.url)
        except ImportError as e:
            raise UserError('MCP(local=...) requires the MCP extra — `pip install "pydantic-ai-slim[mcp]"`.') from e

    def _build_local(self, url: str) -> Tool[AgentDepsT] | AbstractToolset[AgentDepsT]:
        # Merge authorization_token into headers for local connection.
        local_headers = dict(self.headers or {})
        if self.authorization_token:
            local_headers['Authorization'] = self.authorization_token

        # `MCPToolset` infers SSE vs Streamable HTTP from the URL.
        from pydantic_ai.mcp import MCPToolset

        return MCPToolset(url, headers=local_headers or None, include_instructions=True)

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
        local: str | bool | None = None,
        id: str | None = None,
        authorization_token: str | None = None,
        headers: dict[str, str] | None = None,
        allowed_tools: list[str] | None = None,
        description: str | None = None,
        defer_loading: bool = False,
    ) -> MCP[AgentDepsT]:
        """Construct an `MCP` capability from spec-serializable args.

        Restricts the runtime-wide `local=` union to the JSON/YAML-serializable subset
        (`str | bool | None`) so `AgentSpec` schema generation works. Non-serializable runtime
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
            defer_loading=defer_loading,
        )


install_deprecated_kwarg_alias(MCP, old='builtin', new='native')


def _require_url(value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc:
        raise UserError(
            f'MCP(local={value!r}) must be an `http(s)://` URL. For non-URL local clients (script '
            'paths, `fastmcp.Client`, transports, in-process `FastMCP` servers, etc.), pass '
            '`local=MCPToolset(...)` directly.'
        )
