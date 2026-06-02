from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from pydantic_ai.exceptions import UserError
from pydantic_ai.native_tools import MCPServerTool
from pydantic_ai.tools import AgentDepsT, RunContext, Tool
from pydantic_ai.toolsets import AbstractToolset

from .native_or_local import NativeOrLocalTool

if TYPE_CHECKING:
    from pydantic_ai.mcp import MCPToolset, MCPToolsetClient
else:
    try:
        from pydantic_ai.mcp import MCPToolset, MCPToolsetClient
    except ImportError:  # pragma: lax no cover
        MCPToolset = Any
        MCPToolsetClient = Any


@dataclass(init=False)
class MCP(NativeOrLocalTool[AgentDepsT]):
    """MCP server capability.

    The primary entry point for using MCP servers with Pydantic AI. Runs the MCP server
    locally — keeps credentials, hooks, and tracing under your control — and accepts any
    [`MCPToolset`][pydantic_ai.mcp.MCPToolset] input (URL, `fastmcp.Client`, transport,
    in-process `FastMCP` server, script path, etc.) directly via `local=`.

    Pass `url=` for HTTP-based servers; the same URL can also be advertised to providers
    that support native MCP via `native=True`. For non-URL local clients, omit `url=` and
    pass the client/toolset as `local=`. Pass `native=True, local=False` for strict
    native-only (no local at all — works without the `mcp` extra).
    """

    url: str | None
    """The URL of the MCP server.

    Required when using native MCP. Optional when using a local-only client via `local=`."""

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
        url: str | None = None,
        *,
        native: MCPServerTool
        | Callable[[RunContext[AgentDepsT]], Awaitable[MCPServerTool | None] | MCPServerTool | None]
        | bool = False,
        local: MCPToolsetClient | MCPToolset[AgentDepsT] | Callable[..., Any] | bool | None = None,
        id: str | None = None,
        authorization_token: str | None = None,
        headers: dict[str, str] | None = None,
        allowed_tools: list[str] | None = None,
        description: str | None = None,
        defer_loading: bool = False,
    ) -> None:
        # Native MCP requires a URL only when the capability auto-constructs an `MCPServerTool`
        # (i.e. `native=True`). Explicit `MCPServerTool(...)` instances and per-run callables
        # carry their own URL, so the capability's `url=` isn't needed in those cases.
        if url is None and native is True:
            raise UserError(
                'MCP(native=True) requires `url=` — native MCP needs a URL to give the model. '
                "Pass `url='https://…'`, pass an explicit `native=MCPServerTool(url='…', …)` "
                'instance, or for local-only use leave `native` at its default of `False` and '
                'pass `local=` (e.g. an `MCPToolset`, `fastmcp.Client`, transport, in-process '
                '`FastMCP` server, or script path).'
            )

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
        self.id = id
        self.authorization_token = authorization_token
        self.headers = headers
        self.allowed_tools = allowed_tools
        self.description = description
        self.defer_loading = defer_loading
        self.__post_init__()

    @cached_property
    def _resolved_id(self) -> str:
        if self.id:
            return self.id
        # An explicit `native=MCPServerTool(id=...)` carries its own id; key off it so the local
        # fallback's `unless_native` marker matches the native tool that's actually advertised.
        if isinstance(self.native, MCPServerTool):
            return self.native.id
        # Otherwise `_resolved_id` is only read through native paths, which require a URL (enforced in `__init__`).
        assert self.url is not None
        # Include hostname to avoid collisions (e.g. two /sse URLs on different hosts)
        parsed = urlparse(self.url)
        path = parsed.path.rstrip('/')
        slug = path.split('/')[-1] if path else ''
        host = parsed.hostname or ''
        return f'{host}-{slug}' if slug else host or self.url

    def _default_native(self) -> MCPServerTool:
        # `native is True` requires `url is not None` (enforced in `__init__`).
        assert self.url is not None
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
        if self.url is None:
            # No URL → no way to derive a default local; the user must have passed `local=` directly.
            return None
        return self._build_local(self.url)

    def _resolve_local_strategy(self, name: str | bool) -> Tool[AgentDepsT] | AbstractToolset[AgentDepsT]:
        # MCP has no named string strategies. `local=True` uses the URL from `MCP(url=...)`; a
        # string is treated as an override URL and validated to match — we only accept actual URLs
        # here so the same value can roundtrip through `from_spec`/`AgentSpec` and be served as a
        # native MCP tool by models that support it. Local-only inputs that aren't URLs (script
        # paths, `fastmcp.Client` instances, etc.) must be passed as `local=MCPToolset(...)` instead.
        if isinstance(name, str):
            _require_url(name)
            return self._build_local(name)
        if self.url is None:
            raise UserError(
                'MCP(local=True) requires `url=` to derive the local transport from. '
                "Pass `url='https://…'`, or pass a concrete local client/toolset as `local=`."
            )
        return self._build_local(self.url)

    def _build_local(self, url: str) -> Tool[AgentDepsT] | AbstractToolset[AgentDepsT]:
        # Merge authorization_token into headers for local connection.
        local_headers = dict(self.headers or {})
        if self.authorization_token:
            local_headers['Authorization'] = self.authorization_token

        try:
            # `MCPToolset` infers SSE vs Streamable HTTP from the URL.
            from pydantic_ai.mcp import MCPToolset

            return MCPToolset(url, headers=local_headers or None, include_instructions=True)
        except ImportError as e:
            raise UserError(
                'Please install the `mcp` package to run MCP servers locally, you can use the '
                '`mcp` optional group — `pip install "pydantic-ai-slim[mcp]"`. '
                'For native-only MCP (no local — no extra needed), pass '
                "`MCP(url='…', native=True, local=False)`."
            ) from e

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
        native: MCPServerTool | bool = False,
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
        (`str | bool | None`) so `AgentSpec` schema generation works, and requires `url=` (which
        is optional at runtime when `local=` is a concrete non-URL client). Non-serializable
        runtime values like `fastmcp.Client`, `ClientTransport`, or pre-built `MCPToolset`
        instances can still be passed to `MCP(...)` directly — they just can't roundtrip through
        a spec file.
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


def _require_url(value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc:
        raise UserError(
            f'MCP(local={value!r}) must be an `http(s)://` URL. For non-URL local clients (script '
            'paths, `fastmcp.Client`, transports, in-process `FastMCP` servers, etc.), pass '
            '`local=MCPToolset(...)` directly.'
        )
