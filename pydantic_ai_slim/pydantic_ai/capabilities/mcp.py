from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, TypeAlias
from urllib.parse import urlparse

from pydantic_ai import _utils
from pydantic_ai.exceptions import UserError
from pydantic_ai.native_tools import MCPServerTool
from pydantic_ai.tools import AgentDepsT, RunContext, Tool
from pydantic_ai.toolsets import AbstractToolset

from .native_or_local import NativeOrLocalTool

if TYPE_CHECKING:
    # Imported lazily at runtime: `pydantic_ai.mcp` pulls in `fastmcp` and `mcp`, which would
    # otherwise dominate the time it takes to `import pydantic_ai`.
    from pydantic_ai.mcp import MCPHeadersFunc, MCPToolset, MCPToolsetClient


_DEFAULT_PORTS = {'http': 80, 'https': 443}

MCPAuthorizationTokenFunc: TypeAlias = Callable[[RunContext[AgentDepsT]], str | None | Awaitable[str | None]]
"""A sync/async function that takes the run context and returns the `authorization_token` for that run."""


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

    authorization_token: str | MCPAuthorizationTokenFunc[AgentDepsT] | None
    """Authorization header value for MCP server requests. Passed to both native and local.

    Pass a function of the run context instead to connect each agent run as its own user, for
    example with a token from the run's [`deps`][pydantic_ai.tools.RunContext.deps]: it's called at
    the start of each run, and the token it returns is used for that run on both the native and the
    local side. Returning `None` sends no `Authorization` header."""

    headers: dict[str, str] | MCPHeadersFunc[AgentDepsT] | None
    """HTTP headers for MCP server requests. Passed to both native and local.

    Like `authorization_token`, this can be a function of the run context, called at the start of
    each run."""

    allowed_tools: list[str] | None
    """Filter to only these tools. Applied to both native and local."""

    tool_prefix: str | bool | None
    """Prepended with `_` to the names of the server's tools, as in `<prefix>_<tool>`, so they don't collide with other tools.

    - `None` (default) or `False`: keep the names the server gives its tools.
    - `True`: use the explicit `id` as the prefix. Requires `id=`; an `id` derived from the URL is never used.
    - A string: use that string as the prefix.

    Applies to the local tools. `allowed_tools` keeps matching the server's own names, and the toolset's
    `id` doesn't change. Native MCP tools are named by the model provider and aren't prefixed.
    """

    description: str | None = None
    """Description of the MCP server. Native-only; ignored by local tools."""

    _telemetry_url_derived_id: str | None = field(init=False, repr=False, compare=False, default=None)
    """Whether the final configured ID came from the URL (retained by capability merges)."""

    def __init__(
        self,
        url: str | None = None,
        *,
        native: MCPServerTool
        | Callable[[RunContext[AgentDepsT]], Awaitable[MCPServerTool | None] | MCPServerTool | None]
        | bool = False,
        local: MCPToolsetClient | MCPToolset[AgentDepsT] | Callable[..., Any] | bool | None = None,
        id: str | None = None,
        authorization_token: str | MCPAuthorizationTokenFunc[AgentDepsT] | None = None,
        headers: dict[str, str] | MCPHeadersFunc[AgentDepsT] | None = None,
        allowed_tools: list[str] | None = None,
        tool_prefix: str | bool | None = None,
        description: str | None = None,
        defer_loading: bool = False,
    ) -> None:
        if tool_prefix == '':
            raise UserError(
                "MCP(tool_prefix='') would prefix nothing. Pass `tool_prefix=None` to keep the server's tool names."
            )
        if tool_prefix is True and id is None:
            raise UserError(
                'MCP(tool_prefix=True) uses the explicit `id` as the prefix, but no `id=` was passed. '
                "Pass `id='…'`, or pass the prefix itself as `tool_prefix='…'`."
            )
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
        # A server's identity is its host, so the capability takes the same derived id its toolset
        # does rather than leaving the capability anonymous and the leaf named. Without it the run
        # falls back to a positional `mcp` / `mcp_2`, which reorders when the capability list does
        # -- no use as a durable-operation name or an instruction key, and nothing a user could
        # write down. `None` stays `None` when there is nothing to derive from (a local client that
        # carries its own connection, with no `url=` and no explicit `native=MCPServerTool(...)`).
        #
        # Not derived for a deferred capability, which keeps needing an explicit `id`. A deferred
        # capability's id is listed for the model in the `load_capability` catalog. Deriving an id is
        # for naming a durable operation, which is ours to see; putting one in the prompt is the
        # user's call to make deliberately.
        self.id = id if id is not None else (None if defer_loading else self._derive_id(url))
        # Marks an id derived from the URL rather than chosen by the caller. An explicit id,
        # including one supplied by an MCPServerTool, is the caller's chosen label.
        self._telemetry_url_derived_id = (
            self._derive_id(url) if id is None and url is not None and not isinstance(native, MCPServerTool) else None
        )
        # Non-string runtime `local=` inputs the base class doesn't recognize (Path, transport,
        # FastMCP server, pre-built `fastmcp.Client`, `AnyUrl`, etc.) are wrapped into an
        # `MCPToolset` here. Strings flow through `_resolve_local_strategy` below; pre-built
        # toolsets, callables, bools, and `None` pass through to `NativeOrLocalTool` unchanged.
        # Reaching this branch implies a fastmcp-typed object, which can only exist when the `mcp`
        # extra (and hence fastmcp) is installed, so the import below succeeds.
        if (
            local is not None
            and not isinstance(local, (bool, str))
            and not isinstance(local, AbstractToolset)
            and not callable(local)
        ):
            from pydantic_ai.mcp import MCPToolset

            # Stamp the derived id so this leaf can be used with durable execution too. `self.url` is
            # usually `None` here (the input carries its own connection), so the id comes from an
            # explicit `id=` or a native `MCPServerTool`, matching the `_build_local` URL path.
            local = MCPToolset(local, include_instructions=True, id=self._derive_id(self.url))
            # Without a URL of its own the capability has no marker to give, so a client or transport the
            # toolset derived its `id` from keeps the toolset's own.
            if self._telemetry_url_derived_id is not None:
                local._telemetry_url_derived_id = self._telemetry_url_derived_id  # pyright: ignore[reportPrivateUsage]
        self.local = local
        self.authorization_token = authorization_token
        self.headers = headers
        self.allowed_tools = allowed_tools
        self.tool_prefix = tool_prefix
        self.description = description
        self.defer_loading = defer_loading
        self.__post_init__()

    def _derived_id_source(self) -> str | None:
        if self.id is None or self.id != self._telemetry_url_derived_id or self.url is None:
            return None
        # The server endpoint without its userinfo or fragment, which never select a server. The query
        # is kept: it can select a tenant, so two URLs that differ only there may be two servers, and
        # telling them apart takes an explicit `id`.
        parsed = urlparse(self.url)
        scheme = parsed.scheme.lower()
        try:
            port = parsed.port or _DEFAULT_PORTS.get(scheme)
        except ValueError:
            # A malformed port can't be normalized, so it's compared as written.
            host = parsed.netloc.rpartition('@')[2].lower()
        else:
            host = f'{parsed.hostname}:{port}'
        return f'{scheme}://{host}{parsed.path.rstrip("/")}?{parsed.query}'

    def _derive_id(self, url: str | None) -> str | None:
        """Derive a stable id for this capability from `url`.

        Precedence: explicit `id` → the native `MCPServerTool`'s id → the hostname of `url`.
        Returns `None` when there's nothing to derive from — no `id`, no native `MCPServerTool`,
        and no hostname (e.g. a non-URL `local=` client that carries its own connection).

        Only the hostname is used: a URL's path, query or userinfo can carry a credential, and the
        id names durable operations and telemetry and can reach the model. Two servers on one host
        therefore derive the same id, which the agent rejects with a request for an explicit `id`.
        """
        if self.id is not None:
            return self.id
        # An explicit `native=MCPServerTool(id=...)` carries its own id; key off it so the local
        # fallback's `unless_native` marker matches the native tool that's actually advertised.
        if isinstance(self.native, MCPServerTool):
            return self.native.id
        if url is None:
            return None
        return urlparse(url).hostname or None

    @cached_property
    def _resolved_id(self) -> str:
        # Read by `_default_native()` (only when `native is True`, which requires `url=`) and by
        # `_native_unique_id()` (whenever `native is not False`, including a `native=<callable>`
        # factory). The callable-native path can reach here with no `url=`, no `id=`, and a
        # non-`MCPServerTool` native, leaving nothing to derive a stable id from.
        resolved = self._derive_id(self.url)
        if resolved is None:
            raise UserError(
                'MCP(native=<callable>) paired with a local fallback needs a stable `id` to tie the '
                'two together (the local fallback is tagged with the native id via `unless_native`). '
                'Pass `url=`, `id=`, or use an explicit `native=MCPServerTool(...)` instead of a callable.'
            )
        return resolved

    async def _resolve_credentials(self, ctx: RunContext[AgentDepsT]) -> tuple[str | None, dict[str, str] | None]:
        """Call whichever of `authorization_token` and `headers` is a function, for this run."""
        token = self.authorization_token
        if callable(token):
            token = await _utils.await_maybe(token(ctx))
        headers = self.headers
        if callable(headers):
            headers = await _utils.await_maybe(headers(ctx))
        return token, headers

    def _default_native(self) -> MCPServerTool | Callable[[RunContext[AgentDepsT]], Awaitable[MCPServerTool]]:
        # `native is True` requires `url is not None` (enforced in `__init__`).
        url = self.url
        assert url is not None
        token, headers = self.authorization_token, self.headers
        if not callable(token) and not callable(headers):
            return self._native_tool(url, token, headers)

        async def native_for_run(ctx: RunContext[AgentDepsT]) -> MCPServerTool:
            token, headers = await self._resolve_credentials(ctx)
            return self._native_tool(url, token, headers)

        return native_for_run

    def _native_tool(self, url: str, authorization_token: str | None, headers: dict[str, str] | None) -> MCPServerTool:
        tool = MCPServerTool(
            id=self._resolved_id,
            url=url,
            authorization_token=authorization_token,
            headers=headers,
            allowed_tools=self.allowed_tools,
            description=self.description,
        )
        tool._telemetry_url_derived_id = self._telemetry_url_derived_id  # pyright: ignore[reportPrivateUsage]
        return tool

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
        local_headers: dict[str, str] | Callable[[RunContext[AgentDepsT]], Awaitable[dict[str, str] | None]] | None
        token, headers = self.authorization_token, self.headers
        if not callable(token) and not callable(headers):
            local_headers = _local_headers(token, headers)
        else:
            # The toolset resolves these for each run, so each run connects with its own session.
            async def local_headers_for_run(ctx: RunContext[AgentDepsT]) -> dict[str, str] | None:
                return _local_headers(*await self._resolve_credentials(ctx))

            local_headers = local_headers_for_run

        try:
            # `MCPToolset` infers SSE vs Streamable HTTP from the URL.
            from pydantic_ai.mcp import MCPToolset

            # Stamp the derived id onto the local toolset so it can be used with durable execution
            # (which wraps leaf toolsets by `id`). `url` is always concrete here, so derivation from
            # it succeeds even when `self.url` is `None` (the `local='https://…'` override path).
            toolset = MCPToolset(url, headers=local_headers, include_instructions=True, id=self._derive_id(url))
            if self.id is None and not isinstance(self.native, MCPServerTool):
                toolset._telemetry_url_derived_id = toolset.id  # pyright: ignore[reportPrivateUsage]
            else:
                toolset._telemetry_url_derived_id = self._telemetry_url_derived_id  # pyright: ignore[reportPrivateUsage]
            return toolset
        except ImportError as e:
            raise UserError(
                'Please install the `mcp` package to run MCP servers locally, you can use the '
                '`mcp` optional group — `pip install "pydantic-ai-slim[mcp]"`. '
                'For native-only MCP (no local — no extra needed), pass '
                "`MCP(url='…', native=True, local=False)`."
            ) from e

    def get_toolset(self) -> AbstractToolset[AgentDepsT] | None:
        toolset = super().get_toolset()
        if toolset is None:
            return None
        if self.allowed_tools is not None:
            allowed = set(self.allowed_tools)
            toolset = toolset.filtered(lambda _ctx, tool_def: tool_def.name in allowed)
        # Prefixed last, so `allowed_tools` matches the server's own names. `__init__` only accepts
        # `True` beside an explicit `id`, never one derived from the URL.
        prefix = self.id if self.tool_prefix is True else self.tool_prefix
        if prefix:
            toolset = toolset.prefixed(prefix)
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
        tool_prefix: str | bool | None = None,
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
            tool_prefix=tool_prefix,
            description=description,
            defer_loading=defer_loading,
        )


def _local_headers(authorization_token: str | None, headers: dict[str, str] | None) -> dict[str, str] | None:
    """Merge `authorization_token` into the headers for the local connection."""
    local_headers = dict(headers or {})
    if authorization_token:
        local_headers['Authorization'] = authorization_token
    return local_headers or None


def _require_url(value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc:
        raise UserError(
            f'MCP(local={value!r}) must be an `http(s)://` URL. For non-URL local clients (script '
            'paths, `fastmcp.Client`, transports, in-process `FastMCP` servers, etc.), pass '
            '`local=MCPToolset(...)` directly.'
        )
