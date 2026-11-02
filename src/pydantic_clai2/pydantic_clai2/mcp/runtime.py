"""Server lifecycle: which servers exist, which the agent may use, and which hold a connection.

Enabled servers are offered to every run. The first prompt (or `/mcp start`) connects a `ready`
server and holds the connection open between prompts, which the dashboard reports as `running`.
A server that fails to connect is marked `error` and left out, so it cannot fail the prompt.
`stop` releases the connection and disables the server.

A server with `defer_loading` is offered as the on-demand capability `mcp_NAME` instead: the model sees
one catalog line until it loads it, and only then does a run connect the server and list its tools.

An OAuth server connects with its stored sign-in only; `/mcp auth NAME` is the one place that opens the browser.
Signed out, it is marked `error` with that hint instead of holding the prompt up for a sign-in.
"""

import textwrap
import time
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from anyio import to_thread

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, Toolset
from pydantic_ai.toolsets import AbstractToolset, CombinedToolset, ToolsetTool
from pydantic_clai2.mcp.settings import (
    RemoteServer,
    Server,
    Servers,
    SSEServer,
    StdioServer,
    missing,
    resolve,
)
from pydantic_clai2.mcp.store import MCPStore
from pydantic_clai2.mcp.tokens import OAuthSignIn, TokenStore
from pydantic_clai2.plugins.sign_in import SignInRequiredError, not_signed_in

if TYPE_CHECKING:
    from pydantic_ai.mcp import MCPToolset

Source = Literal['user', 'plugin', 'project']
State = Literal['running', 'ready', 'stopped', 'error']

ABOUT_LIMIT = 200
"""The most characters of a server's own instructions an on-demand catalog line shows."""


def capability_id(name: str) -> str:
    """The server's toolset and on-demand capability id; fixed, so a resumed conversation keeps it loaded."""
    return f'mcp_{name}'


@dataclass(frozen=True, kw_only=True)
class ServerEntry:
    """A configured server and the file it came from."""

    name: str
    server: Server
    source: Source
    path: Path | None = None
    """The file that defines the server; `None` for plugin settings."""

    @property
    def origin(self) -> tuple[Source, Path | None, str]:
        """The definition, so session state does not pass to another source's server of the same name."""
        return self.source, self.path, self.name


@dataclass(kw_only=True)
class _Connection:
    server: Server
    toolset: 'MCPToolset[None]'
    prefixed: AbstractToolset[None]
    """`toolset` with the server's name prefix, built once so a run step sees the same toolset again."""
    stack: AsyncExitStack | None = None
    started: float | None = None
    error: str | None = None
    tools: tuple[str, ...] = field(default_factory=tuple[str, ...])
    about: str | None = None
    """What the server said about itself when it last connected, for an on-demand catalog line."""


def _shape(server: Server) -> Server:
    """The configuration a connection depends on; `enabled`, `defer_loading`, and `description` keep it."""
    return server.model_copy(update={'enabled': True, 'defer_loading': False, 'description': None})


class MCPServers:
    """Merge the configured sources and own every held connection."""

    def __init__(self, store: MCPStore, plugin_servers: Servers | None = None) -> None:
        """`plugin_servers` come from `/plugins add mcp` settings, kept for existing configurations."""
        self.store = store
        self._plugin_servers = plugin_servers or {}
        self._overrides: dict[tuple[Source, Path | None, str], bool] = {}
        """Session-only enable state for servers whose file `/mcp` does not write."""
        self._connections: dict[str, _Connection] = {}

    def entries(self) -> list[ServerEntry]:
        """User servers, then plugin settings, then trusted project files; the first name wins."""
        merged: dict[str, ServerEntry] = {}
        sources: list[tuple[Source, Path | None, Servers]] = [
            ('user', self.store.path, self.store.load().servers),
            ('plugin', None, self._plugin_servers),
            *(('project', path, servers) for path, servers in self.store.project_servers().items()),
        ]
        for source, path, servers in sources:
            for name, server in servers.items():
                merged.setdefault(name, ServerEntry(name=name, server=server, source=source, path=path))
        return list(merged.values())

    def get(self, name: str) -> ServerEntry:
        """The named server, or a `ValueError` naming the known ones."""
        entries = self.entries()
        entry = next((entry for entry in entries if entry.name == name), None)
        if entry is None:
            known = ', '.join(entry.name for entry in entries) or 'none; see /mcp install'
            raise ValueError(f'Unknown MCP server: {name}. Known: {known}')
        return entry

    def enabled(self, entry: ServerEntry) -> bool:
        """Whether the agent may use the server."""
        return self._overrides.get(entry.origin, entry.server.enabled)

    def state(self, entry: ServerEntry) -> State:
        """What the dashboard shows."""
        if not self.enabled(entry):
            return 'stopped'
        connection = self._connections.get(entry.name)
        if missing(entry.server) or (connection is not None and connection.error):
            return 'error'
        return 'running' if connection is not None and connection.stack is not None else 'ready'

    def problem(self, entry: ServerEntry) -> str | None:
        """Why the server is in the `error` state."""
        unset = missing(entry.server)
        if unset:
            return f'set {", ".join(unset)} in your environment'
        connection = self._connections.get(entry.name)
        return connection.error if connection is not None else None

    def uptime(self, entry: ServerEntry) -> float | None:
        """Seconds since `start` connected, or `None` without a held connection."""
        connection = self._connections.get(entry.name)
        if connection is None or connection.started is None:
            return None
        return time.monotonic() - connection.started

    def tools(self, entry: ServerEntry) -> tuple[str, ...]:
        """Prefixed tool names recorded when the server last started."""
        connection = self._connections.get(entry.name)
        return connection.tools if connection is not None else ()

    def description(self, entry: ServerEntry) -> str:
        """The catalog line: the configured one, else what the server said about itself, else its name."""
        connection = self._connections.get(entry.name)
        about = connection.about if connection is not None else None
        return entry.server.description or about or f'Tools from the {entry.name} MCP server.'

    async def start(self, name: str) -> str:
        """Enable the server and hold a connection open."""
        entry = self.get(name)
        self._set_enabled(entry, enabled=True)
        connection = await self._connection(entry)
        if connection.stack is not None:
            return f'{name} is already running with {len(connection.tools)} tools.'
        if missing(entry.server):
            return f'{name} is enabled but cannot connect: {self.problem(entry)}.'
        problem = await self._open(name, connection)
        if problem:
            return f'Could not start {name}: {problem.rstrip(".")}. See /mcp logs {name}.'
        return f'Started {name} with {len(connection.tools)} tools. The agent can use them on your next prompt.'

    async def _open(self, name: str, connection: _Connection) -> str | None:
        """Hold the connection and record its tools; the failure, logged, or `None`."""
        signed_out = await _signed_out(name, connection.server)
        if signed_out is not None:
            connection.error = signed_out
            self.log(name, f'start failed: {connection.error}')
            return connection.error
        stack = AsyncExitStack()
        try:
            await stack.enter_async_context(connection.toolset)
            listed = await connection.toolset.list_tools()
        except Exception as exc:  # noqa: BLE001 -- any connection failure is reported, not raised.
            await stack.aclose()
            required = _sign_in_required(exc)
            connection.error = str(required) if required else f'{type(exc).__name__}: {exc}'
            self.log(name, f'start failed: {connection.error}')
            return connection.error
        connection.stack, connection.started, connection.error = stack, time.monotonic(), None
        connection.tools = tuple(f'{name}_{tool.name}' for tool in listed)
        connection.about = _about(connection.toolset)
        self.log(name, f'started with {len(listed)} tools')
        return None

    async def stop(self, name: str) -> str:
        """Release the connection and disable the server."""
        entry = self.get(name)
        self._set_enabled(entry, enabled=False)
        await self._release(name)
        self.log(name, 'stopped')
        return f'Stopped {name}. Its tools are unavailable until /mcp start {name}.'

    async def restart(self, name: str) -> str:
        """Reconnect, picking up configuration or environment changes."""
        self.get(name)
        await self.disconnect(name)
        return await self.start(name)

    async def disconnect(self, name: str) -> None:
        """Close and drop the connection without changing whether the server is enabled."""
        await self._release(name)
        self._connections.pop(name, None)

    async def remove(self, name: str) -> None:
        """Forget a user server; project and plugin servers live in files `/mcp` does not own."""
        reason = not_owned(self.get(name))
        if reason:
            raise ValueError(reason)
        await self.disconnect(name)
        self.store.delete(name)
        await to_thread.run_sync(TokenStore(name).forget)

    async def sync(self) -> None:
        """Drop connections whose server was removed or reconfigured since they opened."""
        current = {entry.name: _shape(entry.server) for entry in self.entries()}
        for name, connection in list(self._connections.items()):
            if current.get(name) != _shape(connection.server):
                await self._release(name)
                del self._connections[name]

    async def close(self) -> None:
        """Release every held connection."""
        for name in list(self._connections):
            await self._release(name)

    async def toolset(self, ctx: RunContext[None]) -> AbstractToolset[None] | None:
        """The enabled eager servers for this run, each prefixed with its name."""
        await self.sync()
        toolsets: list[AbstractToolset[None]] = []
        for entry in self._usable():
            if entry.server.defer_loading:
                continue
            toolset = await self._attach(entry)
            if toolset is not None:
                toolsets.append(toolset)
        return CombinedToolset(toolsets) if toolsets else None

    async def capability(self, ctx: RunContext[None]) -> AbstractCapability[None] | None:
        """The enabled on-demand servers for this run, each its own deferred capability.

        Nothing connects here: a server connects on the first run step after the model loads it.
        """
        await self.sync()
        capabilities: list[AbstractCapability[None]] = [
            Toolset(
                _OnDemandToolset(capability_id=capability_id(entry.name), attach=partial(self._attach, entry)),
                id=capability_id(entry.name),
                description=self.description(entry),
                defer_loading=True,
            )
            for entry in self._usable()
            if entry.server.defer_loading
        ]
        return CombinedCapability(capabilities) if capabilities else None

    def _usable(self) -> list[ServerEntry]:
        return [entry for entry in self.entries() if self.state(entry) in ('running', 'ready')]

    async def _attach(self, entry: ServerEntry) -> AbstractToolset[None] | None:
        """The server's prefixed tools, connecting it first; `None` when it cannot connect."""
        connection = await self._connection(entry)
        if connection.error is not None:
            return None  # Tried and failed: an on-demand server is asked again on each step of the run.
        if connection.stack is None and await self._open(entry.name, connection):
            return None  # A server that cannot connect is marked `error`, not allowed to fail the prompt.
        return connection.prefixed

    async def list_tools(self, name: str) -> list[str]:
        """Connect if needed and list the server's prefixed tool names."""
        entry = self.get(name)
        signed_out = await _signed_out(name, entry.server)
        if signed_out is not None:
            raise ValueError(signed_out)
        connection = await self._connection(entry)
        try:
            async with connection.toolset:
                tools = await connection.toolset.list_tools()
                connection.about = _about(connection.toolset)
        except Exception as exc:
            required = _sign_in_required(exc)
            if required is None:
                raise
            raise ValueError(str(required)) from None
        connection.error = None
        return [f'{name}_{tool.name}' for tool in tools]

    def log(self, name: str, message: str) -> None:
        """Append a lifecycle line to the server's log, next to its captured stderr."""
        self.store.logs.mkdir(parents=True, exist_ok=True)
        with self.log_path(name).open('a') as file:
            file.write(f'{datetime.now().isoformat(timespec="seconds")} [clai] {message}\n')

    def log_path(self, name: str) -> Path:
        """Where the server's stderr and lifecycle lines go."""
        return self.store.logs / f'{name}.log'

    def _set_enabled(self, entry: ServerEntry, *, enabled: bool) -> None:
        if entry.source == 'user':
            self.store.put(entry.name, entry.server.model_copy(update={'enabled': enabled}))
        else:
            self._overrides[entry.origin] = enabled

    async def _connection(self, entry: ServerEntry) -> _Connection:
        connection = self._connections.get(entry.name)
        if connection is not None and _shape(connection.server) == _shape(entry.server):
            return connection
        if connection is not None:
            await self._release(entry.name)
        toolset = self._build(entry)
        created = _Connection(server=entry.server, toolset=toolset, prefixed=toolset.prefixed(entry.name))
        self._connections[entry.name] = created
        return created

    async def _release(self, name: str) -> None:
        connection = self._connections.get(name)
        if connection is None or connection.stack is None:
            return
        stack, connection.stack, connection.started, connection.tools = connection.stack, None, None, ()
        await stack.aclose()

    def _build(self, entry: ServerEntry) -> 'MCPToolset[None]':
        from pydantic_ai.mcp import MCPToolset
        from pydantic_clai2.mcp._client import SSETransport, StdioTransport, StreamableHttpTransport, http_client

        server = entry.server
        transport: StdioTransport | StreamableHttpTransport | SSETransport
        if isinstance(server, StdioServer):
            self.store.logs.mkdir(parents=True, exist_ok=True)
            transport = StdioTransport(
                command=server.command,
                args=server.args,
                env=resolve(server.env),
                # A relative `cwd` is the session's subdirectory; an absolute one stays as it is.
                cwd=str(self.store.workspace / (server.cwd or '')),
                keep_alive=False,
                log_file=self.log_path(entry.name),
            )
            timeout = server.timeout
        else:
            method = sign_in(entry.name, server)
            if method is not None:
                transport = method.transport()
            elif isinstance(server, SSEServer):
                transport = SSETransport(
                    url=str(server.url), headers=resolve(server.headers), httpx_client_factory=http_client
                )
            else:
                transport = StreamableHttpTransport(
                    url=str(server.url), headers=resolve(server.headers), httpx_client_factory=http_client
                )
            timeout = server.timeout
        if timeout is None:
            return MCPToolset(transport, id=capability_id(entry.name))
        return MCPToolset(transport, id=capability_id(entry.name), init_timeout=timeout)


@dataclass(kw_only=True)
class _OnDemandToolset(AbstractToolset[None]):
    """An on-demand server's tools, once the model has loaded its capability; no connection before that.

    The connection is `MCPServers`'s, held between prompts like an eager server's, so the run does not enter it.
    """

    capability_id: str
    attach: Callable[[], Awaitable[AbstractToolset[None] | None]]

    @property
    def id(self) -> str | None:  # pragma: no cover -- only durable execution, which CLAI does not use, asks.
        return None

    async def get_tools(self, ctx: RunContext[None]) -> dict[str, ToolsetTool[None]]:
        if self.capability_id not in ctx.loaded_capability_ids:
            return {}
        toolset = await self.attach()
        return {} if toolset is None else await toolset.get_tools(ctx)

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[None], tool: ToolsetTool[None]
    ) -> Any:
        return await tool.toolset.call_tool(name, tool_args, ctx, tool)


def _about(toolset: 'MCPToolset[None]') -> str | None:
    """The first line of the server's instructions, else the title it gave itself; `None` when it gave neither."""
    lines = (toolset.instructions or '').strip().splitlines()
    if lines:
        return textwrap.shorten(lines[0], ABOUT_LIMIT, placeholder='...')
    try:
        return toolset.server_info.title
    except AttributeError:  # The server sent no server info.
        return None


def sign_in(name: str, server: Server) -> OAuthSignIn | None:
    """The server's browser sign-in, when it uses OAuth; `/mcp auth NAME` runs it."""
    if not isinstance(server, RemoteServer) or server.auth is None:
        return None
    return OAuthSignIn(
        name=name,
        service=name,
        setup=f'/mcp auth {name}',
        url=str(server.url),
        headers=resolve(server.headers),
        sse=isinstance(server, SSEServer),
        init_timeout=server.timeout,
    )


async def _signed_out(name: str, server: Server) -> str | None:
    """How to sign in, when the server uses OAuth and has no stored sign-in; `None` otherwise."""
    method = sign_in(name, server)
    if method is None or await to_thread.run_sync(method.signed_in, abandon_on_cancel=True):
        return None
    return not_signed_in(method)


def _sign_in_required(error: BaseException) -> SignInRequiredError | None:
    """The `SignInRequiredError` FastMCP wrapped, when the stored sign-in was rejected."""
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, SignInRequiredError):
            return current
        current = current.__cause__ or current.__context__
    return None


def not_owned(entry: ServerEntry) -> str | None:
    """Why `/mcp` cannot change this server's saved configuration, or `None` when it can."""
    if entry.source == 'user':
        return None
    if entry.source == 'project':
        return f'{entry.name} is defined in {entry.path}; change that file instead.'
    return f'{entry.name} is configured through /plugins settings for mcp; change it there.'
