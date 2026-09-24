"""Server lifecycle: which servers exist, which the agent may use, and which hold a connection.

Enabled servers are offered to every run. The first prompt (or `/mcp start`) connects a `ready`
server and holds the connection open between prompts, which the dashboard reports as `running`.
A server that fails to connect is marked `error` and left out, so it cannot fail the prompt.
`stop` releases the connection and disables the server.
"""

import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from anyio import to_thread
from fastmcp.client.transports import SSETransport, StdioTransport, StreamableHttpTransport
from pydantic_ai import RunContext
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.toolsets import AbstractToolset, CombinedToolset

from ._settings import Server, Servers, SSEServer, StdioServer, http_client, missing, resolve
from ._store import MCPStore
from ._tokens import TokenStore, oauth

Source = Literal['user', 'plugin', 'project']
State = Literal['running', 'ready', 'stopped', 'error']


@dataclass(frozen=True, kw_only=True)
class ServerEntry:
    """A configured server and the file it came from."""

    name: str
    server: Server
    source: Source


@dataclass(kw_only=True)
class _Connection:
    server: Server
    toolset: MCPToolset[None]
    stack: AsyncExitStack | None = None
    started: float | None = None
    error: str | None = None
    tools: tuple[str, ...] = field(default_factory=tuple[str, ...])


def _shape(server: Server) -> Server:
    """The configuration a connection depends on; toggling `enabled` keeps the connection."""
    return server.model_copy(update={'enabled': True})


class MCPServers:
    """Merge the configured sources and own every held connection."""

    def __init__(self, store: MCPStore, plugin_servers: Servers | None = None) -> None:
        """`plugin_servers` come from `/plugins add mcp` settings, kept for existing configurations."""
        self.store = store
        self._plugin_servers = plugin_servers or {}
        self._overrides: dict[str, bool] = {}
        """Session-only enable state for servers whose file `/mcp` does not write."""
        self._connections: dict[str, _Connection] = {}

    def entries(self) -> list[ServerEntry]:
        """User servers, then plugin settings, then the trusted project file; the first name wins."""
        merged: dict[str, ServerEntry] = {}
        sources: tuple[tuple[Source, Servers], ...] = (
            ('user', self.store.load().servers),
            ('plugin', self._plugin_servers),
            ('project', self.store.project_servers()),
        )
        for source, servers in sources:
            for name, server in servers.items():
                merged.setdefault(name, ServerEntry(name=name, server=server, source=source))
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
        return self._overrides.get(entry.name, entry.server.enabled)

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
            return f'Could not start {name}: {problem}. See /mcp logs {name}.'
        return f'Started {name} with {len(connection.tools)} tools. The agent can use them on your next prompt.'

    async def _open(self, name: str, connection: _Connection) -> str | None:
        """Hold the connection and record its tools; the failure, logged, or `None`."""
        stack = AsyncExitStack()
        try:
            await stack.enter_async_context(connection.toolset)
            listed = await connection.toolset.list_tools()
        except Exception as exc:  # noqa: BLE001 -- any connection failure is reported, not raised.
            await stack.aclose()
            connection.error = f'{type(exc).__name__}: {exc}'
            self.log(name, f'start failed: {connection.error}')
            return connection.error
        connection.stack, connection.started, connection.error = stack, time.monotonic(), None
        connection.tools = tuple(f'{name}_{tool.name}' for tool in listed)
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
        reason = not_owned(self.get(name), self.store)
        if reason:
            raise ValueError(reason)
        await self.disconnect(name)
        self._overrides.pop(name, None)
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
        """The enabled servers for this run, each prefixed with its name."""
        await self.sync()
        toolsets: list[AbstractToolset[None]] = []
        for entry in self.entries():
            if self.state(entry) not in ('running', 'ready'):
                continue
            connection = await self._connection(entry)
            if connection.stack is None and await self._open(entry.name, connection):
                continue  # A server that cannot connect is marked `error`, not allowed to fail the prompt.
            toolsets.append(connection.toolset.prefixed(entry.name))
        return CombinedToolset(toolsets) if toolsets else None

    async def list_tools(self, name: str) -> list[str]:
        """Connect if needed and list the server's prefixed tool names."""
        entry = self.get(name)
        connection = await self._connection(entry)
        async with connection.toolset:
            tools = await connection.toolset.list_tools()
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
            self._overrides[entry.name] = enabled

    async def _connection(self, entry: ServerEntry) -> _Connection:
        connection = self._connections.get(entry.name)
        if connection is not None and _shape(connection.server) == _shape(entry.server):
            return connection
        if connection is not None:
            await self._release(entry.name)
        created = _Connection(server=entry.server, toolset=self._build(entry))
        self._connections[entry.name] = created
        return created

    async def _release(self, name: str) -> None:
        connection = self._connections.get(name)
        if connection is None or connection.stack is None:
            return
        stack, connection.stack, connection.started, connection.tools = connection.stack, None, None, ()
        await stack.aclose()

    def _build(self, entry: ServerEntry) -> MCPToolset[None]:
        server = entry.server
        transport: StdioTransport | StreamableHttpTransport | SSETransport
        if isinstance(server, StdioServer):
            self.store.logs.mkdir(parents=True, exist_ok=True)
            transport = StdioTransport(
                command=server.command,
                args=server.args,
                env=resolve(server.env),
                cwd=server.cwd,
                keep_alive=False,
                log_file=self.log_path(entry.name),
            )
            timeout = server.timeout
        elif isinstance(server, SSEServer):
            transport = SSETransport(
                url=str(server.url),
                headers=resolve(server.headers),
                auth=oauth(entry.name, server),
                httpx_client_factory=http_client,
            )
            timeout = server.init_timeout()
        else:
            transport = StreamableHttpTransport(
                url=str(server.url),
                headers=resolve(server.headers),
                auth=oauth(entry.name, server),
                httpx_client_factory=http_client,
            )
            timeout = server.init_timeout()
        if timeout is None:
            return MCPToolset(transport, id=f'mcp_{entry.name}')
        return MCPToolset(transport, id=f'mcp_{entry.name}', init_timeout=timeout)


def not_owned(entry: ServerEntry, store: MCPStore) -> str | None:
    """Why `/mcp` cannot change this server's saved configuration, or `None` when it can."""
    if entry.source == 'user':
        return None
    if entry.source == 'project':
        return f'{entry.name} is defined in {store.project_file()}; change that file instead.'
    return f'{entry.name} is configured through /plugins settings for mcp; change it there.'
