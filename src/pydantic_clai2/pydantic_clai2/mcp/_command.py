"""The `/mcp` command family, routed like Code Puppy's `MCPCommandHandler`."""

from collections import deque
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from anyio import to_thread

from ..field_menu import TERMINAL, Runners
from ..menu_worker import run_worker
from ._form import Editor, edit_form, edit_in_editor, install_form
from ._runtime import MCPServers, ServerEntry, State, not_owned
from ._settings import RemoteServer, references, target
from ._tokens import TokenStore

_GLYPHS: dict[State, str] = {'running': '+', 'ready': 'o', 'stopped': '-', 'error': '!'}
SERVER_SUBCOMMANDS = ('start', 'stop', 'restart', 'status', 'logs', 'auth', 'edit', 'remove', 'tools')
SUBCOMMANDS = ('list', 'install', 'start-all', 'stop-all', 'trust', 'help', *SERVER_SUBCOMMANDS)

HELP = """MCP server management

Servers
  /mcp                           Status dashboard (also /mcp list, /mcp status)
  /mcp install                   Add a server: name, type (stdio, http, sse), JSON config, OAuth
  /mcp start NAME                Enable and connect now
  /mcp stop NAME                 Disconnect and disable
  /mcp restart NAME              Reconnect, picking up config and environment changes
  /mcp start-all | stop-all      Every server at once
  /mcp status NAME               Details: target, env references, tools, last error
  /mcp tools NAME                Connect and list the tools the agent sees
  /mcp logs NAME [LINES]         Server stderr and lifecycle events (default 20 lines)
  /mcp auth NAME [logout]        Sign in to an OAuth server again, or sign out
  /mcp edit NAME                 Edit a saved server in the same form
  /mcp remove NAME               Stop and forget a saved server
  /mcp trust [status|accept|revoke]
                                 Load this repository's .clai/mcp_servers.json and .mcp.json

States:  + running (connected)  o ready (connects on your next prompt)  - stopped  ! error

Examples
  /mcp install             # opens the add-server form
  /mcp start filesystem    # connect now instead of on the next prompt
  /mcp logs filesystem 50  # the server's stderr
  /mcp edit filesystem     # change its JSON, type, or OAuth"""


def _uptime(seconds: float | None) -> str:
    if seconds is None:
        return '-'
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours}h {minutes}m' if hours else f'{minutes}m {secs}s' if minutes else f'{secs}s'


def _tail(path: Path, limit: int) -> tuple[int, list[str]]:
    """The line count and the last `limit` lines, streamed so a large log is never held in memory."""
    if not path.exists():
        return 0, []
    total, lines = 0, deque[str](maxlen=limit)
    with path.open(errors='replace') as file:
        for line in file:
            total += 1
            lines.append(line.rstrip('\n'))
    return total, list(lines)


def _no_arguments(usage: str, args: list[str]) -> None:
    if args:
        raise ValueError(f'Usage: /mcp {usage} (unexpected {" ".join(args)!r})')


@dataclass(kw_only=True)
class MCPCommand:
    """Parse `/mcp` arguments and dispatch. Menus run through `runners` so tests can script them."""

    servers: MCPServers
    runners: Runners = TERMINAL
    editor: Editor = edit_in_editor

    async def __call__(self, args: list[str]) -> str:
        """Bare `/mcp` shows the dashboard, like Code Puppy."""
        await self.servers.sync()
        if not args or (args[0] in ('list', 'status') and len(args) == 1):
            return self.dashboard()
        action, rest = args[0].lower(), args[1:]
        simple: dict[str, Callable[[list[str]], Awaitable[str]]] = {
            'install': self._install,
            'start-all': self._start_all,
            'stop-all': self._stop_all,
            'trust': self._trust,
        }
        if action in simple:
            return await simple[action](rest)
        if action == 'help':
            return HELP
        if action in SERVER_SUBCOMMANDS:
            if not rest:
                raise ValueError(f'Usage: /mcp {action} NAME')
            return await self._server_action(action, rest[0], rest[1:])
        raise ValueError(f'Unknown MCP subcommand: {action}. Type /mcp help for available commands.')

    def complete(self, args: list[str]) -> Iterable[str]:
        """Subcommands, then server names, trust actions, or `logout`."""
        if len(args) <= 1:
            return SUBCOMMANDS
        if len(args) == 2 and args[0] in SERVER_SUBCOMMANDS:
            try:
                return [entry.name for entry in self.servers.entries()]
            except ValueError:
                return ()
        if len(args) == 2 and args[0] == 'trust':
            return ('status', 'accept', 'revoke')
        if len(args) == 3 and args[0] == 'auth':
            return ('logout',)
        return ()

    def dashboard(self) -> str:
        """One row per server with state, source, uptime, and what it needs."""
        entries = self.servers.entries()
        trust = self._trust_notice()
        if not entries:
            lines = ['No MCP servers yet. /mcp install adds a stdio, http, or sse server; /mcp help lists commands.']
            return '\n'.join([*lines, *([trust] if trust else [])])
        width = max(len(entry.name) for entry in entries)
        lines = ['MCP servers', f'  {"NAME":<{width}}  TYPE   STATE    SOURCE   UPTIME   STATUS']
        for entry in entries:
            state = self.servers.state(entry)
            lines.append(
                f'{_GLYPHS[state]} {entry.name:<{width}}  {entry.server.type:<5}  {state:<7}  '
                f'{entry.source:<7}  {_uptime(self.servers.uptime(entry)):<7}  {self._summary(entry, state)}'
            )
        running = sum(self.servers.state(entry) == 'running' for entry in entries)
        usable = sum(self.servers.state(entry) in ('running', 'ready') for entry in entries)
        lines += ['', f'{running}/{len(entries)} running, {usable} available to the agent. /mcp help lists commands.']
        return '\n'.join([*lines, *([trust] if trust else [])])

    def details(self, entry: ServerEntry) -> str:
        """`/mcp status NAME`."""
        state = self.servers.state(entry)
        source = str(entry.path) if entry.path else '/plugins settings for mcp'
        variables = references(entry.server)
        tools = self.servers.tools(entry)
        return '\n'.join(
            [
                f'{_GLYPHS[state]} {entry.name}',
                f'  state    {state}' + (f' for {_uptime(self.servers.uptime(entry))}' if state == 'running' else ''),
                f'  type     {entry.server.type}',
                f'  target   {target(entry.server)}',
                f'  source   {entry.source} ({source})',
                f'  env      {", ".join(variables) if variables else "none referenced"}',
                *self._oauth_line(entry),
                f'  tools    {", ".join(tools) if tools else "listed after /mcp start"}',
                f'  error    {self.servers.problem(entry) or "none"}',
                f'  log      {self.servers.log_path(entry.name)}',
            ]
        )

    def _oauth_line(self, entry: ServerEntry) -> list[str]:
        if not isinstance(entry.server, RemoteServer) or entry.server.auth is None:
            return []
        state = TokenStore(entry.name).signed_in()
        status = {
            True: 'signed in',
            False: 'not signed in; the browser opens on connect',
            None: 'unknown; the keyring could not be read',
        }[state]
        return [f'  oauth    {status} (/mcp auth {entry.name} [logout])']

    def _summary(self, entry: ServerEntry, state: State) -> str:
        if state == 'running':
            return f'{len(self.servers.tools(entry))} tools'
        if state == 'error':
            return self.servers.problem(entry) or 'error'
        if state == 'ready':
            return 'connects on next prompt'
        return f'/mcp start {entry.name}'

    def _trust_notice(self) -> str | None:
        store = self.servers.store
        states = ((path, store.trust_state(path)) for path in store.project_files())
        notices = [
            f'{path} is {"changed since you trusted it" if state == "changed" else "not trusted"}, '
            'so its servers are not loaded.'
            for path, state in states
            if state != 'trusted'
        ]
        if not notices:
            return None
        return '\n' + '\n'.join([*notices, 'Review the project MCP files, then /mcp trust accept.'])

    async def _server_action(self, action: str, name: str, extra: list[str]) -> str:
        entry = self.servers.get(name)
        if action not in ('logs', 'auth'):
            _no_arguments(f'{action} NAME', extra)
        if action == 'status':
            return self.details(entry)
        if action == 'logs':
            return await self._logs(name, extra)
        if action == 'auth':
            return await self._auth(entry, extra)
        if action == 'tools':
            return '\n'.join(await self.servers.list_tools(name)) or f'No tools provided by {name}.'
        if action == 'remove':
            await self.servers.remove(name)
            return f'Removed {name}.'
        if action == 'edit':
            reason = not_owned(entry)
            if reason:
                raise ValueError(reason)
            running = self.servers.state(entry) == 'running'
            saved = await run_worker(lambda: edit_form(self.servers.store, name, self.runners, self.editor))
            if saved is None:
                return 'No changes.'
            new_name, message = saved
            await self.servers.sync()
            return f'{message}\n{await self.servers.restart(new_name)}' if running else message
        methods = {'start': self.servers.start, 'stop': self.servers.stop, 'restart': self.servers.restart}
        return await methods[action](name)

    async def _auth(self, entry: ServerEntry, extra: list[str]) -> str:
        server, name = entry.server, entry.name
        if not isinstance(server, RemoteServer) or server.auth is None:
            raise ValueError(f'{name} does not use OAuth. Turn on OAuth sign-in with /mcp edit {name}.')
        if extra not in ([], ['logout']):
            raise ValueError('Usage: /mcp auth NAME [logout]')
        await self.servers.disconnect(name)
        await to_thread.run_sync(TokenStore(name).forget)
        if extra:
            return f'Signed out of {name}. Its next connection opens the browser to sign in.'
        return await self.servers.start(name)

    async def _logs(self, name: str, extra: list[str]) -> str:
        if extra and not extra[0].isdigit():
            raise ValueError('Usage: /mcp logs NAME [LINES]')
        limit = int(extra[0]) if extra else 20
        path = self.servers.log_path(name)
        total, lines = await to_thread.run_sync(_tail, path, limit)
        if not total:
            return f'No log entries for {name} yet.'
        return '\n'.join([f'{path} (last {len(lines)} of {total} lines)', *lines])

    async def _install(self, args: list[str]) -> str:
        if args:
            raise ValueError('Usage: /mcp install (opens the add-server form)')
        return await run_worker(lambda: install_form(self.servers.store, self.runners, self.editor))

    async def _start_all(self, args: list[str]) -> str:
        _no_arguments('start-all', args)
        entries = self.servers.entries()
        return '\n'.join([await self.servers.start(entry.name) for entry in entries]) or 'No MCP servers to start.'

    async def _stop_all(self, args: list[str]) -> str:
        _no_arguments('stop-all', args)
        entries = self.servers.entries()
        return '\n'.join([await self.servers.stop(entry.name) for entry in entries]) or 'No MCP servers to stop.'

    async def _trust(self, args: list[str]) -> str:
        store = self.servers.store
        paths = store.project_files()
        action = args[0] if args else 'status'
        if len(args) > 1 or action not in ('status', 'accept', 'revoke'):
            raise ValueError('Usage: /mcp trust [status|accept|revoke]')
        if not paths:
            return 'No .clai/mcp_servers.json or .mcp.json between here and the repository root.'
        listed = ', '.join(map(str, paths))
        if action == 'accept':
            store.trust(*paths)
            names = ', '.join(name for servers in store.project_servers().values() for name in servers) or 'none'
            return f'Trusted {listed}. Servers loaded: {names}. Any edit to a file requires trusting it again.'
        if action == 'revoke':
            revoked = store.revoke(*paths)
            await self.servers.sync()
            return f'Revoked trust in {", ".join(map(str, revoked))}.' if revoked else f'Not trusted: {listed}.'
        return '\n'.join(f'{path}: {store.trust_state(path)}' for path in paths)
