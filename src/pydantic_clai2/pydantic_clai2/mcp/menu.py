"""`/mcp` as a declared menu: each server with its state, managed through `Dialogs`, so the web UI shows it.

The terminal keeps `/mcp`'s dashboard and subcommands. Every action here runs one of those
subcommands through `MCPCommand.run`, asking what it needs through the dialogs it is given.
"""

from collections.abc import Iterator, Mapping
from functools import partial
from typing import Literal

from anyio import to_thread

from pydantic_clai2.mcp.command import MCPCommand
from pydantic_clai2.mcp.runtime import ServerEntry, not_owned, sign_in
from pydantic_clai2.mcp.settings import target
from pydantic_clai2.plugins.menus import Action, Dialogs, FieldRow, Option, SettingsMenu

ADD = 'add'
TRUST = 'trust'
SERVER = 'server.'
"""What starts a server's row key; a server's name has no dot, so no key clashes with `add` or `trust`."""
LOG_LINES = '40'

RowState = Literal['running', 'ready', 'stopped', 'error', 'sign-in']
"""A server row's value: `/mcp`'s state, or `sign-in` for an OAuth server that is on but signed out."""

_SUMMARIES: dict[RowState, str] = {
    'ready': 'Connects on your next prompt',
    'stopped': 'Off until you start it',
    'sign-in': 'Sign in to use it',
}


class ServersSource:
    """The servers `/mcp` lists, a row each with its state, then project-file trust and adding a server.

    Every row is an action: none has a value to type.
    """

    def __init__(self, command: MCPCommand) -> None:
        self._command = command
        self._servers = command.servers

    @property
    def title(self) -> str:
        return 'MCP servers'

    def rows(self) -> list[FieldRow]:
        rows = [self._server_row(entry) for entry in self._servers.entries()]
        if paths := self._servers.store.project_files():
            rows.append(
                FieldRow(
                    key=TRUST,
                    label='Project servers',
                    description="This repository's MCP files load once you trust them.",
                    default='trusted',
                    note=', '.join(map(str, paths)),
                )
            )
        rows.append(
            FieldRow(
                key=ADD,
                label='Add server',
                description='A local program (stdio), or an http or sse endpoint.',
                default='',
            )
        )
        return rows

    def current(self, row: FieldRow) -> str:
        if row.key.startswith(SERVER):
            return self.state(self._servers.get(row.key.removeprefix(SERVER)))
        if row.key == TRUST:
            store = self._servers.store
            states = {store.trust_state(path) for path in store.project_files()}
            return next((state for state in ('untrusted', 'changed') if state in states), 'trusted')
        return ''

    def problem(self, row: FieldRow, text: str) -> str | None:
        return f'{row.label} has no value to set; choose it instead.'

    def apply(self, row: FieldRow, raw: str) -> str:
        raise ValueError(self.problem(row, raw))

    def reset(self, row: FieldRow) -> str:
        raise ValueError(self.problem(row, ''))

    def state(self, entry: ServerEntry) -> RowState:
        """The server's state, as `/mcp` shows it, or `sign-in` when it is on and its OAuth sign-in is missing."""
        state = self._servers.state(entry)
        method = sign_in(entry.name, entry.server)
        if state != 'stopped' and method is not None and method.signed_in() is False:
            return 'sign-in'
        return state

    def _server_row(self, entry: ServerEntry) -> FieldRow:
        state = self.state(entry)
        if state == 'running':
            summary = f'{len(self._servers.tools(entry))} tools'
        elif state == 'error':
            summary = self._servers.problem(entry) or 'Could not connect'
        else:
            summary = _SUMMARIES[state]
        source = None if entry.source == 'user' else f'From {entry.path or "/plugins settings for mcp"}'
        return FieldRow(
            key=f'{SERVER}{entry.name}',
            label=entry.name,
            description=f'{entry.server.type} · {target(entry.server)}',
            default='',
            note=' · '.join(part for part in (summary, source) if part),
        )

    async def manage(self, name: str, dialogs: Dialogs) -> list[str]:
        """What to do with one server: the `/mcp` subcommands that apply to it, run as typing them would."""
        while True:
            entry = self._servers.get(name)
            # Its details and sign-in read the keyring.
            options, details = await to_thread.run_sync(self._choices, entry)
            picked = await dialogs.choose(title=name, options=options, description=details)
            if picked is None:
                return []
            if picked in ('tools', 'logs'):
                await self._show(picked, name, dialogs)
                continue
            if picked == 'remove' and not await dialogs.confirm(
                title=f'Remove {name}?',
                yes='Remove',
                no='Keep it',
                description='CLAI stops it and forgets its configuration and sign-in.',
            ):
                continue
            args = {'sign-in': ['auth', name], 'sign-out': ['auth', name, 'logout']}.get(picked, [picked, name])
            return [await self._command.run(args, dialogs=dialogs, editor=None)]

    def _choices(self, entry: ServerEntry) -> tuple[list[Option], str]:
        state = self._servers.state(entry)
        if state == 'running':
            options = [
                Option(value='restart', label='Restart', description='Reconnect, picking up configuration changes.'),
                Option(value='stop', label='Stop', description='Disconnect, and keep its tools from the agent.'),
            ]
        elif state == 'stopped':
            options = [Option(value='start', label='Start', description='Turn it on and connect now.')]
        else:
            options = [
                Option(value='start', label='Connect now', description='Instead of on your next prompt.'),
                Option(value='stop', label='Turn off', description='Keep its tools from the agent until you start it.'),
            ]
        options += [
            Option(value='tools', label='Show tools', description='Connect and list the tools the agent sees.'),
            Option(value='logs', label='Show logs', description='Its recent output and lifecycle events.'),
        ]
        method = sign_in(entry.name, entry.server)
        if method is not None:
            options.append(Option(value='sign-in', label='Sign in', description='In your browser, then connect.'))
            if method.signed_in():
                options.append(Option(value='sign-out', label='Sign out', description='Forget the stored sign-in.'))
        reason = not_owned(entry)
        if reason is None:
            options += [
                Option(value='edit', label='Edit…', description='Its name, type, JSON configuration, or OAuth.'),
                Option(value='remove', label='Remove…', description='Stop it and forget it.'),
            ]
        details = self._command.details(entry)
        return options, details if reason is None else f'{details}\n\n{reason}'

    async def _show(self, kind: str, name: str, dialogs: Dialogs) -> None:
        """The server's tools or recent logs, until the user goes back to its actions."""
        if kind == 'logs':
            text = await self._command.run(['logs', name, LOG_LINES], dialogs=dialogs, editor=None)
        else:
            try:
                text = await self._command.run(['tools', name], dialogs=dialogs, editor=None)
            except Exception as exc:  # noqa: BLE001 -- a server that cannot connect says why, as `/mcp tools` would.
                text = f'Could not list its tools: {str(exc) or type(exc).__name__}'
        await dialogs.choose(title=f'{name} {kind}', options=[Option(value='back', label='Back')], description=text)

    async def trust(self, dialogs: Dialogs) -> list[str]:
        """Trust this repository's MCP files, loading their servers, or revoke that trust, as `/mcp trust` does."""
        store = self._servers.store

        def review() -> str:
            files = (f'{path} ({store.trust_state(path)})\n{store.review(path)}' for path in store.project_files())
            return '\n\n'.join(files)

        files = await to_thread.run_sync(review)
        picked = await dialogs.choose(
            title='Project MCP servers',
            options=[
                Option(
                    value='accept',
                    label='Trust and load them',
                    description='Any later edit to a file needs trusting again.',
                ),
                Option(value='revoke', label='Revoke trust', description='Stop loading their servers.'),
            ],
            description=f'Review these files first: a stdio server runs a program on this computer.\n\n{files}',
        )
        if picked is None:
            return []
        if picked == 'accept' and await to_thread.run_sync(review) != files:
            # Trust covers the bytes on disk, which must be the ones the user just read.
            return ['A project MCP file changed while you reviewed it; review it again.']
        return [await self._command.run(['trust', picked], dialogs=dialogs, editor=None)]

    async def add(self, dialogs: Dialogs) -> list[str]:
        """The `/mcp install` form."""
        return [await self._command.run(['install'], dialogs=dialogs, editor=None)]


class _Actions(Mapping[str, Action]):
    """Each row's action, found by its key when asked, so a server added since the menu was built has one."""

    def __init__(self, source: ServersSource) -> None:
        self._source = source

    def __getitem__(self, key: str) -> Action:
        if key == ADD:
            return self._source.add
        if key == TRUST:
            return self._source.trust
        if key.startswith(SERVER):
            return partial(self._source.manage, key.removeprefix(SERVER))
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter([row.key for row in self._source.rows()])

    def __len__(self) -> int:
        return len(self._source.rows())


def servers_menu(command: MCPCommand) -> SettingsMenu:
    """The menu bare `/mcp` declares: every server with its state, trust, and adding one."""
    source = ServersSource(command)
    return SettingsMenu(source=source, actions=_Actions(source), searchable=False)
