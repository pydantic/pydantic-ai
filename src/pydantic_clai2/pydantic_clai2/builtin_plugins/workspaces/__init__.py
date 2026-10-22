"""Workspaces: checkouts of a repository, each with its own CLAI, made, opened, and destroyed with /workspaces.

The built-in `workspaces` plugin, off by default. Its settings are the kinds of workspace there are: named
profiles of a backend with its options. This release has one backend, `local-worktree`, a Git worktree on
this machine whose CLAI runs as you with your own settings and sign-ins, so nothing is copied into it.

The live workspaces are kept in a registry in CLAI's state directory, so every CLAI lists the same ones;
see `registry`. `Workspaces` in `manager` is what `/workspaces` and the web UI's orchestrator call.
"""

import html
from collections.abc import Iterable, Sequence

from anyio import to_thread
from pydantic import ValidationError

from pydantic_clai2.builtin_plugins.workspaces.kinds import (
    KindName,
    Teardown,
    WorkspacesSettings,
    WorktreeKind,
)
from pydantic_clai2.builtin_plugins.workspaces.manager import Workspaces
from pydantic_clai2.builtin_plugins.workspaces.registry import (
    SessionProcess,
    Workspace,
    WorkspaceStatus,
    write_private,
)
from pydantic_clai2.commands import Command
from pydantic_clai2.plugins import Plugin, PluginCatalog, PluginHost
from pydantic_clai2.plugins.menus import Dialogs, FieldRow, Option, SettingsMenu
from pydantic_clai2.ui.browser import open_browser

__all__ = (
    'KindName',
    'SessionProcess',
    'Teardown',
    'Workspace',
    'WorkspaceStatus',
    'Workspaces',
    'WorkspacesPlugin',
    'WorkspacesSettings',
    'WorktreeKind',
    'configured',
    'describe',
)

NAME = 'workspaces'
"""The plugin's id, as `/plugins` lists it."""
USAGE = 'Usage: /workspaces [list | create KIND [REPO [REF]] | open ID | stop ID | destroy ID]'
_SUBCOMMANDS = ('list', 'create', 'open', 'stop', 'destroy')


def configured(plugins: PluginCatalog | None) -> Workspaces | None:
    """The registry with the kinds the `workspaces` plugin is configured with, for another plugin; `None` while it is off."""
    rows = plugins.listing() if plugins is not None else []
    row = next((row for row in rows if row.name == NAME and row.status == 'on'), None)
    if row is None or row.settings is None:
        return None
    # Validated again rather than checked with `isinstance`: `/plugins reload` makes a new settings class.
    return Workspaces(kinds=WorkspacesSettings.model_validate(row.settings.model_dump()).kinds)


def describe(workspace: Workspace) -> str:
    """One line for a workspace: its ID, status and address, kind, branch, and folder. Never its token."""
    session = workspace.session
    status = f'running at {session.url}' if workspace.status == 'running' and session else workspace.status
    return f'{workspace.id}  {status}  {workspace.kind}  {workspace.branch}  {workspace.path}'


class WorkspacesPlugin(Plugin[WorkspacesSettings]):
    """Checkouts of a repository, each with its own CLAI: /workspaces makes, opens, stops, and destroys them."""

    scope = 'process'
    """One lock over the shared registry; `host.workspace` follows the session that runs `/workspaces`."""

    def __init__(self, host: PluginHost[None], settings: WorkspacesSettings) -> None:
        super().__init__(host, settings)
        self.workspaces = Workspaces(kinds=settings.kinds)

    def get_commands(self) -> Sequence[Command]:
        return (
            Command(
                name='workspaces',
                description='List workspaces; create KIND [REPO [REF]], open ID, stop ID, or destroy ID',
                handler=self.command,
                complete=self.complete,
                live=True,
            ),
        )

    def settings_menu(self) -> SettingsMenu:
        source = WorkspacesSource(self.host)
        return SettingsMenu(source=source, actions={'add': source.add, 'remove': source.remove})

    def complete(self, args: list[str]) -> Iterable[str]:
        if len(args) <= 1:
            return _SUBCOMMANDS
        if len(args) == 2 and args[0] == 'create':
            return sorted(self.settings.kinds)
        if len(args) == 2 and args[0] in ('open', 'stop', 'destroy'):
            return [workspace.id for workspace in self.workspaces.registry.all()]
        return ()

    async def command(self, args: list[str]) -> str:
        match args:
            case [] | ['list']:
                listing = await self.workspaces.listing()
                if not listing:
                    return 'No workspaces. Make one with /workspaces create KIND [REPO [REF]].'
                return '\n'.join(describe(workspace) for workspace in listing)
            case ['create', kind, *rest] if len(rest) <= 2:
                repo, ref = (*rest, '', '')[:2]
                workspace = await self.workspaces.create(kind=kind, repo=repo, ref=ref, base=self.host.workspace)
                return (
                    f'Created workspace {workspace.id} at {workspace.path} on branch {workspace.branch}. '
                    f'Start its CLAI with /workspaces open {workspace.id}.'
                )
            case ['open', workspace_id]:
                return await self.open(workspace_id)
            case ['stop', workspace_id]:
                workspace = await self.workspaces.stop(workspace_id)
                return f'Workspace {workspace.id}: {workspace.status}.'
            case ['destroy', workspace_id]:
                workspace = await self.workspaces.destroy(workspace_id)
                kept = 'removed' if workspace.teardown == 'remove' else 'kept'
                return f'Destroyed workspace {workspace.id}; its checkout at {workspace.path} was {kept}.'
            case _:
                raise ValueError(USAGE)

    async def open(self, workspace_id: str) -> str:
        """Start the workspace's CLAI if needed and open its page through a file only you can read."""
        workspace = await self.workspaces.start(workspace_id)
        assert workspace.session is not None, 'a started workspace has a session'
        url = f'{workspace.session.url}?token={await self.workspaces.token(workspace_id)}'
        page = self.workspaces.registry.folder(workspace_id) / 'open.html'
        content = f'<!doctype html><meta http-equiv="refresh" content="0;url={html.escape(url)}">'
        await to_thread.run_sync(write_private, page, content)
        opened = await to_thread.run_sync(open_browser, page.as_uri())
        where = 'opened in your browser' if opened else f'open {page} in your browser'
        return f'Workspace {workspace.id} runs at {workspace.session.url}; {where}.'


_FIELDS: dict[str, str] = {
    'repo': 'Repository',
    'ref': 'Start from',
    'root': 'Checkouts folder',
    'branch_prefix': 'Branch prefix',
    'teardown': 'On destroy',
    'model': 'Model',
}
_TEARDOWN_LABELS = {'remove': 'Remove the checkout', 'keep': 'Keep the checkout'}


class WorkspacesSource:
    """The settings menu: each kind's options, then rows to add and remove kinds; a `plugins.menus.FieldSource`."""

    title = 'Workspaces settings'

    def __init__(self, host: PluginHost[None]) -> None:
        self._host = host

    def rows(self) -> list[FieldRow]:
        rows: list[FieldRow] = []
        for name in self._saved().kinds:
            for field, label in _FIELDS.items():
                info = WorktreeKind.model_fields[field]
                rows.append(
                    FieldRow(
                        key=f'{name}.{field}',
                        label=f'{name}: {label}',
                        description=info.description or '',
                        default=str(info.default),
                        choices=tuple(_TEARDOWN_LABELS) if field == 'teardown' else (),
                        choice_labels=_TEARDOWN_LABELS if field == 'teardown' else {},
                        allow_custom=field != 'teardown',
                    )
                )
        rows += [
            FieldRow(key='add', label='Add a kind', description='Name a new local-worktree kind.', default=''),
            FieldRow(key='remove', label='Remove a kind', description='Workspaces made from it stay.', default=''),
        ]
        return rows

    def current(self, row: FieldRow) -> str:
        kinds = self._saved().kinds
        if row.key in ('add', 'remove'):
            return ', '.join(kinds) or 'no kinds'
        name, _, field = row.key.partition('.')
        return str(kinds[name].model_dump()[field])

    def problem(self, row: FieldRow, text: str) -> str | None:
        try:
            self._changed(row, text)
        except ValidationError as exc:
            return exc.errors()[0]['msg']
        return None

    def apply(self, row: FieldRow, raw: str) -> str:
        self._host.save_settings(self._changed(row, raw))
        return f'{row.label}: {row.display(raw)}.'

    def reset(self, row: FieldRow) -> str:
        self._host.save_settings(self._changed(row, row.default))
        return f'{row.label}: {row.display(row.default) or "(empty)"} (default).'

    async def add(self, dialogs: Dialogs) -> list[str]:
        """Ask for a new kind's name and add it with the default options."""
        kinds = (await to_thread.run_sync(self._saved)).kinds

        def check(name: str) -> str | None:
            if name in kinds:
                return f'There is a kind {name!r} already.'
            try:
                WorkspacesSettings(kinds={name: WorktreeKind()})
            except ValidationError:
                return 'Start with a lowercase letter; then lowercase letters, digits, and hyphens.'
            return None

        name = await dialogs.text(title='New workspace kind', label='Name', placeholder='docs-worktree', check=check)
        if not name:
            return []
        await to_thread.run_sync(self._host.save_settings, WorkspacesSettings(kinds={**kinds, name: WorktreeKind()}))
        return [f'Added workspace kind {name}.']

    async def remove(self, dialogs: Dialogs) -> list[str]:
        """Pick a kind and remove it, once confirmed."""
        kinds = (await to_thread.run_sync(self._saved)).kinds
        name = await dialogs.choose(
            title='Remove a workspace kind', options=[Option(value=name, label=name) for name in kinds]
        )
        if name is None or not await dialogs.confirm(
            title=f'Remove {name}?', yes='Remove', no='Keep', description='Workspaces made from it stay.'
        ):
            return []
        remaining = {key: kind for key, kind in kinds.items() if key != name}
        await to_thread.run_sync(self._host.save_settings, WorkspacesSettings(kinds=remaining))
        return [f'Removed workspace kind {name}.']

    def _saved(self) -> WorkspacesSettings:
        return self._host.settings(WorkspacesSettings)

    def _changed(self, row: FieldRow, text: str) -> WorkspacesSettings:
        saved = self._saved()
        name, _, field = row.key.partition('.')
        kind = WorktreeKind.model_validate(saved.kinds[name].model_dump() | {field: text})
        return WorkspacesSettings(kinds={**saved.kinds, name: kind})
