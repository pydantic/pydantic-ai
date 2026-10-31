"""Named credentials: the `/keys` list in the terminal, and the same operations as a declared settings menu.

Adding, replacing, renaming, and deleting a key each ask through `Dialogs`, so the terminal list (with its
single-key shortcuts) and the web UI's Settings window run the same steps.
"""

from collections.abc import Awaitable, Callable, Container
from dataclasses import dataclass
from functools import partial
from typing import Literal

from anyio import to_thread
from keyring.errors import KeyringError
from termflow.tui import MenuBuilder, MenuItem
from termflow.tui.menu import Menu, MenuResult

from pydantic_ai.exceptions import UserError
from pydantic_clai2.config import api_keys
from pydantic_clai2.config.credential_store import credentials_path
from pydantic_clai2.plugins.menus import Action, Dialogs, FieldRow, Option, SettingsMenu
from pydantic_clai2.ui.menus.dialogs import TerminalDialogs
from pydantic_clai2.ui.menus.field_menu import TERMINAL, Runners, picked, save_and_close_item
from pydantic_clai2.ui.menus.menu_worker import full_terminal, menu_key, run_worker
from pydantic_clai2.ui.menus.slash_search import KeyHandler, slash_search
from pydantic_clai2.ui.rendering.rendering import markdown_style

_NOTE = (
    'Only names are displayed. Values are entered masked.\n'
    'Saved-key connections resolve references on each new turn.\n'
    'Replacing a key updates those connections. Deleting it blocks their next use.\n'
    'Keys in use cannot be renamed. Existing inline credentials are unchanged.'
)
ADD = 'add'
"""The settings menu's row that adds a key; saved key names are upper case, so none can be this."""


@dataclass(frozen=True, kw_only=True)
class KeyAction:
    """A menu action, separate from user-defined key names."""

    action: Literal['add', 'rename', 'delete']
    name: str = ''


def storage() -> str:
    """Where keys are kept, as one line; reads the file system, so call it off the loop."""
    path = credentials_path(account='api-keys')
    if path.is_file():
        return f'Plaintext fallback: {path}'
    return 'Storage: OS keyring preferred; private plaintext fallback if unavailable.'


def _used_by(name: str) -> str:
    try:
        users = api_keys.key_users(name=name)
    except UserError as exc:
        return str(exc)
    return f'Used by {", ".join(users)}.' if users else 'No saved connection uses it.'


class KeysSource:
    """The saved keys' names, with where each is used, and a row to add one; never their values."""

    title = 'API keys'

    def names(self) -> list[str]:
        """The saved keys' names, sorted."""
        return sorted(api_keys.load_keys())

    def rows(self) -> list[FieldRow]:
        """A masked row per key, then the row that adds one."""
        rows = [
            FieldRow(key=name, description=_used_by(name), default='(hidden)', secret=True) for name in self.names()
        ]
        add = FieldRow(
            key=ADD, label='Add an API key', description=f'Save a key under a name.\n{storage()}', default=''
        )
        return [*rows, add]

    def current(self, row: FieldRow) -> str:
        """Do not load or expose the current value."""
        return '' if row.key == ADD else '(hidden)'

    def problem(self, row: FieldRow, text: str) -> str | None:
        """Empty input must not remove a secret."""
        return None if text.strip() else 'An API key is required.'

    def apply(self, row: FieldRow, raw: str) -> str:
        """Replace the key's value."""
        if row.key == ADD:
            raise ValueError('Choose Add an API key to name a new one.')
        return api_keys.save_key(name=row.key, value=raw)

    def reset(self, row: FieldRow) -> str:
        """Deletion asks first, through the key's own action."""
        raise ValueError('An API key is required. Delete it from its menu instead.')


async def add_key(dialogs: Dialogs) -> str:
    """Ask for a new name and its masked value, and save them; empty when backed out of."""
    keys = await to_thread.run_sync(api_keys.load_keys)
    name = await dialogs.text(
        title='API key name (automatically uppercased)', label='Name', check=partial(_new_name_problem, keys)
    )
    if not name:
        return ''
    # Checked again: the terminal checks as you type, but a key of that name may have been saved since.
    if (problem := _new_name_problem(await to_thread.run_sync(api_keys.load_keys), name)) is not None:
        raise ValueError(problem)
    return await replace_key(api_keys.normalize_name(name=name), dialogs)


def _new_name_problem(keys: Container[str], text: str) -> str | None:
    try:
        name = api_keys.normalize_name(name=text)
    except ValueError as exc:
        return str(exc)
    if name in keys:
        return 'That key name already exists. Choose it to replace its value.'
    return None


async def replace_key(name: str, dialogs: Dialogs) -> str:
    """Ask for `name`'s new masked value and save it; empty when backed out of."""
    value = await dialogs.text(
        title=f'New value for {name}', label='Value', placeholder='Enter a new secret', secret=True
    )
    if not value:
        return ''
    return await to_thread.run_sync(partial(api_keys.save_key, name=name, value=value))


async def rename_key(name: str, dialogs: Dialogs) -> str:
    """Ask for a new name for `name` and rename it; a key a connection uses cannot be renamed."""
    new_name = await dialogs.text(
        title=f'Rename {name} (automatically uppercased)', label='Name', initial=name, check=_name_problem
    )
    if not new_name:
        return ''
    return await to_thread.run_sync(partial(api_keys.rename_key, name=name, new_name=new_name))


def _name_problem(text: str) -> str | None:
    try:
        api_keys.normalize_name(name=text)
    except ValueError as exc:
        return str(exc)
    return None


async def delete_key(name: str, dialogs: Dialogs) -> str:
    """Delete `name` once the user confirms; empty when they keep it."""
    if not await dialogs.confirm(
        title=f'Delete {name}?',
        yes='Delete key',
        no='Keep key',
        description='Connections referencing this key will fail on their next use. This cannot be undone.',
    ):
        return ''
    return await to_thread.run_sync(partial(api_keys.delete_key, name=name))


_CHANGES: dict[str, tuple[str, Callable[[str, Dialogs], Awaitable[str]]]] = {
    'replace': ('Replace the value...', replace_key),
    'rename': ('Rename...', rename_key),
    'delete': ('Delete...', delete_key),
}


async def change_key(name: str, dialogs: Dialogs) -> str:
    """Ask whether to replace, rename, or delete `name`, then do it; empty when backed out of."""
    users = await to_thread.run_sync(_used_by, name)
    picked_change = await dialogs.choose(
        title=name,
        options=[Option(value=value, label=label) for value, (label, _) in _CHANGES.items()],
        description=users,
    )
    if picked_change is None:
        return ''
    return await _CHANGES[picked_change][1](name, dialogs)


def _failure(exc: Exception) -> str:
    """What the user is told when a key operation fails; never the key's value."""
    if isinstance(exc, ValueError | UserError):
        return str(exc)
    return 'Credential operation failed. Check your credential backend.'


def _reported(change: Callable[[Dialogs], Awaitable[str]]) -> Action:
    """A settings-menu action that runs `change` and reports a failure as an error the page shows."""

    async def action(dialogs: Dialogs) -> list[str]:
        try:
            message = await change(dialogs)
        except (ValueError, UserError, OSError, KeyringError) as exc:
            raise ValueError(_failure(exc)) from exc
        return [message] if message else []

    return action


def keys_menu() -> SettingsMenu:
    """`/keys` as a declared menu: a row per saved key that replaces, renames, or deletes it, and one that adds one.

    Reads the saved names, so call it off the loop.
    """
    actions: dict[str, Action] = {name: _reported(partial(change_key, name)) for name in KeysSource().names()}
    actions[ADD] = _reported(add_key)
    return SettingsMenu(source=KeysSource(), actions=actions, searchable=False)


def build_keys_menu(*, names: list[str], message: str = '') -> Menu:
    """Build a name-only menu without reading credentials or touching the terminal."""
    items = [MenuItem(name, value=name) for name in sorted(names)]
    if not names:
        items.append(MenuItem('No saved API keys', disabled=True))
    items.append(MenuItem('Add API key...', value=KeyAction(action='add')))
    if message:
        items.append(MenuItem(message, disabled=True))
    items.append(save_and_close_item())

    def action(kind: Literal['rename', 'delete'], item: MenuItem) -> MenuResult | None:
        if not item.disabled and isinstance(item.value, str):
            return MenuResult(item=MenuItem(item.label, value=KeyAction(action=kind, name=item.value)))
        return None

    builder = (
        MenuBuilder('API keys')
        .style(markdown_style())
        .items(items)
        .preview(lambda item: f'{item.label}\n\n{_NOTE}\n\n{message}')
    )
    hotkeys: dict[str, KeyHandler] = {
        'a': lambda menu, item: MenuResult(item=MenuItem('Add', value=KeyAction(action='add'))),
        'r': lambda menu, item: action('rename', item),
        'd': lambda menu, item: action('delete', item),
    }
    footer = 'Enter replace - a add - r rename - d delete - Esc close'
    return slash_search(builder, footer=footer, key_source=menu_key, hotkeys=hotkeys)


async def _act(value: object, dialogs: Dialogs) -> str:
    if isinstance(value, KeyAction):
        if value.action == 'delete':
            return await delete_key(value.name, dialogs)
        if value.action == 'rename':
            return await rename_key(value.name, dialogs)
        return await add_key(dialogs)
    if isinstance(value, str):
        return await replace_key(value, dialogs)
    return ''


def _listing(source: KeysSource) -> tuple[list[str], str]:
    try:
        return source.names(), storage()
    except (ValueError, UserError, OSError, KeyringError):
        return [], 'Cannot read API keys. Check your credential backend.'


async def run_keys_flow(*, runners: Runners = TERMINAL) -> None:
    """Apply each action immediately and reopen the list with feedback inside it."""
    source = KeysSource()
    dialogs = TerminalDialogs(runners)
    message = ''
    with full_terminal():
        while True:
            names, status = await to_thread.run_sync(_listing, source)
            menu = build_keys_menu(names=names, message=f'{message}\n{status}')
            item = picked(await run_worker(lambda: runners.run_list(menu)))
            if item is None:
                return
            try:
                message = await _act(item.value, dialogs)
            except (ValueError, UserError, OSError, KeyringError) as exc:
                message = _failure(exc)


async def keys_command(args: list[str]) -> str:
    """Open the manager; do not accept secret values on the command line."""
    if args:
        raise ValueError('Usage: /keys (enter secrets only in the masked editor)')
    await run_keys_flow()
    return ''
