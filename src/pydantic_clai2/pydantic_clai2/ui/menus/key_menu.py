"""Full-screen management of named credentials, with masked shared field editing."""

from dataclasses import dataclass
from typing import Literal

from keyring.errors import KeyringError
from termflow.tui import MenuBuilder, MenuItem, TextInputBuilder
from termflow.tui.menu import Menu, MenuResult

from pydantic_ai.exceptions import UserError
from pydantic_clai2.config import api_keys
from pydantic_clai2.config.credential_store import credentials_path
from pydantic_clai2.ui.menus.field_menu import TERMINAL, FieldMenu, FieldRow, Runners, picked, save_and_close_item
from pydantic_clai2.ui.menus.menu_worker import menu_key, run_worker
from pydantic_clai2.ui.menus.slash_search import KeyHandler, slash_search
from pydantic_clai2.ui.rendering._rendering import markdown_style

_NOTE = (
    'Only names are displayed. Values are entered masked.\n'
    'Saved-key connections resolve references on each new turn.\n'
    'Replacing a key updates those connections. Deleting it blocks their next use.\n'
    'Keys in use cannot be renamed. Existing inline credentials are unchanged.'
)


@dataclass(frozen=True, kw_only=True)
class KeyAction:
    """A menu action, separate from user-defined key names."""

    action: Literal['add', 'rename', 'delete']
    name: str = ''


class KeysSource:
    """The shared field editor sees names and hidden placeholders, not secrets."""

    title = 'API keys'

    def rows(self) -> list[FieldRow]:
        """Read names only for display."""
        return [self.row(name=name) for name in sorted(api_keys.load_keys())]

    def row(self, *, name: str) -> FieldRow:
        """Build a masked field, also usable before a new key is saved."""
        return FieldRow(key=name, description=_NOTE, default='(hidden)', secret=True)

    def current(self, row: FieldRow) -> str:
        """Do not load or expose the current value."""
        return '(hidden)'

    def problem(self, row: FieldRow, text: str) -> str | None:
        """Empty input must not remove a secret."""
        return None if text.strip() else 'An API key is required.'

    def apply(self, row: FieldRow, raw: str) -> str:
        """Persist through the same backend as the compatibility command."""
        return api_keys.save_key(name=row.key, value=raw)

    def reset(self, row: FieldRow) -> str:
        """Deletion requires the explicit confirmation flow."""
        raise ValueError('An API key is required. Use d to delete a saved key.')


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
    footer = 'enter replace · a add · r rename · d delete · esc close'
    return slash_search(builder, footer=footer, key_source=menu_key, hotkeys=hotkeys)


def _name(*, runners: Runners) -> str | None:
    result = runners.run_text(
        TextInputBuilder('API key name (automatically uppercased)')
        .style(markdown_style())
        .prompt('Name: ')
        .footer_hint('Enter continue - Esc cancel')
        .key_source(menu_key)
        .build()
    )
    if result.cancelled or not isinstance(result.value, str):
        return None
    return api_keys.normalize_name(name=result.value)


def _confirm_delete(*, name: str, runners: Runners) -> bool:
    result = runners.run_choice(
        MenuBuilder(f'Delete {name}?')
        .style(markdown_style())
        .items([MenuItem('Keep key', value=False), MenuItem('Delete key', value=True)])
        .preview(lambda item: 'Connections referencing this key will fail on their next use. This cannot be undone.')
        .footer_hint('Enter select - Esc keep key')
        .key_source(menu_key)
        .build()
    )
    return not result.cancelled and result.item is not None and result.item.value is True


def _act(*, value: object, source: KeysSource, runners: Runners) -> str:
    if isinstance(value, KeyAction):
        if value.action == 'delete':
            return api_keys.delete_key(name=value.name) if _confirm_delete(name=value.name, runners=runners) else ''
        name = _name(runners=runners)
        if name is None:
            return ''
        if value.action == 'rename':
            return api_keys.rename_key(name=value.name, new_name=name)
        if name in api_keys.load_keys():
            raise ValueError('That key name already exists. Select it and press Enter to replace its value.')
    elif isinstance(value, str):
        name = value
    else:
        return ''
    editor = FieldMenu(source)
    row = source.row(name=name)
    result = runners.run_text(editor.build_editor(row))
    if result.cancelled or not isinstance(result.value, str):
        return ''
    return editor.apply(row, result.value)


def run_keys_flow(*, runners: Runners = TERMINAL) -> None:
    """Apply each action immediately and reopen the list with feedback inside it."""
    source = KeysSource()
    message = ''
    while True:
        try:
            names = [row.key for row in source.rows()]
        except (ValueError, UserError, OSError, KeyringError):
            names = []
            message = 'Cannot read API keys. Check your credential backend.'
        path = credentials_path(account='api-keys')
        storage = (
            f'Plaintext fallback: {path}'
            if path.is_file()
            else 'Storage: OS keyring preferred; private plaintext fallback if unavailable.'
        )
        item = picked(runners.run_list(build_keys_menu(names=names, message=f'{message}\n{storage}')))
        if item is None:
            return
        try:
            message = _act(value=item.value, source=source, runners=runners)
        except (ValueError, UserError) as exc:
            message = str(exc)
        except (OSError, KeyringError):
            message = 'Credential operation failed. Check your credential backend.'


async def keys_command(args: list[str]) -> str:
    """Open the manager; do not accept secret values on the command line."""
    if args:
        raise ValueError('Usage: /keys (enter secrets only in the masked editor)')
    await run_worker(run_keys_flow)
    return ''
