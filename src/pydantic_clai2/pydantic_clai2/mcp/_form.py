"""The `/mcp install` and `/mcp edit` form, ported from Code Puppy's custom server form.

A menu of fields: the server name, its type (stdio, http, or sse), and its JSON configuration,
edited in `$VISUAL`/`$EDITOR` with a one-line fallback. Remote servers also get an OAuth toggle,
which Code Puppy's HTTP wizard asks about. The preview pane shows the JSON and whether it is valid.
Everything here is synchronous and runs in a menu worker thread.
"""

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import get_args

from pydantic import HttpUrl, JsonValue, TypeAdapter, ValidationError
from termflow.tui import MenuBuilder, MenuItem, TextInputBuilder  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.menu import Menu  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.textinput import TextInput  # pyright: ignore[reportMissingTypeStubs]

from .._rendering import markdown_style
from ..field_menu import TERMINAL, Runners, first_error
from ..menu_worker import menu_key
from ._settings import OAUTH_TIMEOUT, Server, ServerName, ServerType, StdioServer, missing
from ._store import MCPStore
from ._tokens import TokenStore

SERVER_TYPES: tuple[ServerType, ...] = get_args(ServerType)
TYPE_DESCRIPTIONS: dict[ServerType, str] = {
    'stdio': 'Local command (npx, python, uvx) via stdin/stdout',
    'http': 'Streamable HTTP endpoint implementing the MCP protocol',
    'sse': 'Server-Sent Events endpoint (older MCP servers)',
}
EXAMPLES: dict[ServerType, str] = {
    'stdio': json.dumps(
        {
            'type': 'stdio',
            'command': 'npx',
            'args': ['-y', '@modelcontextprotocol/server-filesystem', '/path/to/dir'],
            'env': {'NODE_ENV': 'production'},
            'timeout': 30,
        },
        indent=2,
    ),
    'http': json.dumps(
        {
            'type': 'http',
            'url': 'http://localhost:8080/mcp',
            'headers': {'Authorization': 'Bearer $MY_API_KEY'},
            'timeout': 30,
        },
        indent=2,
    ),
    'sse': json.dumps(
        {'type': 'sse', 'url': 'http://localhost:8080/sse', 'headers': {'Authorization': 'Bearer $MY_API_KEY'}},
        indent=2,
    ),
}

NAME, TYPE, TARGET, OAUTH, CONFIG, EXAMPLE, SAVE, CANCEL = (
    'name',
    'type',
    'target',
    'oauth',
    'json',
    'example',
    'save',
    'cancel',
)
Editor = Callable[[str], str | None]
_NAME: TypeAdapter[str] = TypeAdapter(ServerName)
_SERVER: TypeAdapter[Server] = TypeAdapter(Server)
_OBJECT: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])
_URL: TypeAdapter[HttpUrl] = TypeAdapter(HttpUrl)


class ServerForm:
    """Form state and persistence for adding or editing one user server."""

    def __init__(self, store: MCPStore, *, name: str = '', server: Server | None = None) -> None:
        """Pass the saved `server` to edit it; without one the form adds a new server."""
        self.store = store
        self.editing = server is not None
        self.original = name
        self.name = name
        self.type: ServerType = server.type if server else 'stdio'
        self.config = (
            json.dumps(server.model_dump(mode='json', exclude_defaults=True) | {'type': server.type}, indent=2)
            if server
            else EXAMPLES['stdio']
        )
        self.status: str | None = None
        self._raised_from: int | float | None = None

    def name_problem(self, name: str) -> str | None:
        """Why `name` cannot be saved, or `None`."""
        name = name.strip()
        if not name:
            return 'Server name is required'
        try:
            _NAME.validate_python(name)
        except ValidationError:
            return 'Start with a letter; then letters, digits, or hyphens (max 64)'
        if name != self.original and name in self.store.load().servers:
            return f'{name} already exists'
        return None

    def parse(self) -> Server:
        """The configuration as a server of the selected type; `ValueError` explains a problem."""
        try:
            data = _OBJECT.validate_json(self.config)
        except ValidationError as exc:
            raise ValueError(f'Invalid JSON: {first_error(exc)}') from None
        try:
            return _SERVER.validate_python({**data, 'type': self.type})
        except ValidationError as exc:
            error = exc.errors()[0]
            location = '.'.join(str(part) for part in error['loc'][1:])
            raise ValueError(f'{location}: {error["msg"]}' if location else error['msg']) from None

    def problem(self) -> str | None:
        """Why the configuration cannot be saved, or `None`."""
        try:
            self.parse()
        except ValueError as exc:
            return str(exc)
        return None

    @property
    def oauth(self) -> bool:
        """Whether the configuration asks for OAuth sign-in."""
        try:
            return _OBJECT.validate_json(self.config).get('auth') == 'oauth'
        except ValidationError:
            return False

    def toggle_oauth(self) -> None:
        """Switch browser sign-in on or off.

        A timeout too short for a browser sign-in is raised while OAuth is on, and restored when it is switched off.
        """
        try:
            data = _OBJECT.validate_json(self.config)
        except ValidationError:
            self.status = 'Fix the JSON before switching OAuth'
            return
        timeout = data.get('timeout')
        if data.get('auth') == 'oauth':
            data.pop('auth')
            if self._raised_from is not None and timeout == OAUTH_TIMEOUT:
                data['timeout'] = self._raised_from
            self._raised_from = None
            self.status = None
        else:
            data['auth'] = 'oauth'
            if isinstance(timeout, int | float) and timeout < OAUTH_TIMEOUT:
                self._raised_from, data['timeout'] = timeout, OAUTH_TIMEOUT
            self.status = None
            headers = data.get('headers')
            if isinstance(headers, dict) and any(key.lower() == 'authorization' for key in headers):
                kept = {key: value for key, value in headers.items() if key.lower() != 'authorization'}
                data['headers'] = kept
                if not kept:
                    data.pop('headers')
                self.status = 'Removed the Authorization header; OAuth sets it after sign-in'
        self.config = json.dumps(data, indent=2)

    def select_type(self, new: ServerType) -> None:
        """Swap in the new type's example only while the configuration is the old, untouched example."""
        if self.config.strip() == EXAMPLES[self.type].strip():
            self.config = EXAMPLES[new]
        self.type = new

    def load_example(self) -> None:
        """Replace the configuration with the selected type's example."""
        self.config = EXAMPLES[self.type]
        self.status = None

    def save(self) -> bool:
        """Persist the server; on failure `status` says why."""
        problem = self.name_problem(self.name) or self.problem()
        if problem:
            self.status = f'Save failed: {problem}'
            return False
        name = self.name.strip()
        self.store.put(name, self.parse())
        if self.editing and self.original and name != self.original:
            self.store.delete(self.original)
            TokenStore(self.original).forget()  # Tokens belong to the old name; the renamed server signs in again.
        self.name = name
        return True

    @property
    def target_label(self) -> str:
        """`URL` for remote servers, `Command` for stdio ones."""
        return 'Command' if self.type == 'stdio' else 'URL'

    def target(self) -> str:
        """The URL, or the command line with its arguments, read from the JSON."""
        try:
            data = _OBJECT.validate_json(self.config)
        except ValidationError:
            return ''
        if self.type != 'stdio':
            url = data.get('url')
            return url if isinstance(url, str) else ''
        command, args = data.get('command'), data.get('args')
        if not isinstance(command, str):
            return ''
        extra = [arg for arg in args if isinstance(arg, str)] if isinstance(args, list) else []
        return shlex.join([command, *extra])

    def target_problem(self, text: str) -> str | None:
        """Why `text` is not a usable URL or command line, or `None`."""
        text = text.strip()
        if self.type == 'stdio':
            try:
                return None if shlex.split(text) else 'Enter the program to run, then its arguments'
            except ValueError as exc:
                return str(exc)
        try:
            _URL.validate_python(text)
        except ValidationError:
            return 'Enter an http:// or https:// URL'
        return None

    def set_target(self, text: str) -> None:
        """Write the URL, or split the command line into `command` and `args`."""
        try:
            data = _OBJECT.validate_json(self.config)
        except ValidationError:
            self.status = f'Fix the JSON before editing the {self.target_label}'
            return
        if self.type == 'stdio':
            command, *args = shlex.split(text)
            data['command'] = command
            data['args'] = list[JsonValue](args)
            if not args:
                data.pop('args')
        else:
            data['url'] = text.strip()
        self.config = json.dumps(data, indent=2)
        self.status = None

    def items(self) -> list[MenuItem]:
        """The form's rows; the OAuth row only for remote servers."""
        rows = [
            MenuItem(f'Server Name: {self.name or "(not set)"}', value=NAME),
            MenuItem(f'Server Type: {self.type}', value=TYPE),
            MenuItem(f'{self.target_label}: {self.target() or "(not set)"}', value=TARGET),
        ]
        if self.type != 'stdio':
            rows.append(MenuItem(f'OAuth sign-in: {"on" if self.oauth else "off"}', value=OAUTH))
        return [
            *rows,
            MenuItem(f'JSON Configuration ({"INVALID" if self.problem() else "valid"})', value=CONFIG),
            MenuItem(f'Load example for {self.type}', value=EXAMPLE),
            MenuItem('Save changes' if self.editing else 'Save & Install', value=SAVE),
            MenuItem('Cancel', value=CANCEL),
        ]

    def preview(self) -> str:
        """The right-hand pane: name, type, JSON, and validation status."""
        problem = self.problem()
        lines = [
            'Edit MCP Server' if self.editing else 'Add Custom MCP Server',
            '',
            f'Name   {self.name or "(not set)"}',
            f'Type   {self.type} - {TYPE_DESCRIPTIONS[self.type]}',
            '',
            *self.config.splitlines(),
            '',
            f'Invalid: {problem}' if problem else 'Configuration is valid',
        ]
        if self.status:
            lines.append(self.status)
        lines += ['', 'Use $VAR in env or headers to read secrets from your environment.']
        if self.type != 'stdio':
            lines.append('OAuth signs in through your browser when the server connects.')
        return '\n'.join(lines)


def build_form_menu(form: ServerForm, initial: int = 0) -> Menu:
    """The field list with the live preview."""
    return (
        MenuBuilder('Edit MCP Server' if form.editing else 'Add Custom MCP Server')
        .style(markdown_style())
        .items(form.items())
        .list_width(38)
        .initial_index(initial)
        .preview(lambda _: form.preview())
        .footer_hint('Enter edit field - Esc cancel')
        .key_source(menu_key)
        .build()
    )


def build_type_menu(form: ServerForm) -> Menu:
    """stdio, http, or sse, with descriptions."""
    return (
        MenuBuilder('Server Type')
        .style(markdown_style())
        .items([MenuItem(kind, value=kind, description=TYPE_DESCRIPTIONS[kind]) for kind in SERVER_TYPES])
        .initial_index(SERVER_TYPES.index(form.type))
        .footer_hint('Enter select - Esc keep current')
        .key_source(menu_key)
        .build()
    )


def build_name_input(form: ServerForm) -> TextInput:
    """The server name, validated as you type."""
    return (
        TextInputBuilder('Server Name')
        .style(markdown_style())
        .prompt('Name: ')
        .initial(form.name)
        .placeholder('letters, digits, and hyphens')
        .validator(form.name_problem)
        .footer_hint('Enter save - Esc cancel')
        .key_source(menu_key)
        .build()
    )


def build_target_input(form: ServerForm) -> TextInput:
    """The URL, or the command line for a stdio server."""
    remote = form.type != 'stdio'
    return (
        TextInputBuilder(form.target_label)
        .style(markdown_style())
        .prompt(f'{form.target_label}: ')
        .initial(form.target())
        .placeholder('https://example.com/mcp' if remote else 'uvx my-mcp-server --flag')
        .validator(form.target_problem)
        .footer_hint('Enter save - Esc cancel')
        .key_source(menu_key)
        .build()
    )


def build_json_input(form: ServerForm) -> TextInput:
    """The one-line JSON fallback when no editor could run."""
    try:
        compact = json.dumps(json.loads(form.config))
    except json.JSONDecodeError:
        compact = form.config
    return (
        TextInputBuilder('JSON Configuration (single line)')
        .style(markdown_style())
        .prompt('JSON: ')
        .initial(compact)
        .validator(_json_problem)
        .footer_hint('Enter save - Esc cancel')
        .key_source(menu_key)
        .build()
    )


def _json_problem(text: str) -> str | None:
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        return f'Invalid JSON: {exc.msg}'
    return None


def edit_in_editor(initial: str) -> str | None:
    """Open `$VISUAL` or `$EDITOR` (default `vi`) on the JSON; `None` when it could not run."""
    handle, name = tempfile.mkstemp(suffix='.json', prefix='mcp_server_')
    path = Path(name)
    try:
        editor = shlex.split(os.environ.get('VISUAL') or os.environ.get('EDITOR') or 'vi')
        with os.fdopen(handle, 'w') as file:
            file.write(initial)
        print('\x1b[2J\x1b[H', end='', flush=True, file=sys.__stdout__)
        if subprocess.call([*editor, name]) != 0:
            return None
        return path.read_text()
    except (OSError, ValueError):  # ValueError: an unparsable $EDITOR; the one-line input takes over.
        return None
    finally:
        path.unlink(missing_ok=True)


def run_form(form: ServerForm, runners: Runners = TERMINAL, editor: Editor = edit_in_editor) -> bool:
    """The field-edit loop, until Save succeeds (`True`) or Cancel/Esc (`False`)."""
    cursor = 0
    while True:
        result = runners.run_list(build_form_menu(form, cursor))
        if result.cancelled or result.item is None or result.item.value == CANCEL:
            return False
        value = result.item.value
        values = [item.value for item in form.items()]
        cursor = values.index(value) if value in values else 0
        if value == SAVE:
            if form.save():
                return True
        else:
            _edit_row(form, value, runners, editor)


def _edit_row(form: ServerForm, row: object, runners: Runners, editor: Editor) -> None:
    if row == NAME:
        typed = runners.run_text(build_name_input(form))
        if not typed.cancelled and isinstance(typed.value, str):
            form.name = typed.value.strip()
    elif row == TYPE:
        picked = runners.run_choice(build_type_menu(form))
        if not picked.cancelled and picked.item is not None and picked.item.value in SERVER_TYPES:
            form.select_type(picked.item.value)
    elif row == TARGET:
        typed = runners.run_text(build_target_input(form))
        if not typed.cancelled and isinstance(typed.value, str) and not form.target_problem(typed.value):
            form.set_target(typed.value)
    elif row == OAUTH:
        form.toggle_oauth()
    elif row == CONFIG:
        edited = editor(form.config)
        if edited is None:
            typed = runners.run_text(build_json_input(form))
            if not typed.cancelled and isinstance(typed.value, str) and not _json_problem(typed.value):
                edited = json.dumps(json.loads(typed.value), indent=2)
        if edited is not None:
            form.config = edited
    elif row == EXAMPLE:
        form.load_example()


def saved_message(form: ServerForm, server: Server) -> str:
    """Next steps after saving, plus anything that will stop the server from connecting."""
    verb = 'Updated' if form.editing else 'Added'
    lines = [f'{verb} {form.name}. The agent can use it on your next prompt; /mcp start {form.name} connects now.']
    if isinstance(server, StdioServer) and shutil.which(server.command) is None:
        lines.append(f'{server.command} is not on PATH; install it before starting {form.name}.')
    unset = missing(server)
    if unset:
        lines.append(f'Set {", ".join(unset)} in your environment; the saved file only holds the reference.')
    return '\n'.join(lines)


def install_form(store: MCPStore, runners: Runners = TERMINAL, editor: Editor = edit_in_editor) -> str:
    """`/mcp install`: add a server through the form."""
    form = ServerForm(store)
    if not run_form(form, runners, editor):
        return 'Exited custom server form.'
    return saved_message(form, store.load().servers[form.name])


def edit_form(
    store: MCPStore, name: str, runners: Runners = TERMINAL, editor: Editor = edit_in_editor
) -> tuple[str, str] | None:
    """`/mcp edit NAME`: the same form, prefilled. The saved name and message, or `None` when cancelled."""
    form = ServerForm(store, name=name, server=store.load().servers[name])
    if not run_form(form, runners, editor):
        return None
    return form.name, saved_message(form, store.load().servers[form.name])
