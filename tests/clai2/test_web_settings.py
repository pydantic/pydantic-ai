"""Commands' own menus in the web UI: `/set`, `/theme`, and `/spinner` shown in the page instead of the terminal."""

import io
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from pydantic import JsonValue, TypeAdapter
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_clai2.builtin_plugins.web.protocol import (
    DialogOpened,
    Notice,
    SettingsChanged,
    StatusChanged,
    ThemeChanged,
)
from pydantic_clai2.commands import Command, Commands
from pydantic_clai2.plugins import PluginHost
from pydantic_clai2.plugins.menus import Dialogs, FieldRow, SettingsMenu
from tests.clai2.test_web import TOKEN, MakeWeb, Web, make_web as make_web, receive_until

SOCKET = f'/ws?token={TOKEN}'
ORIGIN = {'origin': 'http://testserver'}


def fields(web_: Web, command: str) -> dict[str, dict[str, JsonValue]]:
    response = web_.get(f'/api/commands/{command}/fields')
    assert response.status_code == 200, response.text
    menu = TypeAdapter(dict[str, JsonValue]).validate_python(response.json())
    rows = TypeAdapter(list[dict[str, JsonValue]]).validate_python(menu['fields'])
    return {str(row['key']): row for row in rows}


def terminal(web_: Web) -> str:
    file = web_.shell.console.file
    assert isinstance(file, io.StringIO)
    return file.getvalue()


def test_a_bare_command_with_a_menu_opens_it_in_the_page(make_web: MakeWeb) -> None:
    """`/set` typed in the browser runs nothing: the page shows its menu. With arguments it runs as typed."""
    web_ = make_web(Agent(TestModel()))
    with web_.client.websocket_connect(SOCKET, headers=ORIGIN) as socket:
        socket.receive_text()
        running = web_.run()
        assert web_.post('/api/input', {'text': ' /set '}).json() == {'menu': 'set'}
        assert web_.post('/api/input', {'text': '/settings'}).json() == {'menu': 'settings'}
        assert web_.post('/api/input', {'text': '/theme'}).json() == {'menu': 'theme'}
        # A command without a menu, an argument, or a path runs as it always did.
        assert web_.post('/api/input', {'text': '/set display.thinking false'}).json() == {'menu': None}
        assert receive_until(socket, Notice)[1].command == '/set display.thinking false', (
            'the bare ones entered nothing'
        )
        # A command typed anywhere may change a setting, so open menus read theirs again.
        receive_until(socket, SettingsChanged)
        assert web_.shell.context.settings.thinking is False
        assert web_.post('/api/input', {'text': '/usr/bin'}).json() == {'menu': None}
        web_.post('/api/input', {'text': '/exit'})
        running.result(timeout=10)


def test_the_set_menu_edits_settings_from_the_page(make_web: MakeWeb) -> None:
    """`/set`'s rows, edited from the page, save and apply as the terminal menu does, and the terminal says so."""
    web_ = make_web(Agent(TestModel()))
    rows = fields(web_, 'set')
    assert 'run.instructions' not in rows, '/system_prompt edits it'
    assert rows['display.thinking'] == {
        'key': 'display.thinking',
        'label': 'display.thinking',
        'description': rows['display.thinking']['description'],
        'value': 'true',
        'default': 'true',
        'choices': [
            {'value': 'true', 'label': 'true', 'description': ''},
            {'value': 'false', 'label': 'false', 'description': ''},
        ],
        'allow_custom': True,
        'secret': False,
        'note': '',
        'action': False,
    }
    assert fields(web_, 'settings').keys() == rows.keys()
    with web_.client.websocket_connect(SOCKET, headers=ORIGIN) as socket:
        socket.receive_text()
        edited = web_.post('/api/commands/set/fields/display.thinking', {'value': 'false'})
        assert edited.status_code == 200, edited.text
        assert edited.json()['messages'] == ['Saved display.thinking. Applied.']
        shown = {field['key']: field['value'] for field in edited.json()['menu']['fields']}
        assert shown['display.thinking'] == 'false'
        assert web_.shell.context.settings.thinking is False
        assert web_.shell.context.store.load().thinking is False, 'saved, as /set saves it'
        messages, _ = receive_until(socket, SettingsChanged)
        assert any(isinstance(message, StatusChanged) for message in messages)
        assert 'Saved display.thinking. Applied.' in terminal(web_)
        # A value the setting rejects says why by its row and changes nothing.
        rejected = web_.post('/api/commands/set/fields/run.request_limit', {'value': 'many'})
        assert rejected.status_code == 400 and rejected.json()['detail']
        # Empty text resets, as an empty edit in the terminal does.
        reset = web_.post('/api/commands/set/fields/display.thinking', {'value': ''})
        assert reset.json()['messages'] == ['Reset display.thinking. Applied.']
        assert web_.shell.context.settings.thinking is True
    assert web_.post('/api/commands/set/fields/nope', {'value': '1'}).status_code == 404
    assert web_.post('/api/commands/set/fields/display.thinking/action').status_code == 404
    # A command with no menu, or none at all, has no fields here.
    assert web_.get('/api/commands/cost/fields').status_code == 404
    assert web_.get('/api/commands/nope/fields').status_code == 404


def test_theme_and_spinner_menus_are_their_rows_of_set(make_web: MakeWeb) -> None:
    web_ = make_web(Agent(TestModel()))
    assert list(fields(web_, 'theme')) == ['display.theme']
    assert list(fields(web_, 'spinner')) == ['display.spinner']
    assert web_.get('/api/commands/theme/fields').json()['title'] == 'Theme'
    with web_.client.websocket_connect(SOCKET, headers=ORIGIN) as socket:
        socket.receive_text()
        assert web_.post('/api/commands/theme/fields/display.theme', {'value': 'tokyo_night'}).status_code == 200
        assert receive_until(socket, ThemeChanged)[1].theme.name == 'tokyo_night'
        web_.post('/api/commands/theme/fields/display.theme', {'value': ''})
        assert receive_until(socket, ThemeChanged)[1].theme.name == 'default'


@dataclass(frozen=True, kw_only=True)
class Counter:
    """A one-row menu whose action asks before it counts."""

    counted: list[str]
    title: str = 'Counter'

    def rows(self) -> Sequence[FieldRow]:
        return [FieldRow(key='count', description='How many.', default='0')]

    def current(self, row: FieldRow) -> str:
        return str(len(self.counted))

    def problem(self, row: FieldRow, text: str) -> str | None:
        return None

    def apply(self, row: FieldRow, raw: str) -> str:
        return 'unused'

    def reset(self, row: FieldRow) -> str:
        return 'unused'


def test_a_command_menu_action_asks_through_the_page(make_web: MakeWeb) -> None:
    web_ = make_web(Agent(TestModel()))
    counted: list[str] = []

    async def count(dialogs: Dialogs) -> list[str]:
        if not await dialogs.confirm(title='Count one more?', yes='Count', no='Keep'):
            return []
        counted.append('one')
        return [f'Counted {len(counted)}.']

    menu = SettingsMenu(source=Counter(counted=counted), actions={'count': count})
    web_.shell.commands.register(Command(name='counter', description='Count.', handler=lambda _: '', menu=lambda: menu))
    assert fields(web_, 'counter')['count']['action'] is True
    with web_.client.websocket_connect(SOCKET, headers=ORIGIN) as socket:
        socket.receive_text()
        with ThreadPoolExecutor(1) as pool:
            action = pool.submit(web_.post, '/api/commands/counter/fields/count/action')
            dialog = receive_until(socket, DialogOpened)[1].dialog
            assert (dialog.kind, dialog.title) == ('confirm', 'Count one more?')
            assert web_.post(f'/api/dialogs/{dialog.id}', {'value': 'yes'}).status_code == 204
            done = action.result(timeout=10).json()
    assert done['messages'] == ['Counted 1.']
    assert done['menu']['fields'][0]['value'] == '1'
    assert 'Counted 1.' in terminal(web_)


def test_a_host_finds_only_declared_available_menus() -> None:
    commands = Commands()
    menu = SettingsMenu(source=Counter(counted=[]))
    commands.register(Command(name='shown', description='', handler=lambda _: '', menu=lambda: menu))
    commands.register(Command(name='plain', description='', handler=lambda _: ''))
    commands.register(
        Command(name='hidden', description='', handler=lambda _: '', menu=lambda: menu, available=lambda: False)
    )
    host = PluginHost[None](name='web', console=Console(file=io.StringIO()), settings={}, commands=commands)
    assert host.command_menu('shown') is menu
    assert host.command_menu('plain') is None
    assert host.command_menu('hidden') is None
    assert host.command_menu('missing') is None
    assert PluginHost[None](name='web', console=Console(file=io.StringIO()), settings={}).command_menu('shown') is None
    assert commands.get('hidden') is None and commands.get('plain') is not None
