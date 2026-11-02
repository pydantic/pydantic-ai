"""Commands' own menus in the web UI: `/set`, `/theme`, `/spinner`, and `/mcp` shown in the page instead of the terminal."""

import io
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from pydantic import JsonValue, TypeAdapter
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_clai2.builtin_plugins.web.protocol import (
    PROTOCOL,
    DialogOpened,
    Notice,
    SettingsChanged,
    StatusChanged,
    ThemeChanged,
    TurnEnded,
)
from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.cli.self_update import Installed, Updates
from pydantic_clai2.commands import Command, Commands
from pydantic_clai2.config import Settings, UpdateChannel
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.mcp import MCPPlugin, MCPSettings, MCPStore
from pydantic_clai2.plugins import PluginHost, collect
from pydantic_clai2.plugins.menus import Dialogs, FieldRow, SettingsMenu
from pydantic_clai2.runtime import session_runtime
from tests.clai2.test_web import TOKEN, MakeWeb, Web, make_web as make_web, receive_until

SOCKET = f'/ws?token={TOKEN}&protocol={PROTOCOL}'
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
        'menu': False,
        'button': '',
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
    """A one-row count, set by typing it or by an action that asks before it counts one more."""

    counted: list[str]
    title: str = 'Counter'

    def rows(self) -> Sequence[FieldRow]:
        return [FieldRow(key='count', description='How many.', default='0')]

    def current(self, row: FieldRow) -> str:
        return str(len(self.counted))

    def problem(self, row: FieldRow, text: str) -> str | None:
        return None if text.isdigit() else 'Type a number.'

    def apply(self, row: FieldRow, raw: str) -> str:
        self.counted[:] = ['one'] * int(raw)
        return f'Set to {raw}.'

    def reset(self, row: FieldRow) -> str:
        self.counted.clear()
        return 'Reset.'


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
    # A row of a command's own menu edits through its source, as in the terminal.
    assert web_.post('/api/commands/counter/fields/count', {'value': 'x'}).json()['detail'] == 'Type a number.'
    assert web_.post('/api/commands/counter/fields/count', {'value': '3'}).json()['messages'] == ['Set to 3.']
    assert web_.post('/api/commands/counter/fields/count', {'value': ''}).json()['messages'] == ['Reset.']
    with web_.client.websocket_connect(SOCKET, headers=ORIGIN) as socket:
        socket.receive_text()
        with ThreadPoolExecutor(1) as pool:
            for answer in ('no', 'yes'):
                action = pool.submit(web_.post, '/api/commands/counter/fields/count/action')
                dialog = receive_until(socket, DialogOpened)[1].dialog
                assert (dialog.kind, dialog.title) == ('confirm', 'Count one more?')
                assert web_.post(f'/api/dialogs/{dialog.id}', {'value': answer}).status_code == 204
                done = action.result(timeout=10).json()
                assert done['messages'] == ([] if answer == 'no' else ['Counted 1.'])
    assert done['menu']['fields'][0]['value'] == '1'
    assert 'Counted 1.' in terminal(web_)


def test_settings_save_one_at_a_time(tmp_path: Path) -> None:
    """The page edits `/set` from a worker thread while the terminal may too: each save holds the context's lock."""
    held: list[bool] = []

    def applied(key: str, settings: Settings) -> None:
        held.append(context.edits.locked())

    context = CommandContext(settings=Settings(), store=SettingsStore(tmp_path / 'config.db'), apply_setting=applied)
    context.set_setting(['display.thinking', 'false'])
    context.reset_setting('display.thinking')
    assert held == [True, True]
    assert not context.edits.locked()


def test_mcp_servers_are_managed_from_the_page(make_web: MakeWeb, tmp_path: Path) -> None:
    """Bare `/mcp` opens its menu in the page, whose actions run `/mcp`'s own form and subcommands there."""
    web_ = make_web(Agent(TestModel()))
    store = MCPStore(tmp_path / 'mcp', workspace=tmp_path)
    host = PluginHost[None](name='mcp', console=Console(file=io.StringIO()), settings={})
    web_.shell.commands.register_many(collect(MCPPlugin(host, MCPSettings(), store=store)).commands)
    assert list(fields(web_, 'mcp')) == ['add']
    with web_.client.websocket_connect(SOCKET, headers=ORIGIN) as socket:
        socket.receive_text()
        assert web_.post('/api/input', {'text': '/mcp'}).json() == {'menu': 'mcp'}
        with ThreadPoolExecutor(1) as pool:
            action = pool.submit(web_.post, '/api/commands/mcp/fields/add/action')
            answers = ['name', 'files', 'json', '{"command": "uvx", "args": ["files"]}', 'save']
            for answer in answers:
                dialog = receive_until(socket, DialogOpened)[1].dialog
                if dialog.kind == 'text' and dialog.label == 'JSON':
                    assert '\n' in dialog.initial, 'the page edits the indented JSON'
                    rejected = web_.post(f'/api/dialogs/{dialog.id}', {'value': '{broken'})
                    assert rejected.status_code == 400 and rejected.json()['detail'].startswith('Invalid JSON')
                assert web_.post(f'/api/dialogs/{dialog.id}', {'value': answer}).status_code == 204
            done = action.result(timeout=10).json()
        receive_until(socket, SettingsChanged)
    [message] = done['messages']
    assert message.startswith('Added files.') and message.splitlines()[0] in terminal(web_)
    assert store.load().servers['files'].model_dump(exclude_defaults=True) == {
        'type': 'stdio',
        'command': 'uvx',
        'args': ['files'],
    }
    shown = {field['key']: field for field in done['menu']['fields']}
    assert shown['server.files']['value'] == 'ready' and shown['server.files']['action'] is True
    assert web_.post('/api/commands/mcp/fields/server.files', {'value': 'off'}).status_code == 400


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


def test_system_prompt_menu_edits_your_instructions_in_the_page(make_web: MakeWeb) -> None:
    """`/system_prompt` opens its menu here: a multi-line dialog edits yours, and a read-only one shows the prompt."""
    web_ = make_web(Agent(TestModel()))
    assert web_.post('/api/input', {'text': '/system_prompt'}).json() == {'menu': 'system_prompt'}
    rows = fields(web_, 'system_prompt')
    assert [(key, row['button'], row['action']) for key, row in rows.items()] == [
        ('instructions', 'Edit…', True),
        ('append', 'Append…', True),
        ('prompt', 'View…', True),
    ]
    with web_.client.websocket_connect(SOCKET, headers=ORIGIN) as socket:
        socket.receive_text()
        with ThreadPoolExecutor(1) as pool:
            action = pool.submit(web_.post, '/api/commands/system_prompt/fields/instructions/action')
            dialog = receive_until(socket, DialogOpened)[1].dialog
            assert (dialog.kind, dialog.multiline, dialog.secret) == ('text', True, False)
            answered = web_.post(f'/api/dialogs/{dialog.id}', {'value': ' Be brief.\n\nUse British spelling. '})
            assert answered.status_code == 204
            done = action.result(timeout=10).json()
            assert done['messages'] == ['Saved your instructions. They apply from your next prompt.']
            assert done['menu']['fields'][0]['value'] == 'Be brief. (3 lines)'
            assert web_.shell.context.settings.instructions == 'Be brief.\n\nUse British spelling.'
            assert 'Saved your instructions.' in terminal(web_)

            viewing = pool.submit(web_.post, '/api/commands/system_prompt/fields/prompt/action')
            dialog = receive_until(socket, DialogOpened)[1].dialog
            assert dialog.kind == 'view' and dialog.text.startswith('Nothing was sent in this conversation yet.')
            assert web_.post(f'/api/dialogs/{dialog.id}', {'value': 'closed'}).status_code == 204
            assert viewing.result(timeout=10).json()['messages'] == []


def test_usage_and_cost_notices_carry_the_figures_they_report(make_web: MakeWeb) -> None:
    web_ = make_web(Agent(TestModel()))
    with web_.client.websocket_connect(SOCKET, headers=ORIGIN) as socket:
        socket.receive_text()
        running = web_.run()
        web_.post('/api/input', {'text': 'hello'})
        receive_until(socket, TurnEnded)
        web_.post('/api/input', {'text': '/usage'})
        usage = receive_until(socket, Notice)[1]
        assert usage.command == '/usage' and usage.usage is not None
        [turn] = usage.usage.turns
        assert usage.usage.total == turn and turn.requests >= 1
        assert usage.usage.total.input_tokens > 0 and usage.usage.unpriced == ['test']
        assert usage.text.startswith('Retained history cost unknown'), 'the terminal text is unchanged'
        web_.post('/api/input', {'text': '/cost'})
        cost = receive_until(socket, Notice)[1]
        assert cost.usage == usage.usage
        web_.post('/api/input', {'text': '/help'})
        assert receive_until(socket, Notice)[1].usage is None, 'other commands carry none'
        web_.post('/api/input', {'text': '/exit'})
        running.result(timeout=10)


def test_the_about_menu_shows_the_version_and_asks_before_updating(
    make_web: MakeWeb, monkeypatch: pytest.MonkeyPatch
) -> None:
    def offline_updates(*, channel: Callable[[], UpdateChannel]) -> Updates:
        return Updates(channel=channel, current=Installed(version='1.0.0'), fetch=lambda channel: '1.0.0', spawn=run)

    def run(work: Callable[[], None]) -> None:
        work()

    monkeypatch.setattr(session_runtime, 'Updates', offline_updates)
    web_ = make_web(Agent(TestModel()))
    assert web_.post('/api/input', {'text': '/update'}).json() == {'menu': 'update'}, 'bare, it only opens the menu'
    rows = fields(web_, 'update')
    version = rows['version']
    assert (version['button'], version['action'], version['value'] == version['default']) == ('Update…', True, True)
    assert version['note'] == 'clai2 is already on the latest version 1.0.0 on the stable channel.'
    assert rows['updates.channel']['value'] == 'stable'
    assert web_.post('/api/commands/update/fields/updates.channel', {'value': 'main'}).status_code == 200
    assert web_.shell.context.settings.update_channel == 'main'
    # The real `/update` reinstalls CLAI: this one only says it ran.
    original = web_.shell.commands.get('update')
    assert original is not None
    web_.shell.commands.unregister(['update'])

    def pretend(args: list[str]) -> str:
        return 'Pretended to update.'

    web_.shell.commands.register(replace(original, handler=pretend))
    with web_.client.websocket_connect(SOCKET, headers=ORIGIN) as socket:
        socket.receive_text()
        running = web_.run()
        with ThreadPoolExecutor(1) as pool:
            for answer, messages in (('no', []), ('yes', ['Running /update now on the main channel.'])):
                action = pool.submit(web_.post, '/api/commands/update/fields/version/action')
                dialog = receive_until(socket, DialogOpened)[1].dialog
                assert (dialog.kind, dialog.yes, dialog.no) == ('confirm', 'Update', 'Cancel')
                assert 'main channel' in dialog.description
                web_.post(f'/api/dialogs/{dialog.id}', {'value': answer})
                assert action.result(timeout=10).json()['messages'] == messages
        notice = receive_until(socket, Notice)[1]
        assert (notice.command, notice.text) == ('/update now', 'Pretended to update.')
        web_.post('/api/input', {'text': '/exit'})
        running.result(timeout=10)
