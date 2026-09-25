"""Interactive application error, cancellation, and input boundaries."""

import asyncio
import io
import signal
import threading
from pathlib import Path
from typing import Generic, TypeVar

import pytest
from menu_script import Script, pick, typed
from prompt_toolkit.styles import BaseStyle
from pydantic_ai import Agent, ModelRequestContext, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel
from rich.color import Color
from rich.console import Console
from rich.text import Text
from termflow.tui import MenuItem  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.menu import MenuResult  # pyright: ignore[reportMissingTypeStubs]

from pydantic_clai2 import api_keys, chat, key_menu, theme
from pydantic_clai2.auth import CodexAuth
from pydantic_clai2.command_context import CommandContext
from pydantic_clai2.commands import Command
from pydantic_clai2.config import Settings
from pydantic_clai2.field_menu import FieldMenu, Runners
from pydantic_clai2.model_menu import ModelSettingsSource, model_settings_command, open_add_model_menu
from pydantic_clai2.settings_store import SettingsStore

PromptT = TypeVar('PromptT')


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def inputs(monkeypatch: pytest.MonkeyPatch, values: list[str | BaseException]) -> None:
    class Prompt(Generic[PromptT]):
        def __init__(self, **kwargs: object) -> None:
            style = kwargs['style']
            assert isinstance(style, BaseStyle)
            for selector in ('class:bottom-toolbar', 'class:bottom-toolbar.text'):
                assert style.get_attrs_for_style_str(selector).color == theme.color(theme.THINKING).lstrip('#')

        async def prompt_async(self, label: str, **kwargs: object) -> str:
            value = values.pop(0)
            if isinstance(value, BaseException):
                raise value
            return value

    monkeypatch.setattr('pydantic_clai2._app.PromptSession', Prompt)


@pytest.mark.parametrize('mode', ['eof', 'interrupt', 'error', 'cancel', 'double', 'structured'])
async def test_chat_boundaries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    output = io.StringIO()
    values: list[str | BaseException] = [' ', '/bad', 'run', '/exit']
    if mode == 'eof':
        values = [EOFError()]
    elif mode == 'interrupt':
        values = [KeyboardInterrupt(), KeyboardInterrupt()]

    class Behaviour(AbstractCapability[None]):
        async def before_model_request(
            self, ctx: RunContext[None], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            if mode == 'error':
                raise ValueError('broken provider')
            if mode in ('cancel', 'double'):
                signal.raise_signal(signal.SIGINT)
                if mode == 'double':
                    signal.raise_signal(signal.SIGINT)
                await asyncio.sleep(0)
            return request_context

    inputs(monkeypatch, values)
    console = Console(file=output, width=20 if mode == 'eof' else 120)
    store = SettingsStore(tmp_path / 'config.db')
    if mode == 'structured':
        await chat(Agent(TestModel(), output_type=list[int]), deps=None, console=console, store=store)
    else:
        await chat(
            Agent(TestModel(), deps_type=type(None), capabilities=[Behaviour()]),
            deps=None,
            console=console,
            store=store,
        )
    if mode == 'error':
        assert 'broken provider' in output.getvalue()
        assert 'Retained history may include partial progress' in output.getvalue()
        assert 'Turn not saved' not in output.getvalue()
    elif mode == 'cancel':
        assert 'Turn cancelled' in output.getvalue()
    elif mode == 'interrupt':
        assert 'Input cleared' in output.getvalue()


async def test_model_string_and_non_command_plugin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs(monkeypatch, ['/set', '/set display.thinking', '/config show', '/plugins list', '/new', '/exit'])

    class Provider(AbstractCapability[None]):
        def get_commands(self, context: CommandContext) -> list[Command]:
            return [Command(name='legacy', description='Legacy command', handler=lambda args: 'ok')]

    await chat(
        Agent('test'),
        deps=None,
        plugins=[AbstractCapability(), Provider()],
        settings=Settings(model='test'),
        console=Console(file=io.StringIO()),
        store=SettingsStore(tmp_path / 'config.db'),
    )


@pytest.mark.parametrize('provider', ['openrouter', 'vllm', 'github-copilot'])
async def test_connected_provider_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str) -> None:
    inputs(monkeypatch, ['hello', '/exit'])
    loop_thread = threading.get_ident()

    def model(name: str) -> TestModel:
        assert threading.get_ident() != loop_thread
        assert name == f'{provider}:test'
        return TestModel(custom_output_text='Connected response')

    monkeypatch.setattr(f'pydantic_clai2.{provider.replace("-", "_")}.model', model)
    output = io.StringIO()
    await chat(
        Agent(TestModel()),
        deps=None,
        settings=Settings(model=f'{provider}:test'),
        store=SettingsStore(tmp_path / 'config.db'),
        console=Console(file=output),
    )
    assert 'Connected response' in output.getvalue()


async def test_keys_command_in_shell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs(monkeypatch, ['/keys', '/help', '/exit'])
    script = Script(
        lists=[pick(key_menu.KeyAction(action='add')), MenuResult(cancelled=True)],
        choices=[],
        texts=[typed('shell_key'), typed('private-value')],
    )
    original = key_menu.run_keys_flow

    def scripted(*, runners: Runners = script.runners) -> None:
        original(runners=runners)

    monkeypatch.setattr(key_menu, 'run_keys_flow', scripted)
    output = io.StringIO()
    await chat(Agent(TestModel()), deps=None, console=Console(file=output), store=SettingsStore(tmp_path / 'config.db'))
    assert api_keys.load_keys()['SHELL_KEY'].get_secret_value() == 'private-value'
    assert '/keys' in output.getvalue()
    assert 'private-value' not in output.getvalue()


async def test_unknown_saved_model_settings_do_not_break_chat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs(monkeypatch, ['hello', '/exit'])
    store = SettingsStore(tmp_path / 'config.db')
    store.save_model_settings('test', {'temperature': 0.5, 'future_setting': {'nested': True}})
    output = io.StringIO()
    await chat(
        Agent(TestModel(custom_output_text='Compatible settings work.')),
        deps=None,
        console=Console(file=output, width=120),
        store=store,
    )
    assert 'Compatible settings work.' in output.getvalue()
    assert 'Invalid saved model settings' not in output.getvalue()
    assert store.model_settings('test') == {'temperature': 0.5, 'future_setting': {'nested': True}}


@pytest.mark.parametrize('invalid_key', ['temperature', '', 'a..b'])
async def test_invalid_saved_model_settings_can_be_repaired_without_exiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid_key: str
) -> None:
    inputs(monkeypatch, ['first attempt', '/model_settings test', 'second attempt', '/exit'])
    store = SettingsStore(tmp_path / 'config.db')
    if invalid_key == 'temperature':
        store.save_model_settings('test', {'temperature': 'private-invalid-value', 'future_setting': True})
    else:
        store.save_model_settings(
            'test', {'custom_params': {invalid_key: 'private-invalid-value'}, 'future_setting': True}
        )
    requests: list[str] = []

    store.add_model(name='test')
    if invalid_key == 'temperature':
        script = Script(lists=[pick('temperature'), MenuResult(cancelled=True)], choices=[], texts=[typed('0.5')])
    else:
        menu = FieldMenu(ModelSettingsSource(store, 'test'))
        reset = menu.reset_marker(object(), MenuItem('Custom params', value='custom_params'))
        script = Script(lists=[reset, MenuResult(cancelled=True)], choices=[], texts=[])

    async def edit_settings(context: CommandContext, args: list[str]) -> str:
        return await model_settings_command(context, args, runners=script.runners)

    monkeypatch.setattr('pydantic_clai2.model_menu.model_settings_command', edit_settings)

    class Repair(AbstractCapability[None]):
        async def before_model_request(
            self, ctx: RunContext[None], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            requests.append('request')
            return request_context

    output = io.StringIO()
    await chat(
        Agent(TestModel(custom_output_text='Recovered successfully.'), deps_type=type(None), capabilities=[Repair()]),
        deps=None,
        console=Console(file=output, width=120),
        store=store,
    )
    assert requests == ['request']
    text = output.getvalue()
    assert 'Invalid saved model settings for test' in text
    assert '/model_settings test' in text
    assert ('temperature:' if invalid_key == 'temperature' else 'custom_params:') in text
    assert 'private-invalid-value' not in text
    assert 'Recovered successfully.' in text
    expected = (
        {'temperature': 0.5, 'future_setting': True} if invalid_key == 'temperature' else {'future_setting': True}
    )
    assert store.model_settings('test') == expected


async def test_codex_login_and_turns_share_lazy_auth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs(monkeypatch, ['/login openai-codex', 'hello', 'again', '/exit'])
    instances: list[CodexAuth] = []

    async def login(self: CodexAuth, args: list[str]) -> str:
        assert args == ['openai-codex']
        instances.append(self)
        return 'Signed in.'

    def model(self: CodexAuth, name: str) -> TestModel:
        assert name == 'openai-codex:test'
        instances.append(self)
        return TestModel(custom_output_text='Connected response')

    monkeypatch.setattr(CodexAuth, 'login', login)
    monkeypatch.setattr(CodexAuth, 'model', model)
    output = io.StringIO()
    await chat(
        Agent(TestModel()),
        deps=None,
        settings=Settings(model='openai-codex:test', session_namer=False),
        store=SettingsStore(tmp_path / 'config.db'),
        console=Console(file=output),
    )
    assert len(instances) == 3
    assert all(instance is instances[0] for instance in instances)
    assert 'Signed in.' in output.getvalue()
    assert output.getvalue().count('Connected response') == 2


async def test_lazy_add_model_menu_and_named_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs(monkeypatch, ['/add_model', '/add_model test', '/exit'])

    async def add_model(context: CommandContext) -> str:
        return await open_add_model_menu(context, run=lambda menu: [])

    monkeypatch.setattr('pydantic_clai2.model_menu.open_add_model_menu', add_model)
    output = io.StringIO()
    store = SettingsStore(tmp_path / 'config.db')
    await chat(
        Agent(TestModel()),
        deps=None,
        settings=Settings(model=None),
        store=store,
        console=Console(file=output),
    )
    assert 'No changes.' in output.getvalue()
    assert 'Saved model. Applied.' in output.getvalue()
    assert store.load().model == 'test'


@pytest.mark.parametrize('name', theme.names())
@pytest.mark.parametrize(
    ('width', 'expected'),
    [
        (160, [theme.LITHIUM, theme.LITHIUM, theme.PURPLE, theme.PURPLE, theme.AI_CYAN, theme.AI_CYAN]),
        (40, [theme.LITHIUM]),
    ],
)
async def test_banner_keeps_brand_colours_under_every_theme(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, width: int, expected: list[str]
) -> None:
    inputs(monkeypatch, ['/exit'])
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, color_system='truecolor', width=width)
    await chat(
        Agent(TestModel()),
        deps=None,
        settings=Settings(model=None, theme=name),
        store=SettingsStore(tmp_path / 'config.db'),
        console=console,
    )
    text = Text.from_ansi(output.getvalue())
    logo_rows = [line for line in text.split() if {'█', '═'} & set(line.plain) or line.plain == 'CLAI 2.0']
    colours = [
        line.get_style_at_offset(console, len(line.plain) - len(line.plain.lstrip())).color for line in logo_rows
    ]
    assert colours == [Color.parse(colour) for colour in expected]
