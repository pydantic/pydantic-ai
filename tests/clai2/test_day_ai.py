"""The built-in `day_ai` plugin: a settings menu for `DayAI`'s options, with any token kept in `/keys`."""

import io
from pathlib import Path
from types import TracebackType

import anyio
import pytest
from fastmcp.client.auth import OAuth
from fastmcp.client.auth.oauth import TokenStorageAdapter
from fastmcp.client.transports import StreamableHttpTransport
from mcp.shared.auth import OAuthToken
from pydantic import JsonValue
from rich.console import Console
from termflow.tui import MenuItem
from termflow.tui.menu import MenuResult

import pydantic_clai2.day_ai as day_ai
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.day_ai import DayAI
from pydantic_clai2 import DEFAULT_PLUGINS, api_keys
from pydantic_clai2.api_keys import KeyReference, SavedKey
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.day_ai import SETUP, DayAISource
from pydantic_clai2.mcp import TokenStore
from pydantic_clai2.plugin_loader import PluginError, PluginLoader
from pydantic_clai2.plugin_menu import Configure, PluginMenu, open_plugins_menu
from pydantic_clai2.plugins import PluginHost, SessionStart
from pydantic_clai2.settings_store import SettingsStore
from tests.clai2.menu_script import Script, pick

BUILTIN = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'day_ai')
CLOSE = MenuResult(cancelled=True)


class SignIn:
    """Stands in for FastMCP's `Client`, whose connection would open the browser."""

    connected: list[StreamableHttpTransport] = []
    error: Exception | None = None

    def __init__(self, transport: StreamableHttpTransport) -> None:
        self.transport = transport

    async def __aenter__(self) -> None:
        if SignIn.error is not None:
            raise SignIn.error
        SignIn.connected.append(self.transport)

    async def __aexit__(
        self, kind: type[BaseException] | None, error: BaseException | None, traceback: TracebackType | None
    ) -> None:
        return None


class Shell:
    """A loader plus what a test inspects: printed output and the plaintext settings file."""

    def __init__(self, tmp_path: Path, *, terminal: bool = False, settings: dict[str, JsonValue] | None = None) -> None:
        self.path = tmp_path / 'settings.db'
        self.store = SettingsStore(self.path)
        self.output = io.StringIO()
        declaration = BUILTIN if settings is None else BUILTIN.model_copy(update={'settings': settings})
        self.loader: PluginLoader[None] = PluginLoader(
            store=self.store,
            console=Console(file=self.output, force_terminal=terminal, width=200),
            commands=Commands(),
            session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=self.store.load()),
            builtin=(declaration,),
        )

    def saved(self) -> dict[str, JsonValue]:
        [declaration] = self.store.plugins()
        assert declaration.enabled
        return declaration.settings

    def host(self) -> PluginHost[None]:
        [entry] = self.loader.entries()
        assert entry.host is not None
        return entry.host

    def client(self) -> object:
        [capability] = self.loader.capabilities()
        assert isinstance(capability, DayAI)
        return capability.client


@pytest.fixture(autouse=True)
def sign_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(day_ai, 'Client', SignIn)
    monkeypatch.setattr(SignIn, 'connected', [])
    monkeypatch.setattr(SignIn, 'error', None)


def script(monkeypatch: pytest.MonkeyPatch, lists: list[MenuResult], choices: list[MenuResult] | None = None) -> Script:
    scripted = Script(lists=[*lists, CLOSE], choices=choices or [], texts=[])
    monkeypatch.setattr(day_ai, 'RUNNERS', scripted.runners)
    return scripted


def key_choice(monkeypatch: pytest.MonkeyPatch, choice: str | KeyReference | None) -> list[str]:
    """Answer `prompt_api_key`, whose saved-key list needs a real terminal, and record its labels."""
    labels: list[str] = []

    async def prompt_api_key(*, prompt: object, label: str, optional: bool = False) -> str | KeyReference | None:
        labels.append(label)
        return choice

    monkeypatch.setattr('pydantic_clai2.plugin_keys.prompt_api_key', prompt_api_key)
    return labels


def with_key(name: str = 'DAY_AI_ACCESS_TOKEN', *, include_instructions: bool = True) -> list[DayAI[None]]:
    return [DayAI[None](auth=SavedKey(name=name, setup=SETUP), include_instructions=include_instructions)]


async def store_sign_in() -> None:
    tokens = TokenStorageAdapter(TokenStore(day_ai.TOKEN_ACCOUNT), server_url=day_ai.DAY_AI_MCP_URL)
    await tokens.set_tokens(OAuthToken(access_token='access', token_type='Bearer', expires_in=3600))


def test_declared_disabled_with_no_settings() -> None:
    assert BUILTIN == PluginSettings(id='day_ai', factory='pydantic_clai2.day_ai', enabled=False)


async def test_enable_opens_the_menu_and_every_option_saves_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('DAY_AI_ACCESS_TOKEN', 'the-environment-is-not-a-source')
    labels = key_choice(monkeypatch, ' day-secret ')
    shown = script(
        monkeypatch, lists=[pick('auth'), pick('include_instructions')], choices=[pick('key'), pick('false')]
    )
    shell = Shell(tmp_path)
    assert await shell.loader.command(['enable', 'day_ai']) == '\n'.join(
        [
            'Enabled day_ai.',
            'Day AI uses the saved key DAY_AI_ACCESS_TOKEN. Manage it in /keys.',
            'Saved Server instructions.',
        ]
    )
    assert f'Day AI is not connected. {SETUP}' in shell.output.getvalue()
    assert labels == ['Day AI access token (saved in /keys as DAY_AI_ACCESS_TOKEN)']
    assert shown.opened == ['list', 'choice', 'list', 'choice', 'list']
    assert api_keys.load_keys()['DAY_AI_ACCESS_TOKEN'].get_secret_value() == 'day-secret'
    assert shell.saved() == {'auth': {'name': 'DAY_AI_ACCESS_TOKEN'}, 'include_instructions': False}
    assert b'day-secret' not in shell.path.read_bytes()
    assert shell.loader.capabilities() == with_key(include_instructions=False)
    assert SignIn.connected == [], 'a token needs no browser sign-in'
    await shell.loader.close('exit')


async def test_conventional_key_is_used_and_resolved_on_every_run(tmp_path: Path) -> None:
    api_keys.save_key(name='DAY_AI_ACCESS_TOKEN', value='first')
    shell = Shell(tmp_path)
    await shell.loader.enable('day_ai')
    assert shell.loader.capabilities() == with_key()
    token = SavedKey(name='DAY_AI_ACCESS_TOKEN', setup=SETUP)
    api_keys.save_key(name='DAY_AI_ACCESS_TOKEN', value='replaced')
    assert token(None) == 'replaced'
    api_keys.delete_key(name='DAY_AI_ACCESS_TOKEN')
    with pytest.raises(UserError, match=r'DAY_AI_ACCESS_TOKEN is missing\. Run /plugins configure day_ai'):
        token(None)
    await shell.loader.close('exit')


async def test_reconfigure_picks_another_saved_key_without_reinstalling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api_keys.save_key(name='WORK_DAY_AI', value='work-secret')
    shell = Shell(tmp_path, settings={'auth': {'name': 'GONE'}})
    await shell.loader.enable('day_ai')
    assert 'Day AI has no token: GONE is not in /keys.' in shell.output.getvalue()
    assert shell.loader.capabilities() == with_key('GONE')
    assert DayAISource(shell.host()).rows()[0].note == 'missing from /keys'
    key_choice(monkeypatch, KeyReference(name='WORK_DAY_AI'))
    script(monkeypatch, lists=[pick('auth')], choices=[pick('key')])
    assert await shell.loader.configure('day_ai') == 'Day AI uses the saved key WORK_DAY_AI. Manage it in /keys.'
    assert shell.saved() == {'auth': {'name': 'WORK_DAY_AI'}, 'include_instructions': True}
    assert b'work-secret' not in shell.path.read_bytes()
    assert shell.loader.capabilities() == with_key('WORK_DAY_AI')
    await shell.loader.close('exit')


async def test_choosing_the_browser_signs_in_and_sticks_with_the_conventional_key_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api_keys.save_key(name='DAY_AI_ACCESS_TOKEN', value='shared-with-another-tool')
    shell = Shell(tmp_path, terminal=True)
    await shell.loader.enable('day_ai')
    source = DayAISource(shell.host())
    assert source.rows()[0].note == 'uses DAY_AI_ACCESS_TOKEN'
    script(monkeypatch, lists=[pick('auth')], choices=[pick('oauth')])
    assert await shell.loader.configure('day_ai') == 'Saved Sign-in.'
    assert shell.saved() == {'auth': 'oauth', 'include_instructions': True}
    transport = shell.client()
    assert isinstance(transport, StreamableHttpTransport) and isinstance(transport.auth, OAuth)
    assert transport.url == day_ai.DAY_AI_MCP_URL
    [signed_in_with] = SignIn.connected
    assert signed_in_with is not transport
    assert 'Opening your browser to sign in to Day AI.' in shell.output.getvalue()
    await shell.loader.close('exit')


async def test_automatic_uses_a_stored_sign_in_without_the_browser(tmp_path: Path) -> None:
    await store_sign_in()
    shell = Shell(tmp_path)
    await shell.loader.enable('day_ai')
    assert isinstance(shell.client(), StreamableHttpTransport)
    assert SignIn.connected == []
    assert DayAISource(shell.host()).rows()[0].note == 'uses the stored browser sign-in'
    await shell.loader.close('exit')


async def test_nothing_chosen_loads_no_capability_and_says_how_to_connect(tmp_path: Path) -> None:
    shell = Shell(tmp_path)
    await shell.loader.enable('day_ai')
    assert shell.loader.capabilities() == []
    assert DayAISource(shell.host()).rows()[0].note == 'not connected'
    assert SignIn.connected == []
    await shell.loader.close('exit')


async def test_failed_sign_in_leaves_nothing_loaded(tmp_path: Path) -> None:
    SignIn.error = RuntimeError('authorization denied')
    shell = Shell(tmp_path, terminal=True, settings={'auth': 'oauth'})
    with pytest.raises(PluginError, match="Plugin 'day_ai': RuntimeError: authorization denied"):
        await shell.loader.enable('day_ai')
    assert shell.loader.capabilities() == []


async def test_headless_browser_sign_in_fails_clearly(tmp_path: Path) -> None:
    shell = Shell(tmp_path, settings={'auth': 'oauth'})
    with pytest.raises(PluginError, match='Save DAY_AI_ACCESS_TOKEN in /keys, or sign in to Day AI'):
        await shell.loader.enable('day_ai')
    assert shell.loader.capabilities() == [] and SignIn.connected == []


async def test_settings_cannot_hold_a_token(tmp_path: Path) -> None:
    shell = Shell(tmp_path, settings={'auth': 'secret'})
    with pytest.raises(PluginError, match='auth'):
        await shell.loader.enable('day_ai')
    assert shell.loader.capabilities() == []


async def test_cancelling_or_resetting_in_the_menu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shell = Shell(tmp_path, settings={'auth': 'oauth', 'include_instructions': False})
    await store_sign_in()
    await shell.loader.enable('day_ai')
    key_choice(monkeypatch, None)
    script(monkeypatch, lists=[pick('auth'), pick('auth')], choices=[CLOSE, pick('key')])
    assert await shell.loader.configure('day_ai') == 'Day AI settings unchanged.'
    assert shell.saved() == {'auth': 'oauth', 'include_instructions': False}
    source = DayAISource(shell.host())
    [auth, instructions] = source.rows()
    assert source.current(auth) == 'oauth' and auth.note == ''
    assert source.reset(instructions) == 'Reset Server instructions.'
    assert source.apply(auth, 'automatic') == 'Saved Sign-in.'
    assert shell.saved() == {'auth': None, 'include_instructions': True}
    assert source.current(auth) == 'automatic'
    await shell.loader.close('exit')


def test_menu_validates_like_saving_would() -> None:
    host = PluginHost[None](name='day_ai', console=Console(file=io.StringIO()), settings={})
    source = DayAISource(host)
    [auth, instructions] = source.rows()
    assert source.problem(instructions, 'false') is None
    assert source.problem(instructions, 'sometimes') is not None
    assert source.problem(auth, '') is not None, 'a key needs a name'
    assert source.apply(auth, 'WORK_DAY_AI') == 'Saved Sign-in.'
    assert source.current(auth) == 'WORK_DAY_AI'


async def test_add_replacing_the_builtin_opens_the_menu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api_keys.save_key(name='DAY_AI_ACCESS_TOKEN', value='saved')
    shell = Shell(tmp_path)
    script(monkeypatch, lists=[pick('include_instructions')], choices=[pick('true')])
    added = await shell.loader.command(['add', 'day_ai', 'pydantic_clai2.day_ai', '{"include_instructions": false}'])
    assert added == 'Replaced built-in day_ai.\nSaved Server instructions.'
    assert shell.loader.capabilities() == with_key()
    await shell.loader.close('exit')


async def test_configure_needs_a_loaded_plugin_with_a_menu(tmp_path: Path) -> None:
    shell = Shell(tmp_path)
    with pytest.raises(ValueError, match='not loaded; enable it before configuring'):
        await shell.loader.command(['configure', 'day_ai'])
    shell.store.save_plugin(BUILTIN.model_copy(update={'id': 'plain', 'factory': 'pydantic_clai2.repo_context'}))
    assert await shell.loader.command(['enable', 'plain']) == 'Enabled plain.'
    with pytest.raises(ValueError, match='no settings menu'):
        await shell.loader.configure('plain')
    await shell.loader.close('exit')


async def test_plugins_menu_enabling_opens_the_settings_menu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api_keys.save_key(name='DAY_AI_ACCESS_TOKEN', value='saved')
    shell = Shell(tmp_path)
    script(monkeypatch, lists=[pick('include_instructions')], choices=[pick('false')])
    presses = iter(['space', 'close'])

    def run(menu: PluginMenu[None]) -> MenuResult:
        item, save_and_close = menu.items()
        if next(presses) == 'close':
            assert menu.notice == 'Saved Server instructions.'
            return MenuResult(item=save_and_close)
        result = menu.toggle(Redraw(), item)
        assert result is not None and result.item is not None and result.item.value == Configure('day_ai')
        return result

    assert await open_plugins_menu(shell.loader, run=run) == 'Saved Server instructions.'
    assert shell.loader.capabilities() == with_key(include_instructions=False)
    await shell.loader.close('exit')


async def test_plugins_menu_stays_open_when_there_is_nothing_to_configure(tmp_path: Path) -> None:
    shell = Shell(tmp_path)
    shell.store.save_plugin(
        BUILTIN.model_copy(update={'id': 'plain', 'factory': 'pydantic_clai2.repo_context', 'enabled': True})
    )
    await shell.loader.load_all()

    def run(menu: PluginMenu[None]) -> MenuResult:
        *rows, save_and_close = menu.items()
        day_ai_row, plain_row = sorted(rows, key=lambda item: str(item.value))
        assert menu.configure(Redraw(), MenuItem('none', value=None)) is None
        assert menu.configure(Redraw(), day_ai_row) is None
        assert menu.notice == 'Enable day_ai to configure it.'
        assert menu.configure(Redraw(), plain_row) is None
        assert menu.notice == 'plain has no settings menu.'
        return MenuResult(item=save_and_close)

    assert await open_plugins_menu(shell.loader, run=run) == ''
    await shell.loader.close('exit')


async def test_cancelling_configure_cancels_an_open_key_picker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shell = Shell(tmp_path)
    await shell.loader.enable('day_ai')
    opened = anyio.Event()
    finished: list[str] = []

    async def prompt_api_key(*, prompt: object, label: str, optional: bool = False) -> str | KeyReference | None:
        opened.set()
        try:
            await anyio.sleep_forever()
        finally:
            finished.append(label)
        return None  # pragma: no cover -- unreachable; keeps the signature honest

    monkeypatch.setattr('pydantic_clai2.plugin_keys.prompt_api_key', prompt_api_key)
    script(monkeypatch, lists=[pick('auth')], choices=[pick('key')])
    with anyio.fail_after(10):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(shell.loader.configure, 'day_ai')
            await opened.wait()
            tasks.cancel_scope.cancel()
    assert finished == ['Day AI access token (saved in /keys as DAY_AI_ACCESS_TOKEN)']
    await shell.loader.close('exit')


class Redraw:
    def replace_items(self, items: object) -> None:
        pass
