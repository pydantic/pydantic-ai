"""The built-in `day_ai` plugin: a settings menu for `DayAI`'s options, with any token kept in `/keys`."""

import io
import threading
import time
from collections.abc import Callable
from pathlib import Path

import anyio
import pytest
from anyio import to_thread
from fastmcp import Client
from fastmcp.client.auth import OAuth
from fastmcp.client.auth.oauth import TokenStorageAdapter
from fastmcp.client.transports import StreamableHttpTransport
from mcp.shared.auth import OAuthToken
from pydantic import JsonValue
from rich.console import Console
from termflow.tui import MenuItem
from termflow.tui.menu import MenuResult
from termflow.tui.textinput import TextInput, TextInputResult

import pydantic_clai2.builtin_plugins.day_ai as day_ai
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.day_ai import DayAI
from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2.builtin_plugins.day_ai import SETUP, DayAISettings, DayAISource
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import PluginSettings, api_keys
from pydantic_clai2.config.api_keys import SavedKey
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.mcp import OAuthSignIn, SignIn as MCPSignIn, TokenStore, http_client
from pydantic_clai2.plugins import PluginHost, SessionStart
from pydantic_clai2.plugins.loader import PluginError, PluginLoader
from pydantic_clai2.plugins.menus import Option
from pydantic_clai2.runtime.session import WorkspacelessCapability
from pydantic_clai2.ui.menus import dialogs
from pydantic_clai2.ui.menus.field_menu import Runners
from pydantic_clai2.ui.menus.menu_worker import worker_stopping
from pydantic_clai2.ui.menus.plugin_menu import Configure, PluginMenu, open_plugins_menu
from tests.clai2.dialog_script import Back, Confirm, Finish, Pick, ScriptedDialogs, Type
from tests.clai2.menu_script import UNTIL_CLOSED, Script, pick, typed

BUILTIN = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'day_ai')
CLOSE = MenuResult(cancelled=True)


class SignIn:
    """Stands in for `OAuthSignIn.sign_in`, which would open the browser; it stores tokens as FastMCP would."""

    attempts: int = 0
    error: Exception | None = None
    unfinished: bool = False
    """The browser sign-in is never completed, so it waits until cancelled."""

    @staticmethod
    async def sign_in(method: OAuthSignIn, *, show: Callable[[str], object]) -> None:
        assert method is day_ai.SIGN_IN
        SignIn.attempts += 1
        if SignIn.error is not None:
            raise SignIn.error
        if SignIn.unfinished:
            show('https://day.ai/authorize')
            await anyio.sleep_forever()
        await store_sign_in()


class Shell:
    """A loader plus what a test inspects: printed output and the plaintext settings file."""

    def __init__(self, tmp_path: Path, *, terminal: bool = True, settings: dict[str, JsonValue] | None = None) -> None:
        self.path = tmp_path / 'settings.db'
        self.store = SettingsStore(self.path)
        self.output = io.StringIO()
        self.commands = Commands()
        declaration = BUILTIN if settings is None else BUILTIN.model_copy(update={'settings': settings})
        self.loader: PluginLoader[None] = PluginLoader(
            store=self.store,
            console=Console(file=self.output, force_terminal=terminal, color_system=None, width=200),
            commands=self.commands,
            session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=self.store.load()),
            builtin=(declaration,),
        )

    def saved(self) -> dict[str, JsonValue]:
        [declaration] = self.store.plugins()
        assert declaration.enabled
        return declaration.settings

    def host(self) -> PluginHost[None]:
        [entry] = self.loader.entries()
        assert entry.loaded is not None
        return entry.loaded.host

    def plugin(self) -> day_ai.DayAIPlugin[None]:
        [entry] = self.loader.entries()
        assert entry.loaded is not None
        plugin = entry.loaded.plugin
        assert isinstance(plugin, day_ai.DayAIPlugin)
        return plugin

    async def choose_auth(self, scripted: ScriptedDialogs) -> list[str]:
        """Run the Sign-in row's action, as the web UI's Change button does."""
        menu = self.plugin().settings_menu()
        return list(await menu.actions['auth'](scripted))

    async def run_capability(self) -> DayAI[None] | None:
        """What a run gets from the browser sign-in's per-run factory."""
        [entry] = self.loader.entries()
        assert entry.loaded is not None
        plugin = entry.loaded.plugin
        assert isinstance(plugin, day_ai.DayAIPlugin)
        assert self.loader.capabilities() == [WorkspacelessCapability(plugin.for_run)]
        capability = await plugin.for_run(RunContext[None](deps=None, model=TestModel(), usage=RunUsage()))
        assert capability is None or capability.defer_loading
        return capability

    async def client(self) -> object:
        capability = await self.run_capability()
        assert capability is not None
        return capability.client


@pytest.fixture(autouse=True)
def sign_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(OAuthSignIn, 'sign_in', SignIn.sign_in)
    monkeypatch.setattr(SignIn, 'attempts', 0)
    monkeypatch.setattr(SignIn, 'error', None)
    monkeypatch.setattr(SignIn, 'unfinished', False)


def script(
    monkeypatch: pytest.MonkeyPatch,
    lists: list[MenuResult],
    choices: list[MenuResult] | None = None,
    texts: list[TextInputResult] | None = None,
) -> Script:
    scripted = Script(lists=[*lists, CLOSE], choices=choices or [], texts=texts or [])
    monkeypatch.setattr(dialogs, 'RUNNERS', scripted.runners)
    return scripted


def with_key(name: str = 'DAY_AI_ACCESS_TOKEN', *, include_instructions: bool = True) -> list[DayAI[None]]:
    return [
        DayAI[None](
            defer_loading=True, auth=SavedKey(name=name, setup=SETUP), include_instructions=include_instructions
        )
    ]


async def store_sign_in() -> None:
    tokens = TokenStorageAdapter(TokenStore(day_ai.TOKEN_ACCOUNT), server_url=day_ai.DAY_AI_MCP_URL)
    await tokens.set_tokens(OAuthToken(access_token='access', token_type='Bearer', expires_in=3600))


def test_declared_disabled_with_no_settings() -> None:
    assert BUILTIN == PluginSettings(id='day_ai', factory='pydantic_clai2.builtin_plugins.day_ai', enabled=False)


async def test_enable_opens_the_menu_and_every_option_saves_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('DAY_AI_ACCESS_TOKEN', 'the-environment-is-not-a-source')
    shown = script(
        monkeypatch,
        lists=[pick('auth'), pick('include_instructions')],
        choices=[pick('key'), pick('false')],
        texts=[typed(' day-secret ')],
    )
    shell = Shell(tmp_path)
    assert await shell.loader.command(['enable', 'day_ai']) == '\n'.join(
        [
            'Enabled day_ai.',
            'Day AI uses the saved key DAY_AI_ACCESS_TOKEN. Manage it in /keys.',
            'Saved Server instructions.',
        ]
    )
    assert 'Not signed in to Day AI. Run /day_ai login to sign in.' in shell.output.getvalue()
    assert shown.opened == ['list', 'choice', 'text', 'list', 'choice', 'list']
    assert api_keys.load_keys()['DAY_AI_ACCESS_TOKEN'].get_secret_value() == 'day-secret'
    assert shell.saved() == {'auth': {'name': 'DAY_AI_ACCESS_TOKEN'}, 'include_instructions': False}
    assert b'day-secret' not in shell.path.read_bytes()
    assert shell.loader.capabilities() == with_key(include_instructions=False)
    assert SignIn.attempts == 0, 'a token needs no browser sign-in'
    await shell.loader.close('exit')


async def test_a_chosen_key_is_resolved_on_every_run(tmp_path: Path) -> None:
    api_keys.save_key(name='DAY_AI_ACCESS_TOKEN', value='first')
    shell = Shell(tmp_path, settings={'auth': {'name': 'DAY_AI_ACCESS_TOKEN'}})
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
    script(monkeypatch, lists=[pick('auth')], choices=[pick('key'), pick('WORK_DAY_AI')])
    assert await shell.loader.configure('day_ai') == 'Day AI uses the saved key WORK_DAY_AI. Manage it in /keys.'
    assert shell.saved() == {'auth': {'name': 'WORK_DAY_AI'}, 'include_instructions': True}
    assert b'work-secret' not in shell.path.read_bytes()
    assert shell.loader.capabilities() == with_key('WORK_DAY_AI')
    await shell.loader.close('exit')


async def test_browser_sign_in_is_the_default_even_with_the_conventional_key_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api_keys.save_key(name='DAY_AI_ACCESS_TOKEN', value='shared-with-another-tool')
    shell = Shell(tmp_path)
    await shell.loader.enable('day_ai')
    assert await shell.run_capability() is None, 'a run without a sign-in leaves Day AI out'
    assert SignIn.attempts == 0, 'neither loading nor a run signs in'
    assert 'Not signed in to Day AI. Run /day_ai login to sign in.' in shell.output.getvalue()
    source = DayAISource(shell.host())
    [auth, _] = source.rows()
    assert source.current(auth) == 'oauth' and auth.note == ''
    script(monkeypatch, lists=[pick('auth')], choices=[pick('oauth'), UNTIL_CLOSED])
    assert await shell.loader.configure('day_ai') == 'Saved Sign-in.\nSigned in to Day AI.'
    assert shell.saved() == {'auth': 'oauth', 'include_instructions': True}
    assert isinstance(await shell.client(), Client)
    assert SignIn.attempts == 1
    await shell.loader.close('exit')


async def test_a_stored_sign_in_is_used_without_the_browser(tmp_path: Path) -> None:
    await store_sign_in()
    shell = Shell(tmp_path)
    await shell.loader.enable('day_ai')
    assert isinstance(await shell.client(), Client)
    assert SignIn.attempts == 0 and shell.output.getvalue() == ''
    await shell.loader.close('exit')


async def test_automatic_saved_by_an_earlier_build_means_the_browser(tmp_path: Path) -> None:
    assert DayAISettings.model_validate({'auth': None}) == DayAISettings()
    await store_sign_in()
    shell = Shell(tmp_path, settings={'auth': None, 'include_instructions': False})
    await shell.loader.enable('day_ai')
    source = DayAISource(shell.host())
    assert source.current(source.rows()[0]) == 'oauth'
    capability = await shell.run_capability()
    assert capability is not None and not capability.include_instructions
    assert SignIn.attempts == 0
    await shell.loader.close('exit')


async def test_failed_sign_in_is_reported_and_adds_no_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    SignIn.error = RuntimeError('Client failed to connect: authorization denied')
    shell = Shell(tmp_path)
    await shell.loader.enable('day_ai')
    script(monkeypatch, lists=[pick('auth')], choices=[pick('oauth'), UNTIL_CLOSED])
    assert await shell.loader.configure('day_ai') == '\n'.join(
        [
            'Saved Sign-in.',
            'Could not sign in to Day AI: Client failed to connect: authorization denied. '
            'Run /day_ai login to try again.',
        ]
    )
    assert shell.saved() == {'auth': 'oauth', 'include_instructions': True}
    assert await shell.run_capability() is None
    await shell.loader.close('exit')


async def test_browser_sign_in_is_an_auth_the_transport_client_accepts() -> None:
    """The client is built in the HTTPX FastMCP drives, so it takes the sign-in rather than rejecting it."""
    transport = day_ai.SIGN_IN.transport()
    assert isinstance(transport, StreamableHttpTransport) and isinstance(transport.auth, OAuth)
    assert transport.url == day_ai.DAY_AI_MCP_URL
    assert transport.httpx_client_factory is http_client
    assert isinstance(transport.auth, MCPSignIn)
    async with http_client(auth=transport.auth) as client:
        assert client.auth is transport.auth
        assert not client.follow_redirects


async def test_an_unfinished_sign_in_never_holds_up_loading(tmp_path: Path) -> None:
    """Loading waits for every plugin, so a browser left open at load would leave CLAI unable to run anything."""
    SignIn.unfinished = True
    for terminal in (True, False):
        folder = tmp_path / f'terminal-{terminal}'
        folder.mkdir()
        shell = Shell(folder, terminal=terminal, settings={'auth': 'oauth'})
        with anyio.fail_after(5):
            await shell.loader.enable('day_ai')
            assert await shell.run_capability() is None
        assert 'Not signed in to Day AI. Run /day_ai login to sign in.' in shell.output.getvalue()
        assert SignIn.attempts == 0
        await shell.loader.close('exit')


async def test_esc_cancels_an_unfinished_sign_in_and_leaves_day_ai_signed_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    SignIn.unfinished = True
    shell = Shell(tmp_path)
    await shell.loader.enable('day_ai')
    script(monkeypatch, lists=[pick('auth')], choices=[pick('oauth'), CLOSE])
    with anyio.fail_after(5):
        message = await shell.loader.configure('day_ai')
    assert message == 'Saved Sign-in.\nDay AI sign-in cancelled. Run /day_ai login to try again.'
    assert SignIn.attempts == 1
    assert await shell.run_capability() is None
    assert 'Not signed in to Day AI. Run /day_ai login to sign in.' in shell.output.getvalue()
    await shell.loader.close('exit')


async def test_day_ai_login_signs_in_and_the_next_run_connects_without_a_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        'pydantic_clai2.plugins.sign_in.RUNNERS', Script(lists=[], choices=[UNTIL_CLOSED], texts=[]).runners
    )
    shell = Shell(tmp_path, settings={'auth': 'oauth'})
    await shell.loader.enable('day_ai')
    assert await shell.run_capability() is None
    commands = shell.commands
    assert await commands.execute_async('/day_ai') == 'Not signed in to Day AI. Run /day_ai login to sign in.'
    assert await commands.execute_async('/day_ai login') == 'Signed in to Day AI.'
    assert isinstance(await shell.client(), Client), 'no reload needed'
    assert await commands.execute_async('/day_ai logout') == 'Signed out of Day AI. Run /day_ai login to sign in again.'
    assert await shell.run_capability() is None
    await shell.loader.close('exit')


async def test_choosing_the_browser_when_already_signed_in_does_not_open_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await store_sign_in()
    shell = Shell(tmp_path)
    await shell.loader.enable('day_ai')
    script(monkeypatch, lists=[pick('auth')], choices=[pick('oauth')])
    assert await shell.loader.configure('day_ai') == 'Saved Sign-in.'
    assert isinstance(await shell.client(), Client)
    assert SignIn.attempts == 0
    await shell.loader.close('exit')


async def test_settings_cannot_hold_a_token(tmp_path: Path) -> None:
    shell = Shell(tmp_path, settings={'auth': 'secret'})
    with pytest.raises(PluginError, match='auth'):
        await shell.loader.enable('day_ai')
    assert shell.loader.capabilities() == []


async def test_cancelling_or_resetting_in_the_menu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shell = Shell(tmp_path, settings={'auth': 'oauth', 'include_instructions': False})
    await store_sign_in()
    await shell.loader.enable('day_ai')
    script(
        monkeypatch,
        lists=[pick('auth'), pick('auth')],
        choices=[CLOSE, pick('key')],
        texts=[TextInputResult(cancelled=True)],
    )
    assert await shell.loader.configure('day_ai') == 'Day AI: nothing changed.'
    assert shell.saved() == {'auth': 'oauth', 'include_instructions': False}
    source = DayAISource(shell.host())
    [auth, instructions] = source.rows()
    assert source.current(auth) == 'oauth' and auth.note == ''
    assert source.reset(instructions) == 'Reset Server instructions.'
    assert source.apply(auth, 'WORK_DAY_AI') == 'Saved Sign-in.'
    assert source.reset(auth) == 'Reset Sign-in.'
    assert shell.saved() == {'auth': 'oauth', 'include_instructions': True}
    assert source.current(auth) == 'oauth'
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
    added = await shell.loader.command(
        [
            'add',
            'day_ai',
            'pydantic_clai2.builtin_plugins.day_ai',
            '{"auth": {"name": "DAY_AI_ACCESS_TOKEN"}, "include_instructions": false}',
        ]
    )
    assert added == 'Replaced built-in day_ai.\nSaved Server instructions.'
    assert shell.loader.capabilities() == with_key()
    await shell.loader.close('exit')


async def test_configure_needs_a_loaded_plugin_with_a_menu(tmp_path: Path) -> None:
    shell = Shell(tmp_path)
    with pytest.raises(ValueError, match='not loaded; enable it before configuring'):
        await shell.loader.command(['configure', 'day_ai'])
    shell.store.save_plugin(
        BUILTIN.model_copy(update={'id': 'plain', 'factory': 'pydantic_clai2.builtin_plugins.repo_context'})
    )
    assert await shell.loader.command(['enable', 'plain']) == 'Enabled plain.'
    with pytest.raises(ValueError, match='no settings menu'):
        await shell.loader.configure('plain')
    await shell.loader.close('exit')


async def test_plugins_menu_enabling_opens_the_settings_menu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api_keys.save_key(name='DAY_AI_ACCESS_TOKEN', value='saved')
    shell = Shell(tmp_path, settings={'auth': {'name': 'DAY_AI_ACCESS_TOKEN'}})
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
        BUILTIN.model_copy(
            update={'id': 'plain', 'factory': 'pydantic_clai2.builtin_plugins.repo_context', 'enabled': True}
        )
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
    opened = threading.Event()
    finished: list[str] = []

    def run_text(widget: TextInput) -> TextInputResult:
        opened.set()
        while not worker_stopping():
            time.sleep(0.01)
        finished.append(widget._title)  # pyright: ignore[reportPrivateUsage]
        return TextInputResult(cancelled=True)

    shown = script(monkeypatch, lists=[pick('auth')], choices=[pick('key')])
    monkeypatch.setattr(
        dialogs, 'RUNNERS', Runners(run_list=shown.run_list, run_choice=shown.run_choice, run_text=run_text)
    )
    with anyio.fail_after(10):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(shell.loader.configure, 'day_ai')
            assert await to_thread.run_sync(opened.wait, 5)
            tasks.cancel_scope.cancel()
    assert finished == ['Day AI access token (saved in /keys as DAY_AI_ACCESS_TOKEN)']
    await shell.loader.close('exit')


AUTH_OPTIONS = (
    Option(value='oauth', label='browser sign-in'),
    Option(value='key', label='choose or enter a key in /keys...'),
)


async def test_the_sign_in_action_signs_in_through_the_dialogs_it_is_given(tmp_path: Path) -> None:
    shell = Shell(tmp_path, terminal=False, settings={'auth': {'name': 'GONE'}})
    await shell.loader.enable('day_ai')
    scripted = ScriptedDialogs([Pick(value='oauth'), Finish()])
    assert await shell.choose_auth(scripted) == ['Saved Sign-in.', 'Signed in to Day AI.']
    choose, wait = scripted.asked
    assert (choose.kind, choose.title, choose.options, choose.current) == (
        'choose',
        'Choose Sign-in',
        AUTH_OPTIONS,
        'GONE',
    )
    assert (wait.kind, wait.title) == ('wait', 'Signing in to Day AI')
    assert shell.saved() == {'auth': 'oauth', 'include_instructions': True}
    assert SignIn.attempts == 1
    scripted = ScriptedDialogs([Pick(value='oauth')])
    assert await shell.choose_auth(scripted) == ['Saved Sign-in.'], 'already signed in: no browser'
    assert scripted.asked[0].current == 'oauth' and SignIn.attempts == 1
    await shell.loader.close('exit')


async def test_backing_out_of_the_sign_in_wait_leaves_day_ai_signed_out(tmp_path: Path) -> None:
    SignIn.unfinished = True
    shell = Shell(tmp_path, terminal=False)
    await shell.loader.enable('day_ai')
    scripted = ScriptedDialogs([Pick(value='oauth'), Back()])
    with anyio.fail_after(5):
        messages = await shell.choose_auth(scripted)
    assert messages == ['Saved Sign-in.', 'Day AI sign-in cancelled. Run /day_ai login to try again.']
    assert await shell.run_capability() is None
    await shell.loader.close('exit')


async def test_the_key_action_saves_a_masked_token_in_keys(tmp_path: Path) -> None:
    shell = Shell(tmp_path, terminal=False)
    await shell.loader.enable('day_ai')
    scripted = ScriptedDialogs([Pick(value='key'), Type(text=' day-secret ')])
    assert await shell.choose_auth(scripted) == ['Day AI uses the saved key DAY_AI_ACCESS_TOKEN. Manage it in /keys.']
    _, token = scripted.asked
    assert (token.kind, token.title, token.secret) == (
        'text',
        'Day AI access token (saved in /keys as DAY_AI_ACCESS_TOKEN)',
        True,
    )
    assert api_keys.load_keys()['DAY_AI_ACCESS_TOKEN'].get_secret_value() == 'day-secret'
    assert shell.saved() == {'auth': {'name': 'DAY_AI_ACCESS_TOKEN'}, 'include_instructions': True}
    assert b'day-secret' not in shell.path.read_bytes()
    scripted = ScriptedDialogs([Pick(value='key'), Pick(value='enter'), Type(text='other'), Confirm(yes=False)])
    assert await shell.choose_auth(scripted) == [], 'keeping the saved token changes nothing'
    assert scripted.asked[-1].title == 'DAY_AI_ACCESS_TOKEN is already in /keys'
    assert api_keys.load_keys()['DAY_AI_ACCESS_TOKEN'].get_secret_value() == 'day-secret'
    await shell.loader.close('exit')


async def test_backing_out_of_the_sign_in_action_changes_nothing(tmp_path: Path) -> None:
    api_keys.save_key(name='WORK_DAY_AI', value='work-secret')
    shell = Shell(tmp_path, terminal=False, settings={'include_instructions': False})
    await shell.loader.enable('day_ai')
    for answers in ([Back()], [Pick(value='key'), Back()]):
        assert await shell.choose_auth(ScriptedDialogs(answers)) == []
    assert shell.saved() == {'include_instructions': False}
    assert SignIn.attempts == 0
    await shell.loader.close('exit')


async def test_configure_without_a_terminal_points_to_one(tmp_path: Path) -> None:
    shell = Shell(tmp_path, terminal=False)
    assert await shell.loader.command(['enable', 'day_ai']) == '\n'.join(
        ['Enabled day_ai.', 'Configure Day AI from a terminal: /plugins configure day_ai']
    )
    await shell.loader.close('exit')


class Redraw:
    def replace_items(self, items: object) -> None:
        pass
