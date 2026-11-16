"""The built-in `notion` plugin: a settings menu for `Notion`'s options, with its key chosen from `/keys` by name."""

import inspect
import io
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import anyio
import pytest
from fastmcp import Client
from fastmcp.client.auth.oauth import TokenStorageAdapter
from fastmcp.client.transports import StreamableHttpTransport
from mcp.shared.auth import OAuthToken
from pydantic import JsonValue
from rich.console import Console
from termflow.tui import MenuItem
from termflow.tui.menu import MenuResult
from termflow.tui.textinput import TextInputResult

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.notion import Notion
from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2.builtin_plugins import notion
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import api_keys
from pydantic_clai2.config.api_keys import KeyReference
from pydantic_clai2.config.credential_store import load_codex_credentials, save_codex_credentials
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.mcp import OAuthSignIn, SignIn as MCPSignIn, TokenStore, http_client
from pydantic_clai2.plugins import PluginHost, SessionStart, load_plugin
from pydantic_clai2.plugins.loader import PluginError, PluginLoader
from pydantic_clai2.plugins.menus import Option, SettingsMenu
from pydantic_clai2.ui.menus import dialogs
from pydantic_clai2.ui.menus.field_menu import FieldMenu
from pydantic_clai2.ui.menus.plugin_menu import PluginMenu, open_plugins_menu
from tests.clai2.conftest import stored_accounts
from tests.clai2.dialog_script import Answer, Asked, Back, Confirm, Finish, Pick, ScriptedDialogs, Type
from tests.clai2.menu_script import UNTIL_CLOSED, Script, pick, typed
from tests.clai2.test_workspace_defaults import per_run

Vault = dict[tuple[str, str], str]
BUILTIN = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'notion')
CLOSE = MenuResult(cancelled=True)
LABEL = 'Notion OAuth access token (saved in /keys as NOTION_API_KEY)'
SIGNED_OUT = 'Not signed in to Notion. Run /notion login to sign in.'
TYPING = Asked(
    kind='text',
    title=LABEL,
    label='Token',
    placeholder='Paste the token; it is saved in /keys as NOTION_API_KEY',
    secret=True,
)
AUTH = Asked(
    kind='choose',
    title='Choose Sign-in',
    options=(Option(value='oauth', label='browser'), Option(value='key', label='key only')),
    current='oauth',
)


class SignIn:
    """Stands in for `OAuthSignIn.sign_in`, which would open the browser; it stores tokens as FastMCP would."""

    attempts: int = 0
    error: Exception | None = None
    unfinished: bool = False
    """The browser sign-in is never completed, so it waits until cancelled."""

    @staticmethod
    async def sign_in(method: OAuthSignIn, *, show: Callable[[str], object]) -> None:
        assert method is notion.SIGN_IN
        SignIn.attempts += 1
        if SignIn.error is not None:
            raise SignIn.error
        if SignIn.unfinished:
            show('https://notion.example/authorize')
            await anyio.sleep_forever()
        await store_sign_in()


@pytest.fixture(autouse=True)
def sign_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(OAuthSignIn, 'sign_in', SignIn.sign_in)
    monkeypatch.setattr(SignIn, 'attempts', 0)
    monkeypatch.setattr(SignIn, 'error', None)
    monkeypatch.setattr(SignIn, 'unfinished', False)


async def store_sign_in() -> None:
    tokens = TokenStorageAdapter(TokenStore(notion.TOKEN_ACCOUNT), server_url=notion.NOTION_MCP_URL)
    await tokens.set_tokens(OAuthToken(access_token='access', token_type='Bearer', expires_in=3600))


class Shell:
    """A loader plus what a test inspects: printed output, commands, and the settings file."""

    def __init__(self, tmp_path: Path, settings: dict[str, JsonValue] | None = None) -> None:
        self.store = SettingsStore(tmp_path / 'settings.db')
        self.output = io.StringIO()
        self.commands = Commands()
        declaration = BUILTIN if settings is None else BUILTIN.model_copy(update={'settings': settings})
        self.loader = PluginLoader[None](
            store=self.store,
            console=Console(file=self.output, width=200, force_terminal=True, color_system=None),
            commands=self.commands,
            session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=self.store.load()),
            builtin=(declaration,),
        )

    def saved(self) -> dict[str, JsonValue]:
        [declaration] = self.store.plugins()
        return declaration.settings

    async def run(self, text: str) -> str:
        result = self.commands.execute(text)
        assert inspect.isawaitable(result), '`/notion` reaches the keyring off the event loop'
        return await result

    async def capability(self) -> Notion[None] | None:
        """What a run gets from the per-run factory the plugin registers."""
        [capability] = self.loader.capabilities()
        result = per_run(capability)(RunContext[None](deps=None, model=TestModel(), usage=RunUsage()))
        assert inspect.isawaitable(result)
        connected = await result
        assert connected is None or (isinstance(connected, Notion) and connected.defer_loading)
        return connected

    async def built(self) -> Notion[None]:
        """The run's `Notion`, which must connect."""
        connected = await self.capability()
        assert connected is not None
        return connected


def script(
    monkeypatch: pytest.MonkeyPatch,
    lists: list[MenuResult],
    choices: list[MenuResult] | None = None,
    texts: list[TextInputResult] | None = None,
) -> Script:
    """Script the terminal settings menu's widgets, its actions' dialogs included."""
    scripted = Script(lists=[*lists, CLOSE], choices=choices or [], texts=texts or [])
    monkeypatch.setattr(dialogs, 'RUNNERS', scripted.runners)
    return scripted


async def act(menu: SettingsMenu, row: str, answers: list[Answer]) -> tuple[list[str], ScriptedDialogs]:
    """Run the menu action for `row` with scripted dialogs, as the web UI runs it; its messages and what it asked."""
    scripted = ScriptedDialogs(answers)
    messages = list(await menu.actions[row](scripted))
    assert scripted.unused == []
    return messages, scripted


def settings_menu(**settings: JsonValue) -> SettingsMenu:
    """The declared menu of the plugin over a host outside the loader, which keeps saved settings for this load."""
    host = PluginHost[None](name='notion', console=Console(file=io.StringIO()), settings=settings)
    menu = load_plugin(notion.NotionPlugin, host).plugin.settings_menu()
    assert menu is not None
    return menu


def source(**settings: JsonValue) -> notion.NotionSource[None]:
    """A settings menu source over a host outside the loader, which keeps saved settings for this load."""
    return notion.NotionSource(PluginHost[None](name='notion', console=Console(file=io.StringIO()), settings=settings))


def reset(key: str) -> MenuResult:
    """What `R` on a row hands back to the settings menu's loop."""
    return FieldMenu(source()).reset_marker(None, MenuItem(key, value=key))


def test_declared_as_a_disabled_builtin_without_settings() -> None:
    assert BUILTIN.factory == 'pydantic_clai2.builtin_plugins.notion'
    assert not BUILTIN.enabled and BUILTIN.settings == {}
    assert all(plugin.factory != 'pydantic_ai_harness.notion:Notion' for plugin in DEFAULT_PLUGINS)


async def test_enable_opens_the_menu_and_every_option_saves_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('NOTION_ACCESS_TOKEN', 'the-environment-is-not-a-source')
    shown = script(
        monkeypatch,
        lists=[pick('key'), pick('auth'), pick('read_only'), pick('include_instructions')],
        choices=[pick('key'), pick('true'), pick('false')],
        texts=[typed(' ntn-new ')],
    )
    shell = Shell(tmp_path)
    assert await shell.loader.command(['enable', 'notion']) == '\n'.join(
        [
            'Enabled notion.',
            'Notion uses the saved key NOTION_API_KEY. Manage it in /keys.',
            'Saved Sign-in: key only.',
            'Saved Sign-in.',
            'Saved Tools.',
            'Saved Server instructions.',
        ]
    )
    assert shown.opened == ['list', 'text', 'list', 'choice', 'list', 'choice', 'list', 'choice', 'list']
    assert api_keys.load_keys()['NOTION_API_KEY'].get_secret_value() == 'ntn-new'
    assert load_codex_credentials(account='notion') == '{"token":{"name":"NOTION_API_KEY"}}'
    assert shell.saved() == {'auth': 'key', 'read_only': True, 'include_instructions': False}
    assert 'ntn-new' not in repr(shell.store.plugins())
    capability = await shell.built()
    assert (capability.auth, capability.read_only, capability.include_instructions) == ('ntn-new', True, False)


async def test_reopening_repicks_a_saved_key_and_resets_options_without_reinstalling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, vault: Vault
) -> None:
    api_keys.save_key(name='NOTION_API_KEY', value='ntn-first')
    api_keys.save_key(name='TEAM_NOTION', value='ntn-team')
    shell = Shell(tmp_path, {'read_only': True})
    script(monkeypatch, lists=[pick('key')], choices=[pick('NOTION_API_KEY')])
    await shell.loader.command(['enable', 'notion'])
    assert (await shell.built()).auth == 'ntn-first'
    script(monkeypatch, lists=[pick('key'), reset('read_only')], choices=[pick('TEAM_NOTION')])
    assert await shell.loader.command(['configure', 'notion']) == (
        'Notion uses the saved key TEAM_NOTION. Manage it in /keys.\nReset Tools.'
    )
    assert shell.saved() == {'auth': 'key', 'read_only': False, 'include_instructions': True}
    assert (await shell.built()).auth == 'ntn-team'
    script(monkeypatch, lists=[reset('key')])
    assert (
        await shell.loader.configure('notion')
        == 'Notion no longer uses a saved key and uses the browser sign-in. The key itself stays in /keys.'
    )
    assert shell.saved() == {'auth': 'oauth', 'read_only': False, 'include_instructions': True}
    assert await shell.capability() is None, 'without a key or a sign-in, the run gets no Notion tools'
    assert set(api_keys.load_keys()) == {'NOTION_API_KEY', 'TEAM_NOTION'}


async def test_a_chosen_key_is_resolved_each_run_and_fails_closed(tmp_path: Path) -> None:
    api_keys.save_key(name='NOTION_API_KEY', value='ntn-first')
    notion.select_key(KeyReference(name='NOTION_API_KEY'))
    shell = Shell(tmp_path, {'auth': 'key'})
    await shell.loader.enable('notion')
    assert (await shell.built()).auth == 'ntn-first'
    api_keys.save_key(name='NOTION_API_KEY', value='ntn-second')
    assert (await shell.built()).auth == 'ntn-second', 'replacing the key in /keys reaches every consumer'
    assert api_keys.key_users(name='NOTION_API_KEY') == ['notion']
    with pytest.raises(ValueError, match='used by notion'):
        api_keys.rename_key(name='NOTION_API_KEY', new_name='OTHER')
    api_keys.delete_key(name='NOTION_API_KEY')
    with pytest.raises(UserError, match='NOTION_API_KEY is missing'):
        await shell.built()


def test_the_menu_declares_its_rows_and_the_rows_that_ask_more() -> None:
    menu = settings_menu()
    assert menu.source.title == 'Notion'
    assert [(row.key, row.label) for row in menu.source.rows()] == [
        ('key', 'Key'),
        ('auth', 'Sign-in'),
        ('read_only', 'Tools'),
        ('include_instructions', 'Server instructions'),
    ]
    assert list(menu.actions) == ['key', 'auth'] and menu.searchable


async def test_configure_outside_a_terminal_explains_how() -> None:
    host = PluginHost[None](name='notion', console=Console(file=io.StringIO()), settings={})
    assert await load_plugin(notion.NotionPlugin, host).plugin.configure() == (
        'Configure Notion from a terminal: /plugins configure notion'
    )


async def test_the_key_action_switches_to_key_only_once(tmp_path: Path) -> None:
    """With nothing in `/keys` it goes straight to a masked input; a later pick keeps key-only as it is."""
    menu = settings_menu()
    messages, asked = await act(menu, 'key', [Type(text='ntn-new')])
    assert messages == [
        'Notion uses the saved key NOTION_API_KEY. Manage it in /keys.',
        'Saved Sign-in: key only.',
    ]
    assert asked.asked == [TYPING]
    assert menu.source.current(menu.source.rows()[1]) == 'key'
    messages, asked = await act(menu, 'key', [Pick(value='NOTION_API_KEY')])
    assert messages == ['Notion uses the saved key NOTION_API_KEY. Manage it in /keys.']
    [choosing] = asked.asked
    assert (choosing.kind, choosing.title) == ('choose', LABEL)
    assert [option.label for option in choosing.options] == ['NOTION_API_KEY', 'Enter a different API key']
    assert api_keys.load_keys()['NOTION_API_KEY'].get_secret_value() == 'ntn-new'


@pytest.mark.parametrize(
    ('confirm', 'value'), [(Confirm(yes=True), 'ntn-new'), (Confirm(yes=False), 'ntn-old'), (Back(), 'ntn-old')]
)
async def test_entering_a_new_value_asks_before_replacing_a_shared_key(confirm: Answer, value: str) -> None:
    api_keys.save_key(name='NOTION_API_KEY', value='ntn-old')
    messages, asked = await act(settings_menu(), 'key', [Pick(value='enter'), Type(text='ntn-new'), confirm])
    assert [dialog.kind for dialog in asked.asked] == ['choose', 'text', 'confirm']
    assert asked.asked[1] == TYPING
    assert (asked.asked[2].title, asked.asked[2].yes) == (
        'NOTION_API_KEY is already in /keys',
        'Replace NOTION_API_KEY for every plugin and connection that uses it',
    )
    assert bool(messages) == (value == 'ntn-new')
    assert api_keys.load_keys()['NOTION_API_KEY'].get_secret_value() == value


@pytest.mark.parametrize('entry', [Back(), Type(text='   ')])
async def test_cancelled_or_blank_entry_changes_nothing(entry: Answer) -> None:
    assert (await act(settings_menu(), 'key', [entry]))[0] == []
    assert load_codex_credentials(account='notion') is None and api_keys.load_keys() == {}


async def test_closing_the_terminal_menu_unchanged_says_so(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shell = Shell(tmp_path)
    await shell.loader.enable('notion')
    script(monkeypatch, lists=[pick('key')], texts=[TextInputResult(cancelled=True)])
    assert await shell.loader.configure('notion') == 'Notion: nothing changed.'


async def test_stored_browser_sign_in_without_a_chosen_key(tmp_path: Path) -> None:
    await store_sign_in()
    shell = Shell(tmp_path, {'read_only': True})
    await shell.loader.enable('notion')
    capability = await shell.built()
    assert capability.auth is None and capability.read_only
    client = capability.client
    assert isinstance(client, Client)
    transport = client.transport
    assert isinstance(transport, StreamableHttpTransport)
    assert transport.url == notion.NOTION_MCP_URL and isinstance(transport.auth, MCPSignIn)
    assert transport.httpx_client_factory is http_client
    assert (await shell.built()).client is not client, 'each run reloads the sign-in, so logout applies to the next'
    assert SignIn.attempts == 0 and SIGNED_OUT not in shell.output.getvalue()


@pytest.mark.parametrize('settings', [{}, {'auth': 'oauth'}])
async def test_signed_out_adds_no_tools_and_says_how_to_sign_in_without_a_browser(
    tmp_path: Path, settings: dict[str, JsonValue]
) -> None:
    """Loading and runs never open the browser, so an unfinished sign-in cannot hold up every prompt."""
    SignIn.unfinished = True
    shell = Shell(tmp_path, settings)
    with anyio.fail_after(5):
        await shell.loader.enable('notion')
        assert await shell.capability() is None
    assert shell.output.getvalue().count(SIGNED_OUT) == 1
    assert SignIn.attempts == 0


async def test_notion_login_signs_in_and_the_next_run_connects_without_a_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        'pydantic_clai2.plugins.sign_in.RUNNERS', Script(lists=[], choices=[UNTIL_CLOSED], texts=[]).runners
    )
    shell = Shell(tmp_path)
    await shell.loader.enable('notion')
    assert await shell.run('/notion') == SIGNED_OUT
    assert await shell.run('/notion login') == 'Signed in to Notion.'
    assert isinstance((await shell.built()).client, Client), 'no reload needed'
    assert await shell.run('/notion status') == 'Signed in to Notion.'
    assert SignIn.attempts == 1


@pytest.mark.parametrize(
    ('error', 'unfinished', 'message'),
    [
        (None, True, 'Notion sign-in cancelled. Run /notion login to try again.'),
        (
            RuntimeError('authorization denied'),
            False,
            'Could not sign in to Notion: authorization denied. Run /notion login to try again.',
        ),
    ],
)
async def test_notion_login_cancelled_or_failed_stays_signed_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception | None, unfinished: bool, message: str
) -> None:
    SignIn.error, SignIn.unfinished = error, unfinished
    monkeypatch.setattr('pydantic_clai2.plugins.sign_in.RUNNERS', Script(lists=[], choices=[CLOSE], texts=[]).runners)
    shell = Shell(tmp_path)
    await shell.loader.enable('notion')
    with anyio.fail_after(5):
        assert await shell.run('/notion login') == message
    assert await shell.capability() is None


async def test_choosing_the_browser_in_the_terminal_menu_signs_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shell = Shell(tmp_path)
    await shell.loader.enable('notion')
    script(monkeypatch, lists=[pick('auth')], choices=[pick('oauth'), UNTIL_CLOSED])
    assert await shell.loader.configure('notion') == 'Saved Sign-in.\nSigned in to Notion.'
    assert shell.saved() == {'auth': 'oauth', 'read_only': False, 'include_instructions': True}
    assert isinstance((await shell.built()).client, Client)
    assert SignIn.attempts == 1


async def test_choosing_the_browser_signs_in_behind_the_waiting_dialog() -> None:
    messages, asked = await act(settings_menu(auth='key'), 'auth', [Pick(value='oauth'), Finish()])
    assert messages == ['Saved Sign-in.', 'Signed in to Notion.']
    assert asked.asked[0] == replace(AUTH, current='key')
    assert (asked.asked[1].kind, asked.asked[1].title) == ('wait', 'Signing in to Notion')
    assert SignIn.attempts == 1


@pytest.mark.parametrize(
    ('error', 'unfinished', 'answer', 'message'),
    [
        (None, True, Back(), 'Notion sign-in cancelled. Run /notion login to try again.'),
        (
            RuntimeError('authorization denied'),
            False,
            Finish(),
            'Could not sign in to Notion: authorization denied. Run /notion login to try again.',
        ),
    ],
)
async def test_menu_sign_in_cancelled_or_failed_stays_signed_out(
    error: Exception | None, unfinished: bool, answer: Answer, message: str
) -> None:
    SignIn.error, SignIn.unfinished = error, unfinished
    with anyio.fail_after(5):
        messages, _ = await act(settings_menu(), 'auth', [Pick(value='oauth'), answer])
    assert messages == ['Saved Sign-in.', message]
    assert SignIn.attempts == 1
    assert not notion.SIGN_IN.signed_in()


async def test_menu_signs_in_only_for_the_browser_when_not_signed_in() -> None:
    menu = settings_menu()
    assert (await act(menu, 'auth', [Back()]))[0] == []
    messages, asked = await act(menu, 'auth', [Pick(value='key')])
    assert messages == ['Saved Sign-in.'] and asked.asked == [AUTH]
    await store_sign_in()
    assert (await act(menu, 'auth', [Pick(value='oauth')]))[0] == ['Saved Sign-in.']
    assert SignIn.attempts == 0


async def test_key_only_mode_warns_and_never_opens_a_browser(tmp_path: Path) -> None:
    shell = Shell(tmp_path, {'auth': 'key'})
    await shell.loader.enable('notion')
    assert 'Notion has no key selected, so runs fail. Choose one with /plugins configure notion.' in (
        shell.output.getvalue()
    )
    with pytest.raises(UserError, match='No Notion key is selected'):
        await shell.built()
    assert await shell.run('/notion status') == 'No Notion key is selected. Choose one with /plugins configure notion.'
    api_keys.save_key(name='NOTION_API_KEY', value='ntn-token')
    notion.select_key(KeyReference(name='NOTION_API_KEY'))
    await shell.loader.reload('notion')
    assert shell.output.getvalue().count('no key selected') == 1, 'a chosen key silences the warning'
    assert await shell.run('/notion') == 'Notion connects with NOTION_API_KEY from /keys.'
    assert SIGNED_OUT not in shell.output.getvalue()
    assert SignIn.attempts == 0


async def test_browser_sign_in_is_the_default_and_ignores_a_chosen_key(tmp_path: Path) -> None:
    api_keys.save_key(name='NOTION_API_KEY', value='ntn-token')
    notion.select_key(KeyReference(name='NOTION_API_KEY'))
    await store_sign_in()
    shell = Shell(tmp_path)
    await shell.loader.enable('notion')
    capability = await shell.built()
    assert capability.auth is None and isinstance(capability.client, Client)
    assert await shell.run('/notion status') == 'Signed in to Notion.'


async def test_automatic_saved_by_an_earlier_build_means_the_browser(tmp_path: Path) -> None:
    assert notion.NotionSettings.model_validate({'auth': None}) == notion.NotionSettings()
    api_keys.save_key(name='NOTION_API_KEY', value='ntn-token')
    notion.select_key(KeyReference(name='NOTION_API_KEY'))
    shell = Shell(tmp_path, {'auth': None, 'read_only': True})
    await shell.loader.enable('notion')
    assert await shell.capability() is None, 'the chosen key is not used, and no browser opens'
    assert shell.output.getvalue().count(SIGNED_OUT) == 1
    await store_sign_in()
    capability = await shell.built()
    assert capability.auth is None and isinstance(capability.client, Client) and capability.read_only
    assert SignIn.attempts == 0


async def test_invalid_selection_fails_closed_and_the_menu_offers_a_new_choice(tmp_path: Path) -> None:
    save_codex_credentials(account='notion', value='{"token": "ntn-inline"}')
    shell = Shell(tmp_path, {'auth': 'key'})
    await shell.loader.enable('notion')
    with pytest.raises(UserError, match='selection is invalid'):
        await shell.built()
    assert shell.output.getvalue() == '', 'the run reports the invalid selection instead'
    menu = source(auth='key')
    assert [menu.current(row) for row in menu.rows()] == ['(invalid; choose again)', 'key', 'false', 'true']


def test_menu_validates_like_saving_would() -> None:
    menu = source(read_only=True)
    [key, auth, tools, _] = menu.rows()
    assert menu.problem(auth, 'browser') is not None
    assert menu.problem(auth, 'auto') is not None, 'automatic is no longer offered'
    assert menu.current(auth) == 'oauth'
    assert (
        menu.reset(key)
        == 'Notion no longer uses a saved key and uses the browser sign-in. The key itself stays in /keys.'
    )
    assert menu.problem(tools, 'maybe') is not None
    assert menu.problem(tools, 'false') is None
    assert menu.current(tools) == 'true', 'checking a value does not save it'


@pytest.mark.parametrize('settings', [{'token': 'ntn-secret'}, {'api_key': {'name': 'NOTION_API_KEY'}}, {'auth': 'x'}])
async def test_settings_cannot_hold_a_secret_or_unknown_options(tmp_path: Path, settings: dict[str, JsonValue]) -> None:
    shell = Shell(tmp_path, settings)
    with pytest.raises(PluginError):
        await shell.loader.enable('notion')
    assert shell.loader.capabilities() == []


async def test_plugins_menu_enabling_opens_the_settings_menu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    await store_sign_in()
    shell = Shell(tmp_path)
    script(monkeypatch, lists=[pick('read_only')], choices=[pick('true')])
    notices: list[str | None] = []

    def run(menu: PluginMenu[None]) -> MenuResult:
        notices.append(menu.notice)
        item, save_and_close = menu.items()
        if len(notices) > 1:
            return MenuResult(item=save_and_close)
        assert menu.configure(Redraw(), item) is None
        assert menu.notice == 'Enable notion to configure it.'
        menu.notice = None
        result = menu.toggle(Redraw(), item)
        assert result is not None
        return result

    assert await open_plugins_menu(shell.loader, run=run) == 'Saved Tools.'
    assert notices == [None, 'Saved Tools.']
    assert (await shell.built()).read_only


async def test_configure_needs_a_loaded_plugin_with_a_menu(tmp_path: Path) -> None:
    shell = Shell(tmp_path)
    with pytest.raises(ValueError, match='not loaded; enable it before configuring'):
        await shell.loader.command(['configure', 'notion'])
    shell.store.save_plugin(
        BUILTIN.model_copy(update={'id': 'plain', 'factory': 'pydantic_clai2.builtin_plugins.repo_context'})
    )
    assert await shell.loader.command(['enable', 'plain']) == 'Enabled plain.'
    with pytest.raises(ValueError, match='no settings menu'):
        await shell.loader.configure('plain')

    def run(menu: PluginMenu[None]) -> MenuResult:
        plain = next(item for item in menu.items() if item.value == 'plain')
        assert menu.configure(Redraw(), MenuItem('none', value=None)) is None
        assert menu.configure(Redraw(), plain) is None
        assert menu.notice == 'plain has no settings menu.'
        return MenuResult(cancelled=True)

    assert await open_plugins_menu(shell.loader, run=run) == ''


async def test_logout_forgets_the_sign_in_and_the_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, vault: Vault
) -> None:
    api_keys.save_key(name='NOTION_API_KEY', value='ntn-token')
    notion.select_key(KeyReference(name='NOTION_API_KEY'))
    await store_sign_in()
    shell = Shell(tmp_path)
    await shell.loader.enable('notion')
    [command] = [command for command in shell.commands if command.name == 'notion']
    assert list(command.complete([''])) == ['login', 'logout', 'status']
    assert list(command.complete(['logout', ''])) == []
    with pytest.raises(ValueError, match=r'Usage: /notion \[login \| logout \| status\]'):
        await shell.run('/notion key')
    assert await shell.run('/notion logout') == (
        'Signed out of Notion. Run /notion login to sign in again. '
        'Cleared the selected key; the key itself stays in /keys.'
    )
    assert stored_accounts() == {'api-keys'}, 'only the named key remains'
    assert await shell.capability() is None
    assert SignIn.attempts == 0


class Redraw:
    def replace_items(self, items: object) -> None:
        pass
