"""The built-in `linear` plugin: declared disabled, set up in its declared settings menu, key named in `/keys`.

The menu's actions run with `ScriptedDialogs` through `PluginLoader.edit_settings`, as the web UI runs them, and the
terminal path through `configure` with scripted widget runners.
"""

import io
import json
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import TypeGuard

import anyio
import pytest
from fastmcp import Client
from fastmcp.client.auth import OAuth
from fastmcp.client.auth.oauth import TokenStorageAdapter
from fastmcp.client.transports import StreamableHttpTransport
from mcp.shared.auth import OAuthToken
from pydantic import JsonValue
from rich.console import Console
from termflow.tui.menu import MenuResult

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.linear import Linear
from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2.builtin_plugins import linear
from pydantic_clai2.builtin_plugins.linear import ACCOUNT, KEY_NAME, TOKEN_ACCOUNT, reference
from pydantic_clai2.commands import Commands
from pydantic_clai2.config.api_keys import delete_key, key_users, load_keys, rename_key, save_key
from pydantic_clai2.config.credential_store import load_codex_credentials, save_codex_credentials
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.mcp import OAuthSignIn, TokenStore, http_client
from pydantic_clai2.plugins import SessionStart
from pydantic_clai2.plugins.loader import PluginError, PluginLoader
from pydantic_clai2.plugins.menus import Option, SettingsMenu
from pydantic_clai2.runtime.session import WorkspacelessCapability
from pydantic_clai2.ui.menus import dialogs
from tests.clai2.conftest import stored_accounts
from tests.clai2.dialog_script import Answer, Back, Confirm, Finish, Pick, ScriptedDialogs, Type
from tests.clai2.menu_script import UNTIL_CLOSED, Script, pick

Vault = dict[tuple[str, str], str]

CLOSED = MenuResult(cancelled=True)
READ_ONLY_URL = 'https://mcp.linear.app/mcp/readonly'
URL = 'https://mcp.linear.app/mcp'


class SignIn:
    """Stands in for `OAuthSignIn.sign_in`, which would open the browser; it stores tokens as FastMCP would."""

    urls: list[str] = []
    error: Exception | None = None
    unfinished: bool = False
    """The browser sign-in is never completed, so it waits, showing its link, until cancelled."""

    @staticmethod
    async def sign_in(method: OAuthSignIn, *, show: Callable[[str], object]) -> None:
        assert method.name == TOKEN_ACCOUNT
        SignIn.urls.append(method.url)
        if SignIn.error is not None:
            raise SignIn.error
        if SignIn.unfinished:
            show('https://linear.app/oauth/authorize')
            await anyio.sleep_forever()
        await store_sign_in(method.url)


@pytest.fixture(autouse=True)
def stub_sign_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """No browser; `/linear login`'s waiting screen stays open until the sign-in ends."""
    monkeypatch.setattr(OAuthSignIn, 'sign_in', SignIn.sign_in)
    monkeypatch.setattr(SignIn, 'urls', [])
    monkeypatch.setattr(SignIn, 'error', None)
    monkeypatch.setattr(SignIn, 'unfinished', False)
    waiting = Script(lists=[], choices=[UNTIL_CLOSED] * 5, texts=[])
    monkeypatch.setattr('pydantic_clai2.plugins.sign_in.RUNNERS', waiting.runners)


async def store_sign_in(url: str) -> None:
    tokens = TokenStorageAdapter(TokenStore(TOKEN_ACCOUNT), server_url=url)
    await tokens.set_tokens(OAuthToken(access_token='access', token_type='Bearer', expires_in=3600))


def make(tmp_path: Path, *, terminal: bool = False) -> tuple[PluginLoader[None], Commands, io.StringIO]:
    store = SettingsStore(tmp_path / 'settings.db')
    commands = Commands()
    output = io.StringIO()
    loader = PluginLoader[None](
        store=store,
        console=Console(file=output, width=400, force_terminal=terminal),
        commands=commands,
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=[plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'linear'],
    )
    return loader, commands, output


async def declare(loader: PluginLoader[None], settings: dict[str, JsonValue]) -> None:
    await loader.remove('linear')
    await loader.command(['add', 'linear', 'pydantic_clai2.builtin_plugins.linear', json.dumps(settings)])


def saved(tmp_path: Path) -> dict[str, JsonValue]:
    [declaration] = [plugin for plugin in SettingsStore(tmp_path / 'settings.db').plugins() if plugin.id == 'linear']
    return declaration.settings


async def act(loader: PluginLoader[None], key: str, *answers: Answer) -> tuple[Sequence[str], ScriptedDialogs]:
    """Run a menu action as the web UI does: one edit, with the plugin reloaded if its settings changed."""
    scripted = ScriptedDialogs(answers)
    messages = await loader.edit_settings('linear', lambda menu: menu.actions[key](scripted))
    assert scripted.unused == []
    return messages, scripted


def menu_of(loader: PluginLoader[None]) -> SettingsMenu:
    """The loaded plugin's declared menu."""
    [entry] = [entry for entry in loader.entries() if entry.name == 'linear']
    assert entry.loaded is not None
    menu = entry.loaded.plugin.settings_menu()
    assert menu is not None
    return menu


def is_linear(capability: object) -> TypeGuard[Linear[None]]:
    return isinstance(capability, Linear)


def only(loader: PluginLoader[None]) -> Linear[None]:
    """The capability the next run gets with an API key."""
    [capability] = loader.capabilities()
    assert is_linear(capability)
    assert capability.defer_loading
    return capability


async def for_run(loader: PluginLoader[None]) -> Linear[None] | None:
    """What a run gets from the browser sign-in's per-run factory."""
    [entry] = [entry for entry in loader.entries() if entry.name == 'linear']
    assert entry.loaded is not None
    plugin = entry.loaded.plugin
    assert isinstance(plugin, linear.LinearPlugin)
    assert loader.capabilities() == [WorkspacelessCapability(plugin.for_run)]
    capability = await plugin.for_run(RunContext[None](deps=None, model=TestModel(), usage=RunUsage()))
    assert capability is None or capability.defer_loading
    return capability


async def connected(loader: PluginLoader[None]) -> Linear[None]:
    capability = await for_run(loader)
    assert capability is not None
    return capability


def token(loader: PluginLoader[None]) -> str | None:
    """What the next run connects with."""
    capability = only(loader)
    assert callable(capability.auth)
    token = capability.auth(RunContext(deps=None, model=TestModel(), usage=RunUsage()))
    assert not isinstance(token, Awaitable)
    return token


async def run(commands: Commands, text: str) -> str:
    result = commands.execute(text)
    return result if isinstance(result, str) else await result


def test_declared_as_a_disabled_built_in_not_the_raw_capability() -> None:
    [declaration] = [plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'linear']
    assert declaration.factory == 'pydantic_clai2.builtin_plugins.linear'
    assert not declaration.enabled
    assert all(not plugin.factory.startswith('pydantic_ai_harness.linear') for plugin in DEFAULT_PLUGINS)


async def test_each_run_resolves_the_named_key(tmp_path: Path, vault: Vault) -> None:
    loader, _, output = make(tmp_path)
    await loader.load_all()
    assert loader.capabilities() == [], 'disabled until the user enables it'
    save_key(name=KEY_NAME, value='lin_1')
    assert await loader.command(['enable', 'linear']) == (
        'Enabled linear.\nConfigure Linear settings from a terminal: /plugins configure linear'
    )
    assert output.getvalue() == ''
    capability = only(loader)
    assert capability.read_only and capability.include_instructions
    assert token(loader) == 'lin_1'

    save_key(name=KEY_NAME, value='lin_2')
    assert token(loader) == 'lin_2', 'replacing the key in /keys reaches the next run without a reload'
    delete_key(name=KEY_NAME)
    with pytest.raises(
        UserError, match=f'{KEY_NAME} is missing. Restore it in /keys or reconfigure through /plugins configure linear'
    ):
        token(loader)
    await loader.close('exit')


async def test_terminal_menu_saves_each_edit_and_reloads(
    tmp_path: Path, vault: Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_key(name=KEY_NAME, value='lin')
    loader, _, _ = make(tmp_path, terminal=True)
    script = Script(
        lists=[pick('read_only'), pick('include_instructions'), pick('include_instructions'), CLOSED],
        choices=[pick('false'), CLOSED, pick('false')],  # Access -> Read and write; Esc; Server instructions -> out
        texts=[],
    )
    monkeypatch.setattr(dialogs, 'RUNNERS', script.runners)
    assert (await loader.command(['enable', 'linear'])).splitlines() == [
        'Enabled linear.',
        'Linear Access: Read and write.',
        'Linear Server instructions: Leave out.',
    ]
    assert script.opened == ['list', 'choice', 'list', 'choice', 'list', 'choice', 'list']
    assert saved(tmp_path) == {'auth': 'api_key', 'read_only': False, 'include_instructions': False}
    capability = only(loader)
    assert not capability.read_only and not capability.include_instructions, 'reloaded with the new settings'
    assert token(loader) == 'lin'

    monkeypatch.setattr(dialogs, 'RUNNERS', Script(lists=[CLOSED], choices=[], texts=[]).runners)
    assert await loader.command(['configure', 'linear']) == 'Linear settings: nothing changed.'
    await loader.close('exit')


async def test_terminal_menu_signs_in_when_browser_sign_in_is_chosen(
    tmp_path: Path, vault: Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader, _, _ = make(tmp_path, terminal=True)
    script = Script(lists=[pick('auth'), CLOSED], choices=[pick('oauth'), UNTIL_CLOSED], texts=[])
    monkeypatch.setattr(dialogs, 'RUNNERS', script.runners)
    with anyio.fail_after(5):
        message = await loader.command(['enable', 'linear'])
    assert message.splitlines() == [
        'Enabled linear.',
        'Linear Sign-in: Browser sign-in (OAuth).',
        'Signed in to Linear.',
    ]
    assert SignIn.urls == [READ_ONLY_URL], 'the waiting screen closed by itself once signed in'
    assert (await connected(loader)).client is not None
    await loader.close('exit')


async def test_rows_follow_the_sign_in(tmp_path: Path, vault: Vault) -> None:
    loader, _, _ = make(tmp_path)
    await loader.enable('linear')
    menu = menu_of(loader)
    assert set(menu.actions) == {'auth', 'api_key', 'read_only'}
    source = menu.source
    assert source.title == 'Linear settings'
    rows = source.rows()
    assert [(row.label, source.current(row), row.note) for row in rows] == [
        ('Sign-in', 'api_key', ''),
        ('API key', KEY_NAME, f'{KEY_NAME} is missing from /keys'),
        ('Access', 'true', ''),
        ('Server instructions', 'true', ''),
    ]
    save_codex_credentials(account=ACCOUNT, value='{"token": "inline-secret"}')
    key_row = source.rows()[1]
    assert source.current(key_row) == '(invalid choice)'
    assert key_row.note == 'The saved Linear key choice is invalid. Choose again with /plugins configure linear.'
    assert source.reset(key_row) == f'Linear uses {KEY_NAME} from /keys from the next run.'
    assert source.reset(rows[3]) == 'Linear Server instructions: Include (default).'

    await act(loader, 'auth', Pick(value='oauth'), Back())
    menu = menu_of(loader)
    assert [row.key for row in menu.source.rows()] == ['auth', 'read_only', 'include_instructions'], 'no key row'
    await loader.close('exit')


async def test_choosing_browser_sign_in_signs_in_for_the_chosen_access(tmp_path: Path, vault: Vault) -> None:
    loader, _, _ = make(tmp_path)
    await loader.enable('linear')
    messages, scripted = await act(loader, 'auth', Pick(value='oauth'), Finish())
    assert messages == ['Linear Sign-in: Browser sign-in (OAuth).', 'Signed in to Linear.']
    [choose, wait] = scripted.asked
    assert (choose.kind, choose.title, choose.current) == ('choose', 'Choose Sign-in', 'api_key')
    assert choose.options == (
        Option(value='api_key', label='API key from /keys'),
        Option(value='oauth', label='Browser sign-in (OAuth)'),
    )
    assert choose.description.startswith('An API key from /keys')
    assert (wait.kind, wait.title) == ('wait', 'Signing in to Linear')
    assert SignIn.urls == [READ_ONLY_URL]
    assert saved(tmp_path) == {'auth': 'oauth', 'read_only': True, 'include_instructions': True}

    messages, scripted = await act(loader, 'read_only', Pick(value='false'), Finish())
    assert messages == ['Linear Access: Read and write.', 'Signed in to Linear.'], 'tokens belong to one endpoint'
    assert (scripted.asked[0].title, scripted.asked[0].current) == ('Choose Access', 'true')
    assert [option.label for option in scripted.asked[0].options] == ['Read-only', 'Read and write']
    assert SignIn.urls == [READ_ONLY_URL, URL]
    client = (await connected(loader)).client
    assert isinstance(client, Client) and isinstance(client.transport, StreamableHttpTransport)
    assert client.transport.url == URL

    messages, _ = await act(loader, 'auth', Pick(value='oauth'))
    assert messages == ['Linear Sign-in: Browser sign-in (OAuth).'], 'already signed in, so no second sign-in'
    messages, _ = await act(loader, 'read_only', Back())
    assert messages == []
    messages, _ = await act(loader, 'auth', Pick(value='api_key'))
    assert messages == ['Linear Sign-in: API key from /keys.'], 'an API key needs no sign-in'
    assert saved(tmp_path) == {'auth': 'api_key', 'read_only': False, 'include_instructions': True}
    assert len(SignIn.urls) == 2
    await loader.close('exit')


@pytest.mark.parametrize(
    ('error', 'unfinished', 'answer', 'message'),
    [
        (None, True, Back(), 'Linear sign-in cancelled. Run /linear login to try again.'),
        (
            RuntimeError('authorization denied'),
            False,
            Finish(),
            'Could not sign in to Linear: authorization denied. Run /linear login to try again.',
        ),
    ],
)
async def test_menu_sign_in_that_does_not_finish_leaves_linear_signed_out(
    tmp_path: Path, vault: Vault, error: Exception | None, unfinished: bool, answer: Answer, message: str
) -> None:
    SignIn.error, SignIn.unfinished = error, unfinished
    loader, _, output = make(tmp_path)
    await loader.enable('linear')
    with anyio.fail_after(5):
        messages, _ = await act(loader, 'auth', Pick(value='oauth'), answer)
    assert messages == ['Linear Sign-in: Browser sign-in (OAuth).', message]
    assert await for_run(loader) is None
    assert 'Not signed in to Linear. Run /linear login to sign in.' in output.getvalue()
    await loader.close('exit')


async def test_key_row_picks_a_saved_key_and_resets(tmp_path: Path, vault: Vault) -> None:
    save_key(name='GITHUB_TOKEN', value='gh')
    save_key(name='SHARED_LINEAR', value='lin_shared')
    loader, _, _ = make(tmp_path)
    await loader.enable('linear')
    messages, scripted = await act(loader, 'api_key', Pick(value='SHARED_LINEAR'))
    assert messages == ['Linear uses SHARED_LINEAR from /keys from the next run.']
    [choose] = scripted.asked
    assert choose.title == 'Linear API key'
    assert [option.label for option in choose.options] == ['GITHUB_TOKEN', 'SHARED_LINEAR', 'Enter a different API key']
    assert token(loader) == 'lin_shared'
    raw = load_codex_credentials(account=ACCOUNT)
    assert raw is not None and json.loads(raw) == {'token': {'name': 'SHARED_LINEAR'}}
    assert 'lin_' not in json.dumps(saved(tmp_path)) and saved(tmp_path) == {}
    assert key_users(name='SHARED_LINEAR') == ['linear']
    with pytest.raises(ValueError, match='used by linear'):
        rename_key(name='SHARED_LINEAR', new_name='OTHER')

    menu = menu_of(loader)
    [key_row] = [row for row in menu.source.rows() if row.key == 'api_key']
    assert menu.source.reset(key_row) == f'Linear uses {KEY_NAME} from /keys from the next run.'
    assert reference().name == KEY_NAME
    messages, _ = await act(loader, 'api_key', Back())
    assert messages == ['Linear key unchanged.']
    await loader.close('exit')


async def test_missing_key_is_reported_at_load_and_typed_in_the_menu(tmp_path: Path, vault: Vault) -> None:
    loader, _, output = make(tmp_path)
    await loader.enable('linear')
    assert f'Linear: Saved API key {KEY_NAME} is missing' in output.getvalue()
    with pytest.raises(UserError, match='is missing'):
        token(loader)

    messages, scripted = await act(loader, 'api_key', Type(text=' lin_typed '))  # No saved keys: straight to typing.
    assert messages == [f'Linear uses {KEY_NAME} from /keys from the next run.']
    [text] = scripted.asked
    assert (text.kind, text.title, text.secret) == ('text', 'Linear API key', True)
    assert token(loader) == 'lin_typed'
    assert 'lin_' not in (load_codex_credentials(account=ACCOUNT) or ''), 'the entry holds a name, never the key'
    await loader.close('exit')


@pytest.mark.parametrize(
    ('replace', 'stored', 'message'), [(True, 'lin_new', 'Linear uses'), (False, 'lin_old', 'Linear key unchanged.')]
)
async def test_entering_a_key_asks_before_replacing_it(
    tmp_path: Path, vault: Vault, replace: bool, stored: str, message: str
) -> None:
    save_key(name=KEY_NAME, value='lin_old')
    loader, _, _ = make(tmp_path)
    await loader.enable('linear')
    messages, scripted = await act(loader, 'api_key', Pick(value='enter'), Type(text='lin_new'), Confirm(yes=replace))
    assert messages[0].startswith(message)
    assert scripted.asked[-1].title == f'{KEY_NAME} is already in /keys'
    assert load_keys()[KEY_NAME].get_secret_value() == stored
    await loader.close('exit')


async def test_key_deleted_while_choosing(tmp_path: Path, vault: Vault) -> None:
    class Deleting(ScriptedDialogs):
        """Another CLAI deletes the key just as it is picked."""

        async def choose(
            self, *, title: str, options: Sequence[Option], current: str | None = None, description: str = ''
        ) -> str | None:
            picked = await super().choose(title=title, options=options, current=current, description=description)
            delete_key(name='GONE')
            return picked

    save_key(name='GONE', value='lin_gone')
    assert await linear.pick_key(Deleting([Pick(value='GONE')])) == [
        'The selected API key no longer exists. Select a saved key again through /plugins configure linear.'
    ]
    assert load_codex_credentials(account=ACCOUNT) is None


@pytest.mark.parametrize('settings', [{'api_key': 'lin_secret'}, {'oauth': True}, {'auth': 'token'}])
async def test_settings_carry_no_credential(tmp_path: Path, settings: dict[str, JsonValue]) -> None:
    loader, _, _ = make(tmp_path)
    with pytest.raises(PluginError, match='validation error'):
        await declare(loader, settings)
    assert loader.capabilities() == []


@pytest.mark.parametrize(
    ('read_only', 'url'), [(True, 'https://mcp.linear.app/mcp/readonly'), (False, 'https://mcp.linear.app/mcp')]
)
async def test_oauth_signs_in_with_keyring_tokens(tmp_path: Path, vault: Vault, read_only: bool, url: str) -> None:
    loader, commands, output = make(tmp_path)
    with anyio.fail_after(5):
        await declare(loader, {'auth': 'oauth', 'read_only': read_only})
        assert await for_run(loader) is None, 'no tools, and no browser, until signed in'
    assert output.getvalue().count('Not signed in to Linear. Run /linear login to sign in.') == 1
    assert SignIn.urls == []
    assert await run(commands, '/linear') == 'Not signed in to Linear. Run /linear login to sign in.'
    assert await run(commands, '/linear login') == 'Signed in to Linear.'
    assert SignIn.urls == [url]
    assert await run(commands, '/linear status') == 'Signed in to Linear.'

    capability = await connected(loader)
    assert (await connected(loader)).client is not capability.client, 'each run connects afresh'
    assert not capability.read_only, 'the URL carries read_only, not tool annotations'
    client = capability.client
    assert isinstance(client, Client)
    transport = client.transport
    assert isinstance(transport, StreamableHttpTransport)
    assert transport.url == url
    assert transport.httpx_client_factory is http_client, 'redirects stay off, as for /mcp servers'
    assert isinstance(transport.auth, OAuth)

    with pytest.raises(ValueError, match=r'Usage: /linear \[login \| logout \| status\]'):
        await run(commands, '/linear other')
    assert await run(commands, '/linear logout') == 'Signed out of Linear. Run /linear login to sign in again.'
    assert stored_accounts() == set()
    assert await for_run(loader) is None

    await loader.disable('linear')
    assert 'linear' not in {command.name for command in commands}
