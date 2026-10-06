"""The built-in `posthog` plugin: its settings menu, its `/keys` reference, and the connection it builds."""

import io
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import anyio
import httpx
import keyring
import pytest
from fastmcp import Client
from fastmcp.client.auth.oauth import TokenStorageAdapter
from fastmcp.client.transports import StreamableHttpTransport
from keyring.errors import KeyringLocked
from mcp.shared.auth import OAuthToken
from pydantic import JsonValue
from rich.console import Console
from termflow.tui.menu import MenuResult
from termflow.tui.textinput import TextInputResult

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.posthog import PostHog
from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2.builtin_plugins import posthog
from pydantic_clai2.builtin_plugins.posthog import EU_URL, EVERY_GROUP, US_URL, PostHogSource, SavedKeyAuth
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import api_keys
from pydantic_clai2.config.api_keys import KeyReference
from pydantic_clai2.config.credential_store import load_codex_credentials, save_codex_credentials
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.mcp import OAuthSignIn, SignIn as MCPSignIn, TokenStore
from pydantic_clai2.plugins import PluginHost, SessionStart
from pydantic_clai2.plugins.loader import PluginError, PluginLoader, PluginSettingsError
from pydantic_clai2.ui.menus.field_menu import CUSTOM, is_save_and_close, save_and_close_item
from pydantic_clai2.ui.menus.plugin_menu import PluginMenu, open_plugins_menu
from tests.clai2.menu_script import UNTIL_CLOSED, Script, pick, typed

pytestmark = pytest.mark.anyio

BUILTIN = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'posthog')
CLOSE = MenuResult(cancelled=True)
KEY = 'POSTHOG_PERSONAL_API_KEY'
NOT_SIGNED_IN = 'Not signed in to PostHog. Run /posthog login to sign in.'


class SignIn:
    """Stands in for `OAuthSignIn.sign_in`, which would open the browser; it stores tokens as FastMCP would."""

    attempts: int = 0
    error: Exception | None = None
    unfinished: bool = False
    """The browser sign-in is never completed, so it waits until cancelled."""

    @staticmethod
    async def sign_in(method: OAuthSignIn, *, show: Callable[[str], object]) -> None:
        assert method.name == posthog.TOKENS and method.service == 'PostHog'
        SignIn.attempts += 1
        if SignIn.error is not None:
            raise SignIn.error
        if SignIn.unfinished:
            await anyio.sleep_forever()
        await store_sign_in(method.url)


@pytest.fixture(autouse=True)
def sign_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(OAuthSignIn, 'sign_in', SignIn.sign_in)
    monkeypatch.setattr(SignIn, 'attempts', 0)
    monkeypatch.setattr(SignIn, 'error', None)
    monkeypatch.setattr(SignIn, 'unfinished', False)


async def store_sign_in(url: str = US_URL) -> None:
    tokens = TokenStorageAdapter(TokenStore(posthog.TOKENS), server_url=url)
    await tokens.set_tokens(OAuthToken(access_token='access', token_type='Bearer', expires_in=3600))


class Shell:
    """A loader in a terminal plus what a test inspects: printed output and the settings file."""

    def __init__(self, tmp_path: Path, settings: dict[str, JsonValue] | None = None, *, terminal: bool = True) -> None:
        self.path = tmp_path / 'config.db'
        self.store = SettingsStore(self.path)
        self.output = io.StringIO()
        self.commands = Commands()
        declaration = BUILTIN if settings is None else BUILTIN.model_copy(update={'settings': settings})
        self.loader: PluginLoader[None] = PluginLoader(
            store=self.store,
            console=Console(file=self.output, width=200, force_terminal=terminal),
            commands=self.commands,
            session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=self.store.load()),
            builtin=(declaration,),
        )

    def saved(self) -> dict[str, JsonValue]:
        [declaration] = self.store.plugins()
        assert declaration.enabled
        return declaration.settings

    def capability(self) -> PostHog[None]:
        [capability] = self.loader.capabilities()
        assert isinstance(capability, PostHog)
        return capability  # pyright: ignore[reportUnknownVariableType]

    async def run_capability(self) -> PostHog[None] | None:
        """What a run gets from the browser sign-in's per-run factory."""
        [entry] = self.loader.entries()
        assert entry.loaded is not None
        plugin = entry.loaded.plugin
        assert isinstance(plugin, posthog.PostHogPlugin)
        assert self.loader.capabilities() == [plugin.for_run]
        return await plugin.for_run(RunContext[None](deps=None, model=TestModel(), usage=RunUsage()))


def script(
    monkeypatch: pytest.MonkeyPatch,
    lists: list[MenuResult],
    choices: list[MenuResult] | None = None,
    texts: list[TextInputResult] | None = None,
) -> Script:
    scripted = Script(lists=[*lists, CLOSE], choices=choices or [], texts=texts or [])
    monkeypatch.setattr(posthog, 'RUNNERS', scripted.runners)
    return scripted


def key_choice(monkeypatch: pytest.MonkeyPatch, choice: str | KeyReference | None) -> list[str]:
    """Answer `prompt_api_key`, whose saved-key list needs a real terminal, and record its labels."""
    labels: list[str] = []

    async def prompt_api_key(*, prompt: object, label: str, optional: bool = False) -> str | KeyReference | None:
        labels.append(label)
        return choice

    monkeypatch.setattr(posthog, 'prompt_api_key', prompt_api_key)
    return labels


def transport(capability: PostHog[None]) -> StreamableHttpTransport:
    client = capability.client
    assert isinstance(client, Client)
    result = client.transport
    assert isinstance(result, StreamableHttpTransport)
    return result


def bearer() -> str:
    request = httpx.Request('GET', US_URL)
    flow = SavedKeyAuth().auth_flow(request)
    return next(flow).headers['Authorization']


async def async_bearer() -> str:
    flow = SavedKeyAuth().async_auth_flow(httpx.Request('GET', US_URL))
    return (await flow.__anext__()).headers['Authorization']


def test_declared_disabled_with_no_settings() -> None:
    assert BUILTIN.factory == 'pydantic_clai2.builtin_plugins.posthog'
    assert not BUILTIN.enabled
    assert BUILTIN.settings == {}


async def test_enable_opens_the_menu_and_every_option_saves_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(KEY, 'phx_the_environment_is_not_a_source')
    labels = key_choice(monkeypatch, ' phx_new ')
    shown = script(
        monkeypatch,
        lists=[
            pick('key'),
            pick('url'),
            pick('read_only'),
            pick('features'),
            pick('mode'),
            pick('project_id'),
            pick('organization_id'),
            pick('include_instructions'),
        ],
        choices=[
            pick(EU_URL),
            pick('false'),
            pick('flags'),
            pick('sql'),
            CLOSE,
            pick('tools'),
            pick('false'),
        ],
        texts=[typed('12345'), typed('0190-abcd')],
    )
    shell = Shell(tmp_path)
    assert await shell.loader.command(['enable', 'posthog']) == '\n'.join(
        [
            'Enabled posthog.',
            f'PostHog connects with {KEY} from /keys.',
            'Saved Region.',
            'Saved Tools.',
            'Saved Feature groups.',
            'Saved Server mode.',
            'Saved Project ID.',
            'Saved Organization ID.',
            'Saved Server instructions.',
        ]
    )
    assert 'PostHog has no key yet' in shell.output.getvalue()
    assert labels == [f'PostHog personal API key (new keys are saved in /keys as {KEY})']
    assert shown.opened.count('list') == 9
    assert api_keys.load_keys()[KEY].get_secret_value() == 'phx_new'
    assert shell.saved() == {
        'auth': 'key',
        'url': EU_URL,
        'read_only': False,
        'features': ['flags', 'sql'],
        'mode': 'tools',
        'project_id': '12345',
        'organization_id': '0190-abcd',
        'include_instructions': False,
    }
    assert b'phx_new' not in shell.path.read_bytes()
    raw = load_codex_credentials(account='posthog')
    assert raw is not None and 'phx_new' not in raw
    capability = shell.capability()
    assert not capability.include_instructions and not capability.read_only and capability.auth is None
    connection = transport(capability)
    assert connection.url == f'{EU_URL}?features=flags%2Csql'
    assert connection.headers == {
        'x-posthog-mcp-mode': 'tools',
        'x-posthog-project-id': '12345',
        'x-posthog-organization-id': '0190-abcd',
    }
    assert isinstance(connection.auth, SavedKeyAuth)
    assert bearer() == 'Bearer phx_new'


async def test_reopening_repicks_a_saved_key_and_region_without_reinstalling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api_keys.save_key(name='SHARED_POSTHOG', value='shared-secret')
    shell = Shell(tmp_path, {'url': EU_URL, 'read_only': False})
    script(monkeypatch, lists=[])
    await shell.loader.command(['enable', 'posthog'])
    key_choice(monkeypatch, KeyReference(name='SHARED_POSTHOG'))
    script(monkeypatch, lists=[pick('key'), pick('url')], choices=[pick(US_URL)])
    assert await shell.loader.command(['configure', 'posthog']) == (
        'PostHog connects with SHARED_POSTHOG from /keys.\nSaved Region.'
    )
    assert shell.saved() == {
        'auth': 'key',
        'url': US_URL,
        'read_only': False,
        'features': None,
        'mode': 'auto',
        'project_id': None,
        'organization_id': None,
        'include_instructions': True,
    }
    assert transport(shell.capability()).url == US_URL
    assert bearer() == 'Bearer shared-secret'
    api_keys.save_key(name='SHARED_POSTHOG', value='replaced')
    assert bearer() == 'Bearer replaced', 'replacing the key in /keys reaches the next request without a reload'
    assert await async_bearer() == 'Bearer replaced', 'async clients resolve it too, off the event loop'
    with pytest.raises(ValueError, match='used by posthog'):
        api_keys.rename_key(name='SHARED_POSTHOG', new_name='OTHER')


async def test_custom_region_url_is_validated_as_typed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shell = Shell(tmp_path)
    script(monkeypatch, lists=[])
    await shell.loader.enable('posthog')
    script(monkeypatch, lists=[pick('url')], choices=[pick(CUSTOM)], texts=[typed('http://localhost:8000/mcp')])
    assert await shell.loader.configure('posthog') == 'Saved Region.'
    assert transport(shell.capability()).url == 'http://localhost:8000/mcp'


async def test_feature_groups_toggle_and_every_group_clears_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shell = Shell(tmp_path, {'features': ['flags', 'custom_group']})
    script(monkeypatch, lists=[])
    await shell.loader.enable('posthog')
    script(monkeypatch, lists=[pick('features'), pick('features')], choices=[pick('flags'), pick(EVERY_GROUP), CLOSE])
    assert await shell.loader.configure('posthog') == 'Saved Feature groups.', 'the second open changed nothing'
    assert shell.saved()['features'] is None
    assert transport(shell.capability()).url == US_URL
    script(monkeypatch, lists=[pick('features')], choices=[pick('custom_group'), CLOSE])
    assert await shell.loader.configure('posthog') == 'Saved Feature groups.'
    assert shell.saved()['features'] == ['custom_group']
    script(
        monkeypatch, lists=[pick('features')], choices=[pick('custom_group'), MenuResult(item=save_and_close_item())]
    )
    await shell.loader.configure('posthog')
    assert shell.saved()['features'] is None, 'unchecking the last group offers every group, never an empty list'
    script(monkeypatch, lists=[pick('features')], choices=[CLOSE])
    assert await shell.loader.configure('posthog') == 'PostHog settings unchanged.'


def test_feature_menu_marks_the_selection_and_keeps_unknown_groups() -> None:
    labels = [item.label for item in posthog.feature_menu(['flags', 'zz_new'], 3)._items]  # pyright: ignore[reportPrivateUsage]
    assert labels[0] == '[ ] every group (no filter)'
    assert '[x] flags' in labels and '[x] zz_new' in labels and '[ ] sql' in labels
    every = posthog.feature_menu(None, 0)._items[0]  # pyright: ignore[reportPrivateUsage]
    assert every.label == '[x] every group (no filter)'
    *_, last = posthog.feature_menu(None, 0)._items  # pyright: ignore[reportPrivateUsage]
    assert is_save_and_close(last)


async def test_new_key_is_typed_masked_when_no_keys_are_saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shell = Shell(tmp_path)
    script(monkeypatch, lists=[])
    await shell.loader.enable('posthog')
    script(monkeypatch, lists=[pick('key')], texts=[typed('phx_typed')])
    assert await shell.loader.configure('posthog') == f'PostHog connects with {KEY} from /keys.'
    assert api_keys.load_keys()[KEY].get_secret_value() == 'phx_typed'
    assert posthog.saved_key() == KeyReference(name=KEY)


@pytest.mark.parametrize('answer', [TextInputResult(cancelled=True), typed('   ')])
async def test_cancelled_or_blank_key_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: TextInputResult
) -> None:
    shell = Shell(tmp_path)
    script(monkeypatch, lists=[])
    await shell.loader.enable('posthog')
    script(monkeypatch, lists=[pick('key')], texts=[answer])
    assert await shell.loader.configure('posthog') == 'PostHog key unchanged.'
    assert api_keys.load_keys() == {} and posthog.saved_key() is None


async def test_a_key_saved_elsewhere_while_typing_is_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shell = Shell(tmp_path)
    script(monkeypatch, lists=[])
    await shell.loader.enable('posthog')

    async def racing_prompt(*, prompt: object, label: str, optional: bool = False) -> str:
        api_keys.save_key(name=KEY, value='saved-by-another-process')
        return 'phx_mine'

    monkeypatch.setattr(posthog, 'prompt_api_key', racing_prompt)
    script(monkeypatch, lists=[pick('key')], choices=[pick(False)])
    assert await shell.loader.configure('posthog') == 'PostHog key unchanged.'
    assert api_keys.load_keys()[KEY].get_secret_value() == 'saved-by-another-process'


async def test_replacing_a_shared_key_needs_confirmation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api_keys.save_key(name=KEY, value='shared')
    shell = Shell(tmp_path)
    script(monkeypatch, lists=[])
    await shell.loader.enable('posthog')
    key_choice(monkeypatch, 'other')
    script(monkeypatch, lists=[pick('key'), pick('key')], choices=[pick(False), pick(True)])
    assert await shell.loader.configure('posthog') == (
        f'PostHog key unchanged.\nPostHog connects with {KEY} from /keys.'
    )
    assert api_keys.load_keys()[KEY].get_secret_value() == 'other'


async def test_a_key_that_vanishes_while_choosing_is_reported_in_the_menu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shell = Shell(tmp_path)
    script(monkeypatch, lists=[])
    await shell.loader.enable('posthog')
    key_choice(monkeypatch, KeyReference(name='GONE'))
    script(monkeypatch, lists=[pick('key')])
    assert await shell.loader.configure('posthog') == (
        'The selected API key no longer exists. Select a saved key again through /plugins configure posthog.'
    )


async def test_saved_key_auth_fails_closed() -> None:
    with pytest.raises(UserError, match=r'PostHog has no key\. Run /plugins configure posthog'):
        bearer()
    with pytest.raises(UserError, match='PostHog has no key'):
        await async_bearer()
    save_codex_credentials(account='posthog', value=f'{{"token": {{"name": "{KEY}"}}}}')
    with pytest.raises(UserError, match=f'{KEY} is missing'):
        bearer()
    save_codex_credentials(account='posthog', value='{"token": "inline-secret"}')
    with pytest.raises(UserError, match='reference is invalid'):
        bearer()


def test_menu_validates_resets_and_flags_a_missing_key() -> None:
    host = PluginHost[None](name='posthog', console=Console(file=io.StringIO()), settings={'read_only': False})
    source = PostHogSource(host)
    rows = {row.key: row for row in source.rows()}
    assert rows['key'].note == 'needs a key' and source.current(rows['key']) == '(none)'
    assert source.problem(rows['url'], 'http://posthog.example.com/mcp') == (
        'Value error, Use an https:// URL (http:// only for localhost) with no query string.'
    )
    for url in (f'{US_URL}?features=sql', f'{US_URL}?', f'{US_URL}#'):
        assert source.problem(rows['url'], url) is not None, 'the feature filter owns the query string'
    for url in ('https://user:phx_secret@mcp.posthog.com/mcp', 'https://phx_secret@mcp.posthog.com/mcp'):
        assert source.problem(rows['url'], url) == 'Value error, Leave credentials out of the URL; keep keys in /keys.'
    assert source.problem(rows['project_id'], '12 34') == (
        'Value error, Use the ID as PostHog shows it: letters, digits, and dashes.'
    )
    assert source.problem(rows['read_only'], 'maybe') == 'Input should be a valid boolean'
    assert source.problem(rows['project_id'], '42') is None
    assert source.current(rows['project_id']) == '(not set)'
    assert source.current(rows['read_only']) == 'false'
    assert source.reset(rows['read_only']) == 'Reset Tools.'
    assert source.current(rows['read_only']) == 'true'
    assert source.current(rows['features']) == 'every group'
    assert source.reset(rows['key']).startswith('The API key has no default')
    api_keys.save_key(name=KEY, value='saved')
    save_codex_credentials(account='posthog', value=f'{{"token": {{"name": "{KEY}"}}}}')
    rows = {row.key: row for row in source.rows()}
    assert rows['key'].note == '' and source.current(rows['key']) == KEY
    api_keys.delete_key(name=KEY)
    assert {row.key: row for row in source.rows()}['key'].note == 'needs a key'
    browser = PostHogSource(PluginHost[None](name='posthog', console=Console(), settings={'auth': 'browser'}))
    assert browser.rows()[0].note == '', 'browser sign-in needs no key'


async def test_missing_key_warns_on_load_but_keeps_the_menu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    save_codex_credentials(account='posthog', value='{"token": {"name": "DELETED"}}')
    shell = Shell(tmp_path, terminal=False)
    script(monkeypatch, lists=[])
    await shell.loader.enable('posthog')
    assert 'PostHog uses DELETED, which is missing from /keys.' in shell.output.getvalue()
    assert len(shell.loader.capabilities()) == 1


async def test_edits_saved_before_the_menu_fails_still_apply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shell = Shell(tmp_path)
    script(monkeypatch, lists=[])
    await shell.loader.enable('posthog')
    scripted = script(monkeypatch, lists=[pick('read_only')], choices=[pick('false')])

    def fail_after_the_edit(menu: object) -> MenuResult:
        if scripted.opened:
            raise RuntimeError('terminal went away')
        return scripted.run_list(menu)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(posthog, 'RUNNERS', replace(scripted.runners, run_list=fail_after_the_edit))
    with pytest.raises(RuntimeError, match='terminal went away'):
        await shell.loader.configure('posthog')
    assert shell.saved()['read_only'] is False
    assert transport(shell.capability()).headers == {}, 'the saved edit is loaded even though the menu failed'


async def test_a_corrupt_key_reference_still_loads_so_the_menu_can_fix_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_codex_credentials(account='posthog', value='{"token": "inline-secret"}')
    shell = Shell(tmp_path)
    script(monkeypatch, lists=[])
    await shell.loader.enable('posthog')
    assert 'The saved PostHog key reference is invalid.' in shell.output.getvalue()
    source = PostHogSource(PluginHost[None](name='posthog', console=Console(), settings={}))
    [key, *_] = source.rows()
    assert key.note == 'needs a key' and source.current(key) == '(invalid)'
    key_choice(monkeypatch, KeyReference(name=KEY))
    api_keys.save_key(name=KEY, value='phx_fixed')
    script(monkeypatch, lists=[pick('key')])
    assert await shell.loader.configure('posthog') == f'PostHog connects with {KEY} from /keys.'
    assert bearer() == 'Bearer phx_fixed'


async def test_headless_configure_explains_instead_of_drawing(tmp_path: Path) -> None:
    shell = Shell(tmp_path, terminal=False)
    assert await shell.loader.command(['enable', 'posthog']) == (
        'Enabled posthog.\nConfigure PostHog from a terminal: Run /plugins configure posthog to choose or enter a key.'
    )


async def test_status_command_in_key_mode(tmp_path: Path) -> None:
    shell = Shell(tmp_path, terminal=False)
    await shell.loader.enable('posthog')
    commands = shell.commands
    assert 'has no key yet' in await commands.execute_async('/posthog')
    api_keys.save_key(name=KEY, value='saved')
    save_codex_credentials(account='posthog', value=f'{{"token": {{"name": "{KEY}"}}}}')
    assert await commands.execute_async('/posthog') == f'PostHog (read-only, {US_URL}) connects with {KEY} from /keys.'
    api_keys.delete_key(name=KEY)
    assert f'PostHog uses {KEY}, which is missing from /keys.' in await commands.execute_async('/posthog')
    with pytest.raises(ValueError, match=r'Usage: /posthog \[login \| logout \| status\]'):
        await commands.execute_async('/posthog key')


async def test_browser_sign_in_status_and_logout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shell = Shell(tmp_path, {'auth': 'browser', 'read_only': False, 'features': ['sql']}, terminal=False)
    await shell.loader.enable('posthog')
    assert await shell.run_capability() is None
    commands = shell.commands
    status = f'PostHog (read-write, {US_URL}) uses browser sign-in.'
    assert await commands.execute_async('/posthog') == f'{status} {NOT_SIGNED_IN}'
    await store_sign_in(f'{US_URL}?features=sql')
    assert await commands.execute_async('/posthog status') == f'{status} Signed in to PostHog.'
    capability = await shell.run_capability()
    assert capability is not None
    connection = transport(capability)
    assert connection.url == f'{US_URL}?features=sql' and connection.headers == {}
    assert isinstance(connection.auth, MCPSignIn)
    assert await commands.execute_async('/posthog logout') == (
        'Signed out of PostHog. Run /posthog login to sign in again.'
    )
    assert await shell.run_capability() is None, 'the next run has no tools rather than reusing the old sign-in'
    [registered] = [command for command in commands if command.name == 'posthog']
    assert list(registered.complete([''])) == ['login', 'logout', 'status']
    assert list(registered.complete(['logout', ''])) == []

    def locked(service: str, account: str) -> str | None:
        raise KeyringLocked('locked')

    monkeypatch.setattr(keyring, 'get_password', locked)
    assert 'the keyring could not be read' in await commands.execute_async('/posthog')
    assert SignIn.attempts == 0


async def test_no_browser_sign_in_loads_without_tools_and_says_how(tmp_path: Path) -> None:
    """Loading waits for every plugin, so a browser left open at load would leave CLAI unable to run anything."""
    SignIn.unfinished = True
    for terminal in (True, False):
        folder = tmp_path / f'terminal-{terminal}'
        folder.mkdir()
        shell = Shell(folder, {'auth': 'browser'}, terminal=terminal)
        with anyio.fail_after(5):
            await shell.loader.enable('posthog')
            assert await shell.run_capability() is None
        assert NOT_SIGNED_IN in shell.output.getvalue()
        assert 'has no key' not in shell.output.getvalue()
        assert SignIn.attempts == 0
        await shell.loader.close('exit')


async def test_posthog_login_signs_in_and_the_next_run_connects_without_a_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        'pydantic_clai2.plugins.sign_in.RUNNERS', Script(lists=[], choices=[UNTIL_CLOSED], texts=[]).runners
    )
    shell = Shell(tmp_path, {'auth': 'browser', 'url': EU_URL}, terminal=False)
    await shell.loader.enable('posthog')
    assert await shell.run_capability() is None
    assert await shell.commands.execute_async('/posthog login') == 'Signed in to PostHog.'
    capability = await shell.run_capability()
    assert capability is not None, 'no reload needed'
    assert transport(capability).url == EU_URL
    assert transport(capability).headers == {'x-posthog-read-only': 'true'}
    assert SignIn.attempts == 1
    await shell.loader.close('exit')


async def test_choosing_the_browser_in_the_menu_signs_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shell = Shell(tmp_path)
    script(monkeypatch, lists=[])
    await shell.loader.enable('posthog')
    script(monkeypatch, lists=[pick('auth')], choices=[pick('browser'), UNTIL_CLOSED])
    assert await shell.loader.configure('posthog') == 'Saved Sign-in.\nSigned in to PostHog.'
    assert shell.saved()['auth'] == 'browser'
    assert await shell.run_capability() is not None
    assert SignIn.attempts == 1
    await shell.loader.close('exit')


async def test_choosing_the_browser_when_already_signed_in_or_cancelling_does_not_open_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await store_sign_in()
    shell = Shell(tmp_path)
    script(monkeypatch, lists=[])
    await shell.loader.enable('posthog')
    shown = script(monkeypatch, lists=[pick('auth'), pick('auth')], choices=[CLOSE, pick('browser')])
    assert await shell.loader.configure('posthog') == 'Saved Sign-in.'
    assert shown.opened == ['list', 'choice', 'list', 'choice', 'list']
    assert SignIn.attempts == 0
    script(monkeypatch, lists=[pick('auth')], choices=[pick('key')])
    assert await shell.loader.configure('posthog') == 'Saved Sign-in.'
    assert shell.saved()['auth'] == 'key'
    assert SignIn.attempts == 0
    await shell.loader.close('exit')


async def test_esc_cancels_an_unfinished_sign_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    SignIn.unfinished = True
    shell = Shell(tmp_path)
    script(monkeypatch, lists=[])
    await shell.loader.enable('posthog')
    script(monkeypatch, lists=[pick('auth')], choices=[pick('browser'), CLOSE])
    with anyio.fail_after(5):
        message = await shell.loader.configure('posthog')
    assert message == 'Saved Sign-in.\nPostHog sign-in cancelled. Run /posthog login to try again.'
    assert SignIn.attempts == 1
    assert await shell.run_capability() is None
    await shell.loader.close('exit')


async def test_failed_sign_in_is_reported_and_adds_no_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    SignIn.error = RuntimeError('Client failed to connect: authorization denied')
    shell = Shell(tmp_path)
    script(monkeypatch, lists=[])
    await shell.loader.enable('posthog')
    script(monkeypatch, lists=[pick('auth')], choices=[pick('browser'), UNTIL_CLOSED])
    assert await shell.loader.configure('posthog') == (
        'Saved Sign-in.\nCould not sign in to PostHog: Client failed to connect: authorization denied. '
        'Run /posthog login to try again.'
    )
    assert await shell.run_capability() is None
    await shell.loader.close('exit')


@pytest.mark.parametrize(
    'settings',
    [
        {'api_key': 'phx_inline_secret'},
        {'token': {'name': KEY}},
        {'auth': 'oauth'},
        {'features': []},
        {'features': ['Flags']},
        {'url': 'http://mcp.posthog.com/mcp'},
        {'mode': 'all'},
    ],
)
async def test_settings_cannot_hold_a_secret_or_invalid_options(tmp_path: Path, settings: dict[str, JsonValue]) -> None:
    shell = Shell(tmp_path, settings)
    with pytest.raises(PluginError):
        await shell.loader.enable('posthog')
    assert shell.loader.capabilities() == []


async def test_add_with_rejected_settings_keeps_no_trace_of_them(tmp_path: Path) -> None:
    shell = Shell(tmp_path, terminal=False)
    with pytest.raises(PluginSettingsError) as raised:
        await shell.loader.command(
            ['add', 'analytics', 'pydantic_clai2.builtin_plugins.posthog', '{"api_key": "phx_pasted"}']
        )
    assert 'phx_pasted' not in str(raised.value), 'errors never echo a rejected value'
    assert [plugin.id for plugin in shell.store.plugins()] == []
    assert b'phx_pasted' not in shell.path.read_bytes(), 'rejected settings are never written, so no page holds them'
    await shell.loader.command(['enable', 'posthog'])
    before = shell.saved()
    with pytest.raises(PluginSettingsError):
        await shell.loader.command(
            ['add', 'posthog', 'pydantic_clai2.builtin_plugins.posthog', '{"token": "phx_pasted"}']
        )
    assert shell.saved() == before, 'replacing with rejected settings restores the previous declaration'
    assert b'phx_pasted' not in shell.path.read_bytes()
    assert len(shell.loader.capabilities()) == 1, 'and the previous plugin is loaded again'


async def test_add_replacing_the_builtin_opens_the_menu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shell = Shell(tmp_path)
    script(monkeypatch, lists=[pick('read_only')], choices=[pick('true')])
    assert await shell.loader.command(
        ['add', 'posthog', 'pydantic_clai2.builtin_plugins.posthog', '{"read_only": false}']
    ) == ('Replaced built-in posthog.\nSaved Tools.')
    assert transport(shell.capability()).headers == {'x-posthog-read-only': 'true'}


async def test_configure_needs_a_loaded_plugin_with_a_menu(tmp_path: Path) -> None:
    shell = Shell(tmp_path)
    with pytest.raises(ValueError, match='not loaded; enable it before configuring'):
        await shell.loader.command(['configure', 'posthog'])
    shell.store.save_plugin(BUILTIN.model_copy(update={'id': 'plain', 'factory': 'pydantic_clai2.repo_context'}))
    assert await shell.loader.command(['enable', 'plain']) == 'Enabled plain.'
    with pytest.raises(ValueError, match='no settings menu'):
        await shell.loader.configure('plain')


async def test_turning_it_on_in_the_plugins_menu_opens_its_settings_menu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shell = Shell(tmp_path)
    script(monkeypatch, lists=[pick('read_only')], choices=[pick('false')])
    runs: list[str | None] = []

    def run(menu: PluginMenu[None]) -> MenuResult:
        runs.append(menu.notice)
        if len(runs) > 1:
            return CLOSE
        opened = menu.toggle(Redraw(), menu.items()[0])
        assert opened is not None
        return opened

    assert await open_plugins_menu(shell.loader, run=run) == 'Saved Tools.'
    assert runs == [None, 'Saved Tools.']
    assert transport(shell.capability()).headers == {}


class Redraw:
    def replace_items(self, items: object) -> None:
        pass
