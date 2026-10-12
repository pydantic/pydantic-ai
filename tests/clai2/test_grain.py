"""The built-in `grain` plugin: harness `Grain` with a token from the environment, `/keys`, or a keyring sign-in."""

import io
from collections.abc import Callable
from pathlib import Path

import anyio
import pytest
from fastmcp import Client
from fastmcp.client.auth import OAuth
from fastmcp.client.auth.oauth import TokenStorageAdapter
from fastmcp.client.transports import SSETransport, StreamableHttpTransport
from mcp.shared.auth import OAuthToken
from pydantic import JsonValue, ValidationError
from rich.console import Console
from termflow.tui import MenuItem
from termflow.tui.menu import MenuResult

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.grain import Grain
from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2._app import create_shell
from pydantic_clai2.builtin_plugins import grain as grain_module
from pydantic_clai2.builtin_plugins.grain import (
    GRAIN_MCP_URL,
    KEY_ACCOUNT,
    KEY_NAME,
    TOKEN_ACCOUNT,
    GrainConnection,
    GrainForm,
    GrainPlugin,
)
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import PluginSettings, api_keys
from pydantic_clai2.config.api_keys import KeyReference
from pydantic_clai2.config.credential_store import load_codex_credentials, save_codex_credentials
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.mcp import OAuthSignIn, SignIn as MCPSignIn, TokenStore, http_client
from pydantic_clai2.plugins import LoadedPlugin, PluginHost, SessionStart, load_plugin
from pydantic_clai2.plugins.loader import PluginLoader
from pydantic_clai2.ui.menus.field_menu import FieldMenu
from tests.clai2.menu_script import UNTIL_CLOSED, Script, pick

RETIRED = 'pydantic_ai_harness.grain:Grain'
"""The raw factory the retired `/plugins` catalog saved under `grain`."""

pytestmark = pytest.mark.anyio

BROWSER_SIGN_IN = OAuthSignIn.sign_in
"""The real sign-in, which the autouse stub replaces."""


class SignIn:
    """Stands in for `OAuthSignIn.sign_in`, which would open the browser; it stores tokens as FastMCP would."""

    attempts: int = 0
    error: Exception | None = None
    unfinished: bool = False
    """The browser sign-in is never completed, so it waits until cancelled."""

    @staticmethod
    async def sign_in(method: OAuthSignIn, *, show: Callable[[str], object]) -> None:
        assert method is grain_module.SIGN_IN
        SignIn.attempts += 1
        if SignIn.error is not None:
            raise SignIn.error
        if SignIn.unfinished:
            await anyio.sleep_forever()
        await store_sign_in()


@pytest.fixture(autouse=True)
def no_grain_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """No token in the environment, so only saved credentials count; no browser, and a waiting screen."""
    monkeypatch.delenv('GRAIN_ACCESS_TOKEN', raising=False)
    monkeypatch.setattr(OAuthSignIn, 'sign_in', SignIn.sign_in)
    monkeypatch.setattr(SignIn, 'attempts', 0)
    monkeypatch.setattr(SignIn, 'error', None)
    monkeypatch.setattr(SignIn, 'unfinished', False)
    waiting = Script(lists=[], choices=[UNTIL_CLOSED] * 5, texts=[])
    monkeypatch.setattr('pydantic_clai2.plugins.sign_in.RUNNERS', waiting.runners)


async def store_sign_in() -> None:
    tokens = TokenStorageAdapter(TokenStore(TOKEN_ACCOUNT), server_url=GRAIN_MCP_URL)
    await tokens.set_tokens(OAuthToken(access_token='access', token_type='Bearer', expires_in=3600))


def host(settings: dict[str, JsonValue] | None = None, *, output: io.StringIO | None = None) -> PluginHost[None]:
    return PluginHost(
        name='grain', console=Console(file=output or io.StringIO(), width=200), settings={**(settings or {})}
    )


def load_grain(plugin_host: PluginHost[None]) -> LoadedPlugin[None]:
    return load_plugin(GrainPlugin, plugin_host)


async def for_run(plugin: LoadedPlugin[None]) -> Grain[None] | None:
    """What the plugin gives the next run, from its per-run factory."""
    grain_plugin = plugin.plugin
    assert isinstance(grain_plugin, GrainPlugin)
    assert list(plugin.capabilities) == [grain_plugin.connection.capability]
    return await grain_plugin.connection.capability(run_context())


async def loader_run(loader: PluginLoader[None]) -> Grain[None] | None:
    """What the loaded plugin gives the next run."""
    [entry] = loader.entries()
    assert entry.loaded is not None
    return await for_run(entry.loaded)


async def grain(plugin: LoadedPlugin[None]) -> Grain[None]:
    """The `Grain` the plugin builds for the next run."""
    capability = await for_run(plugin)
    assert capability is not None
    return capability


def sign_in(capability: Grain[None]) -> MCPSignIn:
    """The run connection's keyring-backed sign-in."""
    assert isinstance(capability.client, Client)
    transport = capability.client.transport
    assert isinstance(transport, StreamableHttpTransport) and transport.url == GRAIN_MCP_URL
    assert transport.httpx_client_factory is http_client
    assert isinstance(transport.auth, MCPSignIn)
    return transport.auth


def test_declared_as_a_disabled_builtin() -> None:
    [declaration] = [plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'grain']
    assert declaration.factory == 'pydantic_clai2.builtin_plugins.grain'
    assert not declaration.enabled


@pytest.mark.parametrize('enabled', [True, False])
def test_a_saved_catalog_declaration_moves_to_the_builtin(tmp_path: Path, enabled: bool) -> None:
    store = SettingsStore(tmp_path / 'settings.db')
    store.save_plugin(PluginSettings(id='grain', factory=RETIRED, enabled=enabled))
    store.save_plugin(PluginSettings(id='other', factory=RETIRED))
    loaded = declarations(store)
    assert loaded['grain'] == PluginSettings(
        id='grain', factory='pydantic_clai2.builtin_plugins.grain', enabled=enabled
    )
    assert loaded['other'].factory == RETIRED, 'only the id the built-in replaced moves'


def test_a_saved_declaration_with_settings_stays(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'settings.db')
    chosen = PluginSettings(id='grain', factory=RETIRED, settings={'read_only': True})
    store.save_plugin(chosen)
    assert declarations(store)['grain'] == chosen


def declarations(store: SettingsStore) -> dict[str, PluginSettings]:
    shell = create_shell(
        Agent(TestModel()),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=io.StringIO()),
        settings=None,
        store=store,
        builtin_plugins=DEFAULT_PLUGINS,
        project=ProjectSettings(),
        headless=True,
    )
    return {entry.name: entry.declaration for entry in shell.loader.entries()}


async def test_enabling_the_builtin_adds_grain_and_the_menu_saves_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SettingsStore(tmp_path / 'settings.db')
    commands = Commands()
    loader: PluginLoader[None] = PluginLoader(
        store=store,
        console=Console(file=io.StringIO()),
        commands=commands,
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=[plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'grain'],
    )
    await loader.load_all()
    assert loader.capabilities() == []

    await loader.enable('grain')
    assert await loader_run(loader) is None, 'no tools until signed in'
    assert 'Not signed in' in await commands.execute_async('/grain status')
    await store_sign_in()
    capability = await loader_run(loader)
    assert capability is not None and capability.read_only

    script = Script(lists=[pick('read_only'), MenuResult(cancelled=True)], choices=[pick('false')], texts=[])
    monkeypatch.setattr(grain_module, 'TERMINAL', script.runners)
    assert 'all tools' in await commands.execute_async('/grain')
    [saved] = store.plugins()
    assert saved.settings == {'read_only': False, 'include_instructions': True}, 'saved at once, for the next load'
    rebuilt = await loader_run(loader)
    assert rebuilt is not None and not rebuilt.read_only, 'and applied to the next prompt without a reload'

    await loader.disable('grain')
    await loader.enable('grain')
    [saved] = store.plugins()
    assert saved.settings == {'read_only': False, 'include_instructions': True}, 'toggling keeps what the menu saved'


async def test_turning_it_on_or_configuring_it_opens_the_menu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = SettingsStore(tmp_path / 'settings.db')
    loader: PluginLoader[None] = PluginLoader(
        store=store,
        console=Console(file=io.StringIO()),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=[plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'grain'],
    )
    await loader.load_all()
    await store_sign_in()
    script = Script(
        lists=[MenuResult(cancelled=True), pick('read_only'), MenuResult(cancelled=True)],
        choices=[pick('false')],
        texts=[],
    )
    monkeypatch.setattr(grain_module, 'TERMINAL', script.runners)

    assert await loader.command(['enable', 'grain']) == 'Enabled grain.\nGrain settings unchanged.'
    [saved] = store.plugins()
    assert saved.settings == {'read_only': True, 'include_instructions': True}, 'closing marks it configured'

    assert 'all tools' in await loader.command(['configure', 'grain'])
    [saved] = store.plugins()
    assert saved.settings == {'read_only': False, 'include_instructions': True}
    rebuilt = await loader_run(loader)
    assert rebuilt is not None and not rebuilt.read_only, 'the reloaded plugin builds from the saved settings'


async def test_token_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('GRAIN_ACCESS_TOKEN', 'grain-token')
    output = io.StringIO()
    plugin = load_grain(host(output=output))
    await plugin.dispatch(SessionStart(agent=Agent(TestModel()), settings=SettingsStore().load()))
    assert output.getvalue() == '', 'a token needs no sign-in, so there is nothing to say'
    plugin = load_grain(host({'read_only': False}))
    capability = await grain(plugin)
    assert capability.client is None and capability.auth is None and not capability.read_only
    assert await grain(plugin) is capability, 'rebuilt only when the token source or a setting changes'
    assert (
        await plugin.commands.execute_async('/grain status')
        == 'Grain uses the GRAIN_ACCESS_TOKEN environment variable.'
    )
    assert 'cannot revoke' in await plugin.commands.execute_async('/grain logout')


async def test_without_a_token_nothing_signs_in_until_asked() -> None:
    """Loading and runs never open the browser: a sign-in nobody finishes would hold up every prompt."""
    SignIn.unfinished = True
    output = io.StringIO()
    with anyio.fail_after(5):
        plugin = load_grain(host(output=output))
        await plugin.dispatch(SessionStart(agent=Agent(TestModel()), settings=SettingsStore().load()))
        assert await for_run(plugin) is None
    assert 'Not signed in to Grain. Run /grain login to sign in.' in output.getvalue()
    assert SignIn.attempts == 0


async def test_grain_login_signs_in_and_the_next_run_connects_without_a_reload() -> None:
    plugin = load_grain(host())
    commands = plugin.commands
    assert await for_run(plugin) is None
    assert await commands.execute_async('/grain status') == 'Not signed in to Grain. Run /grain login to sign in.'
    assert await commands.execute_async('/grain login') == 'Signed in to Grain.'
    assert await commands.execute_async('/grain status') == 'Signed in to Grain.'
    capability = await grain(plugin)
    assert capability.auth is None and capability.read_only
    assert sign_in(capability).tokens.name == TOKEN_ACCOUNT
    assert sign_in(await grain(plugin)) is not sign_in(capability), 'each run connects afresh'

    storage = TokenStorageAdapter(TokenStore(TOKEN_ACCOUNT), server_url=GRAIN_MCP_URL)
    assert await commands.execute_async('/grain logout') == 'Signed out of Grain. Run /grain login to sign in again.'
    assert await storage.get_tokens() is None
    assert await for_run(plugin) is None
    with pytest.raises(ValueError, match=r'Usage: /grain \[status \| key \| login \| logout\]'):
        await commands.execute_async('/grain other')


@pytest.mark.parametrize(
    ('error', 'unfinished', 'message'),
    [
        (None, True, 'Grain sign-in cancelled. Run /grain login to try again.'),
        (
            RuntimeError('authorization denied'),
            False,
            'Could not sign in to Grain: authorization denied. Run /grain login to try again.',
        ),
    ],
)
async def test_a_sign_in_that_does_not_finish_leaves_grain_signed_out(
    monkeypatch: pytest.MonkeyPatch, error: Exception | None, unfinished: bool, message: str
) -> None:
    SignIn.error, SignIn.unfinished = error, unfinished
    escape = Script(lists=[], choices=[MenuResult(cancelled=True)], texts=[])
    monkeypatch.setattr('pydantic_clai2.plugins.sign_in.RUNNERS', escape.runners)
    plugin = load_grain(host())
    with anyio.fail_after(5):
        assert await plugin.commands.execute_async('/grain login') == message
    assert SignIn.attempts == 1
    assert await for_run(plugin) is None


async def test_the_browser_sign_in_redirects_to_localhost_and_shows_the_link(monkeypatch: pytest.MonkeyPatch) -> None:
    url = 'https://api.grain.com/oauth/authorize?state=abc'
    opened: list[str] = []
    shown: list[str] = []

    async def open_browser(self: OAuth, authorization_url: str) -> None:
        opened.append(authorization_url)

    class Connect:
        """Stands in for `fastmcp.Client`: the server asks for authorization as soon as it connects."""

        def __init__(self, transport: StreamableHttpTransport | SSETransport, *, init_timeout: float) -> None:
            self.auth = transport.auth

        async def __aenter__(self) -> None:
            assert isinstance(self.auth, MCPSignIn)
            [redirect] = self.auth.context.client_metadata.redirect_uris or []
            assert redirect.host == 'localhost', 'Grain rejects a 127.0.0.1 redirect'
            await self.auth.redirect_handler(url)

        async def __aexit__(self, *exc_info: object) -> None:
            pass

    monkeypatch.setattr(OAuth, 'redirect_handler', open_browser)
    monkeypatch.setattr('pydantic_clai2.mcp._tokens.Client', Connect)
    await BROWSER_SIGN_IN(grain_module.SIGN_IN, show=shown.append)
    assert opened == [url]
    assert shown == [f'If no browser opened, open this link:\n{url}']
    run_auth = grain_module.SIGN_IN.transport().auth
    assert isinstance(run_auth, MCPSignIn)
    [redirect] = run_auth.context.client_metadata.redirect_uris or []
    assert redirect.host == 'localhost', 'runs refresh with the client registered for localhost'


@pytest.mark.parametrize('settings', [{'readonly': True}, {'auth': 'secret'}, {'token': 'secret'}])
def test_settings_hold_no_secret(settings: dict[str, JsonValue]) -> None:
    with pytest.raises(ValidationError):
        load_grain(host(settings))


def test_completes_subcommands() -> None:
    plugin = load_grain(host())
    [command] = plugin.commands
    assert list(command.complete([])) == ['key', 'login', 'logout', 'status']
    assert list(command.complete(['logout', ''])) == []


class Prompt:
    """The masked prompt `prompt_api_key` reads a typed token from."""

    def __init__(self, *values: str) -> None:
        self.values = iter(values)
        self.labels: list[tuple[str, bool]] = []

    async def prompt_async(self, label: str, *, is_password: bool = False) -> str:
        self.labels.append((label, is_password))
        return next(self.values)


def answer_key_prompt(
    monkeypatch: pytest.MonkeyPatch, *, typed: tuple[str, ...] = (), keys: tuple[str, ...] = ()
) -> Prompt:
    """Answer `/grain key`: `keys` drive the saved-key picker, `typed` the masked prompt."""
    prompt = Prompt(*typed)
    monkeypatch.setattr('pydantic_clai2.builtin_plugins.grain.PromptSession', lambda: prompt)
    pressed = iter(keys)
    monkeypatch.setattr(api_keys, 'menu_key', lambda: next(pressed))
    return prompt


def run_context() -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage())


async def test_a_typed_token_goes_to_keys_and_only_its_name_is_saved(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt = answer_key_prompt(monkeypatch, typed=('typed-secret',))
    plugin = load_grain(host())
    await store_sign_in()
    assert sign_in(await grain(plugin)).tokens.name == TOKEN_ACCOUNT
    result = await plugin.commands.execute_async('/grain key')
    assert result == 'Grain uses the /keys entry GRAIN_ACCESS_TOKEN from the next prompt.'
    assert callable((await grain(plugin)).auth), 'the running session switches without a reload'
    assert prompt.labels == [('Grain access token (saved in /keys as GRAIN_ACCESS_TOKEN; Enter for none): ', True)]
    assert api_keys.load_keys()[KEY_NAME].get_secret_value() == 'typed-secret'
    saved = load_codex_credentials(account=KEY_ACCOUNT)
    assert saved is not None and 'typed-secret' not in saved
    assert api_keys.key_users(name=KEY_NAME) == ['grain'], 'renaming a key Grain uses is refused'

    reloaded = load_grain(host())
    capability = await grain(reloaded)
    assert capability.client is None and callable(capability.auth)
    assert capability.auth(run_context()) == 'typed-secret'
    assert await reloaded.commands.execute_async('/grain status') == 'Grain uses the /keys entry GRAIN_ACCESS_TOKEN.'
    assert 'No API key' in await reloaded.commands.execute_async('/grain logout')

    api_keys.save_key(name=KEY_NAME, value='replaced')
    assert capability.auth(run_context()) == 'replaced', 'the key is resolved on every run'
    api_keys.delete_key(name=KEY_NAME)
    with pytest.raises(UserError, match='Saved API key GRAIN_ACCESS_TOKEN is missing'):
        capability.auth(run_context())


async def test_an_existing_key_is_shared_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    api_keys.save_key(name='SHARED_GRAIN', value='shared-secret')
    answer_key_prompt(monkeypatch, keys=('enter',))
    plugin = load_grain(host())
    assert 'SHARED_GRAIN' in await plugin.commands.execute_async('/grain key')
    auth = (await grain(plugin)).auth
    assert callable(auth) and auth(run_context()) == 'shared-secret'


@pytest.mark.parametrize(
    ('keys', 'expected'),
    [(('down', 'down', 'enter'), 'Grain uses no /keys entry'), (('escape',), 'Grain key unchanged.')],
)
async def test_no_key_or_cancel(monkeypatch: pytest.MonkeyPatch, keys: tuple[str, ...], expected: str) -> None:
    api_keys.save_key(name='SHARED_GRAIN', value='shared-secret')
    answer_key_prompt(monkeypatch, keys=('enter',))
    plugin = load_grain(host())
    await plugin.commands.execute_async('/grain key')
    answer_key_prompt(monkeypatch, keys=keys)
    assert (await plugin.commands.execute_async('/grain key')).startswith(expected)
    reloaded = load_grain(host())
    for loaded in (plugin, reloaded):
        uses_key = (await grain(loaded)).client is None
        assert uses_key == (expected == 'Grain key unchanged.')
    assert SignIn.attempts == (expected != 'Grain key unchanged.'), 'no key signs in through the browser'


def test_an_invalid_saved_choice_fails_closed() -> None:
    save_codex_credentials(account=KEY_ACCOUNT, value='not json')
    with pytest.raises(UserError, match='/grain key'):
        load_grain(host())


async def test_the_menu_picks_a_key_and_changes_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    api_keys.save_key(name='SHARED_GRAIN', value='shared-secret')
    answer_key_prompt(monkeypatch, keys=('enter',))
    output = io.StringIO()
    plugin = load_grain(host(output=output))
    await plugin.dispatch(SessionStart(agent=Agent(TestModel()), settings=SettingsStore().load()))
    assert '/grain picks a /keys token' in output.getvalue(), 'an unconfigured plugin says where to configure it'
    script = Script(
        lists=[
            pick('token'),
            pick('include_instructions'),
            reset('include_instructions'),
            pick('read_only'),
            MenuResult(cancelled=True),
        ],
        choices=[pick('false'), MenuResult(cancelled=True)],
        texts=[],
    )
    monkeypatch.setattr(grain_module, 'TERMINAL', script.runners)
    result = await plugin.commands.execute_async('/grain')
    assert result.splitlines() == [
        'Grain uses the /keys entry SHARED_GRAIN from the next prompt.',
        'Grain server instructions: left out. Saved; applies to the next prompt.',
        'Grain server instructions: included. Saved; applies to the next prompt.',
    ]
    capability = await grain(plugin)
    assert callable(capability.auth) and capability.auth(run_context()) == 'shared-secret'
    assert capability.include_instructions and capability.read_only


async def test_choosing_no_key_in_the_menu_signs_in_unless_signed_in(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = load_grain(host({'read_only': True}))
    chosen = 'Grain uses no /keys entry, so it uses the browser sign-in.'
    for expected in ([chosen, 'Signed in to Grain.'], [chosen]):
        answer_key_prompt(monkeypatch, typed=('',))
        script = Script(lists=[pick('token'), MenuResult(cancelled=True)], choices=[UNTIL_CLOSED], texts=[])
        monkeypatch.setattr(grain_module, 'TERMINAL', script.runners)
        with anyio.fail_after(5):
            assert (await plugin.commands.execute_async('/grain')).splitlines() == expected
    assert SignIn.attempts == 1
    assert sign_in(await grain(plugin)).tokens.name == TOKEN_ACCOUNT


async def test_the_menu_shows_each_token_source(monkeypatch: pytest.MonkeyPatch) -> None:
    form = GrainForm(GrainConnection(host()))
    token, read_only, _ = form.rows()
    assert form.current(token) == 'browser sign-in' and form.current(read_only) == 'true'
    assert 'No API key' in form.reset(token)
    assert form.problem(read_only, 'anything') is None
    form.connection.key = KeyReference(name='SHARED_GRAIN')
    assert form.current(token) == '/keys: SHARED_GRAIN'
    monkeypatch.setenv('GRAIN_ACCESS_TOKEN', 'grain-token')
    assert form.current(token) == 'GRAIN_ACCESS_TOKEN (environment)'


async def test_closing_the_menu_saves_the_defaults_once(monkeypatch: pytest.MonkeyPatch) -> None:
    saved: list[dict[str, JsonValue]] = []
    plugin = load_grain(
        PluginHost[None](name='grain', console=Console(file=io.StringIO()), settings={}, save_settings=saved.append)
    )
    closed = Script(lists=[MenuResult(cancelled=True)] * 2, choices=[], texts=[])
    monkeypatch.setattr(grain_module, 'TERMINAL', closed.runners)
    assert await plugin.commands.execute_async('/grain') == 'Grain settings unchanged.'
    assert await plugin.commands.execute_async('/grain') == 'Grain settings unchanged.'
    assert saved == [{'read_only': True, 'include_instructions': True}]


def reset(key: str) -> MenuResult:
    """What the menu hands back when `r` is pressed on the row `key`."""
    return FieldMenu(GrainForm(GrainConnection(host())), searchable=False).reset_marker(
        object(), MenuItem(key, value=key)
    )
