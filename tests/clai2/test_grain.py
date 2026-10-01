"""The built-in `grain` plugin: harness `Grain` with a token from the environment, `/keys`, or a keyring sign-in."""

import io
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.client.auth import OAuth
from fastmcp.client.auth.oauth import TokenStorageAdapter
from fastmcp.client.transports import StreamableHttpTransport
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
    GrainSignIn,
)
from pydantic_clai2.cli.headless import no_screen
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import PluginSettings, api_keys
from pydantic_clai2.config.api_keys import KeyReference
from pydantic_clai2.config.credential_store import load_codex_credentials, save_codex_credentials
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import FullScreen, LoadedPlugin, PluginHost, SessionStart, bare_screen, load_plugin
from pydantic_clai2.plugins.loader import PluginLoader
from pydantic_clai2.ui.menus.field_menu import FieldMenu
from tests.clai2.menu_script import Script, pick

RETIRED = 'pydantic_ai_harness.grain:Grain'
"""The raw factory the retired `/plugins` catalog saved under `grain`."""

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def no_grain_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """No token in the environment, so only saved credentials count."""
    monkeypatch.delenv('GRAIN_ACCESS_TOKEN', raising=False)


def host(
    settings: dict[str, JsonValue] | None = None,
    *,
    full_screen: FullScreen = bare_screen,
    output: io.StringIO | None = None,
) -> PluginHost[None]:
    return PluginHost(
        name='grain',
        console=Console(file=output or io.StringIO(), width=200),
        settings={**(settings or {})},
        full_screen=full_screen,
    )


def load_grain(plugin_host: PluginHost[None]) -> LoadedPlugin[None]:
    return load_plugin(GrainPlugin, plugin_host)


def grain(plugin: LoadedPlugin[None]) -> Grain[None]:
    """The `Grain` the plugin builds for the next run."""
    [build] = plugin.capabilities
    assert callable(build)
    capability = build(run_context())
    assert isinstance(capability, Grain)
    return capability


def sign_in(capability: Grain[None]) -> GrainSignIn:
    assert isinstance(capability.client, Client)
    transport = capability.client.transport
    assert isinstance(transport, StreamableHttpTransport) and transport.url == GRAIN_MCP_URL
    assert isinstance(transport.auth, GrainSignIn)
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
    [build] = loader.capabilities()
    assert callable(build)
    capability = build(run_context())
    assert isinstance(capability, Grain) and capability.read_only
    assert 'Not signed in' in await commands.execute_async('/grain status')

    script = Script(lists=[pick('read_only'), MenuResult(cancelled=True)], choices=[pick('false')], texts=[])
    monkeypatch.setattr(grain_module, 'TERMINAL', script.runners)
    assert 'all tools' in await commands.execute_async('/grain')
    [saved] = store.plugins()
    assert saved.settings == {'read_only': False, 'include_instructions': True}, 'saved at once, for the next load'
    rebuilt = build(run_context())
    assert isinstance(rebuilt, Grain) and not rebuilt.read_only, 'and applied to the next prompt without a reload'

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
    [build] = loader.capabilities()
    assert callable(build)
    rebuilt = build(run_context())
    assert isinstance(rebuilt, Grain) and not rebuilt.read_only, 'the reloaded plugin builds from the saved settings'


async def test_token_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('GRAIN_ACCESS_TOKEN', 'grain-token')
    plugin = load_grain(host({'read_only': False}))
    capability = grain(plugin)
    assert capability.client is None and capability.auth is None and not capability.read_only
    assert grain(plugin) is capability, 'rebuilt only when the token source or a setting changes'
    assert (
        await plugin.commands.execute_async('/grain status')
        == 'Grain uses the GRAIN_ACCESS_TOKEN environment variable.'
    )
    assert 'cannot revoke' in await plugin.commands.execute_async('/grain logout')


async def test_without_a_token_it_signs_in_through_the_browser_and_keeps_tokens_in_the_keyring() -> None:
    plugin = load_grain(host())
    capability = grain(plugin)
    assert capability.auth is None and capability.read_only
    oauth = sign_in(capability)
    assert oauth.tokens.name == TOKEN_ACCOUNT
    assert 'Not signed in' in await plugin.commands.execute_async('/grain status')

    storage = TokenStorageAdapter(oauth.tokens, server_url=GRAIN_MCP_URL)
    token = OAuthToken(access_token='access', token_type='Bearer', refresh_token='refresh')
    await storage.set_tokens(token)
    oauth.context.current_tokens = token
    assert 'Signed in to Grain' in await plugin.commands.execute_async('/grain status')

    assert 'Signed out of Grain' in await plugin.commands.execute_async('/grain logout')
    assert await storage.get_tokens() is None
    assert oauth.context.current_tokens is None, 'the loaded client cannot keep using the old token'
    with pytest.raises(ValueError, match='Usage: /grain'):
        await plugin.commands.execute_async('/grain login')


async def test_sign_in_prints_the_url_before_opening_the_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []

    async def open_browser(self: OAuth, authorization_url: str) -> None:
        opened.append(authorization_url)

    monkeypatch.setattr(OAuth, 'redirect_handler', open_browser)
    output = io.StringIO()
    plugin = load_grain(host(output=output))
    url = 'https://api.grain.com/oauth/authorize?state=abc'
    await sign_in(grain(plugin)).redirect_handler(url)
    assert opened == [url]
    assert url in output.getvalue()


async def test_headless_sign_in_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    async def open_browser(self: OAuth, authorization_url: str) -> None:
        raise AssertionError('no browser without someone to sign in')  # pragma: no cover

    monkeypatch.setattr(OAuth, 'redirect_handler', open_browser)
    plugin = load_grain(host(full_screen=no_screen))
    with pytest.raises(UserError, match='Run clai2 interactively once to sign in, or set GRAIN_ACCESS_TOKEN'):
        await sign_in(grain(plugin)).redirect_handler('https://api.grain.com/oauth/authorize')


@pytest.mark.parametrize('settings', [{'readonly': True}, {'auth': 'secret'}, {'token': 'secret'}])
def test_settings_hold_no_secret(settings: dict[str, JsonValue]) -> None:
    with pytest.raises(ValidationError):
        load_grain(host(settings))


def test_completes_subcommands() -> None:
    plugin = load_grain(host())
    [command] = plugin.commands
    assert list(command.complete([])) == ['key', 'logout', 'status']
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
    assert sign_in(grain(plugin)).tokens.name == TOKEN_ACCOUNT
    result = await plugin.commands.execute_async('/grain key')
    assert result == 'Grain uses the /keys entry GRAIN_ACCESS_TOKEN from the next prompt.'
    assert callable(grain(plugin).auth), 'the running session switches without a reload'
    assert prompt.labels == [('Grain access token (saved in /keys as GRAIN_ACCESS_TOKEN; Enter for none): ', True)]
    assert api_keys.load_keys()[KEY_NAME].get_secret_value() == 'typed-secret'
    saved = load_codex_credentials(account=KEY_ACCOUNT)
    assert saved is not None and 'typed-secret' not in saved
    assert api_keys.key_users(name=KEY_NAME) == ['grain'], 'renaming a key Grain uses is refused'

    reloaded = load_grain(host())
    capability = grain(reloaded)
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
    auth = grain(plugin).auth
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
        uses_key = grain(loaded).client is None
        assert uses_key == (expected == 'Grain key unchanged.')


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
    capability = grain(plugin)
    assert callable(capability.auth) and capability.auth(run_context()) == 'shared-secret'
    assert capability.include_instructions and capability.read_only


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
