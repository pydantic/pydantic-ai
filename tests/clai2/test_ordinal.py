"""The built-in `ordinal` plugin: a settings menu, a `/keys` token by name, the environment, or a browser sign-in."""

import io
import sys
from pathlib import Path

import keyring
import pytest
from fastmcp import Client
from fastmcp.client.auth import OAuth
from fastmcp.client.auth.oauth import TokenStorageAdapter
from fastmcp.client.transports import StreamableHttpTransport
from keyring.errors import KeyringLocked
from mcp.shared.auth import OAuthToken
from pydantic import JsonValue
from rich.console import Console
from termflow.tui import MenuItem
from termflow.tui.menu import MenuResult
from termflow.tui.textinput import TextInputResult

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.ordinal import Ordinal
from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2._app import create_shell
from pydantic_clai2.builtin_plugins.ordinal import (
    INSTRUCTIONS,
    INVALID,
    KEY,
    KEY_ACCOUNT,
    KEY_NAME,
    SIGN_IN,
    TOKENS,
    URL,
    USAGE,
    OrdinalAuth,
    OrdinalPlugin,
    OrdinalSource,
)
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.config.api_keys import KeyReference, delete_key, load_keys, rename_key, save_key
from pydantic_clai2.config.credential_store import delete_credentials, load_codex_credentials, save_codex_credentials
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.mcp import OAUTH_TIMEOUT, TokenStore
from pydantic_clai2.plugins import LoadedPlugin, PluginHost, SessionStart, load_plugin
from pydantic_clai2.plugins.loader import (
    _RETIRED_BUILTINS,  # pyright: ignore[reportPrivateUsage]
    PluginError,
    PluginLoader,
)
from pydantic_clai2.ui.menus.field_menu import FieldMenu
from pydantic_clai2.ui.menus.plugin_menu import PluginMenu, open_plugins_menu
from tests.clai2.conftest import stored_accounts
from tests.clai2.menu_script import Script, pick, typed

BUILTIN = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'ordinal')
CLOSE = MenuResult(cancelled=True)
Vault = dict[tuple[str, str], str]
Picked = str | KeyReference | None


@pytest.fixture
def vault(monkeypatch: pytest.MonkeyPatch) -> Vault:
    """A fresh keyring, and a terminal on stdin unless a test says otherwise."""
    entries: Vault = {}

    def get(service: str, account: str) -> str | None:
        return entries.get((service, account))

    def set_value(service: str, account: str, value: str) -> None:
        entries[service, account] = value

    monkeypatch.setattr(keyring, 'get_password', get)
    monkeypatch.setattr(keyring, 'set_password', set_value)
    monkeypatch.delenv(KEY_NAME, raising=False)
    terminal(monkeypatch, attached=True)
    return entries


def terminal(monkeypatch: pytest.MonkeyPatch, *, attached: bool) -> None:
    monkeypatch.setattr(sys.stdin, 'isatty', lambda: attached)


def script(
    monkeypatch: pytest.MonkeyPatch,
    lists: list[MenuResult],
    choices: list[MenuResult] | None = None,
    texts: list[TextInputResult] | None = None,
) -> Script:
    """Answer the settings menu's widgets in order; the list closes after `lists`."""
    scripted = Script(lists=[*lists, CLOSE], choices=choices or [], texts=texts or [])
    monkeypatch.setattr('pydantic_clai2.builtin_plugins.ordinal.RUNNERS', scripted.runners)
    return scripted


def key_choice(monkeypatch: pytest.MonkeyPatch, choice: Picked) -> list[str]:
    """Answer `prompt_api_key`, whose saved-key list needs a real terminal, and record its labels."""
    labels: list[str] = []

    async def prompt_api_key(*, prompt: object, label: str, optional: bool = False) -> Picked:
        labels.append(label)
        return choice

    monkeypatch.setattr('pydantic_clai2.builtin_plugins.ordinal.prompt_api_key', prompt_api_key)
    return labels


def choose_saved_key(name: str) -> None:
    save_codex_credentials(account=KEY_ACCOUNT, value=f'{{"token": {{"name": "{name}"}}}}')


class Shell:
    """A loader with only the `ordinal` built-in, plus its output and settings file."""

    def __init__(self, tmp_path: Path, settings: dict[str, JsonValue] | None = None) -> None:
        self.path = tmp_path / 'config.db'
        self.store = SettingsStore(self.path)
        self.output = io.StringIO()
        declaration = BUILTIN if settings is None else BUILTIN.model_copy(update={'settings': settings})
        self.loader: PluginLoader[None] = PluginLoader(
            store=self.store,
            console=Console(file=self.output, width=200),
            commands=Commands(),
            session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=self.store.load()),
            builtin=(declaration,),
        )

    def saved(self) -> dict[str, JsonValue]:
        [declaration] = self.store.plugins()
        assert declaration.enabled
        return declaration.settings

    def auth(self) -> OrdinalAuth[None]:
        [capability] = self.loader.capabilities()
        assert isinstance(capability, OrdinalAuth)
        return capability  # pyright: ignore[reportUnknownVariableType]

    async def next_run(self) -> Ordinal[None]:
        return await self.auth()(RunContext[None](deps=None, model=TestModel(), usage=RunUsage()))


async def enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **settings: JsonValue) -> Shell:
    """A shell with `ordinal` enabled and its first menu closed untouched."""
    shell = Shell(tmp_path, settings or None)
    script(monkeypatch, lists=[])
    await shell.loader.command(['enable', 'ordinal'])
    return shell


def plugin_host(settings: dict[str, JsonValue] | None = None) -> PluginHost[None]:
    return PluginHost[None](name='ordinal', console=Console(file=io.StringIO()), settings=settings or {})


def load_ordinal(plugin_host: PluginHost[None]) -> LoadedPlugin[None]:
    return load_plugin(OrdinalPlugin, plugin_host)


async def run(plugin: LoadedPlugin[None], *args: str) -> str:
    [command] = list(plugin.commands)
    result = command.handler(list(args))
    return result if isinstance(result, str) else await result


def test_declared_as_a_disabled_built_in_with_no_settings() -> None:
    assert BUILTIN.factory == 'pydantic_clai2.builtin_plugins.ordinal'
    assert not BUILTIN.enabled
    assert BUILTIN.settings == {}
    assert _RETIRED_BUILTINS['ordinal'].factory == 'pydantic_ai_harness.ordinal:Ordinal'


def test_url_matches_the_harness_endpoint() -> None:
    toolset = Ordinal[None](auth='token').get_toolset()
    assert isinstance(toolset, MCPToolset)
    assert isinstance(toolset.client, Client)
    transport = toolset.client.transport
    assert isinstance(transport, StreamableHttpTransport)
    assert transport.url == URL


async def test_enable_opens_the_menu_and_every_option_saves_immediately(
    vault: Vault, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shown = script(
        monkeypatch,
        lists=[pick('sign_in'), pick('key'), pick('include_instructions')],
        choices=[pick('browser'), pick('false')],
        texts=[typed(' ord_new ')],
    )
    shell = Shell(tmp_path)
    assert await shell.loader.command(['enable', 'ordinal']) == '\n'.join(
        [
            'Enabled ordinal.',
            'Saved Sign-in.',
            'Ordinal uses ORDINAL_ACCESS_TOKEN from /keys. Manage it there.',
            'Saved Sign-in: Saved key from /keys.',
            'Saved Server instructions.',
        ]
    )
    assert shown.opened == ['list', 'choice', 'list', 'text', 'list', 'choice', 'list']
    assert load_keys()[KEY_NAME].get_secret_value() == 'ord_new'
    assert shell.saved() == {'sign_in': 'key', 'include_instructions': False}
    assert b'ord_new' not in shell.path.read_bytes()
    assert 'ord_new' not in (load_codex_credentials(account=KEY_ACCOUNT) or '')
    ordinal = await shell.next_run()
    assert ordinal.auth == 'ord_new' and not ordinal.include_instructions


async def test_reopening_repicks_a_saved_key_and_options_without_reinstalling(
    vault: Vault, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_key(name='SHARED_ORDINAL', value='shared-token')
    shell = await enabled(tmp_path, monkeypatch, sign_in='environment')
    labels = key_choice(monkeypatch, KeyReference(name='SHARED_ORDINAL'))
    script(monkeypatch, lists=[pick('key'), pick('sign_in')], choices=[pick('auto')])
    assert await shell.loader.command(['configure', 'ordinal']) == '\n'.join(
        [
            'Ordinal uses SHARED_ORDINAL from /keys. Manage it there.',
            'Saved Sign-in: Saved key from /keys.',
            'Saved Sign-in.',
        ]
    )
    assert labels == [f'Ordinal access token (saved in /keys as {KEY_NAME})']
    assert shell.saved() == {'sign_in': 'auto', 'include_instructions': True}
    assert (await shell.next_run()).auth == 'shared-token'
    assert set(load_keys()) == {'SHARED_ORDINAL'}
    with pytest.raises(ValueError, match='used by ordinal'):
        rename_key(name='SHARED_ORDINAL', new_name='ELSEWHERE')


async def test_key_is_resolved_every_run_and_fails_closed(
    vault: Vault, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(KEY_NAME, 'from-the-environment')
    save_key(name=KEY_NAME, value='first')
    choose_saved_key(KEY_NAME)
    shell = await enabled(tmp_path, monkeypatch)
    # With `auto`, a chosen key wins over the environment, as an explicit `auth` does in harness `Ordinal`.
    assert (await shell.next_run()).auth == 'first'
    save_key(name=KEY_NAME, value='replaced')
    assert (await shell.next_run()).auth == 'replaced'
    delete_key(name=KEY_NAME)
    with pytest.raises(UserError, match=f'{KEY_NAME} is missing'):
        await shell.next_run()


async def test_menu_marks_a_missing_or_invalid_key_and_r_resets(
    vault: Vault, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_key(name=KEY_NAME, value='token')
    choose_saved_key(KEY_NAME)
    shell = await enabled(tmp_path, monkeypatch, sign_in='key', include_instructions=False)
    source = OrdinalSource(plugin_host(shell.saved()))
    assert [row.note for row in source.rows()] == ['', '', '']
    delete_key(name=KEY_NAME)
    assert [row.note for row in source.rows()] == ['', 'missing from /keys', '']

    script(monkeypatch, lists=[pick('key'), pick('sign_in')], choices=[pick('Keep current')])
    key_choice(monkeypatch, None)
    assert await shell.loader.configure('ordinal') == 'Ordinal settings unchanged.'

    menu = FieldMenu(source)
    presses = [menu.reset_marker(None, MenuItem(row.label, value=row.key)) for row in (KEY, INSTRUCTIONS, SIGN_IN)]
    script(monkeypatch, lists=presses)
    assert await shell.loader.configure('ordinal') == (
        'Ordinal uses no /keys entry.\nReset Server instructions.\nReset Sign-in.'
    )
    assert shell.saved() == {'sign_in': 'auto', 'include_instructions': True}
    assert source.current(KEY) == KEY.default

    save_codex_credentials(account=KEY_ACCOUNT, value='{"token": "a raw secret"}')
    assert source.current(KEY) == INVALID
    assert [row.note for row in source.rows()] == ['', '', '']


def test_menu_validates_like_saving_would() -> None:
    source = OrdinalSource(plugin_host())
    assert source.problem(SIGN_IN, 'key') is None
    assert source.problem(SIGN_IN, 'sometimes') is not None
    assert source.problem(INSTRUCTIONS, 'false') is None
    assert source.problem(INSTRUCTIONS, 'maybe') is not None
    assert source.current(INSTRUCTIONS) == 'true'
    assert source.apply(SIGN_IN, 'browser') == 'Saved Sign-in.'
    assert source.current(SIGN_IN) == 'browser'
    assert source.title == 'Ordinal'


async def test_typed_token_needs_confirmation_to_replace_a_shared_key(
    vault: Vault, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_key(name=KEY_NAME, value='kept')
    shell = await enabled(tmp_path, monkeypatch)
    key_choice(monkeypatch, 'unwanted')
    script(monkeypatch, lists=[pick('key'), pick('key')], choices=[pick(False), CLOSE])
    assert await shell.loader.configure('ordinal') == 'Ordinal settings unchanged.'
    assert load_keys()[KEY_NAME].get_secret_value() == 'kept'

    key_choice(monkeypatch, 'wanted')
    script(monkeypatch, lists=[pick('key')], choices=[pick(True)])
    assert 'ORDINAL_ACCESS_TOKEN from /keys' in await shell.loader.configure('ordinal')
    assert load_keys()[KEY_NAME].get_secret_value() == 'wanted'


async def test_a_key_saved_meanwhile_by_another_process_is_not_replaced_unasked(
    vault: Vault, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shell = await enabled(tmp_path, monkeypatch)

    async def prompt_api_key(*, prompt: object, label: str, optional: bool = False) -> Picked:
        # The picker listed no keys; another CLAI process saves one before this token is.
        save_key(name=KEY_NAME, value='theirs')
        return 'mine'

    monkeypatch.setattr('pydantic_clai2.builtin_plugins.ordinal.prompt_api_key', prompt_api_key)
    shown = script(monkeypatch, lists=[pick('key')], choices=[pick(False)])
    assert await shell.loader.configure('ordinal') == 'Ordinal settings unchanged.'
    assert shown.opened == ['list', 'choice', 'list']
    assert load_keys()[KEY_NAME].get_secret_value() == 'theirs'


async def test_cancelled_or_blank_token_changes_nothing(
    vault: Vault, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shell = await enabled(tmp_path, monkeypatch)
    # No saved keys: the real `prompt_api_key` goes straight to the masked prompt.
    script(monkeypatch, lists=[pick('key'), pick('key')], texts=[TextInputResult(cancelled=True), typed('  ')])
    assert await shell.loader.configure('ordinal') == 'Ordinal settings unchanged.'
    assert load_keys() == {}
    assert shell.saved() == {}


async def test_each_sign_in_method_is_used_alone(vault: Vault, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    save_key(name=KEY_NAME, value='token')
    choose_saved_key(KEY_NAME)
    monkeypatch.setenv(KEY_NAME, 'from-the-environment')

    environment = await enabled(tmp_path / 'environment', monkeypatch, sign_in='environment')
    ordinal = await environment.next_run()
    assert ordinal.auth is None and ordinal.client is None

    browser = await enabled(tmp_path / 'browser', monkeypatch, sign_in='browser')
    client = (await browser.next_run()).client
    assert isinstance(client, Client)
    assert isinstance(client.transport, StreamableHttpTransport)
    assert client.transport.url == URL
    assert isinstance(client.transport.auth, OAuth)
    # `MCPToolset` gives a bare transport a 5 second handshake, which would end the sign-in early.
    assert client._init_timeout == OAUTH_TIMEOUT  # pyright: ignore[reportPrivateUsage]

    delete_credentials(account=KEY_ACCOUNT)
    key = await enabled(tmp_path / 'key', monkeypatch, sign_in='key')
    with pytest.raises(UserError, match='no /keys entry'):
        await key.next_run()


async def test_status_names_the_credential_in_use(vault: Vault, monkeypatch: pytest.MonkeyPatch) -> None:
    async def status(**settings: JsonValue) -> str:
        plugin = load_ordinal(plugin_host(settings))
        return await run(plugin)

    setup = 'Run /plugins configure ordinal to choose how it signs in.'
    assert await status() == 'Ordinal: not signed in; the browser opens on first use.'
    assert await status(sign_in='key') == f'Ordinal has no /keys entry, so runs fail. {setup}'
    assert await status(sign_in='environment') == f'Ordinal uses `{KEY_NAME}`, which is not set, so runs fail. {setup}'
    monkeypatch.setenv(KEY_NAME, 'token')
    assert await status() == f'Ordinal uses `{KEY_NAME}` from the environment.'
    assert await status(sign_in='browser') == 'Ordinal: not signed in; the browser opens on first use.'
    save_key(name='MINE', value='token')
    choose_saved_key('MINE')
    assert await status() == 'Ordinal uses MINE from /keys.'
    assert await status(sign_in='environment') == f'Ordinal uses `{KEY_NAME}` from the environment.'
    delete_key(name='MINE')
    assert await status() == f'Ordinal uses MINE, which is missing from /keys, so runs fail. {setup}'


async def test_logout_ends_the_browser_session(vault: Vault, monkeypatch: pytest.MonkeyPatch) -> None:
    token = OAuthToken(access_token='access', token_type='Bearer', refresh_token='refresh', expires_in=3600)
    await TokenStorageAdapter(TokenStore(TOKENS), server_url=URL).set_tokens(token)
    terminal(monkeypatch, attached=False)
    plugin = load_ordinal(plugin_host({'sign_in': 'browser'}))
    assert await run(plugin) == 'Ordinal: signed in through the browser. /ordinal logout signs out.'
    [auth] = plugin.capabilities
    assert isinstance(auth, OrdinalAuth)
    context = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())
    signed_in = (await auth(context)).client
    assert 'Signed out' in await run(plugin, 'logout')
    assert stored_accounts() == set()
    signed_out = (await auth(context)).client
    assert isinstance(signed_in, Client) and isinstance(signed_out, Client)
    # FastMCP keeps tokens inside the `OAuth` once connected; the next run must not reuse it.
    assert signed_out is not signed_in and signed_out.transport.auth is not signed_in.transport.auth
    assert await run(plugin) == 'Ordinal: not signed in; the browser opens on first use.'


async def test_unreadable_keyring_is_reported(vault: Vault, monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = load_ordinal(plugin_host())

    def locked(service: str, account: str) -> str | None:
        if account == KEY_ACCOUNT:
            return None
        raise KeyringLocked('locked')

    monkeypatch.setattr(keyring, 'get_password', locked)
    assert await run(plugin) == 'Ordinal: sign-in unknown; the keyring could not be read.'


async def test_invalid_saved_choice_fails_closed(vault: Vault) -> None:
    save_codex_credentials(account=KEY_ACCOUNT, value='{"token": "a raw secret"}')
    plugin = load_ordinal(plugin_host())
    [auth] = plugin.capabilities
    assert isinstance(auth, OrdinalAuth)
    with pytest.raises(UserError, match='/plugins configure ordinal'):
        await auth(RunContext[None](deps=None, model=TestModel(), usage=RunUsage()))


@pytest.mark.parametrize('settings', [{'token': 'ord_secret'}, {'sign_in': 'always'}, {'include_instructions': 'yes'}])
async def test_settings_cannot_hold_a_secret_or_invalid_options(
    vault: Vault, tmp_path: Path, settings: dict[str, JsonValue]
) -> None:
    shell = Shell(tmp_path, settings)
    with pytest.raises(PluginError) as raised:
        await shell.loader.enable('ordinal')
    assert 'ord_secret' not in str(raised.value)
    assert shell.loader.capabilities() == []


async def test_command_usage_and_completion(vault: Vault) -> None:
    plugin = load_ordinal(plugin_host())
    [command] = list(plugin.commands)
    assert command.name == 'ordinal'
    assert await run(plugin, 'nope') == USAGE
    assert list(command.complete([''])) == ['logout']
    assert list(command.complete(['logout', ''])) == []


async def test_no_credential_and_no_terminal_fails_to_enable(
    vault: Vault, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    terminal(monkeypatch, attached=False)
    shell = Shell(tmp_path)
    with pytest.raises(PluginError, match='/plugins configure ordinal'):
        await shell.loader.enable('ordinal')
    assert shell.loader.capabilities() == []

    save_key(name=KEY_NAME, value='token')
    choose_saved_key(KEY_NAME)
    script(monkeypatch, lists=[])
    await shell.loader.enable('ordinal')
    assert (await shell.next_run()).auth == 'token'


async def test_without_a_terminal_only_the_chosen_method_counts(vault: Vault, monkeypatch: pytest.MonkeyPatch) -> None:
    terminal(monkeypatch, attached=False)
    monkeypatch.setenv(KEY_NAME, 'token')
    with pytest.raises(UserError, match='sign-in `key`'):
        load_ordinal(plugin_host({'sign_in': 'key'}))
    plugin = load_ordinal(plugin_host({'sign_in': 'environment'}))
    assert len(plugin.capabilities) == 1
    monkeypatch.delenv(KEY_NAME)
    with pytest.raises(UserError, match='sign-in `environment`'):
        load_ordinal(plugin_host({'sign_in': 'environment'}))


async def test_add_replacing_the_builtin_opens_the_menu(
    vault: Vault, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shell = Shell(tmp_path)
    script(monkeypatch, lists=[pick('include_instructions')], choices=[pick('true')])
    assert await shell.loader.command(
        ['add', 'ordinal', 'pydantic_clai2.builtin_plugins.ordinal', '{"include_instructions": false}']
    ) == ('Replaced built-in ordinal.\nSaved Server instructions.')
    assert (await shell.next_run()).include_instructions


async def test_configure_needs_a_loaded_plugin_with_a_menu(vault: Vault, tmp_path: Path) -> None:
    shell = Shell(tmp_path)
    with pytest.raises(ValueError, match='not loaded; enable it before configuring'):
        await shell.loader.command(['configure', 'ordinal'])
    shell.store.save_plugin(
        BUILTIN.model_copy(update={'id': 'plain', 'factory': 'pydantic_clai2.builtin_plugins.repo_context'})
    )
    assert await shell.loader.command(['enable', 'plain']) == 'Enabled plain.'
    with pytest.raises(ValueError, match='no settings menu'):
        await shell.loader.configure('plain')


class Redraw:
    def replace_items(self, items: object) -> None:
        pass


async def test_enabling_in_the_plugins_menu_opens_the_settings_menu(
    vault: Vault, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shell = Shell(tmp_path)
    script(monkeypatch, lists=[pick('sign_in')], choices=[pick('browser')])
    notices: list[str | None] = []

    def run_menu(menu: PluginMenu[None]) -> MenuResult:
        notices.append(menu.notice)
        item, save_and_close = menu.items()
        result = menu.toggle(Redraw(), item) if len(notices) == 1 else MenuResult(item=save_and_close)
        assert result is not None
        return result

    assert await open_plugins_menu(shell.loader, run=run_menu) == 'Saved Sign-in.'
    assert notices == [None, 'Saved Sign-in.']
    assert shell.saved() == {'sign_in': 'browser', 'include_instructions': True}


async def test_saved_catalog_toggle_becomes_the_built_in(
    vault: Vault, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(KEY_NAME, 'token')
    store = SettingsStore(tmp_path / 'settings.db')

    def shell():
        return create_shell(
            Agent(TestModel()),
            deps=None,
            plugins=(),
            usage_limits=None,
            settings=None,
            project=ProjectSettings(),
            console=Console(file=io.StringIO(), force_terminal=True),
            store=store,
            builtin_plugins=[BUILTIN],
        )

    commands = shell().commands
    [plugins_command] = [command for command in commands if command.name == 'plugins']
    assert 'configure' in list(plugins_command.complete(['']))

    store.save_plugin(PluginSettings(id='ordinal', factory='pydantic_ai_harness.ordinal:Ordinal', enabled=True))
    loader = shell().loader
    [entry] = loader.entries()
    assert entry.builtin and entry.declaration.enabled
    assert entry.declaration.factory == 'pydantic_clai2.builtin_plugins.ordinal'
    await loader.load_all()
    [capability] = loader.capabilities()
    # `/plugins reload` re-imports the module, so compare where the class lives rather than its identity.
    assert type(capability).__module__ == 'pydantic_clai2.builtin_plugins.ordinal'

    customized = PluginSettings(id='ordinal', factory='pydantic_ai_harness.ordinal:Ordinal', settings={'id': 'mine'})
    store.save_plugin(customized)
    [entry] = shell().loader.entries()
    assert entry.declaration == customized and not entry.builtin
