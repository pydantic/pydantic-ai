"""The `logfire_mcp` built-in: its settings menu, keys kept in `/keys`, credential order, and OAuth."""

import io
import threading
import webbrowser
from collections.abc import Awaitable, Callable
from pathlib import Path

import anyio
import pytest
from fastmcp import Client
from keyring.errors import KeyringError
from pydantic import JsonValue, SecretStr
from rich.console import Console
from termflow.tui import MenuItem
from termflow.tui.menu import MenuResult
from termflow.tui.textinput import TextInputResult
from typing_extensions import TypeIs

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.logfire_mcp import LOGFIRE_EU_MCP_URL, LogfireMCP
from pydantic_clai2 import DEFAULT_PLUGINS, api_keys
from pydantic_clai2.api_keys import KeyReference, SavedKey
from pydantic_clai2.commands import Commands
from pydantic_clai2.field_menu import CUSTOM, is_save_and_close
from pydantic_clai2.logfire_mcp import SETUP, LogfireMCPSource, activate
from pydantic_clai2.logfire_oauth import SIGN_IN_TIMEOUT, DeviceAuth, SignInError, Tokens
from pydantic_clai2.plugin_loader import PluginError, PluginLoader
from pydantic_clai2.plugin_menu import PluginMenu, open_plugins_menu
from pydantic_clai2.plugins import PluginHost, SessionStart
from pydantic_clai2.settings_store import SettingsStore
from tests.clai2.menu_script import Script, pick, typed

BUILTIN = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'logfire_mcp')
CLOSE = MenuResult(cancelled=True)
SELF_HOSTED = 'https://logfire.example.com/mcp'


class Shell:
    """A loader plus what a test inspects: printed output and the settings file."""

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
        return declaration.settings

    def capability(self) -> LogfireMCP[None]:
        [capability] = self.loader.capabilities()
        assert is_logfire_mcp(capability)
        return capability


def is_logfire_mcp(capability: object) -> TypeIs[LogfireMCP[None]]:
    return isinstance(capability, LogfireMCP)


def script(
    monkeypatch: pytest.MonkeyPatch,
    lists: list[MenuResult],
    choices: list[MenuResult] | None = None,
    texts: list[TextInputResult] | None = None,
) -> Script:
    scripted = Script(lists=[*lists, CLOSE], choices=choices or [], texts=texts or [])
    monkeypatch.setattr('pydantic_clai2.logfire_mcp.RUNNERS', scripted.runners)
    return scripted


def key_choice(monkeypatch: pytest.MonkeyPatch, choice: str | KeyReference | None) -> list[tuple[str, bool]]:
    """Answer `prompt_api_key`, whose saved-key list needs a real terminal, and record its labels."""
    calls: list[tuple[str, bool]] = []

    async def prompt_api_key(*, prompt: object, label: str, optional: bool = False) -> str | KeyReference | None:
        calls.append((label, optional))
        return choice

    monkeypatch.setattr('pydantic_clai2.logfire_mcp.prompt_api_key', prompt_api_key)
    return calls


@pytest.fixture(autouse=True)
def opened(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record sign-in links instead of opening a browser."""
    links: list[str] = []

    def open_link(url: str) -> bool:  # pragma: no cover - a safety net; tests here stub the sign-in before a link opens
        links.append(url)
        return True

    monkeypatch.setattr(webbrowser, 'open', open_link)
    return links


def keyed(name: str) -> LogfireMCP[None]:
    return LogfireMCP[None](auth=SavedKey(name=name, setup=SETUP), read_only=True)


def test_declared_disabled_with_no_settings() -> None:
    assert (BUILTIN.factory, BUILTIN.enabled, BUILTIN.settings) == ('pydantic_clai2.logfire_mcp', False, {})


async def test_enable_opens_the_menu_and_every_option_saves_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = key_choice(monkeypatch, ' typed-secret ')
    shown = script(
        monkeypatch,
        lists=[pick('key'), pick('url'), pick('read_only'), pick('include_instructions'), pick('oauth')],
        choices=[pick(CUSTOM), pick('false'), pick('false'), pick('false')],
        texts=[typed(SELF_HOSTED)],
    )
    shell = Shell(tmp_path)
    assert await shell.loader.command(['enable', 'logfire_mcp']) == '\n'.join(
        [
            'Enabled logfire_mcp.',
            'Logfire uses the saved key LOGFIRE_API_KEY. Manage it in /keys.',
            'Saved Destination.',
            'Saved Tools.',
            'Saved Server instructions.',
            'Saved Browser sign-in.',
        ]
    )
    assert calls == [('Logfire API key (saved in /keys as LOGFIRE_API_KEY)', True)]
    assert shown.opened.count('list') == 6
    assert api_keys.load_keys()['LOGFIRE_API_KEY'].get_secret_value() == 'typed-secret'
    assert shell.saved() == {
        'key': {'name': 'LOGFIRE_API_KEY'},
        'url': SELF_HOSTED,
        'oauth': False,
        'read_only': False,
        'include_instructions': False,
    }
    assert b'typed-secret' not in shell.path.read_bytes()
    capability = shell.capability()
    assert capability == LogfireMCP[None](
        auth=SavedKey(name='LOGFIRE_API_KEY', setup=SETUP), url=SELF_HOSTED, read_only=False, include_instructions=False
    )


async def test_reopening_repicks_a_saved_key_and_region_without_reinstalling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api_keys.save_key(name='SHARED', value='shared-secret')
    shell = Shell(tmp_path, {'key': {'name': 'LOGFIRE_API_KEY'}})
    script(monkeypatch, lists=[])
    await shell.loader.command(['enable', 'logfire_mcp'])
    assert 'Logfire MCP has no credential: LOGFIRE_API_KEY is not in /keys. Run /plugins' in shell.output.getvalue()
    key_choice(monkeypatch, KeyReference(name='SHARED'))
    script(monkeypatch, lists=[pick('key'), pick('url')], choices=[pick(LOGFIRE_EU_MCP_URL)])
    assert await shell.loader.command(['configure', 'logfire_mcp']) == (
        'Logfire uses the saved key SHARED. Manage it in /keys.\nSaved Destination.'
    )
    assert shell.saved()['key'] == {'name': 'SHARED'}
    assert b'shared-secret' not in shell.path.read_bytes()
    capability = shell.capability()
    assert (capability.auth, capability.url) == (SavedKey(name='SHARED', setup=SETUP), LOGFIRE_EU_MCP_URL)


async def test_no_api_key_clears_the_choice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('LOGFIRE_API_KEY', 'env-key')
    shell = Shell(tmp_path, {'key': {'name': 'OLD'}})
    script(monkeypatch, lists=[])
    await shell.loader.enable('logfire_mcp')
    key_choice(monkeypatch, '')
    script(monkeypatch, lists=[pick('key')])
    assert await shell.loader.configure('logfire_mcp') == (
        'Logfire uses LOGFIRE_API_KEY from the environment or /keys, then browser sign-in.'
    )
    assert shell.saved()['key'] is None
    assert shell.capability() == LogfireMCP[None](read_only=True)


async def test_new_key_is_typed_masked_when_no_keys_are_saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shell = Shell(tmp_path, {'oauth': False})
    script(monkeypatch, lists=[pick('key')], texts=[typed('masked-secret')])
    await shell.loader.command(['enable', 'logfire_mcp'])
    assert api_keys.load_keys()['LOGFIRE_API_KEY'].get_secret_value() == 'masked-secret'


@pytest.mark.parametrize('answer', [TextInputResult(cancelled=True), typed('   ')])
async def test_cancelled_or_blank_key_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: TextInputResult
) -> None:
    shell = Shell(tmp_path, {'oauth': False})
    await shell.loader.enable('logfire_mcp')
    script(monkeypatch, lists=[pick('key')], texts=[answer])
    assert await shell.loader.configure('logfire_mcp') == 'Logfire MCP settings unchanged.'
    assert api_keys.load_keys() == {}


async def test_replacing_a_shared_key_needs_confirmation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api_keys.save_key(name='LOGFIRE_API_KEY', value='shared')
    shell = Shell(tmp_path)
    await shell.loader.enable('logfire_mcp')
    key_choice(monkeypatch, 'other')
    script(monkeypatch, lists=[pick('key'), pick('key')], choices=[pick(False), pick(True)])
    assert await shell.loader.configure('logfire_mcp') == (
        'Logfire uses the saved key LOGFIRE_API_KEY. Manage it in /keys.'
    )
    assert api_keys.load_keys()['LOGFIRE_API_KEY'].get_secret_value() == 'other'


def test_menu_validates_resets_and_notes_where_the_key_comes_from(monkeypatch: pytest.MonkeyPatch) -> None:
    host = PluginHost[None](name='logfire_mcp', console=Console(file=io.StringIO()), settings={'read_only': False})
    source = LogfireMCPSource(host)

    def note() -> str:
        return {row.key: row for row in source.rows()}['key'].note

    rows = {row.key: row for row in source.rows()}
    assert (source.current(rows['key']), note()) == ('(none)', 'browser sign-in, if on')
    assert rows['oauth'].note == 'signed out: signs in on the next run'
    assert (rows['url'].note, rows['read_only'].note) == ('', '')
    api_keys.save_key(name='LOGFIRE_API_KEY', value='saved')
    assert note() == 'LOGFIRE_API_KEY from /keys'
    monkeypatch.setenv('LOGFIRE_API_KEY', 'env')
    assert note() == 'LOGFIRE_API_KEY from the environment'
    source.save(source.settings.model_copy(update={'key': KeyReference(name='GONE')}))
    assert (source.current(rows['key']), note()) == ('GONE', 'missing from /keys')
    source.save(source.settings.model_copy(update={'key': KeyReference(name='LOGFIRE_API_KEY')}))
    assert note() == ''
    bad_url = 'Value error, Use an https:// URL without credentials, a query, or a fragment.'
    for url in ('http://h.test/mcp', 'https://u:pw@h.test/mcp', 'https://h.test/mcp?a=1', 'https://h.test/mcp#staging'):
        assert source.problem(rows['url'], url) == bad_url
    assert source.problem(rows['read_only'], 'maybe') == 'Input should be a valid boolean'
    assert source.problem(rows['url'], SELF_HOSTED) is None
    assert source.current(rows['read_only']) == 'false'
    assert source.reset(rows['read_only']) == 'Reset Tools.'
    assert source.current(rows['read_only']) == 'true'


@pytest.mark.parametrize(
    'settings',
    [
        {'key': 'inline-secret'},
        {'key': {'name': 'X', 'value': 'secret'}},
        {'auth': 'secret'},
        {'region': 'eu'},
        {'url': 'http://logfire-us.pydantic.dev/mcp'},
    ],
)
async def test_settings_cannot_hold_a_secret_or_invalid_options(tmp_path: Path, settings: dict[str, JsonValue]) -> None:
    shell = Shell(tmp_path, settings)
    with pytest.raises(PluginError):
        await shell.loader.enable('logfire_mcp')
    assert shell.loader.capabilities() == []


async def test_environment_key_wins_over_the_conventional_saved_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('LOGFIRE_API_KEY', 'env-key')
    api_keys.save_key(name='LOGFIRE_API_KEY', value='saved')
    shell = Shell(tmp_path)
    await shell.loader.enable('logfire_mcp')
    assert shell.capability() == LogfireMCP[None](read_only=True)


async def test_conventional_saved_key_beats_browser_sign_in_and_resolves_each_run(
    tmp_path: Path,
) -> None:
    api_keys.save_key(name='LOGFIRE_API_KEY', value='first')
    shell = Shell(tmp_path)
    await shell.loader.enable('logfire_mcp')
    capability = shell.capability()
    assert capability == keyed('LOGFIRE_API_KEY')
    assert isinstance(capability.auth, SavedKey)
    api_keys.save_key(name='LOGFIRE_API_KEY', value='second')
    assert capability.auth(None) == 'second'
    api_keys.delete_key(name='LOGFIRE_API_KEY')
    with pytest.raises(UserError, match=r'LOGFIRE_API_KEY is missing\. Run /plugins configure logfire_mcp'):
        capability.auth(None)


async def test_browser_sign_in_is_the_default_and_waits_long_enough_for_the_code(tmp_path: Path) -> None:
    shell = Shell(tmp_path, {'url': LOGFIRE_EU_MCP_URL})
    await shell.loader.enable('logfire_mcp')
    assert shell.output.getvalue() == ''
    client = shell.capability().client
    assert isinstance(client, Client)
    assert client._init_timeout == SIGN_IN_TIMEOUT  # pyright: ignore[reportPrivateUsage]
    assert str(client.transport.url) == LOGFIRE_EU_MCP_URL
    assert isinstance(client.transport.auth, DeviceAuth)


async def test_with_sign_in_off_and_no_key_the_menu_still_loads_and_runs_fail_closed(tmp_path: Path) -> None:
    shell = Shell(tmp_path, {'oauth': False})
    await shell.loader.enable('logfire_mcp')
    assert f'Logfire MCP has no credential: LOGFIRE_API_KEY is not in /keys. {SETUP}' in shell.output.getvalue()
    capability = shell.capability()
    auth = capability.auth
    assert auth == SavedKey(name='LOGFIRE_API_KEY', setup=SETUP)
    assert isinstance(auth, SavedKey)
    with pytest.raises(UserError, match='LOGFIRE_API_KEY is missing'):
        auth(None)


def logfire_command(
    settings: dict[str, JsonValue] | None = None,
) -> tuple[Callable[[list[str]], Awaitable[str]], io.StringIO]:
    output = io.StringIO()
    host = PluginHost[None](name='logfire_mcp', console=Console(file=output, width=200), settings=settings or {})
    activate(host)
    [command] = host.commands

    async def run(args: list[str]) -> str:
        result = command.handler(args)
        return result if isinstance(result, str) else await result

    return run, output


async def test_login_signs_in_now_through_the_browser(monkeypatch: pytest.MonkeyPatch, opened: list[str]) -> None:
    calls: list[tuple[str, bool]] = []

    async def sign_in(auth: DeviceAuth) -> Tokens:
        calls.append((auth._resource, auth._read_only))  # pyright: ignore[reportPrivateUsage]
        auth._announce('Enter code: ABCD-EFGH')  # pyright: ignore[reportPrivateUsage]
        return Tokens(client_id='c', token_endpoint='https://t', access_token='a', expires_at=0)

    monkeypatch.setattr(DeviceAuth, 'sign_in', sign_in)
    command, output = logfire_command({'url': LOGFIRE_EU_MCP_URL, 'read_only': False})
    assert await command(['login']) == 'Logfire runs use this sign-in when no API key is chosen, set, or saved.'
    assert calls == [(LOGFIRE_EU_MCP_URL, False)]
    assert output.getvalue() == 'Enter code: ABCD-EFGH\n'


async def test_sign_in_text_from_the_server_cannot_drive_the_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    async def sign_in(auth: DeviceAuth) -> Tokens:
        auth._announce('Enter code: \x1b]52;c;eA==\x07')  # pyright: ignore[reportPrivateUsage]
        raise SignInError('Logfire refused browser sign-in: \x1b[2J')

    monkeypatch.setattr(DeviceAuth, 'sign_in', sign_in)
    command, output = logfire_command()
    with pytest.raises(ValueError) as raised:
        await command(['login'])
    assert str(raised.value) == 'Logfire refused browser sign-in: \\x1b[2J'
    assert output.getvalue() == 'Enter code: \\x1b]52;c;eA==\\x07\n'


async def test_login_failures_are_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    async def sign_in(auth: DeviceAuth) -> Tokens:
        raise SignInError('Logfire sign-in was denied. Run /logfire_mcp login to retry.')

    monkeypatch.setattr(DeviceAuth, 'sign_in', sign_in)
    command, _ = logfire_command()
    with pytest.raises(ValueError, match='Logfire sign-in was denied'):
        await command(['login'])


async def test_logout_forgets_only_the_sign_in(monkeypatch: pytest.MonkeyPatch) -> None:
    forgotten: list[bool] = [True, False]
    monkeypatch.setattr('pydantic_clai2.logfire_mcp.forget', lambda: forgotten.pop(0))
    api_keys.save_key(name='LOGFIRE_API_KEY', value='kept')
    command, _ = logfire_command()
    assert await command(['logout']) == 'Forgot the Logfire browser sign-in. Keys in /keys are kept.'
    assert await command(['logout']) == 'There was no Logfire browser sign-in to forget.'
    assert 'LOGFIRE_API_KEY' in api_keys.load_keys()
    with pytest.raises(ValueError, match=r'Usage: /logfire_mcp login\|logout'):
        await command([])


@pytest.mark.parametrize('error', [KeyringError('locked'), OSError('read-only')], ids=['keyring', 'file'])
async def test_logout_failures_are_reported(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    def refuse(*, account: str) -> None:
        raise error

    monkeypatch.setattr('pydantic_clai2.logfire_oauth.delete_credentials', refuse)
    command, _ = logfire_command()
    with pytest.raises(
        ValueError, match=r'Could not delete the saved Logfire browser sign-in \((KeyringError|OSError)\)'
    ):
        await command(['logout'])


async def test_registers_the_logout_command(tmp_path: Path) -> None:
    api_keys.save_key(name='LOGFIRE_API_KEY', value='kept')
    shell = Shell(tmp_path)
    await shell.loader.enable('logfire_mcp')
    [host] = [entry.host for entry in shell.loader.entries() if entry.host]
    assert [c.name for c in host.commands] == ['logfire_mcp']


async def test_add_replacing_the_builtin_opens_the_menu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api_keys.save_key(name='LOGFIRE_API_KEY', value='saved')
    shell = Shell(tmp_path)
    script(monkeypatch, lists=[pick('read_only')], choices=[pick('true')])
    added = await shell.loader.command(['add', 'logfire_mcp', 'pydantic_clai2.logfire_mcp', '{"read_only": false}'])
    assert added == 'Replaced built-in logfire_mcp.\nSaved Tools.'
    assert shell.capability() == keyed('LOGFIRE_API_KEY')


async def test_configure_needs_a_loaded_plugin_with_a_menu(tmp_path: Path) -> None:
    shell = Shell(tmp_path)
    with pytest.raises(ValueError, match='not loaded; enable it before configuring'):
        await shell.loader.command(['configure', 'logfire_mcp'])
    shell.store.save_plugin(BUILTIN.model_copy(update={'id': 'plain', 'factory': 'pydantic_clai2.repo_context'}))
    assert await shell.loader.command(['enable', 'plain']) == 'Enabled plain.'
    with pytest.raises(ValueError, match='no settings menu'):
        await shell.loader.configure('plain')


async def test_enabling_in_the_plugins_menu_opens_the_settings_menu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api_keys.save_key(name='LOGFIRE_API_KEY', value='saved')
    shell = Shell(tmp_path)
    script(monkeypatch, lists=[pick('read_only')], choices=[pick('false')])

    def run(menu: PluginMenu[None]) -> MenuResult:
        item, save_and_close = menu.items()
        if menu.notice is not None:  # Back from the settings menu.
            return menu.close(Redraw(), save_and_close)
        opened = menu.toggle(Redraw(), item)
        assert opened is not None
        return opened

    assert await open_plugins_menu(shell.loader, run=run) == 'Saved Tools.'
    assert shell.capability().read_only is False


async def test_plugins_menu_stays_open_when_there_is_nothing_to_configure(tmp_path: Path) -> None:
    shell = Shell(tmp_path)
    shell.store.save_plugin(
        BUILTIN.model_copy(update={'id': 'plain', 'factory': 'pydantic_clai2.repo_context', 'enabled': True})
    )
    await shell.loader.load_all()

    def run(menu: PluginMenu[None]) -> MenuResult:
        assert menu.configure(Redraw(), MenuItem('none', value=None)) is None
        rows = [item for item in menu.items() if not is_save_and_close(item)]
        logfire_row, plain_row = sorted(rows, key=lambda item: str(item.value))
        assert menu.configure(Redraw(), logfire_row) is None
        assert menu.notice == 'Enable logfire_mcp to configure it.'
        assert menu.configure(Redraw(), plain_row) is None
        assert menu.notice == 'plain has no settings menu.'
        return menu.close(Redraw(), plain_row)

    assert await open_plugins_menu(shell.loader, run=run) == ''


async def test_cancelling_configure_waits_for_the_key_picker_to_clean_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shell = Shell(tmp_path, {'oauth': False})
    await shell.loader.enable('logfire_mcp')
    opened = anyio.Event()
    finished: list[str] = []

    async def prompt_api_key(*, prompt: object, label: str, optional: bool = False) -> str | KeyReference | None:
        opened.set()
        try:
            await anyio.sleep_forever()
        finally:
            # Slow cleanup, like a nested menu worker being joined: configure must wait for it.
            with anyio.CancelScope(shield=True):
                await anyio.sleep(0.2)
            finished.append(label)
        return None  # pragma: no cover -- unreachable; keeps the signature honest

    monkeypatch.setattr('pydantic_clai2.logfire_mcp.prompt_api_key', prompt_api_key)
    script(monkeypatch, lists=[pick('key')])
    with anyio.fail_after(10):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(shell.loader.configure, 'logfire_mcp')
            await opened.wait()
            tasks.cancel_scope.cancel()
    assert finished == ['Logfire API key (saved in /keys as LOGFIRE_API_KEY)']


class Redraw:
    def replace_items(self, items: object) -> None:
        pass


async def test_keys_are_read_at_session_start_off_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    threads: list[int] = []

    def load_keys() -> dict[str, SecretStr]:
        threads.append(threading.get_ident())
        return {}

    monkeypatch.setattr('pydantic_clai2.logfire_mcp.load_keys', load_keys)
    host = PluginHost[None](name='logfire_mcp', console=Console(file=io.StringIO()), settings={})
    activate(host)
    assert (host.capabilities, threads) == ([], [])
    shell = Shell(tmp_path)
    await shell.loader.enable('logfire_mcp')
    assert len(threads) == 1
    assert threads[0] != threading.get_ident()
    assert isinstance(shell.capability().client, Client)


async def test_the_settings_menu_reads_keys_and_sign_in_off_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shell = Shell(tmp_path)
    await shell.loader.enable('logfire_mcp')
    threads: list[int] = []

    def status(*, resource: str, read_only: bool) -> str:
        threads.append(threading.get_ident())
        return 'signed out'

    monkeypatch.setattr('pydantic_clai2.logfire_mcp.status', status)
    script(monkeypatch, lists=[])
    assert await shell.loader.command(['configure', 'logfire_mcp']) == 'Logfire MCP settings unchanged.'
    assert threads
    assert threading.get_ident() not in threads
