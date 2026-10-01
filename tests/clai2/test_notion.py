"""The built-in `notion` plugin: a settings menu for `Notion`'s options, with its key chosen from `/keys` by name."""

import inspect
import io
from pathlib import Path

import anyio
import pytest
from fastmcp import Client
from fastmcp.client.auth import OAuth
from fastmcp.client.transports import StreamableHttpTransport
from pydantic import JsonValue, SecretStr
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
from pydantic_clai2.plugins import PluginHost, SessionStart
from pydantic_clai2.plugins.loader import PluginError, PluginLoader
from pydantic_clai2.ui.menus import key_picker
from pydantic_clai2.ui.menus.field_menu import FieldMenu
from pydantic_clai2.ui.menus.plugin_menu import PluginMenu, open_plugins_menu
from tests.clai2.conftest import stored_accounts
from tests.clai2.menu_script import Script, pick, typed

Vault = dict[tuple[str, str], str]
BUILTIN = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'notion')
CLOSE = MenuResult(cancelled=True)
CANCEL = TextInputResult(cancelled=True)
LABEL = 'Notion OAuth access token (saved in /keys as NOTION_API_KEY)'


class Shell:
    """A loader plus what a test inspects: printed output, commands, and the settings file."""

    def __init__(self, tmp_path: Path, settings: dict[str, JsonValue] | None = None) -> None:
        self.store = SettingsStore(tmp_path / 'settings.db')
        self.output = io.StringIO()
        self.commands = Commands()
        declaration = BUILTIN if settings is None else BUILTIN.model_copy(update={'settings': settings})
        self.loader = PluginLoader[None](
            store=self.store,
            console=Console(file=self.output, width=200),
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

    async def built(self) -> Notion[None]:
        """The run's `Notion`, from the per-run factory the plugin registers."""
        [capability] = self.loader.capabilities()
        assert callable(capability)
        result = capability(RunContext[None](deps=None, model=TestModel(), usage=RunUsage()))
        assert inspect.isawaitable(result)
        connected = await result
        assert isinstance(connected, Notion)
        return connected


def script(
    monkeypatch: pytest.MonkeyPatch,
    lists: list[MenuResult],
    choices: list[MenuResult] | None = None,
    texts: list[TextInputResult] | None = None,
    keys: tuple[str, ...] = (),
) -> Script:
    """Script the settings menu's widgets; `keys` drive `prompt_api_key`'s own saved-key list."""
    scripted = Script(lists=[*lists, CLOSE], choices=choices or [], texts=texts or [])
    monkeypatch.setattr(notion, 'RUNNERS', scripted.runners)
    pressed = iter(keys)
    monkeypatch.setattr(api_keys, 'menu_key', lambda: next(pressed))
    return scripted


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
    script(monkeypatch, lists=[pick('key')], keys=('enter',))
    await shell.loader.command(['enable', 'notion'])
    assert (await shell.built()).auth == 'ntn-first'
    script(monkeypatch, lists=[pick('key'), reset('read_only')], keys=('down', 'enter'))
    assert await shell.loader.command(['configure', 'notion']) == (
        'Notion uses the saved key TEAM_NOTION. Manage it in /keys.\nReset Tools.'
    )
    assert shell.saved() == {'auth': None, 'read_only': False, 'include_instructions': True}
    assert (await shell.built()).auth == 'ntn-team'
    script(monkeypatch, lists=[reset('key')])
    assert await shell.loader.configure('notion') == (
        'Notion no longer uses a saved key. The key itself stays in /keys.'
    )
    assert (await shell.built()).auth is None
    assert set(api_keys.load_keys()) == {'NOTION_API_KEY', 'TEAM_NOTION'}


async def test_a_chosen_key_is_resolved_each_run_and_fails_closed(tmp_path: Path) -> None:
    api_keys.save_key(name='NOTION_API_KEY', value='ntn-first')
    notion.select_key(KeyReference(name='NOTION_API_KEY'))
    shell = Shell(tmp_path)
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


@pytest.mark.parametrize(('confirm', 'value'), [(pick(True), 'ntn-new'), (pick(False), 'ntn-old'), (CLOSE, 'ntn-old')])
async def test_entering_a_new_value_asks_before_replacing_a_shared_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, confirm: MenuResult, value: str
) -> None:
    api_keys.save_key(name='NOTION_API_KEY', value='ntn-old')
    shell = Shell(tmp_path)
    await shell.loader.enable('notion')
    script(monkeypatch, lists=[pick('key')], choices=[confirm], texts=[typed('ntn-new')], keys=('down', 'enter'))
    message = await shell.loader.configure('notion')
    assert (message != 'Notion settings unchanged.') == (value == 'ntn-new')
    assert api_keys.load_keys()['NOTION_API_KEY'].get_secret_value() == value


async def test_a_key_another_session_saves_meanwhile_is_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shell = Shell(tmp_path)
    await shell.loader.enable('notion')
    script(monkeypatch, lists=[pick('key')], texts=[typed('ntn-mine')])
    real_load_keys = api_keys.load_keys

    def load_keys_then_another_session_saves() -> dict[str, SecretStr]:
        keys = real_load_keys()
        api_keys.save_key(name='NOTION_API_KEY', value='ntn-theirs')
        return keys

    monkeypatch.setattr(key_picker, 'load_keys', load_keys_then_another_session_saves)
    with pytest.raises(api_keys.KeyExistsError, match='NOTION_API_KEY is already saved'):
        await shell.loader.configure('notion')
    assert api_keys.load_keys()['NOTION_API_KEY'].get_secret_value() == 'ntn-theirs'
    assert load_codex_credentials(account='notion') is None


@pytest.mark.parametrize('entry', [CANCEL, typed('   ')])
async def test_cancelled_or_blank_entry_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: TextInputResult
) -> None:
    shell = Shell(tmp_path)
    await shell.loader.enable('notion')
    script(monkeypatch, lists=[pick('key')], texts=[entry])
    assert await shell.loader.configure('notion') == 'Notion settings unchanged.'
    assert load_codex_credentials(account='notion') is None and api_keys.load_keys() == {}


async def test_browser_sign_in_without_a_chosen_key(tmp_path: Path) -> None:
    shell = Shell(tmp_path, {'read_only': True})
    await shell.loader.enable('notion')
    capability = await shell.built()
    assert capability.auth is None and capability.read_only
    client = capability.client
    assert isinstance(client, Client)
    transport = client.transport
    assert isinstance(transport, StreamableHttpTransport)
    assert transport.url == notion.NOTION_MCP_URL and isinstance(transport.auth, OAuth)
    assert (await shell.built()).client is not client, 'each run reloads the sign-in, so logout applies to the next'


async def test_key_only_mode_warns_and_never_opens_a_browser(tmp_path: Path) -> None:
    shell = Shell(tmp_path, {'auth': 'key'})
    await shell.loader.enable('notion')
    assert 'Notion has no key selected, so runs fail. Choose one with /plugins configure notion.' in (
        shell.output.getvalue()
    )
    with pytest.raises(UserError, match='No Notion key is selected'):
        await shell.built()
    api_keys.save_key(name='NOTION_API_KEY', value='ntn-token')
    notion.select_key(KeyReference(name='NOTION_API_KEY'))
    await shell.loader.reload('notion')
    assert shell.output.getvalue().count('no key selected') == 1, 'a chosen key silences the warning'


async def test_browser_mode_ignores_a_chosen_key(tmp_path: Path) -> None:
    api_keys.save_key(name='NOTION_API_KEY', value='ntn-token')
    notion.select_key(KeyReference(name='NOTION_API_KEY'))
    shell = Shell(tmp_path, {'auth': 'oauth'})
    await shell.loader.enable('notion')
    capability = await shell.built()
    assert capability.auth is None and isinstance(capability.client, Client)


async def test_invalid_selection_fails_closed_and_the_menu_offers_a_new_choice(tmp_path: Path) -> None:
    save_codex_credentials(account='notion', value='{"token": "ntn-inline"}')
    shell = Shell(tmp_path)
    await shell.loader.enable('notion')
    with pytest.raises(UserError, match='selection is invalid'):
        await shell.built()
    menu = source()
    assert [menu.current(row) for row in menu.rows()] == ['(invalid; choose again)', 'auto', 'false', 'true']


def test_menu_validates_like_saving_would() -> None:
    menu = source(read_only=True)
    [_, auth, tools, _] = menu.rows()
    assert menu.problem(auth, 'browser') is not None
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


async def test_cancelling_configure_cancels_an_open_key_picker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shell = Shell(tmp_path)
    await shell.loader.enable('notion')
    opened = anyio.Event()
    finished: list[str] = []

    async def prompt_api_key(*, prompt: object, label: str, optional: bool = False) -> str | KeyReference | None:
        opened.set()
        try:
            await anyio.sleep_forever()
        finally:
            await anyio.sleep(0.2)  # a widget that takes a while to give the screen back
            finished.append(label)
        return None  # pragma: no cover -- unreachable; keeps the signature honest

    monkeypatch.setattr('pydantic_clai2.ui.menus.key_picker.prompt_api_key', prompt_api_key)
    script(monkeypatch, lists=[pick('key')])
    with anyio.fail_after(10):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(shell.loader.configure, 'notion')
            await opened.wait()
            tasks.cancel_scope.cancel()
    assert finished == [LABEL], 'configure returns only after the picker has closed'


async def test_logout_forgets_the_sign_in_and_the_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, vault: Vault
) -> None:
    api_keys.save_key(name='NOTION_API_KEY', value='ntn-token')
    notion.select_key(KeyReference(name='NOTION_API_KEY'))
    await notion.TOKENS.put('token', {'access_token': 'a'}, collection='mcp-oauth-token')
    shell = Shell(tmp_path)
    await shell.loader.enable('notion')
    [command] = [command for command in shell.commands if command.name == 'notion']
    assert list(command.complete([''])) == ['logout'] and list(command.complete(['logout', ''])) == []
    with pytest.raises(ValueError, match='/plugins configure notion'):
        await shell.run('/notion key')
    assert await shell.run('/notion logout') == (
        'Signed out of Notion and cleared the selected key. The key itself stays in /keys.'
    )
    assert stored_accounts() == {'api-keys'}, 'only the named key remains'
    assert (await shell.built()).client is not None


class Redraw:
    def replace_items(self, items: object) -> None:
        pass
