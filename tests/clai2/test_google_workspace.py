"""The `google_workspace` built-in: named-key auth, `/google_workspace`, settings, and its declaration."""

import asyncio
import io
from collections.abc import Coroutine, Sequence
from pathlib import Path

import keyring
import pytest
from pydantic import JsonValue, ValidationError
from rich.console import Console
from termflow.tui import MenuItem
from termflow.tui.menu import MenuResult
from typing_extensions import TypeIs

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.google_workspace import GoogleWorkspace
from pydantic_clai2 import DEFAULT_PLUGINS, api_keys, google_workspace
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.credential_store import load_codex_credentials, save_codex_credentials
from pydantic_clai2.google_workspace import GoogleWorkspaceSettings
from pydantic_clai2.plugin_loader import PluginLoader
from pydantic_clai2.plugin_menu import PluginMenu
from pydantic_clai2.plugins import PluginHost, SessionStart
from pydantic_clai2.settings_store import SettingsStore
from tests.clai2.menu_script import Script, pick


class Prompt:
    def __init__(self, *, values: list[str | BaseException]) -> None:
        self.values = iter(values)
        self.labels: list[tuple[str, bool]] = []

    async def prompt_async(self, label: str, *, is_password: bool = False) -> str:
        self.labels.append((label, is_password))
        value = next(self.values)
        if isinstance(value, BaseException):
            raise value
        return value


class Redraw:
    def replace_items(self, items: Sequence[MenuItem]) -> None:
        self.items = items


def host(settings: dict[str, JsonValue] | None = None) -> PluginHost[None]:
    return PluginHost(name='google_workspace', console=Console(file=io.StringIO()), settings=settings or {})


def deletable_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    entries: dict[tuple[str, str], str] = {}

    def get(service: str, account: str) -> str | None:
        return entries.get((service, account))

    def set_value(service: str, account: str, value: str) -> None:
        entries[service, account] = value

    def delete(service: str, account: str) -> None:
        del entries[service, account]

    monkeypatch.setattr(keyring, 'get_password', get)
    monkeypatch.setattr(keyring, 'set_password', set_value)
    monkeypatch.setattr(keyring, 'delete_password', delete)


def context() -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage())


def loaded(plugin: PluginHost[None]) -> GoogleWorkspace[None]:
    [factory] = plugin.capabilities
    assert not isinstance(factory, AbstractCapability)
    capability = factory(context())
    assert is_workspace(capability)
    return capability


def is_workspace(capability: object) -> TypeIs[GoogleWorkspace[None]]:
    return isinstance(capability, GoogleWorkspace)


def run_token(capability: GoogleWorkspace[None]) -> str | None:
    assert callable(capability.auth)
    return capability.auth(context())


async def configure(plugin: PluginHost[None], *lists: MenuResult, choices: Sequence[MenuResult] = ()) -> str:
    script = Script(lists=[*lists, MenuResult(cancelled=True)], choices=list(choices), texts=[])
    return await google_workspace.configure(plugin, [], runners=script.runners)


def test_the_conventional_key_is_resolved_on_every_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('GOOGLE_ACCESS_TOKEN', 'env-token')
    api_keys.save_key(name='GOOGLE_ACCESS_TOKEN', value='saved-token')
    plugin = host()
    activate_quietly(plugin)
    capability = loaded(plugin)
    assert capability.services == ('gmail', 'calendar', 'drive')
    assert capability.read_only is True
    assert capability.id == 'google-workspace-calendar-drive-gmail'
    assert run_token(capability) == 'saved-token'
    api_keys.save_key(name='GOOGLE_ACCESS_TOKEN', value='refreshed-token')
    assert run_token(capability) == 'refreshed-token'
    api_keys.delete_key(name='GOOGLE_ACCESS_TOKEN')
    with pytest.raises(UserError, match='GOOGLE_ACCESS_TOKEN') as error:
        run_token(capability)
    assert 'env-token' not in str(error.value)


def activate_quietly(plugin: PluginHost[None]) -> str:
    output = io.StringIO()
    plugin.console = Console(file=output)
    google_workspace.activate(plugin)
    return output.getvalue()


def test_activate_without_a_key_warns_and_fails_closed_on_use() -> None:
    plugin = host()
    warning = activate_quietly(plugin)
    assert 'Run /google_workspace' in warning
    assert [command.name for command in plugin.commands] == ['google_workspace']
    with pytest.raises(UserError) as error:
        run_token(loaded(plugin))
    assert str(error.value) == google_workspace.missing('GOOGLE_ACCESS_TOKEN')


def test_settings_choose_services_and_writable_tools() -> None:
    plugin = host({'services': ['gmail'], 'read_only': False})
    activate_quietly(plugin)
    capability = loaded(plugin)
    assert capability.services == ('gmail',)
    assert capability.read_only is False


@pytest.mark.parametrize(
    'settings',
    [
        {'services': []},
        {'services': ['maps']},
        {'auth': 'token'},
        {'token': 'secret'},
        {'token': {'name': 'GOOGLE_ACCESS_TOKEN'}},
        {'read_only': 'yes'},
    ],
)
def test_settings_reject_bad_values_and_credentials(settings: dict[str, JsonValue]) -> None:
    with pytest.raises(ValidationError):
        google_workspace.activate(host(settings))


async def test_menu_saves_an_entered_token_under_the_conventional_label(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = host()
    activate_quietly(plugin)
    prompt = Prompt(values=['entered-token'])
    monkeypatch.setattr(google_workspace, 'PromptSession', lambda: prompt)
    result = await configure(plugin, pick('token'))
    assert result == 'Google Workspace uses the saved key GOOGLE_ACCESS_TOKEN from the next turn.'
    assert prompt.labels == [('Google OAuth access token (saved in /keys as GOOGLE_ACCESS_TOKEN): ', True)]
    assert api_keys.load_keys()['GOOGLE_ACCESS_TOKEN'].get_secret_value() == 'entered-token'
    stored = load_codex_credentials(account='google-workspace')
    assert stored is not None and 'entered-token' not in stored
    assert plugin.settings(GoogleWorkspaceSettings) == GoogleWorkspaceSettings()
    assert run_token(loaded(plugin)) == 'entered-token'


async def test_menu_repicks_a_shared_saved_key_and_resets_to_the_label(monkeypatch: pytest.MonkeyPatch) -> None:
    deletable_keyring(monkeypatch)
    api_keys.save_key(name='WORK_GOOGLE', value='shared-token')
    plugin = host()
    activate_quietly(plugin)
    pressed = iter(['enter'])
    monkeypatch.setattr(api_keys, 'menu_key', lambda: next(pressed))
    monkeypatch.setattr(google_workspace, 'PromptSession', lambda: Prompt(values=[]))
    source = google_workspace.SettingsSource(plugin)
    [token_row, *_] = source.rows()
    assert source.current(token_row) == 'GOOGLE_ACCESS_TOKEN'
    assert (
        await configure(plugin, pick('token')) == 'Google Workspace uses the saved key WORK_GOOGLE from the next turn.'
    )
    assert source.current(token_row) == 'WORK_GOOGLE'
    assert run_token(loaded(plugin)) == 'shared-token'
    assert 'GOOGLE_ACCESS_TOKEN' not in api_keys.load_keys()
    with pytest.raises(ValueError, match='used by google-workspace'):
        api_keys.rename_key(name='WORK_GOOGLE', new_name='OTHER')
    assert source.reset(token_row) == 'Google Workspace uses the saved key GOOGLE_ACCESS_TOKEN again.'
    assert load_codex_credentials(account='google-workspace') is None


@pytest.mark.parametrize('value', [EOFError(), ' '])
async def test_key_cancel_and_empty_entry_leave_no_key(
    monkeypatch: pytest.MonkeyPatch, value: str | BaseException
) -> None:
    plugin = host()
    activate_quietly(plugin)
    monkeypatch.setattr(google_workspace, 'PromptSession', lambda: Prompt(values=[value]))
    if isinstance(value, str):
        with pytest.raises(ValueError, match='required'):
            await configure(plugin, pick('token'))
    else:
        assert await configure(plugin, pick('token')) == 'Google Workspace key unchanged.'
    assert api_keys.load_keys() == {}
    assert load_codex_credentials(account='google-workspace') is None
    with pytest.raises(ValueError, match='Usage'):
        await plugin.commands.execute_async('/google_workspace secret')


async def test_menu_edits_options_and_saves_each_one_immediately(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'settings.db')
    plugins = loader(store)
    await plugins.enable('google_workspace')
    [entry] = [entry for entry in plugins.entries() if entry.name == 'google_workspace']
    assert entry.host is not None
    plugin = entry.host

    result = await configure(
        plugin,
        pick('services'),
        pick('read_only'),
        pick('include_instructions'),
        choices=[
            pick('docs'),
            pick('gmail'),
            MenuResult(cancelled=True),
            pick('false'),
            pick('false'),
        ],
    )
    assert result.splitlines() == [
        'Products: gmail, drive, docs, calendar.',
        'Products: drive, docs, calendar.',
        'Saved Read-only tools: false.',
        'Saved Server instructions: false.',
    ]
    saved = {
        'services': ['drive', 'docs', 'calendar'],
        'read_only': False,
        'include_instructions': False,
    }
    [declaration] = [plugin for plugin in store.plugins() if plugin.id == 'google_workspace']
    assert declaration.settings == saved
    assert declaration.enabled
    workspace = loaded(plugin)
    assert workspace.services == ('drive', 'docs', 'calendar')
    assert (workspace.read_only, workspace.include_instructions) == (False, False)

    source = google_workspace.SettingsSource(plugin)
    rows = {row.key: row for row in source.rows()}
    assert source.current(rows['services']) == 'drive, docs, calendar'
    assert source.problem(rows['read_only'], 'maybe') == 'Choose true or false.'
    assert source.problem(rows['read_only'], 'true') is None
    source.reset(rows['read_only'])
    source.reset(rows['include_instructions'])
    source.reset(rows['services'])
    [declaration] = [plugin for plugin in store.plugins() if plugin.id == 'google_workspace']
    assert declaration.settings == {}
    [entry] = [entry for entry in plugins.entries() if entry.name == 'google_workspace']
    assert entry.builtin
    await plugins.close('exit')


async def test_the_last_product_cannot_be_removed() -> None:
    plugin = host({'services': ['gmail']})
    activate_quietly(plugin)
    result = await configure(plugin, pick('services'), choices=[pick('gmail'), pick('not-a-product')])
    assert result == 'Google Workspace needs at least one product.'
    assert plugin.settings(GoogleWorkspaceSettings).services == ['gmail']
    assert await configure(plugin) == 'No changes.'
    source = google_workspace.SettingsSource(plugin)
    [_, services, *_] = source.rows()
    assert source.reset(services) == 'Reset Products.'
    assert plugin.settings(GoogleWorkspaceSettings) == GoogleWorkspaceSettings()


def test_an_invalid_saved_choice_fails_closed() -> None:
    save_codex_credentials(account='google-workspace', value='{"token": ["inline-secret"]}')
    plugin = host()
    assert 'Run /google_workspace again' in activate_quietly(plugin)
    with pytest.raises(UserError) as error:
        run_token(loaded(plugin))
    assert 'inline-secret' not in str(error.value)
    source = google_workspace.SettingsSource(plugin)
    assert source.current(source.rows()[0]) == '(invalid; Enter to choose again)'
    api_keys.save_key(name='ANY', value='unrelated')
    with pytest.raises(UserError, match='through /google_workspace'):
        api_keys.rename_key(name='ANY', new_name='OTHER')


def loader(store: SettingsStore, builtin: Sequence[PluginSettings] = DEFAULT_PLUGINS) -> PluginLoader[None]:
    return PluginLoader(
        store=store,
        console=Console(file=io.StringIO()),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=builtin,
    )


def test_declared_as_a_disabled_builtin_that_enables_from_the_menu(tmp_path: Path) -> None:
    [declaration] = [plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'google_workspace']
    assert declaration.factory == 'pydantic_clai2.google_workspace'
    assert not declaration.enabled
    assert declaration.settings == {}
    plugins = loader(SettingsStore(tmp_path / 'settings.db'), builtin=(declaration,))

    def apply(action: Coroutine[object, object, object]) -> None:
        asyncio.run(action)

    asyncio.run(plugins.load_all())
    assert plugins.capabilities() == []
    menu = PluginMenu(plugins, apply=apply)
    [item] = menu.items()
    menu.toggle(Redraw(), item)
    assert menu.notice is None
    assert 'enabled, loaded' in menu.details(item)
    [factory] = plugins.capabilities()
    assert not isinstance(factory, AbstractCapability)
    assert isinstance(factory(context()), GoogleWorkspace)
    asyncio.run(plugins.close('exit'))


def test_the_former_catalog_entry_saved_by_the_menu_loads_the_builtin(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'settings.db')
    former = PluginSettings(id='google_workspace', factory='pydantic_ai_harness.google_workspace:GoogleWorkspace')
    store.save_plugin(former)
    [entry] = [entry for entry in loader(store).entries() if entry.name == 'google_workspace']
    assert entry.declaration.factory == 'pydantic_clai2.google_workspace'
    assert entry.declaration.enabled
    assert entry.builtin

    store.save_plugin(former.model_copy(update={'enabled': False}))
    [entry] = [entry for entry in loader(store).entries() if entry.name == 'google_workspace']
    assert entry.declaration.factory == 'pydantic_clai2.google_workspace'
    assert not entry.declaration.enabled


def test_a_former_declaration_with_its_own_settings_is_kept(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'settings.db')
    custom = PluginSettings(
        id='google_workspace',
        factory='pydantic_ai_harness.google_workspace:GoogleWorkspace',
        settings={'services': ['gmail']},
    )
    store.save_plugin(custom)
    [entry] = [entry for entry in loader(store).entries() if entry.name == 'google_workspace']
    assert entry.declaration == custom
    assert not entry.builtin
