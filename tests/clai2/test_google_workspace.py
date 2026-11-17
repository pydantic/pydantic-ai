"""The `google_workspace` built-in: named-key auth, `/google_workspace`, settings, and its declaration."""

import asyncio
import io
from collections.abc import Awaitable, Coroutine, Sequence
from pathlib import Path

import pytest
from pydantic import JsonValue, ValidationError
from rich.console import Console
from termflow.tui import MenuItem
from termflow.tui.menu import MenuResult
from termflow.tui.textinput import TextInputResult
from typing_extensions import TypeIs

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.google_workspace import GoogleWorkspace
from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2.builtin_plugins import google_workspace
from pydantic_clai2.builtin_plugins.google_workspace import GoogleWorkspaceSettings
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import PluginSettings, Settings, api_keys
from pydantic_clai2.config.api_keys import KeyReference
from pydantic_clai2.config.credential_store import load_codex_credentials, save_codex_credentials
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import PluginHost, SessionStart, load_plugin
from pydantic_clai2.plugins.loader import PluginLoader
from pydantic_clai2.plugins.menus import Option
from pydantic_clai2.ui.menus import dialogs
from pydantic_clai2.ui.menus.plugin_menu import Configure, PluginMenu
from tests.clai2.dialog_script import Asked, Back, Confirm, Pick, ScriptedDialogs, Type
from tests.clai2.menu_script import Script, pick, typed
from tests.clai2.test_workspace_defaults import per_run


class Redraw:
    def replace_items(self, items: Sequence[MenuItem]) -> None:
        self.items = items


def host(settings: dict[str, JsonValue] | None = None) -> PluginHost[None]:
    return PluginHost(name='google_workspace', console=Console(file=io.StringIO()), settings=settings or {})


def context() -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage())


def loaded(plugin: PluginHost[None]) -> GoogleWorkspace[None]:
    [factory] = load_plugin(google_workspace.GoogleWorkspacePlugin, plugin).capabilities
    capability = per_run(factory)(context())
    assert is_workspace(capability)
    assert capability.defer_loading
    assert capability.id == 'google_workspace', 'stable whatever the products, so a load survives an edit'
    return capability


def is_workspace(capability: object) -> TypeIs[GoogleWorkspace[None]]:
    return isinstance(capability, GoogleWorkspace)


def run_token(capability: GoogleWorkspace[None]) -> str | None:
    assert callable(capability.auth)
    token = capability.auth(context())
    assert not isinstance(token, Awaitable)
    return token


async def configure(
    monkeypatch: pytest.MonkeyPatch,
    plugin: PluginHost[None],
    *lists: MenuResult,
    choices: Sequence[MenuResult] = (),
    texts: Sequence[TextInputResult] = (),
) -> str:
    """Open the declared menu in a scripted terminal, through the plugin's `configure`."""
    script = Script(lists=[*lists, MenuResult(cancelled=True)], choices=list(choices), texts=list(texts))
    monkeypatch.setattr(dialogs, 'RUNNERS', script.runners)
    plugin.console = Console(file=io.StringIO(), force_terminal=True)
    return await load_plugin(google_workspace.GoogleWorkspacePlugin, plugin).plugin.configure()


def source(plugin: PluginHost[None]) -> google_workspace.SettingsSource[None]:
    menu = load_plugin(google_workspace.GoogleWorkspacePlugin, plugin).plugin.settings_menu()
    assert menu is not None and isinstance(menu.source, google_workspace.SettingsSource)
    assert set(menu.actions) == {'token', 'services'} and not menu.searchable
    return google_workspace.SettingsSource(plugin)


TOKEN_DIALOG = Asked(
    kind='text',
    title='Google OAuth access token',
    label='Token',
    placeholder='Paste the token; it is saved in /keys as GOOGLE_ACCESS_TOKEN',
    secret=True,
)


def products(*chosen: str) -> tuple[Option, ...]:
    return (
        *(
            Option(value=name, label=f'[{"x" if name in chosen else " "}] {name}')
            for name in ('gmail', 'drive', 'docs', 'sheets', 'slides', 'calendar', 'chat', 'people')
        ),
        Option(value='done', label='Done'),
    )


async def test_the_conventional_key_is_resolved_on_every_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('GOOGLE_ACCESS_TOKEN', 'env-token')
    api_keys.save_key(name='GOOGLE_ACCESS_TOKEN', value='saved-token')
    plugin = host()
    await activate_quietly(plugin)
    capability = loaded(plugin)
    assert capability.services == ('gmail', 'calendar', 'drive')
    assert capability.read_only is True
    assert run_token(capability) == 'saved-token'
    api_keys.save_key(name='GOOGLE_ACCESS_TOKEN', value='refreshed-token')
    assert run_token(capability) == 'refreshed-token'
    api_keys.delete_key(name='GOOGLE_ACCESS_TOKEN')
    with pytest.raises(UserError, match='GOOGLE_ACCESS_TOKEN') as error:
        run_token(capability)
    assert 'env-token' not in str(error.value)


async def activate_quietly(plugin: PluginHost[None]) -> str:
    """Load the plugin and start a session, returning what it printed."""
    output = io.StringIO()
    plugin.console = Console(file=output)
    loaded = load_plugin(google_workspace.GoogleWorkspacePlugin, plugin)
    await loaded.dispatch(SessionStart(agent=Agent(TestModel()), settings=Settings()))
    return output.getvalue()


async def test_activate_without_a_key_warns_and_fails_closed_on_use() -> None:
    plugin = host()
    warning = await activate_quietly(plugin)
    assert 'Run /google_workspace' in warning
    commands = load_plugin(google_workspace.GoogleWorkspacePlugin, plugin).commands
    assert [command.name for command in commands] == ['google_workspace']
    with pytest.raises(UserError) as error:
        run_token(loaded(plugin))
    assert str(error.value) == google_workspace.missing('GOOGLE_ACCESS_TOKEN')


async def test_settings_choose_services_and_writable_tools() -> None:
    plugin = host({'services': ['gmail'], 'read_only': False})
    await activate_quietly(plugin)
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
        google_workspace.GoogleWorkspacePlugin.from_host(host(settings))


async def test_menu_saves_an_entered_token_under_the_conventional_label(monkeypatch: pytest.MonkeyPatch) -> None:
    """In the terminal, the key row asks for the token masked, straight away when /keys is empty."""
    plugin = host()
    await activate_quietly(plugin)
    result = await configure(monkeypatch, plugin, pick('token'), texts=[typed('entered-token')])
    assert result == 'Google Workspace uses the saved key GOOGLE_ACCESS_TOKEN from the next turn.'
    assert api_keys.load_keys()['GOOGLE_ACCESS_TOKEN'].get_secret_value() == 'entered-token'
    stored = load_codex_credentials(account='google-workspace')
    assert stored is not None and 'entered-token' not in stored
    assert plugin.settings(GoogleWorkspaceSettings) == GoogleWorkspaceSettings()
    assert run_token(loaded(plugin)) == 'entered-token'


async def test_the_key_action_asks_through_dialogs() -> None:
    plugin = host()
    script = ScriptedDialogs([Type(text='entered-token')])
    assert await google_workspace.pick_key(script) == [
        'Google Workspace uses the saved key GOOGLE_ACCESS_TOKEN from the next turn.'
    ]
    assert script.asked == [TOKEN_DIALOG]
    assert run_token(loaded(plugin)) == 'entered-token'


async def test_menu_repicks_a_shared_saved_key_and_resets_to_the_label() -> None:
    api_keys.save_key(name='WORK_GOOGLE', value='shared-token')
    plugin = host()
    await activate_quietly(plugin)
    settings = source(plugin)
    [token_row, *_] = settings.rows()
    assert settings.current(token_row) == 'GOOGLE_ACCESS_TOKEN'
    script = ScriptedDialogs([Pick(value='WORK_GOOGLE')])
    assert await google_workspace.pick_key(script) == [
        'Google Workspace uses the saved key WORK_GOOGLE from the next turn.'
    ]
    [asked] = script.asked
    assert (asked.kind, asked.title) == ('choose', 'Google OAuth access token')
    assert [option.value for option in asked.options] == ['WORK_GOOGLE', 'enter']
    assert settings.current(token_row) == 'WORK_GOOGLE'
    assert run_token(loaded(plugin)) == 'shared-token'
    assert 'GOOGLE_ACCESS_TOKEN' not in api_keys.load_keys()
    with pytest.raises(ValueError, match='used by google-workspace'):
        api_keys.rename_key(name='WORK_GOOGLE', new_name='OTHER')
    assert settings.reset(token_row) == 'Google Workspace uses the saved key GOOGLE_ACCESS_TOKEN again.'
    assert load_codex_credentials(account='google-workspace') is None


async def test_replacing_the_saved_token_asks_first() -> None:
    api_keys.save_key(name='GOOGLE_ACCESS_TOKEN', value='old-token')
    script = ScriptedDialogs([Pick(value='enter'), Type(text='new-token'), Confirm(yes=False)])
    assert await google_workspace.pick_key(script) == ['Google Workspace key unchanged.']
    assert [asked.kind for asked in script.asked] == ['choose', 'text', 'confirm']
    assert script.asked[2].title == 'GOOGLE_ACCESS_TOKEN is already in /keys'
    assert api_keys.load_keys()['GOOGLE_ACCESS_TOKEN'].get_secret_value() == 'old-token'
    assert load_codex_credentials(account='google-workspace') is None


@pytest.mark.parametrize('answer', [Back(), Type(text=' ')])
async def test_key_cancel_and_empty_entry_leave_no_key(answer: Back | Type) -> None:
    plugin = host()
    await activate_quietly(plugin)
    script = ScriptedDialogs([answer])
    assert await google_workspace.pick_key(script) == ['Google Workspace key unchanged.']
    assert script.asked == [TOKEN_DIALOG]
    assert api_keys.load_keys() == {}
    assert load_codex_credentials(account='google-workspace') is None
    with pytest.raises(ValueError, match='Usage'):
        await load_plugin(google_workspace.GoogleWorkspacePlugin, plugin).commands.execute_async(
            '/google_workspace secret'
        )


class DeletingDialogs(ScriptedDialogs):
    """Delete the picked key from /keys before the choice is saved, as another session might."""

    async def choose(
        self, *, title: str, options: Sequence[Option], current: str | None = None, description: str = ''
    ) -> str | None:
        picked = await super().choose(title=title, options=options, current=current, description=description)
        assert picked is not None
        api_keys.delete_key(name=picked)
        return picked


async def test_a_key_deleted_while_picking_is_reported() -> None:
    api_keys.save_key(name='WORK_GOOGLE', value='shared-token')
    [message] = await google_workspace.pick_key(DeletingDialogs([Pick(value='WORK_GOOGLE')]))
    assert 'through /google_workspace' in message
    assert load_codex_credentials(account='google-workspace') is None


async def test_menu_edits_options_and_saves_each_one_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SettingsStore(tmp_path / 'settings.db')
    plugins = loader(store)
    await plugins.enable('google_workspace')
    [entry] = [entry for entry in plugins.entries() if entry.name == 'google_workspace']
    assert entry.loaded is not None
    plugin = entry.loaded.host

    result = await configure(
        monkeypatch,
        plugin,
        pick('services'),
        pick('read_only'),
        pick('include_instructions'),
        choices=[
            pick('docs'),
            pick('gmail'),
            pick('done'),
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

    settings = source(plugin)
    rows = {row.key: row for row in settings.rows()}
    assert settings.current(rows['services']) == 'drive, docs, calendar'
    assert settings.problem(rows['read_only'], 'maybe') == 'Choose true or false.'
    assert settings.problem(rows['read_only'], 'true') is None
    settings.reset(rows['read_only'])
    settings.reset(rows['include_instructions'])
    settings.reset(rows['services'])
    [declaration] = [plugin for plugin in store.plugins() if plugin.id == 'google_workspace']
    assert declaration.settings == GoogleWorkspaceSettings().model_dump(mode='json')
    await plugins.close('exit')


async def test_the_products_action_toggles_a_checklist() -> None:
    plugin = host()
    script = ScriptedDialogs([Pick(value='docs'), Back()])
    assert await source(plugin).edit_services(script) == ['Products: gmail, drive, docs, calendar.']
    assert [asked.options for asked in script.asked] == [
        products('gmail', 'drive', 'calendar'),
        products('gmail', 'drive', 'docs', 'calendar'),
    ]
    assert script.asked[0].title == 'Google Workspace products'
    assert plugin.settings(GoogleWorkspaceSettings).services == ['gmail', 'drive', 'docs', 'calendar']


async def test_the_last_product_cannot_be_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = host({'services': ['gmail']})
    await activate_quietly(plugin)
    script = ScriptedDialogs([Pick(value='gmail'), Pick(value='gmail'), Pick(value='done')])
    assert await source(plugin).edit_services(script) == ['Google Workspace needs at least one product.']
    assert [asked.description for asked in script.asked][1:] == ['Google Workspace needs at least one product.'] * 2
    assert plugin.settings(GoogleWorkspaceSettings).services == ['gmail']
    assert await configure(monkeypatch, plugin) == 'Google Workspace: nothing changed.'
    settings = source(plugin)
    [_, services, *_] = settings.rows()
    assert settings.reset(services) == 'Reset Products.'
    assert plugin.settings(GoogleWorkspaceSettings) == GoogleWorkspaceSettings()


def test_a_key_deleted_before_saving_the_choice_points_to_google_workspace() -> None:
    with pytest.raises(UserError, match='through /google_workspace'):
        api_keys.save_key_connection(account='google-workspace', token=KeyReference(name='GONE'), value='{}')
    assert load_codex_credentials(account='google-workspace') is None


async def test_an_invalid_saved_choice_fails_closed() -> None:
    save_codex_credentials(account='google-workspace', value='{"token": ["inline-secret"]}')
    plugin = host()
    assert 'Run /google_workspace again' in await activate_quietly(plugin)
    with pytest.raises(UserError) as error:
        run_token(loaded(plugin))
    assert 'inline-secret' not in str(error.value)
    settings = source(plugin)
    assert settings.current(settings.rows()[0]) == '(invalid; Enter to choose again)'
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
    assert declaration.factory == 'pydantic_clai2.builtin_plugins.google_workspace'
    assert not declaration.enabled
    assert declaration.settings == {}
    plugins = loader(SettingsStore(tmp_path / 'settings.db'), builtin=(declaration,))

    def apply(action: Coroutine[object, object, object]) -> None:
        asyncio.run(action)

    asyncio.run(plugins.load_all())
    assert plugins.capabilities() == []
    menu = PluginMenu(plugins, apply=apply)
    [item, _save_and_close] = menu.items()
    opened = menu.toggle(Redraw(), item)
    assert opened is not None and opened.item is not None
    assert opened.item.value == Configure('google_workspace')
    assert menu.notice is None
    assert 'on' in menu.items()[0].description and 'press c to configure' in menu.details(item)
    [factory] = plugins.capabilities()
    assert isinstance(per_run(factory)(context()), GoogleWorkspace)
    redraw = Redraw()
    assert menu.toggle(redraw, item) is None
    assert redraw.items[0].label.startswith('○ google_workspace')
    assert plugins.capabilities() == []
    asyncio.run(plugins.close('exit'))


async def test_plugins_configure_opens_the_google_workspace_menu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`/plugins configure google_workspace` and `/google_workspace` open the same declared menu."""
    script = Script(lists=[MenuResult(cancelled=True)] * 2, choices=[], texts=[])
    monkeypatch.setattr(dialogs, 'RUNNERS', script.runners)
    [declaration] = [plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'google_workspace']
    store = SettingsStore(tmp_path / 'settings.db')
    plugins = PluginLoader[None](
        store=store,
        console=Console(file=io.StringIO(), force_terminal=True),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=(declaration.model_copy(update={'enabled': True}),),
    )
    await plugins.load_all()
    try:
        assert plugins.configurable('google_workspace')
        assert plugins.settings_menu('google_workspace') is not None
        assert await plugins.command(['configure', 'google_workspace']) == 'Google Workspace: nothing changed.'
        [entry] = plugins.entries()
        assert entry.loaded is not None
        assert await entry.loaded.commands.execute_async('/google_workspace') == 'Google Workspace: nothing changed.'
        assert script.opened == ['list', 'list']
    finally:
        await plugins.close('exit')


def test_the_former_catalog_entry_saved_by_the_menu_loads_the_builtin(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'settings.db')
    former = PluginSettings(id='google_workspace', factory='pydantic_ai_harness.google_workspace:GoogleWorkspace')
    store.save_plugin(former)
    [entry] = [entry for entry in loader(store).entries() if entry.name == 'google_workspace']
    assert entry.declaration.factory == 'pydantic_clai2.builtin_plugins.google_workspace'
    assert entry.declaration.enabled
    assert entry.builtin

    store.save_plugin(former.model_copy(update={'enabled': False}))
    [entry] = [entry for entry in loader(store).entries() if entry.name == 'google_workspace']
    assert entry.declaration.factory == 'pydantic_clai2.builtin_plugins.google_workspace'
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
