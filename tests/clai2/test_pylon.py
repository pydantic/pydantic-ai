"""The built-in `pylon` plugin: its declaration, settings menu, `/keys` reference, and connection."""

import io
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.client.auth import OAuth
from fastmcp.client.transports import StreamableHttpTransport
from pydantic import JsonValue, ValidationError
from rich.console import Console
from termflow.tui import MenuItem
from termflow.tui.menu import MenuResult

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.pylon import Pylon
from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2.builtin_plugins import pylon
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import api_keys
from pydantic_clai2.config.credential_store import load_codex_credentials, save_codex_credentials
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.mcp import OAUTH_TIMEOUT
from pydantic_clai2.plugins import LoadedPlugin, PluginHost, SessionStart, load_plugin
from pydantic_clai2.plugins.loader import PluginLoader
from pydantic_clai2.ui.menus.field_menu import FieldMenu
from tests.clai2.menu_script import Script, pick

pytestmark = pytest.mark.anyio

CLOSE = MenuResult(cancelled=True)
CTX = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())


class Prompt:
    def __init__(self, *values: str | BaseException) -> None:
        self.values = iter(values)
        self.labels: list[tuple[str, bool]] = []

    async def prompt_async(self, label: str, *, is_password: bool = False) -> str:
        self.labels.append((label, is_password))
        value = next(self.values)
        if isinstance(value, BaseException):
            raise value
        return value


def make_plugin(settings: dict[str, JsonValue] | None = None) -> LoadedPlugin[None]:
    host: PluginHost[None] = PluginHost(name='pylon', console=Console(file=io.StringIO()), settings=settings or {})
    return load_plugin(pylon.PylonPlugin, host)


def built(loaded: LoadedPlugin[None]) -> Pylon[None]:
    """The `Pylon` the plugin builds for the next run."""
    [capability] = loaded.capabilities
    assert not isinstance(capability, AbstractCapability), 'rebuilt per run from the current settings'
    result = capability(CTX)
    assert isinstance(result, Pylon)
    return result


def config(settings: pylon.PylonSettings | None = None) -> pylon.PylonConfig:
    saved: list[pylon.PylonSettings] = []
    return pylon.PylonConfig(settings or pylon.PylonSettings(), saved.append)


def use_prompt(monkeypatch: pytest.MonkeyPatch, prompt: Prompt) -> None:
    monkeypatch.setattr(pylon, 'PromptSession', lambda: prompt)


def use_key(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    async def choose(**_: object) -> api_keys.KeyReference:
        return api_keys.KeyReference(name=name)

    monkeypatch.setattr(pylon, 'prompt_api_key', choose)


def reset(key: str) -> MenuResult:
    return FieldMenu(config()).reset_marker(None, MenuItem(key, value=key))


def loader(store: SettingsStore) -> PluginLoader[None]:
    [declaration] = [plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'pylon']
    return PluginLoader(
        store=store,
        console=Console(file=io.StringIO()),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=(declaration,),
    )


async def command(loaded: LoadedPlugin[None], *args: str) -> str:
    [registered] = loaded.commands
    result = registered.handler(list(args))
    assert not isinstance(result, str)
    return await result


def pylon_tools(loaded: LoadedPlugin[None]) -> list[str]:
    """Run once and report the Pylon tools the run could see; `TestModel` calls none of them."""
    model = TestModel(call_tools=[])
    Agent(model, deps_type=type(None), capabilities=loaded.capabilities).run_sync('hi')
    assert model.last_model_request_parameters is not None
    return [tool.name for tool in model.last_model_request_parameters.function_tools]


class TestDeclarationAndConnection:
    def test_declared_as_disabled_clai_built_in(self) -> None:
        [declaration] = [plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'pylon']
        assert declaration.factory == 'pydantic_clai2.builtin_plugins.pylon'
        assert not declaration.enabled
        assert declaration.settings == {}, 'nothing secret, or otherwise, is declared'

    def test_no_key_chosen_means_no_pylon_tools(self) -> None:
        assert pylon_tools(make_plugin()) == []

    def test_key_mode_passes_the_capability_options(self) -> None:
        api_keys.save_key(name='SHARED_PYLON', value='secret')
        save_codex_credentials(account='pylon', value='{"token": {"name": "SHARED_PYLON"}}')
        capability = built(make_plugin({'read_only': True, 'include_instructions': False}))
        assert capability.client is None and capability.read_only and not capability.include_instructions

    async def test_shared_key_is_referenced_and_resolved_each_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        api_keys.save_key(name='SHARED_PYLON', value='first')
        use_key(monkeypatch, 'SHARED_PYLON')
        assert await pylon.choose_key() == 'Pylon connects with SHARED_PYLON from /keys.'
        loaded = make_plugin()
        assert built(loaded).auth == 'first'
        api_keys.save_key(name='SHARED_PYLON', value='replaced')
        assert built(loaded).auth == 'replaced', 'replacing the key in /keys reaches the next run without a reload'
        with pytest.raises(ValueError, match='used by pylon'):
            api_keys.rename_key(name='SHARED_PYLON', new_name='OTHER')

    def test_deleted_key_fails_the_run_closed(self) -> None:
        api_keys.save_key(name='PYLON_ACCESS_TOKEN', value='secret')
        save_codex_credentials(account='pylon', value='{"token": {"name": "PYLON_ACCESS_TOKEN"}}')
        api_keys.delete_key(name='PYLON_ACCESS_TOKEN')
        with pytest.raises(UserError, match='PYLON_ACCESS_TOKEN is missing'):
            pylon_tools(make_plugin())

    def test_invalid_saved_reference_fails_closed(self) -> None:
        save_codex_credentials(account='pylon', value='{"token": "inline-secret"}')
        with pytest.raises(UserError, match='/pylon key'):
            pylon.saved_key()
        assert config().current(config().rows()[1]) == '(invalid; choose again)'

    def test_browser_sign_in(self) -> None:
        capability = built(make_plugin({'auth': 'browser'}))
        client = capability.client
        assert isinstance(client, Client)
        transport = client.transport
        assert isinstance(transport, StreamableHttpTransport)
        assert transport.url == pylon.PYLON_MCP_URL == 'https://mcp.usepylon.com'
        assert isinstance(transport.auth, OAuth)
        assert client._init_timeout == OAUTH_TIMEOUT, 'the browser gets as long as `/mcp` gives it'  # pyright: ignore[reportPrivateUsage]

    @pytest.mark.parametrize('settings', [{'token': 'pylon-token'}, {'auth': 'env'}])
    def test_settings_reject_secrets_and_unknown_modes(self, settings: dict[str, JsonValue]) -> None:
        with pytest.raises(ValidationError):
            make_plugin(settings)


class TestChooseKey:
    async def test_new_token_is_saved_in_keys_and_only_its_name_is_referenced(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompt = Prompt(' pylon-secret ')
        use_prompt(monkeypatch, prompt)
        assert await pylon.choose_key() == 'Pylon connects with PYLON_ACCESS_TOKEN from /keys.'
        assert prompt.labels[0][1], 'the value is entered masked'
        assert api_keys.load_keys()['PYLON_ACCESS_TOKEN'].get_secret_value() == 'pylon-secret'
        raw = load_codex_credentials(account='pylon')
        assert raw is not None and 'pylon-secret' not in raw
        assert pylon.saved_key() == api_keys.KeyReference(name='PYLON_ACCESS_TOKEN')

    @pytest.mark.parametrize('answer', ['n', EOFError()])
    async def test_existing_label_is_not_replaced_without_consent(
        self, monkeypatch: pytest.MonkeyPatch, answer: str | BaseException
    ) -> None:
        api_keys.save_key(name='PYLON_ACCESS_TOKEN', value='keep')

        async def enter(**_: object) -> str:
            return 'new'

        monkeypatch.setattr(pylon, 'prompt_api_key', enter)
        use_prompt(monkeypatch, Prompt(answer))
        assert await pylon.choose_key() == 'Pylon key unchanged.'
        assert api_keys.load_keys()['PYLON_ACCESS_TOKEN'].get_secret_value() == 'keep'
        assert pylon.saved_key() is None

    async def test_replacing_the_label_with_consent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        api_keys.save_key(name='PYLON_ACCESS_TOKEN', value='old')

        async def enter(**_: object) -> str:
            return 'new'

        monkeypatch.setattr(pylon, 'prompt_api_key', enter)
        use_prompt(monkeypatch, Prompt('y'))
        assert await pylon.choose_key() == 'Pylon connects with PYLON_ACCESS_TOKEN from /keys.'
        assert api_keys.load_keys()['PYLON_ACCESS_TOKEN'].get_secret_value() == 'new'

    async def test_cancel_and_blank_entries_save_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        use_prompt(monkeypatch, Prompt(KeyboardInterrupt(), '  '))
        assert await pylon.choose_key() == 'Pylon key unchanged.'
        with pytest.raises(ValueError, match='access token is required'):
            await pylon.choose_key()
        assert pylon.saved_key() is None and api_keys.load_keys() == {}


class TestSettingsMenu:
    def test_rows_follow_the_sign_in_mode(self) -> None:
        assert [row.key for row in config().rows()] == ['auth', 'key', 'read_only', 'include_instructions']
        browser = config(pylon.PylonSettings(auth='browser'))
        assert [row.key for row in browser.rows()] == ['auth', 'read_only', 'include_instructions']
        [auth, key, read_only, _] = config().rows()
        assert auth.choices == ('key', 'browser') and not auth.allow_custom
        assert config().current(key) == pylon.NOT_CHOSEN and config().current(read_only) == 'false'

    def test_problem_checks_values_against_the_model(self) -> None:
        [auth, *_] = config().rows()
        assert config().problem(auth, 'browser') is None
        assert config().problem(auth, 'env') is not None

    async def test_edits_save_to_plugin_settings_at_once_and_apply_next_run(self) -> None:
        loaded = make_plugin()
        script = Script(
            lists=[pick('read_only'), pick('include_instructions'), pick('auth'), CLOSE],
            choices=[pick('true'), pick('false'), pick('browser')],
            texts=[],
        )
        edited = pylon.PylonConfig(loaded.host.settings(pylon.PylonSettings), loaded.host.save_settings)
        message = await pylon.configure(edited, script.runners)
        assert message.splitlines() == [
            'Pylon read-only tools: true. Applies from the next run.',
            'Pylon server instructions: false. Applies from the next run.',
            'Pylon sign-in: Browser sign-in (OAuth). Applies from the next run.',
        ]
        assert loaded.host.settings(pylon.PylonSettings) == pylon.PylonSettings(
            auth='browser', read_only=True, include_instructions=False
        )

    async def test_menu_edits_reach_the_running_capability(self, monkeypatch: pytest.MonkeyPatch) -> None:
        api_keys.save_key(name='SHARED_PYLON', value='secret')
        save_codex_credentials(account='pylon', value='{"token": {"name": "SHARED_PYLON"}}')
        loaded = make_plugin()
        script = Script(lists=[pick('read_only'), CLOSE], choices=[pick('true')], texts=[])
        monkeypatch.setattr(pylon, 'TERMINAL', script.runners)
        assert await command(loaded) == 'Pylon read-only tools: true. Applies from the next run.'
        assert built(loaded).read_only, 'no reload needed'

    async def test_key_row_opens_the_key_picker_and_reopens_the_menu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        api_keys.save_key(name='SHARED_PYLON', value='secret')
        use_key(monkeypatch, 'SHARED_PYLON')
        script = Script(lists=[pick('key'), CLOSE], choices=[], texts=[])
        assert await pylon.configure(config(), script.runners) == 'Pylon connects with SHARED_PYLON from /keys.'
        assert script.opened == ['list', 'list'], 'the menu comes back after the picker'

    async def test_key_picker_errors_are_shown_in_the_menu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        use_prompt(monkeypatch, Prompt(' '))
        script = Script(lists=[pick('key'), CLOSE], choices=[], texts=[])
        assert await pylon.configure(config(), script.runners) == 'A Pylon access token is required.'

    async def test_reset_forgets_the_key_and_restores_defaults(self) -> None:
        api_keys.save_key(name='SHARED_PYLON', value='secret')
        save_codex_credentials(account='pylon', value='{"token": {"name": "SHARED_PYLON"}}')
        edited = config(pylon.PylonSettings(read_only=True))
        script = Script(lists=[reset('key'), reset('read_only'), CLOSE], choices=[], texts=[])
        assert (await pylon.configure(edited, script.runners)).splitlines() == [
            'Pylon no longer uses a /keys entry; runs get no Pylon tools until you choose one.',
            'Pylon read-only tools: false. Applies from the next run.',
        ]
        assert pylon.saved_key() is None and edited.settings == pylon.PylonSettings()
        assert 'SHARED_PYLON' in api_keys.load_keys(), 'the key itself stays in /keys'

    async def test_closing_without_changes(self) -> None:
        script = Script(lists=[pick('auth'), CLOSE], choices=[CLOSE], texts=[])
        assert await pylon.configure(config(), script.runners) == 'Pylon settings unchanged.'


class TestShellIntegration:
    async def test_enabling_opens_the_settings_menu(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        store = SettingsStore(tmp_path / 'settings.db')
        script = Script(lists=[pick('read_only'), CLOSE], choices=[pick('true')], texts=[])
        monkeypatch.setattr(pylon, 'TERMINAL', script.runners)
        plugins = loader(store)
        assert await plugins.command(['enable', 'pylon']) == (
            'Enabled pylon.\nPylon read-only tools: true. Applies from the next run.'
        )
        assert len(plugins.capabilities()) == 1
        [saved] = store.plugins()
        assert saved.enabled and pylon.PylonSettings.model_validate(saved.settings).read_only

    async def test_configure_reopens_the_settings_menu(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        plugins = loader(SettingsStore(tmp_path / 'settings.db'))
        await plugins.enable('pylon')
        assert plugins.configurable('pylon')
        monkeypatch.setattr(pylon, 'TERMINAL', Script(lists=[CLOSE], choices=[], texts=[]).runners)
        assert await plugins.command(['configure', 'pylon']) == 'Pylon settings unchanged.'

    async def test_saved_settings_survive_a_reload(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        store = SettingsStore(tmp_path / 'settings.db')
        monkeypatch.setattr(
            pylon, 'TERMINAL', Script(lists=[pick('auth'), CLOSE], choices=[pick('browser')], texts=[]).runners
        )
        plugins = loader(store)
        await plugins.command(['enable', 'pylon'])
        await plugins.reload('pylon')
        [capability] = plugins.capabilities()
        assert not isinstance(capability, AbstractCapability)
        rebuilt = capability(CTX)
        assert isinstance(rebuilt, Pylon) and isinstance(rebuilt.client, Client)

    async def test_command_key_status_and_help(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loaded = make_plugin()
        assert 'no key yet' in await command(loaded, 'status')
        use_prompt(monkeypatch, Prompt('secret'))
        assert await command(loaded, 'key') == 'Pylon connects with PYLON_ACCESS_TOKEN from /keys.'
        assert await command(loaded, 'status') == (
            'Pylon connects with PYLON_ACCESS_TOKEN from /keys; read-only tools false, server instructions true.'
        )
        assert 'browser' in await command(make_plugin({'auth': 'browser'}), 'status')
        with pytest.raises(ValueError, match='Usage: /pylon'):
            await command(loaded, 'nope')
        [registered] = loaded.commands
        assert list(registered.complete([])) == ['key', 'status']
        assert list(registered.complete(['s'])) == ['status'] and list(registered.complete(['key', ''])) == []
