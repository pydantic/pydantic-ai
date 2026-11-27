"""`pax --minimal`, and the startup line naming plugins Pax doesn't ship (#14387).

Not VCR tests: no model request is checked, only which plugins load and what startup prints.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, AgentCapability, Capability
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import AbstractToolset, CombinedToolset, FunctionToolset
from pydantic_ai_harness.peers import InMemoryPeerNetwork, Peers
from pydantic_ai_pax import chat
from pydantic_ai_pax.cli import cli, headless
from pydantic_ai_pax.config import PluginSettings
from pydantic_ai_pax.config.builtins import DEFAULT_PLUGINS, MINIMAL_PLUGINS
from pydantic_ai_pax.config.settings_store import SettingsStore
from pydantic_ai_pax.plugins import Plugin, PluginHost, load_plugin
from pydantic_ai_pax.plugins.process_plugins import ProcessPlugins
from tests.pax.own_plugin_fixture import Unbound, port_scan, whois

SCANNER = PluginSettings(id='scanner', factory='tests.pax.own_plugin_fixture:Scanner')
ASK_USER = PluginSettings(id='ask_user', factory='pydantic_ai_pax.builtin_plugins.ask_user_menu')


async def _start(store: SettingsStore, *, keys: str = '', only_plugins: frozenset[str] | None = None) -> str:
    output = io.StringIO()
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text(f'{keys}/exit\r')
        await chat(
            Agent(TestModel(), deps_type=type(None)),
            deps=None,
            console=Console(file=output, width=200),
            store=store,
            builtin_plugins=(ASK_USER,),
            only_plugins=only_plugins,
        )
    return output.getvalue()


def _no_toolset(ctx: RunContext[None]) -> AbstractToolset[None] | None:
    # Built per run, so never asked at startup.
    return None  # pragma: no cover


def _no_capability(ctx: RunContext[None]) -> Capability[None] | None:
    # Built per run, so never asked at startup.
    return None  # pragma: no cover


class _Mixed(Plugin):
    def get_capabilities(self) -> list[AgentCapability[None]]:
        return [
            Capability(tools=[port_scan]),
            Capability(tools=[port_scan]),
            Capability(),
            Capability(toolsets=[_no_toolset, CombinedToolset([FunctionToolset([whois])])]),
            _no_capability,
        ]


def test_tool_names_are_the_function_tools_known_before_a_run() -> None:
    host: PluginHost[None] = PluginHost(name='mixed', console=Console(file=io.StringIO()), settings={})
    assert load_plugin(_Mixed, host).tool_names() == ['port_scan', 'whois']


@pytest.mark.parametrize(
    'unknown',
    [
        pytest.param(Peers(network=InMemoryPeerNetwork()), id='toolset-built-once-bound'),
        pytest.param(Unbound(), id='get-toolset-raises'),
        pytest.param(Capability(toolsets=[FunctionToolset([whois]).prefixed('lookup')]), id='prefixed'),
        pytest.param(Capability(toolsets=[FunctionToolset([whois]).renamed({'domain_info': 'whois'})]), id='renamed'),
    ],
)
def test_tool_names_are_unknown_when_a_capability_cannot_tell_them_before_a_run(
    unknown: AbstractCapability[None],
) -> None:
    class _Unknown(Plugin):
        def get_capabilities(self) -> list[AgentCapability[None]]:
            return [unknown, Capability(tools=[port_scan])]

    host: PluginHost[None] = PluginHost(name='unknown', console=Console(file=io.StringIO()), settings={})
    assert load_plugin(_Unknown, host).tool_names() is None


class TestOwnPlugins:
    async def test_startup_names_each_plugin_pax_does_not_ship_with_its_tools(self, tmp_path: Path) -> None:
        store = SettingsStore(tmp_path / 'config.db')
        store.save_plugin(SCANNER)
        store.save_plugin(PluginSettings(id='bare', factory='pydantic_ai.capabilities:Capability'))
        store.save_plugin(PluginSettings(id='unbound', factory='tests.pax.own_plugin_fixture:Unbound'))
        text = await _start(store)
        assert 'Plugin scanner: tools port_scan, whois\n' in text
        assert 'Plugin bare: 1 capability\n' in text
        assert 'Plugin unbound: 1 capability\n' in text, 'a failing `get_toolset` never fails the launch'
        assert 'Plugin ask_user' not in text, 'a built-in is not named'

    async def test_a_built_in_with_saved_settings_is_still_not_named(self, tmp_path: Path) -> None:
        store = SettingsStore(tmp_path / 'config.db')
        store.save_plugin(ASK_USER.model_copy(update={'settings': {}}))
        assert 'Plugin ' not in await _start(store)


class TestMinimal:
    def test_declarations_hold_other_plugins_off_without_saving(self, tmp_path: Path) -> None:
        store = SettingsStore(tmp_path / 'config.db')
        store.save_plugin(SCANNER)
        process = ProcessPlugins[None](store=store, builtin=(ASK_USER,), only={'ask_user'})
        declared = process.declared({}, console=Console(file=io.StringIO()))
        assert declared['ask_user'].enabled and not declared['scanner'].enabled
        assert [saved.enabled for saved in store.plugins()] == [True], 'nothing saved changes'
        process.allow('scanner')
        assert process.declared({}, console=Console(file=io.StringIO()))['scanner'].enabled
        unlimited = ProcessPlugins[None](store=store, builtin=(ASK_USER,))
        unlimited.allow('scanner')
        assert unlimited.only is None

    async def test_launch_loads_only_the_named_plugins_until_one_is_enabled(self, tmp_path: Path) -> None:
        store = SettingsStore(tmp_path / 'config.db')
        store.save_plugin(SCANNER)
        text = await _start(
            store, keys='/plugins list\r/plugins enable scanner\r/plugins list\r', only_plugins=frozenset({'ask_user'})
        )
        before, _, after = text.partition('Enabled scanner.')
        assert 'Plugin scanner' not in before, 'held off, so not loaded or named'
        assert 'scanner: tests.pax.own_plugin_fixture:Scanner (disabled)' in before
        assert 'scanner: tests.pax.own_plugin_fixture:Scanner (enabled, loaded)' in after
        assert [saved.enabled for saved in store.plugins()] == [True]

    async def test_a_plugin_enabled_meanwhile_stays_loaded_after_reload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def rebuild(factory: Callable[[], object]) -> object:
            return factory()

        monkeypatch.setattr('pydantic_ai_pax.app.reload_clai', rebuild)
        store = SettingsStore(tmp_path / 'config.db')
        store.save_plugin(SCANNER)
        text = await _start(
            store, keys='/plugins enable scanner\r/reload\r/plugins list\r', only_plugins=frozenset({'ask_user'})
        )
        _, _, after = text.partition('Pax reloaded.')
        assert 'scanner: tests.pax.own_plugin_fixture:Scanner (enabled, loaded)' in after

    def test_minimal_names_the_coding_tools_ask_user_and_sessions(self) -> None:
        shipped = {plugin.id for plugin in DEFAULT_PLUGINS}
        assert MINIMAL_PLUGINS == {'coder', 'ask_user', 'persistence'} and MINIMAL_PLUGINS <= shipped

    @pytest.mark.parametrize(('flags', 'expected'), [(['--minimal'], MINIMAL_PLUGINS), ([], None)])
    def test_flag_reaches_the_terminal_session(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: list[str], expected: frozenset[str] | None
    ) -> None:
        monkeypatch.setattr('sys.argv', ['pax', '--database', str(tmp_path / 'config.db'), *flags])
        seen: list[object] = []

        async def fake_chat(*_: object, only_plugins: frozenset[str] | None, **__: object) -> None:
            seen.append(only_plugins)

        monkeypatch.setattr('pydantic_ai_pax.app.chat', fake_chat)
        cli.run()
        assert seen == [expected]

    def test_flag_reaches_a_headless_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr('sys.argv', ['pax', '--database', str(tmp_path / 'config.db'), '--minimal', '-p', 'hi'])
        seen: list[object] = []

        async def run_headless(*, only_plugins: frozenset[str] | None, **_: object) -> int:
            seen.append(only_plugins)
            return 0

        monkeypatch.setattr(headless, 'run_headless', run_headless)
        with pytest.raises(SystemExit):
            cli.run()
        assert seen == [MINIMAL_PLUGINS]

    def test_flag_reaches_the_web_ui_with_web(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr('sys.argv', ['pax', '--database', str(tmp_path / 'config.db'), '--minimal', '--web'])
        seen: list[object] = []

        async def run_web(*, only_plugins: frozenset[str] | None, **_: object) -> None:
            seen.append(only_plugins)

        monkeypatch.setattr('pydantic_ai_pax.cli.web.run_web', run_web)
        cli.run()
        assert seen == [{*MINIMAL_PLUGINS, 'web'}]

    @pytest.mark.parametrize('other', [['--agent', 'pydantic_ai:Agent'], ['--update'], ['config']])
    def test_flag_refuses_what_loads_no_plugins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], other: list[str]
    ) -> None:
        monkeypatch.setattr('sys.argv', ['pax', '--database', str(tmp_path / 'config.db'), '--minimal', *other])
        with pytest.raises(SystemExit) as error:
            cli.run()
        assert error.value.code == 2
        assert '--minimal' in capsys.readouterr().err
