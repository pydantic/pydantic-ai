"""`pax -x NAME`: load a plugin for one launch, whatever is saved, and save nothing that says it is enabled."""

import argparse
import io
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from pydantic import BaseModel
from rich.console import Console

from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.capabilities import Toolset
from pydantic_ai.models.test import TestModel
from pydantic_ai_pax import app
from pydantic_ai_pax.app import create_stock_agent
from pydantic_ai_pax.cli import cli, headless, web
from pydantic_ai_pax.config import PluginSettings, Settings
from pydantic_ai_pax.config.project_settings import ProjectSettings
from pydantic_ai_pax.config.settings_store import SettingsStore
from pydantic_ai_pax.plugins import AgentCapability, Plugin, SessionStart
from pydantic_ai_pax.runtime.process import ClaiProcess
from pydantic_ai_pax.runtime.session_runtime import SessionRuntime, create_shell
from tests.pax.cli_runner import CliRunner
from tests.pax.test_app_edges import inputs


class ProbeSettings(BaseModel):
    started: int = 0


class Probe(Plugin[ProbeSettings]):
    """Adds a `probe` tool, and counts its starts in its saved settings."""

    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        # Only its presence among the tools is checked.
        def probe() -> str:  # pragma: no cover
            return 'probed'

        return [Toolset(FunctionToolset([probe]))]

    async def on_session_start(self, event: SessionStart) -> None:
        self.host.save_settings(ProbeSettings(started=self.settings.started + 1))


PROBE = PluginSettings(id='probe', factory='tests.pax.test_plugin_flag:Probe', enabled=False)


def _tools(model: TestModel) -> set[str]:
    assert model.last_model_request_parameters is not None
    return {tool.name for tool in model.last_model_request_parameters.function_tools}


@pytest.mark.parametrize('launched', [False, True])
async def test_headless_loads_a_disabled_plugin_for_the_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, launched: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    model = TestModel(call_tools=[], custom_output_text='done')
    monkeypatch.setattr(headless, 'create_agent', lambda: create_stock_agent(model))
    store = SettingsStore(tmp_path / 'config.db')
    store.save_plugin(PROBE)
    assert (
        await headless.run_headless(
            text='go',
            settings=Settings(model=None),
            store=store,
            project=ProjectSettings(),
            launch_plugins=['probe'] if launched else [],
        )
        == 0
    )
    assert ('probe' in _tools(model)) is launched
    # Its settings are saved as it saves them, but it stays disabled for the next launch.
    [saved] = [plugin for plugin in store.plugins() if plugin.id == 'probe']
    assert not saved.enabled
    assert saved.settings == ({'started': 1} if launched else {})


def _shell(
    store: SettingsStore, *, project: ProjectSettings | None = None, process: ClaiProcess[None] | None = None
) -> SessionRuntime[None, str]:
    return create_shell(
        Agent(TestModel(call_tools=[])),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=io.StringIO()),
        settings=None,
        store=None if process is not None else store,
        builtin_plugins=[] if process is not None else [PROBE],
        project=project or ProjectSettings(),
        headless=True,
        launch_plugins=['probe'],
        process=process,
    )


def _loaded(shell: SessionRuntime[None, str]) -> list[str]:
    return [entry.name for entry in shell.loader.entries() if entry.loaded is not None]


async def test_launched_plugin_loads_disabled_and_is_saved_as_declared(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    shell = _shell(store)
    await shell.loader.load_all()
    assert _loaded(shell) == ['probe']
    # Never saved before, it is saved disabled, as it is declared, along with the settings it saved.
    assert store.plugins() == [PROBE.model_copy(update={'settings': {'started': 1}})]
    await shell.close('exit')


@pytest.mark.parametrize(
    ('commands', 'starts'),
    [
        pytest.param(['/reload'], 2, id='kept-through-reload'),
        pytest.param(['/plugins disable probe', '/reload'], 1, id='disabled-stays-off'),
    ],
)
async def test_reload_keeps_the_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, commands: list[str], starts: int
) -> None:
    def rebuild(build: Callable[[], object]) -> object:
        return build()

    monkeypatch.setattr(app, 'reload_clai', rebuild)
    inputs(monkeypatch, [*commands, '/exit'])
    store = SettingsStore(tmp_path / 'config.db')
    await app.chat(
        Agent(TestModel()),
        deps=None,
        settings=Settings(model='test'),
        console=Console(file=io.StringIO(), width=200),
        store=store,
        builtin_plugins=[PROBE],
        launch_plugins=['probe'],
    )
    # Each load counts a start; the rebuilt session loads it again only while the launch still names it.
    [saved] = store.plugins()
    assert (saved.enabled, saved.settings) == (False, {'started': starts})


async def test_reloading_a_launched_plugin_keeps_it_in_the_other_sessions(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    first = _shell(store)
    await first.loader.load_all()
    second = _shell(store, process=first.process)
    await second.loader.load_all()
    await first.loader.reload('probe')
    assert _loaded(first) == _loaded(second) == ['probe']
    await second.close('exit')
    await first.close('exit')


async def test_removing_a_launched_plugin_removes_it_from_the_launch(tmp_path: Path) -> None:
    """`/plugins remove` turns it off in every session, and in the ones started later in the launch."""
    store = SettingsStore(tmp_path / 'config.db')
    first = _shell(store)
    await first.loader.load_all()
    second = _shell(store, process=first.process)
    await second.loader.load_all()
    await first.loader.remove('probe')
    assert _loaded(first) == _loaded(second) == []
    later = _shell(store, process=first.process)
    await later.loader.load_all()
    assert _loaded(later) == []
    for shell in (later, second, first):
        await shell.close('exit')


async def test_a_project_declaration_needs_approval_first(tmp_path: Path) -> None:
    """`-x` doesn't stand in for `/plugins enable`, the approval of what a repository declares."""
    store = SettingsStore(tmp_path / 'config.db')
    project = ProjectSettings(plugins=(PROBE.model_copy(update={'id': 'pinned'}),))
    with pytest.raises(ValueError, match='pinned: declared by the project; approve it with /plugins enable first'):
        create_shell(
            Agent(TestModel(call_tools=[])),
            deps=None,
            plugins=(),
            usage_limits=None,
            console=Console(file=io.StringIO()),
            settings=None,
            store=store,
            builtin_plugins=[],
            project=project,
            headless=True,
            launch_plugins=['pinned'],
        )
    # Named in the launch some other way, it still doesn't load.
    shell = _shell(store, project=project)
    shell.process.plugins.launch |= {'pinned'}
    await shell.loader.load_all()
    assert _loaded(shell) == ['probe']
    await shell.close('exit')


def test_unknown_plugin_lists_the_available_ones(tmp_path: Path, run_cli: CliRunner) -> None:
    result = run_cli('--database', str(tmp_path / 'config.db'), '-x', 'sandbox', '-x', 'nope', cwd=tmp_path)
    assert result.returncode == 2
    assert 'Unknown plugin nope; available: ' in result.stderr
    assert 'sandbox' in result.stderr.split('available: ')[1]


def test_plugins_reach_the_session(tmp_path: Path, run_cli: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    launched: list[object] = []

    async def chat(*args: object, launch_plugins: Sequence[str], **kwargs: object) -> None:
        launched.append(launch_plugins)

    monkeypatch.setattr('pydantic_ai_pax.app.chat', chat)
    result = run_cli('--database', str(tmp_path / 'config.db'), '-x', 'web', '--plugin', 'workspaces', cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert launched == [['web', 'workspaces']]
    assert all(plugin.id not in ('web', 'workspaces') for plugin in SettingsStore(tmp_path / 'config.db').plugins())


@pytest.mark.parametrize(
    ('args', 'error'),
    [
        (['-a', 'm:a', '-x', 'sandbox'], '--plugin cannot be combined with --agent'),
        (['-x', 'sandbox', 'plugins'], 'and --plugin cannot be combined with config, plugins, or sessions'),
        (['--update', '-x', 'sandbox'], '--request-limit, or --plugin'),
    ],
)
def test_plugin_flag_conflicts(tmp_path: Path, run_cli: CliRunner, args: list[str], error: str) -> None:
    result = run_cli('--database', str(tmp_path / 'config.db'), *args, cwd=tmp_path)
    assert result.returncode == 2
    assert error in result.stderr


def test_plugins_reach_headless_and_web(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[object] = []

    async def launch(*, launch_plugins: Sequence[str], **_: object) -> int:
        seen.append(launch_plugins)
        return 0

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(headless, 'run_headless', launch)
    monkeypatch.setattr(web, 'run_web', launch)
    for args in (['-p', 'go'], ['--web']):
        monkeypatch.setattr('sys.argv', ['pax', '--database', 'config.db', '-x', 'web', *args])
        try:
            cli.run()
        except SystemExit as exit:
            assert exit.code == 0
    assert seen == [['web'], ['web']]


async def test_launch_plugins_need_plugins(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='`launch_plugins` needs `load_plugins=True`'):
        await headless.run_headless(
            text='go',
            settings=Settings(model=None),
            store=SettingsStore(tmp_path / 'config.db'),
            project=ProjectSettings(),
            agent=Agent(TestModel(), deps_type=type(None)),
            launch_plugins=['sandbox'],
        )


def test_update_restarts_with_the_plugins_still_launched() -> None:
    """A plugin disabled or removed during the launch stays off after `/update`, as after `/reload`."""
    args = argparse.Namespace(web=False, agent=None, model=None, request_limit=None, database=None, plugin=['a', 'b'])
    assert cli.relaunch_argv(args, executable='/b/pax', session_id=None, launch_plugins=['b']) == [
        '/b/pax',
        '--plugin',
        'b',
    ]
