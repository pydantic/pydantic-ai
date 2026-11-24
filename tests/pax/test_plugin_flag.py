"""`pax -x NAME`: load a plugin for one launch, whatever is saved, and save nothing that says it is enabled."""

import io
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import BaseModel
from rich.console import Console

from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.capabilities import Toolset
from pydantic_ai.models.test import TestModel
from pydantic_ai_pax.app import create_stock_agent
from pydantic_ai_pax.cli import headless
from pydantic_ai_pax.config import PluginSettings, Settings
from pydantic_ai_pax.config.project_settings import ProjectSettings
from pydantic_ai_pax.config.settings_store import SettingsStore
from pydantic_ai_pax.plugins import AgentCapability, Plugin, SessionStart
from pydantic_ai_pax.runtime.session_runtime import create_shell
from tests.pax.cli_runner import CliRunner


class ProbeSettings(BaseModel):
    started: int = 0


class Probe(Plugin[ProbeSettings]):
    """Adds a `probe` tool, and counts its starts in its saved settings."""

    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        def probe() -> str:
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


async def test_a_launched_plugin_loads_in_every_session_until_disabled(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    shell = create_shell(
        Agent(TestModel(call_tools=[])),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=io.StringIO()),
        settings=None,
        store=store,
        builtin_plugins=[PROBE],
        project=ProjectSettings(),
        headless=True,
        launch_plugins=['probe'],
    )
    await shell.loader.load_all()
    assert [entry.name for entry in shell.loader.entries() if entry.loaded is not None] == ['probe']
    # Never saved before, it is saved disabled, as it is declared, along with the settings it saved.
    assert store.plugins() == [PROBE.model_copy(update={'settings': {'started': 1}})]
    await shell.loader.disable('probe')
    assert shell.process.plugins.launch == frozenset()
    await shell.close('exit')


def test_unknown_plugin_lists_the_available_ones(
    tmp_path: Path, run_cli: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def chat(*args: object, **kwargs: object) -> None:  # pragma: no cover - the launch fails first.
        raise AssertionError('started')

    monkeypatch.setattr('pydantic_ai_pax.app.chat', chat)
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
