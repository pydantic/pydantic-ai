"""`clai2 --agent MODULE:ATTR`: resolution, CLI wiring, and the session-only plugin switch-off."""

import io
import sys
from pathlib import Path

import pytest
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.models.test import TestModel
from pydantic_clai2._app import DEFAULT_PLUGINS, create_shell
from pydantic_clai2.cli import _cli, headless
from pydantic_clai2.cli.agent_import import import_agent
from pydantic_clai2.config import PluginSettings, Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore

_MODULE = 'clai_agent_flag_fixture'


@pytest.fixture
def fixture_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An importable module in the launch directory, removed from `sys.modules` afterwards."""
    (tmp_path / f'{_MODULE}.py').write_text(
        'from pydantic_ai import Agent\n'
        'from pydantic_ai.models.test import TestModel\n'
        'agent = Agent(TestModel())\n'
        'modelless = Agent(None)\n'
        'number = 42\n'
        'nothing = None\n'
    )
    (tmp_path / 'clai_agent_flag_broken.py').write_text('import clai_agent_flag_missing_dependency\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'path', [path for path in sys.path if path != str(tmp_path)])
    for name in (_MODULE, 'clai_agent_flag_broken'):
        monkeypatch.delitem(sys.modules, name, raising=False)
    return tmp_path


def test_import_agent_resolves_instance_from_launch_directory(fixture_module: Path) -> None:
    agent = import_agent(f'{_MODULE}:agent')
    assert isinstance(agent, Agent)
    assert sys.path[-1] == str(fixture_module)
    # A second resolution does not add the launch directory again.
    assert import_agent(f'{_MODULE}:agent') is agent
    assert sys.path.count(str(fixture_module)) == 1


@pytest.mark.parametrize(
    ('path', 'error', 'message'),
    [
        ('no_colon', ValueError, 'expects MODULE:ATTR'),
        (f'{_MODULE}:', ValueError, 'expects MODULE:ATTR'),
        ('clai_agent_flag_absent.sub:agent', ImportError, "module 'clai_agent_flag_absent.sub' not found"),
        (f'{_MODULE}:missing', AttributeError, "has no attribute 'missing'"),
        (f'{_MODULE}:Agent', TypeError, "'Agent' is a class"),
        (f'{_MODULE}:number', TypeError, 'got int'),
        (f'{_MODULE}:nothing', TypeError, 'got NoneType'),
        ('clai_agent_flag_broken:agent', ModuleNotFoundError, 'clai_agent_flag_missing_dependency'),
    ],
)
@pytest.mark.usefixtures('fixture_module')
def test_import_agent_errors(path: str, error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        import_agent(path)


@pytest.mark.parametrize(
    'args', [['--agent', 'no_colon'], ['-a', f'{_MODULE}:number'], ['--agent', f'{_MODULE}:agent', 'config']]
)
@pytest.mark.usefixtures('fixture_module')
def test_cli_rejects_bad_agent(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> None:
    monkeypatch.setattr('sys.argv', ['clai2', '--database', 'config.db', *args])
    with pytest.raises(SystemExit) as error:
        _cli.run()
    assert error.value.code == 2


@pytest.mark.parametrize(
    ('attr', 'cli_model', 'expected'),
    [('agent', None, None), ('agent', 'explicit:model', 'explicit:model'), ('modelless', None, 'saved:model')],
)
def test_cli_chats_with_agent_and_no_plugins(
    fixture_module: Path, monkeypatch: pytest.MonkeyPatch, attr: str, cli_model: str | None, expected: str | None
) -> None:
    store = SettingsStore(fixture_module / 'config.db')
    store.set('model', 'saved:model')
    monkeypatch.delenv('CLAI_MODEL', raising=False)
    model_args = ['-m', cli_model] if cli_model else []
    monkeypatch.setattr(
        'sys.argv', ['clai2', '--database', str(store.path), '--agent', f'{_MODULE}:{attr}', *model_args]
    )
    seen: list[tuple[AbstractAgent[None, object], Settings | None, bool]] = []

    async def chat(
        agent: AbstractAgent[None, object], *, settings: Settings | None, load_plugins: bool, **_: object
    ) -> None:
        seen.append((agent, settings, load_plugins))

    monkeypatch.setattr('pydantic_clai2._app.chat', chat)
    _cli.run()
    [(agent, settings, load_plugins)] = seen
    assert agent is getattr(sys.modules[_MODULE], attr)
    assert settings is not None and settings.model == expected
    assert not load_plugins
    assert store.load().model == 'saved:model'


def test_cli_prompt_passes_agent_to_headless(fixture_module: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('sys.argv', ['clai2', '--database', 'config.db', '--agent', f'{_MODULE}:agent', '-p', 'hi'])
    seen: list[AbstractAgent[None, object] | None] = []

    async def run_headless(*, agent: AbstractAgent[None, object] | None, **_: object) -> int:
        seen.append(agent)
        return 0

    monkeypatch.setattr(headless, 'run_headless', run_headless)
    with pytest.raises(SystemExit) as error:
        _cli.run()
    assert error.value.code == 0
    assert seen == [sys.modules[_MODULE].agent]


def _plugin_sources(tmp_path: Path) -> tuple[SettingsStore, ProjectSettings]:
    """A saved user plugin, a drop-in file, and a project plugin that all fail loudly if loaded."""
    store = SettingsStore(tmp_path / 'config.db')
    store.save_plugin(PluginSettings(id='saved', factory='clai_agent_flag_missing_plugin'))
    store.plugins_dir.mkdir(parents=True, exist_ok=True)
    (store.plugins_dir / 'dropin.py').write_text('raise RuntimeError("loaded")\n')
    project = ProjectSettings(plugins=(PluginSettings(id='project', factory='clai_agent_flag_missing_project'),))
    return store, project


@pytest.mark.parametrize('load_plugins', [True, False])
async def test_create_shell_skips_every_plugin_source(tmp_path: Path, load_plugins: bool) -> None:
    store, project = _plugin_sources(tmp_path)
    shell = create_shell(
        Agent(TestModel()),
        deps=None,
        plugins=(),
        usage_limits=None,
        settings=None,
        project=project,
        console=Console(file=io.StringIO()),
        store=store,
        builtin_plugins=DEFAULT_PLUGINS,
        load_plugins=load_plugins,
    )
    names = {entry.name for entry in shell.loader.entries()}
    if load_plugins:
        # Unchanged default: built-ins, saved, drop-in, and project declarations are all listed.
        assert {plugin.id for plugin in DEFAULT_PLUGINS} | {'saved', 'dropin', 'project'} <= names
        return
    assert names == set()
    await shell.loader.load_all()
    assert shell.loader.capabilities() == []
    reply = await shell.commands.execute_async('/plugins add extra some_module')
    assert reply == 'Plugins are off for this session; saved plugin settings are unchanged.'
    assert [plugin.id for plugin in store.plugins()] == ['saved']


async def test_headless_agent_loads_no_plugins(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store, project = _plugin_sources(tmp_path)
    agent = Agent(TestModel(custom_output_text='custom answer'))
    assert (
        await headless.run_headless(text='hi', settings=Settings(model=None), store=store, project=project, agent=agent)
        == 0
    )
    assert capsys.readouterr().out == 'custom answer\n'
