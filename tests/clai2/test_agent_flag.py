"""`clai2 --agent MODULE:ATTR`: resolution, CLI wiring, and the session-only plugin switch-off."""

import importlib
import io
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.models.test import TestModel
from pydantic_clai2.app import DEFAULT_PLUGINS
from pydantic_clai2.cli import cli, headless
from pydantic_clai2.cli.agent_import import import_agent
from pydantic_clai2.config import PluginSettings, Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.runtime.session_runtime import create_shell

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


def test_import_agent_leaves_sys_path_alone_for_an_importable_module(
    fixture_module: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A launch directory module must not replace a package the agent imports after resolution."""
    installed = tmp_path_factory.mktemp('installed')
    (installed / 'clai_agent_flag_installed.py').write_text(
        'from pydantic_ai import Agent\nfrom pydantic_ai.models.test import TestModel\nagent = Agent(TestModel())\n'
    )
    monkeypatch.setattr(sys, 'path', [str(installed), *sys.path])
    monkeypatch.delitem(sys.modules, 'clai_agent_flag_installed', raising=False)

    assert isinstance(import_agent('clai_agent_flag_installed:agent'), Agent)
    assert str(fixture_module) not in sys.path


@pytest.mark.parametrize(
    'package',
    ['clai_agent_flag_installed_namespace', 'clai_agent_flag_installed_namespace.nested'],
    ids=['namespace-package', 'nested-namespace-package'],
)
def test_import_agent_leaves_sys_path_alone_for_a_target_in_an_installed_namespace_package(
    fixture_module: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch, package: str
) -> None:
    """A regular package anywhere on `sys.path` takes precedence over namespace portions, even an earlier one."""
    installed = tmp_path_factory.mktemp('installed')
    package_directory = installed.joinpath(*package.split('.'))
    package_directory.mkdir(parents=True)
    (package_directory / 'agents.py').write_text(
        'from pydantic_ai import Agent\nfrom pydantic_ai.models.test import TestModel\nagent = Agent(TestModel())\n'
    )
    (fixture_module / 'clai_agent_flag_installed_namespace').mkdir()
    (fixture_module / 'clai_agent_flag_installed_namespace' / '__init__.py').write_text('')
    monkeypatch.setattr(sys, 'path', [str(installed), *sys.path])
    for name in ('clai_agent_flag_installed_namespace', package, f'{package}.agents'):
        monkeypatch.delitem(sys.modules, name, raising=False)

    assert isinstance(import_agent(f'{package}.agents:agent'), Agent)
    assert str(fixture_module) not in sys.path


@pytest.mark.parametrize(
    'installed_target_directory', [False, True], ids=['namespace-parent', 'namespace-parent-and-target']
)
@pytest.mark.parametrize('launch_directory_package', [False, True], ids=['namespace-portion', 'regular-package'])
def test_import_agent_resolves_a_launch_directory_target_beside_an_installed_namespace_package(
    fixture_module: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    installed_target_directory: bool,
    launch_directory_package: bool,
) -> None:
    """An installed namespace package, even one named like the target, holds no agent and yields to a regular package."""
    installed = tmp_path_factory.mktemp('installed')
    (installed / 'clai_agent_flag_namespace').mkdir()
    if installed_target_directory:
        (installed / 'clai_agent_flag_namespace' / 'agents').mkdir()
    (fixture_module / 'clai_agent_flag_namespace').mkdir()
    if launch_directory_package:
        (fixture_module / 'clai_agent_flag_namespace' / '__init__.py').write_text('')
    (fixture_module / 'clai_agent_flag_namespace' / 'agents.py').write_text(
        'from pydantic_ai import Agent\nfrom pydantic_ai.models.test import TestModel\nagent = Agent(TestModel())\n'
    )
    monkeypatch.setattr(sys, 'path', [str(installed), *sys.path])
    for name in ('clai_agent_flag_namespace', 'clai_agent_flag_namespace.agents'):
        monkeypatch.delitem(sys.modules, name, raising=False)

    assert isinstance(import_agent('clai_agent_flag_namespace.agents:agent'), Agent)
    assert sys.path[-1] == str(fixture_module)


def package_modules_at_import(monkeypatch: pytest.MonkeyPatch, module_name: str) -> list[list[str]]:
    """Record, each time `module_name` is imported, which modules of its top-level package are already loaded."""
    top_level = module_name.partition('.')[0]
    recorded: list[list[str]] = []
    import_module = importlib.import_module

    def record_and_import(name: str, package: str | None = None) -> ModuleType:
        if name == module_name:
            recorded.append(sorted(loaded for loaded in sys.modules if loaded.partition('.')[0] == top_level))
        return import_module(name, package)

    monkeypatch.setattr(importlib, 'import_module', record_and_import)
    return recorded


def test_import_agent_resolves_a_nested_launch_directory_target_beside_installed_packages(
    fixture_module: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An installed regular subpackage of an installed namespace package yields with it to a launch directory package."""
    installed = tmp_path_factory.mktemp('installed')
    (installed / 'clai_agent_flag_nested' / 'sub').mkdir(parents=True)
    (installed / 'clai_agent_flag_nested' / 'sub' / '__init__.py').write_text('')
    package = fixture_module / 'clai_agent_flag_nested'
    (package / 'sub').mkdir(parents=True)
    (package / '__init__.py').write_text('')
    (package / 'sub' / '__init__.py').write_text('')
    (package / 'sub' / 'agents.py').write_text(
        'from pydantic_ai import Agent\nfrom pydantic_ai.models.test import TestModel\nagent = Agent(TestModel())\n'
    )
    monkeypatch.setattr(sys, 'path', [str(installed), *sys.path])
    for name in ('clai_agent_flag_nested', 'clai_agent_flag_nested.sub', 'clai_agent_flag_nested.sub.agents'):
        monkeypatch.delitem(sys.modules, name, raising=False)
    loaded_before_import = package_modules_at_import(monkeypatch, 'clai_agent_flag_nested.sub.agents')

    assert isinstance(import_agent('clai_agent_flag_nested.sub.agents:agent'), Agent)
    assert loaded_before_import == [[]]
    assert sys.modules['clai_agent_flag_nested.sub.agents'].__file__ == str(package / 'sub' / 'agents.py')


def test_import_agent_prefers_an_installed_regular_package_to_the_launch_directory(
    fixture_module: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An installed regular package with the target's top-level name wins, though only the launch directory has it."""
    installed = tmp_path_factory.mktemp('installed')
    (installed / 'clai_agent_flag_regular').mkdir()
    (installed / 'clai_agent_flag_regular' / '__init__.py').write_text('')
    (fixture_module / 'clai_agent_flag_regular').mkdir()
    (fixture_module / 'clai_agent_flag_regular' / '__init__.py').write_text('')
    (fixture_module / 'clai_agent_flag_regular' / 'agents.py').write_text(
        'from pydantic_ai import Agent\nfrom pydantic_ai.models.test import TestModel\nagent = Agent(TestModel())\n'
    )
    monkeypatch.setattr(sys, 'path', [str(installed), *sys.path])
    for name in ('clai_agent_flag_regular', 'clai_agent_flag_regular.agents'):
        monkeypatch.delitem(sys.modules, name, raising=False)
    loaded_before_import = package_modules_at_import(monkeypatch, 'clai_agent_flag_regular.agents')

    with pytest.raises(ImportError, match=r"module 'clai_agent_flag_regular\.agents' not found"):
        import_agent('clai_agent_flag_regular.agents:agent')
    assert loaded_before_import == [[]]
    assert sys.modules['clai_agent_flag_regular'].__file__ == str(installed / 'clai_agent_flag_regular' / '__init__.py')


class _FinderWithoutFindSpec:
    """A `sys.meta_path` finder from before `find_spec`; Python 3.11 still asks its `find_module`."""

    def find_module(self, fullname: str, path: Sequence[str] | None = None) -> None:
        return None


@pytest.mark.parametrize(
    ('module_name', 'appends_launch_directory'),
    [('clai_agent_flag_installed', False), (_MODULE, True)],
    ids=['installed-target', 'launch-directory-target'],
)
@pytest.mark.filterwarnings(r'ignore:.*find_spec\(\) not found:ImportWarning')
def test_import_agent_skips_a_meta_path_finder_without_find_spec(
    fixture_module: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    appends_launch_directory: bool,
) -> None:
    """The import system skips such a finder, so the lookup does too."""
    installed = tmp_path_factory.mktemp('installed')
    (installed / 'clai_agent_flag_installed.py').write_text(
        'from pydantic_ai import Agent\nfrom pydantic_ai.models.test import TestModel\nagent = Agent(TestModel())\n'
    )
    monkeypatch.setattr(sys, 'path', [str(installed), *sys.path])
    monkeypatch.setattr(sys, 'meta_path', [_FinderWithoutFindSpec(), *sys.meta_path])
    monkeypatch.delitem(sys.modules, 'clai_agent_flag_installed', raising=False)

    assert isinstance(import_agent(f'{module_name}:agent'), Agent)
    assert (str(fixture_module) in sys.path) is appends_launch_directory


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
        cli.run()
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

    monkeypatch.setattr('pydantic_clai2.app.chat', chat)
    cli.run()
    [(agent, settings, load_plugins)] = seen
    assert agent is getattr(sys.modules[_MODULE], attr)
    assert settings is not None and settings.model == expected
    assert not load_plugins
    assert store.load().model == 'saved:model'


class TestCodeMode:
    @pytest.fixture
    def native_tools(self) -> None:
        """The shipped code mode default, not the suite's code-mode-off stand-in."""

    def test_agent_launch_starts_with_code_mode_off(
        self, fixture_module: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = SettingsStore(fixture_module / 'config.db')
        store.set('run.speculative_code_mode', True)
        monkeypatch.setattr('sys.argv', ['clai2', '--database', str(store.path), '-a', f'{_MODULE}:agent'])
        seen: list[Settings] = []

        async def chat(_agent: object, *, settings: Settings, **_: object) -> None:
            seen.append(settings)

        monkeypatch.setattr('pydantic_clai2.app.chat', chat)
        cli.run()
        [settings] = seen
        assert not settings.code_mode and not settings.speculative_code_mode
        assert store.load().speculative_code_mode

    def test_default_launch_keeps_code_mode_on(self, tmp_path: Path) -> None:
        store = SettingsStore(tmp_path / 'config.db')
        settings = cli._settings_for(store, model=None, request_limit=None)(ProjectSettings())  # pyright: ignore[reportPrivateUsage]
        assert settings.code_mode


def test_cli_prompt_passes_agent_to_headless(fixture_module: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('sys.argv', ['clai2', '--database', 'config.db', '--agent', f'{_MODULE}:agent', '-p', 'hi'])
    seen: list[AbstractAgent[None, object] | None] = []

    async def run_headless(*, agent: AbstractAgent[None, object] | None, **_: object) -> int:
        seen.append(agent)
        return 0

    monkeypatch.setattr(headless, 'run_headless', run_headless)
    with pytest.raises(SystemExit) as error:
        cli.run()
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
