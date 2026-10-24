"""`.clai/settings.json`: the walk-up, the layering, and the startup report."""

import io
import json
import os
from pathlib import Path

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic import JsonValue
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_clai2 import chat
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import PluginSettings, resolve_settings
from pydantic_clai2.config.project_settings import (
    PROJECT_FILE,
    ProjectSettings,
    find_project_file,
    load_project_settings,
)
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import SessionStart
from pydantic_clai2.plugins.loader import PluginLoader


def write(directory: Path, content: dict[str, JsonValue]) -> Path:
    path = directory / PROJECT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))
    return path


def test_walk_up_stops_at_the_git_root(tmp_path: Path) -> None:
    repo = tmp_path / 'repo'
    nested = repo / 'src' / 'pkg'
    nested.mkdir(parents=True)
    assert find_project_file(nested) is None

    above = write(tmp_path, {'thinking': False})
    (repo / '.git').mkdir()
    assert find_project_file(nested) is None, 'a file above the git root is not this project'
    (repo / '.git').rmdir()
    assert find_project_file(nested) == above

    (repo / '.git').write_text('gitdir: elsewhere')
    inside = write(repo, {'thinking': True})
    assert find_project_file(nested) == inside
    assert find_project_file(repo) == inside
    assert find_project_file(nested / 'missing') == inside, 'the walk starts from the resolved path even if absent'


def test_permission_failures_are_boundaries_not_absences(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / 'repo'
    nested = repo / 'src'
    nested.mkdir(parents=True)
    write(tmp_path, {'thinking': False})
    real_stat = Path.stat

    def locked(name: str, parent: Path) -> None:
        def stat(self: Path, *, follow_symlinks: bool = True) -> os.stat_result:
            if self.name == name and self.parent == parent:
                raise PermissionError(f'{self} is locked')
            return real_stat(self, follow_symlinks=follow_symlinks)

        monkeypatch.setattr(Path, 'stat', stat)

    locked('.git', repo)
    assert find_project_file(nested) is None, 'an unreadable .git still marks the repository boundary'
    locked('settings.json', repo / '.clai')
    with pytest.raises(ValueError, match=r'repo/\.clai/settings\.json: '):
        load_project_settings(nested)
    (repo / '.clai').touch()
    monkeypatch.setattr(Path, 'stat', real_stat)
    assert find_project_file(nested) == tmp_path / PROJECT_FILE, 'a file named .clai is not a settings folder'


def test_load_layers_between_store_and_flags(tmp_path: Path) -> None:
    path = write(tmp_path, {'thinking': False, 'request_limit': 7, 'colour': 'mauve', 'zeta': 1})
    project = load_project_settings(tmp_path / 'deeper')
    assert project.path == path
    assert project.overrides == {'display.thinking': False, 'run.request_limit': 7}
    assert project.unknown == ('colour', 'zeta')
    assert project.plugins == ()

    store = SettingsStore(tmp_path / 'config.db')
    store.set('run.request_limit', 3)
    store.set('display.splash', False)
    layered = resolve_settings(store.overrides() | project.overrides | {'run.request_limit': 9})
    assert (layered.request_limit, layered.thinking, layered.splash) == (9, False, False)


def test_load_without_a_file_is_empty(tmp_path: Path) -> None:
    assert load_project_settings(tmp_path) == ProjectSettings()


def test_plugins_are_validated_declarations_that_start_off(tmp_path: Path) -> None:
    write(tmp_path, {'plugins': [{'id': 'exa', 'factory': 'pydantic_ai_harness.exa:ExaSearch', 'enabled': True}]})
    project = load_project_settings(tmp_path)
    assert project.plugins == (PluginSettings(id='exa', factory='pydantic_ai_harness.exa:ExaSearch', enabled=False),)
    assert project.overrides == {}


@pytest.mark.parametrize(
    'content',
    [
        {'request_limit': 0},
        {'model': 7},
        {'plugins': [{'id': 'nope'}]},
        {'plugins': 'exa'},
    ],
)
def test_bad_values_fail_startup(tmp_path: Path, content: dict[str, JsonValue]) -> None:
    write(tmp_path, content)
    with pytest.raises(ValueError, match=r'settings.json'):
        load_project_settings(tmp_path)


def test_unreadable_file_fails_startup_with_its_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write(tmp_path, {'thinking': False})

    def unreadable(self: Path) -> bytes:
        raise PermissionError(f'{self} is locked')

    monkeypatch.setattr(Path, 'read_bytes', unreadable)
    with pytest.raises(ValueError, match=r'settings\.json: .* is locked'):
        load_project_settings(tmp_path)


def test_not_an_object_fails_startup(tmp_path: Path) -> None:
    path = tmp_path / PROJECT_FILE
    path.parent.mkdir()
    path.write_text('[1, 2]')
    with pytest.raises(ValueError, match=r'settings.json'):
        load_project_settings(tmp_path)


async def test_startup_reports_unknown_keys_once(tmp_path: Path) -> None:
    plugin: dict[str, JsonValue] = {'id': 'hello', 'factory': 'pydantic_ai.capabilities:Capability'}
    path = write(tmp_path, {'thinking': False, 'colour': 'mauve', 'plugins': [plugin]})
    project = load_project_settings(tmp_path)
    output = io.StringIO()
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text('/set display.thinking true\r/set display.thinking\r/plugins list\r/exit\r')
        await chat(
            Agent(TestModel(), deps_type=type(None)),
            deps=None,
            console=Console(file=output, width=200),
            store=SettingsStore(tmp_path / 'config.db'),
            settings=resolve_settings(project.overrides),
            project=project,
        )
    text = output.getvalue()
    assert f'Project settings: {path}' in text
    assert text.count('Ignoring unknown settings: colour') == 1
    assert 'Project plugins not loaded; approve one with /plugins enable NAME: hello' in text
    assert 'hello: pydantic_ai.capabilities:Capability (project) (disabled)' in text
    assert 'Saved display.thinking. Applied. The project file sets it again at next start.' in text
    assert '\nTrue\n' in text, '/set still applies to this session'


@pytest.mark.parametrize('with_file', [True, False])
async def test_startup_reports_a_clean_file_or_nothing(tmp_path: Path, with_file: bool) -> None:
    output = io.StringIO()
    project = ProjectSettings(path=tmp_path / PROJECT_FILE) if with_file else None
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text('/exit\r')
        await chat(
            Agent(TestModel(), deps_type=type(None)),
            deps=None,
            console=Console(file=output, width=200),
            store=SettingsStore(tmp_path / 'config.db'),
            project=project,
        )
    assert ('Project settings' in output.getvalue()) is with_file
    assert 'Ignoring' not in output.getvalue()


HELPER = """
from pathlib import Path

from pydantic_clai2.plugins import Plugin

Path({marker!r}).write_text('ran')


class Helper(Plugin):
    pass
"""


def repository(root: Path, name: str, *, declared: bool = True) -> Path:
    """A git checkout whose `plugins/helper.py` leaves `<name>.ran` next to it when it runs."""
    directory = root / name
    (directory / '.git').mkdir(parents=True)
    (directory / 'src').mkdir()
    helper = directory / 'plugins' / 'helper.py'
    helper.parent.mkdir()
    helper.write_text(HELPER.format(marker=str(root / f'{name}.ran')))
    if declared:
        write(directory, {'plugins': [{'id': 'helper', 'factory': 'helper', 'path': 'plugins/helper.py'}]})
    return directory


def ran(root: Path) -> list[str]:
    return sorted(marker.stem for marker in root.glob('*.ran'))


def plugin_loader(store: SettingsStore, project: ProjectSettings, output: io.StringIO) -> PluginLoader[None]:
    return PluginLoader(
        store=store,
        console=Console(file=output, width=200),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        project=project.plugins,
    )


@pytest.mark.parametrize('impostor_declares', [True, False])
async def test_approving_a_relative_project_plugin_never_runs_another_repositorys_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, impostor_declares: bool
) -> None:
    """The approval lives in the user store every repository shares, so its path must not follow the launch directory."""
    trusted = repository(tmp_path, 'trusted')
    impostor = repository(tmp_path, 'impostor', declared=impostor_declares)
    store = SettingsStore(tmp_path / 'config.db')

    monkeypatch.chdir(trusted / 'src')
    project = load_project_settings(Path.cwd())
    assert project.plugins == (
        PluginSettings(id='helper', factory='helper', path=str(trusted / 'plugins' / 'helper.py'), enabled=False),
    ), 'anchored to the folder holding .clai, not the launch directory'
    await plugin_loader(store, project, io.StringIO()).command(['enable', 'helper'])
    assert ran(tmp_path) == ['trusted']
    assert [plugin.path for plugin in store.plugins()] == [str(trusted / 'plugins' / 'helper.py')]

    (tmp_path / 'trusted.ran').unlink()
    monkeypatch.chdir(impostor)
    loader = plugin_loader(store, load_project_settings(impostor), io.StringIO())
    await loader.load_all()
    assert ran(tmp_path) == ['trusted'], 'the approved file runs, never the same path in this repository'


async def test_a_relative_path_saved_before_anchoring_is_refused_until_approved_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trusted = repository(tmp_path, 'trusted')
    impostor = repository(tmp_path, 'impostor')
    store = SettingsStore(tmp_path / 'config.db')
    store.save_plugin(PluginSettings(id='helper', factory='helper', path='plugins/helper.py'))

    monkeypatch.chdir(impostor)
    output = io.StringIO()
    await plugin_loader(store, load_project_settings(impostor), output).load_all()
    assert ran(tmp_path) == []
    assert 'plugins/helper.py is relative, so it would load from whichever directory CLAI starts in.' in (
        output.getvalue()
    )

    monkeypatch.chdir(trusted)
    loader = plugin_loader(store, load_project_settings(trusted), io.StringIO())
    await loader.load_all()
    assert ran(tmp_path) == [] and loader.entries()[0].state.startswith('enabled, failed: ValueError:')
    message = await loader.command(['remove', 'helper'])
    assert message.startswith('helper is declared by the project; restored its defaults.')
    assert store.plugins() == [] and loader.entries()[0].state == 'disabled'
    await loader.command(['enable', 'helper'])
    assert ran(tmp_path) == ['trusted']
    assert [plugin.path for plugin in store.plugins()] == [str(trusted / 'plugins' / 'helper.py')]
