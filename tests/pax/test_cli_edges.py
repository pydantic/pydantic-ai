"""Exercise the command line: one installed entry point smoke test, the rest in-process."""

import asyncio
import os
import sqlite3
import subprocess
import sys
import threading
from collections.abc import Coroutine
from contextlib import closing
from pathlib import Path

import pytest

import pydantic_ai_pax.ui.rendering.splash
from pydantic_ai.models import infer_model_profile
from pydantic_ai.profiles import ModelProfile
from pydantic_ai_pax.__main__ import main
from pydantic_ai_pax.cli import cli, self_update
from pydantic_ai_pax.cli.self_update import Installed
from pydantic_ai_pax.config.settings_store import SettingsStore
from pydantic_ai_pax.runtime import session_runtime
from pydantic_ai_pax.ui.rendering.splash import Splash
from tests.pax.cli_runner import CliRunner


@pytest.mark.subprocess(
    reason='smoke-tests the installed `python -m pydantic_ai_pax` entry point; the other CLI tests run in-process'
)
def test_installed_entry_point_starts_and_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`python -m pydantic_ai_pax` without a model: the one launch here that crosses the process boundary."""
    # Subprocess coverage adds import overhead to the CLI startup hang guard.
    for name in tuple(os.environ):
        if name.startswith('COVERAGE_'):
            monkeypatch.delenv(name)
    env = dict(os.environ, CLAI_NO_SPLASH='1')
    env.pop('CLAI_MODEL', None)
    result = subprocess.run(
        [sys.executable, '-m', 'pydantic_ai_pax', '--database', str(tmp_path / 'config.db')],
        input='/exit\n',
        text=True,
        capture_output=True,
        env=env,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ('args', 'returncode'), [([], 0), (['--model', 'test', '--request-limit', '12'], 0), (['--request-limit', '0'], 2)]
)
def test_cli_startup(tmp_path: Path, run_cli: CliRunner, args: list[str], returncode: int) -> None:
    result = run_cli('--database', str(tmp_path / 'config.db'), *args, cwd=tmp_path)
    assert result.returncode == returncode, result.stderr


@pytest.mark.parametrize(
    ('running', 'printed'),
    [
        (Installed(version='0.73.0'), 'pax 0.73.0'),
        (Installed(version='0.74.0.dev3', commit='0123456789abcdef'), 'pax 0.74.0.dev3 (012345678)'),
    ],
    ids=['release', 'main'],
)
def test_version_prints_and_exits(
    tmp_path: Path, run_cli: CliRunner, monkeypatch: pytest.MonkeyPatch, running: Installed, printed: str
) -> None:
    monkeypatch.setattr(self_update, 'installed', lambda: running)
    # Other arguments are ignored, as with argparse's own `version` action: no prompt runs, no database is made.
    result = run_cli('--version', '--database', str(tmp_path / 'config.db'), '-p', 'hello', cwd=tmp_path)
    assert (result.returncode, result.stdout, result.stderr) == (0, f'{printed}\n', '')
    assert not (tmp_path / 'config.db').exists()


def test_version_names_the_legacy_clai2_command(
    tmp_path: Path, run_cli: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `uv tool install pydantic-clai2` environment runs Pax as `clai2`, so `--version` says so."""
    monkeypatch.setattr(sys, 'prefix', str(tmp_path / 'tools' / 'pydantic-clai2'))
    monkeypatch.setattr(self_update, 'installed', lambda: Installed(version='0.73.0'))
    assert run_cli('--version').stdout == 'clai2 0.73.0\n'


def test_version_reads_the_installed_distribution(run_cli: CliRunner) -> None:
    assert run_cli('--version').stdout == f'pax {self_update.installed().version}\n'


def test_first_model_window_lookup_runs_off_the_event_loop(
    tmp_path: Path, run_cli: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first lookup imports the model's provider SDK: on the event loop, keys typed at the new prompt would wait."""
    on_event_loop: list[bool] = []

    def recorded(model: str) -> ModelProfile:
        on_event_loop.append(threading.current_thread() is threading.main_thread())
        return infer_model_profile(model)

    monkeypatch.setattr(session_runtime, 'infer_model_profile', recorded)
    result = run_cli('--database', str(tmp_path / 'config.db'), '--model', 'openai:gpt-5', cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert on_event_loop[0] is False


@pytest.mark.parametrize('args', [[], ['config', 'show']])
def test_cli_recovers_unsupported_theme_without_losing_unknown_settings(
    tmp_path: Path, run_cli: CliRunner, args: list[str]
) -> None:
    path = tmp_path / 'config.db'
    saved = {
        'display.theme': '"light"',
        'future.setting': '{"enabled":true}',
        'display.thinking': 'false',
        'model': '"test"',
    }
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute('PRAGMA user_version = 1')
        connection.execute('CREATE TABLE settings (key TEXT PRIMARY KEY, value_json TEXT NOT NULL)')
        connection.executemany('INSERT INTO settings VALUES (?, ?)', saved.items())
    result = run_cli('--database', str(path), *args, cwd=tmp_path)
    assert result.returncode == 2, result.stderr
    assert 'Unknown theme: light' in result.stderr
    with closing(sqlite3.connect(path)) as connection:
        assert dict(connection.execute('SELECT key, value_json FROM settings')) == saved
    for command in (['config', 'reset', 'display.theme'], args):
        result = run_cli('--database', str(path), *command)
        assert result.returncode == 0, result.stderr
    del saved['display.theme']
    store = SettingsStore(path)
    assert store.overrides() == {'display.thinking': False, 'model': 'test'}
    assert not store.load().thinking
    with closing(sqlite3.connect(path)) as connection:
        assert dict(connection.execute('SELECT key, value_json FROM settings')) == saved


@pytest.mark.parametrize(
    'args',
    [
        ['sessions', 'foo'],
        ['sessions', '-p', 'hi'],
        ['sessions', '--model', 'x'],
        ['--model', 'x', 'sessions'],
        ['--prompt', 'hi', 'sessions'],
        ['--web', 'sessions'],
        ['--request-limit', '3', 'sessions'],
        ['--database', 'unused.db', 'sessions'],
    ],
)
def test_sessions_rejects_ignored_arguments(run_cli: CliRunner, args: list[str]) -> None:
    result = run_cli(*args)
    assert result.returncode == 2
    assert '`sessions` does not accept arguments or chat/settings options' in result.stderr


def test_cli_startup_interrupt(tmp_path: Path, run_cli: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupted(coroutine: Coroutine[object, object, object]) -> None:
        coroutine.close()
        raise KeyboardInterrupt

    monkeypatch.setattr(asyncio, 'run', interrupted)
    assert run_cli('--database', str(tmp_path / 'config.db'), cwd=tmp_path).returncode == 0


@pytest.mark.parametrize(('state', 'enabled'), [('enabled', True), ('disabled', False), ('corrupt', False)])
def test_startup_saved_splash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str, enabled: bool) -> None:
    """A bare launch reads `display.splash` before the heavy imports, and a corrupt database turns it off."""
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    monkeypatch.delenv('CLAI_NO_SPLASH', raising=False)
    monkeypatch.setenv('PYDANTIC_AI_NO_BANNER', '1')  # `main` sets it; this restores it after the test.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['clai2'])
    path = tmp_path / 'pydantic-clai2' / 'config.db'
    store = SettingsStore(path)
    if state == 'corrupt':
        path.write_text('not sqlite')
    else:
        store.set('display.splash', state == 'enabled')
    requested: list[bool] = []

    class RecordingSplash(Splash):
        def __init__(self, *, enabled: bool = True) -> None:
            requested.append(enabled)
            super().__init__(enabled=enabled)

    def run(*, splash: Splash | None = None) -> None:
        assert isinstance(splash, RecordingSplash)

    monkeypatch.setattr(pydantic_ai_pax.ui.rendering.splash, 'Splash', RecordingSplash)
    monkeypatch.setattr(cli, 'run', run)
    main()
    assert requested == [enabled]
