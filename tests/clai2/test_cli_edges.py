"""Exercise the command line: one installed entry point smoke test, the rest in-process."""

import asyncio
import os
import sqlite3
import subprocess
import sys
from collections.abc import Coroutine
from contextlib import closing
from pathlib import Path

import pytest

import pydantic_clai2.ui.rendering.splash
from pydantic_clai2.__main__ import main
from pydantic_clai2.cli import _cli
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.ui.rendering.splash import Splash
from tests.clai2.cli_runner import CliRunner


@pytest.mark.subprocess(
    reason='smoke-tests the installed `python -m pydantic_clai2` entry point; the other CLI tests run in-process'
)
def test_installed_entry_point_starts_and_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`python -m pydantic_clai2` without a model: the one launch here that crosses the process boundary."""
    # Subprocess coverage adds import overhead to the CLI startup hang guard.
    for name in tuple(os.environ):
        if name.startswith('COVERAGE_'):
            monkeypatch.delenv(name)
    env = dict(os.environ, CLAI_NO_SPLASH='1')
    env.pop('CLAI_MODEL', None)
    result = subprocess.run(
        [sys.executable, '-m', 'pydantic_clai2', '--database', str(tmp_path / 'config.db')],
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

    monkeypatch.setattr(pydantic_clai2.ui.rendering.splash, 'Splash', RecordingSplash)
    monkeypatch.setattr(_cli, 'run', run)
    main()
    assert requested == [enabled]
