"""Exercise the installed entry point in isolated subprocesses."""

import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest

from pydantic_clai2.settings_store import SettingsStore


@pytest.mark.parametrize('args', [[], ['--model', 'test', '--request-limit', '12'], ['--request-limit', '0']])
def test_cli_startup(tmp_path: Path, args: list[str]) -> None:
    env = dict(os.environ, CLAI_NO_SPLASH='1')
    env.pop('CLAI_MODEL', None)
    result = subprocess.run(
        [sys.executable, '-m', 'pydantic_clai2', '--database', str(tmp_path / 'config.db'), *args],
        input='/exit\n',
        text=True,
        capture_output=True,
        env=env,
        timeout=15,
        check=False,
    )
    assert result.returncode == (2 if args == ['--request-limit', '0'] else 0), result.stderr


@pytest.mark.parametrize('args', [[], ['config', 'show']])
def test_cli_preserves_unknown_saved_settings(tmp_path: Path, args: list[str]) -> None:
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
    result = subprocess.run(
        [sys.executable, '-m', 'pydantic_clai2', '--database', str(path), *args],
        input='/exit\n',
        text=True,
        capture_output=True,
        env=dict(os.environ, CLAI_NO_SPLASH='1'),
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    store = SettingsStore(path)
    assert store.overrides() == {'display.thinking': False, 'model': 'test'}
    assert not store.load().thinking
    with closing(sqlite3.connect(path)) as connection:
        assert dict(connection.execute('SELECT key, value_json FROM settings')) == saved


def test_cli_startup_interrupt(tmp_path: Path) -> None:
    script = """
import asyncio
import runpy

def interrupted(coroutine):
    coroutine.close()
    raise KeyboardInterrupt

asyncio.run = interrupted
runpy.run_module('pydantic_clai2', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, '-c', script, '--database', str(tmp_path / 'config.db')],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('state', ['enabled', 'disabled', 'corrupt'])
def test_startup_saved_splash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str) -> None:
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    monkeypatch.delenv('CLAI_NO_SPLASH', raising=False)
    monkeypatch.setenv('CLAI_MODEL', 'test')
    path = tmp_path / 'pydantic-clai2' / 'config.db'
    store = SettingsStore(path)
    if state == 'corrupt':
        path.write_text('not sqlite')
    else:
        store.set('display.splash', state == 'enabled')
    result = subprocess.run(
        [sys.executable, '-m', 'pydantic_clai2'],
        input='/exit\n',
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == (1 if state == 'corrupt' else 0)
