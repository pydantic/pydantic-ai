"""Storage and public API error boundaries."""

import io
import sqlite3
import sys
from pathlib import Path

import pytest

import pydantic_clai2
import pydantic_clai2.__main__
from pydantic_clai2.commands import config_completions
from pydantic_clai2.settings_store import SettingsStore
from pydantic_clai2.splash import Splash


def test_public_errors(tmp_path: Path) -> None:
    with pytest.raises(AttributeError):
        assert pydantic_clai2.missing
    assert callable(pydantic_clai2.__main__.main)
    assert list(config_completions(['set', '']))
    path = tmp_path / 'future.db'
    with sqlite3.connect(path) as connection:
        connection.execute('PRAGMA user_version = 99')
    with pytest.raises(ValueError, match='Unsupported'):
        SettingsStore(path)


def test_splash_broken_stream_and_replaced_output(monkeypatch: pytest.MonkeyPatch) -> None:
    class Terminal(io.StringIO):
        def isatty(self) -> bool:
            return True

        def write(self, text: str) -> int:
            raise OSError('closed terminal')

    monkeypatch.setenv('COLUMNS', '80')
    monkeypatch.setenv('LINES', '30')
    monkeypatch.setenv('TERM', 'xterm')
    monkeypatch.delenv('NO_COLOR', raising=False)
    monkeypatch.setenv('COLORTERM', '16color')
    stream = Terminal()
    monkeypatch.setattr(sys, 'stdout', stream)
    splash = Splash()
    assert '\x1b[35m' in splash.frame(10)
    monkeypatch.setenv('COLORTERM', 'truecolor')
    assert '\x1b[38;2;' in splash.frame(10)
    splash.start()
    monkeypatch.setattr(sys, 'stdout', io.StringIO())
    monkeypatch.setattr(sys, 'stderr', io.StringIO())
    with pytest.raises(OSError):
        splash.stop()
