"""Storage and public API error boundaries."""

import io
import sqlite3
import sys
import warnings
from contextlib import closing
from pathlib import Path

import pytest

import pydantic_clai2
import pydantic_clai2.__main__
import pydantic_clai2._cli
from pydantic_clai2.commands import config_completions
from pydantic_clai2.settings_store import SettingsStore
from pydantic_clai2.splash import Splash


def test_public_errors(tmp_path: Path) -> None:
    with pytest.raises(AttributeError):
        assert pydantic_clai2.missing
    assert callable(pydantic_clai2.__main__.main)
    assert list(config_completions(['set', '']))
    path = tmp_path / 'future.db'
    with closing(sqlite3.connect(path)) as connection:
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


@pytest.mark.parametrize(('warnoptions', 'shown'), [([], []), (['default'], ['for developers'])])
def test_entry_point_quiets_user_warnings_unless_requested(
    monkeypatch: pytest.MonkeyPatch, warnoptions: list[str], shown: list[str]
) -> None:
    def run(*, splash: Splash | None = None) -> None:
        warnings.warn('for developers', UserWarning)

    monkeypatch.setenv('PYDANTIC_AI_NO_BANNER', '1')
    monkeypatch.setattr(sys, 'argv', ['clai2', 'config'])
    monkeypatch.setattr(sys, 'warnoptions', warnoptions)
    monkeypatch.setattr(pydantic_clai2._cli, 'run', run)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        filters = list(warnings.filters)
        pydantic_clai2.__main__.main()
        assert warnings.filters == filters
    assert [str(warning.message) for warning in caught] == shown
