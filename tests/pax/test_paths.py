"""Pax's config and state folders are named `pydantic-ai-pax`, and a folder under the old `pydantic-clai2` name moves there."""

import errno
import os
import shutil
import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

import pytest

from pydantic_ai_pax.config.settings_store import SettingsStore
from pydantic_ai_pax.paths import DIRECTORY_NAME, config_dir, state_dir

Rename = Callable[[str | os.PathLike[str], str | os.PathLike[str]], None]


@pytest.fixture
def config(tmp_path: Path) -> Path:
    """`XDG_CONFIG_HOME`, as the autouse `isolated_environment` fixture sets it."""
    return tmp_path / 'config'


def make_legacy(parent: Path) -> Path:
    """A folder under the old name, with a saved setting and a drop-in plugin."""
    legacy = parent / 'pydantic-clai2'
    SettingsStore(legacy / 'config.db').set('display.splash', False)
    (legacy / 'plugins').mkdir()
    (legacy / 'plugins' / 'search.py').write_text('PLUGIN = 1\n')
    return legacy


def test_fresh_install_uses_pax(tmp_path: Path, config: Path) -> None:
    assert config_dir() == config / DIRECTORY_NAME
    assert state_dir() == tmp_path / 'state' / DIRECTORY_NAME
    assert SettingsStore().path == config / DIRECTORY_NAME / 'config.db'
    assert not (config / 'pydantic-clai2').exists()


def test_legacy_folder_moves_and_leaves_a_symlink(config: Path) -> None:
    legacy = make_legacy(config)

    assert config_dir() == config / DIRECTORY_NAME
    assert (config / DIRECTORY_NAME).is_dir() and not (config / DIRECTORY_NAME).is_symlink()
    assert legacy.is_symlink() and os.readlink(legacy) == DIRECTORY_NAME
    assert SettingsStore().overrides() == {'display.splash': False}
    assert (config_dir() / 'plugins' / 'search.py').read_text() == 'PLUGIN = 1\n'
    # An older Pax reads and writes the same database through the old path.
    SettingsStore(legacy / 'config.db').set('display.splash', True)
    assert SettingsStore().overrides() == {'display.splash': True}


def test_legacy_state_folder_moves(tmp_path: Path) -> None:
    legacy = tmp_path / 'state' / 'pydantic-clai2'
    (legacy / 'aoe').mkdir(parents=True)
    (legacy / 'orchestrator.json').write_text('{}')

    assert state_dir() == tmp_path / 'state' / DIRECTORY_NAME
    assert (state_dir() / 'orchestrator.json').read_text() == '{}'
    assert (state_dir() / 'aoe').is_dir()
    assert legacy.is_symlink()


def test_existing_pax_folder_wins_and_legacy_is_left_alone(config: Path) -> None:
    legacy = make_legacy(config)
    SettingsStore(config / DIRECTORY_NAME / 'config.db').set('display.splash', True)

    assert config_dir() == config / DIRECTORY_NAME
    assert SettingsStore().overrides() == {'display.splash': True}
    assert not legacy.is_symlink()
    assert SettingsStore(legacy / 'config.db').overrides() == {'display.splash': False}


def test_legacy_symlink_is_used_where_it_points(tmp_path: Path, config: Path) -> None:
    """A user's own link, such as into a dotfiles checkout, is not moved: Pax reads the folder it points to."""
    dotfiles = make_legacy(tmp_path / 'dotfiles')
    config.mkdir()
    (config / 'pydantic-clai2').symlink_to(dotfiles, target_is_directory=True)

    assert config_dir() == config / 'pydantic-clai2'
    assert SettingsStore().overrides() == {'display.splash': False}
    assert not (config / DIRECTORY_NAME).exists()
    assert os.readlink(config / 'pydantic-clai2') == str(dotfiles)


def test_dangling_legacy_symlink_is_ignored(config: Path) -> None:
    config.mkdir()
    (config / 'pydantic-clai2').symlink_to(config / 'gone', target_is_directory=True)

    assert config_dir() == config / DIRECTORY_NAME
    assert (config / 'pydantic-clai2').is_symlink()


def test_another_pax_moving_it_first(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Another process renames the folder between this one's check and its rename."""
    legacy = make_legacy(config)
    rename: Rename = os.rename

    def moved_by_another_pax(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        rename(source, target)
        raise FileNotFoundError(errno.ENOENT, 'gone', str(source))

    monkeypatch.setattr(os, 'rename', moved_by_another_pax)

    assert config_dir() == config / DIRECTORY_NAME
    assert SettingsStore().overrides() == {'display.splash': False}
    assert not legacy.exists()


def test_old_folder_gone_before_the_move(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The old folder disappears between this process's check and its rename, and no new one exists: it uses the new
    path rather than the old one that is gone.
    """
    legacy = make_legacy(config)

    def removed_by_another_process(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        shutil.rmtree(source)
        raise FileNotFoundError(errno.ENOENT, 'gone', str(source))

    monkeypatch.setattr(os, 'rename', removed_by_another_process)

    assert config_dir() == config / DIRECTORY_NAME
    assert not legacy.exists()
    assert not (config / DIRECTORY_NAME).exists()


def test_new_folder_appearing_during_the_move(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A newer Pax started fresh between this one's check and its rename: both use the new folder."""
    legacy = make_legacy(config)

    def created_by_another_pax(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        SettingsStore(Path(target) / 'config.db')
        raise OSError(errno.ENOTEMPTY, 'not empty', str(target))

    monkeypatch.setattr(os, 'rename', created_by_another_pax)

    assert config_dir() == config / DIRECTORY_NAME
    assert SettingsStore().overrides() == {}
    assert not legacy.is_symlink()


@pytest.mark.parametrize('code', [errno.EXDEV, errno.EACCES])
def test_failed_move_keeps_the_legacy_folder(config: Path, monkeypatch: pytest.MonkeyPatch, code: int) -> None:
    legacy = make_legacy(config)

    def fails(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        raise OSError(code, os.strerror(code), str(source))

    monkeypatch.setattr(os, 'rename', fails)

    assert config_dir() == legacy
    assert SettingsStore().overrides() == {'display.splash': False}
    assert not (config / DIRECTORY_NAME).exists()
    assert not legacy.is_symlink()


def test_failed_symlink_still_uses_the_moved_folder(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Where creating a symlink needs privileges, the folder still moves; only the link is missing."""
    legacy = make_legacy(config)

    def no_symlinks(self: Path, target: str | os.PathLike[str], target_is_directory: bool = False) -> None:
        raise OSError(errno.EPERM, 'symlinks need privileges', str(self))

    monkeypatch.setattr(Path, 'symlink_to', no_symlinks)

    assert config_dir() == config / DIRECTORY_NAME
    assert SettingsStore().overrides() == {'display.splash': False}
    assert not legacy.exists()


def test_open_database_survives_the_move(config: Path) -> None:
    """An older Pax holding the database open keeps reading and writing it, `-wal` and `-shm` included."""
    legacy = make_legacy(config)
    with closing(sqlite3.connect(legacy / 'config.db')) as held:
        assert held.execute('PRAGMA journal_mode=WAL').fetchone() == ('wal',)
        with held:
            held.execute("INSERT INTO settings VALUES ('before', '1')")

        assert config_dir() == config / DIRECTORY_NAME
        assert (config / DIRECTORY_NAME / 'config.db-wal').exists()
        assert (config / DIRECTORY_NAME / 'config.db-shm').exists()

        with held:
            held.execute("INSERT INTO settings VALUES ('after', '2')")
        with closing(sqlite3.connect(config / DIRECTORY_NAME / 'config.db')) as fresh:
            rows = fresh.execute('SELECT key FROM settings ORDER BY key').fetchall()
    assert rows == [('after',), ('before',), ('display.splash',)]
