"""Windows executable selection, exercised with real filesystem entries on each platform."""

import os
from pathlib import Path

import pytest

from pydantic_clai2.plugins import _git
from pydantic_clai2.plugins._git import git_executable

from .test_plugin_loader import Harness


def test_windows_git_ignores_working_directory_and_relative_path_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'git.exe').write_text('untrusted')
    (workspace / 'bin').mkdir()
    (workspace / 'bin' / 'git.exe').write_text('untrusted relative PATH entry')
    empty = tmp_path / 'empty'
    empty.mkdir()
    not_directory = tmp_path / 'file'
    not_directory.touch()
    trusted = tmp_path / 'trusted'
    trusted.mkdir()
    (trusted / 'git.exe').write_text('trusted')
    later = tmp_path / 'later'
    later.mkdir()
    (later / 'git.exe').write_text('later PATH entry')
    monkeypatch.chdir(workspace)
    monkeypatch.setenv(
        'PATH',
        os.pathsep.join(
            (
                '',
                '.',
                str(workspace),
                'bin',
                str(tmp_path / 'missing'),
                str(not_directory),
                str(empty),
                str(trusted),
                str(later),
            )
        ),
    )
    assert git_executable(windows=True) == str(trusted.resolve() / 'git.exe')


async def test_missing_windows_git_never_falls_back_to_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'git.exe').write_text('untrusted')
    monkeypatch.chdir(workspace)
    monkeypatch.setenv('PATH', os.pathsep.join(('', '.', str(workspace))))
    monkeypatch.setattr(_git, 'git_executable', lambda: git_executable(windows=True))
    harness = Harness(tmp_path)
    with pytest.raises(ValueError, match='Git is required'):
        await harness.loader.command(['add', 'https://example.invalid/plugin.git'])
    assert harness.store.plugins() == []
    assert not (harness.store.plugins_dir / '_git' / 'plugin').exists()
    assert (workspace / 'git.exe').read_text() == 'untrusted'


@pytest.mark.skipif(os.name == 'nt', reason='requires creating a directory symlink without special privileges')
def test_windows_git_ignores_aliases_of_the_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'git.exe').touch()
    alias = tmp_path / 'alias'
    alias.symlink_to(workspace, target_is_directory=True)
    monkeypatch.chdir(workspace)
    monkeypatch.setenv('PATH', str(alias))
    with pytest.raises(FileNotFoundError):
        git_executable(windows=True)


@pytest.mark.skipif(os.name == 'nt', reason='uses POSIX executable script fixtures')
async def test_clone_uses_the_resolved_windows_executable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / 'workspace'
    trusted = tmp_path / 'trusted'
    for directory, message in ((workspace, 'hijacked'), (trusted, 'trusted')):
        directory.mkdir()
        executable = directory / 'git.exe'
        executable.write_text(f"#!/bin/sh\nprintf '%s' '{message}' >&2\n")
        executable.chmod(0o755)
    monkeypatch.chdir(workspace)
    monkeypatch.setenv('PATH', os.pathsep.join((str(workspace), str(trusted))))
    monkeypatch.setattr(_git, 'git_executable', lambda: git_executable(windows=True))
    destination = tmp_path / 'checkout'
    destination.mkdir()
    assert await _git.clone_repository('https://example.invalid/plugin.git', destination) == (0, b'trusted')
