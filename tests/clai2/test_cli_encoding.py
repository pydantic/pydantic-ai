"""CLI output is UTF-8 even when the platform's default text codec is not."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import pydantic_clai2.runtime.project_identity as project_identity_module
import pydantic_clai2.runtime.worktrees as worktrees_module
from pydantic_clai2.gh_cli import gh_token, start_login
from pydantic_clai2.runtime.project_identity import ProjectIdentity, project_identity
from pydantic_clai2.runtime.worktrees import current_worktree, open_worktree
from tests.clai2.test_session_browser import repository


@pytest.fixture
def non_utf8_locale(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch the default codec selection, preserving real subprocess pipe decoding
    # even when the interpreter was started in UTF-8 mode.
    monkeypatch.setattr(subprocess, '_text_encoding', lambda: 'cp1251')


@pytest.fixture
def windows_non_utf8_locale(monkeypatch: pytest.MonkeyPatch, non_utf8_locale: None) -> None:
    monkeypatch.setattr(project_identity_module, 'sys', SimpleNamespace(platform='win32'))
    monkeypatch.setattr(worktrees_module, 'sys', SimpleNamespace(platform='win32', stdin=sys.stdin, stderr=sys.stderr))


@pytest.mark.parametrize('name', ['После', 'Иван'])
def test_project_identity_with_non_ascii_paths_and_branch(
    tmp_path: Path, name: str, windows_non_utf8_locale: None
) -> None:
    """Cover both mojibake and the UTF-8 byte 0x98, which cp1251 cannot decode."""
    repo = tmp_path / name
    linked = tmp_path / f'{name}-checkout'
    branch = f'feature/{name}'
    repository(repo, linked=linked, branch=branch)

    assert project_identity(str(linked)) == ProjectIdentity(
        key=str((repo / '.git').resolve()), name=name, root=str(linked.resolve()), checkout=branch
    )


def test_worktree_with_non_ascii_repository_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, windows_non_utf8_locale: None
) -> None:
    repo = tmp_path / 'Иван'
    repository(repo, linked=tmp_path / 'linked', branch='feature')
    monkeypatch.chdir(repo)

    worktree = open_worktree(name='encoding')
    assert worktree.path == repo.resolve() / '.worktrees' / 'encoding'
    assert worktree.created
    assert open_worktree(name='encoding').path == worktree.path
    monkeypatch.chdir(worktree.path)
    current = current_worktree()
    assert current is not None
    assert current.path == worktree.path
    assert current.branch == 'clai-encoding'


@pytest.mark.skipif(sys.platform == 'win32', reason='checks native POSIX text decoding for Git output')
def test_git_output_uses_posix_default_codec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, non_utf8_locale: None
) -> None:
    repo = tmp_path / 'repo'
    linked = tmp_path / 'checkout'
    branch = 'feature/Иван'
    repository(repo, linked=linked, branch=branch)

    subprocess.run(['git', '-C', str(repo), 'pack-refs', '--all'], check=True, capture_output=True)
    packed_refs = repo / '.git' / 'packed-refs'
    packed_refs.write_bytes(packed_refs.read_bytes().replace(branch.encode(), branch.encode('cp1251')))
    (repo / '.git' / 'worktrees' / linked.name / 'HEAD').write_bytes(
        b'ref: refs/heads/' + branch.encode('cp1251') + b'\n'
    )

    identity = project_identity(str(linked))
    assert identity.name == 'repo'
    assert identity.checkout == 'feature/Иван'

    monkeypatch.chdir(linked)
    current = current_worktree()
    assert current is not None
    assert current.path == linked.resolve()
    assert current.branch == 'feature/Иван'


@pytest.mark.parametrize('command', ['token', 'login'])
@pytest.mark.subprocess(reason='Verify UTF-8 decoding of real subprocess pipes.')
def test_gh_with_non_ascii_diagnostics(monkeypatch: pytest.MonkeyPatch, command: str, non_utf8_locale: None) -> None:
    """Run a stand-in CLI so the actual run/Popen pipes decode UTF-8 diagnostics."""
    for name in tuple(os.environ):
        if name.startswith('COVERAGE_'):
            monkeypatch.delenv(name)
    diagnostic = 'Ошибка: Иван\n' if command == 'token' else 'Ошибка: После\n'
    script = (
        'import sys\n'
        f'sys.stderr.buffer.write({diagnostic!r}.encode("utf-8"))\n'
        'if sys.argv[1:3] == ["auth", "token"]: print("gho_saved")\n'
        'sys.exit(0 if sys.argv[1:3] == ["auth", "token"] else 1)\n'
    )
    monkeypatch.setattr('pydantic_clai2.gh_cli.gh_command', lambda: [sys.executable, '-c', script])
    if command == 'token':
        assert gh_token('github.com') == 'gho_saved'
    else:
        assert start_login('github.com') == 'gh auth login failed: Ошибка: После'
