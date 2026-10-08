"""Git and GitHub CLI subprocess UTF-8 decoding under non-UTF-8 ambient locales."""

import subprocess
import sys
from pathlib import Path

import pytest

from pydantic_ai.exceptions import UserError
from pydantic_clai2.gh_cli import gh_token, start_login
from pydantic_clai2.runtime.project_identity import project_identity
from pydantic_clai2.runtime.worktrees import _git


@pytest.fixture(autouse=True)
def non_utf8_subprocess_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a platform whose default subprocess text codec is non-UTF-8 (e.g. Windows cp1251)."""
    original_init = subprocess.Popen.__init__

    def patched_init(self: subprocess.Popen, *args: object, **kwargs: object) -> None:
        if kwargs.get('encoding') is None and (kwargs.get('text') or kwargs.get('universal_newlines')):
            kwargs['encoding'] = 'cp1251'
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(subprocess.Popen, '__init__', patched_init)
    if hasattr(subprocess, '_text_encoding'):
        monkeypatch.setattr(subprocess, '_text_encoding', lambda: 'cp1251')


@pytest.mark.parametrize('repo_name', ['После', 'Иван'])
def test_project_identity_decodes_non_ascii_paths_and_branches_as_utf8(tmp_path: Path, repo_name: str) -> None:
    """Git output is always UTF-8; project_identity decodes it as UTF-8 even when locale codec is cp1251.

    'После' triggers mojibake under cp1251, while 'Иван' contains byte 0x98 which is undefined in cp1251
    and causes UnicodeDecodeError if decoded with the ambient locale codec.
    """
    repo = tmp_path / repo_name
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'config', 'user.name', 'Test'], check=True)
    subprocess.run(['git', '-C', str(repo), 'config', 'user.email', 'test@example.com'], check=True)
    subprocess.run(['git', '-C', str(repo), 'commit', '--allow-empty', '-m', 'init'], check=True)
    branch_name = f'ветка_{repo_name}'
    subprocess.run(['git', '-C', str(repo), 'checkout', '-b', branch_name], check=True)

    identity = project_identity(str(repo))

    assert identity.name == repo_name
    assert identity.root == str(repo.resolve())
    assert identity.checkout == branch_name


def test_worktrees_git_helper_decodes_as_utf8(tmp_path: Path) -> None:
    """The shared worktree _git() helper decodes git output as UTF-8."""
    repo = tmp_path / 'Иван'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'config', 'user.name', 'Test'], check=True)
    subprocess.run(['git', '-C', str(repo), 'config', 'user.email', 'test@example.com'], check=True)
    subprocess.run(['git', '-C', str(repo), 'commit', '--allow-empty', '-m', 'init'], check=True)
    subprocess.run(['git', '-C', str(repo), 'checkout', '-b', 'ветка_ивана'], check=True)

    assert _git('-C', str(repo), 'rev-parse', '--show-toplevel') == str(repo.resolve())
    assert _git('-C', str(repo), 'branch', '--show-current') == 'ветка_ивана'


def test_gh_cli_decodes_output_as_utf8(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """gh auth token and gh auth login decode gh output as UTF-8."""
    fake_gh = tmp_path / 'fake_gh.py'
    fake_gh.write_text(
        'import sys\n'
        'if sys.argv[1:3] == ["auth", "token"]:\n'
        '    sys.stdout.buffer.write("токен_иван\\n".encode("utf-8"))\n'
        '    sys.exit(0)\n'
        'elif sys.argv[1:3] == ["auth", "login"]:\n'
        '    sys.stdout.buffer.write(\n'
        '        "! One-time code (ABCD-1234) copied to clipboard\\n"\n'
        '        "https://github.com/login/device\\n"\n'
        '        "Logged in as Иван\\n".encode("utf-8")\n'
        '    )\n'
        '    sys.exit(0)\n',
        encoding='utf-8',
    )
    monkeypatch.setattr('pydantic_clai2.gh_cli.gh_command', lambda: [sys.executable, str(fake_gh)])

    token = gh_token('github.com')
    assert token == 'токен_иван'

    monkeypatch.setattr('pydantic_clai2.gh_cli.gh_command', lambda: None)
    with pytest.raises(UserError, match='not installed'):
        gh_token('github.com')
    monkeypatch.setattr('pydantic_clai2.gh_cli.gh_command', lambda: [sys.executable, str(fake_gh)])

    login = start_login('github.com')
    assert login is not None and not isinstance(login, str)
    assert login.code == 'ABCD-1234'
    assert login.finish() == 'Signed in to GitHub as Иван.'
