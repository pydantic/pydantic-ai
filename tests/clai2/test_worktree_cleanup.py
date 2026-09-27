"""Worktree shutdown keeps user work unless removal is explicitly confirmed."""

import subprocess
from pathlib import Path

import pytest

from pydantic_clai2.worktrees import offer_worktree_cleanup


def git(directory: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(directory), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    git(repo, 'init')
    git(repo, 'config', 'user.name', 'Test')
    git(repo, 'config', 'user.email', 'test@example.com')
    git(repo, 'commit', '--allow-empty', '-m', 'Initial')
    linked = tmp_path / 'linked checkout'
    git(repo, 'worktree', 'add', '-b', 'feature', str(linked))
    monkeypatch.chdir(linked)
    monkeypatch.setattr('sys.stdin.isatty', lambda: True)
    return linked


@pytest.mark.parametrize('answer', ['', 'n', 'no', 'perhaps'])
def test_keep_is_default(checkout: Path, monkeypatch: pytest.MonkeyPatch, answer: str) -> None:
    def respond(prompt: str) -> str:
        assert str(checkout) in prompt
        assert '[y/N]' in prompt
        return answer

    monkeypatch.setattr('builtins.input', respond)
    offer_worktree_cleanup()
    assert checkout.exists()
    assert Path.cwd() == checkout


@pytest.mark.parametrize('error', [EOFError, KeyboardInterrupt])
def test_cancel_keeps(checkout: Path, monkeypatch: pytest.MonkeyPatch, error: type[BaseException]) -> None:
    def respond(prompt: str) -> str:
        raise error

    monkeypatch.setattr('builtins.input', respond)
    offer_worktree_cleanup()
    assert checkout.exists()


@pytest.mark.parametrize('answer', ['y', ' YES '])
def test_remove_preserves_branch(checkout: Path, monkeypatch: pytest.MonkeyPatch, answer: str) -> None:
    def respond(prompt: str) -> str:
        return answer

    monkeypatch.setattr('builtins.input', respond)
    offer_worktree_cleanup()
    assert not checkout.exists()
    repo = checkout.parent / 'repo'
    assert Path.cwd() == repo
    assert git(repo, 'rev-parse', 'feature') == git(repo, 'rev-parse', 'HEAD')
    assert str(checkout) not in git(repo, 'worktree', 'list')


@pytest.mark.parametrize('condition', ['dirty', 'locked'])
def test_git_refusal_keeps_checkout(
    checkout: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], condition: str
) -> None:
    if condition == 'dirty':
        (checkout / 'unsaved.txt').write_text('keep me')
    else:
        git(checkout, 'worktree', 'lock', str(checkout))

    def respond(prompt: str) -> str:
        return 'yes'

    monkeypatch.setattr('builtins.input', respond)
    offer_worktree_cleanup()
    assert checkout.exists()
    assert Path.cwd() == checkout
    assert 'Worktree kept' in capsys.readouterr().err
    if condition == 'dirty':
        assert (checkout / 'unsaved.txt').read_text() == 'keep me'


def test_missing_git_keeps_checkout(checkout: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PATH', '')
    offer_worktree_cleanup()
    assert checkout.exists()


def test_removal_os_error_keeps_checkout(
    checkout: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def respond(prompt: str) -> str:
        monkeypatch.setenv('PATH', '')
        return 'yes'

    monkeypatch.setattr('builtins.input', respond)
    offer_worktree_cleanup()
    assert checkout.exists()
    assert Path.cwd() == checkout
    assert 'Worktree kept' in capsys.readouterr().err


@pytest.mark.parametrize('location', ['main', 'outside', 'pipe'])
def test_no_prompt(checkout: Path, monkeypatch: pytest.MonkeyPatch, location: str) -> None:
    if location == 'main':
        monkeypatch.chdir(checkout.parent / 'repo')
    elif location == 'outside':
        monkeypatch.chdir(checkout.parent)
    else:
        monkeypatch.setattr('sys.stdin.isatty', lambda: False)

    def respond(prompt: str) -> str:
        pytest.fail('Unexpected cleanup prompt')

    monkeypatch.setattr('builtins.input', respond)
    offer_worktree_cleanup()
    assert checkout.exists()
