"""Worktree shutdown removes an unchanged checkout CLAI created, and keeps changed work unless removal is confirmed."""

import os
import subprocess
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from pydantic_clai2 import _app
from pydantic_clai2.cli import _cli
from pydantic_clai2.cli.self_update import Relaunch
from pydantic_clai2.runtime.worktrees import Worktree, current_worktree, offer_worktree_cleanup, open_worktree


def git(directory: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(directory), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def no_prompt(prompt: str) -> str:
    pytest.fail('Unexpected cleanup prompt')  # pragma: no cover


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    git(repo, 'init')
    git(repo, 'config', 'user.name', 'Test')
    git(repo, 'config', 'user.email', 'test@example.com')
    (repo / 'tracked.txt').write_text('committed')
    git(repo, 'add', 'tracked.txt')
    git(repo, 'commit', '-m', 'Initial')
    monkeypatch.setattr('sys.stdin.isatty', lambda: True)
    return repo


@pytest.fixture
def checkout(repo: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    linked = repo.parent / 'linked checkout'
    git(repo, 'worktree', 'add', '-b', 'feature', str(linked))
    monkeypatch.chdir(linked)
    return linked


@pytest.fixture
def launched(checkout: Path) -> Worktree:
    """The hand-made checkout as CLAI saw it at launch, with a commit made since."""
    worktree = Worktree(path=checkout, branch='feature', head=git(checkout, 'rev-parse', 'HEAD'), created=False)
    assert current_worktree() == worktree
    git(checkout, 'commit', '--allow-empty', '-m', 'Work')
    return worktree


@pytest.fixture
def created(repo: Path, monkeypatch: pytest.MonkeyPatch) -> Worktree:
    monkeypatch.chdir(repo)
    worktree = open_worktree(name='task')
    assert worktree.created and worktree.new_branch
    monkeypatch.chdir(worktree.path)
    return replace(worktree, remove_if_unchanged=True)


def test_notice_names_what_exit_does(created: Worktree) -> None:
    assert created.notice == f'Worktree: {created.path} (branch: clai-task). Removed on exit if unchanged.'
    kept = replace(created, remove_if_unchanged=False)
    assert kept.notice == f'Worktree: {created.path} (branch: clai-task). Kept unless removal is confirmed on exit.'


def test_unchanged_checkout_is_kept_unless_marked(created: Worktree, monkeypatch: pytest.MonkeyPatch) -> None:
    """A checkout the CLI did not mark, as with piped input when it was opened, is kept without asking."""
    monkeypatch.setattr('builtins.input', no_prompt)
    offer_worktree_cleanup(worktree=replace(created, remove_if_unchanged=False))
    assert created.path.exists()


@pytest.mark.parametrize('answer', ['', 'n', 'no', 'perhaps'])
def test_keep_is_default(launched: Worktree, monkeypatch: pytest.MonkeyPatch, answer: str) -> None:
    def respond(prompt: str) -> str:
        assert str(launched.path) in prompt
        assert '[y/N]' in prompt
        return answer

    monkeypatch.setattr('builtins.input', respond)
    offer_worktree_cleanup(worktree=launched)
    assert launched.path.exists()
    assert Path.cwd() == launched.path


@pytest.mark.parametrize('error', [EOFError, KeyboardInterrupt])
def test_cancel_keeps(launched: Worktree, monkeypatch: pytest.MonkeyPatch, error: type[BaseException]) -> None:
    def respond(prompt: str) -> str:
        raise error

    monkeypatch.setattr('builtins.input', respond)
    offer_worktree_cleanup(worktree=launched)
    assert launched.path.exists()


@pytest.mark.parametrize('answer', ['y', ' YES '])
def test_remove_preserves_branch(
    launched: Worktree, repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], answer: str
) -> None:
    def respond(prompt: str) -> str:
        return answer

    work = git(launched.path, 'rev-parse', 'HEAD')
    monkeypatch.setattr('builtins.input', respond)
    offer_worktree_cleanup(worktree=launched)
    assert not launched.path.exists()
    assert Path.cwd() == repo
    assert git(repo, 'rev-parse', 'feature') == work
    assert str(launched.path) not in git(repo, 'worktree', 'list')
    assert 'Branch kept.' in capsys.readouterr().out


@pytest.mark.parametrize('condition', ['dirty', 'locked'])
def test_git_refusal_keeps_checkout(
    launched: Worktree, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], condition: str
) -> None:
    if condition == 'dirty':
        (launched.path / 'unsaved.txt').write_text('keep me')
    else:
        git(launched.path, 'worktree', 'lock', str(launched.path))

    def respond(prompt: str) -> str:
        return 'yes'

    monkeypatch.setattr('builtins.input', respond)
    offer_worktree_cleanup(worktree=launched)
    assert launched.path.exists()
    assert Path.cwd() == launched.path
    output = capsys.readouterr()
    assert 'Worktree kept' in output.err
    assert 'Removed' not in output.out
    if condition == 'dirty':
        assert (launched.path / 'unsaved.txt').read_text() == 'keep me'


def test_missing_git_keeps_checkout(launched: Worktree, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PATH', '')
    monkeypatch.setattr('builtins.input', no_prompt)
    offer_worktree_cleanup(worktree=launched)
    assert launched.path.exists()


def test_removal_os_error_keeps_checkout(
    launched: Worktree, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def respond(prompt: str) -> str:
        monkeypatch.setenv('PATH', '')
        return 'yes'

    monkeypatch.setattr('builtins.input', respond)
    offer_worktree_cleanup(worktree=launched)
    assert launched.path.exists()
    assert Path.cwd() == launched.path
    assert 'Worktree kept' in capsys.readouterr().err


@pytest.mark.parametrize('location', ['main', 'outside'])
def test_only_linked_checkouts_are_worktrees(repo: Path, monkeypatch: pytest.MonkeyPatch, location: str) -> None:
    monkeypatch.chdir(repo if location == 'main' else repo.parent)
    assert current_worktree() is None


def test_detached_checkout(checkout: Path) -> None:
    git(checkout, 'checkout', '--detach')
    worktree = current_worktree()
    assert worktree is not None
    assert worktree.branch == 'detached HEAD'


def test_piped_input_keeps_without_prompt(created: Worktree, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('sys.stdin.isatty', lambda: False)
    monkeypatch.setattr('builtins.input', no_prompt)
    offer_worktree_cleanup(worktree=created)
    assert created.path.exists()


def test_no_worktree_is_a_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('builtins.input', no_prompt)
    offer_worktree_cleanup(worktree=None)


def test_unchanged_hand_made_checkout_is_kept_silently(
    checkout: Path, repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    worktree = current_worktree()
    monkeypatch.setattr('builtins.input', no_prompt)
    offer_worktree_cleanup(worktree=worktree)
    assert checkout.exists()
    assert git(repo, 'branch', '--list', 'feature')
    assert capsys.readouterr() == ('', '')


def test_unchanged_created_checkout_is_removed_silently(
    created: Worktree, repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (created.path / 'ignored.log').write_text('build output')
    (repo / '.git/info/exclude').write_text('*.log\n')
    monkeypatch.setattr('builtins.input', no_prompt)
    offer_worktree_cleanup(worktree=created)
    assert not created.path.exists()
    assert Path.cwd() == repo
    assert str(created.path) not in git(repo, 'worktree', 'list')
    assert not git(repo, 'branch', '--list', created.branch)
    assert capsys.readouterr() == ('', '')


def test_unchanged_created_checkout_keeps_a_reused_branch(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    git(repo, 'branch', 'clai-task')
    monkeypatch.chdir(repo)
    worktree = open_worktree(name='task')
    assert worktree.created and not worktree.new_branch
    monkeypatch.chdir(worktree.path)
    offer_worktree_cleanup(worktree=replace(worktree, remove_if_unchanged=True))
    assert not worktree.path.exists()
    assert git(repo, 'branch', '--list', 'clai-task')


def test_unmerged_branch_is_kept(created: Worktree, repo: Path) -> None:
    """`git branch -d` refuses once the main checkout moves to history without the branch's commit."""
    git(repo, 'checkout', '--orphan', 'elsewhere')
    git(repo, 'commit', '--allow-empty', '-m', 'Unrelated')
    offer_worktree_cleanup(worktree=created)
    assert not created.path.exists()
    assert git(repo, 'branch', '--list', created.branch)


def test_unchanged_locked_checkout_reports_refusal(
    created: Worktree, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    git(created.path, 'worktree', 'lock', str(created.path))
    monkeypatch.setattr('builtins.input', no_prompt)
    offer_worktree_cleanup(worktree=created)
    assert created.path.exists()
    assert Path.cwd() == created.path
    assert 'Worktree kept' in capsys.readouterr().err


@pytest.mark.parametrize('change', ['untracked', 'modified', 'commit'])
def test_changed_created_checkout_still_asks(
    created: Worktree, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], change: str
) -> None:
    if change == 'untracked':
        (created.path / 'new.txt').write_text('draft')
    elif change == 'modified':
        (created.path / 'tracked.txt').write_text('edited')
    else:
        git(created.path, 'commit', '--allow-empty', '-m', 'Work')
    prompts: list[str] = []

    def respond(prompt: str) -> str:
        prompts.append(prompt)
        return ''

    monkeypatch.setattr('builtins.input', respond)
    offer_worktree_cleanup(worktree=created)
    assert len(prompts) == 1
    assert created.path.exists()
    assert git(created.path, 'branch', '--show-current') == created.branch
    assert 'Worktree kept at' in capsys.readouterr().out


def run_cli(
    monkeypatch: pytest.MonkeyPatch,
    *args: str,
    repo: Path,
    session: Callable[[], None] = lambda: None,
    relaunch: bool = False,
) -> list[str]:
    """Run `clai2` with a stand-in shell; `relaunch` ends it as `/update` does. Returns the prompts asked on exit."""
    prompts: list[str] = []

    async def chat(*_: object, **__: object) -> None:
        session()
        if relaunch:
            raise Relaunch(executable='/b/clai2', session_id='s')

    def respond(prompt: str) -> str:
        prompts.append(prompt)
        return ''

    monkeypatch.setattr(_app, 'chat', chat)

    def execv(path: str, argv: list[str]) -> None:
        pass

    monkeypatch.setattr(os, 'execv', execv)
    monkeypatch.setattr('builtins.input', respond)
    # Beside the repository: a database inside the checkout would be an untracked change.
    database = repo.parent / 'config.db'
    monkeypatch.setattr('sys.argv', ['clai2', '--database', str(database), '-m', 'test', *args])
    _cli.run()
    return prompts


@pytest.mark.parametrize('commit', [False, True])
def test_update_keeps_the_launch_snapshot(repo: Path, monkeypatch: pytest.MonkeyPatch, commit: bool) -> None:
    """The relaunched build starts without `--worktree`, yet still removes an unchanged checkout the first one made."""
    monkeypatch.delenv('CLAI_RELAUNCH_WORKTREE', raising=False)
    monkeypatch.chdir(repo)
    workspace = repo / '.worktrees/task'
    assert run_cli(monkeypatch, '-w', 'task', repo=repo, relaunch=True) == []
    assert Path.cwd() == workspace
    head = git(workspace, 'rev-parse', 'HEAD')
    assert os.environ['CLAI_RELAUNCH_WORKTREE'] == f'{head} new-branch'

    def work() -> None:
        assert 'CLAI_RELAUNCH_WORKTREE' not in os.environ
        if commit:
            git(workspace, 'commit', '--allow-empty', '-m', 'Work')

    prompts = run_cli(monkeypatch, repo=repo, session=work)
    assert len(prompts) == commit
    assert workspace.exists() == commit
    assert bool(git(repo, 'branch', '--list', 'clai-task')) == commit


def test_relaunch_from_a_kept_checkout_hands_nothing_over(
    repo: Path, checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('CLAI_RELAUNCH_WORKTREE', raising=False)
    assert run_cli(monkeypatch, repo=repo, relaunch=True) == []
    assert 'CLAI_RELAUNCH_WORKTREE' not in os.environ


def test_session_commits_in_a_hand_made_checkout_still_ask(
    repo: Path, checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The snapshot is taken before the session, so its commits count as changes."""

    def work() -> None:
        git(checkout, 'commit', '--allow-empty', '-m', 'Work')

    prompts = run_cli(monkeypatch, repo=repo, session=work)
    assert len(prompts) == 1
    assert checkout.exists()
