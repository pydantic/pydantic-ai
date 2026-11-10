"""Worktree shutdown removes a checkout with nothing to lose, and keeps unpushed work unless removal is confirmed."""

import io
import os
import sqlite3
import subprocess
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass, field, replace
from pathlib import Path
from unittest.mock import AsyncMock

import anyio
import pytest
from prompt_toolkit.input import create_pipe_input
from rich.console import Console
from rich.text import Text

from pydantic_clai2 import app
from pydantic_clai2.cli import cli
from pydantic_clai2.cli.self_update import Relaunch
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.runtime import worktrees
from pydantic_clai2.runtime.worktrees import Worktree, current_worktree, offer_worktree_cleanup, open_worktree
from pydantic_clai2.ui.menus import worktree_cleanup
from pydantic_clai2.ui.menus.question_menu import QuestionMenu
from pydantic_clai2.ui.prompt.prompt_surface import PromptSurface
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.stderr_relay import terminal_stderr


def git(directory: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(directory), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def no_prompt(prompt: str) -> str:
    pytest.fail('Unexpected cleanup prompt')  # pragma: no cover


def test_suppression_follows_checkout_not_path(launched: Worktree, repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database = repo.parent / 'config.db'
    store = SettingsStore(database)
    picker = AsyncMock(return_value='never')
    monkeypatch.setattr(worktrees, 'ask_removal', picker)
    offer_worktree_cleanup(worktree=launched, store=store)
    assert launched.path.exists()

    # A later launch sees another change, but remembers the same checkout's choice.
    reopened = current_worktree()
    assert reopened is not None
    git(launched.path, 'commit', '--allow-empty', '-m', 'More work')
    offer_worktree_cleanup(worktree=reopened, store=SettingsStore(database))
    assert picker.await_count == 1

    # Another checkout, even of the same repository, gets its own question.
    other = repo.parent / 'other'
    git(repo, 'worktree', 'add', '-b', 'other', str(other))
    (other / 'unsaved').write_text('work')
    offer_worktree_cleanup(worktree=replace(reopened, path=other), store=store)
    assert picker.await_count == 2

    # Recreate the original checkout at exactly the same path and on the same branch.
    monkeypatch.chdir(repo)
    git(repo, 'worktree', 'remove', str(launched.path))
    git(repo, 'worktree', 'add', str(launched.path), 'feature')
    (launched.path / 'unsaved').write_text('new work')
    offer_worktree_cleanup(worktree=reopened, store=store)
    assert picker.await_count == 3
    assert launched.path.exists()
    offer_worktree_cleanup(worktree=reopened, store=SettingsStore(database))
    assert picker.await_count == 3


@pytest.mark.parametrize('failure', ['identity', 'read', 'write'])
def test_suppression_failure_keeps_checkout(
    launched: Worktree,
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: str,
) -> None:
    store = SettingsStore(repo.parent / 'config.db')
    if failure == 'identity':
        original_stat = Path.stat

        def stat(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
            if path == launched.path / '.git':
                raise PermissionError('unreadable checkout identity')
            return original_stat(path, follow_symlinks=follow_symlinks)

        monkeypatch.setattr(Path, 'stat', stat)
    elif failure == 'read':
        with closing(sqlite3.connect(store.path)) as connection, connection:
            connection.execute('DROP TABLE kept_worktrees')
    else:

        def fail_save(path: Path, *, checkout: str) -> None:
            raise sqlite3.OperationalError('read-only database')

        monkeypatch.setattr(store, 'keep_worktree', fail_save)
    picker = AsyncMock(return_value='never')
    monkeypatch.setattr(worktrees, 'ask_removal', picker)
    offer_worktree_cleanup(worktree=launched, store=store)
    assert launched.path.exists()
    picker.assert_awaited_once()
    error = capsys.readouterr().err
    assert ('Cannot save' if failure == 'write' else 'Cannot read') in error


def test_suppression_does_not_change_unchanged_cleanup(
    created: Worktree, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SettingsStore(repo.parent / 'config.db')
    git(created.path, 'commit', '--allow-empty', '-m', 'Work')
    monkeypatch.setattr(worktrees, 'ask_removal', AsyncMock(return_value='never'))
    offer_worktree_cleanup(worktree=created, store=store)
    monkeypatch.setattr(worktrees, 'ask_removal', AsyncMock(side_effect=no_prompt))
    unchanged = replace(created, head=git(created.path, 'rev-parse', 'HEAD'))
    offer_worktree_cleanup(worktree=unchanged, store=store)
    assert not created.path.exists()


def test_never_choice_without_store_keeps_checkout(launched: Worktree, monkeypatch: pytest.MonkeyPatch) -> None:
    """Callers without a settings store can still decline cleanup safely."""
    monkeypatch.setattr(worktrees, 'ask_removal', AsyncMock(return_value='never'))
    offer_worktree_cleanup(worktree=launched)
    assert launched.path.exists()


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
def origin(repo: Path) -> Path:
    """A remote of `repo` that nothing has been pushed to."""
    origin = repo.parent / 'origin.git'
    git(repo, 'init', '--bare', str(origin))
    git(repo, 'remote', 'add', 'origin', str(origin))
    return origin


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
    opened = f'Worktree: {created.path} (branch: clai-task).'
    assert created.notice == f'{opened} Removed on exit if unchanged, or if clean and pushed.'
    kept = replace(created, remove_if_unchanged=False)
    assert kept.notice == (
        f'{opened} Kept on exit while it has uncommitted or unpushed work, unless removal is confirmed.'
    )


def test_unchanged_checkout_is_kept_unless_marked(created: Worktree, monkeypatch: pytest.MonkeyPatch) -> None:
    """A checkout the CLI did not mark, as with piped input when it was opened, is kept without asking."""
    monkeypatch.setattr(worktrees, 'ask_removal', AsyncMock(side_effect=no_prompt))
    offer_worktree_cleanup(worktree=replace(created, remove_if_unchanged=False))
    assert created.path.exists()


def test_keep_is_default(launched: Worktree, monkeypatch: pytest.MonkeyPatch) -> None:
    def respond(prompt: str) -> str:
        assert str(launched.path) in prompt
        assert 'The branch will be kept.' in prompt
        return 'keep'

    monkeypatch.setattr(worktrees, 'ask_removal', AsyncMock(side_effect=respond))
    offer_worktree_cleanup(worktree=launched)
    assert launched.path.exists()
    assert Path.cwd() == launched.path


@pytest.mark.parametrize('error', [EOFError, KeyboardInterrupt])
def test_cancel_keeps(launched: Worktree, monkeypatch: pytest.MonkeyPatch, error: type[BaseException]) -> None:
    def respond(prompt: str) -> str:
        raise error

    monkeypatch.setattr(worktrees, 'ask_removal', AsyncMock(side_effect=respond))
    offer_worktree_cleanup(worktree=launched)
    assert launched.path.exists()


def test_remove_preserves_branch(
    launched: Worktree, repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def respond(prompt: str) -> str:
        return 'remove'

    work = git(launched.path, 'rev-parse', 'HEAD')
    monkeypatch.setattr(worktrees, 'ask_removal', AsyncMock(side_effect=respond))
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
        return 'remove'

    monkeypatch.setattr(worktrees, 'ask_removal', AsyncMock(side_effect=respond))
    offer_worktree_cleanup(worktree=launched)
    assert launched.path.exists()
    assert Path.cwd() == launched.path
    output = capsys.readouterr()
    assert 'Worktree kept' in output.err
    assert 'Removed' not in output.out
    if condition == 'dirty':
        assert (launched.path / 'unsaved.txt').read_text() == 'keep me'


@dataclass
class RemovalCommand:
    action: Callable[[], None] = lambda: None
    calls: list[tuple[list[str], Path, int | None]] = field(default_factory=list[tuple[list[str], Path, int | None]])


@pytest.fixture
def removal_command(monkeypatch: pytest.MonkeyPatch) -> RemovalCommand:
    """Run Git normally, but execute the registered command's filesystem effects in-process."""
    command = RemovalCommand()
    original_run = subprocess.run

    def run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool = False,
        text: bool = False,
        encoding: str | None = None,
        stderr: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        if args[0] == 'git':
            return original_run(
                args, check=check, capture_output=capture_output, text=text, encoding=encoding, stderr=stderr
            )
        # The narrow signature also rejects pipes, stdin overrides, shell=True, and timeouts.
        assert check and not capture_output and not text and encoding is None
        command.calls.append((args, Path.cwd(), stderr))
        command.action()
        return subprocess.CompletedProcess(args, 0)

    def confirm(prompt: str) -> str:
        return 'remove'

    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr(worktrees, 'ask_removal', AsyncMock(side_effect=confirm))
    return command


@pytest.mark.parametrize('branch', ['feature', 'current;branch', ''])
def test_custom_removal_command_uses_safe_argv_and_repo_cwd(
    launched: Worktree,
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    removal_command: RemovalCommand,
    branch: str,
) -> None:
    if branch:
        git(launched.path, 'branch', '-m', branch)
    else:
        git(launched.path, 'checkout', '--detach')

    def remove() -> None:
        launched.path.rename(launched.path.with_name('removed'))

    def confirm(prompt: str) -> str:
        assert prompt.startswith(f'{launched.path}\nRegistered removal command: ')
        assert prompt.endswith('It may delete the branch and uncommitted files.')
        return 'remove'

    monkeypatch.setattr(worktrees, 'ask_removal', AsyncMock(side_effect=confirm))
    removal_command.action = remove
    token = 'literal; $(not-a-shell)'
    commands = {str(repo / '.git'): ['remove', '{path}', '{repo}', '{branch}', token]}
    offer_worktree_cleanup(worktree=launched, read_remove_commands=lambda: commands)
    assert removal_command.calls == [
        (['remove', str(launched.path), str(repo), branch, token], repo, terminal_stderr())
    ]
    assert not launched.path.exists()
    assert Path.cwd() == repo
    output = capsys.readouterr()
    assert f'Removed worktree {launched.path}.' in output.out
    assert 'Branch kept' not in output.out
    assert not output.err


@pytest.mark.parametrize('layout', ['bare', 'bare-dotgit', 'separate-git-dir', 'core.worktree'])
def test_custom_removal_from_repository_layout(
    repo: Path, monkeypatch: pytest.MonkeyPatch, removal_command: RemovalCommand, layout: str
) -> None:
    root = repo.parent / f'{layout}-main'
    common = root / '.git'
    if layout == 'bare':
        common = repo.parent / 'bare; repo\n.git'
        git(repo, 'clone', '--bare', str(repo), str(common))
    elif layout == 'bare-dotgit':
        git(repo, 'clone', '--bare', str(repo), str(common))
    elif layout == 'separate-git-dir':
        common = repo.parent / 'separate-metadata'
        git(repo, 'clone', '--separate-git-dir', str(common), str(repo), str(root))
    else:
        git(repo, 'clone', str(repo), str(root))
        working = repo.parent / 'explicit-working-directory'
        working.mkdir()
        (working / '.git').write_text(f'gitdir: {common}\n')
        git(root, 'config', 'core.worktree', str(working))
    expected_repo = root if layout == 'core.worktree' else common
    path = repo.parent / 'linked checkout'
    git(common, 'worktree', 'add', '-b', 'task', str(path))
    monkeypatch.chdir(path)
    worktree = Worktree(path=path, branch='task', head=git(path, 'rev-parse', 'HEAD'), created=False)
    (path / 'tracked.txt').write_text('changed')

    def remove() -> None:
        path.rename(path.with_name('removed'))

    removal_command.action = remove
    offer_worktree_cleanup(worktree=worktree, read_remove_commands=lambda: {str(common): ['remove', '{repo}']})
    assert removal_command.calls == [(['remove', str(expected_repo)], expected_repo, terminal_stderr())]
    assert Path.cwd() == expected_repo


def test_custom_removal_command_decline_keeps_checkout(
    launched: Worktree, repo: Path, monkeypatch: pytest.MonkeyPatch, removal_command: RemovalCommand
) -> None:
    def decline(prompt: str) -> str:
        return 'keep'

    monkeypatch.setattr(worktrees, 'ask_removal', AsyncMock(side_effect=decline))
    offer_worktree_cleanup(worktree=launched, read_remove_commands=lambda: {str(repo / '.git'): ['remove', '{path}']})
    assert launched.path.exists()
    assert not removal_command.calls


@pytest.mark.parametrize('outcome', ['zero', 'nonzero', 'spawn', 'interrupt', 'partial'])
def test_custom_removal_failure_does_not_fall_back(
    launched: Worktree,
    repo: Path,
    capsys: pytest.CaptureFixture[str],
    removal_command: RemovalCommand,
    outcome: str,
) -> None:
    def fail() -> None:
        if outcome == 'zero':
            return
        if outcome == 'spawn':
            raise OSError('missing command')
        if outcome == 'interrupt':
            raise KeyboardInterrupt
        if outcome == 'partial':
            launched.path.rename(launched.path.with_name('partial'))
        raise subprocess.CalledProcessError(2, ['remove'])

    removal_command.action = fail
    offer_worktree_cleanup(worktree=launched, read_remove_commands=lambda: {str(repo / '.git'): ['remove']})
    assert len(removal_command.calls) == 1
    assert launched.path.exists() == (outcome != 'partial')
    assert Path.cwd() == (repo if outcome == 'partial' else launched.path)
    output = capsys.readouterr()
    assert ('without removing' if outcome == 'zero' else 'failed') in output.err
    assert str(launched.path) in output.err
    assert 'Removed worktree' not in output.out
    assert 'Worktree kept' not in output.err


def test_custom_command_preparation_failure(
    launched: Worktree, repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(*args: str) -> str:
        if '--git-common-dir' in args:
            raise subprocess.CalledProcessError(1, ['git'], stderr='metadata unavailable')
        return git(Path.cwd(), *args)

    monkeypatch.setattr(worktrees, '_git', fail)
    monkeypatch.setattr(worktrees, 'ask_removal', AsyncMock(side_effect=no_prompt))
    offer_worktree_cleanup(worktree=launched, read_remove_commands=lambda: {str(repo / '.git'): ['remove']})
    assert launched.path.exists()
    assert f'Worktree kept at {launched.path}: cannot prepare removal command' in capsys.readouterr().err


def test_custom_command_undecodable_branch_is_reported(
    launched: Worktree, repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def undecodable(*args: str) -> str:
        if args[-2:] == ('branch', '--show-current'):
            raise UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'invalid start byte')
        return git(Path.cwd(), *args)

    monkeypatch.setattr(worktrees, '_git', undecodable)
    monkeypatch.setattr(worktrees, 'ask_removal', AsyncMock(side_effect=no_prompt))
    offer_worktree_cleanup(worktree=launched, read_remove_commands=lambda: {str(repo / '.git'): ['never-run']})
    assert launched.path.exists()
    assert Path.cwd() == launched.path
    error = capsys.readouterr().err
    assert f'Worktree kept at {launched.path}: cannot prepare removal command' in error
    assert "can't decode byte" in error


def test_custom_command_not_offered_without_tty(
    launched: Worktree, repo: Path, monkeypatch: pytest.MonkeyPatch, removal_command: RemovalCommand
) -> None:
    monkeypatch.setattr('sys.stdin.isatty', lambda: False)
    monkeypatch.setattr(worktrees, 'ask_removal', AsyncMock(side_effect=no_prompt))
    offer_worktree_cleanup(worktree=launched, read_remove_commands=lambda: {str(repo / '.git'): ['custom']})
    assert launched.path.exists()
    assert not removal_command.calls


def test_mismatched_repository_uses_git_removal(
    launched: Worktree, repo: Path, monkeypatch: pytest.MonkeyPatch, removal_command: RemovalCommand
) -> None:
    def confirm(prompt: str) -> str:
        assert prompt == f'{launched.path}\nThe branch will be kept. Git refuses to remove uncommitted files.'
        return 'remove'

    monkeypatch.setattr(worktrees, 'ask_removal', AsyncMock(side_effect=confirm))
    offer_worktree_cleanup(worktree=launched, read_remove_commands=lambda: {str(repo / 'other.git'): ['custom']})
    assert not launched.path.exists()
    assert git(repo, 'branch', '--list', 'feature')
    assert not removal_command.calls


def test_unmodified_worktree_ignores_custom_command(
    created: Worktree, repo: Path, monkeypatch: pytest.MonkeyPatch, removal_command: RemovalCommand
) -> None:
    monkeypatch.setattr(worktrees, 'ask_removal', AsyncMock(side_effect=no_prompt))
    offer_worktree_cleanup(worktree=created, read_remove_commands=lambda: {str(repo / '.git'): ['custom']})
    assert not created.path.exists()
    assert not removal_command.calls


@pytest.mark.parametrize('case', ['changed', 'unchanged', 'piped'])
def test_removal_configuration_is_read_only_for_changed_interactive_checkouts(
    created: Worktree,
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    removal_command: RemovalCommand,
    case: str,
) -> None:
    reads: list[bool] = []

    def commands() -> dict[str, list[str]]:
        reads.append(True)
        return {str(repo / '.git'): ['remove']}

    def remove() -> None:
        created.path.rename(created.path.with_name('removed'))

    removal_command.action = remove
    if case == 'changed':
        (created.path / 'tracked.txt').write_text('changed')
    elif case == 'piped':
        monkeypatch.setattr('sys.stdin.isatty', lambda: False)
    offer_worktree_cleanup(worktree=created, read_remove_commands=commands)
    assert bool(reads) == (case == 'changed')
    assert bool(removal_command.calls) == (case == 'changed')
    assert created.path.exists() == (case == 'piped')


def test_forced_removal_ignores_registered_command(
    launched: Worktree, repo: Path, monkeypatch: pytest.MonkeyPatch, removal_command: RemovalCommand
) -> None:
    monkeypatch.setenv('XDG_CONFIG_HOME', str(repo.parent / 'config'))
    SettingsStore().set('worktrees.remove_commands', {str(repo / '.git'): ['custom']})
    (launched.path / 'tracked.txt').write_text('uncommitted changes')
    monkeypatch.chdir(repo)
    worktrees.remove_worktree(path=launched.path)
    assert not launched.path.exists()
    assert git(repo, 'branch', '--list', 'feature')
    assert not removal_command.calls


def test_missing_git_keeps_checkout(launched: Worktree, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PATH', '')
    monkeypatch.setattr(worktrees, 'ask_removal', AsyncMock(side_effect=no_prompt))
    offer_worktree_cleanup(worktree=launched)
    assert launched.path.exists()


def test_removal_os_error_keeps_checkout(
    launched: Worktree, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def respond(prompt: str) -> str:
        monkeypatch.setenv('PATH', '')
        return 'remove'

    monkeypatch.setattr(worktrees, 'ask_removal', AsyncMock(side_effect=respond))
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
    monkeypatch.setattr(worktrees, 'ask_removal', AsyncMock(side_effect=no_prompt))
    offer_worktree_cleanup(worktree=created)
    assert created.path.exists()


def test_no_worktree_is_a_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worktrees, 'ask_removal', AsyncMock(side_effect=no_prompt))
    offer_worktree_cleanup(worktree=None)


def test_cli_uses_custom_command_saved_during_session(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(repo)
    commands: list[list[str]] = []

    def remove(*, path: Path, repo: Path, command: list[str]) -> None:
        commands.append(command)

    monkeypatch.setattr(worktrees, '_remove_with_command', remove)

    def save_command() -> None:
        SettingsStore(repo.parent / 'config.db').set(
            'worktrees.remove_commands', {str(repo / '.git'): ['remove', '{path}']}
        )
        git(Path.cwd(), 'commit', '--allow-empty', '-m', 'Work')

    prompts = run_cli(monkeypatch, '-w', 'task', repo=repo, session=save_command, answer='remove')
    assert len(prompts) == 1
    assert commands == [['remove', str(repo / '.worktrees/task')]]


@pytest.mark.parametrize('linked', [False, True])
def test_exit_ignores_unrelated_settings_changed_by_another_build(
    repo: Path, checkout: Path, monkeypatch: pytest.MonkeyPatch, linked: bool
) -> None:
    monkeypatch.chdir(checkout if linked else repo)
    reads: list[SettingsStore] = []
    original = SettingsStore.worktree_remove_commands

    def read(store: SettingsStore) -> dict[str, list[str]]:
        reads.append(store)
        return original(store)

    monkeypatch.setattr(SettingsStore, 'worktree_remove_commands', read)

    def newer_setting() -> None:
        with closing(sqlite3.connect(repo.parent / 'config.db')) as connection, connection:
            connection.execute('INSERT INTO settings VALUES (?, ?)', ('display.tool_calls', '"newer-format"'))
        git(Path.cwd(), 'commit', '--allow-empty', '-m', 'Work')

    assert len(run_cli(monkeypatch, repo=repo, session=newer_setting)) == linked
    assert len(reads) == linked
    assert checkout.exists()


@pytest.mark.parametrize('failure', ['invalid-setting', 'invalid-json', 'database'])
def test_exit_uses_git_when_removal_setting_cannot_be_read(
    repo: Path,
    checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: str,
) -> None:
    def break_setting() -> None:
        if failure == 'invalid-setting':
            with closing(sqlite3.connect(repo.parent / 'config.db')) as connection, connection:
                connection.execute(
                    'INSERT INTO settings VALUES (?, ?)', ('worktrees.remove_commands', '{"relative": ["remove"]}')
                )
        elif failure == 'invalid-json':
            with closing(sqlite3.connect(repo.parent / 'config.db')) as connection, connection:
                connection.execute('INSERT INTO settings VALUES (?, ?)', ('worktrees.remove_commands', 'not JSON'))
        else:
            with closing(sqlite3.connect(repo.parent / 'config.db')) as connection, connection:
                assert connection.execute(
                    'SELECT COUNT(*) FROM settings WHERE key = ?', ('worktrees.remove_commands',)
                ).fetchone() == (0,)
                connection.execute('DROP TABLE settings')
        git(checkout, 'commit', '--allow-empty', '-m', 'Work')

    assert run_cli(monkeypatch, repo=repo, session=break_setting, answer='remove') == [
        f'{checkout}\nThe branch will be kept. Git refuses to remove uncommitted files.'
    ]
    assert not checkout.exists()
    assert git(repo, 'branch', '--list', 'feature')
    output = capsys.readouterr()
    assert output.err.lower().count('cannot read removal commands') == 1
    assert f'Removed worktree {checkout}. Branch kept.' in output.out


def test_unchanged_hand_made_checkout_is_kept_silently(
    checkout: Path, repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    worktree = current_worktree()
    monkeypatch.setattr(worktrees, 'ask_removal', AsyncMock(side_effect=no_prompt))
    offer_worktree_cleanup(worktree=worktree)
    assert checkout.exists()
    assert git(repo, 'branch', '--list', 'feature')
    assert capsys.readouterr() == ('', '')


def test_unchanged_created_checkout_is_removed_silently(
    created: Worktree, repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (created.path / 'ignored.log').write_text('build output')
    (repo / '.git/info/exclude').write_text('*.log\n')
    monkeypatch.setattr(worktrees, 'ask_removal', AsyncMock(side_effect=no_prompt))
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
    monkeypatch.setattr(worktrees, 'ask_removal', AsyncMock(side_effect=no_prompt))
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
        return 'keep'

    monkeypatch.setattr(worktrees, 'ask_removal', AsyncMock(side_effect=respond))
    offer_worktree_cleanup(worktree=created)
    assert len(prompts) == 1
    assert created.path.exists()
    assert git(created.path, 'branch', '--show-current') == created.branch
    assert 'Worktree kept at' in capsys.readouterr().out


@pytest.mark.parametrize('moved', [False, True])
def test_clean_pushed_checkout_is_removed_without_asking(
    launched: Worktree,
    repo: Path,
    origin: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    removal_command: RemovalCommand,
    moved: bool,
) -> None:
    """Pushed commits are safe on the remote, whether or not `HEAD` moved since launch."""
    git(launched.path, 'push', 'origin', 'feature')
    worktree = launched if moved else current_worktree()
    assert worktree is not None
    monkeypatch.setattr(worktrees, 'ask_removal', AsyncMock(side_effect=no_prompt))
    offer_worktree_cleanup(worktree=worktree, read_remove_commands=lambda: {str(repo / '.git'): ['custom']})
    assert not launched.path.exists()
    assert Path.cwd() == repo
    assert git(repo, 'rev-parse', 'feature') == git(origin, 'rev-parse', 'feature')
    assert not removal_command.calls
    assert capsys.readouterr() == (f'Removed worktree {launched.path}. Branch kept.\n', '')


@pytest.mark.parametrize('change', ['untracked', 'modified', 'commit'])
def test_pushed_checkout_with_later_work_still_asks(
    launched: Worktree, origin: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """Uncommitted files and unpushed commits are work only this checkout or its branch holds."""
    git(launched.path, 'push', 'origin', 'feature')
    if change == 'untracked':
        (launched.path / 'new.txt').write_text('draft')
    elif change == 'modified':
        (launched.path / 'tracked.txt').write_text('edited')
    else:
        git(launched.path, 'commit', '--allow-empty', '-m', 'Unpushed')
    picker = AsyncMock(return_value='keep')
    monkeypatch.setattr(worktrees, 'ask_removal', picker)
    offer_worktree_cleanup(worktree=launched)
    picker.assert_awaited_once()
    assert launched.path.exists()


def test_saved_keep_choice_outlives_a_push(
    launched: Worktree, repo: Path, origin: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = SettingsStore(repo.parent / 'config.db')
    monkeypatch.setattr(worktrees, 'ask_removal', AsyncMock(return_value='never'))
    offer_worktree_cleanup(worktree=launched, store=store)
    git(launched.path, 'push', 'origin', 'feature')
    capsys.readouterr()
    monkeypatch.setattr(worktrees, 'ask_removal', AsyncMock(side_effect=no_prompt))
    offer_worktree_cleanup(worktree=launched, store=store)
    assert launched.path.exists()
    assert capsys.readouterr() == (f'Worktree kept at {launched.path}.\n', '')


def run_cli(
    monkeypatch: pytest.MonkeyPatch,
    *args: str,
    repo: Path,
    session: Callable[[], None] = lambda: None,
    relaunch: bool = False,
    answer: str = 'keep',
) -> list[str]:
    """Run `clai2` with a stand-in shell; `relaunch` ends it as `/update` does. Returns the prompts asked on exit."""
    prompts: list[str] = []

    async def chat(*_: object, **__: object) -> None:
        # The stand-in session commits and edits settings with Git and SQLite, which would block the loop.
        await anyio.to_thread.run_sync(session)
        if relaunch:
            raise Relaunch(executable='/b/clai2', session_id='s')

    def respond(prompt: str) -> str:
        prompts.append(prompt)
        return answer

    monkeypatch.setattr(app, 'chat', chat)

    def execv(path: str, argv: list[str]) -> None:
        pass

    monkeypatch.setattr(os, 'execv', execv)
    monkeypatch.setattr(worktrees, 'ask_removal', AsyncMock(side_effect=respond))
    # Beside the repository: a database inside the checkout would be an untracked change.
    database = repo.parent / 'config.db'
    monkeypatch.setattr('sys.argv', ['clai2', '--database', str(database), '-m', 'test', *args])
    cli.run()
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


@pytest.mark.parametrize(
    ('keys', 'expected'),
    [
        ('\r', 'keep'),
        ('\x1b', 'keep'),
        ('\x03', 'keep'),
        ('1', 'remove'),
        ('2', 'keep'),
        ('3', 'never'),
        ('\x1b[A\r', 'remove'),
        ('\x1b[B\r', 'never'),
        ('\x1b[B\x1b[B\r', 'remove'),
        ('4\r', 'keep'),
    ],
)
async def test_exit_picker(keys: str, expected: str, monkeypatch: pytest.MonkeyPatch) -> None:
    output = io.StringIO()
    console = Console(file=output, width=100, height=30)
    monkeypatch.setattr(worktree_cleanup, 'Console', lambda: console)
    original_frame = QuestionMenu.frame
    frames: list[tuple[str, ...]] = []

    def frame(menu: QuestionMenu, *, width: int, height: int) -> tuple[str, ...]:
        rendered = original_frame(menu, width=width, height=height)
        frames.append(rendered)
        return rendered

    monkeypatch.setattr(QuestionMenu, 'frame', frame)
    with create_pipe_input() as pipe:
        monkeypatch.setattr('pydantic_clai2.ui.prompt.question_input.create_input', lambda: pipe)
        pipe.send_text(keys)
        assert await worktree_cleanup.ask_removal('/tmp/checkout\nThe branch will be kept.') == expected

    initial = frames[0]
    assert 'Remove this worktree?' in initial[0]
    assert initial[1] == theme.sgr(theme.MUTED) + '/tmp/checkout\x1b[0m'
    assert initial[2] == theme.sgr(theme.MUTED) + 'The branch will be kept.\x1b[0m'
    assert initial[3] == '  1. Yes'
    assert '> 2. No' in initial[4]
    assert initial[5] == "  3. No, don't ask again for this worktree"
    assert 'Other' not in output.getvalue()


async def test_exit_picker_accepts_long_details(monkeypatch: pytest.MonkeyPatch) -> None:
    """Complete cleanup details are inspectable in the transcript before a choice can execute them."""
    output = io.StringIO()
    monkeypatch.setattr(worktree_cleanup, 'Console', lambda: Console(file=output, width=80, height=24))
    original_paint = PromptSurface.paint
    transcripts: list[str] = []

    def paint(surface: PromptSurface, rows: tuple[str, ...], *, select_pinned: bool = False) -> None:
        transcripts.append(
            '\n'.join(
                Text.from_ansi(row).plain
                for item in surface.transcript.ids()
                for row in surface.transcript.rows(item, width=80)
            )
        )
        original_paint(surface, rows, select_pinned=select_pinned)

    monkeypatch.setattr(PromptSurface, 'paint', paint)
    details = (
        '/tmp/checkout\nRegistered removal command: '
        + ' '.join(f'argument-{index:02d}' for index in range(80))
        + ' last-argument\x1b[2J\nIt may delete the branch and uncommitted files.'
    )
    with create_pipe_input() as pipe:
        monkeypatch.setattr('pydantic_clai2.ui.prompt.question_input.create_input', lambda: pipe)
        pipe.send_text('\r')
        assert await worktree_cleanup.ask_removal(details) == 'keep'
    assert '/tmp/checkout' in transcripts[0]
    assert transcripts[0].count('argument-') == 80
    assert 'last-argument\\x1b[2J' in transcripts[0]
    assert 'It may delete the branch and uncommitted files.' in transcripts[0]
    assert '\\x1b[2J' in output.getvalue()
