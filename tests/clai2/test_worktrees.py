"""Launch the installed CLI against real repositories without provider requests."""

import os
import subprocess
import sys
from pathlib import Path

import anyio
import pytest
from pydantic_ai_harness.step_persistence.conversations import SqliteConversationStore


def git(directory: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(directory), *args], check=True, capture_output=True, text=True, timeout=15
    ).stdout.strip()


def launch(directory: Path, *args: str, prompt: str = '/exit\n') -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, '-m', 'pydantic_clai2', '--database', 'config.db', '--model', 'test', *args],
        cwd=directory,
        input=prompt,
        text=True,
        capture_output=True,
        env=dict(os.environ, CLAI_NO_SPLASH='1'),
        timeout=30,
        check=False,
    )


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', os.devnull)
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    repo = tmp_path / 'source repo'
    repo.mkdir()
    git(repo, 'init')
    git(repo, 'config', 'user.name', 'CLAI Test')
    git(repo, 'config', 'user.email', 'clai@example.com')
    (repo / '.clai').mkdir()
    (repo / '.clai/settings.json').write_text('{"request_limit": 7}')
    (repo / 'tracked.txt').write_text('committed')
    git(repo, 'add', '.')
    git(repo, 'commit', '-m', 'Initial workspace')
    return repo


@pytest.mark.parametrize(
    'args',
    [['--worktree', 'feature'], ['-w', 'feature'], ['--worktree=feature'], ['-wfeature'], ['--worktree'], ['-w']],
)
def test_launches_in_new_worktree(repository: Path, args: list[str]) -> None:
    original_head = git(repository, 'rev-parse', 'HEAD')
    original_branch = git(repository, 'symbolic-ref', 'HEAD')
    (repository / 'tracked.txt').write_text('uncommitted')
    (repository / '.clai/settings.json').write_text('{"request_limit": 9}')
    (repository / 'untracked.txt').write_text('keep this')
    nested = repository / 'nested'
    nested.mkdir()
    plugins = nested / 'plugins'
    plugins.mkdir()
    (plugins / 'workspace.py').write_text(
        'from pathlib import Path\ndef activate(host):\n    Path("plugin-workspace.txt").write_text(str(Path.cwd()))\n'
    )

    result = launch(nested, *args, prompt='/set run.request_limit\n/exit\n')

    assert result.returncode == 0, result.stderr
    worktrees = list((repository / '.worktrees').iterdir())
    assert len(worktrees) == 1
    workspace = worktrees[0]
    assert (
        workspace.name.startswith('worktree-')
        if len(args) == 1 and args[0] in ('-w', '--worktree')
        else workspace.name == 'feature'
    )
    assert (
        f'Worktree: {workspace} (branch: clai/{workspace.name}). Kept unless removal is confirmed on exit.'
        in result.stdout
    )
    assert git(workspace, 'branch', '--show-current') == f'clai/{workspace.name}'
    assert git(workspace, 'rev-parse', 'HEAD') == original_head
    assert (workspace / 'tracked.txt').read_text() == 'committed'
    assert not (workspace / 'untracked.txt').exists()
    assert 'Project settings:' in result.stdout
    assert '\n7\n' in result.stdout
    assert (workspace / 'plugin-workspace.txt').read_text() == str(workspace)
    assert (nested / 'config.db').exists()
    assert not (workspace / 'config.db').exists()
    assert git(repository, 'symbolic-ref', 'HEAD') == original_branch
    assert (repository / 'tracked.txt').read_text() == 'uncommitted'
    assert (repository / 'untracked.txt').read_text() == 'keep this'
    assert '.worktrees' not in git(repository, 'status', '--porcelain')
    assert not (repository / '.gitignore').exists()


def test_launch_from_linked_worktree(repository: Path) -> None:
    linked = repository.parent / 'linked'
    git(repository, 'worktree', 'add', '-b', 'linked', str(linked))
    (linked / 'linked.txt').write_text('linked commit')
    git(linked, 'add', 'linked.txt')
    git(linked, 'commit', '-m', 'Advance linked checkout')
    result = launch(linked, '-w', 'child')
    assert result.returncode == 0, result.stderr
    child = linked / '.worktrees/child'
    assert git(child, 'rev-parse', 'HEAD') == git(linked, 'rev-parse', 'HEAD')
    assert (child / 'linked.txt').read_text() == 'linked commit'
    assert '.worktrees' not in git(linked, 'status', '--porcelain')


@pytest.mark.parametrize('existing', [None, b'keep-me', b'keep-me\n/.worktrees/\n'])
def test_local_exclude_preserves_rules_without_duplicates(repository: Path, existing: bytes | None) -> None:
    exclude = repository / '.git/info/exclude'
    if existing is None:
        exclude.unlink()
        exclude.parent.rmdir()
    else:
        exclude.write_bytes(existing)
    result = launch(repository, '-w', 'ignored')
    assert result.returncode == 0, result.stderr
    contents = exclude.read_bytes()
    assert contents.startswith(existing or b'')
    assert contents.splitlines().count(b'/.worktrees/') == 1
    assert git(repository, 'check-ignore', '.worktrees/ignored') == '.worktrees/ignored'


def test_unwritable_exclude_is_a_parser_error(repository: Path) -> None:
    exclude = repository / '.git/info/exclude'
    exclude.unlink()
    exclude.mkdir()
    result = launch(repository, '-w', 'ignored')
    assert result.returncode == 2
    assert 'error: Cannot create worktree:' in result.stderr
    assert 'Traceback' not in result.stderr
    assert not (repository / '.worktrees').exists()


def test_session_belongs_to_worktree_and_can_be_resumed(repository: Path) -> None:
    result = launch(
        repository,
        '-w',
        'saved',
        prompt='/plugins disable coder\n/plugins disable ask_user\n/set sessions.naming false\nhello\n/exit\n',
    )
    assert result.returncode == 0, result.stderr
    workspace = repository / '.worktrees/saved'
    store = SqliteConversationStore(database=repository / 'sessions.db')
    summaries = anyio.run(store.listing)
    assert len(summaries) == 1
    assert summaries[0].workspace == str(workspace)
    resumed = launch(workspace, '--database', str(repository / 'config.db'), '--resume', summaries[0].id)
    assert resumed.returncode == 0, resumed.stderr
    assert 'Resumed' in resumed.stdout


def test_startup_error_keeps_created_worktree(repository: Path) -> None:
    result = launch(repository, '-w', 'retained', '--request-limit', '0')
    assert result.returncode == 2
    workspace = repository / '.worktrees/retained'
    assert f'Worktree: {workspace}' in result.stdout
    assert 'Kept unless removal is confirmed on exit.' in result.stdout
    assert (workspace / 'tracked.txt').read_text() == 'committed'
    assert git(workspace, 'branch', '--show-current') == 'clai/retained'


@pytest.mark.parametrize('state', ['outside', 'unborn', 'branch', 'path', 'missing-git'])
def test_git_errors_leave_existing_work_untouched(
    repository: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    directory = repository
    workspace = repository / '.worktrees/feature'
    exclude = repository / '.git/info/exclude'
    original_excludes = exclude.read_bytes()
    if state == 'outside':
        directory = tmp_path
    elif state == 'unborn':
        directory = tmp_path / 'empty'
        directory.mkdir()
        git(directory, 'init')
    elif state == 'branch':
        git(repository, 'branch', 'clai/feature')
    elif state == 'path':
        workspace.mkdir(parents=True)
        (workspace / 'keep.txt').write_text('keep this')
    original_branches = git(repository, 'branch', '--format=%(refname)')
    if state == 'missing-git':
        monkeypatch.setenv('PATH', '')
    result = launch(directory, '-w', 'feature')
    assert result.returncode == 2
    assert 'error: Cannot ' in result.stderr
    assert 'Traceback' not in result.stderr
    assert 'Worktree:' not in result.stdout
    assert (repository / 'tracked.txt').read_text() == 'committed'
    assert exclude.read_bytes() == original_excludes
    if state != 'missing-git':
        assert git(repository, 'branch', '--format=%(refname)') == original_branches
    if state == 'path':
        assert (workspace / 'keep.txt').read_text() == 'keep this'
    else:
        assert not workspace.exists()


def test_failed_checkout_removes_new_branch_and_allows_retry(repository: Path) -> None:
    (repository / '.gitattributes').write_text('tracked.txt filter=fail\n')
    git(repository, 'add', '.gitattributes')
    git(repository, 'commit', '-m', 'Configure checkout filter')
    git(repository, 'config', 'filter.fail.smudge', 'false')
    git(repository, 'config', 'filter.fail.required', 'true')
    result = launch(repository, '-w', 'retry')
    assert result.returncode == 2
    assert 'smudge filter fail failed' in result.stderr
    assert 'clai/retry' not in git(repository, 'branch', '--format=%(refname)')
    assert not (repository / '.worktrees/retry').exists()
    git(repository, 'config', 'filter.fail.smudge', 'cat')
    assert launch(repository, '-w', 'retry').returncode == 0


def test_checkout_hook_failure_preserves_work_and_reports_cleanup_failure(repository: Path) -> None:
    hook = repository / '.git/hooks/post-checkout'
    hook.write_text('#!/bin/sh\nexit 1\n')
    hook.chmod(0o755)
    result = launch(repository, '-w', 'retained')
    workspace = repository / '.worktrees/retained'
    assert result.returncode == 2
    assert f'Cannot create worktree at {workspace}' in result.stderr
    assert 'Branch clai/retained could not be removed' in result.stderr
    assert (workspace / 'tracked.txt').read_text() == 'committed'
    assert git(workspace, 'branch', '--show-current') == 'clai/retained'


def test_exclude_write_failure_reports_retained_worktree(repository: Path) -> None:
    hook = repository / '.git/hooks/post-checkout'
    hook.write_text('#!/bin/sh\nexclude=$(git rev-parse --git-path info/exclude)\nrm "$exclude"\nmkdir "$exclude"\n')
    hook.chmod(0o755)
    result = launch(repository, '-w', 'retained')
    workspace = repository / '.worktrees/retained'
    assert result.returncode == 2
    assert f'Worktree kept at {workspace}, but could not update Git excludes' in result.stderr
    assert (workspace / 'tracked.txt').read_text() == 'committed'
    assert git(workspace, 'branch', '--show-current') == 'clai/retained'


@pytest.mark.parametrize(
    'name', ['../escape', '/absolute', 'nested/name', 'back\\slash', '..', '-b', 'bad name', 'bad.name']
)
def test_invalid_names_are_parser_errors(repository: Path, name: str) -> None:
    result = launch(repository, f'--worktree={name}')
    assert result.returncode == 2
    assert 'error: Worktree names' in result.stderr
    assert 'Traceback' not in result.stderr
    assert not (repository / '.worktrees').exists()


@pytest.mark.parametrize('args', [['--resume'], ['--resume=session'], ['config', 'show'], ['plugins', 'list']])
def test_worktree_rejects_incompatible_commands(repository: Path, args: list[str]) -> None:
    result = launch(repository, '--worktree=feature', *args)
    assert result.returncode == 2
    assert 'cannot be combined' in result.stderr
    assert not (repository / '.worktrees').exists()
    assert not (repository / 'config.db').exists()
