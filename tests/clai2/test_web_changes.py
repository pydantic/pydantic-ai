"""The Changes panel's data: a session's changed files from its own file tools, and from Git."""

import subprocess
from collections.abc import Iterator
from itertools import count
from pathlib import Path
from typing import Literal

import pytest

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai_harness.filesystem import EditFileCallPart, WriteFileCallPart
from pydantic_clai2.builtin_plugins.web import changes as changes_module
from pydantic_clai2.builtin_plugins.web.changes import Changes, ChangeScope, session_changes
from pydantic_clai2.runtime.session import STEERING

from .test_web_sessions import MakeFleet, make_fleet as make_fleet

Call = tuple[type[ToolCallPart[object]], str, dict[str, object]]
"""The part a call is made of, the tool's name, and its arguments."""


class History:
    """A session's history, built a turn at a time, each call given its own ID."""

    def __init__(self) -> None:
        self.messages: list[ModelMessage] = []
        self._ids = count()

    def turn(
        self,
        prompt: str,
        *calls: Call,
        steering: bool = False,
        outcome: Literal['success', 'failed', 'denied', 'interrupted', 'retried'] = 'success',
    ) -> None:
        metadata = {STEERING: True} if steering else None
        self.messages.append(ModelRequest(parts=[UserPromptPart(prompt)], metadata=metadata))
        for part, name, args in calls:
            call_id = f'call{next(self._ids)}'
            self.messages.append(ModelResponse(parts=[part(name, args, call_id)]))
            self.messages.append(ModelRequest(parts=[ToolReturnPart(name, 'done', call_id, outcome=outcome)]))


def write(path: str, content: str) -> Call:
    return WriteFileCallPart, 'write_file', {'path': path, 'content': content}


def edit(path: str, old: str, new: str) -> Call:
    return EditFileCallPart, 'edit_file', {'path': path, 'old_text': old, 'new_text': new}


def listed(found: Changes) -> list[tuple[str, str, int, int]]:
    return [(change.path, change.status, change.added, change.removed) for change in found.files]


def git(repo: Path, *args: str) -> str:
    command = ['git', '-C', str(repo), '-c', 'user.name=Test', '-c', 'user.email=test@example.com', *args]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Iterator[Path]:
    """A Git repository on `main` with one commit of `a.py` and `b.py`."""
    root = tmp_path / 'repo'
    root.mkdir()
    git(root, 'init', '-q', '-b', 'main')
    (root / 'a.py').write_text('def a():\n    return 1\n')
    (root / 'b.py').write_text('def b():\n    return 2\n')
    git(root, 'add', '.')
    git(root, 'commit', '-q', '-m', 'start')
    yield root


A1 = 'def hello():\n    return "a"\n'
A2 = 'def hello():\n    """Say hello."""\n    return "a"\n'
B1 = 'def hello():\n    return "b"\n'
C1 = 'def hello():\n    return "c"\n'


async def test_last_turn_and_session_from_the_file_tools(tmp_path: Path) -> None:
    """The issue's repro: two turns, with `a.py` written then edited; each file shows one net diff."""
    history = History()
    history.turn('create a.py and b.py', write('a.py', A1), write('b.py', B1))
    history.turn(
        'docstring a.py, and create c.py', edit('a.py', '():\n', '():\n    """Say hello."""\n'), write('c.py', C1)
    )
    for name, text in (('a.py', A2), ('b.py', B1), ('c.py', C1)):
        (tmp_path / name).write_text(text)

    session = await session_changes(history.messages, tmp_path, scope='session')
    assert listed(session) == [('a.py', 'A', 3, 0), ('b.py', 'A', 2, 0), ('c.py', 'A', 2, 0)]
    assert session.files[0].diff.diff == (
        '--- /dev/null\n+++ b/a.py\n@@ -0,0 +1,3 @@\n+def hello():\n+    """Say hello."""\n+    return "a"'
    )

    turn = await session_changes(history.messages, tmp_path, scope='turn')
    assert listed(turn) == [('a.py', 'M', 1, 0), ('c.py', 'A', 2, 0)]
    assert turn.files[0].diff.diff == (
        '--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,3 @@\n def hello():\n+    """Say hello."""\n     return "a"'
    )


async def test_a_change_to_the_final_newline_alone_shows(tmp_path: Path) -> None:
    (tmp_path / 'a.py').write_text('x = 1')
    history = History()
    history.turn('one', edit('a.py', 'x = 1\n', 'x = 1'))
    [change] = (await session_changes(history.messages, tmp_path, scope='session')).files
    assert (change.status, change.added, change.removed) == ('M', 1, 1)
    assert change.diff.diff.endswith('-x = 1\n+x = 1\n\\ No newline at end of file')


async def test_edits_of_a_file_the_session_did_not_create_are_undone_from_the_disk(tmp_path: Path) -> None:
    (tmp_path / 'main.py').write_text('x = 3\ny = 4\n')
    history = History()
    history.turn('one', edit('main.py', 'x = 1', 'x = 2'))
    history.turn('steer', edit('main.py', 'x = 2', 'x = 3'), steering=True)
    history.turn('two', edit('main.py', 'y = 1', 'y = 4'))
    history.turn('failed', edit('main.py', 'y = 4', 'y = 5'), outcome='retried')

    session = await session_changes(history.messages, tmp_path, scope='session')
    assert listed(session) == [('main.py', 'M', 2, 2)]
    assert '-x = 1\n-y = 1\n+x = 3\n+y = 4' in session.files[0].diff.diff

    # The last turn holds the failed edit alone, which changed nothing.
    assert listed(await session_changes(history.messages, tmp_path, scope='turn')) == []


async def test_replacements_and_steering_stay_in_their_turn(tmp_path: Path) -> None:
    (tmp_path / 'main.py').write_text('x = 3\ny = 4\n')
    history = History()
    history.turn('one', edit('main.py', 'x = 1', 'x = 2'))
    replacements = [{'old_text': 'x = 2', 'new_text': 'x = 3'}, {'old_text': 'y = 1', 'new_text': 'y = 4'}]
    history.turn('two', (EditFileCallPart, 'edit_file', {'path': 'main.py', 'replacements': replacements}))
    history.turn('more', steering=True)

    assert listed(await session_changes(history.messages, tmp_path, scope='turn')) == [('main.py', 'M', 2, 2)]


async def test_a_file_whose_earlier_content_is_unknown_is_listed_without_a_diff(tmp_path: Path) -> None:
    (tmp_path / 'edited.py').write_text('changed elsewhere\n')
    (tmp_path / 'rewritten.py').write_text('whole\n')
    history = History()
    history.turn(
        'one',
        edit('edited.py', 'a = 1', 'a = 2'),
        edit('rewritten.py', 'a', 'b'),
        write('rewritten.py', 'whole\n'),
        edit('removed.py', 'a = 1', 'a = 2'),
    )

    found = await session_changes(history.messages, tmp_path, scope='session')
    assert listed(found) == [('edited.py', 'M', 0, 0), ('removed.py', 'D', 0, 0), ('rewritten.py', 'M', 0, 0)]
    assert {change.note for change in found.files} == {
        'Its content before these changes is not known, so it has no diff.'
    }


async def test_the_last_commit_is_the_start_the_calls_cannot_give(repo: Path) -> None:
    """A tracked file the session rewrote whole, or edited by deleting text, diffs against its last commit."""
    (repo / 'a.py').write_text('def a():\n    return 100\n')
    (repo / 'b.py').write_text('def b():\n')
    (repo / 'c.py').write_text('changed outside the session\n')
    git(repo, 'add', 'c.py')
    git(repo, 'commit', '-q', '-m', 'c')
    (repo / 'c.py').write_text('changed again\n')
    history = History()
    history.turn(
        'one',
        write('a.py', 'def a():\n    return 100\n'),
        edit('b.py', '    return 2\n', ''),
        edit('c.py', 'changed outside', 'changed inside'),
    )

    found = await session_changes(history.messages, repo, scope='session')
    assert listed(found) == [('a.py', 'M', 1, 1), ('b.py', 'M', 0, 1), ('c.py', 'M', 0, 0)]
    assert found.files[2].note == 'Its content before these changes is not known, so it has no diff.'


async def test_files_the_session_created_stay_created_once_committed(repo: Path) -> None:
    (repo / 'docs').mkdir()
    (repo / 'docs' / 'index.md').write_text('# Docs\n')
    (repo / 'new.py').write_text('new = 1\n')
    (repo / 'with space.py').write_text('space = 1\n')
    (repo / 'b.py').write_bytes(b'\0binary now')
    git(repo, 'add', 'docs', 'new.py')
    git(repo, 'commit', '-q', '-m', 'the session committed new.py')
    history = History()
    history.turn(
        'one',
        write('new.py', 'new = 1\n'),
        write('with space.py', 'space = 1\n'),
        write('docs', ''),
        write('b.py', 'b'),
        write('image.bin', '\0'),
    )
    (repo / 'image.bin').write_bytes(b'\0')
    git(repo, 'add', 'image.bin')
    git(repo, 'commit', '-q', '-m', 'the session committed image.bin')
    history.turn('two', write('line\nbreak.py', 'x = 1\n'))
    (repo / 'line\nbreak.py').write_text('x = 1\n')

    found = await session_changes(history.messages, repo, scope='session')
    assert listed(found) == [
        ('b.py', 'M', 0, 0),
        ('docs', 'A', 0, 0),
        ('image.bin', 'A', 0, 0),
        ('line\nbreak.py', 'A', 1, 0),
        ('new.py', 'A', 1, 0),
        ('with space.py', 'A', 1, 0),
    ]


async def test_deleted_large_binary_and_outside_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / 'work'
    workspace.mkdir()
    (workspace / 'big.txt').write_text('x' * 2000)
    (workspace / 'image.bin').write_bytes(b'\0\1')
    (workspace / 'folder').mkdir()
    (tmp_path / 'outside.py').write_text('out = 2\n')
    (workspace / 'long.py').write_text('\n'.join(f'line {n}' for n in range(30)))
    history = History()
    history.turn(
        'one',
        write('gone.py', 'a = 1\n'),
        write('temporary.py', 'a = 1\n'),
        write('big.txt', 'x'),
        write('image.bin', ''),
        write('folder', ''),
        write('../outside.py', 'out = 1\n'),
        write('long.py', ''),
        (WriteFileCallPart, 'write_file', {'path': 1, 'content': ''}),
        (EditFileCallPart, 'edit_file', {'path': 'x.py', 'replacements': ['no']}),
        (EditFileCallPart, 'edit_file', {'path': 'x.py', 'new_text': 'missing old_text'}),
        # Another tool that happens to be called `write_file`, such as a sandbox's, is not the file system's.
        (ToolCallPart, 'write_file', {'path': 'x.py', 'content': ''}),
    )
    history.turn('two', edit('gone.py', 'a = 1', 'a = 2'), edit('../outside.py', 'out = 1', 'out = 2'))
    monkeypatch.setattr(changes_module, 'MAX_FILE_BYTES', 1000)
    monkeypatch.setattr(changes_module, 'MAX_DIFF_CHARS', 100)

    session = await session_changes(history.messages, workspace, scope='session')
    by_path = {change.path: change for change in session.files}
    assert sorted(by_path) == sorted([str(tmp_path / 'outside.py'), 'big.txt', 'folder', 'image.bin', 'long.py'])
    assert (by_path['big.txt'].status, by_path['big.txt'].note) == ('A', 'Too large to diff.')
    assert by_path['image.bin'].note == 'Binary file.'
    assert by_path['folder'].note == 'Is a directory'
    assert by_path['long.py'].diff.truncated
    assert by_path['long.py'].added == 30
    assert len(by_path['long.py'].diff.diff) <= 100

    turn = await session_changes(history.messages, workspace, scope='turn')
    assert listed(turn) == [(str(tmp_path / 'outside.py'), 'M', 1, 1), ('gone.py', 'D', 0, 1)]


async def test_uncommitted_matches_git_diff_head(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (repo / 'a.py').write_text('def a():\n    return 10\n')
    (repo / 'b.py').unlink()
    (repo / 'c.py').write_text('c = 1\n')
    (repo / 'logo.png').write_bytes(b'\0PNG')
    git(repo, 'add', 'c.py', 'logo.png')
    (repo / 'untracked.py').write_text('ignored by git diff HEAD\n')

    found = await session_changes([], repo, scope='uncommitted')
    numstat = [line.split('\t') for line in git(repo, 'diff', 'HEAD', '--numstat').splitlines()]
    assert [(path, str(added), str(removed)) for path, _, added, removed in listed(found)] == [
        (path, '0' if added == '-' else added, '0' if removed == '-' else removed) for added, removed, path in numstat
    ]
    assert listed(found) == [('a.py', 'M', 1, 1), ('b.py', 'D', 0, 2), ('c.py', 'A', 1, 0), ('logo.png', 'A', 0, 0)]
    assert found.files[0].diff.diff.startswith('--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,2 @@')
    assert found.files[1].diff.diff.startswith('--- a/b.py\n+++ /dev/null\n@@ -1,2 +0,0 @@')
    assert found.files[2].diff.diff.startswith('--- /dev/null\n+++ b/c.py\n')
    assert found.files[3].note == 'Binary file.'

    counted = await session_changes([], repo, scope='uncommitted', diffs=False)
    assert (listed(counted), {change.diff.diff for change in counted.files}) == (listed(found), {''})

    monkeypatch.setattr(changes_module, 'MAX_FILES', 3)
    capped = await session_changes([], repo, scope='uncommitted')
    assert ([change.path for change in capped.files], capped.omitted) == (['a.py', 'b.py', 'c.py'], 1)


async def test_branch_against_its_base(repo: Path) -> None:
    git(repo, 'switch', '-q', '-c', 'feature')
    (repo / 'feature.py').write_text('feature = 1\n')
    git(repo, 'add', '.')
    git(repo, 'commit', '-q', '-m', 'feature')
    (repo / 'a.py').write_text('def a():\n    return 10\n')

    found = await session_changes([], repo, scope='branch')
    assert (found.base, listed(found)) == ('main', [('a.py', 'M', 1, 1), ('feature.py', 'A', 1, 0)])

    # A clone's `origin/HEAD` names the base instead.
    git(repo, 'symbolic-ref', 'refs/remotes/origin/HEAD', 'refs/remotes/origin/trunk')
    assert (await session_changes([], repo, scope='branch')).base == 'main', 'origin/HEAD names no branch yet'
    git(repo, 'update-ref', 'refs/remotes/origin/trunk', 'main')
    assert (await session_changes([], repo, scope='branch')).base == 'origin/trunk'

    # A base with no history in common, as after a shallow clone, says why it cannot compare.
    empty_tree = git(repo, 'hash-object', '-t', 'tree', '/dev/null').strip()
    unrelated_commit = git(repo, 'commit-tree', empty_tree, '-m', 'unrelated').strip()
    git(repo, 'update-ref', 'refs/remotes/origin/trunk', unrelated_commit)
    unrelated = await session_changes([], repo, scope='branch')
    assert (unrelated.base, unrelated.note) == ('origin/trunk', 'Git could not compare: git merge-base failed')


@pytest.mark.parametrize('scope', ['uncommitted', 'branch'])
async def test_git_scopes_say_why_they_show_nothing(tmp_path: Path, scope: ChangeScope) -> None:
    assert (await session_changes([], tmp_path, scope=scope)).note == 'This workspace is not in a Git repository.'
    git(tmp_path, 'init', '-q', '-b', 'trunk')
    assert (await session_changes([], tmp_path, scope=scope)).note == 'This repository has no commits yet.'


async def test_branch_without_a_base(repo: Path) -> None:
    git(repo, 'branch', '-q', '-m', 'trunk')
    found = await session_changes([], repo, scope='branch')
    assert found.note == 'No base branch found: neither origin/HEAD nor main or master.'


@pytest.mark.parametrize('scope', ['uncommitted', 'branch'])
async def test_git_scopes_without_git(repo: Path, monkeypatch: pytest.MonkeyPatch, scope: ChangeScope) -> None:
    monkeypatch.setenv('PATH', '')
    assert (await session_changes([], repo, scope=scope)).note == 'Git is not installed.'


async def test_a_written_file_counts_as_created_without_git(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no Git to ask for the last commit, a file the session wrote whole counts as created."""
    monkeypatch.setenv('PATH', '')
    history = History()
    history.turn('one', write('a.py', 'def a():\n    return 1\n'))
    assert listed(await session_changes(history.messages, repo, scope='session')) == [('a.py', 'A', 2, 0)]


async def test_git_scopes_say_when_git_takes_too_long(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(changes_module, 'GIT_TIMEOUT', 0)
    note = (await session_changes([], repo, scope='uncommitted')).note
    assert note == 'Git could not compare: git rev-parse took longer than 0 seconds'


async def test_a_written_file_counts_as_created_when_git_takes_too_long(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The last commit Git did not give in time is not used, as without Git."""
    monkeypatch.setattr(changes_module, 'GIT_TIMEOUT', 0)
    (repo / 'a.py').write_text('def a():\n    return 10\n')
    history = History()
    history.turn('one', write('a.py', 'def a():\n    return 10\n'))
    assert listed(await session_changes(history.messages, repo, scope='session')) == [('a.py', 'A', 2, 0)]


def test_changes_route(make_fleet: MakeFleet) -> None:
    """`GET /api/sessions/ID/changes` answers for a live session, and for the default one unprefixed."""
    fleet = make_fleet()
    conversation = fleet.new('work')
    workspace = fleet.folder('work')
    history = History()
    history.turn('create a.py', write('a.py', A1))
    fleet.portal.call(fleet.runtimes['work'].session.commit_messages, history.messages)
    (workspace / 'a.py').write_text(A1)

    response = fleet.get(f'/api/sessions/{conversation}/changes?scope=turn')
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body['scope'], [(file['path'], file['status']) for file in body['files']]) == ('turn', [('a.py', 'A')])
    assert fleet.get(f'/api/sessions/{conversation}/changes').json()['scope'] == 'session'
    [counted] = fleet.get(f'/api/sessions/{conversation}/changes?diffs=false').json()['files']
    assert (counted['path'], counted['status'], counted['added'], counted['diff']['diff']) == ('a.py', 'A', 0, '')
    assert fleet.get('/api/changes?scope=uncommitted').json()['note'] == 'This workspace is not in a Git repository.'
    assert fleet.get('/api/sessions/nope/changes').status_code == 404
    assert fleet.get(f'/api/sessions/{conversation}/changes?scope=everything').status_code == 422
