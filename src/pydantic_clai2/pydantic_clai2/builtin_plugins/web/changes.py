"""The Changes panel: what a session changed, as a file list with a unified diff per file.

`Last turn` and `This session` come from the session's own `write_file` and `edit_file` calls, so
they work outside Git too: each file's diff runs from its content before the first call in scope to
its content on disk now, so a file edited three times shows one diff. That earlier content is
replayed from the calls (see `_before`), with the file's last commit as the starting point the
calls cannot give. `Uncommitted` and `Branch vs base` run Git in the session's workspace. The
payloads are mirrored by `web-ui/src/protocol.ts`.
"""

import difflib
import os
import posixpath
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from anyio import fail_after, run_process, to_thread

from pydantic_ai.messages import ModelMessage, ModelRequest, ToolCallPart, ToolReturnPart
from pydantic_ai_harness.filesystem import EditFileCallPart, ReplacementArgs, WriteFileCallPart
from pydantic_clai2.builtin_plugins.web.protocol import Diff
from pydantic_clai2.runtime.imported_history import is_typed_prompt
from pydantic_clai2.runtime.session import STEERING

ChangeScope = Literal['turn', 'session', 'uncommitted', 'branch']
FileStatus = Literal['A', 'M', 'D']

MAX_DIFF_CHARS = 100_000
"""Characters of one file's diff the page receives; the rest is cut, on a line boundary."""
MAX_FILE_BYTES = 2_000_000
"""Files larger than this are listed without a diff."""
GIT_TIMEOUT = 30
"""Seconds one Git command may take before the panel gives up on it."""
MAX_FILES = 300
"""Files listed in one scope; the rest are counted in `Changes.omitted`, so a huge diff cannot stall the page."""
_BASES = ('main', 'master', 'origin/main', 'origin/master')
"""Branches tried, in order, as the base when `origin/HEAD` names none."""


@dataclass(kw_only=True)
class FileChange:
    """One changed file: its status, its added and removed line counts, and its unified diff."""

    path: str
    status: FileStatus
    added: int = 0
    removed: int = 0
    diff: Diff
    note: str | None = None
    """Why the diff is empty, such as a binary file."""


@dataclass(kw_only=True)
class Changes:
    """What a session changed in one scope."""

    scope: ChangeScope
    files: list[FileChange] = field(default_factory=list[FileChange])
    base: str | None = None
    """The branch `Branch vs base` compares with."""
    note: str | None = None
    """Why the scope cannot be shown, such as a workspace outside Git."""
    omitted: int = 0
    """Changed files left out past `MAX_FILES`."""


async def session_changes(
    messages: Sequence[ModelMessage], workspace: Path, *, scope: ChangeScope, diffs: bool = True
) -> Changes:
    """The changes in `scope` of the session whose history is `messages`, working in `workspace`.

    Without `diffs`, each file comes with an empty diff, as for a count of the files: no diff is built, so the
    session scopes leave the line counts at 0, while Git's scopes keep Git's counts.
    """
    if scope == 'uncommitted' or scope == 'branch':
        return await _git_changes(workspace, scope=scope, diffs=diffs)
    calls, turns = _file_calls(messages)
    committed = await _committed(workspace, list(dict.fromkeys(_shown_path(call.path, workspace) for call in calls)))
    last_turn = turns if scope == 'turn' else None
    return await to_thread.run_sync(lambda: _tool_changes(calls, workspace, committed, turn=last_turn, diffs=diffs))


# The session's own file tools.


@dataclass(kw_only=True)
class _FileCall:
    """A `write_file` or `edit_file` call that succeeded, in the turn that made it."""

    turn: int
    path: str
    content: str | None = None
    """What a `write_file` wrote; `None` for an edit."""
    replacements: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    """An edit's `(old_text, new_text)` pairs, in order."""


def _file_calls(messages: Sequence[ModelMessage]) -> tuple[list[_FileCall], int]:
    """Each successful `write_file` and `edit_file` call in `messages`, in order, and the number of turns.

    A turn starts at each prompt the user typed, numbered from 1; steering a running turn starts none.
    """
    turn = 0
    pending: dict[str, _FileCall] = {}
    found: list[_FileCall] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            if not (message.metadata or {}).get(STEERING) and any(is_typed_prompt(part) for part in message.parts):
                turn += 1
            for part in message.parts:
                if isinstance(part, ToolReturnPart) and part.outcome == 'success':
                    if (call := pending.pop(part.tool_call_id, None)) is not None:
                        found.append(call)
            continue
        for part in message.parts:
            if isinstance(part, ToolCallPart) and (call := _file_call(part, turn)) is not None:
                pending[part.tool_call_id] = call
    return found, turn


def _file_call(part: ToolCallPart, turn: int) -> _FileCall | None:
    """The file system's write or edit that `part` makes, by its tool kind; `None` for any other call."""
    if isinstance(part, WriteFileCallPart) and (written := part.typed_args) is not None:
        return _FileCall(turn=turn, path=written['path'], content=written['content'])
    if not isinstance(part, EditFileCallPart) or (edited := part.typed_args) is None:
        return None
    pairs = edited.get('replacements')
    if not pairs:
        old, new = edited.get('old_text'), edited.get('new_text')
        if old is None or new is None:
            # `edit_file` refuses such a call, so none succeeds.
            return None
        pairs = [ReplacementArgs(old_text=old, new_text=new)]
    return _FileCall(
        turn=turn, path=edited['path'], replacements=[(pair['old_text'], pair['new_text']) for pair in pairs]
    )


class _Unknown:
    """A file's content the calls cannot tell."""


_UNKNOWN = _Unknown()
_Content = str | None | _Unknown
"""A file's text, `None` for a file that does not exist, or `_UNKNOWN`."""


def _tool_changes(
    calls: list[_FileCall], workspace: Path, committed: dict[str, str], *, turn: int | None, diffs: bool
) -> Changes:
    """The files `calls` changed, all of them, or only those of `turn`; `committed` has their last commit's content."""
    by_path: dict[str, list[_FileCall]] = {}
    for call in calls:
        by_path.setdefault(_shown_path(call.path, workspace), []).append(call)
    files: list[FileChange] = []
    for path, ops in by_path.items():
        first = next((index for index, op in enumerate(ops) if turn is None or op.turn == turn), None)
        if first is None:
            continue
        base = committed.get(path)
        try:
            now = _read(workspace / path)
        except _Undiffable as exc:
            created = first == 0 and ops[0].content is not None and (base is None or base in _states(base, ops))
            files.append(_undiffed(path, 'A' if created else 'M', str(exc)))
            continue
        before = _before(ops, first, now, base)
        if isinstance(before, _Unknown):
            note = 'Its content before these changes is not known, so it has no diff.'
            files.append(_undiffed(path, 'D' if now is None else 'M', note))
        elif (change := _file_change(path, before, now, diffs=diffs)) is not None:
            files.append(change)
    return _listed(Changes(scope='session' if turn is None else 'turn'), files)


def _listed(changes: Changes, files: list[FileChange]) -> Changes:
    """`changes` with `files` by path, up to `MAX_FILES` of them."""
    changes.files = sorted(files, key=lambda change: change.path)[:MAX_FILES]
    changes.omitted = len(files) - len(changes.files)
    return changes


def _undiffed(path: str, status: FileStatus, note: str) -> FileChange:
    return FileChange(path=path, status=status, diff=Diff(path=path, diff=''), note=note)


def _shown_path(path: str, workspace: Path) -> str:
    """`path` as the workspace-relative POSIX path it names, or the absolute path of one outside the workspace."""
    full = posixpath.normpath(posixpath.join(workspace.as_posix(), path))
    root = workspace.as_posix().rstrip('/') + '/'
    return full.removeprefix(root)


class _Undiffable(Exception):
    """A file that is listed without a diff, for the reason the message gives."""


def _read(path: Path) -> str | None:
    """The file's text now, or `None` when it is gone."""
    try:
        with path.open('rb') as file:
            data = file.read(MAX_FILE_BYTES + 1)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise _Undiffable(exc.strerror or str(exc)) from exc
    if len(data) > MAX_FILE_BYTES:
        raise _Undiffable('Too large to diff.')
    if b'\0' in data:
        raise _Undiffable('Binary file.')
    return data.decode('utf-8', errors='replace')


def _before(ops: list[_FileCall], first: int, now: str | None, base: str | None) -> _Content:
    """The file's content before `ops[first]`, replayed from the calls around it.

    Forward from the last whole write before it, else back from the disk through the edits since.
    When neither tells, forward from `base`, the file's content in the last commit, if that leads to
    what is on disk now. A file the session first wrote whole counts as created when it has no
    commit, or when its commit holds content the session made, as when the session committed it.
    """
    written = next((index for index in range(first - 1, -1, -1) if ops[index].content is not None), None)
    if written is not None:
        return _replay(ops[written].content, ops[written + 1 : first])
    if all(op.content is None for op in ops[first:]):
        undone = now
        for op in reversed(ops[first:]):
            undone = _apply(undone, [(new, old) for old, new in reversed(op.replacements)])
        if not isinstance(undone, _Unknown):
            return undone
    if ops[0].content is not None and (base is None or base in _states(base, ops)):
        return None
    if base is None or _replay(base, ops) != now:
        return _UNKNOWN
    return _replay(base, ops[:first])


def _replay(text: _Content, ops: list[_FileCall]) -> _Content:
    """`text` after `ops`, in order."""
    return _states(text, ops)[-1]


def _states(text: _Content, ops: list[_FileCall]) -> list[_Content]:
    """`text`, then the content after each of `ops`, in order."""
    states = [text]
    for op in ops:
        states.append(op.content if op.content is not None else _apply(states[-1], op.replacements))
    return states[1:] if ops else states


async def _committed(workspace: Path, paths: list[str]) -> dict[str, str]:
    """The content in the last commit of each of `paths` the workspace's Git repository tracks, in one Git call."""
    # Git reads one name a line, so a name with a line break in it is never asked for.
    relative = [path for path in paths if not posixpath.isabs(path) and '\n' not in path and '\r' not in path]
    try:
        # Outside a repository `cat-file` would quit before reading its input, breaking the pipe.
        await _run_git(workspace, 'rev-parse', '--verify', '--quiet', 'HEAD')
        output = await _run_git(
            workspace, 'cat-file', '--batch', input=''.join(f'HEAD:./{path}\n' for path in relative).encode()
        )
    except (_GitError, OSError):
        return {}
    found: dict[str, str] = {}
    at = 0
    for path in relative:
        end = output.index(b'\n', at)
        header, at = output[at:end], end + 1
        if header.endswith((b' missing', b' ambiguous')):
            continue
        kind, size = header.rsplit(b' ', 2)[1:]
        if kind == b'blob' and int(size) <= MAX_FILE_BYTES:
            found[path] = output[at : at + int(size)].decode('utf-8', errors='replace')
        at += int(size) + 1
    return found


def _apply(text: _Content, replacements: list[tuple[str, str]]) -> _Content:
    """`text` with each `(old, new)` replaced in order, as `edit_file` does; `_UNKNOWN` unless each matches once."""
    for old, new in replacements:
        if not isinstance(text, str) or not old or text.count(old) != 1:
            return _UNKNOWN
        text = text.replace(old, new, 1)
    return text


def _file_change(path: str, old: str | None, new: str | None, *, diffs: bool = True) -> FileChange | None:
    """The change from `old` to `new`, each `None` for a file that does not exist; `None` when they are the same.

    Without `diffs`, only its status: no diff is built, and no lines counted.
    """
    if old == new:
        return None
    status: FileStatus = 'A' if old is None else 'D' if new is None else 'M'
    if not diffs:
        return FileChange(path=path, status=status, diff=Diff(path=path, diff=''))
    lines = difflib.unified_diff(
        _lines(old or ''),
        _lines(new or ''),
        fromfile='/dev/null' if old is None else f'a/{path}',
        tofile='/dev/null' if new is None else f'b/{path}',
        lineterm='',
    )
    diff = '\n'.join(line[:-1] + _NO_NEWLINE if line.endswith('\n') else line for line in lines)
    added, removed = _counts(diff)
    return FileChange(path=path, status=status, added=added, removed=removed, diff=_cut(path, diff))


_NO_NEWLINE = '\n\\ No newline at end of file'
"""The marker `git diff` prints after a final line that lacks a newline."""


def _lines(text: str) -> list[str]:
    """`text`'s lines, with a line break kept on the last one when the text does not end in one.

    No other line can hold a line break, so a change to the final newline alone still differs.
    """
    lines = text.splitlines()
    if lines and not text.endswith(('\n', '\r')):
        lines[-1] += '\n'
    return lines


def _counts(diff: str) -> tuple[int, int]:
    """A unified diff's added and removed lines; its `---`/`+++` headers come before the first hunk."""
    added = removed = 0
    in_hunk = False
    for line in diff.splitlines():
        if line.startswith('@@'):
            in_hunk = True
        elif in_hunk and line.startswith('+'):
            added += 1
        elif in_hunk and line.startswith('-'):
            removed += 1
    return added, removed


def _cut(path: str, diff: str) -> Diff:
    """`diff` as the page receives it: cut on a line boundary past `MAX_DIFF_CHARS`."""
    if len(diff) <= MAX_DIFF_CHARS:
        return Diff(path=path, diff=diff)
    cut = diff.rfind('\n', 0, MAX_DIFF_CHARS + 1)
    return Diff(path=path, diff=diff[: cut if cut >= 0 else MAX_DIFF_CHARS], truncated=True)


# Git.


class _GitError(Exception):
    """A Git command failed; its message is what Git printed."""


async def _git(workspace: Path, *args: str) -> str:
    return (await _run_git(workspace, *args)).decode(errors='replace')


async def _run_git(workspace: Path, *args: str, input: bytes | None = None) -> bytes:
    """What Git printed for `args` in `workspace`; `_GitError` with its message when it fails or takes too long."""
    command = ['git', '--no-optional-locks', '-c', 'core.quotepath=off', *args]
    try:
        with fail_after(GIT_TIMEOUT):
            # In English, so a missing repository can be told by its message.
            result = await run_process(
                command, input=input, cwd=workspace, env={**os.environ, 'LC_ALL': 'C'}, check=False
            )
    except TimeoutError:
        raise _GitError(f'git {args[0]} took longer than {GIT_TIMEOUT} seconds') from None
    if result.returncode != 0:
        raise _GitError(result.stderr.decode(errors='replace').strip() or f'git {args[0]} failed')
    return result.stdout


async def _git_changes(workspace: Path, *, scope: Literal['uncommitted', 'branch'], diffs: bool) -> Changes:
    try:
        await _git(workspace, 'rev-parse', '--show-toplevel')
    except OSError:
        return Changes(scope=scope, note='Git is not installed.')
    except _GitError as exc:
        if 'not a git repository' in str(exc):
            return Changes(scope=scope, note='This workspace is not in a Git repository.')
        return Changes(scope=scope, note=f'Git could not compare: {exc}')
    try:
        await _git(workspace, 'rev-parse', '--verify', '--quiet', 'HEAD')
    except _GitError:
        return Changes(scope=scope, note='This repository has no commits yet.')
    base: str | None = None
    against = 'HEAD'
    options = ('--no-color', '--no-ext-diff', '--no-renames')
    try:
        if scope == 'branch':
            base = await _base(workspace)
            if base is None:
                return Changes(scope=scope, note='No base branch found: neither origin/HEAD nor main or master.')
            against = (await _git(workspace, 'merge-base', 'HEAD', base)).strip()
        listing = await _git(workspace, 'diff', '--raw', '--numstat', '-z', *options, against)
        patch = await _git(workspace, 'diff', *options, against) if diffs else None
    except _GitError as exc:
        return Changes(scope=scope, base=base, note=f'Git could not compare: {exc}')
    return _listed(Changes(scope=scope, base=base), _git_files(listing, patch))


async def _base(workspace: Path) -> str | None:
    """The branch this one is compared with: `origin/HEAD`'s, else the first of `_BASES` that exists."""
    try:
        names = [(await _git(workspace, 'symbolic-ref', '--quiet', '--short', 'refs/remotes/origin/HEAD')).strip()]
    except _GitError:
        names = []
    for name in [*names, *_BASES]:
        try:
            await _git(workspace, 'rev-parse', '--verify', '--quiet', f'{name}^{{commit}}')
        except _GitError:
            continue
        return name
    return None


def _git_files(listing: str, patch: str | None) -> list[FileChange]:
    """The files `git diff --raw --numstat -z` lists, one snapshot of statuses and counts, with their patch.

    The counts are Git's own, so the list matches `git diff --stat`. Each diff is found in `patch`, unless `None`,
    by its file's name, so a file that changed between the two Git runs loses its diff rather than taking another's.
    """
    fields = listing.split('\0')
    statuses: dict[str, str] = {}
    counts: list[tuple[str, str, str]] = []
    at = 0
    while at < len(fields) - 1:
        if fields[at].startswith(':'):
            statuses[fields[at + 1]] = fields[at].split()[-1]
            at += 2
        else:
            added, removed, path = fields[at].split('\t', 2)
            counts.append((added, removed, path))
            at += 1
    bodies = {} if patch is None else _patches(patch)
    files: list[FileChange] = []
    for added, removed, path in counts:
        letter = statuses.get(path)
        status: FileStatus = 'A' if letter == 'A' else 'D' if letter == 'D' else 'M'
        if added == '-':
            files.append(FileChange(path=path, status=status, diff=Diff(path=path, diff=''), note='Binary file.'))
        else:
            diff = _cut(path, bodies.get(path, ''))
            files.append(FileChange(path=path, status=status, added=int(added), removed=int(removed), diff=diff))
    return files


def _patches(patch: str) -> dict[str, str]:
    """Each file's part of a `git diff` patch, from its `---` line on, by the name its `+++` (or `---`) line gives."""
    found: dict[str, str] = {}
    for block in patch.split('\ndiff --git '):
        body = block.partition('\n--- ')[2]
        if not body:
            continue
        old, _, rest = body.partition('\n+++ ')
        new = rest.partition('\n')[0]
        name = old if new == '/dev/null' else new
        found[name.removeprefix('a/').removeprefix('b/')] = f'--- {body}'
    return found
