"""The Changes panel: what a session changed, as a file list with a unified diff per file.

`Last turn` and `This session` come from the session's own `write_file` and `edit_file` calls, so
they work outside Git too: each file's diff runs from its content before the first call in scope to
its content on disk now, so a file edited three times shows one diff. That earlier content is
replayed from the calls (see `_before`), with the file's last commit as the starting point the
calls cannot give. `Uncommitted` and `Branch vs base` run Git in the session's workspace. The
payloads are mirrored by `web-ui/src/protocol.ts`.
"""

import difflib
import posixpath
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

from anyio import run_process, to_thread

from pydantic_ai.messages import ModelMessage, ModelRequest, ToolCallPart, ToolReturnPart
from pydantic_clai2.builtin_plugins.web.protocol import Diff
from pydantic_clai2.runtime.imported_history import is_typed_prompt
from pydantic_clai2.runtime.session import STEERING

ChangeScope = Literal['turn', 'session', 'uncommitted', 'branch']
FileStatus = Literal['A', 'M', 'D']

MAX_DIFF_CHARS = 100_000
"""Characters of one file's diff the page receives; the rest is cut, on a line boundary."""
MAX_FILE_BYTES = 2_000_000
"""Files larger than this are listed without a diff."""
MAX_FILES = 300
"""Files listed in one scope; the rest are counted in `Changes.omitted`, so a huge diff cannot stall the page."""
BASES = ('main', 'master', 'origin/main', 'origin/master')
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


async def session_changes(messages: Sequence[ModelMessage], workspace: Path, *, scope: ChangeScope) -> Changes:
    """The changes in `scope` of the session whose history is `messages`, working in `workspace`."""
    if scope == 'uncommitted' or scope == 'branch':
        return await _git_changes(workspace, scope=scope)
    calls, turns = file_calls(messages)
    last_turn = turns if scope == 'turn' else None
    return await to_thread.run_sync(lambda: _tool_changes(calls, workspace, turn=last_turn))


# The session's own file tools.


@dataclass(kw_only=True)
class FileCall:
    """A `write_file` or `edit_file` call that succeeded, in the turn that made it."""

    turn: int
    path: str
    content: str | None = None
    """What a `write_file` wrote; `None` for an edit."""
    replacements: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    """An edit's `(old_text, new_text)` pairs, in order."""


def file_calls(messages: Sequence[ModelMessage]) -> tuple[list[FileCall], int]:
    """Each successful `write_file` and `edit_file` call in `messages`, in order, and the number of turns.

    A turn starts at each prompt the user typed, numbered from 1; steering a running turn starts none.
    """
    turn = 0
    pending: dict[str, FileCall] = {}
    found: list[FileCall] = []
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


def _file_call(part: ToolCallPart, turn: int) -> FileCall | None:
    if part.tool_name not in ('write_file', 'edit_file'):
        return None
    args = part.args_as_dict()
    path = args.get('path')
    if not isinstance(path, str):
        return None
    if part.tool_name == 'write_file':
        content = args.get('content')
        return FileCall(turn=turn, path=path, content=content) if isinstance(content, str) else None
    raw = args.get('replacements')
    pairs = cast(list[object], raw) if isinstance(raw, list) else [args]
    replacements: list[tuple[str, str]] = []
    for pair in pairs:
        if not isinstance(pair, dict):
            return None
        old, new = cast(dict[str, object], pair).get('old_text'), cast(dict[str, object], pair).get('new_text')
        if not isinstance(old, str) or not isinstance(new, str):
            return None
        replacements.append((old, new))
    return FileCall(turn=turn, path=path, replacements=replacements)


class _Unknown:
    """A file's content the calls cannot tell."""


UNKNOWN = _Unknown()
Content = str | None | _Unknown
"""A file's text, `None` for a file that does not exist, or `UNKNOWN`."""


def _tool_changes(calls: list[FileCall], workspace: Path, *, turn: int | None) -> Changes:
    """The files `calls` changed, all of them, or only those of `turn`."""
    by_path: dict[str, list[FileCall]] = {}
    for call in calls:
        by_path.setdefault(_shown_path(call.path, workspace), []).append(call)
    files: list[FileChange] = []
    for path, ops in by_path.items():
        first = next((index for index, op in enumerate(ops) if turn is None or op.turn == turn), None)
        if first is None:
            continue
        created = first == 0 and ops[0].content is not None
        try:
            now = _read(workspace / path)
        except _Undiffable as exc:
            files.append(_undiffed(path, 'A' if created else 'M', str(exc)))
            continue
        before = _before(ops, first, now, lambda: _committed(workspace, path))
        if isinstance(before, _Unknown):
            note = 'Its content before these changes is not known, so it has no diff.'
            files.append(_undiffed(path, 'D' if now is None else 'M', note))
        elif (change := file_change(path, before, now)) is not None:
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
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise _Undiffable(exc.strerror or str(exc)) from exc
    if len(data) > MAX_FILE_BYTES:
        raise _Undiffable('Too large to diff.')
    if b'\0' in data:
        raise _Undiffable('Binary file.')
    return data.decode('utf-8', errors='replace')


def _before(ops: list[FileCall], first: int, now: str | None, committed: Callable[[], str | None]) -> Content:
    """The file's content before `ops[first]`, replayed from the calls around it.

    Forward from the last whole write before it, else back from the disk through the edits since.
    When neither tells, forward from the file's last commit, `committed()`, if that leads to what is on
    disk now; a file with no commit that the session first wrote whole counts as created.
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
    base = committed()
    if base is None:
        return None if ops[0].content is not None else UNKNOWN
    return _replay(base, ops[:first]) if _replay(base, ops) == now else UNKNOWN


def _replay(text: Content, ops: list[FileCall]) -> Content:
    """`text` after `ops`, in order."""
    for op in ops:
        text = op.content if op.content is not None else _apply(text, op.replacements)
    return text


def _committed(workspace: Path, path: str) -> str | None:
    """`path`'s content in the last commit of the workspace's Git repository; `None` without one, or untracked."""
    if posixpath.isabs(path):
        return None
    try:
        result = subprocess.run(
            ['git', '--no-optional-locks', 'show', f'HEAD:./{path}'], cwd=workspace, capture_output=True, check=False
        )
    except OSError:
        return None
    return result.stdout.decode('utf-8', errors='replace') if result.returncode == 0 else None


def _apply(text: Content, replacements: list[tuple[str, str]]) -> Content:
    """`text` with each `(old, new)` replaced in order, as `edit_file` does; `UNKNOWN` unless each matches once."""
    for old, new in replacements:
        if not isinstance(text, str) or not old or text.count(old) != 1:
            return UNKNOWN
        text = text.replace(old, new, 1)
    return text


def file_change(path: str, old: str | None, new: str | None) -> FileChange | None:
    """The change from `old` to `new`, each `None` for a file that does not exist; `None` when they are the same."""
    if old == new:
        return None
    status: FileStatus = 'A' if old is None else 'D' if new is None else 'M'
    lines = difflib.unified_diff(
        (old or '').splitlines(),
        (new or '').splitlines(),
        fromfile='/dev/null' if old is None else f'a/{path}',
        tofile='/dev/null' if new is None else f'b/{path}',
        lineterm='',
    )
    diff = '\n'.join(lines)
    added, removed = _counts(diff)
    return FileChange(path=path, status=status, added=added, removed=removed, diff=_cut(path, diff))


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


class GitError(Exception):
    """A Git command failed; its message is what Git printed."""


async def _git(workspace: Path, *args: str) -> str:
    result = await run_process(
        ['git', '--no-optional-locks', '-c', 'core.quotepath=off', *args], cwd=workspace, check=False
    )
    if result.returncode != 0:
        raise GitError(result.stderr.decode(errors='replace').strip() or f'git {args[0]} failed')
    return result.stdout.decode(errors='replace')


async def _git_changes(workspace: Path, *, scope: Literal['uncommitted', 'branch']) -> Changes:
    try:
        await _git(workspace, 'rev-parse', '--show-toplevel')
    except (GitError, OSError):
        return Changes(scope=scope, note='This workspace is not in a Git repository.')
    try:
        await _git(workspace, 'rev-parse', '--verify', '--quiet', 'HEAD')
    except GitError:
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
        numstat = await _git(workspace, 'diff', '--numstat', '-z', *options, against)
        patch = await _git(workspace, 'diff', *options, against)
    except GitError as exc:
        return Changes(scope=scope, base=base, note=f'Git could not compare: {exc}')
    return _listed(Changes(scope=scope, base=base), git_files(numstat, patch))


async def _base(workspace: Path) -> str | None:
    """The branch this one is compared with: `origin/HEAD`'s, else the first of `BASES` that exists."""
    try:
        names = [(await _git(workspace, 'symbolic-ref', '--quiet', '--short', 'refs/remotes/origin/HEAD')).strip()]
    except GitError:
        names = []
    for name in [*names, *BASES]:
        try:
            await _git(workspace, 'rev-parse', '--verify', '--quiet', f'{name}^{{commit}}')
        except GitError:
            continue
        return name
    return None


def git_files(numstat: str, patch: str) -> list[FileChange]:
    """The files `git diff --numstat -z` lists, each with its part of the same diff's patch, which lists them in order.

    The counts are Git's own, so the list matches `git diff --stat`.
    """
    entries = [entry.split('\t', 2) for entry in numstat.split('\0') if entry]
    blocks = patch.split('\ndiff --git ')
    files: list[FileChange] = []
    for (added, removed, path), block in zip(entries, blocks, strict=False):
        head, _, body = block.partition('\n--- ')
        status: FileStatus = 'A' if '\nnew file mode' in head else 'D' if '\ndeleted file mode' in head else 'M'
        if added == '-':
            files.append(FileChange(path=path, status=status, diff=Diff(path=path, diff=''), note='Binary file.'))
        else:
            diff = _cut(path, f'--- {body}' if body else '')
            files.append(FileChange(path=path, status=status, added=int(added), removed=int(removed), diff=diff))
    return files
