"""Stores for `WorkLog`: where each conversation's log of tool calls and replies is kept.

`WorkLogStore` is the narrow protocol for any backend: append a record under an idempotency key,
read entries back by `seq`, and name the log for a reset notice. `InMemoryWorkLogStore` keeps
each log in this process, `FileWorkLogStore` keeps one JSONL file per log on this machine, and
`WorkspaceWorkLogStore` keeps one file per entry in the run's workspace.

Each store assigns `seq` in memory under a per-log lock, so it supports one writer process per log.
"""

from __future__ import annotations

import hashlib
import os
import posixpath
import re
import stat
import weakref
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import anyio
import anyio.to_thread
from pydantic import TypeAdapter
from typing_extensions import TypedDict

from pydantic_ai.workspaces import Workspace, WorkspaceBackend, WorkspaceError
from pydantic_ai_harness._workspace import METADATA_DIR, metadata_dir, secondary_workspace

WORK_LOG_DIR = 'work-log'
"""The directory under `.pydantic-ai-harness` in the working directory that `WorkspaceWorkLogStore` writes to."""

WorkLogEntryKind = Literal['tool', 'text', 'thinking']
"""What a work log entry records: a tool call, a model reply, or a model thought."""

WorkLogOutcome = Literal['success', 'failed', 'retried']
"""How a recorded tool call ended, in `ToolReturnPart.outcome`'s terms."""


class WorkLogRecord(TypedDict):
    """One tool call, model reply or thought as `WorkLog` records it. JSON-serializable."""

    key: str
    """Idempotency key: a store writes a record whose key the log already holds only once."""
    run_id: str | None
    step: int
    """`RunContext.run_step` when it was recorded."""
    context: int
    """The number of the latest reset notice in the conversation when it was recorded, `0` before any."""
    kind: WorkLogEntryKind
    tool_name: str | None
    tool_call_id: str | None
    parent_tool_call_id: str | None
    """For a call made from another tool's code, such as `run_code`'s, that tool's call id."""
    args: str | None
    """The call's arguments as JSON."""
    outcome: WorkLogOutcome | None
    """`None` for a reply or thought."""
    content: str
    """The tool's full result, its failure, or the reply's text."""
    chars: int
    """`len(content)`."""
    timestamp: str
    """When it was recorded, in ISO 8601, UTC."""


class WorkLogEntry(WorkLogRecord):
    """A `WorkLogRecord` as a store keeps it: with its position in the log."""

    seq: int
    """Assigned by the store: 1-based and dense per log."""


_ENTRY = TypeAdapter(WorkLogEntry)


class WorkLogStore(Protocol):
    """Keep the entries of each conversation's work log, one log per `log_id`."""

    async def append(self, log_id: str, record: WorkLogRecord) -> int:
        """Persist `record` as the log's next entry and return its `seq`.

        When the log already holds an entry with `record['key']`, write nothing and return that
        entry's `seq`: a replayed hook or a retried durable operation appends the same record again.
        """
        ...  # pragma: no cover

    async def read(self, log_id: str, start: int = 1, stop: int | None = None) -> list[WorkLogEntry]:
        """Return the entries with `start <= seq < stop` in `seq` order; `stop=None` reads to the end."""
        ...  # pragma: no cover

    def locate(self, log_id: str) -> str | None:
        """Return the handle a reset notice shows for the log, or `None` when there is nothing the agent could open.

        Pure and synchronous, with no I/O, so it can be called from any context.
        """
        ...  # pragma: no cover


def file_name(log_id: str) -> str:
    """The file or directory name a persistent store keeps `log_id`'s log under: its SHA-256 hex digest.

    Never the id itself, so no id can name a path outside the store directory, and no two ids share a
    name, even on a file system that ignores case or drops a trailing dot.
    """
    return hashlib.sha256(log_id.encode()).hexdigest()


@dataclass
class InMemoryWorkLogStore:
    """Each log lives in this process until `discard` or process exit.

    A long-lived server should call `discard` when a conversation ends. The log is lost when the
    process exits, so a durable run that can resume after a worker restart needs `FileWorkLogStore`
    or `WorkspaceWorkLogStore`. The state is left out of comparisons, so
    `WorkLog(InMemoryWorkLogStore())` stated twice is one log.
    """

    _entries: dict[str, list[WorkLogEntry]] = field(
        default_factory=dict[str, list[WorkLogEntry]], init=False, repr=False, compare=False
    )
    _seqs: dict[str, dict[str, int]] = field(
        default_factory=dict[str, dict[str, int]], init=False, repr=False, compare=False
    )

    async def append(self, log_id: str, record: WorkLogRecord) -> int:
        # No `await` between reading and writing the log, so concurrent appends need no lock.
        seqs = self._seqs.setdefault(log_id, {})
        if (seq := seqs.get(record['key'])) is None:
            entries = self._entries.setdefault(log_id, [])
            seq = seqs[record['key']] = len(entries) + 1
            entries.append(WorkLogEntry(**record, seq=seq))
        return seq

    async def read(self, log_id: str, start: int = 1, stop: int | None = None) -> list[WorkLogEntry]:
        entries = self._entries.get(log_id, [])
        return entries[max(start, 1) - 1 : None if stop is None else max(stop - 1, 0)]

    def locate(self, log_id: str) -> None:
        # The log is in this process only: the agent reaches it through the tools, and the id may be client input.
        return None

    def discard(self, log_id: str) -> None:
        """Forget `log_id`'s log, such as when its conversation has ended."""
        self._entries.pop(log_id, None)
        self._seqs.pop(log_id, None)


def _locks() -> defaultdict[str, anyio.Lock]:
    return defaultdict(anyio.Lock)


@dataclass
class FileWorkLogStore:
    """One JSONL file per log on the machine running the agent, `<directory>/<log>.jsonl`.

    The handle a reset notice shows is the file's absolute path. The keys already written are read
    from the file the first time this process touches a log, then kept in memory. The directory is
    kept owner-only (`0700`) and each file is created `0600`: a log holds full tool results.
    """

    directory: str | os.PathLike[str]
    """The directory holding the log files. Created on the first write."""

    _root: Path = field(init=False, repr=False, compare=False)
    _seqs: dict[str, dict[str, int]] = field(
        default_factory=dict[str, dict[str, int]], init=False, repr=False, compare=False
    )
    _locks: defaultdict[str, anyio.Lock] = field(default_factory=_locks, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._root = Path(self.directory).absolute()

    def _path(self, log_id: str) -> Path:
        return self._root / f'{file_name(log_id)}.jsonl'

    async def append(self, log_id: str, record: WorkLogRecord) -> int:
        async with self._locks[log_id]:
            seqs = self._seqs.get(log_id)
            if seqs is None:
                entries = await anyio.to_thread.run_sync(self._load, log_id)
                seqs = self._seqs[log_id] = {entry['key']: entry['seq'] for entry in entries}
            if (seq := seqs.get(record['key'])) is None:
                seq = len(seqs) + 1
                line = _ENTRY.dump_json(WorkLogEntry(**record, seq=seq)).decode()
                await anyio.to_thread.run_sync(self._write_line, log_id, line)
                seqs[record['key']] = seq
            return seq

    async def read(self, log_id: str, start: int = 1, stop: int | None = None) -> list[WorkLogEntry]:
        # Under the append lock: an append writes from a worker thread, so an unlocked read could see half a line.
        async with self._locks[log_id]:
            entries = await anyio.to_thread.run_sync(self._load, log_id)
        return [entry for entry in entries if start <= entry['seq'] and (stop is None or entry['seq'] < stop)]

    def locate(self, log_id: str) -> str:
        return str(self._path(log_id))

    def _load(self, log_id: str) -> list[WorkLogEntry]:
        path = self._path(log_id)
        if not path.exists():
            return []
        self._check_root()
        # Split the bytes on newlines only: JSON leaves U+2028 and its kin unescaped, and `splitlines` would cut at
        # them. Every entry ends with a newline, so a last segment without one is a write a crash cut short: skip
        # it before decoding, since it can end inside a UTF-8 sequence.
        *lines, _ = path.read_bytes().split(b'\n')
        return [_ENTRY.validate_json(line) for line in lines if line]

    def _check_root(self) -> None:
        """Refuse a symlinked or foreign store directory, and keep it owner-only: logs hold full tool results."""
        st = self._root.lstat()
        if stat.S_ISLNK(st.st_mode):
            raise PermissionError(f'Work log directory {str(self._root)!r} is a symbolic link; refusing to use it.')
        geteuid = getattr(os, 'geteuid', None)
        if geteuid is None:
            return
        if st.st_uid != geteuid():
            raise PermissionError(
                f'Work log directory {str(self._root)!r} is not owned by the current user; refusing to use it.'
            )
        if stat.S_IMODE(st.st_mode) & 0o077:
            self._root.chmod(0o700)

    def _write_line(self, log_id: str, line: str) -> None:
        path = self._path(log_id)
        # Created `0700`, so no other user can reach the directory before `_check_root` runs.
        self._root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._check_root()
        with os.fdopen(os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600), 'a+b') as file:
            size = file.seek(0, os.SEEK_END)
            if size:
                file.seek(size - 1)
                if file.read(1) != b'\n':
                    # A write a crash cut short, which `_load` skips: drop it, so this entry starts its own line.
                    file.seek(0)
                    file.truncate(file.read().rfind(b'\n') + 1)
            file.write(f'{line}\n'.encode())


# The lookbehind keeps a search from retrying inside a run of digits, so scanning stays linear.
_ENTRY_FILE = re.compile(r'(?<!\d)(\d{8,})-([0-9a-f]{16})\.json')
"""An entry file's name: its zero-padded `seq`, then the start of its key's SHA-256 digest."""


@dataclass
class _LogDirectory:
    path: str
    real_path: str
    """`path` with symlinks resolved, to check entry paths against."""
    seqs: dict[str, int] | None = None
    """`seq` by key digest, listed from the directory on the first append."""


@dataclass
class WorkspaceWorkLogStore:
    """One file per entry in the run's workspace, or in a workspace of its own.

    Entries are files under `.pydantic-ai-harness/work-log/<log>/` in the working directory, which
    holds a `.gitignore` so they stay out of version control. The agent's own file and shell tools
    can read them there too. The `Workspace` protocol has no append, so each entry is its own file,
    named after its `seq` and its key's digest: idempotency is checked from file names alone.

    The store caches only file names and keys, never entries: `read` loads the files it returns, so
    a search reads every entry file, one round trip each in a remote sandbox.
    """

    workspace: WorkspaceBackend | None = None
    """A workspace to keep logs in instead of the run's, such as `LocalWorkspaceBackend('/var/work-logs')`.

    A backend, not the `LocalWorkspace` capability.
    """

    _workspace: Workspace | None = field(init=False, repr=False, compare=False)
    _directories: weakref.WeakKeyDictionary[Workspace, dict[str, _LogDirectory]] = field(
        default_factory=weakref.WeakKeyDictionary[Workspace, dict[str, _LogDirectory]],
        init=False,
        repr=False,
        compare=False,
    )
    """Each workspace's log directories; weak, so no finished run's workspace is kept."""
    _locks: defaultdict[str, anyio.Lock] = field(default_factory=_locks, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._workspace = secondary_workspace(self.workspace, 'WorkspaceWorkLogStore')

    def bind(self, workspace: Workspace) -> WorkLogStore:
        """Bind a run's workspace to the common store interface."""
        return _BoundWorkspaceWorkLogStore(self, workspace)

    def locate(self, log_id: str) -> str | None:
        """The log's directory, relative to the working directory, or `None` with a workspace of the store's own.

        The agent's tools reach only the run's workspace, so a log kept elsewhere is reachable only through the
        work log's tools.
        """
        if self._workspace is not None:
            return None
        return f'{METADATA_DIR}/{WORK_LOG_DIR}/{file_name(log_id)}/'

    async def append(self, workspace: Workspace, log_id: str, record: WorkLogRecord) -> int:
        """Write `record` as the next entry of `log_id`'s log and return its `seq`.

        `workspace` is the run's; the store's own `workspace`, when set, is used instead.
        """
        workspace = self._workspace or workspace
        digest = hashlib.sha256(record['key'].encode()).hexdigest()[:16]
        async with self._locks[log_id]:
            directory = await self._directory(workspace, log_id)
            seqs = directory.seqs
            if seqs is None:
                seqs = directory.seqs = {
                    match[2]: int(match[1])
                    for entry in await workspace.list_dir(directory.path)
                    if (match := _ENTRY_FILE.fullmatch(entry.name)) is not None
                }
            if (seq := seqs.get(digest)) is None:
                seq = len(seqs) + 1
                name = f'{seq:08d}-{digest}.json'
                path = posixpath.join(directory.path, name)
                # A symlink planted under the name would carry the write out of the store directory.
                if await workspace.realpath(path) != posixpath.join(directory.real_path, name):
                    raise WorkspaceError(
                        f'`{self.locate(log_id)}{name}` in the workspace is a symlink, so it is not used.'
                    )
                await workspace.write_bytes(path, _ENTRY.dump_json(WorkLogEntry(**record, seq=seq)))
                seqs[digest] = seq
            return seq

    async def read(
        self, workspace: Workspace, log_id: str, start: int = 1, stop: int | None = None
    ) -> list[WorkLogEntry]:
        """Read the entries of `log_id`'s log with `start <= seq < stop`, in `seq` order.

        `workspace` is the run's; the store's own `workspace`, when set, is used instead.
        """
        workspace = self._workspace or workspace
        directory = await self._directory(workspace, log_id)
        if await workspace.realpath(directory.path) != directory.real_path:
            raise WorkspaceError(f'`{self.locate(log_id)}` in the workspace is a symlink, so it is not used.')
        listed = [
            (int(match[1]), entry.path)
            for entry in await workspace.list_dir(directory.path)
            if (match := _ENTRY_FILE.fullmatch(entry.name)) is not None
        ]
        paths = sorted((seq, path) for seq, path in listed if start <= seq and (stop is None or seq < stop))
        entries: list[WorkLogEntry] = []
        for _, path in paths:
            # `read_bytes` follows a symlink planted under an entry's name, which could carry another log's entry.
            name = posixpath.basename(path)
            if await workspace.realpath(path) != posixpath.join(directory.real_path, name):
                raise WorkspaceError(f'`{self.locate(log_id)}{name}` in the workspace is a symlink, so it is not used.')
            entries.append(_ENTRY.validate_json(await workspace.read_bytes(path)))
        return entries

    async def _directory(self, workspace: Workspace, log_id: str) -> _LogDirectory:
        """The log's directory in `workspace`, created and checked on first use, then cached."""
        directories = self._directories.setdefault(workspace, {})
        if (directory := directories.get(log_id)) is None:
            base = await metadata_dir(workspace, WORK_LOG_DIR)
            path = posixpath.join(base, file_name(log_id))
            await workspace.make_dir(path)
            real_path = await workspace.realpath(path)
            if real_path != posixpath.join(await workspace.realpath(base), file_name(log_id)):
                raise WorkspaceError(f'`{self.locate(log_id)}` in the workspace is a symlink, so it is not used.')
            directory = directories[log_id] = _LogDirectory(path, real_path)
        return directory


@dataclass
class _BoundWorkspaceWorkLogStore:
    store: WorkspaceWorkLogStore
    workspace: Workspace

    async def append(self, log_id: str, record: WorkLogRecord) -> int:
        return await self.store.append(self.workspace, log_id, record)

    async def read(self, log_id: str, start: int = 1, stop: int | None = None) -> list[WorkLogEntry]:
        return await self.store.read(self.workspace, log_id, start, stop)

    def locate(self, log_id: str) -> str | None:
        return self.store.locate(log_id)
