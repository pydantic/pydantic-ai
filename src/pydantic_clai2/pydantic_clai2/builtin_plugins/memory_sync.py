"""Personal notebooks synced through Logfire (hackathon), so switching machines loses no notes.

Like sessions, the trace is the storage: every change to a personal note on this machine (by the agent's memory
tools or `/memory edit` and `forget`) is written as a `clai2 memory note` span carrying the note's store path, its
full content or a deletion tombstone, its SHA-256 and when it changed. Changes are found by scanning the notebook
folder after each turn and `/memory` command, and at the end of the session.

At the start of a session, the latest span per note for this user is read back with the query API and reconciled
with the folder, last write wins: a newer remote note is restored, a newer tombstone deletes the local file, and a
local note that is newer or missing in Logfire is written again so Logfire catches up. When Logfire can't be
queried the notebooks stay local, with one quiet warning. Every member of the Logfire project can read the notes.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

from opentelemetry.trace import TracerProvider
from pydantic import BaseModel, TypeAdapter, ValidationError

from pydantic_clai2.builtin_plugins.logfire_sessions import LogfireQuery, sql_attribute, sql_quote

SPAN_NAME = 'clai2 memory note'
SCOPE = 'pydantic-clai2.memory'
WINDOW = timedelta(days=365)
"""How far back notes are read: a note unchanged for longer is re-sent by any machine that still has it."""
MAX_BYTES = 1024 * 1024
"""A larger note stays on this machine; Logfire's attribute budget is meant for notes, not dumps."""


_SYNCED = TypeAdapter(dict[str, str])


def _sha(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


class RemoteNote(BaseModel):
    """The latest change to one note, as Logfire has it."""

    path: str
    content: str = ''
    deleted: bool = False
    sha256: str = ''
    at: float


@dataclass
class SyncState:
    """The SHA-256 each note had when it last matched Logfire, kept beside the other Logfire state."""

    file: Path

    def load(self) -> dict[str, str]:
        try:
            return _SYNCED.validate_json(self.file.read_bytes())
        except (OSError, ValidationError):
            return {}

    def save(self, synced: dict[str, str]) -> None:
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.file.write_text(json.dumps(synced, sort_keys=True))


@dataclass(kw_only=True)
class NotebookSync:
    """Writes personal note changes to Logfire and restores them from it on another machine."""

    directory: Path
    """The personal notebooks' root, as `personal_memory` keeps them."""
    tracer_provider: TracerProvider
    state: SyncState
    owner: Callable[[], str | None]
    query: LogfireQuery | None = None
    """Absent when the key can't query Logfire: notes are still sent, but not restored here."""
    warn: Callable[[str], None] = lambda message: None
    _warned: bool = field(default=False, init=False)

    def local(self) -> dict[str, Path]:
        """Every personal note on this machine, by store path (`<namespace>/personal/<file>.md`)."""
        if not self.directory.is_dir():
            return {}
        return {
            path.relative_to(self.directory).as_posix(): path
            for path in self.directory.rglob('*.md')
            if path.is_file() and path.parent.name == 'personal'
        }

    def emit(self, path: str, content: str | None, *, at: float | None = None) -> bool:
        """Write one note's change (`None` content: deleted) as a span; whether it was sent."""
        owner = self.owner()
        if not owner or (content is not None and len(content.encode()) > MAX_BYTES):
            return False
        tracer = self.tracer_provider.get_tracer(SCOPE)
        with tracer.start_as_current_span(
            SPAN_NAME,
            attributes={
                'clai2.memory.owner': owner,
                'clai2.memory.path': path,
                'clai2.memory.notebook': path.rsplit('/personal/', 1)[0],
                'clai2.memory.content': content or '',
                'clai2.memory.deleted': content is None,
                'clai2.memory.sha256': _sha(content) if content is not None else '',
                'clai2.memory.at': at if at is not None else time.time(),
                'logfire.msg': f'clai2 memory note {path}' + (' deleted' if content is None else ''),
            },
        ):
            pass
        return True

    def push_changed(self) -> int:
        """Send every note that changed or disappeared since it last matched Logfire; how many were sent."""
        synced = self.state.load()
        local = self.local()
        sent = 0
        for path, file in local.items():
            content = file.read_text(encoding='utf-8', errors='replace')
            if synced.get(path) != (sha := _sha(content)) and self.emit(path, content, at=file.stat().st_mtime):
                synced[path] = sha
                sent += 1
        for path in [path for path in synced if path not in local]:
            if self.emit(path, None):
                del synced[path]
                sent += 1
        self.state.save(synced)
        return sent

    async def remote(self) -> dict[str, RemoteNote]:
        """The latest change to each of this user's notes in Logfire."""
        owner = self.owner()
        if self.query is None or not owner:
            return {}
        columns = ', '.join(
            f'{sql_attribute("clai2.memory." + name)} AS {name}'
            for name in ('path', 'content', 'deleted', 'sha256', 'at')
        )
        rows = await self.query.rows(
            f'SELECT {columns} FROM records '
            f'WHERE span_name = {sql_quote(SPAN_NAME)} AND {sql_attribute("clai2.memory.owner")} = {sql_quote(owner)}',
            since=WINDOW,
        )
        latest: dict[str, RemoteNote] = {}
        for row in rows:
            try:
                note = RemoteNote(
                    path=str(row['path']),
                    content=str(row.get('content') or ''),
                    deleted=str(row.get('deleted')).lower() in ('true', '1'),
                    sha256=str(row.get('sha256') or ''),
                    at=float(row['at']),
                )
            except (KeyError, TypeError, ValueError, ValidationError):
                continue
            if note.path not in latest or note.at > latest[note.path].at:
                latest[note.path] = note
        return latest

    async def pull(self) -> None:
        """Reconcile this machine's notebooks with Logfire, last write wins; never fails the session."""
        try:
            remote = await self.remote()
        except Exception as error:  # noqa: BLE001 -- notes stay local when Logfire can't be read
            if not self._warned:
                self._warned = True
                self.warn(
                    f'Personal notes not synced from Logfire ({type(error).__name__}); keeping the notes on this machine.'
                )
            remote = {}
        synced = self.state.load()
        local = self.local()
        for path, note in remote.items():
            file = local.get(path)
            target = self._target(path)
            if target is None:
                continue
            if file is not None:
                content = file.read_text(encoding='utf-8', errors='replace')
                if not note.deleted and _sha(content) == note.sha256:
                    synced[path] = note.sha256
                elif note.at > file.stat().st_mtime:
                    if note.deleted:
                        file.unlink()
                        synced.pop(path, None)
                    else:
                        self._restore(target, note)
                        synced[path] = note.sha256
                else:
                    # This machine's copy is newer: `push_changed` below sends it again.
                    synced.pop(path, None)
            elif not note.deleted and not (path in synced and synced[path] == note.sha256):
                self._restore(target, note)
                synced[path] = note.sha256
            # A note deleted here after it last matched Logfire is sent as a tombstone below.
        self.state.save(synced)
        self.push_changed()

    def _target(self, path: str) -> Path | None:
        """Where a store path lives in the folder, or `None` when it would leave it."""
        target = (self.directory / path).resolve()
        root = self.directory.resolve()
        if root not in target.parents or not path.endswith('.md') or target.parent.name != 'personal':
            return None
        return target

    @staticmethod
    def _restore(target: Path, note: RemoteNote) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(note.content, encoding='utf-8')
        # Stamped with when it changed, so it doesn't look newer than Logfire's copy next time.
        os.utime(target, (note.at, note.at))
