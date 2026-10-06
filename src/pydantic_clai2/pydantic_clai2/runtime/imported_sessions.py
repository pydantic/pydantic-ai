"""Claude Code and Codex sessions for `/resume`, copied into CLAI's store when one is resumed.

A session keeps one CLAI ID derived from its own, so resuming it again finds the same copy. A copy
CLAI has not continued is read again from the original each time it is resumed; once CLAI adds a
turn, the copy is CLAI's own and the original is no longer read.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid5

from anyio.to_thread import run_sync

from pydantic_ai.messages import ModelMessage
from pydantic_ai_harness.step_persistence.conversations import ConversationSummary, SqliteConversationStore
from pydantic_clai2.runtime import claude_code_sessions, codex_sessions
from pydantic_clai2.runtime.imported_history import Header
from pydantic_clai2.ui import telemetry

ImportSource = Literal['claude', 'codex']
IMPORT_SOURCES: tuple[ImportSource, ...] = ('claude', 'codex')

_NAMESPACE = UUID('5d0f0a0e-4f7c-4f43-9a52-6c1f3b1e2a70')
"""Derives a CLAI conversation ID from another agent's session ID."""


@dataclass(frozen=True, kw_only=True)
class _Source:
    label: str
    home: Callable[[], Path]
    files: Callable[[Path], list[Path]]
    find: Callable[[Path, str], Path | None]
    titles: Callable[[Path], dict[str, str]]
    header: Callable[[Path], Header | None]
    messages: Callable[[Path], list[ModelMessage]]


_SOURCES: dict[ImportSource, _Source] = {
    'claude': _Source(
        label='Claude Code',
        home=claude_code_sessions.home,
        files=claude_code_sessions.files,
        find=claude_code_sessions.find,
        titles=claude_code_sessions.titles,
        header=claude_code_sessions.header,
        messages=claude_code_sessions.messages,
    ),
    'codex': _Source(
        label='Codex',
        home=codex_sessions.home,
        files=codex_sessions.files,
        find=codex_sessions.find,
        titles=codex_sessions.titles,
        header=codex_sessions.header,
        messages=codex_sessions.messages,
    ),
}


def import_source(name: str) -> ImportSource | None:
    """The source a `/resume` argument names, if it names one."""
    return next((source for source in IMPORT_SOURCES if source == name), None)


@dataclass(frozen=True, kw_only=True)
class ImportedSession:
    """A session on disk and the summary its CLAI copy starts with."""

    source: ImportSource
    path: Path
    summary: ConversationSummary
    """Revision 0, as it is not yet in CLAI's store."""

    def messages(self) -> list[ModelMessage]:
        """Read the whole transcript. Blocking file IO, so call it off the event loop."""
        return _SOURCES[self.source].messages(self.path)


def _modified(path: Path) -> float | None:
    """A transcript's modification time, or `None` once it is gone, as one deleted since it was listed is."""
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _imported(source: ImportSource, path: Path, *, modified: float, titles: dict[str, str]) -> ImportedSession | None:
    spec = _SOURCES[source]
    try:
        header = spec.header(path)
    except OSError:
        # An unreadable file must not hide every other session from the browser.
        return None
    if header is None:
        return None
    title = titles.get(header.native_id)
    return ImportedSession(
        source=source,
        path=path,
        summary=ConversationSummary(
            id=str(uuid5(_NAMESPACE, f'{source}:{header.native_id}')),
            workspace=str(Path(header.cwd).resolve()),
            updated_at=datetime.fromtimestamp(modified, UTC),
            title=title or header.title,
            subtitle=f'{spec.label} session {header.native_id}',
            tags=(source,),
            # Names the other agent generated are kept until the conversation moves on.
            title_source='generated' if title or header.named else 'fallback',
        ),
    )


class ImportCatalog:
    """One scan of the sessions on disk, read newest first as far as a listing needs.

    Blocking file IO: build and list it off the event loop, as the browser's menu thread does.
    """

    def __init__(self, sources: Sequence[ImportSource]) -> None:
        """List the transcripts and their modification times, reading none of them yet."""
        self._files: list[tuple[float, ImportSource, Path]] = []
        self._titles: dict[ImportSource, dict[str, str]] = {}
        for source in sources:
            spec = _SOURCES[source]
            root = spec.home()
            self._titles[source] = spec.titles(root)
            for path in spec.files(root):
                if (modified := _modified(path)) is not None:
                    self._files.append((modified, source, path))
        self._files.sort(reverse=True)
        self._read: dict[Path, ImportedSession | None] = {}
        self._by_id: dict[str, ImportedSession] = {}

    def listing(self, query: str = '', limit: int = 200) -> list[ConversationSummary]:
        """Up to `limit` sessions whose title or directory contains `query`, newest first."""
        needle = query.casefold()
        found: list[ConversationSummary] = []
        for modified, source, path in self._files:
            if len(found) >= limit:
                break
            if path not in self._read:
                imported = _imported(source, path, modified=modified, titles=self._titles[source])
                self._read[path] = imported
                if imported is not None:
                    self._by_id[imported.summary.id] = imported
            imported = self._read[path]
            if imported is not None and needle in f'{imported.summary.title}\n{imported.summary.workspace}'.casefold():
                found.append(imported.summary)
        return found

    def get(self, conversation_id: str) -> ImportedSession | None:
        """A session a listing has returned, by its CLAI ID."""
        return self._by_id.get(conversation_id)


def find_import(source: ImportSource, native_id: str) -> ImportedSession:
    """A session by the ID its own agent shows. Blocking file IO."""
    spec = _SOURCES[source]
    root = spec.home()
    # The ID becomes part of a glob pattern, so it may hold only the characters session IDs use.
    path = spec.find(root, native_id) if re.fullmatch(r'[\w-]+', native_id) else None
    modified = None if path is None else _modified(path)
    imported = None
    if path is not None and modified is not None:
        imported = _imported(source, path, modified=modified, titles=spec.titles(root))
    if imported is None:
        raise LookupError(f'No {spec.label} session: {native_id}')
    return imported


def merge(
    saved: Sequence[ConversationSummary], imported: Sequence[ConversationSummary], *, limit: int
) -> list[ConversationSummary]:
    """CLAI's sessions and not-yet-imported ones, newest first, a CLAI copy replacing its original."""
    ids = {summary.id for summary in saved}
    entries = [*saved, *(summary for summary in imported if summary.id not in ids)]
    entries.sort(key=lambda summary: summary.updated_at, reverse=True)
    return entries[:limit]


async def save_import(store: SqliteConversationStore, imported: ImportedSession) -> str:
    """Copy a session into the store, or refresh a copy CLAI has not continued, and return its CLAI ID.

    The refresh does not compare times: the original may change while it is being read, after its
    time was taken, and that turn would never be imported.
    """
    try:
        saved = (await store.get(conversation_id=imported.summary.id)).summary
    except LookupError:
        saved = None
    # Importing saves revision 1, and each CLAI turn adds one.
    if saved is not None and saved.revision > 1:
        return saved.id
    messages = await run_sync(imported.messages)
    if not messages:
        raise ValueError(f'{imported.summary.subtitle} has no conversation to import')
    summary = imported.summary
    if saved is not None:
        # Keep the names the user or the namer gave the copy.
        summary = replace(
            summary,
            title=saved.title,
            subtitle=saved.subtitle,
            tags=saved.tags,
            title_source=saved.title_source,
            naming_version=saved.naming_version,
            named_revision=saved.named_revision,
            naming_tokens=saved.naming_tokens,
        )
        # Only once the new transcript has been read, so a failed refresh keeps the previous copy.
        await store.delete(source=saved)
    await store.save(summary=summary, messages=messages)
    telemetry.record('conversation imported', source=imported.source, messages=len(messages))
    return imported.summary.id
