"""Revision-checked conversation heads, separate from per-run step snapshots."""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import subprocess
import sys
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Literal
from uuid import uuid4

from anyio.to_thread import run_sync
from pydantic import TypeAdapter
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelResponse,
    TextContent,
    TextPart,
    UserPromptPart,
)

from pydantic_ai_harness.media import SqliteMediaStore, externalize_media, restore_media


class ConversationConflict(ValueError):
    """Another writer changed or deleted this conversation. Reload before writing."""


@dataclass(frozen=True, kw_only=True)
class ConversationSummary:
    """Browser metadata. Naming edits do not change activity or content revision."""

    schema_version: Literal[1] = 1
    id: str = field(default_factory=lambda: str(uuid4()))
    workspace: str
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    revision: int = 0
    title: str = 'New session'
    subtitle: str = ''
    tags: tuple[str, ...] = ()
    title_source: Literal['fallback', 'generated', 'user'] = 'fallback'
    naming_version: int = 0
    named_revision: int = 0
    model: str | None = None
    outcome: Literal['ready', 'running', 'completed', 'failed', 'cancelled'] = 'ready'
    run_id: str | None = None
    owner_pid: int | None = None
    message_count: int = 0
    total_tokens: int = 0
    naming_tokens: int = 0


def ensure_inactive(summary: ConversationSummary) -> None:
    """Local SQLite sessions cannot be resumed while their recorded process is running.

    PID reuse is conservatively treated as busy. This is not a distributed lease;
    do not share this database between hosts.
    """
    if summary.outcome != 'running' or summary.owner_pid is None:
        return
    if sys.platform == 'win32':
        system_root = PureWindowsPath(os.environ['SystemRoot'])
        if not system_root.is_absolute():
            raise ValueError('SystemRoot must be an absolute Windows path')
        result = subprocess.run(
            [str(system_root / 'System32' / 'tasklist.exe'), '/FI', f'PID eq {summary.owner_pid}', '/FO', 'CSV', '/NH'],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        if not any(len(row) > 1 and row[1] == str(summary.owner_pid) for row in csv.reader(result.stdout.splitlines())):
            return
    else:
        ensure_posix_inactive(pid=summary.owner_pid)
        return
    raise ConversationConflict(f'Session is busy in process {summary.owner_pid}.')


def ensure_posix_inactive(*, pid: int) -> None:
    """Signal zero probes existence without delivering a signal on POSIX."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    except PermissionError:
        pass
    raise ConversationConflict(f'Session is busy in process {pid}.')


@dataclass(kw_only=True)
class SavedConversation:
    """An authoritative history plus its separately editable metadata."""

    summary: ConversationSummary
    messages: list[ModelMessage]


def conversation_text(messages: Sequence[ModelMessage]) -> str:
    """Searchable user/assistant text, excluding tool output and reasoning."""
    lines: list[str] = []
    for message in messages:
        for part in message.parts:
            if isinstance(part, TextPart):
                lines.append(f'assistant: {part.content}')
            elif isinstance(part, UserPromptPart):
                content = part.content
                if isinstance(content, str):
                    lines.append(f'user: {content}')
                else:
                    for item in content:
                        if isinstance(item, str):
                            lines.append(f'user: {item}')
                        elif isinstance(item, TextContent):
                            lines.append(f'user: {item.content}')
    return '\n'.join(lines)


class SqliteConversationStore:
    """Local conversation heads with atomic compare-and-swap writes.

    Messages share the step store's media format. Deletion removes associated top-level run records and checkpoints but leaves shared blobs; only explicitly catalogued conversations are browsable.
    Connections are short-lived and all SQL runs off the caller's event loop.
    """

    def __init__(self, *, database: Path) -> None:
        """Defer database creation until the first operation."""
        self.database = database
        self.media = SqliteMediaStore(database=database)
        self._summary = TypeAdapter(ConversationSummary)

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        self.database.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.database, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
        conn = sqlite3.connect(self.database, timeout=10)
        try:
            conn.create_function('casefold', 1, str.casefold, deterministic=True)
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute(
                'CREATE TABLE IF NOT EXISTS conversations ('
                'id TEXT PRIMARY KEY, revision INTEGER NOT NULL, metadata TEXT NOT NULL, '
                'messages TEXT NOT NULL, search_text TEXT NOT NULL, updated_at TEXT NOT NULL)'
            )
            with conn:
                yield conn
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self._connection():
            pass

    async def save(self, *, summary: ConversationSummary, messages: Sequence[ModelMessage]) -> ConversationSummary:
        """Commit content only if its expected revision is still current."""
        updated = replace(
            summary,
            revision=summary.revision + 1,
            updated_at=datetime.now(timezone.utc),
            message_count=len(messages),
            total_tokens=sum(m.usage.total_tokens for m in messages if isinstance(m, ModelResponse)),
        )
        await run_sync(self._initialize)
        payload = await externalize_media(
            ModelMessagesTypeAdapter.dump_python(list(messages), mode='json'),
            media_store=self.media,
            threshold_bytes=64 * 1024,
        )
        return await run_sync(self._save, summary.revision, updated, json.dumps(payload), conversation_text(messages))

    def _save(self, expected: int, summary: ConversationSummary, messages: str, text: str) -> ConversationSummary:
        with self._connection() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute('SELECT metadata FROM conversations WHERE id = ?', (summary.id,)).fetchone()
            if row is None:
                if expected != 0:
                    raise ConversationConflict('Session was deleted. Start a new session before saving.')
                conn.execute(
                    'INSERT INTO conversations VALUES (?, ?, ?, ?, ?, ?)',
                    (
                        summary.id,
                        summary.revision,
                        self._summary.dump_json(summary).decode(),
                        messages,
                        text,
                        summary.updated_at.isoformat(),
                    ),
                )
                return summary
            prior = self._summary.validate_json(row[0])
            if prior.revision != expected:
                raise ConversationConflict('Session changed in another process. Resume it again before continuing.')
            # Preserve a background naming commit which raced with this content save.
            summary = replace(
                summary,
                title=prior.title,
                subtitle=prior.subtitle,
                tags=prior.tags,
                title_source=prior.title_source,
                naming_version=prior.naming_version,
                named_revision=prior.named_revision,
                naming_tokens=prior.naming_tokens,
            )
            conn.execute(
                'UPDATE conversations SET revision=?, metadata=?, messages=?, search_text=?, updated_at=? WHERE id=?',
                (
                    summary.revision,
                    self._summary.dump_json(summary).decode(),
                    messages,
                    text,
                    summary.updated_at.isoformat(),
                    summary.id,
                ),
            )
            return summary

    async def get(self, *, conversation_id: str) -> SavedConversation:
        """Load exactly one ID, rejecting missing or malformed data."""
        summary, payload = await run_sync(self._get, conversation_id)
        restored = await restore_media(json.loads(payload), media_store=self.media)
        return SavedConversation(summary=summary, messages=ModelMessagesTypeAdapter.validate_python(restored))

    def _get(self, conversation_id: str) -> tuple[ConversationSummary, str]:
        with self._connection() as conn:
            row = conn.execute('SELECT metadata, messages FROM conversations WHERE id=?', (conversation_id,)).fetchone()
            if row is None:
                raise LookupError(f'No saved session: {conversation_id}')
            return self._summary.validate_json(row[0]), str(row[1])

    async def listing(self, *, query: str = '', limit: int = 200, offset: int = 0) -> list[ConversationSummary]:
        """Page summaries without loading transcripts. Search includes title, tags and conversation text."""
        if limit < 1 or offset < 0:
            raise ValueError('limit must be positive and offset nonnegative')
        return await run_sync(self._listing, query, limit, offset)

    def _listing(self, query: str, limit: int, offset: int) -> list[ConversationSummary]:
        with self._connection() as conn:
            rows = conn.execute(
                'SELECT metadata FROM conversations WHERE instr(casefold(search_text || metadata), casefold(?)) > 0 '
                'ORDER BY updated_at DESC, id LIMIT ? OFFSET ?',
                (query, limit, offset),
            ).fetchall()
        return [self._summary.validate_json(row[0]) for row in rows]

    async def name(
        self,
        *,
        source: ConversationSummary,
        title: str,
        subtitle: str = '',
        tags: tuple[str, ...] = (),
        manual: bool = False,
        tokens: int = 0,
    ) -> bool:
        """Apply naming only to the input revision and naming version; never resurrect deleted rows."""
        return await run_sync(self._name, source, title, subtitle, tags, manual, tokens)

    def _name(
        self, source: ConversationSummary, title: str, subtitle: str, tags: tuple[str, ...], manual: bool, tokens: int
    ) -> bool:
        with self._connection() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute('SELECT metadata FROM conversations WHERE id=?', (source.id,)).fetchone()
            if row is None:
                return False
            current = self._summary.validate_json(row[0])
            if tokens:
                current = replace(current, naming_tokens=current.naming_tokens + tokens)
                conn.execute(
                    'UPDATE conversations SET metadata=? WHERE id=?',
                    (self._summary.dump_json(current).decode(), source.id),
                )
            if (current.revision, current.naming_version) != (source.revision, source.naming_version):
                return False
            if current.title_source == 'user' and not manual:
                return False
            updated = replace(
                current,
                title=title,
                subtitle=subtitle,
                tags=tags,
                title_source='user' if manual else 'generated',
                naming_version=current.naming_version + 1,
                named_revision=current.revision,
                naming_tokens=current.naming_tokens,
            )
            conn.execute(
                'UPDATE conversations SET metadata=? WHERE id=?', (self._summary.dump_json(updated).decode(), source.id)
            )
            return True

    async def delete(self, *, source: ConversationSummary) -> None:
        """Delete the expected content revision; concurrent writers cannot be silently erased."""
        await run_sync(self._delete, source)

    def _delete(self, source: ConversationSummary) -> None:
        ensure_inactive(source)
        with self._connection() as conn:
            deleted = conn.execute('DELETE FROM conversations WHERE id=? AND revision=?', (source.id, source.revision))
            if deleted.rowcount != 1:
                raise ConversationConflict('Session changed or was deleted. Refresh the browser.')
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if 'runs' in tables:
                for table in ('events', 'snapshots', 'snapshot_idempotency_keys', 'tool_effects'):
                    if table in tables:
                        # Fixed internal table names, never user input. Keep the catalog and
                        # its run data deletion in the same SQLite transaction.
                        conn.execute(
                            f'DELETE FROM {table} WHERE run_id IN (SELECT run_id FROM runs WHERE conversation_id=?)',
                            (source.id,),
                        )
                conn.execute('DELETE FROM runs WHERE conversation_id=?', (source.id,))
