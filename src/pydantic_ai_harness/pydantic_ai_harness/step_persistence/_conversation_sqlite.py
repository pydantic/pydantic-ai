"""SQLite-backed `ConversationStore`."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path

import anyio

from pydantic_ai.conversation_store import (
    ConversationConflict,
    ConversationEntry,
    ConversationRecord,
    ConversationStore,
)
from pydantic_ai_harness.media import MediaStore, SqliteMediaStore
from pydantic_ai_harness.step_persistence._conversation_codec import (
    RECORD_ADAPTER,
    keyed_entry_messages,
    load_message,
    missing_entries_error,
)
from pydantic_ai_harness.step_persistence._store import (
    _DEFAULT_MEDIA_THRESHOLD_BYTES,  # pyright: ignore[reportPrivateUsage]
    _atomically,  # pyright: ignore[reportPrivateUsage]
    _AutoMedia,  # pyright: ignore[reportPrivateUsage]
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversation_messages (
    sha256 TEXT PRIMARY KEY,
    message TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conversation_entries (
    id TEXT PRIMARY KEY,
    parent_id TEXT REFERENCES conversation_entries (id),
    kind TEXT NOT NULL,
    sha256 TEXT NOT NULL REFERENCES conversation_messages (sha256)
);
CREATE INDEX IF NOT EXISTS conversation_entries_parent ON conversation_entries (parent_id);
CREATE INDEX IF NOT EXISTS conversation_entries_sha256 ON conversation_entries (sha256);
CREATE TABLE IF NOT EXISTS conversation_records (
    conversation_id TEXT PRIMARY KEY,
    revision INTEGER NOT NULL,
    head_id TEXT REFERENCES conversation_entries (id),
    scope TEXT,
    record TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS conversation_records_head ON conversation_records (head_id);
CREATE INDEX IF NOT EXISTS conversation_records_scope ON conversation_records (scope);
"""

# The entries from `head_id` back to the first, deepest first, with each one's message.
_HISTORY_SQL = """
WITH RECURSIVE chain (id, parent_id, kind, sha256, depth) AS (
    SELECT id, parent_id, kind, sha256, 0 FROM conversation_entries WHERE id = ?
    UNION ALL
    SELECT e.id, e.parent_id, e.kind, e.sha256, chain.depth + 1
    FROM conversation_entries e JOIN chain ON e.id = chain.parent_id
)
SELECT chain.id, chain.parent_id, m.message
FROM chain LEFT JOIN conversation_messages m ON m.sha256 = chain.sha256
ORDER BY chain.depth DESC
"""


class SqliteConversationStore(ConversationStore):
    """A [`ConversationStore`][pydantic_ai.conversation_store.ConversationStore] in a SQLite database.

    Pass either `database=` (a path; connections are opened per call) or `connection=` (a
    caller-owned `sqlite3.Connection`, which must be created with `check_same_thread=False`
    because SQL runs on worker threads).

    Each message's JSON is stored once in `conversation_messages`, keyed by its SHA-256, however
    many entries hold it. Binary and text parts at or above `media_threshold_bytes` (default
    64 KiB) are externalized to `media_store`, by default a `SqliteMediaStore` in the same
    database, as the step stores do.

    A save writes its new entries and the record together, in a transaction on a connection the
    store opened and in a savepoint on a caller-owned one. Deleting a conversation also deletes the
    entries and messages no other conversation's history uses; externalized media stays.

    Tables:

    - `conversation_records(conversation_id PK, revision, head_id, scope, record)`
    - `conversation_entries(id PK, parent_id, kind, sha256)`
    - `conversation_messages(sha256 PK, message)`
    """

    def __init__(
        self,
        *,
        database: str | Path | None = None,
        connection: sqlite3.Connection | None = None,
        media_store: MediaStore | None | _AutoMedia = 'auto',
        media_threshold_bytes: int = _DEFAULT_MEDIA_THRESHOLD_BYTES,
    ) -> None:
        if (database is None) == (connection is None):
            raise ValueError('provide exactly one of `database=` or `connection=`')
        self._database = Path(database) if database is not None else None
        self._connection = connection
        self._schema_ready = False
        if media_store == 'auto':
            media_store = (
                SqliteMediaStore(database=self._database)
                if self._database is not None
                else SqliteMediaStore(connection=connection)
            )
        self._media_store: MediaStore | None = media_store
        self._media_threshold_bytes = media_threshold_bytes

    def _open(self) -> sqlite3.Connection:
        if self._connection is not None:
            conn = self._connection
        else:
            assert self._database is not None
            self._database.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self._database, isolation_level=None)
            conn.execute('PRAGMA journal_mode=WAL')
        if not self._schema_ready:
            conn.executescript(_SCHEMA)
            self._schema_ready = True
        return conn

    def _close(self, conn: sqlite3.Connection) -> None:
        if self._connection is None:
            conn.close()

    async def read_record(self, conversation_id: str) -> ConversationRecord | None:
        return await anyio.to_thread.run_sync(self._sync_read_record, conversation_id)

    def _sync_read_record(self, conversation_id: str) -> ConversationRecord | None:
        conn = self._open()
        try:
            row = conn.execute(
                'SELECT record FROM conversation_records WHERE conversation_id = ?', (conversation_id,)
            ).fetchone()
        finally:
            self._close(conn)
        return RECORD_ADAPTER.validate_json(row[0]) if row is not None else None

    async def read_records(self, *, scope: str | None = None) -> Sequence[ConversationRecord]:
        return await anyio.to_thread.run_sync(self._sync_read_records, scope)

    def _sync_read_records(self, scope: str | None) -> list[ConversationRecord]:
        conn = self._open()
        try:
            rows = conn.execute(
                'SELECT record FROM conversation_records WHERE ? IS NULL OR scope = ?', (scope, scope)
            ).fetchall()
        finally:
            self._close(conn)
        return [RECORD_ADAPTER.validate_json(row[0]) for row in rows]

    async def write_record(
        self,
        record: ConversationRecord,
        *,
        new_entries: Sequence[ConversationEntry] = (),
        expected_revision: int | None,
    ) -> None:
        messages = await keyed_entry_messages(
            new_entries, media_store=self._media_store, threshold_bytes=self._media_threshold_bytes
        )
        await anyio.to_thread.run_sync(self._sync_write_record, record, new_entries, messages, expected_revision)

    def _sync_write_record(
        self,
        record: ConversationRecord,
        entries: Sequence[ConversationEntry],
        messages: list[tuple[str, str]],
        expected_revision: int | None,
    ) -> None:
        conn = self._open()
        try:
            with _atomically(conn, store_owned=self._connection is None, begin='BEGIN IMMEDIATE'):
                row = conn.execute(
                    'SELECT revision FROM conversation_records WHERE conversation_id = ?', (record.conversation_id,)
                ).fetchone()
                if (row[0] if row is not None else None) != expected_revision:
                    raise ConversationConflict(record.conversation_id)
                conn.executemany(
                    'INSERT OR IGNORE INTO conversation_messages (sha256, message) VALUES (?, ?)', messages
                )
                conn.executemany(
                    'INSERT INTO conversation_entries (id, parent_id, kind, sha256) VALUES (?, ?, ?, ?)',
                    [
                        (entry.id, entry.parent_id, entry.kind, sha256)
                        for entry, (sha256, _) in zip(entries, messages, strict=True)
                    ],
                )
                conn.execute(
                    'INSERT INTO conversation_records (conversation_id, revision, head_id, scope, record) '
                    'VALUES (?, ?, ?, ?, ?) ON CONFLICT (conversation_id) DO UPDATE SET '
                    'revision = excluded.revision, head_id = excluded.head_id, scope = excluded.scope, '
                    'record = excluded.record',
                    (
                        record.conversation_id,
                        record.revision,
                        record.head_id,
                        record.scope,
                        RECORD_ADAPTER.dump_json(record).decode(),
                    ),
                )
        finally:
            self._close(conn)

    async def delete_record(self, conversation_id: str) -> None:
        await anyio.to_thread.run_sync(self._sync_delete_record, conversation_id)

    def _sync_delete_record(self, conversation_id: str) -> None:
        conn = self._open()
        try:
            with _atomically(conn, store_owned=self._connection is None, begin='BEGIN IMMEDIATE'):
                row = conn.execute(
                    'DELETE FROM conversation_records WHERE conversation_id = ? RETURNING head_id', (conversation_id,)
                ).fetchone()
                entry_id: str | None = row[0] if row is not None else None
                # Walk back from the head, deleting each entry until one that another record's
                # history still reaches: a record's head, or an entry with another child.
                while entry_id is not None:
                    deleted = conn.execute(
                        'DELETE FROM conversation_entries WHERE id = ? '
                        'AND NOT EXISTS (SELECT 1 FROM conversation_records WHERE head_id = ?) '
                        'AND NOT EXISTS (SELECT 1 FROM conversation_entries WHERE parent_id = ?) '
                        'RETURNING parent_id, sha256',
                        (entry_id, entry_id, entry_id),
                    ).fetchone()
                    if deleted is None:
                        break
                    entry_id, sha256 = deleted
                    conn.execute(
                        'DELETE FROM conversation_messages WHERE sha256 = ? '
                        'AND NOT EXISTS (SELECT 1 FROM conversation_entries WHERE sha256 = ?)',
                        (sha256, sha256),
                    )
        finally:
            self._close(conn)

    async def read_entry(self, entry_id: str) -> ConversationEntry:
        row = await anyio.to_thread.run_sync(self._sync_read_entry, entry_id)
        if row is None:
            raise KeyError(entry_id)
        parent_id, message = row
        return ConversationEntry(
            id=entry_id, parent_id=parent_id, message=await load_message(message, media_store=self._media_store)
        )

    def _sync_read_entry(self, entry_id: str) -> tuple[str | None, str] | None:
        conn = self._open()
        try:
            return conn.execute(
                'SELECT e.parent_id, m.message FROM conversation_entries e '
                'JOIN conversation_messages m ON m.sha256 = e.sha256 WHERE e.id = ?',
                (entry_id,),
            ).fetchone()
        finally:
            self._close(conn)

    async def read_history(self, head_id: str | None) -> list[ConversationEntry]:
        """The entries from the first one up to `head_id`, in order, read in one query."""
        if head_id is None:
            return []
        rows = await anyio.to_thread.run_sync(self._sync_read_history, head_id)
        if not rows or rows[0][1] is not None:
            raise missing_entries_error(head_id)
        entries: list[ConversationEntry] = []
        for entry_id, parent_id, message in rows:
            if message is None:
                raise missing_entries_error(head_id)
            entries.append(
                ConversationEntry(
                    id=entry_id, parent_id=parent_id, message=await load_message(message, media_store=self._media_store)
                )
            )
        return entries

    def _sync_read_history(self, head_id: str) -> list[tuple[str, str | None, str | None]]:
        conn = self._open()
        try:
            return conn.execute(_HISTORY_SQL, (head_id,)).fetchall()
        finally:
            self._close(conn)
