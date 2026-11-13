"""SQLite-backed `ConversationStore`."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Generator, Sequence
from contextlib import contextmanager
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
    entry_positions,
    keyed_entry_messages,
    load_message,
    missing_entries_error,
)
from pydantic_ai_harness.step_persistence._store import (
    _BUSY_TIMEOUT_SECONDS,  # pyright: ignore[reportPrivateUsage]
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
    sha256 TEXT NOT NULL REFERENCES conversation_messages (sha256),
    root_id TEXT NOT NULL,
    depth INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS conversation_entries_parent ON conversation_entries (parent_id);
CREATE INDEX IF NOT EXISTS conversation_entries_sha256 ON conversation_entries (sha256);
CREATE TABLE IF NOT EXISTS conversation_records (
    conversation_id TEXT PRIMARY KEY,
    revision TEXT NOT NULL,
    head_id TEXT REFERENCES conversation_entries (id),
    scope TEXT,
    updated_at REAL NOT NULL,
    record TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS conversation_records_head ON conversation_records (head_id);
CREATE INDEX IF NOT EXISTS conversation_records_scope ON conversation_records (scope, updated_at);
CREATE INDEX IF NOT EXISTS conversation_records_updated_at ON conversation_records (updated_at);
CREATE TABLE IF NOT EXISTS conversation_branches (
    conversation_id TEXT NOT NULL,
    entry_id TEXT NOT NULL REFERENCES conversation_entries (id),
    PRIMARY KEY (conversation_id, entry_id)
);
CREATE INDEX IF NOT EXISTS conversation_branches_entry ON conversation_branches (entry_id);
"""

# The entries from `head_id` back to the first, or at most `limit` of them (a negative one is no limit), deepest
# first, with each one's message.
_HISTORY_SQL = """
WITH RECURSIVE chain (id, parent_id, kind, sha256, depth) AS (
    SELECT id, parent_id, kind, sha256, 0 FROM conversation_entries WHERE id = ?
    UNION ALL
    SELECT e.id, e.parent_id, e.kind, e.sha256, chain.depth + 1
    FROM conversation_entries e JOIN chain ON e.id = chain.parent_id
    WHERE ? < 0 OR chain.depth + 1 < ?
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
    store opened and in a savepoint on a caller-owned one. Deleting a conversation also deletes
    the entries and messages no other conversation's history or branch uses; externalized media stays.

    Tables:

    - `conversation_records(conversation_id PK, revision, head_id, scope, updated_at, record)`
    - `conversation_entries(id PK, parent_id, kind, sha256, root_id, depth)`: `root_id` is the first entry of
      the entry's history and `depth` its distance from it.
    - `conversation_messages(sha256 PK, message)`
    - `conversation_branches(conversation_id, entry_id)`: the last entry of each branch a record lists.
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
        if database is not None and str(database) == ':memory:':
            # Each operation opens its own connection, and each would get a new, empty database.
            raise ValueError(
                "`database=':memory:'` is not supported; pass `connection=sqlite3.connect(':memory:', ...)`"
            )
        self._database = Path(database) if database is not None else None
        self._connection = connection
        self._schema_ready = False
        self._lock = threading.Lock()
        if media_store == 'auto':
            media_store = (
                SqliteMediaStore(database=self._database)
                if self._database is not None
                else _LockedSqliteMediaStore(connection=connection, lock=self._lock)
            )
        self._media_store: MediaStore | None = media_store
        self._media_threshold_bytes = media_threshold_bytes

    @contextmanager
    def _connected(self) -> Generator[sqlite3.Connection]:
        """A connection to use for one operation.

        A connection the store opens is its own and closed afterwards. A caller-owned one is shared,
        so operations on it take turns: two at once would interleave their savepoints.
        """
        if self._connection is not None:
            with self._lock:
                self._ensure_schema(self._connection)
                yield self._connection
            return
        assert self._database is not None
        self._database.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._database, isolation_level=None, timeout=_BUSY_TIMEOUT_SECONDS)
        try:
            if not self._schema_ready:
                # SQLite reports two connections switching a new database to WAL at once as locked
                # rather than waiting, so this store's connections set it up one at a time. The
                # journal mode is kept in the file, so later connections needn't set it.
                with self._lock:
                    conn.execute('PRAGMA journal_mode=WAL')
                    self._ensure_schema(conn)
            yield conn
        finally:
            conn.close()

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        if not self._schema_ready:
            # One statement at a time: `executescript` would first commit a transaction the caller has open.
            for statement in _SCHEMA.split(';'):
                if statement.strip():
                    conn.execute(statement)
            # Tables created in a caller's open transaction are gone if it rolls back.
            self._schema_ready = not conn.in_transaction

    async def read_record(self, conversation_id: str) -> ConversationRecord | None:
        return await anyio.to_thread.run_sync(self._sync_read_record, conversation_id)

    def _sync_read_record(self, conversation_id: str) -> ConversationRecord | None:
        with self._connected() as conn:
            row = conn.execute(
                'SELECT record FROM conversation_records WHERE conversation_id = ?', (conversation_id,)
            ).fetchone()
        return RECORD_ADAPTER.validate_json(row[0]) if row is not None else None

    async def read_records(self, *, scope: str | None = None, limit: int | None = None) -> Sequence[ConversationRecord]:
        return await anyio.to_thread.run_sync(self._sync_read_records, scope, limit)

    def _sync_read_records(self, scope: str | None, limit: int | None) -> list[ConversationRecord]:
        # A negative `LIMIT` is no limit in SQLite.
        limit = -1 if limit is None else limit
        with self._connected() as conn:
            if scope is None:
                rows = conn.execute(
                    'SELECT record FROM conversation_records ORDER BY updated_at DESC LIMIT ?', (limit,)
                ).fetchall()
            else:
                rows = conn.execute(
                    'SELECT record FROM conversation_records WHERE scope = ? ORDER BY updated_at DESC LIMIT ?',
                    (scope, limit),
                ).fetchall()
        return [RECORD_ADAPTER.validate_json(row[0]) for row in rows]

    async def write_record(
        self,
        record: ConversationRecord,
        *,
        new_entries: Sequence[ConversationEntry] = (),
        expected_revision: str | None,
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
        expected_revision: str | None,
    ) -> None:
        with self._connected() as conn:
            with _atomically(conn, store_owned=self._connection is None, begin='BEGIN IMMEDIATE'):
                row = conn.execute(
                    'SELECT revision FROM conversation_records WHERE conversation_id = ?',
                    (record.conversation_id,),
                ).fetchone()
                if (row[0] if row is not None else None) != expected_revision:
                    raise ConversationConflict(record.conversation_id)
                conn.executemany(
                    'INSERT OR IGNORE INTO conversation_messages (sha256, message) VALUES (?, ?)', messages
                )
                new_ids = {entry.id for entry in entries}
                parents = sorted({entry.parent_id for entry in entries if entry.parent_id not in (None, *new_ids)})
                stored = {
                    entry_id: (root_id, depth)
                    for entry_id, root_id, depth in conn.execute(
                        'SELECT id, root_id, depth FROM conversation_entries WHERE id IN '
                        f'({", ".join("?" * len(parents))})',
                        parents,
                    )
                }
                if len(stored) < len(parents):
                    # A delete released an entry the new ones continue from.
                    raise ConversationConflict(record.conversation_id)
                conn.executemany(
                    'INSERT INTO conversation_entries (id, parent_id, kind, sha256, root_id, depth) '
                    'VALUES (?, ?, ?, ?, ?, ?)',
                    [
                        (entry.id, entry.parent_id, entry.kind, sha256, root_id, depth)
                        for entry, (sha256, _), (root_id, depth) in zip(
                            entries, messages, entry_positions(entries, stored), strict=True
                        )
                    ],
                )
                # A fork points into another conversation's history, which a concurrent delete or
                # rewrite of that conversation can release before this write.
                if (
                    record.head_id is not None
                    and conn.execute('SELECT 1 FROM conversation_entries WHERE id = ?', (record.head_id,)).fetchone()
                    is None
                ):
                    raise ConversationConflict(record.conversation_id)
                conn.execute(
                    'INSERT INTO conversation_records (conversation_id, revision, head_id, scope, updated_at, record) '
                    'VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (conversation_id) DO UPDATE SET '
                    'revision = excluded.revision, head_id = excluded.head_id, scope = excluded.scope, '
                    'updated_at = excluded.updated_at, record = excluded.record',
                    (
                        record.conversation_id,
                        record.revision,
                        record.head_id,
                        record.scope,
                        record.updated_at.timestamp(),
                        RECORD_ADAPTER.dump_json(record).decode(),
                    ),
                )
                conn.execute('DELETE FROM conversation_branches WHERE conversation_id = ?', (record.conversation_id,))
                conn.executemany(
                    'INSERT INTO conversation_branches (conversation_id, entry_id) VALUES (?, ?)',
                    [(record.conversation_id, entry_id) for entry_id in record.branches],
                )

    async def delete_record(self, conversation_id: str, *, expected_revision: str) -> None:
        await anyio.to_thread.run_sync(self._sync_delete_record, conversation_id, expected_revision)

    def _sync_delete_record(self, conversation_id: str, expected_revision: str) -> None:
        with self._connected() as conn:
            with _atomically(conn, store_owned=self._connection is None, begin='BEGIN IMMEDIATE'):
                row = conn.execute(
                    'DELETE FROM conversation_records WHERE conversation_id = ? AND revision = ? RETURNING head_id',
                    (conversation_id, expected_revision),
                ).fetchone()
                if row is None:
                    raise ConversationConflict(conversation_id)
                tips = conn.execute(
                    'DELETE FROM conversation_branches WHERE conversation_id = ? RETURNING entry_id', (conversation_id,)
                ).fetchall()
                for (entry_id,) in [row, *tips]:
                    _release(conn, entry_id)

    async def read_entry(self, entry_id: str) -> ConversationEntry:
        row = await anyio.to_thread.run_sync(self._sync_read_entry, entry_id)
        if row is None:
            raise KeyError(entry_id)
        parent_id, message = row
        return ConversationEntry(
            id=entry_id, parent_id=parent_id, message=await load_message(message, media_store=self._media_store)
        )

    def _sync_read_entry(self, entry_id: str) -> tuple[str | None, str] | None:
        with self._connected() as conn:
            return conn.execute(
                'SELECT e.parent_id, m.message FROM conversation_entries e '
                'JOIN conversation_messages m ON m.sha256 = e.sha256 WHERE e.id = ?',
                (entry_id,),
            ).fetchone()

    async def read_history(self, head_id: str | None, *, limit: int | None = None) -> list[ConversationEntry]:
        """The entries from the first one up to `head_id`, in order, or only the last `limit` of them.

        Read in one query.
        """
        if head_id is None or limit == 0:
            return []
        rows = await anyio.to_thread.run_sync(self._sync_read_history, head_id, -1 if limit is None else limit)
        if not rows or (rows[0][1] is not None and (limit is None or len(rows) < limit)):
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

    def _sync_read_history(self, head_id: str, limit: int) -> list[tuple[str, str | None, str | None]]:
        with self._connected() as conn:
            return conn.execute(_HISTORY_SQL, (head_id, limit, limit)).fetchall()


def _release(conn: sqlite3.Connection, entry_id: str | None) -> None:
    """Delete `entry_id` and its ancestors, up to the first one another record's history or branch still reaches.

    That's a record's head, the last entry of a branch a record lists, or an entry with another child. A
    message no remaining entry holds goes too.
    """
    while entry_id is not None:
        deleted = conn.execute(
            'DELETE FROM conversation_entries WHERE id = ? '
            'AND NOT EXISTS (SELECT 1 FROM conversation_records WHERE head_id = ?) '
            'AND NOT EXISTS (SELECT 1 FROM conversation_branches WHERE entry_id = ?) '
            'AND NOT EXISTS (SELECT 1 FROM conversation_entries WHERE parent_id = ?) '
            'RETURNING parent_id, sha256',
            (entry_id, entry_id, entry_id, entry_id),
        ).fetchone()
        if deleted is None:
            break
        entry_id, sha256 = deleted
        conn.execute(
            'DELETE FROM conversation_messages WHERE sha256 = ? '
            'AND NOT EXISTS (SELECT 1 FROM conversation_entries WHERE sha256 = ?)',
            (sha256, sha256),
        )


class _LockedSqliteMediaStore(SqliteMediaStore):
    """A `SqliteMediaStore` on a caller-owned connection the conversation store shares, under the store's lock.

    Media is stored before a save takes the lock, so without it a save's media could run on the connection
    in the middle of another save's transaction.
    """

    def __init__(self, *, connection: sqlite3.Connection | None, lock: threading.Lock) -> None:
        super().__init__(connection=connection)
        self._lock = lock

    def _open(self) -> sqlite3.Connection:
        self._lock.acquire()
        return super()._open()

    def _maybe_close(self, conn: sqlite3.Connection) -> None:
        try:
            super()._maybe_close(conn)
        finally:
            self._lock.release()
