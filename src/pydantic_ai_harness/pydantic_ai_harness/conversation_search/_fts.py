"""A SQLite FTS5 index of the search corpus, kept in the database a `SqliteStepStore` writes.

The index stores what the scan path derives on every search: each run's durable record,
one row per message, rendered to the text the search ranks plus the keys the inherited-history
dedup compares. A search refreshes a run's rows when the run's `complete` snapshots changed
since they were indexed, so the write path needs no hook, and a database written before the
index existed is indexed by its first search.

FTS5 `MATCH` only finds the candidate messages. Ranking stays the toolset's BM25 over the
in-scope corpus: FTS5's own `bm25()` takes its statistics from the whole table, every
conversation in the store, and fixes `k1` and `b`.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TypeVar

import anyio.to_thread

from pydantic_ai_harness.conversation_search._source import HistorySource, SnapshotHistorySource
from pydantic_ai_harness.step_persistence import SqliteStepStore

T = TypeVar('T')

_FORMAT_VERSION = 1
"""Stored with each indexed run. Bump it when the rendering or the dedup keys change, so older rows are rebuilt."""

_TABLES = (
    'CREATE TABLE IF NOT EXISTS conversation_search_runs ('
    'run_id TEXT PRIMARY KEY, snapshot_seq INTEGER NOT NULL, snapshot_count INTEGER NOT NULL, '
    'format_version INTEGER NOT NULL) WITHOUT ROWID',
    'CREATE TABLE IF NOT EXISTS conversation_search_messages ('
    'id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, position INTEGER NOT NULL, hash TEXT NOT NULL, '
    'deltas TEXT NOT NULL, length INTEGER NOT NULL, text TEXT NOT NULL, layout TEXT NOT NULL, '
    'UNIQUE (run_id, position))',
)

_CREATE_FTS_TABLE = (
    'CREATE VIRTUAL TABLE IF NOT EXISTS conversation_search_fts USING fts5(text, '
    "content='conversation_search_messages', content_rowid='id', tokenize='unicode61 remove_diacritics 2')"
)
"""The external-content FTS5 table over `conversation_search_messages.text`.

`MATCH` must find every message the toolset's word tokens score. Each query term is matched as an
FTS5 string, which FTS5 tokenizes like the text, so a word `unicode61` splits differently
(`snake_case`) still matches as a phrase. `remove_diacritics 2` folds an accent written as a
separate combining mark into its letter, so the token stays `cafe` as the toolset reads it; it also
lets `cafe` match a precomposed `é`, a candidate the scoring then drops.
"""

_TRIGGERS = (
    'CREATE TRIGGER IF NOT EXISTS conversation_search_messages_insert '
    'AFTER INSERT ON conversation_search_messages BEGIN '
    'INSERT INTO conversation_search_fts (rowid, text) VALUES (NEW.id, NEW.text); END',
    'CREATE TRIGGER IF NOT EXISTS conversation_search_messages_delete '
    'AFTER DELETE ON conversation_search_messages BEGIN '
    "INSERT INTO conversation_search_fts (conversation_search_fts, rowid, text) VALUES ('delete', OLD.id, OLD.text); "
    'END',
    # Deleting a run (as deleting a saved conversation does) removes its text from the index too.
    'CREATE TRIGGER IF NOT EXISTS conversation_search_runs_delete AFTER DELETE ON runs BEGIN '
    'DELETE FROM conversation_search_messages WHERE run_id = OLD.run_id; '
    'DELETE FROM conversation_search_runs WHERE run_id = OLD.run_id; END',
)

_SCHEMA_OBJECTS = (
    'conversation_search_runs',
    'conversation_search_messages',
    'conversation_search_fts',
    'conversation_search_messages_insert',
    'conversation_search_messages_delete',
    'conversation_search_runs_delete',
)


class Fts5Unavailable(Exception):
    """The SQLite library Python links against was built without FTS5."""


@dataclass(frozen=True, kw_only=True)
class IndexedMessage:
    """One message as the index stores it."""

    hash: str
    """`message_hash` of the message, for the inherited-history dedup."""
    deltas: str
    """Keys of the instruction changes the message carries, space separated."""
    length: int
    """Token count of `text`, for BM25 length normalization."""
    text: str
    """The full rendering the search ranks."""
    layout: str
    """JSON `[prefix length, body length, display limit]` per line of `text`, to clip it for display."""


def _ids(run_ids: Sequence[str]) -> str:
    return json.dumps(list(run_ids))


def _match_expression(terms: Sequence[str]) -> str:
    """An FTS5 query for messages holding any of `terms`.

    Each term is an FTS5 string, so query syntax in it (`AND`, `NEAR`, `*`, `^`, `:`, quotes)
    is matched as text instead of being interpreted.
    """
    return ' OR '.join('"' + term.replace('"', '""') + '"' for term in terms)


@contextmanager
def _transaction(conn: sqlite3.Connection, begin: str) -> Generator[None]:
    """Run the statements inside as one transaction, or join a caller's open one through a savepoint."""
    if conn.in_transaction:
        conn.execute('SAVEPOINT conversation_search')
        try:
            yield
        except BaseException:
            conn.execute('ROLLBACK TO conversation_search')
            conn.execute('RELEASE conversation_search')
            raise
        conn.execute('RELEASE conversation_search')
        return
    conn.execute(begin)
    try:
        yield
    except BaseException:
        conn.execute('ROLLBACK')
        raise
    conn.execute('COMMIT')


class IndexReader:
    """Queries the index inside one read transaction, so every query sees the same rows."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def message_keys(self, run_ids: Sequence[str]) -> dict[str, list[tuple[str, str]]]:
        """Each run's `(hash, deltas)` per message, in message order."""
        keys: dict[str, list[tuple[str, str]]] = {run_id: [] for run_id in run_ids}
        for run_id, message_hash, deltas in self._conn.execute(
            'SELECT run_id, hash, deltas FROM conversation_search_messages '
            'WHERE run_id IN (SELECT value FROM json_each(?)) ORDER BY run_id, position',
            (_ids(run_ids),),
        ):
            keys[run_id].append((message_hash, deltas))
        return keys

    def total_length(self, spans: Mapping[str, int]) -> float:
        """Summed token count of the messages in `spans` (run id to the position its corpus starts at)."""
        (total,) = self._conn.execute(
            'SELECT TOTAL(m.length) FROM conversation_search_messages m '
            'JOIN json_each(?) s ON s.key = m.run_id WHERE m.position >= s.value',
            (json.dumps(dict(spans)),),
        ).fetchone()
        return total

    def candidates(self, terms: Sequence[str], spans: Mapping[str, int]) -> list[tuple[str, int, str]]:
        """`(run_id, position, text)` of each message in `spans` that FTS5 finds holding one of `terms`.

        The scope join runs inside the query, so a message outside `spans` never leaves SQLite.
        """
        return self._conn.execute(
            'SELECT m.run_id, m.position, m.text FROM conversation_search_fts f '
            'JOIN conversation_search_messages m ON m.id = f.rowid '
            'JOIN json_each(?) s ON s.key = m.run_id '
            'WHERE conversation_search_fts MATCH ? AND m.position >= s.value',
            (json.dumps(dict(spans)), _match_expression(terms)),
        ).fetchall()

    def window(self, run_id: str, start: int, end: int) -> list[tuple[str, str]]:
        """`(text, layout)` of the run's messages at positions `start` up to `end`, in order."""
        return self._conn.execute(
            'SELECT text, layout FROM conversation_search_messages '
            'WHERE run_id = ? AND position >= ? AND position < ? ORDER BY position',
            (run_id, start, end),
        ).fetchall()


class SqliteSearchIndex:
    """The FTS5 index in a `SqliteStepStore`'s database."""

    def __init__(self, store: SqliteStepStore) -> None:
        self._store = store
        self._ready = False

    @staticmethod
    def for_source(source: HistorySource) -> SqliteSearchIndex | None:
        """The index for a `SnapshotHistorySource` over a `SqliteStepStore`, else `None`."""
        if not isinstance(source, SnapshotHistorySource):
            return None
        store = source._store  # pyright: ignore[reportPrivateUsage]
        return SqliteSearchIndex(store) if isinstance(store, SqliteStepStore) else None

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection]:
        with self._store._connect() as conn:  # pyright: ignore[reportPrivateUsage]
            self._prepare(conn)
            yield conn

    def _prepare(self, conn: sqlite3.Connection) -> None:
        """Create the index on first use; raise `Fts5Unavailable` when SQLite lacks FTS5."""
        if self._ready:
            return
        (fts5,) = conn.execute("SELECT sqlite_compileoption_used('ENABLE_FTS5')").fetchone()
        if not fts5:
            raise Fts5Unavailable('the SQLite library was built without FTS5')
        (present,) = conn.execute(
            'SELECT COUNT(*) FROM sqlite_master WHERE name IN (SELECT value FROM json_each(?))',
            (_ids(_SCHEMA_OBJECTS),),
        ).fetchone()
        # Checked first so a reader takes no write lock once the index exists.
        if present != len(_SCHEMA_OBJECTS):
            with _transaction(conn, 'BEGIN IMMEDIATE'):
                for statement in (*_TABLES, _CREATE_FTS_TABLE, *_TRIGGERS):
                    conn.execute(statement)
        self._ready = True

    async def refresh(self, run_ids: Sequence[str], load: Callable[[str], Awaitable[list[IndexedMessage]]]) -> None:
        """Re-index each of `run_ids` whose `complete` snapshots changed since it was indexed.

        `load` renders a run's current record. Each run is written in its own short
        transaction, so indexing a large database does not hold the write lock other
        processes' snapshot saves wait on.
        """
        stale = await anyio.to_thread.run_sync(self._sync_stale, list(run_ids))
        for run_id, state in stale.items():
            messages = await load(run_id)
            await anyio.to_thread.run_sync(self._sync_replace, run_id, state, messages)

    def _sync_stale(self, run_ids: list[str]) -> dict[str, tuple[int, int]]:
        with self._connect() as conn:
            current: dict[str, tuple[int, int]] = {
                run_id: (seq, count)
                for run_id, seq, count in conn.execute(
                    'SELECT run_id, MAX(seq), COUNT(*) FROM snapshots '
                    "WHERE state = 'complete' AND run_id IN (SELECT value FROM json_each(?)) GROUP BY run_id",
                    (_ids(run_ids),),
                )
            }
            indexed: dict[str, tuple[int, int, int]] = {
                run_id: (seq, count, version)
                for run_id, seq, count, version in conn.execute(
                    'SELECT run_id, snapshot_seq, snapshot_count, format_version FROM conversation_search_runs '
                    'WHERE run_id IN (SELECT value FROM json_each(?))',
                    (_ids(run_ids),),
                )
            }
        states = {run_id: current.get(run_id, (0, 0)) for run_id in run_ids}
        return {run_id: state for run_id, state in states.items() if indexed.get(run_id) != (*state, _FORMAT_VERSION)}

    def _sync_replace(self, run_id: str, state: tuple[int, int], messages: list[IndexedMessage]) -> None:
        with self._connect() as conn, _transaction(conn, 'BEGIN IMMEDIATE'):
            # Another process may have indexed this state meanwhile, or deleted the run; a
            # deleted run's text must not come back through the index.
            indexed = conn.execute(
                'SELECT snapshot_seq, snapshot_count, format_version FROM conversation_search_runs WHERE run_id = ?',
                (run_id,),
            ).fetchone()
            if indexed == (*state, _FORMAT_VERSION):
                return
            if conn.execute('SELECT 1 FROM runs WHERE run_id = ?', (run_id,)).fetchone() is None:
                return
            conn.execute('DELETE FROM conversation_search_messages WHERE run_id = ?', (run_id,))
            conn.executemany(
                'INSERT INTO conversation_search_messages '
                '(run_id, position, hash, deltas, length, text, layout) VALUES (?, ?, ?, ?, ?, ?, ?)',
                [
                    (run_id, position, m.hash, m.deltas, m.length, m.text, m.layout)
                    for position, m in enumerate(messages)
                ],
            )
            conn.execute(
                'INSERT OR REPLACE INTO conversation_search_runs '
                '(run_id, snapshot_seq, snapshot_count, format_version) VALUES (?, ?, ?, ?)',
                (run_id, *state, _FORMAT_VERSION),
            )

    async def read(self, query: Callable[[IndexReader], T]) -> T:
        """Run `query` against the index in one read transaction, on a worker thread."""
        return await anyio.to_thread.run_sync(self._sync_read, query)

    def _sync_read(self, query: Callable[[IndexReader], T]) -> T:
        with self._connect() as conn, _transaction(conn, 'BEGIN'):
            return query(IndexReader(conn))
