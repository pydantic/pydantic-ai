"""Tests for `SqliteConversationStore` and `PostgresConversationStore`.

Both run the shared `ConversationStoreContract`. `PostgresConversationStore` runs against
`SqlitePool`, the in-memory SQLite stand-in for a pool; what only a real server shows is covered
by `pydantic_ai_harness/integration_tests/postgres`.
"""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import closing
from pathlib import Path

import anyio
import pytest

from pydantic_ai import BinaryImage, Conversation, ModelRequest, UserPromptPart
from pydantic_ai.conversation_store import ConversationConflict, ConversationEntry, ConversationRecord
from pydantic_ai_harness.media import SqliteMediaStore
from pydantic_ai_harness.step_persistence import PostgresConversationStore, SqliteConversationStore
from tests.harness._postgres_sqlite import SqlitePool
from tests.harness.step_persistence._conversation_store_contract import Backend, ConversationStoreContract


def _sqlite_backend(store: SqliteConversationStore, db: Callable[[], sqlite3.Connection]) -> Backend:
    def query(sql: str, *params: object) -> int:
        conn = db()
        try:
            row = conn.execute(sql, params).fetchone()
            conn.commit()
            return row[0] if row is not None and row[0] is not None else 0
        finally:
            if conn is not store._connection:  # pyright: ignore[reportPrivateUsage]
                conn.close()

    async def count_entries() -> int:
        return query('SELECT count(*) FROM conversation_entries')

    async def count_messages() -> int:
        return query('SELECT count(*) FROM conversation_messages')

    async def delete_entry(entry_id: str) -> None:
        query('DELETE FROM conversation_entries WHERE id = ?', entry_id)

    async def delete_messages() -> None:
        query('DELETE FROM conversation_messages')

    async def stored_message_bytes() -> int:
        return query('SELECT sum(length(message)) FROM conversation_messages')

    return Backend(store, count_entries, count_messages, delete_entry, delete_messages, stored_message_bytes)


class TestSqliteConversationStoreOnAFile(ConversationStoreContract):
    @pytest.fixture
    def backend(self, tmp_path: Path) -> Backend:
        database = tmp_path / 'conversations.db'
        store = SqliteConversationStore(database=database)
        return _sqlite_backend(store, lambda: sqlite3.connect(database))


class TestSqliteConversationStoreOnACallerConnection(ConversationStoreContract):
    @pytest.fixture
    def backend(self) -> Iterator[Backend]:
        connection = sqlite3.connect(':memory:', check_same_thread=False, isolation_level=None)
        yield _sqlite_backend(SqliteConversationStore(connection=connection), lambda: connection)
        connection.close()


async def test_sqlite_rejects_a_record_whose_history_was_released() -> None:
    """A fork racing a delete of the conversation it forks can't point at entries the delete removed."""
    with closing(sqlite3.connect(':memory:', check_same_thread=False, isolation_level=None)) as connection:
        store = SqliteConversationStore(connection=connection)
        original = Conversation(messages=[ModelRequest(parts=[UserPromptPart('one')])])
        await store.save(original)
        record = await store.read_record(original.conversation_id)
        assert record is not None
        await store.delete(original.conversation_id)

        with pytest.raises(ConversationConflict):
            await store.write_record(
                ConversationRecord(conversation_id='fork', head_id=record.head_id), expected_revision=None
            )
        assert await store.read_record('fork') is None


async def test_sqlite_creates_its_tables_again_after_a_caller_rolls_them_back() -> None:
    with closing(sqlite3.connect(':memory:', check_same_thread=False, isolation_level=None)) as connection:
        store = SqliteConversationStore(connection=connection)
        connection.execute('BEGIN')
        assert await store.list_conversations() == []
        connection.execute('ROLLBACK')

        await store.save(conversation := Conversation())
        assert await store.load(conversation.conversation_id) == conversation


def test_sqlite_media_on_a_caller_connection_takes_the_store_lock() -> None:
    """Media is stored before a save takes the lock, so it takes the lock itself rather than run mid-transaction."""
    with closing(sqlite3.connect(':memory:', check_same_thread=False, isolation_level=None)) as connection:
        store = SqliteConversationStore(connection=connection)
        media = store._media_store  # pyright: ignore[reportPrivateUsage]
        lock = store._lock  # pyright: ignore[reportPrivateUsage]
        assert isinstance(media, SqliteMediaStore)

        conn = media._open()  # pyright: ignore[reportPrivateUsage]
        assert lock.locked()
        media._maybe_close(conn)  # pyright: ignore[reportPrivateUsage]
        assert not lock.locked()


class TestPostgresConversationStore(ConversationStoreContract):
    @pytest.fixture
    async def backend(self) -> AsyncIterator[Backend]:
        pool = SqlitePool()
        store = PostgresConversationStore(pool, table='app')

        async def count(sql: str) -> int:
            async with pool.acquire() as connection:
                value = await connection.fetchval(sql)
            return int(value) if isinstance(value, int) else 0

        async def count_entries() -> int:
            return await pool.count_rows('app_conv_entries')

        async def count_messages() -> int:
            return await pool.count_rows('app_conv_messages')

        async def delete_entry(entry_id: str) -> None:
            async with pool.acquire() as connection:
                await connection.execute('DELETE FROM app_conv_entries WHERE id = $1', entry_id)

        async def delete_messages() -> None:
            async with pool.acquire() as connection:
                await connection.execute('DELETE FROM app_conv_messages')

        async def stored_message_bytes() -> int:
            return await count('SELECT sum(length(message)) FROM app_conv_messages')

        yield Backend(store, count_entries, count_messages, delete_entry, delete_messages, stored_message_bytes)
        pool.close()


async def test_without_a_media_store_large_parts_stay_in_the_message(tmp_path: Path) -> None:
    store = SqliteConversationStore(database=tmp_path / 'conversations.db', media_store=None)
    image = BinaryImage(data=b'\x89PNG' + b'\0' * 200_000, media_type='image/png')
    conversation = Conversation(messages=[ModelRequest(parts=[UserPromptPart(['Look', image])])])

    await store.save(conversation)

    assert await store.load(conversation.conversation_id) == conversation
    with closing(sqlite3.connect(tmp_path / 'conversations.db')) as db:
        (stored,) = db.execute('SELECT max(length(message)) FROM conversation_messages').fetchone()
    assert stored > 200_000


def test_sqlite_store_needs_exactly_one_of_database_or_connection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='exactly one'):
        SqliteConversationStore()
    with closing(sqlite3.connect(':memory:')) as connection, pytest.raises(ValueError, match='exactly one'):
        SqliteConversationStore(database=tmp_path / 'a.db', connection=connection)


def test_postgres_store_rejects_an_invalid_table_name() -> None:
    pool = SqlitePool()
    with pytest.raises(ValueError, match='invalid table name'):
        PostgresConversationStore(pool, table='Bad-Name')
    pool.close()


async def test_postgres_store_creates_its_schema_once_under_concurrent_first_use() -> None:
    pool = SqlitePool()
    store = PostgresConversationStore(pool, media_store=None)

    async with anyio.create_task_group() as tg:
        for _ in range(3):
            tg.start_soon(store.read_record, 'unknown')

    assert sum('CREATE TABLE IF NOT EXISTS step_persistence_conv_records' in s for s in pool.statements) == 1
    pool.close()


async def test_concurrent_saves_on_a_caller_connection_take_turns() -> None:
    with closing(sqlite3.connect(':memory:', check_same_thread=False, isolation_level=None)) as connection:
        store = SqliteConversationStore(connection=connection, media_store=None)
        conversations = [Conversation(messages=[ModelRequest(parts=[UserPromptPart(str(i))])]) for i in range(20)]

        async with anyio.create_task_group() as tg:
            for conversation in conversations:
                tg.start_soon(store.save, conversation)

        assert [await store.load(c.conversation_id) for c in conversations] == conversations


async def test_postgres_a_record_pointing_at_a_released_entry_conflicts() -> None:
    pool = SqlitePool()
    pool.db.execute('PRAGMA foreign_keys = ON')
    store = PostgresConversationStore(pool, media_store=None)

    with pytest.raises(ConversationConflict):
        await store.write_record(ConversationRecord(conversation_id='c', head_id='released'), expected_revision=None)

    assert await store.read_record('c') is None
    pool.close()


async def test_postgres_other_write_errors_are_not_conflicts() -> None:
    pool = SqlitePool()
    store = PostgresConversationStore(pool, media_store=None)
    entry = ConversationEntry(parent_id=None, message=ModelRequest(parts=[UserPromptPart('one')]))
    await store.write_record(
        ConversationRecord(conversation_id='a', head_id=entry.id), new_entries=[entry], expected_revision=None
    )

    with pytest.raises(sqlite3.IntegrityError):
        await store.write_record(
            ConversationRecord(conversation_id='b', head_id=entry.id), new_entries=[entry], expected_revision=None
        )
    pool.close()


def _prompt(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(text)])


async def _positions_after_a_fork(
    store: SqliteConversationStore | PostgresConversationStore,
) -> tuple[str, list[str], str]:
    """Save three messages, fork after the second and add one; return each entry's history and depth."""
    original = Conversation(messages=[_prompt('one'), _prompt('two'), _prompt('three')])
    entries = await store.save(original)
    fork = await store.fork(original.conversation_id, at=2)
    fork.messages.append(_prompt('four'))
    [added] = await store.save(fork)
    return entries[0].id, [entry.id for entry in entries], added.id


async def test_sqlite_entries_record_their_history_and_depth(tmp_path: Path) -> None:
    store = SqliteConversationStore(database=tmp_path / 'c.db', media_store=None)
    root, ids, added = await _positions_after_a_fork(store)

    with closing(sqlite3.connect(tmp_path / 'c.db')) as conn:
        rows = dict(
            (entry_id, (root_id, depth))
            for entry_id, root_id, depth in conn.execute('SELECT id, root_id, depth FROM conversation_entries')
        )
    assert rows == {**{entry_id: (root, depth) for depth, entry_id in enumerate(ids)}, added: (root, 2)}


async def test_postgres_entries_record_their_history_and_depth() -> None:
    pool = SqlitePool()
    store = PostgresConversationStore(pool, media_store=None)
    root, ids, added = await _positions_after_a_fork(store)

    rows = dict(
        (entry_id, (root_id, depth))
        for entry_id, root_id, depth in pool.db.execute('SELECT id, root_id, depth FROM step_persistence_conv_entries')
    )
    assert rows == {**{entry_id: (root, depth) for depth, entry_id in enumerate(ids)}, added: (root, 2)}
    pool.close()


async def test_postgres_writes_and_deletes_in_one_history_take_its_lock() -> None:
    pool = SqlitePool()
    store = PostgresConversationStore(pool, media_store=None)
    original = Conversation(messages=[_prompt('one')])
    await store.save(original)

    def locks() -> int:
        return sum('pg_advisory_xact_lock' in statement for statement in pool.statements)

    before = locks()
    await store.fork(original.conversation_id)
    after_fork = locks()
    await store.delete(original.conversation_id)

    assert (after_fork - before, locks() - after_fork) == (1, 1)
    pool.close()


async def test_postgres_a_write_that_loses_a_race_with_a_delete_conflicts(monkeypatch: pytest.MonkeyPatch) -> None:
    """A delete can release an entry after the write looked it up; the foreign key then turns it into a conflict."""
    pool = SqlitePool()
    pool.db.execute('PRAGMA foreign_keys = ON')
    store = PostgresConversationStore(pool, media_store=None)

    async def released_after_lookup(_connection: object, entry_ids: set[str]) -> dict[str, tuple[str, int]]:
        return {entry_id: (entry_id, 0) for entry_id in entry_ids}

    monkeypatch.setattr(store, '_positions', released_after_lookup)
    entry = ConversationEntry(parent_id='released', message=_prompt('one'))
    with pytest.raises(ConversationConflict):
        await store.write_record(
            ConversationRecord(conversation_id='c', head_id=entry.id), new_entries=[entry], expected_revision=None
        )
    pool.close()


async def test_postgres_a_write_continuing_from_a_released_entry_conflicts() -> None:
    pool = SqlitePool()
    store = PostgresConversationStore(pool, media_store=None)
    entry = ConversationEntry(parent_id='released', message=_prompt('one'))

    with pytest.raises(ConversationConflict):
        await store.write_record(
            ConversationRecord(conversation_id='c', head_id=entry.id), new_entries=[entry], expected_revision=None
        )
    pool.close()


async def test_sqlite_a_write_continuing_from_a_released_entry_conflicts(tmp_path: Path) -> None:
    store = SqliteConversationStore(database=tmp_path / 'c.db', media_store=None)
    entry = ConversationEntry(parent_id='released', message=_prompt('one'))

    with pytest.raises(ConversationConflict):
        await store.write_record(
            ConversationRecord(conversation_id='c', head_id=entry.id), new_entries=[entry], expected_revision=None
        )


def test_sqlite_store_rejects_an_in_memory_database_path() -> None:
    with pytest.raises(ValueError, match="`database=':memory:'` is not supported"):
        SqliteConversationStore(database=':memory:')


async def test_first_use_of_a_caller_connection_leaves_its_open_transaction_to_the_caller() -> None:
    with closing(sqlite3.connect(':memory:', check_same_thread=False, isolation_level=None)) as connection:
        connection.execute('CREATE TABLE app (value TEXT)')
        connection.execute('BEGIN')
        connection.execute("INSERT INTO app VALUES ('uncommitted')")
        store = SqliteConversationStore(connection=connection, media_store=None)

        await store.read_record('unknown')
        connection.execute('ROLLBACK')

        assert connection.execute('SELECT count(*) FROM app').fetchone() == (0,)
