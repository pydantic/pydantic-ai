"""`SqliteStepStore` against a real Turso connection.

Turso is a SQLite fork whose Python client provides the connection extensions the store uses,
so it is passed to the existing store as a caller-owned `connection=` rather than needing a
backend of its own. These tests run the real driver to cover its SQL and error behavior.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import anyio
import pytest
import turso

from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai_harness.media import SqliteMediaStore
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    RunRecord,
    SqliteStepStore,
    StepEvent,
    ToolEffectRecord,
)

TS = datetime(2026, 1, 1, tzinfo=timezone.utc)
_Parameters = Sequence[object] | Mapping[str, object]


class _ConnectionSubclassWithoutLocalErrors(sqlite3.Connection):
    """Model a driver wrapper whose DB-API errors live on its base class module."""

    @property
    def DatabaseError(self) -> type[Exception]:
        raise AttributeError


class _OverlapDetectingConnection:
    """Fail if two worker threads use the wrapped connection at once."""

    def __init__(self, connection: turso.Connection) -> None:
        self._connection = connection
        self._guard = threading.Lock()
        self._active = False

    @contextmanager
    def _exclusive_call(self) -> Iterator[None]:
        with self._guard:
            if self._active:
                raise AssertionError('concurrent connection use')
            self._active = True
        try:
            time.sleep(0.01)
            yield
        finally:
            with self._guard:
                self._active = False

    def execute(self, sql: str, parameters: _Parameters = (), /) -> turso.Cursor:
        with self._exclusive_call():
            return self._connection.execute(sql, parameters)

    def executescript(self, sql_script: str, /) -> turso.Cursor:
        with self._exclusive_call():
            return self._connection.executescript(sql_script)

    def commit(self) -> None:
        with self._exclusive_call():
            self._connection.commit()

    def rollback(self) -> None:
        with self._exclusive_call():
            self._connection.rollback()

    def close(self) -> None:
        self._connection.close()

    @property
    def DatabaseError(self) -> type[Exception]:
        return getattr(self._connection, 'DatabaseError')

    @property
    def in_transaction(self) -> bool:
        with self._exclusive_call():
            return self._connection.in_transaction


@pytest.fixture
def turso_store(tmp_path: Path) -> Iterator[SqliteStepStore]:
    connection = turso.connect(str(tmp_path / 'runs.db'), isolation_level=None)
    yield SqliteStepStore(connection=connection, media_store=None)
    connection.close()


async def test_every_record_type_round_trips_after_reopen(tmp_path: Path) -> None:
    database = tmp_path / 'runs.db'
    connection = turso.connect(str(database), isolation_level=None)
    turso_store = SqliteStepStore(connection=connection, media_store=None)
    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='hello')])]

    await turso_store.register_run(RunRecord(run_id='r1', conversation_id='c1', started_at=TS))
    await turso_store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0, timestamp=TS))
    await turso_store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=1, messages=messages, timestamp=TS))
    await turso_store.record_tool_effect(
        ToolEffectRecord(run_id='r1', tool_call_id='t1', tool_name='get_weather', status='started', started_at=TS)
    )
    connection.close()

    reopened_connection = turso.connect(str(database), isolation_level=None)
    turso_store = SqliteStepStore(connection=reopened_connection, media_store=None)

    run = await turso_store.get_run(run_id='r1')
    assert run is not None
    assert run.conversation_id == 'c1'
    assert [event.kind for event in await turso_store.list_events(run_id='r1')] == ['run_started']
    snapshot = await turso_store.latest_snapshot(run_id='r1')
    assert snapshot is not None
    assert snapshot.messages == messages
    effect = await turso_store.get_tool_effect(run_id='r1', tool_call_id='t1')
    assert effect is not None
    assert effect.status == 'started'
    assert [record.run_id for record in await turso_store.list_runs()] == ['r1']
    assert [record.tool_call_id for record in await turso_store.list_unresolved_tool_effects(run_id='r1')] == ['t1']
    reopened_connection.close()


async def test_reused_run_id_raises_the_drivers_integrity_error(turso_store: SqliteStepStore) -> None:
    """The single-shot `run_id` contract rests on the primary key, not on a `sqlite3` exception class."""
    await turso_store.register_run(RunRecord(run_id='r1'))

    with pytest.raises(turso.IntegrityError):
        await turso_store.register_run(RunRecord(run_id='r1'))


async def test_caller_transaction_is_not_committed_or_rolled_back(tmp_path: Path) -> None:
    connection = turso.connect(str(tmp_path / 'caller-transaction.db'), isolation_level=None)
    store = SqliteStepStore(connection=connection, media_threshold_bytes=1)
    await store.register_run(RunRecord(run_id='r1'))
    connection.execute('CREATE TABLE caller_data (value TEXT)')
    connection.execute('BEGIN')
    connection.execute("INSERT INTO caller_data VALUES ('pending')")

    await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0, timestamp=TS))
    await store.save_snapshot(
        ContinuableSnapshot(
            run_id='r1',
            step_index=0,
            timestamp=TS,
            messages=[ModelRequest(parts=[UserPromptPart(content='hello')])],
        )
    )
    assert connection.in_transaction
    with pytest.raises(turso.IntegrityError):
        await store.register_run(RunRecord(run_id='r1'))
    assert connection.in_transaction

    connection.rollback()
    assert await store.list_events(run_id='r1') == []
    assert await store.latest_snapshot(run_id='r1') is None
    assert connection.execute('SELECT * FROM caller_data').fetchall() == []
    connection.close()


async def test_first_operation_requires_an_idle_connection(tmp_path: Path) -> None:
    connection = turso.connect(str(tmp_path / 'active.db'), isolation_level=None)
    connection.execute('BEGIN')
    store = SqliteStepStore(connection=connection, media_store=None)

    with pytest.raises(RuntimeError, match='must be idle for the first store operation'):
        await store.get_run(run_id='r1')

    connection.rollback()
    connection.close()


async def test_legacy_database_without_state_column_migrates(tmp_path: Path) -> None:
    """The `ALTER TABLE` migrations are attempt-and-catch, and Turso raises a different class.

    `turso.DatabaseError` does not inherit from `sqlite3.OperationalError`, so the migration
    catches the driver's error and checks the schema before deciding whether to re-raise it.
    """
    db = tmp_path / 'runs.db'
    setup = turso.connect(str(db), isolation_level=None)
    setup.executescript(
        'CREATE TABLE snapshots ('
        'seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, step_index INTEGER NOT NULL, '
        'conversation_id TEXT, parent_run_id TEXT, agent_name TEXT, timestamp TEXT NOT NULL, '
        'messages TEXT NOT NULL);'
    )
    setup.execute(
        'INSERT INTO snapshots (run_id, step_index, conversation_id, parent_run_id, agent_name, '
        "timestamp, messages) VALUES ('r1', 3, NULL, NULL, NULL, '2026-01-01T00:00:00+00:00', '[]')"
    )
    setup.commit()
    setup.close()

    connection = turso.connect(str(db), isolation_level=None)
    store = SqliteStepStore(connection=connection, media_store=None)
    migrated = await store.latest_snapshot(run_id='r1')

    assert migrated is not None
    assert migrated.state == 'complete'
    assert migrated.step_index == 3
    connection.close()


async def test_media_round_trips_after_reopen(tmp_path: Path) -> None:
    database = tmp_path / 'media.db'
    connection = turso.connect(str(database), isolation_level=None)
    store = SqliteMediaStore(connection=connection)
    uri = await store.put(b'content')
    connection.close()

    reopened_connection = turso.connect(str(database), isolation_level=None)
    reopened = SqliteMediaStore(connection=reopened_connection)
    assert await reopened.get(uri) == b'content'
    reopened_connection.close()


async def test_media_preserves_a_caller_transaction(tmp_path: Path) -> None:
    connection = turso.connect(str(tmp_path / 'media-transaction.db'), isolation_level=None)
    store = SqliteMediaStore(connection=connection)
    await store.put(b'initialize')
    connection.execute('CREATE TABLE caller_data (value TEXT)')
    connection.execute('BEGIN')
    connection.execute("INSERT INTO caller_data VALUES ('pending')")

    uri = await store.put(b'rolled back')
    assert connection.in_transaction
    connection.rollback()

    assert not await store.exists(uri)
    assert connection.execute('SELECT * FROM caller_data').fetchall() == []
    connection.close()


async def test_media_first_operation_requires_an_idle_connection(tmp_path: Path) -> None:
    connection = turso.connect(str(tmp_path / 'active-media.db'), isolation_level=None)
    connection.execute('BEGIN')
    store = SqliteMediaStore(connection=connection)

    with pytest.raises(RuntimeError, match='must be idle for the first store operation'):
        await store.exists('media+sha256://' + '0' * 64)

    connection.rollback()
    connection.close()


async def test_stdlib_caller_still_controls_transactions(tmp_path: Path) -> None:
    step_connection = sqlite3.connect(tmp_path / 'stdlib-step.db', check_same_thread=False)
    step_store = SqliteStepStore(connection=step_connection, media_store=None)
    await step_store.register_run(RunRecord(run_id='rolled-back'))
    assert step_connection.in_transaction
    step_connection.rollback()
    assert await step_store.get_run(run_id='rolled-back') is None
    step_connection.close()

    media_connection = sqlite3.connect(tmp_path / 'stdlib-media.db', check_same_thread=False)
    media_store = SqliteMediaStore(connection=media_connection)
    assert not await media_store.exists('media+sha256://' + '0' * 64)
    uri = await media_store.put(b'rolled back')
    assert media_connection.in_transaction
    media_connection.rollback()
    assert not await media_store.exists(uri)
    media_connection.close()


async def test_concurrent_step_and_implicit_media_access_is_serialized(tmp_path: Path) -> None:
    inner = turso.connect(str(tmp_path / 'concurrent.db'), isolation_level=None)
    connection = _OverlapDetectingConnection(inner)
    store = SqliteStepStore(connection=connection, media_threshold_bytes=1)
    await store.register_run(RunRecord(run_id='r1'))

    async def append_event(index: int) -> None:
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=index, timestamp=TS))

    async def save_snapshot(index: int) -> None:
        await store.save_snapshot(
            ContinuableSnapshot(
                run_id='r1',
                step_index=index,
                timestamp=TS,
                messages=[ModelRequest(parts=[UserPromptPart(content=f'message {index}')])],
            )
        )

    async with anyio.create_task_group() as task_group:
        for index in range(4):
            task_group.start_soon(append_event, index)
            task_group.start_soon(save_snapshot, index)

    assert len(await store.list_events(run_id='r1')) == 4
    assert len(await store.list_snapshots(run_id='r1')) == 4
    connection.close()


async def test_connection_subclass_resolves_database_errors_from_base_module(tmp_path: Path) -> None:
    connection = sqlite3.connect(
        tmp_path / 'subclass.db', factory=_ConnectionSubclassWithoutLocalErrors, check_same_thread=False
    )
    store = SqliteStepStore(connection=connection, media_store=None)

    assert await store.get_run(run_id='missing') is None
    connection.close()
