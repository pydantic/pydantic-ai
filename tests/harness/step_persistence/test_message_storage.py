"""Each message is stored once, and snapshots reference it, on the file, SQLite, and Postgres stores.

`MongoStepStore` has the same coverage in `test_mongo.py`, which is skipped without `pymongo`.
The Postgres store runs against `SqlitePool`, as in `test_postgres.py`.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    FileStepStore,
    PostgresStepStore,
    SqliteStepStore,
    StepPersistence,
    continue_run,
)
from tests.harness._postgres_sqlite import SqlitePool

KINDS = ['file', 'sqlite', 'postgres']

_Store = FileStepStore | SqliteStepStore | PostgresStepStore


@pytest.fixture
def pool() -> Iterator[SqlitePool]:
    sqlite_pool = SqlitePool()
    yield sqlite_pool
    sqlite_pool.close()


def _store(
    kind: str,
    tmp_path: Path,
    pool: SqlitePool,
    *,
    deduplicate_messages: bool = True,
    max_snapshots_per_run: int | None = None,
    media_store: Literal['auto'] | None = None,
) -> _Store:
    if kind == 'file':
        return FileStepStore(
            tmp_path / 'runs',
            media_store=media_store,
            max_snapshots_per_run=max_snapshots_per_run,
            deduplicate_messages=deduplicate_messages,
        )
    if kind == 'sqlite':
        return SqliteStepStore(
            database=tmp_path / 'runs.db',
            media_store=media_store,
            max_snapshots_per_run=max_snapshots_per_run,
            deduplicate_messages=deduplicate_messages,
        )
    return PostgresStepStore(
        pool,
        table='steps',
        media_store=media_store,
        max_snapshots_per_run=max_snapshots_per_run,
        deduplicate_messages=deduplicate_messages,
    )


async def _stored_messages(kind: str, tmp_path: Path, pool: SqlitePool) -> list[str]:
    if kind == 'file':
        return [path.read_text(encoding='utf-8') for path in (tmp_path / 'runs' / 'messages').glob('*.json')]
    if kind == 'sqlite':
        with closing(sqlite3.connect(tmp_path / 'runs.db')) as conn:
            return [row[0] for row in conn.execute('SELECT message FROM snapshot_messages')]
    async with pool.acquire() as connection:
        return [str(row[0]) for row in await connection.fetch('SELECT message FROM steps_messages')]


async def _drop_messages(kind: str, tmp_path: Path, pool: SqlitePool, containing: str) -> None:
    if kind == 'file':
        for path in (tmp_path / 'runs' / 'messages').glob('*.json'):
            if containing in path.read_text(encoding='utf-8'):
                path.unlink()
    elif kind == 'sqlite':
        with closing(sqlite3.connect(tmp_path / 'runs.db')) as conn, conn:
            conn.execute('DELETE FROM snapshot_messages WHERE instr(message, ?) > 0', (containing,))
    else:
        async with pool.acquire() as connection:
            await connection.execute('DELETE FROM steps_messages WHERE instr(message, $1) > 0', containing)


def _conversation(turns: int) -> list[list[ModelMessage]]:
    """The history after each turn; each is a prefix of the next, as across a conversation's runs."""
    histories: list[list[ModelMessage]] = []
    history: list[ModelMessage] = []
    for turn in range(turns):
        history = [
            *history,
            ModelRequest(parts=[UserPromptPart(content=f'question {turn}')]),
            ModelResponse(parts=[TextPart(content=f'answer {turn}')]),
        ]
        histories.append(history)
    return histories


class TestMessageStorage:
    @pytest.mark.parametrize('kind', KINDS)
    async def test_each_message_is_stored_once_across_a_conversation(
        self, kind: str, tmp_path: Path, pool: SqlitePool
    ) -> None:
        store = _store(kind, tmp_path, pool, max_snapshots_per_run=1)
        histories = _conversation(5)
        for turn, history in enumerate(histories):
            # A second, pruned step per run: its messages stay, as later snapshots share them.
            for step in range(2):
                await store.save_snapshot(ContinuableSnapshot(run_id=f'r{turn}', step_index=step, messages=history))

        assert len(await _stored_messages(kind, tmp_path, pool)) == len(histories[-1])
        for turn, history in enumerate(histories):
            snapshot = await store.latest_snapshot(run_id=f'r{turn}')
            assert snapshot is not None
            assert snapshot.messages == history
            assert [s.messages for s in await store.list_snapshots(run_id=f'r{turn}')] == [history]

    @pytest.mark.parametrize('kind', KINDS)
    async def test_a_repeated_message_is_stored_once_and_restored_in_place(
        self, kind: str, tmp_path: Path, pool: SqlitePool
    ) -> None:
        hi = ModelRequest(parts=[UserPromptPart(content='hi', timestamp=datetime(2026, 1, 1, tzinfo=UTC))])
        history: list[ModelMessage] = [
            hi,
            ModelResponse(parts=[TextPart('one')]),
            hi,
            ModelResponse(parts=[TextPart('two')]),
        ]
        store = _store(kind, tmp_path, pool)
        for step in range(2):
            await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=step, messages=history))

        assert len(await _stored_messages(kind, tmp_path, pool)) == 3
        assert [s.messages for s in await store.list_snapshots(run_id='r1')] == [history, history]

    @pytest.mark.parametrize('kind', KINDS)
    async def test_restored_tool_arguments_keep_their_key_order(
        self, kind: str, tmp_path: Path, pool: SqlitePool
    ) -> None:
        store = _store(kind, tmp_path, pool)
        call = ModelResponse(parts=[ToolCallPart('lookup', {'b': 1, 'a': 2}, tool_call_id='t1')])
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=[call]))

        snapshot = await store.latest_snapshot(run_id='r1')
        assert snapshot is not None
        [part] = snapshot.messages[0].parts
        assert isinstance(part, ToolCallPart) and isinstance(part.args, dict)
        assert list(part.args) == ['b', 'a']

    @pytest.mark.parametrize('kind', KINDS)
    async def test_a_resumed_history_is_not_stored_again(self, kind: str, tmp_path: Path, pool: SqlitePool) -> None:
        """Messages restored from the store, media included, hash to the keys they were saved under."""
        store = _store(kind, tmp_path, pool, media_store='auto')
        agent: Agent[None, str] = Agent(TestModel(), capabilities=[StepPersistence(store=store)])
        image = BinaryContent(data=b'\xab' * 70_000, media_type='image/png')

        history: list[ModelMessage] = []
        for turn in range(3):
            await agent.run([f'question {turn}', image], message_history=history)
            history = await continue_run(store, run_id=(await store.list_runs())[-1].run_id)

        assert len(await _stored_messages(kind, tmp_path, pool)) == len(history)

    @pytest.mark.parametrize('kind', KINDS)
    async def test_reads_snapshots_with_inline_messages(self, kind: str, tmp_path: Path, pool: SqlitePool) -> None:
        owner = {'file': 'FileStepStore', 'sqlite': 'SqliteStepStore', 'postgres': 'PostgresStepStore'}[kind]
        with pytest.warns(
            HarnessDeprecationWarning,
            match=rf'`{owner}\(deduplicate_messages=False\)` is deprecated: .* Remove the argument',
        ):
            inline = _store(kind, tmp_path, pool, deduplicate_messages=False)
        first, second = _conversation(2)
        await inline.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=first))
        assert await _stored_messages(kind, tmp_path, pool) == []

        store = _store(kind, tmp_path, pool)
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=1, messages=second))

        for reader in (inline, store):
            assert [s.messages for s in await reader.list_snapshots(run_id='r1')] == [first, second]
            latest = await reader.latest_snapshot(run_id='r1')
            assert latest is not None
            assert latest.messages == second

    @pytest.mark.parametrize('kind', KINDS)
    async def test_a_missing_message_fails_latest_and_is_skipped_by_list(
        self, kind: str, tmp_path: Path, pool: SqlitePool, caplog: pytest.LogCaptureFixture
    ) -> None:
        store = _store(kind, tmp_path, pool)
        first, second = _conversation(2)
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=first))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=1, messages=second))
        await _drop_messages(kind, tmp_path, pool, 'question 1')

        with pytest.raises(ValueError, match=r'references 1 stored message\(s\) that are missing'):
            await store.latest_snapshot(run_id='r1')
        with caplog.at_level(logging.WARNING):
            assert [s.messages for s in await store.list_snapshots(run_id='r1')] == [first]
        assert 'Skipping unparsable snapshot' in caplog.text

    @pytest.mark.parametrize('kind', KINDS)
    async def test_a_suppressed_retry_stores_no_messages(self, kind: str, tmp_path: Path, pool: SqlitePool) -> None:
        store = _store(kind, tmp_path, pool)
        first, second = _conversation(2)
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=first, idempotency_key='k'))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=1, messages=second, idempotency_key='k'))

        assert len(await _stored_messages(kind, tmp_path, pool)) == len(first)
        assert [s.messages for s in await store.list_snapshots(run_id='r1')] == [first]

    @pytest.mark.parametrize('kind', KINDS)
    async def test_agent_runs_continuing_a_conversation(self, kind: str, tmp_path: Path, pool: SqlitePool) -> None:
        store = _store(kind, tmp_path, pool)
        agent: Agent[None, str] = Agent(TestModel(), capabilities=[StepPersistence(store=store)])

        history: list[ModelMessage] = []
        for turn in range(3):
            result = await agent.run(f'question {turn}', message_history=history)
            history = result.all_messages()

        runs = await store.list_runs()
        assert len(runs) == 3
        snapshot = await store.latest_snapshot(run_id=runs[-1].run_id)
        assert snapshot is not None
        assert snapshot.messages == history
        # Each run's snapshots repeat the earlier turns; each message is stored once.
        assert len(await _stored_messages(kind, tmp_path, pool)) == len(history)


class TestSqliteMessageStorage:
    async def test_a_failed_snapshot_write_rolls_back_its_messages(self, tmp_path: Path) -> None:
        db = tmp_path / 'runs.db'
        store = SqliteStepStore(database=db, media_store=None)
        first, second = _conversation(2)
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=first))
        with closing(sqlite3.connect(db, isolation_level=None)) as conn:
            conn.execute(
                'CREATE TRIGGER reject_step_one BEFORE INSERT ON snapshots WHEN NEW.step_index = 1 '
                "BEGIN SELECT RAISE(ABORT, 'rejected'); END"
            )

        with pytest.raises(sqlite3.IntegrityError, match='rejected'):
            await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=1, messages=second))
        with closing(sqlite3.connect(db)) as conn:
            assert conn.execute('SELECT count(*) FROM snapshot_messages').fetchone()[0] == len(first)

    async def test_a_caller_owned_connection_keeps_transaction_control(self, tmp_path: Path) -> None:
        with closing(sqlite3.connect(tmp_path / 'runs.db', check_same_thread=False, isolation_level=None)) as conn:
            store = SqliteStepStore(connection=conn, media_store=None)
            first, second = _conversation(2)
            await store.save_snapshot(
                ContinuableSnapshot(run_id='r1', step_index=0, messages=first, idempotency_key='k')
            )
            await store.save_snapshot(
                ContinuableSnapshot(run_id='r1', step_index=1, messages=second, idempotency_key='k')
            )
            assert not conn.in_transaction
            assert conn.execute('SELECT count(*) FROM snapshot_messages').fetchone()[0] == len(first)
            snapshot = await store.latest_snapshot(run_id='r1')
            assert snapshot is not None
            assert snapshot.messages == first

    async def test_a_long_history_is_read_back_whole(self, tmp_path: Path) -> None:
        """More distinct messages than one query binds as parameters."""
        store = SqliteStepStore(database=tmp_path / 'runs.db', media_store=None)
        history = _conversation(301)[-1]
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=history))

        snapshot = await store.latest_snapshot(run_id='r1')
        assert snapshot is not None
        assert snapshot.messages == history
