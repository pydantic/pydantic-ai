"""Unit tests for `PostgresStepStore`.

Construction runs against `_UnusedPool`, a stub that is never queried. Runs,
events, snapshots, retention, tool effects, and media externalization run
against `SqlitePool`, an in-memory SQLite stand-in for a pool: there is no
in-process Postgres, and these tests must run in every CI job.

The edges only a real server shows are covered by
`pydantic_ai_harness/integration_tests/postgres`, which runs against Postgres.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import anyio
import pytest

import pydantic_ai_harness.media as media_module
import pydantic_ai_harness.step_persistence as step_persistence_module
from pydantic_ai import Agent
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.conversation_search import SnapshotHistorySource
from pydantic_ai_harness.media import DiskMediaStore, PostgresMediaStore, media_uri_for
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    PostgresConnection,
    PostgresPool,
    PostgresStepStore,
    RunRecord,
    StepEvent,
    StepPersistence,
    StepStore,
    ToolEffectRecord,
    ToolEffectStatus,
)
from tests.harness._postgres_sqlite import SqlitePool


class _UnusedPool:
    """Satisfies `PostgresPool` structurally; construction never acquires."""

    def acquire(self) -> AbstractAsyncContextManager[PostgresConnection]:
        raise AssertionError('the pool is not queried')  # pragma: no cover


class TestPostgresStepStoreConstruction:
    @pytest.mark.parametrize('table', ['step-store', 'x; DROP TABLE users', '1steps', 'a' * 41, '', 'Orders'])
    def test_rejects_invalid_table_name(self, table: str) -> None:
        """Uppercase is rejected: Postgres folds unquoted identifiers, so `'Orders'` would alias `'orders'`."""
        with pytest.raises(ValueError, match='invalid table name'):
            PostgresStepStore(_UnusedPool(), table=table)

    def test_accepts_forty_character_prefix(self) -> None:
        table = 'a' * 40
        store = PostgresStepStore(_UnusedPool(), table=table)
        assert store._table == table  # pyright: ignore[reportPrivateUsage]

    def test_rejects_invalid_max_snapshots_per_run(self) -> None:
        with pytest.raises(ValueError, match='must be an int >= 1 or None'):
            PostgresStepStore(_UnusedPool(), max_snapshots_per_run=1.5)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match='must be >= 1 or None'):
            PostgresStepStore(_UnusedPool(), max_snapshots_per_run=0)

    def test_auto_media_store_shares_pool(self) -> None:
        pool = _UnusedPool()
        store = PostgresStepStore(pool, table='steps')
        media_store = store._media_store  # pyright: ignore[reportPrivateUsage]
        assert isinstance(media_store, PostgresMediaStore)
        assert media_store._pool is pool  # pyright: ignore[reportPrivateUsage]
        assert media_store._table == 'steps_media'  # pyright: ignore[reportPrivateUsage]

    def test_media_store_none_disables_media(self) -> None:
        store = PostgresStepStore(_UnusedPool(), media_store=None)
        assert store._media_store is None  # pyright: ignore[reportPrivateUsage]

    def test_custom_media_store_is_kept(self, tmp_path: Path) -> None:
        media_store = DiskMediaStore(tmp_path / 'media')
        store = PostgresStepStore(_UnusedPool(), media_store=media_store)
        assert store._media_store is media_store  # pyright: ignore[reportPrivateUsage]


class TestPostgresStepStoreProtocol:
    def test_satisfies_step_store_protocol(self) -> None:
        assert isinstance(PostgresStepStore(_UnusedPool()), StepStore)

    def test_stub_pool_satisfies_pool_protocol(self) -> None:
        assert isinstance(_UnusedPool(), PostgresPool)


class TestPostgresStepStoreExports:
    def test_exported_from_step_persistence(self) -> None:
        for name in ('PostgresConnection', 'PostgresPool', 'PostgresStepStore'):
            assert name in step_persistence_module.__all__
        assert step_persistence_module.PostgresStepStore is PostgresStepStore

    def test_pool_protocols_are_the_media_ones(self) -> None:
        assert PostgresPool is media_module.PostgresPool
        assert PostgresConnection is media_module.PostgresConnection


@pytest.fixture
def pool() -> Iterator[SqlitePool]:
    sqlite_pool = SqlitePool()
    yield sqlite_pool
    sqlite_pool.close()


def _user_messages(text: str = 'a') -> list[ModelMessage]:
    return [ModelRequest(parts=[UserPromptPart(content=text)])]


class TestPostgresStepStoreRuns:
    async def test_register_and_get_run(self, pool: SqlitePool) -> None:
        """Every `RunRecord` field survives the round trip, microseconds included."""
        store = PostgresStepStore(pool, media_store=None)
        record = RunRecord(
            run_id='r1',
            conversation_id='c1',
            parent_run_id='p1',
            agent_name='agent',
            metadata={'k': 'v'},
            started_at=datetime(2024, 5, 6, 7, 8, 9, 123456, tzinfo=UTC),
            registration_id='reg-1',
        )

        await store.register_run(record)

        assert await store.get_run(run_id='r1') == record
        assert await store.get_run(run_id='nope') is None

    async def test_register_duplicate_run_raises_value_error(self, pool: SqlitePool) -> None:
        """A reused `run_id` is refused and the first record is kept."""
        store = PostgresStepStore(pool, media_store=None)
        await store.register_run(RunRecord(run_id='r1', agent_name='first'))

        with pytest.raises(ValueError, match='is already in the store'):
            await store.register_run(RunRecord(run_id='r1', agent_name='racing-run'))

        fetched = await store.get_run(run_id='r1')
        assert fetched is not None
        assert fetched.agent_name == 'first'
        assert await pool.count_rows('step_persistence_runs') == 1

    async def test_list_runs_filters(self, pool: SqlitePool) -> None:
        """Each filter narrows on its own and both AND-combine."""
        store = PostgresStepStore(pool, media_store=None)
        base = datetime(2024, 1, 1, tzinfo=UTC)
        await store.register_run(
            RunRecord(run_id='r3', conversation_id='b', parent_run_id='p', started_at=base + timedelta(seconds=3))
        )
        await store.register_run(
            RunRecord(run_id='r1', conversation_id='a', parent_run_id='p', started_at=base + timedelta(seconds=1))
        )
        await store.register_run(
            RunRecord(run_id='r2', conversation_id='a', parent_run_id='q', started_at=base + timedelta(seconds=2))
        )

        assert [r.run_id for r in await store.list_runs()] == ['r1', 'r2', 'r3']
        assert [r.run_id for r in await store.list_runs(conversation_id='a')] == ['r1', 'r2']
        assert [r.run_id for r in await store.list_runs(parent_run_id='p')] == ['r1', 'r3']
        assert [r.run_id for r in await store.list_runs(parent_run_id='p', conversation_id='a')] == ['r1']
        assert await store.list_runs(parent_run_id='q', conversation_id='b') == []

    async def test_list_runs_sorts_by_instant_not_iso_string(self, pool: SqlitePool) -> None:
        """A lexicographic sort of the ISO string would put `late` first."""
        store = PostgresStepStore(pool, media_store=None)
        early = datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone(timedelta(hours=5)))  # 2023-12-31T20:00Z
        late = datetime(2024, 1, 1, 0, 30, 0, tzinfo=UTC)
        await store.register_run(RunRecord(run_id='late', started_at=late))
        await store.register_run(RunRecord(run_id='early', started_at=early))

        assert [r.run_id for r in await store.list_runs()] == ['early', 'late']

    async def test_schema_is_created_once_under_concurrent_first_calls(self, pool: SqlitePool) -> None:
        """The second first call waits on the lock and then finds the schema ready."""
        store = PostgresStepStore(pool, media_store=None)

        async with anyio.create_task_group() as tg:
            tg.start_soon(store.register_run, RunRecord(run_id='first'))
            tg.start_soon(store.register_run, RunRecord(run_id='second'))

        assert {r.run_id for r in await store.list_runs()} == {'first', 'second'}
        assert pool.statements.count('SELECT pg_advisory_xact_lock(hashtext($1))') == 1


class TestPostgresStepStoreEvents:
    async def test_append_and_list_events(self, pool: SqlitePool) -> None:
        """Events come back in write order, not `step_index` order, scoped to the run."""
        store = PostgresStepStore(pool, media_store=None)
        started = StepEvent(run_id='r1', kind='run_started', step_index=5, conversation_id='c1', agent_name='agent')
        tool_call = StepEvent(
            run_id='r1', kind='tool_call_started', step_index=0, tool_call_id='t1', tool_name='add', metadata={'k': 'v'}
        )
        failed = StepEvent(run_id='r1', kind='run_failed', step_index=3, parent_run_id='p1', error='boom')

        for event in (started, tool_call, failed):
            await store.append_event(event)
        await store.append_event(StepEvent(run_id='r2', kind='run_started', step_index=0))

        assert await store.list_events(run_id='r1') == [started, tool_call, failed]
        assert await store.list_events(run_id='nope') == []

    async def test_keyed_event_replay_is_suppressed_but_unkeyed_events_append(self, pool: SqlitePool) -> None:
        """A repeated key is written once per run; events without a key always append."""
        store = PostgresStepStore(pool, media_store=None)
        keyed = StepEvent(run_id='r1', kind='run_started', step_index=0, idempotency_key='event:0')
        await store.append_event(keyed)
        await store.append_event(keyed)
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0))
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0))
        await store.append_event(StepEvent(run_id='r2', kind='run_started', step_index=0, idempotency_key='event:0'))

        assert len(await store.list_events(run_id='r1')) == 3
        assert len(await store.list_events(run_id='r2')) == 1


class TestPostgresStepStoreSnapshots:
    async def test_save_and_load_snapshot(self, pool: SqlitePool) -> None:
        """Every `ContinuableSnapshot` field survives the round trip."""
        store = PostgresStepStore(pool, media_store=None)
        snapshot = ContinuableSnapshot(
            run_id='r1',
            step_index=2,
            messages=[
                ModelRequest(parts=[UserPromptPart(content='hello')]),
                ModelResponse(parts=[TextPart(content='hi back')]),
            ],
            conversation_id='c1',
            parent_run_id='p1',
            agent_name='agent',
            timestamp=datetime(2024, 5, 6, 7, 8, 9, 123456, tzinfo=UTC),
        )

        await store.save_snapshot(snapshot)

        assert await store.latest_snapshot(run_id='r1') == snapshot
        assert await store.latest_snapshot(run_id='nope') is None

    async def test_interrupted_snapshots_are_read_only_on_request(self, pool: SqlitePool) -> None:
        """Snapshots list in write order, with `interrupted` ones only when asked for."""
        store = PostgresStepStore(pool, media_store=None)
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=2, messages=_user_messages()))
        await store.save_snapshot(
            ContinuableSnapshot(run_id='r1', step_index=0, messages=_user_messages(), state='interrupted')
        )
        await store.save_snapshot(ContinuableSnapshot(run_id='r2', step_index=9, messages=_user_messages()))

        default = await store.latest_snapshot(run_id='r1')
        assert default is not None and (default.step_index, default.state) == (2, 'complete')
        opted = await store.latest_snapshot(run_id='r1', include_interrupted=True)
        assert opted is not None and (opted.step_index, opted.state) == (0, 'interrupted')
        assert [s.step_index for s in await store.list_snapshots(run_id='r1')] == [2]
        assert [s.step_index for s in await store.list_snapshots(run_id='r1', include_interrupted=True)] == [2, 0]
        assert await store.list_snapshots(run_id='nope') == []

    async def test_keyed_snapshot_replay_is_suppressed(self, pool: SqlitePool) -> None:
        """Two keys at one `step_index` are two snapshots, and replaying either adds nothing."""
        store = PostgresStepStore(pool, media_store=None)
        complete = ContinuableSnapshot(
            run_id='r1', step_index=2, messages=_user_messages(), state='complete', idempotency_key='2:complete'
        )
        interrupted = ContinuableSnapshot(
            run_id='r1', step_index=2, messages=_user_messages(), state='interrupted', idempotency_key='2:interrupted'
        )
        unkeyed = ContinuableSnapshot(run_id='r1', step_index=3, messages=_user_messages())

        for snapshot in (complete, interrupted, complete, interrupted, unkeyed, unkeyed):
            await store.save_snapshot(snapshot)

        snapshots = await store.list_snapshots(run_id='r1', include_interrupted=True)
        assert snapshots == [complete, interrupted, unkeyed, unkeyed]

    async def test_list_snapshots_skips_an_unparsable_row(
        self, pool: SqlitePool, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One damaged row is logged and skipped; `latest_snapshot` on it raises."""
        store = PostgresStepStore(pool, media_store=None)
        good = ContinuableSnapshot(run_id='r1', step_index=0, messages=_user_messages())
        await store.save_snapshot(good)
        async with pool.acquire() as connection, connection.transaction():
            await connection.execute(
                'INSERT INTO step_persistence_snapshots (run_id, step_index, timestamp, messages) '
                'VALUES ($1, $2, $3, $4)',
                'r1',
                'zero',
                datetime.now(UTC).isoformat(),
                '[]',
            )

        with caplog.at_level(logging.WARNING):
            snapshots = await store.list_snapshots(run_id='r1')

        assert snapshots == [good]
        assert 'Skipping unparsable snapshot row for run r1' in caplog.text
        with pytest.raises(ValueError, match='snapshot row has wrong types'):
            await store.latest_snapshot(run_id='r1')

    async def test_store_is_accepted_as_a_search_substrate(self, pool: SqlitePool) -> None:
        store = PostgresStepStore(pool, media_store=None)
        await store.save_snapshot(
            ContinuableSnapshot(run_id='r1', step_index=0, messages=_user_messages('remember this'))
        )

        source = SnapshotHistorySource(store)

        assert [type(m).__name__ for m in await source.run_history(run_id='r1')] == ['ModelRequest']


class TestPostgresStepStoreRetention:
    async def test_unbounded_keeps_every_snapshot(self, pool: SqlitePool) -> None:
        store = PostgresStepStore(pool, media_store=None)
        for step in range(4):
            await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=step, messages=_user_messages()))

        assert [s.step_index for s in await store.list_snapshots(run_id='r1')] == [0, 1, 2, 3]

    async def test_over_the_limit_prunes_to_the_newest_window(self, pool: SqlitePool) -> None:
        """Past the bound only the newest rows remain, and only for the written run."""
        store = PostgresStepStore(pool, media_store=None, max_snapshots_per_run=2)
        await store.save_snapshot(ContinuableSnapshot(run_id='r2', step_index=0, messages=_user_messages()))
        for step in range(5):
            await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=step, messages=_user_messages()))

        assert [s.step_index for s in await store.list_snapshots(run_id='r1')] == [3, 4]
        assert [s.step_index for s in await store.list_snapshots(run_id='r2')] == [0]
        assert await pool.count_rows('step_persistence_snapshots') == 3

    async def test_keeps_newest_complete_below_an_interrupted_tail(self, pool: SqlitePool) -> None:
        """A `complete` snapshot pushed out of the window by `interrupted` writes survives."""
        store = PostgresStepStore(pool, media_store=None, max_snapshots_per_run=2)
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_user_messages()))
        for step in (1, 2, 3):
            await store.save_snapshot(
                ContinuableSnapshot(run_id='r1', step_index=step, messages=_user_messages(), state='interrupted')
            )

        retained = await store.list_snapshots(run_id='r1', include_interrupted=True)
        assert [s.step_index for s in retained] == [0, 2, 3]
        resumable = await store.latest_snapshot(run_id='r1')
        assert resumable is not None and resumable.step_index == 0

    async def test_keyed_writes_are_idempotent_after_pruning(self, pool: SqlitePool) -> None:
        """The key ledger outlives the snapshot row, so a late replay cannot resurrect it."""
        store = PostgresStepStore(pool, media_store=None, max_snapshots_per_run=1)
        older = ContinuableSnapshot(run_id='r1', step_index=1, messages=[], idempotency_key='0:1:complete')
        newer = ContinuableSnapshot(run_id='r1', step_index=2, messages=[], idempotency_key='1:2:complete')

        await store.save_snapshot(older)
        await store.save_snapshot(newer)
        await store.save_snapshot(older)

        assert await store.list_snapshots(run_id='r1') == [newer]
        assert await pool.count_rows('step_persistence_snapshot_keys') == 2


class TestPostgresStepStoreToolEffects:
    async def test_upsert_and_scope(self, pool: SqlitePool) -> None:
        """A second record for one call replaces the first, per run."""
        store = PostgresStepStore(pool, media_store=None)
        started_at = datetime(2024, 5, 6, 7, 8, 9, tzinfo=UTC)
        completed = ToolEffectRecord(
            tool_call_id='t1',
            tool_name='add',
            run_id='r1',
            status='completed',
            started_at=started_at,
            ended_at=started_at + timedelta(seconds=2),
            idempotency_key='effect-1',
            effect_summary='ok',
        )
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='add', run_id='r1', status='started', started_at=started_at)
        )
        await store.record_tool_effect(completed)
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='add', run_id='r2', status='started')
        )

        assert await store.get_tool_effect(run_id='r1', tool_call_id='t1') == completed
        other = await store.get_tool_effect(run_id='r2', tool_call_id='t1')
        assert other is not None and other.status == 'started' and other.ended_at is None
        assert await store.get_tool_effect(run_id='r1', tool_call_id='nope') is None

    async def test_list_unresolved_tool_effects(self, pool: SqlitePool) -> None:
        """Only `started` effects of the asked run are unresolved."""
        store = PostgresStepStore(pool, media_store=None)
        effects: list[tuple[str, str, ToolEffectStatus]] = [
            ('t1', 'r1', 'started'),
            ('t2', 'r1', 'completed'),
            ('t3', 'r1', 'failed'),
            ('t4', 'r2', 'started'),
        ]
        for tool_call_id, run_id, status in effects:
            await store.record_tool_effect(
                ToolEffectRecord(tool_call_id=tool_call_id, tool_name='add', run_id=run_id, status=status)
            )

        unresolved = await store.list_unresolved_tool_effects(run_id='r1')

        assert [r.tool_call_id for r in unresolved] == ['t1']


def _messages_with_media(big_binary: bytes, big_text: str) -> list[ModelMessage]:
    return [
        ModelRequest(
            parts=[
                UserPromptPart(content=[BinaryContent(data=big_binary, media_type='image/png')]),
                ToolReturnPart(tool_name='scrape', content=big_text, tool_call_id='t1'),
            ]
        ),
        ModelResponse(parts=[TextPart(content='done')]),
    ]


def _assert_media_restored(snapshot: ContinuableSnapshot | None, big_binary: bytes, big_text: str) -> None:
    assert snapshot is not None
    request = snapshot.messages[0]
    assert isinstance(request, ModelRequest)
    prompt, tool_return = request.parts
    assert isinstance(prompt, UserPromptPart) and isinstance(prompt.content, list)
    [binary] = prompt.content
    assert isinstance(binary, BinaryContent) and binary.data == big_binary
    assert isinstance(tool_return, ToolReturnPart) and tool_return.content == big_text


class TestPostgresStepStoreMedia:
    async def test_large_parts_are_externalized_and_restored(self, pool: SqlitePool) -> None:
        """The default media store keeps large parts in `{table}_media`, out of the snapshot row."""
        big_binary = b'\xab' * 100_000
        big_text = 'Z' * 100_000
        store = PostgresStepStore(pool, table='steps', media_threshold_bytes=64 * 1024)
        snapshot = ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages_with_media(big_binary, big_text))

        await store.save_snapshot(snapshot)

        assert await pool.count_rows('steps_media') == 2
        async with pool.acquire() as connection:
            stored_length = await connection.fetchval('SELECT length(messages) FROM steps_snapshots')
        assert isinstance(stored_length, int) and stored_length < 64 * 1024
        _assert_media_restored(await store.latest_snapshot(run_id='r1'), big_binary, big_text)
        [listed] = await store.list_snapshots(run_id='r1')
        _assert_media_restored(listed, big_binary, big_text)
        assert await PostgresMediaStore(pool, table='steps_media').get(media_uri_for(big_binary)) == big_binary

    async def test_agent_run_round_trips_through_step_persistence(self, pool: SqlitePool) -> None:
        """An agent run with a large `BinaryContent` prompt persists and restores through the store."""
        big = b'\xab' * 100_000
        store = PostgresStepStore(pool)
        agent: Agent[None, str] = Agent(TestModel(), capabilities=[StepPersistence(store=store, agent_name='vision')])

        await agent.run(['classify this image', BinaryContent(data=big, media_type='image/png')])

        [run] = await store.list_runs()
        assert run.agent_name == 'vision'
        kinds = [event.kind for event in await store.list_events(run_id=run.run_id)]
        assert (kinds[0], kinds[-1]) == ('run_started', 'run_completed')
        assert await pool.count_rows('step_persistence_media') == 1
        snapshot = await store.latest_snapshot(run_id=run.run_id)
        assert snapshot is not None
        request = snapshot.messages[0]
        assert isinstance(request, ModelRequest)
        prompt = request.parts[0]
        assert isinstance(prompt, UserPromptPart) and isinstance(prompt.content, list)
        assert [p.data for p in prompt.content if isinstance(p, BinaryContent)] == [big]
