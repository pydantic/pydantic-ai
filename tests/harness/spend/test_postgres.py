"""`PostgresSpendStore` against the in-memory SQLite stand-in for an asyncpg pool.

The stand-in runs the store's SQL for real after rewriting the few Postgres-only spellings, so
the increments, the dedup markers, expiry and the sweep are exercised end to end. What only a
server shows -- concurrent writers, row locks, `BIGINT` overflow, identifier folding -- is in
`src/pydantic_ai_harness/integration_tests/postgres`.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Iterator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import timedelta
from decimal import Decimal

import anyio
import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness import SpendLimits, spend
from pydantic_ai_harness.media import PostgresConnection
from pydantic_ai_harness.spend import BatchSpendStore, Budget, PostgresSpendStore, SpendEntry, Spent
from tests.harness._postgres_sqlite import SqlitePool


class _InjectedFailure(Exception):
    pass


class _Connection:
    """Delegates to the stand-in, recording each statement with its arguments.

    `fail_on` raises before a statement that starts with it runs, which is how a test makes the
    database refuse one write in the middle of a transaction.
    """

    def __init__(self, inner: PostgresConnection, pool: _RecordingPool) -> None:
        self._inner = inner
        self._pool = pool

    def transaction(self) -> AbstractAsyncContextManager[object]:
        return self._inner.transaction()

    def _record(self, query: str, args: tuple[object, ...]) -> None:
        self._pool.calls.append((query, args))
        if self._pool.fail_on is not None and query.startswith(self._pool.fail_on):
            self._pool.fail_on = None
            raise _InjectedFailure

    async def execute(self, query: str, *args: object) -> object:
        self._record(query, args)
        return await self._inner.execute(query, *args)

    async def fetchval(self, query: str, *args: object) -> object:
        self._record(query, args)
        return await self._inner.fetchval(query, *args)

    async def fetchrow(self, query: str, *args: object) -> Sequence[object] | None:
        raise NotImplementedError

    async def fetch(self, query: str, *args: object) -> Sequence[Sequence[object]]:
        self._record(query, args)
        return await self._inner.fetch(query, *args)


class _RecordingPool:
    def __init__(self, inner: SqlitePool) -> None:
        self.inner = inner
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.fail_on: str | None = None

    @asynccontextmanager
    async def _acquire(self) -> AsyncGenerator[PostgresConnection]:
        async with self.inner.acquire() as connection:
            yield _Connection(connection, self)

    def acquire(self) -> AbstractAsyncContextManager[PostgresConnection]:
        return self._acquire()


@pytest.fixture
def sqlite() -> Iterator[SqlitePool]:
    pool = SqlitePool()
    yield pool
    pool.close()


@pytest.fixture
def pool(sqlite: SqlitePool) -> _RecordingPool:
    return _RecordingPool(sqlite)


class TestPostgresSpendStore:
    def test_it_is_a_batch_store_exported_from_the_package(self, pool: _RecordingPool):
        assert isinstance(PostgresSpendStore(pool), BatchSpendStore)
        assert 'PostgresSpendStore' in spend.__all__

    async def test_a_key_never_written_reads_as_zero(self, pool: _RecordingPool):
        assert await PostgresSpendStore(pool).get_many(['k']) == {'k': Spent()}

    async def test_every_window_of_a_response_lands_together_and_exactly(self, pool: _RecordingPool):
        store = PostgresSpendStore(pool)

        totals = await store.add_many(
            [
                SpendEntry(key='day', usd=Decimal('0.000123456'), tokens=7, requests=1, unpriced=1),
                SpendEntry(key='month', usd=Decimal('0.000123456'), tokens=7, requests=1, unpriced=1),
            ]
        )

        spent = Spent(usd=Decimal('0.000123456'), tokens=7, requests=1, unpriced_requests=1)
        assert totals == {'day': spent, 'month': spent}
        assert await store.get_many(['day', 'month']) == totals

    async def test_repeated_adds_do_not_drift(self, pool: _RecordingPool):
        """Integer billionths, as `RedisSpendStore` holds them, so a sub-cent price accumulates exactly."""
        store = PostgresSpendStore(pool)
        price = Decimal('0.000000675')
        for _ in range(100):
            await store.add_many([SpendEntry(key='k', usd=price, requests=1)])

        assert (await store.get_many(['k']))['k'] == Spent(usd=price * 100, requests=100)

    async def test_entries_sharing_a_key_collapse_to_the_running_total(self, pool: _RecordingPool):
        totals = await PostgresSpendStore(pool).add_many(
            [SpendEntry(key='k', usd=Decimal('1'), requests=1), SpendEntry(key='k', usd=Decimal('2'), requests=1)]
        )

        assert totals == {'k': Spent(usd=Decimal('3'), requests=2)}

    async def test_windows_are_written_in_key_order(self, pool: _RecordingPool):
        """Two transactions taking the same rows in the same order wait on each other instead of deadlocking."""
        await PostgresSpendStore(pool).add_many(
            [SpendEntry(key='month', requests=1, token='r1'), SpendEntry(key='day', requests=1, token='r1')]
        )

        written = [args[1] for query, args in pool.calls if query.startswith('INSERT INTO spend_counters')]
        assert written == [b'day', b'month']

    async def test_a_key_and_token_of_any_length_are_stored(self, pool: _RecordingPool):
        """Indexed by digest: a B-tree refuses an entry past about 2.7 kB, which a caller's id can exceed."""
        store = PostgresSpendStore(pool)
        entry = SpendEntry(key='b|conversation|*|' + 'x' * 10_000, requests=1, token='t' * 10_000)

        await store.add_many([entry])

        assert await store.add_many([entry]) == {entry.key: Spent(requests=1)}
        indexed = [args[0] for query, args in pool.calls if query.startswith('INSERT INTO')]
        assert {len(str(value)) for value in indexed} == {64}

    async def test_a_separator_inside_a_key_or_token_does_not_merge_two_markers(self, pool: _RecordingPool):
        """`('a|b', 'c')` and `('a', 'b|c')` are two responses on two windows, not a replay."""
        store = PostgresSpendStore(pool)

        await store.add_many([SpendEntry(key='a|b', requests=1, token='c')])
        totals = await store.add_many([SpendEntry(key='a', requests=1, token='b|c')])

        assert totals == {'a': Spent(requests=1)}

    async def test_nothing_to_apply_is_not_a_round_trip(self, pool: _RecordingPool):
        assert await PostgresSpendStore(pool).add_many([]) == {}
        assert pool.calls == []

    async def test_a_failing_entry_leaves_the_whole_response_unapplied(self, pool: _RecordingPool):
        """One transaction: the window written before the failure rolls back with it."""
        store = PostgresSpendStore(pool)
        await store.get_many(['day'])
        pool.calls.clear()

        pool.fail_on = 'INSERT INTO spend_counters (key'
        with pytest.raises(_InjectedFailure):
            await store.add_many([SpendEntry(key='day', requests=1), SpendEntry(key='month', requests=1)])

        assert await store.get_many(['day', 'month']) == {'day': Spent(), 'month': Spent()}

    async def test_a_call_that_fails_does_not_consume_the_token(self, pool: _RecordingPool):
        """The marker rolls back with the increment, so the retry is applied rather than skipped."""
        store = PostgresSpendStore(pool)
        entry = SpendEntry(key='k', requests=1, token='resp-1')

        pool.fail_on = 'INSERT INTO spend_counters (key'
        with pytest.raises(_InjectedFailure):
            await store.add_many([entry])

        assert await store.add_many([entry]) == {'k': Spent(requests=1)}

    async def test_a_repeated_token_is_applied_once(self, pool: _RecordingPool):
        store = PostgresSpendStore(pool)
        entry = SpendEntry(key='k', usd=Decimal('1'), tokens=5, requests=1, token='resp-1')

        first = await store.add_many([entry])
        second = await store.add_many([entry])

        assert first == second == {'k': Spent(usd=Decimal('1'), tokens=5, requests=1)}

    async def test_a_repeat_is_recognised_by_another_store_on_the_same_tables(self, pool: _RecordingPool):
        """What a shared store is for: recovery on a replacement worker finds the first worker's marker."""
        entry = SpendEntry(key='k', requests=1, token='resp-1')
        await PostgresSpendStore(pool).add_many([entry])

        assert await PostgresSpendStore(pool).add_many([entry]) == {'k': Spent(requests=1)}

    async def test_one_token_twice_in_a_call_is_applied_once(self, pool: _RecordingPool):
        entry = SpendEntry(key='k', usd=Decimal('1'), requests=1, token='resp-1')

        assert await PostgresSpendStore(pool).add_many([entry, entry]) == {'k': Spent(usd=Decimal('1'), requests=1)}

    async def test_a_different_token_is_applied(self, pool: _RecordingPool):
        store = PostgresSpendStore(pool)

        await store.add_many([SpendEntry(key='k', requests=1, token='resp-1')])
        totals = await store.add_many([SpendEntry(key='k', requests=1, token='resp-2')])

        assert totals == {'k': Spent(requests=2)}

    async def test_a_token_is_scoped_to_its_window(self, pool: _RecordingPool):
        store = PostgresSpendStore(pool)

        await store.add_many([SpendEntry(key='day', requests=1, token='resp-1')])
        totals = await store.add_many([SpendEntry(key='month', requests=1, token='resp-1')])

        assert totals == {'month': Spent(requests=1)}

    async def test_an_entry_with_no_token_is_always_applied(self, pool: _RecordingPool):
        """A reconciler posting the same correction twice means it twice."""
        store = PostgresSpendStore(pool)
        entry = SpendEntry(key='k', usd=Decimal('-1'))

        await store.add_many([entry])

        assert await store.add_many([entry]) == {'k': Spent(usd=Decimal('-2'))}

    async def test_dedup_can_be_turned_off(self, sqlite: SqlitePool, pool: _RecordingPool):
        store = PostgresSpendStore(pool, dedup_retain=None)
        entry = SpendEntry(key='k', requests=1, token='resp-1')

        await store.add_many([entry])

        assert await store.add_many([entry]) == {'k': Spent(requests=2)}
        assert await sqlite.count_rows('spend_dedup') == 0

    async def test_a_token_past_its_horizon_is_forgotten(self, sqlite: SqlitePool, pool: _RecordingPool):
        """`dedup_retain` is the window a replay is recognised in, not the counter's lifetime."""
        store = PostgresSpendStore(pool, dedup_retain=timedelta(hours=1))
        entry = SpendEntry(key='k', requests=1, token='resp-1')
        await store.add_many([entry])

        sqlite.now += timedelta(hours=2)

        assert await store.add_many([entry]) == {'k': Spent(requests=2)}

    async def test_a_token_is_remembered_no_longer_than_its_counter(self, sqlite: SqlitePool, pool: _RecordingPool):
        """A marker outliving its window would skip a replay against a counter that rolled over."""
        store = PostgresSpendStore(pool, dedup_retain=timedelta(hours=24))
        entry = SpendEntry(key='k', requests=1, ttl=timedelta(hours=1), token='resp-1')
        await store.add_many([entry])

        sqlite.now += timedelta(hours=2)

        assert await store.add_many([entry]) == {'k': Spent(requests=1)}

    @pytest.mark.parametrize('retain', [timedelta(0), timedelta(seconds=-1)])
    def test_a_dedup_horizon_that_holds_nothing_is_refused(self, pool: _RecordingPool, retain: timedelta):
        """A marker written already expired would let the same response count twice in one call."""
        with pytest.raises(UserError, match='dedup_retain` must be positive'):
            PostgresSpendStore(pool, dedup_retain=retain)

    async def test_an_entry_horizon_that_holds_nothing_is_refused_before_any_write(self, pool: _RecordingPool):
        """A row written already expired would be restarted by the next entry for its key."""
        entries = [SpendEntry(key='k', requests=1), SpendEntry(key='k', requests=1, ttl=timedelta(0))]

        with pytest.raises(UserError, match='ttl` must be positive'):
            await PostgresSpendStore(pool).add_many(entries)

        assert pool.calls == []

    async def test_a_window_past_its_horizon_reads_as_zero_and_restarts(self, sqlite: SqlitePool, pool: _RecordingPool):
        store = PostgresSpendStore(pool)
        await store.add_many([SpendEntry(key='k', usd=Decimal('5'), requests=1, ttl=timedelta(hours=1))])

        sqlite.now += timedelta(hours=1)

        assert await store.get_many(['k']) == {'k': Spent()}
        totals = await store.add_many([SpendEntry(key='k', usd=Decimal('1'), requests=1, ttl=timedelta(hours=1))])
        assert totals == {'k': Spent(usd=Decimal('1'), requests=1)}

    async def test_a_write_without_a_horizon_clears_the_previous_one(self, sqlite: SqlitePool, pool: _RecordingPool):
        """A budget moved to `retain='forever'` stops expiring on the schedule it had before."""
        store = PostgresSpendStore(pool)
        await store.add_many([SpendEntry(key='k', requests=1, ttl=timedelta(hours=1))])
        await store.add_many([SpendEntry(key='k', requests=1)])

        sqlite.now += timedelta(days=30)

        assert await store.get_many(['k']) == {'k': Spent(requests=2)}

    async def test_the_sweep_deletes_rolled_over_windows_and_markers(self, sqlite: SqlitePool, pool: _RecordingPool):
        """A day window's row is never written again once the day is over, so only a sweep removes it."""
        store = PostgresSpendStore(pool, sweep_every=3)
        await store.add_many(
            [
                SpendEntry(key='yesterday', requests=1, ttl=timedelta(hours=1), token='r1'),
                SpendEntry(key='total', requests=1, token='r1'),
            ]
        )
        sqlite.now += timedelta(hours=2)

        await store.add_many([SpendEntry(key='today', requests=1, ttl=timedelta(hours=1))])

        rows = sqlite.db.execute('SELECT key FROM spend_counters ORDER BY key').fetchall()
        assert [row[0] for row in rows] == [b'today', b'total']
        assert await sqlite.count_rows('spend_dedup') == 0

    async def test_a_failed_sweep_is_logged_and_the_committed_write_still_returns(
        self, pool: _RecordingPool, caplog: pytest.LogCaptureFixture
    ):
        """The response is already counted; failing the call over housekeeping would end the run."""
        store = PostgresSpendStore(pool, sweep_every=1)
        pool.fail_on = 'DELETE'

        with caplog.at_level(logging.WARNING, logger='pydantic_ai_harness.spend._postgres'):
            totals = await store.add_many([SpendEntry(key='k', requests=1)])

        assert totals == {'k': Spent(requests=1)}
        assert 'could not delete expired rows' in caplog.text

    async def test_the_sweep_is_amortised(self, pool: _RecordingPool):
        store = PostgresSpendStore(pool, sweep_every=3)

        await store.add_many([SpendEntry(key='a', requests=1), SpendEntry(key='b', requests=1)])
        assert not any(query.startswith('DELETE') for query, _ in pool.calls)

        await store.add_many([SpendEntry(key='a', requests=1)])
        assert sum(query.startswith('DELETE') for query, _ in pool.calls) == 2

    async def test_the_table_prefix_names_both_tables(self, sqlite: SqlitePool, pool: _RecordingPool):
        await PostgresSpendStore(pool, table='acme_spend').add_many([SpendEntry(key='k', requests=1, token='r1')])

        assert await sqlite.count_rows('acme_spend_counters') == 1
        assert await sqlite.count_rows('acme_spend_dedup') == 1

    @pytest.mark.parametrize('table', ['Spend', '1spend', 'spend; DROP TABLE x', '', 'a' * 48])
    def test_a_prefix_that_is_not_one_short_lowercase_identifier_is_refused(self, pool: _RecordingPool, table: str):
        with pytest.raises(UserError, match='must be lowercase letters, digits and underscores'):
            PostgresSpendStore(pool, table=table)

    async def test_the_schema_is_created_once_under_concurrency(self, pool: _RecordingPool):
        store = PostgresSpendStore(pool)

        async with anyio.create_task_group() as group:
            for key in ('a', 'b', 'c'):
                group.start_soon(store.get_many, [key])

        assert sum(query.startswith('CREATE TABLE') for query, _ in pool.calls) == 2

    async def test_a_store_on_existing_tables_issues_no_ddl(self, pool: _RecordingPool):
        """A role granted only reads and writes on the tables can use them: there is nothing to create."""
        await PostgresSpendStore(pool).get_many(['k'])
        pool.calls.clear()

        await PostgresSpendStore(pool).add_many([SpendEntry(key='k', requests=1)])

        assert not any(query.startswith(('CREATE', 'SELECT pg_advisory')) for query, _ in pool.calls)

    async def test_spend_limits_enforces_a_budget_through_it(self, pool: _RecordingPool):
        store = PostgresSpendStore(pool)
        limits = SpendLimits[None](budgets=[Budget(tokens=1, window='day')], store=store)
        agent = Agent(TestModel(), deps_type=type(None), capabilities=[limits])

        await agent.run('hello')

        assert (await limits.status())[0].exhausted
