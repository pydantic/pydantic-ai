"""A spend counter shared across processes, backed by PostgreSQL.

Takes a caller-owned asyncpg-compatible pool rather than a driver dependency: the store
depends only on the `PostgresPool` protocol from `pydantic_ai_harness.media`, which an
`asyncpg` pool satisfies, so installing the harness pulls in nothing extra.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import TYPE_CHECKING

import anyio

from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.spend._snapshot import Spent, from_nanos, to_nanos
from pydantic_ai_harness.spend._store import DEFAULT_DEDUP_RETAIN, SpendEntry

if TYPE_CHECKING:
    from pydantic_ai_harness.media import PostgresConnection, PostgresPool

_logger = logging.getLogger(__name__)

# Postgres truncates identifiers past 63 bytes, which would let two distinct prefixes
# collide. The longest derived name is the index `{table}_counters_expiry` (47 + 16 = 63).
# Lowercase only: the interpolated identifiers are unquoted, so Postgres folds them to
# lowercase and `'Spend'` would share tables with `'spend'`.
_TABLE_RE = re.compile(r'[a-z_][a-z0-9_]{0,46}')


def _int(value: object) -> int:
    assert isinstance(value, int)
    return value


class PostgresSpendStore:
    """Spend counters in PostgreSQL, so every worker enforces one budget.

    One row per window in `{table}_counters`, holding the four counters as integers, and one
    row per applied `SpendEntry.token` and window in `{table}_dedup`. Both tables are created
    on first use. Takes a caller-owned asyncpg-compatible pool; the store never closes it.

    Horizons are measured on the server's clock, `now()`, the way Redis measures expiry on its
    own, so every worker agrees on when a window rolls over and a worker whose clock runs ahead
    cannot restart a counter another worker is still adding to.

    A read and the increment that follows it are separate round trips, so concurrent runs
    can each observe a budget as unexhausted and push past it together, as with
    [`RedisSpendStore`][pydantic_ai_harness.spend.RedisSpendStore]; see the README on what the
    gate does and does not guarantee.
    """

    def __init__(
        self,
        pool: PostgresPool,
        *,
        table: str = 'spend',
        dedup_retain: timedelta | None = DEFAULT_DEDUP_RETAIN,
        sweep_every: int = 256,
    ) -> None:
        """Validate the configuration; nothing touches the database until the first call.

        Args:
            pool: An asyncpg-compatible pool. The application owns its lifecycle.
            table: Prefix for the store's tables, `{table}_counters` and `{table}_dedup`.
                Lowercase letters, digits and underscores, at most 47 characters, so every
                derived name stays inside Postgres's 63-byte identifier limit.
            dedup_retain: How long an applied `SpendEntry.token` is remembered, or `None` to
                apply every entry. This is the window a replay is recognised in, not the
                counter's lifetime: a response replayed later than this is counted again,
                because the marker it would have matched has expired. Each remembered token is
                one small row per response per window until it is swept. Must be positive: a
                zero horizon writes a marker that has already expired, so a second entry of the
                same response in one call would take it over and count again.
            sweep_every: Entries written by this instance between deletions of expired rows.
                A day window produces a new key each day, so yesterday's row is never written
                again and has to be deleted by a sweep rather than on its next write.
        """
        if not _TABLE_RE.fullmatch(table):
            raise UserError(
                f'PostgresSpendStore.table must be lowercase letters, digits and underscores, start with a letter '
                f'or underscore, and be at most 47 characters; got {table!r}.'
            )
        if dedup_retain is not None and dedup_retain <= timedelta(0):
            raise UserError(
                f'PostgresSpendStore.dedup_retain must be positive, or `None` to apply every entry; '
                f'got {dedup_retain!r}.'
            )
        self._pool = pool
        self._counters = f'{table}_counters'
        self._dedup = f'{table}_dedup'
        self._dedup_retain = dedup_retain
        self._sweep_every = sweep_every
        self._schema_ready = False
        self._schema_lock = anyio.Lock()
        self._writes_since_sweep = 0

    async def get_many(self, keys: Sequence[str]) -> Mapping[str, Spent]:
        """What each key has accumulated. An absent or expired row reads as zero."""
        await self._ensure_schema()
        async with self._pool.acquire() as connection:
            return await self._totals(connection, keys)

    async def add_many(self, entries: Sequence[SpendEntry]) -> Mapping[str, Spent]:
        """Apply every entry in one transaction and return each key's new total.

        Every window of the response commits together or none does, and a failure part-way,
        such as a counter passing the signed 64-bit range, rolls back the windows before it too.
        A connection that drops after the commit was sent leaves the outcome unknown rather than
        untried, the same as `RedisSpendStore`, which is why `SpendLimits` does not retry.

        An entry carrying a `SpendEntry.token` claims a marker row first, and is skipped when a
        live marker for that key and token already exists. Claiming it in the same transaction as
        the increment is what makes the order safe: a failed increment rolls the claim back with
        it, so the retry that could have recorded the response is not skipped as a replay. A
        concurrent claim of the same marker waits on the first transaction and, once that commits,
        finds the marker and skips. That wait is `READ COMMITTED` behavior, so the transaction
        sets that level rather than inheriting a stricter default, under which the second writer
        would fail with a serialization error instead.

        Entries are applied in key order so two transactions taking the same windows lock their
        rows in the same order rather than deadlocking. Entries sharing a key keep their relative
        order.
        """
        if not entries:
            return {}
        if any(entry.ttl is not None and entry.ttl <= timedelta(0) for entry in entries):
            # `Budget.retain` is refused unless positive, so only a direct caller reaches this. A row
            # written already expired would be restarted by the next entry for its key in the call.
            raise UserError('`SpendEntry.ttl` must be positive, or `None` to keep the key, for `PostgresSpendStore`.')
        await self._ensure_schema()
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute('SET TRANSACTION ISOLATION LEVEL READ COMMITTED')
                for entry in sorted(entries, key=lambda entry: entry.key):
                    if await self._claim(connection, entry):
                        await self._increment(connection, entry)
                totals = await self._totals(connection, [entry.key for entry in entries])
            self._writes_since_sweep += len(entries)
            if self._writes_since_sweep >= self._sweep_every:
                self._writes_since_sweep = 0
                await self._sweep(connection)
        return totals

    async def _claim(self, connection: PostgresConnection, entry: SpendEntry) -> bool:
        """Record that this entry's response reached its window, or report that it already had.

        A marker past its horizon that the sweep has not reached yet is taken over rather than
        honoured, so expiry does not depend on when the sweep runs.
        """
        horizon = self._marker_horizon(entry)
        if entry.token is None or horizon is None:
            return True
        claimed = await connection.fetchval(
            f'INSERT INTO {self._dedup} (key, token, expires_at) VALUES ($1, $2, now() + $3::interval) '
            f'ON CONFLICT (key, token) DO UPDATE SET expires_at = EXCLUDED.expires_at '
            f'WHERE {self._dedup}.expires_at <= now() RETURNING 1',
            entry.key,
            entry.token,
            horizon,
        )
        return claimed is not None

    async def _increment(self, connection: PostgresConnection, entry: SpendEntry) -> None:
        """Add the entry to its window, starting from zero if the window's row has expired.

        `ttl=None` clears an existing horizon rather than keeping it, so a budget moved from a
        finite `retain` to `'forever'` stops expiring on the old schedule, as with the other stores.
        """
        table = self._counters
        increments = ', '.join(
            f'{column} = CASE WHEN {table}.expires_at <= now() THEN 0 ELSE {table}.{column} END + EXCLUDED.{column}'
            for column in ('usd_nanos', 'tokens', 'requests', 'unpriced')
        )
        await connection.execute(
            f'INSERT INTO {table} (key, usd_nanos, tokens, requests, unpriced, expires_at) '
            f'VALUES ($1, $2, $3, $4, $5, now() + $6::interval) ON CONFLICT (key) DO UPDATE SET {increments}, '
            'expires_at = EXCLUDED.expires_at',
            entry.key,
            to_nanos(entry.usd),
            entry.tokens,
            entry.requests,
            entry.unpriced,
            entry.ttl,
        )

    async def _totals(self, connection: PostgresConnection, keys: Sequence[str]) -> dict[str, Spent]:
        rows = await connection.fetch(
            f'SELECT key, usd_nanos, tokens, requests, unpriced FROM {self._counters} '
            'WHERE key = ANY($1::text[]) AND (expires_at IS NULL OR expires_at > now())',
            list(keys),
        )
        found = {
            str(row[0]): Spent(
                usd=from_nanos(_int(row[1])),
                tokens=_int(row[2]),
                requests=_int(row[3]),
                unpriced_requests=_int(row[4]),
            )
            for row in rows
        }
        return {key: found.get(key, Spent()) for key in keys}

    def _marker_horizon(self, entry: SpendEntry) -> timedelta | None:
        """How long this entry's marker is held, never past the counter it guards.

        What the horizon does and does not promise, and which way it errs, is settled in
        `RedisSpendStore._marker_seconds`.
        """
        if self._dedup_retain is None:
            return None
        return self._dedup_retain if entry.ttl is None else min(self._dedup_retain, entry.ttl)

    async def _sweep(self, connection: PostgresConnection) -> None:
        """Delete every window that has rolled over and every marker past its horizon.

        Runs after the write's transaction has committed, so a sweep cannot hold back a write,
        and a sweep that fails is logged rather than raised: the response is already counted,
        and reporting that write as failed would end the run over housekeeping. The next sweep
        is due `sweep_every` entries later. `SKIP LOCKED` leaves a row another transaction is
        renewing to that transaction instead of waiting on it, so a sweep cannot deadlock with a
        concurrent `add_many`, and the outer condition is checked again on the row it locked.
        """
        try:
            await connection.execute(
                f'DELETE FROM {self._counters} WHERE expires_at <= now() AND key IN '
                f'(SELECT key FROM {self._counters} WHERE expires_at <= now() FOR UPDATE SKIP LOCKED)'
            )
            await connection.execute(
                f'DELETE FROM {self._dedup} WHERE expires_at <= now() AND (key, token) IN '
                f'(SELECT key, token FROM {self._dedup} WHERE expires_at <= now() FOR UPDATE SKIP LOCKED)'
            )
        except Exception:
            _logger.warning('PostgresSpendStore could not delete expired rows from %r', self._counters, exc_info=True)

    async def _ensure_schema(self) -> None:
        """Create the tables once, serialised in-process and across processes.

        `CREATE TABLE IF NOT EXISTS` is not race-free in Postgres -- concurrent callers can
        collide on `pg_type`'s unique index -- so the statements run under an advisory lock
        keyed on the counters table name.
        """
        if self._schema_ready:
            return
        async with self._schema_lock:
            if self._schema_ready:
                return
            async with self._pool.acquire() as connection, connection.transaction():
                await connection.fetchval('SELECT pg_advisory_xact_lock(hashtext($1))', self._counters)
                await connection.execute(
                    f'CREATE TABLE IF NOT EXISTS {self._counters} ('
                    'key TEXT PRIMARY KEY, usd_nanos BIGINT NOT NULL, tokens BIGINT NOT NULL, '
                    'requests BIGINT NOT NULL, unpriced BIGINT NOT NULL, expires_at TIMESTAMPTZ)'
                )
                await connection.execute(
                    f'CREATE INDEX IF NOT EXISTS {self._counters}_expiry ON {self._counters} (expires_at) '
                    'WHERE expires_at IS NOT NULL'
                )
                await connection.execute(
                    f'CREATE TABLE IF NOT EXISTS {self._dedup} ('
                    'key TEXT NOT NULL, token TEXT NOT NULL, expires_at TIMESTAMPTZ NOT NULL, '
                    'PRIMARY KEY (key, token))'
                )
                await connection.execute(
                    f'CREATE INDEX IF NOT EXISTS {self._dedup}_expiry ON {self._dedup} (expires_at)'
                )
            self._schema_ready = True
