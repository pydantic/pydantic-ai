"""A spend counter shared across processes, backed by PostgreSQL.

Takes a caller-owned asyncpg-compatible pool rather than a driver dependency: the store
depends only on the `PostgresPool` protocol from `pydantic_ai_harness.media`, which an
`asyncpg` pool satisfies, so installing the harness pulls in nothing extra.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import TYPE_CHECKING

import anyio

from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.spend._budget import delimited
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


def _digest(text: str) -> str:
    """The SHA-256 hex digest the tables are keyed by.

    A budget key can carry a caller-supplied conversation or run id of any length, and a
    B-tree index refuses an entry past about 2.7 kB, so indexing the text itself would let an
    oversized id fail the write after the model had already been paid for. A digest is a
    fixed 64 characters whatever it stands for.
    """
    return hashlib.sha256(text.encode()).hexdigest()


class PostgresSpendStore:
    """Spend counters in PostgreSQL, so every worker enforces one budget.

    One row per window in `{table}_counters`, holding the four counters as integers, and one
    row per applied `SpendEntry.token` and window in `{table}_dedup`, both keyed by a SHA-256
    digest so a key or token of any length fits the index. Both tables are created on first
    use. Takes a caller-owned asyncpg-compatible pool; the store never closes it.

    Horizons are measured on the server's clock, the way Redis measures expiry on its
    own, so every worker agrees on when a window rolls over and a worker whose clock runs ahead
    cannot restart a counter another worker is still adding to.

    A read and the increment that follows it are separate round trips, so concurrent runs
    can each observe a budget as unexhausted and push past it together, as with
    [`RedisSpendStore`][pydantic_ai_harness.spend.RedisSpendStore]; see "What the gate guarantees"
    in the Spend docs.
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
                f'`PostgresSpendStore.table` must be lowercase letters, digits and underscores, start with a letter '
                f'or underscore, and be at most 47 characters; got {table!r}.'
            )
        if dedup_retain is not None and dedup_retain <= timedelta(0):
            raise UserError(
                f'`PostgresSpendStore.dedup_retain` must be positive, or `None` to apply every entry; '
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

        Every marker is claimed before any counter is written, markers in key and token order and
        counters in key order, so two transactions lock rows in one global order and wait on each
        other rather than deadlocking. Claiming a marker after a counter instead would let a
        transaction hold the counter while it waits on another's marker for the same key, while
        that one waits on the counter. Entries sharing a key are applied in their given order.
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
                by_marker = sorted(range(len(entries)), key=lambda i: (entries[i].key, entries[i].token or ''))
                applied = [i for i in by_marker if await self._claim(connection, entries[i])]
                for entry in sorted((entries[i] for i in sorted(applied)), key=lambda entry: entry.key):
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
        honoured, so expiry does not depend on when the sweep runs. Horizons written here and in
        `_increment` are measured from `clock_timestamp()`, the moment the row is written, rather
        than `now()`, the transaction's start: a write that waited on another transaction's lock
        for longer than a short `ttl` would otherwise commit a row that had already expired.
        """
        horizon = self._marker_horizon(entry)
        if entry.token is None or horizon is None:
            return True
        claimed = await connection.fetchval(
            f'INSERT INTO {self._dedup} (marker_sha256, expires_at) VALUES ($1, clock_timestamp() + $2::interval) '
            f'ON CONFLICT (marker_sha256) DO UPDATE SET expires_at = clock_timestamp() + $2::interval '
            f'WHERE {self._dedup}.expires_at <= clock_timestamp() RETURNING 1',
            # Length-prefixed, as `RedisSpendStore` names its markers, so `('a|b', 'c')` and
            # `('a', 'b|c')` stay two markers.
            _digest(delimited(entry.key, entry.token)),
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
            f'{column} = CASE WHEN {table}.expires_at <= clock_timestamp() THEN 0 ELSE {table}.{column} END '
            f'+ EXCLUDED.{column}'
            for column in ('usd_nanos', 'tokens', 'requests', 'unpriced')
        )
        await connection.execute(
            f'INSERT INTO {table} (key_sha256, key, usd_nanos, tokens, requests, unpriced, expires_at) '
            f'VALUES ($1, $2, $3, $4, $5, $6, clock_timestamp() + $7::interval) ON CONFLICT (key_sha256) '
            f'DO UPDATE SET {increments}, expires_at = clock_timestamp() + $7::interval',
            _digest(entry.key),
            # Bytes, so a key carrying a NUL, which `TEXT` refuses, is stored rather than failing the write.
            entry.key.encode(),
            to_nanos(entry.usd),
            entry.tokens,
            entry.requests,
            entry.unpriced,
            entry.ttl,
        )

    async def _totals(self, connection: PostgresConnection, keys: Sequence[str]) -> dict[str, Spent]:
        digests = {_digest(key): key for key in keys}
        rows = await connection.fetch(
            f'SELECT key_sha256, usd_nanos, tokens, requests, unpriced FROM {self._counters} '
            'WHERE key_sha256 = ANY($1::text[]) AND (expires_at IS NULL OR expires_at > now())',
            list(digests),
        )
        found = {
            digests[str(row[0])]: Spent(
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
                f'DELETE FROM {self._counters} WHERE expires_at <= now() AND key_sha256 IN '
                f'(SELECT key_sha256 FROM {self._counters} WHERE expires_at <= now() FOR UPDATE SKIP LOCKED)'
            )
            await connection.execute(
                f'DELETE FROM {self._dedup} WHERE expires_at <= now() AND marker_sha256 IN '
                f'(SELECT marker_sha256 FROM {self._dedup} WHERE expires_at <= now() FOR UPDATE SKIP LOCKED)'
            )
        except Exception:
            _logger.warning('`PostgresSpendStore` could not delete expired rows from %r', self._counters, exc_info=True)

    async def _ensure_schema(self) -> None:
        """Create the tables once, serialised in-process and across processes.

        Skipped when both tables already exist, so a role granted only reads and writes on them
        can use the store: Postgres checks `CREATE` on the schema before it reads `IF NOT EXISTS`.
        `CREATE TABLE IF NOT EXISTS` is not race-free in Postgres -- concurrent callers can
        collide on `pg_type`'s unique index -- so the statements run under an advisory lock
        keyed on the counters table name.
        """
        if self._schema_ready:
            return
        async with self._schema_lock:
            if self._schema_ready:
                return
            async with self._pool.acquire() as connection:
                exists = await connection.fetchval(
                    'SELECT to_regclass($1) IS NOT NULL AND to_regclass($2) IS NOT NULL', self._counters, self._dedup
                )
                if exists:
                    self._schema_ready = True
                    return
            async with self._pool.acquire() as connection, connection.transaction():
                await connection.fetchval('SELECT pg_advisory_xact_lock(hashtext($1))', self._counters)
                await connection.execute(
                    f'CREATE TABLE IF NOT EXISTS {self._counters} ('
                    'key_sha256 TEXT PRIMARY KEY, key BYTEA NOT NULL, usd_nanos BIGINT NOT NULL, tokens BIGINT NOT NULL, '
                    'requests BIGINT NOT NULL, unpriced BIGINT NOT NULL, expires_at TIMESTAMPTZ)'
                )
                await connection.execute(
                    f'CREATE INDEX IF NOT EXISTS {self._counters}_expiry ON {self._counters} (expires_at) '
                    'WHERE expires_at IS NOT NULL'
                )
                await connection.execute(
                    f'CREATE TABLE IF NOT EXISTS {self._dedup} ('
                    'marker_sha256 TEXT PRIMARY KEY, expires_at TIMESTAMPTZ NOT NULL)'
                )
                await connection.execute(
                    f'CREATE INDEX IF NOT EXISTS {self._dedup}_expiry ON {self._dedup} (expires_at)'
                )
            self._schema_ready = True
