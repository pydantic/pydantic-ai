"""PostgreSQL-backed step-persistence store.

Mirrors `SqliteStepStore`: an append-only event log, continuable snapshots,
and a tool-effect ledger, over a caller-owned asyncpg-compatible pool. Binary
and text parts at or above `media_threshold_bytes` are externalized to a
`MediaStore` (default: a `PostgresMediaStore` on the same pool).

The store never imports `asyncpg`: it depends only on the `PostgresPool` and
`PostgresConnection` protocols from `pydantic_ai_harness.media`, so the harness
carries no database driver dependency and the application owns the pool's
lifecycle.

`table` is a prefix. Tables (created on first use):

- `{table}_runs` -- `run_id` primary key; the insert enforces the single-shot
  `run_id` contract (duplicates surface as `ValueError`).
- `{table}_events` -- one row per event, ordered by an identity `seq`. A
  unique partial index on `(run_id, idempotency_key)` drops keyed replays.
- `{table}_snapshots` -- one row per snapshot; latest-per-run is the highest
  `seq`, matching `SqliteStepStore`'s `AUTOINCREMENT seq` so a reused `run_id`
  whose `step_index` reset to 0 cannot clobber an earlier snapshot.
- `{table}_snapshot_keys` -- `(run_id, idempotency_key)` primary key; replay
  suppression retained after pruning.
- `{table}_tool_effects` -- upsert per `(run_id, tool_call_id)`.
- `{table}_media` -- the default `PostgresMediaStore` table.

Timestamps are stored as ISO 8601 text and JSON as text, column for column
like the SQLite schema, so rows written by either backend read the same way.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from datetime import datetime

import anyio

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_ai_harness.media import (
    MediaStore,
    PostgresConnection,
    PostgresMediaStore,
    PostgresPool,
    externalize_media,
    restore_media,
)
from pydantic_ai_harness.step_persistence._store import (
    _DEFAULT_MEDIA_THRESHOLD_BYTES,  # pyright: ignore[reportPrivateUsage]
    _AutoMedia,  # pyright: ignore[reportPrivateUsage]
    _event_from_row,  # pyright: ignore[reportPrivateUsage]
    _opt_str,  # pyright: ignore[reportPrivateUsage]
    _retained_seqs,  # pyright: ignore[reportPrivateUsage]
    _run_from_row,  # pyright: ignore[reportPrivateUsage]
    _snapshot_state,  # pyright: ignore[reportPrivateUsage]
    _tool_effect_from_row,  # pyright: ignore[reportPrivateUsage]
    _validate_max_snapshots,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai_harness.step_persistence._types import (
    ContinuableSnapshot,
    RunRecord,
    SnapshotState,
    StepEvent,
    ToolEffectRecord,
)

_logger = logging.getLogger(__name__)

# Postgres truncates identifiers past 63 bytes, which would let two distinct
# names collide. The prefix is capped at 40 characters so every derived name
# fits: the longest is the implicit primary-key constraint
# `{table}_snapshot_keys_pkey` (40 + 19 = 59), ahead of the longest index
# `{table}_snapshots_run_idx` and the identity sequence
# `{table}_snapshots_seq_seq` (40 + 18 = 58 each). Lowercase only: the
# interpolated identifiers are unquoted, so Postgres folds them to lowercase
# and `'Orders'` would share tables with `'orders'`.
_TABLE_RE = re.compile(r'[a-z_][a-z0-9_]{0,39}')

_RUN_COLUMNS = 'run_id, conversation_id, parent_run_id, agent_name, metadata, started_at, registration_id'
_EVENT_COLUMNS = (
    'run_id, kind, step_index, timestamp, conversation_id, parent_run_id, '
    'agent_name, tool_call_id, tool_name, error, metadata, idempotency_key'
)
_SNAPSHOT_COLUMNS = (
    'run_id, step_index, conversation_id, parent_run_id, agent_name, timestamp, state, messages, idempotency_key'
)
_TOOL_EFFECT_COLUMNS = 'run_id, tool_call_id, tool_name, status, started_at, ended_at, idempotency_key, effect_summary'


class PostgresStepStore:
    """PostgreSQL-backed step-persistence store; see the module docstring for layout.

    Takes a caller-owned asyncpg-compatible pool; the store never closes it.
    Each operation acquires one connection, and writes run in a transaction.

    `table` is a prefix for the store's tables (`{table}_runs`,
    `{table}_events`, ...), limited to 40 characters so the derived names stay
    inside Postgres's 63 byte identifier limit.

    `media_store` defaults to `'auto'`, which builds a `PostgresMediaStore` on
    the same pool with the table `{table}_media`. Pass `None` to keep payloads
    inline in the snapshot row, or pass any `MediaStore` (e.g. `S3MediaStore`)
    to redirect. Parts whose byte length is >= `media_threshold_bytes` are
    externalized.

    The `{table}_runs.run_id` primary key enforces the "explicit `run_id` is
    single-shot" contract: `register_run` raises `ValueError` on reuse.

    `max_snapshots_per_run` (default `None`, unbounded) bounds per-run
    snapshot growth: after each write, one `DELETE` prunes the rows outside
    the retain set (see `_prune_snapshots`).
    """

    def __init__(
        self,
        pool: PostgresPool,
        *,
        table: str = 'step_persistence',
        media_store: MediaStore | None | _AutoMedia = 'auto',
        media_threshold_bytes: int = _DEFAULT_MEDIA_THRESHOLD_BYTES,
        max_snapshots_per_run: int | None = None,
    ) -> None:
        if not _TABLE_RE.fullmatch(table):
            raise ValueError(f'invalid table name: {table!r}')
        _validate_max_snapshots(max_snapshots_per_run)
        self._max_snapshots_per_run = max_snapshots_per_run
        self._pool = pool
        self._table = table
        self._runs_table = f'{table}_runs'
        self._events_table = f'{table}_events'
        self._snapshots_table = f'{table}_snapshots'
        self._snapshot_keys_table = f'{table}_snapshot_keys'
        self._tool_effects_table = f'{table}_tool_effects'
        resolved: MediaStore | None
        if media_store == 'auto':
            resolved = PostgresMediaStore(pool, table=f'{table}_media')
        else:
            resolved = media_store
        self._media_store: MediaStore | None = resolved
        self._media_threshold_bytes = media_threshold_bytes
        self._schema_ready = False
        self._schema_lock = anyio.Lock()

    async def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        async with self._schema_lock:
            if self._schema_ready:
                return
            async with self._pool.acquire() as connection, connection.transaction():
                # `CREATE TABLE IF NOT EXISTS` is not race-free in Postgres: concurrent
                # callers can collide on `pg_type`'s unique index.
                await connection.fetchval('SELECT pg_advisory_xact_lock(hashtext($1))', self._table)
                await connection.execute(
                    f'CREATE TABLE IF NOT EXISTS {self._runs_table} ('
                    'run_id TEXT PRIMARY KEY, '
                    'conversation_id TEXT, '
                    'parent_run_id TEXT, '
                    'agent_name TEXT, '
                    'metadata TEXT NOT NULL, '
                    'started_at TEXT NOT NULL, '
                    'registration_id TEXT)'
                )
                await connection.execute(
                    f'CREATE INDEX IF NOT EXISTS {self._runs_table}_conv_idx ON {self._runs_table} (conversation_id)'
                )
                await connection.execute(
                    f'CREATE INDEX IF NOT EXISTS {self._runs_table}_parent_idx ON {self._runs_table} (parent_run_id)'
                )
                await connection.execute(
                    f'CREATE TABLE IF NOT EXISTS {self._events_table} ('
                    'seq BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, '
                    'run_id TEXT NOT NULL, '
                    'kind TEXT NOT NULL, '
                    'step_index INTEGER NOT NULL, '
                    'timestamp TEXT NOT NULL, '
                    'conversation_id TEXT, '
                    'parent_run_id TEXT, '
                    'agent_name TEXT, '
                    'tool_call_id TEXT, '
                    'tool_name TEXT, '
                    'error TEXT, '
                    'metadata TEXT NOT NULL, '
                    'idempotency_key TEXT)'
                )
                await connection.execute(
                    f'CREATE INDEX IF NOT EXISTS {self._events_table}_run_idx ON {self._events_table} (run_id, seq)'
                )
                await connection.execute(
                    f'CREATE UNIQUE INDEX IF NOT EXISTS {self._events_table}_key_idx '
                    f'ON {self._events_table} (run_id, idempotency_key) WHERE idempotency_key IS NOT NULL'
                )
                await connection.execute(
                    f'CREATE TABLE IF NOT EXISTS {self._snapshots_table} ('
                    'seq BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, '
                    'run_id TEXT NOT NULL, '
                    'step_index INTEGER NOT NULL, '
                    'conversation_id TEXT, '
                    'parent_run_id TEXT, '
                    'agent_name TEXT, '
                    'timestamp TEXT NOT NULL, '
                    "state TEXT NOT NULL DEFAULT 'complete', "
                    'messages TEXT NOT NULL, '
                    'idempotency_key TEXT)'
                )
                await connection.execute(
                    f'CREATE INDEX IF NOT EXISTS {self._snapshots_table}_run_idx '
                    f'ON {self._snapshots_table} (run_id, seq)'
                )
                await connection.execute(
                    f'CREATE TABLE IF NOT EXISTS {self._snapshot_keys_table} ('
                    'run_id TEXT NOT NULL, '
                    'idempotency_key TEXT NOT NULL, '
                    'PRIMARY KEY (run_id, idempotency_key))'
                )
                await connection.execute(
                    f'CREATE TABLE IF NOT EXISTS {self._tool_effects_table} ('
                    'run_id TEXT NOT NULL, '
                    'tool_call_id TEXT NOT NULL, '
                    'tool_name TEXT NOT NULL, '
                    'status TEXT NOT NULL, '
                    'started_at TEXT NOT NULL, '
                    'ended_at TEXT, '
                    'idempotency_key TEXT, '
                    'effect_summary TEXT, '
                    'PRIMARY KEY (run_id, tool_call_id))'
                )
            self._schema_ready = True

    async def register_run(self, record: RunRecord) -> None:
        """Insert the run; a `run_id` already in the store raises `ValueError`.

        The `run_id` primary key is what enforces the "explicit `run_id` is
        single-shot" contract, so two workers racing on the same id cannot
        both register it.
        """
        await self._ensure_schema()
        async with self._pool.acquire() as connection, connection.transaction():
            inserted = await connection.fetchval(
                f'INSERT INTO {self._runs_table} ({_RUN_COLUMNS}) VALUES ($1, $2, $3, $4, $5, $6, $7) '
                'ON CONFLICT (run_id) DO NOTHING RETURNING run_id',
                record.run_id,
                record.conversation_id,
                record.parent_run_id,
                record.agent_name,
                json.dumps(dict(record.metadata)),
                record.started_at.isoformat(),
                record.registration_id,
            )
        if inserted is None:
            raise ValueError(f'run_id {record.run_id!r} is already in the store')

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        await self._ensure_schema()
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(f'SELECT {_RUN_COLUMNS} FROM {self._runs_table} WHERE run_id = $1', run_id)
        if row is None:
            return None
        return _run_from_row(tuple(row))

    async def list_runs(
        self,
        *,
        parent_run_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[RunRecord]:
        await self._ensure_schema()
        clauses: list[str] = []
        params: list[object] = []
        if parent_run_id is not None:
            params.append(parent_run_id)
            clauses.append(f'parent_run_id = ${len(params)}')
        if conversation_id is not None:
            params.append(conversation_id)
            clauses.append(f'conversation_id = ${len(params)}')
        sql = f'SELECT {_RUN_COLUMNS} FROM {self._runs_table}'
        if clauses:
            sql += ' WHERE ' + ' AND '.join(clauses)
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(sql, *params)
        # Sort by the parsed instant, not the stored ISO string: a lexicographic
        # sort would misorder mixed-offset timestamps, and the `StepStore`
        # contract (matching the in-memory/sqlite/file stores) is instant order.
        return sorted((_run_from_row(tuple(row)) for row in rows), key=lambda r: r.started_at)

    async def append_event(self, event: StepEvent) -> None:
        await self._ensure_schema()
        async with self._pool.acquire() as connection, connection.transaction():
            await connection.execute(
                f'INSERT INTO {self._events_table} ({_EVENT_COLUMNS}) '
                'VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12) '
                'ON CONFLICT (run_id, idempotency_key) WHERE idempotency_key IS NOT NULL DO NOTHING',
                event.run_id,
                event.kind,
                event.step_index,
                event.timestamp.isoformat(),
                event.conversation_id,
                event.parent_run_id,
                event.agent_name,
                event.tool_call_id,
                event.tool_name,
                event.error,
                json.dumps(dict(event.metadata)),
                event.idempotency_key,
            )

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        await self._ensure_schema()
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(
                f'SELECT {_EVENT_COLUMNS} FROM {self._events_table} WHERE run_id = $1 ORDER BY seq ASC', run_id
            )
        return [_event_from_row(tuple(row)) for row in rows]

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        await self._ensure_schema()
        messages_json: object = json.loads(ModelMessagesTypeAdapter.dump_json(snapshot.messages).decode('utf-8'))
        # Externalize before opening the transaction: the default media store
        # acquires its own connection from the same pool, so doing it inside
        # would hold two connections per save and can exhaust a small pool.
        if self._media_store is not None:
            messages_json = await externalize_media(
                messages_json,
                media_store=self._media_store,
                threshold_bytes=self._media_threshold_bytes,
            )
        async with self._pool.acquire() as connection, connection.transaction():
            if snapshot.idempotency_key is not None:
                # This ledger is independent of retained snapshot rows, so pruning cannot
                # make a previously applied key eligible again. Claiming the key and
                # inserting the snapshot commit together, so a failed insert releases it.
                claimed = await connection.fetchval(
                    f'INSERT INTO {self._snapshot_keys_table} (run_id, idempotency_key) VALUES ($1, $2) '
                    'ON CONFLICT DO NOTHING RETURNING run_id',
                    snapshot.run_id,
                    snapshot.idempotency_key,
                )
                if claimed is None:
                    return
            await connection.execute(
                f'INSERT INTO {self._snapshots_table} ({_SNAPSHOT_COLUMNS}) '
                'VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)',
                snapshot.run_id,
                snapshot.step_index,
                snapshot.conversation_id,
                snapshot.parent_run_id,
                snapshot.agent_name,
                snapshot.timestamp.isoformat(),
                snapshot.state,
                json.dumps(messages_json),
                snapshot.idempotency_key,
            )
            await self._prune_snapshots(connection, snapshot.run_id)

    async def _prune_snapshots(self, connection: PostgresConnection, run_id: str) -> None:
        """Delete this run's snapshot rows outside the retain set when bounded.

        No-op when `max_snapshots_per_run` is `None`. Externalized media is
        content-addressed and may be shared across snapshots and runs, so a
        deleted snapshot row never removes a media row -- orphaned-blob GC is
        a separate concern (see the capability README non-goals).

        The retain set comes from `_retained_seqs`, so the newest overall and
        the newest `complete` survive even when the newest `keep` rows are
        all `interrupted`. It is bound as one array parameter, so its size
        does not grow the statement.

        The `DELETE` is also limited to the rows this prune enumerated. Under
        `READ COMMITTED` each statement takes its own snapshot, so a save that
        commits between the `SELECT` and the `DELETE` would otherwise have its
        row deleted without the retain rule having seen it. Concurrent saves
        can leave a run above the bound until the next save prunes it.
        """
        if self._max_snapshots_per_run is None:
            return
        rows = await connection.fetch(f'SELECT seq, state FROM {self._snapshots_table} WHERE run_id = $1', run_id)
        entries: list[tuple[int, SnapshotState]] = []
        for row in rows:
            seq = row[0]
            assert isinstance(seq, int)
            entries.append((seq, _snapshot_state(row[1])))
        retained = _retained_seqs(entries, self._max_snapshots_per_run)
        await connection.execute(
            f'DELETE FROM {self._snapshots_table} '
            'WHERE run_id = $1 AND NOT (seq = ANY($2::bigint[])) AND seq = ANY($3::bigint[])',
            run_id,
            sorted(retained),
            sorted(seq for seq, _ in entries),
        )

    async def _snapshot_from_row(self, run_id: str, row: Sequence[object]) -> ContinuableSnapshot:
        step_index, conv_id, parent_id, agent_name, timestamp_iso, state_raw, messages_text, key = row
        if not (isinstance(step_index, int) and isinstance(timestamp_iso, str) and isinstance(messages_text, str)):
            raise ValueError('snapshot row has wrong types')
        messages_json: object = json.loads(messages_text)
        if self._media_store is not None:
            messages_json = await restore_media(messages_json, media_store=self._media_store)
        messages: list[ModelMessage] = ModelMessagesTypeAdapter.validate_python(messages_json)
        return ContinuableSnapshot(
            run_id=run_id,
            step_index=step_index,
            messages=messages,
            conversation_id=_opt_str(conv_id),
            parent_run_id=_opt_str(parent_id),
            agent_name=_opt_str(agent_name),
            timestamp=datetime.fromisoformat(timestamp_iso),
            state=_snapshot_state(state_raw),
            idempotency_key=_opt_str(key),
        )

    def _select_snapshots_sql(self, include_interrupted: bool) -> str:
        sql = (
            'SELECT step_index, conversation_id, parent_run_id, agent_name, timestamp, state, messages, '
            f'idempotency_key FROM {self._snapshots_table} WHERE run_id = $1'
        )
        if not include_interrupted:
            sql += " AND state = 'complete'"
        return sql

    async def latest_snapshot(self, *, run_id: str, include_interrupted: bool = False) -> ContinuableSnapshot | None:
        await self._ensure_schema()
        sql = self._select_snapshots_sql(include_interrupted) + ' ORDER BY seq DESC LIMIT 1'
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(sql, run_id)
        if row is None:
            return None
        # The connection is released before media is restored, for the same
        # reason `save_snapshot` externalizes before its transaction.
        return await self._snapshot_from_row(run_id, row)

    async def list_snapshots(self, *, run_id: str, include_interrupted: bool = False) -> list[ContinuableSnapshot]:
        """Return retained snapshots for `run_id` in write order.

        Mirrors the `latest_snapshot` gate: `interrupted` snapshots are skipped
        unless `include_interrupted=True`. Rows that fail to parse are skipped
        and logged, so one damaged row does not hide the rest of the run's
        history. Not part of the `StepStore` protocol: the `conversation_search`
        capability consumes it through its narrower `SnapshotStore` protocol.
        """
        await self._ensure_schema()
        sql = self._select_snapshots_sql(include_interrupted) + ' ORDER BY seq ASC'
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(sql, run_id)
        snapshots: list[ContinuableSnapshot] = []
        for row in rows:
            try:
                snapshots.append(await self._snapshot_from_row(run_id, row))
            except Exception:
                _logger.warning('Skipping unparsable snapshot row for run %s', run_id, exc_info=True)
        return snapshots

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        await self._ensure_schema()
        async with self._pool.acquire() as connection, connection.transaction():
            await connection.execute(
                f'INSERT INTO {self._tool_effects_table} ({_TOOL_EFFECT_COLUMNS}) '
                'VALUES ($1, $2, $3, $4, $5, $6, $7, $8) '
                'ON CONFLICT (run_id, tool_call_id) DO UPDATE SET '
                'tool_name = EXCLUDED.tool_name, status = EXCLUDED.status, started_at = EXCLUDED.started_at, '
                'ended_at = EXCLUDED.ended_at, idempotency_key = EXCLUDED.idempotency_key, '
                'effect_summary = EXCLUDED.effect_summary',
                record.run_id,
                record.tool_call_id,
                record.tool_name,
                record.status,
                record.started_at.isoformat(),
                record.ended_at.isoformat() if record.ended_at is not None else None,
                record.idempotency_key,
                record.effect_summary,
            )

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> ToolEffectRecord | None:
        await self._ensure_schema()
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(
                f'SELECT {_TOOL_EFFECT_COLUMNS} FROM {self._tool_effects_table} '
                'WHERE run_id = $1 AND tool_call_id = $2',
                run_id,
                tool_call_id,
            )
        if row is None:
            return None
        return _tool_effect_from_row(tuple(row))

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]:
        await self._ensure_schema()
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(
                f'SELECT {_TOOL_EFFECT_COLUMNS} FROM {self._tool_effects_table} '
                "WHERE run_id = $1 AND status = 'started'",
                run_id,
            )
        return [_tool_effect_from_row(tuple(row)) for row in rows]
