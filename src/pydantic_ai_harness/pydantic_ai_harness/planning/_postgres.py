"""PostgreSQL plan storage over a caller-owned asyncpg-compatible pool.

The store never imports `asyncpg`: it depends only on the `PostgresPool` and
`PostgresConnection` protocols below, so the harness carries no database driver
dependency and the backend is testable with an in-memory fake pool. Pass your
own `asyncpg` pool (which already satisfies these protocols) at construction.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from typing import Protocol, runtime_checkable

import anyio

from pydantic_ai_harness.planning._events import PlanEventEmitter
from pydantic_ai_harness.planning._store import (
    apply_updates,
    emit_created,
    emit_deleted,
    emit_mutation,
    validate_table_name,
)
from pydantic_ai_harness.planning._types import PlanItem, TaskStatus


@runtime_checkable
class PostgresConnection(Protocol):
    """The acquired asyncpg-compatible connection surface used by the store."""

    def transaction(self) -> AbstractAsyncContextManager[object]:
        """Return an async transaction context manager."""
        ...  # pragma: no cover

    async def execute(self, query: str, *args: object) -> object:
        """Execute a statement and return its status string."""
        ...  # pragma: no cover

    async def fetchval(self, query: str, *args: object) -> object:
        """Return the first column of the first row."""
        ...  # pragma: no cover

    async def fetchrow(self, query: str, *args: object) -> Sequence[object] | None:
        """Return the first row, or `None`."""
        ...  # pragma: no cover

    async def fetch(self, query: str, *args: object) -> Sequence[Sequence[object]]:
        """Return all rows."""
        ...  # pragma: no cover


@runtime_checkable
class PostgresPool(Protocol):
    """The asyncpg-compatible pool surface used by `PostgresPlanStore`."""

    def acquire(self) -> AbstractAsyncContextManager[PostgresConnection]:
        """Acquire one connection for an operation or transaction."""
        ...  # pragma: no cover


class PostgresPlanStore:
    """PostgreSQL plan storage scoped to a `session` for multi-tenancy.

    `set_items` runs in a transaction and is atomic. The granular ops are not
    wrapped in a shared transaction -- `add_item` picks its sequence with a
    non-transactional `MAX(seq) + 1`, and `update_item`/`remove_item` are
    read-modify-write -- so concurrent writers on the same session can race.
    Drive one session from one task, or use `set_items` for atomic writes.
    """

    def __init__(
        self,
        pool: PostgresPool,
        *,
        session: str = 'default',
        table: str = 'plan_items',
        event_emitter: PlanEventEmitter | None = None,
    ) -> None:
        validate_table_name(table)
        self._pool = pool
        self._session = session
        self._table = table
        self._emitter = event_emitter
        self._ready = False
        self._schema_lock = anyio.Lock()

    _SELECT_COLUMNS = 'id, content, status, active_form, parent_id, depends_on'

    async def _ensure_schema(self) -> None:
        """Create the table once, serialised in-process and across processes.

        `CREATE TABLE IF NOT EXISTS` is not race-free in Postgres -- concurrent
        callers can collide on `pg_type`'s unique index -- so the statement runs
        under an advisory lock keyed on the table name.
        """
        if self._ready:
            return
        async with self._schema_lock:
            if self._ready:
                return
            async with self._pool.acquire() as connection, connection.transaction():
                await connection.fetchval('SELECT pg_advisory_xact_lock(hashtext($1))', self._table)
                await connection.execute(
                    f'CREATE TABLE IF NOT EXISTS {self._table} ('
                    'session TEXT NOT NULL, seq BIGINT NOT NULL, id TEXT NOT NULL, '
                    'content TEXT NOT NULL, status TEXT NOT NULL, active_form TEXT NOT NULL, '
                    'parent_id TEXT, depends_on TEXT NOT NULL, PRIMARY KEY (session, id))'
                )
            self._ready = True

    def _row_to_item(self, row: Sequence[object]) -> PlanItem:
        return PlanItem(
            id=str(row[0]),
            content=str(row[1]),
            status=TaskStatus(str(row[2])),
            active_form=str(row[3]),
            parent_id=None if row[4] is None else str(row[4]),
            depends_on=list(json.loads(str(row[5]))),
        )

    async def get_items(self) -> list[PlanItem]:
        """Return every step for this session in insertion order."""
        await self._ensure_schema()
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(
                f'SELECT {self._SELECT_COLUMNS} FROM {self._table} WHERE session = $1 ORDER BY seq',
                self._session,
            )
        return [self._row_to_item(row) for row in rows]

    async def set_items(self, items: list[PlanItem]) -> None:
        """Replace the whole list for this session with `items`, in one transaction."""
        await self._ensure_schema()
        async with self._pool.acquire() as connection, connection.transaction():
            await connection.execute(f'DELETE FROM {self._table} WHERE session = $1', self._session)
            for seq, item in enumerate(items):
                await connection.execute(self._insert_sql(), *self._insert_params(seq, item))

    async def get_item(self, item_id: str) -> PlanItem | None:
        """Return the step with `item_id` for this session, or `None`."""
        await self._ensure_schema()
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(
                f'SELECT {self._SELECT_COLUMNS} FROM {self._table} WHERE session = $1 AND id = $2',
                self._session,
                item_id,
            )
        return None if row is None else self._row_to_item(row)

    async def add_item(self, item: PlanItem) -> PlanItem:
        """Append `item` for this session and return it."""
        await self._ensure_schema()
        async with self._pool.acquire() as connection:
            next_seq = await connection.fetchval(
                f'SELECT COALESCE(MAX(seq) + 1, 0) FROM {self._table} WHERE session = $1',
                self._session,
            )
            await connection.execute(self._insert_sql(), *self._insert_params(int(str(next_seq)), item))
        await emit_created(self._emitter, item)
        return item

    async def update_item(
        self,
        item_id: str,
        *,
        content: str | None = None,
        status: TaskStatus | None = None,
        active_form: str | None = None,
        parent_id: str | None = None,
        depends_on: list[str] | None = None,
    ) -> PlanItem | None:
        """Apply the non-`None` fields to `item_id`; return the updated step or `None`."""
        await self._ensure_schema()
        item = await self.get_item(item_id)
        if item is None:
            return None
        previous = item.model_copy() if self._emitter is not None else None
        apply_updates(
            item,
            content=content,
            status=status,
            active_form=active_form,
            parent_id=parent_id,
            depends_on=depends_on,
        )
        async with self._pool.acquire() as connection:
            await connection.execute(
                f'UPDATE {self._table} SET content = $1, status = $2, active_form = $3, '
                'parent_id = $4, depends_on = $5 WHERE session = $6 AND id = $7',
                item.content,
                item.status.value,
                item.active_form,
                item.parent_id,
                json.dumps(item.depends_on),
                self._session,
                item.id,
            )
        await emit_mutation(self._emitter, item, previous)
        return item

    async def remove_item(self, item_id: str) -> bool:
        """Delete `item_id` for this session; return whether it existed."""
        await self._ensure_schema()
        removed = await self.get_item(item_id) if self._emitter is not None else None
        async with self._pool.acquire() as connection:
            status = await connection.execute(
                f'DELETE FROM {self._table} WHERE session = $1 AND id = $2',
                self._session,
                item_id,
            )
        existed = _deleted_count(status) > 0
        if existed and removed is not None:
            await emit_deleted(self._emitter, removed)
        return existed

    def _insert_sql(self) -> str:
        return (
            f'INSERT INTO {self._table} '
            '(session, seq, id, content, status, active_form, parent_id, depends_on) '
            'VALUES ($1, $2, $3, $4, $5, $6, $7, $8)'
        )

    def _insert_params(self, seq: int, item: PlanItem) -> tuple[object, ...]:
        return (
            self._session,
            seq,
            item.id,
            item.content,
            item.status.value,
            item.active_form,
            item.parent_id,
            json.dumps(item.depends_on),
        )


def _deleted_count(status: object) -> int:
    """Parse the row count from an asyncpg `DELETE <n>` command tag."""
    parts = str(status).split()
    if len(parts) >= 2 and parts[-1].isdigit():
        return int(parts[-1])
    return 0
