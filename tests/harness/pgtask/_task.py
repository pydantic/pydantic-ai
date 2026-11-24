"""Run code inside real pgtask tasks, the way a worker does.

`run_in_task` enqueues a task on a queue of its own and works it with a real `Worker`. Every
attempt but the last ends in a simulated crash after the body returns, so the next attempt is
pgtask's own retry, replayed from the checkpoints in Postgres. `checkpoints` reads them back.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from uuid import uuid4

import anyio
import pytest

pytest.importorskip('pgtask')

from pgtask import Client, JSONValue, Task, TaskRegistry, Worker
from psycopg import AsyncConnection
from psycopg.types.json import Jsonb

StepKey = tuple[str, int]
"""A checkpoint's step name and occurrence."""


@dataclass(frozen=True)
class PGTaskDatabase:
    """A migrated pgtask database and a client on it."""

    dsn: str
    client: Client


class _SimulatedCrash(Exception):
    pass


async def run_in_task(
    db: PGTaskDatabase,
    body: Callable[[int], Awaitable[object]],
    *,
    attempts: int = 1,
    seed: Mapping[StepKey, JSONValue] | None = None,
) -> str:
    """Run `body(attempt)` in a pgtask task `attempts` times and return the task's id.

    `seed` checkpoints are stored before the first attempt, as if an earlier worker had written them.
    """
    registry = TaskRegistry(queue_name=f'test_{uuid4().hex[:8]}')
    runs = 0

    @registry.task('run', retry_delay=0)
    async def run(task: Task, payload: JSONValue) -> JSONValue:
        nonlocal runs
        runs += 1
        await body(runs)
        if runs < attempts:
            raise _SimulatedCrash
        return None

    result = None
    handle = await db.client.enqueue(run.request(None, max_attempts=attempts))
    if seed:
        async with await AsyncConnection.connect(db.dsn, autocommit=True) as conn:
            for (name, occurrence), value in seed.items():
                await conn.execute(
                    'INSERT INTO pgtask.checkpoints (task_id, handler_version, step_name, occurrence, value) '
                    'VALUES (%s, 1, %s, %s, %s)',
                    (handle.id, name, occurrence, Jsonb(value)),
                )

    worker = Worker(db.dsn, registry, concurrency=1, poll_interval=0.05, max_query_connections=4)
    async with anyio.create_task_group() as tg:
        tg.start_soon(worker.run)
        try:
            result = await handle.result(timeout=30)
        finally:
            worker.shutdown()

    assert result is not None
    assert result.state == 'succeeded', result.error
    assert runs == attempts
    return handle.id


async def checkpoints(db: PGTaskDatabase, task_id: str) -> dict[StepKey, JSONValue]:
    """The task's stored checkpoints, in the order they were written."""
    async with await AsyncConnection.connect(db.dsn) as conn:
        cursor = await conn.execute(
            'SELECT step_name, occurrence, value FROM pgtask.checkpoint_view WHERE task_id = %s '
            'ORDER BY created_at, step_name, occurrence',
            (task_id,),
        )
        return {(name, occurrence): value for name, occurrence, value in await cursor.fetchall()}
