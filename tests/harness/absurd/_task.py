"""Enter real Absurd task contexts, the way a worker does.

`running_task_context` spawns and claims a task and enters its `AsyncTaskContext`, so the agent
runs as it would inside a handler. `reenter_running_task` fails that run and re-claims the task,
as Absurd does on a retry: the new context is hydrated from the checkpoints in Postgres, so a
replay is Absurd's own. `checkpoints` reads the checkpoint table back, and `claim_expires_at` the
running run's claim.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime

import pytest

pytest.importorskip('absurd_sdk')

from absurd_sdk import (
    AsyncAbsurd,
    AsyncTaskContext,
    ClaimedTask,
    JsonValue,
    _create_async_task_context,  # pyright: ignore[reportPrivateUsage]
    _current_task_context,  # pyright: ignore[reportPrivateUsage]
)
from psycopg import AsyncConnection, sql
from psycopg.rows import TupleRow


def _conn(absurd: AsyncAbsurd) -> AsyncConnection[TupleRow]:
    conn: AsyncConnection[TupleRow] | None = absurd._conn  # pyright: ignore[reportPrivateUsage]
    assert conn is not None
    return conn


def _queue(absurd: AsyncAbsurd) -> str:
    return absurd._queue_name  # pyright: ignore[reportPrivateUsage]


async def _noop(params: JsonValue, ctx: AsyncTaskContext) -> JsonValue:  # pragma: no cover
    return None


@asynccontextmanager
async def _entered(
    absurd: AsyncAbsurd, task: ClaimedTask, *, claim_timeout: int = 120
) -> AsyncGenerator[AsyncTaskContext]:
    ctx = await _create_async_task_context(task['task_id'], _conn(absurd), _queue(absurd), task, claim_timeout)
    token = _current_task_context.set(ctx)
    try:
        yield ctx
    finally:
        _current_task_context.reset(token)


@asynccontextmanager
async def running_task_context(
    absurd: AsyncAbsurd, task_name: str = 'noop', *, claim_timeout: int = 120
) -> AsyncGenerator[AsyncTaskContext]:
    """Spawn, claim and enter a task context for the duration of the block.

    The task stays `running` when the block exits, so `reenter_running_task` can retry it.
    """
    absurd.register_task(name=task_name)(_noop)
    spawned = await absurd.spawn(task_name, None, max_attempts=2)
    [task] = await absurd.claim_tasks(batch_size=1, claim_timeout=claim_timeout)
    assert task['task_id'] == spawned['task_id']
    async with _entered(absurd, task, claim_timeout=claim_timeout) as ctx:
        yield ctx


@asynccontextmanager
async def reenter_running_task(absurd: AsyncAbsurd, task_id: str) -> AsyncGenerator[AsyncTaskContext]:
    """Fail the task's current run and enter the retry, whose checkpoints come from Postgres."""
    conn = _conn(absurd)
    queue = _queue(absurd)
    cursor = await conn.execute(
        sql.SQL('SELECT run_id FROM absurd.{} WHERE task_id = %s AND state = %s').format(sql.Identifier(f'r_{queue}')),
        (task_id, 'running'),
    )
    row = await cursor.fetchone()
    assert row is not None
    await conn.execute('SELECT absurd.fail_run(%s, %s, %s, %s)', (queue, row[0], '{"type": "test.Replay"}', None))
    [task] = await absurd.claim_tasks(batch_size=1)
    assert task['task_id'] == task_id
    async with _entered(absurd, task) as ctx:
        yield ctx


async def checkpoints(absurd: AsyncAbsurd, task_id: str) -> dict[str, JsonValue]:
    """The task's stored checkpoints, by step name, in the order they were written."""
    cursor = await _conn(absurd).execute(
        sql.SQL('SELECT checkpoint_name, state FROM absurd.{} WHERE task_id = %s ORDER BY updated_at').format(
            sql.Identifier(f'c_{_queue(absurd)}')
        ),
        (task_id,),
    )
    return {name: state for name, state in await cursor.fetchall()}


async def claim_expires_at(absurd: AsyncAbsurd, task_id: str) -> datetime:
    """When the task's running run's claim expires."""
    expires_at = await claim_expiry(absurd, task_id)
    assert expires_at is not None
    return expires_at


async def claim_expiry(absurd: AsyncAbsurd, task_id: str) -> datetime | None:
    """The task's running run's claim expiry, or `None` for a claim that never expires."""
    cursor = await _conn(absurd).execute(
        sql.SQL('SELECT claim_expires_at FROM absurd.{} WHERE task_id = %s AND state = %s').format(
            sql.Identifier(f'r_{_queue(absurd)}')
        ),
        (task_id, 'running'),
    )
    row = await cursor.fetchone()
    assert row is not None
    return row[0]
