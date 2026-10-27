"""Connecting the live PostgreSQL tests under `pydantic_ai_harness/integration_tests/postgres` to a server.

Each test opens its own pool through `live_tables` and works in tables whose names start with a
prefix generated for it, dropped afterwards, so the suite can share a database with other data.
Without a reachable server the tests skip, unless `POSTGRES_REQUIRE_LIVE` is set (CI does), where
an unreachable server fails instead. `POSTGRES_TEST_URL` carries credentials, so it never appears in
a skip or failure message: only the exception type does.
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import NoReturn
from uuid import uuid4

import anyio
import asyncpg
import pytest

from pydantic_ai_harness.media import PostgresPool

# Retrying is only worth it where a server is promised but may still be starting.
# Locally an absent server is the normal case, so one attempt is enough and the
# suite skips in seconds instead of stalling. In CI the window only has to cover
# the gap between the service container passing its health check and the server
# accepting connections -- a container that never becomes healthy fails the job
# before any step runs, so a longer window would only make the never-reachable
# case burn the job's whole timeout.
CONNECT_RETRY_SECONDS = 30.0
CONNECT_TIMEOUT_SECONDS = 5.0

# Nothing listens on port 1, so a connection is refused rather than left to time out.
UNREACHABLE_URL = 'postgresql://127.0.0.1:1/none'

# URLs that already exhausted a connection attempt, with the failure they ended on.
# Every test opens its own pool, so without this an absent server would cost each
# of them the full attempt (and in CI the full retry window) instead of only the first.
unreachable: dict[str, str] = {}


def requires_live() -> bool:
    return os.environ.get('POSTGRES_REQUIRE_LIVE', '').lower() in {'1', 'true', 'yes'}


def postgres_url() -> str:
    return os.environ.get('POSTGRES_TEST_URL', 'postgresql://postgres:postgres@127.0.0.1:5432/postgres')


def unavailable(message: str) -> NoReturn:
    if requires_live():
        pytest.fail(message)
    pytest.skip(message)


async def connect(*, retry_seconds: float = CONNECT_RETRY_SECONDS) -> asyncpg.Pool:
    """Return an open pool, retrying until the deadline when a server is required.

    A container health check reports the process listening; the server may still
    refuse the first connection while it finishes starting. Retrying pool
    creation is what makes the suite independent of that gap.
    """
    url = postgres_url()
    deadline = time.monotonic() + (retry_seconds if requires_live() else 0.0)
    while url not in unreachable:
        try:
            return await asyncpg.create_pool(  # pyright: ignore[reportUnknownMemberType]
                url, min_size=1, max_size=4, timeout=CONNECT_TIMEOUT_SECONDS
            )
        except (TimeoutError, OSError, asyncpg.PostgresError, asyncpg.InterfaceError) as exc:
            # Only the type: the message of a connection error can quote the URL.
            failure = type(exc).__name__
        if time.monotonic() >= deadline:
            unreachable[url] = failure
            break
        await anyio.sleep(1.0)
    unavailable(f'No PostgreSQL server reachable through POSTGRES_TEST_URL ({unreachable[url]}).')


def derived_tables(prefix: str) -> list[str]:
    """Every table a test may create from `prefix`, and nothing else."""
    # Referencing tables first, so dropping in order never trips a foreign key.
    return [
        prefix,
        f'{prefix}_conv_records',
        f'{prefix}_conv_entries',
        f'{prefix}_conv_messages',
        f'{prefix}_runs',
        f'{prefix}_events',
        f'{prefix}_snapshots',
        f'{prefix}_snapshot_keys',
        f'{prefix}_messages',
        f'{prefix}_tool_effects',
        f'{prefix}_media',
    ]


def as_store_pool(pool: asyncpg.Pool) -> PostgresPool:
    """The single type-boundary shim between asyncpg and the stores' pool protocol.

    An asyncpg pool hands out a `PoolConnectionProxy`, which forwards the
    connection methods at runtime through a metaclass, so a type checker sees
    none of them and cannot match the pool against `PostgresPool`.
    """
    return pool  # pyright: ignore[reportReturnType]


@asynccontextmanager
async def live_tables() -> AsyncGenerator[tuple[PostgresPool, str], None]:
    """Yield a pool and a table prefix unique to one test; drop its tables afterwards."""
    driver_pool = await connect()
    pool = as_store_pool(driver_pool)
    prefix = f'it_{uuid4().hex[:12]}'
    try:
        yield pool, prefix
    finally:
        try:
            async with pool.acquire() as connection:
                for table in derived_tables(prefix):
                    await connection.execute(f'DROP TABLE IF EXISTS {table}')
        finally:
            await driver_pool.close()


async def count_rows(pool: PostgresPool, table: str) -> int:
    async with pool.acquire() as connection:
        count = await connection.fetchval(f'SELECT count(*) FROM {table}')
    assert isinstance(count, int)
    return count


async def table_exists(pool: PostgresPool, table: str) -> bool:
    async with pool.acquire() as connection:
        exists = await connection.fetchval('SELECT to_regclass($1) IS NOT NULL', table)
    assert isinstance(exists, bool)
    return exists
