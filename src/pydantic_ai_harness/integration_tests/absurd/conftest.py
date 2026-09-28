"""PostgreSQL fixtures for the live `AbsurdDurability` tests.

The unit suite in `tests/harness/absurd` drives the capability through a fake task context. These
tests run it on a real Absurd worker instead, to prove that checkpoints survive a worker crash and
that checkpoints recorded by `pydantic-ai-absurd` 0.8.0 resume here.

Run against a local server with `make integration-absurd` after starting one, e.g.
`docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:16`. Without a reachable server
the tests skip, unless `ABSURD_REQUIRE_LIVE` is set (CI does), where an unreachable server fails
instead.

`fixtures/absurd.sql` is Absurd's `sql/absurd.sql` (https://github.com/earendil-works/absurd,
Apache-2.0) as vendored by `pydantic-ai-absurd` 0.8.0 in `tests/fixtures/absurd.sql`. Refresh it
from the Absurd release matching the `absurd-sdk` floor in `pydantic-ai-harness[absurd]`.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from absurd_sdk import AsyncAbsurd
from psycopg import AsyncConnection
from psycopg.rows import TupleRow

ABSURD_SQL = (Path(__file__).parent / 'fixtures' / 'absurd.sql').read_text()


@pytest.fixture
def anyio_backend() -> str:
    """Run live server tests once under asyncio."""
    return 'asyncio'


@pytest.fixture(scope='session')
def db_dsn() -> str:
    """Install Absurd's schema in the database named by `ABSURD_TEST_DATABASE_URL`."""
    dsn = os.environ.get('ABSURD_TEST_DATABASE_URL', 'postgresql://postgres:postgres@127.0.0.1:5432/postgres')
    try:
        with psycopg.connect(dsn, autocommit=True, connect_timeout=5) as conn:
            # The schema script is not re-runnable, so a database kept between local runs is reused.
            installed = conn.execute("SELECT to_regnamespace('absurd') IS NOT NULL").fetchone()
            if installed != (True,):
                # This is a trusted, checked-in schema fixture, not user input.
                conn.execute(ABSURD_SQL)  # pyright: ignore[reportCallIssue, reportArgumentType]
    except psycopg.OperationalError as exc:
        message = f'PostgreSQL is unreachable at ABSURD_TEST_DATABASE_URL: {exc}'
        if os.environ.get('ABSURD_REQUIRE_LIVE', '').lower() in {'1', 'true', 'yes'}:
            pytest.fail(message)
        pytest.skip(message)
    return dsn


@pytest.fixture
async def async_conn(db_dsn: str) -> AsyncIterator[AsyncConnection[TupleRow]]:
    """Open an autocommit connection to the test database."""
    async with await AsyncConnection.connect(db_dsn, autocommit=True) as conn:
        yield conn


@pytest.fixture
async def absurd(async_conn: AsyncConnection[TupleRow]) -> AsyncIterator[AsyncAbsurd]:
    """Yield an Absurd client with an isolated queue."""
    client = AsyncAbsurd(async_conn, queue_name=f'test_{uuid4().hex[:8]}')
    await client.create_queue()
    try:
        yield client
    finally:
        await client.drop_queue()
