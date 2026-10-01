"""A real Absurd schema on PostgreSQL, and task contexts entered the way an Absurd worker enters them.

Every test runs against Absurd's own SQL, so step naming, encounter-order disambiguation and
checkpoint storage are Absurd's, not a stand-in's. The database is named by `ABSURD_TEST_DATABASE_URL`:

    docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:16
    export ABSURD_TEST_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5432/postgres

Without it the tests skip. With it but no reachable server they also skip, unless
`ABSURD_REQUIRE_LIVE` is set (CI does), where they fail instead.

`fixtures/absurd.sql` is Absurd's `sql/absurd.sql` (https://github.com/earendil-works/absurd,
Apache-2.0). Refresh it from the Absurd release matching the `absurd-sdk` floor in
`pydantic-ai-harness[absurd]`.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from ...conftest import try_import

with try_import() as imports_successful:
    from absurd_sdk import AsyncAbsurd
    from psycopg import AsyncConnection
    from psycopg.rows import TupleRow

if TYPE_CHECKING:
    AsyncConn = AsyncConnection[TupleRow]

ABSURD_SQL = Path(__file__).parent / 'fixtures' / 'absurd.sql'


@pytest.fixture(scope='session')
def db_dsn() -> str:
    """Install Absurd's schema, once, in the database named by `ABSURD_TEST_DATABASE_URL`."""
    import psycopg

    dsn = os.environ.get('ABSURD_TEST_DATABASE_URL')
    if not dsn:
        pytest.skip('ABSURD_TEST_DATABASE_URL is not set')
    try:
        with psycopg.connect(dsn, autocommit=True, connect_timeout=5) as conn:
            # xdist workers share the database, and the schema script is not re-runnable.
            conn.execute('SELECT pg_advisory_lock(8126)')
            if conn.execute("SELECT to_regnamespace('absurd') IS NULL").fetchone() == (True,):
                conn.execute(ABSURD_SQL.read_bytes())
            conn.execute('SELECT pg_advisory_unlock(8126)')
    # Only when the configured server is unreachable, which CI never is.
    except psycopg.OperationalError as exc:  # pragma: no cover
        message = f'PostgreSQL is unreachable at ABSURD_TEST_DATABASE_URL: {exc}'
        if os.environ.get('ABSURD_REQUIRE_LIVE', '').lower() in {'1', 'true', 'yes'}:
            pytest.fail(message)
        pytest.skip(message)
    return dsn


@pytest.fixture
async def async_conn(db_dsn: str) -> AsyncGenerator[AsyncConn]:
    """An autocommit connection to the test database."""
    async with await AsyncConnection.connect(db_dsn, autocommit=True) as conn:
        yield conn


@pytest.fixture
def queue_name() -> str:
    """A queue name unique to the test, so runs on a kept database cannot collide."""
    return f'test_{uuid4().hex[:8]}'


@pytest.fixture
async def absurd(async_conn: AsyncConn, queue_name: str) -> AsyncGenerator[AsyncAbsurd]:
    """An Absurd client on a queue of its own, dropped after the test."""
    client = AsyncAbsurd(async_conn, queue_name=queue_name)
    await client.create_queue()
    try:
        yield client
    finally:
        await client.drop_queue()
