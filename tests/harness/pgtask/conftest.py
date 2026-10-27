"""A real pgtask schema on PostgreSQL.

Every test runs its agent inside a task claimed by a real pgtask `Worker`, so step naming,
occurrence numbering and checkpoint storage are pgtask's own. The database is named by
`PGTASK_TEST_DATABASE_URL`:

    docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:16
    export PGTASK_TEST_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5432/postgres

Without it the tests skip. With it but no reachable server they also skip, unless
`PGTASK_REQUIRE_LIVE` is set (CI does), where they fail instead.
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from ...conftest import try_import

with try_import() as imports_successful:
    import psycopg
    from pgtask import Client

    from ._task import PGTaskDatabase


def _migrate(dsn: str) -> None:
    async def migrate() -> None:
        await (await Client.connect(dsn, max_query_connections=1)).migrate()

    # On a thread of its own, so its event loop never becomes the tests' current one.
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(asyncio.run, migrate()).result()


@pytest.fixture(scope='session')
def pgtask_dsn() -> str:
    """Migrate pgtask's schema, once, in the database named by `PGTASK_TEST_DATABASE_URL`."""
    dsn = os.environ.get('PGTASK_TEST_DATABASE_URL')
    if not dsn:
        pytest.skip('PGTASK_TEST_DATABASE_URL is not set')
    try:
        with psycopg.connect(dsn, autocommit=True, connect_timeout=5) as conn:
            # xdist workers share the database, so they take turns migrating it.
            conn.execute('SELECT pg_advisory_lock(8127)')
            _migrate(dsn)
            conn.execute('SELECT pg_advisory_unlock(8127)')
    # Only when the configured server is unreachable, which CI never is.
    except psycopg.OperationalError as exc:  # pragma: no cover
        message = f'PostgreSQL is unreachable at PGTASK_TEST_DATABASE_URL: {exc}'
        if os.environ.get('PGTASK_REQUIRE_LIVE', '').lower() in {'1', 'true', 'yes'}:
            pytest.fail(message)
        pytest.skip(message)
    return dsn


# pgtask's `Client` cannot be closed, so each test process reuses one rather than leaving a pool
# behind per test.
_clients: dict[str, Client] = {}


@pytest.fixture
async def pgtask_db(pgtask_dsn: str) -> PGTaskDatabase:
    """The test process's pgtask client on the migrated database."""
    if pgtask_dsn not in _clients:
        _clients[pgtask_dsn] = await Client.connect(pgtask_dsn, max_query_connections=1)
    return PGTaskDatabase(pgtask_dsn, _clients[pgtask_dsn])
