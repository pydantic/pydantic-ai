"""Live PostgreSQL tests for `PostgresStepStore` and `PostgresMediaStore`.

Both stores take a caller-owned asyncpg-compatible pool and speak SQL to it, and
there is no in-process Postgres to run them against. The unit suites under
`tests/harness` run the stores against an in-memory SQLite stand-in for the pool,
which rewrites the few Postgres-only spellings. This file runs every read, write,
ordering, idempotency and retention rule against a real server instead, along
with the edges only a real server shows:

- parameter binding: a `run_id` and metadata holding a quote, a backslash and the
  text `$1` are stored and returned unchanged;
- a NUL byte in a `TEXT` value is refused by the server rather than truncated;
- `CREATE TABLE IF NOT EXISTS` racing from two store instances on one pool;
- ordering by an identity `seq`, not by `step_index` or arrival in a heap scan.

Run against a local server with `make integration-postgres` after starting one,
e.g. `docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:17`, or
point `POSTGRES_TEST_URL` at an existing database. Without a reachable server the
tests skip, unless `POSTGRES_REQUIRE_LIVE` is set (CI does), where an unreachable
server fails instead -- a service container that never came up must not pass as
a silent skip.

Every test works in tables whose names start with a prefix generated for that
test, and drops exactly those tables afterwards, so the suite can share a
database with other data. `POSTGRES_TEST_URL` carries credentials, so it never
appears in a skip or failure message: only the exception type does.

External assumptions:

- PostgreSQL text types cannot store the NUL character; the server rejects it
  with SQLSTATE 22021 (`character_not_in_repertoire`), which asyncpg raises as
  `CharacterNotInRepertoireError`. Source:
  <https://www.postgresql.org/docs/current/datatype-character.html>.
- Identifiers longer than 63 bytes are truncated, which is why table names are
  length-checked at construction. Source:
  <https://www.postgresql.org/docs/current/sql-syntax-lexical.html#SQL-SYNTAX-IDENTIFIERS>.
- `CREATE TABLE IF NOT EXISTS` is not safe against a concurrent creator of the
  same table: the loser can fail on a system catalog unique index instead of
  seeing the table. The stores serialize schema creation for that reason.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta, timezone
from typing import NoReturn
from uuid import uuid4

import anyio
import asyncpg
import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.conversation_search import SnapshotHistorySource
from pydantic_ai_harness.media import (
    MediaContext,
    MediaStore,
    PostgresMediaStore,
    PostgresPool,
    make_static_public_url,
    media_uri_for,
    parse_media_uri,
)
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    PostgresStepStore,
    RunRecord,
    StepEvent,
    StepPersistence,
    StepStore,
    ToolEffectRecord,
    ToolEffectStatus,
)

# Retrying is only worth it where a server is promised but may still be starting.
# Locally an absent server is the normal case, so one attempt is enough and the
# suite skips in seconds instead of stalling. In CI the window only has to cover
# the gap between the service container passing its health check and the server
# accepting connections -- a container that never becomes healthy fails the job
# before any step runs, so a longer window would only make the never-reachable
# case burn the job's whole timeout.
_CONNECT_RETRY_SECONDS = 30.0
_CONNECT_TIMEOUT_SECONDS = 5.0

# Nothing listens on port 1, so a connection is refused rather than left to time out.
_UNREACHABLE_URL = 'postgresql://127.0.0.1:1/none'

# URLs that already exhausted a connection attempt, with the failure they ended on.
# Every test opens its own pool, so without this an absent server would cost each
# of them the full attempt (and in CI the full retry window) instead of only the first.
_unreachable: dict[str, str] = {}

_MISSING_URI = 'media+sha256://' + ('0' * 64)

# Far past anything the driver would send inline as one small message, and past
# the 2 kB point where Postgres moves a value out of line (TOAST).
_LARGE_BLOB_BYTES = 4 * 1024 * 1024

# Keys and values that are only safe because they travel as bound parameters and
# as JSON: a dotted key, a `$`-prefixed key, quotes, a backslash, and non-ASCII.
_AWKWARD_METADATA = {
    'plain': 'value',
    'dotted.key': 'a.b.c',
    '$dollar': 'not a parameter',
    'it\'s "quoted"': 'back\\slash',
    'ünïcø∂é': 'naïve café \U0001f389',
}

# A quote, a backslash and a parameter placeholder: each would change the meaning
# of a statement that was assembled by string formatting.
_HOSTILE_TEXT = 'it\'s a "run" \\ $1; DROP TABLE runs; --'


@pytest.fixture
def anyio_backend() -> str:
    """Run live server tests once under asyncio, the only loop asyncpg supports."""
    return 'asyncio'


def _requires_live() -> bool:
    return os.environ.get('POSTGRES_REQUIRE_LIVE', '').lower() in {'1', 'true', 'yes'}


def _postgres_url() -> str:
    return os.environ.get('POSTGRES_TEST_URL', 'postgresql://postgres:postgres@127.0.0.1:5432/postgres')


def _unavailable(message: str) -> NoReturn:
    if _requires_live():
        pytest.fail(message)
    pytest.skip(message)


async def _connect(*, retry_seconds: float = _CONNECT_RETRY_SECONDS) -> asyncpg.Pool:
    """Return an open pool, retrying until the deadline when a server is required.

    A container health check reports the process listening; the server may still
    refuse the first connection while it finishes starting. Retrying pool
    creation is what makes the suite independent of that gap.
    """
    url = _postgres_url()
    deadline = time.monotonic() + (retry_seconds if _requires_live() else 0.0)
    while url not in _unreachable:
        try:
            return await asyncpg.create_pool(  # pyright: ignore[reportUnknownMemberType]
                url, min_size=1, max_size=4, timeout=_CONNECT_TIMEOUT_SECONDS
            )
        except (TimeoutError, OSError, asyncpg.PostgresError, asyncpg.InterfaceError) as exc:
            # Only the type: the message of a connection error can quote the URL.
            failure = type(exc).__name__
        if time.monotonic() >= deadline:
            _unreachable[url] = failure
            break
        await anyio.sleep(1.0)
    _unavailable(f'No PostgreSQL server reachable through POSTGRES_TEST_URL ({_unreachable[url]}).')


def _derived_tables(prefix: str) -> list[str]:
    """Every table a test may create from `prefix`, and nothing else."""
    return [
        prefix,
        f'{prefix}_runs',
        f'{prefix}_events',
        f'{prefix}_snapshots',
        f'{prefix}_snapshot_keys',
        f'{prefix}_tool_effects',
        f'{prefix}_media',
    ]


def _as_store_pool(pool: asyncpg.Pool) -> PostgresPool:
    """The single type-boundary shim between asyncpg and the stores' pool protocol.

    An asyncpg pool hands out a `PoolConnectionProxy`, which forwards the
    connection methods at runtime through a metaclass, so a type checker sees
    none of them and cannot match the pool against `PostgresPool`.
    """
    return pool  # pyright: ignore[reportReturnType]


@asynccontextmanager
async def _live_tables() -> AsyncGenerator[tuple[PostgresPool, str], None]:
    """Yield a pool and a table prefix unique to one test; drop its tables afterwards."""
    driver_pool = await _connect()
    pool = _as_store_pool(driver_pool)
    prefix = f'it_{uuid4().hex[:12]}'
    try:
        yield pool, prefix
    finally:
        try:
            async with pool.acquire() as connection:
                for table in _derived_tables(prefix):
                    await connection.execute(f'DROP TABLE IF EXISTS {table}')
        finally:
            await driver_pool.close()


async def _count_rows(pool: PostgresPool, table: str) -> int:
    async with pool.acquire() as connection:
        count = await connection.fetchval(f'SELECT count(*) FROM {table}')
    assert isinstance(count, int)
    return count


async def _table_exists(pool: PostgresPool, table: str) -> bool:
    async with pool.acquire() as connection:
        exists = await connection.fetchval('SELECT to_regclass($1) IS NOT NULL', table)
    assert isinstance(exists, bool)
    return exists


def _user_messages(text: str = 'a') -> list[ModelMessage]:
    return [ModelRequest(parts=[UserPromptPart(content=text)])]


# ---------------------------------------------------------------------------
# The suite's own availability handling
# ---------------------------------------------------------------------------


async def test_live_tests_skip_locally_without_a_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer without a running Postgres gets a skip, not a failure."""
    monkeypatch.delenv('POSTGRES_REQUIRE_LIVE', raising=False)
    monkeypatch.setenv('POSTGRES_TEST_URL', _UNREACHABLE_URL)
    monkeypatch.setattr(sys.modules[__name__], '_unreachable', {})

    with pytest.raises(pytest.skip.Exception) as skipped:
        await _connect()

    assert _UNREACHABLE_URL not in str(skipped.value)


async def test_live_tests_fail_in_ci_without_a_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """CI sets POSTGRES_REQUIRE_LIVE so a service container that never came up goes red."""
    monkeypatch.setenv('POSTGRES_REQUIRE_LIVE', '1')
    monkeypatch.setenv('POSTGRES_TEST_URL', _UNREACHABLE_URL)
    monkeypatch.setattr(sys.modules[__name__], '_unreachable', {})

    with pytest.raises(pytest.fail.Exception) as failed:
        await _connect(retry_seconds=0.0)

    assert _UNREACHABLE_URL not in str(failed.value)


def test_live_tests_run_on_asyncio_backend(anyio_backend: str) -> None:
    """Live server tests should not duplicate slow round trips under trio."""
    assert anyio_backend == 'asyncio'


# ---------------------------------------------------------------------------
# PostgresMediaStore
# ---------------------------------------------------------------------------


async def test_media_put_get_round_trip() -> None:
    """Bytes that are not valid UTF-8 come back identical from a `BYTEA` column."""
    async with _live_tables() as (pool, prefix):
        store = PostgresMediaStore(pool, table=f'{prefix}_media')
        data = b'hello postgres bytes \x00\xff'

        uri = await store.put(data, context=MediaContext(media_type='application/octet-stream'))

        assert uri == media_uri_for(data)
        assert await store.get(uri) == data


async def test_media_second_put_is_a_no_op() -> None:
    """The digest is the primary key, so a repeat `put` neither fails nor overwrites."""
    async with _live_tables() as (pool, prefix):
        table = f'{prefix}_media'
        store = PostgresMediaStore(pool, table=table)
        data = b'duplicate me'

        first = await store.put(data, context=MediaContext(media_type='text/plain', metadata={'writer': 'first'}))
        second = await store.put(data, context=MediaContext(media_type='image/png', metadata={'writer': 'second'}))

        assert first == second
        assert await _count_rows(pool, table) == 1
        assert await store.get_metadata(first) == {'writer': 'first'}
        assert await store.get(first) == data


async def test_media_empty_blob_round_trips() -> None:
    """Zero bytes are a stored row of size 0, not a missing one."""
    async with _live_tables() as (pool, prefix):
        table = f'{prefix}_media'
        store = PostgresMediaStore(pool, table=table)

        uri = await store.put(b'')

        assert await store.get(uri) == b''
        assert await store.exists(uri) is True
        async with pool.acquire() as connection:
            size = await connection.fetchval(f'SELECT size_bytes FROM {table} WHERE sha256 = $1', parse_media_uri(uri))
        assert size == 0


async def test_media_multi_megabyte_blob_round_trips() -> None:
    """A blob far past the out-of-line storage threshold reads back whole."""
    data = os.urandom(_LARGE_BLOB_BYTES)
    async with _live_tables() as (pool, prefix):
        table = f'{prefix}_media'
        store = PostgresMediaStore(pool, table=table)

        uri = await store.put(data)

        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                f'SELECT size_bytes, octet_length(bytes) FROM {table} WHERE sha256 = $1', parse_media_uri(uri)
            )
        assert row is not None
        assert list(row) == [_LARGE_BLOB_BYTES, _LARGE_BLOB_BYTES]
        assert await store.get(uri) == data


async def test_media_exists() -> None:
    """`exists` tells a stored digest from an absent one."""
    async with _live_tables() as (pool, prefix):
        store = PostgresMediaStore(pool, table=f'{prefix}_media')

        uri = await store.put(b'present')

        assert await store.exists(uri) is True
        assert await store.exists(_MISSING_URI) is False


async def test_media_missing_get_and_get_metadata_raise() -> None:
    """An absent digest raises `FileNotFoundError` from both readers."""
    async with _live_tables() as (pool, prefix):
        store = PostgresMediaStore(pool, table=f'{prefix}_media')

        with pytest.raises(FileNotFoundError):
            await store.get(_MISSING_URI)
        with pytest.raises(FileNotFoundError):
            await store.get_metadata(_MISSING_URI)


async def test_media_metadata_round_trips() -> None:
    """Metadata is returned as written, and is empty when none was supplied."""
    async with _live_tables() as (pool, prefix):
        store = PostgresMediaStore(pool, table=f'{prefix}_media')

        tagged = await store.put(b'tagged', context=MediaContext(media_type='image/png', metadata=_AWKWARD_METADATA))
        untagged = await store.put(b'no tags')

        assert await store.get_metadata(tagged) == _AWKWARD_METADATA
        assert await store.get_metadata(untagged) == {}


async def test_media_custom_table_name() -> None:
    """`table=` is the whole table name, so the bare prefix is a valid one."""
    async with _live_tables() as (pool, prefix):
        store = PostgresMediaStore(pool, table=prefix)

        uri = await store.put(b'in a custom table')

        assert await _count_rows(pool, prefix) == 1
        assert await _table_exists(pool, f'{prefix}_media') is False
        assert await store.get(uri) == b'in a custom table'


async def test_media_public_url_resolvers() -> None:
    """`public_url` is `None` without a resolver and awaits an async one."""

    async def signed(uri: str, context: MediaContext) -> str | None:
        return f'https://signed.example.com/{parse_media_uri(uri)}'

    async with _live_tables() as (pool, prefix):
        table = f'{prefix}_media'
        uri = await PostgresMediaStore(pool, table=table).put(b'p')
        digest = parse_media_uri(uri)

        assert await PostgresMediaStore(pool, table=table).public_url(uri) is None
        static = PostgresMediaStore(pool, table=table, public_url=make_static_public_url('https://cdn.example.com'))
        assert await static.public_url(uri) == f'https://cdn.example.com/{digest}.bin'
        assert (
            await PostgresMediaStore(pool, table=table, public_url=signed).public_url(uri)
            == f'https://signed.example.com/{digest}'
        )


async def test_media_concurrent_schema_initialization() -> None:
    """Two instances creating the same table at once both succeed."""
    async with _live_tables() as (pool, prefix):
        table = f'{prefix}_media'
        first = PostgresMediaStore(pool, table=table)
        second = PostgresMediaStore(pool, table=table)

        async with anyio.create_task_group() as tg:
            tg.start_soon(first.put, b'from the first store')
            tg.start_soon(second.put, b'from the second store')

        assert await _count_rows(pool, table) == 2
        assert await second.get(media_uri_for(b'from the first store')) == b'from the first store'
        assert await first.get(media_uri_for(b'from the second store')) == b'from the second store'


async def test_media_hostile_metadata_is_bound_not_interpolated() -> None:
    """A quote, a backslash and `$1` are stored verbatim."""
    async with _live_tables() as (pool, prefix):
        table = f'{prefix}_media'
        store = PostgresMediaStore(pool, table=table)

        uri = await store.put(
            b'x', context=MediaContext(media_type=_HOSTILE_TEXT, metadata={_HOSTILE_TEXT: _HOSTILE_TEXT})
        )

        assert await store.get_metadata(uri) == {_HOSTILE_TEXT: _HOSTILE_TEXT}
        async with pool.acquire() as connection:
            media_type = await connection.fetchval(
                f'SELECT media_type FROM {table} WHERE sha256 = $1', parse_media_uri(uri)
            )
        assert media_type == _HOSTILE_TEXT


async def test_media_null_byte_in_text_is_rejected_by_the_server() -> None:
    """A NUL byte in a `TEXT` column fails loudly."""
    async with _live_tables() as (pool, prefix):
        store = PostgresMediaStore(pool, table=f'{prefix}_media')
        data = b'never stored'

        with pytest.raises(asyncpg.CharacterNotInRepertoireError):
            await store.put(data, context=MediaContext(media_type='image/\x00png'))

        # The failed statement rolled back and the connection went back to the pool usable.
        assert await store.exists(media_uri_for(data)) is False


# ---------------------------------------------------------------------------
# PostgresStepStore: runs, events, snapshots, tool effects
# ---------------------------------------------------------------------------


async def test_step_store_satisfies_protocols() -> None:
    """The stores pass the runtime protocol checks the capability relies on."""
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix)

        assert isinstance(store, StepStore)
        assert isinstance(PostgresMediaStore(pool, table=f'{prefix}_media'), MediaStore)


async def test_step_store_creates_only_prefixed_tables() -> None:
    """Every table the store creates is one derived from `table=`."""
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix)
        await store.register_run(RunRecord(run_id='r1'))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_user_messages()))

        async with pool.acquire() as connection:
            rows = await connection.fetch(
                "SELECT table_name FROM information_schema.tables WHERE table_name LIKE $1 ESCAPE '\\'",
                prefix.replace('_', '\\_') + '%',
            )
        created = {str(row[0]) for row in rows}
        assert created <= set(_derived_tables(prefix))
        assert {f'{prefix}_runs', f'{prefix}_snapshots'} <= created


async def test_register_and_get_run() -> None:
    """Every `RunRecord` field survives the round trip, microseconds included."""
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix, media_store=None)
        started_at = datetime(2024, 5, 6, 7, 8, 9, 123456, tzinfo=UTC)
        record = RunRecord(
            run_id='r1',
            conversation_id='c1',
            parent_run_id='p1',
            agent_name='agent',
            metadata={'k': 'v'},
            started_at=started_at,
            registration_id='reg-1',
        )

        await store.register_run(record)

        assert await store.get_run(run_id='r1') == record


async def test_register_duplicate_run_raises_value_error() -> None:
    """A reused `run_id` is refused and the first record is kept."""
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix, media_store=None)
        await store.register_run(RunRecord(run_id='r1', agent_name='first'))

        with pytest.raises(ValueError, match='is already in the store'):
            await store.register_run(RunRecord(run_id='r1', agent_name='racing-run'))

        fetched = await store.get_run(run_id='r1')
        assert fetched is not None
        assert fetched.agent_name == 'first'
        assert await _count_rows(pool, f'{prefix}_runs') == 1


async def test_list_runs_chronological() -> None:
    """Runs come back by `started_at`, not by insertion order."""
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix, media_store=None)
        base = datetime(2024, 1, 1, tzinfo=UTC)
        await store.register_run(RunRecord(run_id='r3', started_at=base + timedelta(seconds=3)))
        await store.register_run(RunRecord(run_id='r1', started_at=base + timedelta(seconds=1)))
        await store.register_run(RunRecord(run_id='r2', started_at=base + timedelta(seconds=2)))

        assert [r.run_id for r in await store.list_runs()] == ['r1', 'r2', 'r3']


async def test_list_runs_filters() -> None:
    """Each filter narrows on its own and both AND-combine."""
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix, media_store=None)
        await store.register_run(RunRecord(run_id='r1', conversation_id='a', parent_run_id='p'))
        await store.register_run(RunRecord(run_id='r2', conversation_id='a', parent_run_id='q'))
        await store.register_run(RunRecord(run_id='r3', conversation_id='b', parent_run_id='p'))

        assert {r.run_id for r in await store.list_runs(conversation_id='a')} == {'r1', 'r2'}
        assert {r.run_id for r in await store.list_runs(parent_run_id='p')} == {'r1', 'r3'}
        assert [r.run_id for r in await store.list_runs(parent_run_id='p', conversation_id='a')] == ['r1']
        assert await store.list_runs(parent_run_id='q', conversation_id='b') == []


async def test_list_runs_sorts_by_instant_not_iso_string() -> None:
    """Mixed-offset timestamps sort by instant, matching the base stores' contract.

    A lexicographic sort of the ISO string would put `late` first; by instant
    `early` (an earlier moment behind a +05:00 offset) comes first.
    """
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix, media_store=None)
        early_instant = datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone(timedelta(hours=5)))  # 2023-12-31T20:00Z
        late_instant = datetime(2024, 1, 1, 0, 30, 0, tzinfo=UTC)
        await store.register_run(RunRecord(run_id='late', started_at=late_instant))
        await store.register_run(RunRecord(run_id='early', started_at=early_instant))

        runs = await store.list_runs()

        assert [r.run_id for r in runs] == ['early', 'late']
        assert [r.started_at for r in runs] == [early_instant, late_instant]


async def test_append_and_list_events() -> None:
    """Events come back in write order with every field, scoped to the run."""
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix, media_store=None)
        await store.register_run(RunRecord(run_id='r1'))
        started = StepEvent(run_id='r1', kind='run_started', step_index=0, conversation_id='c1', agent_name='agent')
        tool_call = StepEvent(
            run_id='r1',
            kind='tool_call_started',
            step_index=1,
            tool_call_id='t1',
            tool_name='add',
            metadata={'k': 'v'},
        )
        failed = StepEvent(run_id='r1', kind='run_failed', step_index=2, parent_run_id='p1', error='boom')

        for event in (started, tool_call, failed):
            await store.append_event(event)
        await store.append_event(StepEvent(run_id='r2', kind='run_started', step_index=0))

        assert await store.list_events(run_id='r1') == [started, tool_call, failed]


async def test_events_are_ordered_by_write_not_step_index() -> None:
    """The identity `seq` orders events, whatever `step_index` says."""
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix, media_store=None)
        for step in (5, 0, 3):
            await store.append_event(StepEvent(run_id='r1', kind='model_request_started', step_index=step))

        assert [e.step_index for e in await store.list_events(run_id='r1')] == [5, 0, 3]


async def test_keyed_event_replay_is_suppressed_but_unkeyed_events_append() -> None:
    """A repeated key is written once per run; events without a key always append."""
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix, media_store=None)
        keyed = StepEvent(run_id='r1', kind='run_started', step_index=0, idempotency_key='event:0')
        await store.append_event(keyed)
        await store.append_event(keyed)
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0))
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0))
        # The key is scoped to the run: the same key under another run is a new event.
        await store.append_event(StepEvent(run_id='r2', kind='run_started', step_index=0, idempotency_key='event:0'))

        assert len(await store.list_events(run_id='r1')) == 3
        assert len(await store.list_events(run_id='r2')) == 1
        assert await _count_rows(pool, f'{prefix}_events') == 4


async def test_save_and_load_snapshot() -> None:
    """Every `ContinuableSnapshot` field survives the round trip."""
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix, media_store=None)
        await store.register_run(RunRecord(run_id='r1'))
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='hello')]),
            ModelResponse(parts=[TextPart(content='hi back')]),
        ]
        snapshot = ContinuableSnapshot(
            run_id='r1',
            step_index=2,
            messages=messages,
            conversation_id='c1',
            parent_run_id='p1',
            agent_name='agent',
            timestamp=datetime(2024, 5, 6, 7, 8, 9, 123456, tzinfo=UTC),
        )

        await store.save_snapshot(snapshot)

        assert await store.latest_snapshot(run_id='r1') == snapshot


async def test_latest_snapshot_default_skips_newer_interrupted() -> None:
    """The default read returns the newest `complete`; opting in returns the newest of any state."""
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix, media_store=None)
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_user_messages('settled')))
        await store.save_snapshot(
            ContinuableSnapshot(run_id='r1', step_index=1, messages=_user_messages('frontier'), state='interrupted')
        )

        default = await store.latest_snapshot(run_id='r1')
        assert default is not None and default.state == 'complete' and default.step_index == 0
        opted = await store.latest_snapshot(run_id='r1', include_interrupted=True)
        assert opted is not None and opted.state == 'interrupted' and opted.step_index == 1


async def test_only_interrupted_defaults_to_none() -> None:
    """A run with only `interrupted` snapshots has no default resume point."""
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix, media_store=None)
        await store.save_snapshot(
            ContinuableSnapshot(run_id='r1', step_index=0, messages=_user_messages(), state='interrupted')
        )

        assert await store.latest_snapshot(run_id='r1') is None
        assert await store.latest_snapshot(run_id='r1', include_interrupted=True) is not None
        assert await store.list_snapshots(run_id='r1') == []


async def test_snapshot_seq_monotonic_across_reset_step() -> None:
    """A reused run_id whose step_index resets to 0 must not clobber the prior snapshot."""
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix, media_store=None)
        await store.register_run(RunRecord(run_id='r1'))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=5, messages=_user_messages()))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_user_messages()))

        snap = await store.latest_snapshot(run_id='r1')

        assert snap is not None
        assert snap.step_index == 0  # last write wins, not the highest step_index
        assert [s.step_index for s in await store.list_snapshots(run_id='r1')] == [5, 0]


async def test_keyed_snapshot_replays_preserve_complete_and_interrupted_at_same_step() -> None:
    """Two keys at one `step_index` are two snapshots, and replaying either adds nothing."""
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix, media_store=None)
        complete = ContinuableSnapshot(
            run_id='r1', step_index=2, messages=_user_messages(), state='complete', idempotency_key='2:complete'
        )
        interrupted = ContinuableSnapshot(
            run_id='r1', step_index=2, messages=_user_messages(), state='interrupted', idempotency_key='2:interrupted'
        )

        for snapshot in (complete, interrupted, complete, interrupted):
            await store.save_snapshot(snapshot)

        snapshots = await store.list_snapshots(run_id='r1', include_interrupted=True)
        assert [(s.step_index, s.state) for s in snapshots] == [(2, 'complete'), (2, 'interrupted')]
        assert await store.latest_snapshot(run_id='r1') == complete


async def test_unkeyed_snapshots_always_append() -> None:
    """Without a key there is nothing to deduplicate on."""
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix, media_store=None)
        snapshot = ContinuableSnapshot(run_id='r1', step_index=0, messages=_user_messages())

        await store.save_snapshot(snapshot)
        await store.save_snapshot(snapshot)

        assert await _count_rows(pool, f'{prefix}_snapshots') == 2


async def test_keyed_writes_are_idempotent_after_snapshot_pruning() -> None:
    """The key ledger outlives the snapshot row, so a late replay cannot resurrect it."""
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix, media_store=None, max_snapshots_per_run=1)
        event = StepEvent(run_id='r1', kind='run_started', step_index=0, idempotency_key='event:0')
        older = ContinuableSnapshot(run_id='r1', step_index=1, messages=[], idempotency_key='0:1:complete')
        newer = ContinuableSnapshot(run_id='r1', step_index=2, messages=[], idempotency_key='1:2:complete')

        await store.append_event(event)
        await store.append_event(event)
        await store.save_snapshot(older)
        await store.save_snapshot(newer)
        await store.save_snapshot(older)

        assert await store.list_events(run_id='r1') == [event]
        assert await store.latest_snapshot(run_id='r1') == newer
        assert await store.list_snapshots(run_id='r1') == [newer]
        assert await _count_rows(pool, f'{prefix}_snapshots') == 1
        assert await _count_rows(pool, f'{prefix}_snapshot_keys') == 2


async def test_tool_effect_upsert_and_scope() -> None:
    """A second record for one call replaces the first, per run."""
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix, media_store=None)
        started_at = datetime(2024, 5, 6, 7, 8, 9, tzinfo=UTC)
        completed = ToolEffectRecord(
            tool_call_id='t1',
            tool_name='add',
            run_id='r1',
            status='completed',
            started_at=started_at,
            ended_at=started_at + timedelta(seconds=2),
            idempotency_key='effect-1',
            effect_summary='ok',
        )
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='add', run_id='r1', status='started', started_at=started_at)
        )
        await store.record_tool_effect(completed)
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='add', run_id='r2', status='started')
        )

        assert await store.get_tool_effect(run_id='r1', tool_call_id='t1') == completed
        other = await store.get_tool_effect(run_id='r2', tool_call_id='t1')
        assert other is not None and other.status == 'started'
        assert await _count_rows(pool, f'{prefix}_tool_effects') == 2


async def test_list_unresolved_tool_effects() -> None:
    """Only `started` effects of the asked run are unresolved."""
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix, media_store=None)
        effects: list[tuple[str, str, ToolEffectStatus]] = [
            ('t1', 'r1', 'started'),
            ('t2', 'r1', 'completed'),
            ('t3', 'r1', 'failed'),
            ('t4', 'r2', 'started'),
        ]
        for tool_call_id, run_id, status in effects:
            await store.record_tool_effect(
                ToolEffectRecord(tool_call_id=tool_call_id, tool_name='add', run_id=run_id, status=status)
            )

        unresolved = await store.list_unresolved_tool_effects(run_id='r1')

        assert [r.tool_call_id for r in unresolved] == ['t1']


async def test_missing_lookups_return_none_or_empty() -> None:
    """Reads on an empty store create the schema and find nothing."""
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix, media_store=None)

        assert await store.get_run(run_id='nope') is None
        assert await store.latest_snapshot(run_id='nope') is None
        assert await store.latest_snapshot(run_id='nope', include_interrupted=True) is None
        assert await store.list_snapshots(run_id='nope') == []
        assert await store.get_tool_effect(run_id='nope', tool_call_id='x') is None
        assert await store.list_events(run_id='nope') == []
        assert await store.list_unresolved_tool_effects(run_id='nope') == []
        assert await store.list_runs() == []


async def test_non_ascii_and_awkward_metadata_round_trips() -> None:
    """Non-ASCII and awkward keys survive in runs and events."""
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix, media_store=None)
        metadata = {'user': 'Ada Lovelace ✨', 'ключ': 'ジョブ完了', 'emoji': '🙂', **_AWKWARD_METADATA}
        await store.register_run(RunRecord(run_id='r1', agent_name='agént', metadata=metadata))
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0, metadata=metadata))

        run = await store.get_run(run_id='r1')

        assert run is not None
        assert run.metadata == metadata
        assert run.agent_name == 'agént'
        assert [e.metadata for e in await store.list_events(run_id='r1')] == [metadata]


async def test_hostile_identifiers_are_bound_not_interpolated() -> None:
    """A quote, a backslash and `$1` in ids and metadata come back unchanged."""
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix, media_store=None)
        run = RunRecord(
            run_id=_HOSTILE_TEXT,
            conversation_id=_HOSTILE_TEXT,
            parent_run_id=_HOSTILE_TEXT,
            agent_name=_HOSTILE_TEXT,
            metadata={_HOSTILE_TEXT: _HOSTILE_TEXT},
        )
        event = StepEvent(
            run_id=_HOSTILE_TEXT,
            kind='tool_call_failed',
            step_index=0,
            tool_call_id=_HOSTILE_TEXT,
            tool_name=_HOSTILE_TEXT,
            error=_HOSTILE_TEXT,
            metadata={_HOSTILE_TEXT: _HOSTILE_TEXT},
            idempotency_key=_HOSTILE_TEXT,
        )
        snapshot = ContinuableSnapshot(
            run_id=_HOSTILE_TEXT,
            step_index=0,
            messages=_user_messages(_HOSTILE_TEXT),
            idempotency_key=_HOSTILE_TEXT,
        )
        effect = ToolEffectRecord(
            tool_call_id=_HOSTILE_TEXT,
            tool_name=_HOSTILE_TEXT,
            run_id=_HOSTILE_TEXT,
            status='started',
            effect_summary=_HOSTILE_TEXT,
        )

        await store.register_run(run)
        await store.register_run(RunRecord(run_id='plain'))
        await store.append_event(event)
        await store.save_snapshot(snapshot)
        await store.record_tool_effect(effect)

        assert await store.get_run(run_id=_HOSTILE_TEXT) == run
        assert await store.list_runs(conversation_id=_HOSTILE_TEXT, parent_run_id=_HOSTILE_TEXT) == [run]
        assert await store.list_events(run_id=_HOSTILE_TEXT) == [event]
        assert await store.latest_snapshot(run_id=_HOSTILE_TEXT) == snapshot
        assert await store.get_tool_effect(run_id=_HOSTILE_TEXT, tool_call_id=_HOSTILE_TEXT) == effect
        assert await store.list_unresolved_tool_effects(run_id=_HOSTILE_TEXT) == [effect]
        assert await _count_rows(pool, f'{prefix}_runs') == 2


async def test_null_byte_in_text_is_rejected_by_the_server() -> None:
    """A NUL byte in a `TEXT` column fails loudly."""
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix, media_store=None)

        with pytest.raises(asyncpg.CharacterNotInRepertoireError):
            await store.register_run(RunRecord(run_id='run-null', agent_name='a\x00b'))

        # Nothing was written, and the connection went back to the pool usable.
        assert await store.get_run(run_id='run-null') is None
        assert await store.list_runs() == []


async def test_concurrent_schema_initialization() -> None:
    """Two instances creating the same tables at once both succeed."""
    async with _live_tables() as (pool, prefix):
        first = PostgresStepStore(pool, table=prefix)
        second = PostgresStepStore(pool, table=prefix)

        async with anyio.create_task_group() as tg:
            tg.start_soon(first.register_run, RunRecord(run_id='from-first'))
            tg.start_soon(second.register_run, RunRecord(run_id='from-second'))

        assert {r.run_id for r in await first.list_runs()} == {'from-first', 'from-second'}
        assert {r.run_id for r in await second.list_runs()} == {'from-first', 'from-second'}


async def test_concurrent_keyed_snapshot_is_written_once() -> None:
    """The same keyed save racing from two instances leaves one row."""
    async with _live_tables() as (pool, prefix):
        first = PostgresStepStore(pool, table=prefix, media_store=None)
        second = PostgresStepStore(pool, table=prefix, media_store=None)
        snapshot = ContinuableSnapshot(
            run_id='r1', step_index=0, messages=_user_messages(), idempotency_key='0:0:complete'
        )
        event = StepEvent(run_id='r1', kind='run_started', step_index=0, idempotency_key='event:0')

        async with anyio.create_task_group() as tg:
            for store in (first, second):
                tg.start_soon(store.save_snapshot, snapshot)
                tg.start_soon(store.append_event, event)

        assert await first.list_snapshots(run_id='r1') == [snapshot]
        assert await first.list_events(run_id='r1') == [event]


# ---------------------------------------------------------------------------
# PostgresStepStore: list_snapshots
# ---------------------------------------------------------------------------


async def test_list_snapshots_write_order_and_interrupted_filter() -> None:
    """Snapshots list in write order, with `interrupted` ones only on request."""
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix, media_store=None)
        # Descending then ascending `step_index` so write order is neither a
        # `step_index` sort nor its reverse.
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=2, messages=_user_messages()))
        await store.save_snapshot(
            ContinuableSnapshot(run_id='r1', step_index=0, messages=_user_messages(), state='interrupted')
        )
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=1, messages=_user_messages()))
        await store.save_snapshot(ContinuableSnapshot(run_id='r2', step_index=9, messages=_user_messages()))

        assert [s.step_index for s in await store.list_snapshots(run_id='r1')] == [2, 1]
        opted = await store.list_snapshots(run_id='r1', include_interrupted=True)
        assert [s.step_index for s in opted] == [2, 0, 1]
        assert [s.state for s in opted] == ['complete', 'interrupted', 'complete']
        assert await store.list_snapshots(run_id='nope') == []


async def test_store_is_accepted_as_a_search_substrate() -> None:
    """`SnapshotHistorySource` rejects stores lacking `list_snapshots` at construction."""
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix, media_store=None)
        await store.save_snapshot(
            ContinuableSnapshot(run_id='r1', step_index=0, messages=_user_messages('remember this'))
        )

        source = SnapshotHistorySource(store)

        assert [type(m).__name__ for m in await source.run_history(run_id='r1')] == ['ModelRequest']


# ---------------------------------------------------------------------------
# PostgresStepStore: retention
# ---------------------------------------------------------------------------


async def test_retention_none_keeps_every_snapshot() -> None:
    """Without a bound nothing is pruned."""
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix, media_store=None)
        for step in range(4):
            await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=step, messages=_user_messages()))

        assert [s.step_index for s in await store.list_snapshots(run_id='r1')] == [0, 1, 2, 3]


async def test_retention_under_the_limit_keeps_every_snapshot() -> None:
    """A bound that was not reached prunes nothing."""
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix, media_store=None, max_snapshots_per_run=5)
        for step in range(3):
            await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=step, messages=_user_messages()))

        assert [s.step_index for s in await store.list_snapshots(run_id='r1')] == [0, 1, 2]


async def test_retention_over_the_limit_prunes_to_the_newest_window() -> None:
    """Past the bound only the newest rows remain."""
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix, media_store=None, max_snapshots_per_run=2)
        for step in range(5):
            await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=step, messages=_user_messages()))

        assert [s.step_index for s in await store.list_snapshots(run_id='r1')] == [3, 4]
        assert await _count_rows(pool, f'{prefix}_snapshots') == 2


async def test_retention_keeps_newest_complete_below_an_interrupted_tail() -> None:
    """A `complete` snapshot pushed out of the window by `interrupted` writes survives."""
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix, media_store=None, max_snapshots_per_run=2)
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_user_messages()))
        for step in (1, 2, 3):
            await store.save_snapshot(
                ContinuableSnapshot(run_id='r1', step_index=step, messages=_user_messages(), state='interrupted')
            )

        retained = await store.list_snapshots(run_id='r1', include_interrupted=True)
        assert [s.step_index for s in retained] == [0, 2, 3]
        resumable = await store.latest_snapshot(run_id='r1')
        assert resumable is not None
        assert resumable.step_index == 0


async def test_retention_keep_one_with_newest_interrupted_keeps_two() -> None:
    """Both read modes stay served: the newest overall and the newest `complete`."""
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix, media_store=None, max_snapshots_per_run=1)
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_user_messages()))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=1, messages=_user_messages()))
        await store.save_snapshot(
            ContinuableSnapshot(run_id='r1', step_index=2, messages=_user_messages(), state='interrupted')
        )

        retained = await store.list_snapshots(run_id='r1', include_interrupted=True)
        assert [(s.step_index, s.state) for s in retained] == [(1, 'complete'), (2, 'interrupted')]
        default = await store.latest_snapshot(run_id='r1')
        assert default is not None and default.step_index == 1
        opted = await store.latest_snapshot(run_id='r1', include_interrupted=True)
        assert opted is not None and opted.step_index == 2


async def test_retention_is_scoped_to_the_written_run() -> None:
    """Pruning one run leaves another run's snapshots alone."""
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix, media_store=None, max_snapshots_per_run=1)
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_user_messages()))
        await store.save_snapshot(ContinuableSnapshot(run_id='r2', step_index=0, messages=_user_messages()))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=1, messages=_user_messages()))

        assert [s.step_index for s in await store.list_snapshots(run_id='r1')] == [1]
        assert [s.step_index for s in await store.list_snapshots(run_id='r2')] == [0]


# ---------------------------------------------------------------------------
# PostgresStepStore: media externalization
# ---------------------------------------------------------------------------


def _messages_with_media(big_binary: bytes, big_text: str) -> list[ModelMessage]:
    return [
        ModelRequest(
            parts=[
                UserPromptPart(content=[BinaryContent(data=big_binary, media_type='image/png')]),
                ToolReturnPart(tool_name='scrape', content=big_text, tool_call_id='t1'),
            ]
        ),
        ModelResponse(parts=[TextPart(content='done')]),
    ]


def _assert_media_restored(snapshot: ContinuableSnapshot | None, big_binary: bytes, big_text: str) -> None:
    assert snapshot is not None
    request = snapshot.messages[0]
    assert isinstance(request, ModelRequest)
    prompt = request.parts[0]
    assert isinstance(prompt, UserPromptPart)
    assert isinstance(prompt.content, list)
    binary = prompt.content[0]
    assert isinstance(binary, BinaryContent)
    assert binary.data == big_binary
    tool_return = request.parts[1]
    assert isinstance(tool_return, ToolReturnPart)
    assert tool_return.content == big_text


async def _stored_messages_length(pool: PostgresPool, prefix: str) -> int:
    async with pool.acquire() as connection:
        length = await connection.fetchval(f'SELECT octet_length(messages::text) FROM {prefix}_snapshots')
    assert isinstance(length, int)
    return length


async def test_large_binary_and_text_externalized_and_restored() -> None:
    """Both a large binary part and a large text tool-return round-trip through media."""
    big_binary = b'\xab' * 100_000
    big_text = 'Z' * 100_000
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix, media_threshold_bytes=64 * 1024)
        await store.register_run(RunRecord(run_id='r1'))

        await store.save_snapshot(
            ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages_with_media(big_binary, big_text))
        )

        # Two blobs (binary + text) live in the sibling media table, not in the snapshot row.
        assert await _count_rows(pool, f'{prefix}_media') == 2
        assert await _stored_messages_length(pool, prefix) < 64 * 1024
        _assert_media_restored(await store.latest_snapshot(run_id='r1'), big_binary, big_text)
        listed = await store.list_snapshots(run_id='r1')
        assert len(listed) == 1
        _assert_media_restored(listed[0], big_binary, big_text)


async def test_below_threshold_stays_inline() -> None:
    """A payload under the threshold is stored in the snapshot row."""
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix, media_threshold_bytes=64 * 1024)
        messages: list[ModelMessage] = [ModelResponse(parts=[TextPart(content='small')])]

        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=messages))

        # Nothing crossed the threshold, so the media store never ran a statement.
        if await _table_exists(pool, f'{prefix}_media'):
            assert await _count_rows(pool, f'{prefix}_media') == 0
        snap = await store.latest_snapshot(run_id='r1')
        assert snap is not None
        response = snap.messages[0]
        assert isinstance(response, ModelResponse)
        text = response.parts[0]
        assert isinstance(text, TextPart)
        assert text.content == 'small'


async def test_media_store_none_keeps_payloads_inline() -> None:
    """Without a media store no media table is created."""
    big_binary = b'\xab' * 100_000
    big_text = 'Z' * 100_000
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix, media_store=None)

        await store.save_snapshot(
            ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages_with_media(big_binary, big_text))
        )

        assert await _table_exists(pool, f'{prefix}_media') is False
        assert await _stored_messages_length(pool, prefix) > 200_000
        _assert_media_restored(await store.latest_snapshot(run_id='r1'), big_binary, big_text)


async def test_explicit_media_store_is_used() -> None:
    """A caller-supplied store replaces the `{table}_media` default."""
    big_binary = b'\xab' * 100_000
    big_text = 'Z' * 100_000
    async with _live_tables() as (pool, prefix):
        media_store = PostgresMediaStore(pool, table=prefix)
        store = PostgresStepStore(pool, table=prefix, media_store=media_store)

        await store.save_snapshot(
            ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages_with_media(big_binary, big_text))
        )

        assert await _count_rows(pool, prefix) == 2
        assert await _table_exists(pool, f'{prefix}_media') is False
        assert await media_store.get(media_uri_for(big_binary)) == big_binary
        _assert_media_restored(await store.latest_snapshot(run_id='r1'), big_binary, big_text)


async def test_agent_run_round_trips_through_step_persistence() -> None:
    """An Agent run with a large BinaryContent prompt persists and restores via Postgres."""
    big = b'\xab' * 100_000
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix)
        agent: Agent[None, str] = Agent(
            TestModel(),
            capabilities=[StepPersistence(store=store, agent_name='vision')],
        )

        result = await agent.run(['classify this image', BinaryContent(data=big, media_type='image/png')])

        assert isinstance(result.output, str)
        runs = await store.list_runs()
        assert len(runs) == 1
        assert runs[0].agent_name == 'vision'
        kinds = [event.kind for event in await store.list_events(run_id=runs[0].run_id)]
        assert kinds[0] == 'run_started'
        assert kinds[-1] == 'run_completed'
        assert await _count_rows(pool, f'{prefix}_media') == 1
        snap = await store.latest_snapshot(run_id=runs[0].run_id)
        assert snap is not None
        request = snap.messages[0]
        assert isinstance(request, ModelRequest)
        prompt = request.parts[0]
        assert isinstance(prompt, UserPromptPart)
        assert isinstance(prompt.content, list)
        binary = next(p for p in prompt.content if isinstance(p, BinaryContent))
        assert binary.data == big


# ---------------------------------------------------------------------------
# Rows and tables the stores did not write
# ---------------------------------------------------------------------------


async def test_media_one_instance_initializes_its_schema_once_under_concurrency() -> None:
    """Two first calls on one instance: the second waits on the lock and finds the schema ready."""
    async with _live_tables() as (pool, prefix):
        store = PostgresMediaStore(pool, table=f'{prefix}_media')

        async with anyio.create_task_group() as tg:
            tg.start_soon(store.put, b'first caller')
            tg.start_soon(store.put, b'second caller')

        assert await _count_rows(pool, f'{prefix}_media') == 2


async def test_media_foreign_table_layout_fails_loudly_on_get() -> None:
    """An existing table with another layout is left as is, and a read of it raises."""
    async with _live_tables() as (pool, prefix):
        table = f'{prefix}_media'
        digest = 'a' * 64
        async with pool.acquire() as connection:
            await connection.execute(
                f'CREATE TABLE {table} (sha256 TEXT PRIMARY KEY, bytes TEXT NOT NULL, metadata BYTEA)'
            )
            await connection.execute(
                f'INSERT INTO {table} (sha256, bytes, metadata) VALUES ($1, $2, $3)', digest, 'not bytea', b'not text'
            )
        store = PostgresMediaStore(pool, table=table)

        with pytest.raises(ValueError, match='has wrong types'):
            await store.get(f'media+sha256://{digest}')
        with pytest.raises(ValueError, match='has wrong types'):
            await store.get_metadata(f'media+sha256://{digest}')


async def test_list_snapshots_skips_an_unparsable_row(caplog: pytest.LogCaptureFixture) -> None:
    """One damaged row is logged and skipped, so the rest of the run's history stays readable."""
    async with _live_tables() as (pool, prefix):
        store = PostgresStepStore(pool, table=prefix, media_store=None)
        good = ContinuableSnapshot(run_id='r1', step_index=0, messages=_user_messages())
        await store.save_snapshot(good)
        async with pool.acquire() as connection:
            await connection.execute(
                f'INSERT INTO {prefix}_snapshots (run_id, step_index, timestamp, messages) VALUES ($1, $2, $3, $4)',
                'r1',
                1,
                datetime.now(UTC).isoformat(),
                'not json',
            )

        with caplog.at_level('WARNING'):
            snapshots = await store.list_snapshots(run_id='r1')

        assert snapshots == [good]
        assert 'Skipping unparsable snapshot row for run r1' in caplog.text


async def test_snapshot_table_with_a_foreign_layout_fails_loudly_on_read() -> None:
    """A pre-existing snapshots table with other column types is left as is, and a read of it raises."""
    async with _live_tables() as (pool, prefix):
        async with pool.acquire() as connection:
            await connection.execute(
                f'CREATE TABLE {prefix}_snapshots ('
                'seq BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, run_id TEXT NOT NULL, '
                'step_index TEXT NOT NULL, conversation_id TEXT, parent_run_id TEXT, agent_name TEXT, '
                "timestamp TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'complete', "
                'messages TEXT NOT NULL, idempotency_key TEXT)'
            )
            await connection.execute(
                f'INSERT INTO {prefix}_snapshots (run_id, step_index, timestamp, messages) VALUES ($1, $2, $3, $4)',
                'r1',
                'zero',
                datetime.now(UTC).isoformat(),
                '[]',
            )
        store = PostgresStepStore(pool, table=prefix, media_store=None)

        with pytest.raises(ValueError, match='snapshot row has wrong types'):
            await store.latest_snapshot(run_id='r1')
