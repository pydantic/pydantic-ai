"""Live mongod conformance tests for `MongoStepStore` and `MongoMediaStore`.

The unit suites under `tests/step_persistence` and `tests/media` run against
`mongomock-motor`, which models the collection API but not the storage engine.
Four behaviors it does not reproduce decide whether these backends are correct:

- the 16MB BSON document cap, which is the reason snapshots externalize media;
- real BSON key encoding (non-ASCII, dotted, and `$`-prefixed metadata keys);
- collection-scan ordering, which is why chunk reassembly sorts explicitly;
- server-side unique index enforcement on `tool_effects`.

This file covers exactly those. It is not a second copy of the unit suites --
API-shape coverage belongs there, where it runs on every matrix leg.

Run against a local server with `make integration-mongodb` after starting one,
e.g. `docker run -d -p 27017:27017 mongo:8`. Without a reachable server the
tests skip, unless `MONGODB_REQUIRE_LIVE` is set (CI does), where an unreachable
server fails instead -- a service container that never came up must not pass as
a silent skip.

External assumptions, verified 2026-07-28 against mongod 8.2.12 (`mongo:8`):

- Field names may contain `.` and may start with `$` since MongoDB 5.0; they are
  stored and returned verbatim. Source:
  <https://www.mongodb.com/docs/manual/core/dot-dollar-considerations/>.
- A NULL byte in a field name is rejected by the BSON encoder, so it never
  reaches the server.
- The document cap is 16MiB, but it is enforced in two places with a gap
  between them. PyMongo checks against `maxBsonObjectSize` plus a 16KiB command
  allowance (16793598 bytes on 8.2.12) and raises `DocumentTooLarge` without a
  round trip; the server rejects anything over 16777216 with a `WriteError`
  (code 10334). A document inside that ~16KiB window therefore fails on the
  server, not in the driver -- both loudly, but as different exception types.
  Source:
  <https://www.mongodb.com/docs/manual/reference/limits/#mongodb-limit-BSON-Document-Size>.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import NoReturn

import anyio
import pytest
from bson.errors import InvalidDocument
from pydantic import TypeAdapter
from pydantic_ai.messages import ModelResponse, TextPart
from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import DocumentTooLarge, DuplicateKeyError, PyMongoError, WriteError

from pydantic_ai_harness.media import MongoMediaStore, parse_media_uri
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    MongoStepStore,
    RunRecord,
    StepEvent,
    ToolEffectRecord,
)

_MongoDocument = dict[str, object]

_INDEX_INFO = TypeAdapter(dict[str, dict[str, object]])

# Retrying is only worth it where a server is promised but may still be starting.
# Locally an absent server is the normal case, so one attempt is enough and the
# suite skips in seconds instead of stalling. In CI the window only has to cover
# the gap between the service container passing its health check and this client
# finishing topology discovery -- a container that never becomes healthy fails
# the job before any step runs, so a longer window would only make the
# never-reachable case burn the job's whole timeout.
_CONNECT_RETRY_SECONDS = 30.0
_SERVER_SELECTION_TIMEOUT_MS = 2000

# One byte over the server's 16MiB cap: inside PyMongo's 16KiB command allowance,
# so this size reaches the wire and is refused by mongod.
_SERVER_REFUSED_BYTES = 16 * 1024 * 1024 + 1
# Past the driver's allowance too, so PyMongo refuses it without a round trip.
_DRIVER_REFUSED_BYTES = 16 * 1024 * 1024 + 64 * 1024
# Three chunks at `MongoMediaStore`'s untouched 8MB default, which also makes the
# blob itself larger than a single BSON document could hold.
_MULTI_CHUNK_BLOB_BYTES = 20 * 1024 * 1024

# Metadata keys that real BSON accepts but that a JSON-shaped fake need not
# distinguish: a dotted key, a `$`-prefixed key, and non-ASCII in both key and
# value.
_AWKWARD_METADATA = {
    'plain': 'value',
    'dotted.key': 'a.b.c',
    '$dollar': 'not an operator',
    'ünïcø∂é': 'naïve café \U0001f389',
}


@pytest.fixture
def anyio_backend() -> str:
    """Run live server tests once under asyncio."""
    return 'asyncio'


def _requires_live() -> bool:
    return os.environ.get('MONGODB_REQUIRE_LIVE', '').lower() in {'1', 'true', 'yes'}


def _mongodb_url() -> str:
    return os.environ.get('MONGODB_TEST_URL', 'mongodb://127.0.0.1:27017')


def _unavailable(message: str) -> NoReturn:
    if _requires_live():
        pytest.fail(message)
    pytest.skip(message)


async def _connect() -> AsyncMongoClient[_MongoDocument]:
    """Return a client that has completed a `ping`, retrying until the deadline.

    A container health check reports the process listening; it does not report
    that replica-set/topology discovery finished for this client. Retrying the
    first command is what makes the suite independent of that gap.
    """
    deadline = time.monotonic() + (_CONNECT_RETRY_SECONDS if _requires_live() else 0.0)
    # One client retried, rather than a fresh one per attempt: a discarded client
    # leaves its monitor socket to the garbage collector, and `filterwarnings =
    # ['error']` turns that finalization into a test error.
    client = AsyncMongoClient[_MongoDocument](_mongodb_url(), serverSelectionTimeoutMS=_SERVER_SELECTION_TIMEOUT_MS)
    failure = 'no connection attempt was made'
    while True:
        try:
            await client.admin.command('ping')
        except PyMongoError as exc:
            failure = f'{type(exc).__name__}: {exc}'
        else:
            return client
        if time.monotonic() >= deadline:
            break
        await anyio.sleep(1.0)
    await client.close()
    _unavailable(f'No MongoDB server at {_mongodb_url()} ({failure}).')


@asynccontextmanager
async def _live_database() -> AsyncGenerator[AsyncDatabase[_MongoDocument], None]:
    """Yield a database unique to one test, dropped afterwards."""
    client = await _connect()
    name = f'harness_it_{uuid.uuid4().hex}'
    try:
        yield client[name]
    finally:
        await client.drop_database(name)
        await client.close()


def _text_snapshot(run_id: str, text: str) -> ContinuableSnapshot:
    return ContinuableSnapshot(run_id=run_id, step_index=0, messages=[ModelResponse(parts=[TextPart(content=text)])])


def test_live_tests_skip_locally_without_a_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer without a running mongod gets a skip, not a failure."""
    monkeypatch.delenv('MONGODB_REQUIRE_LIVE', raising=False)

    with pytest.raises(pytest.skip.Exception):
        _unavailable('no server')


def test_live_tests_fail_in_ci_without_a_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """CI sets MONGODB_REQUIRE_LIVE so a service container that never came up goes red."""
    monkeypatch.setenv('MONGODB_REQUIRE_LIVE', '1')

    with pytest.raises(pytest.fail.Exception):
        _unavailable('no server')


def test_live_tests_run_on_asyncio_backend(anyio_backend: str) -> None:
    """Live server tests should not duplicate slow round trips under trio."""
    assert anyio_backend == 'asyncio'


@pytest.mark.parametrize(
    ('payload_bytes', 'expected'),
    [
        pytest.param(_SERVER_REFUSED_BYTES, WriteError, id='server-refused'),
        pytest.param(_DRIVER_REFUSED_BYTES, DocumentTooLarge, id='driver-refused'),
    ],
)
@pytest.mark.anyio(backends=['asyncio'])
async def test_oversized_snapshot_fails_loudly_without_media_externalization(
    payload_bytes: int, expected: type[PyMongoError]
) -> None:
    """A snapshot past the 16MB BSON cap raises instead of silently truncating."""
    async with _live_database() as db:
        store = MongoStepStore(client=db.client, database=db.name, media_store=None)
        snapshot = _text_snapshot('run-too-large', 'x' * payload_bytes)

        with pytest.raises(expected):
            await store.save_snapshot(snapshot)

        assert await store.latest_snapshot(run_id='run-too-large') is None


@pytest.mark.anyio(backends=['asyncio'])
async def test_media_externalization_saves_a_snapshot_past_the_bson_cap() -> None:
    """The same payload round-trips once large text is offloaded to the media store."""
    text = 'x' * _SERVER_REFUSED_BYTES
    async with _live_database() as db:
        store = MongoStepStore(client=db.client, database=db.name)

        await store.save_snapshot(_text_snapshot('run-externalized', text))

        stored = await db['snapshots'].find_one({'run_id': 'run-externalized'})
        assert stored is not None
        messages_field = stored['messages']
        assert isinstance(messages_field, str)
        assert len(messages_field) < 64 * 1024

        restored = await store.latest_snapshot(run_id='run-externalized')
        assert restored is not None
        # Compared by content: `ModelResponse` carries its own timestamp, so the
        # rebuilt message is not equal to a freshly constructed one.
        part = restored.messages[0].parts[0]
        assert isinstance(part, TextPart)
        assert part.content == text


@pytest.mark.anyio(backends=['asyncio'])
async def test_multi_chunk_blob_round_trips_at_the_default_chunk_size() -> None:
    """A blob larger than one BSON document survives the default 8MB chunking."""
    data = os.urandom(_MULTI_CHUNK_BLOB_BYTES)
    async with _live_database() as db:
        store = MongoMediaStore(client=db.client, database=db.name)

        uri = await store.put(data)

        manifest = await db['media'].find_one({'_id': parse_media_uri(uri)})
        assert manifest is not None
        assert manifest['size_bytes'] == _MULTI_CHUNK_BLOB_BYTES
        assert manifest['chunk_count'] == 3
        assert await db['media_chunks'].count_documents({'files_id': parse_media_uri(uri)}) == 3

        assert await store.get(uri) == data


@pytest.mark.anyio(backends=['asyncio'])
async def test_chunk_reassembly_ignores_collection_scan_order() -> None:
    """Reassembly follows the sort on `n`, not the order mongod stores chunks in."""
    data = os.urandom(5 * 1024)
    async with _live_database() as db:
        store = MongoMediaStore(client=db.client, database=db.name, chunk_size_bytes=1024)
        uri = await store.put(data)
        digest = parse_media_uri(uri)
        chunks = db['media_chunks']

        # Re-inserting the first chunk gives it a fresh record id, which moves it
        # to the end of the collection scan while its `n` stays 0.
        first = await chunks.find_one({'files_id': digest, 'n': 0})
        assert first is not None
        await chunks.delete_one({'_id': first['_id']})
        await chunks.insert_one(first)

        # `hint` forces the collection scan; the `(files_id, n)` index would
        # otherwise hand back `n` order and hide the divergence.
        scan_order = [doc['n'] async for doc in chunks.find({'files_id': digest}).hint([('$natural', 1)])]
        assert scan_order == [1, 2, 3, 4, 0]

        assert await store.get(uri) == data


@pytest.mark.anyio(backends=['asyncio'])
async def test_awkward_metadata_keys_round_trip_through_real_bson() -> None:
    """Dotted, `$`-prefixed, and non-ASCII metadata keys are stored verbatim."""
    async with _live_database() as db:
        store = MongoStepStore(client=db.client, database=db.name)
        await store.register_run(RunRecord(run_id='run-metadata', metadata=dict(_AWKWARD_METADATA)))
        await store.append_event(
            StepEvent(run_id='run-metadata', kind='run_started', step_index=0, metadata=dict(_AWKWARD_METADATA))
        )

        run = await store.get_run(run_id='run-metadata')
        assert run is not None
        assert run.metadata == _AWKWARD_METADATA
        events = await store.list_events(run_id='run-metadata')
        assert [event.metadata for event in events] == [_AWKWARD_METADATA]

        # The server keeps the key characters as given rather than escaping them,
        # so a reader outside this library sees the same names.
        raw = await db['runs'].find_one({'_id': 'run-metadata'})
        assert raw is not None
        assert raw['metadata'] == _AWKWARD_METADATA


@pytest.mark.anyio(backends=['asyncio'])
async def test_null_byte_metadata_key_is_rejected_by_the_bson_encoder() -> None:
    """The one metadata key shape BSON cannot carry fails at encode time, before the wire."""
    async with _live_database() as db:
        store = MongoStepStore(client=db.client, database=db.name)

        with pytest.raises(InvalidDocument):
            await store.register_run(RunRecord(run_id='run-null-key', metadata={'a\x00b': 'v'}))


@pytest.mark.anyio(backends=['asyncio'])
async def test_tool_effect_unique_index_is_created_and_enforced() -> None:
    """`(run_id, tool_call_id)` is a real server-side constraint, not a convention."""
    async with _live_database() as db:
        store = MongoStepStore(client=db.client, database=db.name)
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='call-1', tool_name='search', run_id='run-effects', status='started')
        )

        indexes = _INDEX_INFO.validate_python(await db['tool_effects'].index_information())
        assert indexes['run_id_1_tool_call_id_1']['unique'] is True

        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='call-1', tool_name='search', run_id='run-effects', status='completed')
        )
        assert await db['tool_effects'].count_documents({'run_id': 'run-effects'}) == 1
        effect = await store.get_tool_effect(run_id='run-effects', tool_call_id='call-1')
        assert effect is not None
        assert effect.status == 'completed'

        with pytest.raises(DuplicateKeyError):
            await db['tool_effects'].insert_one({'run_id': 'run-effects', 'tool_call_id': 'call-1'})
