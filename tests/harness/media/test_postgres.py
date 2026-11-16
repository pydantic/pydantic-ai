"""Unit tests for `PostgresMediaStore`.

Construction, table name validation, protocol conformance, `public_url`, and the
URI check that precedes every query run against `_StubPool`, which satisfies
`PostgresPool` structurally and is never queried. Round trips, dedup, and schema
creation run against `SqlitePool`, an in-memory SQLite stand-in for a pool: there
is no in-process Postgres, and these tests must run in every CI job.

The edges only a real server shows are covered by
`src/pydantic_ai_harness/integration_tests/postgres`, which runs against Postgres.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractAsyncContextManager

import anyio
import pytest

import pydantic_ai_harness.media as media
from pydantic_ai_harness.media import (
    MediaContext,
    MediaStore,
    PostgresConnection,
    PostgresMediaStore,
    PostgresPool,
    make_static_public_url,
    media_uri_for,
    parse_media_uri,
)
from tests.harness._postgres_sqlite import SqlitePool

_MALFORMED_URI = 'https://example.com/not-a-media-uri'


class _StubPool:
    """A `PostgresPool` whose `acquire` fails the test if anything reaches it."""

    def acquire(self) -> AbstractAsyncContextManager[PostgresConnection]:
        raise AssertionError('the pool must not be used')  # pragma: no cover


class TestPostgresMediaStoreConstruction:
    @pytest.mark.parametrize('table', ['media-store', 'x; DROP TABLE users', '1media', 'a' * 64, '', 'Media_2'])
    def test_rejects_invalid_table_name(self, table: str) -> None:
        """Uppercase is rejected: Postgres folds unquoted identifiers, so `'Media'` would alias `'media'`."""
        with pytest.raises(ValueError, match='invalid table name'):
            PostgresMediaStore(_StubPool(), table=table)

    @pytest.mark.parametrize('table', ['a' * 63, '_media', 'media_2'])
    def test_accepts_valid_table_name(self, table: str) -> None:
        """63 characters is the longest identifier Postgres keeps without truncating."""
        store = PostgresMediaStore(_StubPool(), table=table)
        assert isinstance(store, PostgresMediaStore)

    def test_stub_pool_satisfies_pool_protocol(self) -> None:
        """The pool is accepted structurally, so no driver import is needed to build a store."""
        assert isinstance(_StubPool(), PostgresPool)


class TestPostgresMediaStoreMalformedUri:
    """A URI that is not `media+sha256://<digest>` is refused before any query."""

    async def test_get_raises(self) -> None:
        store = PostgresMediaStore(_StubPool())
        with pytest.raises(ValueError):
            await store.get(_MALFORMED_URI)

    async def test_exists_raises(self) -> None:
        store = PostgresMediaStore(_StubPool())
        with pytest.raises(ValueError):
            await store.exists(_MALFORMED_URI)

    async def test_get_metadata_raises(self) -> None:
        store = PostgresMediaStore(_StubPool())
        with pytest.raises(ValueError):
            await store.get_metadata(_MALFORMED_URI)


class TestPostgresMediaStorePublicUrl:
    async def test_without_resolver_returns_none(self) -> None:
        store = PostgresMediaStore(_StubPool())
        assert await store.public_url(media_uri_for(b'x')) is None

    async def test_with_sync_resolver_uses_it(self) -> None:
        store = PostgresMediaStore(_StubPool(), public_url=make_static_public_url('https://cdn.example.com'))
        uri = media_uri_for(b'p')
        digest = parse_media_uri(uri)
        assert await store.public_url(uri) == f'https://cdn.example.com/{digest}.bin'

    async def test_with_async_resolver_uses_it(self) -> None:
        seen: list[tuple[str, MediaContext]] = []

        async def resolver(uri: str, context: MediaContext) -> str | None:
            seen.append((uri, context))
            return f'https://signed.example.com/{parse_media_uri(uri)}'

        store = PostgresMediaStore(_StubPool(), public_url=resolver)
        uri = media_uri_for(b'p')
        context = MediaContext(media_type='image/png')
        assert await store.public_url(uri, context=context) == f'https://signed.example.com/{parse_media_uri(uri)}'
        assert seen == [(uri, context)]


class TestPostgresMediaStoreProtocol:
    def test_satisfies_media_store_protocol(self) -> None:
        store: MediaStore = PostgresMediaStore(_StubPool())
        assert isinstance(store, MediaStore)


class TestMediaPostgresExport:
    def test_postgres_names_are_exported(self) -> None:
        assert media.PostgresMediaStore is PostgresMediaStore
        assert media.PostgresPool is PostgresPool
        assert media.PostgresConnection is PostgresConnection
        assert {'PostgresMediaStore', 'PostgresPool', 'PostgresConnection'} <= set(media.__all__)


@pytest.fixture
def pool() -> Iterator[SqlitePool]:
    sqlite_pool = SqlitePool()
    yield sqlite_pool
    sqlite_pool.close()


_MISSING_URI = 'media+sha256://' + '0' * 64


class TestPostgresMediaStoreRoundTrip:
    async def test_put_get_round_trip(self, pool: SqlitePool) -> None:
        """Bytes that are not valid UTF-8 come back identical."""
        store = PostgresMediaStore(pool)
        data = b'hello postgres bytes \x00\xff'

        uri = await store.put(data, context=MediaContext(media_type='application/octet-stream'))

        assert uri == media_uri_for(data)
        assert await store.get(uri) == data
        assert await store.exists(uri) is True
        assert await store.exists(_MISSING_URI) is False

    async def test_second_put_is_a_no_op(self, pool: SqlitePool) -> None:
        """The digest is the primary key, so a repeat `put` neither fails nor overwrites."""
        store = PostgresMediaStore(pool)
        data = b'duplicate me'

        first = await store.put(data, context=MediaContext(media_type='text/plain', metadata={'writer': 'first'}))
        second = await store.put(data, context=MediaContext(media_type='image/png', metadata={'writer': 'second'}))

        assert first == second
        assert await pool.count_rows('media') == 1
        assert await store.get_metadata(first) == {'writer': 'first'}

    async def test_metadata_defaults_to_empty(self, pool: SqlitePool) -> None:
        store = PostgresMediaStore(pool)

        uri = await store.put(b'no tags')

        assert await store.get_metadata(uri) == {}

    async def test_custom_table_name(self, pool: SqlitePool) -> None:
        store = PostgresMediaStore(pool, table='blobs')

        uri = await store.put(b'in a custom table')

        assert await pool.count_rows('blobs') == 1
        assert await store.get(uri) == b'in a custom table'

    async def test_missing_digest_raises_file_not_found(self, pool: SqlitePool) -> None:
        store = PostgresMediaStore(pool)

        with pytest.raises(FileNotFoundError, match='media not found'):
            await store.get(_MISSING_URI)
        with pytest.raises(FileNotFoundError, match='media not found'):
            await store.get_metadata(_MISSING_URI)

    async def test_row_with_wrong_types_raises(self, pool: SqlitePool) -> None:
        """A row another writer stored with the wrong column types is refused, not returned."""
        store = PostgresMediaStore(pool)
        await store.exists(_MISSING_URI)  # creates the table
        async with pool.acquire() as connection, connection.transaction():
            await connection.execute(
                'INSERT INTO media (sha256, bytes, size_bytes, metadata) VALUES ($1, $2, $3, $4)',
                parse_media_uri(_MISSING_URI),
                'not bytes',
                9,
                b'not text',
            )

        with pytest.raises(ValueError, match='has wrong types'):
            await store.get(_MISSING_URI)
        with pytest.raises(ValueError, match='has wrong types'):
            await store.get_metadata(_MISSING_URI)


class TestPostgresMediaStoreSchema:
    async def test_schema_is_created_once_under_concurrent_first_calls(self, pool: SqlitePool) -> None:
        """The second first call waits on the lock and then finds the schema ready."""
        store = PostgresMediaStore(pool)

        async with anyio.create_task_group() as tg:
            tg.start_soon(store.put, b'first caller')
            tg.start_soon(store.put, b'second caller')
        await store.put(b'later caller')

        assert await pool.count_rows('media') == 3
        creates = [statement for statement in pool.statements if statement.startswith('CREATE TABLE')]
        assert len(creates) == 1
        assert pool.statements.count('SELECT pg_advisory_xact_lock(hashtext($1))') == 1
