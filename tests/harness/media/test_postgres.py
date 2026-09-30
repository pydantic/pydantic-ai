"""SQL-free tests for `PostgresMediaStore`.

Everything here runs without a database: construction, table name validation,
protocol conformance, `public_url`, and the URI check that precedes every
query. `_StubPool` satisfies `PostgresPool` structurally and is never queried.

The behavior that needs a server (round trips, dedup, schema creation) is
covered by `src/pydantic_ai_harness/integration_tests/postgres`, which runs
against a real Postgres.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager

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
