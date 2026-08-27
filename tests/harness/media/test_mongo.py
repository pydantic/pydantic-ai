"""Tests for `MongoMediaStore` against an in-memory `mongomock-motor` client.

`MongoMediaStore` stores blobs as manual sha256-addressed chunks over plain
async collection operations (no GridFS), so it reaches full coverage against
`mongomock-motor` without a running mongod. `_mock_client` is the single
type-boundary shim: the fake is not a `pymongo.AsyncMongoClient`, but the
store only uses the async collection surface both share.
"""

from __future__ import annotations

import pytest
from bson.binary import Binary
from mongomock_motor import AsyncMongoMockClient
from pymongo import AsyncMongoClient

import pydantic_ai_harness.media as media
from pydantic_ai_harness.media import (
    MediaContext,
    MediaStore,
    MongoMediaStore,
    make_static_public_url,
    media_uri_for,
    parse_media_uri,
)

pytestmark = pytest.mark.anyio

_MISSING_URI = 'media+sha256://' + ('0' * 64)


def _mock_client() -> AsyncMongoClient[dict[str, object]]:
    return AsyncMongoMockClient()  # pyright: ignore[reportUnknownVariableType, reportReturnType]


@pytest.fixture
def close_calls(monkeypatch: pytest.MonkeyPatch) -> list[AsyncMongoClient[dict[str, object]]]:
    """Record every `AsyncMongoClient.close()` so `aclose` ownership is observable.

    Both `aclose` cases hinge on whether `close` was called, which is invisible
    from the store's own surface -- without this the "shared client" case passes
    even when `aclose` closes a client it does not own.
    """
    calls: list[AsyncMongoClient[dict[str, object]]] = []

    async def record_close(self: AsyncMongoClient[dict[str, object]]) -> None:
        calls.append(self)

    monkeypatch.setattr(AsyncMongoClient, 'close', record_close)
    return calls


class TestMongoMediaStoreConstruction:
    def test_requires_exactly_one_of_client_or_db_url(self) -> None:
        with pytest.raises(ValueError, match='exactly one'):
            MongoMediaStore(database='t')
        with pytest.raises(ValueError, match='exactly one'):
            MongoMediaStore(client=_mock_client(), db_url='mongodb://x', database='t')

    def test_requires_database(self) -> None:
        with pytest.raises(ValueError, match='`database=` is required'):
            MongoMediaStore(client=_mock_client())

    def test_rejects_invalid_collection_name(self) -> None:
        with pytest.raises(ValueError, match='invalid collection name'):
            MongoMediaStore(client=_mock_client(), database='t', collection='bad-name')

    def test_rejects_non_positive_chunk_size(self) -> None:
        with pytest.raises(ValueError, match='`chunk_size_bytes` must be positive'):
            MongoMediaStore(client=_mock_client(), database='t', chunk_size_bytes=0)

    def test_rejects_oversized_chunk_size(self) -> None:
        """A chunk size above the BSON headroom ceiling would build a rejected chunk doc."""
        with pytest.raises(ValueError, match='must not exceed'):
            MongoMediaStore(client=_mock_client(), database='t', chunk_size_bytes=20 * 1024 * 1024)

    async def test_db_url_owns_client_and_aclose_closes_it(
        self,
        monkeypatch: pytest.MonkeyPatch,
        close_calls: list[AsyncMongoClient[dict[str, object]]],
    ) -> None:
        store = MongoMediaStore(db_url='mongodb://localhost:59017', database='t')
        await store.aclose()
        assert len(close_calls) == 1

        monkeypatch.undo()
        await close_calls[0].close()  # the recorded close was a stub; release it for real

    async def test_shared_client_aclose_is_noop(
        self,
        monkeypatch: pytest.MonkeyPatch,
        close_calls: list[AsyncMongoClient[dict[str, object]]],
    ) -> None:
        client = AsyncMongoClient[dict[str, object]]('mongodb://localhost:59017')
        store = MongoMediaStore(client=client, database='t')
        await store.aclose()
        assert close_calls == []  # a caller-supplied client stays open

        monkeypatch.undo()
        await client.close()


class TestMongoMediaStoreRoundTrip:
    async def test_put_get_round_trip(self) -> None:
        store = MongoMediaStore(client=_mock_client(), database='t')
        data = b'hello mongo bytes'
        uri = await store.put(data, context=MediaContext(media_type='application/octet-stream'))
        assert uri == media_uri_for(data)
        assert await store.get(uri) == data

    async def test_multi_chunk_split_and_reassembly(self) -> None:
        """A blob larger than `chunk_size_bytes` splits into ordered chunks and rejoins.

        Chunk 0 is deleted and re-inserted so that storage order no longer matches
        `n`. Reassembly can then only produce the original bytes by sorting on `n`:
        the in-memory fake returns insertion order for an unsorted `find`, and the
        `size_bytes` guard cannot catch a scramble because the total length is
        unchanged. Real MongoDB gives no order guarantee for an unsorted query.
        """
        client = _mock_client()
        store = MongoMediaStore(client=client, database='t', chunk_size_bytes=4)
        data = bytes(range(256)) * 40  # 10_240 bytes -> 2560 chunks of 4
        uri = await store.put(data)
        digest = parse_media_uri(uri)
        expected_chunks = (len(data) + 3) // 4
        chunks = client['t']['media_chunks']
        assert await chunks.count_documents({}) == expected_chunks
        manifest = await client['t']['media'].find_one({'_id': digest})
        assert manifest is not None
        assert manifest['chunk_count'] == expected_chunks

        first_chunk = await chunks.find_one({'_id': f'{digest}:0'})
        assert first_chunk is not None
        await chunks.delete_one({'_id': f'{digest}:0'})
        await chunks.insert_one(first_chunk)

        assert await store.get(uri) == data

    async def test_dedup_short_circuits_the_write_path(self) -> None:
        """A repeat `put` returns on the manifest lookup without touching the chunks.

        A chunk is deleted between the two writes: `$setOnInsert` is idempotent by
        itself, so the deleted chunk staying absent is what pins the early return
        rather than merely the absence of duplicates.
        """
        client = _mock_client()
        store = MongoMediaStore(client=client, database='t', chunk_size_bytes=4)
        data = b'duplicate me'
        first = await store.put(data)
        digest = parse_media_uri(first)
        expected_chunks = (len(data) + 3) // 4
        assert await client['t']['media_chunks'].count_documents({}) == expected_chunks
        await client['t']['media_chunks'].delete_one({'_id': f'{digest}:0'})

        second = await store.put(data)
        assert first == second
        assert await client['t']['media'].count_documents({}) == 1
        assert await client['t']['media_chunks'].count_documents({}) == expected_chunks - 1

    async def test_empty_blob_round_trips(self) -> None:
        """Zero bytes produce a manifest with no chunks and still read back."""
        client = _mock_client()
        store = MongoMediaStore(client=client, database='t')
        uri = await store.put(b'')
        digest = parse_media_uri(uri)
        manifest = await client['t']['media'].find_one({'_id': digest})
        assert manifest is not None
        assert manifest['chunk_count'] == 0
        assert await client['t']['media_chunks'].count_documents({}) == 0
        assert await store.get(uri) == b''
        assert await store.exists(uri) is True

    async def test_multi_byte_text_survives_chunk_boundaries(self) -> None:
        """Chunking splits raw bytes, so a multi-byte character straddles two chunks."""
        store = MongoMediaStore(client=_mock_client(), database='t', chunk_size_bytes=7)
        text = 'héllo wörld 日本語 🎉 ' * 500
        data = text.encode('utf-8')
        uri = await store.put(data, context=MediaContext(media_type='text/plain'))
        fetched = await store.get(uri)
        assert fetched == data
        assert fetched.decode('utf-8') == text

    async def test_exists(self) -> None:
        store = MongoMediaStore(client=_mock_client(), database='t')
        uri = await store.put(b'present')
        assert await store.exists(uri) is True
        assert await store.exists(_MISSING_URI) is False

    async def test_get_missing_raises(self) -> None:
        store = MongoMediaStore(client=_mock_client(), database='t')
        with pytest.raises(FileNotFoundError):
            await store.get(_MISSING_URI)

    async def test_reassembly_mismatch_raises(self) -> None:
        """A `size_bytes` that disagrees with the chunk bytes surfaces as a loud error."""
        client = _mock_client()
        store = MongoMediaStore(client=client, database='t')
        uri = await store.put(b'twelve chars')
        digest = parse_media_uri(uri)
        await client['t']['media'].update_one({'_id': digest}, {'$set': {'size_bytes': 999}})
        with pytest.raises(ValueError, match='reassembly mismatch'):
            await store.get(uri)

    async def test_stale_chunks_from_crashed_write_never_silently_corrupt(self) -> None:
        """Stale chunks from a crashed prior write at a different chunk size cannot corrupt a read.

        Chunk `_id`s are `<digest>:<n>`, so a retry at a different
        `chunk_size_bytes` leaves the old chunks in place (`$setOnInsert` will
        not overwrite them). Every chunk under a digest is a slice of the SAME
        content, so a mismatched-layout retry can only ever reassemble the
        identical bytes (length matches) or fail the `size_bytes` guard -- it
        can never return wrong bytes silently. In this constructed layout the
        stale 8-byte chunks plus the fresh 4-byte chunks over-count the length,
        so `get` fails loudly, which is the outcome asserted here. This is the
        evidence for Macroscope 446-1.
        """
        client = _mock_client()
        data = bytes(range(64))
        digest = parse_media_uri(media_uri_for(data))
        # Simulate a crashed write that used an 8-byte chunk size and left its
        # chunks behind for this digest without ever writing the manifest.
        chunks = client['t']['media_chunks']
        for n, start in enumerate(range(0, len(data), 8)):
            await chunks.update_one(
                {'_id': f'{digest}:{n}'},
                {'$setOnInsert': {'files_id': digest, 'n': n, 'data': Binary(data[start : start + 8])}},
                upsert=True,
            )
        # A fresh write re-puts the same content at a 4-byte chunk size.
        store = MongoMediaStore(client=client, database='t', chunk_size_bytes=4)
        await store.put(data)
        with pytest.raises(ValueError, match='reassembly mismatch'):
            await store.get(media_uri_for(data))

    async def test_metadata_round_trips(self) -> None:
        store = MongoMediaStore(client=_mock_client(), database='t')
        uri = await store.put(
            b'tagged',
            context=MediaContext(media_type='image/png', metadata={'origin': 'user', 'tenant': 'acme'}),
        )
        assert await store.get_metadata(uri) == {'origin': 'user', 'tenant': 'acme'}

    async def test_metadata_empty_when_not_supplied(self) -> None:
        store = MongoMediaStore(client=_mock_client(), database='t')
        uri = await store.put(b'no tags')
        assert await store.get_metadata(uri) == {}

    async def test_get_metadata_missing_raises(self) -> None:
        store = MongoMediaStore(client=_mock_client(), database='t')
        with pytest.raises(FileNotFoundError):
            await store.get_metadata(_MISSING_URI)

    async def test_custom_collection_name(self) -> None:
        client = _mock_client()
        store = MongoMediaStore(client=client, database='t', collection='blobs')
        uri = await store.put(b'in a custom collection')
        assert await client['t']['blobs'].count_documents({}) == 1
        assert await client['t']['blobs_chunks'].count_documents({}) >= 1
        assert await store.get(uri) == b'in a custom collection'


class TestMongoMediaStorePublicUrl:
    async def test_without_resolver_returns_none(self) -> None:
        store = MongoMediaStore(client=_mock_client(), database='t')
        assert await store.public_url(media_uri_for(b'x')) is None

    async def test_with_resolver_uses_it(self) -> None:

        store = MongoMediaStore(
            client=_mock_client(),
            database='t',
            public_url=make_static_public_url('https://cdn.example.com'),
        )
        uri = media_uri_for(b'p')
        digest = parse_media_uri(uri)
        assert await store.public_url(uri) == f'https://cdn.example.com/{digest}.bin'


class TestMongoMediaStoreProtocol:
    def test_satisfies_media_store_protocol(self) -> None:
        store: MediaStore = MongoMediaStore(client=_mock_client(), database='t')
        assert isinstance(store, MediaStore)


class TestMediaLazyExport:
    def test_mongo_media_store_lazily_exported(self) -> None:

        assert media.MongoMediaStore is MongoMediaStore

    def test_unknown_attribute_raises(self) -> None:

        with pytest.raises(AttributeError, match='has no attribute'):
            _ = media.NoSuchStore  # type: ignore[attr-defined]
