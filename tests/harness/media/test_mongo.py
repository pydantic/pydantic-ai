"""Tests for `MongoMediaStore` against an in-memory `mongomock-motor` client.

`MongoMediaStore` stores blobs as manual sha256-addressed chunks over plain
async collection operations (no GridFS), so it reaches full coverage against
`mongomock-motor` without a running mongod. `_mock_client` is the single
type-boundary shim: the fake is not a `pymongo.AsyncMongoClient`, but the
store only uses the async collection surface both share.
"""

from __future__ import annotations

import anyio.to_thread
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
        # Building a client touches the filesystem; users build stores outside the event loop.
        store = await anyio.to_thread.run_sync(
            lambda: MongoMediaStore(db_url='mongodb://localhost:59017', database='t')
        )
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
        the in-memory fake returns insertion order for an unsorted `find`. Real
        MongoDB gives no order guarantee for an unsorted query.
        """
        client = _mock_client()
        store = MongoMediaStore(client=client, database='t', chunk_size_bytes=4)
        data = bytes(range(42))  # 42 bytes -> 10 full chunks of 4 plus a 2-byte tail
        uri = await store.put(data)
        digest = parse_media_uri(uri)
        expected_chunks = (len(data) + 3) // 4
        chunks = client['t']['media_chunks']
        assert await chunks.count_documents({}) == expected_chunks
        manifest = await client['t']['media'].find_one({'_id': digest})
        assert manifest is not None
        assert manifest['chunk_count'] == expected_chunks

        first_chunk = await chunks.find_one({'files_id': digest, 'n': 0})
        assert first_chunk is not None
        await chunks.delete_one({'_id': first_chunk['_id']})
        await chunks.insert_one(first_chunk)

        assert await store.get(uri) == data

    async def test_dedup_short_circuits_the_write_path(self) -> None:
        """A repeat `put` returns on the manifest lookup without touching the chunks.

        A chunk is deleted between the two writes, so the deleted chunk staying
        absent is what pins the early return rather than merely the absence of
        duplicates.
        """
        client = _mock_client()
        store = MongoMediaStore(client=client, database='t', chunk_size_bytes=4)
        data = b'duplicate me'
        first = await store.put(data)
        digest = parse_media_uri(first)
        expected_chunks = (len(data) + 3) // 4
        assert await client['t']['media_chunks'].count_documents({}) == expected_chunks
        await client['t']['media_chunks'].delete_one({'files_id': digest, 'n': 0})

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
        text = 'héllo wörld 日本語 🎉 ' * 2
        data = text.encode('utf-8')
        # At least one chunk boundary lands on a UTF-8 continuation byte, mid-character.
        assert any(data[i] & 0xC0 == 0x80 for i in range(7, len(data), 7))
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
        """A chunk whose bytes no longer hash to the digest surfaces as a loud error."""
        client = _mock_client()
        store = MongoMediaStore(client=client, database='t')
        uri = await store.put(b'twelve chars')
        digest = parse_media_uri(uri)
        await client['t']['media_chunks'].update_one(
            {'files_id': digest, 'n': 0}, {'$set': {'data': Binary(b'tweLve chars')}}
        )
        with pytest.raises(ValueError, match='reassembly mismatch'):
            await store.get(uri)

    @pytest.mark.parametrize(
        ('stale_size', 'retry_size'),
        [pytest.param(8, 4, id='stale-chunks-larger'), pytest.param(2, 4, id='stale-chunks-smaller')],
    )
    async def test_retry_after_a_crashed_write_at_another_chunk_size_reads_back(
        self, stale_size: int, retry_size: int
    ) -> None:
        """A retry at another chunk size reads back past the chunks a crashed write left behind.

        The stale chunks stay in the collection. `get` reads only the chunks
        that match the manifest's `chunk_size` and `chunk_count`.
        """
        client = _mock_client()
        data = bytes(range(64))
        digest = parse_media_uri(media_uri_for(data))
        chunks = client['t']['media_chunks']
        # A crashed write left its chunks for this digest without writing the manifest.
        for n, start in enumerate(range(0, len(data), stale_size)):
            await chunks.insert_one(
                {
                    '_id': f'{digest}:{stale_size}:{n}',
                    'files_id': digest,
                    'chunk_size': stale_size,
                    'n': n,
                    'data': Binary(data[start : start + stale_size]),
                }
            )
        store = MongoMediaStore(client=client, database='t', chunk_size_bytes=retry_size)
        uri = await store.put(data)
        assert await store.get(uri) == data
        stale_count = len(range(0, len(data), stale_size))
        assert await chunks.count_documents({'chunk_size': stale_size}) == stale_count

    async def test_concurrent_writer_at_another_chunk_size_cannot_corrupt_a_published_blob(self) -> None:
        """A writer that passed the manifest check before another published writes chunks of its own.

        Chunks are keyed by size, so the late writer's chunks never replace the
        ones the published manifest points at.
        """
        client = _mock_client()
        data = bytes(range(64))
        first = MongoMediaStore(client=client, database='t', chunk_size_bytes=4)
        uri = await first.put(data)
        files = client['t']['media']
        published = await files.find_one({'_id': parse_media_uri(uri)})
        assert published is not None
        # Replay the late writer: it saw no manifest, so it writes all its chunks.
        await files.delete_one({'_id': parse_media_uri(uri)})
        await MongoMediaStore(client=client, database='t', chunk_size_bytes=8).put(data)
        await files.replace_one({'_id': parse_media_uri(uri)}, published)

        assert await first.get(uri) == data
        assert await client['t']['media_chunks'].count_documents({'files_id': parse_media_uri(uri)}) == 16 + 8

    async def test_manifest_from_before_size_keyed_chunks_reads_back(self) -> None:
        client = _mock_client()
        data = b'legacy layout'
        digest = parse_media_uri(media_uri_for(data))
        await client['t']['media'].insert_one(
            {'_id': digest, 'media_type': None, 'size_bytes': len(data), 'chunk_count': 2, 'metadata': '{}'}
        )
        for n, piece in enumerate([data[:7], data[7:]]):
            await client['t']['media_chunks'].insert_one(
                {'_id': f'{digest}:{n}', 'files_id': digest, 'n': n, 'data': Binary(piece)}
            )
        store = MongoMediaStore(client=client, database='t', chunk_size_bytes=4)
        assert await store.get(media_uri_for(data)) == data

    async def test_manifest_without_integer_chunk_count_raises(self) -> None:
        client = _mock_client()
        store = MongoMediaStore(client=client, database='t')
        uri = await store.put(b'counted')
        await client['t']['media'].update_one({'_id': parse_media_uri(uri)}, {'$unset': {'chunk_count': ''}})
        with pytest.raises(ValueError, match='no integer chunk_count'):
            await store.get(uri)

    async def test_chunk_without_bytes_raises(self) -> None:
        client = _mock_client()
        store = MongoMediaStore(client=client, database='t')
        uri = await store.put(b'chunked')
        await client['t']['media_chunks'].update_one(
            {'files_id': parse_media_uri(uri), 'n': 0}, {'$set': {'data': 'text'}}
        )
        with pytest.raises(ValueError, match='holds str, not bytes'):
            await store.get(uri)

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

    async def test_get_metadata_rejects_non_string_metadata(self) -> None:
        client = _mock_client()
        store = MongoMediaStore(client=client, database='t')
        uri = await store.put(b'tagged', context=MediaContext(metadata={'k': 'v'}))
        await client['t']['media'].update_one({'_id': parse_media_uri(uri)}, {'$set': {'metadata': {'k': 'v'}}})
        with pytest.raises(ValueError, match='non-string metadata'):
            await store.get_metadata(uri)

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
            _ = media.NoSuchStore
