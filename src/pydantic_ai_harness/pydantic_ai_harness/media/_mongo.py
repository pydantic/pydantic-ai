"""MongoDB-backed `MediaStore` with content-addressed manual chunking.

Blobs are split into fixed-size chunks stored across two collections -- a
`files` manifest document keyed by the sha256 hex digest plus one document
per chunk in a sibling `<collection>_chunks` collection. Chunking bounds the
size of each BSON document, not memory: `put` takes the whole payload and
`get` reassembles it, so a blob must fit in process memory either way. The
bound is achieved without depending on the GridFS driver. The manifest itself stores
`MediaContext.metadata` inline and is not chunked, so callers should keep
per-blob metadata small (it is not covered by the chunk-size bound).

Manual chunking is chosen over GridFS deliberately:

- Content-addressed dedup: the digest is the files `_id`, so identical bytes
  written twice resolve to the same URI and the second write is a no-op.
  GridFS keys files by `ObjectId` and does no dedup, so a sha256 layer would
  sit on top of it anyway.
- The async collection surface used here (`find_one`, `update_one` upsert,
  sorted `find`) is fakeable in-memory, so the store reaches full test
  coverage without a running mongod. The native async GridFS driver rejects
  in-memory fakes.

Chunking is unconditional (a small blob is one chunk) -- there is no
size-branch, so the write path is the same for a 1KB and a 100MB payload.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping

from bson.binary import Binary
from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase

from pydantic_ai_harness.media._store import (
    _EMPTY_CONTEXT,  # pyright: ignore[reportPrivateUsage]
    MediaContext,
    PublicUrlResolver,
    _coerce_metadata_mapping,  # pyright: ignore[reportPrivateUsage]
    _resolve_public_url,  # pyright: ignore[reportPrivateUsage]
    media_uri_for,
    parse_media_uri,
)

_DEFAULT_CHUNK_SIZE_BYTES = 8 * 1024 * 1024
# 16MiB BSON document cap minus headroom for the chunk document's own framing
# (`_id`, `files_id`, `n`, the `Binary` subtype/length prefix). A larger value
# would build a chunk document that MongoDB rejects on insert.
_MAX_CHUNK_SIZE_BYTES = 16 * 1024 * 1024 - 64 * 1024
_COLLECTION_NAME_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

_MongoDocument = dict[str, object]


class MongoMediaStore:
    """MongoDB `MediaStore`; content-addressed, chunked across two collections.

    Pass `client=` (a shared `AsyncMongoClient`) or `db_url=` (the store
    creates and owns its own client). `database=` is always required. When
    `client=` is used the caller owns its lifecycle; when `db_url=` is used,
    call `aclose()` to release the owned client.

    Layout, for the default `collection='media'`:

    - `media` -- one manifest document per blob, `_id = <sha256-hex>`, with
      fields `media_type`, `size_bytes`, `chunk_count`, and `metadata`
      (canonical JSON of `context.metadata`). The manifest is not chunked, so
      unbounded `metadata` could itself approach the 16MB cap -- keep per-blob
      metadata small.
    - `media_chunks` -- one document per chunk, `_id = '<digest>:<n>'`, with
      `files_id = <digest>`, `n` (0-based order), and `data` (`Binary`).

    Writes are idempotent via `$setOnInsert` upserts: re-`put`-ing the same
    bytes is a no-op. `chunk_size_bytes` (default 8MB) bounds each chunk
    document below the 16MB cap; it is rejected outside `1..~16MB`. Lower it
    in tests to exercise the multi-chunk split cheaply.
    """

    def __init__(
        self,
        *,
        client: AsyncMongoClient[_MongoDocument] | None = None,
        db_url: str | None = None,
        database: str | None = None,
        collection: str = 'media',
        chunk_size_bytes: int = _DEFAULT_CHUNK_SIZE_BYTES,
        public_url: PublicUrlResolver | None = None,
    ) -> None:
        if (client is None) == (db_url is None):
            raise ValueError('provide exactly one of `client=` or `db_url=`')
        if database is None:
            raise ValueError('`database=` is required')
        if not _COLLECTION_NAME_RE.fullmatch(collection):
            raise ValueError(f'invalid collection name: {collection!r}')
        if chunk_size_bytes < 1:
            raise ValueError('`chunk_size_bytes` must be positive')
        if chunk_size_bytes > _MAX_CHUNK_SIZE_BYTES:
            raise ValueError(f'`chunk_size_bytes` must not exceed {_MAX_CHUNK_SIZE_BYTES} (MongoDB 16MB document cap)')

        if client is not None:
            self._client = client
            self._own_client = False
        else:
            assert db_url is not None
            self._client = AsyncMongoClient[_MongoDocument](db_url)
            self._own_client = True

        self._db: AsyncDatabase[_MongoDocument] = self._client[database]
        self._files_name = collection
        self._chunks_name = f'{collection}_chunks'
        self._chunk_size = chunk_size_bytes
        self._public_url_resolver = public_url
        self._indexes_ready = False

    async def aclose(self) -> None:
        """Close the client if this store created it; no-op for a shared client."""
        if self._own_client:
            await self._client.close()

    async def _ensure_indexes(self) -> None:
        if self._indexes_ready:
            return
        # `get` reassembles a blob via `media_chunks.find({'files_id': ...}).sort('n')`;
        # without this index that read is a full collection scan.
        await self._db[self._chunks_name].create_index([('files_id', 1), ('n', 1)])
        self._indexes_ready = True

    async def put(self, data: bytes, *, context: MediaContext = _EMPTY_CONTEXT) -> str:
        await self._ensure_indexes()
        uri = media_uri_for(data)
        digest = parse_media_uri(uri)
        files = self._db[self._files_name]
        if await files.find_one({'_id': digest}, {'_id': 1}) is not None:
            return uri
        chunks = self._db[self._chunks_name]
        chunk_count = 0
        for n, start in enumerate(range(0, len(data), self._chunk_size)):
            piece = data[start : start + self._chunk_size]
            await chunks.update_one(
                {'_id': f'{digest}:{n}'},
                {'$setOnInsert': {'files_id': digest, 'n': n, 'data': Binary(piece)}},
                upsert=True,
            )
            chunk_count = n + 1
        await files.update_one(
            {'_id': digest},
            {
                '$setOnInsert': {
                    'media_type': context.media_type,
                    'size_bytes': len(data),
                    'chunk_count': chunk_count,
                    'metadata': json.dumps(dict(context.metadata)),
                }
            },
            upsert=True,
        )
        return uri

    async def get(self, uri: str, *, context: MediaContext = _EMPTY_CONTEXT) -> bytes:
        await self._ensure_indexes()
        digest = parse_media_uri(uri)
        files_doc = await self._db[self._files_name].find_one({'_id': digest}, {'size_bytes': 1})
        if files_doc is None:
            raise FileNotFoundError(f'media not found: {uri}')
        buffer = bytearray()
        async for doc in self._db[self._chunks_name].find({'files_id': digest}).sort('n', 1):
            piece = doc['data']
            assert isinstance(piece, (bytes, bytearray))
            buffer.extend(piece)
        # A chunk gone missing or truncated would silently return short bytes; the
        # stored `size_bytes` is the reassembly checksum that turns that into a
        # loud failure instead of a corrupt round-trip.
        expected = files_doc.get('size_bytes')
        if len(buffer) != expected:
            raise ValueError(f'media reassembly mismatch for {uri}: expected {expected!r} bytes, got {len(buffer)}')
        return bytes(buffer)

    async def exists(self, uri: str, *, context: MediaContext = _EMPTY_CONTEXT) -> bool:
        digest = parse_media_uri(uri)
        return await self._db[self._files_name].find_one({'_id': digest}, {'_id': 1}) is not None

    async def public_url(self, uri: str, *, context: MediaContext = _EMPTY_CONTEXT) -> str | None:
        return await _resolve_public_url(self._public_url_resolver, uri, context)

    async def get_metadata(self, uri: str, *, context: MediaContext = _EMPTY_CONTEXT) -> Mapping[str, str]:
        digest = parse_media_uri(uri)
        doc = await self._db[self._files_name].find_one({'_id': digest}, {'metadata': 1})
        if doc is None:
            raise FileNotFoundError(f'media not found: {uri}')
        raw = doc.get('metadata', '{}')
        assert isinstance(raw, str)
        return _coerce_metadata_mapping(json.loads(raw))
