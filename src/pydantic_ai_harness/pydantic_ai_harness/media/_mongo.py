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

import hashlib
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
    coerce_metadata_mapping,
    media_uri_for,
    parse_media_uri,
    resolve_public_url,
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
      fields `media_type`, `size_bytes`, `chunk_count`, `chunk_size`, and
      `metadata` (JSON of `context.metadata`). The manifest is not chunked, so
      unbounded `metadata` could itself approach the 16MB cap -- keep per-blob
      metadata small.
    - `media_chunks` -- one document per chunk, `_id =
      '<digest>:<chunk_size>:<n>'`, with `files_id = <digest>`, `chunk_size`,
      `n` (0-based order), and `data` (`Binary`).

    Writes are idempotent: once the manifest exists, re-`put`-ing the same
    bytes is a no-op. Chunks are written before the manifest and keyed by
    their chunk size, so writers with different `chunk_size_bytes` (a retry
    after a crash, or two processes at once) never share a chunk, and `get`
    reads only the chunks of the manifest's size and count. `get` checks the
    sha256 of the result, so a missing or corrupt chunk raises. Manifests
    written before chunks were keyed by size carry no `chunk_size` and read
    the unkeyed chunks. `chunk_size_bytes` (default 8MB) bounds each chunk
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
        size = self._chunk_size
        chunk_count = 0
        for n, start in enumerate(range(0, len(data), size)):
            await chunks.update_one(
                {'_id': f'{digest}:{size}:{n}'},
                {
                    '$setOnInsert': {
                        'files_id': digest,
                        'chunk_size': size,
                        'n': n,
                        'data': Binary(data[start : start + size]),
                    }
                },
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
                    'chunk_size': size,
                    'metadata': json.dumps(dict(context.metadata)),
                }
            },
            upsert=True,
        )
        return uri

    async def get(self, uri: str, *, context: MediaContext = _EMPTY_CONTEXT) -> bytes:
        await self._ensure_indexes()
        digest = parse_media_uri(uri)
        files_doc = await self._db[self._files_name].find_one(
            {'_id': digest}, {'size_bytes': 1, 'chunk_count': 1, 'chunk_size': 1}
        )
        if files_doc is None:
            raise FileNotFoundError(f'media not found: {uri}')
        chunk_count = files_doc.get('chunk_count')
        if not isinstance(chunk_count, int):
            raise ValueError(f'media manifest for {uri} has no integer chunk_count: {chunk_count!r}')
        chunk_size = files_doc.get('chunk_size')
        query: _MongoDocument = {
            'files_id': digest,
            # A manifest from before chunks were keyed by size reads the unkeyed chunks.
            'chunk_size': chunk_size if isinstance(chunk_size, int) else {'$exists': False},
            'n': {'$lt': chunk_count},
        }
        buffer = bytearray()
        async for doc in self._db[self._chunks_name].find(query).sort('n', 1):
            piece = doc['data']
            if not isinstance(piece, bytes):
                raise ValueError(f'media chunk for {uri} holds {type(piece).__name__}, not bytes')
            buffer.extend(piece)
        # A missing, truncated, or stale chunk would otherwise return wrong bytes.
        # The digest is the content hash, so checking it catches all three.
        if hashlib.sha256(buffer).hexdigest() != digest:
            expected = files_doc.get('size_bytes')
            raise ValueError(
                f'media reassembly mismatch for {uri}: expected {expected!r} bytes '
                f'with that sha256, got {len(buffer)} bytes that do not match'
            )
        return bytes(buffer)

    async def exists(self, uri: str, *, context: MediaContext = _EMPTY_CONTEXT) -> bool:
        digest = parse_media_uri(uri)
        return await self._db[self._files_name].find_one({'_id': digest}, {'_id': 1}) is not None

    async def public_url(self, uri: str, *, context: MediaContext = _EMPTY_CONTEXT) -> str | None:
        return await resolve_public_url(self._public_url_resolver, uri, context)

    async def get_metadata(self, uri: str, *, context: MediaContext = _EMPTY_CONTEXT) -> Mapping[str, str]:
        digest = parse_media_uri(uri)
        doc = await self._db[self._files_name].find_one({'_id': digest}, {'metadata': 1})
        if doc is None:
            raise FileNotFoundError(f'media not found: {uri}')
        raw = doc.get('metadata', '{}')
        if not isinstance(raw, str):
            raise ValueError(f'media manifest for {uri} has non-string metadata: {raw!r}')
        return coerce_metadata_mapping(json.loads(raw))
