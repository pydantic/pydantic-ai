---
title: Media Externalization
description: Content-addressed stores and walker helpers that move large binary and text payloads out of message history into deduplicated storage and put them back on demand.
---

# Media Externalization

A conversation that carries images, audio, or other `BinaryContent` inlines those bytes into every message, and a large text part (a big tool-return string, say) is just as heavy. Persist that history and each snapshot re-serializes the payloads; the same image referenced by ten messages is ten copies of the bytes. Media externalization solves that: content-addressed stores write each payload once, keyed by its own hash, and leave a short `media+sha256://` URI in its place. Reach for it whenever large binary or text payloads would otherwise balloon what you store or send.

!!! note "Import path"
    Import these helpers from their submodule -- there is no top-level `pydantic_ai_harness` re-export:

    ```python
    from pydantic_ai_harness.media import (
        DiskMediaStore,
        S3MediaStore,
        SqliteMediaStore,
        externalize_media,
        restore_media,
    )
    ```

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Building blocks, not a capability

These are building blocks. There is no class you add to `Agent(capabilities=[...])` yet. [`StepPersistence`](step-persistence.md) already uses them to keep snapshots small when messages carry `BinaryContent` or large text (e.g. a big tool-return string), and a forthcoming `MediaExternalizer` capability ([#254](https://github.com/pydantic/pydantic-ai-harness/issues/254)) will reuse the same stores to rewrite `BinaryContent` into URL parts before the model sees them.

## Why content-addressing

The URI is derived from the payload hash, so identical bytes deduplicate automatically. The same bytes are stored once no matter how many messages or snapshots reference them, and moving the underlying storage is a one-line swap because the URI does not change.

## Stores

Every store implements the `MediaStore` protocol -- `put`, `get`, `exists`, `public_url`, and `get_metadata`, all async and content-addressed.

| Store | Backed by | Use when |
|---|---|---|
| `DiskMediaStore(directory=...)` | A directory on disk | Local runs and tests |
| `SqliteMediaStore(database=...)` | A SQLite database | A single-file store that travels with the data |
| `S3MediaStore(bucket=, endpoint=, region=, ...)` | S3 or an S3-compatible bucket | Shared or production storage |
| `MongoMediaStore(client= or db_url=, database=, ...)` | MongoDB (sha256-addressed manual chunking) | A MongoDB deployment; blobs larger than one BSON document |

`S3MediaStore` uses path-style URLs plus handrolled SigV4, so it is compatible with AWS S3, Cloudflare R2 (`region='auto'`), MinIO, and other S3-compatible providers. `SqliteMediaStore` also accepts `connection=` instead of `database=` to share a `sqlite3.Connection`.

`MongoMediaStore` needs the `mongodb` extra (`pip install pydantic-ai-harness[mongodb]`, which installs `pymongo>=4.17.0`). Pass a shared `AsyncMongoClient` as `client=`, or a connection string as `db_url=` (the store then owns the client -- call `await store.aclose()` to release it); `database=` is always required. Each blob is stored as sha256-addressed chunks in a `media_chunks` collection, with a `media` manifest document per blob (`_id = <digest>`). The chunking bounds each BSON document, so a blob larger than MongoDB's 16 MiB document cap still stores and reads back. It does not bound memory: `put` takes the whole payload as `bytes` and `get` reassembles every chunk into one `bytearray`, so a blob has to fit in process memory in both directions -- there is no streaming API. The manifest holds `MediaContext.metadata` inline and is not chunked, so keep per-blob metadata small. Manual chunking is used rather than the GridFS driver on purpose: the digest is the manifest `_id`, so identical bytes deduplicate (GridFS keys files by `ObjectId` and does no dedup), and the plain-collection surface stays fully testable in-memory.

Two constructor knobs shape that layout. `collection=` (default `'media'`) names the manifest collection and derives the chunk collection as `<collection>_chunks`; names outside `[A-Za-z_][A-Za-z0-9_]*` are rejected. `chunk_size_bytes=` (default 8 MiB) sets the split size and is rejected below 1 byte or above 16 MiB minus 64 KiB of headroom for the chunk document's own fields, since a larger chunk would build a document MongoDB refuses on insert.

On its first `put` or `get`, the store issues `createIndex` for a compound `(files_id, n)` index on the chunk collection, without which reassembly is a collection scan. The connecting user therefore needs the privilege to create indexes -- a restricted Atlas role may not have it -- and pointing the store at an already-populated collection pays the index build on that first call.

```python
from pymongo import AsyncMongoClient

from pydantic_ai_harness.media import MongoMediaStore

client = AsyncMongoClient('mongodb://localhost:27017')
store = MongoMediaStore(client=client, database='agent_media')
```

## Walker helpers

`externalize_media` and `restore_media` walk a message node and swap payloads for URIs and back:

```python
from pydantic_ai_harness.media import DiskMediaStore, externalize_media, restore_media

store = DiskMediaStore(directory='./media')

# Replace binary and text payloads at or above the threshold with media+sha256:// URIs.
lean = await externalize_media(message, media_store=store, threshold_bytes=32_000)

# Later, rehydrate the URIs back into the original parts.
full = await restore_media(lean, media_store=store)
```

`externalize_media` externalizes both large `BinaryContent` and large text: any message part whose string `content` reaches `threshold_bytes` UTF-8 bytes (`TextPart`, `ThinkingPart`, a string-returning `ToolReturnPart`, a string-valued `UserPromptPart`), plus any `TextContent` element travelling inside a `UserPromptPart.content` sequence or a `ToolReturn`. The same `threshold_bytes` governs binary and text, and payloads below it stay inline. Round-trip is transparent -- `restore_media` re-inlines binary bytes and text symmetrically. If you need to key media yourself, `media_uri_for` and `parse_media_uri` give you the raw URI round-trip.

The current reader restores binary markers written before text externalization. That compatibility is upgrade-only: a release that predates text externalization treats every marker as binary, so it cannot validate a snapshot containing an externalized text marker. Keep a current reader for persisted snapshots that contain those markers.

If a payload uses the same namespaced keys as the marker format, the writer moves those values into a versioned reserved mapping and the current reader restores them. A marker written before this escaping format was added still restores. The reader treats the reserved keys as its own only when the reserved mapping has the shape the writer produces and the version key carries a stamp from the format's own namespace, so a payload that happened to carry either or both is left alone. A marker carrying both, stamped with a version this reader does not know, is rejected rather than restored with the reserved values stripped. Compatibility the other way is upgrade-only. A reader that predates the escaping format re-inlines the externalized field correctly, but it leaves the caller's values in the reserved mapping instead of restoring them to their own keys, and leaves the two reserved keys in the payload. Keep a current reader for persisted snapshots that contain escaped markers.

## Public URLs

When a store is fronted by a CDN, a local HTTP server, or a signed-URL service, pass a `public_url=` resolver (or use `make_static_public_url`) to turn a stored `media+sha256://` URI into a URL the model can fetch directly. Without a resolver, `public_url(...)` returns `None`.

A static base URL, for a public bucket or CDN:

```python
from pydantic_ai_harness.media import S3MediaStore, make_static_public_url

store = S3MediaStore(
    bucket='my-bucket',
    endpoint='https://<acc>.r2.cloudflarestorage.com',
    region='auto',
    access_key_id=..., secret_access_key=...,
    key_prefix='media/',
    public_url=make_static_public_url('https://pub-abc.r2.dev', key_prefix='media/'),
)
```

A presigned or rotating-signature URL -- pass any async callable that takes `(uri, MediaContext)`:

```python
from pydantic_ai_harness.media import MediaContext, S3MediaStore


async def presign(uri: str, ctx: MediaContext) -> str:
    key = 'media/' + uri.removeprefix('media+sha256://') + '.bin'
    return await my_signer.generate(key, ttl=3600, content_type=ctx.media_type)


store = S3MediaStore(..., public_url=presign)
```

This is what the forthcoming `MediaExternalizer` will use to swap `BinaryContent` parts for `ImageUrl` / `AudioUrl` / other URL parts before the model sees the message, letting providers fetch big media over the wire without re-encoding bytes into the request body. Emitting a URL is always safe: pydantic-ai providers transparently download the bytes when the target model does not natively accept that URL type, so you only ever lose wire savings, never correctness.

## `MediaContext`

Every store method and both user-supplied callables (`PublicUrlResolver`, `KeyStrategy`) accept a `MediaContext` -- an extensible per-operation bag:

```python
from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, kw_only=True)
class MediaContext:
    media_type: str | None = None                    # e.g. 'image/png'
    filename: str | None = None                      # original filename, when known
    metadata: Mapping[str, str] = field(default_factory=dict)  # user-supplied tags
```

All fields default, so you pass what you have and ignore the rest; new fields are added non-breakingly as use cases emerge. `get_metadata(uri)` round-trips the user-supplied `metadata` mapping on all four stores; `media_type` is persisted separately (as the byte payload's `Content-Type`).

## `KeyStrategy`

The default on-store key layout is `<sha256>.bin`. `DiskMediaStore` and `S3MediaStore` accept a `key_strategy=` override to fit an existing layout. `SqliteMediaStore` and `MongoMediaStore` do not, since the digest is their primary key -- use `table=` / `collection=` to move the rows or documents instead:

```python
from pydantic_ai_harness.media import DiskMediaStore, MediaContext


def by_media_type(uri: str, ctx: MediaContext) -> str:
    digest = uri.removeprefix('media+sha256://')
    ext = {'image/png': '.png', 'image/jpeg': '.jpg'}.get(ctx.media_type or '', '.bin')
    return f'images/{digest}{ext}'


store = DiskMediaStore('runs', key_strategy=by_media_type)
```

If your strategy depends on `ctx.media_type`, the same context must be supplied at read time for `get`/`exists` to find the blob. `DiskMediaStore` rejects strategies that produce absolute paths or `..` segments, to keep writes inside the store directory. `default_key_strategy` is exported if you want to build on it.

## API

| Symbol | Purpose |
|---|---|
| `MediaStore` | Async content-addressed store protocol (`put` / `get` / `exists` / `public_url` / `get_metadata`) |
| `DiskMediaStore`, `SqliteMediaStore`, `S3MediaStore`, `MongoMediaStore` | Concrete stores (`MongoMediaStore` needs the `mongodb` extra) |
| `MediaContext` | Per-operation context (media type, filename, tags) threaded through store operations |
| `KeyStrategy`, `default_key_strategy` | On-store key layout |
| `PublicUrlResolver`, `make_static_public_url` | Resolve a stored URI to a public URL |
| `externalize_media`, `restore_media` | Walk a message node to externalize / rehydrate large binary and text payloads |
| `media_uri_for`, `parse_media_uri` | Compute and parse a `media+sha256://` URI |

Source: [`pydantic_ai_harness/media/`](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/media/).

## Related

- [Step Persistence](step-persistence.md) -- the first consumer of these stores, externalizing large `BinaryContent` and text parts in run snapshots.
