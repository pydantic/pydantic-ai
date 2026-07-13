---
title: Media Externalization
description: Content-addressed stores and walker helpers that move large BinaryContent payloads out of message history into deduplicated storage and put them back on demand.
---

# Media Externalization

A conversation that carries images, audio, or other `BinaryContent` inlines those bytes into every message. Persist that history and each snapshot re-serializes the payloads; the same image referenced by ten messages is ten copies of the bytes. Media externalization solves that: content-addressed stores write each payload once, keyed by its own hash, and leave a short `media+sha256://` URI in its place. Reach for it whenever binary payloads would otherwise balloon what you store or send.

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

> The API may change between releases. Where practical, breaking changes ship with a deprecation warning.

## Building blocks, not a capability

These are building blocks. There is no class you add to `Agent(capabilities=[...])` yet. [`StepPersistence`](step-persistence.md) already uses them to keep snapshots small when messages carry `BinaryContent`, and a forthcoming `MediaExternalizer` capability will reuse the same stores to rewrite `BinaryContent` into URL parts before the model sees them.

## Why content-addressing

The URI is derived from the payload hash, so identical bytes deduplicate automatically. The same bytes are stored once no matter how many messages or snapshots reference them, and moving the underlying storage is a one-line swap because the URI does not change.

## Stores

Every store implements the `MediaStore` protocol -- `put`, `get`, `exists`, `public_url`, and `get_metadata`, all async and content-addressed.

| Store | Backed by | Use when |
|---|---|---|
| `DiskMediaStore(directory=...)` | A directory on disk | Local runs and tests |
| `SqliteMediaStore(database=...)` | A SQLite database | A single-file store that travels with the data |
| `S3MediaStore(bucket=, endpoint=, region=, ...)` | S3 or an S3-compatible bucket | Shared or production storage |

`S3MediaStore` uses path-style URLs plus handrolled SigV4, so it is compatible with AWS S3, Cloudflare R2 (`region='auto'`), MinIO, and other S3-compatible providers. `SqliteMediaStore` also accepts `connection=` instead of `database=` to share a `sqlite3.Connection`.

## Walker helpers

`externalize_media` and `restore_media` walk a message node and swap payloads for URIs and back:

```python
from pydantic_ai_harness.media import DiskMediaStore, externalize_media, restore_media

store = DiskMediaStore(directory='./media')

# Replace BinaryContent larger than the threshold with media+sha256:// URIs.
lean = await externalize_media(message, media_store=store, threshold_bytes=32_000)

# Later, rehydrate the URIs back into BinaryContent.
full = await restore_media(lean, media_store=store)
```

`externalize_media` only externalizes payloads over `threshold_bytes`; smaller ones stay inline. Round-trip is transparent -- `restore_media` returns `BinaryContent` with the original bytes. If you need to key media yourself, `media_uri_for` and `parse_media_uri` give you the raw URI round-trip.

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

All fields default, so you pass what you have and ignore the rest; new fields are added non-breakingly as use cases emerge. `get_metadata(uri)` round-trips the user-supplied `metadata` mapping on all three stores; `media_type` is persisted separately (as the byte payload's `Content-Type`).

## `KeyStrategy`

The default on-store key layout is `<sha256>.bin`. `DiskMediaStore` and `S3MediaStore` accept a `key_strategy=` override to fit an existing layout; `SqliteMediaStore` does not, since its primary key is the digest:

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
| `DiskMediaStore`, `SqliteMediaStore`, `S3MediaStore` | Concrete stores |
| `MediaContext` | Per-operation context (media type, filename, tags) threaded through store operations |
| `KeyStrategy`, `default_key_strategy` | On-store key layout |
| `PublicUrlResolver`, `make_static_public_url` | Resolve a stored URI to a public URL |
| `externalize_media`, `restore_media` | Walk a message node to externalize / rehydrate payloads |
| `media_uri_for`, `parse_media_uri` | Compute and parse a `media+sha256://` URI |

Source: [`pydantic_ai_harness/media/`](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/media/).

## Related

- [Step Persistence](step-persistence.md) -- the first consumer of these stores, externalizing `BinaryContent` in run snapshots.
