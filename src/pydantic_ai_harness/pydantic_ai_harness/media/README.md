# Media Externalization

> [!NOTE]
> Import these helpers from their submodule -- there is no top-level `pydantic_ai_harness` re-export:
>
> ```python
> from pydantic_ai_harness.media import (
>     DiskMediaStore,
>     S3MediaStore,
>     SqliteMediaStore,
>     externalize_media,
>     restore_media,
> )
> ```
>
> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

Content-addressed stores for large `BinaryContent` and text payloads, plus helpers that swap those payloads for `media+sha256://<digest>` URIs in serialized message history and back. [`StepPersistence`](../step_persistence/) uses them to keep run snapshots small; use them directly when you configure a store yourself or serialize history some other way.

This package does not define an agent capability and emits no telemetry.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/media/)

## Quick start

The helpers take the JSON form of a message history, not message objects:

```python
import json

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_ai_harness.media import DiskMediaStore, externalize_media, restore_media

store = DiskMediaStore('./media')


async def save(messages: list[ModelMessage]) -> str:
    tree = json.loads(ModelMessagesTypeAdapter.dump_json(messages))
    lean = await externalize_media(tree, media_store=store, threshold_bytes=32_000)
    return json.dumps(lean)


async def load(saved: str) -> list[ModelMessage]:
    restored = await restore_media(json.loads(saved), media_store=store)
    return ModelMessagesTypeAdapter.validate_python(restored)
```

A payload's URI is its sha256, so identical bytes are stored once however many messages or snapshots reference them.

## Stores

Every store implements the `MediaStore` protocol: `put`, `get`, `exists`, `public_url`, and `get_metadata`, all async.

| Store | Backed by | Use when |
|---|---|---|
| `DiskMediaStore(directory)` | A directory on disk | Local runs and tests |
| `SqliteMediaStore(database=...)` | A SQLite database | A single-file store that travels with the data |
| `S3MediaStore(bucket=, endpoint=, region=, ...)` | S3 or an S3-compatible bucket | Shared or production storage |
| `MongoMediaStore(client= or db_url=, database=, ...)` | MongoDB, chunked documents | A MongoDB deployment |
| `PostgresMediaStore(pool, table=...)` | PostgreSQL, one `BYTEA` row per blob | A PostgreSQL deployment; blobs up to 1 GB |

None of the stores stream: `put` takes the whole payload and `get` returns it, so a blob has to fit in process memory.

`SqliteMediaStore` also accepts `connection=` instead of `database=` to share a `sqlite3.Connection` (created with `check_same_thread=False`). On that connection, `put` commits its insert when no transaction is open and otherwise leaves the commit to you.

`S3MediaStore` uses path-style URLs and its own SigV4 signing, so it works with AWS S3, Cloudflare R2 (`region='auto'`), MinIO, and other S3-compatible providers. It sends one PUT per blob, so a blob is limited to S3's 5 GB single-PUT cap.

`MongoMediaStore` needs the `mongodb` extra.

uv:

```bash
uv add "pydantic-ai-harness[mongodb]"
```

pip:

```bash
pip install "pydantic-ai-harness[mongodb]"
```

- Pass a shared `AsyncMongoClient` as `client=`, or a connection string as `db_url=` (the store then owns the client; call `await store.aclose()`). `database=` is required.
- Each blob is a manifest document in `media` (`_id` is the digest, `collection=` renames it) plus chunks in `media_chunks`. `chunk_size_bytes` (default 8 MiB, at most 16 MiB minus 64 KiB) keeps each chunk under MongoDB's 16 MiB document cap. The manifest holds `MediaContext.metadata` unchunked, so keep metadata small.
- `get` reads the manifest's chunks and checks their sha256 against the URI, so a missing or corrupt chunk raises `ValueError`.
- The first `put` or `get` creates a `(files_id, n)` index on the chunk collection, so the user needs the privilege to create indexes.
- GridFS is not used because it keys files by `ObjectId` and cannot deduplicate by content.

`PostgresMediaStore` takes a caller-owned asyncpg-compatible pool (the `PostgresPool` protocol) and never closes it. The harness defines no extra for the driver.

uv:

```bash
uv add asyncpg
```

pip:

```bash
pip install asyncpg
```

- Each blob is one row in `media`, keyed by its digest; a second `put` of the same bytes is a no-op. `table=` renames the table and accepts lowercase names matching `[a-z_][a-z0-9_]*`, up to 63 characters.
- The first operation runs `CREATE TABLE IF NOT EXISTS` under a `pg_advisory_xact_lock`, so the role needs `CREATE` on the schema. An existing table of the same name with another layout is not migrated, and the first `put` or `get` fails on a missing column; pass `table=` in that case.

```python
import asyncpg

from pydantic_ai_harness.media import PostgresMediaStore


async def build_media_store() -> tuple[PostgresMediaStore, asyncpg.Pool]:
    pool = await asyncpg.create_pool('postgres://localhost/app')
    return PostgresMediaStore(pool), pool
```

Close the returned pool during application shutdown.

## Metadata

`put(data, context=MediaContext(metadata={...}))` stores a `str` to `str` mapping that `get_metadata(uri)` returns. `get_metadata` raises `FileNotFoundError` for a blob that was never stored.

| Store | Stored as | A later `put` with other metadata |
|---|---|---|
| `DiskMediaStore` | `<blob>.meta.json` sidecar | Replaces it |
| `SqliteMediaStore`, `PostgresMediaStore` | JSON column | Is ignored |
| `MongoMediaStore` | JSON string on the manifest | Is ignored |
| `S3MediaStore` | Signed `x-amz-meta-*` headers | Replaces it |

S3 keys must match `[A-Za-z0-9-]+` and come back lowercased. `get_metadata` never returns `media_type`: S3 sends it as `Content-Type`, the database stores keep it in a column, and `DiskMediaStore` does not persist it.

## Walker helpers

`externalize_media` replaces each payload of at least `threshold_bytes` with a marker that carries its URI, and `restore_media` puts the payloads back. Both walk a JSON tree such as `json.loads(ModelMessagesTypeAdapter.dump_json(messages))` or `dump_python(messages, mode='json')`; anything else, including message objects, raises `TypeError`.

They externalize:

- `BinaryContent` data, measured in decoded bytes. `data` that is not canonical base64 stays inline, since a tool can return a dict of the same shape. The restored `data` uses the source's base64 alphabet.
- Any message part whose string `content` is at least `threshold_bytes` UTF-8 bytes: `TextPart`, `ThinkingPart`, a string-returning `ToolReturnPart`, a string-valued `UserPromptPart`.
- `TextContent` inside a `UserPromptPart.content` sequence or a `ToolReturn`.

Markers use reserved `__harness_external_*` keys. Caller data that uses those keys, such as a tool return, is escaped on write and restored on read, so it is never read back as a marker. `media_uri_for` and `parse_media_uri` compute and parse URIs if you key media yourself.

Older snapshots restore with a newer release; the reverse is not supported. Read persisted snapshots with a release at least as new as the one that wrote them. The [Step Persistence README](../step_persistence/) lists the marker-format changes.

## Public URLs

Pass a `public_url=` resolver to turn a stored URI into a URL the model can fetch, for a store behind a CDN, a local HTTP server, or a signed-URL service. Without one, `public_url(...)` returns `None`.

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

A presigned URL, from any sync or async callable that takes `(uri, MediaContext)`:

```python
from pydantic_ai_harness.media import MediaContext, S3MediaStore


async def presign(uri: str, ctx: MediaContext) -> str:
    key = 'media/' + uri.removeprefix('media+sha256://') + '.bin'
    return await my_signer.generate(key, ttl=3600, content_type=ctx.media_type)


store = S3MediaStore(..., public_url=presign)
```

## `MediaContext`

Every store method and both user-supplied callables (`PublicUrlResolver`, `KeyStrategy`) take a `MediaContext`. All fields are optional, and new ones are added without breaking callers.

| Field | Default | Used by |
|---|---|---|
| `media_type` | `None` | S3 `Content-Type`, key strategies, resolvers |
| `filename` | `None` | Your key strategy or resolver |
| `metadata` | `{}` | Persisted per store; read with `get_metadata` |

## `KeyStrategy`

`DiskMediaStore` and `S3MediaStore` store each blob at `<sha256>.bin` by default. Pass `key_strategy=` to fit an existing layout. The SQLite, MongoDB, and PostgreSQL stores key on the digest, so they take `table=` or `collection=` instead.

```python
from pydantic_ai_harness.media import DiskMediaStore, MediaContext, parse_media_uri


def by_media_type(uri: str, ctx: MediaContext) -> str:
    ext = {'image/png': '.png', 'image/jpeg': '.jpg'}.get(ctx.media_type or '', '.bin')
    return f'images/{parse_media_uri(uri)}{ext}'


store = DiskMediaStore('runs', key_strategy=by_media_type)
```

A strategy that reads `ctx.media_type` needs the same context on `get` and `exists`; `restore_media` passes the part's media type. `DiskMediaStore` rejects URIs that are not `media+sha256://` URIs and keys that are absolute or contain `..`. `default_key_strategy` is exported if you want to build on it.

## API

| Symbol | Purpose |
|---|---|
| `MediaStore` | Async content-addressed store protocol (`put` / `get` / `exists` / `public_url` / `get_metadata`) |
| `DiskMediaStore`, `SqliteMediaStore`, `S3MediaStore`, `MongoMediaStore`, `PostgresMediaStore` | Concrete stores (`MongoMediaStore` needs the `mongodb` extra) |
| `PostgresPool`, `PostgresConnection` | The asyncpg-compatible pool and connection protocols `PostgresMediaStore` accepts |
| `MediaContext` | Per-operation context (media type, filename, tags) threaded through store operations |
| `KeyStrategy`, `default_key_strategy` | On-store key layout |
| `PublicUrlResolver`, `make_static_public_url` | Resolve a stored URI to a public URL |
| `externalize_media`, `restore_media` | Walk a JSON message tree to externalize / rehydrate large binary and text payloads |
| `media_uri_for`, `parse_media_uri` | Compute and parse a `media+sha256://` URI |
