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
> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

Content-addressed stores and walker helpers that move large binary and text payloads out of message history and put them back on demand.

These are building blocks, not a capability. There is no class you add to `Agent(capabilities=[...])` yet. [`StepPersistence`](../step_persistence/) uses them to keep snapshots small when messages carry `BinaryContent` or large text (e.g. a big tool-return string). A forthcoming `MediaExternalizer` capability ([#254](https://github.com/pydantic/pydantic-ai-harness/issues/254)) will reuse the same stores to rewrite `BinaryContent` into URL parts before the model sees them.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/media/)

## Why

A conversation that carries images, audio, or other `BinaryContent` inlines those bytes into every message, and a large text part (a big tool-return string, say) is just as heavy. Persist that history and each snapshot re-serializes the payloads. Content-addressed storage writes each payload once, keyed by its own hash, and leaves a short `media+sha256://` URI in its place. The same bytes are stored once no matter how many messages or snapshots reference them.

## Stores

Every store implements the `MediaStore` protocol: `put`, `get`, `exists`, `public_url`, and `get_metadata`, all async and content-addressed (the URI is derived from the payload hash, so identical bytes deduplicate).

| Store | Backed by | Use when |
|---|---|---|
| `DiskMediaStore(directory=...)` | A directory on disk | Local runs and tests |
| `SqliteMediaStore(...)` | A SQLite database | A single-file store that travels with the data |
| `S3MediaStore(...)` | S3 or an S3-compatible bucket | Shared or production storage |
| `MongoMediaStore(...)` | MongoDB (sha256-addressed manual chunking) | A MongoDB deployment; blobs larger than one BSON document, split so no chunk hits the 16 MiB cap |

`MongoMediaStore` needs the `mongodb` extra (`pip install pydantic-ai-harness[mongodb]`, which installs `pymongo>=4.17.0`) and is imported the same way (`from pydantic_ai_harness.media import MongoMediaStore`). It stores each blob as sha256-addressed chunks in a `media_chunks` collection, with a `media` manifest document per blob. The chunking bounds each BSON document, so a blob larger than MongoDB's 16 MiB document cap still stores and reads back; it does not bound memory, since `put` takes the whole payload as `bytes` and `get` reassembles every chunk into one `bytearray` (there is no streaming API). The manifest itself holds `MediaContext.metadata` inline and is not chunked, so keep per-blob metadata small. Manual chunking is used instead of GridFS: it keeps content-addressed dedup (GridFS keys files by `ObjectId` and does none) and stays fully testable in-memory.

`collection=` (default `'media'`) names the manifest collection and derives the chunk collection as `<collection>_chunks`. `chunk_size_bytes=` (default 8 MiB) sets the split size and is rejected below 1 byte or above 16 MiB minus 64 KiB of headroom for the chunk document's own fields. On the first `put` or `get` the store issues `createIndex` for a compound `(files_id, n)` index on the chunk collection, without which reassembly is a collection scan -- so the connecting user needs the privilege to create indexes, and an already-populated collection pays the index build on that first call.

A `KeyStrategy` controls the on-store layout for `DiskMediaStore` and `S3MediaStore` (`SqliteMediaStore` and `MongoMediaStore` key on the digest itself, so they take `table=` / `collection=` instead), and a `PublicUrlResolver` (or `make_static_public_url`) turns a stored URI into a public URL when the store is served over HTTP.

## Walker helpers

`externalize_media` and `restore_media` walk a message node and swap payloads for URIs and back.

```python
store = DiskMediaStore(directory='./media')

# Replace binary and text payloads at or above the threshold with media+sha256:// URIs.
lean = await externalize_media(message, media_store=store, threshold_bytes=32_000)

# Later, rehydrate the URIs back into the original parts.
full = await restore_media(lean, media_store=store)
```

`externalize_media` externalizes both large `BinaryContent` and large text: any message part whose string `content` is at least `threshold_bytes` UTF-8 bytes (`TextPart`, `ThinkingPart`, a string-returning `ToolReturnPart`, a string-valued `UserPromptPart`), plus any `TextContent` element travelling inside a `UserPromptPart.content` sequence or a `ToolReturn`. The same `threshold_bytes` governs both; there is no separate text knob. Payloads below the threshold stay inline, and `restore_media` re-inlines binary and text symmetrically. `media_uri_for` and `parse_media_uri` give you the raw URI round-trip if you need to key media yourself.

The current reader restores binary markers written before text externalization. That compatibility is upgrade-only: a release that predates text externalization treats every marker as binary, so it cannot validate a snapshot containing an externalized text marker. Keep a current reader for persisted snapshots that contain those markers.

## API

| Symbol | Purpose |
|---|---|
| `MediaStore` | Async content-addressed store protocol (`put` / `get` / `exists` / `public_url` / `get_metadata`) |
| `DiskMediaStore`, `SqliteMediaStore`, `S3MediaStore`, `MongoMediaStore` | Concrete stores (`MongoMediaStore` needs the `mongodb` extra) |
| `MediaContext` | Per-call context (e.g. tenant) threaded through store operations |
| `KeyStrategy`, `default_key_strategy` | On-store key layout |
| `PublicUrlResolver`, `make_static_public_url` | Resolve a stored URI to a public URL |
| `externalize_media`, `restore_media` | Walk a message node to externalize / rehydrate large binary and text payloads |
| `media_uri_for`, `parse_media_uri` | Compute and parse a `media+sha256://` URI |
