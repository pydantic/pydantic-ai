# Media

> **Experimental.** Imported from `pydantic_ai_harness.experimental.media`, this module can change or be removed in any release. Importing it emits a `HarnessExperimentalWarning`.

```python
from pydantic_ai_harness.experimental.media import (
    DiskMediaStore,
    S3MediaStore,
    SqliteMediaStore,
    externalize_media,
    restore_media,
)
```

Content-addressed stores and walker helpers that move large binary payloads out of message history and put them back on demand.

These are building blocks, not a capability. There is no class you add to `Agent(capabilities=[...])` yet. [`StepPersistence`](../step_persistence/) uses them to keep snapshots small when messages carry `BinaryContent`. A forthcoming `MediaExternalizer` capability will reuse the same stores to rewrite `BinaryContent` into URL parts before the model sees them.

## Why

A conversation that carries images, audio, or other `BinaryContent` inlines those bytes into every message. Persist that history and each snapshot re-serializes the payloads. Content-addressed storage writes each payload once, keyed by its own hash, and leaves a short `media://` URI in its place. The same bytes are stored once no matter how many messages or snapshots reference them.

## Stores

Every store implements the `MediaStore` protocol: `put`, `get`, `exists`, `public_url`, and `get_metadata`, all async and content-addressed (the URI is derived from the payload hash, so identical bytes deduplicate).

| Store | Backed by | Use when |
|---|---|---|
| `DiskMediaStore(directory=...)` | A directory on disk | Local runs and tests |
| `SqliteMediaStore(...)` | A SQLite database | A single-file store that travels with the data |
| `S3MediaStore(...)` | S3 or an S3-compatible bucket | Shared or production storage |

A `KeyStrategy` controls the on-store layout, and a `PublicUrlResolver` (or `make_static_public_url`) turns a stored URI into a public URL when the store is served over HTTP.

## Walker helpers

`externalize_media` and `restore_media` walk a message node and swap payloads for URIs and back.

```python
store = DiskMediaStore(directory='./media')

# Replace BinaryContent larger than the threshold with media:// URIs.
lean = await externalize_media(message, media_store=store, threshold_bytes=32_000)

# Later, rehydrate the URIs back into BinaryContent.
full = await restore_media(lean, media_store=store)
```

`externalize_media` only externalizes payloads over `threshold_bytes`; smaller ones stay inline. `media_uri_for` and `parse_media_uri` give you the raw URI round-trip if you need to key media yourself.

## API

| Symbol | Purpose |
|---|---|
| `MediaStore` | Async content-addressed store protocol (`put` / `get` / `exists` / `public_url` / `get_metadata`) |
| `DiskMediaStore`, `SqliteMediaStore`, `S3MediaStore` | Concrete stores |
| `MediaContext` | Per-call context (e.g. tenant) threaded through store operations |
| `KeyStrategy`, `default_key_strategy` | On-store key layout |
| `PublicUrlResolver`, `make_static_public_url` | Resolve a stored URI to a public URL |
| `externalize_media`, `restore_media` | Walk a message node to externalize / rehydrate payloads |
| `media_uri_for`, `parse_media_uri` | Compute and parse a `media://` URI |
