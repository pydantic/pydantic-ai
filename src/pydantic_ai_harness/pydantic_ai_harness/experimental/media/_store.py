"""Local `MediaStore` protocol + `DiskMediaStore` / `SqliteMediaStore`.

The shared URI scheme is `media+sha256://<lowercase-hex>` -- content-addressed,
so the same blob written through any store resolves the same way and dedup
is automatic.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import sqlite3
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, TypeGuard, runtime_checkable

import anyio.to_thread

_URI_SCHEME = 'media+sha256://'
_HEX_RE = re.compile(r'^[0-9a-f]{64}$')

# Sentinel: empty, shareable, immutable. Used as the default `context` on every
# `MediaStore` method. Pulled into a module constant so a default-bound empty
# context is always the same instance (cheap identity checks, no per-call
# allocation).
_EMPTY_METADATA: Mapping[str, str] = MappingProxyType({})


@dataclass(frozen=True, kw_only=True)
class MediaContext:
    """Per-operation context threaded through `MediaStore` methods + callables.

    Extensible bag -- new use cases (TTL hints, response-header overrides,
    origin run ids for audit, etc.) add a field here without breaking any
    method signature or user-supplied callable. Fields default to `None` /
    empty so callers and resolvers can ignore what they don't care about.

    Conventions:

    - `media_type` -- IANA media type (`image/png`, `audio/wav`, ...). When
      absent, callbacks may guess from `filename` (stdlib `mimetypes`) or
      fall back to a generic default. Never required.
    - `filename` -- original filename for the bytes, if known. Useful for
      key strategies that want a recognisable extension, and for resolvers
      that need to set `Content-Disposition` on presigned URLs.
    - `metadata` -- free-form `dict[str, str]` of user-supplied tags.
      Persisted by all three concrete stores (`SqliteMediaStore` → JSON
      column, `S3MediaStore` → signed `x-amz-meta-*` headers,
      `DiskMediaStore` → sidecar JSON file). Read back via
      `MediaStore.get_metadata(uri)`.
    """

    media_type: str | None = None
    filename: str | None = None
    metadata: Mapping[str, str] = field(default_factory=lambda: _EMPTY_METADATA)


_EMPTY_CONTEXT = MediaContext()


PublicUrlResolver = Callable[[str, MediaContext], 'str | None | Awaitable[str | None]']
"""User-supplied callable that turns a `media+sha256://<hex>` URI into a fetchable URL.

Sync or async; the store auto-detects via `inspect.isawaitable` on the result.
Return `None` to signal "no URL for this URI" (the resolver may be content-aware
and choose to inline some payloads).

Receives the full `MediaContext` so the resolver can vary its output by media
type, response headers, TTL, etc. without breaking the signature later.
"""

KeyStrategy = Callable[[str, MediaContext], str]
"""User-supplied callable that turns a URI + context into a backend storage key.

Used by `DiskMediaStore` and `S3MediaStore` to derive the on-disk path / S3
object key from the canonical URI. `SqliteMediaStore` does not take one: its
primary key is the content digest, so a user-chosen key would break dedup.
Default strategy is `<sha256>.bin` (see `default_key_strategy`); override when
an existing bucket layout or naming convention dictates otherwise.

**Caveat**: if your strategy depends on `context.media_type` (e.g. to pick an
extension), `get(uri)` and `exists(uri)` won't find the blob unless callers
pass the same context at read time. For pure path-organisation strategies
(e.g. `'media/' + digest + '.bin'`) the context can be ignored entirely.
"""


def default_key_strategy(uri: str, context: MediaContext) -> str:
    """Default storage key: `<sha256>.bin`. Ignores context."""
    return f'{parse_media_uri(uri)}.bin'


async def _resolve_public_url(
    resolver: PublicUrlResolver | None,
    uri: str,
    context: MediaContext,
) -> str | None:
    if resolver is None:
        return None
    result = resolver(uri, context)
    if inspect.isawaitable(result):
        return await result
    return result


def make_static_public_url(
    base: str,
    *,
    key_prefix: str = '',
    extension: str = '.bin',
) -> PublicUrlResolver:
    """Build a static-URL resolver for the common "public bucket / CDN" case.

    The returned callable takes a `media+sha256://<hex>` URI and returns
    `{base}/{key_prefix}<hex>{extension}`. Use this when the bytes live at a
    known stable URL -- e.g. an R2 public bucket (`https://pub-xxx.r2.dev`)
    or a CDN in front of S3. For presigned URLs or any logic that runs per
    request, pass your own callable instead.

    Ignores `MediaContext` -- pass your own callable if URL needs to vary
    with media type, TTL, etc.
    """
    base = base.rstrip('/')

    def resolver(uri: str, context: MediaContext) -> str:
        digest = parse_media_uri(uri)
        return f'{base}/{key_prefix}{digest}{extension}'

    return resolver


def media_uri_for(data: bytes) -> str:
    """Return the canonical `media+sha256://<hex>` URI for `data`.

    The URI is the same regardless of which store the bytes are written to --
    content-addressed so two stores holding the same bytes can be queried
    interchangeably.
    """
    return f'{_URI_SCHEME}{hashlib.sha256(data).hexdigest()}'


def parse_media_uri(uri: str) -> str:
    """Return the lowercase hex digest from a `media+sha256://<hex>` URI.

    Raises `ValueError` if `uri` does not match the scheme or the digest is
    not 64 lowercase hex characters.
    """
    if not uri.startswith(_URI_SCHEME):
        raise ValueError(f'not a media URI: {uri!r}')
    digest = uri[len(_URI_SCHEME) :]
    if not _HEX_RE.fullmatch(digest):
        raise ValueError(f'invalid sha256 digest in {uri!r}')
    return digest


@runtime_checkable
class MediaStore(Protocol):
    """Async content-addressed bytes store.

    `put` returns the canonical URI (`media+sha256://<hex>`) for the bytes --
    callers do not pick the key. Implementations may dedup on the hash.

    Every method takes an optional `context: MediaContext` for forward
    extensibility. New context fields don't break existing call sites or
    user-supplied callables.

    `public_url(uri)` returns a URL the model can fetch directly, or `None`
    if the store can't or hasn't been configured to produce one. The forthcoming
    `MediaExternalizer` capability uses this to rewrite `BinaryContent` parts
    to `ImageUrl`/`AudioUrl`/etc. before the model sees the message.
    """

    async def put(self, data: bytes, *, context: MediaContext = _EMPTY_CONTEXT) -> str: ...  # pragma: no cover

    async def get(self, uri: str, *, context: MediaContext = _EMPTY_CONTEXT) -> bytes: ...  # pragma: no cover

    async def exists(self, uri: str, *, context: MediaContext = _EMPTY_CONTEXT) -> bool: ...  # pragma: no cover

    async def public_url(
        self, uri: str, *, context: MediaContext = _EMPTY_CONTEXT
    ) -> str | None: ...  # pragma: no cover

    async def get_metadata(
        self, uri: str, *, context: MediaContext = _EMPTY_CONTEXT
    ) -> Mapping[str, str]: ...  # pragma: no cover


class DiskMediaStore:
    """Filesystem store. One file per blob; default layout is `<directory>/<sha256>.bin`.

    Directory is created on first write. Dedup is automatic via the
    content-hash filename.

    Customisation:

    - `key_strategy=` -- `Callable[[str, MediaContext], str]` overriding the
      relative path inside `directory` (e.g. `'images/<digest>.png'`).
      The returned path must be relative and free of `..` segments
      (absolute paths and `..` are both rejected with `ValueError`).
      Caveat: if the strategy uses `context.media_type` etc. to pick the
      filename, the same context must be supplied on `get`/`exists` to
      locate the blob.
    - `public_url=` -- resolver exposing a URL the model can fetch.
      Without it `public_url(...)` returns `None` (local filesystem paths
      are not addressable from a model).

    **Metadata persistence**: `context.metadata` is written to a sidecar
    JSON file (`<resolved>.meta.json`) alongside the blob and read back
    via `get_metadata(uri)`. Sidecars are written atomically (tmp +
    rename) and are missing only when no metadata was supplied.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        key_strategy: KeyStrategy = default_key_strategy,
        public_url: PublicUrlResolver | None = None,
    ) -> None:
        self._root = Path(directory)
        self._key_strategy = key_strategy
        self._public_url_resolver = public_url

    def _path_for(self, uri: str, context: MediaContext) -> Path:
        relative = self._key_strategy(uri, context)
        candidate = Path(relative)
        # `Path('/root') / '/abs'` discards `/root` and returns `/abs` -- an absolute
        # key would silently escape the store directory. Reject both shapes.
        if candidate.is_absolute() or '..' in candidate.parts:
            raise ValueError(f'key_strategy produced traversal-unsafe path: {relative!r}')
        return self._root / candidate

    async def put(self, data: bytes, *, context: MediaContext = _EMPTY_CONTEXT) -> str:
        return await anyio.to_thread.run_sync(self._sync_put, data, context)

    def _sync_put(self, data: bytes, context: MediaContext) -> str:
        uri = media_uri_for(data)
        path = self._path_for(uri, context)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            tmp = path.with_suffix(path.suffix + '.tmp')
            tmp.write_bytes(data)
            tmp.replace(path)
        if context.metadata:
            meta_path = path.with_name(path.name + '.meta.json')
            meta_tmp = meta_path.with_suffix('.json.tmp')
            meta_tmp.write_text(json.dumps(dict(context.metadata)))
            meta_tmp.replace(meta_path)
        return uri

    async def get(self, uri: str, *, context: MediaContext = _EMPTY_CONTEXT) -> bytes:
        return await anyio.to_thread.run_sync(self._sync_get, uri, context)

    def _sync_get(self, uri: str, context: MediaContext) -> bytes:
        path = self._path_for(uri, context)
        if not path.exists():
            raise FileNotFoundError(f'media not found: {uri}')
        return path.read_bytes()

    async def exists(self, uri: str, *, context: MediaContext = _EMPTY_CONTEXT) -> bool:
        return await anyio.to_thread.run_sync(self._sync_exists, uri, context)

    def _sync_exists(self, uri: str, context: MediaContext) -> bool:
        return self._path_for(uri, context).exists()

    async def public_url(self, uri: str, *, context: MediaContext = _EMPTY_CONTEXT) -> str | None:
        return await _resolve_public_url(self._public_url_resolver, uri, context)

    async def get_metadata(self, uri: str, *, context: MediaContext = _EMPTY_CONTEXT) -> Mapping[str, str]:
        return await anyio.to_thread.run_sync(self._sync_get_metadata, uri, context)

    def _sync_get_metadata(self, uri: str, context: MediaContext) -> Mapping[str, str]:
        path = self._path_for(uri, context)
        meta_path = path.with_name(path.name + '.meta.json')
        if not meta_path.exists():
            return _EMPTY_METADATA
        return _coerce_metadata_mapping(json.loads(meta_path.read_text()))


def _is_object_dict(raw: object) -> TypeGuard[dict[object, object]]:
    return isinstance(raw, dict)


def _coerce_metadata_mapping(raw: object) -> Mapping[str, str]:
    if not _is_object_dict(raw):
        raise ValueError(f'metadata payload must be a JSON object, got {type(raw).__name__}')
    out: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f'metadata entries must be str→str, got {key!r}: {value!r}')
        out[key] = value
    return out


class SqliteMediaStore:
    """SQLite store. One row per blob in a `media` table keyed by sha256 hex.

    Pass either a path to a SQLite file (created on demand) or an existing
    `sqlite3.Connection`. When a connection is passed the caller owns its
    lifecycle; when a path is passed each call opens a short-lived connection
    inside the worker thread (safe across event-loop threads).

    A caller-owned connection **must** be created with
    `check_same_thread=False`. Store methods dispatch SQL onto worker threads
    via `anyio.to_thread`, so the stdlib default (`check_same_thread=True`)
    raises `sqlite3.ProgrammingError` on first use. The path form sets this
    internally; a passed-in connection cannot, so it is the caller's
    responsibility.

    The table layout is:

    ```sql
    CREATE TABLE IF NOT EXISTS media (
        sha256 TEXT PRIMARY KEY,
        media_type TEXT,
        bytes BLOB NOT NULL,
        size_bytes INTEGER NOT NULL,
        metadata TEXT
    );
    ```

    `INSERT OR IGNORE` makes writes idempotent -- the second `put` with the
    same content is a no-op, not an overwrite. `metadata` is stored as
    canonical JSON of `context.metadata` (empty mapping → `'{}'`) and
    read back via `get_metadata(uri)`.

    There is no `key_strategy=` parameter: SqliteMediaStore is
    content-addressed and the digest IS the primary key -- any user-chosen
    storage key would either break dedup or be a no-op. Use `table=` if
    you need a non-default table name.
    """

    def __init__(
        self,
        *,
        database: str | Path | None = None,
        connection: sqlite3.Connection | None = None,
        table: str = 'media',
        public_url: PublicUrlResolver | None = None,
    ) -> None:
        if (database is None) == (connection is None):
            raise ValueError('provide exactly one of `database=` or `connection=`')
        self._database = Path(database) if database is not None else None
        self._connection = connection
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', table):
            raise ValueError(f'invalid table name: {table!r}')
        self._table = table
        self._schema_ready = False
        self._public_url_resolver = public_url

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        if self._schema_ready:
            return
        conn.execute(
            f'CREATE TABLE IF NOT EXISTS {self._table} ('
            'sha256 TEXT PRIMARY KEY, '
            'media_type TEXT, '
            'bytes BLOB NOT NULL, '
            'size_bytes INTEGER NOT NULL, '
            'metadata TEXT)'
        )
        conn.commit()
        self._schema_ready = True

    def _open(self) -> sqlite3.Connection:
        if self._connection is not None:
            return self._connection
        assert self._database is not None
        self._database.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._database, isolation_level=None, check_same_thread=False)
        conn.execute('PRAGMA journal_mode=WAL').close()
        return conn

    def _maybe_close(self, conn: sqlite3.Connection) -> None:
        if self._connection is None:
            conn.close()

    async def put(self, data: bytes, *, context: MediaContext = _EMPTY_CONTEXT) -> str:
        return await anyio.to_thread.run_sync(self._sync_put, data, context)

    def _sync_put(self, data: bytes, context: MediaContext) -> str:
        uri = media_uri_for(data)
        digest = parse_media_uri(uri)
        metadata_json = json.dumps(dict(context.metadata))
        conn = self._open()
        try:
            self._ensure_schema(conn)
            conn.execute(
                f'INSERT OR IGNORE INTO {self._table} '
                '(sha256, media_type, bytes, size_bytes, metadata) VALUES (?, ?, ?, ?, ?)',
                (digest, context.media_type, data, len(data), metadata_json),
            )
        finally:
            self._maybe_close(conn)
        return uri

    async def get(self, uri: str, *, context: MediaContext = _EMPTY_CONTEXT) -> bytes:
        digest = parse_media_uri(uri)
        return await anyio.to_thread.run_sync(self._sync_get, digest)

    def _sync_get(self, digest: str) -> bytes:
        conn = self._open()
        try:
            self._ensure_schema(conn)
            row = conn.execute(f'SELECT bytes FROM {self._table} WHERE sha256 = ?', (digest,)).fetchone()
        finally:
            self._maybe_close(conn)
        if row is None:
            raise FileNotFoundError(f'media not found: {digest}')
        return bytes(row[0])

    async def exists(self, uri: str, *, context: MediaContext = _EMPTY_CONTEXT) -> bool:
        digest = parse_media_uri(uri)
        return await anyio.to_thread.run_sync(self._sync_exists, digest)

    def _sync_exists(self, digest: str) -> bool:
        conn = self._open()
        try:
            self._ensure_schema(conn)
            row = conn.execute(f'SELECT 1 FROM {self._table} WHERE sha256 = ?', (digest,)).fetchone()
        finally:
            self._maybe_close(conn)
        return row is not None

    async def public_url(self, uri: str, *, context: MediaContext = _EMPTY_CONTEXT) -> str | None:
        return await _resolve_public_url(self._public_url_resolver, uri, context)

    async def get_metadata(self, uri: str, *, context: MediaContext = _EMPTY_CONTEXT) -> Mapping[str, str]:
        digest = parse_media_uri(uri)
        return await anyio.to_thread.run_sync(self._sync_get_metadata, digest)

    def _sync_get_metadata(self, digest: str) -> Mapping[str, str]:
        conn = self._open()
        try:
            self._ensure_schema(conn)
            row = conn.execute(f'SELECT metadata FROM {self._table} WHERE sha256 = ?', (digest,)).fetchone()
        finally:
            self._maybe_close(conn)
        if row is None:
            raise FileNotFoundError(f'media not found: {digest}')
        return _coerce_metadata_mapping(json.loads(row[0]))
