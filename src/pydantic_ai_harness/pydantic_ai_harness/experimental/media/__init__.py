"""Content-addressed media stores for offloading large binary parts.

Used by `pydantic_ai_harness.experimental.step_persistence` to keep snapshots small when
messages carry `BinaryContent` payloads. A forthcoming `MediaExternalizer`
capability will reuse these stores for in-flight wire-payload reduction
(rewriting `BinaryContent` to URL parts before the model sees them).
"""

from pydantic_ai_harness.experimental._warn import warn_experimental
from pydantic_ai_harness.experimental.media._s3 import S3MediaStore
from pydantic_ai_harness.experimental.media._store import (
    DiskMediaStore,
    KeyStrategy,
    MediaContext,
    MediaStore,
    PublicUrlResolver,
    SqliteMediaStore,
    default_key_strategy,
    make_static_public_url,
    media_uri_for,
    parse_media_uri,
)
from pydantic_ai_harness.experimental.media._walker import externalize_media, restore_media

warn_experimental('media')

__all__ = [
    'DiskMediaStore',
    'KeyStrategy',
    'MediaContext',
    'MediaStore',
    'PublicUrlResolver',
    'S3MediaStore',
    'SqliteMediaStore',
    'default_key_strategy',
    'externalize_media',
    'make_static_public_url',
    'media_uri_for',
    'parse_media_uri',
    'restore_media',
]
