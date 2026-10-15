"""Deprecated import location for `pydantic_ai_harness.media`.

This capability graduated out of `experimental`; importing from here still works but
emits a `DeprecationWarning`. Import from `pydantic_ai_harness.media` instead.
"""

from pydantic_ai_harness.experimental._warn import warn_moved
from pydantic_ai_harness.media import (
    DiskMediaStore,
    KeyStrategy,
    MediaContext,
    MediaStore,
    PublicUrlResolver,
    S3MediaStore,
    SqliteMediaStore,
    default_key_strategy,
    externalize_media,
    make_static_public_url,
    media_uri_for,
    parse_media_uri,
    restore_media,
)

warn_moved('media', 'media')

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
