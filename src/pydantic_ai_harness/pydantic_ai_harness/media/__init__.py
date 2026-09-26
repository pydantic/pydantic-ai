"""Content-addressed media stores for offloading large binary/text parts.

Used by `pydantic_ai_harness.step_persistence` to keep snapshots small when
messages carry `BinaryContent` payloads or large text (e.g. big tool-return
strings). A forthcoming `MediaExternalizer` capability (#254) will reuse
these stores for in-flight wire-payload reduction (rewriting `BinaryContent`
to URL parts before the model sees them).

`MongoMediaStore` needs the `mongodb` extra (`pip install
pydantic-ai-harness[mongodb]`); it is imported lazily so the rest of this
module stays usable without `pymongo` installed.
"""

from typing import TYPE_CHECKING

from pydantic_ai_harness.media._s3 import S3MediaStore
from pydantic_ai_harness.media._store import (
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
from pydantic_ai_harness.media._walker import externalize_media, restore_media

if TYPE_CHECKING:
    from pydantic_ai_harness.media._mongo import MongoMediaStore

__all__ = [
    'DiskMediaStore',
    'KeyStrategy',
    'MediaContext',
    'MediaStore',
    'MongoMediaStore',
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


def __getattr__(name: str) -> object:
    if name == 'MongoMediaStore':
        from pydantic_ai_harness.media._mongo import MongoMediaStore

        return MongoMediaStore
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
