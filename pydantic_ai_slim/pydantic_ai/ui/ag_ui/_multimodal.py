"""Multimodal input content conversion for AG-UI protocol >= 0.1.15.

This module is lazy-imported only when the caller has verified that
ag-ui-protocol >= 0.1.15 is installed, so these imports will succeed.
"""

from __future__ import annotations

from base64 import b64decode
from typing import Any

from ag_ui.core import (
    AudioInputContent,
    DocumentInputContent,
    ImageInputContent,
    InputContentDataSource,
    InputContentUrlSource,
    VideoInputContent,
)
from pydantic import TypeAdapter

from ...messages import AudioUrl, BinaryContent, DocumentUrl, ImageUrl, VideoUrl

_URL_TYPE_MAP: dict[type, type] = {
    ImageUrl: ImageInputContent,
    AudioUrl: AudioInputContent,
    VideoUrl: VideoInputContent,
    DocumentUrl: DocumentInputContent,
}

# `vendor_metadata` is carried under a dedicated key inside the input content's generic
# `metadata` field, mirroring the `vendor_metadata` key the `UploadedFile` round-trip already
# uses, so we only ever read back our own value and ignore unrelated client metadata.
_VENDOR_METADATA_KEY = 'vendor_metadata'

# The `metadata` field is typed as `Any`; coerce a client-supplied value to a known `dict` shape
# (mirroring how the Vercel adapter reads from the typed `load_provider_metadata` result) so the
# extracted `vendor_metadata` is statically typed rather than `Unknown`.
_METADATA_ADAPTER: TypeAdapter[dict[str, Any]] = TypeAdapter(dict[str, Any])


def _dump_vendor_metadata(
    item: ImageUrl | AudioUrl | VideoUrl | DocumentUrl | BinaryContent,
) -> dict[str, object] | None:
    return {_VENDOR_METADATA_KEY: item.vendor_metadata} if item.vendor_metadata is not None else None


def media_url_to_multimodal(
    item: ImageUrl | AudioUrl | VideoUrl | DocumentUrl,
) -> ImageInputContent | AudioInputContent | VideoInputContent | DocumentInputContent:
    """Convert a media URL to typed multimodal AG-UI input content."""
    source = InputContentUrlSource(type='url', value=item.url, mime_type=item.media_type or '')
    return _URL_TYPE_MAP[type(item)](source=source, metadata=_dump_vendor_metadata(item))


_MEDIA_PREFIX_TO_CONTENT: dict[str, type] = {
    'image': ImageInputContent,
    'audio': AudioInputContent,
    'video': VideoInputContent,
}


def binary_to_multimodal(
    item: BinaryContent,
) -> ImageInputContent | AudioInputContent | VideoInputContent | DocumentInputContent:
    """Convert BinaryContent to typed multimodal AG-UI input content based on media type prefix."""
    source = InputContentDataSource(type='data', value=item.base64, mime_type=item.media_type)
    content_cls = _MEDIA_PREFIX_TO_CONTENT.get(item.media_type.split('/', 1)[0], DocumentInputContent)
    return content_cls(source=source, metadata=_dump_vendor_metadata(item))


def multimodal_input_to_content(
    part: ImageInputContent | AudioInputContent | VideoInputContent | DocumentInputContent,
) -> ImageUrl | AudioUrl | VideoUrl | DocumentUrl | BinaryContent:
    """Convert a typed multimodal AG-UI input content back to a Pydantic AI content type."""
    source = part.source
    # `metadata` is client-controlled and typed as `Any`; the value is passed to the validating
    # constructors below, so a malformed (non-`dict`) `vendor_metadata` is rejected there, matching
    # the Vercel adapter. The adapter coerces the `Any` value to a statically-typed `dict` rather
    # than relying on `isinstance` narrowing.
    metadata = part.metadata
    vendor_metadata: dict[str, Any] | None = None
    if isinstance(metadata, dict):
        vendor_metadata = _METADATA_ADAPTER.validate_python(metadata).get(_VENDOR_METADATA_KEY)
    if isinstance(source, InputContentUrlSource):
        media_type = source.mime_type or None
        if isinstance(part, ImageInputContent):
            return ImageUrl(url=source.value, media_type=media_type, vendor_metadata=vendor_metadata)
        elif isinstance(part, AudioInputContent):
            return AudioUrl(url=source.value, media_type=media_type, vendor_metadata=vendor_metadata)
        elif isinstance(part, VideoInputContent):
            return VideoUrl(url=source.value, media_type=media_type, vendor_metadata=vendor_metadata)
        else:
            return DocumentUrl(url=source.value, media_type=media_type, vendor_metadata=vendor_metadata)
    else:
        return BinaryContent(data=b64decode(source.value), media_type=source.mime_type, vendor_metadata=vendor_metadata)
