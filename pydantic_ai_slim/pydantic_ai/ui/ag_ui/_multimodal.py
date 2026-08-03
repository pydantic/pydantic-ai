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

from ..._utils import is_str_dict
from ...messages import AudioUrl, BinaryContent, DocumentUrl, ForceDownloadMode, ImageUrl, VideoUrl

AGUIContentTypes = ImageInputContent | AudioInputContent | VideoInputContent | DocumentInputContent
PydanticAIUrlType = ImageUrl | AudioUrl | VideoUrl | DocumentUrl

_URL_TYPE_MAP: dict[type[PydanticAIUrlType], type[AGUIContentTypes]] = {
    ImageUrl: ImageInputContent,
    AudioUrl: AudioInputContent,
    VideoUrl: VideoInputContent,
    DocumentUrl: DocumentInputContent,
}

_CONTENT_TYPE_MAP: dict[type[AGUIContentTypes], type[PydanticAIUrlType]] = {
    ImageInputContent: ImageUrl,
    AudioInputContent: AudioUrl,
    VideoInputContent: VideoUrl,
    DocumentInputContent: DocumentUrl,
}

# `vendor_metadata` and (for URL types) `force_download` are carried under dedicated keys inside
# the input content's generic `metadata` field, mirroring the `vendor_metadata` key the
# `UploadedFile` round-trip already uses, so we only ever read back our own values and ignore
# unrelated client metadata.
_VENDOR_METADATA_KEY = 'vendor_metadata'
_FORCE_DOWNLOAD_KEY = 'force_download'


def dump_metadata(
    item: PydanticAIUrlType | BinaryContent,
) -> dict[str, object] | None:
    metadata: dict[str, object] = {}
    if item.vendor_metadata is not None:
        metadata[_VENDOR_METADATA_KEY] = item.vendor_metadata
    if isinstance(item, PydanticAIUrlType) and item.force_download is not False:
        metadata[_FORCE_DOWNLOAD_KEY] = item.force_download
    return metadata or None


def media_url_to_multimodal(
    item: PydanticAIUrlType,
) -> AGUIContentTypes:
    """Convert a media URL to typed multimodal AG-UI input content."""
    source = InputContentUrlSource(type='url', value=item.url, mime_type=item.media_type or '')
    return _URL_TYPE_MAP[type(item)](source=source, metadata=dump_metadata(item))


_MEDIA_PREFIX_TO_CONTENT: dict[str, type] = {
    'image': ImageInputContent,
    'audio': AudioInputContent,
    'video': VideoInputContent,
}


def binary_to_multimodal(
    item: BinaryContent,
) -> AGUIContentTypes:
    """Convert BinaryContent to typed multimodal AG-UI input content based on media type prefix."""
    source = InputContentDataSource(type='data', value=item.base64, mime_type=item.media_type)
    content_cls = _MEDIA_PREFIX_TO_CONTENT.get(item.media_type.split('/', 1)[0], DocumentInputContent)
    return content_cls(source=source, metadata=dump_metadata(item))


def multimodal_input_to_content(
    part: AGUIContentTypes,
) -> PydanticAIUrlType | BinaryContent:
    """Convert a typed multimodal AG-UI input content back to a Pydantic AI content type."""
    source = part.source
    # `metadata` is client-controlled and typed as `Any`; a non-`dict` value is ignored, and a
    # malformed (non-`dict`) `vendor_metadata` inside it is rejected by the validating constructors
    # below, matching the Vercel adapter.
    metadata = part.metadata
    vendor_metadata: dict[str, Any] | None = None
    force_download: ForceDownloadMode = False
    if is_str_dict(metadata):
        vendor_metadata = metadata.get(_VENDOR_METADATA_KEY)
        force_download = metadata.get(_FORCE_DOWNLOAD_KEY, False)
    if isinstance(source, InputContentUrlSource):
        return _CONTENT_TYPE_MAP[type(part)](
            url=source.value,
            media_type=source.mime_type or None,
            force_download=force_download,
            vendor_metadata=vendor_metadata,
        )
    else:
        return BinaryContent(data=b64decode(source.value), media_type=source.mime_type, vendor_metadata=vendor_metadata)
