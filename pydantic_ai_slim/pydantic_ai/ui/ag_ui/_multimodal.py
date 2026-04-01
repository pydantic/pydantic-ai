# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false
# pyright: reportUnknownArgumentType=false
"""Multimodal input content conversion for AG-UI protocol >= 0.1.15.

This module is lazy-imported only when the caller has verified that
ag-ui-protocol >= 0.1.15 is installed, so these imports will succeed.
Pyright suppressions are needed because the development dependency pins 0.1.13.
"""

from __future__ import annotations

from base64 import b64decode

from ag_ui.core import (
    AudioInputContent,
    DocumentInputContent,
    ImageInputContent,
    InputContentUrlSource,
    VideoInputContent,
)

from ...messages import AudioUrl, BinaryContent, DocumentUrl, ImageUrl, VideoUrl

_URL_TYPE_MAP: dict[type, type] = {
    ImageUrl: ImageInputContent,
    AudioUrl: AudioInputContent,
    VideoUrl: VideoInputContent,
    DocumentUrl: DocumentInputContent,
}


def media_url_to_multimodal(
    item: ImageUrl | AudioUrl | VideoUrl | DocumentUrl,
) -> ImageInputContent | AudioInputContent | VideoInputContent | DocumentInputContent:
    """Convert a media URL to typed multimodal AG-UI input content."""
    source = InputContentUrlSource(type='url', value=item.url, mime_type=item.media_type or '')
    return _URL_TYPE_MAP[type(item)](source=source)


def multimodal_input_to_content(
    part: ImageInputContent | AudioInputContent | VideoInputContent | DocumentInputContent,
) -> ImageUrl | AudioUrl | VideoUrl | DocumentUrl | BinaryContent:
    """Convert a typed multimodal AG-UI input content back to a Pydantic AI content type."""
    source = part.source
    if isinstance(source, InputContentUrlSource):
        media_type = source.mime_type or None
        if isinstance(part, ImageInputContent):
            return ImageUrl(url=source.value, media_type=media_type)
        elif isinstance(part, AudioInputContent):
            return AudioUrl(url=source.value, media_type=media_type)
        elif isinstance(part, VideoInputContent):
            return VideoUrl(url=source.value, media_type=media_type)
        else:
            return DocumentUrl(url=source.value, media_type=media_type)
    else:
        return BinaryContent(data=b64decode(source.value), media_type=source.mime_type)
