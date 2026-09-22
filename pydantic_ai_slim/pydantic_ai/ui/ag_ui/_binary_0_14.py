"""Legacy `binary` input part for peers below 0.1.15.

`BinaryInputContent` was the untyped media part before typed multimodal content. AG-UI 1.0 retired it
from the `InputContent` union and keeps the class importable for one release only, so it is gated
here; `HAS_BINARY_INPUT_PART` in `_forward_compat` says whether the installed SDK still accepts the part.
"""

from __future__ import annotations

from base64 import b64decode
from typing import TYPE_CHECKING, cast

from ag_ui.core import InputContent

from ...messages import AudioUrl, BinaryContent, DocumentUrl, ImageUrl, UserContent, VideoUrl
from ._utils import MediaPartType, media_part_type

if TYPE_CHECKING:
    from ag_ui.core import BinaryInputContent
else:
    try:
        from ag_ui.core import BinaryInputContent
    except ImportError:  # pragma: lax no cover
        # Once the SDK drops the retired class, no inbound part can match it.

        class BinaryInputContent:
            """Stub for SDKs without the retired `binary` input part."""


__all__ = ['BinaryInputContent', 'legacy_binary_input', 'legacy_binary_to_content']

_URL_CONTENT_TYPES: dict[MediaPartType, type[ImageUrl | VideoUrl | AudioUrl | DocumentUrl]] = {
    'image': ImageUrl,
    'video': VideoUrl,
    'audio': AudioUrl,
    'document': DocumentUrl,
}


def legacy_binary_to_content(part: BinaryInputContent) -> UserContent:
    """Convert a legacy `binary` input part to Pydantic AI content."""
    if part.url:
        try:
            return BinaryContent.from_data_uri(part.url)
        except ValueError:
            return _URL_CONTENT_TYPES[media_part_type(part.mime_type)](url=part.url, media_type=part.mime_type)
    elif part.data:
        return BinaryContent(data=b64decode(part.data), media_type=part.mime_type)
    else:  # pragma: no cover
        raise ValueError('BinaryInputContent must have either a `url` or `data` field.')


def legacy_binary_input(*, mime_type: str, url: str | None = None, data: str | None = None) -> InputContent:
    """Build the retired `binary` part for peers before typed multimodal content.

    Only called when `HAS_BINARY_INPUT_PART`; the cast is for 1.0, where `BinaryInputContent` is a
    deprecated class outside the `InputContent` union.
    """
    return cast(InputContent, BinaryInputContent(type='binary', mime_type=mime_type, url=url, data=data))
