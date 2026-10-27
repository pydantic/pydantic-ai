"""Legacy `binary` input part for peers below 0.1.15.

`BinaryInputContent` was the untyped media part before typed multimodal content. AG-UI 1.0 retired it
from the `InputContent` union and keeps the class importable for one release only, so it is gated
here, with `HAS_BINARY_INPUT_PART` saying whether the installed SDK still accepts the part.
"""

from __future__ import annotations

from base64 import b64decode
from typing import TYPE_CHECKING

from ag_ui.core import InputContent

from ..._utils import get_union_args
from ...messages import AudioUrl, BinaryContent, DocumentUrl, ImageUrl, UserContent, VideoUrl

if TYPE_CHECKING:
    from ag_ui.core import BinaryInputContent
else:
    try:
        from ag_ui.core import BinaryInputContent
    except ImportError:  # pragma: lax no cover
        # Once the SDK drops the retired class, no inbound part can match it.

        class BinaryInputContent:
            """Stub for SDKs without the retired `binary` input part."""


__all__ = ['HAS_BINARY_INPUT_PART', 'BinaryInputContent', 'legacy_binary_to_content']

HAS_BINARY_INPUT_PART = BinaryInputContent in get_union_args(InputContent)
"""Whether the installed SDK still accepts the retired `binary` input part."""


def legacy_binary_to_content(part: BinaryInputContent) -> UserContent:
    """Convert a legacy `binary` input part to Pydantic AI content."""
    if part.url:
        try:
            return BinaryContent.from_data_uri(part.url)
        except ValueError:
            media_type_constructors = {'image': ImageUrl, 'video': VideoUrl, 'audio': AudioUrl}
            constructor = media_type_constructors.get(part.mime_type.split('/', 1)[0], DocumentUrl)
            return constructor(url=part.url, media_type=part.mime_type)
    if part.data:
        return BinaryContent(data=b64decode(part.data), media_type=part.mime_type)
    raise ValueError('BinaryInputContent must have either a `url` or `data` field.')
