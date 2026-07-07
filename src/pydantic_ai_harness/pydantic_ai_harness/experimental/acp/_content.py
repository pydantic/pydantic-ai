"""Translation between ACP content blocks and Pydantic AI user content."""

from __future__ import annotations

import base64
from collections.abc import Sequence

from acp import schema
from pydantic_ai.messages import BinaryContent, ImageUrl, UserContent

# The content block variants ACP may send in a `session/prompt` request.
PromptContentBlock = (
    schema.TextContentBlock
    | schema.ImageContentBlock
    | schema.AudioContentBlock
    | schema.ResourceContentBlock
    | schema.EmbeddedResourceContentBlock
)

# Used when an embedded binary resource arrives without a declared media type; an empty
# media type would later raise when the content is formatted for a model request.
_DEFAULT_BINARY_MEDIA_TYPE = 'application/octet-stream'


def prompt_blocks_to_user_content(blocks: Sequence[PromptContentBlock]) -> list[UserContent]:
    """Convert ACP prompt content blocks into Pydantic AI user content.

    Text and embedded text resources become plain strings; images and audio become
    [`BinaryContent`][pydantic_ai.messages.BinaryContent] (or [`ImageUrl`][pydantic_ai.messages.ImageUrl]
    when the image is referenced by URL); a resource link contributes its URI as text, prefixed with
    its title or name when the client provided one (for example `My File (file:///a.txt)`).
    """
    content: list[UserContent] = []
    for block in blocks:
        if isinstance(block, schema.TextContentBlock):
            content.append(block.text)
        elif isinstance(block, schema.ImageContentBlock):
            # ACP requires inline `data`; `uri` is only an optional reference to the image's source.
            # Prefer the bytes the client actually sent, falling back to the URL only when no inline
            # data is present (a client sending both must not have its image silently replaced by a
            # link the model may be unable to fetch).
            if not block.data and block.uri is not None:
                content.append(ImageUrl(url=block.uri))
            else:
                content.append(BinaryContent(data=base64.b64decode(block.data), media_type=block.mime_type))
        elif isinstance(block, schema.AudioContentBlock):
            content.append(BinaryContent(data=base64.b64decode(block.data), media_type=block.mime_type))
        elif isinstance(block, schema.EmbeddedResourceContentBlock):
            resource = block.resource
            if isinstance(resource, schema.TextResourceContents):
                content.append(resource.text)
            else:
                media_type = resource.mime_type or _DEFAULT_BINARY_MEDIA_TYPE
                content.append(BinaryContent(data=base64.b64decode(resource.blob), media_type=media_type))
        else:
            # The only remaining variant is a resource link, which carries no inline content;
            # pass its URI through as text, prefixed with a human-readable label when present.
            label = block.title or block.name
            content.append(f'{label} ({block.uri})' if label else block.uri)
    return content
