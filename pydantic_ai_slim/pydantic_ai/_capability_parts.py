"""Rendering [`CapabilityPart`][pydantic_ai.messages.CapabilityPart]s into the parts every model already maps.

A capability part is rendered before any model sees the history: by the agent graph ahead of
[`Model.prepare_messages`][pydantic_ai.models.Model.prepare_messages], so a custom model that maps parts itself, or
overrides `prepare_messages`, never meets one, and again at the start of `prepare_messages` for a caller that drives
a model directly. Rendering is a pure function of the stored part: no id, timestamp or metadata of the part reaches
the model, so the same history always renders to the same bytes.
"""

from __future__ import annotations as _annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Final, Literal, TypeAlias

from . import _utils
from ._framework_frame import framework_frame_text
from ._framing import capability_frame_tags
from .messages import (
    _PYDANTIC_AI_METADATA_KEY,  # pyright: ignore[reportPrivateUsage]
    CapabilityPart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    StructuredContent,
    SystemPromptPart,
    TextContent,
    UserContent,
    UserPromptPart,
)

RENDERED_CAPABILITY_PART_KEY: Final = 'capability_part'
"""The key under the reserved `__pydantic_ai__` namespace of a `SystemPromptPart`'s metadata that marks it as rendered from a
[`CapabilityPart`][pydantic_ai.messages.CapabilityPart], so it's never taken for the standing system prompt."""


def has_capability_parts(messages: Sequence[ModelMessage]) -> bool:
    """Whether any request in `messages` holds a [`CapabilityPart`][pydantic_ai.messages.CapabilityPart]."""
    return any(_has_capability_parts(message) for message in messages)


def render_capability_parts(messages: list[ModelMessage]) -> list[ModelMessage]:
    """`messages` with every [`CapabilityPart`][pydantic_ai.messages.CapabilityPart] rendered in place, or `messages` itself when there's none."""
    if not has_capability_parts(messages):
        return messages
    return [
        replace(message, parts=_render_parts(message.parts))
        if isinstance(message, ModelRequest) and _has_capability_parts(message)
        else message
        for message in messages
    ]


def is_rendered_capability_part(part: SystemPromptPart) -> bool:
    """Whether `part` was rendered from a system-voice capability part, or is a capability part's note."""
    namespace = part.metadata.get(_PYDANTIC_AI_METADATA_KEY) if _utils.is_str_dict(part.metadata) else None
    return _utils.is_str_dict(namespace) and namespace.get(RENDERED_CAPABILITY_PART_KEY) is True


def _has_capability_parts(message: ModelMessage) -> bool:
    return isinstance(message, ModelRequest) and any(isinstance(part, CapabilityPart) for part in message.parts)


def _render_parts(parts: Sequence[ModelRequestPart]) -> list[ModelRequestPart]:
    rendered: list[ModelRequestPart] = []
    for part in parts:
        if isinstance(part, CapabilityPart):
            rendered.extend(_render(part))
        else:
            rendered.append(part)
    return rendered


def _render(part: CapabilityPart) -> list[SystemPromptPart | UserPromptPart]:
    """The parts `part` reaches the model as.

    An instruction part is a system message, in its own scope. A data part is its content in a plain tag, whatever its
    source. Its note, which vouches for the content, goes before it only when the source is trusted. The tags are
    marked as the framework's, so the lookalike check skips them; text in the body that resembles a frame gets the
    lookalike note from `prepare_messages`, naming who wrote it (see `content_author`).
    """
    rendering = part.rendering
    content = part.model_content()
    metadata = {_PYDANTIC_AI_METADATA_KEY: {RENDERED_CAPABILITY_PART_KEY: True}}
    if rendering.voice == 'instruction':
        # Checked when the part was created.
        assert isinstance(content, str)
        return [SystemPromptPart(content, timestamp=part.timestamp, scope=part.scope, metadata=metadata)]
    assert rendering.tag is not None
    # TODO(#11029): deliver content from a source the operator doesn't control on the tool channel instead, as a
    # framework-inserted read and its result, rebuilding #11837.
    open_tag, close_tag = capability_frame_tags(rendering.tag, attrs=part.rendered_attributes())
    # Only the tags are marked as the framework's: the body is the part's content, which `prepare_messages` checks
    # for lookalike frames like any other text, so a body that closes the frame early is noted rather than trusted.
    author = _AUTHORS[part.source.source_kind]
    items = [content] if isinstance(content, str | StructuredContent) else content
    body = [_with_author(item, author) for item in items]
    framed = UserPromptPart(
        [framework_frame_text(open_tag), *body, framework_frame_text(close_tag)], timestamp=part.timestamp
    )
    if rendering.note is None or not part.source.trusted:
        return [framed]
    reference, _ = capability_frame_tags(rendering.tag, attrs={})
    note = rendering.note.replace('{frame}', reference)
    return [SystemPromptPart(note, timestamp=part.timestamp, metadata=metadata), framed]


ContentAuthor: TypeAlias = Literal['user', 'tool', 'file', 'application']
"""Who wrote a capability part's content, for the lookalike note: the lookalike sources, or the application itself."""

_AUTHORS: Final[dict[str, ContentAuthor]] = {
    'operator': 'application',
    'workspace': 'file',
    'tool-result': 'tool',
    'user': 'user',
}
"""The author of the content from each kind of `CapabilityPartSource`, by its `source_kind`."""

CONTENT_AUTHOR_KEY: Final = 'content_author'
"""The key under the reserved `__pydantic_ai__` namespace of a rendered body's `TextContent` naming its author."""


def _with_author(item: UserContent, author: ContentAuthor) -> UserContent:
    """`item` marked with who wrote it, when it's text; other content, such as a file, as it is."""
    if isinstance(item, str):
        return TextContent(item, metadata={_PYDANTIC_AI_METADATA_KEY: {CONTENT_AUTHOR_KEY: author}})
    if not isinstance(item, TextContent):
        return item
    metadata = item.metadata if _utils.is_str_dict(item.metadata) else {}
    namespace = metadata.get(_PYDANTIC_AI_METADATA_KEY)
    namespace = namespace if _utils.is_str_dict(namespace) else {}
    return replace(item, metadata={**metadata, _PYDANTIC_AI_METADATA_KEY: {**namespace, CONTENT_AUTHOR_KEY: author}})


def content_author(item: TextContent) -> ContentAuthor:
    """Who wrote `item`: the author a rendered capability part's body was marked with, or else the user."""
    namespace = item.metadata.get(_PYDANTIC_AI_METADATA_KEY) if _utils.is_str_dict(item.metadata) else None
    author = namespace.get(CONTENT_AUTHOR_KEY) if _utils.is_str_dict(namespace) else None
    return author if author in ('tool', 'file', 'application') else 'user'
