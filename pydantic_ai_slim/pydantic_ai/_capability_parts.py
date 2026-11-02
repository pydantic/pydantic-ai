"""Rendering [`CapabilityPart`][pydantic_ai.messages.CapabilityPart]s into the parts every model already maps.

A capability part stays a typed part in the history and in the messages handed to
[`Model.prepare_messages`][pydantic_ai.models.Model.prepare_messages], which renders it in its lookalike pass. There
the part's own fields say who wrote its body and which text is the framework's frame, so nothing about it is written
into the rendered parts' metadata. Rendering is a pure function of the stored part: no id, timestamp or metadata of
the part reaches the model, so the same history always renders to the same bytes.
"""

from __future__ import annotations as _annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from . import _utils
from ._framing import LookalikeSource, capability_frame_tags, lookalike_tags
from .messages import (
    _FRAME_PLACEHOLDER,  # pyright: ignore[reportPrivateUsage]
    BinaryContent,
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


def has_capability_parts(messages: Sequence[ModelMessage]) -> bool:
    """Whether any request in `messages` holds a [`CapabilityPart`][pydantic_ai.messages.CapabilityPart]."""
    return any(_has_capability_parts(message) for message in messages)


@dataclass(frozen=True)
class RenderedCapabilityPart:
    """What a [`CapabilityPart`][pydantic_ai.messages.CapabilityPart] reaches the model as, within one `prepare_messages` pass.

    Never stored: what the lookalike pass needs to know about the rendered parts travels here, next to them, rather
    than in their metadata.
    """

    parts: list[SystemPromptPart | UserPromptPart]
    """The rendered parts, in order."""

    body: UserPromptPart | None
    """A data part's framed content on the user channel, which `prepare_messages` counts as a frame to announce."""

    lookalikes: list[tuple[LookalikeSource, list[str]]]
    """Who wrote text in the body that resembles a framework frame, with the system-voice tags it opens."""


def render_capability_part(part: CapabilityPart) -> RenderedCapabilityPart:
    """The parts `part` reaches the model as.

    An instruction part is a system message, in its own scope. A data part is its content in a plain tag, whatever its
    source. Its note, which vouches for the content, goes before it only when the source is trusted. The body is the
    content as written: text in it that resembles a frame gets a lookalike note from `prepare_messages`, naming the
    author the part's source gives.
    """
    rendering = part.rendering
    content = part.model_content()
    if rendering.voice == 'instruction':
        # Checked when the part was created.
        assert isinstance(content, str)
        system = SystemPromptPart(content, timestamp=part.timestamp, scope=part.scope)
        return RenderedCapabilityPart(parts=[system], body=None, lookalikes=[])
    assert rendering.tag is not None
    # TODO(#11029): deliver content from a source the operator doesn't control on the tool channel instead, as a
    # framework-inserted read and its result, rebuilding #11837.
    attrs = part._rendered_attributes()  # pyright: ignore[reportPrivateUsage]
    open_tag, close_tag = capability_frame_tags(rendering.tag, attrs=attrs)
    items: list[UserContent] = [content] if isinstance(content, str | StructuredContent) else list(content)
    # Text goes out as `TextContent`, as in any user prompt with more than one item.
    body = UserPromptPart(
        [
            TextContent(open_tag),
            *(TextContent(item) if isinstance(item, str) else item for item in items),
            TextContent(close_tag),
        ],
        timestamp=part.timestamp,
    )
    lookalikes = _body_lookalikes(part, items)
    if rendering.note is None or not part.source.trusted:
        return RenderedCapabilityPart(parts=[body], body=body, lookalikes=lookalikes)
    reference, _ = capability_frame_tags(rendering.tag, attrs={})
    note = SystemPromptPart(rendering.note.replace(_FRAME_PLACEHOLDER, reference), timestamp=part.timestamp)
    return RenderedCapabilityPart(parts=[note, body], body=body, lookalikes=lookalikes)


def render_capability_parts(messages: list[ModelMessage]) -> list[ModelMessage]:
    """`messages` with every capability part rendered in place, with no lookalike notes, or `messages` itself when there's none.

    For the paths with no lookalike pass: a second model reading a transcript, and a realtime session's seeded history.
    """
    if not has_capability_parts(messages):
        return messages
    return [
        replace(message, parts=_render_parts(message.parts))
        if isinstance(message, ModelRequest) and _has_capability_parts(message)
        else message
        for message in messages
    ]


def _has_capability_parts(message: ModelMessage) -> bool:
    return isinstance(message, ModelRequest) and any(isinstance(part, CapabilityPart) for part in message.parts)


def _render_parts(parts: Sequence[ModelRequestPart]) -> list[ModelRequestPart]:
    rendered: list[ModelRequestPart] = []
    for part in parts:
        if isinstance(part, CapabilityPart):
            rendered.extend(render_capability_part(part).parts)
        else:
            rendered.append(part)
    return rendered


def _body_lookalikes(part: CapabilityPart, items: Sequence[UserContent]) -> list[tuple[LookalikeSource, list[str]]]:
    """Who wrote text in a data part's body that resembles a framework frame, with the system-voice tags it opens.

    The body's text is the source's author's; text from an `OperatorSource` is the application's own and gets no note.
    A text file in the body is its author's, as in a user prompt.
    """
    author = part.source._author  # pyright: ignore[reportPrivateUsage]
    found: dict[LookalikeSource, list[str]] = {}

    def check(source: LookalikeSource, text: str) -> None:
        if tags := lookalike_tags(text):
            found.setdefault(source, []).extend(tag for tag in tags if tag not in found.get(source, ()))

    for item in items:
        if author != 'application' and isinstance(item, str | TextContent):
            check(author, item if isinstance(item, str) else item.content)
    for item in items:
        if isinstance(item, BinaryContent) and _utils.is_text_like_media_type(item.media_type):
            check('file', item.data.decode(errors='replace'))
    return list(found.items())
