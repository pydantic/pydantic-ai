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
from typing import Final

from . import _utils
from ._frame_seal import framework_frame_text
from ._framing import capability_frame_tags
from .messages import (
    _PYDANTIC_AI_METADATA_KEY,  # pyright: ignore[reportPrivateUsage]
    CapabilityPart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
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
        replace(message, parts=_render_parts(message.parts)) if _has_capability_parts(message) else message
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

    A `'system'` part is a system message, in its own scope. A `'framed'` part from a trusted source is a frame
    carrying the frame seal, preceded by its note when the kind has one; from any other source, the frame carries
    no seal and the note is left out, so the content is data like any other text from a user or a tool, and gets
    the same lookalike note from `prepare_messages` when it contains something that resembles a frame.
    """
    render = part.render
    metadata = {_PYDANTIC_AI_METADATA_KEY: {RENDERED_CAPABILITY_PART_KEY: True}}
    if render.voice == 'system':
        # Checked when the part was created.
        assert isinstance(part.content, str)
        return [SystemPromptPart(part.content, timestamp=part.timestamp, scope=part.scope, metadata=metadata)]
    assert render.tag is not None
    # TODO(#11029): deliver content from a source the operator doesn't control on the tool channel instead, as a
    # framework-inserted read and its result, rebuilding #11837.
    sealed = part.source.trusted
    open_tag, close_tag = capability_frame_tags(render.tag, sealed=sealed, attrs=part.rendered_attributes())
    content: str | list[UserContent]
    if isinstance(part.content, str):
        text = f'{open_tag}{part.content}{close_tag}'
        content = [framework_frame_text(text)] if sealed else text
    else:
        content = [_frame_text(open_tag, sealed=sealed), *part.content, _frame_text(close_tag, sealed=sealed)]
    framed = UserPromptPart(content, timestamp=part.timestamp)
    if render.note is None or not sealed:
        return [framed]
    reference, _ = capability_frame_tags(render.tag, sealed=False, attrs={})
    note = render.note.replace('{frame}', reference)
    return [SystemPromptPart(note, timestamp=part.timestamp, metadata=metadata), framed]


def _frame_text(text: str, *, sealed: bool) -> str | TextContent:
    return framework_frame_text(text) if sealed else text
