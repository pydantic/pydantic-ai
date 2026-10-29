"""The conversation's frame seal: the value every frame core renders on the user channel carries as `seal="..."`.

A model can't tell a frame the application wrote from text that only looks like one, such as a `<system>` tag a
user pasted or a web page contains. Rather than rewriting that text, every frame core writes into a user turn
carries the conversation's seal, the model is told which seal that is
([`frame_seal_announcement`][pydantic_ai._framing.frame_seal_announcement]), and a lookalike gets a note saying whose
text it is. The seal is a hint to the model, not a guarantee.

The seal has to be the same on every request of a conversation, or every frame would render differently and the
provider's prompt cache would miss from the first one on. So it's taken once and stored in the reserved
`__pydantic_ai__` namespace of [`ModelRequest.metadata`][pydantic_ai.messages.ModelRequest.metadata], on every
request from the first one that needs it, which keeps it through compaction, trimming the oldest turns, forks,
dump and load, and the drop of request-scoped parts. When no request carries one yet, it's derived from the
conversation's first user prompt, so a history that lost the stored value (a UI client that kept no metadata, or
a direct `Model.request` caller who keeps none) gets the same seal back as long as that prompt is still there.
Being a pure function of the history, it also replays the same in a durable workflow.
"""

from __future__ import annotations as _annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Final, TypeGuard

from . import _utils
from .messages import (
    _PYDANTIC_AI_METADATA_KEY,  # pyright: ignore[reportPrivateUsage]
    ModelMessage,
    ModelRequest,
    TextContent,
    UserPromptPart,
)

FRAME_SEAL_KEY: Final = 'frame_seal'
"""The key under the reserved `__pydantic_ai__` namespace of a request's metadata that stores the conversation's seal."""

FRAMEWORK_FRAME_KEY: Final = 'framework_frame'
"""The key under the reserved `__pydantic_ai__` namespace of a `TextContent`'s metadata that marks it as a frame the
framework or a capability wrote, such as a background tool's `<tool_result>`. See `framework_frame_text`."""

_OPENING_TAG: Final = re.compile(r'<([a-z]+(?:_[a-z]+)*)(?=[\s>])')
_SEAL_ON_FRAME: Final = re.compile(r'<[a-z_]+\s(?:[^<>]*\s)?seal="')
_FRAME_SEAL: Final = re.compile(r'[0-9a-f]{12}')


def framework_frame_text(text: str, *, metadata: dict[str, Any] | None = None) -> TextContent:
    """`text`, a frame the framework or a capability wrote for the user channel, as `TextContent` marked as such.

    `prepare_messages` puts the conversation's frame seal on the frame's opening tag when it has none, on every
    request, so the stored text stays as written and always goes out with the current seal. It also doesn't take the
    frame for text that resembles one. The mark lives in `TextContent.metadata`, which the UI adapters drop from
    client-submitted history, so a client can't claim it.
    """
    namespace = (metadata or {}).get(_PYDANTIC_AI_METADATA_KEY)
    return TextContent(
        text,
        metadata={
            **(metadata or {}),
            _PYDANTIC_AI_METADATA_KEY: {
                **(namespace if _utils.is_str_dict(namespace) else {}),
                FRAMEWORK_FRAME_KEY: True,
            },
        },
    )


def is_framework_frame(item: object) -> TypeGuard[TextContent]:
    """Whether `item` is a `TextContent` marked by `framework_frame_text`."""
    if not isinstance(item, TextContent):
        return False
    # `TextContent.metadata` is whatever the application put there, not necessarily a dict.
    namespace = item.metadata.get(_PYDANTIC_AI_METADATA_KEY) if _utils.is_str_dict(item.metadata) else None
    return _utils.is_str_dict(namespace) and namespace.get(FRAMEWORK_FRAME_KEY) is True


def with_frame_seal_on_frame(item: TextContent, seal: str) -> TextContent:
    """`item` with `seal` as the first attribute of its first opening tag, or `item` itself when that tag has a `seal`."""
    if _SEAL_ON_FRAME.match(item.content.lstrip()) or not (match := _OPENING_TAG.search(item.content)):
        return item
    end = match.end()
    return replace(item, content=f'{item.content[:end]} seal="{seal}"{item.content[end:]}')


def is_frame_seal(value: str) -> bool:
    """Whether `value` has the shape of a frame seal, which is what a stored or client-supplied one must have to be used."""
    return _FRAME_SEAL.fullmatch(value) is not None


def stored_frame_seal(message: ModelMessage) -> str | None:
    """The frame seal stored on `message`, if it's a request that carries one."""
    if not isinstance(message, ModelRequest):
        return None
    namespace = (message.metadata or {}).get(_PYDANTIC_AI_METADATA_KEY)
    seal = namespace.get(FRAME_SEAL_KEY) if _utils.is_str_dict(namespace) else None
    return seal if isinstance(seal, str) and is_frame_seal(seal) else None


def latest_stored_frame_seal(messages: Sequence[ModelMessage]) -> str | None:
    """The frame seal the latest request in `messages` that stores one stores, or `None` when none does."""
    return next((seal for message in reversed(messages) if (seal := stored_frame_seal(message)) is not None), None)


def frame_seal(messages: Sequence[ModelMessage]) -> str:
    """The conversation's frame seal: the one its latest request stores, or else the one derived from its first prompt."""
    return latest_stored_frame_seal(messages) or _derived_frame_seal(messages)


def has_stored_frame_seal(messages: Sequence[ModelMessage]) -> bool:
    """Whether any request in `messages` stores a frame seal."""
    return latest_stored_frame_seal(messages) is not None


def with_frame_seal(request: ModelRequest, seal: str) -> ModelRequest:
    """`request` storing `seal`, or `request` itself when it already does."""
    if stored_frame_seal(request) == seal:
        return request
    metadata = request.metadata or {}
    namespace = metadata.get(_PYDANTIC_AI_METADATA_KEY)
    return replace(
        request,
        metadata={
            **metadata,
            _PYDANTIC_AI_METADATA_KEY: {**(namespace if _utils.is_str_dict(namespace) else {}), FRAME_SEAL_KEY: seal},
        },
    )


def store_frame_seal(request: ModelRequest, seal: str) -> None:
    """Store `seal` on `request` in place, as the agent graph does on the request it's about to send."""
    request.metadata = with_frame_seal(request, seal).metadata


def _derived_frame_seal(messages: Sequence[ModelMessage]) -> str:
    """A seal derived from the text of the conversation's first user prompt, for a history that stores none.

    Only text counts: a UI round trip can give back a file with a different identifier, and a single text item as a
    plain string, so neither may change the seal.
    """
    seed = next(
        (
            text
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, UserPromptPart) and (text := _prompt_text(part))
        ),
        '',
    )
    return hashlib.sha256(f'pydantic-ai frame seal\0{seed}'.encode()).hexdigest()[:12]


def _prompt_text(part: UserPromptPart) -> str:
    if isinstance(part.content, str):
        return part.content
    return '\n'.join(
        item if isinstance(item, str) else item.content for item in part.content if isinstance(item, str | TextContent)
    )
