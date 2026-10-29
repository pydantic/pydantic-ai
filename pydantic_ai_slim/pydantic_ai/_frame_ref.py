"""The conversation's frame ref: the value every frame core renders on the user channel carries as `ref="..."`.

A model can't tell a frame the application wrote from text that only looks like one, such as a `<system>` tag a
user pasted or a web page contains. Rather than rewriting that text, every frame core writes into a user turn
carries the conversation's ref, the model is told which ref that is
([`frame_ref_announcement`][pydantic_ai._framing.frame_ref_announcement]), and a lookalike gets a note saying whose
text it is. The ref is a hint to the model, not a guarantee.

The ref has to be the same on every request of a conversation, or every frame would render differently and the
provider's prompt cache would miss from the first one on. So it's taken once and stored in the reserved
`__pydantic_ai__` namespace of [`ModelRequest.metadata`][pydantic_ai.messages.ModelRequest.metadata], on every
request from the first one that needs it, which keeps it through compaction, trimming the oldest turns, forks,
dump and load, and the drop of request-scoped parts. When no request carries one yet, it's derived from the
conversation's first user prompt, so a history that lost the stored value (a UI client that kept no metadata, or
a direct `Model.request` caller who keeps none) gets the same ref back as long as that prompt is still there.
Being a pure function of the history, it also replays the same in a durable workflow.
"""

from __future__ import annotations as _annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import replace
from typing import Final

from . import _utils
from .messages import (
    _PYDANTIC_AI_METADATA_KEY,  # pyright: ignore[reportPrivateUsage]
    ModelMessage,
    ModelRequest,
    TextContent,
    UserPromptPart,
)

FRAME_REF_KEY: Final = 'frame_ref'
"""The key under the reserved `__pydantic_ai__` namespace of a request's metadata that stores the conversation's ref."""

_FRAME_REF: Final = re.compile(r'[0-9a-f]{12}')


def is_frame_ref(value: str) -> bool:
    """Whether `value` has the shape of a frame ref, which is what a stored or client-supplied one must have to be used."""
    return _FRAME_REF.fullmatch(value) is not None


def stored_frame_ref(message: ModelMessage) -> str | None:
    """The frame ref stored on `message`, if it's a request that carries one."""
    if not isinstance(message, ModelRequest):
        return None
    namespace = (message.metadata or {}).get(_PYDANTIC_AI_METADATA_KEY)
    ref = namespace.get(FRAME_REF_KEY) if _utils.is_str_dict(namespace) else None
    return ref if isinstance(ref, str) and is_frame_ref(ref) else None


def latest_stored_frame_ref(messages: Sequence[ModelMessage]) -> str | None:
    """The frame ref the latest request in `messages` that stores one stores, or `None` when none does."""
    return next((ref for message in reversed(messages) if (ref := stored_frame_ref(message)) is not None), None)


def frame_ref(messages: Sequence[ModelMessage]) -> str:
    """The conversation's frame ref: the one its latest request stores, or else the one derived from its first prompt."""
    return latest_stored_frame_ref(messages) or _derived_frame_ref(messages)


def has_stored_frame_ref(messages: Sequence[ModelMessage]) -> bool:
    """Whether any request in `messages` stores a frame ref."""
    return latest_stored_frame_ref(messages) is not None


def with_frame_ref(request: ModelRequest, ref: str) -> ModelRequest:
    """`request` storing `ref`, or `request` itself when it already does."""
    if stored_frame_ref(request) == ref:
        return request
    metadata = request.metadata or {}
    namespace = metadata.get(_PYDANTIC_AI_METADATA_KEY)
    return replace(
        request,
        metadata={
            **metadata,
            _PYDANTIC_AI_METADATA_KEY: {**(namespace if _utils.is_str_dict(namespace) else {}), FRAME_REF_KEY: ref},
        },
    )


def store_frame_ref(request: ModelRequest, ref: str) -> None:
    """Store `ref` on `request` in place, as the agent graph does on the request it's about to send."""
    request.metadata = with_frame_ref(request, ref).metadata


def _derived_frame_ref(messages: Sequence[ModelMessage]) -> str:
    """A ref derived from the text of the conversation's first user prompt, for a history that stores none.

    Only text counts: a UI round trip can give back a file with a different identifier, and a single text item as a
    plain string, so neither may change the ref.
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
    return hashlib.sha256(f'pydantic-ai frame ref\0{seed}'.encode()).hexdigest()[:12]


def _prompt_text(part: UserPromptPart) -> str:
    if isinstance(part.content, str):
        return part.content
    return '\n'.join(
        item if isinstance(item, str) else item.content for item in part.content if isinstance(item, str | TextContent)
    )
