# pyright: reportPrivateUsage=false
"""Reasoning event handlers for AG-UI protocol >= 0.1.13 (REASONING_* events).

These are extracted class methods of `AGUIEventStream` — the `self` parameter
is the event stream instance, and access to its private fields is intentional.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from ag_ui.core import (
    BaseEvent,
    ReasoningEncryptedValueEvent,
    ReasoningEndEvent,
    ReasoningMessageContentEvent,
    ReasoningMessageEndEvent,
    ReasoningMessageStartEvent,
    ReasoningStartEvent,
)

from ...messages import ThinkingPart, ThinkingPartDelta
from ._utils import REASONING_MESSAGE_ROLE, thinking_encrypted_metadata

if TYPE_CHECKING:
    from ...output import OutputDataT
    from ...tools import AgentDepsT
    from ._event_stream import AGUIEventStream


async def handle_thinking_start(
    self: AGUIEventStream[AgentDepsT, OutputDataT], part: ThinkingPart
) -> AsyncIterator[BaseEvent]:
    assert self._reasoning_message_id is not None
    if part.content:
        yield ReasoningStartEvent(message_id=self._reasoning_message_id)
        self._reasoning_started = True
        yield ReasoningMessageStartEvent(message_id=self._reasoning_message_id, role=REASONING_MESSAGE_ROLE)  # pyright: ignore[reportArgumentType]
        yield ReasoningMessageContentEvent(message_id=self._reasoning_message_id, delta=part.content)
        self._reasoning_text = True


async def handle_thinking_delta(
    self: AGUIEventStream[AgentDepsT, OutputDataT], delta: ThinkingPartDelta
) -> AsyncIterator[BaseEvent]:
    assert self._reasoning_message_id is not None
    assert delta.content_delta is not None
    message_id = self._reasoning_message_id

    if not self._reasoning_started:
        yield ReasoningStartEvent(message_id=message_id)
        self._reasoning_started = True

    if not self._reasoning_text:
        yield ReasoningMessageStartEvent(message_id=message_id, role=REASONING_MESSAGE_ROLE)  # pyright: ignore[reportArgumentType]
        self._reasoning_text = True

    yield ReasoningMessageContentEvent(message_id=message_id, delta=delta.content_delta)


async def handle_thinking_end(
    self: AGUIEventStream[AgentDepsT, OutputDataT], part: ThinkingPart
) -> AsyncIterator[BaseEvent]:
    assert self._reasoning_message_id is not None
    message_id = self._reasoning_message_id
    encrypted = thinking_encrypted_metadata(part)

    if not self._reasoning_started and not encrypted:
        self._reasoning_message_id = None
        return

    if not self._reasoning_started:
        yield ReasoningStartEvent(message_id=message_id)

    if self._reasoning_text:
        yield ReasoningMessageEndEvent(message_id=message_id)
        self._reasoning_text = False

    if encrypted:
        yield ReasoningEncryptedValueEvent(
            subtype='message',
            entity_id=message_id,
            encrypted_value=json.dumps(encrypted),
        )

    yield ReasoningEndEvent(message_id=message_id)
    self._reasoning_message_id = None
