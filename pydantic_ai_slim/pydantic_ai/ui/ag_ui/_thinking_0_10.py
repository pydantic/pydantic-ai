# pyright: reportPrivateUsage=false
"""Legacy `THINKING_*` event handlers for peers below 0.1.11.

These are extracted class methods of `AGUIEventStream` — the `self` parameter is the event stream
instance, and access to its private fields is intentional.

The `THINKING_*` event models are defined here because `ag-ui-protocol>=1.0` no longer ships them.
Below 1.0 the SDK's own classes are used instead, so the native event stream keeps yielding the
same types it did before; the fields match, so the wire is the same either way.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Literal

from ag_ui.core import BaseEvent

from ...messages import ThinkingPart, ThinkingPartDelta

if TYPE_CHECKING:
    from ...output import OutputDataT
    from ...tools import AgentDepsT
    from ._event_stream import AGUIEventStream


class ThinkingStartEvent(BaseEvent):
    """Legacy `THINKING_START` event."""

    type: Literal['THINKING_START'] = 'THINKING_START'  # pyright: ignore[reportIncompatibleVariableOverride]
    title: str | None = None


class ThinkingEndEvent(BaseEvent):
    """Legacy `THINKING_END` event."""

    type: Literal['THINKING_END'] = 'THINKING_END'  # pyright: ignore[reportIncompatibleVariableOverride]


class ThinkingTextMessageStartEvent(BaseEvent):
    """Legacy thinking message start event."""

    type: Literal['THINKING_TEXT_MESSAGE_START'] = 'THINKING_TEXT_MESSAGE_START'  # pyright: ignore[reportIncompatibleVariableOverride]


class ThinkingTextMessageContentEvent(BaseEvent):
    """Legacy thinking message content event."""

    type: Literal['THINKING_TEXT_MESSAGE_CONTENT'] = 'THINKING_TEXT_MESSAGE_CONTENT'  # pyright: ignore[reportIncompatibleVariableOverride]
    delta: str


class ThinkingTextMessageEndEvent(BaseEvent):
    """Legacy thinking message end event."""

    type: Literal['THINKING_TEXT_MESSAGE_END'] = 'THINKING_TEXT_MESSAGE_END'  # pyright: ignore[reportIncompatibleVariableOverride]


if not TYPE_CHECKING:  # pragma: lax no cover
    try:
        from ag_ui.core import (
            ThinkingEndEvent,
            ThinkingStartEvent,
            ThinkingTextMessageContentEvent,
            ThinkingTextMessageEndEvent,
            ThinkingTextMessageStartEvent,
        )
    except ImportError:
        pass


async def handle_thinking_start(
    self: AGUIEventStream[AgentDepsT, OutputDataT], part: ThinkingPart
) -> AsyncIterator[BaseEvent]:
    if part.content:
        yield ThinkingStartEvent()
        self._reasoning_started = True
        yield ThinkingTextMessageStartEvent()
        yield ThinkingTextMessageContentEvent(delta=part.content)
        self._reasoning_text = True


async def handle_thinking_delta(
    self: AGUIEventStream[AgentDepsT, OutputDataT], delta: ThinkingPartDelta
) -> AsyncIterator[BaseEvent]:
    assert delta.content_delta is not None

    if not self._reasoning_started:
        yield ThinkingStartEvent()
        self._reasoning_started = True

    if not self._reasoning_text:
        yield ThinkingTextMessageStartEvent()
        self._reasoning_text = True

    yield ThinkingTextMessageContentEvent(delta=delta.content_delta)


async def handle_thinking_end(
    self: AGUIEventStream[AgentDepsT, OutputDataT], part: ThinkingPart
) -> AsyncIterator[BaseEvent]:
    if not self._reasoning_started and not part.content:
        self._reasoning_message_id = None
        return

    if not self._reasoning_started:
        yield ThinkingStartEvent()

    if self._reasoning_text:
        yield ThinkingTextMessageEndEvent()
        self._reasoning_text = False

    yield ThinkingEndEvent()
    self._reasoning_message_id = None
