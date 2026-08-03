# pyright: reportPrivateUsage=false
"""Thinking event handlers for AG-UI protocol < 0.1.13 (THINKING_* events).

These are extracted class methods of `AGUIEventStream` — the `self` parameter
is the event stream instance, and access to its private fields is intentional.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from ag_ui.core import (
    BaseEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ThinkingTextMessageContentEvent,
    ThinkingTextMessageEndEvent,
    ThinkingTextMessageStartEvent,
)

from ...messages import ThinkingPart, ThinkingPartDelta

if TYPE_CHECKING:
    from ...output import OutputDataT
    from ...tools import AgentDepsT
    from ._event_stream import AGUIEventStream


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
