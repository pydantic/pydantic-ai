"""Shared assertion helpers for the realtime WebSocket tests."""

from __future__ import annotations as _annotations

import json
from typing import Any

import pytest

from pydantic_ai.realtime import RealtimeError, RealtimeEvent, RealtimeSession, RealtimeSessionErrorEvent
from pydantic_ai.realtime.codec import RealtimeCodecEvent, RealtimeConnection

from .ws_cassettes import CassetteMessage, RealtimeCassette


async def collect_codec_events(connection: RealtimeConnection, *, sideband: bool = False) -> list[RealtimeCodecEvent]:
    """Drain a connection through the end of its scripted conversation.

    Both the fakes and the recordings end with the server hanging up, which a WebSocket-backed
    connection reports as a final non-recoverable `RealtimeSessionErrorEvent` (see
    `test_clean_close_is_reported_as_a_fatal_error`). Asserting and stripping it here keeps every
    caller's expectations about the conversation rather than its ending.

    Pass `sideband=True` for a WebRTC sideband, where a clean close is the browser hanging up — the
    normal end of a call — so the stream simply ends with nothing to strip.
    """
    events = [event async for event in connection]
    if sideband:
        return events
    closed = events.pop()
    assert isinstance(closed, RealtimeSessionErrorEvent), closed
    assert not closed.recoverable and 'connection closed' in closed.message, closed
    return events


async def collect_session_events(session: RealtimeSession) -> list[RealtimeEvent]:
    """Drain a session through the end of its scripted conversation, absorbing the server's hangup."""
    events: list[RealtimeEvent] = []
    with pytest.raises(RealtimeError, match='connection closed'):
        async for event in session:
            events.append(event)
    return events


def collapse_event_types(events: list[Any]) -> list[str]:
    """Collapse consecutive runs of the same event type into a single entry.

    Audio and transcript arrive as long runs of `PartDeltaEvent`s whose exact count depends on the
    recording; collapsing keeps the asserted event *shape* stable and readable.
    """
    collapsed: list[str] = []
    for name in (type(event).__name__ for event in events):
        if not collapsed or collapsed[-1] != name:
            collapsed.append(name)
    return collapsed


def sent_frames_containing(cassette: RealtimeCassette, needle: str) -> list[dict[str, Any]]:
    """The outbound frames in `cassette` whose serialized JSON contains `needle`."""
    return [
        message.data
        for message in cassette.interactions
        if isinstance(message, CassetteMessage) and message.direction == 'sent' and needle in json.dumps(message.data)
    ]
