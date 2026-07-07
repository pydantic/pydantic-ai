"""Shared assertion helpers for the realtime WebSocket cassette tests."""

from __future__ import annotations as _annotations

import json
from typing import Any

from .ws_cassettes import CassetteMessage, RealtimeCassette


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
