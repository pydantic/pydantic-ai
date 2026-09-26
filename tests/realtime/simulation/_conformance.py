"""The lifecycle contract a connection's codec event stream must obey, checked event by event.

Used two ways: on the live stream of every simulated session (wrapped around the session's pump), and
over every recorded WebSocket cassette in `tests/realtime/cassettes/` (see `test_conformance.py`), so
the adapters are held to the same rules against real provider traces.

The rules are the ones the current (v1) codec vocabulary can express; each names what the session
would get wrong if an adapter broke it:

- `codec.duplicate_tool_call`: a tool call id is reported twice (the tool would run twice);
- `codec.unknown_cancellation`: a `ToolCallCancelled` names a call never reported;
- `codec.content_after_terminal`: audio, transcript, a tool call, or usage for a response arrives after
  that response's `ResponseDone` (it would land on whatever response the session is assembling next);
- `codec.duplicate_terminal`: a response's `ResponseDone` arrives twice (the second would close
  whatever response the session is assembling next);
- `codec.event_after_fatal`: anything follows a non-recoverable `RealtimeSessionErrorEvent`.
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass, field

from pydantic_ai.messages import RealtimeSessionErrorEvent, RealtimeSessionReconnectEvent
from pydantic_ai.realtime.codec import (
    AudioDelta,
    OutputTranscript,
    RealtimeCodecEvent,
    ResponseDone,
    SessionUsage,
    ToolCall,
    ToolCallCancelled,
)


@dataclass
class ConformanceIssue:
    code: str
    detail: str
    position: int


@dataclass
class LifecycleChecker:
    """Feed it a connection's codec events in order; it collects every contract violation."""

    issues: list[ConformanceIssue] = field(default_factory=list[ConformanceIssue])
    events: int = 0
    _tool_calls: set[str] = field(default_factory=set[str])
    _ended: set[str] = field(default_factory=set[str])
    _fatal: bool = False

    def feed(self, event: RealtimeCodecEvent) -> list[ConformanceIssue]:
        position = self.events
        self.events += 1
        found: list[ConformanceIssue] = []

        def issue(code: str, detail: str) -> None:
            found.append(ConformanceIssue(code, detail, position))

        if self._fatal:
            issue('codec.event_after_fatal', f'{type(event).__name__} after a non-recoverable error')
        if isinstance(event, RealtimeSessionReconnectEvent):
            # A new connection: ids are only unique per server session.
            self._ended.clear()
        if isinstance(event, ToolCall):
            if event.tool_call_id in self._tool_calls:
                issue('codec.duplicate_tool_call', f'tool call {event.tool_call_id!r} reported twice')
            self._tool_calls.add(event.tool_call_id)
        if isinstance(event, ToolCallCancelled):
            unknown = [call_id for call_id in event.tool_call_ids if call_id not in self._tool_calls]
            if unknown:
                issue('codec.unknown_cancellation', f'cancellation of calls never reported: {unknown}')
        response_id = _content_response_id(event)
        if response_id is not None and response_id in self._ended:
            issue('codec.content_after_terminal', f'{type(event).__name__} for {response_id!r} after its ResponseDone')
        if isinstance(event, ResponseDone) and event.provider_response_id is not None:
            if event.provider_response_id in self._ended:
                issue('codec.duplicate_terminal', f'second ResponseDone for {event.provider_response_id!r}')
            self._ended.add(event.provider_response_id)
        if isinstance(event, RealtimeSessionErrorEvent) and not event.recoverable:
            self._fatal = True
        self.issues.extend(found)
        return found


def _content_response_id(event: RealtimeCodecEvent) -> str | None:
    if isinstance(event, (AudioDelta, OutputTranscript, ToolCall)):
        return event.response_id
    if isinstance(event, SessionUsage) and event.response_scoped:
        return event.provider_response_id
    return None
