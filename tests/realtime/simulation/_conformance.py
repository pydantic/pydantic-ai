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
- `codec.overlapping_responses`: content for one response arrives while another is still open (neither
  its `ResponseDone` nor its usage seen yet): a provider runs one response at a time;
- `codec.event_after_fatal`: anything follows a non-recoverable `RealtimeSessionErrorEvent`.

The three rules keyed on response ids (`content_after_terminal`, `duplicate_terminal`,
`overlapping_responses`) only apply to the OpenAI-protocol adapters, which report the provider's response
ids. Gemini Live and GPT-Live have none, so for them those rules never fire; their turn boundaries are
checked by the simulator's invariants against the fake server's ground truth instead.

A connection on the second version of the lifecycle contract (`_lifecycle.py`) is also held to the rules
its lifecycle events promise, checked on its lifecycle stream (`LifecycleChecker(lifecycle=True)`):

- `lifecycle.duplicate_start` / `lifecycle.end_without_start` / `lifecycle.duplicate_end`: a response
  isn't bracketed by exactly one `ResponseStarted` and one `ResponseEnded`;
- `lifecycle.content_outside_response`: content, usage, or a terminal for a response that isn't between
  its start and its end;
- `lifecycle.unknown_answer`: a response answers an input the connection was never sent;
- `lifecycle.input_settled_twice`: an input is answered, refused, or lost more than once;
- `lifecycle.input_added_twice`: an input joins the conversation twice;
- `lifecycle.turn_started_twice` / `lifecycle.turn_end_without_start`: a spoken turn isn't bracketed by
  one start and one end (or discard);
- `lifecycle.unended_at_close` / `lifecycle.turn_unended_at_close`: the stream ended with a response or
  a spoken turn still open.
"""

from __future__ import annotations as _annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic_ai.messages import RealtimeSessionErrorEvent, RealtimeSessionReconnectEvent
from pydantic_ai.realtime._lifecycle import (
    LIFECYCLE_EVENT_TYPES,
    InputAdded,
    LifecycleEvent,
    ResponseEnded,
    ResponseStarted,
    UserTurnDiscarded,
    UserTurnEnded,
    UserTurnStarted,
)
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
    """Feed it a connection's events in order; it collects every contract violation.

    With `lifecycle`, it is fed a connection's lifecycle stream, and holds it to the lifecycle rules too.
    """

    lifecycle: bool = False
    inputs_sent: Callable[[], int] | None = None
    """How many inputs the connection has been sent so far, for `lifecycle.unknown_answer`."""
    issues: list[ConformanceIssue] = field(default_factory=list[ConformanceIssue])
    events: int = 0
    _tool_calls: set[str] = field(default_factory=set[str])
    _ended: set[str] = field(default_factory=set[str])
    _open: str | None = None
    _fatal: bool = False
    _started: set[str] = field(default_factory=set[str])
    _open_responses: set[str] = field(default_factory=set[str])
    _responses_ended: set[str] = field(default_factory=set[str])
    _settled_inputs: set[int] = field(default_factory=set[int])
    _added_inputs: set[int] = field(default_factory=set[int])
    _turns: set[str] = field(default_factory=set[str])
    _open_turns: set[str] = field(default_factory=set[str])
    _joined_turns: set[str] = field(default_factory=set[str])

    def feed(self, event: RealtimeCodecEvent | LifecycleEvent) -> list[ConformanceIssue]:
        position = self.events
        self.events += 1
        found: list[ConformanceIssue] = []

        def issue(code: str, detail: str) -> None:
            found.append(ConformanceIssue(code, detail, position))

        if isinstance(event, LIFECYCLE_EVENT_TYPES):
            self._feed_lifecycle(event, issue)
        elif self.lifecycle:
            # The codec rules are checked on the codec stream; this one carries the same events, minus stale ones.
            self._feed_lifecycle_content(event, issue)
        else:
            self._feed_codec(event, issue)
        self.issues.extend(found)
        return found

    def _feed_lifecycle_content(self, event: RealtimeCodecEvent, issue: Callable[[str, str], None]) -> None:
        response_id = event.provider_response_id if isinstance(event, ResponseDone) else _content_response_id(event)
        if response_id is not None and response_id not in self._open_responses:
            issue('lifecycle.content_outside_response', f'{type(event).__name__} for {response_id!r} outside it')

    def _feed_codec(self, event: RealtimeCodecEvent, issue: Callable[[str, str], None]) -> None:
        if self._fatal:
            issue('codec.event_after_fatal', f'{type(event).__name__} after a non-recoverable error')
        if isinstance(event, RealtimeSessionReconnectEvent):
            # A new connection: ids are only unique per server session, and the open response is gone.
            self._ended.clear()
            self._open = None
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
        elif response_id is not None:
            if self._open is not None and self._open != response_id:
                issue('codec.overlapping_responses', f'content for {response_id!r} while {self._open!r} is still open')
            # A response's usage comes with its terminal: a function-call-only response has no `ResponseDone`.
            self._open = None if isinstance(event, SessionUsage) else response_id
        if isinstance(event, ResponseDone) and event.provider_response_id is not None:
            if event.provider_response_id in self._ended:
                issue('codec.duplicate_terminal', f'second ResponseDone for {event.provider_response_id!r}')
            self._ended.add(event.provider_response_id)
            if self._open == event.provider_response_id:
                self._open = None
        if isinstance(event, RealtimeSessionErrorEvent) and not event.recoverable:
            self._fatal = True

    def _feed_lifecycle(self, event: LifecycleEvent, issue: Callable[[str, str], None]) -> None:
        if isinstance(event, ResponseStarted):
            if event.response_id in self._started:
                issue('lifecycle.duplicate_start', f'{event.response_id!r} started twice')
            self._started.add(event.response_id)
            self._open_responses.add(event.response_id)
            sent = self.inputs_sent() if self.inputs_sent is not None else 0
            if unknown := [input_id for input_id in event.answers if not 0 <= input_id < sent]:
                issue('lifecycle.unknown_answer', f'{event.response_id!r} answers inputs never sent: {unknown}')
            self._settle(event.answers, issue)
        elif isinstance(event, ResponseEnded):
            if event.response_id in self._responses_ended:
                issue('lifecycle.duplicate_end', f'{event.response_id!r} ended twice')
            elif event.response_id not in self._open_responses:
                issue('lifecycle.end_without_start', f'{event.response_id!r} ended without starting')
            self._open_responses.discard(event.response_id)
            self._responses_ended.add(event.response_id)
        elif isinstance(event, UserTurnStarted):
            if event.turn_id in self._turns:
                issue('lifecycle.turn_started_twice', f'spoken turn {event.turn_id!r} started twice')
            self._turns.add(event.turn_id)
            self._open_turns.add(event.turn_id)
        elif isinstance(event, UserTurnEnded):
            if event.turn_id not in self._open_turns:
                issue('lifecycle.turn_end_without_start', f'spoken turn {event.turn_id!r} ended without starting')
            self._open_turns.discard(event.turn_id)
            self._joined_turns.add(event.turn_id)
        elif isinstance(event, UserTurnDiscarded):
            # A turn can be discarded once it joined, too: it just gets no more audio.
            if event.turn_id not in self._open_turns and event.turn_id not in self._joined_turns:
                issue('lifecycle.turn_end_without_start', f'spoken turn {event.turn_id!r} discarded without starting')
            self._open_turns.discard(event.turn_id)
            self._joined_turns.discard(event.turn_id)
        elif isinstance(event, InputAdded):
            if event.input_id in self._added_inputs:
                issue('lifecycle.input_added_twice', f'input {event.input_id} joined the conversation twice')
            self._added_inputs.add(event.input_id)
        else:
            self._settle(event.input_ids, issue)

    def _settle(self, input_ids: tuple[int, ...], issue: Callable[[str, str], None]) -> None:
        # Checked one at a time, so an event naming an input twice settles it twice too.
        twice: list[int] = []
        for input_id in input_ids:
            if input_id in self._settled_inputs:
                twice.append(input_id)
            self._settled_inputs.add(input_id)
        if twice:
            issue('lifecycle.input_settled_twice', f'inputs settled a second time: {twice}')

    def finish(self) -> list[ConformanceIssue]:
        """The stream ended: nothing may still be open."""
        found = [
            ConformanceIssue('lifecycle.unended_at_close', f'{response_id!r} never ended', self.events)
            for response_id in sorted(self._open_responses)
        ]
        found += [
            ConformanceIssue('lifecycle.turn_unended_at_close', f'spoken turn {turn_id!r} never ended', self.events)
            for turn_id in sorted(self._open_turns)
        ]
        self.issues.extend(found)
        return found


def _content_response_id(event: RealtimeCodecEvent) -> str | None:
    if isinstance(event, (AudioDelta, OutputTranscript, ToolCall)):
        return event.response_id
    if isinstance(event, SessionUsage) and event.response_scoped:
        return event.provider_response_id
    return None
