"""`StreamGuardrail`: screen each streamed text, thinking and tool-call part before the caller sees it."""

from __future__ import annotations

from collections.abc import AsyncIterable, Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from typing import assert_never

from typing_extensions import TypeIs

from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, Instrumentation
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    ModelResponseRejectedEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolCallPartDelta,
)
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai_harness._durable import in_durable_context
from pydantic_ai_harness.guardrails._exceptions import OutputBlocked
from pydantic_ai_harness.guardrails._shared import GuardOutcome, as_guards, evaluate_all, trace_block, trace_redaction

StreamGuardrailPart = TextPart | ThinkingPart | ToolCallPart
"""The streamed parts a `StreamGuardrail` holds back and screens."""

StreamGuardrailFunc = (
    Callable[[StreamGuardrailPart], GuardOutcome | Awaitable[GuardOutcome]]
    | Callable[[RunContext[AgentDepsT], StreamGuardrailPart], GuardOutcome | Awaitable[GuardOutcome]]
)
"""Signature of the callable passed to `StreamGuardrail`.

The callable receives a completed `TextPart`, `ThinkingPart` or `ToolCallPart` and returns
`True` / `GuardrailResult`. It may optionally take a [`RunContext`][pydantic_ai.tools.RunContext]
first, and may be sync or async.
"""

_DEFAULT_BLOCK_MESSAGE = 'Streamed part blocked by stream guardrail.'


def _is_part(value: object) -> TypeIs[StreamGuardrailPart]:
    return isinstance(value, StreamGuardrailPart)


def _replacement_check(part: StreamGuardrailPart) -> Callable[[object, int], None]:
    def check(replacement: object, position: int) -> None:
        if isinstance(part, ToolCallPart):
            raise UserError(
                f'A StreamGuardrail guard cannot return GuardrailResult.replace() for a tool call; guard at position '
                f'{position} did. The run would still execute the original arguments: rewrite them with ToolGuardrail.'
            )
        if not _is_part(replacement) or type(replacement) is not type(part):
            raise UserError(
                f'GuardrailResult.replace() for a StreamGuardrail must provide a {type(part).__name__} to replace '
                f'the streamed one; guard at position {position} returned {type(replacement).__name__}.'
            )

    return check


def _latest_start(events: list[AgentStreamEvent], index: int) -> int:
    """Where the latest part at `index` starts in `events`, leaving an earlier part at that index alone."""
    return max(i for i, event in enumerate(events) if isinstance(event, PartStartEvent) and event.index == index)


def _replace_part(events: list[AgentStreamEvent], index: int, part: StreamGuardrailPart) -> list[AgentStreamEvent]:
    """`events` with the latest part at `index` replaced by `part`, its deltas dropped."""
    start = _latest_start(events, index)
    return events[:start] + [
        replace(event, part=part)
        if isinstance(event, PartStartEvent | PartEndEvent) and event.index == index
        else event
        for event in events[start:]
        if not (isinstance(event, PartDeltaEvent) and event.index == index)
    ]


def _drop_part(events: list[AgentStreamEvent], index: int) -> list[AgentStreamEvent]:
    """`events` without those of the latest part at `index`."""
    start = _latest_start(events, index)
    return events[:start] + [
        event
        for event in events[start:]
        if not (isinstance(event, PartStartEvent | PartDeltaEvent | PartEndEvent) and event.index == index)
    ]


_Screen = Callable[[int, StreamGuardrailPart, list[AgentStreamEvent]], Awaitable[list[AgentStreamEvent]]]


@dataclass
class _Hold:
    """The events of one model response's stream that wait on a verdict.

    Text and thinking parts are screened at their end event. Tool-call parts are screened when the
    stream ends, as a provider can still stream arguments to a call after another part started.
    A delta for a part already released is dropped, as the guard never saw it.
    """

    screen: _Screen
    held: list[AgentStreamEvent] = field(default_factory=list[AgentStreamEvent])
    streaming: set[int] = field(default_factory=set[int])
    calls: dict[int, ToolCallPart] = field(default_factory=dict[int, ToolCallPart])
    released: set[int] = field(default_factory=set[int])

    async def feed(self, event: AgentStreamEvent) -> list[AgentStreamEvent]:
        """Take `event` and return the events that can be released now."""
        if isinstance(event, PartStartEvent):
            self._start(event)
        elif isinstance(event, PartDeltaEvent) and not self._delta(event):
            return []
        if not (self.streaming or self.calls):
            return [event]
        self.held.append(event)
        if isinstance(event, PartEndEvent) and event.index in self.streaming and _is_part(event.part):
            self.streaming.discard(event.index)
            if isinstance(event.part, ToolCallPart):
                self.calls[event.index] = event.part
            else:
                await self._screen(event.index, event.part)
        return self._release()

    async def finish(self) -> list[AgentStreamEvent]:
        """Screen the held tool calls once the stream has ended, and return what can be released."""
        for index in self.streaming:
            self.held = _drop_part(self.held, index)
        self.streaming.clear()
        for index, call in self.calls.items():
            await self._screen(index, call)
        self.calls.clear()
        return self._release()

    def _delta(self, event: PartDeltaEvent) -> bool:
        """Apply a delta to a held tool call, returning whether the delta is kept."""
        if event.index in self.released:
            return False
        call = self.calls.get(event.index)
        if call is not None and isinstance(event.delta, ToolCallPartDelta):
            updated = event.delta.apply(call)
            if isinstance(updated, ToolCallPart):  # pragma: no branch
                self.calls[event.index] = updated
        return True

    def _start(self, event: PartStartEvent) -> None:
        """Track a part that starts, dropping a held part at its index that will now never complete."""
        if event.index in self.streaming or event.index in self.calls:
            self.held = _drop_part(self.held, event.index)
            self.streaming.discard(event.index)
            self.calls.pop(event.index, None)
        self.released.discard(event.index)
        if _is_part(event.part):
            self.streaming.add(event.index)

    async def _screen(self, index: int, part: StreamGuardrailPart) -> None:
        self.held = await self.screen(index, part, self.held)
        self.released.add(index)

    def _release(self) -> list[AgentStreamEvent]:
        if self.streaming or self.calls:
            return []
        released, self.held = self.held, []
        return released


@dataclass
class StreamGuardrail(AbstractCapability[AgentDepsT]):
    """Screen each streamed text, thinking and tool-call part before the caller receives it.

    The events of each such part are held back until it is complete, when the `guard` receives the
    whole part: `allow` releases the events as they arrived, `block` raises
    [`OutputBlocked`][pydantic_ai_harness.guardrails.OutputBlocked] without releasing them, and
    `replace` releases a start and end event carrying the replacement part instead, which must be
    of the same type. An allowed tool call is likewise released as one start and end event carrying
    the screened part, as its argument deltas can overwrite earlier keys. Events that arrive while a
    part is held are held with it, so the caller sees them in their original order. `retry` and
    `approve` raise
    [`UserError`][pydantic_ai.exceptions.UserError], as does `replace` for a tool call, whose
    original arguments the run would still execute: rewrite those with `ToolGuardrail`.

    ```python
    from dataclasses import replace

    from pydantic_ai import Agent
    from pydantic_ai.messages import TextPart
    from pydantic_ai_harness import GuardrailResult, StreamGuardrail
    from pydantic_ai_harness.guardrails import StreamGuardrailPart


    def no_ssn(part: StreamGuardrailPart) -> GuardrailResult:
        if isinstance(part, TextPart) and 'SSN' in part.content:
            return GuardrailResult.replace(replace(part, content='[redacted]'))
        return GuardrailResult.allow()


    agent = Agent('openai:gpt-5.4', capabilities=[StreamGuardrail(guard=no_ssn)])
    ```

    A text or thinking part is complete at its end event, so its first chunk is delayed by the
    part's generation time plus the guard's latency. A tool call is complete only when the model
    response ends, as a provider can still stream its arguments after another part started, and
    a delta for a part already released is dropped.

    The guard screens the run's event stream: `run_stream_events()`, the `event_stream_handler`
    argument to a run, `node.stream()` under `iter()`, and UI adapters. A
    [`ModelResponseRejectedEvent`][pydantic_ai.messages.ModelResponseRejectedEvent] passes with its
    response's text, thinking and tool-call parts left out, as the stream already carried what the
    guard released of them. A `ProcessEventStream`
    capability and an `on_event` hook see each event before the guard, so they are not screened.
    `run_stream()` is not screened: its `stream_text()`, `stream_output()` and `stream_response()`
    read the accumulated response, which never carries a replacement and can hold the start of a
    part before the guard decides on it, though a `block` still raises. A replacement changes only
    what the event stream carries, not the message history or the run's output: pair it with
    `OutputGuardrail` and `ToolGuardrail` for those.

    Adding it makes `run()` and `run_sync()` stream every model request, so the guard runs even when
    nothing reads the events, and a model that cannot stream fails. It raises `UserError` under
    durable execution, where the model request streams its events to the durability's own handler
    before any capability wraps them, and in a realtime session, whose single event stream would
    stay held from the first tool call until the session closes.
    """

    guard: StreamGuardrailFunc[AgentDepsT] | Sequence[StreamGuardrailFunc[AgentDepsT]]
    """One callable, or several run in order, deciding what happens to each streamed part.

    In a chain the first `block` ends it, and a `replace` substitutes the part the remaining guards see.
    """

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Exclude a callable policy from YAML and JSON agent specifications."""
        return None

    def get_ordering(self) -> CapabilityOrdering:
        """Sit outermost (inside `Instrumentation`) so the guard screens what every other capability streams."""
        return CapabilityOrdering(position='outermost', wrapped_by=[Instrumentation])

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        """Refuse a run whose events the guard cannot hold back, before any model stream opens."""
        if in_durable_context(ctx):
            raise UserError(
                'StreamGuardrail cannot run under durable execution: the durable model request streams its events '
                "to the durability's event stream handler before any capability wraps them, so they would not be "
                'screened.'
            )
        if ctx.realtime:
            raise UserError(
                'StreamGuardrail cannot run in a realtime session: its event stream lasts the whole session, so a '
                'held tool call would hold back every later event until the session closes.'
            )

    async def wrap_run_event_stream(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        stream: AsyncIterable[AgentStreamEvent],
    ) -> AsyncIterable[AgentStreamEvent]:
        """Hold each screened part's events until the guard has decided on the complete part."""
        hold = _Hold(partial(self._screen, ctx))
        async for event in stream:
            if isinstance(event, ModelResponseRejectedEvent):
                # The verdicts apply to the part events, so the rejected response's own parts are as the model
                # streamed them: a value the guard redacted, or a part cut short before the guard saw it complete.
                parts = [part for part in event.response.parts if not _is_part(part)]
                event = replace(event, response=replace(event.response, parts=parts))
            for released in await hold.feed(event):
                yield released
        for released in await hold.finish():
            yield released

    async def _screen(
        self, ctx: RunContext[AgentDepsT], index: int, part: StreamGuardrailPart, held: list[AgentStreamEvent]
    ) -> list[AgentStreamEvent]:
        """Run the guard chain on the complete `part` at `index`, returning the `held` events to release."""
        verdict, position = await evaluate_all(
            as_guards(self.guard, capability='StreamGuardrail'), ctx, part, check_replacement=_replacement_check(part)
        )
        match verdict.action:
            case 'allow':
                return _replace_part(held, index, part) if isinstance(part, ToolCallPart) else held
            case 'replace':
                assert _is_part(verdict.replacement)
                trace_redaction(ctx, direction='stream', replacement=verdict.replacement)
                return _replace_part(held, index, verdict.replacement)
            case 'block':
                message = verdict.message or _DEFAULT_BLOCK_MESSAGE
                tool_name = part.tool_name if isinstance(part, ToolCallPart) else None
                trace_block(ctx, direction='stream', message=message, tool_name=tool_name)
                raise OutputBlocked(message)
            case 'retry' | 'approve':
                raise UserError(
                    f'A StreamGuardrail guard cannot return GuardrailResult.{verdict.action}(); guard at position '
                    f'{position} did. A streamed part can only be allowed, blocked or replaced.'
                )
            case _:  # pragma: no cover
                assert_never(verdict.action)
