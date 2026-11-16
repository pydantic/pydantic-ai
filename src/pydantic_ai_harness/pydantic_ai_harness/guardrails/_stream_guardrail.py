"""`StreamGuardrail`: screen each streamed text, thinking and tool-call part before the caller sees it."""

from __future__ import annotations

from collections.abc import AsyncIterable, Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import assert_never

from typing_extensions import TypeIs

from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, Instrumentation
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
)
from pydantic_ai.tools import AgentDepsT, RunContext
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


def _replacement_check(part: StreamGuardrailPart) -> Callable[[object, int], StreamGuardrailPart]:
    def check(replacement: object, position: int) -> StreamGuardrailPart:
        if not _is_part(replacement) or type(replacement) is not type(part):
            raise UserError(
                f'GuardrailResult.replace() for a StreamGuardrail must provide a {type(part).__name__} to replace '
                f'the streamed one; guard at position {position} returned {type(replacement).__name__}.'
            )
        return replacement

    return check


def _replace_part(events: list[AgentStreamEvent], index: int, part: StreamGuardrailPart) -> list[AgentStreamEvent]:
    """`events` with the part at `index` replaced by `part`, its deltas dropped."""
    return [
        replace(event, part=part)
        if isinstance(event, PartStartEvent | PartEndEvent) and event.index == index
        else event
        for event in events
        if not (isinstance(event, PartDeltaEvent) and event.index == index)
    ]


def _drop_part(events: list[AgentStreamEvent], index: int) -> list[AgentStreamEvent]:
    """`events` without those of the part at `index`."""
    return [
        event
        for event in events
        if not (isinstance(event, PartStartEvent | PartDeltaEvent | PartEndEvent) and event.index == index)
    ]


@dataclass
class StreamGuardrail(AbstractCapability[AgentDepsT]):
    """Screen each streamed text, thinking and tool-call part before the caller receives it.

    The events of each such part are held back from its start until it is complete, when the
    `guard` receives the whole part: `allow` releases the events as they arrived, `block` raises
    [`OutputBlocked`][pydantic_ai_harness.guardrails.OutputBlocked] without releasing them, and
    `replace` releases a start and end event carrying the replacement part instead, which must be
    of the same type. Events that arrive while a part is held are held with it, so the caller sees
    them in their original order. `retry` and `approve` raise
    [`UserError`][pydantic_ai.exceptions.UserError].

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

    Each part reaches the caller only once it is complete and screened, which adds the part's
    generation time and the guard's latency before its first chunk.

    The guard screens the run's event stream: `run_stream_events()`, an `event_stream_handler`,
    `node.stream()` under `iter()`, `StreamedRunResult.stream_text()` and UI adapters.
    `stream_output()` and `stream_response()` read the accumulated response rather than the events,
    and a replacement changes only what the event stream carries, not the message history or the
    run's output: pair it with `OutputGuardrail` and `ToolGuardrail` for those. The guard runs on
    the live stream, so it is not a durable operation.
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

    async def wrap_run_event_stream(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        stream: AsyncIterable[AgentStreamEvent],
    ) -> AsyncIterable[AgentStreamEvent]:
        """Hold each screened part's events until the guard has decided on the complete part.

        A part that starts while another is held means the held one will not complete, as when a
        model stream is replaced mid-part, so the held part's events are dropped unscreened.
        """
        held: list[AgentStreamEvent] = []
        open_index: int | None = None
        async for event in stream:
            if isinstance(event, PartStartEvent) and isinstance(event.part, StreamGuardrailPart):
                if open_index is not None:
                    held = _drop_part(held, open_index)
                open_index = event.index
            if open_index is None:
                yield event
                continue
            held.append(event)
            if (
                isinstance(event, PartEndEvent)
                and event.index == open_index
                and isinstance(event.part, StreamGuardrailPart)
            ):
                released = await self._screen(ctx, open_index, event.part, held)
                held, open_index = [], None
                for released_event in released:
                    yield released_event

    async def _screen(
        self, ctx: RunContext[AgentDepsT], index: int, part: StreamGuardrailPart, held: list[AgentStreamEvent]
    ) -> list[AgentStreamEvent]:
        """Run the guard chain on a complete `part`, returning the `held` events to release."""
        check = _replacement_check(part)
        verdict, position = await evaluate_all(
            as_guards(self.guard, capability='StreamGuardrail'), ctx, part, check_replacement=check
        )
        match verdict.action:
            case 'allow':
                return held
            case 'replace':
                trace_redaction(ctx, direction='stream', replacement=verdict.replacement)
                return _replace_part(held, index, check(verdict.replacement, position))
            case 'block':
                message = verdict.message or _DEFAULT_BLOCK_MESSAGE
                trace_block(ctx, direction='stream', message=message)
                raise OutputBlocked(message)
            case 'retry' | 'approve':
                raise UserError(
                    f'A StreamGuardrail guard cannot return GuardrailResult.{verdict.action}(); guard at position '
                    f'{position} did. A streamed part can only be allowed, blocked or replaced.'
                )
            case _:  # pragma: no cover - assert_never exhaustiveness guard
                assert_never(verdict.action)
