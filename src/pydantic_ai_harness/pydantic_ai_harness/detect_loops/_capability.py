"""Detect repeated-action loops in a run and interrupt them."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from collections import Counter
from collections.abc import AsyncIterable, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from hashlib import sha256
from itertools import takewhile
from typing import TYPE_CHECKING, Literal, TypeAlias

import anyio

from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, WrapModelRequestHandler
from pydantic_ai.exceptions import AgentRunError, ModelRetry
from pydantic_ai.messages import (
    AgentStreamEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelResponsePartDelta,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
)
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai_harness._durable import in_durable_context
from pydantic_ai_harness.detect_loops._detectors import (
    LoopDetector,
    Repetition,
    RepetitionDetector,
    StreamChannel,
    StreamSample,
)

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext

__all__ = ['DetectLoops', 'DetectedLoop', 'LoopDetectedError', 'LoopHandler', 'LoopPattern']

LoopPattern: TypeAlias = Literal[
    'repeated_call',
    'repeated_result',
    'alternating_calls',
    'repeated_text',
    'repeated_thinking',
    'repeated_stream_text',
    'repeated_tool_args',
    'repeated_parallel_call',
]
"""Which kind of loop was detected.

Across the run's responses:

- `'repeated_call'`: one tool call (same tool, same arguments) made many times among the recent calls.
- `'repeated_result'`: one tool call that returned the same result each of the last several times it was made.
- `'alternating_calls'`: two different tool calls made in turn, A-B-A-B.
- `'repeated_text'`: consecutive responses without tool calls whose text is the same.

Within one response, as it streams:

- `'repeated_thinking'`: a thinking part that keeps repeating the same passage.
- `'repeated_stream_text'`: a text part that keeps repeating the same passage.
- `'repeated_tool_args'`: one tool call's arguments that keep repeating the same passage.
- `'repeated_parallel_call'`: the same tool call (same tool, same arguments) made many times in one response.
"""

_IN_RESPONSE_PATTERNS: dict[StreamChannel, LoopPattern] = {
    'thinking': 'repeated_thinking',
    'text': 'repeated_stream_text',
    'tool_args': 'repeated_tool_args',
}

_NUDGE_ORIGIN = 'pydantic-ai-harness.detect_loops.nudge.v1'
"""The `origin` in the `metadata` of the `SystemPromptPart` a nudge is delivered as."""

_UNSAFE_NAME_CHARACTERS = re.compile(r'[^A-Za-z0-9_]')

_CHANGE_APPROACH = 'Change your approach, or explain what is blocking you.'

_REJECTED = 'Your previous response was rejected because it kept repeating itself.'
"""The `ModelRetry` message a nudge for a loop within a response retries the request with."""


@dataclass(frozen=True, kw_only=True)
class DetectedLoop:
    """A loop `DetectLoops` found in the run.

    Passed to a `LoopHandler`, and carried by `LoopDetectedError` as `loop`.
    """

    pattern: LoopPattern
    """Which kind of loop it is."""

    tool_names: tuple[str, ...]
    """The tools involved: one for `'repeated_call'`, `'repeated_result'`, `'repeated_tool_args'` and
    `'repeated_parallel_call'`, two for `'alternating_calls'`, none for the others."""

    count: int
    """How many times the pattern repeated: calls for `'repeated_call'`, `'repeated_result'` and
    `'repeated_parallel_call'`, A-B cycles for `'alternating_calls'`, responses for `'repeated_text'`, and
    repeats of the passage for the other patterns within one response."""

    window: int
    """How much the detection looked at: recent tool calls for the call patterns, responses for
    `'repeated_text'`, the response's tool calls so far for `'repeated_parallel_call'`, and characters
    of the part for the other patterns within one response."""

    fingerprints: tuple[str, ...]
    """Stable hashes of what repeated.

    For the call patterns, each covers a tool name and its canonical arguments: one for `'repeated_call'`,
    `'repeated_result'` and `'repeated_parallel_call'`, two for `'alternating_calls'`. For `'repeated_text'`,
    it's the hash of the normalized text, and for the other patterns the hash of the passage that repeated.
    They don't contain argument values or text in the clear, but a short argument such as a user id can be
    recovered by hashing candidate values.
    """

    message: str
    """The explanation sent to the model as the nudge, and used as the `LoopDetectedError` message."""

    repeated_text: str | None = None
    """The passage that repeated, for `'repeated_thinking'`, `'repeated_stream_text'` and
    `'repeated_tool_args'`; `None` for the other patterns. Unlike `fingerprints`, it's the model's own text."""


class LoopDetectedError(AgentRunError):
    """Raised by `DetectLoops(action='error')` when it detects a loop.

    Subclasses `AgentRunError`, so code that already stops on agent run errors stops on this one too.
    """

    loop: DetectedLoop
    """The loop that was detected."""

    def __init__(self, loop: DetectedLoop) -> None:
        self.loop = loop
        super().__init__(loop.message)


LoopHandler: TypeAlias = Callable[[RunContext[AgentDepsT], DetectedLoop], Awaitable[None] | None]
"""A sync or async function `DetectLoops` calls with each loop it detects.

Raise from it to stop the run, or call `ctx.enqueue(...)` to steer the model your own way.
"""


@dataclass
class _Call:
    tool_name: str
    tool_call_id: str
    fingerprint: str
    result: str | None = None
    """A hash of what the call returned, or `None` until its result arrives."""


@dataclass(frozen=True)
class _Step:
    """What one model response in the run did."""

    timestamp: datetime
    """The response's timestamp, which finds it in history even after a capability replaced its parts."""

    calls: tuple[_Call, ...]
    text: str | None
    """The normalized text of a response without tool calls, or `None`."""


@dataclass
class _Part:
    """What one thinking part, text part or tool call's arguments in a response has had so far."""

    channel: StreamChannel
    tool_name: str | None
    keep: int
    """How many of the most recent characters `text` holds."""
    text: str = ''
    total: int = 0
    unchecked: int = 0
    """How many characters arrived since the part was last checked."""


@dataclass
class _ResponseCheck:
    """The checks on one model response, fed as it streams, or all at once when it arrives whole."""

    parts: dict[int, _Part] = field(default_factory=dict[int, _Part])
    calls: Counter[str] = field(default_factory=Counter[str])
    """How many times each tool call fingerprint appears in the response so far, since it was last reported."""
    call_count: int = 0
    streamed: bool = False
    found: DetectedLoop | None = None
    """A loop found in the stream, waiting for `wrap_model_request` to act on it."""
    detected: asyncio.Event = field(default_factory=asyncio.Event)
    resumed: asyncio.Event = field(default_factory=asyncio.Event)
    stopped: bool = False
    """Whether `wrap_model_request` stopped the stream, so the stream should pass nothing more on."""

    async def hand_over(self, loop: DetectedLoop) -> bool:
        """Hold the stream until `wrap_model_request` has acted on `loop`, and return whether it stopped the stream."""
        self.found = loop
        self.detected.set()
        await self.resumed.wait()
        self.resumed.clear()
        return self.stopped

    def take(self) -> DetectedLoop:
        loop = self.found
        assert loop is not None
        self.found = None
        self.detected.clear()
        return loop

    def release(self, *, stopped: bool) -> None:
        self.stopped = self.stopped or stopped
        self.resumed.set()


@dataclass(kw_only=True)
class DetectLoops(AbstractCapability[AgentDepsT]):
    """Detect when an agent repeats the same actions without making progress, and interrupt it.

    Each model response's tool calls and text are recorded as it arrives, and each call's result from the
    request that follows it. Before each model request, the run so far is checked for four patterns: the
    same tool call made `repeated_call_threshold` times within the last `window` tool calls, the same
    tool call returning the same result `repeated_result_threshold` times in a row, two tool calls
    alternating for `alternating_calls_threshold` cycles, and `repeated_text_threshold` consecutive
    responses without tool calls whose text is the same. Set a threshold to `None` to turn that check off.

    Each response is also checked on its own, as it streams: its thinking, its text and each tool call's
    arguments are handed to `stream_detector` every `stream_check_every` characters, to catch a model
    repeating the same passage, and the same tool call made `repeated_parallel_call_threshold` times in the
    response is reported before any of them run. A response that isn't streamed gets the same checks when
    it arrives.

    On detection, `action` runs: `'nudge'` (the default) adds a system prompt to the conversation, from
    the current request on, asking the model to change its approach, and for a loop within a response
    also stops the stream and makes the request again; `'error'` raises `LoopDetectedError`; and a
    `LoopHandler` is called with the `DetectedLoop`. The calls or text that formed the loop are then not
    counted again, so the same loop has to build up again before it is reported a second time.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import DetectLoops

    agent = Agent('anthropic:claude-opus-5-5', capabilities=[DetectLoops()])
    ```

    Only the current run's responses are recorded, so a run that continues an earlier conversation
    starts with nothing counted. Compaction that later drops or rewrites messages doesn't change what
    was recorded.
    """

    window: int = 10
    """How many of the most recent tool calls the call-based checks look at."""

    repeated_call_threshold: int | None = 5
    """Report a tool call (same tool, same arguments) made this many times within `window`."""

    repeated_result_threshold: int | None = 3
    """Report a tool call that returned the same result each of the last this-many times it was made
    within `window`, with no different result in between."""

    alternating_calls_threshold: int | None = 3
    """Report two different tool calls made in turn (A-B-A-B) for this many A-B cycles."""

    repeated_text_threshold: int | None = 3
    """Report this many consecutive responses without tool calls whose text is the same, ignoring case
    and whitespace."""

    stream_detector: LoopDetector[AgentDepsT] | None = field(default_factory=lambda: RepetitionDetector[AgentDepsT]())
    """How each response's thinking, text and tool call arguments are checked for a repeating passage.

    `RepetitionDetector` by default, a text heuristic. Pass `LoopJudge` to have a model confirm or find loops,
    your own `LoopDetector`, or `None` to turn these checks off.
    """

    stream_channels: Sequence[StreamChannel] = ('thinking', 'text', 'tool_args')
    """Which parts of a response `stream_detector` checks."""

    stream_check_every: int = 256
    """Check each part every this many characters, and once more when it ends."""

    repeated_parallel_call_threshold: int | None = 5
    """Report a tool call (same tool, same arguments) made this many times in one response, before any of
    them run."""

    action: Literal['nudge', 'error'] | LoopHandler[AgentDepsT] = 'nudge'
    """What to do when a loop is detected.

    - `'nudge'`: enqueue a system prompt telling the model which loop it is in and asking it to change
      its approach. It joins the current request and keeps the default conversation scope, so it keeps
      applying on later requests instead of being shown once. For a loop within a response, the stream
      is also stopped and the request made again with the nudge, through a `ModelRetry`.
    - `'error'`: raise `LoopDetectedError`, ending the run.
    - a `LoopHandler`: call it with the run context and the `DetectedLoop`. For a loop within a response,
      the response continues when it returns; raise `ModelRetry` from it to stop the stream and make the
      request again.
    """

    _steps: list[_Step] = field(default_factory=list[_Step], init=False, repr=False, compare=False)
    """What each of this run's model responses did, in order."""

    _results_pending: bool = field(default=False, init=False, repr=False, compare=False)
    """Whether the last recorded response's results are still to be read."""

    _counted_steps: int = field(default=0, init=False, repr=False, compare=False)
    """How many of `_steps` a reported loop already used up."""

    _streaming: _ResponseCheck | None = field(default=None, init=False, repr=False, compare=False)
    """The checks on the response being streamed, while one is."""

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError(f'`window` must be at least 2, got {self.window}.')
        thresholds = {
            'repeated_call_threshold': self.repeated_call_threshold,
            'repeated_result_threshold': self.repeated_result_threshold,
            'alternating_calls_threshold': self.alternating_calls_threshold,
            'repeated_text_threshold': self.repeated_text_threshold,
            'repeated_parallel_call_threshold': self.repeated_parallel_call_threshold,
        }
        if self.stream_check_every < 1:
            raise ValueError(f'`stream_check_every` must be at least 1, got {self.stream_check_every}.')
        for name, threshold in thresholds.items():
            if threshold is not None and threshold < 2:
                raise ValueError(f'`{name}` must be at least 2 or `None`, got {threshold}.')
        alternating_calls = self.alternating_calls_threshold
        calls_needed = {
            'repeated_call_threshold': self.repeated_call_threshold,
            'repeated_result_threshold': self.repeated_result_threshold,
            'alternating_calls_threshold': None if alternating_calls is None else alternating_calls * 2,
        }
        for name, needed in calls_needed.items():
            if needed is not None and needed > self.window:
                raise ValueError(
                    f'`{name}` needs {needed} tool calls, more than the {self.window} that `window` keeps, '
                    'so it could never fire.'
                )

    def get_ordering(self) -> CapabilityOrdering:
        """Run outermost, so the newest results are read before other capabilities, such as compaction, rewrite them."""
        return CapabilityOrdering(position='outermost')

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> DetectLoops[AgentDepsT]:
        """Return a fresh instance, so runs don't share what they recorded."""
        return replace(self)

    async def after_model_request(
        self, ctx: RunContext[AgentDepsT], *, request_context: ModelRequestContext, response: ModelResponse
    ) -> ModelResponse:
        """Record the response's tool calls, or its text when it has none."""
        calls = tuple(
            _Call(tool_name=tool_name, tool_call_id=tool_call_id, fingerprint=fingerprint)
            for tool_name, tool_call_id, fingerprint in _call_keys(response)
        )
        text = None if calls else ' '.join((response.text or '').split()).lower() or None
        self._steps.append(_Step(timestamp=response.timestamp, calls=calls, text=text))
        self._results_pending = True
        return response

    async def before_model_request(
        self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Record the results of the previous response's tool calls, then act on the first loop found."""
        if self._results_pending:
            _record_results(self._steps[-1], request_context.messages)
            self._results_pending = False
        loop = self._detect(self._steps[self._counted_steps :])
        if loop is not None:
            self._counted_steps = len(self._steps)
            await self._act(ctx, loop)
        return request_context

    @property
    def has_wrap_run_event_stream(self) -> bool:
        """`False`, so a run isn't made to stream for these checks: a response that isn't streamed is checked whole."""
        return False

    async def wrap_run_event_stream(
        self, ctx: RunContext[AgentDepsT], *, stream: AsyncIterable[AgentStreamEvent]
    ) -> AsyncIterable[AgentStreamEvent]:
        """Check the response being streamed as its events pass, holding an event that completes a loop until it's acted on."""
        async for event in stream:
            check = self._streaming
            if check is not None and not check.stopped:
                for loop in await self._observe(ctx, check, event):
                    if await check.hand_over(loop):
                        break
                if check.stopped:
                    # The request is being torn down, so the stream ends without this event.
                    continue
            yield event

    async def wrap_model_request(
        self, ctx: RunContext[AgentDepsT], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
    ) -> ModelResponse:
        """Act on loops within the response: as it streams, or once it has arrived whole."""
        if self.stream_detector is None and self.repeated_parallel_call_threshold is None:
            return await handler(request_context)
        check = _ResponseCheck()
        # Under durable execution the response streams inside an activity or step, and the workflow
        # only replays it, so it's checked whole here, in workflow code.
        if request_context.streaming and not in_durable_context(ctx):
            self._streaming = check
            try:
                response = await self._watch_stream(ctx, check, request_context, handler)
            finally:
                self._streaming = None
        else:
            response = await handler(request_context)
        if not check.streamed:
            for loop in await self._whole_response_loops(ctx, check, response):
                await self._act(ctx, loop, in_response=True)
        return response

    async def _watch_stream(
        self,
        ctx: RunContext[AgentDepsT],
        check: _ResponseCheck,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        """Run the request while acting on each loop the stream hands over, stopping the request when the action raises."""

        async def run_request() -> ModelResponse:
            return await handler(request_context)

        request = asyncio.create_task(run_request())
        try:
            while True:
                detected = asyncio.create_task(check.detected.wait())
                try:
                    await asyncio.wait({request, detected}, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    detected.cancel()
                if not check.detected.is_set():
                    return request.result()
                try:
                    await self._act(ctx, check.take(), in_response=True)
                except BaseException:
                    # Stopping the request makes the agent stop pulling the stream, before the stream is released.
                    await _stop(request)
                    raise
                finally:
                    check.release(stopped=request.done())
        finally:
            await _stop(request)
            check.release(stopped=True)

    async def _observe(
        self, ctx: RunContext[AgentDepsT], check: _ResponseCheck, event: AgentStreamEvent
    ) -> list[DetectedLoop]:
        """Feed one stream event to the checks, returning the loops it completes."""
        check.streamed = True
        if isinstance(event, PartStartEvent):
            text = self._start_part(check, event.index, event.part)
            return [] if text is None else await self._feed(ctx, check, event.index, text)
        if isinstance(event, PartDeltaEvent):
            if event.index not in check.parts or (text := _delta_text(event.delta)) is None:
                return []
            return await self._feed(ctx, check, event.index, text)
        if isinstance(event, PartEndEvent):
            return await self._end_part(ctx, check, event.index, event.part)
        return []

    async def _whole_response_loops(
        self, ctx: RunContext[AgentDepsT], check: _ResponseCheck, response: ModelResponse
    ) -> list[DetectedLoop]:
        """Run the checks a stream gets over a response that arrived whole, part by part."""
        loops: list[DetectedLoop] = []
        for index, part in enumerate(response.parts):
            if (text := self._start_part(check, index, part)) is not None:
                loops += await self._feed(ctx, check, index, text)
            loops += await self._end_part(ctx, check, index, part)
        return loops

    def _start_part(self, check: _ResponseCheck, index: int, part: ModelResponsePart) -> str | None:
        """Start checking a part that `stream_detector` checks, returning the text it starts with."""
        detector = self.stream_detector
        if detector is None:
            return None
        channel: StreamChannel
        if isinstance(part, ThinkingPart):
            channel, tool_name, text = 'thinking', None, part.content
        elif isinstance(part, TextPart):
            channel, tool_name, text = 'text', None, part.content
        elif isinstance(part, ToolCallPart):
            channel, tool_name, text = 'tool_args', part.tool_name, _args_text(part.args)
        else:
            return None
        if channel not in self.stream_channels:
            return None
        check.parts[index] = _Part(channel=channel, tool_name=tool_name, keep=detector.sample_chars)
        return text

    async def _feed(
        self, ctx: RunContext[AgentDepsT], check: _ResponseCheck, index: int, text: str
    ) -> list[DetectedLoop]:
        """Add `text` to a part, checking it each time another `stream_check_every` characters have arrived."""
        part = check.parts[index]
        loops: list[DetectedLoop] = []
        every = self.stream_check_every
        while text:
            split = every - part.unchecked
            piece, text = text[:split], text[split:]
            part.text = (part.text + piece)[-part.keep :]
            part.total += len(piece)
            part.unchecked += len(piece)
            if part.unchecked == every and (loop := await self._check_part(ctx, part)) is not None:
                loops.append(loop)
        return loops

    async def _end_part(
        self, ctx: RunContext[AgentDepsT], check: _ResponseCheck, index: int, part: ModelResponsePart
    ) -> list[DetectedLoop]:
        """Check the rest of a finished part, and count it if it's a tool call."""
        loops: list[DetectedLoop] = []
        checked = check.parts.pop(index, None)
        if checked is not None and checked.unchecked and (loop := await self._check_part(ctx, checked)) is not None:
            loops.append(loop)
        threshold = self.repeated_parallel_call_threshold
        if threshold is None or not isinstance(part, ToolCallPart):
            return loops
        fingerprint = _call_fingerprint(part)
        check.calls[fingerprint] += 1
        check.call_count += 1
        count = check.calls[fingerprint]
        if count >= threshold:
            check.calls[fingerprint] = 0
            loops.append(
                DetectedLoop(
                    pattern='repeated_parallel_call',
                    tool_names=(part.tool_name,),
                    count=count,
                    window=check.call_count,
                    fingerprints=(fingerprint,),
                    message=(
                        f'You called `{_model_facing(part.tool_name)}` with the same arguments {count} times in one '
                        f'response, which is not making progress. {_CHANGE_APPROACH}'
                    ),
                )
            )
        return loops

    async def _check_part(self, ctx: RunContext[AgentDepsT], part: _Part) -> DetectedLoop | None:
        detector = self.stream_detector
        assert detector is not None
        sample = StreamSample(
            channel=part.channel,
            text=part.text,
            total_chars=part.total,
            new_chars=part.unchecked,
            tool_name=part.tool_name,
        )
        part.unchecked = 0
        repetition = await detector.detect(ctx, sample)
        if repetition is None:
            return None
        # The passages that formed the loop are not counted again.
        part.text = ''
        return _in_part_loop(part, repetition, window=len(sample.text))

    def _detect(self, steps: Sequence[_Step]) -> DetectedLoop | None:
        recent = [call for step in steps for call in step.calls][-self.window :]
        return (
            self._repeated_result(recent)
            or self._alternating_calls(recent)
            or self._repeated_call(recent)
            or self._repeated_text(steps)
        )

    def _repeated_result(self, recent: Sequence[_Call]) -> DetectedLoop | None:
        threshold = self.repeated_result_threshold
        if threshold is None:
            return None
        streaks: dict[str, tuple[str | None, int]] = {}
        for call in recent:
            previous, streak = streaks.get(call.fingerprint, (None, 0))
            same = call.result is not None and call.result == previous
            streaks[call.fingerprint] = (call.result, streak + 1 if same else 1)
        for call in reversed(recent):
            _, streak = streaks[call.fingerprint]
            if streak >= threshold:
                return DetectedLoop(
                    pattern='repeated_result',
                    tool_names=(call.tool_name,),
                    count=streak,
                    window=len(recent),
                    fingerprints=(call.fingerprint,),
                    message=(
                        f'Your last {streak} calls to `{_model_facing(call.tool_name)}` used the same arguments '
                        f'and returned the same result, so calling it again will not change the outcome. '
                        f'{_CHANGE_APPROACH}'
                    ),
                )
        return None

    def _alternating_calls(self, recent: Sequence[_Call]) -> DetectedLoop | None:
        cycles = self.alternating_calls_threshold
        if cycles is None or len(recent) < cycles * 2:
            return None
        tail = recent[-cycles * 2 :]
        first, second = tail[0], tail[1]
        expected = (first.fingerprint, second.fingerprint)
        if first.fingerprint == second.fingerprint or any(
            call.fingerprint != expected[index % 2] for index, call in enumerate(tail)
        ):
            return None
        first_name, second_name = _model_facing(first.tool_name), _model_facing(second.tool_name)
        calls = (
            f'alternated between two calls to `{first_name}` with different arguments'
            if first_name == second_name
            else f'alternated between the same calls to `{first_name}` and `{second_name}`'
        )
        return DetectedLoop(
            pattern='alternating_calls',
            tool_names=(first.tool_name, second.tool_name),
            count=cycles,
            window=len(recent),
            fingerprints=expected,
            message=f'You have {calls} {cycles} times without making progress. {_CHANGE_APPROACH}',
        )

    def _repeated_call(self, recent: Sequence[_Call]) -> DetectedLoop | None:
        threshold = self.repeated_call_threshold
        if threshold is None or not recent:
            return None
        fingerprint, count = Counter(call.fingerprint for call in recent).most_common(1)[0]
        if count < threshold:
            return None
        tool_name = next(call.tool_name for call in recent if call.fingerprint == fingerprint)
        return DetectedLoop(
            pattern='repeated_call',
            tool_names=(tool_name,),
            count=count,
            window=len(recent),
            fingerprints=(fingerprint,),
            message=(
                f'You have called `{_model_facing(tool_name)}` with the same arguments {count} times in your last '
                f'{len(recent)} tool calls, which is not making progress. {_CHANGE_APPROACH}'
            ),
        )

    def _repeated_text(self, steps: Sequence[_Step]) -> DetectedLoop | None:
        threshold = self.repeated_text_threshold
        if threshold is None or not steps or (text := steps[-1].text) is None:
            return None
        count = 0
        for step in reversed(steps):
            if step.text != text:
                break
            count += 1
        if count < threshold:
            return None
        return DetectedLoop(
            pattern='repeated_text',
            tool_names=(),
            count=count,
            window=count,
            fingerprints=(_digest(text),),
            message=(
                f'Your last {count} responses had the same text and no tool calls, so repeating it will not '
                f'change the outcome. {_CHANGE_APPROACH}'
            ),
        )

    async def _act(self, ctx: RunContext[AgentDepsT], loop: DetectedLoop, *, in_response: bool = False) -> None:
        """Act on a loop: across the run's responses before a model request, or within the response `in_response`."""
        self._trace(ctx, loop)
        action = self.action
        if not isinstance(action, str):
            result = action(ctx, loop)
            if inspect.isawaitable(result):
                await result
        elif action == 'error':
            raise LoopDetectedError(loop)
        else:
            metadata = {'origin': _NUDGE_ORIGIN, 'pattern': loop.pattern, 'tool_names': list(loop.tool_names)}
            ctx.enqueue(SystemPromptPart(loop.message, metadata=metadata))
            if in_response:
                # Rejecting the response makes the request again, and the nudge joins it.
                raise ModelRetry(_REJECTED)

    def _trace(self, ctx: RunContext[AgentDepsT], loop: DetectedLoop) -> None:
        action = self.action
        attributes: dict[str, str | int | tuple[str, ...]] = {
            'detect_loops.pattern': loop.pattern,
            'detect_loops.action': action if isinstance(action, str) else 'handler',
            'detect_loops.count': loop.count,
            'detect_loops.window': loop.window,
            'detect_loops.tool_names': loop.tool_names,
        }
        if ctx.trace_include_content:
            # Hashes of short arguments, such as a user id, can be reversed by trying candidate values.
            attributes['detect_loops.fingerprints'] = loop.fingerprints
            if loop.repeated_text is not None:
                attributes['detect_loops.repeated_text'] = loop.repeated_text
        ctx.tracer.start_span('loop detected', attributes=attributes).end()


async def _stop(request: asyncio.Task[ModelResponse]) -> None:
    """Cancel the model request and wait for it to finish tearing its stream down."""
    request.cancel()
    # Shielded, so an outer cancellation doesn't leave the stream half torn down.
    with anyio.CancelScope(shield=True):
        await asyncio.wait({request})
    if not request.cancelled():
        request.exception()


def _in_part_loop(part: _Part, repetition: Repetition, *, window: int) -> DetectedLoop:
    count = repetition.count
    if part.channel == 'thinking':
        what = 'Your thinking'
    elif part.channel == 'text':
        what = 'Your response'
    else:
        assert part.tool_name is not None
        what = f'The arguments of your call to `{_model_facing(part.tool_name)}`'
    return DetectedLoop(
        pattern=_IN_RESPONSE_PATTERNS[part.channel],
        tool_names=() if part.tool_name is None else (part.tool_name,),
        count=count,
        window=window,
        fingerprints=(_digest(repetition.text),),
        message=f'{what} repeated the same passage {count} times in a row, which is not making progress. {_CHANGE_APPROACH}',
        repeated_text=repetition.text,
    )


def _delta_text(delta: ModelResponsePartDelta) -> str | None:
    if isinstance(delta, ToolCallPartDelta):
        return _args_text(delta.args_delta)
    # Only thinking parts, text parts and tool calls are checked.
    assert isinstance(delta, TextPartDelta | ThinkingPartDelta)
    return delta.content_delta


def _args_text(args: str | Mapping[str, object] | None) -> str:
    """Tool call arguments as the text the model wrote: as streamed, or as JSON when they came as an object."""
    if args is None:
        return ''
    return args if isinstance(args, str) else json.dumps(args)


def _call_keys(response: ModelResponse) -> list[tuple[str, str, str]]:
    """The tool name, tool call id, and fingerprint of each tool call in `response`."""
    return [(part.tool_name, part.tool_call_id, _call_fingerprint(part)) for part in response.tool_calls]


def _call_fingerprint(part: ToolCallPart) -> str:
    """A hash of a tool call's name and canonical arguments."""
    return _digest(_canonical([part.tool_name, part.args_as_dict()]))


def _record_results(step: _Step, messages: Sequence[ModelMessage]) -> None:
    """Fill in the results of `step`'s calls from the requests that follow its response in `messages`.

    The response is the last one with the step's timestamp and tool call ids, because a capability that
    short-circuits a model request can put a response DetectLoops never recorded after it, and some models
    reuse tool call ids across responses. If none matches, the last response is used. Only the requests up
    to the next response are read.
    """
    ids = [(call.tool_name, call.tool_call_id) for call in step.calls]
    responses = [(index, message) for index, message in enumerate(messages) if isinstance(message, ModelResponse)]
    matching = (
        index
        for index, response in reversed(responses)
        if response.timestamp == step.timestamp
        and [(part.tool_name, part.tool_call_id) for part in response.tool_calls] == ids
    )
    start = next(matching, responses[-1][0] if responses else len(messages)) + 1
    digests: dict[str, list[str]] = {}
    for message in takewhile(lambda message: isinstance(message, ModelRequest), messages[start:]):
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                digests.setdefault(part.tool_call_id, []).append(_result_digest(part))
    for call in step.calls:
        # Calls that share an id within one response take their results in order.
        results = digests.get(call.tool_call_id)
        call.result = results.pop(0) if results else None


def _result_digest(part: ToolReturnPart) -> str:
    """A hash of what a tool returned: its outcome, its structured content, and the identifiers of its files."""
    files = [file.identifier for file in part.files]
    return _digest(_canonical([part.outcome, part.model_response_object(), files]))


def _canonical(value: object) -> str:
    """JSON with sorted keys, so equal structured values serialize identically."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=str)


def _digest(text: str) -> str:
    return sha256(text.encode()).hexdigest()[:16]


def _model_facing(tool_name: str) -> str:
    """The tool name as it may appear in framework text: only `[A-Za-z0-9_]`."""
    return _UNSAFE_NAME_CHARACTERS.sub('_', tool_name)
