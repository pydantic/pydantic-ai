"""Detect repeated-action loops in a run and interrupt them."""

from __future__ import annotations

import inspect
import json
import re
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from hashlib import sha256
from itertools import takewhile
from typing import TYPE_CHECKING, Literal, TypeAlias

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import AgentRunError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, SystemPromptPart, ToolReturnPart
from pydantic_ai.tools import AgentDepsT, RunContext

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext

__all__ = ['DetectLoops', 'DetectedLoop', 'LoopDetectedError', 'LoopHandler', 'LoopPattern']

LoopPattern: TypeAlias = Literal['repeated_call', 'repeated_result', 'alternating_calls', 'repeated_text']
"""Which kind of loop was detected.

- `'repeated_call'`: one tool call (same tool, same arguments) made many times among the recent calls.
- `'repeated_result'`: one tool call that returned the same result each of the last several times it was made.
- `'alternating_calls'`: two different tool calls made in turn, A-B-A-B.
- `'repeated_text'`: consecutive responses without tool calls whose text is the same.
"""

_NUDGE_ORIGIN = 'pydantic-ai-harness.detect_loops.nudge.v1'
"""The `origin` in the `metadata` of the `SystemPromptPart` a nudge is delivered as."""

_UNSAFE_NAME_CHARACTERS = re.compile(r'[^A-Za-z0-9_]')

_CHANGE_APPROACH = 'Change your approach, or explain what is blocking you.'


@dataclass(frozen=True, kw_only=True)
class DetectedLoop:
    """A loop `DetectLoops` found in the run.

    Passed to a `LoopHandler`, and carried by `LoopDetectedError` as `loop`.
    """

    pattern: LoopPattern
    """Which kind of loop it is."""

    tool_names: tuple[str, ...]
    """The tools involved: one for `'repeated_call'` and `'repeated_result'`, two for
    `'alternating_calls'`, none for `'repeated_text'`."""

    count: int
    """How many times the pattern repeated: calls for `'repeated_call'` and `'repeated_result'`,
    A-B cycles for `'alternating_calls'`, responses for `'repeated_text'`."""

    window: int
    """How many recent tool calls (responses, for `'repeated_text'`) the detection looked at."""

    fingerprints: tuple[str, ...]
    """Stable hashes of what repeated.

    For the call patterns, each covers a tool name and its canonical arguments: one for `'repeated_call'`
    and `'repeated_result'`, two for `'alternating_calls'`. For `'repeated_text'`, it's the hash of the
    normalized text. They contain no argument values or text, so they are safe to log.
    """

    message: str
    """The explanation sent to the model as the nudge, and used as the `LoopDetectedError` message."""


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


@dataclass(frozen=True)
class _Call:
    tool_name: str
    fingerprint: str
    result: str | None
    """A hash of what the call returned, or `None` while history holds no result for it."""


@dataclass(frozen=True)
class _Step:
    """What one model response in the run did."""

    calls: tuple[_Call, ...]
    text: str | None
    """The normalized text of a response without tool calls, or `None`."""


@dataclass(kw_only=True)
class DetectLoops(AbstractCapability[AgentDepsT]):
    """Detect when an agent repeats the same actions without making progress, and interrupt it.

    Before each model request, the run's history is checked for four patterns: the same tool call
    made `repeated_call_threshold` times within the last `window` tool calls, the same tool call
    returning the same result `repeated_result_threshold` times in a row, two tool calls alternating
    for `alternating_calls_threshold` cycles, and `repeated_text_threshold` consecutive responses
    without tool calls whose text is the same. Set a threshold to `None` to turn that check off.

    On detection, `action` runs: `'nudge'` (the default) adds a system prompt to the request asking the
    model to change its approach, `'error'` raises `LoopDetectedError`, and a `LoopHandler` is called
    with the `DetectedLoop`. The calls that formed the loop are then not counted again, so the same loop
    has to build up again before it is reported a second time.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import DetectLoops

    agent = Agent('anthropic:claude-opus-5-5', capabilities=[DetectLoops()])
    ```

    Detection reads only the current run's messages, so a run that continues an earlier conversation
    starts with nothing counted, and a run replayed by a durable execution engine reaches the same
    decisions.
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

    action: Literal['nudge', 'error'] | LoopHandler[AgentDepsT] = 'nudge'
    """What to do when a loop is detected.

    - `'nudge'`: add a system prompt to the current request telling the model which loop it is in and
      asking it to change its approach.
    - `'error'`: raise `LoopDetectedError`, ending the run.
    - a `LoopHandler`: call it with the run context and the `DetectedLoop`.
    """

    _counted_responses: int = field(default=0, init=False, repr=False, compare=False)
    """How many of this run's responses a reported loop already used up."""

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError(f'`window` must be at least 2, got {self.window}.')
        thresholds = {
            'repeated_call_threshold': self.repeated_call_threshold,
            'repeated_result_threshold': self.repeated_result_threshold,
            'alternating_calls_threshold': self.alternating_calls_threshold,
            'repeated_text_threshold': self.repeated_text_threshold,
        }
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

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> DetectLoops[AgentDepsT]:
        """Return a fresh instance, so runs don't share what has already been reported."""
        return replace(self)

    async def before_model_request(
        self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Check the run's history for a loop, and act on the first one found."""
        steps = _run_steps(request_context.messages, ctx.run_id)
        loop = self._detect(steps[self._counted_responses :])
        if loop is not None:
            self._counted_responses = len(steps)
            await self._act(ctx, loop)
        return request_context

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
        return DetectedLoop(
            pattern='alternating_calls',
            tool_names=(first.tool_name, second.tool_name),
            count=cycles,
            window=len(recent),
            fingerprints=expected,
            message=(
                f'You have alternated between the same calls to `{_model_facing(first.tool_name)}` and '
                f'`{_model_facing(second.tool_name)}` {cycles} times without making progress. {_CHANGE_APPROACH}'
            ),
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

    async def _act(self, ctx: RunContext[AgentDepsT], loop: DetectedLoop) -> None:
        action = self.action
        ctx.tracer.start_span(
            'loop detected',
            attributes={
                'detect_loops.pattern': loop.pattern,
                'detect_loops.action': action if isinstance(action, str) else 'handler',
                'detect_loops.count': loop.count,
                'detect_loops.window': loop.window,
                'detect_loops.tool_names': loop.tool_names,
                'detect_loops.fingerprints': loop.fingerprints,
            },
        ).end()
        if not isinstance(action, str):
            result = action(ctx, loop)
            if inspect.isawaitable(result):
                await result
        elif action == 'error':
            raise LoopDetectedError(loop)
        else:
            metadata = {'origin': _NUDGE_ORIGIN, 'pattern': loop.pattern, 'tool_names': list(loop.tool_names)}
            ctx.enqueue(SystemPromptPart(loop.message, metadata=metadata))


def _run_steps(messages: Sequence[ModelMessage], run_id: str | None) -> list[_Step]:
    """What each of this run's model responses did, with each tool call matched to its result.

    A call's result is looked up only in the requests between its response and the next one, because
    some models reuse tool call ids across responses.
    """
    steps: list[_Step] = []
    for index, message in enumerate(messages):
        if not isinstance(message, ModelResponse) or message.run_id != run_id:
            continue
        results: dict[str, str] = {}
        for request in takewhile(lambda later: isinstance(later, ModelRequest), messages[index + 1 :]):
            for part in request.parts:
                if isinstance(part, ToolReturnPart):
                    results.setdefault(part.tool_call_id, _result_digest(part))
        calls = tuple(
            _Call(
                tool_name=part.tool_name,
                fingerprint=_digest(_canonical([part.tool_name, part.args_as_dict()])),
                result=results.get(part.tool_call_id),
            )
            for part in message.tool_calls
        )
        text = None if calls else ' '.join((message.text or '').split()).lower() or None
        steps.append(_Step(calls=calls, text=text))
    return steps


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
