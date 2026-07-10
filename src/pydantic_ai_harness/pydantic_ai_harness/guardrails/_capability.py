"""Input and output guardrail capabilities.

`InputGuard` intercepts the first model request and lets a user-supplied
callable decide what to do with the user prompt. `OutputGuard` runs as the
model output is processed and decides what to do with the agent output.

A guard returns a bare `bool` (`True` = allow) or a
[`GuardResult`][pydantic_ai_harness.guardrails.GuardResult] — one of four
outcomes:

- `allow` — let the value through unchanged.
- `block` — refuse: `InputGuard` short-circuits the model call with a refusal
  message via [`SkipModelRequest`][pydantic_ai.exceptions.SkipModelRequest];
  `OutputGuard` raises [`OutputBlocked`][pydantic_ai_harness.guardrails.OutputBlocked].
- `replace` — substitute a sanitized value (redaction) and continue.
- `retry` — send the output back to the model to try again (`OutputGuard` only).

A guard that raises propagates the exception so the caller sees a hard
failure. Guards may be sync or async and may optionally take a
[`RunContext`][pydantic_ai.tools.RunContext] as their first argument.

`replace` and `block` are recorded as spans on the active OpenTelemetry
tracer, so a redaction or refusal is visible in Logfire traces. Content
attributes (the original/replacement values and the refusal message) are
attached only when `RunContext.trace_include_content` is set.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, Instrumentation, WrapModelRequestHandler
from pydantic_ai.exceptions import ModelRetry, SkipModelRequest, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.tools import AgentDepsT, RunContext
from typing_extensions import assert_never

from pydantic_ai_harness.guardrails._exceptions import OutputBlocked

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.output import OutputContext


_DEFAULT_INPUT_BLOCK_MESSAGE = 'Request blocked by input guardrail.'
_DEFAULT_OUTPUT_BLOCK_MESSAGE = 'Output blocked by output guardrail.'
_DEFAULT_OUTPUT_RETRY_MESSAGE = 'Output rejected by output guardrail.'


@dataclass(frozen=True, kw_only=True)
class GuardResult:
    """The outcome a guard reports for the value it inspected.

    Construct one with the classmethods — `GuardResult.allow()`,
    `GuardResult.block()`, `GuardResult.replace()`, `GuardResult.retry()` —
    rather than the raw fields. A guard may also return a bare `bool`: `True`
    is `allow()`, `False` is `block()`.
    """

    action: Literal['allow', 'block', 'replace', 'retry']
    """What the capability should do with the inspected value."""

    message: str | None = None
    """For `block`, the refusal text. For `retry`, the instruction sent back to the model."""

    replacement: object | None = None
    """For `replace`, the value substituted for the inspected one."""

    def __post_init__(self) -> None:
        """Reject field combinations the four-outcome contract does not allow."""
        match self.action:
            case 'allow':
                if self.message is not None or self.replacement is not None:
                    raise UserError("GuardResult(action='allow') must not set `message` or `replacement`.")
            case 'replace':
                if self.replacement is None:
                    raise UserError("GuardResult(action='replace') requires a `replacement` value.")
            case 'retry':
                if self.message is None:
                    raise UserError("GuardResult(action='retry') requires a `message`.")
            case 'block':
                # `message=None` is valid: a default is supplied at the use site.
                pass
            case _:  # pragma: no cover - assert_never exhaustiveness guard
                assert_never(self.action)

    @classmethod
    def allow(cls) -> GuardResult:
        """Let the value through unchanged."""
        return cls(action='allow')

    @classmethod
    def block(cls, message: str | None = None) -> GuardResult:
        """Refuse the value. `message` is the refusal text; `None` uses a default."""
        return cls(action='block', message=message)

    @classmethod
    def replace(cls, value: object) -> GuardResult:
        """Substitute `value` for the inspected one and continue.

        For `InputGuard`, `value` is the replacement prompt text sent to the
        model. For `OutputGuard`, it is the agent output returned to the caller.
        """
        return cls(action='replace', replacement=value)

    @classmethod
    def retry(cls, message: str) -> GuardResult:
        """Send the output back to the model to try again — `OutputGuard` only.

        `message` is the instruction the model sees on the retry.
        """
        return cls(action='retry', message=message)


GuardOutcome = bool | GuardResult
"""What a guard callable returns: a bare `bool` (`True` = allow), or a `GuardResult`."""


InputGuardFunc = (
    Callable[[str], GuardOutcome | Awaitable[GuardOutcome]]
    | Callable[[RunContext[AgentDepsT], str], GuardOutcome | Awaitable[GuardOutcome]]
)
"""Signature of the callable passed to `InputGuard`.

The callable receives the user prompt and returns `True` / `GuardResult`. It
may optionally take a [`RunContext`][pydantic_ai.tools.RunContext] as a first
argument — for `deps`, message history, or other run state — and may be sync
or async. Raising an exception is treated as a hard failure and propagates up
to the caller.
"""

OutputGuardFunc = (
    Callable[[object], GuardOutcome | Awaitable[GuardOutcome]]
    | Callable[[RunContext[AgentDepsT], object], GuardOutcome | Awaitable[GuardOutcome]]
)
"""Signature of the callable passed to `OutputGuard`.

The callable receives the agent output unchanged — for typed outputs this is
the Pydantic model — and returns `True` / `GuardResult`. It may optionally take
a [`RunContext`][pydantic_ai.tools.RunContext] first, and may be sync or async.
"""


def _takes_ctx(func: Callable[..., object]) -> bool:
    """Return `True` when `func` declares a leading `RunContext` parameter.

    Detected by parameter count, not annotation: a guard always takes the
    guarded value, so a second parameter means it also wants the run context.
    This matches pydantic-ai's own optional-`ctx` convention for output
    validators. A callable whose signature cannot be introspected is treated
    as taking the value only.
    """
    try:
        parameters = inspect.signature(func).parameters
    except ValueError:  # pragma: no cover - callable without an introspectable signature
        return False
    return len(parameters) > 1


async def _evaluate(
    guard: Callable[..., GuardOutcome | Awaitable[GuardOutcome]],
    ctx: RunContext[AgentDepsT],
    value: object,
) -> GuardResult:
    """Call `guard` (passing `ctx` when declared), await it, and normalize to `GuardResult`."""
    outcome = guard(ctx, value) if _takes_ctx(guard) else guard(value)
    if inspect.isawaitable(outcome):
        outcome = await outcome
    if isinstance(outcome, GuardResult):
        return outcome
    return GuardResult.allow() if outcome else GuardResult.block()


def _extract_prompt(ctx: RunContext[AgentDepsT], messages: Sequence[ModelMessage]) -> str | None:
    """Return the text of the most recent user prompt, or `None` if absent.

    Prefers `ctx.prompt` (set at run start) and falls back to scanning the
    message history for the last [`UserPromptPart`][pydantic_ai.messages.UserPromptPart]
    so that sub-agent calls or resumed runs without a fresh prompt still work.
    """
    if ctx.prompt is not None:
        return ctx.prompt if isinstance(ctx.prompt, str) else str(ctx.prompt)
    for message in reversed(messages):
        for part in reversed(message.parts):
            if isinstance(part, UserPromptPart):
                return part.content if isinstance(part.content, str) else str(part.content)
    return None


def _replace_prompt(messages: Sequence[ModelMessage], new_content: str) -> bool:
    """Rewrite the most recent user prompt to `new_content`. Returns whether one was found."""
    for message in reversed(messages):
        for part in reversed(message.parts):
            if isinstance(part, UserPromptPart):
                part.content = new_content
                return True
    return False


def _trace_block(ctx: RunContext[AgentDepsT], *, direction: str, message: str) -> None:
    """Record a zero-duration span marking a guardrail refusal.

    The refusal message is attached only when `ctx.trace_include_content` is
    set — it can quote sensitive content from the guarded value, and ops
    audiences are broader than the user who sees the refusal text.
    """
    attributes: dict[str, str] = {'guardrail.direction': direction, 'guardrail.action': 'block'}
    if ctx.trace_include_content:
        attributes['guardrail.message'] = message
    ctx.tracer.start_span(f'guardrail blocked {direction}', attributes=attributes).end()


def _trace_redaction(ctx: RunContext[AgentDepsT], *, direction: str, original: object, replacement: object) -> None:
    """Record a zero-duration span marking a guardrail redaction.

    The original and replacement values are attached only when
    `ctx.trace_include_content` is set, since a redacted value is often the
    sensitive content the guard exists to keep out of traces.
    """
    attributes: dict[str, str] = {'guardrail.direction': direction, 'guardrail.action': 'replace'}
    if ctx.trace_include_content:
        attributes['guardrail.original'] = str(original)
        attributes['guardrail.replacement'] = str(replacement)
    ctx.tracer.start_span(f'guardrail redacted {direction}', attributes=attributes).end()


@dataclass
class InputGuard(AbstractCapability[AgentDepsT]):
    """Validate the user prompt before it reaches the model.

    The `guard` callable receives the prompt text and returns one of the four
    outcomes (see the module docstring). `replace` rewrites the prompt sent to
    the model and also overwrites the original in the run's message history,
    so a redacted secret is not retained; a `str` replacement overwrites a
    multimodal prompt's other parts. `retry` is not valid for an input guard.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import GuardResult, InputGuard


    def no_secrets(prompt: str) -> GuardResult:
        if 'api_key' in prompt.lower():
            return GuardResult.block('Your message looks like it contains an API key.')
        return GuardResult.allow()


    agent = Agent('openai:gpt-5.4', capabilities=[InputGuard(guard=no_secrets)])
    ```

    The guard may take a [`RunContext`][pydantic_ai.tools.RunContext] as a
    first parameter when it needs run state — `deps` for tenant/role-aware
    policy, `messages` for conversation-aware checks. The parameter is detected
    from the signature, so prompt-only guards stay as-is.

    Set `parallel=True` to run the guard concurrently with the model call
    rather than before it, overlapping a slow guard (an LLM classifier, a
    network call) with the model round-trip so it adds no latency on the pass
    path. The model call is cancelled the moment the guard reports a
    violation. Trade-off: sequential mode never calls the model on a blocked
    prompt, whereas parallel mode has already started it — if the guard trips
    only after the model has responded, those tokens were still spent. Prefer
    sequential for fast local guards. `replace` (redaction) is incompatible
    with `parallel=True`, since the model call has already started with the
    original prompt.

    Scope: the guard runs exactly once per run — on the first model request —
    and evaluates the original user prompt. Subsequent model requests in the
    same run (e.g. after tool calls) are not re-checked, since the user input
    has not changed.

    Ordering: declares `position='innermost'` so any capability that morphs
    the messages (a prompt rewriter, a context manager) runs first and the
    guard sees the final prompt that will reach the model.
    """

    guard: InputGuardFunc[AgentDepsT]
    """Callable that decides what to do with the prompt before it reaches the model."""

    parallel: bool = False
    """Run the guard concurrently with the model request and cancel the model call on failure."""

    def get_ordering(self) -> CapabilityOrdering:
        """Sit innermost so message-morphing capabilities run first and the guard sees the final prompt."""
        return CapabilityOrdering(position='innermost')

    async def _run_guard(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
        prompt: str,
    ) -> None:
        """Evaluate the guard and act on its verdict.

        `allow` returns; `block` raises `SkipModelRequest`; `replace` rewrites
        the prompt in `request_context`; `retry` and `replace` under
        `parallel=True` raise `UserError`.
        """
        verdict = await _evaluate(self.guard, ctx, prompt)
        match verdict.action:
            case 'allow':
                return
            case 'retry':
                raise UserError(
                    'An InputGuard guard cannot return GuardResult.retry() — retry applies to model output only.'
                )
            case 'block':
                message = verdict.message or _DEFAULT_INPUT_BLOCK_MESSAGE
                _trace_block(ctx, direction='input', message=message)
                raise SkipModelRequest(ModelResponse(parts=[TextPart(content=message)]))
            case 'replace':
                if self.parallel:
                    raise UserError(
                        'InputGuard(parallel=True) is incompatible with GuardResult.replace(): the model call has '
                        'already started with the original prompt. Use sequential mode for prompt redaction.'
                    )
                replacement = verdict.replacement
                if not isinstance(replacement, str):
                    raise UserError(
                        'GuardResult.replace() for an input guard must provide replacement prompt text (str).'
                    )
                if not _replace_prompt(request_context.messages, replacement):
                    raise UserError('InputGuard could not find a user prompt to redact in the request.')
                _trace_redaction(ctx, direction='input', original=prompt, replacement=replacement)
            case _:  # pragma: no cover - assert_never exhaustiveness guard
                assert_never(verdict.action)

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        """Check the prompt before the first model call.

        Sequential mode runs the guard then the model. `parallel=True` races
        the guard against the model call and cancels it on a violation.
        """
        if ctx.run_step > 1:
            return await handler(request_context)
        prompt = _extract_prompt(ctx, request_context.messages)
        if prompt is None:
            return await handler(request_context)
        if not self.parallel:
            await self._run_guard(ctx, request_context, prompt)
            return await handler(request_context)

        async def run_handler() -> ModelResponse:
            return await handler(request_context)

        guard_task: asyncio.Task[None] = asyncio.create_task(self._run_guard(ctx, request_context, prompt))
        handler_task: asyncio.Task[ModelResponse] = asyncio.create_task(run_handler())
        try:
            done, _ = await asyncio.wait(
                {guard_task, handler_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if guard_task in done:
                await guard_task
                return await handler_task

            response = await handler_task
            await guard_task
            return response
        finally:
            for task in (guard_task, handler_task):
                if not task.done():
                    task.cancel()

            await asyncio.gather(guard_task, handler_task, return_exceptions=True)


@dataclass
class OutputGuard(AbstractCapability[AgentDepsT]):
    """Validate the agent output as it is produced.

    The `guard` callable receives the output — no automatic stringification, so
    a typed output arrives as the Pydantic model instance — and returns one of
    the four outcomes (see the module docstring): `allow` exposes the output,
    `block` raises [`OutputBlocked`][pydantic_ai_harness.guardrails.OutputBlocked],
    `replace` substitutes a sanitized output (redaction), and `retry` sends the
    output back to the model to try again.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import GuardResult, OutputGuard


    def no_pii(output: object) -> GuardResult:
        if 'SSN' in str(output):
            return GuardResult.retry('Do not include personal identifiers.')
        return GuardResult.allow()


    agent = Agent('openai:gpt-5.4', capabilities=[OutputGuard(guard=no_pii)])
    ```

    The guard runs as the output is processed, so `retry` reuses pydantic-ai's
    normal retry machinery and counts against the run's output-retry budget.
    Like `InputGuard`, the guard may take a
    [`RunContext`][pydantic_ai.tools.RunContext] as a first parameter; it is
    detected from the signature.

    Ordering: declares `position='outermost'` so the guard sees the final
    output after every inner capability has processed it, and `wrapped_by=
    [Instrumentation]` so an enclosing `Instrumentation` span always captures
    the guard's block/redact spans regardless of user list order.

    Streaming caveats. `retry` is supported with
    [`run()`][pydantic_ai.Agent.run] / `run_sync()` only — pydantic-ai does not
    support output retries during [`run_stream()`][pydantic_ai.Agent.run_stream],
    where a `retry` verdict surfaces as
    [`UnexpectedModelBehavior`][pydantic_ai.exceptions.UnexpectedModelBehavior].
    The guard inspects only the final output, not partial chunks, so during a
    streamed run the caller may already have received partial output before a
    `block` or `replace` verdict is reached — use `run()` when the output must
    be screened before any of it is exposed.
    """

    guard: OutputGuardFunc[AgentDepsT]
    """Callable that decides what to do with the agent output."""

    def get_ordering(self) -> CapabilityOrdering:
        """Sit outermost (inside `Instrumentation`) so the guard sees the final processed output."""
        return CapabilityOrdering(position='outermost', wrapped_by=[Instrumentation])

    async def after_output_process(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: Any,
    ) -> Any:
        """Evaluate the guard against the processed output and act on its verdict."""
        if ctx.partial_output:
            return output
        verdict = await _evaluate(self.guard, ctx, output)
        match verdict.action:
            case 'allow':
                return output
            case 'block':
                message = verdict.message or _DEFAULT_OUTPUT_BLOCK_MESSAGE
                _trace_block(ctx, direction='output', message=message)
                raise OutputBlocked(message)
            case 'retry':
                raise ModelRetry(verdict.message or _DEFAULT_OUTPUT_RETRY_MESSAGE)
            case 'replace':
                _trace_redaction(ctx, direction='output', original=output, replacement=verdict.replacement)
                return verdict.replacement
            case _:  # pragma: no cover - assert_never exhaustiveness guard
                assert_never(verdict.action)
