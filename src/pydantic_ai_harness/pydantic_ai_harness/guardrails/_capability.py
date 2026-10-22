"""Input and output guardrail capabilities.

`InputGuardrail` intercepts the first model request and lets a user-supplied
callable decide what to do with the user prompt. `OutputGuardrail` runs as the
model output is processed and decides what to do with the agent output.

A guard returns a bare `bool` (`True` = allow) or a
[`GuardrailResult`][pydantic_ai_harness.guardrails.GuardrailResult] -- one of five
outcomes:

- `allow`: let the value through unchanged.
- `block`: refuse. `InputGuardrail` short-circuits the model call with a refusal
  message via [`SkipModelRequest`][pydantic_ai.exceptions.SkipModelRequest];
  `OutputGuardrail` raises [`OutputBlocked`][pydantic_ai_harness.guardrails.OutputBlocked].
- `replace`: substitute a sanitized value (redaction) and continue.
- `retry`: send the output back to the model to try again (not valid for input).
- `approve`: defer a tool call for human approval
  ([`ToolGuardrail`][pydantic_ai_harness.ToolGuardrail] arguments only).

A guard that raises propagates the exception so the caller sees a hard
failure. Guards may be sync or async and may optionally take a
[`RunContext`][pydantic_ai.tools.RunContext] as their first argument.

`replace` and `block` are recorded as spans on the active OpenTelemetry
tracer, so a redaction or refusal is visible in Logfire traces. Content
attributes (the replacement value and the refusal message) are attached only
when `RunContext.trace_include_content` is set. The original value is never recorded.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, assert_never

import anyio

from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, Instrumentation, WrapModelRequestHandler
from pydantic_ai.exceptions import ModelRetry, SkipModelRequest, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserContent, UserPromptPart
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai_harness._background_completion import get_background_completion
from pydantic_ai_harness.guardrails._exceptions import OutputBlocked
from pydantic_ai_harness.guardrails._shared import (
    GuardOutcome,
    as_guards,
    evaluate_all,
    trace_block,
    trace_redaction,
)

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.output import OutputContext


_DEFAULT_INPUT_BLOCK_MESSAGE = 'Request blocked by input guardrail.'
_DEFAULT_OUTPUT_BLOCK_MESSAGE = 'Output blocked by output guardrail.'
_DEFAULT_OUTPUT_RETRY_MESSAGE = 'Output rejected by output guardrail.'


InputGuardrailFunc = (
    Callable[[str], GuardOutcome | Awaitable[GuardOutcome]]
    | Callable[[RunContext[AgentDepsT], str], GuardOutcome | Awaitable[GuardOutcome]]
)
"""Signature of the callable passed to `InputGuardrail`.

The callable receives the user prompt and returns `True` / `GuardrailResult`. It
may optionally take a [`RunContext`][pydantic_ai.tools.RunContext] as a first
argument -- for `deps`, message history, or other run state -- and may be sync
or async. Raising an exception is treated as a hard failure and propagates up
to the caller.
"""

OutputGuardrailFunc = (
    Callable[[object], GuardOutcome | Awaitable[GuardOutcome]]
    | Callable[[RunContext[AgentDepsT], object], GuardOutcome | Awaitable[GuardOutcome]]
)
"""Signature of the callable passed to `OutputGuardrail`.

The callable receives the agent output unchanged -- for typed outputs this is
the Pydantic model -- and returns `True` / `GuardrailResult`. It may optionally take
a [`RunContext`][pydantic_ai.tools.RunContext] first, and may be sync or async.
"""


def _require_prompt_text(replacement: object, position: int) -> str:
    """A prompt replacement has to be text, since it is written back into the message."""
    if not isinstance(replacement, str):
        raise UserError(
            f'GuardrailResult.replace() for an input guardrail must provide replacement prompt text (str); '
            f'guard at position {position} returned {type(replacement).__name__}.'
        )
    return replacement


def _find_prompt_part(
    ctx: RunContext[AgentDepsT], messages: Sequence[ModelMessage]
) -> tuple[UserPromptPart | None, UserPromptPart | None]:
    """The user's prompt for this run as it appears in this request, and its original in the history.

    The prompt is the most recent `UserPromptPart` in the run's history, `ctx.messages`. `messages`
    is this request's copy of the history, where an outer capability can add a `UserPromptPart` for
    this request only (`SystemReminders`, `Planning` and `Memory` append one to the last request).
    That part is not the user's prompt, so the prompt is found by identity: the guard neither
    screens nor overwrites the added part. A prompt an outer capability rewrote in place is still
    the same object, so the guard reads the rewritten text, and the two returned parts are the same.

    When a capability replaced the prompt with a copy, the copy is the part at the prompt's
    position: the same part index in the message the same distance from the end. Parts appended
    after it and earlier turns are not taken for it. The copy is returned with the history part it
    was made from, so a redaction can overwrite both when the copy still has the original's text.
    Only when that position holds no user prompt, because the request was rebuilt, is the most
    recent part in `messages` used, with no original: nothing ties that part to the history.
    """
    for from_end, message in enumerate(reversed(ctx.messages), start=1):
        prompts = [(index, part) for index, part in enumerate(message.parts) if isinstance(part, UserPromptPart)]
        if prompts:
            index, prompt = prompts[-1]
            break
    else:
        return _latest_prompt_part(messages), None
    if any(part is prompt for request in messages for part in request.parts):
        return prompt, prompt
    parts = messages[-from_end].parts if from_end <= len(messages) else ()
    copy = parts[index] if index < len(parts) else None
    if isinstance(copy, UserPromptPart):
        return copy, prompt
    return _latest_prompt_part(messages), None


def _latest_prompt_part(messages: Sequence[ModelMessage]) -> UserPromptPart | None:
    """The most recent `UserPromptPart` in `messages`, if any."""
    return next(
        (
            part
            for message in reversed(messages)
            for part in reversed(message.parts)
            if isinstance(part, UserPromptPart)
        ),
        None,
    )


def _prompt_text(content: str | Sequence[UserContent]) -> str:
    """The text a guard sees for a prompt: a background completion's text, or the prompt rendered as text."""
    completion = get_background_completion(content)
    if completion is not None:
        return completion.content
    return content if isinstance(content, str) else str(content)


def _replace_prompt(part: UserPromptPart, new_content: str) -> None:
    """Rewrite the user prompt `part` to `new_content`.

    A multimodal prompt is refused rather than rewritten. Its `content` is a
    sequence of parts, and a `str` written over it would send the model the
    replacement text alone -- the images, documents and audio the user attached
    would be gone, with nothing in the run to say so.
    Singleton native background completions are rewritten in place, preserving
    their metadata.
    """
    if isinstance(part.content, str):
        part.content = new_content
        return
    completion = get_background_completion(part.content)
    if completion is not None:
        completion.content = new_content
        return
    raise UserError(
        'An InputGuardrail returned GuardrailResult.replace() for a multimodal prompt. The prompt '
        f'holds {type(part.content).__name__} rather than text, so writing the replacement over it '
        'would drop the attached parts. Return GuardrailResult.allow() or GuardrailResult.block() '
        'for prompts that are not plain text, or guard the output instead.'
    )


@dataclass
class InputGuardrail(AbstractCapability[AgentDepsT]):
    """Validate the user prompt before it reaches the model.

    The `guard` callable receives the prompt text and returns one of the
    outcomes (see the module docstring). `replace` rewrites the prompt sent to
    the model. When that is the history's own prompt part, or a copy of it with
    the same text, the original in the run's message history is rewritten too,
    so a redacted secret is not retained. When an outer capability changed the
    copy's text, or rebuilt the request with a new prompt part, only the part
    sent to the model is rewritten: the history keeps the original prompt,
    including anything the guard redacted from it. A later guard in a chain
    that blocks ends the chain before any rewrite happens.

    `retry` is not valid for an input guardrail, and neither is `replace` on a
    multimodal prompt: the guard sees the whole prompt rendered as text, and
    writing one string back over a prompt built from several parts would drop
    the attached ones, so it raises
    [`UserError`][pydantic_ai.exceptions.UserError] instead.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import GuardrailResult, InputGuardrail


    def no_secrets(prompt: str) -> GuardrailResult:
        if 'api_key' in prompt.lower():
            return GuardrailResult.block('Your message looks like it contains an API key.')
        return GuardrailResult.allow()


    agent = Agent('openai:gpt-5.4', capabilities=[InputGuardrail(guard=no_secrets)])
    ```

    The guard may take a [`RunContext`][pydantic_ai.tools.RunContext] as a
    first parameter when it needs run state -- `deps` for tenant/role-aware
    policy, `messages` for conversation-aware checks. The parameter is detected
    from the signature, so prompt-only guards stay as-is.

    Set `parallel=True` to run the guard concurrently with the model call
    rather than before it, overlapping a slow guard (an LLM classifier, a
    network call) with the model round-trip so it adds no latency on the pass
    path. The model call is cancelled the moment the guard reports a
    violation. Trade-off: sequential mode never calls the model on a blocked
    prompt, whereas parallel mode has already started it -- if the guard trips
    only after the model has responded, those tokens were still spent. Prefer
    sequential for fast local guards. `replace` (redaction) is incompatible
    with `parallel=True`, since the model call has already started with the
    original prompt.

    Scope: the guard runs once per run, on the first model request, and
    evaluates the most recent user prompt in the run's history. Subsequent
    model requests in the same run (e.g. after tool calls) are not re-checked,
    since the user input has not changed. A run resumed without a new prompt
    (to answer deferred tool calls, say) has no new user input either, so its
    first request screens the most recent prompt in the history again.

    Ordering: declares `position='innermost'` so any capability that rewrites
    the user prompt (a prompt rewriter, a history processor) runs first and the
    guard sees the rewritten text. A `UserPromptPart` that an outer capability
    adds to this request only, such as a `SystemReminders` reminder, is not the
    user's prompt: the guard neither screens nor rewrites it.
    """

    guard: InputGuardrailFunc[AgentDepsT] | Sequence[InputGuardrailFunc[AgentDepsT]]
    """One callable, or several run in order, deciding what happens to the prompt.

    In a chain the first `block` ends it, and a `replace` substitutes the text
    the remaining guards see -- so a redactor placed first cleans the prompt the
    ones after it inspect.
    """

    parallel: bool = False
    """Run the guard concurrently with the model request and cancel the model call on failure."""

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Exclude a callable policy from YAML and JSON agent specifications."""
        return None

    def get_ordering(self) -> CapabilityOrdering:
        """Sit innermost so message-morphing capabilities run first and the guard sees the final prompt."""
        return CapabilityOrdering(position='innermost')

    async def _run_guard(
        self,
        ctx: RunContext[AgentDepsT],
        part: UserPromptPart | None,
        original: UserPromptPart | None,
        prompt: str,
    ) -> None:
        """Evaluate the guard and act on its verdict.

        `allow` returns; `block` raises `SkipModelRequest`; `replace` rewrites
        the prompt `part` sent to the model and its `original` in the run's
        history when that is a different object with the same text; `retry`
        and `replace` under `parallel=True` raise `UserError`.
        """
        verdict, position = await evaluate_all(
            as_guards(self.guard, capability='InputGuardrail'),
            ctx,
            prompt,
            check_replacement=_require_prompt_text,
        )
        match verdict.action:
            case 'allow':
                return
            case 'retry':
                raise UserError(
                    f'An InputGuardrail guard cannot return GuardrailResult.retry(); guard at position {position} '
                    'did. Retry applies to model output only.'
                )
            case 'approve':
                raise UserError(
                    'An InputGuardrail guard cannot return GuardrailResult.approve() -- approval applies to tool '
                    'calls only.'
                )
            case 'block':
                message = verdict.message or _DEFAULT_INPUT_BLOCK_MESSAGE
                trace_block(ctx, direction='input', message=message)
                raise SkipModelRequest(ModelResponse(parts=[TextPart(content=message)]))
            case 'replace':
                if self.parallel:
                    raise UserError(
                        'InputGuardrail(parallel=True) is incompatible with GuardrailResult.replace(): the model call has '
                        'already started with the original prompt. Use sequential mode for prompt redaction.'
                    )
                replacement = _require_prompt_text(verdict.replacement, position)
                if part is None:
                    raise UserError('InputGuardrail could not find a user prompt to redact in the request.')
                # A copy with the original's text puts the same secret in the run's history, so the
                # redaction goes there too. A copy whose text a capability changed for this request
                # says nothing about what the original should become, so the history is left alone.
                same_text = original is not None and original is not part and original.content == part.content
                history_part = original if same_text else None
                _replace_prompt(part, replacement)
                if history_part is not None:
                    _replace_prompt(history_part, replacement)
                trace_redaction(ctx, direction='input', replacement=replacement)
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
        part, original = _find_prompt_part(ctx, request_context.messages)
        content = ctx.prompt if part is None else part.content
        if content is None:
            return await handler(request_context)
        prompt = _prompt_text(content)
        if not self.parallel:
            await self._run_guard(ctx, part, original, prompt)
            return await handler(request_context)

        async def run_handler() -> ModelResponse:
            return await handler(request_context)

        guard_task: asyncio.Task[None] = asyncio.create_task(self._run_guard(ctx, part, original, prompt))
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

            # Run cancellation (`cancel_run()`, `anyio.fail_after`) can land here with an
            # enclosing anyio scope already cancelled, and that scope re-cancels its tasks on
            # every event-loop cycle -- each delivery either aborts the drain below outright
            # (abandoning the guard and handler tasks mid-unwind) or is forwarded through the
            # `gather` into both tasks, breaking any await their cleanup performs. Shield the
            # drain so both tasks fully unwind before the cancellation propagates. The shield
            # holds for anyio-scope cancellation; a raw second `Task.cancel()` can still
            # pierce it.
            #
            # Not an anyio task group, deliberately. The children being raw asyncio tasks is
            # what confines the re-delivery to this host task: anyio only re-cancels tasks its
            # scopes track, so each child sees exactly one `.cancel()` and unwinds cleanly. As
            # task-group children they would be re-cancelled inside their own cleanup on every
            # cycle -- the failure above, one level down. A task group also cannot shield only
            # its drain (`tg.cancel_scope.shield = True` covers the whole race, blocking
            # `cancel_run()` outright), and anyio >= 4 wraps child exceptions in
            # `BaseExceptionGroup`, which would keep the guard's `SkipModelRequest` from
            # reaching pydantic-ai unwrapped.
            with anyio.CancelScope(shield=True):
                await asyncio.gather(guard_task, handler_task, return_exceptions=True)


@dataclass
class OutputGuardrail(AbstractCapability[AgentDepsT]):
    """Validate the agent output as it is produced.

    The `guard` callable receives the output -- no automatic stringification, so
    a typed output arrives as the Pydantic model instance -- and returns one of
    the outcomes (see the module docstring): `allow` exposes the output,
    `block` raises [`OutputBlocked`][pydantic_ai_harness.guardrails.OutputBlocked],
    `replace` substitutes a sanitized output (redaction), and `retry` sends the
    output back to the model to try again.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import GuardrailResult, OutputGuardrail


    def no_pii(output: object) -> GuardrailResult:
        if 'SSN' in str(output):
            return GuardrailResult.retry('Do not include personal identifiers.')
        return GuardrailResult.allow()


    agent = Agent('openai:gpt-5.4', capabilities=[OutputGuardrail(guard=no_pii)])
    ```

    The guard runs as the output is processed, so `retry` reuses pydantic-ai's
    normal retry machinery and counts against the run's output-retry budget.
    Like `InputGuardrail`, the guard may take a
    [`RunContext`][pydantic_ai.tools.RunContext] as a first parameter; it is
    detected from the signature.

    Ordering: declares `position='outermost'` so the guard sees the final
    output after every inner capability has processed it, and `wrapped_by=
    [Instrumentation]` so an enclosing `Instrumentation` span always captures
    the guard's block/redact spans regardless of user list order.

    Streaming caveats. `retry` is supported with
    [`run()`][pydantic_ai.Agent.run] / `run_sync()` only -- pydantic-ai does not
    support output retries during [`run_stream()`][pydantic_ai.Agent.run_stream],
    where a `retry` verdict surfaces as
    [`UnexpectedModelBehavior`][pydantic_ai.exceptions.UnexpectedModelBehavior].
    The guard inspects only the final output, not partial chunks, so during a
    streamed run the caller may already have received partial output before a
    `block` or `replace` verdict is reached -- use `run()` when the output must
    be screened before any of it is exposed.
    """

    guard: OutputGuardrailFunc[AgentDepsT] | Sequence[OutputGuardrailFunc[AgentDepsT]]
    """One callable, or several run in order, deciding what happens to the output.

    In a chain the first `block` or `retry` ends it, and a `replace` substitutes
    the output the remaining guards see.
    """

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Exclude a callable policy from YAML and JSON agent specifications."""
        return None

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
        verdict, _ = await evaluate_all(as_guards(self.guard, capability='OutputGuardrail'), ctx, output)
        match verdict.action:
            case 'allow':
                return output
            case 'block':
                message = verdict.message or _DEFAULT_OUTPUT_BLOCK_MESSAGE
                trace_block(ctx, direction='output', message=message)
                raise OutputBlocked(message)
            case 'retry':
                raise ModelRetry(verdict.message or _DEFAULT_OUTPUT_RETRY_MESSAGE)
            case 'approve':
                raise UserError(
                    'An OutputGuardrail guard cannot return GuardrailResult.approve() -- approval applies to tool '
                    'calls only.'
                )
            case 'replace':
                trace_redaction(ctx, direction='output', replacement=verdict.replacement)
                return verdict.replacement
            case _:  # pragma: no cover - assert_never exhaustiveness guard
                assert_never(verdict.action)
