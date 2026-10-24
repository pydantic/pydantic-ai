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
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, assert_never

import anyio

from pydantic_ai.capabilities import (
    AbstractCapability,
    CapabilityOrdering,
    Instrumentation,
    WrapModelRequestHandler,
    WrapperCapability,
)
from pydantic_ai.exceptions import ModelRetry, SkipModelRequest, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserContent, UserPromptPart
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai_harness._background_completion import get_background_completion
from pydantic_ai_harness._task_report import is_task_report
from pydantic_ai_harness.guardrails._exceptions import OutputBlocked
from pydantic_ai_harness.guardrails._shared import (
    GuardOutcome,
    GuardrailResult,
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
) -> tuple[list[UserPromptPart], UserPromptPart | None]:
    """The user's prompt for this run as it appears in this request, and its history original if separate.

    The parts are usually one. The original is set only when it is a different object from the
    parts sent to the model.

    The prompt is the most recent `UserPromptPart` in the run's history, `ctx.messages`. `messages`
    is this request's copy of the history, where an outer capability can add a `UserPromptPart` for
    this request only (`SystemReminders`, `Planning` and `Memory` append one to the last request).
    That part is not the user's prompt, so the prompt is found by identity: the guard neither
    screens nor overwrites the added part. A prompt an outer capability rewrote in place is still
    the same object, so the guard reads the rewritten text, and there is no separate original.

    When a capability replaced the prompt with a copy, the copies are found by `_find_copies` in
    the most recent message holding a part with the prompt's timestamp, wherever other parts or
    messages were put around it, so neither a message appended for this request nor history
    trimmed from the start moves them; several copies with the prompt's text are all returned, so
    a redaction reaches each of them. The copies are returned with the history part they were made
    from, so a redaction can reach it too. Only when no part can be told to be the copy, because
    the request was rebuilt or several parts carry the prompt's timestamp and none has its text, is
    the most recent part in `messages` used, with no original: nothing ties that part to the
    history.
    """
    skip_task_reports = ctx.prompt is None
    latest = _latest_prompt_parts(ctx.messages, skip_task_reports=skip_task_reports)
    if not latest:
        return _latest_prompt_parts(messages, skip_task_reports=skip_task_reports), None
    prompt = latest[0]
    if any(part is prompt for request in messages for part in request.parts):
        return [prompt], None
    holder = next(
        (
            message
            for message in reversed(messages)
            if any(isinstance(part, UserPromptPart) and part.timestamp == prompt.timestamp for part in message.parts)
        ),
        None,
    )
    copies = [] if holder is None else _find_copies(prompt, holder.parts)
    if copies:
        return copies, prompt
    return _latest_prompt_parts(messages, skip_task_reports=skip_task_reports), None


def _find_copies(prompt: UserPromptPart, parts: Sequence[object]) -> list[UserPromptPart]:
    """The copies of `prompt` among `parts`, told apart by their timestamp, or none.

    A copy made with `dataclasses.replace` keeps the prompt's timestamp, even when its text was
    changed, while a part an outer capability creates for this request gets its own, so neither
    the copy's position nor its text decides. When several parts carry the timestamp, those with
    the prompt's text are taken, all of them, since the guard's verdict on that text applies to
    each; a changed copy beside them is not, as the guard did not screen its text. When none of
    them has the prompt's text, the copy cannot be told apart.
    """
    candidates = [part for part in parts if isinstance(part, UserPromptPart) and part.timestamp == prompt.timestamp]
    same_text = [part for part in candidates if part.content == prompt.content]
    if same_text:
        return same_text
    return candidates if len(candidates) == 1 else []


def _history_prompt_parts(ctx: RunContext[AgentDepsT]) -> int:
    """How many `UserPromptPart`s the latest request in the run's history holding one carries.

    A run resuming from a request reports all of them combined as `ctx.prompt`. A part an outer
    capability adds for this request only is not in the history, so it is not counted.
    """
    return next(
        (
            count
            for message in reversed(ctx.messages)
            if (count := sum(isinstance(part, UserPromptPart) for part in message.parts))
        ),
        0,
    )


def _latest_prompt_parts(messages: Sequence[ModelMessage], *, skip_task_reports: bool) -> list[UserPromptPart]:
    """The most recent `UserPromptPart` in `messages`, as a list of at most one.

    `skip_task_reports` is set only for a run without a prompt of its own. `ctx.prompt` is always
    guarded, whatever metadata it carries, including a prompt taken from a request that ends
    `message_history`. Earlier history is trusted as the caller's own (untrusted client history
    goes through a UI adapter first), so a part there that carries a `SubAgents` task report's
    metadata is taken for one: a child agent's output, not the prompt.
    """
    for message in reversed(messages):
        for part in reversed(message.parts):
            if isinstance(part, UserPromptPart) and not (skip_task_reports and is_task_report(part)):
                return [part]
    return []


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
    the model and the original in the run's message history, so a redacted
    secret is not retained or replayed on a later step. When an outer
    capability sent the model a copy of the prompt part with changed text, the
    guard is called again on the original's own text, so a guard with side
    effects or a per-call cost (an LLM classifier) runs twice for that request.
    Only a redaction from that call is written to the history, so the copy's
    request-only text never reaches it; `allow` leaves the original as it was,
    and `block` blocks the run before anything is rewritten. A request rebuilt
    with a new prompt part is not tied to the history, so only the part sent to
    the model is rewritten. A later guard in a chain that blocks ends the chain
    before any rewrite happens.

    `retry` is not valid for an input guardrail, and neither is `replace` on a
    multimodal prompt, including the history's original of a text copy: the
    guard sees the whole prompt rendered as text, and writing one string back
    over a prompt built from several parts would drop the attached ones, so it
    raises
    [`UserError`][pydantic_ai.exceptions.UserError] instead. The same holds for
    a run resuming from a request with several prompt parts, which the guard
    screens together, as `ctx.prompt` reports them.

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
    only after the model has responded, those tokens were still spent. Tool
    calls still wait for the verdict, including the nested ones `CodeMode`
    starts while the model streams. Prefer sequential for fast local guards.
    `replace` (redaction) is incompatible with `parallel=True`, since the
    model call has already started with the original prompt.

    Scope: the guard runs once per run, on the first model request (twice
    when it redacts a changed copy of the prompt an outer capability sent the
    model, see above), and evaluates the most recent user prompt in the run's
    history. Subsequent model requests in the same run (e.g. after tool calls)
    are not re-checked, since the user input has not changed. A run resumed
    without a new prompt (to answer deferred tool calls, say) has no new user
    input either, so its first request screens the most recent prompt in the
    history again.

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

    _parallel_verdict: asyncio.Task[None] | None = field(default=None, init=False, repr=False)

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Exclude a callable policy from YAML and JSON agent specifications."""
        return None

    def get_ordering(self) -> CapabilityOrdering:
        """Sit innermost so message-morphing capabilities run first and the guard sees the final prompt."""
        return CapabilityOrdering(position='innermost')

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> InputGuardrail[AgentDepsT]:
        """Return a fresh instance for a parallel guard so concurrent runs don't share a pending verdict."""
        return replace(self) if self.parallel else self

    async def _run_guard(
        self,
        ctx: RunContext[AgentDepsT],
        parts: Sequence[UserPromptPart],
        original: UserPromptPart | None,
        prompt: str,
        *,
        prompt_parts: int,
    ) -> None:
        """Evaluate the guard and act on its verdict.

        `allow` returns; `block` raises `SkipModelRequest`; `replace` rewrites
        the prompt `parts` sent to the model and their `original` in the run's
        history when there is a separate one (see `_history_replacement`);
        `retry`, `replace` under `parallel=True`, and `replace` on a prompt of
        more than one part (`prompt_parts`) raise `UserError`. Every
        evaluation happens before anything is rewritten, and the redaction spans
        are recorded once every rewrite has succeeded.
        """
        verdict, position = await self._screen(ctx, prompt)
        if verdict.action == 'allow':
            return
        if self.parallel:
            raise UserError(
                'InputGuardrail(parallel=True) is incompatible with GuardrailResult.replace(): the model call has '
                'already started with the original prompt. Use sequential mode for prompt redaction.'
            )
        replacement = _require_prompt_text(verdict.replacement, position)
        if prompt_parts > 1:
            raise UserError(
                f'An InputGuardrail returned GuardrailResult.replace() for a prompt made of {prompt_parts} '
                'parts. The guard saw them as one text, so one replacement cannot be written back over '
                'them without dropping or merging parts. Return GuardrailResult.allow() or '
                'GuardrailResult.block() for such prompts, or guard the output instead.'
            )
        if not parts:
            raise UserError('InputGuardrail could not find a user prompt to redact in the request.')
        # A copy with the original's text gets the same replacement, under the same span.
        same_text = original is not None and original.content == parts[0].content
        own_replacement = None
        if original is not None and not same_text:
            own_replacement = await self._history_replacement(ctx, original)
        for part in parts:
            _replace_prompt(part, replacement)
        if original is not None and (same_text or own_replacement is not None):
            _replace_prompt(original, replacement if own_replacement is None else own_replacement)
        trace_redaction(ctx, direction='input', replacement=replacement)
        if own_replacement is not None:
            trace_redaction(ctx, direction='input', replacement=own_replacement)

    async def _screen(self, ctx: RunContext[AgentDepsT], prompt: str) -> tuple[GuardrailResult, int]:
        """Run the guard chain on `prompt`, returning an `allow` or `replace` verdict and raising for the others.

        `block` raises `SkipModelRequest`; `retry` and `approve` raise `UserError`.
        """
        verdict, position = await evaluate_all(
            as_guards(self.guard, capability='InputGuardrail'),
            ctx,
            prompt,
            check_replacement=_require_prompt_text,
        )
        match verdict.action:
            case 'allow' | 'replace':
                return verdict, position
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
            case _:  # pragma: no cover - assert_never exhaustiveness guard
                assert_never(verdict.action)

    async def _history_replacement(self, ctx: RunContext[AgentDepsT], original: UserPromptPart) -> str | None:
        """The redaction of the history's `original` of a prompt the model was sent a changed copy of.

        The history is what later requests in the run, and any stored conversation, replay, so a
        secret left there reaches the model on the next step. A copy whose text a capability
        changed for this request says nothing about what the original should become, so the guard
        screens the original on its own text, as it screened the copy: a redaction it returns is
        written there, so the copy's request-only text never reaches the history, `allow` leaves
        the original as it is, and `block` blocks the run before anything is rewritten.

        Returns `None` when the original is left as it is.
        """
        verdict, position = await self._screen(ctx, _prompt_text(original.content))
        if verdict.action == 'allow':
            return None
        return _require_prompt_text(verdict.replacement, position)

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
        parts, original = _find_prompt_part(ctx, request_context.messages)
        content = parts[0].content if parts else ctx.prompt
        prompt_parts = _history_prompt_parts(ctx) if ctx.prompt is not None else 0
        if prompt_parts > 1:
            # A run resuming from a request with several prompt parts reports them combined as
            # `ctx.prompt`; the guard screens all of them, not just the last.
            content = ctx.prompt
        if content is None:
            return await handler(request_context)
        prompt = _prompt_text(content)
        if not self.parallel:
            await self._run_guard(ctx, parts, original, prompt, prompt_parts=prompt_parts)
            return await handler(request_context)

        async def run_handler() -> ModelResponse:
            return await handler(request_context)

        guard_task: asyncio.Task[None] = asyncio.create_task(
            self._run_guard(ctx, parts, original, prompt, prompt_parts=prompt_parts)
        )
        self._parallel_verdict = guard_task
        handler_task: asyncio.Task[ModelResponse] = asyncio.create_task(run_handler())
        try:
            # The verdict comes first, even when the handler finished or raised before it: a
            # `ModelRetry`, a `SkipModelRequest` or an error an outer capability recovers from would
            # otherwise continue the run without one. `asyncio.wait` rather than `await`, so a
            # cancellation of this call reaches the guard only through the drain below.
            await asyncio.wait({guard_task})
            await guard_task
            return await handler_task
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


async def input_guardrails_allow(ctx: RunContext[object]) -> bool:
    """Wait for this run's pending parallel `InputGuardrail` verdicts and report whether they allowed the prompt.

    Tool calls normally run only after `wrap_model_request` has awaited a parallel guard's verdict.
    Work that runs tools while the model is still streaming awaits the same verdict here first.
    """
    for capability in ctx.capabilities.values():
        # A guard returned from a `CapabilityFunc` or given `prefix_tools()` is registered as its wrapper.
        while isinstance(capability, WrapperCapability):
            capability = capability.wrapped
        verdict = capability._parallel_verdict if isinstance(capability, InputGuardrail) else None  # pyright: ignore[reportPrivateUsage]
        if verdict is None:
            continue
        if not verdict.done():
            # `asyncio.wait` rather than `await`: cancelling this waiter must not cancel the guard.
            await asyncio.wait({verdict})
        # A guard cancelled before it decided, as when the stream fails while a call waits, never
        # allowed the prompt.
        if verdict.cancelled() or verdict.exception() is not None:
            return False
    return True


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
