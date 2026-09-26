"""Verdict vocabulary shared by the input, output, and tool guardrails.

`GuardrailResult` is the single result type every guard returns, so the same five
outcomes read the same way on all three edges of a run. The helpers here
normalize a guard's return value, run a chain of guards over one value, and
record the non-`allow` outcomes as spans.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from typing_extensions import TypeIs, assert_never


class _Unset:
    """Sentinel for a `replacement` that was never supplied.

    `None` is a legitimate replacement: a tool may return `None`, and a result
    guard may want to sanitize a sensitive result down to it. A `None` default
    could not tell that apart from "this verdict carries no replacement".
    """

    def __repr__(self) -> str:
        return '<unset>'


_UNSET = _Unset()


@dataclass(frozen=True, kw_only=True)
class GuardrailResult:
    """The outcome a guard reports for the value it inspected.

    Construct one with the classmethods -- `GuardrailResult.allow()`,
    `GuardrailResult.block()`, `GuardrailResult.replace()`, `GuardrailResult.retry()`,
    `GuardrailResult.approve()` -- rather than the raw fields. A guard may also
    return a bare `bool`: `True` is `allow()`, `False` is `block()`.

    Not every outcome applies to every guard. `retry` and `approve` are
    rejected by [`InputGuardrail`][pydantic_ai_harness.InputGuardrail]; `approve` is
    rejected by [`OutputGuardrail`][pydantic_ai_harness.OutputGuardrail] and by the
    result stage of [`ToolGuardrail`][pydantic_ai_harness.ToolGuardrail]. Each raises
    `UserError` naming the guard and the outcome.
    """

    action: Literal['allow', 'block', 'replace', 'retry', 'approve']
    """What the capability should do with the inspected value."""

    message: str | None = None
    """For `block`, the refusal text. For `retry`, the instruction sent back to the model."""

    replacement: object = _UNSET
    """For `replace`, the value substituted for the inspected one.

    Defaults to a private sentinel rather than `None`, so `replace(None)` is a
    valid verdict.
    """

    def __post_init__(self) -> None:
        """Reject field combinations the outcome contract does not allow."""
        match self.action:
            case 'allow':
                if self.message is not None or self.replacement is not _UNSET:
                    raise UserError("GuardrailResult(action='allow') must not set `message` or `replacement`.")
            case 'approve':
                if self.message is not None or self.replacement is not _UNSET:
                    raise UserError("GuardrailResult(action='approve') must not set `message` or `replacement`.")
            case 'replace':
                if self.replacement is _UNSET:
                    raise UserError("GuardrailResult(action='replace') requires a `replacement` value.")
                if self.message is not None:
                    raise UserError("GuardrailResult(action='replace') must not set `message`.")
            case 'retry':
                if self.message is None:
                    raise UserError("GuardrailResult(action='retry') requires a `message`.")
                if self.replacement is not _UNSET:
                    raise UserError("GuardrailResult(action='retry') must not set `replacement`.")
            case 'block':
                # `message=None` is valid: a default is supplied at the use site. A
                # `replacement` is not: nothing reads it here, so accepting one would
                # silently discard a substitution the guard believed it had made.
                if self.replacement is not _UNSET:
                    raise UserError("GuardrailResult(action='block') must not set `replacement`.")
            case _:  # pragma: no cover - assert_never exhaustiveness guard
                assert_never(self.action)

    @classmethod
    def allow(cls) -> GuardrailResult:
        """Let the value through unchanged."""
        return cls(action='allow')

    @classmethod
    def block(cls, message: str | None = None) -> GuardrailResult:
        """Refuse the value. `message` is the refusal text; `None` uses a default."""
        return cls(action='block', message=message)

    @classmethod
    def replace(cls, value: object) -> GuardrailResult:
        """Substitute `value` for the inspected one and continue.

        For `InputGuardrail`, `value` is the replacement prompt text sent to the
        model. For `OutputGuardrail`, it is the agent output returned to the caller.
        For `ToolGuardrail`, it is the replacement tool arguments (a mapping) or the
        replacement tool result, depending on the stage. `None` is a valid
        replacement.
        """
        return cls(action='replace', replacement=value)

    @classmethod
    def retry(cls, message: str) -> GuardrailResult:
        """Send the value back to the model to try again.

        `message` is the instruction the model sees on the retry. Valid for
        `OutputGuardrail` and for both stages of `ToolGuardrail`.
        """
        return cls(action='retry', message=message)

    @classmethod
    def approve(cls) -> GuardrailResult:
        """Defer the tool call for human approval -- `ToolGuardrail` argument stage only.

        Raises [`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired], so
        the call surfaces in
        [`DeferredToolRequests`][pydantic_ai.DeferredToolRequests] and the
        application decides. To attach approval metadata, raise
        `ApprovalRequired(metadata=...)` from the guard directly.
        """
        return cls(action='approve')


GuardOutcome = bool | GuardrailResult
"""What a guard callable returns: a bare `bool` (`True` = allow), or a `GuardrailResult`."""


GuardCallable = Callable[..., object]
"""A guard as the helpers below handle it. `evaluate` normalizes whatever it returns.

The public `*GuardrailFunc` aliases pin down what a guard may return; by the time
one reaches here it has already been checked against that field type.
"""


def takes_ctx(func: Callable[..., object]) -> bool:
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


async def evaluate(
    guard: GuardCallable,
    ctx: RunContext[AgentDepsT],
    value: object,
) -> GuardrailResult:
    """Call `guard` (passing `ctx` when declared), await it, and normalize to `GuardrailResult`."""
    outcome = guard(ctx, value) if takes_ctx(guard) else guard(value)
    if inspect.isawaitable(outcome):
        outcome = await outcome
    if isinstance(outcome, GuardrailResult):
        return outcome
    return GuardrailResult.allow() if outcome else GuardrailResult.block()


def is_guard_chain(guard: object) -> TypeIs[Sequence[GuardCallable]]:
    """Whether a `guard` field holds several guards rather than one.

    Callability decides first: a guard that is itself a sequence would
    otherwise be taken apart and its elements invoked. `str` and `bytes` are
    excluded because they are sequences, so a string handed over by mistake
    would be split into characters and each one "called".

    A `Sequence` rather than any iterable, matching the declared field type. A
    set has no order for a chain to run in, and a one-shot iterator is spent
    after the first run -- both are refused here by name rather than reordering
    the chain or emptying it between runs.

    `TypeIs` rather than `TypeGuard` so the negative branch narrows too, which
    is what lets the single-guard path stay cast-free.
    """
    return not callable(guard) and not isinstance(guard, (str, bytes)) and isinstance(guard, Sequence)


def as_guards(guard: object, *, capability: str) -> tuple[GuardCallable, ...]:
    """Normalize a `guard` field to a tuple, refusing shapes that would misbehave later.

    Everything wrong here is caught by name. Left alone, a non-callable reaches
    `inspect.signature` and surfaces as a bare `TypeError` about an object being
    uncallable, with nothing to say which guard or which position.
    """
    if not is_guard_chain(guard):
        if not callable(guard):
            raise UserError(f'{capability} needs a guard callable, or a sequence of them; got {type(guard).__name__}.')
        return (guard,)
    guards = tuple(guard)
    if not guards:
        raise UserError(f'{capability} was given an empty sequence of guards, so it would inspect nothing.')
    for position, entry in enumerate(guards):
        if not callable(entry):
            raise UserError(
                f'{capability} needs a guard callable at position {position} of its guard sequence; '
                f'got {type(entry).__name__}.'
            )
    return guards


async def evaluate_all(
    guards: Sequence[GuardCallable],
    ctx: RunContext[AgentDepsT],
    value: object,
    *,
    check_replacement: Callable[[object, int], None] | None = None,
) -> tuple[GuardrailResult, int]:
    """Run each guard in order over the value, threading replacements through.

    `allow` moves on to the next guard. `replace` substitutes the value the rest
    of the chain inspects, so a redactor followed by a checker sees the redacted
    text. Every other outcome ends the chain, since none of them leaves a value
    for a later guard to judge. When every guard allowed and at least one
    replaced, the accumulated replacement is the verdict.

    `check_replacement` validates each substitution as it is made rather than
    only at the end, so a guard returning the wrong type is named instead of
    handing something unusable to the next one.

    Returns the verdict and the position of the guard that produced it, so an
    error about an inapplicable verdict can say which guard to go and look at.
    """
    replaced = False
    for position, guard in enumerate(guards):
        verdict = await evaluate(guard, ctx, value)
        if verdict.action == 'allow':
            continue
        if verdict.action == 'replace':
            if check_replacement is not None:
                check_replacement(verdict.replacement, position)
            value = verdict.replacement
            replaced = True
            continue
        return verdict, position
    return (GuardrailResult.replace(value) if replaced else GuardrailResult.allow()), len(guards) - 1


def direction_attributes(direction: str, action: str, tool_name: str | None) -> dict[str, str]:
    """Attributes every guardrail span carries, regardless of content settings."""
    attributes = {'guardrail.direction': direction, 'guardrail.action': action}
    if tool_name is not None:
        attributes['guardrail.tool'] = tool_name
    return attributes


def trace_block(
    ctx: RunContext[AgentDepsT],
    *,
    direction: str,
    message: str,
    tool_name: str | None = None,
) -> None:
    """Record a zero-duration span marking a guardrail refusal.

    The refusal message is attached only when `ctx.trace_include_content` is
    set -- it can quote sensitive content from the guarded value, and ops
    audiences are broader than the user who sees the refusal text.
    """
    attributes = direction_attributes(direction, 'block', tool_name)
    if ctx.trace_include_content:
        attributes['guardrail.message'] = message
    ctx.tracer.start_span(f'guardrail blocked {direction}', attributes=attributes).end()


def trace_redaction(
    ctx: RunContext[AgentDepsT],
    *,
    direction: str,
    original: object,
    replacement: object,
    tool_name: str | None = None,
) -> None:
    """Record a zero-duration span marking a guardrail redaction.

    The original and replacement values are attached only when
    `ctx.trace_include_content` is set, since a redacted value is often the
    sensitive content the guard exists to keep out of traces.
    """
    attributes = direction_attributes(direction, 'replace', tool_name)
    if ctx.trace_include_content:
        attributes['guardrail.original'] = str(original)
        attributes['guardrail.replacement'] = str(replacement)
    ctx.tracer.start_span(f'guardrail redacted {direction}', attributes=attributes).end()


def trace_approval(
    ctx: RunContext[AgentDepsT],
    *,
    direction: str,
    tool_name: str,
    tool_call_id: str,
    args: object,
) -> None:
    """Record a zero-duration span marking a tool call deferred for approval.

    Deferring means the tool never executes, so no `execute_tool` span is ever
    created and this is the only record of what was asked for. It therefore
    carries the call id, to correlate with the `DeferredToolRequests` an
    application answers, and the arguments when `trace_include_content` allows
    it -- the one case where an operator most needs to see them.
    """
    attributes = direction_attributes(direction, 'approve', tool_name)
    attributes['guardrail.tool_call_id'] = tool_call_id
    if ctx.trace_include_content:
        attributes['guardrail.arguments'] = str(args)
    ctx.tracer.start_span(f'guardrail deferred {direction}', attributes=attributes).end()
