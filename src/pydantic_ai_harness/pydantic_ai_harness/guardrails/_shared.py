"""Verdict vocabulary shared by the input, output, and tool guardrails.

`GuardrailResult` is the single result type every guard returns, so the same five
outcomes read the same way on all three edges of a run. The helpers here
normalize a guard's return value, run a chain of guards over one value, and
record the non-`allow` outcomes as spans.
"""

from __future__ import annotations

import functools
import inspect
import re
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import UnionType
from typing import Annotated, Any, ForwardRef, Literal, Union, assert_never, get_args, get_origin

from typing_extensions import TypeIs

from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai_harness._warn import HarnessDeprecationWarning


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


TextDetector = Callable[[str], GuardrailResult]
"""A check over text. Adapt one with `for_text` or `for_tool_result_text` at non-text guard boundaries."""


GuardOutcome = bool | GuardrailResult
"""What a guard callable returns: a bare `bool` (`True` = allow), or a `GuardrailResult`."""


GuardCallable = Callable[..., object]
"""A guard as the helpers below handle it.

Deliberately loose: whether a guard takes `RunContext` is decided from its signature at call
time, which no static callable type can express, and `evaluate` checks what it returns at
runtime. The public `*GuardrailFunc` aliases carry the precise types for users.
"""


def takes_ctx(func: Callable[..., object]) -> bool:
    """Return `True` when `func` declares a leading `RunContext` parameter.

    Detected from the positional parameters: a guard always takes the guarded
    value, so a second required positional parameter means it also wants the
    run context, whether or not it is annotated, as does a `*args` parameter
    alongside another parameter.
    Keyword-only parameters are the guard's own options and never count on their own. A
    second positional parameter with a default is read from the first
    parameter's annotation: `RunContext` (also in a union or `Annotated`)
    means the run context, and any other type means the guarded value. A first
    parameter with no annotation, `Any`, `object`, or a type that cannot be
    resolved still gets the run context, as it always has, with a
    `HarnessDeprecationWarning`, unless the unresolved annotation names
    `RunContext`, which needs no warning. A callable whose signature cannot be
    introspected is treated as taking the value only.
    """
    try:
        parameters = inspect.signature(func).parameters.values()
    except ValueError:  # pragma: no cover - callable without an introspectable signature
        return False
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) < 2:
        has_args = any(parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters)
        return has_args and len(parameters) > 1
    if positional[1].default is inspect.Parameter.empty:
        return True
    annotation = _resolve(func, positional[0].annotation)
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    either = annotation is inspect.Parameter.empty or annotation is Any or annotation is object
    unresolved = isinstance(annotation, str) or _has_forward_ref(annotation)
    if either or (unresolved and not _is_run_context(annotation)):
        warnings.warn(
            f'Guard {_guard_name(func)} has a second positional parameter with a default and a first parameter '
            'that is unannotated, `Any`, `object` or unresolvable, so it is passed the RunContext first, as before. '
            "Annotate the first parameter, with `RunContext[...]` or with the guarded value's type: a future "
            'release will pass the guarded value to it.',
            HarnessDeprecationWarning,
            stacklevel=2,
        )
        return True
    return _is_run_context(annotation)


def _resolve(func: Callable[..., object], annotation: object) -> object:
    """A string annotation (under `from __future__ import annotations`) evaluated in `func`'s module.

    Only this one annotation is evaluated, so a type elsewhere in the signature that is imported only
    for type checking does not stop it resolving. When it cannot be evaluated itself, the string is
    kept rather than turning a working guard into an error.
    """
    if not isinstance(annotation, str):
        return annotation
    target = inspect.unwrap(func.func if isinstance(func, functools.partial) else func)
    # A callable instance's annotations come from its class's `__call__`.
    namespace = getattr(target, '__globals__', None) or getattr(type(target).__call__, '__globals__', {})
    try:
        return eval(annotation, namespace)  # same as `typing.get_type_hints`
    except Exception:
        return annotation


def _is_run_context(annotation: object) -> bool:
    """Whether a type annotation is `RunContext`, a subscripted `RunContext`, or one in a union or `Annotated`.

    A string annotation that could not be resolved is matched by name.
    """
    if isinstance(annotation, str):
        return _names_run_context(annotation)
    if isinstance(annotation, ForwardRef):
        return _names_run_context(annotation.__forward_arg__)
    if annotation is RunContext:
        return True
    origin = get_origin(annotation)
    if origin is RunContext:
        return True
    if origin is Annotated:
        return _is_run_context(get_args(annotation)[0])
    if origin is Union or origin is UnionType:
        return any(_is_run_context(arg) for arg in get_args(annotation) if arg is not type(None))
    return False


def _has_forward_ref(annotation: object) -> bool:
    """Whether a type annotation holds a quoted name that was never resolved, as in `Optional['Deps']`."""
    return isinstance(annotation, ForwardRef) or any(_has_forward_ref(arg) for arg in get_args(annotation))


def _names_run_context(annotation: str) -> bool:
    """Whether a string annotation names `RunContext`, qualified or not, as in `'RunContext[D] | None'`."""
    return re.search(r'(?<!\w)RunContext\b', annotation) is not None


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
    if not isinstance(outcome, bool):
        # Still read for truthiness, as before, but flagged: a classifier label such as `'unsafe'`
        # is truthy and lets the value through.
        warnings.warn(
            f'A guard should return True, False, or a GuardrailResult; {_guard_name(guard)} returned '
            f"{type(outcome).__name__}, which is read for truthiness, so a truthy label such as 'unsafe' "
            'allows the value. A future release will raise UserError instead. Return bool(...) or a '
            'GuardrailResult.',
            HarnessDeprecationWarning,
            stacklevel=2,
        )
    return GuardrailResult.allow() if outcome else GuardrailResult.block()


def _guard_name(guard: GuardCallable) -> str:
    """A guard's name for a message, without its `repr`, which can show bound arguments such as keys."""
    target: object = guard.func if isinstance(guard, functools.partial) else guard
    return getattr(target, '__qualname__', None) or type(target).__name__


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
    check_replacement: Callable[[object, int], object] | None = None,
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
    replacement: object,
    tool_name: str | None = None,
) -> None:
    """Record a zero-duration span marking a guardrail redaction.

    The replacement value is attached only when `ctx.trace_include_content` is
    set. The original value never is: it is the content the guard removed, and
    recording it would put a redacted secret into the trace that core's own
    spans, which see the rewritten value, keep out.
    """
    attributes = direction_attributes(direction, 'replace', tool_name)
    if ctx.trace_include_content:
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
