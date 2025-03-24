from __future__ import annotations as _annotations

from collections.abc import Mapping
from datetime import timedelta
from typing import Any, cast

from pydantic_ai import models

from ..otel.span_tree import SpanQuery, as_predicate
from .context import EvaluatorContext
from .spec import EvaluatorFunction, EvaluatorOutputValue, EvaluatorResult

__all__ = (
    'equals',
    'equals_expected',
    'contains',
    'is_instance',
    'max_duration',
    'llm_judge',
    'span_query',
    'python',
)


async def equals(ctx: EvaluatorContext[object, object, object], value: Any) -> bool:
    """Check if the output exactly equals the provided value."""
    return ctx.output == value


async def equals_expected(ctx: EvaluatorContext[object, object, object]) -> Mapping[str, bool]:
    """Check if the output exactly equals the expected output."""
    if ctx.expected_output is None:
        return {}  # Only compare if expected output is provided

    return {'equals_expected': ctx.output == ctx.expected_output}


_MAX_REASON_LENGTH = 500


async def contains(  # noqa C901
    ctx: EvaluatorContext[object, object, object], value: Any, case_sensitive: bool = True, as_strings: bool = False
) -> EvaluatorResult:
    """Check if the output contains the expected output.

    For strings, checks if expected_output is a substring of output.
    For lists/tuples, checks if expected_output is in output.
    For dicts, checks if all key-value pairs in expected_output are in output.

    Note: case_sensitive only applies when both the value and output are strings.
    """
    # Convert objects to strings if requested
    failure_reason: str | None = None
    as_strings = as_strings or (isinstance(value, str) and isinstance(ctx.output, str))
    if as_strings:
        output_str = str(ctx.output)
        expected_str = str(value)

        if not case_sensitive:
            output_str = output_str.lower()
            expected_str = expected_str.lower()

        failure_reason: str | None = None
        if expected_str not in output_str:
            failure_reason = f'Output string {output_str!r} does not contain expected string {expected_str!r}'
            if (
                len(failure_reason) > _MAX_REASON_LENGTH
            ):  # Only include the strings in the reason if it doesn't make it too long
                failure_reason = 'Output string does not contain expected string'
        return EvaluatorResult(value=failure_reason is None, reason=failure_reason)

    try:
        # Handle different collection types
        if isinstance(ctx.output, dict):
            if isinstance(value, dict):
                # Cast to Any to avoid type checking issues
                output_dict = cast(dict[Any, Any], ctx.output)  # pyright: ignore[reportUnknownMemberType]
                expected_dict = cast(dict[Any, Any], value)
                for k in expected_dict:
                    if k not in output_dict:
                        failure_reason = f'Output dictionary does not contain expected key {k!r}'
                        break
                    elif output_dict[k] != expected_dict[k]:
                        failure_reason = f'Output dictionary has different value for key {k!r}: {output_dict[k]!r} != {expected_dict[k]!r}'
                        if (
                            len(failure_reason) > _MAX_REASON_LENGTH
                        ):  # Only include the strings in the reason if it doesn't make it too long
                            failure_reason = f'Output dictionary has different value for key {k!r}'
                        break
            else:
                if value not in ctx.output:  # pyright: ignore[reportUnknownMemberType]
                    failure_reason = f'Output {ctx.output!r} does not contain expected item as key'  # pyright: ignore[reportUnknownMemberType]
                    if len(failure_reason) > _MAX_REASON_LENGTH:
                        failure_reason = 'Output does not contain expected item as key'
        elif value not in ctx.output:  # pyright: ignore[reportOperatorIssue]  # will be handled by except block
            failure_reason = f'Output {ctx.output!r} does not contain expected item {value!r}'
            if len(failure_reason) > _MAX_REASON_LENGTH:
                failure_reason = 'Output does not contain expected item'
    except (TypeError, ValueError) as e:
        failure_reason = f'Containment check failed: {e}'

    return EvaluatorResult(value=failure_reason is None, reason=failure_reason)


async def is_instance(ctx: EvaluatorContext[object, object, object], type_name: str) -> EvaluatorResult:
    """Check if the output is an instance of a type with the given name."""
    output = ctx.output
    for cls in type(output).__mro__:
        if cls.__name__ == type_name or cls.__qualname__ == type_name:
            return EvaluatorResult(value=True)

    reason = f'output is of type {type(output).__name__}'
    if type(output).__qualname__ != type(output).__name__:
        reason += f' (qualname: {type(output).__qualname__})'
    return EvaluatorResult(value=False, reason=reason)


async def max_duration(ctx: EvaluatorContext[object, object, object], seconds: float | timedelta) -> bool:
    """Check if the execution time is under the specified maximum."""
    duration = timedelta(seconds=ctx.duration)
    if not isinstance(seconds, timedelta):
        seconds = timedelta(seconds=seconds)
    return duration <= seconds


async def llm_judge(
    ctx: EvaluatorContext[object, object, object],
    rubric: str,
    model: models.KnownModelName = 'gpt-4o',
    include_input: bool = False,
) -> EvaluatorResult:
    """Judge whether the output of a language model meets the criteria of a provided rubric."""
    if include_input:
        from .llm_as_a_judge import judge_input_output

        grading_output = await judge_input_output(ctx.inputs, ctx.output, rubric, model)
    else:
        from .llm_as_a_judge import judge_output

        grading_output = await judge_output(ctx.output, rubric, model)
    return EvaluatorResult(value=grading_output.pass_, reason=grading_output.reason)


async def span_query(
    ctx: EvaluatorContext[object, object, object],
    query: SpanQuery,
) -> bool:
    """Check if the span tree contains a span with the specified name."""
    return ctx.span_tree.find_first(as_predicate(query)) is not None


# TODO: Consider moving this to docs rather than providing it with the library, given the security implications
async def python(
    ctx: EvaluatorContext[object, object, object], expression: str
) -> Mapping[str, EvaluatorOutputValue | EvaluatorResult]:
    """Return the result of evaluating a Python expression, to be used as an evaluator's output.

    ***WARNING***: this evaluator runs arbitrary Python code, so you should ***NEVER*** use it with untrusted inputs.
    """
    # Evaluate the condition, exposing access to the evaluator context as `ctx`.
    return eval(expression, {'ctx': ctx})


DEFAULT_EVALUATORS: tuple[EvaluatorFunction[object, object, object], ...] = (
    equals,
    equals_expected,
    contains,
    is_instance,
    max_duration,
    llm_judge,
    span_query,
    # python,  # not included by default for security reasons
)
