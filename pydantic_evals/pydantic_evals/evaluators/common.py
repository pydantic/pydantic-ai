from __future__ import annotations as _annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, cast

from pydantic_ai import models

from ..otel.span_tree import SpanQuery as SpanNodeQuery, as_predicate
from .context import EvaluatorContext
from .evaluator import EvaluationReason, Evaluator, EvaluatorOutput

__all__ = (
    'Equals',
    'EqualsExpected',
    'Contains',
    'IsInstance',
    'MaxDuration',
    'LlmJudge',
    'SpanQuery',
    'Python',
)


@dataclass
class Equals(Evaluator[object, object, object]):
    """Check if the output exactly equals the provided value."""

    value: Any

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluatorOutput:
        return ctx.output == self.value


@dataclass
class EqualsExpected(Evaluator[object, object, object]):
    """Check if the output exactly equals the expected output."""

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluatorOutput:
        if ctx.expected_output is None:
            return {}  # Only compare if expected output is provided
        return ctx.output == ctx.expected_output


_MAX_REASON_LENGTH = 500


@dataclass
class Contains(Evaluator[object, object, object]):
    """Check if the output contains the expected output.

    For strings, checks if expected_output is a substring of output.
    For lists/tuples, checks if expected_output is in output.
    For dicts, checks if all key-value pairs in expected_output are in output.

    Note: case_sensitive only applies when both the value and output are strings.
    """

    value: Any
    case_sensitive: bool = True
    as_strings: bool = False

    def evaluate(  # noqa C901
        self,
        ctx: EvaluatorContext[object, object, object],
    ) -> EvaluatorOutput:
        # Convert objects to strings if requested
        failure_reason: str | None = None
        as_strings = self.as_strings or (isinstance(self.value, str) and isinstance(ctx.output, str))
        if as_strings:
            output_str = str(ctx.output)
            expected_str = str(self.value)

            if not self.case_sensitive:
                output_str = output_str.lower()
                expected_str = expected_str.lower()

            failure_reason: str | None = None
            if expected_str not in output_str:
                failure_reason = f'Output string {output_str!r} does not contain expected string {expected_str!r}'
                if (
                    len(failure_reason) > _MAX_REASON_LENGTH
                ):  # Only include the strings in the reason if it doesn't make it too long
                    failure_reason = 'Output string does not contain expected string'
            return EvaluationReason(value=failure_reason is None, reason=failure_reason)

        try:
            # Handle different collection types
            if isinstance(ctx.output, dict):
                if isinstance(self.value, dict):
                    # Cast to Any to avoid type checking issues
                    output_dict = cast(dict[Any, Any], ctx.output)  # pyright: ignore[reportUnknownMemberType]
                    expected_dict = cast(dict[Any, Any], self.value)  # pyright: ignore[reportUnknownMemberType]
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
                    if self.value not in ctx.output:  # pyright: ignore[reportUnknownMemberType]
                        failure_reason = f'Output {ctx.output!r} does not contain expected item as key'  # pyright: ignore[reportUnknownMemberType]
                        if len(failure_reason) > _MAX_REASON_LENGTH:
                            failure_reason = 'Output does not contain expected item as key'
            elif self.value not in ctx.output:  # pyright: ignore[reportOperatorIssue]  # will be handled by except block
                failure_reason = f'Output {ctx.output!r} does not contain expected item {self.value!r}'
                if len(failure_reason) > _MAX_REASON_LENGTH:
                    failure_reason = 'Output does not contain expected item'
        except (TypeError, ValueError) as e:
            failure_reason = f'Containment check failed: {e}'

        return EvaluationReason(value=failure_reason is None, reason=failure_reason)


@dataclass
class IsInstance(Evaluator[object, object, object]):
    """Check if the output is an instance of a type with the given name."""

    type_name: str

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluatorOutput:
        output = ctx.output
        for cls in type(output).__mro__:
            if cls.__name__ == self.type_name or cls.__qualname__ == self.type_name:
                return EvaluationReason(value=True)

        reason = f'output is of type {type(output).__name__}'
        if type(output).__qualname__ != type(output).__name__:
            reason += f' (qualname: {type(output).__qualname__})'
        return EvaluationReason(value=False, reason=reason)


@dataclass
class MaxDuration(Evaluator[object, object, object]):
    """Check if the execution time is under the specified maximum."""

    seconds: float | timedelta

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluatorOutput:
        duration = timedelta(seconds=ctx.duration)
        seconds = self.seconds
        if not isinstance(seconds, timedelta):
            seconds = timedelta(seconds=seconds)
        return duration <= seconds


@dataclass
class LlmJudge(Evaluator[object, object, object]):
    """Judge whether the output of a language model meets the criteria of a provided rubric."""

    rubric: str
    model: models.KnownModelName = 'gpt-4o'
    include_input: bool = False

    async def evaluate_async(
        self,
        ctx: EvaluatorContext[object, object, object],
    ) -> EvaluatorOutput:
        if self.include_input:
            from .llm_as_a_judge import judge_input_output

            grading_output = await judge_input_output(ctx.inputs, ctx.output, self.rubric, self.model)
        else:
            from .llm_as_a_judge import judge_output

            grading_output = await judge_output(ctx.output, self.rubric, self.model)
        return EvaluationReason(value=grading_output.pass_, reason=grading_output.reason)


@dataclass
class SpanQuery(Evaluator[object, object, object]):
    """Check if the span tree contains a span with the specified name."""

    query: SpanNodeQuery

    def evaluate(
        self,
        ctx: EvaluatorContext[object, object, object],
    ) -> EvaluatorOutput:
        return ctx.span_tree.find_first(as_predicate(self.query)) is not None


# TODO: Consider moving this to docs rather than providing it with the library, given the security implications
@dataclass
class Python(Evaluator[object, object, object]):
    """The output of this evaluator is the result of evaluating the provided Python expression.

    ***WARNING***: this evaluator runs arbitrary Python code, so you should ***NEVER*** use it with untrusted inputs.
    """

    expression: str

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluatorOutput:
        # Evaluate the condition, exposing access to the evaluator context as `ctx`.
        return eval(self.expression, {'ctx': ctx})


DEFAULT_EVALUATORS: tuple[type[Evaluator[object, object, object]], ...] = (
    Equals,
    EqualsExpected,
    Contains,
    IsInstance,
    MaxDuration,
    LlmJudge,
    SpanQuery,
    # Python,  # not included by default for security reasons
)
