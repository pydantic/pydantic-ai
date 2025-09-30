from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Literal, cast

from typing_extensions import TypedDict

from pydantic_ai import models
from pydantic_ai.settings import ModelSettings

from ..otel.span_tree import SpanQuery
from .context import EvaluatorContext
from .evaluator import EvaluationReason, EvaluationScalar, Evaluator, EvaluatorOutput

__all__ = (
    'Equals',
    'EqualsExpected',
    'Contains',
    'IsInstance',
    'MaxDuration',
    'LLMJudge',
    'HasMatchingSpan',
    'OutputConfig',
)


@dataclass(repr=False)
class Equals(Evaluator[object, object, object]):
    """Check if the output exactly equals the provided value."""

    value: Any
    evaluation_name: str | None = field(default=None)

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> bool:
        return ctx.output == self.value


@dataclass(repr=False)
class EqualsExpected(Evaluator[object, object, object]):
    """Check if the output exactly equals the expected output."""

    evaluation_name: str | None = field(default=None)

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> bool | dict[str, bool]:
        if ctx.expected_output is None:
            return {}  # Only compare if expected output is provided
        return ctx.output == ctx.expected_output


# _MAX_REASON_LENGTH = 500
# _MAX_REASON_KEY_LENGTH = 30


def _truncated_repr(value: Any, max_length: int = 100) -> str:
    repr_value = repr(value)
    if len(repr_value) > max_length:
        repr_value = repr_value[: max_length // 2] + '...' + repr_value[-max_length // 2 :]
    return repr_value


@dataclass(repr=False)
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
    evaluation_name: str | None = field(default=None)

    def evaluate(
        self,
        ctx: EvaluatorContext[object, object, object],
    ) -> EvaluationReason:
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
                output_trunc = _truncated_repr(output_str, max_length=100)
                expected_trunc = _truncated_repr(expected_str, max_length=100)
                failure_reason = f'Output string {output_trunc} does not contain expected string {expected_trunc}'
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
                            k_trunc = _truncated_repr(k, max_length=30)
                            failure_reason = f'Output dictionary does not contain expected key {k_trunc}'
                            break
                        elif output_dict[k] != expected_dict[k]:
                            k_trunc = _truncated_repr(k, max_length=30)
                            output_v_trunc = _truncated_repr(output_dict[k], max_length=100)
                            expected_v_trunc = _truncated_repr(expected_dict[k], max_length=100)
                            failure_reason = f'Output dictionary has different value for key {k_trunc}: {output_v_trunc} != {expected_v_trunc}'
                            break
                else:
                    if self.value not in ctx.output:  # pyright: ignore[reportUnknownMemberType]
                        output_trunc = _truncated_repr(ctx.output, max_length=200)  # pyright: ignore[reportUnknownMemberType]
                        failure_reason = f'Output {output_trunc} does not contain provided value as a key'
            elif self.value not in ctx.output:  # pyright: ignore[reportOperatorIssue]  # will be handled by except block
                output_trunc = _truncated_repr(ctx.output, max_length=200)
                failure_reason = f'Output {output_trunc} does not contain provided value'
        except (TypeError, ValueError) as e:
            failure_reason = f'Containment check failed: {e}'

        return EvaluationReason(value=failure_reason is None, reason=failure_reason)


@dataclass(repr=False)
class IsInstance(Evaluator[object, object, object]):
    """Check if the output is an instance of a type with the given name."""

    type_name: str
    evaluation_name: str | None = field(default=None)

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluationReason:
        output = ctx.output
        for cls in type(output).__mro__:
            if cls.__name__ == self.type_name or cls.__qualname__ == self.type_name:
                return EvaluationReason(value=True)

        reason = f'output is of type {type(output).__name__}'
        if type(output).__qualname__ != type(output).__name__:
            reason += f' (qualname: {type(output).__qualname__})'
        return EvaluationReason(value=False, reason=reason)


@dataclass(repr=False)
class MaxDuration(Evaluator[object, object, object]):
    """Check if the execution time is under the specified maximum."""

    seconds: float | timedelta

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> bool:
        duration = timedelta(seconds=ctx.duration)
        seconds = self.seconds
        if not isinstance(seconds, timedelta):
            seconds = timedelta(seconds=seconds)
        return duration <= seconds


class OutputConfig(TypedDict, total=False):
    """Configuration for the score and assertion outputs of the LLMJudge evaluator."""

    evaluation_name: str
    include_reason: bool


def _update_combined_output(
    combined_output: dict[str, EvaluationScalar | EvaluationReason],
    value: EvaluationScalar,
    reason: str | None,
    config: OutputConfig,
    default_name: str,
) -> None:
    name = config.get('evaluation_name') or default_name
    if config.get('include_reason') and reason is not None:
        combined_output[name] = EvaluationReason(value=value, reason=reason)
    else:
        combined_output[name] = value


@dataclass(repr=False)
class LLMJudge(Evaluator[object, object, object]):
    """Judge whether the output of a language model meets the criteria of a provided rubric.

    If you do not specify a model, it uses the default model for judging. This starts as 'openai:gpt-4o', but can be
    overridden by calling [`set_default_judge_model`][pydantic_evals.evaluators.llm_as_a_judge.set_default_judge_model].
    """

    rubric: str
    model: models.Model | models.KnownModelName | None = None
    include_input: bool = False
    include_expected_output: bool = False
    model_settings: ModelSettings | None = None
    score: OutputConfig | Literal[False] = False
    assertion: OutputConfig | Literal[False] = field(default_factory=lambda: OutputConfig(include_reason=True))

    async def evaluate(
        self,
        ctx: EvaluatorContext[object, object, object],
    ) -> EvaluatorOutput:
        if self.include_input:
            if self.include_expected_output:
                from .llm_as_a_judge import judge_input_output_expected

                grading_output = await judge_input_output_expected(
                    ctx.inputs, ctx.output, ctx.expected_output, self.rubric, self.model, self.model_settings
                )
            else:
                from .llm_as_a_judge import judge_input_output

                grading_output = await judge_input_output(
                    ctx.inputs, ctx.output, self.rubric, self.model, self.model_settings
                )
        else:
            if self.include_expected_output:
                from .llm_as_a_judge import judge_output_expected

                grading_output = await judge_output_expected(
                    ctx.output, ctx.expected_output, self.rubric, self.model, self.model_settings
                )
            else:
                from .llm_as_a_judge import judge_output

                grading_output = await judge_output(ctx.output, self.rubric, self.model, self.model_settings)

        output: dict[str, EvaluationScalar | EvaluationReason] = {}
        include_both = self.score is not False and self.assertion is not False
        evaluation_name = self.get_default_evaluation_name()

        if self.score is not False:
            default_name = f'{evaluation_name}_score' if include_both else evaluation_name
            _update_combined_output(output, grading_output.score, grading_output.reason, self.score, default_name)

        if self.assertion is not False:
            default_name = f'{evaluation_name}_pass' if include_both else evaluation_name
            _update_combined_output(output, grading_output.pass_, grading_output.reason, self.assertion, default_name)

        return output

    def build_serialization_arguments(self):
        result = super().build_serialization_arguments()
        # always serialize the model as a string when present; use its name if it's a KnownModelName
        if (model := result.get('model')) and isinstance(model, models.Model):  # pragma: no branch
            result['model'] = f'{model.system}:{model.model_name}'

        # Note: this may lead to confusion if you try to serialize-then-deserialize with a custom model.
        # I expect that is rare enough to be worth not solving yet, but common enough that we probably will want to
        # solve it eventually. I'm imagining some kind of model registry, but don't want to work out the details yet.
        return result


@dataclass(repr=False)
class HasMatchingSpan(Evaluator[object, object, object]):
    """Check if the span tree contains a span that matches the specified query."""

    query: SpanQuery
    evaluation_name: str | None = field(default=None)

    def evaluate(
        self,
        ctx: EvaluatorContext[object, object, object],
    ) -> bool:
        return ctx.span_tree.any(self.query)


DEFAULT_EVALUATORS: tuple[type[Evaluator[object, object, object]], ...] = (
    Equals,
    EqualsExpected,
    Contains,
    IsInstance,
    MaxDuration,
    LLMJudge,
    HasMatchingSpan,
)


def __getattr__(name: str):
    if name == 'Python':
        raise ImportError(
            'The `Python` evaluator has been removed for security reasons. See https://github.com/pydantic/pydantic-ai/pull/2808 for more details and a workaround.'
        )
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
