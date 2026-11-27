from __future__ import annotations as _annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from pydantic import Field, GetPydanticSchema, RootModel, TypeAdapter
from pydantic.json_schema import SkipJsonSchema
from pydantic_core import core_schema
from typing_extensions import TypedDict, TypeForm

from pydantic_ai import models
from pydantic_ai._utils import is_model_like
from pydantic_ai.settings import ModelSettings

from .._utils import decode_duration
from ..otel.span_tree import SpanQuery
from .agentic import ArgumentCorrectness, MaxModelRequests, MaxToolCalls, ToolCorrectness, TrajectoryMatch
from .context import EvaluatorContext
from .evaluator import EvaluationReason, EvaluationScalar, Evaluator, EvaluatorOutput
from .spec import EvaluatorSpec

if TYPE_CHECKING:
    from .llm_as_a_judge import JudgeAnswer

__all__ = (
    'Equals',
    'EqualsExpected',
    'Contains',
    'IsInstance',
    'MaxDuration',
    'LLMJudge',
    'StructuredJudge',
    'GEval',
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

    def get_default_evaluation_name(self) -> str:
        return self.evaluation_name if isinstance(self.evaluation_name, str) else self.get_serialization_name()


@dataclass(repr=False)
class EqualsExpected(Evaluator[object, object, object]):
    """Check if the output exactly equals the expected output."""

    evaluation_name: str | None = field(default=None)

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> bool | dict[str, bool]:
        if ctx.expected_output is None:
            return {}  # Only compare if expected output is provided
        return ctx.output == ctx.expected_output

    def get_default_evaluation_name(self) -> str:
        return self.evaluation_name if isinstance(self.evaluation_name, str) else self.get_serialization_name()


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
    For model-like types (BaseModel, dataclasses), converts to a dict and checks key-value pairs.

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
            output_type = type(ctx.output)
            output_is_model_like = is_model_like(output_type)
            if isinstance(ctx.output, dict) or output_is_model_like:
                if output_is_model_like:
                    adapter: TypeAdapter[Any] = TypeAdapter(output_type)
                    output_dict = adapter.dump_python(ctx.output)  # pyright: ignore[reportUnknownMemberType]
                else:
                    # Cast to Any to avoid type checking issues
                    output_dict = cast(dict[Any, Any], ctx.output)  # pyright: ignore[reportUnknownMemberType]
                failure_reason = self._dict_failure_reason(output_dict)
            elif self.value not in ctx.output:  # pyright: ignore[reportOperatorIssue]  # will be handled by except block
                output_trunc = _truncated_repr(ctx.output, max_length=200)
                failure_reason = f'Output {output_trunc} does not contain provided value'
        except (TypeError, ValueError) as e:
            failure_reason = f'Containment check failed: {e}'

        return EvaluationReason(value=failure_reason is None, reason=failure_reason)

    def _dict_failure_reason(self, output_dict: dict[Any, Any]) -> str | None:
        if isinstance(self.value, dict):
            # Cast to Any to avoid type checking issues
            expected_dict = cast(dict[Any, Any], self.value)  # pyright: ignore[reportUnknownMemberType]
            for k in expected_dict:
                if k not in output_dict:
                    k_trunc = _truncated_repr(k, max_length=30)
                    return f'Output does not contain expected key {k_trunc}'
                elif output_dict[k] != expected_dict[k]:
                    k_trunc = _truncated_repr(k, max_length=30)
                    output_v_trunc = _truncated_repr(output_dict[k], max_length=100)
                    expected_v_trunc = _truncated_repr(expected_dict[k], max_length=100)
                    return f'Output has different value for key {k_trunc}: {output_v_trunc} != {expected_v_trunc}'
        elif self.value not in output_dict:
            output_trunc = _truncated_repr(output_dict, max_length=200)
            return f'Output {output_trunc} does not contain provided value as a key'
        return None

    def get_default_evaluation_name(self) -> str:
        return self.evaluation_name if isinstance(self.evaluation_name, str) else self.get_serialization_name()


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

    def get_default_evaluation_name(self) -> str:
        return self.evaluation_name if isinstance(self.evaluation_name, str) else self.get_serialization_name()


@dataclass(repr=False)
class MaxDuration(Evaluator[object, object, object]):
    """Check if the execution time is under the specified maximum."""

    seconds: float | timedelta

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> bool:
        return timedelta(seconds=ctx.duration) <= decode_duration(self.seconds)


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


def _serialize_model_as_string(arguments: dict[str, Any]) -> dict[str, Any]:
    """Replace a `model` argument's `Model` instance with its `model_id` string, so specs round-trip cleanly."""
    # always serialize the model as a string when present; use its name if it's a KnownModelName
    if (model := arguments.get('model')) and isinstance(model, models.Model):
        arguments['model'] = model.model_id

    # Note: this may lead to confusion if you try to serialize-then-deserialize with a custom model.
    # I expect that is rare enough to be worth not solving yet, but common enough that we probably will want to
    # solve it eventually. I'm imagining some kind of model registry, but don't want to work out the details yet.
    return arguments


@dataclass(repr=False)
class LLMJudge(Evaluator[object, object, object]):
    """Judge whether the output of a language model meets the criteria of a provided rubric.

    If you do not specify a model, it uses the default model for judging. This starts as 'openai:gpt-5.2', but can be
    overridden by calling [`set_default_judge_model`][pydantic_evals.evaluators.llm_as_a_judge.set_default_judge_model].

    A judge whose profile has `supports_text_output=False` returns only the typed pass/fail verdict. Its reason is
    unavailable, and its score is `1.0` for pass or `0.0` for fail.
    """

    rubric: str
    model: models.Model | models.KnownModelName | str | None = None
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
                from .llm_as_a_judge import _judge_input_output_expected  # pyright: ignore[reportPrivateUsage]

                grading_output = await _judge_input_output_expected(
                    ctx.inputs,
                    ctx.output,
                    ctx.expected_output,
                    self.rubric,
                    self.model,
                    self.model_settings,
                    allow_reasonless=True,
                )
            else:
                from .llm_as_a_judge import _judge_input_output  # pyright: ignore[reportPrivateUsage]

                grading_output = await _judge_input_output(
                    ctx.inputs,
                    ctx.output,
                    self.rubric,
                    self.model,
                    self.model_settings,
                    allow_reasonless=True,
                )
        else:
            if self.include_expected_output:
                from .llm_as_a_judge import _judge_output_expected  # pyright: ignore[reportPrivateUsage]

                grading_output = await _judge_output_expected(
                    ctx.output,
                    ctx.expected_output,
                    self.rubric,
                    self.model,
                    self.model_settings,
                    allow_reasonless=True,
                )
            else:
                from .llm_as_a_judge import _judge_output  # pyright: ignore[reportPrivateUsage]

                grading_output = await _judge_output(
                    ctx.output, self.rubric, self.model, self.model_settings, allow_reasonless=True
                )

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
        return _serialize_model_as_string(super().build_serialization_arguments())


@dataclass(repr=False)
class StructuredJudge(Evaluator[object, object, object]):
    """Ask a judge typed questions about one output in a single model request, and report each answer.

    Where [`LLMJudge`][pydantic_evals.evaluators.LLMJudge] grades an output against a rubric and explains its
    verdict, `StructuredJudge` asks questions and reports the answers themselves, with no prose: one question, or
    several in one request where one `LLMJudge` per measure would send the same output to the judge once per
    question. Several questions are reported exactly like the measures of any evaluator that returns a mapping,
    each named after its question, so three questions read as three results with one evaluator as their source.

    The questions come in one of three shapes. A single question is answered with its `output_type`: a `bool`,
    the default, is a yes or a no; a `Literal` or `Enum` of strings picks one option; and a `float` is a score,
    bounded by the type you pass, such as `Annotated[float, Field(ge=0, le=1)]`:

    ```python {title="structured_judge_question.py"}
    from typing import Literal

    from pydantic_evals.evaluators import StructuredJudge

    StructuredJudge('Is the reply polite?')
    StructuredJudge(
        'How does the reply treat the customer?',
        output_type=Literal['warm', 'neutral', 'curt'],
        evaluation_name='tone',
    )
    ```

    A mapping of measure name to a rubric is answered with one pass/fail assertion per rubric:

    ```python {title="structured_judge_rubrics.py"}
    from pydantic_evals.evaluators import StructuredJudge

    StructuredJudge(
        {
            'follows_policy': 'The reply follows the support policy.',
            'gives_next_step': 'The reply gives the customer a concrete next step.',
            'never_asks_for_secrets': 'The reply never asks for a password or a login code.',
        }
    )
    ```

    A Pydantic model, dataclass or `TypedDict` asks one question per field, which is what to reach for when
    several measures are scores or labels rather than a yes or a no:

    ```python {title="structured_judge_output_type.py"}
    from typing import Literal

    from pydantic import BaseModel, Field

    from pydantic_evals.evaluators import StructuredJudge


    class ReplyReview(BaseModel):
        policy: Literal['compliant', 'minor_issue', 'violation'] = Field(
            description='Does the reply comply with the support policy?'
        )
        completeness: float = Field(
            ge=0, le=1, description='How completely does the reply address the request?'
        )
        asks_for_secret: bool = Field(
            description='Does the reply ask for a password or a login code?'
        )


    StructuredJudge(ReplyReview)
    ```

    A `bool` answer is reported as an assertion, an `int` or `float` as a score, and a `str` as a label.
    A report groups labels by value, so ask for a `Literal` or an `Enum` when the answer is a category: free
    text gives each case a label of its own, and never becomes the reason for another answer. An `Enum` is reported as whatever its value is, so an `Enum` of strings is a label and an `IntEnum` a
    score. An answer of `None` is reported as no measure at all, the way
    [`EqualsExpected`][pydantic_evals.evaluators.EqualsExpected] reports nothing without an expected
    output. Any other answer is an error, since there is no measure to report it as. When the judge reports how
    confident it is in an answer, as a [decision model](https://pydantic.dev/docs/ai/models/decision/) such as
    TypeSafe's Jev does, the confidence becomes the answer's reason.

    Give every field a description, and the model a docstring saying what is being judged: the field
    descriptions are the questions, and a judge that takes its questions from the output type has
    nothing to ask without them.

    One request answering every question also fails as one: a request that errors leaves an
    [`EvaluatorFailure`][pydantic_evals.evaluators.EvaluatorFailure] in place of all of the measures,
    where one judge per measure fails only its own. Asking the questions together can also move the
    answers, since each question is answered in the presence of the others, so measure both shapes on
    your own cases before switching a suite over.

    If you do not specify a model, it uses the default model for judging. This starts as 'openai:gpt-5.2', but can be
    overridden by calling [`set_default_judge_model`][pydantic_evals.evaluators.llm_as_a_judge.set_default_judge_model].

    A type cannot be written in a YAML or JSON dataset file, so only a question answered yes or no and the
    mapping of rubrics round-trip through [`Dataset.to_file`][pydantic_evals.dataset.Dataset.to_file]. A dataset
    with any other `output_type` or with a model of questions can't be saved, so build a dataset that uses either in
    Python.
    """

    # A Python class has no JSON schema, so only a question and the mapping form are offered to a dataset file. The
    # schema asks for at least one rubric, as `__post_init__` refuses an empty mapping.
    questions: str | Annotated[Mapping[str, str], Field(min_length=1)] | SkipJsonSchema[type[object]]
    # A type form has no core schema for a dataset file to validate against, and no JSON schema either.
    output_type: SkipJsonSchema[
        Annotated[TypeForm[object], GetPydanticSchema(lambda _source, _handler: core_schema.any_schema())]
    ] = bool
    model: models.Model | models.KnownModelName | str | None = None
    include_input: bool = False
    include_expected_output: bool = False
    model_settings: ModelSettings | None = None
    evaluation_name: str | None = field(default=None)

    def __post_init__(self):
        if isinstance(self.questions, str):
            if isinstance(self.output_type, type) and is_model_like(self.output_type):
                raise ValueError(
                    '`output_type` is the type of the answer to a single question. To ask a question per field, '
                    f'pass the model as `questions`: `StructuredJudge({self.output_type.__name__})`'
                )
            return
        if self.output_type is not bool:
            raise ValueError(
                '`output_type` only applies to a single question: a rubric is answered yes or no, '
                'and a field is answered with its own type'
            )
        if isinstance(self.questions, Mapping):
            names: Mapping[str, str] = self.questions
        elif not is_model_like(self.questions):
            raise ValueError(
                '`questions` must be a question, a mapping of measure name to rubric, '
                'or a Pydantic model, dataclass or `TypedDict` whose fields are the questions'
            )
        elif issubclass(self.questions, RootModel):
            # A root model has one unnamed value, so there is no question to name a measure after.
            raise ValueError('`questions` must be a model with named fields, not a `RootModel`')
        else:
            from .llm_as_a_judge import _question_fields  # pyright: ignore[reportPrivateUsage]

            names = _question_fields(self.questions)
        if not names:
            raise ValueError('`questions` must contain at least one question')

    async def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluatorOutput:
        inputs = ctx.inputs if self.include_input else None
        expected_output = ctx.expected_output if self.include_expected_output else None
        if isinstance(self.questions, str):
            from .llm_as_a_judge import _judge_question  # pyright: ignore[reportPrivateUsage]

            answer = await _judge_question(
                ctx.output, self.questions, self.output_type, inputs, expected_output, self.model, self.model_settings
            )
            measure = _measure(self.get_default_evaluation_name(), answer)
            return {} if measure is None else measure

        from .llm_as_a_judge import judge_questions

        answers = await judge_questions(
            ctx.output,
            self.questions,
            inputs=inputs,
            expected_output=expected_output,
            model=self.model,
            model_settings=self.model_settings,
        )
        return {name: measure for name, answer in answers.items() if (measure := _measure(name, answer)) is not None}

    def get_default_evaluation_name(self) -> str:
        return self.evaluation_name if isinstance(self.evaluation_name, str) else self.get_serialization_name()

    def _serialization_spec(self) -> EvaluatorSpec:
        # Written by name, a type would load back as a string: a model of questions as a single question that is the
        # model's name, and an `output_type` as a name no judge can answer with.
        if isinstance(self.questions, type):
            raise TypeError(
                f'`StructuredJudge({self.questions.__name__})` cannot be serialized, as its questions are a Python '
                'type. To save a dataset to a file, use a mapping of rubrics, or build the dataset in Python.'
            )
        if self.output_type is not bool:
            raise TypeError(
                '`StructuredJudge` with an `output_type` other than `bool` cannot be serialized, as its `output_type` '
                'is a Python type. To save a dataset to a file, ask a question answered yes or no, or build the dataset '
                'in Python.'
            )
        return super()._serialization_spec()

    def build_serialization_arguments(self):
        arguments = _serialize_model_as_string(super().build_serialization_arguments())
        # A type does not survive a YAML round trip; its name keeps the spec serializable, as the JSON of a
        # result's `source` must be.
        if isinstance(questions := arguments.get('questions'), type):
            arguments['questions'] = questions.__name__
        if (output_type := arguments.get('output_type')) is not None:
            arguments['output_type'] = output_type.__name__ if isinstance(output_type, type) else str(output_type)
        return arguments


def _measure(name: str, answer: JudgeAnswer) -> EvaluationScalar | EvaluationReason | None:
    """The measure a judge's answer reports, with the judge's confidence as its reason, or `None` for no measure."""
    value = answer.value
    if value is None:
        return None
    # The answer is the value its field validated to, so an `Enum` arrives as the member and a value that is not a
    # scalar at all — a nested model, a list, a date — arrives as itself and is refused here, rather than as the
    # string a JSON dump would quietly turn it into.
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, (bool, int, float, str)):
        raise ValueError(
            f'The judge answered {name!r} with {value!r}, which cannot be reported as a measure. '
            'Every question has to be answered with a `bool`, `int`, `float`, `str`, or an `Enum` of those.'
        )
    if answer.confidence is None:
        return value
    return EvaluationReason(value=value, reason=f'confidence {answer.confidence:.2f}')


@dataclass(repr=False)
class GEval(Evaluator[object, object, object]):
    """G-Eval-style chain-of-thought evaluator (Liu et al., 2023).

    The judge is shown the evaluation `criteria` and a list of explicit `evaluation_steps`,
    produces a short reasoning trace, and emits an integer score within `score_range` (inclusive),
    returned as an [`EvaluationReason`][pydantic_evals.evaluators.EvaluationReason]. Because the
    criteria and steps are user-supplied, `GEval` puts no structural requirements on `ctx.inputs`
    or `ctx.output`.

    If you do not specify a model, it uses the default model for judging. This starts as 'openai:gpt-5.2', but can be
    overridden by calling [`set_default_judge_model`][pydantic_evals.evaluators.llm_as_a_judge.set_default_judge_model].

    A judge whose profile has `supports_text_output=False` returns the same integer score scale without a reason;
    its `score_range` may contain at most 20 levels.

    !!! note "Simplified G-Eval"
        The paper computes a probability-weighted expectation over score tokens using log-probs.
        We ask the model for a direct integer score instead, trading some correlation with human
        judgment for provider-agnostic simplicity.
    """

    criteria: str
    evaluation_steps: list[str]
    score_range: tuple[int, int] = (1, 5)
    include_input: bool = False
    model: models.Model | models.KnownModelName | str | None = None
    model_settings: ModelSettings | None = None
    evaluation_name: str | None = field(default=None)

    def __post_init__(self):
        if self.score_range[0] >= self.score_range[1]:
            raise ValueError(f'`score_range` must satisfy min < max, got {self.score_range!r}')
        if not self.evaluation_steps:
            raise ValueError('`evaluation_steps` must contain at least one step')

    async def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluatorOutput:
        from .llm_as_a_judge import _judge_g_eval  # pyright: ignore[reportPrivateUsage]

        g_eval_output = await _judge_g_eval(
            ctx.output,
            self.criteria,
            self.evaluation_steps,
            self.score_range,
            inputs=ctx.inputs if self.include_input else None,
            model=self.model,
            model_settings=self.model_settings,
            allow_reasonless=True,
        )
        return EvaluationReason(value=g_eval_output.score, reason=g_eval_output.reason)

    def get_default_evaluation_name(self) -> str:
        return self.evaluation_name if isinstance(self.evaluation_name, str) else self.get_serialization_name()

    def build_serialization_arguments(self):
        return _serialize_model_as_string(super().build_serialization_arguments())


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

    def get_default_evaluation_name(self) -> str:
        return self.evaluation_name if isinstance(self.evaluation_name, str) else self.get_serialization_name()


DEFAULT_EVALUATORS: tuple[type[Evaluator[object, object, object]], ...] = (
    Equals,
    EqualsExpected,
    Contains,
    IsInstance,
    MaxDuration,
    LLMJudge,
    StructuredJudge,
    HasMatchingSpan,
    ToolCorrectness,
    TrajectoryMatch,
    ArgumentCorrectness,
    MaxToolCalls,
    MaxModelRequests,
    GEval,
)


def __getattr__(name: str):
    if name == 'Python':
        raise ImportError(
            'The `Python` evaluator has been removed for security reasons. See https://github.com/pydantic/pydantic-ai/pull/2808 for more details and a workaround.'
        )
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
