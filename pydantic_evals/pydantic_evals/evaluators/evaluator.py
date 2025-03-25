from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, dataclass, fields
from typing import Any, Generic, cast

from pydantic import (
    TypeAdapter,
    ValidationError,
    model_serializer,
)
from pydantic.alias_generators import to_snake
from pydantic_core import to_jsonable_python
from typing_extensions import TypeVar

from .._utils import run_until_complete
from ._spec import EvaluatorSpec
from .context import EvaluatorContext

EvaluationScalar = bool | int | float | str
"""The most primitive output allowed as an output from an Evaluator.

`int` and `float` are treated as scores, `str` as labels, and `bool` as assertions.
"""


@dataclass
class EvaluationReason:
    """The result of running an evaluator with an optional explanation.

    Contains a scalar value and an optional "reason" explaining the value.

    Args:
        value: The scalar result of the evaluation (boolean, integer, float, or string).
        reason: An optional explanation of the evaluation result.
    """

    value: EvaluationScalar
    reason: str | None = None


EvaluatorOutput = EvaluationScalar | EvaluationReason | Mapping[str, EvaluationScalar | EvaluationReason]
"""Type for the output of an evaluator, which can be a scalar, an EvaluationReason, or a mapping of names to either."""


# TODO(DavidM): Add bound=EvaluationScalar to the following typevar after we upgrade to pydantic 2.11
EvaluationScalarT = TypeVar('EvaluationScalarT', default=EvaluationScalar, covariant=True)
"""Type variable for the scalar result type of an evaluation."""

T = TypeVar('T')


@dataclass
class EvaluationResult(Generic[EvaluationScalarT]):
    """The details of an individual evaluation result.

    Contains the name, value, reason, and source evaluator for a single evaluation.

    Args:
        name: The name of the evaluation.
        value: The scalar result of the evaluation.
        reason: An optional explanation of the evaluation result.
        source: The evaluator that produced this result.
    """

    name: str
    value: EvaluationScalarT
    reason: str | None
    source: Evaluator

    def downcast(self, *value_types: type[T]) -> EvaluationResult[T] | None:
        """Attempt to downcast this result to a more specific type.

        Args:
            *value_types: The types to check the value against.

        Returns:
            A downcast version of this result if the value is an instance of one of the given types,
            otherwise None.
        """
        if isinstance(self.value, value_types):
            return cast(EvaluationResult[T], self)
        return None


# Evaluators are contravariant in all of its parameters.
InputsT = TypeVar('InputsT', default=Any, contravariant=True)
"""Type variable for the inputs type of the task being evaluated."""

OutputT = TypeVar('OutputT', default=Any, contravariant=True)
"""Type variable for the output type of the task being evaluated."""

MetadataT = TypeVar('MetadataT', default=Any, contravariant=True)
"""Type variable for the metadata type of the task being evaluated."""


@dataclass
class Evaluator(Generic[InputsT, OutputT, MetadataT]):
    """Base class for all evaluators.

    Evaluators can assess the performance of a task in a variety of ways, as a function of the EvaluatorContext.

    Subclasses must implement either `evaluate` or `evaluate_async` (or both).

    Example:
    ```python
    @dataclass
    class ExactMatch(Evaluator[Any, Any, Any]):
        def evaluate(self, ctx: EvaluatorContext) -> bool:
            return ctx.actual_output == ctx.expected_output
    ```
    """

    def __post_init__(self):
        if type(self) is Evaluator:  # pragma: no cover
            raise TypeError('You should not instantiate Evaluator directly.')

    def __init_subclass__(cls):
        # Ensure that at least one of `evaluate` and `evaluate_async` is implemented on this class
        # and not inherited from the base class.
        # It's okay if it's on a parent class as long as the class isn't BaseEvaluator.
        if cls.evaluate is Evaluator.evaluate and cls.evaluate_async is Evaluator.evaluate_async:
            raise TypeError(f'{cls.__name__} must implement either `evaluate` or `evaluate_async`')

    @classmethod
    def name(cls) -> str:
        """Return the 'name' of this Evaluator to use during serialization.

        Returns:
            The snake-cased name of the evaluator class.
        """
        return to_snake(cls.__name__)

    def evaluate(self, ctx: EvaluatorContext[InputsT, OutputT, MetadataT]) -> EvaluatorOutput:
        """Evaluate the task output in the given context.

        This is a synchronous method that calls `evaluate_async` and runs it to completion.
        Subclasses should either override this method or `evaluate_async`, but not both.

        Args:
            ctx: The context containing the inputs, outputs, and metadata for evaluation.

        Returns:
            The evaluation result, which can be a scalar value, an EvaluationReason, or a mapping
            of evaluation names to either of those.
        """
        return run_until_complete(self.evaluate_async(ctx))

    async def evaluate_async(self, ctx: EvaluatorContext[InputsT, OutputT, MetadataT]) -> EvaluatorOutput:
        """Evaluate the task output in the given context asynchronously.

        This is an asynchronous method that by default calls `evaluate`. Subclasses should
        either override this method or `evaluate`, but not both.

        Args:
            ctx: The context containing the inputs, outputs, and metadata for evaluation.

        Returns:
            The evaluation result, which can be a scalar value, an EvaluationReason, or a mapping
            of evaluation names to either of those.

        Note:
            If you need to prevent this from blocking, you can override this with:
            `return await anyio.to_thread.run_sync(self.evaluate, ctx)`
        """
        # If you need to prevent this from blocking, you can override this with:
        # return await anyio.to_thread.run_sync(self.evaluate, ctx)
        return self.evaluate(ctx)

    @model_serializer(mode='plain')
    def serialize(self) -> Any:
        """Serialize this Evaluator to a JSON-serializable form.

        Returns:
            A JSON-serializable representation of this evaluator as an EvaluatorSpec.
        """
        raw_arguments: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            # always exclude defaults:
            if field.default is not MISSING:
                if value == field.default:
                    continue
            if field.default_factory is not MISSING:
                if value == field.default_factory():
                    continue
            raw_arguments[field.name] = value

        arguments: None | tuple[Any,] | dict[str, Any]
        if len(raw_arguments) == 0:
            arguments = None
        elif len(raw_arguments) == 1:
            arguments = (next(iter(raw_arguments.values())),)
        else:
            arguments = raw_arguments
        return to_jsonable_python(EvaluatorSpec(name=self.name(), arguments=arguments))


async def run_evaluator(
    evaluator: Evaluator[InputsT, OutputT, MetadataT], ctx: EvaluatorContext[InputsT, OutputT, MetadataT]
) -> list[EvaluationResult]:
    """Run an evaluator and return the results.

    This function runs an evaluator on the given context and processes the results into
    a standardized format.

    Args:
        evaluator: The evaluator to run.
        ctx: The context containing the inputs, outputs, and metadata for evaluation.

    Returns:
        A list of evaluation results.

    Raises:
        ValueError: If the evaluator returns a value of an invalid type.
    """
    raw_results = await evaluator.evaluate_async(ctx)

    try:
        results = _EVALUATOR_OUTPUT_ADAPTER.validate_python(raw_results)
    except ValidationError as e:
        raise ValueError(f'{evaluator!r}.evaluate returned a value of an invalid type: {raw_results!r}.') from e

    results = _convert_to_mapping(results, scalar_name=evaluator.name())

    details: list[EvaluationResult] = []
    for name, result in results.items():
        if not isinstance(result, EvaluationReason):
            result = EvaluationReason(value=result)
        details.append(EvaluationResult(name=name, value=result.value, reason=result.reason, source=evaluator))

    return details


_EVALUATOR_OUTPUT_ADAPTER = TypeAdapter[EvaluatorOutput](EvaluatorOutput)


def _convert_to_mapping(
    result: EvaluatorOutput, *, scalar_name: str
) -> Mapping[str, EvaluationScalar | EvaluationReason]:
    """Convert an evaluator output to a mapping from names to scalar values or evaluation reasons.

    Args:
        result: The evaluator output to convert.
        scalar_name: The name to use for a scalar result.

    Returns:
        A mapping from names to scalar values or evaluation reasons.
    """
    if isinstance(result, Mapping):
        return result
    return {scalar_name: result}
