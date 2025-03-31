from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import (
    TypeAdapter,
    ValidationError,
)
from typing_extensions import TypeVar

from .context import EvaluatorContext
from .evaluator import EvaluationReason, EvaluationResult, EvaluationScalar, Evaluator, EvaluatorOutput

InputsT = TypeVar('InputsT', default=Any, contravariant=True)
OutputT = TypeVar('OutputT', default=Any, contravariant=True)
MetadataT = TypeVar('MetadataT', default=Any, contravariant=True)


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
