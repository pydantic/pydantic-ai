from __future__ import annotations

import traceback
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import (
    TypeAdapter,
    ValidationError,
)
from typing_extensions import TypeVar

from pydantic_evals._utils import logfire_span

from .context import EvaluatorContext
from .evaluator import (
    EvaluationReason,
    EvaluationResult,
    EvaluationScalar,
    Evaluator,
    EvaluatorFailure,
    EvaluatorOutput,
)

if TYPE_CHECKING:
    from pydantic_ai.retries import RetryConfig


InputsT = TypeVar('InputsT', default=Any, contravariant=True)
OutputT = TypeVar('OutputT', default=Any, contravariant=True)
MetadataT = TypeVar('MetadataT', default=Any, contravariant=True)


async def run_evaluator(
    evaluator: Evaluator[InputsT, OutputT, MetadataT],
    ctx: EvaluatorContext[InputsT, OutputT, MetadataT],
    retry: RetryConfig | None = None,
) -> list[EvaluationResult] | EvaluatorFailure:
    """Run an evaluator and return the results.

    This function runs an evaluator on the given context and processes the results into
    a standardized format.

    Args:
        evaluator: The evaluator to run.
        ctx: The context containing the inputs, outputs, and metadata for evaluation.
        retry: The retry configuration to use for running the evaluator.

    Returns:
        A list of evaluation results, or an evaluator failure if an exception is raised during its execution.

    Raises:
        ValueError: If the evaluator returns a value of an invalid type.
    """
    evaluate = evaluator.evaluate_async
    if retry is not None:
        # import from pydantic_ai.retries to trigger more descriptive import error if tenacity is missing
        from pydantic_ai.retries import retry as tenacity_retry

        evaluate = tenacity_retry(**retry)(evaluate)

    try:
        with logfire_span(
            'evaluator: {evaluator_name}',
            evaluator_name=evaluator.get_default_evaluation_name(),
        ):
            raw_results = await evaluate(ctx)

            try:
                results = _EVALUATOR_OUTPUT_ADAPTER.validate_python(raw_results)
            except ValidationError as e:
                raise ValueError(f'{evaluator!r}.evaluate returned a value of an invalid type: {raw_results!r}.') from e
    except Exception as e:
        return EvaluatorFailure(
            name=evaluator.get_default_evaluation_name(),
            error_message=f'{type(e).__name__}: {e}',
            error_stacktrace=traceback.format_exc(),
            source=evaluator.as_spec(),
        )

    results = _convert_to_mapping(results, scalar_name=evaluator.get_default_evaluation_name())

    details: list[EvaluationResult] = []
    for name, result in results.items():
        if not isinstance(result, EvaluationReason):
            result = EvaluationReason(value=result)
        details.append(
            EvaluationResult(name=name, value=result.value, reason=result.reason, source=evaluator.as_spec())
        )

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
