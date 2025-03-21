from __future__ import annotations as _annotations

from pydantic_ai import models

from .context import ScoringContext
from .spec import AssessmentResult


async def llm_rubric(
    ctx: ScoringContext[object, object, object],
    rubric: str,
    model: models.KnownModelName = 'gpt-4o',
    include_input: bool = False,
) -> AssessmentResult:
    """Judge whether the output of a language model meets the criteria of a provided rubric."""
    if include_input:
        from .llm_as_a_judge import judge_input_output

        grading_output = await judge_input_output(ctx.inputs, ctx.output, rubric, model)
    else:
        from .llm_as_a_judge import judge_output

        grading_output = await judge_output(ctx.output, rubric, model)
    return AssessmentResult(value=grading_output.pass_, reason=grading_output.reason)


async def is_instance(ctx: ScoringContext[object, object, object], type_name: str) -> AssessmentResult:
    """Check if the output is an instance of a type with the given name."""
    output = ctx.output
    for cls in type(output).__mro__:
        if cls.__name__ == type_name or cls.__qualname__ == type_name:
            return AssessmentResult(value=True)

    return AssessmentResult(
        value=False, reason=f'output is of type {type(output).__name__} ({type(output).__qualname__})'
    )
