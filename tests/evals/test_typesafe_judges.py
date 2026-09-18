from __future__ import annotations

import pytest
from pydantic_core import to_jsonable_python

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.typesafe import TypeSafeModel
    from pydantic_evals.evaluators import EvaluationReason, EvaluatorContext, GEval, LLMJudge
    from pydantic_evals.otel._errors import SpanTreeRecordingError

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='pydantic-evals or typesafe-sdk not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


def _context(output: str) -> EvaluatorContext[None, str, None]:
    return EvaluatorContext(
        name='urgent-ticket',
        inputs=None,
        metadata=None,
        expected_output=None,
        output=output,
        duration=0,
        _span_tree=SpanTreeRecordingError('spans were not recorded'),
        attributes={},
        metrics={},
    )


async def test_llm_judge_with_typesafe(allow_model_requests: None, env: TestEnv, typesafe_api_key: str):
    """A string model id resolves Jev's profile and selects the verdict-only output shape."""
    env.set('TYPESAFE_API_KEY', typesafe_api_key)
    evaluator = LLMJudge(rubric='The ticket is urgent.', model='typesafe:jev-latest')

    result = await evaluator.evaluate(_context('Checkout returns a 500 for every customer right now.'))

    assert to_jsonable_python(result) == {'LLMJudge': True}


async def test_g_eval_with_typesafe(allow_model_requests: None, env: TestEnv, typesafe_api_key: str):
    """Jev scores the normalized rubric and GEval returns the requested 0-4 scale without a reason."""
    env.set('TYPESAFE_API_KEY', typesafe_api_key)
    model = TypeSafeModel('jev-latest')
    evaluator = GEval(
        criteria='urgency',
        evaluation_steps=['Decide whether someone is blocked now or many users are affected.', 'Assign the score.'],
        score_range=(0, 4),
        model=model,
    )

    result = await evaluator.evaluate(_context('Checkout returns a 500 for every customer right now.'))

    assert result == EvaluationReason(value=2, reason=None)
