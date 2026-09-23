from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel, Field
from pydantic_core import to_jsonable_python
from vcr.cassette import Cassette

from .._inline_snapshot import snapshot
from ..cassette_utils import single_request_body
from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.typesafe import TypeSafeModel
    from pydantic_evals.evaluators import EvaluationReason, EvaluatorContext, GEval, LLMJudge, StructuredJudge
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


class SupportReview(BaseModel):
    """Judge a support reply against the support policy."""

    policy: Literal['compliant', 'violation'] = Field(description='Does the reply comply with the support policy?')
    completeness: float = Field(ge=0, le=1, description='How completely does the reply address the request?')
    asks_for_secret: bool = Field(description='Does the reply ask for a password or a login code?')


async def test_structured_judge_with_typesafe(
    allow_model_requests: None, env: TestEnv, typesafe_api_key: str, vcr: Cassette
):
    """Three measures come back from one Jev request, which is the shape Jev is built for."""
    env.set('TYPESAFE_API_KEY', typesafe_api_key)
    evaluator = StructuredJudge(SupportReview, model='typesafe:jev-latest')

    result = await evaluator.evaluate(
        _context('I cannot reset your password, but you can do it yourself from the account page.')
    )

    assert to_jsonable_python(result) == snapshot(
        {'policy': 'compliant', 'completeness': 0.6, 'asks_for_secret': False}
    )
    # `single_request_body` asserts the cassette holds exactly one request, which is the point of the
    # evaluator: all three questions were answered by it.
    assert single_request_body(vcr) == snapshot(
        {
            'state': """\
<Output>
I cannot reset your password, but you can do it yourself from the account page.
</Output>\
""",
            'model': 'jev-latest',
            'questions': {
                'policy': {
                    'type': 'choice',
                    'criteria': {'compliant': None, 'violation': None},
                    'instructions': {
                        'field': 'policy',
                        'question': 'Does the reply comply with the support policy?',
                        'goal': 'Judge a support reply against the support policy.',
                        'instructions': 'Answer each question about <Output>.',
                    },
                },
                'completeness': {
                    'type': 'noul',
                    'instructions': {
                        'field': 'completeness',
                        'question': 'How completely does the reply address the request?',
                        'goal': 'Judge a support reply against the support policy.',
                        'instructions': 'Answer each question about <Output>.',
                    },
                },
                'asks_for_secret': {
                    'type': 'noul',
                    'instructions': {
                        'field': 'asks_for_secret',
                        'question': 'Does the reply ask for a password or a login code?',
                        'goal': 'Judge a support reply against the support policy.',
                        'instructions': 'Answer each question about <Output>.',
                    },
                },
            },
        }
    )


async def test_structured_judge_rubrics_with_typesafe(
    allow_model_requests: None, env: TestEnv, typesafe_api_key: str, vcr: Cassette
):
    """The rubrics that would have been three `LLMJudge` evaluators are three questions in one request."""
    env.set('TYPESAFE_API_KEY', typesafe_api_key)
    evaluator = StructuredJudge(
        {
            'follows_policy': 'The reply follows the support policy.',
            'gives_next_step': 'The reply gives the customer a concrete next step.',
            'never_asks_for_secrets': 'The reply never asks for a password or a login code.',
        },
        model='typesafe:jev-latest',
    )

    result = await evaluator.evaluate(
        _context('I cannot reset your password, but you can do it yourself from the account page.')
    )

    assert to_jsonable_python(result) == snapshot(
        {'follows_policy': True, 'gives_next_step': True, 'never_asks_for_secrets': True}
    )
    assert single_request_body(vcr) == snapshot(
        {
            'state': """\
<Output>
I cannot reset your password, but you can do it yourself from the account page.
</Output>\
""",
            'model': 'jev-latest',
            'questions': {
                'follows_policy': {
                    'type': 'noul',
                    'instructions': {
                        'field': 'follows_policy',
                        'question': 'Is this statement true? The reply follows the support policy.',
                        'goal': 'Judge an output against each rubric.',
                        'instructions': 'Answer each question about <Output>.',
                    },
                },
                'gives_next_step': {
                    'type': 'noul',
                    'instructions': {
                        'field': 'gives_next_step',
                        'question': 'Is this statement true? The reply gives the customer a concrete next step.',
                        'goal': 'Judge an output against each rubric.',
                        'instructions': 'Answer each question about <Output>.',
                    },
                },
                'never_asks_for_secrets': {
                    'type': 'noul',
                    'instructions': {
                        'field': 'never_asks_for_secrets',
                        'question': 'Is this statement true? The reply never asks for a password or a login code.',
                        'goal': 'Judge an output against each rubric.',
                        'instructions': 'Answer each question about <Output>.',
                    },
                },
            },
        }
    )
