"""Integration tests for conversation-level evaluators that exercise the judge agent end-to-end.

We deliberately avoid VCR cassettes here: instead of replaying recorded HTTP traffic from a
hosted LLM, we use [`FunctionModel`][pydantic_ai.models.function.FunctionModel] as the judge
model. That keeps the tests hermetic (no API key, no network) while still driving the full
stack — `evaluator.evaluate()` → `extract_conversation_turns` → `judge_conversation_goal` /
`judge_role_adherence` → Pydantic AI agent run — so we catch regressions that a mock of the
judge helper would miss.

Span trees are built with real `logfire.span` calls so that the `pydantic_ai.all_messages`
attribute is inspected the same way it would be in production.
"""

from __future__ import annotations as _annotations

import json
from typing import Any

import pytest

from .._inline_snapshot import snapshot
from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel
    from pydantic_evals.evaluators import (
        ConversationGoalAchievement,
        EvaluationReason,
        EvaluatorContext,
        RoleAdherence,
    )

with try_import() as logfire_import_successful:
    import logfire
    from logfire.testing import CaptureLogfire

    from pydantic_evals.otel._context_in_memory_span_exporter import context_subtree

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed'),
    pytest.mark.skipif(not logfire_import_successful(), reason='logfire not installed'),
    pytest.mark.anyio,
]


def _make_grading_function(*, reason: str, pass_: bool, score: float) -> Any:
    """Return a judge function that responds with a fixed GradingOutput JSON payload."""

    async def judge_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # The judge prompt is always rendered as plain text by `_build_conversation_prompt`,
        # so we don't need to inspect it — just produce a structured grading response.
        del messages, info  # unused; grading output is deterministic for this test
        payload = json.dumps({'reason': reason, 'pass': pass_, 'score': score})
        return ModelResponse(parts=[TextPart(content=payload)])

    return judge_function


def _build_context_with_run_span(
    all_messages: list[dict[str, Any]],
    *,
    output: Any,
    new_message_index: int | None = None,
) -> EvaluatorContext[Any, Any, Any]:
    """Build an `EvaluatorContext` whose `span_tree` contains a real logfire span with the
    `pydantic_ai.all_messages` attribute set, matching the shape that Pydantic AI emits at
    the end of an agent run.
    """
    attrs: dict[str, Any] = {'pydantic_ai.all_messages': json.dumps(all_messages)}
    if new_message_index is not None:
        attrs['pydantic_ai.new_message_index'] = new_message_index

    with context_subtree() as tree:
        with logfire.span('agent run', **attrs):
            pass

    return EvaluatorContext(
        name='integration',
        inputs={},
        metadata=None,
        expected_output=None,
        output=output,
        duration=0.0,
        _span_tree=tree,
        attributes={},
        metrics={},
    )


async def test_conversation_goal_achievement_integration(capfire: CaptureLogfire):
    """End-to-end: extraction + judge + threshold wiring, with the judge producing a high score."""
    assert capfire  # keeps logfire configured for the test

    ctx = _build_context_with_run_span(
        [
            {'role': 'system', 'parts': [{'type': 'text', 'content': 'Help the user reset their password.'}]},
            {'role': 'user', 'parts': [{'type': 'text', 'content': 'I forgot my password.'}]},
            {
                'role': 'assistant',
                'parts': [
                    {'type': 'text', 'content': "I'll send a reset link to your email. Check your inbox."},
                ],
            },
        ],
        output='Reset link sent.',
    )

    judge_model = FunctionModel(
        _make_grading_function(reason='The agent resolved the reset request cleanly.', pass_=True, score=0.9)
    )
    evaluator = ConversationGoalAchievement(
        goal='Reset the user password and tell them what to do next.',
        threshold=0.8,
        model=judge_model,
    )
    result = await evaluator.evaluate_async(ctx)
    assert result == snapshot(
        {
            'goal_achieved': EvaluationReason(value=True, reason='The agent resolved the reset request cleanly.'),
            'goal_achievement_score': 0.9,
        }
    )


async def test_role_adherence_integration(capfire: CaptureLogfire):
    """End-to-end: extraction + role-adherence judge, judge flags a specific turn."""
    assert capfire

    ctx = _build_context_with_run_span(
        [
            {'role': 'system', 'parts': [{'type': 'text', 'content': 'You are a customer support assistant.'}]},
            {'role': 'user', 'parts': [{'type': 'text', 'content': 'Are stocks a good buy right now?'}]},
            {
                'role': 'assistant',
                'parts': [{'type': 'text', 'content': "You should definitely buy tech stocks; they're undervalued."}],
            },
        ],
        output="You should definitely buy tech stocks; they're undervalued.",
    )

    judge_model = FunctionModel(
        _make_grading_function(
            reason='At turn 2 the assistant gave financial advice, breaking the support-only role.',
            pass_=False,
            score=0.4,
        )
    )
    evaluator = RoleAdherence(
        role='customer support assistant; never gives financial advice.',
        threshold=0.7,
        model=judge_model,
    )
    result = await evaluator.evaluate_async(ctx)
    assert result == snapshot(
        {
            'role_adhered': EvaluationReason(
                value=False,
                reason='At turn 2 the assistant gave financial advice, breaking the support-only role.',
            ),
            'role_adherence_score': 0.4,
        }
    )
