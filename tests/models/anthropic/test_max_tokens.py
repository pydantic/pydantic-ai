"""Tests for the `max_tokens` Anthropic receives when the request doesn't set one."""

from __future__ import annotations as _annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, ThinkingPart

from ...conftest import RequestCapture, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModelSettings

if TYPE_CHECKING:
    from pydantic_ai.models.anthropic import AnthropicModel

    ANTHROPIC_MODEL_FIXTURE = Callable[..., AnthropicModel]

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.vcr,
]


@dataclass(frozen=True)
class MaxTokensCase:
    model_name: str
    model_settings: AnthropicModelSettings
    thinking: dict[str, object] | None
    max_tokens: int


MAX_TOKENS_CASES = {
    'no-thinking': MaxTokensCase('claude-sonnet-4-5', {}, None, 4096),
    # Unified thinking maps to an extended thinking budget above the 4096 default.
    'unified-extended': MaxTokensCase(
        'claude-sonnet-4-5', {'thinking': 'high'}, {'type': 'enabled', 'budget_tokens': 16384}, 16384 + 4096
    ),
    'explicit-budget': MaxTokensCase(
        'claude-sonnet-4-5',
        {'anthropic_thinking': {'type': 'enabled', 'budget_tokens': 8000}},
        {'type': 'enabled', 'budget_tokens': 8000},
        8000 + 4096,
    ),
    # Adaptive thinking has no budget, so the default stays.
    'unified-adaptive': MaxTokensCase('claude-sonnet-4-6', {'thinking': 'high'}, {'type': 'adaptive'}, 4096),
    'explicit-max-tokens': MaxTokensCase(
        'claude-sonnet-4-5',
        {'anthropic_thinking': {'type': 'enabled', 'budget_tokens': 4096}, 'max_tokens': 15000},
        {'type': 'enabled', 'budget_tokens': 4096},
        15000,
    ),
}


@pytest.mark.parametrize('case', MAX_TOKENS_CASES.values(), ids=MAX_TOKENS_CASES.keys())
async def test_default_max_tokens_leaves_room_for_the_thinking_budget(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
    request_capture: RequestCapture,
    case: MaxTokensCase,
) -> None:
    """Without `max_tokens`, an extended thinking budget is added on top of the 4096 default.

    Anthropic counts `budget_tokens` toward `max_tokens` and answers a request whose `max_tokens` isn't greater
    than the budget with a 400 (`max_tokens` must be greater than `thinking.budget_tokens`), so unified
    `thinking=True`, `'medium'`, `'high'` and `'xhigh'` failed on every model with extended thinking unless
    `max_tokens` was set too. An explicit `max_tokens` is sent as is.
    """
    agent = Agent(anthropic_model(case.model_name, capture=True))
    result = await agent.run('What is 17 * 23? Answer with just the number.', model_settings=case.model_settings)

    assert result.output == '391'
    body = request_capture.body('/v1/messages')
    assert (body.get('thinking'), body['max_tokens']) == (case.thinking, case.max_tokens)
    response = result.all_messages()[1]
    assert isinstance(response, ModelResponse)
    assert any(isinstance(part, ThinkingPart) for part in response.parts) is (case.thinking is not None)
