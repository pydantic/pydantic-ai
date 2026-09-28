"""Tests for the `max_tokens` Anthropic receives when the request doesn't set one."""

from __future__ import annotations as _annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from pydantic_ai import Agent

from ...conftest import RequestCapture, try_import
from ..test_anthropic import MockAnthropic, completion_message, get_mock_chat_completion_kwargs

with try_import() as imports_successful:
    from anthropic.types.beta import BetaTextBlock, BetaUsage

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

if TYPE_CHECKING:
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
    streamed: bool = False


MAX_TOKENS_CASES = {
    'no-thinking': MaxTokensCase('claude-sonnet-4-5', {}, None, 16384),
    # Thinks by default, and adaptive thinking has no budget.
    'default-thinking': MaxTokensCase('claude-opus-5', {}, None, 16384),
    'unified-adaptive': MaxTokensCase('claude-sonnet-4-6', {'thinking': 'high'}, {'type': 'adaptive'}, 16384),
    # A budget that leaves the default enough room for the answer keeps it.
    'explicit-budget': MaxTokensCase(
        'claude-sonnet-4-5',
        {'anthropic_thinking': {'type': 'enabled', 'budget_tokens': 8000}},
        {'type': 'enabled', 'budget_tokens': 8000},
        16384,
    ),
    # Larger budgets raise it, which Claude Opus 4.1's 32,000-token maximum output still fits at `'high'`.
    'unified-extended': MaxTokensCase(
        'claude-sonnet-4-5', {'thinking': 'high'}, {'type': 'enabled', 'budget_tokens': 16384}, 16384 + 4096
    ),
    # Above the SDK's non-streaming limit, the request is streamed instead.
    'unified-extended-xhigh': MaxTokensCase(
        'claude-sonnet-4-5',
        {'thinking': 'xhigh'},
        {'type': 'enabled', 'budget_tokens': 32768},
        32768 + 4096,
        streamed=True,
    ),
    'explicit-max-tokens': MaxTokensCase(
        'claude-sonnet-4-5',
        {'anthropic_thinking': {'type': 'enabled', 'budget_tokens': 4096}, 'max_tokens': 15000},
        {'type': 'enabled', 'budget_tokens': 4096},
        15000,
    ),
}


@pytest.mark.parametrize('case', MAX_TOKENS_CASES.values(), ids=MAX_TOKENS_CASES.keys())
async def test_default_max_tokens(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
    request_capture: RequestCapture,
    case: MaxTokensCase,
) -> None:
    """Without `max_tokens`, Anthropic gets 16384, raised to leave 4096 beyond an extended thinking budget.

    Anthropic requires `max_tokens`, counts thinking toward it (including the adaptive thinking current models do by
    default), and answers a request whose `max_tokens` isn't greater than `budget_tokens` with a 400. An explicit
    `max_tokens` is sent as is.
    """
    agent = Agent(anthropic_model(case.model_name, capture=True))
    result = await agent.run('What is 17 * 23? Answer with just the number.', model_settings=case.model_settings)

    assert result.output == '391'
    body = request_capture.body('/v1/messages')
    assert (body.get('thinking'), body['max_tokens'], body.get('stream', False)) == (
        case.thinking,
        case.max_tokens,
        case.streamed,
    )


@pytest.mark.parametrize(
    ('model_name', 'model_settings', 'max_tokens'),
    [
        pytest.param('claude-sonnet-4-20250514', {}, 4096, id='sonnet-4'),
        pytest.param('us.anthropic.claude-sonnet-4-20250514-v1:0', {}, 4096, id='bedrock-sonnet-4'),
        pytest.param('claude-opus-4-1@20250805', {}, 4096, id='vertex-opus-4-1'),
        pytest.param('claude-sonnet-4-20250514', {'thinking': 'high'}, 16384 + 4096, id='sonnet-4-extended-thinking'),
        pytest.param('claude-sonnet-4-5', {}, 16384, id='sonnet-4-5'),
    ],
)
async def test_default_max_tokens_on_models_that_reject_overflowing_the_context_window(
    allow_model_requests: None, model_name: str, model_settings: AnthropicModelSettings, max_tokens: int
) -> None:
    """Models older than Claude Sonnet 4.5 keep the lower default of 4096.

    They answer a request whose input plus `max_tokens` exceeds the context window with a 400 (Bedrock's
    `claude-sonnet-4-20250514` rejects 192K input tokens plus 16384, and accepts 192K plus 4096), so the higher
    default would break conversations close to the window. This is mocked because these models can't be recorded
    anymore: the Anthropic API has retired them, and Bedrock only serves them to accounts that used them recently.
    """
    mock_client = MockAnthropic.create_mock(
        completion_message([BetaTextBlock(text='391', type='text')], BetaUsage(input_tokens=5, output_tokens=2))
    )
    agent = Agent(AnthropicModel(model_name, provider=AnthropicProvider(anthropic_client=mock_client)))
    await agent.run('What is 17 * 23?', model_settings=model_settings)

    assert get_mock_chat_completion_kwargs(mock_client)[0]['max_tokens'] == max_tokens
