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
    # The model's maximum output is above the SDK's non-streaming limit, so `run()` streams the request.
    'no-thinking': MaxTokensCase('claude-sonnet-4-5', {}, None, 64_000, streamed=True),
    # Thinks by default, and adaptive thinking has no budget.
    'default-thinking': MaxTokensCase('claude-opus-5', {}, None, 128_000, streamed=True),
    'unified-adaptive': MaxTokensCase(
        'claude-sonnet-4-6', {'thinking': 'high'}, {'type': 'adaptive'}, 128_000, streamed=True
    ),
    'unified-extended': MaxTokensCase(
        'claude-sonnet-4-5', {'thinking': 'high'}, {'type': 'enabled', 'budget_tokens': 16384}, 64_000, streamed=True
    ),
    # With a custom timeout the SDK doesn't require streaming, so `AnthropicModel` streams the default itself.
    'custom-timeout': MaxTokensCase('claude-sonnet-4-5', {'timeout': 120}, None, 64_000, streamed=True),
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
    """Without `max_tokens`, Anthropic gets the model's maximum output, as APIs where the output limit is optional do.

    Anthropic requires `max_tokens` and counts thinking toward it, including the adaptive thinking current models do
    by default. The SDK refuses a non-streaming request above about 21,000 tokens, so `run()` streams it instead. An
    explicit `max_tokens` is sent as is.
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
        pytest.param('claude-unknown', {}, 16384, id='unknown-model'),
        pytest.param(
            'claude-unknown',
            {'anthropic_thinking': {'type': 'enabled', 'budget_tokens': 16000}},
            16000 + 4096,
            id='unknown-model-extended-thinking',
        ),
        pytest.param('us.anthropic.claude-haiku-4-5-20251001-v1:0', {}, 64_000, id='bedrock-haiku-4-5'),
    ],
)
async def test_default_max_tokens_without_a_known_safe_maximum(
    allow_model_requests: None, model_name: str, model_settings: AnthropicModelSettings, max_tokens: int
) -> None:
    """Models older than Claude Sonnet 4.5 keep 4096, and models with an unknown maximum output get 16384.

    Older models answer a request whose input plus `max_tokens` exceeds the context window with a 400 (Bedrock's
    `claude-sonnet-4-20250514` rejects 192K input tokens plus 16384, and accepts 192K plus 4096), so a higher default
    would break conversations close to the window. An extended thinking budget raises either default to leave 4096
    beyond it. This is mocked because the older models can't be recorded anymore: the Anthropic API has retired them,
    and Bedrock only serves them to accounts that used them recently.
    """
    mock_client = MockAnthropic.create_mock(
        completion_message([BetaTextBlock(text='391', type='text')], BetaUsage(input_tokens=5, output_tokens=2))
    )
    agent = Agent(AnthropicModel(model_name, provider=AnthropicProvider(anthropic_client=mock_client)))
    await agent.run('What is 17 * 23?', model_settings=model_settings)

    assert get_mock_chat_completion_kwargs(mock_client)[0]['max_tokens'] == max_tokens
