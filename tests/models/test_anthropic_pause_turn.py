from __future__ import annotations as _annotations

import pytest

from pydantic_ai import Agent, ModelResponse
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from ..conftest import try_import
from .test_anthropic import MockAnthropic, completion_message

with try_import() as imports_successful:
    from anthropic.types.beta import BetaTextBlock, BetaUsage

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.anyio,
]


async def test_pause_turn_continues_run(allow_model_requests: None):
    c1 = completion_message([BetaTextBlock(text='paused', type='text')], BetaUsage(input_tokens=10, output_tokens=5))
    c1.stop_reason = 'pause_turn'  # type: ignore[assignment]
    c2 = completion_message([BetaTextBlock(text='final', type='text')], BetaUsage(input_tokens=10, output_tokens=5))

    mock_client = MockAnthropic.create_mock([c1, c2])
    model = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(model)

    result = await agent.run('test prompt')

    assert result.output == 'final'

    pause_response = next(
        m
        for m in result.all_messages()
        if isinstance(m, ModelResponse) and (m.provider_details or {}).get('finish_reason') == 'pause_turn'
    )
    assert pause_response.finish_reason == 'incomplete'

    assert len(mock_client.chat_completion_kwargs) == 2  # type: ignore[arg-type]
    messages_2 = mock_client.chat_completion_kwargs[1]['messages']  # type: ignore[index]
    assert len(messages_2) == 2
    assert messages_2[1]['role'] == 'assistant'
    content_blocks = messages_2[1]['content']
    assert isinstance(content_blocks, list)
    assert content_blocks[0]['type'] == 'text'
    assert content_blocks[0]['text'] == 'paused'
