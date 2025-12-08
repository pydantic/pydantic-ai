from __future__ import annotations as _annotations

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from ..conftest import try_import
from .test_anthropic import MockAnthropic, completion_message

with try_import() as imports_successful:
    from anthropic.types.beta import (
        BetaTextBlock,
        BetaUsage,
        BetaMessage,
    )

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.anyio,
]

async def test_pause_turn_retry_loop(allow_model_requests: None):
    # Mock a sequence of responses:
    # 1. pause_turn response
    # 2. final response
    
    c1 = completion_message(
        [BetaTextBlock(text='paused', type='text')],
        usage=BetaUsage(input_tokens=10, output_tokens=5),
    )
    c1.stop_reason = 'pause_turn' # type: ignore
    
    c2 = completion_message(
        [BetaTextBlock(text='final', type='text')],
        usage=BetaUsage(input_tokens=10, output_tokens=5),
    )
    
    mock_client = MockAnthropic.create_mock([c1, c2])
    m = AnthropicModel('claude-3-5-sonnet-20241022', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    result = await agent.run('test prompt')
    
    # Verify the agent received the final response
    assert result.output == 'final'
    
    # Verify the loop happened (2 requests)
    assert len(mock_client.chat_completion_kwargs) == 2
    
    # Verify history in second request includes the paused message
    messages_2 = mock_client.chat_completion_kwargs[1]['messages']
    # Should be: User -> Assistant(paused)
    assert len(messages_2) == 2
    assert messages_2[1]['role'] == 'assistant'
    # Content is a list of BetaContentBlock objects, get the text from first block
    content_blocks = messages_2[1]['content']
    assert len(content_blocks) > 0
    first_block = content_blocks[0]
    assert hasattr(first_block, 'text') and first_block.text == 'paused'
