from __future__ import annotations as _annotations

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, ModelResponse
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from ..conftest import try_import
from .test_anthropic import MockAnthropic, completion_message

with try_import() as imports_successful:
    from anthropic.types.beta import (
        BetaBashCodeExecutionToolResultBlock,
        BetaTextBlock,
        BetaUsage,
        BetaMessage,
    )

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.anyio,
]

@pytest.mark.filterwarnings('ignore::UserWarning')
async def test_bash_code_execution_tool_result(allow_model_requests: None):
    # Mock response with BetaBashCodeExecutionToolResultBlock
    # Use model_construct to bypass validation entirely for testing
    
    # BetaBashCodeExecutionToolResultBlock has content list
    content_block = {'type': 'text', 'text': 'output'}
    
    block = BetaBashCodeExecutionToolResultBlock.model_construct(
        tool_use_id='tool-123',
        type='bash_code_execution_tool_result',
        content=[content_block],
    )
    
    c = completion_message(
        [block, BetaTextBlock(text='final response', type='text')],
        usage=BetaUsage(input_tokens=10, output_tokens=5),
    )
    
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-3-5-sonnet-20241022', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    result = await agent.run('test prompt')
    
    last_message = result.all_messages()[-1]
    assert isinstance(last_message, ModelResponse)
    # We expect 2 parts now: tool return and text
    assert len(last_message.parts) == 2
    
    from pydantic_ai.messages import BuiltinToolReturnPart
    part = next(p for p in last_message.parts if isinstance(p, BuiltinToolReturnPart))
    assert isinstance(part, BuiltinToolReturnPart)
    assert part.tool_name == 'code_execution'
    # We expect the content to be dumped as json, without is_error
    assert part.content == snapshot({'content': [{'error_code': None, 'type': 'text', 'text': 'output'}]})
