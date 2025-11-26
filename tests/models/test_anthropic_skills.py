from __future__ import annotations as _annotations

from typing import Any, cast
from dataclasses import dataclass, field
from datetime import timezone

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, ModelResponse
from pydantic_ai.builtin_tools import CodeExecutionTool
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider

from ..conftest import try_import
from .test_anthropic import MockAnthropic, completion_message, get_mock_chat_completion_kwargs

with try_import() as imports_successful:
    from anthropic.types.beta import (
        BetaCodeExecutionToolResultBlock,
        BetaBashCodeExecutionToolResultBlock,
        BetaTextBlock,
        BetaUsage,
        BetaMessage,
    )

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.anyio,
]

async def test_code_execution_with_skills(allow_model_requests: None):
    c = completion_message(
        [BetaTextBlock(text='response', type='text')],
        usage=BetaUsage(input_tokens=10, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-3-5-sonnet-20241022', provider=AnthropicProvider(anthropic_client=mock_client))
    
    # Create agent with CodeExecutionTool having skills
    # Skills should be a list of skill_id strings
    tool = CodeExecutionTool(skills=['pptx', 'xlsx'])
    agent = Agent(m, builtin_tools=[tool])

    await agent.run('test prompt')

    # Verify skills were passed in container, NOT in tool definition
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    tools = completion_kwargs['tools']
    extra_headers = completion_kwargs['extra_headers']
    container = completion_kwargs['container']
    
    assert tools == snapshot(
        [
            {
                'name': 'code_execution',
                'type': 'code_execution_20250825',
                # 'skills' should NOT be here
            }
        ]
    )
    assert 'skills' not in tools[0]
    assert 'skills-2025-10-02' in extra_headers['anthropic-beta']
    assert container == snapshot({
        'skills': [
            {'type': 'anthropic', 'skill_id': 'pptx', 'version': 'latest'},
            {'type': 'anthropic', 'skill_id': 'xlsx', 'version': 'latest'}
        ]
    })

async def test_container_persistence(allow_model_requests: None):
    # Mock response with container ID
    # We need to mock the response object to have 'container' attribute
    # Since BetaMessage is a Pydantic model, we can't easily add attributes if they are not defined.
    # But we can use a custom mock or rely on the fact that we updated AnthropicModel to check hasattr.
    
    # Create a mock message that has a container attribute
    class MockMessageWithContainer(BetaMessage):
        container: Any = None

    c = MockMessageWithContainer(
        id='123',
        content=[BetaTextBlock(text='response', type='text')],
        model='claude-3-5-sonnet-20241022',
        role='assistant',
        stop_reason='end_turn',
        type='message',
        usage=BetaUsage(input_tokens=10, output_tokens=5),
        container=type('Container', (), {'id': 'container-123'}),
    )
    
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-3-5-sonnet-20241022', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    result = await agent.run('test prompt')
    
    # Verify container ID is stored in provider_details
    last_message = result.all_messages()[-1]
    assert isinstance(last_message, ModelResponse)
    assert last_message.provider_details['anthropic_container_id'] == 'container-123'

    # Now run again passing the container ID
    # We need to manually pass it in model_settings as per our implementation
    # The user (or agent loop) needs to pass it.
    
    mock_client.index = 0
    await agent.run(
        'next prompt', 
        model_settings=AnthropicModelSettings(anthropic_container={'id': 'container-123'})
    )
    
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[1]
    assert completion_kwargs['container'] == {'id': 'container-123'}

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
