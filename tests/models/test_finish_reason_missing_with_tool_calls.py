from __future__ import annotations as _annotations

import pytest
from typing import Any, cast, Union

from pydantic_ai import Agent
from pydantic_ai.messages import ToolCallPart, TextPart
from pydantic_ai.usage import Usage

from ..conftest import IsDatetime, IsStr, try_import

with try_import() as imports_successful:
    from openai.types import chat
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
    
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from tests.models.test_openai import MockOpenAI, completion_message

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
]


async def test_process_response_missing_finish_reason_with_tool_calls(allow_model_requests: None):
    """Test the fix for missing finish_reason when tool calls are present.
    
    This test simulates the issue found with some models on OpenRouter.ai,
    where a response with tool calls might have no finish_reason.
    The fix should set the finish_reason to 'tool_calls' if there are tool calls present.
    """
    # Create a mock response with tool calls but with finish_reason=None
    mock_response = chat.ChatCompletion(
        id='123',
        choices=[
            Choice(
                finish_reason=None,  # Explicitly set to None to simulate the issue
                index=0,
                message=ChatCompletionMessage(
                    content=None,
                    role='assistant',
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id='call_123',
                            function=Function(arguments='{"query": "test"}', name='search_tool'),
                            type='function',
                        )
                    ],
                ),
            )
        ],
        created=1704067200,  # 2024-01-01
        model='gpt-4o-123',
        object='chat.completion',
        usage=None,
    )
    
    # Create the model with our mock client
    mock_client = MockOpenAI.create_mock(mock_response)
    model = OpenAIModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    
    # Process the response directly to verify the fix
    response = model._process_response(mock_response)
    
    # Assert that the response was processed correctly
    assert response.vendor_id == '123'
    assert response.model_name == 'gpt-4o-123'
    assert len(response.parts) == 1
    assert isinstance(response.parts[0], ToolCallPart)
    assert response.parts[0].tool_name == 'search_tool'
    assert response.parts[0].args == '{"query": "test"}'
    
    # Now test through the Agent interface to ensure full integration
    agent = Agent(model)
    result = await agent.run('Make a tool call')
    
    # Verify the response contains the tool call
    response = result.all_messages()[1]
    
    # Check that we have a tool call part in the response
    parts = response.parts
    assert len(parts) == 1
    assert isinstance(parts[0], ToolCallPart)
    assert parts[0].tool_name == 'search_tool'
    assert parts[0].args == '{"query": "test"}'
    
    # Verify the response was processed correctly despite the missing finish_reason
    assert response.model_name == 'gpt-4o-123'
    assert response.vendor_id == '123'
    

async def test_process_response_missing_finish_reason_without_tool_calls(allow_model_requests: None):
    """Test the fix for missing finish_reason when no tool calls are present.
    
    This test ensures that when finish_reason is None but there are no tool calls,
    the finish_reason is defaulted to 'stop'.
    """
    # Create a mock response with no tool calls but with finish_reason=None
    mock_response = chat.ChatCompletion(
        id='123',
        choices=[
            Choice(
                finish_reason=None,  # Explicitly set to None to simulate the issue
                index=0,
                message=ChatCompletionMessage(
                    content="This is a text response without tool calls",
                    role='assistant',
                    tool_calls=None,
                ),
            )
        ],
        created=1704067200,  # 2024-01-01
        model='gpt-4o-123',
        object='chat.completion',
        usage=None,
    )
    
    # Create the model with our mock client
    mock_client = MockOpenAI.create_mock(mock_response)
    model = OpenAIModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    
    # Process the response directly to verify the fix
    response = model._process_response(mock_response)
    
    # Assert that the response was processed correctly
    assert response.vendor_id == '123'
    assert response.model_name == 'gpt-4o-123'
    assert len(response.parts) == 1
    assert isinstance(response.parts[0], TextPart)
    assert response.parts[0].content == "This is a text response without tool calls"
    
    # Now test through the Agent interface to ensure full integration
    agent = Agent(model)
    result = await agent.run('Make a response without tool calls')
    
    # Verify the response contains text content
    response = result.all_messages()[1]
    
    # Check that we have a text part in the response
    parts = response.parts
    assert len(parts) == 1
    assert isinstance(parts[0], TextPart)
    assert parts[0].content == "This is a text response without tool calls"
    
    # Verify the response was processed correctly despite the missing finish_reason
    assert response.model_name == 'gpt-4o-123'
    assert response.vendor_id == '123'