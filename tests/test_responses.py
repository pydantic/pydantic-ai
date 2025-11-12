"""Tests for Responses implementation."""

from __future__ import annotations

import json
from http import HTTPStatus
from typing import Any

import httpx
import pytest
from asgi_lifespan import LifespanManager
from pydantic import BaseModel

from pydantic_ai import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai import models
from pydantic_ai.agent import Agent
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from .conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.ui.responses import ResponsesAdapter, ResponsesApp

models.ALLOW_MODEL_REQUESTS = False

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not imports_successful(), reason='starlette not installed'),
]


class WeatherResult(BaseModel):
    """Weather result model for testing."""

    location: str
    temperature: int


def get_weather(location: str) -> WeatherResult:
    """Get the weather for a location.

    Args:
        location: The location to get weather for.

    Returns:
        Weather information for the location.
    """
    return WeatherResult(location=location, temperature=20)


# ============================================================================
# Message Conversion Tests
# ============================================================================


async def test_load_messages_simple_string() -> None:
    """Test converting simple string input from Responses to Pydantic AI."""
    result = ResponsesAdapter.load_messages(["Hello, how are you?"])
    
    assert len(result) == 1
    assert isinstance(result[0], ModelRequest)
    assert len(result[0].parts) == 1
    assert isinstance(result[0].parts[0], UserPromptPart)
    assert result[0].parts[0].content == "Hello, how are you?"


async def test_load_messages_with_instructions() -> None:
    """Test converting string input with instructions from Responses to Pydantic AI."""
    result = ResponsesAdapter.load_messages([
        [{'role': 'system', 'content': 'You are a helpful assistant.'}],
        "Hello"
    ])
    
    assert len(result) == 1
    assert isinstance(result[0], ModelRequest)
    assert len(result[0].parts) == 2
    assert isinstance(result[0].parts[0], SystemPromptPart)
    assert result[0].parts[0].content == "You are a helpful assistant."
    assert isinstance(result[0].parts[1], UserPromptPart)
    assert result[0].parts[1].content == "Hello"


async def test_load_messages_user_message() -> None:
    """Test converting user message from Responses to Pydantic AI."""
    messages = [[
        {'type': 'message', 'role': 'user', 'content': 'What is the weather?'}
    ]]
    
    result = ResponsesAdapter.load_messages(messages)
    
    assert len(result) == 1
    assert isinstance(result[0], ModelRequest)
    assert len(result[0].parts) == 1
    assert isinstance(result[0].parts[0], UserPromptPart)
    assert result[0].parts[0].content == 'What is the weather?'


async def test_load_messages_system_message() -> None:
    """Test converting system message from Responses to Pydantic AI."""
    messages = [[
        {'type': 'message', 'role': 'system', 'content': 'You are a helpful assistant.'}
    ]]
    
    result = ResponsesAdapter.load_messages(messages)
    
    assert len(result) == 1
    assert isinstance(result[0], ModelRequest)
    assert len(result[0].parts) == 1
    assert isinstance(result[0].parts[0], SystemPromptPart)
    assert result[0].parts[0].content == 'You are a helpful assistant.'


async def test_load_messages_assistant_message() -> None:
    """Test converting assistant message from Responses to Pydantic AI."""
    messages = [[
        {'type': 'message', 'role': 'assistant', 'content': 'I can help with that.'}
    ]]
    
    result = ResponsesAdapter.load_messages(messages)
    
    assert len(result) == 1
    assert isinstance(result[0], ModelResponse)
    assert len(result[0].parts) == 1
    assert isinstance(result[0].parts[0], TextPart)
    assert result[0].parts[0].content == 'I can help with that.'


async def test_load_messages_function_call_output() -> None:
    """Test converting function call output from Responses to Pydantic AI."""
    messages = [[
        {
            'type': 'function_call_output',
            'call_id': 'call_123',
            'output': '{"location": "London", "temperature": 20}'
        }
    ]]
    
    result = ResponsesAdapter.load_messages(messages)
    
    assert len(result) == 1
    assert isinstance(result[0], ModelRequest)
    assert len(result[0].parts) == 1
    assert isinstance(result[0].parts[0], ToolReturnPart)
    assert result[0].parts[0].tool_call_id == 'call_123'
    assert result[0].parts[0].content == '{"location": "London", "temperature": 20}'


async def test_load_messages_conversation() -> None:
    """Test converting a full conversation from Responses to Pydantic AI."""
    messages = [[
        {'type': 'message', 'role': 'system', 'content': 'You are a helpful assistant.'},
        {'type': 'message', 'role': 'user', 'content': 'What is the weather in London?'},
        {'type': 'message', 'role': 'assistant', 'content': 'Let me check that for you.'},
        {
            'type': 'function_call_output',
            'call_id': 'call_123',
            'output': '{"location": "London", "temperature": 20}'
        },
        {'type': 'message', 'role': 'assistant', 'content': 'The weather in London is 20 degrees.'},
    ]]
    
    result = ResponsesAdapter.load_messages(messages)
    
    assert len(result) == 4
    # First request with system + user
    assert isinstance(result[0], ModelRequest)
    assert isinstance(result[0].parts[0], SystemPromptPart)
    assert isinstance(result[0].parts[1], UserPromptPart)
    # First assistant response
    assert isinstance(result[1], ModelResponse)
    assert isinstance(result[1].parts[0], TextPart)
    # Tool return
    assert isinstance(result[2], ModelRequest)
    assert isinstance(result[2].parts[0], ToolReturnPart)
    # Final assistant response
    assert isinstance(result[3], ModelResponse)
    assert isinstance(result[3].parts[0], TextPart)


async def test_load_messages_multimodal_user() -> None:
    """Test converting multimodal user message from Responses to Pydantic AI."""
    messages = [[
        {
            'type': 'message',
            'role': 'user',
            'content': [
                {'type': 'input_text', 'text': 'What is in this image?'},
                {'type': 'input_image', 'image_url': 'https://example.com/image.jpg'},
            ],
        }
    ]]
    
    result = ResponsesAdapter.load_messages(messages)
    
    assert len(result) == 1
    assert isinstance(result[0], ModelRequest)
    assert len(result[0].parts) == 1
    assert isinstance(result[0].parts[0], UserPromptPart)
    # Content should be a list
    content = result[0].parts[0].content
    assert isinstance(content, list)
    assert len(content) == 2


async def test_load_messages_assistant_content_array() -> None:
    """Test converting assistant message with content array from Responses to Pydantic AI."""
    messages = [[
        {
            'type': 'message',
            'role': 'assistant',
            'content': [
                {'type': 'text', 'text': 'First part. '},
                {'type': 'text', 'text': 'Second part.'},
            ],
        }
    ]]
    
    result = ResponsesAdapter.load_messages(messages)
    
    assert len(result) == 1
    assert isinstance(result[0], ModelResponse)
    assert len(result[0].parts) == 1
    assert isinstance(result[0].parts[0], TextPart)
    assert result[0].parts[0].content == 'First part. Second part.'


async def test_load_messages_developer_role() -> None:
    """Test converting developer role message (alias for system) from Responses to Pydantic AI."""
    messages = [[
        {'type': 'message', 'role': 'developer', 'content': 'You are a helpful assistant.'}
    ]]
    
    result = ResponsesAdapter.load_messages(messages)
    
    assert len(result) == 1
    assert isinstance(result[0], ModelRequest)
    assert len(result[0].parts) == 1
    assert isinstance(result[0].parts[0], SystemPromptPart)
    assert result[0].parts[0].content == 'You are a helpful assistant.'


async def test_load_messages_system_content_array() -> None:
    """Test converting system message with content array from Responses to Pydantic AI."""
    messages = [[
        {
            'type': 'message',
            'role': 'system',
            'content': [
                {'type': 'text', 'text': 'Part 1. '},
                {'type': 'text', 'text': 'Part 2.'},
            ],
        }
    ]]
    
    result = ResponsesAdapter.load_messages(messages)
    
    assert len(result) == 1
    assert isinstance(result[0], ModelRequest)
    assert len(result[0].parts) == 1
    assert isinstance(result[0].parts[0], SystemPromptPart)
    assert result[0].parts[0].content == 'Part 1. Part 2.'


async def test_load_messages_image_with_detail() -> None:
    """Test converting image with detail parameter from Responses to Pydantic AI."""
    from pydantic_ai.messages import ImageUrl
    
    messages = [[
        {
            'type': 'message',
            'role': 'user',
            'content': [
                {
                    'type': 'input_image',
                    'image_url': 'https://example.com/image.jpg',
                    'detail': 'high',
                },
            ],
        }
    ]]
    
    result = ResponsesAdapter.load_messages(messages)
    
    assert len(result) == 1
    assert isinstance(result[0], ModelRequest)
    assert len(result[0].parts) == 1
    assert isinstance(result[0].parts[0], UserPromptPart)
    content = result[0].parts[0].content
    assert isinstance(content, list)
    assert len(content) == 1
    assert isinstance(content[0], ImageUrl)
    assert content[0].url == 'https://example.com/image.jpg'
    assert content[0].vendor_metadata == {'detail': 'high'}


async def test_load_messages_image_data_uri() -> None:
    """Test converting image from data URI from Responses to Pydantic AI."""
    from pydantic_ai.messages import BinaryContent
    
    messages = [[
        {
            'type': 'message',
            'role': 'user',
            'content': [
                {
                    'type': 'input_image',
                    'image_url': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
                },
            ],
        }
    ]]
    
    result = ResponsesAdapter.load_messages(messages)
    
    assert len(result) == 1
    assert isinstance(result[0], ModelRequest)
    assert len(result[0].parts) == 1
    assert isinstance(result[0].parts[0], UserPromptPart)
    content = result[0].parts[0].content
    assert isinstance(content, list)
    assert len(content) == 1
    assert isinstance(content[0], BinaryContent)
    assert content[0].media_type == 'image/png'


async def test_load_messages_image_data_uri_with_detail() -> None:
    """Test converting image from data URI with detail parameter from Responses to Pydantic AI."""
    from pydantic_ai.messages import BinaryContent
    
    messages = [[
        {
            'type': 'message',
            'role': 'user',
            'content': [
                {
                    'type': 'input_image',
                    'image_url': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
                    'detail': 'low',
                },
            ],
        }
    ]]
    
    result = ResponsesAdapter.load_messages(messages)
    
    assert len(result) == 1
    assert isinstance(result[0], ModelRequest)
    assert len(result[0].parts) == 1
    assert isinstance(result[0].parts[0], UserPromptPart)
    content = result[0].parts[0].content
    assert isinstance(content, list)
    assert len(content) == 1
    assert isinstance(content[0], BinaryContent)
    assert content[0].media_type == 'image/png'
    assert content[0].vendor_metadata == {'detail': 'low'}


async def test_load_messages_file_data_uri() -> None:
    """Test converting file from data URI from Responses to Pydantic AI."""
    from pydantic_ai.messages import BinaryContent
    
    messages = [[
        {
            'type': 'message',
            'role': 'user',
            'content': [
                {
                    'type': 'input_file',
                    'file_data': 'data:application/pdf;base64,JVBERi0xLjQKJeLjz9MK',
                },
            ],
        }
    ]]
    
    result = ResponsesAdapter.load_messages(messages)
    
    assert len(result) == 1
    assert isinstance(result[0], ModelRequest)
    assert len(result[0].parts) == 1
    assert isinstance(result[0].parts[0], UserPromptPart)
    content = result[0].parts[0].content
    assert isinstance(content, list)
    assert len(content) == 1
    assert isinstance(content[0], BinaryContent)
    assert content[0].media_type == 'application/pdf'


async def test_load_messages_empty_assistant_content() -> None:
    """Test converting assistant message with empty content from Responses to Pydantic AI."""
    messages = [[
        {'type': 'message', 'role': 'assistant', 'content': ''}
    ]]
    
    result = ResponsesAdapter.load_messages(messages)
    
    # Empty content still creates a ModelResponse with empty TextPart
    assert len(result) == 1
    assert isinstance(result[0], ModelResponse)
    assert len(result[0].parts) == 1
    assert isinstance(result[0].parts[0], TextPart)
    assert result[0].parts[0].content == ''


async def test_load_messages_user_empty_content_list() -> None:
    """Test converting user message with empty content list from Responses to Pydantic AI."""
    messages = [[
        {
            'type': 'message',
            'role': 'user',
            'content': [],
        }
    ]]
    
    result = ResponsesAdapter.load_messages(messages)
    
    # Should skip empty user messages
    assert len(result) == 0


async def test_load_messages_mixed_valid_and_invalid_content_parts() -> None:
    """Test converting user message with mix of valid and invalid content parts from Responses to Pydantic AI."""
    messages = [[
        {
            'type': 'message',
            'role': 'user',
            'content': [
                {'type': 'input_text', 'text': 'Hello'},
                {'type': 'unknown_type', 'data': 'should be ignored'},
                {'type': 'input_text', 'text': ' World'},
            ],
        }
    ]]
    
    result = ResponsesAdapter.load_messages(messages)
    
    assert len(result) == 1
    assert isinstance(result[0], ModelRequest)
    assert len(result[0].parts) == 1
    assert isinstance(result[0].parts[0], UserPromptPart)
    content = result[0].parts[0].content
    assert isinstance(content, list)
    assert len(content) == 2
    assert content[0] == 'Hello'
    assert content[1] == ' World'


# ============================================================================
# Toolset Tests
# ============================================================================


async def test_toolset_parsing_single_function() -> None:
    """Test parsing a single function tool from Responses input."""
    body = json.dumps({
        'model': 'test',
        'input': 'Hello',
        'tools': [
            {
                'type': 'function',
                'function': {
                    'name': 'get_weather',
                    'description': 'Get the weather for a location.',
                    'strict': True,
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'location': {'type': 'string'},
                        },
                        'required': ['location'],
                    },
                },
            }
        ],
        'stream': True,
    }).encode()
    
    agent = Agent(TestModel())
    run_input = ResponsesAdapter.build_run_input(body)
    adapter = ResponsesAdapter(agent=agent, run_input=run_input, accept='text/event-stream')
    
    toolset = adapter.toolset
    assert toolset is not None
    assert len(toolset.tool_defs) == 1
    assert toolset.tool_defs[0].name == 'get_weather'
    assert toolset.tool_defs[0].description == 'Get the weather for a location.'
    assert toolset.tool_defs[0].parameters_json_schema == {
        'type': 'object',
        'properties': {
            'location': {'type': 'string'},
        },
        'required': ['location'],
    }


async def test_toolset_parsing_multiple_functions() -> None:
    """Test parsing multiple function tools from Responses input."""
    body = json.dumps({
        'model': 'test',
        'input': 'Hello',
        'tools': [
            {
                'type': 'function',
                'function': {
                    'name': 'get_weather',
                    'description': 'Get the weather.',
                    'parameters': {'type': 'object', 'properties': {}},
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'get_time',
                    'description': 'Get the current time.',
                    'parameters': {'type': 'object', 'properties': {}},
                },
            },
        ],
        'stream': True,
    }).encode()
    
    agent = Agent(TestModel())
    run_input = ResponsesAdapter.build_run_input(body)
    adapter = ResponsesAdapter(agent=agent, run_input=run_input, accept='text/event-stream')
    
    toolset = adapter.toolset
    assert toolset is not None
    assert len(toolset.tool_defs) == 2
    assert toolset.tool_defs[0].name == 'get_weather'
    assert toolset.tool_defs[1].name == 'get_time'


async def test_toolset_parsing_no_tools() -> None:
    """Test that toolset is None when no tools are provided."""
    body = json.dumps({
        'model': 'test',
        'input': 'Hello',
        'stream': True,
    }).encode()
    
    agent = Agent(TestModel())
    run_input = ResponsesAdapter.build_run_input(body)
    adapter = ResponsesAdapter(agent=agent, run_input=run_input, accept='text/event-stream')
    
    assert adapter.toolset is None


async def test_toolset_parsing_empty_tools_list() -> None:
    """Test that toolset is None when tools list is empty."""
    body = json.dumps({
        'model': 'test',
        'input': 'Hello',
        'tools': [],
        'stream': True,
    }).encode()
    
    agent = Agent(TestModel())
    run_input = ResponsesAdapter.build_run_input(body)
    adapter = ResponsesAdapter(agent=agent, run_input=run_input, accept='text/event-stream')
    
    assert adapter.toolset is None


async def test_toolset_parsing_non_function_tools() -> None:
    """Test that non-function type tools are ignored."""
    body = json.dumps({
        'model': 'test',
        'input': 'Hello',
        'tools': [
            {
                'type': 'other',
                'name': 'something',
            }
        ],
        'stream': True,
    }).encode()
    
    agent = Agent(TestModel())
    run_input = ResponsesAdapter.build_run_input(body)
    adapter = ResponsesAdapter(agent=agent, run_input=run_input, accept='text/event-stream')
    
    # Non-function tools create an empty toolset
    toolset = adapter.toolset
    assert toolset is not None
    assert len(toolset.tool_defs) == 0


# ============================================================================
# State Tests
# ============================================================================


async def test_state_from_metadata() -> None:
    """Test extracting state from metadata in Responses input."""
    body = json.dumps({
        'model': 'test',
        'input': 'Hello',
        'metadata': {
            'user_id': '123',
            'session_id': 'abc',
            'custom_data': 'some_value',
        },
        'stream': True,
    }).encode()
    
    agent = Agent(TestModel())
    run_input = ResponsesAdapter.build_run_input(body)
    adapter = ResponsesAdapter(agent=agent, run_input=run_input, accept='text/event-stream')
    
    state = adapter.state
    assert state is not None
    assert state == {
        'user_id': '123',
        'session_id': 'abc',
        'custom_data': 'some_value',
    }


async def test_state_no_metadata() -> None:
    """Test that state is None when no metadata is provided."""
    body = json.dumps({
        'model': 'test',
        'input': 'Hello',
        'stream': True,
    }).encode()
    
    agent = Agent(TestModel())
    run_input = ResponsesAdapter.build_run_input(body)
    adapter = ResponsesAdapter(agent=agent, run_input=run_input, accept='text/event-stream')
    
    assert adapter.state is None


async def test_state_empty_metadata() -> None:
    """Test that state is None when metadata is empty."""
    body = json.dumps({
        'model': 'test',
        'input': 'Hello',
        'metadata': {},
        'stream': True,
    }).encode()
    
    agent = Agent(TestModel())
    run_input = ResponsesAdapter.build_run_input(body)
    adapter = ResponsesAdapter(agent=agent, run_input=run_input, accept='text/event-stream')
    
    assert adapter.state is None


# ============================================================================
# Streaming Tests
# ============================================================================


async def test_responses_streaming() -> None:
    """Test streaming responses endpoint."""
    agent = Agent(TestModel())
    app = ResponsesApp(agent)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as client:
            client.base_url = 'http://test'
            response = await client.post(
                '/v1/responses',
                json={
                    'model': 'test',
                    'input': 'Hello',
                    'stream': True
                },
            )

            assert response.status_code == HTTPStatus.OK
            assert response.headers['content-type'] == 'text/event-stream; charset=utf-8'

            chunks: list[dict[str, Any]] = []
            async for line in response.aiter_lines():
                if line.startswith('data: ') and line != 'data: [DONE]':
                    chunk_data = json.loads(line[6:])  # Remove "data: " prefix
                    chunks.append(chunk_data)

            assert len(chunks) > 0
            # Check structure of chunks
            for chunk in chunks:
                assert 'type' in chunk
                if chunk['type'] == 'response.done':
                    assert 'response' in chunk


async def test_responses_with_instructions() -> None:
    """Test responses endpoint with instructions."""
    agent = Agent(TestModel())
    app = ResponsesApp(agent)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as client:
            client.base_url = 'http://test'
            response = await client.post(
                '/v1/responses',
                json={
                    'model': 'test',
                    'input': 'Hello',
                    'instructions': 'You are a helpful assistant.',
                    'stream': True
                },
            )

            assert response.status_code == HTTPStatus.OK


async def test_responses_with_list_input() -> None:
    """Test responses endpoint with list input."""
    agent = Agent(TestModel())
    app = ResponsesApp(agent)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as client:
            client.base_url = 'http://test'
            response = await client.post(
                '/v1/responses',
                json={
                    'model': 'test',
                    'input': [
                        {'type': 'message', 'role': 'user', 'content': 'Hello'}
                    ],
                    'stream': True
                },
            )

            assert response.status_code == HTTPStatus.OK


# ============================================================================
# Tool Tests
# ============================================================================


async def test_responses_with_function_call_output() -> None:
    """Test responses endpoint with function call output in conversation history."""
    def stream_function(messages: list[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
        # Check that tool return was properly loaded
        assert len(messages) >= 2
        last_request = messages[-1]
        assert isinstance(last_request, ModelRequest)
        assert any(isinstance(part, ToolReturnPart) for part in last_request.parts)
        
        return ModelResponse(
            parts=[TextPart(content='The weather in London is 20 degrees.')]
        )

    agent = Agent(FunctionModel(stream_function), tools=[get_weather])
    app = ResponsesApp(agent)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as client:
            client.base_url = 'http://test'
            response = await client.post(
                '/v1/responses',
                json={
                    'model': 'test',
                    'input': [
                        {'type': 'message', 'role': 'user', 'content': 'What is the weather in London?'},
                        {
                            'type': 'function_call_output',
                            'call_id': 'call_123',
                            'output': '{"location": "London", "temperature": 20}'
                        }
                    ],
                    'stream': True,
                },
            )

            assert response.status_code == HTTPStatus.OK


# ============================================================================
# State and Callback Tests
# ============================================================================


async def test_responses_callback_sync() -> None:
    """Test responses endpoint with synchronous callback."""
    callback_called = False
    
    def sync_callback(run_result: Any) -> None:
        nonlocal callback_called
        callback_called = True

    agent = Agent(TestModel())
    app = ResponsesApp(agent, on_complete=sync_callback)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as client:
            client.base_url = 'http://test'
            response = await client.post(
                '/v1/responses',
                json={
                    'model': 'test',
                    'input': 'Hello',
                    'stream': True
                },
            )

            assert response.status_code == HTTPStatus.OK
            # Consume the response
            async for _ in response.aiter_lines():
                pass

    assert callback_called


async def test_responses_callback_async() -> None:
    """Test responses endpoint with asynchronous callback."""
    callback_called = False
    
    async def async_callback(run_result: Any) -> None:
        nonlocal callback_called
        callback_called = True

    agent = Agent(TestModel())
    app = ResponsesApp(agent, on_complete=async_callback)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as client:
            client.base_url = 'http://test'
            response = await client.post(
                '/v1/responses',
                json={
                    'model': 'test',
                    'input': 'Hello',
                    'stream': True
                },
            )

            assert response.status_code == HTTPStatus.OK
            # Consume the response
            async for _ in response.aiter_lines():
                pass

    assert callback_called


# ============================================================================
# Error Handling Tests
# ============================================================================


async def test_responses_missing_required_fields() -> None:
    """Test responses endpoint with missing required fields."""
    agent = Agent(TestModel())
    app = ResponsesApp(agent)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as client:
            client.base_url = 'http://test'
            response = await client.post(
                '/v1/responses',
                json={
                    'model': 'test',
                    # Missing 'input' field
                    'stream': True
                },
            )
            assert response.status_code == 422
