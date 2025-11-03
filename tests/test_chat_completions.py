"""Tests for Chat Completions implementation."""

from __future__ import annotations

import json
from http import HTTPStatus
from typing import Any

import httpx
import pytest
from asgi_lifespan import LifespanManager
from inline_snapshot import snapshot
from openai import AsyncOpenAI
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
    from pydantic_ai.ui.chat_completions import ChatCompletionsAdapter, ChatCompletionsApp

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


async def test_load_messages_user_text() -> None:
    """Test converting user text message from Chat Completions to Pydantic AI."""
    messages = [
        {'role': 'user', 'content': 'Hello, how are you?'}
    ]
    
    result = ChatCompletionsAdapter.load_messages(messages)
    
    assert len(result) == 1
    assert isinstance(result[0], ModelRequest)
    assert len(result[0].parts) == 1
    assert isinstance(result[0].parts[0], UserPromptPart)
    assert result[0].parts[0].content == 'Hello, how are you?'


async def test_load_messages_system() -> None:
    """Test converting system message from Chat Completions to Pydantic AI."""
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'}
    ]
    
    result = ChatCompletionsAdapter.load_messages(messages)
    
    assert len(result) == 1
    assert isinstance(result[0], ModelRequest)
    assert len(result[0].parts) == 1
    assert isinstance(result[0].parts[0], SystemPromptPart)
    assert result[0].parts[0].content == 'You are a helpful assistant.'


async def test_load_messages_assistant_text() -> None:
    """Test converting assistant text message from Chat Completions to Pydantic AI."""
    messages = [
        {'role': 'assistant', 'content': 'I am doing well, thank you!'}
    ]
    
    result = ChatCompletionsAdapter.load_messages(messages)
    
    assert len(result) == 1
    assert isinstance(result[0], ModelResponse)
    assert len(result[0].parts) == 1
    assert isinstance(result[0].parts[0], TextPart)
    assert result[0].parts[0].content == 'I am doing well, thank you!'


async def test_load_messages_assistant_with_tool_calls() -> None:
    """Test converting assistant message with tool calls from Chat Completions to Pydantic AI."""
    messages = [
        {
            'role': 'assistant',
            'content': None,
            'tool_calls': [
                {
                    'id': 'call_123',
                    'type': 'function',
                    'function': {
                        'name': 'get_weather',
                        'arguments': '{"location": "London"}',
                    },
                }
            ],
        }
    ]
    
    result = ChatCompletionsAdapter.load_messages(messages)
    
    assert len(result) == 1
    assert isinstance(result[0], ModelResponse)
    assert len(result[0].parts) == 1
    assert isinstance(result[0].parts[0], ToolCallPart)
    assert result[0].parts[0].tool_call_id == 'call_123'
    assert result[0].parts[0].tool_name == 'get_weather'
    assert result[0].parts[0].args == {'location': 'London'}


async def test_load_messages_tool_response() -> None:
    """Test converting tool response message from Chat Completions to Pydantic AI."""
    messages = [
        {
            'role': 'tool',
            'tool_call_id': 'call_123',
            'content': '{"location": "London", "temperature": 20}',
        }
    ]
    
    result = ChatCompletionsAdapter.load_messages(messages)
    
    assert len(result) == 1
    assert isinstance(result[0], ModelRequest)
    assert len(result[0].parts) == 1
    assert isinstance(result[0].parts[0], ToolReturnPart)
    assert result[0].parts[0].tool_call_id == 'call_123'
    assert result[0].parts[0].content == '{"location": "London", "temperature": 20}'


async def test_load_messages_conversation() -> None:
    """Test converting a full conversation from Chat Completions to Pydantic AI."""
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'What is the weather in London?'},
        {
            'role': 'assistant',
            'content': None,
            'tool_calls': [
                {
                    'id': 'call_123',
                    'type': 'function',
                    'function': {
                        'name': 'get_weather',
                        'arguments': '{"location": "London"}',
                    },
                }
            ],
        },
        {
            'role': 'tool',
            'tool_call_id': 'call_123',
            'content': '{"location": "London", "temperature": 20}',
        },
        {'role': 'assistant', 'content': 'The weather in London is 20 degrees.'},
    ]
    
    result = ChatCompletionsAdapter.load_messages(messages)
    
    # MessagesBuilder combines consecutive request parts into a single ModelRequest
    assert len(result) == 4
    assert isinstance(result[0], ModelRequest)
    assert isinstance(result[0].parts[0], SystemPromptPart)
    assert isinstance(result[0].parts[1], UserPromptPart)
    assert isinstance(result[1], ModelResponse)
    assert isinstance(result[1].parts[0], ToolCallPart)
    assert isinstance(result[2], ModelRequest)
    assert isinstance(result[2].parts[0], ToolReturnPart)
    assert isinstance(result[3], ModelResponse)
    assert isinstance(result[3].parts[0], TextPart)


async def test_load_messages_multimodal() -> None:
    """Test converting multimodal user message from Chat Completions to Pydantic AI."""
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'What is in this image?'},
                {'type': 'image_url', 'image_url': {'url': 'https://example.com/image.jpg'}},
            ],
        }
    ]
    
    result = ChatCompletionsAdapter.load_messages(messages)
    
    assert len(result) == 1
    assert isinstance(result[0], ModelRequest)
    assert len(result[0].parts) == 1
    assert isinstance(result[0].parts[0], UserPromptPart)
    # Content should be a list with text and image
    content = result[0].parts[0].content
    assert isinstance(content, list)
    assert len(content) == 2
    assert content[0] == 'What is in this image?'


async def test_load_messages_assistant_content_array() -> None:
    """Test converting assistant message with content array from Chat Completions to Pydantic AI."""
    messages = [
        {
            'role': 'assistant',
            'content': [
                {'type': 'text', 'text': 'First part'},
                {'type': 'text', 'text': 'Second part'},
            ],
        }
    ]
    
    result = ChatCompletionsAdapter.load_messages(messages)
    
    assert len(result) == 1
    assert isinstance(result[0], ModelResponse)
    assert len(result[0].parts) == 1
    assert isinstance(result[0].parts[0], TextPart)
    assert result[0].parts[0].content == 'First part Second part'


async def test_load_messages_function_deprecated() -> None:
    """Test converting deprecated function message from Chat Completions to Pydantic AI."""
    messages = [
        {
            'role': 'function',
            'name': 'get_weather',
            'content': '{"location": "London", "temperature": 20}',
        }
    ]
    
    result = ChatCompletionsAdapter.load_messages(messages)
    
    assert len(result) == 1
    assert isinstance(result[0], ModelRequest)
    assert len(result[0].parts) == 1
    assert isinstance(result[0].parts[0], ToolReturnPart)
    assert result[0].parts[0].tool_name == 'get_weather'




# ============================================================================
# Streaming Tests
# ============================================================================


async def test_chat_completions_streaming() -> None:
    """Test streaming chat completions endpoint."""
    agent = Agent(TestModel())
    app = ChatCompletionsApp(agent)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as client:
            client.base_url = 'http://test'
            response = await client.post(
                '/v1/chat/completions',
                json={'model': 'test', 'messages': [{'role': 'user', 'content': 'Hello'}], 'stream': True},
            )

            assert response.status_code == HTTPStatus.OK
            assert response.headers['content-type'] == 'text/event-stream; charset=utf-8'

            chunks: list[dict[str, Any]] = []
            async for line in response.aiter_lines():
                if line.startswith('data: ') and line != 'data: [DONE]':
                    chunk_data = json.loads(line[6:])  # Remove "data: " prefix
                    chunks.append(chunk_data)

            assert len(chunks) > 0
            # First chunk should have role
            assert chunks[0]['choices'][0]['delta'].get('role') == 'assistant'
            # Last chunk should have finish_reason
            assert chunks[-1]['choices'][0]['finish_reason'] == 'stop'
            # All chunks should have correct structure
            for chunk in chunks:
                assert chunk['object'] == 'chat.completion.chunk'
                assert chunk['model'] == 'test'


# ============================================================================
# Tool Tests
# ============================================================================


async def test_chat_completions_tool_call_streaming() -> None:
    """Test streaming chat completions with tool calls."""
    def stream_function(messages: list[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_call_id='call_123',
                    tool_name='get_weather',
                    args={'location': 'London'},
                )
            ]
        )

    agent = Agent(FunctionModel(stream_function), tools=[get_weather])
    app = ChatCompletionsApp(agent)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as client:
            client.base_url = 'http://test'
            response = await client.post(
                '/v1/chat/completions',
                json={
                    'model': 'test',
                    'messages': [{'role': 'user', 'content': 'What is the weather in London?'}],
                    'stream': True,
                    'tools': [
                        {
                            'type': 'function',
                            'function': {
                                'name': 'get_weather',
                                'description': 'Get the weather for a location.',
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
                },
            )

            assert response.status_code == HTTPStatus.OK

            tool_calls_found = False
            async for line in response.aiter_lines():
                if line.startswith('data: ') and line != 'data: [DONE]':
                    chunk_data = json.loads(line[6:])
                    delta = chunk_data['choices'][0]['delta']
                    if 'tool_calls' in delta:
                        tool_calls_found = True

            assert tool_calls_found




# ============================================================================
# OpenAI Client Integration Tests
# ============================================================================


async def test_openai_client_streaming() -> None:
    """Test using OpenAI client with streaming chat completions."""
    agent = Agent(TestModel())
    app = ChatCompletionsApp(agent)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as httpx_client:
            httpx_client.base_url = 'http://test'
            client = AsyncOpenAI(api_key='sk-test', http_client=httpx_client)

            stream = await client.chat.completions.create(
                model='test',
                messages=[{'role': 'user', 'content': 'Hello'}],
                stream=True,
            )

            chunks = []
            async for chunk in stream:
                chunks.append(chunk)

            assert len(chunks) > 0
            assert chunks[0].object == 'chat.completion.chunk'
            assert chunks[-1].choices[0].finish_reason == 'stop'
