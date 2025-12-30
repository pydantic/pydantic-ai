"""Tests for xAI model integration.

Note on builtin tools testing:
------------------------------
xAI's builtin tools (code_execution, web_search, mcp_server) are executed server-side via gRPC.
Since VCR doesn't support gRPC, we cannot record/replay these interactions like we do with HTTP APIs.

For builtin tool tests, we use simplified mocks that verify:
1. Tools are properly registered with the xAI SDK
2. The agent can process responses when builtin tools are enabled
3. Builtin tools can coexist with custom (client-side) tools
"""

from __future__ import annotations as _annotations

import json
import os
from datetime import timezone
from typing import Any

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel
from typing_extensions import TypedDict

from pydantic_ai import (
    Agent,
    AgentRunResult,
    AgentRunResultEvent,
    AudioUrl,
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    CodeExecutionTool,
    DocumentUrl,
    FinalResultEvent,
    ImageUrl,
    MCPServerTool,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelRetry,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
    WebSearchTool,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    BuiltinToolResultEvent,  # pyright: ignore[reportDeprecated]
    CachePoint,
)
from pydantic_ai.models import ModelRequestParameters, ToolDefinition
from pydantic_ai.output import NativeOutput, PromptedOutput, ToolOutput
from pydantic_ai.profiles.grok import GrokModelProfile
from pydantic_ai.result import RunUsage
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

from ..conftest import IsDatetime, IsInstance, IsNow, IsStr, try_import
from .mock_xai import (
    MockXai,
    create_code_execution_responses,
    create_failed_builtin_tool_response,
    create_logprob,
    create_mcp_server_responses,
    create_mixed_tools_response,
    create_response,
    create_response_with_tool_calls,
    create_response_without_usage,
    create_server_tool_call,
    create_stream_chunk,
    create_tool_call,
    create_web_search_responses,
    get_mock_chat_create_kwargs,
)

with try_import() as imports_successful:
    import xai_sdk.chat as chat_types
    from xai_sdk.proto.v6 import chat_pb2, usage_pb2

    from pydantic_ai.models.xai import XaiModel, XaiModelSettings
    from pydantic_ai.providers.xai import XaiProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='xai_sdk not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
    pytest.mark.filterwarnings(
        'ignore:`BuiltinToolCallEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolCallPart` instead.:DeprecationWarning'
    ),
    pytest.mark.filterwarnings(
        'ignore:`BuiltinToolResultEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolReturnPart` instead.:DeprecationWarning'
    ),
]

# Test model constants
XAI_NON_REASONING_MODEL = 'grok-4-1-fast-non-reasoning'
XAI_REASONING_MODEL = 'grok-4-1-fast-reasoning'


def create_usage(
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    reasoning_tokens: int = 0,
    cached_prompt_text_tokens: int = 0,
    server_side_tools_used: list[usage_pb2.ServerSideTool] | None = None,
) -> usage_pb2.SamplingUsage:
    """Helper to create xAI SamplingUsage protobuf objects for tests with all required fields."""
    return usage_pb2.SamplingUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        cached_prompt_text_tokens=cached_prompt_text_tokens,
        server_side_tools_used=server_side_tools_used or [],
    )


def test_xai_init():
    provider = XaiProvider(api_key='foobar')
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=provider)

    assert m.model_name == XAI_NON_REASONING_MODEL
    assert m.system == 'xai'


def test_xai_init_with_fixture_api_key(xai_api_key: str):
    """Test that xai_api_key fixture is properly used."""
    provider = XaiProvider(api_key=xai_api_key)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=provider)

    assert m.model_name == XAI_NON_REASONING_MODEL
    assert m.system == 'xai'


def test_create_tool_call_part_failed_status(allow_model_requests: None):
    """Ensure failed server-side tool calls carry provider status/error into return parts."""

    response = create_failed_builtin_tool_response(
        tool_name=CodeExecutionTool.kind,
        tool_type=chat_pb2.TOOL_CALL_TYPE_CODE_EXECUTION_TOOL,
        tool_call_id='code_exec_1',
        error_message='sandbox error',
        content='tool failed',
    )

    mock_client = MockXai.create_mock([response])
    model = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(model)

    result = agent.run_sync('hello')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(content='tool failed'),
                    BuiltinToolReturnPart(
                        tool_name=CodeExecutionTool.kind,
                        content='tool failed',
                        tool_call_id='code_exec_1',
                        provider_name='xai',
                        provider_details={'status': 'failed', 'error': 'sandbox error'},
                        timestamp=IsDatetime(),
                    ),
                ],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_request_simple_success(allow_model_requests: None):
    response = create_response(content='world')
    mock_client = MockXai.create_mock([response, response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    result = await agent.run('hello')
    assert result.output == 'world'
    assert result.usage() == snapshot(RunUsage(requests=1))

    result = await agent.run('hello', message_history=result.new_messages())
    assert result.output == 'world'
    assert result.usage() == snapshot(RunUsage(requests=1))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_request_simple_usage(allow_model_requests: None):
    response = create_response(
        content='world',
        usage=create_usage(prompt_tokens=2, completion_tokens=1),
    )
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    result = await agent.run('Hello')
    assert result.output == 'world'
    assert result.usage() == snapshot(
        RunUsage(
            requests=1,
            input_tokens=2,
            output_tokens=1,
        )
    )


async def test_xai_image_input(allow_model_requests: None):
    """Test that xAI model handles image inputs (text is extracted from content)."""
    response = create_response(content='done')
    mock_client = MockXai.create_mock([response])
    model = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(model)

    image_url = ImageUrl('https://example.com/image.png')
    binary_image = BinaryContent(b'\x89PNG', media_type='image/png')

    result = await agent.run(['Describe these inputs.', image_url, binary_image])
    assert result.output == 'done'

    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [
                    {
                        'content': [
                            {'text': 'Describe these inputs.'},
                            {'image_url': {'image_url': 'https://example.com/image.png', 'detail': 'DETAIL_AUTO'}},
                            {'image_url': {'image_url': 'data:image/png;base64,iVBORw==', 'detail': 'DETAIL_AUTO'}},
                        ],
                        'role': 'ROLE_USER',
                    }
                ],
                'tools': None,
                'tool_choice': None,
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            }
        ]
    )


async def test_xai_request_structured_response_tool_output(allow_model_requests: None):
    """Test structured output using ToolOutput (tool-based structured output)."""

    class CityLocation(BaseModel):
        city: str
        country: str

    # First response: call the get_user_country tool
    response1 = create_response(
        tool_calls=[create_tool_call('call_get_country', 'get_user_country', {})],
        usage=create_usage(prompt_tokens=70, completion_tokens=12),
    )
    # Second response: return structured output via final_result tool
    response2 = create_response(
        tool_calls=[create_tool_call('call_final', 'final_result', {'city': 'Mexico City', 'country': 'Mexico'})],
        usage=create_usage(prompt_tokens=90, completion_tokens=15),
    )
    mock_client = MockXai.create_mock([response1, response2])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    # Use ToolOutput explicitly to use tool-based structured output (not native)
    agent = Agent(m, output_type=ToolOutput(CityLocation))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run('What is the largest city in the user country?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country?',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_user_country', args='{}', tool_call_id='call_get_country')],
                usage=RequestUsage(input_tokens=70, output_tokens=12),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='call_get_country',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"city": "Mexico City", "country": "Mexico"}',
                        tool_call_id='call_final',
                    )
                ],
                usage=RequestUsage(input_tokens=90, output_tokens=15),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='call_final',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )

    # With ToolOutput, we should send tool definitions, not response_format
    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': 'grok-4-1-fast-non-reasoning',
                'messages': [
                    {'content': [{'text': 'What is the largest city in the user country?'}], 'role': 'ROLE_USER'}
                ],
                'tools': [
                    {
                        'function': {
                            'name': 'get_user_country',
                            'parameters': '{"additionalProperties": false, "properties": {}, "type": "object"}',
                        }
                    },
                    {
                        'function': {
                            'name': 'final_result',
                            'description': 'The final response which ends this conversation',
                            'parameters': '{"properties": {"city": {"type": "string"}, "country": {"type": "string"}}, "required": ["city", "country"], "title": "CityLocation", "type": "object"}',
                        }
                    },
                ],
                'tool_choice': 'required',
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            },
            {
                'model': 'grok-4-1-fast-non-reasoning',
                'messages': [
                    {'content': [{'text': 'What is the largest city in the user country?'}], 'role': 'ROLE_USER'},
                    {
                        'content': [{'text': ''}],
                        'role': 'ROLE_ASSISTANT',
                        'tool_calls': [
                            {
                                'id': 'call_get_country',
                                'type': 'TOOL_CALL_TYPE_CLIENT_SIDE_TOOL',
                                'status': 'TOOL_CALL_STATUS_COMPLETED',
                                'function': {'name': 'get_user_country', 'arguments': '{}'},
                            }
                        ],
                    },
                    {
                        'content': [{'text': 'Mexico'}],
                        'role': 'ROLE_TOOL',
                    },
                ],
                'tools': [
                    {
                        'function': {
                            'name': 'get_user_country',
                            'parameters': '{"additionalProperties": false, "properties": {}, "type": "object"}',
                        }
                    },
                    {
                        'function': {
                            'name': 'final_result',
                            'description': 'The final response which ends this conversation',
                            'parameters': '{"properties": {"city": {"type": "string"}, "country": {"type": "string"}}, "required": ["city", "country"], "title": "CityLocation", "type": "object"}',
                        }
                    },
                ],
                'tool_choice': 'required',
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            },
        ]
    )


async def test_xai_request_structured_response_native_output(allow_model_requests: None):
    """Test structured output using native JSON schema output (the default for xAI)."""

    class CityLocation(BaseModel):
        city: str
        country: str

    # First response: call the get_user_country tool
    response1 = create_response(
        tool_calls=[create_tool_call('call_get_country', 'get_user_country', {})],
        usage=create_usage(prompt_tokens=70, completion_tokens=12),
    )
    # Second response: native output returns JSON text (not a tool call)
    response2 = create_response(
        content='{"city":"Mexico City","country":"Mexico"}',
        usage=create_usage(prompt_tokens=90, completion_tokens=15),
    )
    mock_client = MockXai.create_mock([response1, response2])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    # Plain output_type uses native output by default for xAI (per GrokModelProfile)
    agent = Agent(m, output_type=CityLocation)

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run('What is the largest city in the user country?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country?',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_user_country', args='{}', tool_call_id='call_get_country')],
                usage=RequestUsage(input_tokens=70, output_tokens=12),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='call_get_country',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"city":"Mexico City","country":"Mexico"}')],
                usage=RequestUsage(input_tokens=90, output_tokens=15),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    # Verify API request parameters
    kwargs = get_mock_chat_create_kwargs(mock_client)
    # With native output + tools, both requests should have response_format with JSON schema
    assert kwargs == snapshot(
        [
            {
                'model': 'grok-4-1-fast-non-reasoning',
                'messages': [
                    {'content': [{'text': 'What is the largest city in the user country?'}], 'role': 'ROLE_USER'}
                ],
                'tools': [
                    {
                        'function': {
                            'name': 'get_user_country',
                            'parameters': '{"additionalProperties": false, "properties": {}, "type": "object"}',
                        }
                    }
                ],
                'tool_choice': 'auto',
                'response_format': {
                    'format_type': 'FORMAT_TYPE_JSON_SCHEMA',
                    'schema': '{"properties": {"city": {"type": "string"}, "country": {"type": "string"}}, "required": ["city", "country"], "title": "CityLocation", "type": "object"}',
                },
                'use_encrypted_content': False,
                'include': [],
            },
            {
                'model': 'grok-4-1-fast-non-reasoning',
                'messages': [
                    {'content': [{'text': 'What is the largest city in the user country?'}], 'role': 'ROLE_USER'},
                    {
                        'content': [{'text': ''}],
                        'role': 'ROLE_ASSISTANT',
                        'tool_calls': [
                            {
                                'id': 'call_get_country',
                                'type': 'TOOL_CALL_TYPE_CLIENT_SIDE_TOOL',
                                'status': 'TOOL_CALL_STATUS_COMPLETED',
                                'function': {'name': 'get_user_country', 'arguments': '{}'},
                            }
                        ],
                    },
                    {
                        'content': [{'text': 'Mexico'}],
                        'role': 'ROLE_TOOL',
                    },
                ],
                'tools': [
                    {
                        'function': {
                            'name': 'get_user_country',
                            'parameters': '{"additionalProperties": false, "properties": {}, "type": "object"}',
                        }
                    }
                ],
                'tool_choice': 'auto',
                'response_format': {
                    'format_type': 'FORMAT_TYPE_JSON_SCHEMA',
                    'schema': '{"properties": {"city": {"type": "string"}, "country": {"type": "string"}}, "required": ["city", "country"], "title": "CityLocation", "type": "object"}',
                },
                'use_encrypted_content': False,
                'include': [],
            },
        ]
    )


async def test_xai_request_tool_call(allow_model_requests: None):
    responses = [
        create_response(
            tool_calls=[create_tool_call(id='1', name='get_location', arguments={'loc_name': 'San Fransisco'})],
            usage=create_usage(prompt_tokens=2, completion_tokens=1),
        ),
        create_response(
            tool_calls=[create_tool_call(id='2', name='get_location', arguments={'loc_name': 'London'})],
            usage=create_usage(prompt_tokens=3, completion_tokens=2),
        ),
        create_response(content='final response'),
    ]
    mock_client = MockXai.create_mock(responses)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, system_prompt='this is the system prompt')

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        if loc_name == 'London':
            return json.dumps({'lat': 51, 'lng': 0})
        else:
            raise ModelRetry('Wrong location, please try again')

    result = await agent.run('Hello')
    assert result.output == 'final response'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='this is the system prompt', timestamp=IsNow(tz=timezone.utc)),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args='{"loc_name": "San Fransisco"}',
                        tool_call_id='1',
                    )
                ],
                usage=RequestUsage(
                    input_tokens=2,
                    output_tokens=1,
                ),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Wrong location, please try again',
                        tool_name='get_location',
                        tool_call_id='1',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args='{"loc_name": "London"}',
                        tool_call_id='2',
                    )
                ],
                usage=RequestUsage(
                    input_tokens=3,
                    output_tokens=2,
                ),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_location',
                        content='{"lat": 51, "lng": 0}',
                        tool_call_id='2',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='final response')],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )
    assert result.usage() == snapshot(RunUsage(requests=3, input_tokens=5, output_tokens=3, tool_calls=1))

    # Verify tool definitions were passed correctly to the API
    kwargs = get_mock_chat_create_kwargs(mock_client)[0]
    assert kwargs['tools'] == snapshot(
        [
            {
                'function': {
                    'name': 'get_location',
                    'parameters': '{"additionalProperties": false, "properties": {"loc_name": {"type": "string"}}, "required": ["loc_name"], "type": "object"}',
                }
            }
        ]
    )


# Helpers for creating Grok streaming chunks
def grok_text_chunk(text: str, finish_reason: str = 'stop') -> tuple[chat_types.Response, chat_types.Chunk]:
    """Create a text streaming chunk for Grok.

    Note: For streaming, Response accumulates content, Chunk is the delta.
    Since we can't easily track state across calls, we pass full accumulated text as response.content
    and the delta as chunk.content.

    Returns:
        Tuple of (accumulated Response, delta Chunk) - both using real SDK types
    """
    # Create chunk (delta) - the incremental content
    chunk = create_stream_chunk(
        content=text,
        finish_reason=finish_reason if finish_reason else None,  # type: ignore[arg-type]
    )

    # Create response (accumulated) - for simplicity in mocks, we'll just use the same text
    # In real usage, the Response object would accumulate over multiple chunks
    response = create_response(
        content=text,
        finish_reason=finish_reason if finish_reason else 'stop',  # type: ignore[arg-type]
        usage=create_usage(prompt_tokens=2, completion_tokens=1) if finish_reason else None,
    )

    return (response, chunk)


def grok_reasoning_text_chunk(
    text: str, reasoning_content: str = '', encrypted_content: str = '', finish_reason: str = 'stop'
) -> tuple[chat_types.Response, chat_types.Chunk]:
    """Create a text streaming chunk for Grok with reasoning content.

    Args:
        text: The text content delta
        reasoning_content: The reasoning trace delta
        encrypted_content: The encrypted reasoning signature delta
        finish_reason: The finish reason

    Returns:
        Tuple of (accumulated Response, delta Chunk) - both using real SDK types
    """
    # Create chunk (delta) - includes reasoning content as delta
    chunk = create_stream_chunk(
        content=text,
        reasoning_content=reasoning_content,
        encrypted_content=encrypted_content,
        finish_reason=finish_reason if finish_reason else None,  # type: ignore[arg-type]
    )

    # Create response (accumulated) - includes reasoning content
    response = create_response(
        content=text,
        finish_reason=finish_reason if finish_reason else 'stop',  # type: ignore[arg-type]
        usage=create_usage(prompt_tokens=2, completion_tokens=1) if finish_reason else None,
        reasoning_content=reasoning_content,
        encrypted_content=encrypted_content,
    )

    return (response, chunk)


def _generate_tool_result_content(tool_call: chat_pb2.ToolCall) -> str:
    """Generate appropriate JSON content for completed server-side tool calls.

    Args:
        tool_call: The chat_pb2.ToolCall proto object

    Returns:
        JSON string with tool result content
    """
    tool_name = tool_call.function.name
    tool_type = tool_call.type

    # Code execution tool
    if tool_type == chat_pb2.ToolCallType.TOOL_CALL_TYPE_CODE_EXECUTION_TOOL:
        return json.dumps({'output': 'Code execution result for: ', 'return_code': 0, 'stderr': ''})
    # MCP tool
    elif tool_type == chat_pb2.ToolCallType.TOOL_CALL_TYPE_MCP_TOOL:
        return json.dumps(
            {'result': {'content': [{'type': 'text', 'text': f'MCP tool {tool_name} executed successfully'}]}}
        )
    # Web search tool
    elif tool_type == chat_pb2.ToolCallType.TOOL_CALL_TYPE_WEB_SEARCH_TOOL:
        return json.dumps(
            {
                'status': 'completed',
                'results': [
                    {
                        'title': 'Sample Search Result',
                        'url': 'https://example.com',
                        'snippet': 'Sample search result snippet',
                    }
                ],
            }
        )
    # Default for other tool types (x_search, collections_search, etc.)
    else:
        return json.dumps({'status': 'completed'})


def grok_builtin_tool_chunk(
    tool_call: chat_pb2.ToolCall,
    response_id: str = 'grok-builtin',
    finish_reason: str = '',
) -> tuple[chat_types.Response, chat_types.Chunk]:
    """Create a streaming chunk for Grok with a builtin (server-side) tool call.

    Args:
        tool_call: The server-side tool call proto object (chat_pb2.ToolCall)
        response_id: The response ID
        finish_reason: The finish reason (usually empty for tool call chunks)

    Returns:
        Tuple of (accumulated Response, delta Chunk) - both using real SDK types
    """
    # Generate content for completed tools based on tool type
    content = ''
    if tool_call.status == chat_pb2.ToolCallStatus.TOOL_CALL_STATUS_COMPLETED:
        content = _generate_tool_result_content(tool_call)

    # Create chunk (delta) - the tool call
    chunk = create_stream_chunk(
        content='',
        tool_calls=[tool_call],
        finish_reason=finish_reason if finish_reason else None,  # type: ignore[arg-type]
    )

    # Create response (accumulated) - same tool call with content
    response = create_response(
        content=content,
        tool_calls=[tool_call],
        finish_reason=finish_reason if finish_reason else 'stop',  # type: ignore[arg-type]
        usage=create_usage(prompt_tokens=2, completion_tokens=1) if finish_reason else None,
    )

    return (response, chunk)


async def test_xai_stream_text(allow_model_requests: None):
    stream = [grok_text_chunk('hello '), grok_text_chunk('world')]
    mock_client = MockXai.create_mock_stream([stream])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(['hello ', 'hello world'])
        assert result.is_complete
        assert result.usage() == snapshot(RunUsage(requests=1, input_tokens=2, output_tokens=1))


async def test_xai_stream_text_finish_reason(allow_model_requests: None):
    # Create streaming chunks with finish reasons
    stream = [
        grok_text_chunk('hello ', ''),
        grok_text_chunk('world', ''),
        grok_text_chunk('.', 'stop'),
    ]
    mock_client = MockXai.create_mock_stream([stream])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(
            ['hello ', 'hello world', 'hello world.']
        )
        assert result.is_complete
        async for response, is_last in result.stream_responses(debounce_by=None):
            if is_last:
                assert response == snapshot(
                    ModelResponse(
                        parts=[TextPart(content='hello world.')],
                        usage=RequestUsage(input_tokens=2, output_tokens=1),
                        model_name=XAI_NON_REASONING_MODEL,
                        timestamp=IsDatetime(),
                        provider_name='xai',
                        provider_url='https://api.x.ai/v1',
                        provider_response_id='grok-123',
                        finish_reason='stop',
                    )
                )


def grok_tool_chunk(
    tool_name: str | None, tool_arguments: str | None, finish_reason: str = '', accumulated_args: str = ''
) -> tuple[chat_types.Response, chat_types.Chunk]:
    """Create a tool call streaming chunk for Grok.

    Args:
        tool_name: The tool name (should be provided in all chunks for proper tracking)
        tool_arguments: The delta of arguments for this chunk
        finish_reason: The finish reason (only in last chunk)
        accumulated_args: The accumulated arguments string up to and including this chunk

    Note: Unlike the real xAI SDK which only sends the tool name in the first chunk,
    our mock includes it in every chunk to ensure proper tool call tracking.

    Returns:
        Tuple of (accumulated Response, delta Chunk) - both using real SDK types
    """
    # Infer tool name from accumulated state if not provided
    effective_tool_name = tool_name or ('final_result' if accumulated_args else None)

    # Create the chunk tool call (delta) - using real proto type
    chunk_tool_calls: list[chat_pb2.ToolCall] = []
    if effective_tool_name is not None or tool_arguments is not None:
        chunk_tool_call = chat_pb2.ToolCall(
            id='tool-123',
            type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL,
            function=chat_pb2.FunctionCall(
                name=effective_tool_name or '',
                arguments=tool_arguments if tool_arguments is not None else '',
            ),
        )
        chunk_tool_calls = [chunk_tool_call]

    # Chunk (delta) - using real proto type
    chunk = create_stream_chunk(
        content='',
        tool_calls=chunk_tool_calls,
        finish_reason=finish_reason if finish_reason else None,  # type: ignore[arg-type]
    )

    # Response (accumulated) - create tool calls for response
    response_tool_calls: list[chat_pb2.ToolCall] = []
    if effective_tool_name is not None or accumulated_args:
        response_tool_call = chat_pb2.ToolCall(
            id='tool-123',
            type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL,
            function=chat_pb2.FunctionCall(
                name=effective_tool_name or '',
                arguments=accumulated_args,  # Full accumulated arguments
            ),
        )
        response_tool_calls = [response_tool_call]

    response = create_response(
        content='',
        tool_calls=response_tool_calls,
        finish_reason=finish_reason if finish_reason else 'stop',  # type: ignore[arg-type]
        usage=create_usage(prompt_tokens=20, completion_tokens=1) if finish_reason else None,
    )

    return (response, chunk)


class MyTypedDict(TypedDict, total=False):
    first: str
    second: str


def test_grok_tool_chunk_empty_params():
    """Test grok_tool_chunk with all None/empty params to cover edge case branches."""
    # This exercises the branches where tool_name=None, tool_arguments=None, accumulated_args=''
    response, chunk = grok_tool_chunk(None, None, '', '')
    # Should produce empty tool call lists
    assert response.tool_calls == []
    assert chunk.tool_calls == []


async def test_xai_stream_structured(allow_model_requests: None):
    stream = [
        grok_tool_chunk('final_result', None, accumulated_args=''),
        grok_tool_chunk(None, '{"first": "One', accumulated_args='{"first": "One'),
        grok_tool_chunk(None, '", "second": "Two"', accumulated_args='{"first": "One", "second": "Two"'),
        grok_tool_chunk(None, '}', finish_reason='stop', accumulated_args='{"first": "One", "second": "Two"}'),
    ]
    mock_client = MockXai.create_mock_stream([stream])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, output_type=MyTypedDict)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [dict(c) async for c in result.stream_output(debounce_by=None)] == snapshot(
            [{'first': 'One', 'second': 'Two'}]
        )
        assert result.is_complete
        assert result.usage() == snapshot(RunUsage(requests=1, input_tokens=20, output_tokens=1))


async def test_xai_stream_structured_finish_reason(allow_model_requests: None):
    stream = [
        grok_tool_chunk('final_result', None, accumulated_args=''),
        grok_tool_chunk(None, '{"first": "One', accumulated_args='{"first": "One'),
        grok_tool_chunk(None, '", "second": "Two"', accumulated_args='{"first": "One", "second": "Two"'),
        grok_tool_chunk(None, '}', accumulated_args='{"first": "One", "second": "Two"}'),
        grok_tool_chunk(None, None, finish_reason='stop', accumulated_args='{"first": "One", "second": "Two"}'),
    ]
    mock_client = MockXai.create_mock_stream([stream])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, output_type=MyTypedDict)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [dict(c) async for c in result.stream_output(debounce_by=None)] == snapshot(
            [{'first': 'One', 'second': 'Two'}]
        )
        assert result.is_complete


async def test_xai_stream_native_output(allow_model_requests: None):
    stream = [
        grok_text_chunk('{"first": "One'),
        grok_text_chunk('", "second": "Two"'),
        grok_text_chunk('}'),
    ]
    mock_client = MockXai.create_mock_stream([stream])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, output_type=NativeOutput(MyTypedDict))

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [dict(c) async for c in result.stream_output(debounce_by=None)] == snapshot(
            [{'first': 'One'}, {'first': 'One', 'second': 'Two'}, {'first': 'One', 'second': 'Two'}]
        )
        assert result.is_complete


async def test_xai_stream_tool_call_with_empty_text(allow_model_requests: None):
    stream = [
        grok_tool_chunk('final_result', None, accumulated_args=''),
        grok_tool_chunk(None, '{"first": "One', accumulated_args='{"first": "One'),
        grok_tool_chunk(None, '", "second": "Two"', accumulated_args='{"first": "One", "second": "Two"'),
        grok_tool_chunk(None, '}', finish_reason='stop', accumulated_args='{"first": "One", "second": "Two"}'),
    ]
    mock_client = MockXai.create_mock_stream([stream])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, output_type=[str, MyTypedDict])

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_output(debounce_by=None)] == snapshot(
            [{'first': 'One'}, {'first': 'One', 'second': 'Two'}, {'first': 'One', 'second': 'Two'}]
        )
    assert await result.get_output() == snapshot({'first': 'One', 'second': 'Two'})


async def test_xai_no_delta(allow_model_requests: None):
    stream = [
        grok_text_chunk('hello '),
        grok_text_chunk('world'),
    ]
    mock_client = MockXai.create_mock_stream([stream])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(['hello ', 'hello world'])
        assert result.is_complete
        assert result.usage() == snapshot(RunUsage(requests=1, input_tokens=2, output_tokens=1))


async def test_xai_none_delta(allow_model_requests: None):
    # Test handling of chunks without deltas
    stream = [
        grok_text_chunk('hello '),
        grok_text_chunk('world'),
    ]
    mock_client = MockXai.create_mock_stream([stream])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(['hello ', 'hello world'])
        assert result.is_complete
        assert result.usage() == snapshot(RunUsage(requests=1, input_tokens=2, output_tokens=1))


@pytest.mark.parametrize('parallel_tool_calls', [True, False])
async def test_xai_parallel_tool_calls(allow_model_requests: None, parallel_tool_calls: bool) -> None:
    tool_call = create_tool_call(
        id='123',
        name='final_result',
        arguments={'response': [1, 2, 3]},
    )
    response = create_response(content='', tool_calls=[tool_call], finish_reason='tool_call')
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, output_type=list[int], model_settings=ModelSettings(parallel_tool_calls=parallel_tool_calls))

    await agent.run('Hello')
    assert get_mock_chat_create_kwargs(mock_client)[0]['parallel_tool_calls'] == parallel_tool_calls


async def test_xai_penalty_parameters(allow_model_requests: None) -> None:
    response = create_response(content='test response')
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))

    settings = ModelSettings(
        temperature=0.7,
        presence_penalty=0.5,
        frequency_penalty=0.3,
        parallel_tool_calls=False,
    )

    agent = Agent(m, model_settings=settings)
    result = await agent.run('Hello')

    # Check that all settings were passed to the xAI SDK
    kwargs = get_mock_chat_create_kwargs(mock_client)[0]
    assert kwargs['temperature'] == 0.7
    assert kwargs['presence_penalty'] == 0.5
    assert kwargs['frequency_penalty'] == 0.3
    assert kwargs['parallel_tool_calls'] is False
    assert result.output == 'test response'


async def test_xai_instructions(allow_model_requests: None):
    """Test that instructions are passed through to xAI SDK as a system message."""
    response = create_response(content='The capital of France is Paris.')
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, instructions='You are a helpful assistant.')

    result = await agent.run('What is the capital of France?')
    # Verify the message history has instructions
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is the capital of France?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='The capital of France is Paris.')],
                usage=RequestUsage(),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    # Verify instructions are passed as a system message in the API request
    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [
                    {'content': [{'text': 'You are a helpful assistant.'}], 'role': 'ROLE_SYSTEM'},
                    {'content': [{'text': 'What is the capital of France?'}], 'role': 'ROLE_USER'},
                ],
                'tools': None,
                'tool_choice': None,
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            }
        ]
    )


async def test_xai_image_url_input(allow_model_requests: None):
    response = create_response(content='world')
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    result = await agent.run(
        [
            'hello',
            ImageUrl(url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg'),
        ]
    )
    assert result.output == 'world'

    # Verify the generated API payload contains the image URL
    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [
                    {
                        'content': [
                            {'text': 'hello'},
                            {
                                'image_url': {
                                    'image_url': 'https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg',
                                    'detail': 'DETAIL_AUTO',
                                }
                            },
                        ],
                        'role': 'ROLE_USER',
                    }
                ],
                'tools': None,
                'tool_choice': None,
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            }
        ]
    )


async def test_xai_image_detail_vendor_metadata(allow_model_requests: None):
    """Test that xAI model handles image detail setting from vendor_metadata for ImageUrl."""
    response = create_response(content='done')
    mock_client = MockXai.create_mock([response])
    model = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(model)

    # Test both 'high' and 'low' detail settings
    image_high = ImageUrl('https://example.com/high.png', vendor_metadata={'detail': 'high'})
    image_low = ImageUrl('https://example.com/low.png', vendor_metadata={'detail': 'low'})

    await agent.run(['Describe these images.', image_high, image_low])

    # Verify the generated API payload contains the correct detail settings
    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [
                    {
                        'content': [
                            {'text': 'Describe these images.'},
                            {'image_url': {'image_url': 'https://example.com/high.png', 'detail': 'DETAIL_HIGH'}},
                            {'image_url': {'image_url': 'https://example.com/low.png', 'detail': 'DETAIL_LOW'}},
                        ],
                        'role': 'ROLE_USER',
                    }
                ],
                'tools': None,
                'tool_choice': None,
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            }
        ]
    )


async def test_xai_image_url_tool_response(allow_model_requests: None):
    """Test xAI with image URL from tool response."""
    # First response: model calls tool
    tool_call_response = create_response(
        content='',
        tool_calls=[create_tool_call(id='tool_001', name='get_image', arguments={})],
    )
    # Second response: model responds after seeing image
    final_response = create_response(content='The image shows a potato.')

    mock_client = MockXai.create_mock([tool_call_response, final_response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    @agent.tool_plain
    async def get_image() -> ImageUrl:
        return ImageUrl(url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg')

    result = await agent.run(['What food is in the image you can get from the get_image tool?'])

    # Verify the complete message history with snapshot
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=['What food is in the image you can get from the get_image tool?'],
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_image', args='{}', tool_call_id='tool_001')],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_image',
                        content='See file bd38f5',
                        tool_call_id='tool_001',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content=[
                            'This is file bd38f5:',
                            ImageUrl(
                                url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg'
                            ),
                        ],
                        timestamp=IsDatetime(),
                    ),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='The image shows a potato.')],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_image_as_binary_content_tool_response(allow_model_requests: None, image_content: BinaryContent):
    """Test xAI with binary image content from tool response."""
    # First response: model calls tool
    tool_call_response = create_response(
        content='',
        tool_calls=[create_tool_call(id='tool_001', name='get_image', arguments={})],
    )
    # Second response: model responds after seeing image
    final_response = create_response(content='The image shows a kiwi fruit.')

    mock_client = MockXai.create_mock([tool_call_response, final_response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    @agent.tool_plain
    async def get_image() -> BinaryContent:
        return image_content

    result = await agent.run(['What fruit is in the image you can get from the get_image tool?'])

    # Verify the complete message history with snapshot
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=['What fruit is in the image you can get from the get_image tool?'],
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_image', args='{}', tool_call_id='tool_001')],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_image',
                        content='See file 241a70',
                        tool_call_id='tool_001',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content=[
                            'This is file 241a70:',
                            IsInstance(BinaryContent),
                        ],
                        timestamp=IsDatetime(),
                    ),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='The image shows a kiwi fruit.')],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_image_as_binary_content_input(allow_model_requests: None, image_content: BinaryContent):
    """Test passing binary image content directly as input (not from a tool)."""
    response = create_response(content='The image shows a kiwi fruit.')

    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    result = await agent.run(['What fruit is in the image?', image_content])
    assert result.output == 'The image shows a kiwi fruit.'

    # Verify the generated API payload contains the image as a data URI
    kwargs = get_mock_chat_create_kwargs(mock_client)
    messages = kwargs[0]['messages']
    assert len(messages) == 1
    content = messages[0]['content']
    assert content[0] == {'text': 'What fruit is in the image?'}
    # Verify the image is base64-encoded as a data URI (don't snapshot the full base64)
    assert 'image_url' in content[1]
    assert content[1]['image_url']['image_url'].startswith('data:image/jpeg;base64,')
    assert content[1]['image_url']['detail'] == 'DETAIL_AUTO'


async def test_xai_document_url_input(allow_model_requests: None):
    """Test passing a document URL to the xAI model."""
    response = create_response(content='This document is a dummy PDF file.')

    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    document_url = DocumentUrl(url='https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf')

    result = await agent.run(['What is the main content on this document?', document_url])
    assert result.output
    # The document contains "Dummy PDF file"
    response_text = result.output.lower()
    assert 'dummy' in response_text or 'pdf' in response_text


async def test_xai_binary_content_document_input(allow_model_requests: None):
    """Test passing a document as BinaryContent to the xAI model."""
    response = create_response(content='The document discusses testing.')
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    document_content = BinaryContent(
        data=b'%PDF-1.4\nTest document content',
        media_type='application/pdf',
    )

    result = await agent.run(['What is in this document?', document_content])
    assert result.output == 'The document discusses testing.'

    # Verify the generated API payload contains the file reference
    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [
                    {
                        'content': [
                            {'text': 'What is in this document?'},
                            {'file': {'file_id': 'file-86a6ad'}},
                        ],
                        'role': 'ROLE_USER',
                    }
                ],
                'tools': None,
                'tool_choice': None,
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            }
        ]
    )


async def test_xai_audio_url_not_supported(allow_model_requests: None):
    """Test that AudioUrl raises NotImplementedError."""
    response = create_response(content='This should not be reached')
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    audio_url = AudioUrl(url='https://example.com/audio.mp3')

    with pytest.raises(NotImplementedError, match='AudioUrl is not supported by xAI SDK'):
        await agent.run(['What is in this audio?', audio_url])


async def test_xai_video_url_not_supported(allow_model_requests: None):
    """Test that VideoUrl raises NotImplementedError."""
    response = create_response(content='This should not be reached')
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    video_url = VideoUrl(url='https://example.com/video.mp4')

    with pytest.raises(NotImplementedError, match='VideoUrl is not supported by xAI SDK'):
        await agent.run(['What is in this video?', video_url])


async def test_xai_binary_content_audio_not_supported(allow_model_requests: None):
    """Test that BinaryContent with audio raises NotImplementedError."""
    response = create_response(content='This should not be reached')
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    audio_content = BinaryContent(
        data=b'fake audio data',
        media_type='audio/mpeg',
    )

    with pytest.raises(NotImplementedError, match='AudioUrl/BinaryContent with audio is not supported by xAI SDK'):
        await agent.run(['What is in this audio?', audio_content])


async def test_xai_response_with_logprobs(allow_model_requests: None):
    """Test that logprobs are correctly extracted from xAI responses."""
    response = create_response(
        content='Belo Horizonte.',
        logprobs=[
            create_logprob('Belo', -0.5),
            create_logprob(' Horizonte', -0.25),
            create_logprob('.', -0.0),
        ],
    )
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    result = await agent.run('What is the capital of Minas Gerais?')
    messages = result.all_messages()
    response_msg = messages[1]
    assert isinstance(response_msg, ModelResponse)
    text_part = response_msg.parts[0]
    assert isinstance(text_part, TextPart)
    assert text_part.provider_details is not None
    assert 'logprobs' in text_part.provider_details
    assert text_part.provider_details['logprobs'] == snapshot(
        {
            'content': [
                {'token': 'Belo', 'logprob': -0.5, 'bytes': [66, 101, 108, 111], 'top_logprobs': []},
                {
                    'token': ' Horizonte',
                    'logprob': -0.25,
                    'bytes': [32, 72, 111, 114, 105, 122, 111, 110, 116, 101],
                    'top_logprobs': [],
                },
                {'token': '.', 'logprob': -0.0, 'bytes': [46], 'top_logprobs': []},
            ]
        }
    )


# Grok built-in tools tests
# Built-in tools are executed server-side by xAI's infrastructure
# Based on: https://github.com/xai-org/xai-sdk-python/blob/main/examples/aio/server_side_tools.py


async def test_xai_builtin_web_search_tool(allow_model_requests: None):
    """Test xAI's built-in web_search tool."""
    # Create response with outputs
    response = create_web_search_responses(
        query='date of Jan 1 in 2026',
        content='Thursday',
        tool_call_id='ws_001',
    )
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, builtin_tools=[WebSearchTool()])

    result = await agent.run('Return just the day of week for the date of Jan 1 in 2026?')
    assert result.output
    assert 'thursday' in result.output.lower()

    # Verify the builtin tool call and result appear in message history
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Return just the day of week for the date of Jan 1 in 2026?',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'date of Jan 1 in 2026'},
                        tool_call_id='ws_001',
                        provider_name='xai',
                    ),
                    TextPart(content='Thursday'),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content='Thursday',
                        tool_call_id='ws_001',
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                ],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-ws_001',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_builtin_web_search_tool_stream(allow_model_requests: None):
    """Test xAI's built-in web_search tool with streaming."""
    # Create a mock web search server-side tool call
    web_search_tool_call = create_server_tool_call(
        tool_name='web_search',
        arguments={'query': 'weather San Francisco'},
        tool_call_id='ws_stream_001',
        tool_type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_WEB_SEARCH_TOOL,
    )

    # For streaming with builtin tools, the tool call appears in the first chunk
    # and then subsequent chunks contain the text response
    stream = [
        # First chunk: builtin tool call
        grok_builtin_tool_chunk(web_search_tool_call, response_id='grok-ws_stream_001'),
        # Subsequent chunks: text response
        grok_text_chunk('The weather '),
        grok_text_chunk('is sunny.', finish_reason='stop'),
    ]

    mock_client = MockXai.create_mock_stream([stream])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, builtin_tools=[WebSearchTool()])

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='What is the weather in San Francisco?') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        # Capture all events for validation
                        event_parts.append(event)

    # Verify we got the expected builtin tool call events with snapshot
    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=BuiltinToolReturnPart(
                    tool_name='web_search',
                    content={
                        'status': 'completed',
                        'results': [
                            {
                                'title': 'Sample Search Result',
                                'url': 'https://example.com',
                                'snippet': 'Sample search result snippet',
                            }
                        ],
                    },
                    tool_call_id='ws_stream_001',
                    timestamp=IsDatetime(),
                    provider_name='xai',
                ),
            ),
            PartStartEvent(index=1, part=TextPart(content='The weather '), previous_part_kind='builtin-tool-return'),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='is sunny.')),
            PartEndEvent(index=1, part=TextPart(content='The weather is sunny.')),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='web_search',
                    content={
                        'status': 'completed',
                        'results': [
                            {
                                'title': 'Sample Search Result',
                                'url': 'https://example.com',
                                'snippet': 'Sample search result snippet',
                            }
                        ],
                    },
                    tool_call_id='ws_stream_001',
                    timestamp=IsDatetime(),
                    provider_name='xai',
                )
            ),
        ]
    )


async def test_xai_builtin_x_search_tool(allow_model_requests: None):
    """Test xAI's built-in x_search tool (X/Twitter search)."""
    # Note: XSearchTool is not yet implemented in pydantic-ai
    # This test documents the expected behavior when it is implemented
    pytest.skip('XSearchTool not yet implemented in pydantic-ai')


async def test_xai_builtin_code_execution_tool(allow_model_requests: None):
    """Test xAI's built-in code_execution tool."""
    # Create response with outputs
    response = create_code_execution_responses(
        code='65465 - 6544 * 65464 - 6 + 1.02255',
        content='The result is -428,050,955.97745',
        tool_call_id='code_001',
    )
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, builtin_tools=[CodeExecutionTool()])

    result = await agent.run('What is 65465 - 6544 * 65464 - 6 + 1.02255? Use code to calculate this.')

    # Verify the builtin tool call and result appear in message history
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is 65465 - 6544 * 65464 - 6 + 1.02255? Use code to calculate this.',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={'code': '65465 - 6544 * 65464 - 6 + 1.02255'},
                        tool_call_id='code_001',
                        provider_name='xai',
                    ),
                    TextPart(content='The result is -428,050,955.97745'),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content='The result is -428,050,955.97745',
                        tool_call_id='code_001',
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                ],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-code_001',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_builtin_code_execution_tool_stream(allow_model_requests: None):
    """Test xAI's built-in code_execution tool with streaming."""
    # Create a mock code execution server-side tool call
    code_tool_call = create_server_tool_call(
        tool_name='code_execution',
        arguments={'code': '2 + 2'},
        tool_call_id='code_stream_001',
        tool_type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_CODE_EXECUTION_TOOL,
    )

    # For streaming with builtin tools, the tool call appears in the first chunk
    stream = [
        # First chunk: builtin tool call
        grok_builtin_tool_chunk(code_tool_call, response_id='grok-code_stream_001'),
        # Subsequent chunks: text response
        grok_text_chunk('The result '),
        grok_text_chunk('is 4', finish_reason='stop'),
    ]

    mock_client = MockXai.create_mock_stream([stream])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, builtin_tools=[CodeExecutionTool()])

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='Calculate 2 + 2 using code') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    # Verify we got the expected builtin tool call events with snapshot
    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'output': 'Code execution result for: ', 'return_code': 0, 'stderr': ''},
                    tool_call_id='code_stream_001',
                    timestamp=IsDatetime(),
                    provider_name='xai',
                ),
            ),
            PartStartEvent(index=1, part=TextPart(content='The result '), previous_part_kind='builtin-tool-return'),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='is 4')),
            PartEndEvent(index=1, part=TextPart(content='The result is 4')),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'output': 'Code execution result for: ', 'return_code': 0, 'stderr': ''},
                    tool_call_id='code_stream_001',
                    timestamp=IsDatetime(),
                    provider_name='xai',
                )
            ),
        ]
    )


async def test_xai_builtin_multiple_tools(allow_model_requests: None):
    """Test using multiple built-in tools together."""
    # Create a response that simulates using both web search and code execution
    web_search_tool = create_server_tool_call(
        tool_name='web_search',
        arguments={'query': 'current price of Bitcoin'},
        tool_call_id='ws_002',
        tool_type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_WEB_SEARCH_TOOL,
    )
    code_tool = create_server_tool_call(
        tool_name='code_execution',
        arguments={'code': '((65000 - 50000) / 50000) * 100'},
        tool_call_id='code_002',
        tool_type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_CODE_EXECUTION_TOOL,
    )

    response = create_mixed_tools_response(
        server_tools=[web_search_tool, code_tool],
        text_content='Bitcoin has increased by 30.0% from last week.',
    )

    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(
        m,
        instructions='You are a helpful assistant.',
        builtin_tools=[WebSearchTool(), CodeExecutionTool()],
    )

    result = await agent.run(
        'Search for the current price of Bitcoin and calculate its percentage change if it was $50000 last week.'
    )

    # Verify both builtin tool calls appear in message history
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Search for the current price of Bitcoin and calculate its percentage change if it was $50000 last week.',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsDatetime(),
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content=None,
                        tool_call_id='ws_002',
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content=None,
                        tool_call_id='code_002',
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    TextPart(content='Bitcoin has increased by 30.0% from last week.'),
                ],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-multi-tool',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_builtin_tools_with_custom_tools(allow_model_requests: None):
    """Test mixing xAI's built-in tools with custom (client-side) tools.

    This test verifies that both builtin tools (web_search) and custom tools
    (get_local_temperature) can be used in the same conversation.
    """
    # Response 1: Model calls the custom tool
    response1 = create_response_with_tool_calls(
        tool_calls=[
            chat_pb2.ToolCall(
                id='custom_001',
                type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL,
                function=chat_pb2.FunctionCall(
                    name='get_local_temperature',
                    arguments='{"city": "Tokyo"}',
                ),
            )
        ],
        finish_reason='tool_call',
    )
    # Response 2: Model uses builtin web_search and provides final answer
    response2 = create_web_search_responses(
        query='Tokyo weather forecast',
        content='Based on the local temperature of 72°F and the forecast, Tokyo weather is sunny.',
        tool_call_id='ws_003',
    )

    mock_client = MockXai.create_mock([response1, response2])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, builtin_tools=[WebSearchTool()])

    # Track if custom tool was called
    tool_was_called = False

    @agent.tool_plain
    def get_local_temperature(city: str) -> str:
        """Get the local temperature for a city."""
        nonlocal tool_was_called
        tool_was_called = True
        return f'The local temperature in {city} is 72°F'

    result = await agent.run('What is the weather in Tokyo?')

    # Verify custom tool was actually called
    assert tool_was_called, 'Custom tool get_local_temperature should have been called'

    # Verify full message history with both custom and builtin tool calls
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is the weather in Tokyo?', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_local_temperature',
                        args='{"city": "Tokyo"}',
                        tool_call_id='custom_001',
                    )
                ],
                usage=RequestUsage(),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_local_temperature',
                        content='The local temperature in Tokyo is 72°F',
                        tool_call_id='custom_001',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'Tokyo weather forecast'},
                        tool_call_id='ws_003',
                        provider_name='xai',
                    ),
                    TextPart(
                        content='Based on the local temperature of 72°F and the forecast, Tokyo weather is sunny.'
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content='Based on the local temperature of 72°F and the forecast, Tokyo weather is sunny.',
                        tool_call_id='ws_003',
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                ],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-ws_003',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_builtin_mcp_server_tool(allow_model_requests: None):
    """Test xAI's MCP server tool with Linear."""
    # Create response with outputs
    response = create_mcp_server_responses(
        server_id='linear',
        tool_name='list_issues',
        content='No issues found.',
        tool_call_id='mcp_linear_001',
    )
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(
        m,
        instructions='You are a helpful assistant.',
        builtin_tools=[
            MCPServerTool(
                id='linear',
                url='https://mcp.linear.app/mcp',
                description='MCP server for Linear the project management tool.',
                authorization_token='mock-token',  # Won't be used with mock
            ),
        ],
    )

    result = await agent.run('Can you list my Linear issues? Keep your answer brief.')

    # Verify the response and check builtin tool parts appear in message history
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Can you list my Linear issues? Keep your answer brief.',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsDatetime(),
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='mcp_server:linear',
                        args={},
                        tool_call_id='mcp_linear_001',
                        provider_name='xai',
                    ),
                    TextPart(content='No issues found.'),
                    BuiltinToolReturnPart(
                        tool_name='mcp_server:linear',
                        content='No issues found.',
                        tool_call_id='mcp_linear_001',
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                ],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-mcp_linear_001',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_builtin_mcp_server_tool_stream(allow_model_requests: None):
    """Test xAI's MCP server tool with Linear using streaming."""
    # Create a mock MCP server tool call
    mcp_tool_call = create_server_tool_call(
        tool_name='linear.list_issues',  # xAI format: server_label.tool_name
        arguments={},
        tool_call_id='mcp_stream_001',
        tool_type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_MCP_TOOL,
    )

    # For streaming with builtin tools, the tool call appears in the first chunk
    stream = [
        # First chunk: builtin tool call
        grok_builtin_tool_chunk(mcp_tool_call, response_id='grok-mcp_stream_001'),
        # Subsequent chunks: text response
        grok_text_chunk('No issues '),
        grok_text_chunk('found.', finish_reason='stop'),
    ]

    mock_client = MockXai.create_mock_stream([stream])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(
        m,
        instructions='You are a helpful assistant.',
        builtin_tools=[
            MCPServerTool(
                id='linear',
                url='https://mcp.linear.app/mcp',
                description='MCP server for Linear the project management tool.',
                authorization_token='mock-token',
            ),
        ],
    )

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='Can you list my Linear issues?') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    # Verify we got the expected builtin tool call events with snapshot
    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=BuiltinToolReturnPart(
                    tool_name='mcp_server:linear',
                    content={
                        'result': {
                            'content': [{'type': 'text', 'text': 'MCP tool linear.list_issues executed successfully'}]
                        }
                    },
                    tool_call_id='mcp_stream_001',
                    timestamp=IsDatetime(),
                    provider_name='xai',
                ),
            ),
            PartStartEvent(index=1, part=TextPart(content='No issues '), previous_part_kind='builtin-tool-return'),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='found.')),
            PartEndEvent(index=1, part=TextPart(content='No issues found.')),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='mcp_server:linear',
                    content={
                        'result': {
                            'content': [{'type': 'text', 'text': 'MCP tool linear.list_issues executed successfully'}]
                        }
                    },
                    tool_call_id='mcp_stream_001',
                    timestamp=IsDatetime(),
                    provider_name='xai',
                )
            ),
        ]
    )


async def test_xai_model_retries(allow_model_requests: None):
    """Test xAI model with retries."""
    # Create error response then success
    success_response = create_response(content='Success after retry')

    mock_client = MockXai.create_mock([success_response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)
    result = await agent.run('hello')
    assert result.output == 'Success after retry'


async def test_xai_model_settings(allow_model_requests: None):
    """Test xAI model with various settings."""
    response = create_response(content='response with settings')
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(
        m,
        model_settings=ModelSettings(
            temperature=0.5,
            max_tokens=100,
            top_p=0.9,
        ),
    )

    result = await agent.run('hello')
    assert result.output == 'response with settings'

    # Verify settings were passed to the mock
    kwargs = get_mock_chat_create_kwargs(mock_client)[0]
    assert kwargs['temperature'] == 0.5
    assert kwargs['max_tokens'] == 100
    assert kwargs['top_p'] == 0.9


async def test_xai_specific_model_settings(allow_model_requests: None):
    """Test xAI-specific model settings are correctly mapped to SDK parameters."""
    response = create_response(content='response with xai settings')
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(
        m,
        model_settings=XaiModelSettings(
            # Standard settings
            temperature=0.7,
            max_tokens=200,
            top_p=0.95,
            presence_penalty=0.1,
            frequency_penalty=0.2,
            # xAI-specific settings
            xai_logprobs=True,
            xai_top_logprobs=5,
            xai_user='test-user-123',
            xai_store_messages=True,
            xai_previous_response_id='prev-resp-456',
        ),
    )

    result = await agent.run('hello')
    assert result.output == 'response with xai settings'

    # Verify all settings were correctly mapped and passed to the mock
    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': 'grok-4-1-fast-non-reasoning',
                'messages': [{'content': [{'text': 'hello'}], 'role': 'ROLE_USER'}],
                'tools': None,
                'tool_choice': None,
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
                # Standard settings
                'temperature': 0.7,
                'max_tokens': 200,
                'top_p': 0.95,
                'presence_penalty': 0.1,
                'frequency_penalty': 0.2,
                # xAI-specific settings (mapped from xai_* to SDK parameter names)
                'logprobs': True,
                'top_logprobs': 5,
                'user': 'test-user-123',
                'store_messages': True,
                'previous_response_id': 'prev-resp-456',
            }
        ]
    )


async def test_xai_model_multiple_tool_calls(allow_model_requests: None):
    """Test xAI model with multiple tool calls in sequence."""
    # Three responses: two tool calls, then final answer
    responses = [
        create_response(
            tool_calls=[create_tool_call(id='1', name='get_data', arguments={'key': 'value1'})],
        ),
        create_response(
            tool_calls=[create_tool_call(id='2', name='process_data', arguments={'data': 'result1'})],
        ),
        create_response(content='Final processed result'),
    ]

    mock_client = MockXai.create_mock(responses)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    @agent.tool_plain
    async def get_data(key: str) -> str:
        return f'data for {key}'

    @agent.tool_plain
    async def process_data(data: str) -> str:
        return f'processed {data}'

    result = await agent.run('Get and process data')
    assert result.output == 'Final processed result'
    assert result.usage().requests == 3
    assert result.usage().tool_calls == 2

    # Verify kwargs - should have 3 calls (initial + 2 tool results)
    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [{'content': [{'text': 'Get and process data'}], 'role': 'ROLE_USER'}],
                'tools': [
                    {
                        'function': {
                            'name': 'get_data',
                            'parameters': '{"additionalProperties": false, "properties": {"key": {"type": "string"}}, "required": ["key"], "type": "object"}',
                        }
                    },
                    {
                        'function': {
                            'name': 'process_data',
                            'parameters': '{"additionalProperties": false, "properties": {"data": {"type": "string"}}, "required": ["data"], "type": "object"}',
                        }
                    },
                ],
                'tool_choice': 'auto',
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            },
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [
                    {'content': [{'text': 'Get and process data'}], 'role': 'ROLE_USER'},
                    {
                        'content': [{'text': ''}],
                        'role': 'ROLE_ASSISTANT',
                        'tool_calls': [
                            {
                                'id': '1',
                                'type': 'TOOL_CALL_TYPE_CLIENT_SIDE_TOOL',
                                'status': 'TOOL_CALL_STATUS_COMPLETED',
                                'function': {'name': 'get_data', 'arguments': '{"key": "value1"}'},
                            }
                        ],
                    },
                    {'content': [{'text': 'data for value1'}], 'role': 'ROLE_TOOL'},
                ],
                'tools': [
                    {
                        'function': {
                            'name': 'get_data',
                            'parameters': '{"additionalProperties": false, "properties": {"key": {"type": "string"}}, "required": ["key"], "type": "object"}',
                        }
                    },
                    {
                        'function': {
                            'name': 'process_data',
                            'parameters': '{"additionalProperties": false, "properties": {"data": {"type": "string"}}, "required": ["data"], "type": "object"}',
                        }
                    },
                ],
                'tool_choice': 'auto',
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            },
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [
                    {'content': [{'text': 'Get and process data'}], 'role': 'ROLE_USER'},
                    {
                        'content': [{'text': ''}],
                        'role': 'ROLE_ASSISTANT',
                        'tool_calls': [
                            {
                                'id': '1',
                                'type': 'TOOL_CALL_TYPE_CLIENT_SIDE_TOOL',
                                'status': 'TOOL_CALL_STATUS_COMPLETED',
                                'function': {'name': 'get_data', 'arguments': '{"key": "value1"}'},
                            }
                        ],
                    },
                    {'content': [{'text': 'data for value1'}], 'role': 'ROLE_TOOL'},
                    {
                        'content': [{'text': ''}],
                        'role': 'ROLE_ASSISTANT',
                        'tool_calls': [
                            {
                                'id': '2',
                                'type': 'TOOL_CALL_TYPE_CLIENT_SIDE_TOOL',
                                'status': 'TOOL_CALL_STATUS_COMPLETED',
                                'function': {'name': 'process_data', 'arguments': '{"data": "result1"}'},
                            }
                        ],
                    },
                    {'content': [{'text': 'processed result1'}], 'role': 'ROLE_TOOL'},
                ],
                'tools': [
                    {
                        'function': {
                            'name': 'get_data',
                            'parameters': '{"additionalProperties": false, "properties": {"key": {"type": "string"}}, "required": ["key"], "type": "object"}',
                        }
                    },
                    {
                        'function': {
                            'name': 'process_data',
                            'parameters': '{"additionalProperties": false, "properties": {"data": {"type": "string"}}, "required": ["data"], "type": "object"}',
                        }
                    },
                ],
                'tool_choice': 'auto',
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            },
        ]
    )


# Test for error handling
@pytest.mark.skipif(os.getenv('XAI_API_KEY') is not None, reason='Skipped when XAI_API_KEY is set')
async def test_xai_model_invalid_api_key():
    """Test xAI provider with invalid API key."""
    with pytest.raises(UserError, match='Set the `XAI_API_KEY` environment variable'):
        XaiProvider(api_key='')


async def test_xai_model_properties():
    """Test xAI model properties."""
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(api_key='test-key'))

    assert m.model_name == XAI_NON_REASONING_MODEL
    assert m.system == 'xai'


# Tests for reasoning/thinking content (similar to OpenAI Responses tests)


async def test_xai_reasoning_simple(allow_model_requests: None):
    """Test xAI model with simple reasoning content."""
    response = create_response(
        content='The answer is 4',
        reasoning_content='Let me think: 2+2 equals 4',
        usage=create_usage(prompt_tokens=10, completion_tokens=20),
    )
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    result = await agent.run('What is 2+2?')

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is 2+2?', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content='Let me think: 2+2 equals 4', signature=None),
                    TextPart(content='The answer is 4'),
                ],
                usage=RequestUsage(input_tokens=10, output_tokens=20),
                model_name=XAI_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_encrypted_content_only(allow_model_requests: None):
    """Test xAI model with encrypted content (signature) only."""
    response = create_response(
        content='4',
        encrypted_content='abc123signature',
        usage=create_usage(prompt_tokens=10, completion_tokens=5),
    )
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    result = await agent.run('What is 2+2?')

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is 2+2?', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content='', signature='abc123signature', provider_name='xai'),
                    TextPart(content='4'),
                ],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name=XAI_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_reasoning_without_summary(allow_model_requests: None):
    """Test xAI model with encrypted content but no reasoning summary."""
    response = create_response(
        content='4',
        encrypted_content='encrypted123',
    )
    mock_client = MockXai.create_mock([response])
    model = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))

    agent = Agent(model=model)
    result = await agent.run('What is 2+2?')

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is 2+2?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content='', signature='encrypted123', provider_name='xai'),
                    TextPart(content='4'),
                ],
                model_name=XAI_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_reasoning_with_tool_calls(allow_model_requests: None):
    """Test xAI model with reasoning content and tool calls."""
    responses = [
        create_response(
            tool_calls=[create_tool_call(id='1', name='calculate', arguments={'expression': '2+2'})],
            reasoning_content='I need to use the calculate tool to solve this',
            usage=create_usage(prompt_tokens=10, completion_tokens=30),
        ),
        create_response(
            content='The calculation shows that 2+2 equals 4',
            usage=create_usage(prompt_tokens=15, completion_tokens=10),
        ),
    ]
    mock_client = MockXai.create_mock(responses)
    m = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    @agent.tool_plain
    async def calculate(expression: str) -> str:
        return '4'

    result = await agent.run('What is 2+2?')

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is 2+2?', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content='I need to use the calculate tool to solve this', signature=None),
                    ToolCallPart(
                        tool_name='calculate',
                        args='{"expression": "2+2"}',
                        tool_call_id='1',
                    ),
                ],
                usage=RequestUsage(input_tokens=10, output_tokens=30),
                model_name=XAI_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='calculate',
                        content='4',
                        tool_call_id='1',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='The calculation shows that 2+2 equals 4')],
                usage=RequestUsage(input_tokens=15, output_tokens=10),
                model_name=XAI_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_reasoning_with_encrypted_and_tool_calls(allow_model_requests: None):
    """Test xAI model with encrypted reasoning content and tool calls."""
    responses = [
        create_response(
            tool_calls=[create_tool_call(id='1', name='get_weather', arguments={'city': 'San Francisco'})],
            encrypted_content='encrypted_reasoning_abc123',
            usage=create_usage(prompt_tokens=20, completion_tokens=40),
        ),
        create_response(
            content='The weather in San Francisco is sunny',
            usage=create_usage(prompt_tokens=25, completion_tokens=12),
        ),
    ]
    mock_client = MockXai.create_mock(responses)
    m = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    @agent.tool_plain
    async def get_weather(city: str) -> str:
        return 'sunny'

    result = await agent.run('What is the weather in San Francisco?')

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='What is the weather in San Francisco?', timestamp=IsNow(tz=timezone.utc))
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content='', signature='encrypted_reasoning_abc123', provider_name='xai'),
                    ToolCallPart(
                        tool_name='get_weather',
                        args='{"city": "San Francisco"}',
                        tool_call_id='1',
                    ),
                ],
                usage=RequestUsage(input_tokens=20, output_tokens=40),
                model_name=XAI_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_weather',
                        content='sunny',
                        tool_call_id='1',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='The weather in San Francisco is sunny')],
                usage=RequestUsage(input_tokens=25, output_tokens=12),
                model_name=XAI_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_stream_with_reasoning(allow_model_requests: None):
    """Test xAI streaming with reasoning content."""
    stream = [
        grok_reasoning_text_chunk('The answer', reasoning_content='Let me think about this...', finish_reason=''),
        grok_reasoning_text_chunk(' is 4', reasoning_content='Let me think about this...', finish_reason='stop'),
    ]
    mock_client = MockXai.create_mock_stream([stream])
    m = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('What is 2+2?') as result:
        assert not result.is_complete
        text_chunks = [c async for c in result.stream_text(debounce_by=None)]
        assert text_chunks == snapshot(['The answer', 'The answer is 4'])
        assert result.is_complete

    # Verify the final response includes both reasoning and text
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is 2+2?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ThinkingPart(content='Let me think about this...'), TextPart(content='The answer is 4')],
                usage=RequestUsage(input_tokens=2, output_tokens=1),
                model_name=XAI_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_stream_with_encrypted_reasoning(allow_model_requests: None):
    """Test xAI streaming with encrypted reasoning content."""
    stream = [
        grok_reasoning_text_chunk('The weather', encrypted_content='encrypted_abc123', finish_reason=''),
        grok_reasoning_text_chunk(' is sunny', encrypted_content='encrypted_abc123', finish_reason='stop'),
    ]
    mock_client = MockXai.create_mock_stream([stream])
    m = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('What is the weather?') as result:
        assert not result.is_complete
        text_chunks = [c async for c in result.stream_text(debounce_by=None)]
        assert text_chunks == snapshot(['The weather', 'The weather is sunny'])
        assert result.is_complete

    # Verify the final response includes both encrypted reasoning and text
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the weather?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content='', signature='encrypted_abc123', provider_name='xai'),
                    TextPart(content='The weather is sunny'),
                ],
                usage=RequestUsage(input_tokens=2, output_tokens=1),
                model_name=XAI_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_stream_events_with_reasoning(allow_model_requests: None):
    """Test xAI streaming events with reasoning content."""
    stream = [
        grok_reasoning_text_chunk('The answer', reasoning_content='Let me think about this...', finish_reason=''),
        grok_reasoning_text_chunk(' is 4', reasoning_content='Let me think about this...', finish_reason='stop'),
    ]
    mock_client = MockXai.create_mock_stream([stream])
    m = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    events = [event async for event in agent.run_stream_events('What is 2+2?')]

    # Verify the events include both ThinkingPart and TextPart events
    assert events == snapshot(
        [
            PartStartEvent(index=0, part=ThinkingPart(content='Let me think about this...')),
            PartEndEvent(index=0, part=ThinkingPart(content='Let me think about this...'), next_part_kind='text'),
            PartStartEvent(index=1, part=TextPart(content='The answer'), previous_part_kind='thinking'),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' is 4')),
            PartEndEvent(index=1, part=TextPart(content='The answer is 4')),
            AgentRunResultEvent(result=AgentRunResult(output='The answer is 4')),
        ]
    )


async def test_xai_stream_events_with_encrypted_reasoning(allow_model_requests: None):
    """Test xAI streaming events with encrypted reasoning content."""
    stream = [
        grok_reasoning_text_chunk('The weather', encrypted_content='encrypted_abc123', finish_reason=''),
        grok_reasoning_text_chunk(' is sunny', encrypted_content='encrypted_abc123', finish_reason='stop'),
    ]
    mock_client = MockXai.create_mock_stream([stream])
    m = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    events = [event async for event in agent.run_stream_events('What is the weather?')]

    # Verify the events include both ThinkingPart (encrypted) and TextPart events
    assert events == snapshot(
        [
            PartStartEvent(index=0, part=ThinkingPart(content='', signature='encrypted_abc123', provider_name='xai')),
            PartEndEvent(
                index=0,
                part=ThinkingPart(content='', signature='encrypted_abc123', provider_name='xai'),
                next_part_kind='text',
            ),
            PartStartEvent(index=1, part=TextPart(content='The weather'), previous_part_kind='thinking'),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' is sunny')),
            PartEndEvent(index=1, part=TextPart(content='The weather is sunny')),
            AgentRunResultEvent(result=AgentRunResult(output='The weather is sunny')),
        ]
    )


async def test_xai_usage_with_reasoning_tokens(allow_model_requests: None):
    """Test that xAI model properly extracts reasoning_tokens and cache_read_tokens from usage."""
    # Create a mock usage object with reasoning_tokens and cached_prompt_text_tokens
    mock_usage = create_usage(
        prompt_tokens=100,
        completion_tokens=50,
        reasoning_tokens=25,
        cached_prompt_text_tokens=30,
    )
    response = create_response(
        content='The answer is 42',
        reasoning_content='Let me think deeply about this...',
        usage=mock_usage,
    )
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    result = await agent.run('What is the meaning of life?')
    assert result.output == 'The answer is 42'

    # Verify usage includes details
    assert result.usage() == snapshot(
        RunUsage(
            input_tokens=100, output_tokens=50, details={'reasoning_tokens': 25, 'cache_read_tokens': 30}, requests=1
        )
    )


async def test_xai_usage_without_details(allow_model_requests: None):
    """Test that xAI model handles usage without reasoning_tokens or cached tokens."""
    mock_usage = create_usage(prompt_tokens=20, completion_tokens=10)
    response = create_response(
        content='Simple answer',
        usage=mock_usage,
    )
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    result = await agent.run('Simple question')
    assert result.output == 'Simple answer'

    # Verify usage without details (empty dict when no additional usage info)
    assert result.usage() == snapshot(RunUsage(input_tokens=20, output_tokens=10, requests=1))


async def test_xai_usage_with_server_side_tools(allow_model_requests: None):
    """Test that xAI model properly extracts server_side_tools_used from usage."""
    # Create a mock usage object with server_side_tools_used
    # In the real SDK, server_side_tools_used is a repeated field (list-like)
    mock_usage = create_usage(
        prompt_tokens=50,
        completion_tokens=30,
        server_side_tools_used=[usage_pb2.SERVER_SIDE_TOOL_WEB_SEARCH, usage_pb2.SERVER_SIDE_TOOL_WEB_SEARCH],
    )
    response = create_response(
        content='The answer based on web search',
        usage=mock_usage,
    )
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    result = await agent.run('Search for something')
    assert result.output == 'The answer based on web search'

    # Verify usage includes server_side_tools_used in details
    assert result.usage() == snapshot(
        RunUsage(input_tokens=50, output_tokens=30, details={'server_side_tools_web_search': 2}, requests=1)
    )


async def test_xai_native_output_with_tools(allow_model_requests: None):
    """Test that native output works with tools - tools should be called first, then native output."""
    from pydantic import BaseModel

    class CityLocation(BaseModel):
        """A city and its country."""

        city: str
        country: str

    # First response: tool call
    response1 = create_response(
        tool_calls=[create_tool_call('call_get_country', 'get_user_country', {})],
        usage=create_usage(prompt_tokens=70, completion_tokens=12),
    )
    # Second response: native output (JSON)
    response2 = create_response(
        content='{"city":"Mexico City","country":"Mexico"}',
        usage=create_usage(prompt_tokens=90, completion_tokens=15),
    )
    mock_client = MockXai.create_mock([response1, response2])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))

    agent = Agent(m, output_type=NativeOutput(CityLocation))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run('What is the largest city in the user country?')

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_user_country', args='{}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=70, output_tokens=12),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"city":"Mexico City","country":"Mexico"}')],
                usage=RequestUsage(input_tokens=90, output_tokens=15),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    # Verify response_format was passed correctly to the API (both requests should have the JSON schema)
    kwargs = get_mock_chat_create_kwargs(mock_client)
    assert kwargs == snapshot(
        [
            {
                'model': 'grok-4-1-fast-non-reasoning',
                'messages': [
                    {'content': [{'text': 'What is the largest city in the user country?'}], 'role': 'ROLE_USER'}
                ],
                'tools': [
                    {
                        'function': {
                            'name': 'get_user_country',
                            'parameters': '{"additionalProperties": false, "properties": {}, "type": "object"}',
                        }
                    }
                ],
                'tool_choice': 'auto',
                'response_format': {
                    'format_type': 'FORMAT_TYPE_JSON_SCHEMA',
                    'schema': '{"properties": {"city": {"type": "string"}, "country": {"type": "string"}}, "required": ["city", "country"], "title": "CityLocation", "type": "object"}',
                },
                'use_encrypted_content': False,
                'include': [],
            },
            {
                'model': 'grok-4-1-fast-non-reasoning',
                'messages': [
                    {'content': [{'text': 'What is the largest city in the user country?'}], 'role': 'ROLE_USER'},
                    {
                        'content': [{'text': ''}],
                        'role': 'ROLE_ASSISTANT',
                        'tool_calls': [
                            {
                                'id': 'call_get_country',
                                'type': 'TOOL_CALL_TYPE_CLIENT_SIDE_TOOL',
                                'status': 'TOOL_CALL_STATUS_COMPLETED',
                                'function': {'name': 'get_user_country', 'arguments': '{}'},
                            }
                        ],
                    },
                    {'content': [{'text': 'Mexico'}], 'role': 'ROLE_TOOL'},
                ],
                'tools': [
                    {
                        'function': {
                            'name': 'get_user_country',
                            'parameters': '{"additionalProperties": false, "properties": {}, "type": "object"}',
                        }
                    }
                ],
                'tool_choice': 'auto',
                'response_format': {
                    'format_type': 'FORMAT_TYPE_JSON_SCHEMA',
                    'schema': '{"properties": {"city": {"type": "string"}, "country": {"type": "string"}}, "required": ["city", "country"], "title": "CityLocation", "type": "object"}',
                },
                'use_encrypted_content': False,
                'include': [],
            },
        ]
    )


async def test_tool_choice_fallback(allow_model_requests: None) -> None:
    """Test that tool_choice falls back to 'auto' when 'required' is not supported."""
    # Create a profile that doesn't support tool_choice='required'
    profile = GrokModelProfile(grok_supports_tool_choice_required=False)

    response = create_response(content='ok', usage=create_usage(prompt_tokens=10, completion_tokens=5))
    mock_client = MockXai.create_mock([response])
    model = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client), profile=profile)

    params = ModelRequestParameters(function_tools=[ToolDefinition(name='x')], allow_text_output=False)

    await model._create_chat(  # pyright: ignore[reportPrivateUsage]
        messages=[],
        model_settings={},
        model_request_parameters=params,
    )

    # Verify tool_choice was set to 'auto' (not 'required')
    kwargs = get_mock_chat_create_kwargs(mock_client)
    assert kwargs == snapshot(
        [
            {
                'model': XAI_REASONING_MODEL,
                'messages': [],
                'tools': [{'function': {'name': 'x', 'parameters': '{"type": "object", "properties": {}}'}}],
                'tool_choice': 'auto',
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            }
        ]
    )


async def test_mock_xai_index_error(allow_model_requests: None) -> None:
    """Test that MockChatInstance raises IndexError when responses are exhausted."""
    responses = [create_response(content='first')]
    mock_client = MockXai.create_mock(responses)
    model = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(model)

    await agent.run('Hello')

    with pytest.raises(IndexError, match='Mock response index 1 out of range'):
        await agent.run('Hello again')


async def test_xai_logprobs(allow_model_requests: None) -> None:
    """Test logprobs in response."""
    response = create_response(
        content='Test',
        logprobs=[create_logprob('Test', -0.1)],
    )
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    result = await agent.run('Say test')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Say test', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='Test',
                        provider_details={
                            'logprobs': {
                                'content': [
                                    {
                                        'token': 'Test',
                                        'logprob': -0.10000000149011612,
                                        'bytes': [84, 101, 115, 116],
                                        'top_logprobs': [],
                                    }
                                ]
                            }
                        },
                    )
                ],
                usage=RequestUsage(),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_code_execution_default_output(allow_model_requests: None) -> None:
    """Test code execution with default example output."""
    response = create_code_execution_responses(code='print(2+2)')
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, builtin_tools=[CodeExecutionTool()])

    result = await agent.run('Calculate 2+2')
    messages = result.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Calculate 2+2', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={'code': 'print(2+2)'},
                        tool_call_id='code_exec_001',
                        provider_name='xai',
                    ),
                    TextPart(content='{"stdout": "4\\n", "stderr": "", "output_files": {}, "error": "", "ret": ""}'),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'stdout': '4\n', 'stderr': '', 'output_files': {}, 'error': '', 'ret': ''},
                        tool_call_id='code_exec_001',
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                ],
                usage=RequestUsage(),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_web_search_default_output(allow_model_requests: None) -> None:
    """Test web search with default example output."""
    response = create_web_search_responses(query='test query')
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, builtin_tools=[WebSearchTool()])

    result = await agent.run('Search for test')
    messages = result.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Search for test', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'test query'},
                        tool_call_id='web_search_001',
                        provider_name='xai',
                    ),
                    TextPart(content='{}'),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={},
                        tool_call_id='web_search_001',
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                ],
                usage=RequestUsage(),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_mcp_server_default_output(allow_model_requests: None) -> None:
    """Test MCP server tool with default example output."""
    response = create_mcp_server_responses(server_id='linear', tool_name='list_issues')
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(
        m,
        builtin_tools=[MCPServerTool(id='linear', url='https://mcp.linear.app/mcp', description='Linear MCP server')],
    )

    result = await agent.run('List issues')
    messages = result.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='List issues', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='mcp_server:linear', args={}, tool_call_id='mcp_001', provider_name='xai'
                    ),
                    TextPart(
                        content='[{"id": "issue_001", "identifier": "PROJ-123", "title": "example-issue", "description": "example-issue description", "status": "Todo", "priority": {"value": 3, "name": "Medium"}, "url": "https://linear.app/team/issue/PROJ-123/example-issue"}]'
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_server:linear',
                        content=[
                            {
                                'id': 'issue_001',
                                'identifier': 'PROJ-123',
                                'title': 'example-issue',
                                'description': 'example-issue description',
                                'status': 'Todo',
                                'priority': {'value': 3, 'name': 'Medium'},
                                'url': 'https://linear.app/team/issue/PROJ-123/example-issue',
                            }
                        ],
                        tool_call_id='mcp_001',
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                ],
                usage=RequestUsage(),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_retry_prompt_as_user_message(allow_model_requests: None):
    """Test that RetryPromptPart with tool_name=None is sent as a user message."""
    # First response triggers a ModelRetry
    response1 = create_response(content='Invalid')
    # Second response succeeds
    response2 = create_response(content='Valid response')
    mock_client = MockXai.create_mock([response1, response2])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    # Use a result validator that forces a retry without a tool_name
    @agent.output_validator
    async def validate_output(ctx: Any, output: str) -> str:
        if output == 'Invalid':
            raise ModelRetry('Please provide a valid response')
        return output

    result = await agent.run('Hello')
    assert result.output == 'Valid response'

    # Verify the kwargs sent to xAI - second call should have RetryPrompt mapped as user message
    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': 'grok-4-1-fast-non-reasoning',
                'messages': [{'content': [{'text': 'Hello'}], 'role': 'ROLE_USER'}],
                'tools': None,
                'tool_choice': None,
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            },
            {
                'model': 'grok-4-1-fast-non-reasoning',
                'messages': [
                    {'content': [{'text': 'Hello'}], 'role': 'ROLE_USER'},
                    {'content': [{'text': 'Invalid'}], 'role': 'ROLE_ASSISTANT'},
                    {
                        'content': [
                            {
                                'text': """\
Validation feedback:
Please provide a valid response

Fix the errors and try again.\
"""
                            }
                        ],
                        'role': 'ROLE_USER',
                    },
                ],
                'tools': None,
                'tool_choice': None,
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            },
        ]
    )

    # Verify the retry prompt was sent as a user message
    messages = result.all_messages()
    assert len(messages) == 4  # UserPrompt, ModelResponse, RetryPrompt, ModelResponse
    assert isinstance(messages[2].parts[0], RetryPromptPart)
    assert messages[2].parts[0].tool_name is None


async def test_xai_thinking_part_in_message_history(allow_model_requests: None):
    """Test that ThinkingPart in message history is properly mapped."""
    # First response with reasoning
    response1 = create_response(
        content='first response',
        reasoning_content='First reasoning',
        usage=create_usage(prompt_tokens=10, completion_tokens=5),
    )
    # Second response
    response2 = create_response(
        content='second response',
        usage=create_usage(prompt_tokens=20, completion_tokens=5),
    )

    mock_client = MockXai.create_mock([response1, response2])
    m = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    # Run twice to test message history containing ThinkingPart
    result1 = await agent.run('First question')
    result2 = await agent.run('Second question', message_history=result1.new_messages())

    # Verify kwargs - second call should have ThinkingPart mapped with reasoning_content
    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': XAI_REASONING_MODEL,
                'messages': [{'content': [{'text': 'First question'}], 'role': 'ROLE_USER'}],
                'tools': None,
                'tool_choice': None,
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            },
            {
                'model': XAI_REASONING_MODEL,
                'messages': [
                    {'content': [{'text': 'First question'}], 'role': 'ROLE_USER'},
                    {'content': [{'text': 'first response'}], 'role': 'ROLE_ASSISTANT'},
                    {'content': [{'text': 'Second question'}], 'role': 'ROLE_USER'},
                ],
                'tools': None,
                'tool_choice': None,
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            },
        ]
    )

    assert result2.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='First question', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ThinkingPart(content='First reasoning'), TextPart(content='first response')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name=XAI_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='Second question', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='second response')],
                usage=RequestUsage(input_tokens=20, output_tokens=5),
                model_name=XAI_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_thinking_part_with_content_and_signature_in_history(allow_model_requests: None):
    """Test that ThinkingPart with BOTH content AND signature in history is properly mapped."""
    # First response with BOTH reasoning content AND encrypted signature
    # This is needed because provider_name is only set to 'xai' when there's a signature
    # And content is only mapped when provider_name matches
    response1 = create_response(
        content='first response',
        reasoning_content='First reasoning',
        encrypted_content='encrypted_signature_123',
        usage=create_usage(prompt_tokens=10, completion_tokens=5),
    )
    # Second response
    response2 = create_response(
        content='second response',
        usage=create_usage(prompt_tokens=20, completion_tokens=5),
    )

    mock_client = MockXai.create_mock([response1, response2])
    m = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    # Run twice to test message history containing ThinkingPart with content AND signature
    result1 = await agent.run('First question')
    result2 = await agent.run('Second question', message_history=result1.new_messages())

    # Verify kwargs - second call should have ThinkingPart mapped with both reasoning_content AND encrypted_content
    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': XAI_REASONING_MODEL,
                'messages': [{'content': [{'text': 'First question'}], 'role': 'ROLE_USER'}],
                'tools': None,
                'tool_choice': None,
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            },
            {
                'model': XAI_REASONING_MODEL,
                'messages': [
                    {'content': [{'text': 'First question'}], 'role': 'ROLE_USER'},
                    # ThinkingPart with BOTH content and signature
                    {
                        'content': [{'text': ''}],
                        'reasoning_content': 'First reasoning',
                        'encrypted_content': 'encrypted_signature_123',
                        'role': 'ROLE_ASSISTANT',
                    },
                    {'content': [{'text': 'first response'}], 'role': 'ROLE_ASSISTANT'},
                    {'content': [{'text': 'Second question'}], 'role': 'ROLE_USER'},
                ],
                'tools': None,
                'tool_choice': None,
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            },
        ]
    )

    assert result2.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='First question', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content='First reasoning', signature='encrypted_signature_123', provider_name='xai'),
                    TextPart(content='first response'),
                ],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name=XAI_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='Second question', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='second response')],
                usage=RequestUsage(input_tokens=20, output_tokens=5),
                model_name=XAI_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_thinking_part_with_signature_only_in_history(allow_model_requests: None):
    """Test that ThinkingPart with ONLY encrypted signature in history is properly mapped."""
    # First response with ONLY encrypted reasoning (no readable content)
    response1 = create_response(
        content='first response',
        encrypted_content='encrypted_signature_123',
        usage=create_usage(prompt_tokens=10, completion_tokens=5),
    )
    # Second response
    response2 = create_response(
        content='second response',
        usage=create_usage(prompt_tokens=20, completion_tokens=5),
    )

    mock_client = MockXai.create_mock([response1, response2])
    m = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    # Run twice to test message history containing ThinkingPart with signature
    result1 = await agent.run('First question')
    result2 = await agent.run('Second question', message_history=result1.new_messages())

    # Verify kwargs - second call should have ThinkingPart mapped with encrypted_content
    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': XAI_REASONING_MODEL,
                'messages': [{'content': [{'text': 'First question'}], 'role': 'ROLE_USER'}],
                'tools': None,
                'tool_choice': None,
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            },
            {
                'model': XAI_REASONING_MODEL,
                'messages': [
                    {'content': [{'text': 'First question'}], 'role': 'ROLE_USER'},
                    {
                        'content': [{'text': ''}],
                        'encrypted_content': 'encrypted_signature_123',
                        'role': 'ROLE_ASSISTANT',
                    },
                    {'content': [{'text': 'first response'}], 'role': 'ROLE_ASSISTANT'},
                    {'content': [{'text': 'Second question'}], 'role': 'ROLE_USER'},
                ],
                'tools': None,
                'tool_choice': None,
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            },
        ]
    )

    assert result2.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='First question', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content='', signature='encrypted_signature_123', provider_name='xai'),
                    TextPart(content='first response'),
                ],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name=XAI_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='Second question', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='second response')],
                usage=RequestUsage(input_tokens=20, output_tokens=5),
                model_name=XAI_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_builtin_tool_call_in_history(allow_model_requests: None):
    """Test that BuiltinToolCallPart and BuiltinToolReturnPart in history are mapped."""
    # First response with code execution
    response1 = create_code_execution_responses(code='print(2+2)')
    # Second response
    response2 = create_response(content='The result was 4')

    mock_client = MockXai.create_mock([response1, response2])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, builtin_tools=[CodeExecutionTool()])

    # Run once, then continue with history
    result1 = await agent.run('Calculate 2+2')
    result2 = await agent.run('What was the result?', message_history=result1.new_messages())

    # Verify kwargs - second call should have builtin tool call in history
    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [{'content': [{'text': 'Calculate 2+2'}], 'role': 'ROLE_USER'}],
                'tools': [{'code_execution': {}}],
                'tool_choice': 'auto',
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            },
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [
                    {'content': [{'text': 'Calculate 2+2'}], 'role': 'ROLE_USER'},
                    {
                        'content': [{'text': ''}],
                        'role': 'ROLE_ASSISTANT',
                        'tool_calls': [
                            {
                                'id': 'code_exec_001',
                                'type': 'TOOL_CALL_TYPE_CODE_EXECUTION_TOOL',
                                'status': 'TOOL_CALL_STATUS_COMPLETED',
                                'function': {'name': 'code_execution', 'arguments': '{"code":"print(2+2)"}'},
                            }
                        ],
                    },
                    {
                        'content': [
                            {'text': '{"stdout": "4\\n", "stderr": "", "output_files": {}, "error": "", "ret": ""}'}
                        ],
                        'role': 'ROLE_ASSISTANT',
                    },
                    {'content': [{'text': 'What was the result?'}], 'role': 'ROLE_USER'},
                ],
                'tools': [{'code_execution': {}}],
                'tool_choice': 'auto',
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            },
        ]
    )

    assert result2.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Calculate 2+2', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={'code': 'print(2+2)'},
                        tool_call_id='code_exec_001',
                        provider_name='xai',
                    ),
                    TextPart(content='{"stdout": "4\\n", "stderr": "", "output_files": {}, "error": "", "ret": ""}'),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'stdout': '4\n', 'stderr': '', 'output_files': {}, 'error': '', 'ret': ''},
                        tool_call_id='code_exec_001',
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                ],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-code_exec_001',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='What was the result?', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='The result was 4')],
                usage=RequestUsage(),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_builtin_tool_failed_in_history(allow_model_requests: None):
    """Test that failed BuiltinToolReturnPart in history updates call status.

    This test creates a message history with BOTH BuiltinToolCallPart AND BuiltinToolReturnPart
    with matching tool_call_id, where the return part has status='failed'.
    where the call status is updated to FAILED.
    """
    # Create a response for the second call
    response = create_response(content='I understand the tool failed')
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, builtin_tools=[CodeExecutionTool()])

    # Manually construct a message history with:
    # 1. BuiltinToolCallPart (populates builtin_calls dict in _map_response_parts)
    # 2. BuiltinToolReturnPart with status='failed'
    message_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Run some code')]),
        ModelResponse(
            parts=[
                BuiltinToolCallPart(
                    tool_name='code_execution',
                    args={'code': 'print("test")'},
                    tool_call_id='code_fail_1',
                    provider_name='xai',  # Must match self.system
                ),
                BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content='Error: execution failed',
                    tool_call_id='code_fail_1',  # Same ID as BuiltinToolCallPart
                    provider_name='xai',  # Must match self.system
                    provider_details={'status': 'failed', 'error': 'Execution timeout'},
                ),
            ],
            model_name=XAI_NON_REASONING_MODEL,
        ),
    ]

    result = await agent.run('What happened?', message_history=message_history)

    # Verify kwargs - the call should have the failed builtin tool with FAILED status and error_message
    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [
                    {'content': [{'text': 'Run some code'}], 'role': 'ROLE_USER'},
                    {
                        'content': [{'text': ''}],
                        'role': 'ROLE_ASSISTANT',
                        'tool_calls': [
                            {
                                'id': 'code_fail_1',
                                'type': 'TOOL_CALL_TYPE_CODE_EXECUTION_TOOL',
                                'status': 'TOOL_CALL_STATUS_COMPLETED',
                                'function': {'name': 'code_execution', 'arguments': '{"code":"print(\\"test\\")"}'},
                            }
                        ],
                    },
                    {'content': [{'text': 'What happened?'}], 'role': 'ROLE_USER'},
                ],
                'tools': [{'code_execution': {}}],
                'tool_choice': 'auto',
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            },
        ]
    )

    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Run some code', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={'code': 'print("test")'},
                        tool_call_id='code_fail_1',
                        provider_name='xai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content='Error: execution failed',
                        tool_call_id='code_fail_1',
                        timestamp=IsDatetime(),
                        provider_name='xai',
                        provider_details={'status': 'failed', 'error': 'Execution timeout'},
                    ),
                ],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='What happened?', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='I understand the tool failed')],
                usage=RequestUsage(),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_include_settings(allow_model_requests: None):
    """Test xAI include settings for encrypted content and tool outputs."""
    response = create_response(content='test', usage=create_usage(prompt_tokens=10, completion_tokens=5))
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    # Run with all include settings enabled
    settings: XaiModelSettings = {
        'xai_include_encrypted_content': True,
        'xai_include_code_execution_outputs': True,
        'xai_include_web_search_outputs': True,
        'xai_include_mcp_outputs': True,
    }
    result = await agent.run('Hello', model_settings=settings)
    assert result.output == 'test'

    # Verify settings were passed to API
    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': XAI_REASONING_MODEL,
                'messages': [{'content': [{'text': 'Hello'}], 'role': 'ROLE_USER'}],
                'tools': None,
                'tool_choice': None,
                'response_format': None,
                'use_encrypted_content': True,
                'include': [
                    3,
                    1,
                    6,
                ],
            }
        ]
    )


async def test_xai_prompted_output_json_object(allow_model_requests: None):
    """Test prompted output uses json_object format."""

    class SimpleResult(BaseModel):
        answer: str

    response = create_response(content='{"answer": "42"}', usage=create_usage(prompt_tokens=10, completion_tokens=5))
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    # Use PromptedOutput explicitly - uses json_object mode when no tools
    agent: Agent[None, SimpleResult] = Agent(m, output_type=PromptedOutput(SimpleResult))

    result = await agent.run('What is the meaning of life?')
    assert result.output == SimpleResult(answer='42')

    # Verify response_format was set to json_object (not json_schema since it's prompted output)
    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [
                    {
                        'content': [
                            {
                                'text': """\

Always respond with a JSON object that's compatible with this schema:

{"properties": {"answer": {"type": "string"}}, "required": ["answer"], "title": "SimpleResult", "type": "object"}

Don't include any text or Markdown fencing before or after.
"""
                            }
                        ],
                        'role': 'ROLE_SYSTEM',
                    },
                    {'content': [{'text': 'What is the meaning of life?'}], 'role': 'ROLE_USER'},
                ],
                'tools': None,
                'tool_choice': None,
                'response_format': {'format_type': 'FORMAT_TYPE_JSON_OBJECT'},
                'use_encrypted_content': False,
                'include': [],
            }
        ]
    )


async def test_xai_cache_point_filtered(allow_model_requests: None):
    """Test that CachePoint in user prompt is filtered out."""
    response = create_response(content='Hello', usage=create_usage(prompt_tokens=5, completion_tokens=2))
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    # Run with a user prompt that includes a CachePoint (which should be filtered)
    result = await agent.run(['Hello', CachePoint(), ' world'])
    assert result.output == 'Hello'

    # Verify message was sent (CachePoint filtered out - only text items remain)
    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [{'content': [{'text': 'Hello'}, {'text': ' world'}], 'role': 'ROLE_USER'}],
                'tools': None,
                'tool_choice': None,
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            }
        ]
    )


async def test_xai_user_prompt_cache_point_only_skipped(allow_model_requests: None):
    """Test that UserPromptPart with only CachePoint returns None and is skipped."""
    response1 = create_response(content='First', usage=create_usage(prompt_tokens=5, completion_tokens=2))
    response2 = create_response(content='Second', usage=create_usage(prompt_tokens=5, completion_tokens=2))
    mock_client = MockXai.create_mock([response1, response2])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    # First run with normal message
    result1 = await agent.run('First question')

    # Create a message history where we manually insert a UserPromptPart with only CachePoint
    # The next run should handle this gracefully
    result2 = await agent.run([CachePoint()], message_history=result1.new_messages())

    # Verify kwargs - the second request should have the history but the CachePoint-only prompt is skipped
    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [{'content': [{'text': 'First question'}], 'role': 'ROLE_USER'}],
                'tools': None,
                'tool_choice': None,
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            },
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [
                    {'content': [{'text': 'First question'}], 'role': 'ROLE_USER'},
                    {'content': [{'text': 'First'}], 'role': 'ROLE_ASSISTANT'},
                    # CachePoint-only user prompt is skipped (returns None from _map_user_prompt)
                ],
                'tools': None,
                'tool_choice': None,
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            },
        ]
    )

    assert result2.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='First question', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='First')],
                usage=RequestUsage(input_tokens=5, output_tokens=2),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content=[CachePoint()], timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Second')],
                usage=RequestUsage(input_tokens=5, output_tokens=2),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_unknown_tool_type_in_response(allow_model_requests: None):
    """Test handling of unknown tool types like x_search or collections_search."""
    # Create a server-side tool call with an unknown/other type
    unknown_tool_call = chat_pb2.ToolCall(
        id='unknown_001',
        type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_X_SEARCH_TOOL,  # x_search is not directly mapped
        status=chat_pb2.ToolCallStatus.TOOL_CALL_STATUS_COMPLETED,
        function=chat_pb2.FunctionCall(
            name='x_search',
            arguments='{"query": "test"}',
        ),
    )

    # Create response with unknown tool
    response = create_mixed_tools_response([unknown_tool_call], text_content='Search results here')
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    result = await agent.run('Search for something')

    # Verify kwargs sent to xAI
    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [{'content': [{'text': 'Search for something'}], 'role': 'ROLE_USER'}],
                'tools': None,
                'tool_choice': None,
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            }
        ]
    )

    # Verify the unknown tool type is handled gracefully using the function name
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Search for something', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolReturnPart(
                        tool_name='x_search',  # Uses function name for unknown tool types
                        content=None,
                        tool_call_id='unknown_001',
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    TextPart(content='Search results here'),
                ],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_empty_usage_response(allow_model_requests: None):
    """Test handling of response with no usage data."""
    # Create response explicitly without usage data
    response = create_response_without_usage(content='No usage tracked')
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    result = await agent.run('Hello')

    # Verify kwargs sent to xAI
    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [{'content': [{'text': 'Hello'}], 'role': 'ROLE_USER'}],
                'tools': None,
                'tool_choice': None,
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            }
        ]
    )

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='No usage tracked')],
                usage=RequestUsage(),  # Empty usage
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )
    assert result.usage() == snapshot(RunUsage(requests=1))


async def test_xai_parse_tool_args_invalid_json(allow_model_requests: None):
    """Test that invalid JSON in tool arguments returns empty dict."""
    # Create a server-side tool call with invalid JSON arguments
    invalid_tool_call = chat_pb2.ToolCall(
        id='invalid_001',
        type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_WEB_SEARCH_TOOL,
        status=chat_pb2.ToolCallStatus.TOOL_CALL_STATUS_IN_PROGRESS,
        function=chat_pb2.FunctionCall(
            name='web_search',
            arguments='not valid json {{{',  # Invalid JSON
        ),
    )

    response = create_mixed_tools_response([invalid_tool_call], text_content='Search complete')
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    # Should handle gracefully, parsing args as empty dict
    result = await agent.run('Search for something')

    # Verify kwargs sent to xAI
    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [{'content': [{'text': 'Search for something'}], 'role': 'ROLE_USER'}],
                'tools': None,
                'tool_choice': None,
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            }
        ]
    )

    # Verify the tool call part has empty args (due to parse failure)
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Search for something', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={},  # Empty due to JSON parse failure
                        tool_call_id='invalid_001',
                        provider_name='xai',
                    ),
                    TextPart(content='Search complete'),
                ],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_stream_empty_tool_call_name(allow_model_requests: None):
    """Test streaming skips tool calls with empty function name."""
    # Create a tool call with empty name
    empty_name_tool_call = chat_pb2.ToolCall(
        id='empty_name_001',
        type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL,
        function=chat_pb2.FunctionCall(name='', arguments='{}'),  # Empty name
    )

    # Create a streaming response with a tool call that has an empty name
    chunk = create_stream_chunk(content='Hello', finish_reason='stop')
    response = create_response_with_tool_calls(
        content='Hello',
        tool_calls=[empty_name_tool_call],
        finish_reason='stop',
        usage=create_usage(prompt_tokens=5, completion_tokens=2),
    )

    stream = [(response, chunk)]
    mock_client = MockXai.create_mock_stream([stream])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('Hello') as result:
        text_chunks = [c async for c in result.stream_text(debounce_by=None)]
        # Should get text, but skip the empty-name tool call
        assert 'Hello' in text_chunks[-1]


async def test_xai_stream_no_usage_no_finish_reason(allow_model_requests: None):
    """Test streaming handles responses without usage or finish reason."""
    # Create streaming chunks where intermediate chunks have no usage/finish_reason
    # First chunk: no usage, no finish_reason (UNSPECIFIED = 0 = falsy)
    chunk1 = create_stream_chunk(content='Hello', finish_reason=None)
    response1 = create_response_without_usage(content='Hello', finish_reason=None)

    # Second chunk: with usage and finish_reason to complete the stream
    chunk2 = create_stream_chunk(content=' world', finish_reason='stop')
    response2 = create_response(
        content='Hello world', finish_reason='stop', usage=create_usage(prompt_tokens=5, completion_tokens=2)
    )

    stream = [(response1, chunk1), (response2, chunk2)]
    mock_client = MockXai.create_mock_stream([stream])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('Hello') as result:
        [c async for c in result.stream_text(debounce_by=None)]

    # Verify kwargs
    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [{'content': [{'text': 'Hello'}], 'role': 'ROLE_USER'}],
                'tools': None,
                'tool_choice': None,
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            }
        ]
    )

    # Should complete without errors
    assert result.is_complete


async def test_xai_provider_string_initialization(allow_model_requests: None, monkeypatch: pytest.MonkeyPatch):
    """Test that provider can be initialized with a string."""
    # This test verifies the infer_provider path when provider is a string
    monkeypatch.setenv('XAI_API_KEY', 'test-key-for-coverage')
    m = XaiModel(XAI_NON_REASONING_MODEL, provider='xai')
    assert m.model_name == XAI_NON_REASONING_MODEL
    assert m.system == 'xai'


async def test_xai_web_search_tool_in_history(allow_model_requests: None):
    """Test that WebSearchTool builtin calls in history are mapped."""
    # First response with web search
    response1 = create_web_search_responses(query='test query', content='Search results')
    # Second response
    response2 = create_response(content='The search found results')

    mock_client = MockXai.create_mock([response1, response2])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, builtin_tools=[WebSearchTool()])

    # Run once, then continue with history
    result1 = await agent.run('Search for test')
    result2 = await agent.run('What did you find?', message_history=result1.new_messages())

    # Verify kwargs - second call should have WebSearchTool builtin call mapped
    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [{'content': [{'text': 'Search for test'}], 'role': 'ROLE_USER'}],
                'tools': [{'web_search': {'enable_image_understanding': False}}],
                'tool_choice': 'auto',
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            },
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [
                    {'content': [{'text': 'Search for test'}], 'role': 'ROLE_USER'},
                    {
                        'content': [{'text': ''}],
                        'role': 'ROLE_ASSISTANT',
                        'tool_calls': [
                            {
                                'id': 'web_search_001',
                                'type': 'TOOL_CALL_TYPE_WEB_SEARCH_TOOL',
                                'status': 'TOOL_CALL_STATUS_COMPLETED',
                                'function': {'name': 'web_search', 'arguments': '{"query":"test query"}'},
                            }
                        ],
                    },
                    {'content': [{'text': 'Search results'}], 'role': 'ROLE_ASSISTANT'},
                    {'content': [{'text': 'What did you find?'}], 'role': 'ROLE_USER'},
                ],
                'tools': [{'web_search': {'enable_image_understanding': False}}],
                'tool_choice': 'auto',
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            },
        ]
    )

    assert result2.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Search for test', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'test query'},
                        tool_call_id='web_search_001',
                        provider_name='xai',
                    ),
                    TextPart(content='Search results'),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content='Search results',
                        tool_call_id='web_search_001',
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                ],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-web_search_001',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='What did you find?', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='The search found results')],
                usage=RequestUsage(),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_mcp_server_tool_in_history(allow_model_requests: None):
    """Test that MCPServerTool builtin calls in history are mapped."""
    # First response with MCP server tool
    response1 = create_mcp_server_responses(
        server_id='my-server', tool_name='get_data', content={'data': 'MCP result'}, tool_input={'param': 'value'}
    )
    # Second response
    response2 = create_response(content='MCP returned data')

    mock_client = MockXai.create_mock([response1, response2])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, builtin_tools=[MCPServerTool(id='my-server', url='https://example.com/mcp')])

    # Run once, then continue with history
    result1 = await agent.run('Get MCP data')
    result2 = await agent.run('What did MCP return?', message_history=result1.new_messages())

    # Verify kwargs - second call should have MCPServerTool builtin call mapped
    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [{'content': [{'text': 'Get MCP data'}], 'role': 'ROLE_USER'}],
                'tools': [{'mcp': {'server_label': 'my-server', 'server_url': 'https://example.com/mcp'}}],
                'tool_choice': 'auto',
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            },
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [
                    {'content': [{'text': 'Get MCP data'}], 'role': 'ROLE_USER'},
                    {
                        'content': [{'text': ''}],
                        'role': 'ROLE_ASSISTANT',
                        'tool_calls': [
                            {
                                'id': 'mcp_001',
                                'type': 'TOOL_CALL_TYPE_MCP_TOOL',
                                'status': 'TOOL_CALL_STATUS_COMPLETED',
                                'function': {'name': 'mcp_server:my-server', 'arguments': '{"param":"value"}'},
                            }
                        ],
                    },
                    {'content': [{'text': '{"data": "MCP result"}'}], 'role': 'ROLE_ASSISTANT'},
                    {'content': [{'text': 'What did MCP return?'}], 'role': 'ROLE_USER'},
                ],
                'tools': [{'mcp': {'server_label': 'my-server', 'server_url': 'https://example.com/mcp'}}],
                'tool_choice': 'auto',
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            },
        ]
    )

    assert result2.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Get MCP data', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='mcp_server:my-server',
                        args={'param': 'value'},
                        tool_call_id='mcp_001',
                        provider_name='xai',
                    ),
                    TextPart(content='{"data": "MCP result"}'),
                    BuiltinToolReturnPart(
                        tool_name='mcp_server:my-server',
                        content={'data': 'MCP result'},
                        tool_call_id='mcp_001',
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                ],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-mcp_001',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='What did MCP return?', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='MCP returned data')],
                usage=RequestUsage(),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_builtin_tool_without_tool_call_id(allow_model_requests: None):
    """Test that BuiltinToolCallPart without tool_call_id returns None."""
    # Create a response for the call
    response = create_response(content='Done')
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, builtin_tools=[CodeExecutionTool()])

    # Manually construct message history with BuiltinToolCallPart that has empty tool_call_id
    # This directly tests the case when tool_call_id is empty
    message_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Run code')]),
        ModelResponse(
            parts=[
                BuiltinToolCallPart(
                    tool_name='code_execution',
                    args={},
                    tool_call_id='',  # Empty - should be skipped
                    provider_name='xai',
                ),
                TextPart(content='Code ran'),
            ],
            model_name=XAI_NON_REASONING_MODEL,
        ),
    ]

    result = await agent.run('What happened?', message_history=message_history)

    # Verify kwargs - the builtin tool call with empty id is skipped
    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [
                    {'content': [{'text': 'Run code'}], 'role': 'ROLE_USER'},
                    # BuiltinToolCallPart with empty tool_call_id is skipped
                    {'content': [{'text': 'Code ran'}], 'role': 'ROLE_ASSISTANT'},
                    {'content': [{'text': 'What happened?'}], 'role': 'ROLE_USER'},
                ],
                'tools': [{'code_execution': {}}],
                'tool_choice': 'auto',
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            },
        ]
    )

    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Run code', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(tool_name='code_execution', args={}, tool_call_id='', provider_name='xai'),
                    TextPart(content='Code ran'),
                ],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='What happened?', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Done')],
                usage=RequestUsage(),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


def test_grok_builtin_tool_chunk_not_completed():
    """Test grok_builtin_tool_chunk when status is not COMPLETED."""
    # Create a tool call with IN_PROGRESS status
    tool_call = chat_pb2.ToolCall(
        id='in_progress_001',
        type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_CODE_EXECUTION_TOOL,
        status=chat_pb2.ToolCallStatus.TOOL_CALL_STATUS_IN_PROGRESS,  # Not completed
        function=chat_pb2.FunctionCall(name='code_execution', arguments='{}'),
    )

    response, chunk = grok_builtin_tool_chunk(tool_call)

    # When status is not COMPLETED, content should be empty
    # The response should still have the tool call but without generated content
    assert chunk.tool_calls == [tool_call]
    # Verify response has the tool call but empty content (status is in progress)
    assert response.content == ''
    # Verify the tool call status is preserved
    assert response.tool_calls[0].status == chat_pb2.ToolCallStatus.TOOL_CALL_STATUS_IN_PROGRESS


def test_generate_tool_result_content_unknown_type():
    """Test _generate_tool_result_content for unknown tool types."""
    # Create a tool call with an unknown/other type (like x_search)
    unknown_tool_call = chat_pb2.ToolCall(
        id='unknown_001',
        type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_X_SEARCH_TOOL,
        status=chat_pb2.ToolCallStatus.TOOL_CALL_STATUS_COMPLETED,
        function=chat_pb2.FunctionCall(name='x_search', arguments='{}'),
    )

    content = _generate_tool_result_content(unknown_tool_call)

    # For unknown types, should return default status
    assert content == snapshot('{"status": "completed"}')


async def test_xai_thinking_part_content_only_with_provider_in_history(allow_model_requests: None):
    """Test ThinkingPart with content and provider_name but NO signature in history."""
    # Create a response for the continuation
    response = create_response(content='Got it', usage=create_usage(prompt_tokens=10, completion_tokens=5))
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    # Manually construct history with ThinkingPart that has content and provider_name='xai' but NO signature
    # This triggers the branch where item.signature is falsy at line 236 in xai.py
    message_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='First question')]),
        ModelResponse(
            parts=[
                ThinkingPart(
                    content='I am reasoning about this',
                    signature=None,  # No signature - this is the key for branch coverage
                    provider_name='xai',  # Must be 'xai' to enter the if block
                ),
                TextPart(content='First answer'),
            ],
            model_name=XAI_REASONING_MODEL,
        ),
    ]

    await agent.run('Follow up', message_history=message_history)

    # Verify kwargs - ThinkingPart with content only should map to reasoning_content without encrypted_content
    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': XAI_REASONING_MODEL,
                'messages': [
                    {'content': [{'text': 'First question'}], 'role': 'ROLE_USER'},
                    # ThinkingPart with content only → reasoning_content set, no encrypted_content
                    {
                        'content': [{'text': ''}],
                        'reasoning_content': 'I am reasoning about this',
                        'role': 'ROLE_ASSISTANT',
                    },
                    {'content': [{'text': 'First answer'}], 'role': 'ROLE_ASSISTANT'},
                    {'content': [{'text': 'Follow up'}], 'role': 'ROLE_USER'},
                ],
                'tools': None,
                'tool_choice': None,
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            }
        ]
    )


async def test_xai_builtin_tool_failed_without_error_in_history(allow_model_requests: None):
    """Test failed BuiltinToolReturnPart without error message in history."""
    response = create_response(content='I see it failed')
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, builtin_tools=[CodeExecutionTool()])

    # Construct history with failed builtin tool but NO 'error' key in provider_details
    # This triggers the branch at line 261 where error_msg is falsy
    message_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Run code')]),
        ModelResponse(
            parts=[
                BuiltinToolCallPart(
                    tool_name='code_execution',
                    args={},
                    tool_call_id='fail_no_error_1',
                    provider_name='xai',
                ),
                BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content='Failed',
                    tool_call_id='fail_no_error_1',
                    provider_name='xai',
                    provider_details={'status': 'failed'},  # No 'error' key!
                ),
            ],
            model_name=XAI_NON_REASONING_MODEL,
        ),
    ]

    await agent.run('What happened?', message_history=message_history)

    # Verify kwargs - status is FAILED but no error_message since 'error' key was missing
    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [
                    {'content': [{'text': 'Run code'}], 'role': 'ROLE_USER'},
                    {
                        'content': [{'text': ''}],
                        'role': 'ROLE_ASSISTANT',
                        'tool_calls': [
                            {
                                'id': 'fail_no_error_1',
                                'type': 'TOOL_CALL_TYPE_CODE_EXECUTION_TOOL',
                                'status': 'TOOL_CALL_STATUS_COMPLETED',
                                'function': {'name': 'code_execution', 'arguments': '{}'},
                            }
                        ],
                    },
                    {'content': [{'text': 'What happened?'}], 'role': 'ROLE_USER'},
                ],
                'tools': [{'code_execution': {}}],
                'tool_choice': 'auto',
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            }
        ]
    )


async def test_xai_document_url_without_data_type(allow_model_requests: None, monkeypatch: pytest.MonkeyPatch):
    """Test DocumentUrl handling when data_type is missing or empty."""
    response = create_response(content='Document processed')
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    # Mock download_item to return empty data_type (simulating unknown content type)
    async def mock_download_item(item: Any, data_format: str = 'bytes', type_format: str = 'mime') -> dict[str, Any]:
        return {'data': b'%PDF-1.4 test', 'data_type': ''}  # Empty data_type

    monkeypatch.setattr('pydantic_ai.models.xai.download_item', mock_download_item)

    document_url = DocumentUrl(url='https://example.com/unknown-file')
    result = await agent.run(['Process this document', document_url])

    # Should succeed - filename won't have extension when data_type is empty
    assert result.output == 'Document processed'

    # Verify kwargs - file should be uploaded without extension (no data_type means no extension added)
    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [
                    {
                        'content': [
                            {'text': 'Process this document'},
                            {'file': {'file_id': 'file-69b5dc'}},  # Note: no extension since data_type was empty
                        ],
                        'role': 'ROLE_USER',
                    }
                ],
                'tools': None,
                'tool_choice': None,
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            }
        ]
    )


# End of tests
