"""Tests for xAI model integration.

The xAI SDK uses gRPC for all calls (including executing built-in tools like `code_execution`,
`web_search`, and `mcp_server` server-side). Since VCR doesn't support gRPC, we cannot
record/replay these interactions like we do with HTTP APIs.

For all xAI tests, we use simplified mocks that verify:
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
    AudioUrl,
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    CodeExecutionTool,
    DocumentUrl,
    FilePart,
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
from pydantic_ai.messages import (
    BuiltinToolCallEvent,  # pyright: ignore[reportDeprecated]
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
    create_code_execution_response,
    create_failed_builtin_tool_response,
    create_logprob,
    create_mcp_server_response,
    create_mixed_tools_response,
    create_response,
    create_response_with_tool_calls,
    create_response_without_usage,
    create_stream_chunk,
    create_tool_call,
    create_web_search_response,
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
XAI_NON_REASONING_MODEL = 'grok-4-fast-non-reasoning'
XAI_REASONING_MODEL = 'grok-4-fast-reasoning'


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
    binary_image = BinaryContent(b'\x89PNG', media_type='image/png', vendor_metadata={'detail': 'high'})

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
                            {'image_url': {'image_url': 'data:image/png;base64,iVBORw==', 'detail': 'DETAIL_HIGH'}},
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


async def test_xai_image_url_force_download(allow_model_requests: None) -> None:
    """Test that force_download=True calls download_item for ImageUrl in XaiModel."""
    from unittest.mock import AsyncMock, patch

    response = create_response(content='done')
    mock_client = MockXai.create_mock([response])
    model = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(model)

    with patch('pydantic_ai.models.xai.download_item', new_callable=AsyncMock) as mock_download:
        mock_download.return_value = {'data': 'data:image/png;base64,iVBORw==', 'data_type': 'png'}
        await agent.run(
            [
                'Test image',
                ImageUrl(
                    url='https://example.com/image.png',
                    media_type='image/png',
                    force_download=True,
                    vendor_metadata={'detail': 'high'},
                ),
            ]
        )

        mock_download.assert_called_once()
        assert mock_download.call_args[0][0].url == 'https://example.com/image.png'
        assert mock_download.call_args[1]['data_format'] == 'base64_uri'
        assert mock_download.call_args[1]['type_format'] == 'extension'

    # Ensure the data URI is what gets sent to xAI, not the original URL
    assert get_mock_chat_create_kwargs(mock_client)[0]['messages'] == snapshot(
        [
            {
                'content': [
                    {'text': 'Test image'},
                    {'image_url': {'image_url': 'data:image/png;base64,iVBORw==', 'detail': 'DETAIL_HIGH'}},
                ],
                'role': 'ROLE_USER',
            }
        ]
    )


async def test_xai_image_url_no_force_download(allow_model_requests: None) -> None:
    """Test that force_download=False does not call download_item for ImageUrl in XaiModel."""
    from unittest.mock import AsyncMock, patch

    response = create_response(content='done')
    mock_client = MockXai.create_mock([response])
    model = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(model)

    with patch('pydantic_ai.models.xai.download_item', new_callable=AsyncMock) as mock_download:
        await agent.run(
            [
                'Test image',
                ImageUrl(
                    url='https://example.com/image.png',
                    media_type='image/png',
                    force_download=False,
                    vendor_metadata={'detail': 'high'},
                ),
            ]
        )
        mock_download.assert_not_called()

    assert get_mock_chat_create_kwargs(mock_client)[0]['messages'] == snapshot(
        [
            {
                'content': [
                    {'text': 'Test image'},
                    {'image_url': {'image_url': 'https://example.com/image.png', 'detail': 'DETAIL_HIGH'}},
                ],
                'role': 'ROLE_USER',
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


async def test_xai_multiple_tool_calls_in_history_are_grouped(allow_model_requests: None):
    """Test that multiple client-side ToolCallParts in history are grouped into one assistant message."""
    response1 = create_response(
        tool_calls=[
            create_tool_call('call_a', 'tool_a', {}),
            create_tool_call('call_b', 'tool_b', {}),
        ],
        finish_reason='tool_call',
        usage=create_usage(prompt_tokens=10, completion_tokens=5),
    )
    response2 = create_response(
        content='done',
        usage=create_usage(prompt_tokens=20, completion_tokens=5),
    )
    mock_client = MockXai.create_mock([response1, response2])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    @agent.tool_plain
    async def tool_a() -> str:
        return 'a'

    @agent.tool_plain
    async def tool_b() -> str:
        return 'b'

    result = await agent.run('Run tools')
    assert result.output == 'done'

    kwargs = get_mock_chat_create_kwargs(mock_client)
    assert len(kwargs) == 2
    second_messages = kwargs[1]['messages']
    assistant_tool_call_msgs = [m for m in second_messages if m.get('role') == 'ROLE_ASSISTANT' and m.get('tool_calls')]
    assert assistant_tool_call_msgs == snapshot(
        [
            {
                'content': [{'text': ''}],
                'role': 'ROLE_ASSISTANT',
                'tool_calls': [
                    {
                        'id': 'call_a',
                        'type': 'TOOL_CALL_TYPE_CLIENT_SIDE_TOOL',
                        'status': 'TOOL_CALL_STATUS_COMPLETED',
                        'function': {'name': 'tool_a', 'arguments': '{}'},
                    },
                    {
                        'id': 'call_b',
                        'type': 'TOOL_CALL_TYPE_CLIENT_SIDE_TOOL',
                        'status': 'TOOL_CALL_STATUS_COMPLETED',
                        'function': {'name': 'tool_b', 'arguments': '{}'},
                    },
                ],
            }
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
    reasoning_content: str = '',
    encrypted_content: str = '',
) -> tuple[chat_types.Response, chat_types.Chunk]:
    """Create a streaming chunk for Grok with a builtin (server-side) tool call.

    Args:
        tool_call: The server-side tool call proto object (chat_pb2.ToolCall)
        response_id: The response ID
        finish_reason: The finish reason (usually empty for tool call chunks)
        reasoning_content: Optional reasoning content to attach to the response/chunk
        encrypted_content: Optional encrypted reasoning signature to attach to the response/chunk

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
        reasoning_content=reasoning_content,
        encrypted_content=encrypted_content,
        finish_reason=finish_reason if finish_reason else None,  # type: ignore[arg-type]
    )

    # Create response (accumulated) - same tool call with content
    response = create_response(
        content=content,
        tool_calls=[tool_call],
        finish_reason=finish_reason if finish_reason else 'stop',  # type: ignore[arg-type]
        usage=create_usage(prompt_tokens=2, completion_tokens=1) if finish_reason else None,
        reasoning_content=reasoning_content,
        encrypted_content=encrypted_content,
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
            [
                {'first': 'One'},
                {'first': 'One', 'second': 'Two'},
                {'first': 'One', 'second': 'Two'},
                {'first': 'One', 'second': 'Two'},
            ]
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
            [
                {'first': 'One'},
                {'first': 'One', 'second': 'Two'},
                {'first': 'One', 'second': 'Two'},
                {'first': 'One', 'second': 'Two'},
            ]
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


async def test_xai_document_url_input(allow_model_requests: None, xai_provider: XaiProvider):
    """Test passing a document URL to the xAI model."""
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=xai_provider)
    agent = Agent(m)

    document_url = DocumentUrl(url='https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf')

    result = await agent.run(['What is the main content on this document?', document_url])
    assert result.output == snapshot(
        'The main content of the document is a simple placeholder text reading "Dummy PDF file" centered on the page, with no additional substantive information or sections. It appears to be a minimal or test PDF.'
    )


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


async def test_xai_builtin_web_search_tool(allow_model_requests: None, xai_provider: XaiProvider):
    """Test xAI's built-in web_search tool (non-streaming, recorded via proto cassette)."""
    m = XaiModel(XAI_REASONING_MODEL, provider=xai_provider)
    agent = Agent(
        m,
        builtin_tools=[WebSearchTool()],
        model_settings=XaiModelSettings(
            xai_include_encrypted_content=True,  # Encrypted reasoning and tool calls
            xai_include_web_search_output=True,
        ),
    )

    result = await agent.run('Return just the day of week for the date of Jan 1 in 2026?')
    assert result.output == snapshot('**Thursday**')

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
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='xai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'January 1 2026 day of week'},
                        tool_call_id='call_66722885',
                        provider_name='xai',
                    ),
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='xai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content=None,
                        tool_call_id='call_66722885',
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='xai',
                    ),
                    TextPart(content='**Thursday**'),
                ],
                usage=RequestUsage(
                    input_tokens=2484,
                    output_tokens=36,
                    details={'reasoning_tokens': 324, 'cache_read_tokens': 1335, 'server_side_tools_web_search': 1},
                ),
                model_name='grok-4-fast-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_builtin_web_search_tool_stream(allow_model_requests: None, xai_provider: XaiProvider):
    """Test xAI's built-in web_search tool with streaming (recorded via proto cassette)."""
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=xai_provider)

    # Create an agent that includes encrypted content and web search output
    agent = Agent(
        m,
        builtin_tools=[WebSearchTool()],
        model_settings=XaiModelSettings(
            xai_include_encrypted_content=True,  # Encrypted tool calls
            xai_include_web_search_output=True,
        ),
    )

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='What is the weather in San Francisco today in celsius?') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        # Capture all events for validation
                        event_parts.append(event)

    assert agent_run.result is not None
    messages = agent_run.result.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the weather in San Francisco today in celsius?', timestamp=IsDatetime()
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'current weather in San Francisco in Celsius', 'num_results': 5},
                        tool_call_id='call_41803280',
                        provider_name='xai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content=None,
                        tool_call_id='call_41803280',
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    TextPart(
                        content='The current weather in San Francisco is clear with a temperature of 15°C (feels like 14°C). High today around 15°C, low around 8°C.'
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=2312,
                    output_tokens=81,
                    details={'cache_read_tokens': 826, 'server_side_tools_web_search': 1},
                ),
                model_name='grok-4-fast-non-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id='50bdfe4b-2f58-d0b4-8c49-3dc24aeeaf1a',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    # Verify we got the expected builtin tool call events with snapshot.
    # NOTE: IDs/signatures come from xAI, so we use matchers.
    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=BuiltinToolCallPart(
                    tool_name='web_search',
                    args={'query': 'current weather in San Francisco in Celsius', 'num_results': 5},
                    tool_call_id='call_41803280',
                    provider_name='xai',
                ),
            ),
            PartEndEvent(
                index=0,
                part=BuiltinToolCallPart(
                    tool_name='web_search',
                    args={'query': 'current weather in San Francisco in Celsius', 'num_results': 5},
                    tool_call_id='call_41803280',
                    provider_name='xai',
                ),
                next_part_kind='builtin-tool-return',
            ),
            PartStartEvent(
                index=1,
                part=BuiltinToolReturnPart(
                    tool_name='web_search',
                    content=None,
                    tool_call_id='call_41803280',
                    timestamp=IsDatetime(),
                    provider_name='xai',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartStartEvent(index=2, part=TextPart(content='The'), previous_part_kind='builtin-tool-return'),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' current')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' weather')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' San')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' Francisco')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' is')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' clear')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' temperature')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta='15')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta='°C')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta='fe')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta='els')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' like')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta='14')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta='°C')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=').')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' High')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' today')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' around')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta='15')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta='°C')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' low')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' around')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta='8')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta='°C')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta='.')),
            PartEndEvent(
                index=2,
                part=TextPart(
                    content='The current weather in San Francisco is clear with a temperature of 15°C (feels like 14°C). High today around 15°C, low around 8°C.'
                ),
            ),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='web_search',
                    args={'query': 'current weather in San Francisco in Celsius', 'num_results': 5},
                    tool_call_id='call_41803280',
                    provider_name='xai',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='web_search',
                    content=None,
                    tool_call_id='call_41803280',
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


async def test_xai_builtin_code_execution_tool(allow_model_requests: None, xai_provider: XaiProvider):
    """Test xAI's built-in code_execution tool (non-streaming, recorded via proto cassette)."""
    m = XaiModel(XAI_REASONING_MODEL, provider=xai_provider)
    agent = Agent(
        m,
        builtin_tools=[CodeExecutionTool()],
        model_settings=XaiModelSettings(
            xai_include_encrypted_content=False,
            xai_include_code_execution_output=True,
            max_tokens=20,
        ),
    )

    prompt = (
        'Use the builtin tool `code_execution` to compute:\n'
        '  65465 - 6544 * 65464 - 6 + 1.02255\n'
        'Return just the numeric result and nothing else.'
    )
    result = await agent.run(prompt)
    assert result.output == snapshot('-428330955.97745')

    # Verify the builtin tool call and result appear in message history
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=prompt,
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
                        args={'code': 'print(65465 - 6544 * 65464 - 6 + 1.02255)'},
                        tool_call_id='call_65643640',
                        provider_name='xai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={
                            'stdout': '-428330955.97745\n',
                            'stderr': '',
                            'output_files': {},
                            'error': '',
                            'ret': '',
                        },
                        tool_call_id='call_65643640',
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    TextPart(content='-428330955.97745'),
                ],
                usage=RequestUsage(
                    input_tokens=1878,
                    output_tokens=52,
                    details={'reasoning_tokens': 151, 'cache_read_tokens': 1838, 'server_side_tools_code_execution': 1},
                ),
                model_name='grok-4-fast-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_builtin_code_execution_tool_stream(allow_model_requests: None, xai_provider: XaiProvider):
    """Test xAI's built-in code_execution tool with streaming (recorded via proto cassette)."""
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=xai_provider)
    agent = Agent(
        m,
        builtin_tools=[CodeExecutionTool()],
        model_settings=XaiModelSettings(xai_include_code_execution_output=True),
    )

    event_parts: list[Any] = []
    async with agent.iter(
        user_prompt='Use the builtin tool `code_execution` to compute 2 + 2. Return just the numeric result and nothing else.'
    ) as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert agent_run.result is not None
    assert agent_run.result.output == snapshot('4')
    assert agent_run.result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Use the builtin tool `code_execution` to compute 2 + 2. Return just the numeric result and nothing else.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={'code': 'print(2 + 2)'},
                        tool_call_id='call_94648273',
                        provider_name='xai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'stdout': '4\n', 'stderr': '', 'output_files': {}, 'error': '', 'ret': ''},
                        tool_call_id='call_94648273',
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    TextPart(content='4'),
                ],
                usage=RequestUsage(
                    input_tokens=1718,
                    output_tokens=31,
                    details={'cache_read_tokens': 1668, 'server_side_tools_code_execution': 1},
                ),
                model_name='grok-4-fast-non-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id='07064506-c80a-6578-ca78-4dbe34b931b7',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )
    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args={'code': 'print(2 + 2)'},
                    tool_call_id='call_94648273',
                    provider_name='xai',
                ),
            ),
            PartEndEvent(
                index=0,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args={'code': 'print(2 + 2)'},
                    tool_call_id='call_94648273',
                    provider_name='xai',
                ),
                next_part_kind='builtin-tool-return',
            ),
            PartStartEvent(
                index=1,
                part=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'stdout': '4\n', 'stderr': '', 'output_files': {}, 'error': '', 'ret': ''},
                    tool_call_id='call_94648273',
                    timestamp=IsDatetime(),
                    provider_name='xai',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartStartEvent(index=2, part=TextPart(content='4'), previous_part_kind='builtin-tool-return'),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartEndEvent(index=2, part=TextPart(content='4')),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args={'code': 'print(2 + 2)'},
                    tool_call_id='call_94648273',
                    provider_name='xai',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'stdout': '4\n', 'stderr': '', 'output_files': {}, 'error': '', 'ret': ''},
                    tool_call_id='call_94648273',
                    timestamp=IsDatetime(),
                    provider_name='xai',
                )
            ),
        ]
    )


async def test_xai_builtin_x_search_tool_stream_with_in_progress(allow_model_requests: None):
    """Test streaming x_search tool with IN_PROGRESS then COMPLETED status.

    This test covers:
    1. grok_builtin_tool_chunk with IN_PROGRESS status (no content generated)
    2. grok_builtin_tool_chunk with COMPLETED status
    3. _generate_tool_result_content for unknown tool types (x_search)

    Note: x_search is a valid xAI tool type but not explicitly mapped in our helpers,
    so it exercises the 'else' branch returning default status.
    """
    # First chunk: IN_PROGRESS status (simulates tool starting)
    x_search_in_progress = chat_pb2.ToolCall(
        id='xs_001',
        type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_X_SEARCH_TOOL,
        status=chat_pb2.ToolCallStatus.TOOL_CALL_STATUS_IN_PROGRESS,
        function=chat_pb2.FunctionCall(name='x_search', arguments='{"query": "test"}'),
    )
    # Second chunk: COMPLETED status (tool finished)
    x_search_completed = chat_pb2.ToolCall(
        id='xs_001',
        type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_X_SEARCH_TOOL,
        status=chat_pb2.ToolCallStatus.TOOL_CALL_STATUS_COMPLETED,
        function=chat_pb2.FunctionCall(name='x_search', arguments='{"query": "test"}'),
    )

    stream = [
        # First chunk: IN_PROGRESS - no content generated
        grok_builtin_tool_chunk(x_search_in_progress, response_id='grok-xs_001'),
        # Second chunk: COMPLETED - content generated with default status
        grok_builtin_tool_chunk(x_search_completed, response_id='grok-xs_001'),
        # Final text response
        grok_text_chunk('Here are the x search results.', finish_reason='stop'),
    ]

    mock_client = MockXai.create_mock_stream([stream])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    # Note: No builtin_tools needed since x_search is server-side only
    agent = Agent(m)

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='Search x for test') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    # Verify we got the expected events - the IN_PROGRESS chunk should be processed
    # but the BuiltinToolReturnPart only appears when COMPLETED
    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=BuiltinToolCallPart(
                    tool_name='x_search', args={'query': 'test'}, tool_call_id='xs_001', provider_name='xai'
                ),
            ),
            PartEndEvent(
                index=0,
                part=BuiltinToolCallPart(
                    tool_name='x_search', args={'query': 'test'}, tool_call_id='xs_001', provider_name='xai'
                ),
                next_part_kind='builtin-tool-return',
            ),
            PartStartEvent(
                index=1,
                part=BuiltinToolReturnPart(
                    tool_name='x_search',
                    content={'status': 'completed'},
                    tool_call_id='xs_001',
                    timestamp=IsDatetime(),
                    provider_name='xai',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=2,
                part=TextPart(content='Here are the x search results.'),
                previous_part_kind='builtin-tool-return',
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartEndEvent(index=2, part=TextPart(content='Here are the x search results.')),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='x_search', args={'query': 'test'}, tool_call_id='xs_001', provider_name='xai'
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='x_search',
                    content={'status': 'completed'},
                    tool_call_id='xs_001',
                    timestamp=IsDatetime(),
                    provider_name='xai',
                )
            ),
        ]
    )


async def test_xai_builtin_multiple_tools(allow_model_requests: None, xai_provider: XaiProvider):
    """Test using multiple built-in tools together (recorded via proto cassette)."""
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=xai_provider)
    agent = Agent(
        m,
        instructions='You are a helpful assistant.',
        builtin_tools=[WebSearchTool(), CodeExecutionTool()],
        model_settings=XaiModelSettings(
            xai_include_encrypted_content=True,
            xai_include_web_search_output=True,
            xai_include_code_execution_output=True,
        ),
    )

    prompt = (
        'You MUST do these steps in order:\n'
        '1) Use the builtin tool `web_search` to find the release year of Python 3.0.\n'
        '2) Use the builtin tool `code_execution` to compute (year + 1).\n'
        'Return just the final number with no other text.'
    )
    result = await agent.run(prompt)
    assert result.output == snapshot('2009')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content="""\
You MUST do these steps in order:
1) Use the builtin tool `web_search` to find the release year of Python 3.0.
2) Use the builtin tool `code_execution` to compute (year + 1).
Return just the final number with no other text.\
""",
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'release year of Python 3.0'},
                        tool_call_id='call_73579919',
                        provider_name='xai',
                    ),
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='xai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content=None,
                        tool_call_id='call_73579919',
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={'code': 'print(2008 + 1)'},
                        tool_call_id='call_17343356',
                        provider_name='xai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'stdout': '2009\n', 'stderr': '', 'output_files': {}, 'error': '', 'ret': ''},
                        tool_call_id='call_17343356',
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    TextPart(content='2009'),
                ],
                usage=RequestUsage(
                    input_tokens=11140,
                    output_tokens=68,
                    details={
                        'cache_read_tokens': 6768,
                        'server_side_tools_web_search': 1,
                        'server_side_tools_code_execution': 1,
                    },
                ),
                model_name='grok-4-fast-non-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='4372f92c-6672-8612-78f0-c2bb68a0ac6c',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_builtin_tools_with_custom_tools(allow_model_requests: None, xai_provider: XaiProvider):
    """Test mixing xAI's built-in tools with custom (client-side) tools (recorded via proto cassette).

    This test verifies that both builtin tools (`web_search`) and custom tools
    (`get_local_temperature`) can be used in the same conversation.
    """
    m = XaiModel(XAI_REASONING_MODEL, provider=xai_provider)
    agent = Agent(
        m,
        instructions=(
            'Use tools to get the users city and then use the web search tool to find a famous landmark in that city.'
        ),
        builtin_tools=[WebSearchTool()],
        model_settings=XaiModelSettings(
            xai_include_encrypted_content=True,  # Encrypted tool calls
            xai_include_web_search_output=True,
            parallel_tool_calls=False,
        ),
    )

    # Track if custom tool was called
    tool_was_called = False

    @agent.tool_plain
    def guess_city() -> str:
        """The city to guess"""
        nonlocal tool_was_called
        tool_was_called = True
        return 'Chicago'

    result = await agent.run('I am thinking of a city, can you tell me about a famours landmark in this city?')
    assert result.output == snapshot(
        "One of Chicago's most iconic landmarks is **Cloud Gate**, often nicknamed \"The Bean,\" located in Millennium Park. This massive, reflective stainless-steel sculpture by artist Anish Kapoor was unveiled in 2006 and has become a must-see attraction. Shaped like a kidney bean, it mirrors the city's skyline, surrounding architecture, and visitors, creating surreal and interactive photo opportunities. It's free to visit, draws millions annually, and symbolizes Chicago's blend of modern art and urban energy. If you're planning a trip, it's best viewed at sunrise or sunset for stunning reflections!"
    )

    # Verify custom tool was actually called
    assert tool_was_called, 'Custom tool guess_city should have been called'

    # Verify full message history with both custom and builtin tool calls
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='I am thinking of a city, can you tell me about a famours landmark in this city?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions='Use tools to get the users city and then use the web search tool to find a famous landmark in that city.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='xai',
                    ),
                    ToolCallPart(tool_name='guess_city', args='{}', tool_call_id='call_90503665'),
                ],
                usage=RequestUsage(
                    input_tokens=743, output_tokens=15, details={'reasoning_tokens': 135, 'cache_read_tokens': 179}
                ),
                model_name='grok-4-fast-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='2ac934ed-1ba8-40e2-48c5-332c14c7ca71',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='guess_city',
                        content='Chicago',
                        tool_call_id='call_90503665',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions='Use tools to get the users city and then use the web search tool to find a famous landmark in that city.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='xai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'famous landmarks in Chicago', 'num_results': 5},
                        tool_call_id='call_77070235',
                        provider_name='xai',
                    ),
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='xai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content=None,
                        tool_call_id='call_77070235',
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='xai',
                    ),
                    TextPart(
                        content="One of Chicago's most iconic landmarks is **Cloud Gate**, often nicknamed \"The Bean,\" located in Millennium Park. This massive, reflective stainless-steel sculpture by artist Anish Kapoor was unveiled in 2006 and has become a must-see attraction. Shaped like a kidney bean, it mirrors the city's skyline, surrounding architecture, and visitors, creating surreal and interactive photo opportunities. It's free to visit, draws millions annually, and symbolizes Chicago's blend of modern art and urban energy. If you're planning a trip, it's best viewed at sunrise or sunset for stunning reflections!"
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=2280,
                    output_tokens=153,
                    details={'reasoning_tokens': 142, 'cache_read_tokens': 1161, 'server_side_tools_web_search': 1},
                ),
                model_name='grok-4-fast-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='c242ba61-01bb-e362-c878-bdbe1bd1573a',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_builtin_mcp_server_tool(allow_model_requests: None, xai_provider: XaiProvider):
    """Test xAI's MCP server tool with Linear (non-streaming, recorded via proto cassette)."""
    # NOTE: Recording requires a real Linear token so xAI can call the MCP server.
    # Replay mode uses the recorded proto cassette, so the token isn't required there.
    linear_access_token = os.getenv('LINEAR_ACCESS_TOKEN') or 'mock-token'

    m = XaiModel(XAI_NON_REASONING_MODEL, provider=xai_provider)
    agent = Agent(
        m,
        instructions='You are a helpful assistant.',
        builtin_tools=[
            MCPServerTool(
                id='linear',
                url='https://mcp.linear.app/mcp',
                description='MCP server for Linear the project management tool.',
                authorization_token=linear_access_token or 'mock-token',
                allowed_tools=['list_issues'],
            ),
        ],
        model_settings=XaiModelSettings(
            xai_include_encrypted_content=False,
            xai_include_mcp_output=False,
        ),
    )

    prompt = 'What whas the identifier of my last issue opened?'
    result = await agent.run(prompt)
    assert result.output == snapshot('The identifier of your last opened issue is **PAPI-955**.')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What whas the identifier of my last issue opened?',
                        timestamp=IsDatetime(),
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
                        args={'limit': 1, 'orderBy': 'createdAt', 'assignee': 'me'},
                        tool_call_id='call_45974691',
                        provider_name='xai',
                    ),
                    TextPart(content='The identifier of your last opened issue is **PAPI-955**.'),
                ],
                usage=RequestUsage(
                    input_tokens=2097,
                    output_tokens=64,
                    details={'cache_read_tokens': 1007, 'server_side_tools_mcp_server': 1},
                ),
                model_name='grok-4-fast-non-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='9ab72fa8-1674-06f7-6936-412dcf83bcb3',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_builtin_mcp_server_tool_stream(allow_model_requests: None, xai_provider: XaiProvider):
    """Test xAI's MCP server tool with Linear using streaming (recorded via proto cassette)."""
    linear_access_token = os.getenv('LINEAR_ACCESS_TOKEN') or 'mock-token'

    m = XaiModel(XAI_NON_REASONING_MODEL, provider=xai_provider)
    agent = Agent(
        m,
        instructions='You are a helpful assistant.',
        builtin_tools=[
            MCPServerTool(
                id='linear',
                url='https://mcp.linear.app/mcp',
                description='MCP server for Linear the project management tool.',
                authorization_token=linear_access_token,
                allowed_tools=['list_issues'],
            ),
        ],
        model_settings=XaiModelSettings(
            xai_include_encrypted_content=False,
            xai_include_mcp_output=False,
        ),
    )

    event_parts: list[Any] = []
    prompt = 'What whas the identifier of my last issue opened?'
    async with agent.iter(prompt) as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert agent_run.result is not None
    assert agent_run.result.output == snapshot('PAPI-955')
    assert agent_run.result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='What whas the identifier of my last issue opened?', timestamp=IsDatetime())
                ],
                timestamp=IsDatetime(),
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='mcp_server:linear',
                        args={'limit': 1, 'orderBy': 'createdAt', 'assignee': 'me'},
                        tool_call_id='call_11137098',
                        provider_name='xai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_server:linear',
                        content=None,
                        tool_call_id='call_11137098',
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    TextPart(content='PAPI-955'),
                ],
                usage=RequestUsage(
                    input_tokens=2097,
                    output_tokens=54,
                    details={'cache_read_tokens': 1007, 'server_side_tools_mcp_server': 1},
                ),
                model_name='grok-4-fast-non-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id='23540c94-612d-0208-947f-34635d2d86be',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )
    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=BuiltinToolCallPart(
                    tool_name='mcp_server:linear',
                    args={'limit': 1, 'orderBy': 'createdAt', 'assignee': 'me'},
                    tool_call_id='call_11137098',
                    provider_name='xai',
                ),
            ),
            PartEndEvent(
                index=0,
                part=BuiltinToolCallPart(
                    tool_name='mcp_server:linear',
                    args={'limit': 1, 'orderBy': 'createdAt', 'assignee': 'me'},
                    tool_call_id='call_11137098',
                    provider_name='xai',
                ),
                next_part_kind='builtin-tool-return',
            ),
            PartStartEvent(
                index=1,
                part=BuiltinToolReturnPart(
                    tool_name='mcp_server:linear',
                    content=None,
                    tool_call_id='call_11137098',
                    timestamp=IsDatetime(),
                    provider_name='xai',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartStartEvent(index=2, part=TextPart(content='P'), previous_part_kind='builtin-tool-return'),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta='API')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta='955')),
            PartEndEvent(index=2, part=TextPart(content='PAPI-955')),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='mcp_server:linear',
                    args={'limit': 1, 'orderBy': 'createdAt', 'assignee': 'me'},
                    tool_call_id='call_11137098',
                    provider_name='xai',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='mcp_server:linear',
                    content=None,
                    tool_call_id='call_11137098',
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


async def test_xai_model_multiple_tool_calls(allow_model_requests: None, xai_provider: XaiProvider):
    """Test xAI model with multiple tool calls in sequence (recorded via proto cassette)."""
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=xai_provider)
    agent = Agent(
        m,
        model_settings=XaiModelSettings(parallel_tool_calls=False),
    )

    @agent.tool_plain
    async def get_data(key: str) -> str:
        nonlocal tool_was_called_get
        tool_was_called_get = True
        return 'HELLO'

    @agent.tool_plain
    async def process_data(data: str) -> str:
        nonlocal tool_was_called_process
        tool_was_called_process = True
        return f'the result is: {len(data)}'

    tool_was_called_get = False
    tool_was_called_process = False

    result = await agent.run('Get data for KEY_1 and process data returning the output')
    assert result.output == snapshot('the result is: 5')
    assert result.usage().requests == 3
    assert result.usage().tool_calls == 2
    assert tool_was_called_get
    assert tool_was_called_process

    # Message ordering/IDs are provider-dependent; record once and accept the snapshot.
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Get data for KEY_1 and process data returning the output', timestamp=IsDatetime()
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_data', args='{"key":"KEY_1"}', tool_call_id='call_12963444')],
                usage=RequestUsage(input_tokens=393, output_tokens=27, details={'cache_read_tokens': 392}),
                model_name='grok-4-fast-non-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='8defb9c2-64d8-3771-ae1a-86383005640b',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_data',
                        content='HELLO',
                        tool_call_id='call_12963444',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='process_data', args='{"data":"HELLO"}', tool_call_id='call_79562088')],
                usage=RequestUsage(input_tokens=433, output_tokens=26, details={'cache_read_tokens': 432}),
                model_name='grok-4-fast-non-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='28a4ddad-525a-fe0e-9b93-ce0bf84800a0',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='process_data',
                        content='the result is: 5',
                        tool_call_id='call_79562088',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='the result is: 5')],
                usage=RequestUsage(input_tokens=476, output_tokens=6, details={'cache_read_tokens': 467}),
                model_name='grok-4-fast-non-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='3a0535ad-371c-01cd-c65c-d3f78f0bb22b',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_model_properties():
    """Test xAI model properties."""
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(api_key='test-key'))

    assert m.model_name == XAI_NON_REASONING_MODEL
    assert m.system == 'xai'


async def test_xai_reasoning_simple(allow_model_requests: None, xai_provider: XaiProvider):
    """Test xAI reasoning model with encrypted content enabled (recorded via proto cassette)."""
    m = XaiModel(XAI_REASONING_MODEL, provider=xai_provider)
    agent = Agent(
        m,
        model_settings=XaiModelSettings(
            xai_include_encrypted_content=True,
            max_tokens=20,
        ),
    )

    result = await agent.run('What is 2+2? Return just number.')
    assert result.output == snapshot('4')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is 2+2? Return just number.', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='xai',
                    ),
                    TextPart(content='4'),
                ],
                usage=RequestUsage(
                    input_tokens=167, output_tokens=1, details={'reasoning_tokens': 104, 'cache_read_tokens': 151}
                ),
                model_name='grok-4-fast-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='6517dd91-329a-6103-53b0-9a189dc42373',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_encrypted_content_only(allow_model_requests: None, xai_provider: XaiProvider):
    """Test encrypted content (signature) appears when enabled (recorded via proto cassette)."""
    m = XaiModel(XAI_REASONING_MODEL, provider=xai_provider)
    agent = Agent(
        m,
        model_settings=XaiModelSettings(
            xai_include_encrypted_content=True,
            max_tokens=20,
        ),
    )
    result = await agent.run('What is 2+2? Return just "4".')
    assert result.output == snapshot('4')
    assert result.all_messages() == snapshot([])


async def test_xai_reasoning_without_summary(allow_model_requests: None, xai_provider: XaiProvider):
    """Test encrypted reasoning signature without requiring a reasoning summary (recorded via proto cassette)."""
    m = XaiModel(XAI_REASONING_MODEL, provider=xai_provider)
    agent = Agent(
        m,
        model_settings=XaiModelSettings(
            xai_include_encrypted_content=True,
            max_tokens=20,
        ),
    )
    result = await agent.run('What is 2+2? Return just "4".')
    assert result.output == snapshot('4')
    assert result.all_messages() == snapshot([])


async def test_xai_reasoning_with_tool_calls(allow_model_requests: None, xai_provider: XaiProvider):
    """Test reasoning model using client-side tool calls (recorded via proto cassette)."""
    m = XaiModel(XAI_REASONING_MODEL, provider=xai_provider)
    agent = Agent(
        m,
        instructions='Call the tool `calculate` to solve the expression, then answer with just the final number.',
        model_settings=XaiModelSettings(parallel_tool_calls=False, max_tokens=10),
    )

    @agent.tool_plain
    async def calculate(expression: str) -> str:
        return str(eval(expression))  # pragma: no cover

    result = await agent.run('What is 2+2?')
    assert result.output == snapshot('4')
    assert result.all_messages() == snapshot([])


async def test_xai_reasoning_with_encrypted_and_tool_calls(allow_model_requests: None, xai_provider: XaiProvider):
    """Test encrypted reasoning + client-side tool calls (recorded via proto cassette)."""
    m = XaiModel(XAI_REASONING_MODEL, provider=xai_provider)
    agent = Agent(
        m,
        instructions='Call the tool `get_weather` first, then answer with just the tool result.',
        model_settings=XaiModelSettings(
            parallel_tool_calls=False,
            xai_include_encrypted_content=True,
            max_tokens=20,
        ),
    )

    @agent.tool_plain
    async def get_weather(city: str) -> str:
        assert city  # pragma: no cover
        return 'sunny'

    result = await agent.run('What is the weather in San Francisco?')
    assert result.output == snapshot('sunny')
    assert result.all_messages() == snapshot([])


async def test_xai_stream_with_reasoning(allow_model_requests: None, xai_provider: XaiProvider):
    """Test xAI streaming with reasoning model (recorded via proto cassette)."""
    m = XaiModel(XAI_REASONING_MODEL, provider=xai_provider)
    agent = Agent(m, model_settings=XaiModelSettings(xai_include_encrypted_content=True, max_tokens=20))

    async with agent.run_stream('What is 2+2?') as result:
        assert not result.is_complete
        text_chunks = [c async for c in result.stream_text(debounce_by=None)]
        assert text_chunks == snapshot(['4'])
        assert result.is_complete

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is 2+2?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='xai',
                    ),
                    TextPart(content='4'),
                ],
                usage=RequestUsage(
                    input_tokens=163, output_tokens=1, details={'reasoning_tokens': 78, 'cache_read_tokens': 151}
                ),
                model_name='grok-4-fast-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id='89397156-d7fd-08ff-db2f-49eba0cd5b35',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_stream_with_encrypted_reasoning(allow_model_requests: None, xai_provider: XaiProvider):
    """Test xAI streaming with encrypted reasoning enabled (recorded via proto cassette)."""
    m = XaiModel(XAI_REASONING_MODEL, provider=xai_provider)
    agent = Agent(m, model_settings=XaiModelSettings(xai_include_encrypted_content=True, max_tokens=30))

    async with agent.run_stream('Count to 10') as result:
        assert not result.is_complete
        text_chunks = [c async for c in result.stream_text(debounce_by=None)]
        assert text_chunks == snapshot(
            [
                '1',
                '1,',
                '1, ',
                '1, 2',
                '1, 2,',
                '1, 2, ',
                '1, 2, 3',
                '1, 2, 3,',
                '1, 2, 3, ',
                '1, 2, 3, 4',
                '1, 2, 3, 4,',
                '1, 2, 3, 4, ',
                '1, 2, 3, 4, 5',
                '1, 2, 3, 4, 5,',
                '1, 2, 3, 4, 5, ',
                '1, 2, 3, 4, 5, 6',
                '1, 2, 3, 4, 5, 6,',
                '1, 2, 3, 4, 5, 6, ',
                '1, 2, 3, 4, 5, 6, 7',
                '1, 2, 3, 4, 5, 6, 7,',
                '1, 2, 3, 4, 5, 6, 7, ',
                '1, 2, 3, 4, 5, 6, 7, 8',
                '1, 2, 3, 4, 5, 6, 7, 8,',
                '1, 2, 3, 4, 5, 6, 7, 8, ',
                '1, 2, 3, 4, 5, 6, 7, 8, 9',
                '1, 2, 3, 4, 5, 6, 7, 8, 9,',
                '1, 2, 3, 4, 5, 6, 7, 8, 9, ',
                '1, 2, 3, 4, 5, 6, 7, 8, 9, 10',
                '1, 2, 3, 4, 5, 6, 7, 8, 9, 10!',
            ]
        )
        assert result.is_complete

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Count to 10', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='xai',
                    ),
                    TextPart(content='1, 2, 3, 4, 5, 6, 7, 8, 9, 10!'),
                ],
                usage=RequestUsage(
                    input_tokens=160, output_tokens=29, details={'reasoning_tokens': 87, 'cache_read_tokens': 151}
                ),
                model_name='grok-4-fast-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id='4fcd3093-0dcb-82ab-9f3d-3c0a37b6dcdb',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_stream_events_with_reasoning(allow_model_requests: None, xai_provider: XaiProvider):
    """Test xAI streaming events with reasoning model (recorded via proto cassette)."""
    m = XaiModel(XAI_REASONING_MODEL, provider=xai_provider)
    # When a summarised thinking trace is enabled, it will be included as delta events.
    agent = Agent(m, model_settings=XaiModelSettings(xai_include_encrypted_content=True, max_tokens=100))

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='What is the 10th prime number?') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert agent_run.result is not None
    assert agent_run.result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is the 10th prime number?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='xai',
                    ),
                    TextPart(
                        content="""\
29

The first 10 prime numbers are: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29.\
"""
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=165, output_tokens=40, details={'reasoning_tokens': 157, 'cache_read_tokens': 164}
                ),
                model_name='grok-4-fast-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id='8e9d29fd-7a2d-36d6-8b61-44c088703d3d',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ThinkingPart(content='', signature=None, provider_name='xai'),
            ),
            PartEndEvent(
                index=0,
                part=ThinkingPart(
                    content='',
                    signature=IsStr(),
                    provider_name='xai',
                ),
                next_part_kind='text',
            ),
            PartStartEvent(index=1, part=TextPart(content='29'), previous_part_kind='thinking'),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\


"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='The')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' first')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='10')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' prime')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' numbers')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' are')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='3')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='5')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='7')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='11')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='13')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='17')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='19')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='23')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='29')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartEndEvent(
                index=1,
                part=TextPart(
                    content="""\
29

The first 10 prime numbers are: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29.\
"""
                ),
            ),
        ]
    )


async def test_xai_stream_events_with_encrypted_reasoning(allow_model_requests: None, xai_provider: XaiProvider):
    """Test xAI streaming events with encrypted reasoning enabled (recorded via proto cassette)."""
    m = XaiModel(XAI_REASONING_MODEL, provider=xai_provider)
    agent = Agent(m, model_settings=XaiModelSettings(xai_include_encrypted_content=True, max_tokens=30))
    events = [event async for event in agent.run_stream_events('What is the weather? Reply briefly.')]
    assert events == snapshot([])


async def test_xai_usage_with_reasoning_tokens(allow_model_requests: None, xai_provider: XaiProvider):
    """Test that xAI usage extraction includes reasoning/cache tokens when available (recorded via proto cassette)."""
    m = XaiModel(XAI_REASONING_MODEL, provider=xai_provider)
    agent = Agent(
        m,
        model_settings=XaiModelSettings(
            xai_include_encrypted_content=True,
            max_tokens=20,
        ),
    )

    result = await agent.run('What is the meaning of life? Keep it very short.')
    assert isinstance(result.usage(), RunUsage)
    # These fields are provider-dependent but should be non-zero in normal runs.
    assert result.usage().requests >= 1
    assert result.usage().input_tokens >= 0
    assert result.usage().output_tokens >= 0


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
    response = create_code_execution_response(code='print(2+2)')
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
    response = create_web_search_response(query='test query')
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
    response = create_mcp_server_response(server_id='linear', tool_name='list_issues')
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
    # Add a foreign, empty thinking part to the message history, it should be ignored when mapping messages
    # (covers the empty-thinking branch in xAI thinking mapping).
    message_history: list[ModelMessage] = [
        *result1.new_messages(),
        ModelResponse(parts=[ThinkingPart(content='')], provider_name='other', model_name='other-model'),
    ]
    # Include user-supplied `<think>` tags to confirm they are treated as plain user text.
    result2 = await agent.run('Second question <think>user think</think>', message_history=message_history)

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
                    {
                        'content': [
                            {
                                'text': """\
<think>
First reasoning
</think>\
"""
                            }
                        ],
                        'role': 'ROLE_ASSISTANT',
                    },
                    {'content': [{'text': 'first response'}], 'role': 'ROLE_ASSISTANT'},
                    {'content': [{'text': 'Second question <think>user think</think>'}], 'role': 'ROLE_USER'},
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
            ModelResponse(
                parts=[ThinkingPart(content='')],
                usage=RequestUsage(),
                model_name='other-model',
                timestamp=IsDatetime(),
                provider_name='other',
                provider_response_id=None,
                finish_reason=None,
                run_id=None,
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Second question <think>user think</think>', timestamp=IsNow(tz=timezone.utc)
                    )
                ],
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
                    ThinkingPart(content='First reasoning', signature=IsStr(), provider_name='xai'),
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
                    ThinkingPart(content='', signature=IsStr(), provider_name='xai'),
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
    response1 = create_code_execution_response(code='print(2+2)')
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
        'xai_include_code_execution_output': True,
        'xai_include_web_search_output': True,
        'xai_include_mcp_output': True,
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
                    chat_pb2.IncludeOption.INCLUDE_OPTION_CODE_EXECUTION_CALL_OUTPUT,
                    chat_pb2.IncludeOption.INCLUDE_OPTION_WEB_SEARCH_CALL_OUTPUT,
                    chat_pb2.IncludeOption.INCLUDE_OPTION_MCP_CALL_OUTPUT,
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
    response1 = create_web_search_response(query='test query', content='Search results')
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
    response1 = create_mcp_server_response(
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


async def test_xai_file_part_in_history_skipped(allow_model_requests: None):
    """Test that FilePart in message history is silently skipped.

    Files generated by models (e.g., from CodeExecutionTool) are stored in the
    message history but should not be sent back to the API.
    """
    response = create_response(content='Got it', usage=create_usage(prompt_tokens=10, completion_tokens=5))
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    # Create a message history with a FilePart (as if generated by CodeExecutionTool)
    message_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Generate a file')]),
        ModelResponse(
            parts=[
                TextPart(content='Here is your file'),
                # This FilePart simulates output from CodeExecutionTool
                FilePart(
                    content=BinaryContent(data=b'\x89PNG\r\n\x1a\n', media_type='image/png'),
                    id='file_001',
                    provider_name='xai',
                ),
            ],
            model_name=XAI_NON_REASONING_MODEL,
        ),
    ]

    result = await agent.run('What was in that file?', message_history=message_history)
    assert result.output == 'Got it'

    # Verify kwargs - the FilePart should be silently skipped (not sent to API)
    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [
                    {'content': [{'text': 'Generate a file'}], 'role': 'ROLE_USER'},
                    {
                        'content': [{'text': 'Here is your file'}],
                        'role': 'ROLE_ASSISTANT',
                        # Note: FilePart is NOT included here - it's silently skipped
                    },
                    {'content': [{'text': 'What was in that file?'}], 'role': 'ROLE_USER'},
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
