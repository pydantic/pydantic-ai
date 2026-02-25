"""Tests for xAI model integration.

The xAI SDK uses gRPC for all calls (including executing built-in tools like `code_execution`,
`web_search`, and `mcp_server` server-side). Since VCR doesn't support gRPC, we cannot
record/replay these interactions like we do with HTTP APIs.

Instead, we use two strategies:
- A **custom recorder** for xAI SDK interactions where possible (gRPC-aware recording/replay)
- **Mocks** (`MockXai` + real SDK proto objects) for edge cases and streaming scenarios that are hard to record

Across these tests, we verify:
1. Tools are properly registered with the xAI SDK
2. The agent can process responses when builtin tools are enabled
3. Builtin tools can coexist with custom (client-side) tools
"""

from __future__ import annotations as _annotations

import json
from datetime import timezone
from decimal import Decimal
from typing import Any

import pytest
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
    TextContent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
    WebSearchTool,
)
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    BuiltinToolCallEvent,  # pyright: ignore[reportDeprecated]
    BuiltinToolResultEvent,  # pyright: ignore[reportDeprecated]
    CachePoint,
)
from pydantic_ai.models import ModelRequestParameters, ToolDefinition
from pydantic_ai.output import NativeOutput, PromptedOutput, ToolOutput
from pydantic_ai.profiles.grok import GrokModelProfile
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage, RunUsage

from .._inline_snapshot import snapshot
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
    create_server_tool_call,
    create_stream_chunk,
    create_tool_call,
    create_usage,
    create_web_search_response,
    get_grok_reasoning_text_chunk,
    get_grok_text_chunk,
    get_grok_tool_chunk,
    get_mock_chat_create_kwargs,
)

with try_import() as imports_successful:
    import xai_sdk.chat as chat_types
    from xai_sdk.proto import chat_pb2, usage_pb2

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
                provider_url='https://api.x.ai/v1',
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
                provider_url='https://api.x.ai/v1',
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
    assert result.usage() == snapshot(RunUsage(input_tokens=2, output_tokens=1, requests=1))


async def test_xai_cost_calculation(allow_model_requests: None):
    """Test that cost calculation works with genai-prices for xAI models."""
    response = create_response(
        content='world',
        usage=create_usage(prompt_tokens=100, completion_tokens=50),
    )
    mock_client = MockXai.create_mock([response])
    # Use grok-4-1-fast as grok-4-fast is not supported by genai-prices yet
    m = XaiModel('grok-4-1-fast-non-reasoning', provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    result = await agent.run('Hello')
    assert result.output == 'world'

    # Verify cost is calculated via genai-prices
    last_message = result.all_messages()[-1]
    assert isinstance(last_message, ModelResponse)
    assert last_message.cost().total_price == snapshot(Decimal('0.000045'))


async def test_xai_request_structured_response_tool_output(allow_model_requests: None, xai_provider: XaiProvider):
    """ToolOutput with client-side tools (recorded via xAI proto cassette).

    This is closer to OpenAI's recorded tests (`test_openai_tool_output`) since it exercises the
    real provider integration (tool call / tool return / final_result loop).
    """

    m = XaiModel(XAI_NON_REASONING_MODEL, provider=xai_provider)

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(
        m,
        output_type=ToolOutput(CityLocation),
        instructions='Call `get_user_country` first, then call `final_result` with the JSON result.',
        model_settings=XaiModelSettings(parallel_tool_calls=False, max_tokens=80),
    )

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run('What is the largest city in the user country?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is the largest city in the user country?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                instructions='Call `get_user_country` first, then call `final_result` with the JSON result.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_user_country', args='{}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=420, output_tokens=16, details={'cache_read_tokens': 157}),
                model_name='grok-4-fast-non-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
                finish_reason='tool_call',
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
                instructions='Call `get_user_country` first, then call `final_result` with the JSON result.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"city":"Mexico City","country":"Mexico"}',
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=448, output_tokens=36, details={'cache_read_tokens': 436}),
                model_name='grok-4-fast-non-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
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


async def test_xai_reorders_tool_return_parts_by_tool_call_id(allow_model_requests: None):
    response = create_response(
        content='done',
        usage=create_usage(prompt_tokens=20, completion_tokens=5),
    )
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))

    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Run tools')]),
        ModelResponse(
            parts=[
                # Deliberately non-alphabetical order to ensure we don't sort tool results by name/content.
                ToolCallPart(tool_name='tool_a', args='{}', tool_call_id='tool_a'),
                ToolCallPart(tool_name='tool_c', args='{}', tool_call_id='tool_c'),
                ToolCallPart(tool_name='tool_b', args='{}', tool_call_id='tool_b'),
            ],
            finish_reason='tool_call',
        ),
        # Intentionally shuffled: xAI expects tool results in the order the calls were requested.
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name='tool_b', content='tool_b', tool_call_id='tool_b'),
                ToolReturnPart(tool_name='tool_a', content='tool_a', tool_call_id='tool_a'),
                ToolReturnPart(tool_name='tool_c', content='tool_c', tool_call_id='tool_c'),
            ]
        ),
    ]

    await m.request(messages, model_settings=None, model_request_parameters=ModelRequestParameters())

    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [
                    {'content': [{'text': 'Run tools'}], 'role': 'ROLE_USER'},
                    {
                        'content': [{'text': ''}],
                        'role': 'ROLE_ASSISTANT',
                        'tool_calls': [
                            {
                                'id': 'tool_a',
                                'type': 'TOOL_CALL_TYPE_CLIENT_SIDE_TOOL',
                                'status': 'TOOL_CALL_STATUS_COMPLETED',
                                'function': {'name': 'tool_a', 'arguments': '{}'},
                            },
                            {
                                'id': 'tool_c',
                                'type': 'TOOL_CALL_TYPE_CLIENT_SIDE_TOOL',
                                'status': 'TOOL_CALL_STATUS_COMPLETED',
                                'function': {'name': 'tool_c', 'arguments': '{}'},
                            },
                            {
                                'id': 'tool_b',
                                'type': 'TOOL_CALL_TYPE_CLIENT_SIDE_TOOL',
                                'status': 'TOOL_CALL_STATUS_COMPLETED',
                                'function': {'name': 'tool_b', 'arguments': '{}'},
                            },
                        ],
                    },
                    {'content': [{'text': 'tool_a'}], 'role': 'ROLE_TOOL'},
                    {'content': [{'text': 'tool_c'}], 'role': 'ROLE_TOOL'},
                    {'content': [{'text': 'tool_b'}], 'role': 'ROLE_TOOL'},
                ],
                'tools': None,
                'tool_choice': None,
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            }
        ]
    )


async def test_xai_reorders_retry_prompt_tool_results_by_tool_call_id(allow_model_requests: None):
    response = create_response(
        content='done',
        usage=create_usage(prompt_tokens=20, completion_tokens=5),
    )
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))

    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Run tools')]),
        ModelResponse(
            parts=[
                # Deliberately non-alphabetical order to ensure we don't sort tool results by name/content.
                ToolCallPart(tool_name='tool_a', args='{}', tool_call_id='tool_a'),
                ToolCallPart(tool_name='tool_c', args='{}', tool_call_id='tool_c'),
                ToolCallPart(tool_name='tool_b', args='{}', tool_call_id='tool_b'),
            ],
            finish_reason='tool_call',
        ),
        # Intentionally shuffled, but these are tool-results too (tool_name is set).
        ModelRequest(
            parts=[
                RetryPromptPart(content='retry tool_b', tool_name='tool_b', tool_call_id='tool_b'),
                RetryPromptPart(content='retry tool_a', tool_name='tool_a', tool_call_id='tool_a'),
                RetryPromptPart(content='retry tool_c', tool_name='tool_c', tool_call_id='tool_c'),
            ]
        ),
    ]

    await m.request(messages, model_settings=None, model_request_parameters=ModelRequestParameters())

    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [
                    {'content': [{'text': 'Run tools'}], 'role': 'ROLE_USER'},
                    {
                        'content': [{'text': ''}],
                        'role': 'ROLE_ASSISTANT',
                        'tool_calls': [
                            {
                                'id': 'tool_a',
                                'type': 'TOOL_CALL_TYPE_CLIENT_SIDE_TOOL',
                                'status': 'TOOL_CALL_STATUS_COMPLETED',
                                'function': {'name': 'tool_a', 'arguments': '{}'},
                            },
                            {
                                'id': 'tool_c',
                                'type': 'TOOL_CALL_TYPE_CLIENT_SIDE_TOOL',
                                'status': 'TOOL_CALL_STATUS_COMPLETED',
                                'function': {'name': 'tool_c', 'arguments': '{}'},
                            },
                            {
                                'id': 'tool_b',
                                'type': 'TOOL_CALL_TYPE_CLIENT_SIDE_TOOL',
                                'status': 'TOOL_CALL_STATUS_COMPLETED',
                                'function': {'name': 'tool_b', 'arguments': '{}'},
                            },
                        ],
                    },
                    {
                        'content': [{'text': 'retry tool_a\n\nFix the errors and try again.'}],
                        'role': 'ROLE_TOOL',
                    },
                    {
                        'content': [{'text': 'retry tool_c\n\nFix the errors and try again.'}],
                        'role': 'ROLE_TOOL',
                    },
                    {
                        'content': [{'text': 'retry tool_b\n\nFix the errors and try again.'}],
                        'role': 'ROLE_TOOL',
                    },
                ],
                'tools': None,
                'tool_choice': None,
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            }
        ]
    )


async def test_xai_request_structured_response_native_output(allow_model_requests: None, xai_provider: XaiProvider):
    """Test structured output using native JSON schema output (the default for xAI)."""

    m = XaiModel(XAI_NON_REASONING_MODEL, provider=xai_provider)

    # Plain output_type uses native output by default for xAI (per GrokModelProfile)
    class CityLocation(BaseModel):
        city: str
        country: str

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
                parts=[ToolCallPart(tool_name='get_user_country', args='{}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=439, output_tokens=16, details={'cache_read_tokens': 314}),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"city": "Mexico City", "country": "Mexico"}')],
                usage=RequestUsage(input_tokens=467, output_tokens=13, details={'cache_read_tokens': 455}),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_request_tool_call(allow_model_requests: None, xai_provider: XaiProvider):
    """Test tool call with retry."""
    m = XaiModel(XAI_REASONING_MODEL, provider=xai_provider)
    agent = Agent(m)

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        if loc_name == 'London':
            return json.dumps({'lat': 51, 'lng': 0})
        else:
            raise ModelRetry('Wrong location, I only know about "London".')

    result = await agent.run('What is the location of Lodon and London?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is the location of Lodon and London?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='get_location', args='{"loc_name":"Lodon"}', tool_call_id=IsStr()),
                    ToolCallPart(tool_name='get_location', args='{"loc_name":"London"}', tool_call_id=IsStr()),
                ],
                usage=RequestUsage(
                    input_tokens=351, output_tokens=53, details={'reasoning_tokens': 223, 'cache_read_tokens': 148}
                ),
                model_name='grok-4-fast-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Wrong location, I only know about "London".',
                        tool_name='get_location',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    ),
                    ToolReturnPart(
                        tool_name='get_location',
                        content='{"lat": 51, "lng": 0}',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    ),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='Lodon appears to be a misspelling or non-standard variant of "London," which doesn\'t correspond to a known location based on available data. London (the capital of England and the United Kingdom) is located at approximately 51° N latitude and 0° W longitude, in southeastern England along the River Thames.'
                    )
                ],
                usage=RequestUsage(
                    input_tokens=670, output_tokens=63, details={'reasoning_tokens': 83, 'cache_read_tokens': 601}
                ),
                model_name='grok-4-fast-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )
    assert result.usage() == snapshot(
        RunUsage(
            requests=2,
            input_tokens=1021,
            details={'reasoning_tokens': 306, 'cache_read_tokens': 749},
            output_tokens=116,
            tool_calls=1,
        )
    )


async def test_xai_model_multiple_tool_calls(allow_model_requests: None):
    """Test xAI model with multiple tool calls in sequence (mocked)."""
    responses = [
        create_response(
            tool_calls=[create_tool_call('call_get', 'get_data', {'key': 'KEY_1'})],
            finish_reason='tool_call',
        ),
        create_response(
            tool_calls=[create_tool_call('call_process', 'process_data', {'data': 'HELLO'})],
            finish_reason='tool_call',
        ),
        create_response(content='the result is: 5'),
    ]
    mock_client = MockXai.create_mock(responses)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, model_settings=XaiModelSettings(parallel_tool_calls=False))

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
    assert result.output == 'the result is: 5'
    assert result.usage() == snapshot(RunUsage(requests=3, tool_calls=2))
    assert tool_was_called_get
    assert tool_was_called_process


async def test_xai_native_output_with_tools(allow_model_requests: None):
    """Test that native output works with tools - tools should be called first, then native output (mocked)."""
    from pydantic import BaseModel

    class CityLocation(BaseModel):
        """A city and its country."""

        city: str
        country: str

    responses = [
        create_response(
            tool_calls=[create_tool_call('call_country', 'get_user_country', {})],
            finish_reason='tool_call',
        ),
        create_response(content='{"city":"Mexico City","country":"Mexico"}'),
    ]
    mock_client = MockXai.create_mock(responses)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(
        m,
        output_type=NativeOutput(CityLocation),
        instructions=(
            'You MUST call the tool `get_user_country` first. '
            'Then respond with JSON matching the schema (no extra keys, no prose).'
        ),
        model_settings=XaiModelSettings(parallel_tool_calls=False, max_tokens=60),
    )

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run('What is the largest city in the user country?')

    assert result.output.model_dump() == snapshot({'city': 'Mexico City', 'country': 'Mexico'})

    # Ensure the request used JSON schema response_format and included the tool definition.
    kwargs = get_mock_chat_create_kwargs(mock_client)
    assert kwargs[0]['response_format'] is not None
    assert kwargs[0]['tools'] is not None


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


async def test_xai_stream_text(allow_model_requests: None):
    stream = [get_grok_text_chunk('hello '), get_grok_text_chunk('world')]
    mock_client = MockXai.create_mock_stream([stream])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(['hello ', 'hello world'])
        assert result.is_complete
        assert result.usage() == snapshot(RunUsage(input_tokens=2, output_tokens=1, requests=1))


async def test_xai_stream_text_finish_reason(allow_model_requests: None):
    # Create streaming chunks with finish reasons
    stream = [
        get_grok_text_chunk('hello ', ''),
        get_grok_text_chunk('world', ''),
        get_grok_text_chunk('.', 'stop'),
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


class MyTypedDict(TypedDict, total=False):
    first: str
    second: str


def test_xai_tool_chunk_empty_params():
    """Test grok_tool_chunk with all None/empty params to cover edge case branches."""
    # This exercises the branches where tool_name=None, tool_arguments=None, accumulated_args=''
    response, chunk = get_grok_tool_chunk(None, None, '', '')
    # Should produce empty tool call lists
    assert response.tool_calls == []
    assert chunk.tool_calls == []


async def test_xai_stream_structured(allow_model_requests: None):
    """Test structured output streaming, verifying args come as deltas (not repeated PartStartEvents)."""
    stream = [
        get_grok_tool_chunk('final_result', None, accumulated_args=''),
        get_grok_tool_chunk(None, '{"first": "One', accumulated_args='{"first": "One'),
        get_grok_tool_chunk(None, '", "second": "Two"', accumulated_args='{"first": "One", "second": "Two"'),
        get_grok_tool_chunk(None, '}', finish_reason='stop', accumulated_args='{"first": "One", "second": "Two"}'),
    ]
    mock_client = MockXai.create_mock_stream([stream])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, output_type=MyTypedDict)

    # Capture events while streaming, then verify both output and event types
    events: list[Any] = []
    async with agent.iter(user_prompt='') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        events.append(event)

    assert agent_run.result is not None
    assert agent_run.result.output == snapshot({'first': 'One', 'second': 'Two'})
    assert agent_run.usage() == snapshot(RunUsage(input_tokens=20, output_tokens=1, requests=1))

    # Verify event types: one PartStartEvent, then PartDeltaEvents for args
    # (UI adapters like Vercel AI and AG-UI expect deltas, not repeated starts)
    tool_events = [
        e
        for e in events
        if isinstance(e, (PartStartEvent, PartDeltaEvent))
        and (
            isinstance(getattr(e, 'part', None), ToolCallPart)
            or isinstance(getattr(e, 'delta', None), ToolCallPartDelta)
        )
    ]
    assert tool_events == snapshot(
        [
            PartStartEvent(index=0, part=ToolCallPart(tool_name='final_result', tool_call_id='tool-123')),
            PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='{"first": "One', tool_call_id='tool-123')),
            PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='", "second": "Two"', tool_call_id='tool-123')),
            PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='}', tool_call_id='tool-123')),
        ]
    )


async def test_xai_stream_structured_finish_reason(allow_model_requests: None):
    stream = [
        get_grok_tool_chunk('final_result', None, accumulated_args=''),
        get_grok_tool_chunk(None, '{"first": "One', accumulated_args='{"first": "One'),
        get_grok_tool_chunk(None, '", "second": "Two"', accumulated_args='{"first": "One", "second": "Two"'),
        get_grok_tool_chunk(None, '}', accumulated_args='{"first": "One", "second": "Two"}'),
        get_grok_tool_chunk(None, None, finish_reason='stop', accumulated_args='{"first": "One", "second": "Two"}'),
    ]
    mock_client = MockXai.create_mock_stream([stream])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, output_type=MyTypedDict)

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


async def test_xai_stream_native_output(allow_model_requests: None):
    stream = [
        get_grok_text_chunk('{"first": "One'),
        get_grok_text_chunk('", "second": "Two"'),
        get_grok_text_chunk('}'),
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
        get_grok_tool_chunk('final_result', None, accumulated_args=''),
        get_grok_tool_chunk(None, '{"first": "One', accumulated_args='{"first": "One'),
        get_grok_tool_chunk(None, '", "second": "Two"', accumulated_args='{"first": "One", "second": "Two"'),
        get_grok_tool_chunk(None, '}', finish_reason='stop', accumulated_args='{"first": "One", "second": "Two"}'),
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
        get_grok_text_chunk('hello '),
        get_grok_text_chunk('world'),
    ]
    mock_client = MockXai.create_mock_stream([stream])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(['hello ', 'hello world'])
        assert result.is_complete
        assert result.usage() == snapshot(RunUsage(input_tokens=2, output_tokens=1, requests=1))


async def test_xai_none_delta(allow_model_requests: None):
    # Test handling of chunks without deltas
    stream = [
        get_grok_text_chunk('hello '),
        get_grok_text_chunk('world'),
    ]
    mock_client = MockXai.create_mock_stream([stream])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(['hello ', 'hello world'])
        assert result.is_complete
        assert result.usage() == snapshot(RunUsage(input_tokens=2, output_tokens=1, requests=1))


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


async def test_xai_instructions(allow_model_requests: None, xai_provider: XaiProvider):
    """Test that instructions are passed through to xAI SDK as a system message."""
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=xai_provider)
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
                parts=[
                    TextPart(
                        content="Paris is the capital of France. It's the largest city in the country and a major global center for art, fashion, and culture."
                    )
                ],
                usage=RequestUsage(input_tokens=181, output_tokens=27, details={'cache_read_tokens': 162}),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_system_prompt(allow_model_requests: None, xai_provider: XaiProvider):
    """Test that instructions are passed through to xAI SDK as a system message."""
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=xai_provider)
    agent = Agent(m, system_prompt='You are a helpful assistant.')

    result = await agent.run('What is the capital of France?')
    # Verify the message history has system prompt
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='You are a helpful assistant.', timestamp=IsDatetime()),
                    UserPromptPart(content='What is the capital of France?', timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="Paris is the capital of France. It's the largest city in the country and a major global center for art, fashion, and culture."
                    )
                ],
                usage=RequestUsage(input_tokens=181, output_tokens=27, details={'cache_read_tokens': 180}),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
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


async def test_xai_image_url_tool_response(allow_model_requests: None, xai_provider: XaiProvider):
    """Test xAI with image URL from tool response."""

    m = XaiModel(XAI_NON_REASONING_MODEL, provider=xai_provider)
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
                parts=[ToolCallPart(tool_name='get_image', args='{}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=356, output_tokens=15, details={'cache_read_tokens': 314}),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_image',
                        content='See file bd38f5',
                        tool_call_id=IsStr(),
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
                parts=[TextPart(content='The image shows a single raw potato.')],
                usage=RequestUsage(input_tokens=657, output_tokens=8, details={'cache_read_tokens': 371}),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_image_as_binary_content_tool_response(
    allow_model_requests: None, image_content: BinaryContent, xai_provider: XaiProvider
):
    """Test xAI with binary image content from tool response."""
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=xai_provider)
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
                parts=[ToolCallPart(tool_name='get_image', args='{}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=356, output_tokens=15, details={'cache_read_tokens': 314}),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_image',
                        content='See file 241a70',
                        tool_call_id=IsStr(),
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
                parts=[TextPart(content='Kiwi')],
                usage=RequestUsage(input_tokens=657, output_tokens=2, details={'cache_read_tokens': 371}),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_image_as_binary_content_input(
    allow_model_requests: None, image_content: BinaryContent, xai_provider: XaiProvider
):
    """Test passing binary image content directly as input (not from a tool)."""
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=xai_provider)
    agent = Agent(m)

    result = await agent.run(['What fruit is in the image? Keep it short and concise.', image_content])
    assert result.output == snapshot('Kiwi.')


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


async def test_xai_document_url_with_data_type_adds_extension(
    allow_model_requests: None, monkeypatch: pytest.MonkeyPatch
):
    """Test DocumentUrl handling when download returns a data_type (extension should be added)."""
    response = create_response(content='Document processed')
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    async def mock_download_item(item: Any, data_format: str = 'bytes', type_format: str = 'mime') -> dict[str, Any]:
        return {'data': b'%PDF-1.4 test', 'data_type': 'pdf'}

    monkeypatch.setattr('pydantic_ai.models.xai.download_item', mock_download_item)

    # Provide an explicit identifier so we can assert the uploaded filename deterministically.
    document_url = DocumentUrl(url='https://example.com/file', identifier='mydoc')
    result = await agent.run(['Process this document', document_url])
    assert result.output == 'Document processed'

    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [
                    {
                        'content': [
                            {'text': 'Process this document'},
                            {'file': {'file_id': 'file-mydoc.pdf'}},
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


async def test_xai_binary_content_unknown_media_type_raises(allow_model_requests: None):
    """Cover the unsupported BinaryContent media type branch."""
    response = create_response(content='ok', usage=create_usage(prompt_tokens=1, completion_tokens=1))
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    # Neither image/*, audio/*, nor a known document type => should fail during prompt mapping.
    bc = BinaryContent(b'123', media_type='video/mp4')
    with pytest.raises(RuntimeError, match='Unsupported binary content type: video/mp4'):
        await agent.run(['hello', bc])


async def test_xai_stream_empty_response_raises(allow_model_requests: None):
    """Cover the streamed-response empty-stream guard."""
    mock_client = MockXai.create_mock_stream([[]])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    with pytest.raises(UnexpectedModelBehavior, match='Streamed response ended without content or tool calls'):
        async with agent.run_stream(''):
            pass


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
    assert result.output == snapshot('Thursday')

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
                        args={'query': 'what day of the week is January 1, 2026'},
                        tool_call_id=IsStr(),
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
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='xai',
                    ),
                    TextPart(content='Thursday'),
                ],
                usage=RequestUsage(
                    input_tokens=2332,
                    output_tokens=38,
                    details={'reasoning_tokens': 310, 'cache_read_tokens': 1540, 'server_side_tools_web_search': 1},
                ),
                model_name='grok-4-fast-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_builtin_web_search_tool_stream(allow_model_requests: None, xai_provider: XaiProvider):
    """Test xAI's built-in web_search tool with streaming (recorded via proto cassette)."""
    m = XaiModel(XAI_REASONING_MODEL, provider=xai_provider)

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
                    ThinkingPart(
                        content='',
                        signature='DCSdJov0KqNTAiL+A6ioyn70PGRCLsIt3PRq4htAiEqlpTrAZvYxgY3Z22z69UNf30XmThQ6i8OzHILTc5t260s0YS8mrTBzlwtf60lp1o+5SE3BrAtT/iA9DMmMUfgvSe75iVaqrk54sGmxmpbUKzvnpcqBU01Nl1l3l+lssigniKZgD0VB7W/7fGSasp/ysO9BVAgrVTfn8aDGMYh7FOH8ItJCW5AdzPERnITXiL8YmiaeieqdlZBGCLg2datmkj4IldOyhIjF4AAfv+0p8Lv1vcWVAEv35ZI1PF7NMDMyxmyANUBDS+6ZanmMMeQB4hfFFf86d5cQUIF6VItRf4uahuDnmczDMo4W7Ho2xCFdPU8AEKOMndXA8yNeq8pwX3VRguYPzKCTDgaCIn3zBX+YWIfdXujB87L6rZ04FqlLoN1BPtoC+hal6O4OsyfZj3NVh6/P2nwJlgi7ntop4j/S7FxnttWDCtxWxSKMnrBrAO4V+fDaitEtokkxAnID8sPqdWXqN4vk49ZuBufUAG62ASqg88sfZq9up6afYkfONwnhRgv8kqmpqoSDABG79ZRLAvb/ipDrDkSjkfGd/jB6dGQAesTUGyzVLLC5v/NAkiLxVQQP9ADTymxSdJ/MlmScf6xlEIH1RhVsR2XdAst0aJENkWjtH5HjBJIemghkd4LQeIX9JFEd6XWqR5mjA9wMKHKAez7P/uQgD4SU4Yq1HFGHpync4NAOwD1/dLlNp1/qrrEUhGBMXM6uZokb2PYxCBVK4zPRinHfb+DnIvxjFQ6aSAtD88LZDeTpQYgGgflq9o8seGYnMGiLyv6faHyz4TUtmKE0X5T0PtS2iNqGDKn4xPqVxPZc5ErRm2JglnUs6XVkFAo',
                        provider_name='xai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'San Francisco weather today Celsius'},
                        tool_call_id=IsStr(),
                        provider_name='xai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content=None,
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'url': 'https://www.theweathernetwork.com/en/city/us/california/san-francisco/current'},
                        tool_call_id=IsStr(),
                        provider_name='xai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content=None,
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    TextPart(
                        content='Today in San Francisco, the current temperature is 7°C. Expect a high of 16°C and a low of 7°C, with partly cloudy conditions.'
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=4441,
                    output_tokens=135,
                    details={'reasoning_tokens': 631, 'cache_read_tokens': 2530, 'server_side_tools_web_search': 2},
                ),
                model_name='grok-4-fast-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    # Verify we got the expected builtin tool call events with snapshot.
    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ThinkingPart(
                    content='',
                    signature='qBsbOpXt0iGLuYsVE7NyapPz/aM46HSgPDWPirBkao7k1PC/Kr8cCkd8X7kIKzvoyNsXaXRuUlaX/HIeSWM+Z1WX4D7zkUugOj+9wpiI3yuOVvpcgH9ZR8jHXDVl2l3wOuJ1MJLN3ksDfzEe8jXdp2eFqjphqMe4L2F+0CztjBAA8t3JrChA5WOMNCi+7J/ilrMj/Vb5ziC2OYdnoZgKo0tw5g8MD79sSBZI4gccCkyWtepj/Tq5raR5HZbXVXKLbPJSHBe2OHbN3QNnm8Kru+D268Y4g0FviU0FfVGF+VaQvkbHyfj0khheg/e0haj7AssXkH5b6YYOVbxe2qCtQOCSUAw37gAW77nqG9EQc74kk4eobmi8eoeDCVMnQ1C2TqhMaL7bvJ1/YIEQoQ9MZKIMyHSYgsrn6GsEZXza01NAqKrO8tK03UFGR19TphM7ybDzs9dB5VMyg0OhlS+4ZJFTSNg0mrz578hqhp3/iNfgAMD04L8EOA5HdoCS60khfF+LHir0syt+4aN6Dxotm5UK3SMtbnAf26kPLAp/C1TIIBoC2cU+rHOyylXY47w7VrgQVPrYDmuOnt4C2bqGb3W/HYcgg/f4c9pG1nqWdEu8CgvyNtOsHGFilTJJ/7wnZYeA48YtV5yP1GpWKmK1ukSod6YuAjyHFiguBIbLnqVN/c1gJcSm40COikR9B87vzOawR0IgdQjf04ASyucZpTW7MJ3DM+NT5FOHMIg6bqQt2xdkwtPaQSrTMvvFBpIO4FMip103f/1DrFpW+8sQqR7GXcpJi824jHQmb+rqIEjQ1BgsyQNtZuInhXNx31snZgpTkG5cFVl9W/2DaH9PTZkhQP1xibQqpQaPgF83c7j+o2fWYH39kgbEMZMlfxa4BIjHqm2ryFwzhLByfoafpnAyS4u2IRenO3CRr3mFatUGssLs0GjB/sG7zJjqfn7aORhBK4M03uQaW4UaAW4H+p9dbZRomOsh2mubWZG3rrDnjQoMoJv9YqvE78IhNkfInhP98IaFxHRTQpWMTzIkAlBdXyYlz/zxKfV4N4/4ZwQ4kwk1PhH769O7xiQa0PiNSg5dK28LULR0Z8WHz1KMSC6F3IA',
                    provider_name='xai',
                ),
            ),
            PartEndEvent(
                index=0,
                part=ThinkingPart(
                    content='',
                    signature='qBsbOpXt0iGLuYsVE7NyapPz/aM46HSgPDWPirBkao7k1PC/Kr8cCkd8X7kIKzvoyNsXaXRuUlaX/HIeSWM+Z1WX4D7zkUugOj+9wpiI3yuOVvpcgH9ZR8jHXDVl2l3wOuJ1MJLN3ksDfzEe8jXdp2eFqjphqMe4L2F+0CztjBAA8t3JrChA5WOMNCi+7J/ilrMj/Vb5ziC2OYdnoZgKo0tw5g8MD79sSBZI4gccCkyWtepj/Tq5raR5HZbXVXKLbPJSHBe2OHbN3QNnm8Kru+D268Y4g0FviU0FfVGF+VaQvkbHyfj0khheg/e0haj7AssXkH5b6YYOVbxe2qCtQOCSUAw37gAW77nqG9EQc74kk4eobmi8eoeDCVMnQ1C2TqhMaL7bvJ1/YIEQoQ9MZKIMyHSYgsrn6GsEZXza01NAqKrO8tK03UFGR19TphM7ybDzs9dB5VMyg0OhlS+4ZJFTSNg0mrz578hqhp3/iNfgAMD04L8EOA5HdoCS60khfF+LHir0syt+4aN6Dxotm5UK3SMtbnAf26kPLAp/C1TIIBoC2cU+rHOyylXY47w7VrgQVPrYDmuOnt4C2bqGb3W/HYcgg/f4c9pG1nqWdEu8CgvyNtOsHGFilTJJ/7wnZYeA48YtV5yP1GpWKmK1ukSod6YuAjyHFiguBIbLnqVN/c1gJcSm40COikR9B87vzOawR0IgdQjf04ASyucZpTW7MJ3DM+NT5FOHMIg6bqQt2xdkwtPaQSrTMvvFBpIO4FMip103f/1DrFpW+8sQqR7GXcpJi824jHQmb+rqIEjQ1BgsyQNtZuInhXNx31snZgpTkG5cFVl9W/2DaH9PTZkhQP1xibQqpQaPgF83c7j+o2fWYH39kgbEMZMlfxa4BIjHqm2ryFwzhLByfoafpnAyS4u2IRenO3CRr3mFatUGssLs0GjB/sG7zJjqfn7aORhBK4M03uQaW4UaAW4H+p9dbZRomOsh2mubWZG3rrDnjQoMoJv9YqvE78IhNkfInhP98IaFxHRTQpWMTzIkAlBdXyYlz/zxKfV4N4/4ZwQ4kwk1PhH769O7xiQa0PiNSg5dK28LULR0Z8WHz1KMSC6F3IA',
                    provider_name='xai',
                ),
                next_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=1,
                part=BuiltinToolCallPart(
                    tool_name='web_search',
                    args={'query': 'San Francisco weather today Celsius'},
                    tool_call_id=IsStr(),
                    provider_name='xai',
                ),
                previous_part_kind='thinking',
            ),
            PartEndEvent(
                index=1,
                part=BuiltinToolCallPart(
                    tool_name='web_search',
                    args={'query': 'San Francisco weather today Celsius'},
                    tool_call_id=IsStr(),
                    provider_name='xai',
                ),
                next_part_kind='builtin-tool-return',
            ),
            PartStartEvent(
                index=2,
                part=BuiltinToolReturnPart(
                    tool_name='web_search',
                    content=None,
                    tool_call_id=IsStr(),
                    timestamp=IsDatetime(),
                    provider_name='xai',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    signature_delta='ech7t8FxCgEqI/eDeHo8Xd6VMDvsoYAuJZ7CwcXDJzLoflFyJRVo6l+sVq0kcryWOARaEtn8D8CTYhIH0NBDmnzAyPE25xNQdqkdVOQTxeQnRPFrbnBmjgXcEJmiIF3Ym2wuQm5t2c02XDd/O2va9ZjBMwGz/OR0SWUg7GlohlTMvpxvDmYbWl1/Co21yBuxk/7ghDX+E1WP3wlML7G6aAkQYM8GFUmk3Nz4Y+xM8JywizY8PPsbF7gsRs5vGqho3z5HAddTOKANUJzLcPPbeKips2K9ctbS2Wj6yYyD/yODpFzyf77fMk2w4UxMHfugzOmYej8GTRFHjMw5VDM9LGWNOYLVnE/08GTsuCSYgzww+iWdQ+Z1+BkjkFQKlOB9hTYmybKfhw8nvTi7DJEMIoRPaYn725I181Q8jHNwZmdRQcXAu+Gdnee9vdFBUILoB4zvKN+gn5KvtYFnFQ3frIURiJZXTy73g7CIQdhXGNRhJKbsQASvtyINBFTscbmnkX871dWA7UYqLrxy5ePhnXjrhyQd84Bco0JwUqaPt8Rqw44EBkeF5fC+zj7lzsjoR2NjvwXI3xkjJQ+uBnbZ+4uGGMvW1vwmi+aAoNFlYDIaXP0KIiyyGrZoGmjm/X+B1O614OkrZK6cisdbvtu9BHogQ8bzXAKB6kIIJWWt6j6j7oASgb5yAtiS7yCkYg3+L+EJUCOjaeecfBNtSCzIDb3i8RTrvquW/ZiEJj7ZSgiZL134zzEWO6ENL5tZWcE3kWm2LeFb3mOOuQnmzIDT0iDK0LxepAwnv4L8FgE7JD/qVZvMaVGUs0e2qnUsmbw7P4rKMcU0NtVFNLGYVRJaNSqZxxB/l0xfzcv0trmBVhw/YzjSlJUt77T2ZMZm0Y9KQJATd9EdVifzuZrjHUr1DPDRz3cfnBifTZTAIr/UcysQUGr6dAB6wEBRiQK0KoT/F2rhgIyJ2N0sqE0sUpKLGN1wcWkO+cbNER0rtyq2tapCREaZPsDgjZ4T+SLMxMraePnJodzfpDB1mLC3sBmYP66gx0Ay9iTo+2CwnueAFNjsq+fEtTTKu/WAGEEl/fVBvLDZnaNstXqxpBskHK6RKc+y+1TlXXfmWC38grI1oGC7VvzY+TJRGVBvtxU0YJ17IwWdtHgbjqCAx2btanfTWvigsObzRztA10ifWVLQwHKW5GYCn3YGjVzhwFwQJp7jj1/hmwCWwFM4Ijivolc5kPsWVUylbB6TUjIg4Y7zZO6ZryMXKNIJhbA7t6Vaej/E+0cfTFn6ACzFGnqqXxu62GLyte8NBGCEgvh4Bxhc+DoenauWKlCNktSR+i1M+mjg3pkgDcR0J5QiLNHvKhRblh5NGdHapYOphUDbWcwqdJ07VOwo9CHZ1WTox4VuVVSVz84Siag9ibuf18Z7XKd56MortTH7WAATHE6LB6RY8OG5+n1ZY9TWiHE3Mo6vKbHRKcEkbB2k/S5L28CL1a4C4yuAy5nNERL/+IRFhRpEUSH6GFQ4uIWcOcD62Zc'
                ),
            ),
            PartStartEvent(
                index=3,
                part=BuiltinToolCallPart(
                    tool_name='web_search',
                    args={'url': 'https://www.theweathernetwork.com/en/city/us/california/san-francisco/current'},
                    tool_call_id=IsStr(),
                    provider_name='xai',
                ),
                previous_part_kind='builtin-tool-return',
            ),
            PartEndEvent(
                index=3,
                part=BuiltinToolCallPart(
                    tool_name='web_search',
                    args={'url': 'https://www.theweathernetwork.com/en/city/us/california/san-francisco/current'},
                    tool_call_id=IsStr(),
                    provider_name='xai',
                ),
                next_part_kind='builtin-tool-return',
            ),
            PartStartEvent(
                index=4,
                part=BuiltinToolReturnPart(
                    tool_name='web_search',
                    content=None,
                    tool_call_id=IsStr(),
                    timestamp=IsDatetime(),
                    provider_name='xai',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    signature_delta='DCSdJov0KqNTAiL+A6ioyn70PGRCLsIt3PRq4htAiEqlpTrAZvYxgY3Z22z69UNf30XmThQ6i8OzHILTc5t260s0YS8mrTBzlwtf60lp1o+5SE3BrAtT/iA9DMmMUfgvSe75iVaqrk54sGmxmpbUKzvnpcqBU01Nl1l3l+lssigniKZgD0VB7W/7fGSasp/ysO9BVAgrVTfn8aDGMYh7FOH8ItJCW5AdzPERnITXiL8YmiaeieqdlZBGCLg2datmkj4IldOyhIjF4AAfv+0p8Lv1vcWVAEv35ZI1PF7NMDMyxmyANUBDS+6ZanmMMeQB4hfFFf86d5cQUIF6VItRf4uahuDnmczDMo4W7Ho2xCFdPU8AEKOMndXA8yNeq8pwX3VRguYPzKCTDgaCIn3zBX+YWIfdXujB87L6rZ04FqlLoN1BPtoC+hal6O4OsyfZj3NVh6/P2nwJlgi7ntop4j/S7FxnttWDCtxWxSKMnrBrAO4V+fDaitEtokkxAnID8sPqdWXqN4vk49ZuBufUAG62ASqg88sfZq9up6afYkfONwnhRgv8kqmpqoSDABG79ZRLAvb/ipDrDkSjkfGd/jB6dGQAesTUGyzVLLC5v/NAkiLxVQQP9ADTymxSdJ/MlmScf6xlEIH1RhVsR2XdAst0aJENkWjtH5HjBJIemghkd4LQeIX9JFEd6XWqR5mjA9wMKHKAez7P/uQgD4SU4Yq1HFGHpync4NAOwD1/dLlNp1/qrrEUhGBMXM6uZokb2PYxCBVK4zPRinHfb+DnIvxjFQ6aSAtD88LZDeTpQYgGgflq9o8seGYnMGiLyv6faHyz4TUtmKE0X5T0PtS2iNqGDKn4xPqVxPZc5ErRm2JglnUs6XVkFAo'
                ),
            ),
            PartStartEvent(index=5, part=TextPart(content='Today'), previous_part_kind='builtin-tool-return'),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta=' San')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta=' Francisco')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta=' current')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta=' temperature')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta=' is')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta='7')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta='°C')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta=' Expect')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta=' high')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta='16')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta='°C')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta=' low')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta='7')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta='°C')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta=' partly')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta=' cloudy')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta=' conditions')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta='.')),
            PartEndEvent(
                index=5,
                part=TextPart(
                    content='Today in San Francisco, the current temperature is 7°C. Expect a high of 16°C and a low of 7°C, with partly cloudy conditions.'
                ),
            ),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='web_search',
                    args={'query': 'San Francisco weather today Celsius'},
                    tool_call_id=IsStr(),
                    provider_name='xai',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='web_search',
                    content=None,
                    tool_call_id=IsStr(),
                    timestamp=IsDatetime(),
                    provider_name='xai',
                )
            ),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='web_search',
                    args={'url': 'https://www.theweathernetwork.com/en/city/us/california/san-francisco/current'},
                    tool_call_id=IsStr(),
                    provider_name='xai',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='web_search',
                    content=None,
                    tool_call_id=IsStr(),
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
                        tool_call_id=IsStr(),
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
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    TextPart(content='-428330955.97745'),
                ],
                usage=RequestUsage(
                    input_tokens=1889,
                    output_tokens=52,
                    details={'reasoning_tokens': 161, 'cache_read_tokens': 1347, 'server_side_tools_code_execution': 1},
                ),
                model_name='grok-4-fast-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
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
                        tool_call_id=IsStr(),
                        provider_name='xai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'stdout': '4\n', 'stderr': '', 'output_files': {}, 'error': '', 'ret': ''},
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    TextPart(content='4'),
                ],
                usage=RequestUsage(
                    input_tokens=1718,
                    output_tokens=31,
                    details={'cache_read_tokens': 1037, 'server_side_tools_code_execution': 1},
                ),
                model_name='grok-4-fast-non-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
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
                    tool_call_id=IsStr(),
                    provider_name='xai',
                ),
            ),
            PartEndEvent(
                index=0,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args={'code': 'print(2 + 2)'},
                    tool_call_id=IsStr(),
                    provider_name='xai',
                ),
                next_part_kind='builtin-tool-return',
            ),
            PartStartEvent(
                index=1,
                part=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'stdout': '4\n', 'stderr': '', 'output_files': {}, 'error': '', 'ret': ''},
                    tool_call_id=IsStr(),
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
                    tool_call_id=IsStr(),
                    provider_name='xai',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'stdout': '4\n', 'stderr': '', 'output_files': {}, 'error': '', 'ret': ''},
                    tool_call_id=IsStr(),
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
                        tool_call_id=IsStr(),
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
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={'code': 'print(2008 + 1)'},
                        tool_call_id=IsStr(),
                        provider_name='xai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'stdout': '2009\n', 'stderr': '', 'output_files': {}, 'error': '', 'ret': ''},
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    TextPart(content='2009'),
                ],
                usage=RequestUsage(
                    input_tokens=11140,
                    output_tokens=68,
                    details={
                        'cache_read_tokens': 6347,
                        'server_side_tools_web_search': 1,
                        'server_side_tools_code_execution': 1,
                    },
                ),
                model_name='grok-4-fast-non-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
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
        "One of the most famous landmarks in Chicago is **Cloud Gate**, often nicknamed \"The Bean.\" It's a massive, reflective stainless steel sculpture in Millennium Park, designed by artist Anish Kapoor and unveiled in 2006. The 110-ton, bean-shaped installation mirrors the city's skyline, Lake Michigan, and visitors, creating surreal and interactive photo opportunities. It's become an iconic symbol of Chicago, drawing millions of visitors annually for its playful design and free public access. If that's not the city you had in mind, feel free to give me a hint!"
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
                    ToolCallPart(tool_name='guess_city', args='{}', tool_call_id=IsStr()),
                ],
                usage=RequestUsage(
                    input_tokens=743, output_tokens=15, details={'reasoning_tokens': 483, 'cache_read_tokens': 170}
                ),
                model_name='grok-4-fast-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='guess_city',
                        content='Chicago',
                        tool_call_id=IsStr(),
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
                        tool_call_id=IsStr(),
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
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='xai',
                    ),
                    TextPart(
                        content="One of the most famous landmarks in Chicago is **Cloud Gate**, often nicknamed \"The Bean.\" It's a massive, reflective stainless steel sculpture in Millennium Park, designed by artist Anish Kapoor and unveiled in 2006. The 110-ton, bean-shaped installation mirrors the city's skyline, Lake Michigan, and visitors, creating surreal and interactive photo opportunities. It's become an iconic symbol of Chicago, drawing millions of visitors annually for its playful design and free public access. If that's not the city you had in mind, feel free to give me a hint!"
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=2973,
                    output_tokens=150,
                    details={'reasoning_tokens': 168, 'cache_read_tokens': 1506, 'server_side_tools_web_search': 1},
                ),
                model_name='grok-4-fast-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_builtin_mcp_server_tool(allow_model_requests: None, xai_provider: XaiProvider):
    m = XaiModel(XAI_REASONING_MODEL, provider=xai_provider)
    agent = Agent(
        m,
        instructions='You are a helpful assistant.',
        builtin_tools=[
            MCPServerTool(
                id='deepwiki',
                url='https://mcp.deepwiki.com/mcp',
                description='DeepWiki MCP server',
                allowed_tools=['ask_question'],
                headers={'custom-header-key': 'custom-header-value'},
            ),
        ],
        model_settings=XaiModelSettings(
            xai_include_encrypted_content=True,
            xai_include_mcp_output=True,
        ),
    )

    result = await agent.run('Can you tell me more about the pydantic/pydantic-ai repo? Keep your answer short')
    assert result.output == snapshot(
        "Pydantic AI (`pydantic/pydantic-ai`) is a Python framework for building production-grade LLM applications. It emphasizes type safety, structured outputs, dependency injection, and observability, with model-agnostic support for 20+ providers. Key features include graph-based agent execution, evaluation tools, and integrations like OpenTelemetry. It's structured as a monorepo with core packages for agents, graphs, and evals."
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Can you tell me more about the pydantic/pydantic-ai repo? Keep your answer short',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions='You are a helpful assistant.',
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
                        tool_name='mcp_server:deepwiki',
                        args={
                            'action': 'call_tool',
                            'tool_name': 'ask_question',
                            'tool_args': {
                                'repoName': 'pydantic/pydantic-ai',
                                'question': 'What is this repository about? Provide a short summary.',
                            },
                        },
                        tool_call_id=IsStr(),
                        provider_name='xai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_server:deepwiki',
                        content="""\
This repository, `pydantic/pydantic-ai`, is a Python agent framework designed for building production-grade applications with Large Language Models (LLMs) . It emphasizes type safety, structured outputs, dependency injection, and observability, providing a model-agnostic interface for over 20 LLM providers . The framework also includes comprehensive evaluation and testing infrastructure .

## Core Purpose and Features

The primary purpose of Pydantic AI is to simplify the development of reliable AI applications by offering a robust framework that integrates type-safety and an intuitive developer experience . It aims to provide a unified approach to interacting with various LLM providers and managing complex agent workflows .

Key features include:
*   **Type-Safe Agents**: Agents are generic `Agent[Deps, Output]` for compile-time validation, leveraging Pydantic for output validation and dependency injection .
*   **Model-Agnostic Design**: It supports over 15 LLM providers through a unified `Model` interface, allowing for easy switching between different models and providers .
*   **Structured Outputs**: Automatic Pydantic validation and reflection/self-correction ensure structured and reliable outputs from LLMs .
*   **Comprehensive Observability**: Integration with OpenTelemetry and native Logfire provides real-time debugging, performance monitoring, and cost tracking .
*   **Production-Ready Tooling**: This includes an evaluation framework (`pydantic-evals`), durable execution capabilities, and various protocol integrations like MCP, A2A, and AG-UI .
*   **Graph Execution**: The `pydantic_graph.Graph` module provides a graph-based state machine for orchestrating agent execution, using nodes like `UserPromptNode`, `ModelRequestNode`, and `CallToolsNode` .

## Framework Architecture

The framework is structured as a UV workspace monorepo, containing several packages .

### Core Packages
*   `pydantic-ai-slim`: Contains the core framework components such as `Agent`, `Model`, and tools .
*   `pydantic-ai`: A meta-package that includes all optional extras .

### Supporting Packages
*   `pydantic-graph`: Provides the graph execution engine with `Graph` and `BaseNode` .
*   `pydantic-evals`: An evaluation framework for datasets and evaluators .
*   `examples`: Contains example applications .
*   `clai`: Provides a CLI interface .

## Agent Execution Flow

The `Agent` class serves as the primary orchestrator . Agent execution is graph-based, utilizing a state machine from `pydantic_graph.Graph` . The execution involves three core node types:
*   `UserPromptNode`: Processes user input and creates initial `ModelRequest` .
*   `ModelRequestNode`: Calls `model.request()` or `model.request_stream()` and handles retries .
*   `CallToolsNode`: Executes tool functions via `RunContext[Deps]` .

The `Agent` provides methods like `run()`, `run_sync()`, and `run_stream()` for different execution scenarios .

## Notes
The `CLAUDE.md` file provides guidance for Claude Code when working with the repository, including development commands and an overview of core components and design patterns . The `mkdocs.yml` file defines the structure and content of the project's documentation, including navigation, plugins, and watch directories for various packages . The `docs/install.md` file details how to install the `pydantic-ai` package and its optional components, including a "slim" installation option for specific model dependencies .

Wiki pages you might want to explore:
- [Overview (pydantic/pydantic-ai)](/wiki/pydantic/pydantic-ai#1)

View this search on DeepWiki: https://deepwiki.com/search/what-is-this-repository-about_10565715-352a-4a8a-9260-3ed39b0b3226
""",
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='xai',
                    ),
                    TextPart(
                        content="Pydantic AI (`pydantic/pydantic-ai`) is a Python framework for building production-grade LLM applications. It emphasizes type safety, structured outputs, dependency injection, and observability, with model-agnostic support for 20+ providers. Key features include graph-based agent execution, evaluation tools, and integrations like OpenTelemetry. It's structured as a monorepo with core packages for agents, graphs, and evals."
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=1844,
                    output_tokens=140,
                    details={'reasoning_tokens': 202, 'cache_read_tokens': 771, 'server_side_tools_mcp_server': 1},
                ),
                model_name='grok-4-fast-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_builtin_mcp_server_tool_stream(allow_model_requests: None, xai_provider: XaiProvider):
    m = XaiModel(XAI_REASONING_MODEL, provider=xai_provider)
    agent = Agent(
        m,
        instructions='You are a helpful assistant.',
        builtin_tools=[
            MCPServerTool(
                id='deepwiki',
                url='https://mcp.deepwiki.com/mcp',
                description='DeepWiki MCP server',
                allowed_tools=['ask_question'],
                headers={'custom-header-key': 'custom-header-value'},
            ),
        ],
        model_settings=XaiModelSettings(
            xai_include_encrypted_content=True,
            xai_include_mcp_output=True,
        ),
    )

    event_parts: list[Any] = []
    async with agent.iter(
        user_prompt='Can you tell me more about the pydantic/pydantic-ai repo? Keep your answer short'
    ) as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert agent_run.result is not None
    assert agent_run.result.output == snapshot(
        "Pydantic/pydantic-ai is a GenAI Agent Framework built on Pydantic for creating type-safe Generative AI applications. It unifies interactions with LLMs from providers like OpenAI, Anthropic, Google, and others; supports agent orchestration, graph-based execution, tools, durable workflows, and multi-agent patterns. It's a monorepo with core packages for slim framework, graphs, and evals."
    )
    assert agent_run.result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Can you tell me more about the pydantic/pydantic-ai repo? Keep your answer short',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions='You are a helpful assistant.',
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
                        tool_name='mcp_server:deepwiki',
                        args={
                            'action': 'call_tool',
                            'tool_name': 'ask_question',
                            'tool_args': {
                                'repoName': 'pydantic/pydantic-ai',
                                'question': 'Provide a short summary of the repository, including its purpose and main features.',
                            },
                        },
                        tool_call_id=IsStr(),
                        provider_name='xai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_server:deepwiki',
                        content="""\
This repository, `pydantic/pydantic-ai`, is a GenAI Agent Framework that leverages Pydantic for building Generative AI applications . Its main purpose is to provide a unified and type-safe way to interact with various large language models (LLMs) from different providers, manage agent execution flows, and integrate with external tools and services .

## Purpose
The primary purpose of `pydantic-ai` is to simplify the development of robust and reliable Generative AI applications by providing a structured, type-safe, and extensible framework . It aims to abstract away the complexities of interacting with different LLM providers and managing agent workflows, allowing developers to focus on application logic .

## Main Features
The `pydantic-ai` repository offers several core features:
*   **Agent System**: The `Agent` class serves as the main orchestrator for managing interactions with LLMs and executing tasks . Agents can be configured with generic types for dependency injection and output validation, ensuring type safety throughout the application .
*   **Model Integration**: The framework provides a unified interface for integrating with various LLM providers, including OpenAI, Anthropic, Google, Groq, Cohere, Mistral, Bedrock, and HuggingFace . Each model integration follows a consistent settings pattern with provider-specific prefixes .
*   **Graph-based Execution**: Pydantic AI uses `pydantic-graph` to manage the execution flow of agents, representing it as a finite state machine .
*   **Tool System**: Function tools enable models to perform actions and retrieve additional information . Tools can be registered using decorators like `@agent.tool` or `@agent.tool_plain` .
*   **Output Handling**: The framework supports various output types, including `TextOutput`, `ToolOutput`, `NativeOutput`, and `PromptedOutput` .
*   **Durable Execution**: Pydantic AI integrates with durable execution systems like DBOS and Temporal, allowing agents to maintain state and resume execution after failures or restarts .
*   **Multi-Agent Patterns and Integrations**: The repository supports multi-agent applications and various integrations, including Pydantic Evals, Pydantic Graph, Logfire, Agent-User Interaction (AG-UI), Agent2Agent (A2A), and Clai .

## Notes
The repository is organized as a monorepo with core packages like `pydantic-ai-slim` (core framework), `pydantic-graph` (execution engine), and `pydantic-evals` (evaluation tools) . The documentation is built using MkDocs  and includes API references and examples .

Wiki pages you might want to explore:
- [OpenAI Models (pydantic/pydantic-ai)](/wiki/pydantic/pydantic-ai#3.2)
- [Google Gemini and Vertex AI Models (pydantic/pydantic-ai)](/wiki/pydantic/pydantic-ai#3.4)

View this search on DeepWiki: https://deepwiki.com/search/provide-a-short-summary-of-the_72abe8b9-cee5-4e55-80ce-3f1117e36815
""",
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    TextPart(
                        content="Pydantic/pydantic-ai is a GenAI Agent Framework built on Pydantic for creating type-safe Generative AI applications. It unifies interactions with LLMs from providers like OpenAI, Anthropic, Google, and others; supports agent orchestration, graph-based execution, tools, durable workflows, and multi-agent patterns. It's a monorepo with core packages for slim framework, graphs, and evals."
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=1783,
                    output_tokens=141,
                    details={'reasoning_tokens': 262, 'cache_read_tokens': 853, 'server_side_tools_mcp_server': 1},
                ),
                model_name='grok-4-fast-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )
    assert event_parts == snapshot(
        [
            PartStartEvent(index=0, part=ThinkingPart(content='', signature=IsStr(), provider_name='xai')),
            PartEndEvent(
                index=0,
                part=ThinkingPart(
                    content='',
                    signature=IsStr(),
                    provider_name='xai',
                ),
                next_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=1,
                part=BuiltinToolCallPart(
                    tool_name='mcp_server:deepwiki',
                    args={
                        'action': 'call_tool',
                        'tool_name': 'ask_question',
                        'tool_args': {
                            'repoName': 'pydantic/pydantic-ai',
                            'question': 'Provide a short summary of the repository, including its purpose and main features.',
                        },
                    },
                    tool_call_id=IsStr(),
                    provider_name='xai',
                ),
                previous_part_kind='thinking',
            ),
            PartEndEvent(
                index=1,
                part=BuiltinToolCallPart(
                    tool_name='mcp_server:deepwiki',
                    args={
                        'action': 'call_tool',
                        'tool_name': 'ask_question',
                        'tool_args': {
                            'repoName': 'pydantic/pydantic-ai',
                            'question': 'Provide a short summary of the repository, including its purpose and main features.',
                        },
                    },
                    tool_call_id=IsStr(),
                    provider_name='xai',
                ),
                next_part_kind='builtin-tool-return',
            ),
            PartStartEvent(
                index=2,
                part=BuiltinToolReturnPart(
                    tool_name='mcp_server:deepwiki',
                    content="""\
This repository, `pydantic/pydantic-ai`, is a GenAI Agent Framework that leverages Pydantic for building Generative AI applications . Its main purpose is to provide a unified and type-safe way to interact with various large language models (LLMs) from different providers, manage agent execution flows, and integrate with external tools and services .

## Purpose
The primary purpose of `pydantic-ai` is to simplify the development of robust and reliable Generative AI applications by providing a structured, type-safe, and extensible framework . It aims to abstract away the complexities of interacting with different LLM providers and managing agent workflows, allowing developers to focus on application logic .

## Main Features
The `pydantic-ai` repository offers several core features:
*   **Agent System**: The `Agent` class serves as the main orchestrator for managing interactions with LLMs and executing tasks . Agents can be configured with generic types for dependency injection and output validation, ensuring type safety throughout the application .
*   **Model Integration**: The framework provides a unified interface for integrating with various LLM providers, including OpenAI, Anthropic, Google, Groq, Cohere, Mistral, Bedrock, and HuggingFace . Each model integration follows a consistent settings pattern with provider-specific prefixes .
*   **Graph-based Execution**: Pydantic AI uses `pydantic-graph` to manage the execution flow of agents, representing it as a finite state machine .
*   **Tool System**: Function tools enable models to perform actions and retrieve additional information . Tools can be registered using decorators like `@agent.tool` or `@agent.tool_plain` .
*   **Output Handling**: The framework supports various output types, including `TextOutput`, `ToolOutput`, `NativeOutput`, and `PromptedOutput` .
*   **Durable Execution**: Pydantic AI integrates with durable execution systems like DBOS and Temporal, allowing agents to maintain state and resume execution after failures or restarts .
*   **Multi-Agent Patterns and Integrations**: The repository supports multi-agent applications and various integrations, including Pydantic Evals, Pydantic Graph, Logfire, Agent-User Interaction (AG-UI), Agent2Agent (A2A), and Clai .

## Notes
The repository is organized as a monorepo with core packages like `pydantic-ai-slim` (core framework), `pydantic-graph` (execution engine), and `pydantic-evals` (evaluation tools) . The documentation is built using MkDocs  and includes API references and examples .

Wiki pages you might want to explore:
- [OpenAI Models (pydantic/pydantic-ai)](/wiki/pydantic/pydantic-ai#3.2)
- [Google Gemini and Vertex AI Models (pydantic/pydantic-ai)](/wiki/pydantic/pydantic-ai#3.4)

View this search on DeepWiki: https://deepwiki.com/search/provide-a-short-summary-of-the_72abe8b9-cee5-4e55-80ce-3f1117e36815
""",
                    tool_call_id=IsStr(),
                    timestamp=IsDatetime(),
                    provider_name='xai',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(signature_delta=IsStr()),
            ),
            PartStartEvent(index=3, part=TextPart(content='P'), previous_part_kind='builtin-tool-return'),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='yd')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='antic')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='/p')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='yd')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='antic')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='-ai')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' is')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' Gen')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='AI')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' Agent')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' Framework')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' built')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' on')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' P')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='yd')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='antic')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' creating')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' type')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='-safe')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' Gener')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='ative')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' AI')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' applications')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' It')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' un')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='ifies')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' interactions')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' LL')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='Ms')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' from')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' providers')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' like')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' Open')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='AI')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' Anthrop')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='ic')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' Google')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' others')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=';')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' supports')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' agent')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' orchestration')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' graph')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='-based')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' execution')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' tools')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' durable')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' workflows')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' multi')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='-agent')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' patterns')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=" It's")),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' mon')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='ore')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='po')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' core')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' packages')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' slim')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' framework')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' graphs')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' ev')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='als')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='.')),
            PartEndEvent(
                index=3,
                part=TextPart(
                    content="Pydantic/pydantic-ai is a GenAI Agent Framework built on Pydantic for creating type-safe Generative AI applications. It unifies interactions with LLMs from providers like OpenAI, Anthropic, Google, and others; supports agent orchestration, graph-based execution, tools, durable workflows, and multi-agent patterns. It's a monorepo with core packages for slim framework, graphs, and evals."
                ),
            ),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='mcp_server:deepwiki',
                    args={
                        'action': 'call_tool',
                        'tool_name': 'ask_question',
                        'tool_args': {
                            'repoName': 'pydantic/pydantic-ai',
                            'question': 'Provide a short summary of the repository, including its purpose and main features.',
                        },
                    },
                    tool_call_id=IsStr(),
                    provider_name='xai',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='mcp_server:deepwiki',
                    content="""\
This repository, `pydantic/pydantic-ai`, is a GenAI Agent Framework that leverages Pydantic for building Generative AI applications . Its main purpose is to provide a unified and type-safe way to interact with various large language models (LLMs) from different providers, manage agent execution flows, and integrate with external tools and services .

## Purpose
The primary purpose of `pydantic-ai` is to simplify the development of robust and reliable Generative AI applications by providing a structured, type-safe, and extensible framework . It aims to abstract away the complexities of interacting with different LLM providers and managing agent workflows, allowing developers to focus on application logic .

## Main Features
The `pydantic-ai` repository offers several core features:
*   **Agent System**: The `Agent` class serves as the main orchestrator for managing interactions with LLMs and executing tasks . Agents can be configured with generic types for dependency injection and output validation, ensuring type safety throughout the application .
*   **Model Integration**: The framework provides a unified interface for integrating with various LLM providers, including OpenAI, Anthropic, Google, Groq, Cohere, Mistral, Bedrock, and HuggingFace . Each model integration follows a consistent settings pattern with provider-specific prefixes .
*   **Graph-based Execution**: Pydantic AI uses `pydantic-graph` to manage the execution flow of agents, representing it as a finite state machine .
*   **Tool System**: Function tools enable models to perform actions and retrieve additional information . Tools can be registered using decorators like `@agent.tool` or `@agent.tool_plain` .
*   **Output Handling**: The framework supports various output types, including `TextOutput`, `ToolOutput`, `NativeOutput`, and `PromptedOutput` .
*   **Durable Execution**: Pydantic AI integrates with durable execution systems like DBOS and Temporal, allowing agents to maintain state and resume execution after failures or restarts .
*   **Multi-Agent Patterns and Integrations**: The repository supports multi-agent applications and various integrations, including Pydantic Evals, Pydantic Graph, Logfire, Agent-User Interaction (AG-UI), Agent2Agent (A2A), and Clai .

## Notes
The repository is organized as a monorepo with core packages like `pydantic-ai-slim` (core framework), `pydantic-graph` (execution engine), and `pydantic-evals` (evaluation tools) . The documentation is built using MkDocs  and includes API references and examples .

Wiki pages you might want to explore:
- [OpenAI Models (pydantic/pydantic-ai)](/wiki/pydantic/pydantic-ai#3.2)
- [Google Gemini and Vertex AI Models (pydantic/pydantic-ai)](/wiki/pydantic/pydantic-ai#3.4)

View this search on DeepWiki: https://deepwiki.com/search/provide-a-short-summary-of-the_72abe8b9-cee5-4e55-80ce-3f1117e36815
""",
                    tool_call_id=IsStr(),
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
                'model': 'grok-4-fast-non-reasoning',
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


async def test_xai_model_properties():
    """Test xAI model properties."""
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(api_key='test-key'))

    assert m.model_name == XAI_NON_REASONING_MODEL
    assert m.system == 'xai'


async def test_xai_reasoning_simple(allow_model_requests: None):
    """Test reasoning output mapping to ThinkingPart (mocked)."""
    response = create_response(content='4', reasoning_content='...', encrypted_content='sig-123')
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, model_settings=XaiModelSettings(xai_include_encrypted_content=True, max_tokens=20))

    result = await agent.run('What is 2+2? Return just number.')
    assert result.output == '4'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is 2+2? Return just number.', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content='...', signature='sig-123', provider_name='xai'),
                    TextPart(content='4'),
                ],
                model_name=XAI_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )
    assert get_mock_chat_create_kwargs(mock_client)[0]['use_encrypted_content'] is True


async def test_xai_encrypted_content_only(allow_model_requests: None):
    """Test encrypted content (signature) appears when enabled"""
    response = create_response(content='4', encrypted_content='sig-abc')
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, model_settings=XaiModelSettings(xai_include_encrypted_content=True, max_tokens=20))

    result = await agent.run('What is 2+2? Return just "4".')
    assert result.output == '4'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is 2+2? Return just "4".', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ThinkingPart(content='', signature='sig-abc', provider_name='xai'), TextPart(content='4')],
                model_name=XAI_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_stream_with_encrypted_reasoning(allow_model_requests: None):
    """Test xAI streaming with reasoning + encrypted reasoning signature enabled."""
    stream = [
        [
            get_grok_reasoning_text_chunk(
                '1, 2, 3',
                reasoning_content='...',
                encrypted_content='sig',
                finish_reason='stop',
            ),
        ]
    ]
    mock_client = MockXai.create_mock_stream(stream)
    m = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, model_settings=XaiModelSettings(xai_include_encrypted_content=True, max_tokens=30))

    async with agent.run_stream('Count to 3') as result:
        assert not result.is_complete
        text_chunks = [c async for c in result.stream_text(debounce_by=None)]
        assert text_chunks == snapshot(['1, 2, 3'])
        assert result.is_complete
        # Ensure the final accumulated response contains the expected ThinkingPart (reasoning + signature).
        final_response: ModelResponse | None = None
        async for response, is_last in result.stream_responses(debounce_by=None):
            if is_last:
                final_response = response
        assert final_response is not None
        assert any(
            isinstance(p, ThinkingPart) and p.content == '...' and p.signature == 'sig' and p.provider_name == 'xai'
            for p in final_response.parts
        )


async def test_xai_stream_events_with_reasoning(allow_model_requests: None, xai_provider: XaiProvider):
    """Test xAI streaming events with reasoning model (recorded via proto cassette)."""
    m = XaiModel(XAI_REASONING_MODEL, provider=xai_provider)
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
                    input_tokens=165, output_tokens=40, details={'reasoning_tokens': 121, 'cache_read_tokens': 151}
                ),
                model_name='grok-4-fast-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    assert event_parts == snapshot(
        [
            PartStartEvent(index=0, part=ThinkingPart(content='', signature=IsStr(), provider_name='xai')),
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


async def test_xai_usage_with_reasoning_tokens(allow_model_requests: None):
    """Test that xAI usage extraction includes reasoning tokens when available (mocked)."""
    response = create_response(
        content='42',
        encrypted_content='sig',
        usage=create_usage(prompt_tokens=10, completion_tokens=2, reasoning_tokens=7),
    )
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, model_settings=XaiModelSettings(xai_include_encrypted_content=True, max_tokens=20))

    result = await agent.run('What is the meaning of life? Keep it very short.')
    assert result.output == '42'
    assert result.usage() == snapshot(
        RunUsage(
            input_tokens=10,
            output_tokens=2,
            requests=1,
            details={'reasoning_tokens': 7},
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
                provider_url='https://api.x.ai/v1',
                provider_name='xai',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_code_execution_default_output(allow_model_requests: None) -> None:
    """Test code execution with default example output."""
    response = create_code_execution_response(code='print(2+2)', assistant_text='Tool completed successfully.')
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
                        tool_call_id=IsStr(),
                        provider_name='xai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'stdout': '4\n', 'stderr': '', 'output_files': {}, 'error': '', 'ret': ''},
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    TextPart(content='Tool completed successfully.'),
                ],
                usage=RequestUsage(),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_url='https://api.x.ai/v1',
                provider_name='xai',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_web_search_default_output(allow_model_requests: None) -> None:
    """Test web search with default example output."""
    response = create_web_search_response(query='test query', assistant_text='Tool completed successfully.')
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
                        tool_call_id=IsStr(),
                        provider_name='xai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={},
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    TextPart(content='Tool completed successfully.'),
                ],
                usage=RequestUsage(),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_url='https://api.x.ai/v1',
                provider_name='xai',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_mcp_server_default_output(allow_model_requests: None) -> None:
    """Test MCP server tool with default example output."""
    response = create_mcp_server_response(
        server_id='linear', tool_name='list_issues', assistant_text='Tool completed successfully.'
    )
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
                        tool_name='mcp_server:linear',
                        args={'action': 'call_tool', 'tool_name': 'list_issues', 'tool_args': {}},
                        tool_call_id=IsStr(),
                        provider_name='xai',
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
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    TextPart(content='Tool completed successfully.'),
                ],
                usage=RequestUsage(),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_url='https://api.x.ai/v1',
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
                'model': 'grok-4-fast-non-reasoning',
                'messages': [{'content': [{'text': 'Hello'}], 'role': 'ROLE_USER'}],
                'tools': None,
                'tool_choice': None,
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            },
            {
                'model': 'grok-4-fast-non-reasoning',
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
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
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
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
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
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
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
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
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
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
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
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_builtin_tool_call_in_history(allow_model_requests: None):
    """Test that BuiltinToolCallPart and BuiltinToolReturnPart in history are mapped."""
    # First response with code execution
    response1 = create_code_execution_response(code='print(2+2)', assistant_text='Tool completed successfully.')
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
                        'content': [{'text': 'Tool completed successfully.'}],
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
                        tool_call_id=IsStr(),
                        provider_name='xai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'stdout': '4\n', 'stderr': '', 'output_files': {}, 'error': '', 'ret': ''},
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    TextPart(content='Tool completed successfully.'),
                ],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
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
                provider_url='https://api.x.ai/v1',
                provider_name='xai',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


def test_builtin_tool_call_part_failed_status(allow_model_requests: None):
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
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
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
                provider_url='https://api.x.ai/v1',
                provider_name='xai',
                provider_response_id=IsStr(),
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
        'xai_include_inline_citations': True,
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
                    chat_pb2.IncludeOption.INCLUDE_OPTION_INLINE_CITATIONS,
                    chat_pb2.IncludeOption.INCLUDE_OPTION_MCP_CALL_OUTPUT,
                ],
            }
        ]
    )


async def test_xai_stream_server_side_tool_call_and_return_dedupes(allow_model_requests: None):
    """Server-side tool call/return deltas should only produce one call and one return part each."""

    # Intentionally rely on MockXai defaults for `tool_type` and `status` here to keep mock_xai.py covered without
    # a dedicated "coverage-only" test.
    server_tool_call = create_server_tool_call(
        tool_name='web_search',
        arguments={'query': 'x'},
        tool_call_id='server_tool_1',
    )

    tool_output_json = json.dumps({'status': 'ok'})

    stream: list[tuple[chat_types.Response, chat_types.Chunk]] = [
        (
            create_response(content='', tool_calls=[server_tool_call], finish_reason='stop'),
            create_stream_chunk(role=chat_pb2.MessageRole.ROLE_ASSISTANT, tool_calls=[server_tool_call]),
        ),
        # Duplicate call (should be ignored)
        (
            create_response(content='', tool_calls=[server_tool_call], finish_reason='stop'),
            create_stream_chunk(role=chat_pb2.MessageRole.ROLE_ASSISTANT, tool_calls=[server_tool_call]),
        ),
        (
            create_response(content=tool_output_json, tool_calls=[server_tool_call], finish_reason='stop'),
            create_stream_chunk(
                role=chat_pb2.MessageRole.ROLE_TOOL, tool_calls=[server_tool_call], content=tool_output_json
            ),
        ),
        # Duplicate return with same content (should be ignored)
        (
            create_response(content=tool_output_json, tool_calls=[server_tool_call], finish_reason='stop'),
            create_stream_chunk(
                role=chat_pb2.MessageRole.ROLE_TOOL, tool_calls=[server_tool_call], content=tool_output_json
            ),
        ),
        # Add assistant text so the agent has a stable final output.
        (
            create_response(content='done', finish_reason='stop'),
            create_stream_chunk(role=chat_pb2.MessageRole.ROLE_ASSISTANT, content='done'),
        ),
    ]

    mock_client = MockXai.create_mock_stream([stream])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        final_response: ModelResponse | None = None
        async for response, is_last in result.stream_responses(debounce_by=None):
            if is_last:
                final_response = response

    assert final_response is not None
    builtin_calls = [p for p in final_response.parts if isinstance(p, BuiltinToolCallPart)]
    builtin_returns = [p for p in final_response.parts if isinstance(p, BuiltinToolReturnPart)]
    assert len(builtin_calls) == 1
    assert builtin_calls[0].tool_name == 'web_search'
    assert builtin_calls[0].args == {'query': 'x'}
    assert builtin_calls[0].tool_call_id == 'server_tool_1'
    assert len(builtin_returns) == 1
    assert builtin_returns[0].tool_name == 'web_search'
    assert builtin_returns[0].content == {'status': 'ok'}
    assert builtin_returns[0].tool_call_id == 'server_tool_1'


async def test_xai_stream_server_side_tool_call_ignored_for_unknown_role(allow_model_requests: None):
    """Server-side tool deltas with an unknown role should be ignored."""

    server_tool_call = create_server_tool_call(
        tool_name='web_search',
        arguments={'query': 'x'},
        tool_call_id='server_tool_1',
        tool_type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_WEB_SEARCH_TOOL,
        status=chat_pb2.ToolCallStatus.TOOL_CALL_STATUS_COMPLETED,
    )

    stream: list[tuple[chat_types.Response, chat_types.Chunk]] = [
        (
            create_response(content='', tool_calls=[server_tool_call], finish_reason='stop'),
            create_stream_chunk(role=chat_pb2.MessageRole.ROLE_USER, tool_calls=[server_tool_call]),
        ),
        (
            create_response(content='done', finish_reason='stop'),
            create_stream_chunk(role=chat_pb2.MessageRole.ROLE_ASSISTANT, content='done'),
        ),
    ]

    mock_client = MockXai.create_mock_stream([stream])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        final_response: ModelResponse | None = None
        async for response, is_last in result.stream_responses(debounce_by=None):
            if is_last:
                final_response = response

    assert final_response is not None
    assert not any(isinstance(p, BuiltinToolCallPart) for p in final_response.parts)
    assert not any(isinstance(p, BuiltinToolReturnPart) for p in final_response.parts)


async def test_xai_stream_tool_call_without_name_ignored(allow_model_requests: None):
    """Tool deltas with an empty function name should be ignored."""

    no_name_tool_call = chat_pb2.ToolCall(
        id='no-name',
        type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL,
        function=chat_pb2.FunctionCall(name='', arguments='{"x": 1}'),
    )

    stream: list[tuple[chat_types.Response, chat_types.Chunk]] = [
        (
            create_response(content='', tool_calls=[no_name_tool_call], finish_reason='stop'),
            create_stream_chunk(tool_calls=[no_name_tool_call]),
        ),
        (create_response(content='done', finish_reason='stop'), create_stream_chunk(content='done')),
    ]

    mock_client = MockXai.create_mock_stream([stream])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        final_response: ModelResponse | None = None
        async for response, is_last in result.stream_responses(debounce_by=None):
            if is_last:
                final_response = response

    assert final_response is not None
    assert not any(isinstance(p, ToolCallPart) for p in final_response.parts)


async def test_xai_stream_client_side_tool_call_prefers_delta_when_accumulated_missing_or_empty(
    allow_model_requests: None,
):
    """When accumulated tool-call args are missing/empty, we should keep the delta args."""

    delta_client_call = chat_pb2.ToolCall(
        id='tool-123',
        type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL,
        function=chat_pb2.FunctionCall(name='final_result', arguments='{"first": "One"}'),
    )
    other_client_call = chat_pb2.ToolCall(
        id='other-1',
        type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL,
        function=chat_pb2.FunctionCall(name='final_result', arguments='{"other": true}'),
    )
    matching_empty_accumulated = chat_pb2.ToolCall(
        id='tool-123',
        type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL,
        function=chat_pb2.FunctionCall(name='final_result', arguments=''),
    )

    # Frame 1: no matching accumulated tool call id (forces delta).
    response_no_match = create_response(content='', tool_calls=[other_client_call], finish_reason='stop')
    # Frame 2: matching id exists but accumulated args are empty (forces delta).
    response_match_empty_args = create_response(
        content='', tool_calls=[other_client_call, matching_empty_accumulated], finish_reason='stop'
    )

    stream: list[tuple[chat_types.Response, chat_types.Chunk]] = [
        (response_no_match, create_stream_chunk(tool_calls=[delta_client_call])),
        (response_match_empty_args, create_stream_chunk(tool_calls=[delta_client_call])),
        # Add assistant text so the agent has a stable final output.
        (
            create_response(content='done', finish_reason='stop'),
            create_stream_chunk(content='done', tool_calls=[], finish_reason='stop'),
        ),
    ]

    mock_client = MockXai.create_mock_stream([stream])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        final_response: ModelResponse | None = None
        async for response, is_last in result.stream_responses(debounce_by=None):
            if is_last:
                final_response = response

    assert final_response is not None
    tool_calls = [p for p in final_response.parts if isinstance(p, ToolCallPart)]
    assert tool_calls, 'expected at least one client-side ToolCallPart'
    assert any(p.tool_name == 'final_result' and p.args == '{"first": "One"}' for p in tool_calls)


async def test_xai_stream_client_tool_args_non_prefix_path(allow_model_requests: None):
    """Force the client-side tool args fallback path where accumulated args reset mid-stream.

    This tests when accumulated_args doesn't start with prev_args, we use the
    full accumulated_args as the delta. This is a defensive fallback for edge cases where
    the server's accumulated view changes unexpectedly (e.g., corrections/resets).
    """
    # Frame 1: Tool call with initial args 'ABC'
    # Frame 2: Same tool call but accumulated args change to 'XYZ' (not a prefix of 'ABC')
    # This triggers: args_delta = accumulated_args or None

    stream = [
        # Frame 1: New tool call starts with args 'ABC'
        get_grok_tool_chunk('final_result', 'ABC', accumulated_args='ABC'),
        # Frame 2: Accumulated args reset to 'XYZ' (doesn't start with 'ABC')
        get_grok_tool_chunk(None, 'XYZ', accumulated_args='XYZ'),
        # Frame 3: Final text response to finish
        get_grok_text_chunk('done', finish_reason='stop'),
    ]
    mock_client = MockXai.create_mock_stream([stream])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    # Use plain agent (no output_type) to avoid JSON validation on the tool args
    agent = Agent(m)

    async with agent.run_stream('') as result:
        final_response: ModelResponse | None = None
        async for response, is_last in result.stream_responses(debounce_by=None):
            if is_last:
                final_response = response

    assert final_response is not None
    # The tool call part should have args that include both delta applications
    # (the behavior is to concatenate, so we get 'ABCXYZ')
    tool_calls = [p for p in final_response.parts if isinstance(p, ToolCallPart)]
    assert tool_calls, 'expected at least one client-side ToolCallPart'
    assert any(p.tool_name == 'final_result' and p.args == 'ABCXYZ' for p in tool_calls)


async def test_xai_stream_reasoning_delta_non_prefix_path(allow_model_requests: None):
    """Force the reasoning-delta fallback path where accumulated reasoning resets mid-stream."""
    # Frame 1: reasoning starts.
    r1 = create_response(content='', reasoning_content='abc')
    c1 = create_stream_chunk(reasoning_content='abc')

    # Frame 2: accumulated reasoning changes to a different non-prefix string, forcing the fallback branch.
    r2 = create_response(content='done', reasoning_content='XYZ', finish_reason='stop')
    c2 = create_stream_chunk(content='done', reasoning_content='XYZ', finish_reason='stop')

    mock_client = MockXai.create_mock_stream([[(r1, c1), (r2, c2)]])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert [t async for t in result.stream_text(debounce_by=None)] == ['done']


async def test_xai_map_builtin_tool_call_part_unknown_tool_name_ignored(allow_model_requests: None):
    """Cover the fallback path where a builtin tool call part has an unknown tool name."""
    response = create_response(content='ok', usage=create_usage(prompt_tokens=1, completion_tokens=1))
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    message_history: list[ModelMessage] = [
        ModelResponse(
            parts=[
                BuiltinToolCallPart(
                    tool_name='unknown_builtin_tool',
                    args={'x': 1},
                    tool_call_id='unknown_tool_1',
                    provider_name='xai',
                )
            ],
            model_name=XAI_NON_REASONING_MODEL,
            timestamp=IsNow(tz=timezone.utc),
            provider_name='xai',
        )
    ]

    result = await agent.run('hello', message_history=message_history)
    assert result.output == 'ok'


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
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
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
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
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
                    BuiltinToolCallPart(
                        tool_name='x_search', args={'query': 'test'}, tool_call_id=IsStr(), provider_name='xai'
                    ),
                    TextPart(content='Search results here'),
                ],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
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
                provider_url='https://api.x.ai/v1',
                provider_name='xai',
                provider_response_id=IsStr(),
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
                        tool_call_id=IsStr(),
                        provider_name='xai',
                    ),
                    TextPart(content='Search complete'),
                ],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
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
    response1 = create_web_search_response(
        query='test query', content='Search results', assistant_text='Tool completed successfully.'
    )
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
                    {'content': [{'text': 'Tool completed successfully.'}], 'role': 'ROLE_ASSISTANT'},
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
                        tool_call_id=IsStr(),
                        provider_name='xai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content='Search results',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    TextPart(content='Tool completed successfully.'),
                ],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
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
                provider_url='https://api.x.ai/v1',
                provider_name='xai',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_mcp_server_tool_in_history(allow_model_requests: None):
    """Test that MCPServerTool builtin calls in history are mapped."""
    # First response with MCP server tool
    response1 = create_mcp_server_response(
        server_id='my-server',
        tool_name='get_data',
        content={'data': 'MCP result'},
        tool_input={'param': 'value'},
        assistant_text='Tool completed successfully.',
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
                                'function': {'name': 'my-server.get_data', 'arguments': '{"param": "value"}'},
                            }
                        ],
                    },
                    {'content': [{'text': 'Tool completed successfully.'}], 'role': 'ROLE_ASSISTANT'},
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
                        args={'action': 'call_tool', 'tool_name': 'get_data', 'tool_args': {'param': 'value'}},
                        tool_call_id=IsStr(),
                        provider_name='xai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_server:my-server',
                        content={'data': 'MCP result'},
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    TextPart(content='Tool completed successfully.'),
                ],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
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
                provider_url='https://api.x.ai/v1',
                provider_name='xai',
                provider_response_id=IsStr(),
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
                provider_url='https://api.x.ai/v1',
                provider_name='xai',
                provider_response_id=IsStr(),
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
    # This triggers the branch where item.signature is falsy
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
    # This triggers the branch where error_msg is falsy
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


async def test_map_user_prompt_with_text_content():
    response = create_response(content='test response')
    mock_client = MockXai.create_mock([response])
    model = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))

    m = await model._map_user_prompt(  # pyright: ignore[reportPrivateUsage]
        part=UserPromptPart(
            content=['Hi there', TextContent(content='This is a test', metadata={'format': 'markdown'})],
            timestamp=IsDatetime(),
        )
    )

    assert repr(m) == snapshot("""\
content {
  text: "Hi there"
}
content {
  text: "This is a test"
}
role: ROLE_USER
""")


# End of tests
