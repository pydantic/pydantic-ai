"""Tests for xAI search tool integrations (XSearchTool, FileSearchTool, grok profiles)."""

from __future__ import annotations as _annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from pydantic_ai import (
    Agent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    FileSearchTool,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    UserPromptPart,
    XSearchTool,
)
from pydantic_ai.messages import RequestUsage
from pydantic_ai.profiles.grok import GrokModelProfile, grok_model_profile
from pydantic_ai.usage import RunUsage

from ..._inline_snapshot import snapshot
from ...conftest import IsDatetime, IsNow, IsStr, try_import
from ..mock_xai import (
    MockXai,
    create_collections_search_response,
    create_mixed_tools_response,
    create_response,
    create_usage,
    create_x_search_response,
    get_mock_chat_create_kwargs,
)

with try_import() as imports_successful:
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

XAI_NON_REASONING_MODEL = 'grok-4-fast-non-reasoning'
XAI_REASONING_MODEL = 'grok-4-fast-reasoning'


# =============================================================================
# Grok model profile tests
# =============================================================================


@pytest.mark.parametrize(
    'model_name,expected_thinking,expected_always',
    [
        ('grok-4-fast-reasoning', True, True),
        ('grok-4-1-reasoning', True, True),
        ('grok-4-fast-non-reasoning', False, False),
        ('grok-4-1-fast-non-reasoning', False, False),
        ('grok-3-mini', True, False),
        ('grok-3-mini-fast', True, False),
        ('grok-3', False, False),
    ],
    ids=[
        'grok-4-fast-reasoning',
        'grok-4-1-reasoning',
        'grok-4-fast-non-reasoning',
        'grok-4-1-fast-non-reasoning',
        'grok-3-mini',
        'grok-3-mini-fast',
        'grok-3',
    ],
)
def test_grok_model_profile_thinking(model_name: str, expected_thinking: bool, expected_always: bool) -> None:
    profile = grok_model_profile(model_name)
    assert profile is not None
    assert profile.supports_thinking == expected_thinking
    assert profile.thinking_always_enabled == expected_always


def test_grok_model_profile_builtin_tools() -> None:
    grok4_profile = grok_model_profile('grok-4-fast-non-reasoning')
    assert grok4_profile is not None
    assert isinstance(grok4_profile, GrokModelProfile)
    assert grok4_profile.grok_supports_builtin_tools is True

    grok3_profile = grok_model_profile('grok-3')
    assert grok3_profile is not None
    assert isinstance(grok3_profile, GrokModelProfile)
    assert grok3_profile.grok_supports_builtin_tools is False


# =============================================================================
# XSearchTool validation tests
# =============================================================================


def test_x_search_tool_validation():
    """Test XSearchTool validation rules."""
    with pytest.raises(ValueError, match='Cannot specify both allowed_x_handles and excluded_x_handles'):
        XSearchTool(allowed_x_handles=['foo'], excluded_x_handles=['bar'])

    with pytest.raises(ValueError, match='allowed_x_handles cannot contain more than 10 handles'):
        XSearchTool(allowed_x_handles=['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11'])

    with pytest.raises(ValueError, match='excluded_x_handles cannot contain more than 10 handles'):
        XSearchTool(excluded_x_handles=['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11'])

    tool = XSearchTool(allowed_x_handles=['handle1', 'handle2'])
    assert tool.allowed_x_handles == ['handle1', 'handle2']
    assert tool.excluded_x_handles is None

    tool = XSearchTool(excluded_x_handles=['spam1', 'spam2'])
    assert tool.excluded_x_handles == ['spam1', 'spam2']
    assert tool.allowed_x_handles is None

    tool = XSearchTool()
    assert tool.allowed_x_handles is None
    assert tool.excluded_x_handles is None

    tool = XSearchTool(from_date=datetime(2024, 6, 1), to_date=datetime(2024, 12, 31))
    assert tool.from_date == datetime(2024, 6, 1)
    assert tool.to_date == datetime(2024, 12, 31)


# =============================================================================
# XSearchTool → x_search VCR tests
# =============================================================================


async def test_xai_builtin_x_search_tool(allow_model_requests: None, xai_provider: XaiProvider):
    """Test xAI's built-in x_search tool (non-streaming, recorded via proto cassette)."""
    m = XaiModel(XAI_REASONING_MODEL, provider=xai_provider)
    agent = Agent(
        m,
        builtin_tools=[XSearchTool()],
        model_settings=XaiModelSettings(
            xai_include_encrypted_content=True,
            xai_include_x_search_output=True,
        ),
    )

    result = await agent.run('What are the latest posts about PydanticAI on X? Reply with just the key topic.')
    assert result.output == snapshot('Building AI agents and workflows with PydanticAI')

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What are the latest posts about PydanticAI on X? Reply with just the key topic.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
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
                        tool_name='x_search',
                        args={'query': 'PydanticAI', 'mode': 'Latest'},
                        tool_call_id=IsStr(),
                        provider_name='xai',
                    ),
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='xai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='x_search',
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
                    TextPart(content='Building AI agents and workflows with PydanticAI'),
                ],
                usage=RequestUsage(
                    input_tokens=3774,
                    cache_read_tokens=1639,
                    output_tokens=50,
                    details={'reasoning_tokens': 439, 'server_side_tools_x_search': 1},
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


async def test_xai_builtin_x_search_tool_stream(allow_model_requests: None, xai_provider: XaiProvider):
    """Test xAI's built-in x_search tool with streaming (recorded via proto cassette)."""
    m = XaiModel(XAI_REASONING_MODEL, provider=xai_provider)
    agent = Agent(
        m,
        builtin_tools=[XSearchTool()],
        model_settings=XaiModelSettings(
            xai_include_encrypted_content=True,
            xai_include_x_search_output=True,
        ),
    )

    event_parts: list[Any] = []
    async with agent.iter(
        user_prompt='Search X for the latest PydanticAI updates. Reply with just the key topic.'
    ) as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert agent_run.result is not None
    messages = agent_run.result.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Search X for the latest PydanticAI updates. Reply with just the key topic.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
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
                        tool_name='x_search',
                        args={'query': 'PydanticAI OR "Pydantic AI"', 'limit': 10, 'mode': 'Latest'},
                        tool_call_id=IsStr(),
                        provider_name='xai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='x_search',
                        content=None,
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    TextPart(
                        content='Pydantic sponsoring To The Americas hackathon with prizes for best Pydantic AI usage.'
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=5991,
                    cache_read_tokens=2720,
                    output_tokens=74,
                    details={'reasoning_tokens': 770, 'server_side_tools_x_search': 1},
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


# =============================================================================
# XSearchTool → x_search mock tests (SDK parameter verification)
# =============================================================================


async def test_xai_builtin_x_search_tool_with_handles(allow_model_requests: None):
    """Test that XSearchTool handle filtering params are sent to the xAI SDK."""
    response = create_x_search_response(
        query='AI updates',
        content={'results': [{'text': 'AI news from @OpenAI'}]},
        assistant_text='Found filtered posts.',
    )
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(
        m,
        builtin_tools=[XSearchTool(allowed_x_handles=['OpenAI', 'AnthropicAI'])],
    )

    await agent.run('What are OpenAI and Anthropic tweeting about?')

    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [
                    {'content': [{'text': 'What are OpenAI and Anthropic tweeting about?'}], 'role': 'ROLE_USER'}
                ],
                'tools': [
                    {
                        'x_search': {
                            'allowed_x_handles': ['OpenAI', 'AnthropicAI'],
                            'enable_image_understanding': False,
                            'enable_video_understanding': False,
                        }
                    }
                ],
                'tool_choice': 'auto',
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            }
        ]
    )


async def test_xai_builtin_x_search_tool_with_date_range(allow_model_requests: None):
    """Test that XSearchTool date params are sent to the xAI SDK."""
    response = create_x_search_response(
        query='PydanticAI release',
        content={'results': []},
        assistant_text='No posts found in date range.',
    )
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(
        m,
        builtin_tools=[
            XSearchTool(
                from_date=datetime(2024, 1, 1),
                to_date=datetime(2024, 12, 31),
            )
        ],
    )

    await agent.run('Any PydanticAI posts in 2024?')

    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [{'content': [{'text': 'Any PydanticAI posts in 2024?'}], 'role': 'ROLE_USER'}],
                'tools': [
                    {
                        'x_search': {
                            'from_date': '2024-01-01T00:00:00Z',
                            'to_date': '2024-12-31T00:00:00Z',
                            'enable_image_understanding': False,
                            'enable_video_understanding': False,
                        }
                    }
                ],
                'tool_choice': 'auto',
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            }
        ]
    )


async def test_xai_x_search_tool_type_in_response(allow_model_requests: None):
    """Test handling of x_search tool type in responses (without agent-side XSearchTool)."""
    x_search_tool_call = chat_pb2.ToolCall(
        id='x_search_001',
        type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_X_SEARCH_TOOL,
        status=chat_pb2.ToolCallStatus.TOOL_CALL_STATUS_COMPLETED,
        function=chat_pb2.FunctionCall(
            name='x_search',
            arguments='{"query": "test"}',
        ),
    )

    response = create_mixed_tools_response([x_search_tool_call], text_content='Search results here')
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    result = await agent.run('Search for something')

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


async def test_xai_x_search_builtin_tool_call_in_history(allow_model_requests: None):
    """Test that XSearchTool BuiltinToolCallPart in history is properly mapped back to xAI."""
    response1 = create_x_search_response(query='pydantic updates', assistant_text='Found posts about PydanticAI.')
    response2 = create_response(content='The posts were about PydanticAI releases.')

    mock_client = MockXai.create_mock([response1, response2])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, builtin_tools=[XSearchTool()])

    result1 = await agent.run('Search for pydantic updates')
    result2 = await agent.run('What were the posts about?', message_history=result1.new_messages())

    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [{'content': [{'text': 'Search for pydantic updates'}], 'role': 'ROLE_USER'}],
                'tools': [{'x_search': {'enable_image_understanding': False, 'enable_video_understanding': False}}],
                'tool_choice': 'auto',
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            },
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [
                    {'content': [{'text': 'Search for pydantic updates'}], 'role': 'ROLE_USER'},
                    {
                        'content': [{'text': ''}],
                        'role': 'ROLE_ASSISTANT',
                        'tool_calls': [
                            {
                                'id': 'x_search_001',
                                'type': 'TOOL_CALL_TYPE_X_SEARCH_TOOL',
                                'status': 'TOOL_CALL_STATUS_COMPLETED',
                                'function': {'name': 'x_search', 'arguments': '{"query":"pydantic updates"}'},
                            }
                        ],
                    },
                    {
                        'content': [{'text': 'Found posts about PydanticAI.'}],
                        'role': 'ROLE_ASSISTANT',
                    },
                    {'content': [{'text': 'What were the posts about?'}], 'role': 'ROLE_USER'},
                ],
                'tools': [{'x_search': {'enable_image_understanding': False, 'enable_video_understanding': False}}],
                'tool_choice': 'auto',
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            },
        ]
    )

    assert result2.output == 'The posts were about PydanticAI releases.'


async def test_xai_x_search_include_option(allow_model_requests: None):
    """Test that xai_include_x_search_output maps correctly."""
    response = create_response(content='test', usage=create_usage(prompt_tokens=10, completion_tokens=5))
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    settings: XaiModelSettings = {
        'xai_include_x_search_output': True,
    }
    await agent.run('Hello', model_settings=settings)

    kwargs = get_mock_chat_create_kwargs(mock_client)
    assert kwargs[0]['include'] == [chat_pb2.IncludeOption.INCLUDE_OPTION_X_SEARCH_CALL_OUTPUT]


async def test_xai_x_search_usage_mapping(allow_model_requests: None):
    """Test that SERVER_SIDE_TOOL_X_SEARCH maps to x_search in usage."""
    mock_usage = create_usage(
        prompt_tokens=50,
        completion_tokens=30,
        server_side_tools_used=[usage_pb2.SERVER_SIDE_TOOL_X_SEARCH],
    )
    response = create_response(content='Found it', usage=mock_usage)
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    result = await agent.run('Search X')
    assert result.usage() == snapshot(
        RunUsage(
            input_tokens=50,
            output_tokens=30,
            details={'server_side_tools_x_search': 1},
            requests=1,
        )
    )


# =============================================================================
# FileSearchTool → collections_search tests
# =============================================================================


async def test_xai_builtin_file_search_tool(allow_model_requests: None):
    """Test xAI's built-in file_search tool (mapped to collections_search)."""
    response = create_collections_search_response(
        query='quarterly report',
        content={'results': [{'chunk': 'Q3 revenue increased by 15%', 'score': 0.92}]},
        assistant_text='According to your documents, Q3 revenue increased by 15%.',
    )
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(
        m,
        builtin_tools=[FileSearchTool(file_store_ids=['collection-abc', 'collection-xyz'])],
    )

    result = await agent.run('What does the quarterly report say?')
    assert result.output == 'According to your documents, Q3 revenue increased by 15%.'

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What does the quarterly report say?',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='file_search',
                        args={'query': 'quarterly report'},
                        tool_call_id=IsStr(),
                        provider_name='xai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='file_search',
                        content={
                            'results': [{'chunk': 'Q3 revenue increased by 15%', 'score': 0.92}],
                        },
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    TextPart(content='According to your documents, Q3 revenue increased by 15%.'),
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


async def test_xai_file_search_sends_collection_ids(allow_model_requests: None):
    """Test that FileSearchTool passes collection_ids to the xAI SDK."""
    response = create_response(content='result', usage=create_usage(prompt_tokens=10, completion_tokens=5))
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(
        m,
        builtin_tools=[FileSearchTool(file_store_ids=['col-1', 'col-2'])],
    )

    await agent.run('Search my docs')

    kwargs = get_mock_chat_create_kwargs(mock_client)
    assert len(kwargs) == 1
    tools = kwargs[0]['tools']
    assert tools is not None
    assert len(tools) == 1
    tool_dict = tools[0]
    assert 'collections_search' in tool_dict


async def test_xai_file_search_include_option(allow_model_requests: None):
    """Test that xai_include_collections_search_output maps correctly."""
    response = create_response(content='test', usage=create_usage(prompt_tokens=10, completion_tokens=5))
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    settings: XaiModelSettings = {
        'xai_include_collections_search_output': True,
    }
    await agent.run('Hello', model_settings=settings)

    kwargs = get_mock_chat_create_kwargs(mock_client)
    assert kwargs[0]['include'] == [chat_pb2.IncludeOption.INCLUDE_OPTION_COLLECTIONS_SEARCH_CALL_OUTPUT]


async def test_xai_file_search_builtin_tool_call_in_history(allow_model_requests: None):
    """Test that FileSearchTool BuiltinToolCallPart in history is properly mapped back to xAI."""
    response1 = create_collections_search_response(query='quarterly report', assistant_text='Found relevant documents.')
    response2 = create_response(content='The report showed 15% revenue increase.')

    mock_client = MockXai.create_mock([response1, response2])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, builtin_tools=[FileSearchTool(file_store_ids=['col-abc'])])

    result1 = await agent.run('Search my documents for quarterly report')
    result2 = await agent.run('What did it say?', message_history=result1.new_messages())

    assert get_mock_chat_create_kwargs(mock_client) == snapshot(
        [
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [{'content': [{'text': 'Search my documents for quarterly report'}], 'role': 'ROLE_USER'}],
                'tools': [{'collections_search': {'collection_ids': ['col-abc']}}],
                'tool_choice': 'auto',
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            },
            {
                'model': XAI_NON_REASONING_MODEL,
                'messages': [
                    {'content': [{'text': 'Search my documents for quarterly report'}], 'role': 'ROLE_USER'},
                    {
                        'content': [{'text': ''}],
                        'role': 'ROLE_ASSISTANT',
                        'tool_calls': [
                            {
                                'id': 'collections_search_001',
                                'type': 'TOOL_CALL_TYPE_COLLECTIONS_SEARCH_TOOL',
                                'status': 'TOOL_CALL_STATUS_COMPLETED',
                                'function': {
                                    'name': 'file_search',
                                    'arguments': '{"query":"quarterly report"}',
                                },
                            }
                        ],
                    },
                    {
                        'content': [{'text': 'Found relevant documents.'}],
                        'role': 'ROLE_ASSISTANT',
                    },
                    {'content': [{'text': 'What did it say?'}], 'role': 'ROLE_USER'},
                ],
                'tools': [{'collections_search': {'collection_ids': ['col-abc']}}],
                'tool_choice': 'auto',
                'response_format': None,
                'use_encrypted_content': False,
                'include': [],
            },
        ]
    )

    assert result2.output == 'The report showed 15% revenue increase.'


async def test_xai_file_search_usage_mapping(allow_model_requests: None):
    """Test that SERVER_SIDE_TOOL_COLLECTIONS_SEARCH maps to file_search in usage."""
    mock_usage = create_usage(
        prompt_tokens=50,
        completion_tokens=30,
        server_side_tools_used=[usage_pb2.SERVER_SIDE_TOOL_COLLECTIONS_SEARCH],
    )
    response = create_response(content='Found it', usage=mock_usage)
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    result = await agent.run('Search collections')
    assert result.usage() == snapshot(
        RunUsage(
            input_tokens=50,
            output_tokens=30,
            details={'server_side_tools_file_search': 1},
            requests=1,
        )
    )
