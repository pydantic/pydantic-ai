"""Tests for xAI search tool integrations (FileSearchTool, XSearchTool profiles)."""

from __future__ import annotations as _annotations

from datetime import timezone

import pytest

from pydantic_ai import (
    Agent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    FileSearchTool,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.profiles.grok import GrokModelProfile, grok_model_profile
from pydantic_ai.usage import RunUsage

from ..._inline_snapshot import snapshot
from ...conftest import IsDatetime, IsNow, IsStr, try_import
from ..mock_xai import (
    MockXai,
    create_collections_search_response,
    create_response,
    create_usage,
    get_mock_chat_create_kwargs,
)

with try_import() as imports_successful:
    from xai_sdk.proto import chat_pb2, usage_pb2

    from pydantic_ai.models.xai import XaiModel, XaiModelSettings
    from pydantic_ai.providers.xai import XaiProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='xai_sdk not installed'),
    pytest.mark.anyio,
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
    # The tool should be a collections_search proto with the collection_ids
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
