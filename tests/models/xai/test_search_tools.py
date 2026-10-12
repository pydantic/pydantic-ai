"""Tests for xAI search tool integrations (XSearchTool, FileSearchTool, grok profiles)."""

from __future__ import annotations as _annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from pydantic_ai import (
    Agent,
    FileSearchTool,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    TextPart,
    ThinkingPart,
    UserPromptPart,
    XSearchTool,
)
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.messages import PartStartEvent, RequestUsage
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles.grok import grok_model_profile
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
    from xai_sdk import chat as chat_types
    from xai_sdk.proto import chat_pb2, sample_pb2, usage_pb2

    from pydantic_ai.models.xai import XaiModel, XaiModelSettings
    from pydantic_ai.providers.xai import XaiProvider
    from tests.models.xai_proto_cassettes import XaiProtoCassetteClient


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='xai_sdk not installed'),
    pytest.mark.vcr,
]

XAI_NON_REASONING_MODEL = 'grok-4-fast-non-reasoning'
XAI_REASONING_MODEL = 'grok-4-fast-reasoning'


# =============================================================================
# Grok model profile tests
# =============================================================================


@pytest.mark.parametrize(
    'model_name,expected_thinking,expected_always_enabled',
    [
        ('grok-4.3', True, False),
        ('grok-4.3-latest', True, False),
        # `grok-latest` is the floating alias for the newest Grok (currently 4.3), so it mirrors its efforts.
        ('grok-latest', True, False),
        ('grok-4-fast-reasoning', True, False),
        ('grok-4-fast-non-reasoning', True, False),
        ('grok-4-1-fast-non-reasoning', True, False),
        # `grok-4.20`'s effort knob controls agent count, not thinking depth, so unified thinking is unsupported.
        ('grok-4.20', False, False),
        ('grok-4.20-multi-agent', False, False),
        ('grok-4.20-reasoning', False, False),
        # `grok-code-fast-1` redirects to `grok-build-0.1`, not Grok 4.3, so they get no reasoning effort.
        ('grok-code-fast-1', False, False),
        ('grok-build-0.1', False, False),
        ('grok-3', True, False),
        ('grok-3-mini', True, True),
        ('grok-3-mini-fast', True, True),
        ('grok-3-fast', False, False),
        ('grok-4-1-reasoning', False, False),
    ],
    ids=[
        'grok-4.3',
        'grok-4.3-latest',
        'grok-latest',
        'grok-4-fast-reasoning',
        'grok-4-fast-non-reasoning',
        'grok-4-1-fast-non-reasoning',
        'grok-4.20',
        'grok-4.20-multi-agent',
        'grok-4.20-reasoning',
        'grok-code-fast-1',
        'grok-build-0.1',
        'grok-3',
        'grok-3-mini',
        'grok-3-mini-fast',
        'grok-3-fast',
        'grok-4-1-reasoning',
    ],
)
def test_grok_model_profile_thinking(model_name: str, expected_thinking: bool, expected_always_enabled: bool) -> None:
    profile = grok_model_profile(model_name)
    assert profile is not None
    assert profile.get('supports_thinking', False) == expected_thinking
    # Only models whose `reasoning_effort` set lacks `'none'` (the grok-3-mini family) are always-on;
    # Grok 4.3 and its redirect slugs accept `'none'`, so `thinking=False` disables reasoning there.
    assert profile.get('thinking_always_enabled', False) == expected_always_enabled


async def test_grok_4_reasoning_model_forwards_reasoning_effort(allow_model_requests: None) -> None:
    """Retired grok-4 reasoning slugs redirect to grok-4.3 and accept `reasoning_effort`."""
    response = create_response(content='ok')
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    settings: XaiModelSettings = {'thinking': 'high'}
    agent = Agent(m, model_settings=settings)

    await agent.run('hi')

    kwargs = get_mock_chat_create_kwargs(mock_client)
    assert len(kwargs) == 1
    assert kwargs[0]['reasoning_effort'] == 'high'


async def test_xai_thinking_false_with_non_always_on_profile_is_dropped(allow_model_requests: None) -> None:
    """Defensive guard: no `reasoning_effort` is emitted when `thinking=False` survives the gate
    (only possible under a profile with `supports_thinking=True` and `thinking_always_enabled=False`).
    The profile here exposes no `reasoning_effort` values, so `_map_reasoning_effort` returns `None`
    and the parameter is omitted rather than forwarded."""
    response = create_response(content='ok')
    mock_client = MockXai.create_mock([response])
    custom_profile = ModelProfile(supports_thinking=True, thinking_always_enabled=False)
    m = XaiModel('grok-3-mini', provider=XaiProvider(xai_client=mock_client), profile=custom_profile)
    settings: XaiModelSettings = {'thinking': False}
    agent = Agent(m, model_settings=settings)

    await agent.run('hi')

    kwargs = get_mock_chat_create_kwargs(mock_client)
    assert len(kwargs) == 1
    assert 'reasoning_effort' not in kwargs[0]


def test_grok_model_profile_builtin_tools() -> None:
    grok4_profile = grok_model_profile('grok-4-fast-non-reasoning')
    assert grok4_profile is not None
    assert isinstance(grok4_profile, dict)
    assert grok4_profile.get('grok_supports_builtin_tools', False) is True

    # `grok-3` redirects to Grok 4.3, so it's builtin-capable despite not matching the `grok-4`/`code` patterns.
    grok3_profile = grok_model_profile('grok-3')
    assert grok3_profile is not None
    assert isinstance(grok3_profile, dict)
    assert grok3_profile.get('grok_supports_builtin_tools', False) is True

    # `grok-build-0.1` is a coding model (the `grok-code-fast-1` redirect target) and supports builtin tools.
    grok_build_profile = grok_model_profile('grok-build-0.1')
    assert grok_build_profile is not None
    assert isinstance(grok_build_profile, dict)
    assert grok_build_profile.get('grok_supports_builtin_tools', False) is True

    grok3_mini_profile = grok_model_profile('grok-3-mini')
    assert grok3_mini_profile is not None
    assert isinstance(grok3_mini_profile, dict)
    assert grok3_mini_profile.get('grok_supports_builtin_tools', False) is False


# =============================================================================
# XSearchTool validation tests
# =============================================================================


def test_x_search_tool_validation():
    """Test XSearchTool validation rules."""
    with pytest.raises(ValueError, match='Cannot specify both allowed_x_handles and excluded_x_handles'):
        XSearchTool(allowed_x_handles=['foo'], excluded_x_handles=['bar'])

    handles = [f'h{i}' for i in range(1, 21)]
    assert XSearchTool(allowed_x_handles=handles).allowed_x_handles == handles
    assert XSearchTool(excluded_x_handles=handles).excluded_x_handles == handles

    handles = [f'h{i}' for i in range(1, 22)]
    with pytest.raises(ValueError, match='allowed_x_handles cannot contain more than 20 handles'):
        XSearchTool(allowed_x_handles=handles)

    with pytest.raises(ValueError, match='excluded_x_handles cannot contain more than 20 handles'):
        XSearchTool(excluded_x_handles=handles)

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
        capabilities=[NativeTool(XSearchTool())],
        model_settings=XaiModelSettings(
            xai_include_encrypted_content=True,
            xai_include_x_search_output=True,
        ),
    )

    result = await agent.run('What are the latest posts about PydanticAI on X? Reply with just the key topic.')
    assert result.output == snapshot('PydanticAI v1.80 updates for AI agent development')

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What are the latest posts about PydanticAI on X? Reply with just the key topic.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=UTC),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='xai',
                    ),
                    NativeToolCallPart(
                        tool_name='x_search',
                        args={'query': 'PydanticAI', 'limit': 10, 'mode': 'Latest'},
                        tool_call_id=IsStr(),
                        provider_name='xai',
                        provider_details={'function_name': 'x_keyword_search'},
                    ),
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='xai',
                    ),
                    NativeToolReturnPart(
                        tool_name='x_search',
                        content={
                            'citations': [
                                'https://x.com/i/status/2042562199843987834',
                                'https://x.com/i/status/2042535641490096426',
                                'https://x.com/i/status/2042981439357227193',
                                'https://x.com/i/status/2042935940440822230',
                                'https://x.com/i/status/2043733929694232605',
                                'https://x.com/i/status/2043307387835342915',
                                'https://x.com/i/status/2042600007912820765',
                                'https://x.com/i/status/2043737344478527731',
                                'https://x.com/i/status/2043307391111024980',
                                'https://x.com/i/status/2043548524416217320',
                                'https://x.com/i/status/2042444002595889482',
                                'https://x.com/i/status/2042149152801620346',
                                'https://x.com/i/status/2042935942454087800',
                            ]
                        },
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='xai',
                    ),
                    TextPart(content='PydanticAI v1.80 updates for AI agent development'),
                ],
                usage=RequestUsage(
                    input_tokens=5821,
                    cache_read_tokens=2692,
                    output_tokens=586,
                    output_reasoning_tokens=524,
                    details={'reasoning_tokens': 524, 'server_side_tools_x_search': 1},
                    cost=Decimal('0.0010534'),
                ),
                model_name='grok-4-fast-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_xai_builtin_x_search_tool_stream(allow_model_requests: None, xai_provider: XaiProvider):
    """Test xAI's built-in x_search tool with streaming (recorded via proto cassette)."""
    m = XaiModel(XAI_REASONING_MODEL, provider=xai_provider)
    agent = Agent(
        m,
        capabilities=[NativeTool(XSearchTool())],
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
                timestamp=IsNow(tz=UTC),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='xai',
                    ),
                    NativeToolCallPart(
                        tool_name='x_search',
                        args={'query': 'PydanticAI', 'limit': 10, 'mode': 'Latest'},
                        tool_call_id=IsStr(),
                        provider_name='xai',
                        provider_details={'function_name': 'x_keyword_search'},
                    ),
                    ThinkingPart(
                        content='',
                        signature='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',
                        provider_name='xai',
                    ),
                    NativeToolReturnPart(
                        tool_name='x_search',
                        content={
                            'citations': [
                                'https://x.com/i/status/2042935942454087800',
                                'https://x.com/i/status/2042444002595889482',
                                'https://x.com/i/status/2042981439357227193',
                                'https://x.com/i/status/2042149152801620346',
                                'https://x.com/i/status/2043307391111024980',
                                'https://x.com/i/status/2042562199843987834',
                                'https://x.com/i/status/2043307387835342915',
                                'https://x.com/i/status/2043548524416217320',
                                'https://x.com/i/status/2043733929694232605',
                                'https://x.com/i/status/2042935940440822230',
                                'https://x.com/i/status/2043737344478527731',
                                'https://x.com/i/status/2042535641490096426',
                                'https://x.com/i/status/2042600007912820765',
                            ]
                        },
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    ThinkingPart(
                        content='',
                        signature='LEJY1c5PJ5/S6gqkTSigRrynG3l+sS18vS54TohBpjBkB8hVRKB81oZ/LnRbjY2FsHnxdbzBMvwmFVOLlcIlBvAhoXHoX+YcP2+/LTGkG72iVCs2/1y2n0jm9aFf6YKrO2E6DgXXKWSYlElekhxjXSa5zC1LVP+JbRJ+YAt9rqJxOTblrqpFjtwpbPs2FbMMzIHxbGm+iR2wM4vHUG2aSMkwwSYsnsbG7Z9vdEIgb7dU6bxuZddd4AJ730NQb/6Oxi0HqNmbjz+O6NpcDBGNowhxdDNO8uNSfsR5myNJJbVhkyDhV8/qyquyF+jsvPRoYaSUuRNfQXZ7YqHJ6nyvM6r6MGia+LIh141U1cqJpe3qbmWLDw4eVaRupjXeM7gYGZMU+Rbq6HsptsPl4oMz/Ti/bAUHW65PyPkixYCc+bfrysdP8opoEUSFEAKYui9qWMMHw19MeJBTxATO2n8Ywu4SrPpaVy/tn719LqkmkKd+m18efkVauFNAw+JGqHxhGV7SQSUoySldWWDFN29cmsq1LrQdtBEyXJazO9r/iRB2wXSr3ASGpOwlQWZrO0oRrZ6GUufPNd1IxONbtK1VTLWnlfJh9acMIjGt3+zOfnimbEOQZOZVDGiVDuh0HWj8FvRlKsVIiyMxJougu7rpqZZA6dSp5bpU+hjQAf92pncnIXRzGXnpSEwYTog5ZEQKlV/GTEk5Mildro7O6f/7NyTXHXJHNB6QQzAi1XOp0TQ/UDpeMuU5j8qK4mtJXvwg6dYgmlfnBfCGusbfj7hQENxXPiuMAZVCiVXZ09woFmEjlX/Kmzhjbb04Jp6qlTl7cXd4t22E9siMoMEb2rxFI7TS4Rx7JSboGJ+8k6pY7IMQFp6kN19EcBGkTFrCjcf2Yg88Y8NsxhGNkfmJs312DMSnJpV/OgsKLeDOiJ0RLd3vAmcq7EhTjHYfvBF91DF452VmHGjmLyMPhkqrUQ1p6c6Ad9HkRrawtpgHrDIgwAkmEzR7HR0x1J38Etr52C3HHQgymz2G8I0/iEe8ViemaOQFHqmo92ee8ghMuCznDMQ1zYPvzIOMENuG3xOom2dMTkjvlSOSEaoSQ0pvW7A8u80yj5Y14EkO6RyvxYaykoXSqXpI5tgIXlaGblfoQfZMK7CqoMQbhhamE6Qou0vCnQghZ+TCi8QNycdACpgi9vlYe8VWgJrl3NOwZnzA+hNKsrF++L0YJFdfXMAajwj/Whaaw0U0lMxr6fHE41Gp3VD+RWVWXNOMdMOVPmfirWG06Bk9f/i77ik0flge3QvpCgZHrkPqtlwCYIzfaDLwg0DNfyeViHRlmuzz/FXlbuf8dlEMDf/Dn3liPwLpVcMvb5CgRDivJO5VvB6o4d8WWf2jGzbzxIhk59zRq5rlld0cLmrTkoFr2lQiTgeEN09TDK8hJoNGz0w2V5yQoUCZeiwjWErRaYSKmgAGDvWukPuvesuF/WpeSovRfQLp3ET+nkEdaI/O1cOV0qMffZ+4lg/5ta6t1OxcDmZ8Ki28YN2VOXeT4Tqarl1aIPbvbN0GCHlayiQUTE4+sdNzPdCMM4d1HnqosL9qxGnbHTAbAgZMwSiSMyUAhlCePhGz5CyKGIMfnfVhUUviaKs8W3d4k1wKMzTelNqrdNpLp1y58OIGzZSBtgZ1fej1Hqh+2eMuCGurH14MUY4QMqsgsjOUQ0GctlrGSuca6AmtyNKtXYOMYs5FaEmmULmWUGMVxH649Rx5E5s49U+NPv3aY76sFkKb+BAtoNjjr9pziFfpBFlegFec4wUV7G0N8SZ159i4DWFahK1zvEg089HccrMAGtdvBRyCmFcPfyUO+saXFGkQR0PT7imJSp+syVIJG5vrrpOt91jAXvcE7EV+4dBqeKZTFICWFYm1igiXlrS1',
                        provider_name='xai',
                    ),
                    TextPart(content='PydanticAI v1.80: Tool call retry fixes and capability ordering primitives'),
                ],
                usage=RequestUsage(
                    input_tokens=5828,
                    cache_read_tokens=2701,
                    output_tokens=664,
                    output_reasoning_tokens=598,
                    details={'reasoning_tokens': 598, 'server_side_tools_x_search': 1},
                    cost=Decimal('0.00109245'),
                ),
                model_name='grok-4-fast-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_xai_x_search_streaming_citations_no_duplicate_part_start_event(allow_model_requests: None):
    """Regression: streaming x_search citation backfill must not emit a duplicate `PartStartEvent`.

    xAI returns x_search results as top-level `response.citations` that only arrive with the
    final stream chunk, so we backfill them onto the already-emitted `NativeToolReturnPart`.
    The fix mutates the part in place rather than re-calling `_parts_manager.handle_part`,
    which would have emitted a second `PartStartEvent` at the same index. This test exercises
    that path with mocked stream chunks (citation arrives only on the final chunk) and asserts:
    1. the final return part's `content` ends up populated with the citations, and
    2. exactly one `PartStartEvent` is emitted for the x_search return part vendor id.
    """
    tool_call_id = 'x_search_stream_001'
    citations = ['https://x.com/i/status/1', 'https://x.com/i/status/2']

    def _build_x_search_tool_call(status: chat_pb2.ToolCallStatus) -> chat_pb2.ToolCall:
        return chat_pb2.ToolCall(
            id=tool_call_id,
            type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_X_SEARCH_TOOL,
            status=status,
            function=chat_pb2.FunctionCall(name='x_keyword_search', arguments='{"query":"PydanticAI"}'),
        )

    def _build_chunk(
        *,
        role: chat_pb2.MessageRole,
        tool_calls: list[chat_pb2.ToolCall] | None = None,
        content: str = '',
        finish_reason: str | None = None,
    ) -> chat_types.Chunk:
        proto = chat_pb2.GetChatCompletionChunk(id='grok-stream')
        proto.created.GetCurrentTime()
        output_chunk = chat_pb2.CompletionOutputChunk(
            index=0,
            delta=chat_pb2.Delta(role=role, tool_calls=tool_calls or [], content=content),
        )
        if finish_reason == 'stop':
            output_chunk.finish_reason = sample_pb2.FinishReason.REASON_STOP
        elif finish_reason == 'tool_calls':
            output_chunk.finish_reason = sample_pb2.FinishReason.REASON_TOOL_CALLS
        proto.outputs.append(output_chunk)
        return chat_types.Chunk(proto, index=None)

    def _build_response(
        *,
        tool_calls: list[chat_pb2.ToolCall] | None = None,
        content: str = '',
        finish_reason: str = 'stop',
        with_citations: bool = False,
    ) -> chat_types.Response:
        proto = chat_pb2.GetChatCompletionResponse(id='grok-stream')
        proto.created.GetCurrentTime()
        proto.outputs.append(
            chat_pb2.CompletionOutput(
                index=0,
                finish_reason=sample_pb2.FinishReason.REASON_STOP
                if finish_reason == 'stop'
                else sample_pb2.FinishReason.REASON_TOOL_CALLS,
                message=chat_pb2.CompletionMessage(
                    role=chat_pb2.MessageRole.ROLE_ASSISTANT, content=content, tool_calls=tool_calls or []
                ),
            )
        )
        if with_citations:
            proto.citations.extend(citations)
        return chat_types.Response(proto, index=None)

    completed_call = _build_x_search_tool_call(chat_pb2.ToolCallStatus.TOOL_CALL_STATUS_COMPLETED)

    stream = [
        # Assistant emits the x_search call.
        (
            _build_response(tool_calls=[completed_call], finish_reason='tool_calls'),
            _build_chunk(
                role=chat_pb2.MessageRole.ROLE_ASSISTANT,
                tool_calls=[completed_call],
                finish_reason='tool_calls',
            ),
        ),
        # ROLE_TOOL message marks the tool result. Note: no `content` and no `citations` yet.
        (
            _build_response(tool_calls=[completed_call], finish_reason='tool_calls'),
            _build_chunk(role=chat_pb2.MessageRole.ROLE_TOOL, tool_calls=[completed_call]),
        ),
        # Final chunk: assistant reply + citations populated on the accumulated response.
        (
            _build_response(content='done', with_citations=True),
            _build_chunk(role=chat_pb2.MessageRole.ROLE_ASSISTANT, content='done', finish_reason='stop'),
        ),
    ]

    mock_client = MockXai.create_mock_stream([stream])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, capabilities=[NativeTool(XSearchTool())])

    events: list[Any] = []
    async with agent.iter(user_prompt='find PydanticAI posts') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        events.append(event)

    assert agent_run.result is not None
    parts = agent_run.result.all_messages()[1].parts
    return_parts = [p for p in parts if isinstance(p, NativeToolReturnPart) and p.tool_name == XSearchTool.kind]
    assert len(return_parts) == 1
    assert return_parts[0].content == {'citations': citations}

    # Locate the return part by index in the final parts list, then verify exactly one
    # `PartStartEvent` was emitted for that index.
    return_part_index = parts.index(return_parts[0])
    start_events_at_return_index = [
        e
        for e in events
        if isinstance(e, PartStartEvent) and e.index == return_part_index and isinstance(e.part, NativeToolReturnPart)
    ]
    assert len(start_events_at_return_index) == 1


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
        capabilities=[NativeTool(XSearchTool(allowed_x_handles=['OpenAI', 'AnthropicAI']))],
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
        capabilities=[
            NativeTool(
                XSearchTool(
                    from_date=datetime(2024, 1, 1),
                    to_date=datetime(2024, 12, 31),
                )
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
                parts=[UserPromptPart(content='Search for something', timestamp=IsNow(tz=UTC))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    NativeToolCallPart(
                        tool_name='x_search',
                        args={'query': 'test'},
                        tool_call_id=IsStr(),
                        provider_name='xai',
                        provider_details={'function_name': 'x_search'},
                    ),
                    TextPart(content='Search results here'),
                ],
                usage=RequestUsage(cost=Decimal('0.00')),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_url='https://api.x.ai/v1',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_xai_x_search_builtin_tool_call_in_history(allow_model_requests: None):
    """Test that XSearchTool NativeToolCallPart in history is properly mapped back to xAI."""
    response1 = create_x_search_response(query='pydantic updates', assistant_text='Found posts about PydanticAI.')
    response2 = create_response(content='The posts were about PydanticAI releases.')

    mock_client = MockXai.create_mock([response1, response2])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, capabilities=[NativeTool(XSearchTool())])

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
                                'function': {'name': 'x_keyword_search', 'arguments': '{"query":"pydantic updates"}'},
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


async def test_xai_x_search_function_name_round_trip(allow_model_requests: None):
    """Test that the xAI-specific function name (e.g. 'x_keyword_search') survives the round-trip.

    The xAI API uses function names like 'x_keyword_search' or 'collections_search' that differ
    from PydanticAI's normalized tool_name ('x_search', 'file_search'). The original function name
    must be preserved in provider_details and sent back when replaying history.
    """
    response1 = create_x_search_response(query='test query', assistant_text='Found results.')
    response2 = create_response(content='Follow-up answer.')

    mock_client = MockXai.create_mock([response1, response2])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, capabilities=[NativeTool(XSearchTool())])

    result1 = await agent.run('Search for something')

    # Verify provider_details stores the original function name
    call_parts = [p for p in result1.all_messages()[1].parts if isinstance(p, NativeToolCallPart)]
    assert len(call_parts) == 1
    assert call_parts[0].tool_name == 'x_search'
    assert call_parts[0].provider_details == snapshot({'function_name': 'x_keyword_search'})

    # Verify round-trip: the original function name is sent back in history
    result2 = await agent.run('Follow up', message_history=result1.new_messages())
    kwargs = get_mock_chat_create_kwargs(mock_client)
    history_tool_calls = kwargs[1]['messages'][1]['tool_calls']
    assert history_tool_calls[0]['function']['name'] == 'x_keyword_search'

    assert result2.output == 'Follow-up answer.'


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
    assert result.usage == snapshot(
        RunUsage(
            input_tokens=50,
            output_tokens=30,
            details={'server_side_tools_x_search': 1},
            requests=1,
            cost=Decimal('0.000025'),
        )
    )


# =============================================================================
# FileSearchTool → collections_search tests
# =============================================================================


async def test_xai_builtin_file_search_tool(
    allow_model_requests: None,
    xai_provider: XaiProvider,
    monkeypatch: pytest.MonkeyPatch,
):
    """End-to-end `FileSearchTool` -> xAI `collections_search` round-trip (recorded via proto cassette).

    Creates a real collection, uploads a test document, runs an agent query, and cleans up.
    All four interactions (create, upload_document, chat.sample, delete) are captured for offline replay.

    Re-recording requires `XAI_MANAGEMENT_KEY` in addition to `XAI_API_KEY` — the SDK reads it from env
    when creating the management gRPC channel used by `client.collections.*`.
    """
    import asyncio
    from datetime import timedelta
    from uuid import uuid4

    from xai_sdk.aio.collections import Client as _AioCollectionsClient
    from xai_sdk.poll_timer import PollTimer
    from xai_sdk.proto import collections_pb2

    # xai-sdk (through 1.11.0) raises on unknown DocumentStatus values. The xAI backend has added a
    # status beyond the ones the SDK recognizes, so patch polling to treat unknown statuses as
    # "still processing" during recording. Replay path never calls into the real SDK, so this patch
    # is a no-op offline.
    async def _tolerant_wait_for_indexing(  # pragma: no cover
        self: _AioCollectionsClient,
        collection_id: str,
        file_id: str,
        poll_interval: timedelta,
        timeout: timedelta,
    ) -> collections_pb2.DocumentMetadata:
        timer = PollTimer(timeout, poll_interval)
        while True:
            doc = await self.get_document(file_id, collection_id)
            if doc.status == collections_pb2.DocumentStatus.DOCUMENT_STATUS_PROCESSED:
                return doc
            if doc.status == collections_pb2.DocumentStatus.DOCUMENT_STATUS_FAILED:
                raise ValueError(f'Document indexing failed: {doc.error_message}')
            await asyncio.sleep(timer.sleep_interval_or_raise())

    monkeypatch.setattr(_AioCollectionsClient, '_wait_for_indexing', _tolerant_wait_for_indexing)

    paragraph = (
        'Zorblax Research Memo 7742. '
        'The Zorblax Protocol is a fictional encryption scheme invented by the Zorblax Research Collective '
        'in the year 2187. Its defining property is the use of heptapod-prime key rotation, which cycles '
        'every 7919 milliseconds across the primary substrate. The Zorblax Protocol was adopted as the '
        'galactic standard by the Outer Rim Treaty of 2193. Researchers cite three principal inventors: '
        'Dr. Mira Calyx, Dr. Taren Ko, and Dr. Silas Rhen. '
    )
    doc_text = ('\n\n'.join([f'Section {i}. {paragraph}' for i in range(1, 11)])).encode('utf-8')

    client = xai_provider.client
    collection = await client.collections.create(
        name=f'pydantic-ai-test-{uuid4().hex[:8]}',
        chunk_configuration={
            'chars_configuration': {'max_chunk_size_chars': 256, 'chunk_overlap_chars': 32},
        },
    )
    try:
        await client.collections.upload_document(
            collection_id=collection.collection_id,
            name='zorblax-memo-7742.txt',
            data=doc_text,
            wait_for_indexing=True,
            timeout=timedelta(seconds=180),
        )
        if not isinstance(client, XaiProtoCassetteClient):  # pragma: no cover
            # PROCESSED status doesn't guarantee the live search index is fully propagated; give it a moment.
            await asyncio.sleep(5)

        m = XaiModel(XAI_NON_REASONING_MODEL, provider=xai_provider)
        agent = Agent(
            m,
            capabilities=[
                NativeTool(
                    FileSearchTool(
                        file_store_ids=[collection.collection_id],
                        max_num_results=1,
                        instructions='Prioritize exact factual matches from the uploaded research memo.',
                        retrieval_mode='semantic',
                    )
                )
            ],
            model_settings=XaiModelSettings(xai_include_collections_search_output=True),
        )

        result = await agent.run(
            'Using the uploaded Zorblax Research Memo, in what year was the Zorblax Protocol invented '
            'and who are its three principal inventors?'
        )
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Using the uploaded Zorblax Research Memo, in what year was the Zorblax Protocol invented and who are its three principal inventors?',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        NativeToolCallPart(
                            tool_name='file_search',
                            args={
                                'search_request': '{"query": "Zorblax Protocol invented year principal inventors", "limit": 10, "retrieval_mode": "semantic"}'
                            },
                            tool_call_id=IsStr(),
                            provider_name='xai',
                            provider_details={'function_name': 'collections_search'},
                        ),
                        NativeToolReturnPart(
                            tool_name='file_search',
                            content={
                                'search_matches': [
                                    {
                                        'file_id': 'file_e9ef3a06-160e-4a51-a3e2-762cd070cc32',
                                        'chunk_id': 'file_e9ef3a06-160e-4a51-a3e2-762cd070cc32_5',
                                        'chunk_content': 'ary substrate. The Zorblax Protocol was adopted as the galactic standard by the Outer Rim Treaty of 2193. Researchers cite three principal inventors: Dr. Mira Calyx, Dr. Taren Ko, and Dr. Silas Rhen. \\n\\nSection 10. Zorblax Research Memo 7742. The Zorblax Protocol is a fictional encryption scheme invented by the Zorblax Research Collective in the year 2187. Its defining property is the use of heptapod-prime key rotation, which cycles every 7919 milliseconds across the primary substrate. The Zorblax Protocol was adopted as the galactic standard by the Outer Rim Treaty of 2193. Researchers cite three principal inventors: Dr. Mira Calyx, Dr. Taren Ko, and Dr. Silas Rhen. "}]',
                                        'score': 0.7739996314048767,
                                        'collection_ids': ['collection_744aab7b-44f2-41ab-a982-9c49d1690c2f'],
                                    }
                                ]
                            },
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                            provider_name='xai',
                        ),
                        TextPart(
                            content="""\
**2187**, by **Dr. Mira Calyx, Dr. Taren Ko, and Dr. Silas Rhen**. \n\

This is stated directly in the uploaded Zorblax Research Memo (Section 10), which describes the Zorblax Protocol as a fictional encryption scheme invented by the Zorblax Research Collective in 2187, with those three researchers cited as the principal inventors.\
"""
                        ),
                    ],
                    usage=RequestUsage(
                        input_tokens=2417,
                        cache_read_tokens=1152,
                        output_tokens=120,
                        details={'server_side_tools_file_search': 1},
                        cost=Decimal('0.0003706'),
                    ),
                    model_name='grok-4-fast-non-reasoning',
                    timestamp=IsDatetime(),
                    provider_name='xai',
                    provider_url='https://api.x.ai/v1',
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )
    finally:
        await client.collections.delete(collection.collection_id)


async def test_xai_file_search_sends_collection_ids(allow_model_requests: None):
    """Test that FileSearchTool passes collection_ids to the xAI SDK."""
    response = create_response(content='result', usage=create_usage(prompt_tokens=10, completion_tokens=5))
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(
        m,
        capabilities=[NativeTool(FileSearchTool(file_store_ids=['col-1', 'col-2']))],
    )

    await agent.run('Search my docs')

    kwargs = get_mock_chat_create_kwargs(mock_client)
    assert len(kwargs) == 1
    tools = kwargs[0]['tools']
    assert tools is not None
    assert len(tools) == 1
    tool_dict = tools[0]
    assert 'collections_search' in tool_dict


async def test_xai_file_search_options_forwarded(allow_model_requests: None):
    """FileSearchTool option fields are forwarded to xAI's collections search payload."""
    response = create_response(content='result', usage=create_usage(prompt_tokens=10, completion_tokens=5))
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(
        m,
        capabilities=[
            NativeTool(
                FileSearchTool(
                    file_store_ids=['col-1', 'col-2'],
                    max_num_results=5,
                    instructions='Focus on recent documents.',
                    retrieval_mode='hybrid',
                )
            )
        ],
    )

    await agent.run('Search my docs')

    kwargs = get_mock_chat_create_kwargs(mock_client)
    assert kwargs[0]['tools'] == snapshot(
        [
            {
                'collections_search': {
                    'collection_ids': ['col-1', 'col-2'],
                    'limit': 5,
                    'instructions': 'Focus on recent documents.',
                    'hybrid_retrieval': {},
                }
            }
        ]
    )


async def test_xai_file_search_options_omitted_when_none(allow_model_requests: None):
    """Unset FileSearchTool options are omitted from the outgoing collections search payload."""
    response = create_response(content='result', usage=create_usage(prompt_tokens=10, completion_tokens=5))
    mock_client = MockXai.create_mock([response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(
        m,
        capabilities=[NativeTool(FileSearchTool(file_store_ids=['col-1']))],
    )

    await agent.run('Search my docs')

    kwargs = get_mock_chat_create_kwargs(mock_client)
    assert kwargs[0]['tools'] == snapshot(
        [
            {
                'collections_search': {
                    'collection_ids': ['col-1'],
                }
            }
        ]
    )


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
    """Test that FileSearchTool NativeToolCallPart in history is properly mapped back to xAI."""
    response1 = create_collections_search_response(query='quarterly report', assistant_text='Found relevant documents.')
    response2 = create_response(content='The report showed 15% revenue increase.')

    mock_client = MockXai.create_mock([response1, response2])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, capabilities=[NativeTool(FileSearchTool(file_store_ids=['col-abc']))])

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
                                    'name': 'collections_search',
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
    assert result.usage == snapshot(
        RunUsage(
            input_tokens=50,
            output_tokens=30,
            details={'server_side_tools_file_search': 1},
            requests=1,
            cost=Decimal('0.000025'),
        )
    )
