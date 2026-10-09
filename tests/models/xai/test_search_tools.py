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
                        provider_details={
                            'encrypted_content': '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'
                        },
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
                        provider_details={
                            'encrypted_content': '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'
                        },
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
