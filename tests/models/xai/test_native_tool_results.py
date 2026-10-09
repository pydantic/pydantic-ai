"""Tests for how xAI server-side tool results are stored and replayed.

xAI returns each server-side tool result as a `ROLE_TOOL` output carrying the result as `encrypted_content`.
It is stored on the `NativeToolReturnPart` and replayed as a `ROLE_TOOL` message paired with its call by
`tool_call_id`, which is how `xai_sdk`'s `Chat.append` replays it.
"""

from __future__ import annotations as _annotations

from typing import Any

import pytest

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    TextPart,
    ThinkingPart,
    UserPromptPart,
    WebSearchTool,
)
from pydantic_ai.capabilities import NativeTool

from ..._inline_snapshot import snapshot
from ...conftest import try_import
from ..mock_xai import MockXai, create_response, get_mock_chat_create_kwargs

with try_import() as imports_successful:
    from xai_sdk import chat as chat_types
    from xai_sdk.proto import chat_pb2, sample_pb2

    from pydantic_ai.models.xai import XaiModel
    from pydantic_ai.providers.xai import XaiProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='xai_sdk not installed')

XAI_REASONING_MODEL = 'grok-4-fast-reasoning'


def _web_search_call(tool_call_id: str, status: chat_pb2.ToolCallStatus) -> chat_pb2.ToolCall:
    return chat_pb2.ToolCall(
        id=tool_call_id,
        type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_WEB_SEARCH_TOOL,
        status=status,
        function=chat_pb2.FunctionCall(name='web_search', arguments='{"query":"pydantic"}'),
    )


def _chunk(index: int, delta: chat_pb2.Delta, finish_reason: sample_pb2.FinishReason | None = None) -> chat_types.Chunk:
    proto = chat_pb2.GetChatCompletionChunk(id='grok-stream')
    proto.created.GetCurrentTime()
    output = chat_pb2.CompletionOutputChunk(index=index, delta=delta)
    if finish_reason is not None:
        output.finish_reason = finish_reason
    proto.outputs.append(output)
    return chat_types.Chunk(proto, index=None)


def _accumulated() -> chat_types.Response:
    proto = chat_pb2.GetChatCompletionResponse(id='grok-stream')
    proto.created.GetCurrentTime()
    return chat_types.Response(proto, index=None)


async def test_xai_streamed_tool_result_split_across_chunks(allow_model_requests: None):
    """A tool result that arrives in pieces ends up whole on the return part, and never as thinking."""
    in_progress = _web_search_call('call_1', chat_pb2.ToolCallStatus.TOOL_CALL_STATUS_IN_PROGRESS)
    completed = _web_search_call('call_1', chat_pb2.ToolCallStatus.TOOL_CALL_STATUS_COMPLETED)
    stream = [
        (
            _accumulated(),
            _chunk(0, chat_pb2.Delta(role=chat_pb2.MessageRole.ROLE_ASSISTANT, tool_calls=[in_progress])),
        ),
        (
            _accumulated(),
            _chunk(
                1,
                chat_pb2.Delta(
                    role=chat_pb2.MessageRole.ROLE_TOOL, tool_calls=[completed], encrypted_content='result-part-1-'
                ),
            ),
        ),
        (_accumulated(), _chunk(1, chat_pb2.Delta(encrypted_content='result-part-2'))),
        (
            _accumulated(),
            _chunk(
                2,
                chat_pb2.Delta(role=chat_pb2.MessageRole.ROLE_ASSISTANT, content='done'),
                finish_reason=sample_pb2.FinishReason.REASON_STOP,
            ),
        ),
    ]
    model = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=MockXai.create_mock_stream([stream])))
    agent = Agent(model, capabilities=[NativeTool(WebSearchTool())])

    async with agent.run_stream('Search for pydantic') as result:
        await result.get_output()

    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert [type(part).__name__ for part in response.parts] == snapshot(
        ['NativeToolCallPart', 'NativeToolReturnPart', 'TextPart']
    )
    return_part = response.parts[1]
    assert isinstance(return_part, NativeToolReturnPart)
    assert return_part.provider_details == {'encrypted_content': 'result-part-1-result-part-2'}


def _search(tool_call_id: str) -> NativeToolCallPart:
    return NativeToolCallPart(
        tool_name='web_search',
        args={'query': 'pydantic'},
        tool_call_id=tool_call_id,
        provider_name='xai',
        provider_details={'function_name': 'web_search'},
    )


def _result(tool_call_id: str, encrypted_content: str | None = None) -> NativeToolReturnPart:
    return NativeToolReturnPart(
        tool_name='web_search',
        content=None,
        tool_call_id=tool_call_id,
        provider_name='xai',
        provider_details={'encrypted_content': encrypted_content} if encrypted_content else None,
    )


def _reasoning(signature: str) -> ThinkingPart:
    return ThinkingPart(content='', signature=signature, provider_name='xai')


# How histories were stored before results moved to the return part: each result as a `ThinkingPart`
# signature right after its call, and a return part with nothing to replay.
OLD_SHAPE_RESPONSE = ModelResponse(
    parts=[
        _reasoning('reasoning-1'),
        _search('call_1'),
        _reasoning('result-1'),
        _result('call_1'),
        _search('call_2'),
        _reasoning('result-2'),
        _result('call_2'),
        TextPart(content='First answer.'),
    ],
    model_name=XAI_REASONING_MODEL,
    provider_name='xai',
)

NEW_SHAPE_RESPONSE = ModelResponse(
    parts=[
        _reasoning('reasoning-3'),
        _search('call_3'),
        _result('call_3', 'result-3'),
        _search('call_4'),
        _result('call_4', 'result-4'),
        TextPart(content='Second answer.'),
    ],
    model_name=XAI_REASONING_MODEL,
    provider_name='xai',
)


async def _sent_messages(history: list[ModelMessage]) -> list[dict[str, Any]]:
    client = MockXai.create_mock([create_response(content='ok')])
    agent = Agent(
        XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=client)),
        capabilities=[NativeTool(WebSearchTool())],
    )
    await agent.run('And now?', message_history=history)
    return get_mock_chat_create_kwargs(client)[0]['messages']


async def test_xai_old_and_new_tool_result_histories_replay(allow_model_requests: None):
    """An old-shape turn replays as before, and a new-shape turn pairs each result with its call.

    A call that follows a replayed result starts its own assistant message rather than joining the tool message.
    """
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='First question')]),
        OLD_SHAPE_RESPONSE,
        ModelRequest(parts=[UserPromptPart(content='Second question')]),
        NEW_SHAPE_RESPONSE,
    ]

    assert await _sent_messages(history) == snapshot(
        [
            {'content': [{'text': 'First question'}], 'role': 'ROLE_USER'},
            {
                'content': [{'text': ''}],
                'role': 'ROLE_ASSISTANT',
                'tool_calls': [
                    {
                        'id': 'call_1',
                        'type': 'TOOL_CALL_TYPE_WEB_SEARCH_TOOL',
                        'status': 'TOOL_CALL_STATUS_COMPLETED',
                        'function': {'name': 'web_search', 'arguments': '{"query":"pydantic"}'},
                    }
                ],
                'encrypted_content': 'reasoning-1',
            },
            {
                'content': [{'text': ''}],
                'role': 'ROLE_ASSISTANT',
                'tool_calls': [
                    {
                        'id': 'call_2',
                        'type': 'TOOL_CALL_TYPE_WEB_SEARCH_TOOL',
                        'status': 'TOOL_CALL_STATUS_COMPLETED',
                        'function': {'name': 'web_search', 'arguments': '{"query":"pydantic"}'},
                    }
                ],
                'encrypted_content': 'result-1',
            },
            {'content': [{'text': ''}], 'role': 'ROLE_ASSISTANT', 'encrypted_content': 'result-2'},
            {'content': [{'text': 'First answer.'}], 'role': 'ROLE_ASSISTANT'},
            {'content': [{'text': 'Second question'}], 'role': 'ROLE_USER'},
            {
                'content': [{'text': ''}],
                'role': 'ROLE_ASSISTANT',
                'tool_calls': [
                    {
                        'id': 'call_3',
                        'type': 'TOOL_CALL_TYPE_WEB_SEARCH_TOOL',
                        'status': 'TOOL_CALL_STATUS_COMPLETED',
                        'function': {'name': 'web_search', 'arguments': '{"query":"pydantic"}'},
                    }
                ],
                'encrypted_content': 'reasoning-3',
            },
            {
                'content': [{'text': ''}],
                'role': 'ROLE_TOOL',
                'tool_calls': [
                    {
                        'id': 'call_3',
                        'type': 'TOOL_CALL_TYPE_WEB_SEARCH_TOOL',
                        'status': 'TOOL_CALL_STATUS_COMPLETED',
                        'function': {'name': 'web_search', 'arguments': '{"query":"pydantic"}'},
                    }
                ],
                'encrypted_content': 'result-3',
                'tool_call_id': 'call_3',
            },
            {
                'content': [{'text': ''}],
                'role': 'ROLE_ASSISTANT',
                'tool_calls': [
                    {
                        'id': 'call_4',
                        'type': 'TOOL_CALL_TYPE_WEB_SEARCH_TOOL',
                        'status': 'TOOL_CALL_STATUS_COMPLETED',
                        'function': {'name': 'web_search', 'arguments': '{"query":"pydantic"}'},
                    }
                ],
            },
            {
                'content': [{'text': ''}],
                'role': 'ROLE_TOOL',
                'tool_calls': [
                    {
                        'id': 'call_4',
                        'type': 'TOOL_CALL_TYPE_WEB_SEARCH_TOOL',
                        'status': 'TOOL_CALL_STATUS_COMPLETED',
                        'function': {'name': 'web_search', 'arguments': '{"query":"pydantic"}'},
                    }
                ],
                'encrypted_content': 'result-4',
                'tool_call_id': 'call_4',
            },
            {'content': [{'text': 'Second answer.'}], 'role': 'ROLE_ASSISTANT'},
            {'content': [{'text': 'And now?'}], 'role': 'ROLE_USER'},
        ]
    )


async def test_xai_other_provider_tool_results_are_not_replayed(allow_model_requests: None):
    """Another provider's server-side call and result in the history are not sent to xAI."""
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='First question')]),
        ModelResponse(
            parts=[
                NativeToolCallPart(
                    tool_name='web_search', args={'query': 'pydantic'}, tool_call_id='ws_1', provider_name='openai'
                ),
                NativeToolReturnPart(
                    tool_name='web_search',
                    content=None,
                    tool_call_id='ws_1',
                    provider_name='openai',
                    provider_details={'encrypted_content': 'not-for-xai'},
                ),
                TextPart(content='First answer.'),
            ],
            model_name='gpt-5',
            provider_name='openai',
        ),
    ]

    assert await _sent_messages(history) == snapshot(
        [
            {'content': [{'text': 'First question'}], 'role': 'ROLE_USER'},
            {'content': [{'text': 'First answer.'}], 'role': 'ROLE_ASSISTANT'},
            {'content': [{'text': 'And now?'}], 'role': 'ROLE_USER'},
        ]
    )
