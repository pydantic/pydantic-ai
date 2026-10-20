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
    PartStartEvent,
    TextPart,
    ThinkingPart,
    UserPromptPart,
    WebSearchTool,
    capture_run_messages,
)
from pydantic_ai.capabilities import NativeTool

from ..._inline_snapshot import snapshot
from ...conftest import try_import
from ..mock_xai import MockXai, create_response, get_mock_chat_create_kwargs

with try_import() as imports_successful:
    from xai_sdk import chat as chat_types
    from xai_sdk.proto import chat_pb2, sample_pb2

    from pydantic_ai.models.xai import XaiModel, XaiModelSettings
    from pydantic_ai.providers.xai import XaiProvider

    from ..xai_proto_cassettes import get_sent_request_messages

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


def _tool_turns(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The sent messages that carry a server-side call or its result, with each id replaced by whether it pairs."""
    turns: list[dict[str, Any]] = []
    for message in messages:
        call_ids = [call['id'] for call in message.get('tool_calls', [])]
        if not call_ids:
            continue
        turn: dict[str, Any] = {
            'role': message['role'],
            'calls': len(call_ids),
            'encrypted_content': 'encrypted_content' in message,
        }
        if 'tool_call_id' in message:
            turn['paired'] = message['tool_call_id'] == call_ids[0]
        turns.append(turn)
    return turns


# The recorded follow-up answers, which quote rows only the search result held.
FOLLOW_UP_OUTPUTS: dict[bool, str] = {
    False: """\
1. [Python 3.15.0](/downloads/release/python-3150/) Oct. 9, 2026
2. [Python 3.14.8](/downloads/release/python-3148/) Sept. 30, 2026
3. [Python 3.14.7](/downloads/release/python-3147/) Aug. 5, 2026\
""",
    True: """\
1. [Python 3.15.0](/downloads/release/python-3150/) Oct. 9, 2026 [Download](/downloads/release/python-3150/) [Release notes](https://docs.python.org/release/3.15.0/whatsnew/changelog.html)
2. [Python 3.14.8](/downloads/release/python-3148/) Sept. 30, 2026 [Download](/downloads/release/python-3148/) [Release notes](https://docs.python.org/release/3.14.8/whatsnew/changelog.html)
3. [Python 3.14.7](/downloads/release/python-3147/) Aug. 5, 2026 [Download](/downloads/release/python-3147/) [Release notes](https://docs.python.org/release/3.14.7/whatsnew/changelog.html)\
""",
}


@pytest.mark.vcr
@pytest.mark.parametrize('stream', [False, True])
async def test_xai_native_tool_result_replayed_paired(
    allow_model_requests: None, xai_provider: XaiProvider, stream: bool
):
    """A web search result is stored on its return part, and the next turn sends it back as a tool turn paired with its call.

    The follow-up asks for rows of the page that only the search result holds, so the recorded answer shows xAI
    accepted the paired tool turn and read the result from it without searching again.
    """
    agent = Agent(
        XaiModel(XAI_REASONING_MODEL, provider=xai_provider),
        capabilities=[NativeTool(WebSearchTool())],
        model_settings=XaiModelSettings(xai_include_encrypted_content=True),
    )
    prompt = 'Open python.org/downloads and tell me the latest Python release. One line.'
    if stream:
        async with agent.run_stream(prompt) as streamed:
            await streamed.get_output()
        history = streamed.all_messages()
    else:
        history = (await agent.run(prompt)).all_messages()

    response = history[-1]
    assert isinstance(response, ModelResponse)
    return_parts = [part for part in response.parts if isinstance(part, NativeToolReturnPart)]
    assert return_parts
    assert all((part.provider_details or {}).get('encrypted_content') for part in return_parts)
    # The result is no longer stored as reasoning: no thinking signature repeats a return part's result.
    results = {(part.provider_details or {})['encrypted_content'] for part in return_parts}
    assert not [part for part in response.parts if isinstance(part, ThinkingPart) and part.signature in results]

    follow_up = await agent.run(
        'Without searching or opening anything again, list the first three rows of the release table on that page '
        '(version and release date), exactly as the page showed them.',
        message_history=history,
    )

    follow_up_response = follow_up.all_messages()[-1]
    assert isinstance(follow_up_response, ModelResponse)
    assert not [part for part in follow_up_response.parts if isinstance(part, NativeToolCallPart)]
    assert follow_up.output == FOLLOW_UP_OUTPUTS[stream]
    assert _tool_turns(get_sent_request_messages(xai_provider.client)[-1]) == snapshot(
        [
            {'role': 'ROLE_ASSISTANT', 'calls': 1, 'encrypted_content': True},
            {'role': 'ROLE_TOOL', 'calls': 1, 'paired': True, 'encrypted_content': True},
        ]
    )


async def test_xai_streamed_tool_result_split_across_chunks(allow_model_requests: None):
    """A tool result that arrives in pieces ends up whole on the return part, and never as thinking.

    Not a recorded test: whether xAI splits a result across chunks isn't something a request can control.
    """
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


async def test_xai_interrupted_stream_keeps_tool_result(allow_model_requests: None):
    """A stream cut short right after a tool result arrives still records the result on the return part."""
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
                chat_pb2.Delta(role=chat_pb2.MessageRole.ROLE_TOOL, tool_calls=[completed], encrypted_content='result'),
            ),
        ),
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

    with capture_run_messages() as messages, pytest.raises(RuntimeError, match='stop'):
        async with agent.run_stream_events('Search for pydantic') as events:
            async for event in events:  # pragma: no branch
                if isinstance(event, PartStartEvent) and isinstance(event.part, NativeToolReturnPart):
                    raise RuntimeError('stop')

    response = messages[-1]
    assert isinstance(response, ModelResponse)
    assert response.state == 'interrupted'
    assert [type(part).__name__ for part in response.parts] == snapshot(['NativeToolCallPart', 'NativeToolReturnPart'])
    assert response.parts[1].provider_details == {'encrypted_content': 'result'}


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

    Not a recorded test: the old shape is history stored by earlier releases, which no current response produces.
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
    """Another provider's server-side call and result in the history are not sent to xAI.

    Not a recorded test: the history comes from another provider, and only what is sent to xAI is asserted.
    """
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
