"""Filtering for Anthropic server-tool calls that were not enabled by the request."""

from __future__ import annotations

from typing import Literal

import pytest

from pydantic_ai import Agent, ModelMessage, ModelRequest, NativeToolCallPart, NativeToolReturnPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.native_tools import AbstractNativeTool, CodeExecutionTool, WebFetchTool
from pydantic_ai.native_tools._tool_search import ToolSearchTool

from ...conftest import try_import
from ..test_anthropic import MockAnthropic, completion_message, get_mock_chat_completion_kwargs

with try_import() as imports_successful:
    from anthropic.types.beta import (
        BetaCodeExecutionResultBlock,
        BetaCodeExecutionToolResultBlock,
        BetaDirectCaller,
        BetaInputJSONDelta,
        BetaMessage,
        BetaMessageDeltaUsage,
        BetaRawContentBlockDeltaEvent,
        BetaRawContentBlockStartEvent,
        BetaRawContentBlockStopEvent,
        BetaRawMessageDeltaEvent,
        BetaRawMessageStartEvent,
        BetaRawMessageStopEvent,
        BetaRawMessageStreamEvent,
        BetaServerToolUseBlock,
        BetaTextBlock,
        BetaUsage,
    )
    from anthropic.types.beta.beta_raw_message_delta_event import Delta

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.anyio,
]

_TOOL_CALL_ID = 'srvtoolu_test'
ServerToolName = Literal[
    'advisor',
    'web_search',
    'web_fetch',
    'code_execution',
    'bash_code_execution',
    'text_editor_code_execution',
    'tool_search_tool_regex',
    'tool_search_tool_bm25',
]


def _server_tool_call(name: ServerToolName, input: dict[str, object]) -> BetaServerToolUseBlock:
    return BetaServerToolUseBlock(
        id=_TOOL_CALL_ID,
        name=name,
        input=input,
        type='server_tool_use',
        caller=BetaDirectCaller(type='direct'),
    )


def _server_tool_stream(
    name: ServerToolName, input_delta: str = '{"code":"print(1)"}'
) -> list[BetaRawMessageStreamEvent]:
    return [
        BetaRawMessageStartEvent(
            type='message_start',
            message=BetaMessage(
                id='msg_test',
                content=[],
                model='claude-sonnet-4-6',
                role='assistant',
                stop_reason=None,
                type='message',
                usage=BetaUsage(input_tokens=1, output_tokens=0),
            ),
        ),
        BetaRawContentBlockStartEvent(
            type='content_block_start',
            index=0,
            content_block=BetaServerToolUseBlock(
                id=_TOOL_CALL_ID,
                name=name,
                input={},
                type='server_tool_use',
                caller=BetaDirectCaller(type='direct'),
            ),
        ),
        BetaRawContentBlockDeltaEvent(
            type='content_block_delta',
            index=0,
            delta=BetaInputJSONDelta(type='input_json_delta', partial_json=input_delta),
        ),
        BetaRawContentBlockStopEvent(type='content_block_stop', index=0),
        BetaRawMessageDeltaEvent(
            type='message_delta',
            delta=Delta(stop_reason='end_turn'),
            usage=BetaMessageDeltaUsage(output_tokens=1),
        ),
        BetaRawMessageStopEvent(type='message_stop'),
    ]


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize(
    ('server_tool_name', 'input', 'input_delta', 'native_tools', 'expected_call'),
    [
        pytest.param('code_execution', {'code': 'print(1)'}, '{"code":"print(1)"}', [], False, id='unconfigured'),
        pytest.param(
            'code_execution',
            {'code': 'print(1)'},
            '{"code":"print(1)"}',
            [CodeExecutionTool()],
            True,
            id='configured',
        ),
        pytest.param(
            'code_execution',
            {'code': 'print(1)'},
            '{"code":"print(1)"}',
            [WebFetchTool()],
            True,
            id='implicitly-enabled-by-web-fetch',
        ),
        pytest.param('advisor', {}, '{}', [], False, id='unsupported-advisor'),
        pytest.param(
            'bash_code_execution',
            {'code': 'print(1)'},
            '{"code":"print(1)"}',
            [CodeExecutionTool()],
            True,
            id='code-execution-sub-tool',
        ),
        pytest.param(
            'tool_search_tool_bm25',
            {'query': 'weather'},
            '{"query":"weather"}',
            [ToolSearchTool(strategy='custom')],
            False,
            id='custom-tool-search',
        ),
    ],
)
async def test_anthropic_filters_unconfigured_native_tool_calls(
    allow_model_requests: None,
    stream: bool,
    server_tool_name: ServerToolName,
    input: dict[str, object],
    input_delta: str,
    native_tools: list[AbstractNativeTool],
    expected_call: bool,
):
    """Unit test because an unconfigured server-tool call cannot be requested reliably from the live API."""
    if stream:
        mock_client = MockAnthropic.create_stream_mock(_server_tool_stream(server_tool_name, input_delta))
    else:
        response = completion_message(
            [_server_tool_call(server_tool_name, input)], BetaUsage(input_tokens=1, output_tokens=1)
        )
        mock_client = MockAnthropic.create_mock(response)

    model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=mock_client))
    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='hello')])]
    request_parameters = ModelRequestParameters(native_tools=native_tools)

    if stream:
        async with model.request_stream(messages, None, request_parameters) as streamed_response:
            async for _ in streamed_response:
                pass
            model_response = streamed_response.get()
    else:
        model_response = await model.request(messages, None, request_parameters)

    calls = [part for part in model_response.parts if isinstance(part, NativeToolCallPart)]
    assert bool(calls) is expected_call
    if expected_call:
        assert calls[0].tool_name == 'code_execution'
        assert calls[0].tool_call_id == _TOOL_CALL_ID
        assert calls[0].args_as_dict() == {'code': 'print(1)'}


async def test_anthropic_keeps_server_tool_call_with_matching_result(allow_model_requests: None):
    """A matching result proves the server executed the call despite the request's declared tool set."""
    result = BetaCodeExecutionToolResultBlock(
        tool_use_id=_TOOL_CALL_ID,
        type='code_execution_tool_result',
        content=BetaCodeExecutionResultBlock(
            content=[], return_code=0, stderr='', stdout='1\n', type='code_execution_result'
        ),
    )
    response = completion_message(
        [_server_tool_call('code_execution', {'code': 'print(1)'}), result],
        BetaUsage(input_tokens=1, output_tokens=1),
    )
    mock_client = MockAnthropic.create_mock(response)
    model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=mock_client))

    model_response = await model.request(
        [ModelRequest(parts=[UserPromptPart(content='hello')])], None, ModelRequestParameters()
    )

    assert [type(part) for part in model_response.parts] == [NativeToolCallPart, NativeToolReturnPart]


async def test_anthropic_agent_recovers_from_unconfigured_native_tool_call(allow_model_requests: None):
    """The regression test uses a mock because the initial hallucination is nondeterministic."""
    first_response = completion_message(
        [_server_tool_call('code_execution', {'code': 'print(1)'})], BetaUsage(input_tokens=1, output_tokens=1)
    )
    second_response = completion_message(
        [BetaTextBlock(text='ok', type='text')], BetaUsage(input_tokens=2, output_tokens=1)
    )
    mock_client = MockAnthropic.create_mock([first_response, second_response])
    model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=mock_client))

    result = await Agent(model).run('hello')

    assert result.output == 'ok'
    assert not any(isinstance(part, NativeToolCallPart) for message in result.all_messages() for part in message.parts)
    assert get_mock_chat_completion_kwargs(mock_client)[1]['messages'] == [
        {'role': 'user', 'content': [{'text': 'hello', 'type': 'text'}]},
        {
            'role': 'user',
            'content': [
                {
                    'text': 'Validation feedback:\nPlease return text.\n\nFix the errors and try again.',
                    'type': 'text',
                }
            ],
        },
    ]
