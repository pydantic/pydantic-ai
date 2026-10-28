"""Filtering for Anthropic server-tool calls that were not enabled by the request."""

from __future__ import annotations

from typing import Literal

import pytest

from pydantic_ai import Agent, ModelMessage, ModelRequest, NativeToolCallPart, NativeToolReturnPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.native_tools import AbstractNativeTool, CodeExecutionTool, WebFetchTool, WebSearchTool
from pydantic_ai.native_tools._tool_search import ToolSearchTool

from ..._inline_snapshot import snapshot
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

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
]

_TOOL_CALL_ID = 'srvtoolu_test'
_CODE_EXECUTION_WIRE_TOOL: dict[str, object] = {'type': 'code_execution_20250825', 'name': 'code_execution'}
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


async def _server_tool_calls_kept(
    stream: bool,
    server_tool_name: ServerToolName,
    input: dict[str, object],
    input_delta: str,
    native_tools: list[AbstractNativeTool],
    model_settings: AnthropicModelSettings | None,
) -> list[NativeToolCallPart]:
    """The server-tool calls left in a response to a request with `native_tools`, out of one call to `server_tool_name`."""
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
        async with model.request_stream(messages, model_settings, request_parameters) as streamed_response:
            async for _ in streamed_response:
                pass
            model_response = streamed_response.get()
    else:
        model_response = await model.request(messages, model_settings, request_parameters)

    return [part for part in model_response.parts if isinstance(part, NativeToolCallPart)]


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
    calls = await _server_tool_calls_kept(stream, server_tool_name, input, input_delta, native_tools, None)
    assert bool(calls) is expected_call
    if expected_call:
        assert calls[0].tool_name == 'code_execution'
        assert calls[0].tool_call_id == _TOOL_CALL_ID
        assert calls[0].args_as_dict() == {'code': 'print(1)'}


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize(
    ('native_tools', 'model_settings', 'expected_call'),
    [
        pytest.param(
            [WebFetchTool()],
            {'anthropic_web_tools_allowed_callers': ['direct']},
            False,
            id='setting-direct',
        ),
        pytest.param(
            [WebFetchTool()],
            {'anthropic_web_tools_allowed_callers': ['code_execution_20260120']},
            True,
            id='setting-code-execution',
        ),
        pytest.param([WebSearchTool()], {'citations': True}, False, id='citations-web-search'),
        pytest.param([WebFetchTool()], {'citations': True}, False, id='citations-web-fetch'),
        pytest.param([WebFetchTool(enable_citations=True)], None, False, id='web-fetch-enable-citations'),
        pytest.param(
            [WebSearchTool(), WebFetchTool(enable_citations=True)], None, True, id='uncited-web-search-keeps-default'
        ),
        pytest.param(
            [WebFetchTool()],
            {'citations': True, 'anthropic_web_tools_allowed_callers': ['code_execution_20260120']},
            True,
            id='setting-takes-precedence-over-citations',
        ),
        pytest.param(
            [WebSearchTool()],
            {'citations': True, 'extra_body': {'tools': [_CODE_EXECUTION_WIRE_TOOL]}},
            True,
            id='extra-body-code-execution-with-cited-web-search',
        ),
        pytest.param(
            [],
            {'citations': True, 'extra_body': {'tools': [_CODE_EXECUTION_WIRE_TOOL]}},
            True,
            id='extra-body-code-execution-alone',
        ),
        pytest.param(
            [WebSearchTool()],
            {'extra_body': {'tools': [{'type': 'web_search_20250305', 'name': 'web_search'}]}},
            False,
            id='extra-body-replaces-code-execution',
        ),
        pytest.param([WebSearchTool()], {'extra_body': {'tools': None}}, False, id='extra-body-null-tools'),
    ],
)
async def test_anthropic_web_tool_callers_gate_implicit_code_execution(
    allow_model_requests: None,
    stream: bool,
    native_tools: list[AbstractNativeTool],
    model_settings: AnthropicModelSettings | None,
    expected_call: bool,
):
    """`code_execution` is implicitly enabled by the web tools only while one of them allows a code execution caller.

    Tools in `extra_body` replace the built ones on the wire, so they decide instead.

    Unit test because a sandbox `code_execution` call can't be requested reliably from the live API when the web tools
    only allow direct calls.
    """
    calls = await _server_tool_calls_kept(
        stream, 'code_execution', {'code': 'print(1)'}, '{"code":"print(1)"}', native_tools, model_settings
    )
    assert [call.tool_name for call in calls] == (['code_execution'] if expected_call else [])


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize(
    ('native_tools', 'tool', 'expected_call'),
    [
        pytest.param(
            [], {'type': 'code_execution_20250522', 'name': 'code_execution'}, True, id='code-execution-20250522'
        ),
        pytest.param([], _CODE_EXECUTION_WIRE_TOOL, True, id='code-execution-20250825'),
        pytest.param(
            [], {'type': 'code_execution_20260120', 'name': 'code_execution'}, True, id='code-execution-20260120'
        ),
        pytest.param(
            [], {'type': 'code_execution_20260521', 'name': 'code_execution'}, True, id='code-execution-20260521'
        ),
        pytest.param([], {'type': 'web_search_20260209', 'name': 'web_search'}, True, id='web-search-20260209'),
        pytest.param([], {'type': 'web_search_20260318', 'name': 'web_search'}, True, id='web-search-20260318'),
        pytest.param([], {'type': 'web_fetch_20260209', 'name': 'web_fetch'}, True, id='web-fetch-20260209'),
        pytest.param([], {'type': 'web_fetch_20260309', 'name': 'web_fetch'}, True, id='web-fetch-20260309'),
        pytest.param([], {'type': 'web_fetch_20260318', 'name': 'web_fetch'}, True, id='web-fetch-20260318'),
        pytest.param([], {'type': 'web_search_20250305', 'name': 'web_search'}, False, id='web-search-20250305'),
        pytest.param([], {'type': 'web_fetch_20250910', 'name': 'web_fetch'}, False, id='web-fetch-20250910'),
        pytest.param(
            [],
            {'type': 'web_fetch_20260209', 'name': 'web_fetch', 'allowed_callers': None},
            True,
            id='null-callers',
        ),
        pytest.param(
            [],
            {'type': 'web_search_20260318', 'name': 'web_search', 'allowed_callers': ['direct']},
            False,
            id='direct-callers',
        ),
        pytest.param(
            [],
            {'type': 'web_fetch_20260209', 'name': 'web_fetch', 'allowed_callers': 'direct'},
            False,
            id='callers-as-string',
        ),
        pytest.param(
            [],
            {'type': 'web_search_20250305', 'name': 'web_search', 'allowed_callers': ['code_execution_20260120']},
            True,
            id='previous-version-code-execution-caller',
        ),
        pytest.param(
            [],
            {'name': 'code_execution', 'input_schema': {'type': 'object'}},
            False,
            id='function-tool-named-code-execution',
        ),
        pytest.param([], {'type': ['code_execution_20250825'], 'name': 'odd'}, False, id='unhashable-type'),
        pytest.param(
            [CodeExecutionTool()],
            {'type': 'code_execution_20260521', 'name': 'code_execution'},
            True,
            id='native-code-execution-newer-extra-body',
        ),
        pytest.param(
            [WebSearchTool()],
            {'type': 'web_search_20260318', 'name': 'web_search'},
            True,
            id='native-web-search-newer-extra-body',
        ),
    ],
)
async def test_anthropic_extra_body_tool_versions_gate_implicit_code_execution(
    allow_model_requests: None,
    stream: bool,
    native_tools: list[AbstractNativeTool],
    tool: dict[str, object],
    expected_call: bool,
):
    """Code execution in `extra_body['tools']` is recognized by tool family, not by the versions Pydantic AI builds.

    Any `code_execution_*` tool provisions it, and so does a web search or fetch tool newer than `web_search_20250305`
    and `web_fetch_20250910` unless its `allowed_callers` only allow direct calls.

    Unit test: a hand-rolled `extra_body['tools']` list can carry versions and shapes no recording has.
    """
    calls = await _server_tool_calls_kept(
        stream,
        'bash_code_execution',
        {'command': 'ls'},
        '{"command":"ls"}',
        native_tools,
        {'extra_body': {'tools': [tool]}},
    )
    assert [call.tool_name for call in calls] == (['code_execution'] if expected_call else [])


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
    assert get_mock_chat_completion_kwargs(mock_client)[1]['messages'] == snapshot(
        [
            {'role': 'user', 'content': [{'text': 'hello', 'type': 'text'}]},
            {
                'role': 'user',
                'content': [{'text': '<system>Please return text.</system>', 'type': 'text'}],
            },
        ]
    )
