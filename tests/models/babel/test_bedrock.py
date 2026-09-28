from __future__ import annotations as _annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast

import pytest
from inline_snapshot import snapshot

from pydantic_ai import (
    Agent,
    BinaryContent,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
)
from pydantic_ai.exceptions import ModelHTTPError, UserError
from pydantic_ai.messages import (
    FinalResultEvent,
    InstructionPart,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    ToolReturnPart,
    UploadedFile,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.profiles import DEFAULT_PROFILE
from pydantic_ai.providers import Provider
from pydantic_ai.usage import RequestUsage

from ...conftest import try_import

with try_import() as imports_successful:
    from botocore.exceptions import ClientError
    from botocore.hooks import HierarchicalEmitter

    from pydantic_ai.models.babel.bedrock import BabelBedrockConverseModel, BabelBedrockStreamedResponse
    from pydantic_ai.models.bedrock import BedrockModelSettings
    from pydantic_ai.providers.bedrock import BedrockModelProfile

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='boto3 or llm-babel not installed'),
    pytest.mark.anyio,
]


class _EventStream:
    def __init__(self, events: list[dict[str, Any]], error: Exception | None = None):
        self._events = events
        self._error = error

    def __iter__(self) -> Iterator[dict[str, Any]]:
        yield from self._events
        if self._error is not None:
            raise self._error


class _StubBedrockClient:
    def __init__(
        self,
        responses: list[dict[str, Any]] | None = None,
        stream: list[dict[str, Any]] | None = None,
        request_id: str | None = 'req_1',
        stream_error: Exception | None = None,
    ):
        self._responses = iter(responses or [])
        self._stream = stream or []
        self._stream_error = stream_error
        self._request_id = request_id
        self.calls: list[dict[str, Any]] = []
        self.meta = SimpleNamespace(endpoint_url='https://bedrock.stub', events=HierarchicalEmitter())

    def converse(self, **params: Any) -> dict[str, Any]:
        self.calls.append(params)
        return next(self._responses)

    def converse_stream(self, **params: Any) -> dict[str, Any]:
        self.calls.append(params)
        return {
            'stream': _EventStream(self._stream, self._stream_error),
            'ResponseMetadata': {'RequestId': self._request_id},
        }


class _StubBedrockProvider(Provider[Any]):
    def __init__(self, client: _StubBedrockClient):
        self._client = client

    @property
    def name(self) -> str:
        return 'bedrock'

    @property
    def base_url(self) -> str:
        return 'https://bedrock.stub'

    @property
    def client(self) -> _StubBedrockClient:
        return self._client

    @staticmethod
    def model_profile(model_name: str):
        return DEFAULT_PROFILE


def converse_response(content: list[dict[str, Any]], stop_reason: str = 'end_turn') -> dict[str, Any]:
    return {
        'output': {'message': {'role': 'assistant', 'content': content}},
        'stopReason': stop_reason,
        'usage': {'inputTokens': 10, 'outputTokens': 4, 'totalTokens': 14, 'cacheReadInputTokens': 6},
        'ResponseMetadata': {'RequestId': 'req_1'},
    }


def make_model(
    client: _StubBedrockClient,
    profile: BedrockModelProfile | None = None,
    model_name: str = 'us.anthropic.claude-sonnet-4-5',
) -> BabelBedrockConverseModel:
    return BabelBedrockConverseModel(model_name, provider=_StubBedrockProvider(client), profile=profile)


def tool_loop_responses() -> list[dict[str, Any]]:
    return [
        converse_response(
            [
                {'reasoningContent': {'reasoningText': {'text': 'lookup', 'signature': 'SIG'}}},
                {'toolUse': {'toolUseId': 'tool_1', 'name': 'get_weather', 'input': {'city': 'Paris'}}},
            ],
            'tool_use',
        ),
        converse_response([{'text': 'It is sunny in Paris.'}]),
    ]


async def test_tool_loop(allow_model_requests: None):
    client = _StubBedrockClient(responses=tool_loop_responses())
    # The profile decides whether the thinking block is replayed, as it does for the native model.
    model = make_model(client, BedrockModelProfile(bedrock_send_back_thinking_parts=True))
    agent = Agent(model, system_prompt='You are a weather assistant.', instructions='Be terse.')

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return f'{city}: sunny'

    result = await agent.run('What is the weather in Paris?')
    assert result.output == 'It is sunny in Paris.'
    response = result.all_messages()[1]
    assert isinstance(response, ModelResponse)
    assert response.parts == snapshot(
        [
            ThinkingPart(content='lookup', signature='SIG', provider_name='bedrock'),
            ToolCallPart(tool_name='get_weather', args={'city': 'Paris'}, tool_call_id='tool_1'),
        ]
    )
    assert response.usage == snapshot(RequestUsage(input_tokens=16, cache_read_tokens=6, output_tokens=4))
    assert response.model_name == 'us.anthropic.claude-sonnet-4-5'
    assert response.provider_name == 'bedrock'
    assert response.provider_url == 'https://bedrock.stub'
    assert response.provider_response_id == 'req_1'
    assert response.finish_reason == 'tool_call'
    assert client.calls[1]['system'] == snapshot([{'text': 'You are a weather assistant.'}, {'text': 'Be terse.'}])
    assert client.calls[1]['messages'] == snapshot(
        [
            {'role': 'user', 'content': [{'text': 'What is the weather in Paris?'}]},
            {
                'role': 'assistant',
                'content': [
                    {'reasoningContent': {'reasoningText': {'text': 'lookup', 'signature': 'SIG'}}},
                    {'toolUse': {'toolUseId': 'tool_1', 'name': 'get_weather', 'input': {'city': 'Paris'}}},
                ],
            },
            {
                'role': 'user',
                'content': [
                    {'toolResult': {'toolUseId': 'tool_1', 'content': [{'text': 'Paris: sunny'}], 'status': 'success'}}
                ],
            },
        ]
    )


async def test_thinking_is_not_replayed_unless_the_profile_says_so(allow_model_requests: None):
    client = _StubBedrockClient(responses=tool_loop_responses())
    agent = Agent(make_model(client))

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return f'{city}: sunny'

    await agent.run('What is the weather in Paris?')
    assert client.calls[1]['messages'][1] == snapshot(
        {
            'role': 'assistant',
            'content': [{'toolUse': {'toolUseId': 'tool_1', 'name': 'get_weather', 'input': {'city': 'Paris'}}}],
        }
    )


async def test_stream(allow_model_requests: None):
    client = _StubBedrockClient(
        stream=[
            {'messageStart': {'role': 'assistant'}},
            {'contentBlockDelta': {'contentBlockIndex': 0, 'delta': {'text': 'It is '}}},
            {'contentBlockDelta': {'contentBlockIndex': 0, 'delta': {'text': 'sunny.'}}},
            {'contentBlockStop': {'contentBlockIndex': 0}},
            {'contentBlockStart': {'contentBlockIndex': 1, 'start': {'toolUse': {'toolUseId': 'tool_1', 'name': 'f'}}}},
            {'contentBlockDelta': {'contentBlockIndex': 1, 'delta': {'toolUse': {'input': '{"a": 1}'}}}},
            {'contentBlockStop': {'contentBlockIndex': 1}},
            {'messageStop': {'stopReason': 'tool_use'}},
            {
                'metadata': {
                    'usage': {'inputTokens': 10, 'outputTokens': 4, 'totalTokens': 14},
                    'metrics': {'latencyMs': 1},
                }
            },
            {'metadata': {'metrics': {'latencyMs': 2}}},
        ]
    )
    model = make_model(client)
    async with model.request_stream([ModelRequest.user_text_prompt('hi')], None, ModelRequestParameters()) as response:
        assert isinstance(response, BabelBedrockStreamedResponse)
        events = [event async for event in response]
    assert [type(event) for event in events] == snapshot(
        [
            PartStartEvent,
            FinalResultEvent,
            PartDeltaEvent,
            PartEndEvent,
            PartStartEvent,
            PartDeltaEvent,
            PartEndEvent,
        ]
    )
    assert response.get().parts == snapshot(
        [TextPart(content='It is sunny.'), ToolCallPart(tool_name='f', args='{"a":1}', tool_call_id='tool_1')]
    )
    assert response.finish_reason == 'tool_call'
    assert response.provider_response_id == 'req_1'
    assert response.usage == snapshot(RequestUsage(input_tokens=10, output_tokens=4))


async def test_stream_api_error_is_mapped(allow_model_requests: None):
    error = ClientError(
        {
            'Error': {'Code': 'ThrottlingException', 'Message': 'slow down'},
            'ResponseMetadata': {
                'HTTPStatusCode': 429,
                'HTTPHeaders': {},
                'RequestId': 'req_1',
                'HostId': '',
                'RetryAttempts': 0,
            },
        },
        'ConverseStream',
    )
    client = _StubBedrockClient(
        stream=[
            {'messageStart': {'role': 'assistant'}},
            {'contentBlockDelta': {'contentBlockIndex': 0, 'delta': {'text': 'hi'}}},
        ],
        stream_error=error,
    )
    model = make_model(client)
    with pytest.raises(ModelHTTPError) as exc_info:
        async with model.request_stream(
            [ModelRequest.user_text_prompt('hi')], None, ModelRequestParameters()
        ) as response:
            _ = [event async for event in response]
    assert exc_info.value.status_code == 429
    assert exc_info.value.body['Error']['Message'] == 'slow down'  # type: ignore[index]


async def test_stream_without_request_id(allow_model_requests: None):
    client = _StubBedrockClient(
        stream=[
            {'messageStart': {'role': 'assistant'}},
            {'contentBlockDelta': {'contentBlockIndex': 0, 'delta': {'text': 'hi'}}},
            {'contentBlockStop': {'contentBlockIndex': 0}},
            {'messageStop': {'stopReason': 'end_turn'}},
        ],
        request_id=None,
    )
    model = make_model(client)
    async with model.request_stream([ModelRequest.user_text_prompt('hi')], None, ModelRequestParameters()) as response:
        _ = [event async for event in response]
    assert response.provider_response_id is None
    assert response.get().parts == [TextPart(content='hi')]
    assert response.finish_reason == 'stop'


async def test_guardrail_trace_and_raw_stop_reason(allow_model_requests: None):
    trace = {'guardrail': {'modelOutput': ['blocked']}}
    client = _StubBedrockClient(
        responses=[{**converse_response([{'text': 'hello'}], 'guardrail_intervened'), 'trace': trace}]
    )
    response = await make_model(client).request([ModelRequest.user_text_prompt('hi')], None, ModelRequestParameters())
    assert response.finish_reason == 'content_filter'
    assert response.provider_details == {'finish_reason': 'guardrail_intervened', 'trace': trace}


async def test_stream_guardrail_trace(allow_model_requests: None):
    trace = {'guardrail': {'modelOutput': ['blocked']}}
    client = _StubBedrockClient(
        stream=[
            {'messageStart': {'role': 'assistant'}},
            {'contentBlockDelta': {'contentBlockIndex': 0, 'delta': {'text': 'hello'}}},
            {'contentBlockStop': {'contentBlockIndex': 0}},
            {'messageStop': {'stopReason': 'guardrail_intervened'}},
            {'metadata': {'usage': {'inputTokens': 1, 'outputTokens': 1, 'totalTokens': 2}, 'trace': trace}},
        ]
    )
    model = make_model(client)
    async with model.request_stream([ModelRequest.user_text_prompt('hi')], None, ModelRequestParameters()) as response:
        _ = [event async for event in response]
    assert response.finish_reason == 'content_filter'
    assert response.provider_details == {'finish_reason': 'guardrail_intervened', 'trace': trace}


async def test_s3_urls_are_passed_through(allow_model_requests: None):
    client = _StubBedrockClient(responses=[converse_response([{'text': 'an image'}])])
    await Agent(make_model(client)).run(['look', ImageUrl(url='s3://bucket/a.png')])
    assert client.calls[0]['messages'][0]['content'] == snapshot(
        [{'text': 'look'}, {'image': {'format': 'png', 'source': {'s3Location': {'uri': 's3://bucket/a.png'}}}}]
    )


async def test_stream_ignores_leading_whitespace_when_the_profile_says_so(allow_model_requests: None):
    client = _StubBedrockClient(
        stream=[
            {'messageStart': {'role': 'assistant'}},
            {'contentBlockDelta': {'contentBlockIndex': 0, 'delta': {'text': '\n\n'}}},
            {'contentBlockDelta': {'contentBlockIndex': 0, 'delta': {'text': 'Paris'}}},
            {'contentBlockStop': {'contentBlockIndex': 0}},
            {'messageStop': {'stopReason': 'end_turn'}},
        ]
    )
    model = BabelBedrockConverseModel(
        'qwen.qwen3-coder-next',
        provider=_StubBedrockProvider(client),
        profile=BedrockModelProfile(ignore_streamed_leading_whitespace=True),
    )
    async with model.request_stream([ModelRequest.user_text_prompt('hi')], None, ModelRequestParameters()) as response:
        _ = [event async for event in response]
    assert response.get().parts == [TextPart(content='Paris')]


def tool_result_history(content: Any, outcome: str = 'success') -> list[ModelRequest | ModelResponse]:
    return [
        ModelRequest.user_text_prompt('hi'),
        ModelResponse(parts=[ToolCallPart(tool_name='f', args={}, tool_call_id='t1')]),
        ModelRequest(parts=[ToolReturnPart(tool_name='f', content=content, tool_call_id='t1', outcome=outcome)]),  # pyright: ignore[reportArgumentType]
    ]


async def sent_messages(
    model: BabelBedrockConverseModel, history: list[ModelRequest | ModelResponse]
) -> list[dict[str, Any]]:
    await model.request(history, None, ModelRequestParameters())
    client = cast(_StubBedrockClient, model.client)
    return client.calls[0]['messages']


def one_reply() -> _StubBedrockClient:
    return _StubBedrockClient(responses=[converse_response([{'text': 'ok'}])])


async def test_tool_result_block_follows_the_profile(allow_model_requests: None):
    # The family default: every result as a text block, a structured one serialized.
    messages = await sent_messages(make_model(one_reply()), tool_result_history({'temp': 18}))
    assert messages[-1]['content'] == snapshot(
        [{'toolResult': {'toolUseId': 't1', 'content': [{'text': '{"temp":18}'}], 'status': 'success'}}]
    )
    # A family that wants structured results as `json` blocks (Mistral) still sends a string as text.
    json_profile = BedrockModelProfile(bedrock_tool_result_format='json')
    messages = await sent_messages(make_model(one_reply(), json_profile), tool_result_history({'temp': 18}))
    assert messages[-1]['content'] == snapshot(
        [{'toolResult': {'toolUseId': 't1', 'content': [{'json': {'temp': 18}}], 'status': 'success'}}]
    )
    messages = await sent_messages(make_model(one_reply(), json_profile), tool_result_history('ok'))
    assert messages[-1]['content'] == snapshot(
        [{'toolResult': {'toolUseId': 't1', 'content': [{'text': 'ok'}], 'status': 'success'}}]
    )


async def test_failed_tool_result_takes_the_status_channel_or_folds_into_the_content(allow_model_requests: None):
    messages = await sent_messages(make_model(one_reply()), tool_result_history('boom', outcome='failed'))
    assert messages[-1]['content'] == snapshot(
        [{'toolResult': {'toolUseId': 't1', 'content': [{'text': 'boom'}], 'status': 'error'}}]
    )
    # A family that rejects `status` (Writer) gets the failure inside the content instead, as from the native model.
    no_status = BedrockModelProfile(bedrock_supports_tool_result_status=False)
    messages = await sent_messages(make_model(one_reply(), no_status), tool_result_history('boom', outcome='failed'))
    assert messages[-1]['content'] == snapshot(
        [{'toolResult': {'toolUseId': 't1', 'content': [{'text': '{"error":"boom"}'}]}}]
    )
    # A denial is an ordinary result: its content says what happened.
    messages = await sent_messages(make_model(one_reply()), tool_result_history('not allowed', outcome='denied'))
    assert messages[-1]['content'][0]['toolResult']['status'] == 'success'


async def test_leading_assistant_turn_is_padded_unless_the_profile_allows_it(allow_model_requests: None):
    history: list[ModelRequest | ModelResponse] = [
        ModelResponse(parts=[TextPart(content='Hello!')]),
        ModelRequest.user_text_prompt('hi'),
    ]
    messages = await sent_messages(make_model(one_reply()), history)
    assert [(m['role'], m['content']) for m in messages] == snapshot(
        [
            ('user', [{'text': '.'}]),
            ('assistant', [{'text': 'Hello!'}]),
            ('user', [{'text': 'hi'}]),
        ]
    )
    allows = BedrockModelProfile(bedrock_supports_leading_assistant_message=True)
    messages = await sent_messages(make_model(one_reply(), allows), history)
    assert [m['role'] for m in messages] == ['assistant', 'user']


async def test_cache_settings_place_the_breakpoints_the_native_model_places(allow_model_requests: None):
    client = _StubBedrockClient(responses=[converse_response([{'text': 'ok'}])] * 2)
    model = make_model(client, BedrockModelProfile(bedrock_supports_prompt_caching=True))
    agent = Agent(model, system_prompt='You are terse.', instructions='Static.')

    @agent.instructions
    def dynamic() -> str:
        return 'Dynamic.'

    settings = BedrockModelSettings(bedrock_cache_instructions=True, bedrock_cache_messages='1h')
    await agent.run('hi', model_settings=settings)
    # The instructions breakpoint follows the last static instruction; the messages one ends the last user turn.
    assert client.calls[0]['system'] == snapshot(
        [{'text': 'You are terse.'}, {'text': 'Static.'}, {'cachePoint': {'type': 'default'}}, {'text': 'Dynamic.'}]
    )
    assert client.calls[0]['messages'] == snapshot(
        [{'role': 'user', 'content': [{'text': 'hi'}, {'cachePoint': {'type': 'default', 'ttl': '1h'}}]}]
    )
    # A profile without prompt caching leaves the settings without effect, as the native model does.
    await Agent(make_model(client), system_prompt='You are terse.').run('hi', model_settings=settings)
    assert client.calls[1]['system'] == [{'text': 'You are terse.'}]
    assert client.calls[1]['messages'] == [{'role': 'user', 'content': [{'text': 'hi'}]}]


async def test_audio_is_not_supported(allow_model_requests: None):
    request = ModelRequest(parts=[UserPromptPart(content=[BinaryContent(data=b'mp3', media_type='audio/mpeg')])])
    with pytest.raises(NotImplementedError, match='Audio content is not supported by this model'):
        await make_model(one_reply()).request([request], None, ModelRequestParameters())


def test_cache_point_placement_edge_cases(allow_model_requests: None):
    model = make_model(one_reply(), BedrockModelProfile(bedrock_supports_prompt_caching=True))
    add = model._add_cache_points  # pyright: ignore[reportPrivateUsage]
    point = {'cachePoint': {'type': 'default'}}
    # Only static instructions: the breakpoint ends the whole system prompt.
    system: list[Any] = [{'text': 'sys'}, {'text': 'static'}]
    add(system, [], [InstructionPart(content='static')], BedrockModelSettings(bedrock_cache_instructions=True))
    assert system == [{'text': 'sys'}, {'text': 'static'}, point]
    # Only dynamic instructions and no system prompt: nothing static precedes them, so no breakpoint.
    system = [{'text': 'dynamic'}]
    add(
        system,
        [],
        [InstructionPart(content='dynamic', dynamic=True)],
        BedrockModelSettings(bedrock_cache_instructions=True),
    )
    assert system == [{'text': 'dynamic'}]
    # No system prompt, and no user turn for the messages breakpoint to end: neither is placed.
    messages: list[Any] = [{'role': 'assistant', 'content': [{'text': 'hi'}]}]
    add([], messages, [], BedrockModelSettings(bedrock_cache_instructions=True, bedrock_cache_messages=True))
    assert messages == [{'role': 'assistant', 'content': [{'text': 'hi'}]}]


async def test_uploaded_file_from_another_provider_is_rejected(allow_model_requests: None):
    request = ModelRequest(parts=[UserPromptPart(content=[UploadedFile(file_id='file-1', provider_name='openai')])])
    with pytest.raises(UserError, match=r"provider_name='openai'.*cannot be used with BabelBedrockConverseModel"):
        await make_model(one_reply()).request([request], None, ModelRequestParameters())
