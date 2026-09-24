from __future__ import annotations as _annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, ModelRequest, ModelResponse, TextPart, ThinkingPart, ToolCallPart
from pydantic_ai.messages import FinalResultEvent, PartDeltaEvent, PartEndEvent, PartStartEvent
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.profiles import DEFAULT_PROFILE
from pydantic_ai.providers import Provider
from pydantic_ai.usage import RequestUsage

from ...conftest import try_import

with try_import() as imports_successful:
    from botocore.hooks import HierarchicalEmitter

    from pydantic_ai.models.babel.bedrock import BabelBedrockConverseModel, BabelBedrockStreamedResponse

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='boto3 or llm-babel not installed'),
    pytest.mark.anyio,
]


class _EventStream:
    def __init__(self, events: list[dict[str, Any]]):
        self._events = events

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._events)


class _StubBedrockClient:
    def __init__(
        self,
        responses: list[dict[str, Any]] | None = None,
        stream: list[dict[str, Any]] | None = None,
        request_id: str | None = 'req_1',
    ):
        self._responses = iter(responses or [])
        self._stream = stream or []
        self._request_id = request_id
        self.calls: list[dict[str, Any]] = []
        self.meta = SimpleNamespace(endpoint_url='https://bedrock.stub', events=HierarchicalEmitter())

    def converse(self, **params: Any) -> dict[str, Any]:
        self.calls.append(params)
        return next(self._responses)

    def converse_stream(self, **params: Any) -> dict[str, Any]:
        self.calls.append(params)
        return {'stream': _EventStream(self._stream), 'ResponseMetadata': {'RequestId': self._request_id}}


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


def make_model(client: _StubBedrockClient) -> BabelBedrockConverseModel:
    return BabelBedrockConverseModel('us.anthropic.claude-sonnet-4-5', provider=_StubBedrockProvider(client))


async def test_tool_loop(allow_model_requests: None):
    client = _StubBedrockClient(
        responses=[
            converse_response(
                [
                    {'reasoningContent': {'reasoningText': {'text': 'lookup', 'signature': 'SIG'}}},
                    {'toolUse': {'toolUseId': 'tool_1', 'name': 'get_weather', 'input': {'city': 'Paris'}}},
                ],
                'tool_use',
            ),
            converse_response([{'text': 'It is sunny in Paris.'}]),
        ]
    )
    agent = Agent(make_model(client), system_prompt='You are a weather assistant.', instructions='Be terse.')

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
            ToolCallPart(tool_name='get_weather', args='{"city":"Paris"}', tool_call_id='tool_1'),
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
                    {'toolResult': {'toolUseId': 'tool_1', 'content': [{'json': 'Paris: sunny'}], 'status': 'success'}}
                ],
            },
        ]
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
