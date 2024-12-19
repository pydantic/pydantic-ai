# pyright: reportPrivateUsage=false
from __future__ import annotations as _annotations

import json
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from datetime import timezone

import httpx
import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel
from typing_extensions import Literal, TypeAlias

from pydantic_ai import Agent, ModelRetry, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import (
    ArgsDict,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.gemini import (
    ApiKeyAuth,
    GeminiModel,
    _content_model_response,
    _function_call_part_from_call,
    _gemini_response_ta,
    _gemini_streamed_response_ta,
    _GeminiCandidates,
    _GeminiContent,
    _GeminiFunction,
    _GeminiFunctionCallingConfig,
    _GeminiResponse,
    _GeminiTextPart,
    _GeminiToolConfig,
    _GeminiTools,
    _GeminiUsageMetaData,
)
from pydantic_ai.result import Usage
from pydantic_ai.tools import ToolDefinition

from ..conftest import ClientWithHandler, IsNow, TestEnv

pytestmark = pytest.mark.anyio


def test_api_key_arg(env: TestEnv):
    env.set('GEMINI_API_KEY', 'via-env-var')
    m = GeminiModel('gemini-1.5-flash', api_key='via-arg')
    assert isinstance(m.auth, ApiKeyAuth)
    assert m.auth.api_key == 'via-arg'


def test_api_key_env_var(env: TestEnv):
    env.set('GEMINI_API_KEY', 'via-env-var')
    m = GeminiModel('gemini-1.5-flash')
    assert isinstance(m.auth, ApiKeyAuth)
    assert m.auth.api_key == 'via-env-var'


def test_api_key_not_set(env: TestEnv):
    env.remove('GEMINI_API_KEY')
    with pytest.raises(UserError, match='API key must be provided or set in the GEMINI_API_KEY environment variable'):
        GeminiModel('gemini-1.5-flash')


def test_api_key_empty(env: TestEnv):
    env.set('GEMINI_API_KEY', '')
    with pytest.raises(UserError, match='API key must be provided or set in the GEMINI_API_KEY environment variable'):
        GeminiModel('gemini-1.5-flash')


async def test_agent_model_simple(allow_model_requests: None):
    m = GeminiModel('gemini-1.5-flash', api_key='via-arg')
    agent_model = await m.agent_model(function_tools=[], allow_text_result=True, result_tools=[])
    assert isinstance(agent_model.http_client, httpx.AsyncClient)
    assert agent_model.model_name == 'gemini-1.5-flash'
    assert isinstance(agent_model.auth, ApiKeyAuth)
    assert agent_model.auth.api_key == 'via-arg'
    assert agent_model.tools is None
    assert agent_model.tool_config is None


async def test_agent_model_tools(allow_model_requests: None):
    m = GeminiModel('gemini-1.5-flash', api_key='via-arg')
    tools = [
        ToolDefinition(
            'foo',
            'This is foo',
            {'type': 'object', 'title': 'Foo', 'properties': {'bar': {'type': 'number', 'title': 'Bar'}}},
        ),
        ToolDefinition(
            'apple',
            'This is apple',
            {
                'type': 'object',
                'properties': {
                    'banana': {'type': 'array', 'title': 'Banana', 'items': {'type': 'number', 'title': 'Bar'}}
                },
            },
        ),
    ]
    result_tool = ToolDefinition(
        'result',
        'This is the tool for the final Result',
        {'type': 'object', 'title': 'Result', 'properties': {'spam': {'type': 'number'}}, 'required': ['spam']},
    )
    agent_model = await m.agent_model(function_tools=tools, allow_text_result=True, result_tools=[result_tool])
    assert agent_model.tools == snapshot(
        _GeminiTools(
            function_declarations=[
                _GeminiFunction(
                    name='foo',
                    description='This is foo',
                    parameters={'type': 'object', 'properties': {'bar': {'type': 'number'}}},
                ),
                _GeminiFunction(
                    name='apple',
                    description='This is apple',
                    parameters={
                        'type': 'object',
                        'properties': {'banana': {'type': 'array', 'items': {'type': 'number'}}},
                    },
                ),
                _GeminiFunction(
                    name='result',
                    description='This is the tool for the final Result',
                    parameters={
                        'type': 'object',
                        'properties': {'spam': {'type': 'number'}},
                        'required': ['spam'],
                    },
                ),
            ]
        )
    )
    assert agent_model.tool_config is None


async def test_require_response_tool(allow_model_requests: None):
    m = GeminiModel('gemini-1.5-flash', api_key='via-arg')
    result_tool = ToolDefinition(
        'result',
        'This is the tool for the final Result',
        {'type': 'object', 'title': 'Result', 'properties': {'spam': {'type': 'number'}}},
    )
    agent_model = await m.agent_model(function_tools=[], allow_text_result=False, result_tools=[result_tool])
    assert agent_model.tools == snapshot(
        _GeminiTools(
            function_declarations=[
                _GeminiFunction(
                    name='result',
                    description='This is the tool for the final Result',
                    parameters={
                        'type': 'object',
                        'properties': {'spam': {'type': 'number'}},
                    },
                ),
            ]
        )
    )
    assert agent_model.tool_config == snapshot(
        _GeminiToolConfig(
            function_calling_config=_GeminiFunctionCallingConfig(mode='ANY', allowed_function_names=['result'])
        )
    )


async def test_json_def_replaced(allow_model_requests: None):
    class Location(BaseModel):
        lat: float
        lng: float = 1.1

    class Locations(BaseModel):
        locations: list[Location]

    json_schema = Locations.model_json_schema()
    assert json_schema == snapshot(
        {
            '$defs': {
                'Location': {
                    'properties': {
                        'lat': {'title': 'Lat', 'type': 'number'},
                        'lng': {'default': 1.1, 'title': 'Lng', 'type': 'number'},
                    },
                    'required': ['lat'],
                    'title': 'Location',
                    'type': 'object',
                }
            },
            'properties': {'locations': {'items': {'$ref': '#/$defs/Location'}, 'title': 'Locations', 'type': 'array'}},
            'required': ['locations'],
            'title': 'Locations',
            'type': 'object',
        }
    )

    m = GeminiModel('gemini-1.5-flash', api_key='via-arg')
    result_tool = ToolDefinition(
        'result',
        'This is the tool for the final Result',
        json_schema,
    )
    agent_model = await m.agent_model(function_tools=[], allow_text_result=True, result_tools=[result_tool])
    assert agent_model.tools == snapshot(
        _GeminiTools(
            function_declarations=[
                _GeminiFunction(
                    name='result',
                    description='This is the tool for the final Result',
                    parameters={
                        'properties': {
                            'locations': {
                                'items': {
                                    'properties': {
                                        'lat': {'type': 'number'},
                                        'lng': {'type': 'number'},
                                    },
                                    'required': ['lat'],
                                    'type': 'object',
                                },
                                'type': 'array',
                            }
                        },
                        'required': ['locations'],
                        'type': 'object',
                    },
                )
            ]
        )
    )


async def test_json_def_replaced_any_of(allow_model_requests: None):
    class Location(BaseModel):
        lat: float
        lng: float

    class Locations(BaseModel):
        op_location: Location | None = None

    json_schema = Locations.model_json_schema()

    m = GeminiModel('gemini-1.5-flash', api_key='via-arg')
    result_tool = ToolDefinition(
        'result',
        'This is the tool for the final Result',
        json_schema,
    )
    agent_model = await m.agent_model(function_tools=[], allow_text_result=True, result_tools=[result_tool])
    assert agent_model.tools == snapshot(
        _GeminiTools(
            function_declarations=[
                _GeminiFunction(
                    name='result',
                    description='This is the tool for the final Result',
                    parameters={
                        'properties': {
                            'op_location': {
                                'anyOf': [
                                    {
                                        'properties': {'lat': {'type': 'number'}, 'lng': {'type': 'number'}},
                                        'required': ['lat', 'lng'],
                                        'type': 'object',
                                    },
                                    {'type': 'null'},
                                ]
                            }
                        },
                        'type': 'object',
                    },
                )
            ]
        )
    )


async def test_json_def_recursive(allow_model_requests: None):
    class Location(BaseModel):
        lat: float
        lng: float
        nested_locations: list[Location]

    json_schema = Location.model_json_schema()
    assert json_schema == snapshot(
        {
            '$defs': {
                'Location': {
                    'properties': {
                        'lat': {'title': 'Lat', 'type': 'number'},
                        'lng': {'title': 'Lng', 'type': 'number'},
                        'nested_locations': {
                            'items': {'$ref': '#/$defs/Location'},
                            'title': 'Nested Locations',
                            'type': 'array',
                        },
                    },
                    'required': ['lat', 'lng', 'nested_locations'],
                    'title': 'Location',
                    'type': 'object',
                }
            },
            '$ref': '#/$defs/Location',
        }
    )

    m = GeminiModel('gemini-1.5-flash', api_key='via-arg')
    result_tool = ToolDefinition(
        'result',
        'This is the tool for the final Result',
        json_schema,
    )
    with pytest.raises(UserError, match=r'Recursive `\$ref`s in JSON Schema are not supported by Gemini'):
        await m.agent_model(function_tools=[], allow_text_result=True, result_tools=[result_tool])


@dataclass
class AsyncByteStreamList(httpx.AsyncByteStream):
    data: list[bytes]

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.data:
            yield chunk


ResOrList: TypeAlias = '_GeminiResponse | httpx.AsyncByteStream | Sequence[_GeminiResponse | httpx.AsyncByteStream]'
GetGeminiClient: TypeAlias = 'Callable[[ResOrList], httpx.AsyncClient]'


@pytest.fixture
async def get_gemini_client(
    client_with_handler: ClientWithHandler, env: TestEnv, allow_model_requests: None
) -> GetGeminiClient:
    env.set('GEMINI_API_KEY', 'via-env-var')

    def create_client(response_or_list: ResOrList) -> httpx.AsyncClient:
        index = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal index

            ua = request.headers.get('User-Agent')
            assert isinstance(ua, str) and ua.startswith('pydantic-ai')

            if isinstance(response_or_list, Sequence):
                response = response_or_list[index]
                index += 1
            else:
                response = response_or_list

            if isinstance(response, httpx.AsyncByteStream):
                content: bytes | None = None
                stream: httpx.AsyncByteStream | None = response
            else:
                content = _gemini_response_ta.dump_json(response, by_alias=True)
                stream = None

            return httpx.Response(
                200,
                content=content,
                stream=stream,
                headers={'Content-Type': 'application/json'},
            )

        return client_with_handler(handler)

    return create_client


def gemini_response(content: _GeminiContent, finish_reason: Literal['STOP'] | None = 'STOP') -> _GeminiResponse:
    candidate = _GeminiCandidates(content=content, index=0, safety_ratings=[])
    if finish_reason:  # pragma: no cover
        candidate['finish_reason'] = finish_reason
    return _GeminiResponse(candidates=[candidate], usage_metadata=example_usage())


def example_usage() -> _GeminiUsageMetaData:
    return _GeminiUsageMetaData(prompt_token_count=1, candidates_token_count=2, total_token_count=3)


async def test_text_success(get_gemini_client: GetGeminiClient):
    response = gemini_response(_content_model_response(ModelResponse.from_text('Hello world')))
    gemini_client = get_gemini_client(response)
    m = GeminiModel('gemini-1.5-flash', http_client=gemini_client)
    agent = Agent(m)

    result = await agent.run('Hello')
    assert result.data == 'Hello world'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse.from_text(content='Hello world', timestamp=IsNow(tz=timezone.utc)),
        ]
    )
    assert result.usage() == snapshot(Usage(requests=1, request_tokens=1, response_tokens=2, total_tokens=3))

    result = await agent.run('Hello', message_history=result.new_messages())
    assert result.data == 'Hello world'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse.from_text(content='Hello world', timestamp=IsNow(tz=timezone.utc)),
            ModelRequest(parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse.from_text(content='Hello world', timestamp=IsNow(tz=timezone.utc)),
        ]
    )


async def test_request_structured_response(get_gemini_client: GetGeminiClient):
    response = gemini_response(
        _content_model_response(
            ModelResponse(parts=[ToolCallPart.from_raw_args('final_result', {'response': [1, 2, 123]})])
        )
    )
    gemini_client = get_gemini_client(response)
    m = GeminiModel('gemini-1.5-flash', http_client=gemini_client)
    agent = Agent(m, result_type=list[int])

    result = await agent.run('Hello')
    assert result.data == [1, 2, 123]
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args=ArgsDict(args_dict={'response': [1, 2, 123]}),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result', content='Final result processed.', timestamp=IsNow(tz=timezone.utc)
                    )
                ]
            ),
        ]
    )


async def test_request_tool_call(get_gemini_client: GetGeminiClient):
    responses = [
        gemini_response(
            _content_model_response(
                ModelResponse(parts=[ToolCallPart.from_raw_args('get_location', {'loc_name': 'San Fransisco'})])
            )
        ),
        gemini_response(
            _content_model_response(
                ModelResponse(parts=[ToolCallPart.from_raw_args('get_location', {'loc_name': 'London'})])
            )
        ),
        gemini_response(_content_model_response(ModelResponse.from_text('final response'))),
    ]
    gemini_client = get_gemini_client(responses)
    m = GeminiModel('gemini-1.5-flash', http_client=gemini_client)
    agent = Agent(m, system_prompt='this is the system prompt')

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        if loc_name == 'London':
            return json.dumps({'lat': 51, 'lng': 0})
        else:
            raise ModelRetry('Wrong location, please try again')

    result = await agent.run('Hello')
    assert result.data == 'final response'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='this is the system prompt'),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args=ArgsDict(args_dict={'loc_name': 'San Fransisco'}),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Wrong location, please try again',
                        tool_name='get_location',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args=ArgsDict(args_dict={'loc_name': 'London'}),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_location', content='{"lat": 51, "lng": 0}', timestamp=IsNow(tz=timezone.utc)
                    )
                ]
            ),
            ModelResponse.from_text(content='final response', timestamp=IsNow(tz=timezone.utc)),
        ]
    )
    assert result.usage() == snapshot(Usage(requests=3, request_tokens=3, response_tokens=6, total_tokens=9))


async def test_unexpected_response(client_with_handler: ClientWithHandler, env: TestEnv, allow_model_requests: None):
    env.set('GEMINI_API_KEY', 'via-env-var')

    def handler(_: httpx.Request):
        return httpx.Response(401, content='invalid request')

    gemini_client = client_with_handler(handler)
    m = GeminiModel('gemini-1.5-flash', http_client=gemini_client)
    agent = Agent(m, system_prompt='this is the system prompt')

    with pytest.raises(UnexpectedModelBehavior) as exc_info:
        await agent.run('Hello')

    assert str(exc_info.value) == snapshot('Unexpected response from gemini 401, body:\ninvalid request')


async def test_heterogeneous_responses(get_gemini_client: GetGeminiClient):
    response = gemini_response(
        _GeminiContent(
            role='model',
            parts=[
                _GeminiTextPart(text='foo'),
                _function_call_part_from_call(
                    ToolCallPart(
                        tool_name='get_location',
                        args=ArgsDict(args_dict={'loc_name': 'San Fransisco'}),
                    )
                ),
            ],
        )
    )
    gemini_client = get_gemini_client(response)
    m = GeminiModel('gemini-1.5-flash', http_client=gemini_client)
    agent = Agent(m)
    result = await agent.run('Hello')
    assert result.data == 'foo'


async def test_stream_text(get_gemini_client: GetGeminiClient):
    responses = [
        gemini_response(_content_model_response(ModelResponse.from_text('Hello '))),
        gemini_response(_content_model_response(ModelResponse.from_text('world'))),
    ]
    json_data = _gemini_streamed_response_ta.dump_json(responses, by_alias=True)
    stream = AsyncByteStreamList([json_data[:100], json_data[100:200], json_data[200:]])
    gemini_client = get_gemini_client(stream)
    m = GeminiModel('gemini-1.5-flash', http_client=gemini_client)
    agent = Agent(m)

    async with agent.run_stream('Hello') as result:
        chunks = [chunk async for chunk in result.stream(debounce_by=None)]
        assert chunks == snapshot(['Hello ', 'Hello world'])
    assert result.usage() == snapshot(Usage(requests=1, request_tokens=2, response_tokens=4, total_tokens=6))

    async with agent.run_stream('Hello') as result:
        chunks = [chunk async for chunk in result.stream_text(delta=True, debounce_by=None)]
        assert chunks == snapshot(['', 'Hello ', 'world'])
    assert result.usage() == snapshot(Usage(requests=1, request_tokens=2, response_tokens=4, total_tokens=6))


async def test_stream_text_no_data(get_gemini_client: GetGeminiClient):
    responses = [_GeminiResponse(candidates=[], usage_metadata=example_usage())]
    json_data = _gemini_streamed_response_ta.dump_json(responses, by_alias=True)
    stream = AsyncByteStreamList([json_data[:100], json_data[100:200], json_data[200:]])
    gemini_client = get_gemini_client(stream)
    m = GeminiModel('gemini-1.5-flash', http_client=gemini_client)
    agent = Agent(m)
    with pytest.raises(UnexpectedModelBehavior, match='Streamed response ended without con'):
        async with agent.run_stream('Hello'):
            pass


async def test_stream_structured(get_gemini_client: GetGeminiClient):
    responses = [
        gemini_response(
            _content_model_response(
                ModelResponse(parts=[ToolCallPart.from_raw_args('final_result', {'response': [1, 2]})])
            ),
        ),
    ]
    json_data = _gemini_streamed_response_ta.dump_json(responses, by_alias=True)
    stream = AsyncByteStreamList([json_data[:100], json_data[100:200], json_data[200:]])
    gemini_client = get_gemini_client(stream)
    model = GeminiModel('gemini-1.5-flash', http_client=gemini_client)
    agent = Agent(model, result_type=tuple[int, int])

    async with agent.run_stream('Hello') as result:
        chunks = [chunk async for chunk in result.stream(debounce_by=None)]
        assert chunks == snapshot([(1, 2), (1, 2), (1, 2)])
    assert result.usage() == snapshot(Usage(requests=1, request_tokens=1, response_tokens=2, total_tokens=3))


async def test_stream_structured_tool_calls(get_gemini_client: GetGeminiClient):
    first_responses = [
        gemini_response(
            _content_model_response(ModelResponse(parts=[ToolCallPart.from_raw_args('foo', {'x': 'a'})])),
        ),
        gemini_response(
            _content_model_response(ModelResponse(parts=[ToolCallPart.from_raw_args('bar', {'y': 'b'})])),
        ),
    ]
    d1 = _gemini_streamed_response_ta.dump_json(first_responses, by_alias=True)
    first_stream = AsyncByteStreamList([d1[:100], d1[100:200], d1[200:300], d1[300:]])

    second_responses = [
        gemini_response(
            _content_model_response(
                ModelResponse(parts=[ToolCallPart.from_raw_args('final_result', {'response': [1, 2]})])
            ),
        ),
    ]
    d2 = _gemini_streamed_response_ta.dump_json(second_responses, by_alias=True)
    second_stream = AsyncByteStreamList([d2[:100], d2[100:]])

    gemini_client = get_gemini_client([first_stream, second_stream])
    model = GeminiModel('gemini-1.5-flash', http_client=gemini_client)
    agent = Agent(model, result_type=tuple[int, int])
    tool_calls: list[str] = []

    @agent.tool_plain
    async def foo(x: str) -> str:
        tool_calls.append(f'foo({x=!r})')
        return x

    @agent.tool_plain
    async def bar(y: str) -> str:
        tool_calls.append(f'bar({y=!r})')
        return y

    async with agent.run_stream('Hello') as result:
        response = await result.get_data()
        assert response == snapshot((1, 2))
    assert result.usage() == snapshot(Usage(requests=2, request_tokens=3, response_tokens=6, total_tokens=9))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='foo', args=ArgsDict(args_dict={'x': 'a'})),
                    ToolCallPart(tool_name='bar', args=ArgsDict(args_dict={'y': 'b'})),
                ],
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name='foo', content='a', timestamp=IsNow(tz=timezone.utc)),
                    ToolReturnPart(tool_name='bar', content='b', timestamp=IsNow(tz=timezone.utc)),
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args=ArgsDict(args_dict={'response': [1, 2]}),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result', content='Final result processed.', timestamp=IsNow(tz=timezone.utc)
                    )
                ]
            ),
        ]
    )
    assert tool_calls == snapshot(["foo(x='a')", "bar(y='b')"])


async def test_stream_text_heterogeneous(get_gemini_client: GetGeminiClient):
    responses = [
        gemini_response(_content_model_response(ModelResponse.from_text('Hello '))),
        gemini_response(
            _GeminiContent(
                role='model',
                parts=[
                    _GeminiTextPart(text='foo'),
                    _function_call_part_from_call(
                        ToolCallPart(
                            tool_name='get_location',
                            args=ArgsDict(args_dict={'loc_name': 'San Fransisco'}),
                        )
                    ),
                ],
            )
        ),
    ]
    json_data = _gemini_streamed_response_ta.dump_json(responses, by_alias=True)
    stream = AsyncByteStreamList([json_data[:100], json_data[100:200], json_data[200:]])
    gemini_client = get_gemini_client(stream)
    m = GeminiModel('gemini-1.5-flash', http_client=gemini_client)
    agent = Agent(m)

    msg = 'Streamed response with unexpected content, expected all parts to be text'
    async with agent.run_stream('Hello') as result:
        with pytest.raises(UnexpectedModelBehavior, match=msg):
            await result.get_data()
