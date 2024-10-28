from __future__ import annotations as _annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

import httpx
import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel
from typing_extensions import TypeAlias

from pydantic_ai import Agent, AgentError, ModelRetry, UnexpectedModelBehaviour, UserError
from pydantic_ai._utils import ObjectJsonSchema
from pydantic_ai.messages import (
    ArgsObject,
    LLMResponse,
    LLMToolCalls,
    RetryPrompt,
    SystemPrompt,
    ToolCall,
    ToolReturn,
    UserPrompt,
)
from pydantic_ai.models.gemini import (
    GeminiModel,
    _gemini_response_ta,  # pyright: ignore[reportPrivateUsage]
    _GeminiCandidates,  # pyright: ignore[reportPrivateUsage]
    _GeminiContent,  # pyright: ignore[reportPrivateUsage]
    _GeminiFunction,  # pyright: ignore[reportPrivateUsage]
    _GeminiFunctionCallingConfig,  # pyright: ignore[reportPrivateUsage]
    _GeminiFunctionCallPart,  # pyright: ignore[reportPrivateUsage]
    _GeminiResponse,  # pyright: ignore[reportPrivateUsage]
    _GeminiTextPart,  # pyright: ignore[reportPrivateUsage]
    _GeminiToolConfig,  # pyright: ignore[reportPrivateUsage]
    _GeminiTools,  # pyright: ignore[reportPrivateUsage]
    _GeminiUsageMetaData,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai.shared import Cost
from tests.conftest import ClientWithHandler, IsNow, TestEnv

pytestmark = pytest.mark.anyio


def test_api_key_arg(env: TestEnv):
    env.set('GEMINI_API_KEY', 'via-env-var')
    m = GeminiModel('gemini-1.5-flash', api_key='via-arg')
    assert m.api_key == 'via-arg'


def test_api_key_env_var(env: TestEnv):
    env.set('GEMINI_API_KEY', 'via-env-var')
    m = GeminiModel('gemini-1.5-flash')
    assert m.api_key == 'via-env-var'


def test_api_key_not_set(env: TestEnv):
    with pytest.raises(UserError, match='API key must be provided or set in the GEMINI_API_KEY environment variable'):
        GeminiModel('gemini-1.5-flash')


def test_api_key_empty(env: TestEnv):
    env.set('GEMINI_API_KEY', '')
    with pytest.raises(UserError, match='API key must be provided or set in the GEMINI_API_KEY environment variable'):
        GeminiModel('gemini-1.5-flash')


def test_agent_model_simple():
    m = GeminiModel('gemini-1.5-flash', api_key='via-arg')
    agent_model = m.agent_model({}, True, None)
    assert isinstance(agent_model.http_client, httpx.AsyncClient)
    assert agent_model.model_name == 'gemini-1.5-flash'
    assert agent_model.api_key == 'via-arg'
    assert agent_model.tools is None
    assert agent_model.tool_config is None


@dataclass
class TestToolDefinition:
    __test__ = False
    name: str
    description: str
    json_schema: ObjectJsonSchema
    outer_typed_dict_key: str | None = None


def test_agent_model_tools():
    m = GeminiModel('gemini-1.5-flash', api_key='via-arg')
    retrievers = {
        'foo': TestToolDefinition(
            'foo',
            'This is foo',
            {'type': 'object', 'title': 'Foo', 'properties': {'bar': {'type': 'number', 'title': 'Bar'}}},
        ),
        'apple': TestToolDefinition(
            'apple',
            'This is apple',
            {
                'type': 'object',
                'properties': {
                    'banana': {'type': 'array', 'title': 'Banana', 'items': {'type': 'number', 'title': 'Bar'}}
                },
            },
        ),
    }
    result_tool = TestToolDefinition(
        'result',
        'This is the tool for the final Result',
        {'type': 'object', 'title': 'Result', 'properties': {'spam': {'type': 'number'}}, 'required': ['spam']},
    )
    agent_model = m.agent_model(retrievers, True, [result_tool])
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


def test_require_response_tool():
    m = GeminiModel('gemini-1.5-flash', api_key='via-arg')
    result_tool = TestToolDefinition(
        'result',
        'This is the tool for the final Result',
        {'type': 'object', 'title': 'Result', 'properties': {'spam': {'type': 'number'}}},
    )
    agent_model = m.agent_model({}, False, [result_tool])
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


def test_json_def_replaced():
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
    result_tool = TestToolDefinition(
        'result',
        'This is the tool for the final Result',
        json_schema,
    )
    agent_model = m.agent_model({}, True, [result_tool])
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


def test_json_def_replaced_any_of():
    class Location(BaseModel):
        lat: float
        lng: float

    class Locations(BaseModel):
        op_location: Location | None = None

    json_schema = Locations.model_json_schema()

    m = GeminiModel('gemini-1.5-flash', api_key='via-arg')
    result_tool = TestToolDefinition(
        'result',
        'This is the tool for the final Result',
        json_schema,
    )
    agent_model = m.agent_model({}, True, [result_tool])
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


def test_json_def_recursive():
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
    result_tool = TestToolDefinition(
        'result',
        'This is the tool for the final Result',
        json_schema,
    )
    with pytest.raises(UserError, match=r'Recursive `\$ref`s in JSON Schema are not supported by Gemini'):
        m.agent_model({}, True, [result_tool])


@pytest.fixture
async def get_gemini_client(client_with_handler: ClientWithHandler, env: TestEnv):
    env.set('GEMINI_API_KEY', 'via-env-var')

    def create_client(response_data: _GeminiResponse | list[_GeminiResponse]) -> httpx.AsyncClient:
        index = 0

        def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal index

            if isinstance(response_data, list):
                r = response_data[index]
            else:
                r = response_data
            index += 1

            content = _gemini_response_ta.dump_json(r, by_alias=True)
            return httpx.Response(200, content=content, headers={'Content-Type': 'application/json'})

        return client_with_handler(handler)

    return create_client


GetGeminiClient: TypeAlias = 'Callable[[_GeminiResponse | list[_GeminiResponse]], httpx.AsyncClient]'


def gemini_response(content: _GeminiContent) -> _GeminiResponse:
    return _GeminiResponse(
        candidates=[_GeminiCandidates(content=content, finish_reason='STOP', index=0, safety_ratings=[])],
        usage_metadata=_GeminiUsageMetaData(1, 2, 3),
    )


async def test_request_simple_success(get_gemini_client: GetGeminiClient):
    response = gemini_response(_GeminiContent.model_text('Hello world'))
    gemini_client = get_gemini_client(response)
    m = GeminiModel('gemini-1.5-flash', http_client=gemini_client)
    agent = Agent(m, deps=None)

    result = await agent.run('Hello')
    assert result.response == 'Hello world'
    assert result.message_history == snapshot(
        [
            UserPrompt(content='Hello', timestamp=IsNow()),
            LLMResponse(content='Hello world', timestamp=IsNow()),
        ]
    )
    assert result.cost == snapshot(Cost(request_tokens=1, response_tokens=2, total_tokens=3))


async def test_request_structured_response(get_gemini_client: GetGeminiClient):
    response = gemini_response(
        _GeminiContent.function_call(
            LLMToolCalls(calls=[ToolCall.from_object('final_result', {'response': [1, 2, 123]})])
        )
    )
    gemini_client = get_gemini_client(response)
    m = GeminiModel('gemini-1.5-flash', http_client=gemini_client)
    agent = Agent(m, deps=None, result_type=list[int])

    result = await agent.run('Hello')
    assert result.response == [1, 2, 123]
    assert result.message_history == snapshot(
        [
            UserPrompt(content='Hello', timestamp=IsNow()),
            LLMToolCalls(
                calls=[
                    ToolCall(
                        tool_name='final_result',
                        args=ArgsObject(args_object={'response': [1, 2, 123]}),
                    )
                ],
                timestamp=IsNow(),
            ),
        ]
    )


async def test_request_tool_call(get_gemini_client: GetGeminiClient):
    responses = [
        gemini_response(
            _GeminiContent.function_call(
                LLMToolCalls(calls=[ToolCall.from_object('get_location', {'loc_name': 'San Fransisco'})])
            )
        ),
        gemini_response(
            _GeminiContent.function_call(
                LLMToolCalls(calls=[ToolCall.from_object('get_location', {'loc_name': 'London'})])
            )
        ),
        gemini_response(_GeminiContent.model_text('final response')),
    ]
    gemini_client = get_gemini_client(responses)
    m = GeminiModel('gemini-1.5-flash', http_client=gemini_client)
    agent = Agent(m, deps=None, system_prompt='this is the system prompt')

    @agent.retriever_plain
    async def get_location(loc_name: str) -> str:
        if loc_name == 'London':
            return json.dumps({'lat': 51, 'lng': 0})
        else:
            raise ModelRetry('Wrong location, please try again')

    result = await agent.run('Hello')
    assert result.response == 'final response'
    assert result.message_history == snapshot(
        [
            SystemPrompt(content='this is the system prompt'),
            UserPrompt(content='Hello', timestamp=IsNow()),
            LLMToolCalls(
                calls=[
                    ToolCall(
                        tool_name='get_location',
                        args=ArgsObject(args_object={'loc_name': 'San Fransisco'}),
                    )
                ],
                timestamp=IsNow(),
            ),
            RetryPrompt(tool_name='get_location', content='Wrong location, please try again', timestamp=IsNow()),
            LLMToolCalls(
                calls=[
                    ToolCall(
                        tool_name='get_location',
                        args=ArgsObject(args_object={'loc_name': 'London'}),
                    )
                ],
                timestamp=IsNow(),
            ),
            ToolReturn(tool_name='get_location', content='{"lat": 51, "lng": 0}', timestamp=IsNow()),
            LLMResponse(content='final response', timestamp=IsNow()),
        ]
    )
    assert result.cost == snapshot(Cost(request_tokens=3, response_tokens=6, total_tokens=9))


async def test_unexpected_response(client_with_handler: ClientWithHandler, env: TestEnv):
    env.set('GEMINI_API_KEY', 'via-env-var')

    def handler(_: httpx.Request):
        return httpx.Response(401, content='invalid request')

    gemini_client = client_with_handler(handler)
    m = GeminiModel('gemini-1.5-flash', http_client=gemini_client)
    agent = Agent(m, deps=None, system_prompt='this is the system prompt')

    with pytest.raises(AgentError, match='Error while running model gemini-1.5-flash') as exc_info:
        await agent.run('Hello')

    assert str(exc_info.value) == snapshot(
        'Error while running model gemini-1.5-flash\n'
        '  caused by unexpected model behavior: Unexpected response from gemini 401'
    )

    cause = exc_info.value.cause()
    assert isinstance(cause, UnexpectedModelBehaviour)
    assert str(cause) == snapshot('Unexpected response from gemini 401, body:\ninvalid request')


async def test_heterogeneous_responses(get_gemini_client: GetGeminiClient):
    response = gemini_response(
        _GeminiContent(
            role='model',
            parts=[
                _GeminiTextPart(text='foo'),
                _GeminiFunctionCallPart.from_call(
                    ToolCall(
                        tool_name='get_location',
                        args=ArgsObject(args_object={'loc_name': 'San Fransisco'}),
                    )
                ),
            ],
        )
    )
    gemini_client = get_gemini_client(response)
    m = GeminiModel('gemini-1.5-flash', http_client=gemini_client)
    agent = Agent(m, deps=None)
    with pytest.raises(AgentError, match='Error while running model gemini-1.5-flash') as exc_info:
        await agent.run('Hello')

    cause = exc_info.value.cause()
    assert isinstance(cause, UnexpectedModelBehaviour)
    assert str(cause) == snapshot(
        'Unexpected response from Gemini, expected all parts to be function calls or text, got: '
        "[_GeminiTextPart(text='foo'), _GeminiFunctionCallPart(function_call="
        "_GeminiFunctionCall(name='get_location', args={'loc_name': 'San Fransisco'}))]"
    )
