# pyright: reportPrivateUsage=false
# pyright: reportDeprecated=false
from __future__ import annotations as _annotations

import datetime
import json
import re
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from datetime import timezone
from enum import IntEnum
from typing import Annotated, Literal, TypeAlias

import httpx
import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel, Field

from pydantic_ai import (
    Agent,
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    ModelRetry,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UnexpectedModelBehavior,
    UserError,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.gemini import (
    GeminiModel,
    GeminiModelSettings,
    _content_model_response,
    _gemini_response_ta,
    _gemini_streamed_response_ta,
    _GeminiCandidates,
    _GeminiContent,
    _GeminiFunctionCall,
    _GeminiFunctionCallingConfig,
    _GeminiFunctionCallPart,
    _GeminiModalityTokenCount,
    _GeminiResponse,
    _GeminiSafetyRating,
    _GeminiTextPart,
    _GeminiThoughtPart,
    _GeminiToolConfig,
    _GeminiUsageMetaData,
    _metadata_as_usage,
)
from pydantic_ai.output import NativeOutput, PromptedOutput, TextOutput, ToolOutput
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from pydantic_ai.result import RunUsage
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

from ..conftest import ClientWithHandler, IsDatetime, IsNow, IsStr, TestEnv

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.filterwarnings('ignore:Use `GoogleModel` instead.:DeprecationWarning'),
    pytest.mark.filterwarnings('ignore:`GoogleGLAProvider` is deprecated.:DeprecationWarning'),
]


async def test_model_simple(allow_model_requests: None):
    m = GeminiModel('gemini-1.5-flash', provider=GoogleGLAProvider(api_key='via-arg'))
    assert isinstance(m.client, httpx.AsyncClient)
    assert m.model_name == 'gemini-1.5-flash'
    assert 'x-goog-api-key' in m.client.headers

    mrp = ModelRequestParameters(
        function_tools=[], allow_text_output=True, output_tools=[], output_mode='text', output_object=None
    )
    mrp = m.customize_request_parameters(mrp)
    tools = m._get_tools(mrp)
    tool_config = m._get_tool_config(mrp, tools)
    assert tools is None
    assert tool_config is None


async def test_model_tools(allow_model_requests: None):
    m = GeminiModel('gemini-1.5-flash', provider=GoogleGLAProvider(api_key='via-arg'))
    tools = [
        ToolDefinition(
            name='foo',
            description='This is foo',
            parameters_json_schema={
                'type': 'object',
                'title': 'Foo',
                'properties': {'bar': {'type': 'number', 'title': 'Bar'}},
            },
        ),
        ToolDefinition(
            name='apple',
            description='This is apple',
            parameters_json_schema={
                'type': 'object',
                'properties': {
                    'banana': {'type': 'array', 'title': 'Banana', 'items': {'type': 'number', 'title': 'Bar'}}
                },
            },
        ),
    ]
    output_tool = ToolDefinition(
        name='result',
        description='This is the tool for the final Result',
        parameters_json_schema={
            'type': 'object',
            'title': 'Result',
            'properties': {'spam': {'type': 'number'}},
            'required': ['spam'],
        },
    )

    mrp = ModelRequestParameters(
        function_tools=tools,
        allow_text_output=True,
        output_tools=[output_tool],
        output_mode='text',
        output_object=None,
    )
    mrp = m.customize_request_parameters(mrp)
    tools = m._get_tools(mrp)
    tool_config = m._get_tool_config(mrp, tools)
    assert tools == snapshot(
        {
            'function_declarations': [
                {
                    'name': 'foo',
                    'description': 'This is foo',
                    'parameters_json_schema': {'type': 'object', 'properties': {'bar': {'type': 'number'}}},
                },
                {
                    'name': 'apple',
                    'description': 'This is apple',
                    'parameters_json_schema': {
                        'type': 'object',
                        'properties': {'banana': {'type': 'array', 'items': {'type': 'number'}}},
                    },
                },
                {
                    'name': 'result',
                    'description': 'This is the tool for the final Result',
                    'parameters_json_schema': {
                        'type': 'object',
                        'properties': {'spam': {'type': 'number'}},
                        'required': ['spam'],
                    },
                },
            ]
        }
    )
    assert tool_config is None


async def test_require_response_tool(allow_model_requests: None):
    m = GeminiModel('gemini-1.5-flash', provider=GoogleGLAProvider(api_key='via-arg'))
    output_tool = ToolDefinition(
        name='result',
        description='This is the tool for the final Result',
        parameters_json_schema={'type': 'object', 'title': 'Result', 'properties': {'spam': {'type': 'number'}}},
    )
    mrp = ModelRequestParameters(
        function_tools=[],
        allow_text_output=False,
        output_tools=[output_tool],
        output_mode='tool',
        output_object=None,
    )
    mrp = m.customize_request_parameters(mrp)
    tools = m._get_tools(mrp)
    tool_config = m._get_tool_config(mrp, tools)
    assert tools == snapshot(
        {
            'function_declarations': [
                {
                    'name': 'result',
                    'description': 'This is the tool for the final Result',
                    'parameters_json_schema': {'type': 'object', 'properties': {'spam': {'type': 'number'}}},
                }
            ]
        }
    )
    assert tool_config == snapshot(
        _GeminiToolConfig(
            function_calling_config=_GeminiFunctionCallingConfig(mode='ANY', allowed_function_names=['result'])
        )
    )


async def test_json_def_replaced(allow_model_requests: None):
    class Axis(BaseModel):
        label: str = Field(default='<unlabeled axis>', description='The label of the axis')

    class Chart(BaseModel):
        x_axis: Axis
        y_axis: Axis

    class Location(BaseModel):
        lat: float
        lng: float = 1.1
        chart: Chart

    class Locations(BaseModel):
        locations: list[Location]

    json_schema = Locations.model_json_schema()
    assert json_schema == snapshot(
        {
            '$defs': {
                'Axis': {
                    'properties': {
                        'label': {
                            'default': '<unlabeled axis>',
                            'description': 'The label of the axis',
                            'title': 'Label',
                            'type': 'string',
                        }
                    },
                    'title': 'Axis',
                    'type': 'object',
                },
                'Chart': {
                    'properties': {'x_axis': {'$ref': '#/$defs/Axis'}, 'y_axis': {'$ref': '#/$defs/Axis'}},
                    'required': ['x_axis', 'y_axis'],
                    'title': 'Chart',
                    'type': 'object',
                },
                'Location': {
                    'properties': {
                        'lat': {'title': 'Lat', 'type': 'number'},
                        'lng': {'default': 1.1, 'title': 'Lng', 'type': 'number'},
                        'chart': {'$ref': '#/$defs/Chart'},
                    },
                    'required': ['lat', 'chart'],
                    'title': 'Location',
                    'type': 'object',
                },
            },
            'properties': {'locations': {'items': {'$ref': '#/$defs/Location'}, 'title': 'Locations', 'type': 'array'}},
            'required': ['locations'],
            'title': 'Locations',
            'type': 'object',
        }
    )

    m = GeminiModel('gemini-1.5-flash', provider=GoogleGLAProvider(api_key='via-arg'))
    output_tool = ToolDefinition(
        name='result',
        description='This is the tool for the final Result',
        parameters_json_schema=json_schema,
    )
    mrp = ModelRequestParameters(
        function_tools=[],
        allow_text_output=True,
        output_tools=[output_tool],
        output_mode='text',
        output_object=None,
    )
    mrp = m.customize_request_parameters(mrp)
    assert m._get_tools(mrp) == snapshot(
        {
            'function_declarations': [
                {
                    'name': 'result',
                    'description': 'This is the tool for the final Result',
                    'parameters_json_schema': {
                        'properties': {'locations': {'items': {'$ref': '#/$defs/Location'}, 'type': 'array'}},
                        'required': ['locations'],
                        'type': 'object',
                        '$defs': {
                            'Axis': {
                                'properties': {
                                    'label': {
                                        'default': '<unlabeled axis>',
                                        'description': 'The label of the axis',
                                        'type': 'string',
                                    }
                                },
                                'type': 'object',
                            },
                            'Chart': {
                                'properties': {'x_axis': {'$ref': '#/$defs/Axis'}, 'y_axis': {'$ref': '#/$defs/Axis'}},
                                'required': ['x_axis', 'y_axis'],
                                'type': 'object',
                            },
                            'Location': {
                                'properties': {
                                    'lat': {'type': 'number'},
                                    'lng': {'default': 1.1, 'type': 'number'},
                                    'chart': {'$ref': '#/$defs/Chart'},
                                },
                                'required': ['lat', 'chart'],
                                'type': 'object',
                            },
                        },
                    },
                }
            ]
        }
    )


async def test_json_def_enum(allow_model_requests: None):
    class ProgressEnum(IntEnum):
        DONE = 100
        ALMOST_DONE = 80
        IN_PROGRESS = 60
        BARELY_STARTED = 40
        NOT_STARTED = 20

    class QueryDetails(BaseModel):
        progress: list[ProgressEnum] | None = None

    json_schema = QueryDetails.model_json_schema()
    assert json_schema == snapshot(
        {
            '$defs': {'ProgressEnum': {'enum': [100, 80, 60, 40, 20], 'title': 'ProgressEnum', 'type': 'integer'}},
            'properties': {
                'progress': {
                    'anyOf': [{'items': {'$ref': '#/$defs/ProgressEnum'}, 'type': 'array'}, {'type': 'null'}],
                    'default': None,
                    'title': 'Progress',
                }
            },
            'title': 'QueryDetails',
            'type': 'object',
        }
    )
    m = GeminiModel('gemini-1.5-flash', provider=GoogleGLAProvider(api_key='via-arg'))
    output_tool = ToolDefinition(
        name='result',
        description='This is the tool for the final Result',
        parameters_json_schema=json_schema,
    )
    mrp = ModelRequestParameters(
        function_tools=[],
        output_mode='text',
        allow_text_output=True,
        output_tools=[output_tool],
        output_object=None,
    )
    mrp = m.customize_request_parameters(mrp)

    # This tests that the enum values are properly converted to strings for Gemini
    assert m._get_tools(mrp) == snapshot(
        {
            'function_declarations': [
                {
                    'name': 'result',
                    'description': 'This is the tool for the final Result',
                    'parameters_json_schema': {
                        'properties': {
                            'progress': {
                                'default': None,
                                'anyOf': [
                                    {'items': {'$ref': '#/$defs/ProgressEnum'}, 'type': 'array'},
                                    {'type': 'null'},
                                ],
                            }
                        },
                        'type': 'object',
                        '$defs': {'ProgressEnum': {'enum': [100, 80, 60, 40, 20], 'type': 'integer'}},
                    },
                }
            ]
        }
    )


async def test_json_def_replaced_any_of(allow_model_requests: None):
    class Location(BaseModel):
        lat: float
        lng: float

    class Locations(BaseModel):
        op_location: Location | None = None

    json_schema = Locations.model_json_schema()

    m = GeminiModel('gemini-1.5-flash', provider=GoogleGLAProvider(api_key='via-arg'))
    output_tool = ToolDefinition(
        name='result',
        description='This is the tool for the final Result',
        parameters_json_schema=json_schema,
    )
    mrp = ModelRequestParameters(
        function_tools=[],
        allow_text_output=True,
        output_tools=[output_tool],
        output_mode='text',
        output_object=None,
    )
    mrp = m.customize_request_parameters(mrp)
    assert m._get_tools(mrp) == snapshot(
        {
            'function_declarations': [
                {
                    'name': 'result',
                    'description': 'This is the tool for the final Result',
                    'parameters_json_schema': {
                        'properties': {
                            'op_location': {'default': None, 'anyOf': [{'$ref': '#/$defs/Location'}, {'type': 'null'}]}
                        },
                        'type': 'object',
                        '$defs': {
                            'Location': {
                                'properties': {'lat': {'type': 'number'}, 'lng': {'type': 'number'}},
                                'required': ['lat', 'lng'],
                                'type': 'object',
                            }
                        },
                    },
                }
            ]
        }
    )


async def test_json_def_date(allow_model_requests: None):
    class FormattedStringFields(BaseModel):
        d: datetime.date
        dt: datetime.datetime
        t: datetime.time = Field(description='')
        td: datetime.timedelta = Field(description='my timedelta')

    json_schema = FormattedStringFields.model_json_schema()
    assert json_schema == snapshot(
        {
            'properties': {
                'd': {'format': 'date', 'title': 'D', 'type': 'string'},
                'dt': {'format': 'date-time', 'title': 'Dt', 'type': 'string'},
                't': {'format': 'time', 'title': 'T', 'type': 'string', 'description': ''},
                'td': {'format': 'duration', 'title': 'Td', 'type': 'string', 'description': 'my timedelta'},
            },
            'required': ['d', 'dt', 't', 'td'],
            'title': 'FormattedStringFields',
            'type': 'object',
        }
    )

    m = GeminiModel('gemini-1.5-flash', provider=GoogleGLAProvider(api_key='via-arg'))
    output_tool = ToolDefinition(
        name='result',
        description='This is the tool for the final Result',
        parameters_json_schema=json_schema,
    )
    mrp = ModelRequestParameters(
        function_tools=[],
        allow_text_output=True,
        output_tools=[output_tool],
        output_mode='text',
        output_object=None,
    )
    mrp = m.customize_request_parameters(mrp)
    assert m._get_tools(mrp) == snapshot(
        {
            'function_declarations': [
                {
                    'name': 'result',
                    'description': 'This is the tool for the final Result',
                    'parameters_json_schema': {
                        'properties': {
                            'd': {'type': 'string', 'description': 'Format: date'},
                            'dt': {'type': 'string', 'description': 'Format: date-time'},
                            't': {'description': 'Format: time', 'type': 'string'},
                            'td': {'description': 'my timedelta (format: duration)', 'type': 'string'},
                        },
                        'required': ['d', 'dt', 't', 'td'],
                        'type': 'object',
                    },
                }
            ]
        }
    )


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
    if finish_reason:  # pragma: no branch
        candidate['finish_reason'] = finish_reason
    return _GeminiResponse(candidates=[candidate], usage_metadata=example_usage(), model_version='gemini-1.5-flash-123')


def example_usage() -> _GeminiUsageMetaData:
    return _GeminiUsageMetaData(prompt_token_count=1, candidates_token_count=2, total_token_count=3)


async def test_text_success(get_gemini_client: GetGeminiClient):
    response = gemini_response(_content_model_response(ModelResponse(parts=[TextPart('Hello world')])))
    gemini_client = get_gemini_client(response)
    m = GeminiModel('gemini-1.5-flash', provider=GoogleGLAProvider(http_client=gemini_client))
    agent = Agent(m)

    result = await agent.run('Hello')
    assert result.output == 'Hello world'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Hello world')],
                usage=RequestUsage(input_tokens=1, output_tokens=2),
                model_name='gemini-1.5-flash-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_url='https://generativelanguage.googleapis.com/v1beta/models/',
                provider_details={'finish_reason': 'STOP'},
                run_id=IsStr(),
            ),
        ]
    )
    assert result.usage() == snapshot(RunUsage(requests=1, input_tokens=1, output_tokens=2))

    result = await agent.run('Hello', message_history=result.new_messages())
    assert result.output == 'Hello world'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Hello world')],
                usage=RequestUsage(input_tokens=1, output_tokens=2),
                model_name='gemini-1.5-flash-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_url='https://generativelanguage.googleapis.com/v1beta/models/',
                provider_details={'finish_reason': 'STOP'},
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Hello world')],
                usage=RequestUsage(input_tokens=1, output_tokens=2),
                model_name='gemini-1.5-flash-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_url='https://generativelanguage.googleapis.com/v1beta/models/',
                provider_details={'finish_reason': 'STOP'},
                run_id=IsStr(),
            ),
        ]
    )


async def test_request_structured_response(get_gemini_client: GetGeminiClient):
    response = gemini_response(
        _content_model_response(ModelResponse(parts=[ToolCallPart('final_result', {'response': [1, 2, 123]})]))
    )
    gemini_client = get_gemini_client(response)
    m = GeminiModel('gemini-1.5-flash', provider=GoogleGLAProvider(http_client=gemini_client))
    agent = Agent(m, output_type=list[int])

    result = await agent.run('Hello')
    assert result.output == [1, 2, 123]
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='final_result', args={'response': [1, 2, 123]}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=1, output_tokens=2),
                model_name='gemini-1.5-flash-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_url='https://generativelanguage.googleapis.com/v1beta/models/',
                provider_details={'finish_reason': 'STOP'},
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        timestamp=IsNow(tz=timezone.utc),
                        tool_call_id=IsStr(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )


async def test_request_tool_call(get_gemini_client: GetGeminiClient):
    responses = [
        gemini_response(
            _content_model_response(ModelResponse(parts=[ToolCallPart('get_location', {'loc_name': 'San Fransisco'})]))
        ),
        gemini_response(
            _content_model_response(
                ModelResponse(
                    parts=[
                        ToolCallPart('get_location', {'loc_name': 'London'}),
                        ToolCallPart('get_location', {'loc_name': 'New York'}),
                    ],
                )
            )
        ),
        gemini_response(_content_model_response(ModelResponse(parts=[TextPart('final response')]))),
    ]
    gemini_client = get_gemini_client(responses)
    m = GeminiModel('gemini-1.5-flash', provider=GoogleGLAProvider(http_client=gemini_client))
    agent = Agent(m, system_prompt='this is the system prompt')

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        if loc_name == 'London':
            return json.dumps({'lat': 51, 'lng': 0})
        elif loc_name == 'New York':
            return json.dumps({'lat': 41, 'lng': -74})
        else:
            raise ModelRetry('Wrong location, please try again')

    result = await agent.run('Hello')
    assert result.output == 'final response'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='this is the system prompt', timestamp=IsNow(tz=timezone.utc)),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='get_location', args={'loc_name': 'San Fransisco'}, tool_call_id=IsStr())
                ],
                usage=RequestUsage(input_tokens=1, output_tokens=2),
                model_name='gemini-1.5-flash-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_url='https://generativelanguage.googleapis.com/v1beta/models/',
                provider_details={'finish_reason': 'STOP'},
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Wrong location, please try again',
                        tool_name='get_location',
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='get_location', args={'loc_name': 'London'}, tool_call_id=IsStr()),
                    ToolCallPart(tool_name='get_location', args={'loc_name': 'New York'}, tool_call_id=IsStr()),
                ],
                usage=RequestUsage(input_tokens=1, output_tokens=2),
                model_name='gemini-1.5-flash-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_url='https://generativelanguage.googleapis.com/v1beta/models/',
                provider_details={'finish_reason': 'STOP'},
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_location',
                        content='{"lat": 51, "lng": 0}',
                        timestamp=IsNow(tz=timezone.utc),
                        tool_call_id=IsStr(),
                    ),
                    ToolReturnPart(
                        tool_name='get_location',
                        content='{"lat": 41, "lng": -74}',
                        timestamp=IsNow(tz=timezone.utc),
                        tool_call_id=IsStr(),
                    ),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='final response')],
                usage=RequestUsage(input_tokens=1, output_tokens=2),
                model_name='gemini-1.5-flash-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_url='https://generativelanguage.googleapis.com/v1beta/models/',
                provider_details={'finish_reason': 'STOP'},
                run_id=IsStr(),
            ),
        ]
    )
    assert result.usage() == snapshot(RunUsage(requests=3, input_tokens=3, output_tokens=6, tool_calls=2))


async def test_unexpected_response(client_with_handler: ClientWithHandler, env: TestEnv, allow_model_requests: None):
    env.set('GEMINI_API_KEY', 'via-env-var')

    def handler(_: httpx.Request):
        return httpx.Response(401, content='invalid request')

    gemini_client = client_with_handler(handler)
    m = GeminiModel('gemini-1.5-flash', provider=GoogleGLAProvider(http_client=gemini_client))
    agent = Agent(m, system_prompt='this is the system prompt')

    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('Hello')

    assert str(exc_info.value) == snapshot('status_code: 401, model_name: gemini-1.5-flash, body: invalid request')


async def test_stream_text(get_gemini_client: GetGeminiClient):
    responses = [
        gemini_response(_content_model_response(ModelResponse(parts=[TextPart('Hello ')]))),
        gemini_response(_content_model_response(ModelResponse(parts=[TextPart('world')]))),
    ]
    json_data = _gemini_streamed_response_ta.dump_json(responses, by_alias=True)
    stream = AsyncByteStreamList([json_data[:100], json_data[100:200], json_data[200:]])
    gemini_client = get_gemini_client(stream)
    m = GeminiModel('gemini-1.5-flash', provider=GoogleGLAProvider(http_client=gemini_client))
    agent = Agent(m)

    async with agent.run_stream('Hello') as result:
        chunks = [chunk async for chunk in result.stream_output(debounce_by=None)]
        assert chunks == snapshot(['Hello ', 'Hello world', 'Hello world'])
    assert result.usage() == snapshot(RunUsage(requests=1, input_tokens=1, output_tokens=2))

    async with agent.run_stream('Hello') as result:
        chunks = [chunk async for chunk in result.stream_text(delta=True, debounce_by=None)]
        assert chunks == snapshot(['Hello ', 'world'])
    assert result.usage() == snapshot(RunUsage(requests=1, input_tokens=1, output_tokens=2))


async def test_stream_invalid_unicode_text(get_gemini_client: GetGeminiClient):
    # Probably safe to remove this test once https://github.com/pydantic/pydantic-core/issues/1633 is resolved
    responses = [
        gemini_response(_content_model_response(ModelResponse(parts=[TextPart('abc')]))),
        gemini_response(_content_model_response(ModelResponse(parts=[TextPart('€def')]))),
    ]
    json_data = _gemini_streamed_response_ta.dump_json(responses, by_alias=True)

    for i in range(10, 1000):
        parts = [json_data[:i], json_data[i:]]
        try:
            parts[0].decode()
        except UnicodeDecodeError:
            break
    else:  # pragma: no cover
        assert False, 'failed to find a spot in payload that would break unicode parsing'

    with pytest.raises(UnicodeDecodeError):
        # Ensure the first part is _not_ valid unicode
        parts[0].decode()

    stream = AsyncByteStreamList(parts)
    gemini_client = get_gemini_client(stream)
    m = GeminiModel('gemini-1.5-flash', provider=GoogleGLAProvider(http_client=gemini_client))
    agent = Agent(m)

    async with agent.run_stream('Hello') as result:
        chunks = [chunk async for chunk in result.stream_output(debounce_by=None)]
        assert chunks == snapshot(['abc', 'abc€def', 'abc€def'])
    assert result.usage() == snapshot(RunUsage(requests=1, input_tokens=1, output_tokens=2))


async def test_stream_text_no_data(get_gemini_client: GetGeminiClient):
    responses = [_GeminiResponse(candidates=[], usage_metadata=example_usage())]
    json_data = _gemini_streamed_response_ta.dump_json(responses, by_alias=True)
    stream = AsyncByteStreamList([json_data[:100], json_data[100:200], json_data[200:]])
    gemini_client = get_gemini_client(stream)
    m = GeminiModel('gemini-1.5-flash', provider=GoogleGLAProvider(http_client=gemini_client))
    agent = Agent(m)
    with pytest.raises(UnexpectedModelBehavior, match='Streamed response ended without con'):
        async with agent.run_stream('Hello'):
            pass


async def test_stream_structured(get_gemini_client: GetGeminiClient):
    responses = [
        gemini_response(
            _content_model_response(ModelResponse(parts=[ToolCallPart('final_result', {'response': [1, 2]})])),
        ),
    ]
    json_data = _gemini_streamed_response_ta.dump_json(responses, by_alias=True)
    stream = AsyncByteStreamList([json_data[:100], json_data[100:200], json_data[200:]])
    gemini_client = get_gemini_client(stream)
    model = GeminiModel('gemini-1.5-flash', provider=GoogleGLAProvider(http_client=gemini_client))
    agent = Agent(model, output_type=tuple[int, int])

    async with agent.run_stream('Hello') as result:
        chunks = [chunk async for chunk in result.stream_output(debounce_by=None)]
        assert chunks == snapshot([(1, 2), (1, 2)])
    assert result.usage() == snapshot(RunUsage(requests=1, input_tokens=1, output_tokens=2))


async def test_stream_structured_tool_calls(get_gemini_client: GetGeminiClient):
    first_responses = [
        gemini_response(
            _content_model_response(ModelResponse(parts=[ToolCallPart('foo', {'x': 'a'})])),
        ),
        gemini_response(
            _content_model_response(ModelResponse(parts=[ToolCallPart('bar', {'y': 'b'})])),
        ),
    ]
    d1 = _gemini_streamed_response_ta.dump_json(first_responses, by_alias=True)
    first_stream = AsyncByteStreamList([d1[:100], d1[100:200], d1[200:300], d1[300:]])

    second_responses = [
        gemini_response(
            _content_model_response(ModelResponse(parts=[ToolCallPart('final_result', {'response': [1, 2]})])),
        ),
    ]
    d2 = _gemini_streamed_response_ta.dump_json(second_responses, by_alias=True)
    second_stream = AsyncByteStreamList([d2[:100], d2[100:]])

    gemini_client = get_gemini_client([first_stream, second_stream])
    model = GeminiModel('gemini-1.5-flash', provider=GoogleGLAProvider(http_client=gemini_client))
    agent = Agent(model, output_type=tuple[int, int])
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
        response = await result.get_output()
        assert response == snapshot((1, 2))
    assert result.usage() == snapshot(RunUsage(requests=2, input_tokens=2, output_tokens=4, tool_calls=2))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='foo', args={'x': 'a'}, tool_call_id=IsStr()),
                    ToolCallPart(tool_name='bar', args={'y': 'b'}, tool_call_id=IsStr()),
                ],
                usage=RequestUsage(input_tokens=1, output_tokens=2),
                model_name='gemini-1.5-flash',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/v1beta/models/',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='foo', content='a', timestamp=IsNow(tz=timezone.utc), tool_call_id=IsStr()
                    ),
                    ToolReturnPart(
                        tool_name='bar', content='b', timestamp=IsNow(tz=timezone.utc), tool_call_id=IsStr()
                    ),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='final_result', args={'response': [1, 2]}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=1, output_tokens=2),
                model_name='gemini-1.5-flash',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/v1beta/models/',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        timestamp=IsNow(tz=timezone.utc),
                        tool_call_id=IsStr(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )
    assert tool_calls == snapshot(["foo(x='a')", "bar(y='b')"])


async def test_stream_text_heterogeneous(get_gemini_client: GetGeminiClient):
    responses = [
        gemini_response(_content_model_response(ModelResponse(parts=[TextPart('Hello ')]))),
        gemini_response(
            _GeminiContent(
                role='model',
                parts=[
                    _GeminiThoughtPart(thought=True, thought_signature='test-signature-value'),
                    _GeminiTextPart(text='foo'),
                    _GeminiFunctionCallPart(
                        function_call=_GeminiFunctionCall(name='get_location', args={'loc_name': 'San Fransisco'})
                    ),
                ],
            )
        ),
    ]
    json_data = _gemini_streamed_response_ta.dump_json(responses, by_alias=True)
    stream = AsyncByteStreamList([json_data[:100], json_data[100:200], json_data[200:]])
    gemini_client = get_gemini_client(stream)
    m = GeminiModel('gemini-1.5-flash', provider=GoogleGLAProvider(http_client=gemini_client))
    agent = Agent(m)

    @agent.tool_plain()
    def get_location(loc_name: str) -> str:
        return f'Location for {loc_name}'  # pragma: no cover

    async with agent.run_stream('Hello') as result:
        data = await result.get_output()

    assert data == 'Hello foo'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Hello',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(content='Hello foo'),
                    ToolCallPart(
                        tool_name='get_location',
                        args={'loc_name': 'San Fransisco'},
                        tool_call_id=IsStr(),
                    ),
                ],
                usage=RequestUsage(input_tokens=1, output_tokens=2),
                model_name='gemini-1.5-flash',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/v1beta/models/',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_location',
                        content='Tool not executed - a final result was already processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )


async def test_empty_text_ignored():
    content = _content_model_response(
        ModelResponse(parts=[ToolCallPart('final_result', {'response': [1, 2, 123]}), TextPart(content='xxx')])
    )
    # text included
    assert content == snapshot(
        {
            'role': 'model',
            'parts': [
                {
                    'function_call': {'name': 'final_result', 'args': {'response': [1, 2, 123]}},
                    'thought_signature': b'skip_thought_signature_validator',
                },
                {'text': 'xxx'},
            ],
        }
    )

    content = _content_model_response(
        ModelResponse(parts=[ToolCallPart('final_result', {'response': [1, 2, 123]}), TextPart(content='')])
    )
    # text skipped
    assert content == snapshot(
        {
            'role': 'model',
            'parts': [
                {
                    'function_call': {'name': 'final_result', 'args': {'response': [1, 2, 123]}},
                    'thought_signature': b'skip_thought_signature_validator',
                }
            ],
        }
    )


async def test_model_settings(client_with_handler: ClientWithHandler, env: TestEnv, allow_model_requests: None) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        generation_config = json.loads(request.content)['generationConfig']
        assert generation_config == {
            'max_output_tokens': 1,
            'temperature': 0.1,
            'top_p': 0.2,
            'presence_penalty': 0.3,
            'frequency_penalty': 0.4,
        }
        return httpx.Response(
            200,
            content=_gemini_response_ta.dump_json(
                gemini_response(_content_model_response(ModelResponse(parts=[TextPart('world')]))),
                by_alias=True,
            ),
            headers={'Content-Type': 'application/json'},
        )

    gemini_client = client_with_handler(handler)
    m = GeminiModel('gemini-1.5-flash', provider=GoogleGLAProvider(http_client=gemini_client, api_key='mock'))
    agent = Agent(m)

    result = await agent.run(
        'hello',
        model_settings={
            'max_tokens': 1,
            'temperature': 0.1,
            'top_p': 0.2,
            'presence_penalty': 0.3,
            'frequency_penalty': 0.4,
        },
    )
    assert result.output == 'world'


def gemini_no_content_response(
    safety_ratings: list[_GeminiSafetyRating], finish_reason: Literal['SAFETY'] | None = 'SAFETY'
) -> _GeminiResponse:
    candidate = _GeminiCandidates(safety_ratings=safety_ratings)
    if finish_reason:  # pragma: no branch
        candidate['finish_reason'] = finish_reason
    return _GeminiResponse(candidates=[candidate], usage_metadata=example_usage())


async def test_safety_settings_unsafe(
    client_with_handler: ClientWithHandler, env: TestEnv, allow_model_requests: None
) -> None:
    try:

        def handler(request: httpx.Request) -> httpx.Response:
            safety_settings = json.loads(request.content)['safetySettings']
            assert safety_settings == [
                {'category': 'HARM_CATEGORY_CIVIC_INTEGRITY', 'threshold': 'BLOCK_LOW_AND_ABOVE'},
                {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_LOW_AND_ABOVE'},
            ]

            return httpx.Response(
                200,
                content=_gemini_response_ta.dump_json(
                    gemini_no_content_response(
                        finish_reason='SAFETY',
                        safety_ratings=[
                            {'category': 'HARM_CATEGORY_HARASSMENT', 'probability': 'MEDIUM', 'blocked': True}
                        ],
                    ),
                    by_alias=True,
                ),
                headers={'Content-Type': 'application/json'},
            )

        gemini_client = client_with_handler(handler)

        m = GeminiModel('gemini-1.5-flash', provider=GoogleGLAProvider(http_client=gemini_client, api_key='mock'))
        agent = Agent(m)

        await agent.run(
            'a request for something rude',
            model_settings=GeminiModelSettings(
                gemini_safety_settings=[
                    {'category': 'HARM_CATEGORY_CIVIC_INTEGRITY', 'threshold': 'BLOCK_LOW_AND_ABOVE'},
                    {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_LOW_AND_ABOVE'},
                ]
            ),
        )
    except UnexpectedModelBehavior as e:
        assert repr(e) == "UnexpectedModelBehavior('Safety settings triggered')"


async def test_safety_settings_safe(
    client_with_handler: ClientWithHandler, env: TestEnv, allow_model_requests: None
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        safety_settings = json.loads(request.content)['safetySettings']
        assert safety_settings == [
            {'category': 'HARM_CATEGORY_CIVIC_INTEGRITY', 'threshold': 'BLOCK_LOW_AND_ABOVE'},
            {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_LOW_AND_ABOVE'},
        ]

        return httpx.Response(
            200,
            content=_gemini_response_ta.dump_json(
                gemini_response(_content_model_response(ModelResponse(parts=[TextPart('world')]))),
                by_alias=True,
            ),
            headers={'Content-Type': 'application/json'},
        )

    gemini_client = client_with_handler(handler)
    m = GeminiModel('gemini-1.5-flash', provider=GoogleGLAProvider(http_client=gemini_client, api_key='mock'))
    agent = Agent(m)

    result = await agent.run(
        'hello',
        model_settings=GeminiModelSettings(
            gemini_safety_settings=[
                {'category': 'HARM_CATEGORY_CIVIC_INTEGRITY', 'threshold': 'BLOCK_LOW_AND_ABOVE'},
                {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_LOW_AND_ABOVE'},
            ]
        ),
    )
    assert result.output == 'world'


@pytest.mark.vcr()
async def test_image_as_binary_content_tool_response(
    allow_model_requests: None, gemini_api_key: str, image_content: BinaryContent
) -> None:
    m = GeminiModel('gemini-3-pro-preview', provider=GoogleGLAProvider(api_key=gemini_api_key))
    agent = Agent(m)

    @agent.tool_plain
    async def get_image() -> BinaryContent:
        return image_content

    result = await agent.run(['What fruit is in the image you can get from the get_image tool?'])
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=['What fruit is in the image you can get from the get_image tool?'],
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_image', args={}, tool_call_id=IsStr())],
                usage=RequestUsage(
                    input_tokens=33, output_tokens=155, details={'thoughts_tokens': 145, 'text_prompt_tokens': 33}
                ),
                model_name='gemini-3-pro-preview',
                timestamp=IsDatetime(),
                provider_url='https://generativelanguage.googleapis.com/v1beta/models/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_image',
                        content='See file 241a70',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content=[
                            'This is file 241a70:',
                            image_content,
                        ],
                        timestamp=IsDatetime(),
                    ),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(content='6'),
                    TextPart(
                        content='The fruit in the image is a **kiwi** (specifically, a cross-section showing its green flesh, black seeds, and white core).'
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=1166,
                    output_tokens=215,
                    details={'thoughts_tokens': 184, 'image_prompt_tokens': 1088, 'text_prompt_tokens': 78},
                ),
                model_name='gemini-3-pro-preview',
                timestamp=IsDatetime(),
                provider_url='https://generativelanguage.googleapis.com/v1beta/models/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                run_id=IsStr(),
            ),
        ]
    )


@pytest.mark.vcr()
async def test_labels_are_ignored_with_gla_provider(allow_model_requests: None, gemini_api_key: str) -> None:
    m = GeminiModel('gemini-2.0-flash', provider=GoogleGLAProvider(api_key=gemini_api_key))
    agent = Agent(m)

    result = await agent.run(
        'What is the capital of France?',
        model_settings=GeminiModelSettings(gemini_labels={'environment': 'test', 'team': 'analytics'}),
    )
    assert result.output == snapshot('The capital of France is **Paris**.\n')


@pytest.mark.vcr()
async def test_image_as_binary_content_input(
    allow_model_requests: None, gemini_api_key: str, image_content: BinaryContent
) -> None:
    m = GeminiModel('gemini-2.0-flash', provider=GoogleGLAProvider(api_key=gemini_api_key))
    agent = Agent(m)

    result = await agent.run(['What is the name of this fruit?', image_content])
    assert result.output == snapshot('The fruit in the image is a kiwi.')


@pytest.mark.vcr()
async def test_image_url_input(allow_model_requests: None, gemini_api_key: str) -> None:
    m = GeminiModel('gemini-2.0-flash-exp', provider=GoogleGLAProvider(api_key=gemini_api_key))
    agent = Agent(m)

    image_url = ImageUrl(url='https://goo.gle/instrument-img')

    result = await agent.run(['What is the name of this fruit?', image_url])
    assert result.output == snapshot("This is not a fruit; it's a pipe organ console.")


@pytest.mark.vcr()
async def test_video_as_binary_content_input(
    allow_model_requests: None, gemini_api_key: str, video_content: BinaryContent
) -> None:
    m = GeminiModel('gemini-1.5-flash', provider=GoogleGLAProvider(api_key=gemini_api_key))
    agent = Agent(m, system_prompt='You are a helpful chatbot.')

    result = await agent.run(['Explain me this video', video_content])
    assert result.output.strip() == snapshot(
        "That's a picture of a small, portable monitor attached to a camera, likely used for filming. The monitor displays a scene of a canyon or similar rocky landscape.  This suggests the camera is being used to film this landscape. The camera itself is mounted on a tripod, indicating a stable and likely professional setup.  The background is out of focus, but shows the same canyon as seen on the monitor. This makes it clear that the image shows the camera's viewfinder or recording output, rather than an unrelated display."
    )


@pytest.mark.vcr()
async def test_video_url_input(allow_model_requests: None, gemini_api_key: str) -> None:
    m = GeminiModel('gemini-1.5-flash', provider=GoogleGLAProvider(api_key=gemini_api_key))
    agent = Agent(m, system_prompt='You are a helpful chatbot.')

    video_url = VideoUrl(url='https://data.grepit.app/assets/tiny_video.mp4')

    result = await agent.run(['Explain me this video', video_url])
    assert result.output.strip() == snapshot(
        """That's a lovely picture!  It shows a picturesque outdoor cafe or restaurant situated in a narrow, whitewashed alleyway.


Here's a breakdown of what we see:

* **Location:** The cafe is nestled between two white buildings, typical of Greek island architecture (possibly Mykonos or a similar island, judging by the style).  The alleyway opens up to a view of the Aegean Sea, which is visible in the background. The sea appears somewhat choppy.

* **Setting:** The cafe has several wooden tables and chairs set out along the alley. The tables are simple and seem to be made of light-colored wood. There are cushions on a built-in bench along one wall providing seating. Small potted plants are on some tables, adding to the ambiance. The cobblestone ground in the alley adds to the charming, traditional feel.

* **Atmosphere:** The overall feel is relaxed and serene, despite the somewhat windy conditions indicated by the sea. The bright white buildings and the blue sea create a classic Mediterranean vibe. The picture evokes a sense of calmness and escape.

In short, the image depicts an idyllic scene of a charming seaside cafe in a picturesque Greek island setting."""
    )


@pytest.mark.vcr()
async def test_document_url_input(allow_model_requests: None, gemini_api_key: str) -> None:
    m = GeminiModel('gemini-2.0-flash-thinking-exp-01-21', provider=GoogleGLAProvider(api_key=gemini_api_key))
    agent = Agent(m)

    document_url = DocumentUrl(url='https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf')

    result = await agent.run(['What is the main content on this document?', document_url])
    assert result.output == snapshot('The main content of this document is that it is a **dummy PDF file**.')


@pytest.mark.vcr()
async def test_gemini_drop_exclusive_maximum(allow_model_requests: None, gemini_api_key: str) -> None:
    m = GeminiModel('gemini-2.0-flash', provider=GoogleGLAProvider(api_key=gemini_api_key))
    agent = Agent(m)

    @agent.tool_plain
    async def get_chinese_zodiac(age: Annotated[int, Field(gt=18)]) -> str:
        return 'Dragon'

    result = await agent.run('I want to know my chinese zodiac. I am 20 years old.')
    assert result.output == snapshot('Your Chinese zodiac is Dragon.\n')

    result = await agent.run('I want to know my chinese zodiac. I am 17 years old.')
    assert result.output == snapshot('I am sorry, I cannot fulfill this request. The age needs to be greater than 18.')


@pytest.mark.vcr()
async def test_gemini_model_instructions(allow_model_requests: None, gemini_api_key: str):
    m = GeminiModel('gemini-1.5-flash', provider=GoogleGLAProvider(api_key=gemini_api_key))
    agent = Agent(m, instructions='You are a helpful assistant.')

    result = await agent.run('What is the capital of France?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is the capital of France?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='The capital of France is Paris.\n')],
                usage=RequestUsage(
                    input_tokens=13, output_tokens=8, details={'text_prompt_tokens': 13, 'text_candidates_tokens': 8}
                ),
                model_name='gemini-1.5-flash',
                timestamp=IsDatetime(),
                provider_url='https://generativelanguage.googleapis.com/v1beta/models/',
                provider_details={'finish_reason': 'STOP'},
                run_id=IsStr(),
            ),
        ]
    )


class CurrentLocation(BaseModel, extra='forbid'):
    city: str
    country: str


@pytest.mark.vcr()
async def test_gemini_additional_properties_is_false(allow_model_requests: None, gemini_api_key: str):
    m = GeminiModel('gemini-2.0-flash', provider=GoogleGLAProvider(api_key=gemini_api_key))
    agent = Agent(m)

    @agent.tool_plain
    async def get_temperature(location: CurrentLocation) -> float:  # pragma: no cover
        return 20.0

    result = await agent.run('What is the temperature in Tokyo?')
    assert result.output == snapshot(
        'I need the country to find the temperature in Tokyo. Could you please tell me which country Tokyo is in?\n'
    )


@pytest.mark.vcr()
async def test_gemini_additional_properties_is_true(allow_model_requests: None, gemini_api_key: str):
    """Test that additionalProperties with schemas now work natively (no warning since Nov 2025 announcement)."""
    m = GeminiModel('gemini-2.5-flash', provider=GoogleGLAProvider(api_key=gemini_api_key))
    agent = Agent(m)

    @agent.tool_plain
    async def get_temperature(location: dict[str, CurrentLocation]) -> float:  # pragma: no cover
        return 20.0

    result = await agent.run('What is the temperature in Tokyo?')
    assert result.output == snapshot('I need the latitude and longitude for Tokyo. Can you please provide them?')


@pytest.mark.vcr()
async def test_gemini_model_thinking_part(allow_model_requests: None, gemini_api_key: str):
    model = GeminiModel('gemini-2.5-flash', provider=GoogleGLAProvider(api_key=gemini_api_key))
    agent = Agent(model)

    result = await agent.run(
        'What is 2+2?',
        model_settings=GeminiModelSettings(
            gemini_thinking_config={'thinking_budget': 1024, 'include_thoughts': True},
        ),
    )

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is 2+2?',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="""\
**Calculating the Simple**

Okay, here we go. Someone's asking a pretty straightforward arithmetic question. It's really just a simple calculation. Nothing fancy, just plug in the numbers and get the result.  No need to overthink it. It's a quick win, a chance to flex some basic math muscles before getting into anything more complex. Just a matter of applying the right operation and moving on.
"""
                    ),
                    TextPart(content='2 + 2 = 4'),
                ],
                usage=RequestUsage(
                    input_tokens=8, output_tokens=24, details={'thoughts_tokens': 16, 'text_prompt_tokens': 8}
                ),
                model_name='gemini-2.5-flash',
                timestamp=IsNow(tz=timezone.utc),
                provider_url='https://generativelanguage.googleapis.com/v1beta/models/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id='lghYaaSmK7eomtkP_KDT6A0',
                run_id=IsStr(),
            ),
        ]
    )


@pytest.mark.vcr()
async def test_gemini_youtube_video_url_input(allow_model_requests: None, gemini_api_key: str) -> None:
    url = VideoUrl(url='https://youtu.be/lCdaVNyHtjU')

    m = GeminiModel('gemini-2.0-flash', provider=GoogleGLAProvider(api_key=gemini_api_key))
    agent = Agent(m)
    result = await agent.run(['What is the main content of this URL?', url])

    assert result.output == snapshot(
        'The main content of the URL is an analysis of recent 404 HTTP responses. The analysis identifies several patterns, including the most common endpoints with 404 errors, request patterns (such as all requests being GET requests), timeline-related issues, and configuration/authentication problems. The analysis also provides recommendations for addressing the 404 errors.'
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content=['What is the main content of this URL?', url], timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='The main content of the URL is an analysis of recent 404 HTTP responses. The analysis identifies several patterns, including the most common endpoints with 404 errors, request patterns (such as all requests being GET requests), timeline-related issues, and configuration/authentication problems. The analysis also provides recommendations for addressing the 404 errors.'
                    )
                ],
                usage=RequestUsage(
                    input_tokens=9,
                    output_tokens=72,
                    details={
                        'text_prompt_tokens': 9,
                        'video_prompt_tokens': 0,
                        'audio_prompt_tokens': 0,
                        'text_candidates_tokens': 72,
                    },
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_url='https://generativelanguage.googleapis.com/v1beta/models/',
                provider_details={'finish_reason': 'STOP'},
                run_id=IsStr(),
            ),
        ]
    )


async def test_gemini_no_finish_reason(get_gemini_client: GetGeminiClient):
    response = gemini_response(
        _content_model_response(ModelResponse(parts=[TextPart('Hello world')])), finish_reason=None
    )
    gemini_client = get_gemini_client(response)
    m = GeminiModel('gemini-1.5-flash', provider=GoogleGLAProvider(http_client=gemini_client))
    agent = Agent(m)

    result = await agent.run('Hello World')

    for message in result.all_messages():
        if isinstance(message, ModelResponse):
            assert message.provider_details is None


async def test_response_with_thought_part(get_gemini_client: GetGeminiClient):
    """Tests that a response containing a 'thought' part can be parsed."""
    content_with_thought = _GeminiContent(
        role='model',
        parts=[
            _GeminiThoughtPart(thought=True, thought_signature='test-signature-value'),
            _GeminiTextPart(text='Hello from thought test'),
        ],
    )
    response = gemini_response(content_with_thought)
    gemini_client = get_gemini_client(response)
    m = GeminiModel('gemini-1.5-flash', provider=GoogleGLAProvider(http_client=gemini_client))
    agent = Agent(m)

    result = await agent.run('Test with thought')

    assert result.output == 'Hello from thought test'
    assert result.usage() == snapshot(RunUsage(requests=1, input_tokens=1, output_tokens=2))


@pytest.mark.vcr()
async def test_gemini_tool_config_any_with_tool_without_args(allow_model_requests: None, gemini_api_key: str):
    class Foo(BaseModel):
        bar: str

    m = GeminiModel('gemini-2.0-flash', provider=GoogleGLAProvider(api_key=gemini_api_key))
    agent = Agent(m, output_type=Foo)

    @agent.tool_plain
    async def bar() -> str:
        return 'hello'

    result = await agent.run('run bar for me please')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='run bar for me please',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='bar', args={}, tool_call_id=IsStr())],
                usage=RequestUsage(
                    input_tokens=21, output_tokens=1, details={'text_prompt_tokens': 21, 'text_candidates_tokens': 1}
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_url='https://generativelanguage.googleapis.com/v1beta/models/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='bar',
                        content='hello',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args={'bar': 'hello'},
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(
                    input_tokens=27, output_tokens=5, details={'text_prompt_tokens': 27, 'text_candidates_tokens': 5}
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_url='https://generativelanguage.googleapis.com/v1beta/models/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )


@pytest.mark.vcr()
async def test_gemini_tool_output(allow_model_requests: None, gemini_api_key: str):
    m = GeminiModel('gemini-2.0-flash', provider=GoogleGLAProvider(api_key=gemini_api_key))

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(m, output_type=ToolOutput(CityLocation))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run('What is the largest city in the user country?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_user_country', args={}, tool_call_id=IsStr())],
                usage=RequestUsage(
                    input_tokens=32, output_tokens=5, details={'text_prompt_tokens': 32, 'text_candidates_tokens': 5}
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_url='https://generativelanguage.googleapis.com/v1beta/models/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args={'country': 'Mexico', 'city': 'Mexico City'},
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(
                    input_tokens=46, output_tokens=8, details={'text_prompt_tokens': 46, 'text_candidates_tokens': 8}
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_url='https://generativelanguage.googleapis.com/v1beta/models/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )


@pytest.mark.vcr()
async def test_gemini_text_output_function(allow_model_requests: None, gemini_api_key: str):
    m = GeminiModel('gemini-2.5-pro-preview-05-06', provider=GoogleGLAProvider(api_key=gemini_api_key))

    def upcase(text: str) -> str:
        return text.upper()

    agent = Agent(m, output_type=TextOutput(upcase))

    result = await agent.run('What is the largest city in Mexico?')
    assert result.output == snapshot("""\
THE LARGEST CITY IN MEXICO IS **MEXICO CITY (CIUDAD DE MÉXICO, CDMX)**.

IT'S THE CAPITAL OF MEXICO AND ONE OF THE LARGEST METROPOLITAN AREAS IN THE WORLD, BOTH BY POPULATION AND LAND AREA.\
""")

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in Mexico?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
The largest city in Mexico is **Mexico City (Ciudad de México, CDMX)**.

It's the capital of Mexico and one of the largest metropolitan areas in the world, both by population and land area.\
"""
                    )
                ],
                usage=RequestUsage(
                    input_tokens=9, output_tokens=589, details={'thoughts_tokens': 545, 'text_prompt_tokens': 9}
                ),
                model_name='models/gemini-2.5-pro-preview-05-06',
                timestamp=IsDatetime(),
                provider_url='https://generativelanguage.googleapis.com/v1beta/models/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id='TT9IaNfGN_DmqtsPzKnE4AE',
                run_id=IsStr(),
            ),
        ]
    )


@pytest.mark.vcr()
async def test_gemini_native_output_with_tools(allow_model_requests: None, gemini_api_key: str):
    m = GeminiModel('gemini-2.0-flash', provider=GoogleGLAProvider(api_key=gemini_api_key))

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(m, output_type=NativeOutput(CityLocation))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'  # pragma: no cover

    with pytest.raises(
        UserError,
        match=re.escape(
            'Gemini does not support `NativeOutput` and tools at the same time. Use `output_type=ToolOutput(...)` instead.'
        ),
    ):
        await agent.run('What is the largest city in the user country?')


@pytest.mark.vcr()
async def test_gemini_native_output(allow_model_requests: None, gemini_api_key: str):
    m = GeminiModel('gemini-2.0-flash', provider=GoogleGLAProvider(api_key=gemini_api_key))

    class CityLocation(BaseModel):
        """A city and its country."""

        city: str
        country: str

    agent = Agent(m, output_type=NativeOutput(CityLocation))

    result = await agent.run('What is the largest city in Mexico?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in Mexico?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
{
  "city": "Mexico City",
  "country": "Mexico"
}\
"""
                    )
                ],
                usage=RequestUsage(
                    input_tokens=8, output_tokens=20, details={'text_prompt_tokens': 8, 'text_candidates_tokens': 20}
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_url='https://generativelanguage.googleapis.com/v1beta/models/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                run_id=IsStr(),
            ),
        ]
    )


@pytest.mark.vcr()
async def test_gemini_native_output_multiple(allow_model_requests: None, gemini_api_key: str):
    m = GeminiModel('gemini-2.0-flash', provider=GoogleGLAProvider(api_key=gemini_api_key))

    class CityLocation(BaseModel):
        city: str
        country: str

    class CountryLanguage(BaseModel):
        country: str
        language: str

    agent = Agent(m, output_type=NativeOutput([CityLocation, CountryLanguage]))

    result = await agent.run('What is the primarily language spoken in Mexico?')
    assert result.output == snapshot(CountryLanguage(country='Mexico', language='Spanish'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the primarily language spoken in Mexico?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
{
  "result": {
    "data": {
      "country": "Mexico",
      "language": "Spanish"
    },
    "kind": "CountryLanguage"
  }
}\
"""
                    )
                ],
                usage=RequestUsage(
                    input_tokens=46, output_tokens=46, details={'text_prompt_tokens': 46, 'text_candidates_tokens': 46}
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_url='https://generativelanguage.googleapis.com/v1beta/models/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                run_id=IsStr(),
            ),
        ]
    )


@pytest.mark.vcr()
async def test_gemini_prompted_output(allow_model_requests: None, gemini_api_key: str):
    m = GeminiModel('gemini-2.0-flash', provider=GoogleGLAProvider(api_key=gemini_api_key))

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(m, output_type=PromptedOutput(CityLocation))

    result = await agent.run('What is the largest city in Mexico?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in Mexico?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='{"properties": {"city": {"type": "string"}, "country": {"type": "string"}}, "required": ["city", "country"], "title": "CityLocation", "type": "object", "city": "Mexico City", "country": "Mexico"}'
                    )
                ],
                usage=RequestUsage(
                    input_tokens=80, output_tokens=56, details={'text_prompt_tokens': 80, 'text_candidates_tokens': 56}
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_url='https://generativelanguage.googleapis.com/v1beta/models/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                run_id=IsStr(),
            ),
        ]
    )


@pytest.mark.vcr()
async def test_gemini_prompted_output_with_tools(allow_model_requests: None, gemini_api_key: str):
    m = GeminiModel('gemini-2.5-pro-preview-05-06', provider=GoogleGLAProvider(api_key=gemini_api_key))

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(m, output_type=PromptedOutput(CityLocation))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run(
        'What is the largest city in the user country? Use the get_user_country tool and then your own world knowledge.'
    )
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country? Use the get_user_country tool and then your own world knowledge.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_user_country', args={}, tool_call_id=IsStr())],
                usage=RequestUsage(
                    input_tokens=123, output_tokens=330, details={'thoughts_tokens': 318, 'text_prompt_tokens': 123}
                ),
                model_name='models/gemini-2.5-pro-preview-05-06',
                timestamp=IsDatetime(),
                provider_url='https://generativelanguage.googleapis.com/v1beta/models/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"city": "Mexico City", "country": "Mexico"}')],
                usage=RequestUsage(
                    input_tokens=154, output_tokens=107, details={'thoughts_tokens': 94, 'text_prompt_tokens': 154}
                ),
                model_name='models/gemini-2.5-pro-preview-05-06',
                timestamp=IsDatetime(),
                provider_url='https://generativelanguage.googleapis.com/v1beta/models/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                run_id=IsStr(),
            ),
        ]
    )


@pytest.mark.vcr()
async def test_gemini_prompted_output_multiple(allow_model_requests: None, gemini_api_key: str):
    m = GeminiModel('gemini-2.0-flash', provider=GoogleGLAProvider(api_key=gemini_api_key))

    class CityLocation(BaseModel):
        city: str
        country: str

    class CountryLanguage(BaseModel):
        country: str
        language: str

    agent = Agent(m, output_type=PromptedOutput([CityLocation, CountryLanguage]))

    result = await agent.run('What is the largest city in Mexico?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in Mexico?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='{"result": {"kind": "CityLocation", "data": {"city": "Mexico City", "country": "Mexico"}}}'
                    )
                ],
                usage=RequestUsage(
                    input_tokens=253,
                    output_tokens=27,
                    details={'text_prompt_tokens': 253, 'text_candidates_tokens': 27},
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_url='https://generativelanguage.googleapis.com/v1beta/models/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                run_id=IsStr(),
            ),
        ]
    )


def test_map_usage():
    response = gemini_response(_content_model_response(ModelResponse(parts=[TextPart('Hello world')])))
    assert 'usage_metadata' in response
    response['usage_metadata']['cached_content_token_count'] = 9100
    response['usage_metadata']['prompt_tokens_details'] = [
        _GeminiModalityTokenCount(modality='AUDIO', token_count=9200)
    ]
    response['usage_metadata']['cache_tokens_details'] = [
        _GeminiModalityTokenCount(modality='AUDIO', token_count=9300),
    ]
    response['usage_metadata']['candidates_tokens_details'] = [
        _GeminiModalityTokenCount(modality='AUDIO', token_count=9400)
    ]
    response['usage_metadata']['thoughts_token_count'] = 9500
    response['usage_metadata']['tool_use_prompt_token_count'] = 9600

    assert _metadata_as_usage(response) == snapshot(
        RequestUsage(
            input_tokens=1,
            cache_read_tokens=9100,
            output_tokens=9502,
            input_audio_tokens=9200,
            cache_audio_read_tokens=9300,
            output_audio_tokens=9400,
            details={
                'cached_content_tokens': 9100,
                'audio_prompt_tokens': 9200,
                'audio_cache_tokens': 9300,
                'audio_candidates_tokens': 9400,
                'thoughts_tokens': 9500,
                'tool_use_prompt_tokens': 9600,
            },
        )
    )


def test_map_empty_usage():
    response = gemini_response(_content_model_response(ModelResponse(parts=[TextPart('Hello world')])))
    assert 'usage_metadata' in response
    del response['usage_metadata']

    assert _metadata_as_usage(response) == RequestUsage()
