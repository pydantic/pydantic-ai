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
from pydantic_ai.messages import BinaryImage
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
from pydantic_ai.usage import RequestUsage, UsageLimits

from ..conftest import ClientWithHandler, IsBytes, IsDatetime, IsInstance, IsNow, IsStr, TestEnv

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
                        content=BinaryImage(data=IsBytes(), media_type='image/jpeg'),
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
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
async def test_gemini_model_thinking_part(allow_model_requests: None, gemini_api_key: str, openai_api_key: str):
    gemini_model = GeminiModel('gemini-2.5-flash', provider=GoogleGLAProvider(api_key=gemini_api_key))
    agent = Agent(gemini_model)

    # We call OpenAI to get the thinking parts, because Google disabled the thoughts in the API.
    # See https://github.com/pydantic/pydantic-ai/issues/793 for more details.
    result = await agent.run(
        'How do I cross the street?',
        model_settings=GeminiModelSettings(gemini_thinking_config={'thinking_budget': 1024, 'include_thoughts': True}),
        usage_limits=UsageLimits(total_tokens_limit=5000),
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='How do I cross the street?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    IsInstance(ThinkingPart),
                    TextPart(
                        content="""\
Crossing the street safely is a fundamental skill, and it's great you're asking! Here's a breakdown of how to do it, from basic principles to specific scenarios:

## The Golden Rules of Crossing the Street:

1.  **Stop at the Curb:** Always stop at the edge of the sidewalk or street. Don't step into the street until you're ready to cross.
2.  **Look Left, Right, and Left Again:** This is crucial. Traffic patterns can vary, so make sure to check all directions. Look left first for oncoming traffic, then right, and then left again as you start to step out, as new traffic might have appeared.
3.  **Make Eye Contact with Drivers:** Don't just look at cars; try to make eye contact with drivers. This confirms that they see you. Never assume a driver sees you or will stop.
4.  **Wait for a Safe Gap or Signal:** Only cross when there's plenty of time and space for you to get to the other side safely.
5.  **Walk, Don't Run:** Cross the street at a steady, predictable pace. Running can make you lose balance or make it harder for drivers to react.
6.  **Stay Alert - No Distractions!** Put away your phone, take out your headphones, and pay full attention to your surroundings.

---

## Crossing in Different Scenarios:

### 1. At a Marked Crosswalk with a Traffic Light / Pedestrian Signal:

*   **Wait for the "WALK" Signal:** Press the pedestrian button if there is one. Wait for the signal to change to a "WALK" symbol (a white walking person or the word "WALK").
*   **Look L-R-L:** Even with a signal, always check for turning vehicles or drivers running a red light.
*   **Cross Promptly:** Walk across the street within the crosswalk lines.
*   **Beware of Flashing "DON'T WALK":** If the "DON'T WALK" hand starts flashing while you're still on the curb, don't start crossing. If you've already started, continue to the other side quickly but safely.

### 2. At a Marked Crosswalk (No Traffic Light):

*   **Stop at the Curb:** Position yourself clearly visible at the edge of the crosswalk.
*   **Look L-R-L:** Scan for approaching vehicles.
*   **Wait for Vehicles to Stop:** Drivers are generally required to yield to pedestrians in marked crosswalks. Wait until traffic in all lanes has come to a complete stop before you begin.
*   **Make Eye Contact:** Confirm drivers see you and intend to stop.
*   **Cross Carefully:** Even when cars have stopped, remain vigilant for other vehicles, especially in multi-lane roads where one car might stop and another in an adjacent lane might not.

### 3. At an Unmarked Intersection (No Crosswalk Lines):

*   **Treat it Like a Crosswalk:** In many places, any intersection *is* legally considered a crosswalk, even if it's not painted. However, drivers might be less aware.
*   **Choose Wisely:** Only cross if you have clear visibility of traffic and there's a significant gap in vehicles.
*   **Extra Vigilance:** Be even more cautious than at a marked crosswalk, as drivers may be less expecting you.
*   **Look L-R-L and Make Eye Contact:** Absolutely critical here.

### 4. Crossing Mid-Block (Not at an Intersection or Crosswalk - "Jaywalking"):

*   **Discouraged and Often Illegal:** It's generally safer and often legally required to cross at intersections or marked crosswalks.
*   **If You Must (and it's safe/legal in your area):**
    *   **Find a Clear Spot:** Choose an area where you have excellent visibility of oncoming traffic in both directions, and drivers can see you easily. Avoid crossing from behind parked cars or obstacles.
    *   **Wide Gap:** Wait for a very wide gap in traffic.
    *   **Cross Swiftly:** Get across the street as directly and quickly as possible once it's clear.
    *   **Be Prepared to Wait:** You might need to wait a long time for a safe opportunity.

### 5. Using an Overpass or Underpass:

*   These are the safest options, as they completely separate pedestrians from vehicle traffic.
*   Always use them if they are available, especially on very busy or high-speed roads.

---

## Important Safety Considerations:

*   **Be Visible:** Wear bright or reflective clothing, especially at dawn, dusk, or night.
*   **Be Predictable:** Don't dart into traffic or suddenly change direction.
*   **Watch for Turning Vehicles:** Even if you have a "WALK" signal, always check for cars making right or left turns.
*   **Never Walk While Intoxicated/Impaired:** Your judgment and reaction time will be severely affected.
*   **Teach Children:** Always hold a child's hand when crossing and teach them these rules.
*   **Be Patient:** It's better to wait an extra minute than to take a risk.

By following these guidelines, you can cross the street safely and confidently!\
"""
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=8, output_tokens=1987, details={'thoughts_tokens': 849, 'text_prompt_tokens': 8}
                ),
                model_name='gemini-2.5-flash',
                timestamp=IsDatetime(),
                provider_url='https://generativelanguage.googleapis.com/v1beta/models/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id='Fxhfad0_29DPsg-UlsFJ',
                run_id=IsStr(),
            ),
        ]
    )

    result = await agent.run(
        'Considering the way to cross the street, analogously, how do I cross the river?',
        model=gemini_model,
        message_history=result.all_messages(),
        model_settings=GeminiModelSettings(
            gemini_thinking_config={'thinking_budget': 1024, 'include_thoughts': True},
        ),
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='How do I cross the street?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    IsInstance(ThinkingPart),
                    IsInstance(TextPart),
                ],
                usage=RequestUsage(
                    input_tokens=8, output_tokens=1987, details={'thoughts_tokens': 849, 'text_prompt_tokens': 8}
                ),
                model_name='gemini-2.5-flash',
                timestamp=IsDatetime(),
                provider_url='https://generativelanguage.googleapis.com/v1beta/models/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id='Fxhfad0_29DPsg-UlsFJ',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Considering the way to cross the street, analogously, how do I cross the river?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="""\
**Deconstructing "Crossing the River": An Analogical Approach**

Alright, let's break this down. The user wants to map "crossing the street" principles onto "crossing a river." Essentially, it's about taking that fundamental safety mindset and adapting it to a totally different environment. First, I need to jog my memory of the street crossing principles, from that earlier answer: stop at the curb, look all directions, make eye contact, safe gap, steady pace, stay alert, different crossing scenarios. Got it. Now, how does that translate?

Okay, let's brainstorm river crossing methods - that's the equivalent of thinking about crosswalks, overpasses, etc. Swimming, boating, a bridge, a ford, a ferry, wading, maybe even a rope traverse if we're getting technical. And, of course, the hazards: current, depth, temperature, rocks, wildlife, weather - the whole gamut. Also, prep - gear, knowledge, planning, a buddy.

Now, the core of the analogy. "Stop at the Curb" - that needs to become "Assess from the Bank." You don't just leap into a river; you take a minute and observe. Like 'before you get wet, stop at the curb'.

"Look L-R-L" becomes "Look Upstream, Downstream, and At the Water Itself." It's about spotting those river hazards and conditions. Instead of traffic, it's the current, the rapids, the rocks, the depth.

"Make Eye Contact with Drivers" is a tricky one. No drivers, but you need to "read the water." That's the river's "intent." Understand its power.

Then there's "Wait for a Safe Gap/Signal" - which becomes waiting for "Safe Conditions/Right Time." Low water, calm conditions, good weather. It's about knowing when to go.

And finally, "Walk, Don't Run" is now "Move Deliberately and Appropriately for Method." Steady swim, careful wade, purposeful paddle - it's about being in control. Just as you do not run across the street, you do not rush the river.

The final piece is "Stay Alert, No Distractions" - that's fundamental situational awareness regardless of the environment. Focus on the task at hand. Keep an eye out.
"""
                    ),
                    TextPart(
                        content="""\
This is a fantastic analogy! Let's break down how to "cross the river" using the principles of "crossing the street."

The core idea is **assessment, preparation, timing, and respectful engagement with the force you're crossing.**

---

## Crossing the River: An Analogous Guide

### The Golden Rules of "Crossing the River":

1.  **Stop at the Bank (Analogous to "Stop at the Curb"):**
    *   **Street:** Don't step into the street until you're ready.
    *   **River:** Don't just jump in. Get to the edge of the river (the bank) and take time to observe it. Never assume it's safe without looking.

2.  **Look Upstream, Downstream, and At the Water Itself (Analogous to "Look L-R-L"):**
    *   **Street:** Check for traffic coming from all directions.
    *   **River:**
        *   **Upstream (Left):** What's coming *towards* you? Rapids, waterfalls, debris, other boats, changing weather? What might influence the current from upriver?
        *   **Downstream (Right):** What's *below* you? Are there hazards you could be swept into? Strainers (fallen trees), rocks, rapids, dams, cold-water shock points? What are the potential consequences of losing control?
        *   **At the Water Itself (Left again):** What are the immediate conditions? Current speed, depth (if visible), clarity, temperature, hidden rocks, eddies, sandbars?

3.  **Read the Water's "Intent" / Understand its Power (Analogous to "Make Eye Contact with Drivers"):**
    *   **Street:** Confirm drivers see you and intend to stop.
    *   **River:** The river isn't a sentient being, but it *has* power and intent (its natural flow). You need to "read" the water's signals. Can you see rocks just below the surface? Is the current swirling strongly? Are there visible hazards? Understand that it will not stop for you, and you must respect its power.

4.  **Wait for Safe Conditions or the Right Opportunity (Analogous to "Wait for a Safe Gap or Signal"):**
    *   **Street:** Wait for a break in traffic or a "WALK" signal.
    *   **River:** Wait for:
        *   **Optimal Conditions:** Lower water levels, calmer weather, slower current (often in the morning or specific seasons).
        *   **A Clear "Path":** A visible ford, a calm eddy, a suitable landing spot on the other side.
        *   **The Right "Signal":** Perhaps you have a guide, a clear map of a known crossing point, or the weather forecast is perfect.

5.  **Move Deliberately and with Purpose (Analogous to "Walk, Don't Run"):**
    *   **Street:** Cross at a steady, predictable pace.
    *   **River:** Whether swimming, wading, or paddling, move with clear intention and control. Don't panic or make sudden, erratic movements. Every action should be planned and executed carefully.

6.  **Maintain Focus - No Distractions! (Analogous to "Stay Alert - No Distractions!"):**
    *   **Street:** Put away your phone, remove headphones.
    *   **River:** Your entire focus must be on the river, your technique, your equipment, and your surroundings. This is not the time for sightseeing or daydreaming.

---

## "Crossing the River" in Different Scenarios:

### 1. Using a Bridge (Analogous to "Marked Crosswalk with a Traffic Light / Pedestrian Signal"):
*   **Description:** The safest and most controlled way. A dedicated, engineered structure.
*   **Analogy:** You are completely separated from the river's immediate flow. It's the "green light" equivalent - follow the path, and you're generally safe from the main "traffic" (current). Still, pay attention to the bridge's condition and other users.

### 2. Using a Ferry or Known Boat Crossing (Analogous to "Marked Crosswalk, No Light"):
*   **Description:** You're using a specific, designated method of transport across.
*   **Analogy:** You're relying on a system (the ferry/boat operator) that knows the river's "traffic patterns" (currents, hazards). You wait for the "vehicle" to arrive and take you across. You've ceded control to a trusted system. Still, stay alert and follow instructions.

### 3. Fording / Wading Across a Shallow Section (Analogous to "Unmarked Intersection"):
*   **Description:** You're walking directly through the river where it's shallow enough.
*   **Analogy:** This requires heightened awareness. There are no "lines" or "signals." You must:
    *   **Verify Depth and Bottom:** Can you see the bottom? Is it rocky, muddy, slippery?
    *   **Assess Current:** Even shallow water can have a strong current. Use a stick for balance.
    *   **Choose Your Path:** Find the clearest, most direct, and safest route across.
    *   **Look L-R-L (Upstream, Downstream, Water):** Especially critical here, as no one is expecting you.

### 4. Swimming Directly Across (Analogous to "Crossing Mid-Block / Jaywalking"):
*   **Description:** The most direct, often riskiest, method, using your own ability.
*   **Analogy:** This is often discouraged unless you're highly skilled and prepared.
    *   **High Visibility & Clear Spot:** Like finding a clear spot to jaywalk, you need to ensure you can be seen and have a clear "path."
    *   **Wide Gap in "Traffic":** You need ample space and time, meaning very calm conditions, no immediate hazards, and plenty of time to reach the other side.
    *   **Preparedness:** You *must* be an excellent swimmer, understand river dynamics, and potentially have a personal flotation device (PFD). This is like needing specific gear for a risky street crossing (e.g., a high-visibility vest for a night crossing).
    *   **It Might Be Illegal/Dangerous:** Many rivers are unsafe for swimming, just as jaywalking is illegal/dangerous in many places.

---

## Important "River Crossing" Safety Considerations:

*   **Be Visible:** Wear bright colors, especially if others (boaters) are on the river.
*   **Be Predictable:** Don't suddenly change direction or stop mid-stream.
*   **Watch for Changing Conditions:** Rivers are dynamic. What was safe an hour ago might not be now (e.g., rain upstream, dam release).
*   **Never Cross Alone (If possible):** "Crossing with a buddy" is like having someone to spot for you on the street.
*   **Have the Right Gear:** PFD, appropriate footwear for wading, dry bag, map, communication device. This is like having appropriate clothing and awareness for street crossing.
*   **Know Your Limits:** Don't attempt to cross if you're not confident in your skills or the conditions.
*   **Be Patient:** Waiting for safer conditions or finding a better crossing point is always preferable to taking a risk.

In essence, crossing a river, like crossing a street, is about **respecting the environment's power, understanding its dynamics, and making informed, cautious decisions.**\
"""
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=1166, output_tokens=2581, details={'thoughts_tokens': 940, 'text_prompt_tokens': 1166}
                ),
                model_name='gemini-2.5-flash',
                timestamp=IsDatetime(),
                provider_url='https://generativelanguage.googleapis.com/v1beta/models/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id='KBhfab3gGIbOz7IPyJnhqAY',
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
