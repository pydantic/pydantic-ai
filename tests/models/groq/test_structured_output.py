from __future__ import annotations as _annotations

import json
from dataclasses import dataclass
from typing import Literal

import httpx
import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, UserError
from pydantic_ai.output import NativeOutput, PromptedOutput
from pydantic_ai.profiles.groq import GroqModelProfile

from ..._inline_snapshot import snapshot
from ...conftest import try_import

with try_import() as imports_successful:
    from groq.types.chat.chat_completion_message import ChatCompletionMessage

    from pydantic_ai.models import create_async_http_client
    from pydantic_ai.models.groq import GroqModel, GroqModelSettings
    from pydantic_ai.providers.groq import GroqProvider

    from ..test_groq import MockGroq, completion_message, text_chunk

pytestmark = pytest.mark.skipif(not imports_successful(), reason='groq not installed')


class CityLocation(BaseModel):
    city: str
    country: str


CITY_LOCATION_JSON = '{"city": "Mexico City", "country": "Mexico"}'


class Country(BaseModel):
    name: str


class CityInCountry(BaseModel):
    city: str
    country: Country
    neighbours: list[Country]


class CityWithNickname(BaseModel):
    city: str
    nickname: str = 'CDMX'


def lookup_country(location: CityLocation) -> str:
    # Unreachable: the guard raises before the model is called, and the request-body mocks never call a tool.
    return location.country  # pragma: no cover


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('native_from', ['marker', 'profile'])
async def test_groq_native_output_with_function_tools(
    allow_model_requests: None, stream: bool, native_from: Literal['marker', 'profile']
):
    """`NativeOutput` with a function tool raises before any request, since Groq returns a 400 for the combination.

    It raises streamed or not, and whether native output comes from the `NativeOutput` marker or from the profile's
    `default_structured_output_mode`. A pre-request guard, so there's no response to record. A request that did go
    out would get the mock's completion instead of this error.
    """
    c = completion_message(ChatCompletionMessage(content=CITY_LOCATION_JSON, role='assistant'))
    provider = GroqProvider(groq_client=MockGroq.create_mock(c))
    if native_from == 'marker':
        model = GroqModel('openai/gpt-oss-120b', provider=provider)
        agent = Agent(model, output_type=NativeOutput(CityLocation), tools=[lookup_country])
    else:
        profile = GroqModelProfile(default_structured_output_mode='native')
        model = GroqModel('openai/gpt-oss-120b', provider=provider, profile=profile)
        agent = Agent(model, output_type=CityLocation, tools=[lookup_country])

    prompt = 'What is the largest city in the user country?'
    with pytest.raises(UserError) as exc_info:
        if stream:
            async with agent.run_stream_events(prompt) as events:
                _ = [event async for event in events]
        else:
            await agent.run(prompt)

    assert str(exc_info.value) == snapshot(
        'This model does not support `NativeOutput` and function tools at the same time. Use `output_type=ToolOutput(...)` instead.'
    )


GroqResponseFormatOutput = NativeOutput[CityInCountry] | PromptedOutput[CityLocation]


@dataclass(frozen=True)
class GroqResponseFormatCase:
    id: str
    model_name: str
    output_type: GroqResponseFormatOutput
    expected: object
    """Snapshot of the request body's `response_format`."""
    output: CityInCountry | CityLocation


GROQ_RESPONSE_FORMAT_CASES = [
    GroqResponseFormatCase(
        # Groq's strict mode requires `additionalProperties: false` on every object, nested ones included.
        id='qwen3.8-native-strict-nested',
        model_name='qwen/qwen3.8-27b',
        output_type=NativeOutput(CityInCountry),
        expected=snapshot(
            {
                'type': 'json_schema',
                'json_schema': {
                    'name': CityInCountry.__qualname__,
                    'schema': {
                        'properties': {
                            'city': {'type': 'string'},
                            'country': {
                                'properties': {'name': {'type': 'string'}},
                                'required': ['name'],
                                'type': 'object',
                                'additionalProperties': False,
                            },
                            'neighbours': {
                                'items': {
                                    'properties': {'name': {'type': 'string'}},
                                    'required': ['name'],
                                    'type': 'object',
                                    'additionalProperties': False,
                                },
                                'type': 'array',
                            },
                        },
                        'required': ['city', 'country', 'neighbours'],
                        'type': 'object',
                        'additionalProperties': False,
                    },
                    'strict': True,
                },
            }
        ),
        output=CityInCountry(
            city='Mexico City',
            country=Country(name='Mexico'),
            neighbours=[Country(name='United States'), Country(name='Guatemala')],
        ),
    ),
    GroqResponseFormatCase(
        id='qwen3.8-prompted',
        model_name='qwen/qwen3.8-27b',
        output_type=PromptedOutput(CityLocation),
        expected=snapshot({'type': 'json_object'}),
        output=CityLocation(city='Mexico City', country='Mexico'),
    ),
]


@pytest.mark.vcr
@pytest.mark.parametrize('case', [pytest.param(c, id=c.id) for c in GROQ_RESPONSE_FORMAT_CASES])
async def test_groq_response_format(allow_model_requests: None, groq_api_key: str, case: GroqResponseFormatCase):
    """Groq answers the `response_format` qwen3.8 gets: a strict `json_schema` for `NativeOutput`, with
    `additionalProperties: false` on every object, and `json_object` for `PromptedOutput`.

    Cassette matching ignores the request body, so a request hook reads the body off the wire, on replay too.
    """
    sent: list[httpx.Request] = []

    async def record(request: httpx.Request) -> None:
        sent.append(request)

    async with create_async_http_client() as http_client:
        http_client.event_hooks = {'request': [record], 'response': []}
        provider = GroqProvider(api_key=groq_api_key, http_client=http_client)
        agent = Agent(GroqModel(case.model_name, provider=provider), output_type=case.output_type)
        result = await agent.run('What is the largest city in Mexico? Name two of its neighbouring countries.')

    assert result.output == case.output
    [request] = sent
    assert json.loads(request.content)['response_format'] == case.expected


GroqRequestBodyOutput = (
    NativeOutput[CityLocation] | NativeOutput[CityWithNickname] | PromptedOutput[CityLocation] | type[str]
)


@dataclass(frozen=True)
class GroqRequestBodyCase:
    id: str
    model_name: str
    expected: object
    """Snapshot of the request body's `stream`, `response_format`, `reasoning_format`, `reasoning_effort` and `tools`."""
    output_type: GroqRequestBodyOutput = str
    response: str = CITY_LOCATION_JSON
    stream: bool = False
    function_tool: bool = False
    profile: GroqModelProfile | None = None
    thinking: bool | None = None


GROQ_REQUEST_BODY_CASES = [
    GroqRequestBodyCase(
        id='gpt-oss-native-stream',
        model_name='openai/gpt-oss-120b',
        output_type=NativeOutput(CityLocation),
        stream=True,
        expected=snapshot(
            {
                'stream': True,
                'response_format': {
                    'type': 'json_schema',
                    'json_schema': {
                        'name': CityLocation.__qualname__,
                        'schema': {
                            'properties': {'city': {'type': 'string'}, 'country': {'type': 'string'}},
                            'required': ['city', 'country'],
                            'type': 'object',
                            'additionalProperties': False,
                        },
                        'strict': True,
                    },
                },
            }
        ),
    ),
    GroqRequestBodyCase(
        # Groq's strict mode requires every field to be required, so a defaulted field sends `strict: false`.
        id='qwen3.8-native-defaulted-field',
        model_name='qwen/qwen3.8-27b',
        output_type=NativeOutput(CityWithNickname),
        response='{"city": "Mexico City"}',
        expected=snapshot(
            {
                'stream': False,
                'response_format': {
                    'type': 'json_schema',
                    'json_schema': {
                        'name': CityWithNickname.__qualname__,
                        'schema': {
                            'properties': {
                                'city': {'type': 'string'},
                                'nickname': {'default': 'CDMX', 'type': 'string'},
                            },
                            'required': ['city'],
                            'type': 'object',
                            'additionalProperties': False,
                        },
                        'strict': False,
                    },
                },
            }
        ),
    ),
    GroqRequestBodyCase(
        # Groq doesn't list qwen3.6 for strict mode, so its output schema goes out best-effort.
        id='qwen3.6-native',
        model_name='qwen/qwen3.6-27b',
        output_type=NativeOutput(CityLocation),
        expected=snapshot(
            {
                'stream': False,
                'response_format': {
                    'type': 'json_schema',
                    'json_schema': {
                        'name': CityLocation.__qualname__,
                        'schema': {
                            'properties': {'city': {'type': 'string'}, 'country': {'type': 'string'}},
                            'required': ['city', 'country'],
                            'title': CityLocation.__qualname__,
                            'type': 'object',
                        },
                        'strict': False,
                    },
                },
            }
        ),
    ),
    GroqRequestBodyCase(
        id='muse-native',
        model_name='meta-models/muse-spark-open',
        output_type=NativeOutput(CityLocation),
        expected=snapshot(
            {
                'stream': False,
                'response_format': {
                    'type': 'json_schema',
                    'json_schema': {
                        'name': CityLocation.__qualname__,
                        'schema': {
                            'properties': {'city': {'type': 'string'}, 'country': {'type': 'string'}},
                            'required': ['city', 'country'],
                            'title': CityLocation.__qualname__,
                            'type': 'object',
                        },
                        'strict': None,
                    },
                },
            }
        ),
    ),
    GroqRequestBodyCase(
        id='qwen3.8-prompted-stream',
        model_name='qwen/qwen3.8-27b',
        output_type=PromptedOutput(CityLocation),
        stream=True,
        expected=snapshot({'stream': True, 'response_format': {'type': 'json_object'}}),
    ),
    GroqRequestBodyCase(
        # JSON mode can't be combined with function tools, so the request carries no `response_format`. Groq never
        # receives `strict` on a tool, so the tool schema keeps qwen's family transformer: it keeps its `title` and gets
        # no `additionalProperties`.
        id='qwen3.8-prompted-with-function-tool',
        model_name='qwen/qwen3.8-27b',
        output_type=PromptedOutput(CityLocation),
        function_tool=True,
        expected=snapshot(
            {
                'stream': False,
                'tools': [
                    {
                        'type': 'function',
                        'function': {
                            'name': lookup_country.__qualname__,
                            'description': '',
                            'parameters': {
                                'properties': {'city': {'type': 'string'}, 'country': {'type': 'string'}},
                                'required': ['city', 'country'],
                                'title': CityLocation.__qualname__,
                                'type': 'object',
                            },
                        },
                    }
                ],
            }
        ),
    ),
    GroqRequestBodyCase(
        id='qwen3.8-prompted-json-mode-off',
        model_name='qwen/qwen3.8-27b',
        output_type=PromptedOutput(CityLocation),
        profile=GroqModelProfile(supports_json_object_output=False),
        expected=snapshot({'stream': False}),
    ),
    GroqRequestBodyCase(
        id='muse-thinking',
        model_name='meta-models/muse-spark-open',
        thinking=True,
        expected=snapshot({'stream': False, 'reasoning_format': 'parsed'}),
    ),
    GroqRequestBodyCase(
        id='muse-no-thinking',
        model_name='meta-models/muse-spark-open',
        thinking=False,
        expected=snapshot({'stream': False}),
    ),
    GroqRequestBodyCase(
        id='minimax-thinking',
        model_name='minimaxai/minimax-m2.7',
        thinking=True,
        expected=snapshot({'stream': False, 'reasoning_format': 'parsed'}),
    ),
    GroqRequestBodyCase(
        id='minimax-no-thinking',
        model_name='minimaxai/minimax-m2.7',
        thinking=False,
        expected=snapshot({'stream': False}),
    ),
]


@pytest.mark.parametrize('case', [pytest.param(c, id=c.id) for c in GROQ_REQUEST_BODY_CASES])
async def test_groq_request_body(allow_model_requests: None, case: GroqRequestBodyCase):
    """What `GroqModel` sends for structured output and reasoning settings beyond `test_groq_response_format`.

    `NativeOutput` sends a `json_schema` response format, strict when the schema meets strict mode's requirements.
    `PromptedOutput` sends `json_object` unless function tools are present or the profile turns JSON mode off.
    `thinking` on the always-on reasoning models sends `reasoning_format`.

    A mock transport serves the responses and the body is read off the wire. The stream, tool and profile cases vary
    how the request is built on models Groq answers in the recordings. Groq offers Muse Spark Open and MiniMax M2.7
    only as an Enterprise preview, and has shut qwen3.6 down for its free and developer tiers, so those cases have no
    recording. `GroqProvider` takes a legacy `httpx.AsyncClient`, so the shared `request_capture` fixture, an
    `httpx2` client, can't be injected.
    """
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        if case.stream:
            sse_chunk = text_chunk(case.response, finish_reason='stop').model_dump_json()
            content = f'data: {sse_chunk}\n\ndata: [DONE]\n\n'
            return httpx.Response(200, content=content, headers={'content-type': 'text/event-stream'})
        c = completion_message(ChatCompletionMessage(content=case.response, role='assistant'))
        return httpx.Response(200, json=c.model_dump(mode='json'))

    settings = GroqModelSettings()
    if case.thinking is not None:
        settings['thinking'] = case.thinking
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider = GroqProvider(api_key='test-key', base_url='http://localhost', http_client=http_client)
        agent = Agent(
            GroqModel(case.model_name, provider=provider, profile=case.profile),
            output_type=case.output_type,
            tools=[lookup_country] if case.function_tool else [],
            model_settings=settings,
        )
        prompt = 'What is the largest city in Mexico?'
        if case.stream:
            async with agent.run_stream(prompt) as result:
                await result.get_output()
        else:
            await agent.run(prompt)

    [request] = sent
    body = json.loads(request.content)
    keys = ('stream', 'response_format', 'reasoning_format', 'reasoning_effort', 'tools')
    assert {key: body[key] for key in keys if key in body} == case.expected
