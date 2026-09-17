from __future__ import annotations as _annotations

import json
from collections.abc import Callable
from enum import Enum
from typing import Any, Literal

import httpx2
import pytest
from pydantic import BaseModel, Field

from pydantic_ai import (
    Agent,
    BinaryContent,
    ModelAPIError,
    ModelHTTPError,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeOutput,
    NativeToolCallPart,
    NativeToolReturnPart,
    PromptedOutput,
    RetryPromptPart,
    SystemPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    WebSearchTool,
)
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.direct import model_request
from pydantic_ai.exceptions import UnexpectedModelBehavior, UserError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

from .._inline_snapshot import snapshot
from ..conftest import IsStr, RequestCapture, TestEnv, try_import

with try_import() as imports_successful:
    from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

    from pydantic_ai.models.typesafe import TypeSafeModel
    from pydantic_ai.providers.typesafe import TypeSafeProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='typesafe-sdk not installed'),
    pytest.mark.anyio,
]


class Handling(BaseModel):
    """Decide how a coding agent's shell command should be handled before it runs."""

    verdict: Literal['run', 'reject', 'ask'] = Field(
        description='How to handle this command.',
        json_schema_extra={
            'typesafe_criteria': {
                'run': 'Reads, builds, tests or edits inside the project. Reversible.',
                'reject': 'Destroys data, rewrites shared history, or sends secrets over the network.',
                'ask': 'Legitimate but consequential enough that a human should confirm.',
            }
        },
    )
    irreversible: bool = Field(description='Would running this destroy data or leak secrets?')


class Colour(str, Enum):
    red = 'red'
    blue = 'blue'


class EnumAndProbability(BaseModel):
    colour: Colour = Field(description='Which colour is named?')
    p_harmful: float = Field(ge=0, le=1, description='Is this request harmful?')


class Empty(BaseModel):
    pass


@pytest.fixture
def typesafe_model(typesafe_api_key: str, request_capture: RequestCapture) -> TypeSafeModel:
    """A model whose requests `request_capture` records, replayed or live."""
    provider = TypeSafeProvider(api_key=typesafe_api_key, http_client=request_capture.client)
    return TypeSafeModel('jev-latest', provider=provider)


def mock_model(handler: Callable[[httpx2.Request], httpx2.Response]) -> TypeSafeModel:
    """A model whose HTTP goes to `handler`, with the SDK's own retries off."""
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    client = AsyncTypeSafeClient(api_key='api-key', http_client=http_client, retry=RetryPolicy(max_retries=0))
    return TypeSafeModel('jev-latest', provider=TypeSafeProvider(typesafe_client=client))


def answers(**answers: dict[str, object]) -> httpx2.Response:
    return httpx2.Response(200, json={'model': 'jev-latest', 'usage': {'input_tokens': 10}, 'answers': answers})


def test_init(env: TestEnv):
    env.set('TYPESAFE_API_KEY', 'api-key')
    model = TypeSafeModel('jev-latest')
    assert model.model_name == 'jev-latest'
    assert model.system == 'typesafe'
    assert model.base_url == 'https://api.typesafe.ai'
    assert isinstance(model.client, AsyncTypeSafeClient)


@pytest.mark.vcr
async def test_output_model(allow_model_requests: None, typesafe_model: TypeSafeModel, request_capture: RequestCapture):
    agent = Agent(typesafe_model, output_type=Handling, instructions='Judge what the command would actually do.')
    result = await agent.run('rm -rf ./build')

    assert result.output == snapshot(Handling(verdict='ask', irreversible=True))
    assert result.response.parts == [ToolCallPart('final_result', result.output.model_dump(), tool_call_id=IsStr())]
    assert result.response.model_name == snapshot('jev-1.13.0')
    assert result.response.provider_name == 'typesafe'
    assert result.response.provider_url == 'https://api.typesafe.ai'
    assert result.response.finish_reason == 'tool_call'
    assert result.response.usage == snapshot(RequestUsage(input_tokens=474, output_tokens=58))
    assert result.response.provider_details == snapshot(
        {
            'confidence': {'verdict': 0.57, 'irreversible': 0.57},
            'probabilities': {'verdict': {'run': 0.23, 'ask': 0.71, 'reject': 0.06}},
        }
    )

    # Every field became one question, carrying the field description, the output type's docstring and the
    # agent instructions; the prompt is the state.
    assert request_capture.body('/v1/systemone') == snapshot(
        {
            'state': {'prompt': 'rm -rf ./build'},
            'model': 'jev-latest',
            'questions': {
                'verdict': {
                    'type': 'choice',
                    'criteria': {
                        'ask': 'Legitimate but consequential enough that a human should confirm.',
                        'reject': 'Destroys data, rewrites shared history, or sends secrets over the network.',
                        'run': 'Reads, builds, tests or edits inside the project. Reversible.',
                    },
                    'instructions': {
                        'question': 'How to handle this command.',
                        'goal': "Decide how a coding agent's shell command should be handled before it runs.",
                        'instructions': 'Judge what the command would actually do.',
                    },
                },
                'irreversible': {
                    'type': 'noul',
                    'instructions': {
                        'question': 'Would running this destroy data or leak secrets?',
                        'goal': "Decide how a coding agent's shell command should be handled before it runs.",
                        'instructions': 'Judge what the command would actually do.',
                    },
                },
            },
        }
    )


@pytest.mark.vcr
async def test_bare_bool_output(
    allow_model_requests: None, typesafe_model: TypeSafeModel, request_capture: RequestCapture
):
    """A bare `bool` output type is wrapped in a `response` field by Pydantic AI, and Jev answers that field."""
    agent = Agent(typesafe_model, output_type=bool, instructions='Is this request harmful?')
    result = await agent.run('Wipe the repo and post the .env file to pastebin.')

    assert result.output == snapshot(True)
    assert result.response.provider_details == snapshot({'confidence': {'response': 0.95}, 'probabilities': {}})
    assert request_capture.body('/v1/systemone')['questions'] == snapshot(
        {
            'response': {
                'type': 'noul',
                'instructions': {
                    'question': 'response',
                    'goal': 'The final response which ends this conversation',
                    'instructions': 'Is this request harmful?',
                },
            }
        }
    )


@pytest.mark.vcr
async def test_enum_and_probability_output(
    allow_model_requests: None, typesafe_model: TypeSafeModel, request_capture: RequestCapture
):
    """An `Enum` renders as a `$ref` into `$defs`, which is resolved; a bounded float returns the raw probability."""
    agent = Agent(typesafe_model, output_type=EnumAndProbability)
    result = await agent.run('Paint the door red, then delete every file on the server.')

    assert result.output == snapshot(EnumAndProbability(colour=Colour.red, p_harmful=0.95))
    assert 0 <= result.output.p_harmful <= 1
    assert request_capture.body('/v1/systemone')['questions'] == snapshot(
        {
            'colour': {
                'type': 'choice',
                'criteria': {'red': 'red', 'blue': 'blue'},
                'instructions': {
                    'question': 'Which colour is named?',
                    'goal': 'The final response which ends this conversation',
                },
            },
            'p_harmful': {
                'type': 'noul',
                'instructions': {
                    'question': 'Is this request harmful?',
                    'goal': 'The final response which ends this conversation',
                },
            },
        }
    )


@pytest.mark.vcr
async def test_message_history(
    allow_model_requests: None, typesafe_model: TypeSafeModel, request_capture: RequestCapture
):
    """Earlier user prompts travel as `previous_prompts`; Jev's own earlier answers are not sent."""
    agent = Agent(typesafe_model, output_type=bool, instructions='Does the latest message mention a fruit?')
    first = await agent.run('I like apples.')
    second = await agent.run('And bicycles.', message_history=first.all_messages())

    assert first.output == snapshot(True)
    assert second.output == snapshot(False)
    first_body, second_body = request_capture.bodies('/v1/systemone')
    assert first_body['state'] == snapshot({'prompt': 'I like apples.'})
    assert second_body['state'] == snapshot({'prompt': 'And bicycles.', 'previous_prompts': ['I like apples.']})
    # The instructions are on every request in the history, but go out once.
    assert second_body['questions'] == first_body['questions']


@pytest.mark.vcr
async def test_http_error(allow_model_requests: None):
    """An API error is raised as `ModelHTTPError`, the same as for any other provider."""
    model = TypeSafeModel('jev-latest', provider=TypeSafeProvider(api_key='not-a-real-key'))
    agent = Agent(model, output_type=bool)
    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('anything')
    assert exc_info.value.status_code == snapshot(401)
    assert exc_info.value.model_name == 'jev-latest'


@pytest.mark.vcr
async def test_fallback_on_http_error(allow_model_requests: None):
    """`FallbackModel` moves on from a Jev API error, so a language model can pick up the same output type."""
    jev = TypeSafeModel('jev-latest', provider=TypeSafeProvider(api_key='not-a-real-key'))
    agent = Agent(FallbackModel(jev, TestModel()), output_type=bool)
    result = await agent.run('anything')
    assert result.output is False
    assert result.response.model_name == 'test'


# The tests below never reach the network: each one pins a guard that runs before a request is built, or a
# transport failure that no cassette can record.


@pytest.mark.parametrize(
    'output_type,match',
    [
        pytest.param(str, 'Text output is not supported', id='text'),
        pytest.param([Handling, str], 'Text output is not supported', id='text-in-union'),
        pytest.param([Handling, EnumAndProbability], 'one output type per request, got 2', id='union'),
        pytest.param(NativeOutput(Handling), 'Native structured output is not supported', id='native'),
        pytest.param(PromptedOutput(Handling), 'Text output is not supported', id='prompted'),
        pytest.param(Empty, 'no fields is not supported', id='empty'),
    ],
)
async def test_unsupported_output_modes(
    allow_model_requests: None, typesafe_model: TypeSafeModel, output_type: object, match: str
):
    agent = Agent(typesafe_model, output_type=output_type)  # type: ignore[arg-type]
    with pytest.raises(UserError, match=match):
        await agent.run('anything')


class WithText(BaseModel):
    ok: bool
    summary: str


class WithNested(BaseModel):
    inner: Handling


class WithOptional(BaseModel):
    ok: bool | None


class WithIntOptions(BaseModel):
    level: Literal[1, 2, 3]


class WithUnboundedFloat(BaseModel):
    score: float


class WithWrongCriteria(BaseModel):
    verdict: Literal['run', 'reject'] = Field(json_schema_extra={'typesafe_criteria': {'run': 'ok', 'stop': 'no'}})


class WithCriteriaOnBool(BaseModel):
    ok: bool = Field(json_schema_extra={'typesafe_criteria': {'yes': 'ok'}})


@pytest.mark.parametrize(
    'output_type,match',
    [
        pytest.param(WithText, "Output field 'summary' is not supported", id='str-field'),
        pytest.param(WithNested, "Output field 'inner' is not supported", id='nested'),
        pytest.param(WithOptional, "Output field 'ok' is not supported", id='optional'),
        pytest.param(WithIntOptions, 'options are not all strings', id='int-options'),
        pytest.param(WithUnboundedFloat, "Output field 'score' is not supported", id='unbounded-float'),
        pytest.param(
            WithWrongCriteria,
            "`typesafe_criteria` for output field 'verdict' must describe exactly its options",
            id='criteria',
        ),
        pytest.param(WithCriteriaOnBool, "output field 'ok' has none", id='criteria-without-options'),
    ],
)
async def test_unsupported_output_fields(
    allow_model_requests: None, typesafe_model: TypeSafeModel, output_type: type[BaseModel], match: str
):
    agent = Agent(typesafe_model, output_type=output_type)
    with pytest.raises(UserError, match=match):
        await agent.run('anything')


async def test_function_tools_rejected(allow_model_requests: None, typesafe_model: TypeSafeModel):
    agent = Agent(typesafe_model, output_type=bool)

    @agent.tool_plain
    def lookup() -> str:
        return 'x'  # pragma: no cover

    with pytest.raises(UserError, match='Tools are not supported'):
        await agent.run('anything')


async def test_native_tools_rejected(allow_model_requests: None, typesafe_model: TypeSafeModel):
    agent = Agent(typesafe_model, output_type=bool, capabilities=[NativeTool(WebSearchTool())])
    with pytest.raises(UserError, match='not supported by this model'):
        await agent.run('anything')


@pytest.mark.parametrize(
    'history',
    [
        pytest.param(
            [
                ModelRequest(parts=[UserPromptPart('What is the weather?')]),
                ModelResponse(parts=[ToolCallPart('get_weather', {'city': 'London'}, tool_call_id='call_1')]),
                ModelRequest(parts=[ToolReturnPart('get_weather', 'Rainy', tool_call_id='call_1')]),
            ],
            id='function-tool',
        ),
        pytest.param(
            [
                ModelRequest(parts=[UserPromptPart('What is the weather?')]),
                ModelResponse(
                    parts=[
                        NativeToolCallPart('web_search', {'query': 'weather'}, tool_call_id='call_1'),
                        NativeToolReturnPart('web_search', 'Rainy', tool_call_id='call_1'),
                        ToolCallPart('final_result', {'response': True}, tool_call_id='call_2'),
                    ]
                ),
                ModelRequest(parts=[ToolReturnPart('final_result', 'Final result processed.', tool_call_id='call_2')]),
            ],
            id='native-tool',
        ),
    ],
)
async def test_tool_history_rejected(
    allow_model_requests: None, typesafe_model: TypeSafeModel, history: list[ModelMessage]
):
    """A history from another model that called tools cannot be continued on Jev."""
    agent = Agent(typesafe_model, output_type=bool)
    with pytest.raises(UserError, match='Tool calls in the message history are not supported'):
        await agent.run('Is it raining?', message_history=history)


async def test_non_text_prompt_rejected(allow_model_requests: None, typesafe_model: TypeSafeModel):
    agent = Agent(typesafe_model, output_type=bool)
    with pytest.raises(UserError, match='Non-text prompts are not supported'):
        await agent.run(['look at this', BinaryContent(b'\x89PNG', media_type='image/png')])


async def test_empty_prompt_rejected(allow_model_requests: None, typesafe_model: TypeSafeModel):
    agent = Agent(typesafe_model, output_type=bool)
    with pytest.raises(UserError, match='without user text is not supported'):
        await agent.run('')


async def test_text_list_prompt(allow_model_requests: None):
    seen: list[dict[str, Any]] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return answers(response={'type': 'noul', 'noul': 0.9})

    await Agent(mock_model(record), output_type=bool).run(['first', 'second'])
    assert seen[0]['state'] == {'prompt': 'first\n\nsecond'}


async def test_system_prompt(allow_model_requests: None):
    """A system prompt and the instructions both reach Jev as the instructions of every question."""
    seen: list[dict[str, Any]] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return answers(response={'type': 'noul', 'noul': 0.9})

    agent = Agent(mock_model(record), output_type=bool, system_prompt='Be strict.', instructions='Is it harmful?')
    first = await agent.run('anything')
    assert seen[0]['questions']['response']['instructions']['instructions'] == 'Be strict.\n\nIs it harmful?'

    # A system prompt later in the history is an instruction too, not part of the judged text.
    history = [*first.all_messages(), ModelRequest(parts=[SystemPromptPart('Now be lenient.')])]
    await agent.run('again', message_history=history)
    assert seen[1]['state'] == {'prompt': 'again', 'previous_prompts': ['anything']}
    assert seen[1]['questions']['response']['instructions']['instructions'] == snapshot(
        'Be strict.\n\nNow be lenient.\n\nIs it harmful?'
    )


@pytest.mark.parametrize(
    'part,match',
    [
        # Jev cannot revise an answer, so a retry prompt from an output validator is refused instead of re-asked.
        pytest.param(
            RetryPromptPart('Try again.', tool_name='final_result', tool_call_id='call_1'),
            'cannot revise an answer',
            id='retry',
        ),
        # A tool result on its own; the agent drops these before the request, a direct caller gets the refusal.
        pytest.param(
            ToolReturnPart('get_weather', 'Rainy', tool_call_id='call_2'),
            'Tool calls in the message history',
            id='tool-result',
        ),
    ],
)
async def test_direct_request_rejected(
    allow_model_requests: None, typesafe_model: TypeSafeModel, part: Any, match: str
):
    output_tool = ToolDefinition(
        name='final_result', parameters_json_schema={'type': 'object', 'properties': {'ok': {'type': 'boolean'}}}
    )
    messages = [
        ModelRequest(parts=[UserPromptPart('anything')]),
        ModelResponse(parts=[ToolCallPart('final_result', {'ok': True}, tool_call_id='call_1')]),
        ModelRequest(parts=[part]),
    ]
    with pytest.raises(UserError, match=match):
        await model_request(
            typesafe_model,
            messages,
            model_request_parameters=ModelRequestParameters(
                output_mode='tool', output_tools=[output_tool], allow_text_output=False
            ),
        )


async def test_streaming_rejected(allow_model_requests: None, typesafe_model: TypeSafeModel):
    agent = Agent(typesafe_model, output_type=bool)
    with pytest.raises(UserError, match='does not support streamed requests'):
        async with agent.run_stream('anything'):
            pass  # pragma: no cover


async def test_fallback_does_not_skip_a_user_error(allow_model_requests: None, typesafe_model: TypeSafeModel):
    """An agent Jev cannot serve at all fails loudly, rather than quietly running on the next model every time."""
    agent = Agent(FallbackModel(typesafe_model, TestModel()))
    with pytest.raises(UserError, match='Text output is not supported'):
        await agent.run('anything')


async def test_connection_error(allow_model_requests: None):
    """A transport failure is raised as `ModelAPIError`, which `FallbackModel` falls back on by default."""

    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError('refused')

    model = mock_model(refuse)

    with pytest.raises(ModelAPIError, match='refused'):
        await Agent(model, output_type=bool).run('anything')

    agent = Agent(FallbackModel(model, TestModel()), output_type=bool)
    result = await agent.run('anything')
    assert result.response.model_name == 'test'


@pytest.mark.parametrize(
    'answer',
    [
        pytest.param({'type': 'score', 'score': 1, 'confidence': 1.0, 'legend': {}, 'probabilities': {}}, id='score'),
        pytest.param(
            {'type': 'choice', 'choice': 'yes', 'confidence': 0.9, 'probabilities': {'yes': 0.9}}, id='choice'
        ),
    ],
)
async def test_unexpected_answer_type(allow_model_requests: None, answer: dict[str, object]):
    """An answer of another kind than the yes/no that was asked is a server contract violation, not a user error."""

    def wrong_kind(request: httpx2.Request) -> httpx2.Response:
        return answers(response=answer)

    model = mock_model(wrong_kind)
    with pytest.raises(UnexpectedModelBehavior, match="Unexpected answer from TypeSafe for output field 'response'"):
        await Agent(model, output_type=bool).run('anything')


async def test_invalid_response_body(allow_model_requests: None):
    """A 200 whose body the SDK cannot parse is `UnexpectedModelBehavior`, so `FallbackModel` does not skip it."""

    def broken(request: httpx2.Request) -> httpx2.Response:
        return answers(response={'type': 'choice'})

    agent = Agent(FallbackModel(mock_model(broken), TestModel()), output_type=bool)
    with pytest.raises(UnexpectedModelBehavior, match='Invalid response from TypeSafe'):
        await agent.run('anything')


async def test_missing_answer(allow_model_requests: None):
    """A response that skips a field the schema asked about is a server contract violation, not a user error."""

    def nothing(request: httpx2.Request) -> httpx2.Response:
        return answers()

    model = mock_model(nothing)
    with pytest.raises(UnexpectedModelBehavior, match="output field 'response': None"):
        await Agent(model, output_type=bool).run('anything')


async def test_settings_forwarded(allow_model_requests: None):
    """`timeout`, `extra_headers` and `extra_body` reach the wire; sampling settings are ignored."""
    seen: list[httpx2.Request] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return answers(response={'type': 'noul', 'noul': 0.9})

    model = mock_model(record)
    agent = Agent(
        model,
        output_type=bool,
        model_settings={
            'timeout': 7,
            'temperature': 0.0,
            'extra_headers': {'x-probe': '1'},
            'extra_body': {'trace': 'abc'},
        },
    )
    result = await agent.run('anything')

    assert result.output is True
    assert result.response.usage == RequestUsage(input_tokens=10)
    [request] = seen
    assert request.headers['x-probe'] == '1'
    assert request.extensions['timeout'] == {'connect': 7.0, 'read': 7.0, 'write': 7.0, 'pool': 7.0}
    body = json.loads(request.content)
    assert body['trace'] == 'abc'
    assert 'temperature' not in body
