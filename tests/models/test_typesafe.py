from __future__ import annotations as _annotations

import json
from collections.abc import Callable
from enum import Enum, IntEnum
from typing import Any, Literal

import httpx2
import pytest
from pydantic import BaseModel, Field

from pydantic_ai import (
    Agent,
    BinaryContent,
    CachePoint,
    CompactionPart,
    FilePart,
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
    TextContent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    WebSearchTool,
)
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.direct import model_request
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior, UserError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

from .._inline_snapshot import snapshot
from ..conftest import IsStr, RequestCapture, TestEnv, try_import

with try_import() as evals_imports_successful:
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import Classifier

with try_import() as imports_successful:
    from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

    from pydantic_ai.models.typesafe import TypeSafeModel
    from pydantic_ai.providers.typesafe import TypeSafeProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='typesafe-sdk not installed'),
    pytest.mark.anyio,
]


class Verdict(str, Enum):
    """How to handle this command."""

    run = 'run'
    """Reads, builds, tests or edits inside the project. Reversible."""
    reject = 'reject'
    """Destroys data, rewrites shared history, or sends secrets over the network."""
    ask = 'ask'
    """Legitimate but consequential enough that a human should confirm."""


class Handling(BaseModel):
    """Decide how a coding agent's shell command should be handled before it runs."""

    verdict: Verdict
    irreversible: bool = Field(description='Would running this destroy data or leak secrets?')


class Colour(str, Enum):
    red = 'red'
    blue = 'blue'


class EnumAndProbability(BaseModel):
    colour: Colour = Field(description='Which colour is named?')
    p_harmful: float = Field(ge=0, le=1, description='Is this request harmful?')


class Clarity(IntEnum):
    """How clearly does the text explain itself?"""

    unclear = 0
    """Leaves a reader who did not already know none the wiser."""
    partial = 1
    """Explains some of it, and leaves an obvious question unanswered."""
    clear = 2
    """A reader who did not already know could act on it."""


class Review(BaseModel):
    """Grade a piece of writing."""

    clarity: Clarity


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

    assert result.output == snapshot(Handling(verdict=Verdict.ask, irreversible=True))
    assert result.response.parts == [ToolCallPart('final_result', result.output.model_dump(), tool_call_id=IsStr())]
    assert result.response.model_name == snapshot('jev-1.13.0')
    assert result.response.provider_name == 'typesafe'
    assert result.response.provider_url == 'https://api.typesafe.ai'
    assert result.response.finish_reason == 'tool_call'
    assert result.response.usage == snapshot(RequestUsage(input_tokens=474, output_tokens=58))
    assert result.response.provider_details == snapshot(
        {
            'confidence': {'verdict': 0.55, 'irreversible': 0.10000000000000009},
            'probabilities': {'verdict': {'run': 0.17, 'ask': 0.69, 'reject': 0.14}},
            'scores': {},
        }
    )

    # Every field became one question, carrying the field description, the output type's docstring and the
    # agent instructions; the enum member docstrings describe the options; the prompt is the state.
    assert request_capture.body('/v1/systemone') == snapshot(
        {
            'state': 'rm -rf ./build',
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
                        'field': 'verdict',
                        'question': 'How to handle this command.',
                        'goal': "Decide how a coding agent's shell command should be handled before it runs.",
                        'instructions': 'Judge what the command would actually do.',
                    },
                },
                'irreversible': {
                    'type': 'noul',
                    'instructions': {
                        'field': 'irreversible',
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
    assert result.response.provider_details == snapshot(
        {'confidence': {'response': 0.9199999999999999}, 'probabilities': {}, 'scores': {}}
    )
    assert request_capture.body('/v1/systemone')['questions'] == snapshot(
        {
            'response': {
                'type': 'noul',
                'instructions': 'Is this request harmful?',
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

    assert result.output == snapshot(EnumAndProbability(colour=Colour.red, p_harmful=0.96))
    assert 0 <= result.output.p_harmful <= 1
    # A bounded float asks for the probability itself, so the probability is the answer and not also a
    # confidence in it; only the pick-one field reports one.
    assert result.response.provider_details == snapshot(
        {'confidence': {'colour': 1.0}, 'probabilities': {'colour': {'blue': 0.0, 'red': 1.0}}, 'scores': {}}
    )
    assert request_capture.body('/v1/systemone')['questions'] == snapshot(
        {
            'colour': {
                'type': 'choice',
                'criteria': {'red': None, 'blue': None},
                'instructions': {'field': 'colour', 'question': 'Which colour is named?'},
            },
            'p_harmful': {
                'type': 'noul',
                'instructions': {'field': 'p_harmful', 'question': 'Is this request harmful?'},
            },
        }
    )


@pytest.mark.vcr
async def test_rubric_output(
    allow_model_requests: None, typesafe_model: TypeSafeModel, request_capture: RequestCapture
):
    """An `IntEnum` from 0 upwards is Jev's third primitive, a rubric: its member docstrings are the levels."""
    agent = Agent(typesafe_model, output_type=Review)
    result = await agent.run('Jevantic gives Python programs typed, probabilistic decisions from Jev.')

    # Jev put 0.84 on the lowest level for a single sentence out of context, so that is the answer.
    assert result.output == snapshot(Review(clarity=Clarity.unclear))
    # The answer is the level Jev thought most likely; `scores` keeps the expectation across the rubric,
    # which falls between levels and is the number to average over a dataset.
    assert result.response.provider_details == snapshot(
        {
            'confidence': {'clarity': 0.76},
            'probabilities': {'clarity': {'0': 0.84, '1': 0.16, '2': 0.0}},
            'scores': {'clarity': 0.16},
        }
    )

    assert request_capture.body('/v1/systemone')['questions'] == snapshot(
        {
            'clarity': {
                'type': 'score',
                'criteria': [
                    'Leaves a reader who did not already know none the wiser.',
                    'Explains some of it, and leaves an obvious question unanswered.',
                    'A reader who did not already know could act on it.',
                ],
                'instructions': {
                    'field': 'clarity',
                    'question': 'How clearly does the text explain itself?',
                    'goal': 'Grade a piece of writing.',
                },
            }
        }
    )


@pytest.mark.vcr
async def test_message_history(
    allow_model_requests: None, typesafe_model: TypeSafeModel, request_capture: RequestCapture
):
    """Everything before the latest prompt goes along as `history`, Jev's own earlier answer included."""
    agent = Agent(typesafe_model, output_type=bool, instructions='Does the latest message mention a fruit?')
    first = await agent.run('I like apples.')
    second = await agent.run('And bicycles.', message_history=first.all_messages())

    assert first.output == snapshot(True)
    assert second.output == snapshot(False)
    # Jev answered `noul` 0.99 to the first and 0.05 to the second: a confident yes and a confident no. What
    # is reported is confidence in the answer given, not the probability of yes, so both read as ~0.95+ and a
    # threshold means the same thing whichever way the answer went.
    assert first.response.provider_details == snapshot(
        {'confidence': {'response': 0.98}, 'probabilities': {}, 'scores': {}}
    )
    assert second.response.provider_details == snapshot(
        {'confidence': {'response': 0.9}, 'probabilities': {}, 'scores': {}}
    )
    first_body, second_body = request_capture.bodies('/v1/systemone')
    assert first_body['state'] == snapshot('I like apples.')
    assert second_body['state'] == snapshot(
        {
            'history': [
                {'user': 'I like apples.'},
                {'tool_call': {'name': 'final_result', 'args': {'response': True}}},
                {'tool_return': {'name': 'final_result', 'content': 'Final result processed.'}},
            ],
            'text': 'And bicycles.',
        }
    )
    # The instructions are on every request in the history, but go out once.
    assert second_body['questions'] == first_body['questions']


@pytest.mark.vcr
async def test_http_error(allow_model_requests: None):
    """An API error is raised as `ModelHTTPError`, the same as for any other provider."""
    model = TypeSafeModel('jev-latest', provider=TypeSafeProvider(api_key='not-a-real-key'))
    agent = Agent(model, output_type=bool, instructions='Is this fine?')
    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('anything')
    assert exc_info.value.status_code == snapshot(401)
    assert exc_info.value.model_name == 'jev-latest'


@pytest.mark.vcr
async def test_fallback_on_http_error(allow_model_requests: None):
    """`FallbackModel` moves on from a Jev API error, so a language model can pick up the same output type."""
    jev = TypeSafeModel('jev-latest', provider=TypeSafeProvider(api_key='not-a-real-key'))
    agent = Agent(FallbackModel(jev, TestModel()), output_type=bool, instructions='Is this fine?')
    result = await agent.run('anything')
    assert result.output is False
    assert result.response.model_name == 'test'


@pytest.mark.parametrize(
    'noul,answered_by',
    [pytest.param(0.55, 'test', id='unsure'), pytest.param(0.95, 'jev-latest', id='sure')],
)
async def test_fallback_on_low_confidence(allow_model_requests: None, noul: float, answered_by: str):
    """A response handler reads Jev's confidence off the response, so only an unsure answer moves to the next model."""
    jev = mock_model(lambda _: answers(response={'type': 'noul', 'noul': noul}))

    def unsure(response: ModelResponse) -> bool:
        confidence = (response.provider_details or {}).get('confidence', {})
        return any(value < 0.8 for value in confidence.values())

    agent = Agent(FallbackModel(jev, TestModel(), fallback_on=unsure), output_type=bool, instructions='Is this fine?')
    result = await agent.run('anything')
    # `TestModel` reports no confidence, so the same handler passes its answer through.
    assert result.response.model_name == answered_by


# The tests below never reach the network: each one pins a guard that runs before a request is built, or a
# transport failure that no cassette can record.


@pytest.mark.parametrize(
    'output_type,match',
    [
        pytest.param(str, 'Text output is not supported', id='text'),
        pytest.param([Handling, str], 'Text output is not supported', id='text-in-union'),
        pytest.param([Handling, EnumAndProbability], 'Multiple output types are not supported.*got 2', id='union'),
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


class WithUndescribedLevels(BaseModel):
    level: Literal[0, 1, 2]


class OutOfOrder(IntEnum):
    """Declared out of level order; the numbers are what count."""

    clear = 2
    """Top of the rubric."""
    unclear = 0
    """Bottom of the rubric."""
    partial = 1
    """The middle."""


class WithOutOfOrderRubric(BaseModel):
    level: OutOfOrder


class OnlyOne(str, Enum):
    only = 'only'


class WithOneOption(BaseModel):
    only: OnlyOne


class WithUnboundedFloat(BaseModel):
    score: float


class WithUndescribedBool(BaseModel):
    ok: bool


@pytest.mark.parametrize(
    'output_type,match',
    [
        pytest.param(WithText, "Output field 'summary' is not supported", id='str-field'),
        pytest.param(WithNested, "Output field 'inner' is not supported", id='nested'),
        pytest.param(WithOptional, "Output field 'ok' is not supported", id='optional'),
        pytest.param(WithIntOptions, 'a rubric must be the whole numbers from 0 upwards', id='rubric-not-from-0'),
        pytest.param(WithUndescribedLevels, 'every level needs to say what it means', id='rubric-undescribed'),
        pytest.param(WithOneOption, 'options are not two or more strings', id='one-option'),
        pytest.param(WithUnboundedFloat, "Output field 'score' is not supported", id='unbounded-float'),
        pytest.param(bool, "Output field 'response' asks Jev nothing", id='bare-bool-no-question'),
    ],
)
async def test_unsupported_output_fields(
    allow_model_requests: None, typesafe_model: TypeSafeModel, output_type: type[BaseModel] | type[bool], match: str
):
    agent = Agent(typesafe_model, output_type=output_type)
    with pytest.raises(UserError, match=match):
        await agent.run('anything')


async def test_rubric_levels_are_read_in_level_order(allow_model_requests: None):
    """A rubric's levels carry their own numbers, so the order they are declared in says nothing."""
    seen: list[dict[str, Any]] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return answers(
            level={'type': 'score', 'score': 2.0, 'confidence': 0.9, 'legend': {}, 'probabilities': {'2': 1.0}}
        )

    result = await Agent(mock_model(record), output_type=WithOutOfOrderRubric).run('anything')
    assert result.output.level is OutOfOrder.clear
    assert seen[0]['questions']['level']['criteria'] == snapshot(
        ['Bottom of the rubric.', 'The middle.', 'Top of the rubric.']
    )


async def test_unencodable_extra_body_is_a_user_error(allow_model_requests: None):
    """The SDK refusing to send what it was given is the caller's to fix, not a model failure."""

    def unreachable(request: httpx2.Request) -> httpx2.Response:  # pragma: no cover
        raise AssertionError('the request should never be sent')

    agent = Agent(
        mock_model(unreachable),
        output_type=bool,
        instructions='Is this fine?',
        model_settings={'extra_body': {'nope': object()}},
    )
    with pytest.raises(UserError, match='TypeSafe could not send this request'):
        await agent.run('anything')


async def test_function_tools_rejected(allow_model_requests: None, typesafe_model: TypeSafeModel):
    agent = Agent(typesafe_model, output_type=bool, instructions='Is this fine?')

    @agent.tool_plain
    def lookup() -> str:
        return 'x'  # pragma: no cover

    with pytest.raises(UserError, match='Function tools are not supported'):
        await agent.run('anything')


async def test_native_tools_rejected(allow_model_requests: None, typesafe_model: TypeSafeModel):
    agent = Agent(typesafe_model, output_type=bool, capabilities=[NativeTool(WebSearchTool())])
    with pytest.raises(UserError, match='not supported by this model'):
        await agent.run('anything')


async def test_output_validator_retry_gets_the_same_answer(allow_model_requests: None):
    """Jev cannot revise: a `ModelRetry` goes out as history and the same question gets the same answer."""
    seen: list[dict[str, Any]] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return answers(response={'type': 'noul', 'noul': 0.9})

    agent = Agent(mock_model(record), output_type=bool, instructions='Is this request harmful?')

    @agent.output_validator
    def be_sure(output: bool) -> bool:
        if len(seen) == 1:
            raise ModelRetry('Be sure.')
        return output

    result = await agent.run('Delete everything.')

    assert result.output is True
    assert len(seen) == 2
    assert seen[1]['state'] == snapshot(
        {
            'history': [
                {'user': 'Delete everything.'},
                {'tool_call': {'name': 'final_result', 'args': {'response': True}}},
                {
                    'retry': """\
Be sure.

Fix the errors and try again.\
"""
                },
            ]
        }
    )


async def test_history_from_another_model(allow_model_requests: None):
    """A history from a model that called tools is the text under judgment: every part is sent, in order."""
    seen: list[dict[str, Any]] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return answers(response={'type': 'noul', 'noul': 0.9})

    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('What is the weather?')]),
        ModelResponse(
            parts=[
                ThinkingPart('Let me check.'),
                NativeToolCallPart('web_search', {'query': 'weather'}, tool_call_id='call_1'),
                NativeToolReturnPart('web_search', 'Rainy', tool_call_id='call_1'),
                ToolCallPart('get_weather', {'city': 'London'}, tool_call_id='call_2'),
            ]
        ),
        ModelRequest(parts=[ToolReturnPart('get_weather', 'Rainy', tool_call_id='call_2')]),
        ModelResponse(parts=[TextPart('Rain.'), CompactionPart(content=None)]),
        ModelRequest(parts=[RetryPromptPart('Say more.')]),
        ModelResponse(parts=[TextPart('It is raining.'), CompactionPart(content='Weather was discussed.')]),
    ]
    agent = Agent(mock_model(record), output_type=bool, instructions='Was the user told the weather?')
    result = await agent.run('Did the assistant answer?', message_history=history)

    assert result.output is True
    assert seen[0]['state'] == snapshot(
        {
            'history': [
                {'user': 'What is the weather?'},
                {'tool_call': {'name': 'web_search', 'args': {'query': 'weather'}}},
                {'tool_return': {'name': 'web_search', 'content': 'Rainy'}},
                {'tool_call': {'name': 'get_weather', 'args': {'city': 'London'}}},
                {'tool_return': {'name': 'get_weather', 'content': 'Rainy'}},
                {'assistant': 'Rain.'},
                {
                    'retry': """\
Validation feedback:
Say more.

Fix the errors and try again.\
"""
                },
                {'assistant': 'It is raining.'},
                {'summary': 'Weather was discussed.'},
            ],
            'text': 'Did the assistant answer?',
        }
    )


async def test_file_in_history_rejected(allow_model_requests: None, typesafe_model: TypeSafeModel):
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('Draw a cat.')]),
        ModelResponse(parts=[FilePart(BinaryContent(b'\x89PNG', media_type='image/png'))]),
    ]
    agent = Agent(typesafe_model, output_type=bool, instructions='Is this fine?')
    with pytest.raises(UserError, match='Files are not supported'):
        await agent.run('Is it a cat?', message_history=history)


async def test_non_text_prompt_rejected(allow_model_requests: None, typesafe_model: TypeSafeModel):
    agent = Agent(typesafe_model, output_type=bool, instructions='Is this fine?')
    with pytest.raises(UserError, match='Files are not supported'):
        await agent.run(['look at this', BinaryContent(b'\x89PNG', media_type='image/png')])


async def test_empty_prompt_rejected(allow_model_requests: None, typesafe_model: TypeSafeModel):
    agent = Agent(typesafe_model, output_type=bool, instructions='Is this fine?')
    with pytest.raises(UserError, match='without user text is not supported'):
        await agent.run('')


async def test_text_list_prompt(allow_model_requests: None):
    seen: list[dict[str, Any]] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return answers(response={'type': 'noul', 'noul': 0.9})

    await Agent(mock_model(record), output_type=bool, instructions='Is this fine?').run(['first', 'second'])
    # With nothing but the latest text, the state is that text, as TypeSafe's own examples pass it.
    assert seen[0]['state'] == 'first\n\nsecond'


async def test_text_content_and_cache_points_are_text(allow_model_requests: None):
    """`TextContent` is text with metadata attached and a `CachePoint` marks a prefix to cache; neither is a file."""
    seen: list[dict[str, Any]] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return answers(response={'type': 'noul', 'noul': 0.9})

    agent = Agent(mock_model(record), output_type=bool, instructions='Is this fine?')
    await agent.run([TextContent('first', metadata={'source': 'form'}), CachePoint(), 'second'])
    assert seen[0]['state'] == 'first\n\nsecond'


@pytest.mark.parametrize(
    'history',
    [
        pytest.param(
            [
                ModelRequest(parts=[UserPromptPart('Take a photo.')]),
                ModelResponse(parts=[ToolCallPart('camera', {}, 'call-1')]),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            'camera', ['A cat.', BinaryContent(b'\x89PNG', media_type='image/png')], 'call-1'
                        )
                    ]
                ),
            ],
            id='tool_return',
        ),
        pytest.param(
            [
                ModelRequest(parts=[UserPromptPart('Take a photo.')]),
                ModelResponse(
                    parts=[
                        NativeToolReturnPart(
                            'camera', ['A cat.', BinaryContent(b'\x89PNG', media_type='image/png')], 'call-1'
                        )
                    ]
                ),
            ],
            id='native_tool_return',
        ),
    ],
)
async def test_file_in_tool_result_rejected(
    allow_model_requests: None, typesafe_model: TypeSafeModel, history: list[ModelMessage]
):
    """`model_response_str` leaves a tool result's files out, so a result carrying one is refused rather than sent short."""
    agent = Agent(typesafe_model, output_type=bool, instructions='Is this fine?')
    with pytest.raises(UserError, match='a file in a tool result'):
        await agent.run('Is it a cat?', message_history=history)


async def test_system_prompts_are_judged_not_asked(allow_model_requests: None):
    """A system prompt is something that was said, so it joins the state; the question is the instructions.

    Whoever wrote it. Hoisting it into the question meant that judging another agent's run folded that
    agent's persona into what Jev was asked, and nothing on a `SystemPromptPart` says whose it is.
    """
    seen: list[dict[str, Any]] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return answers(response={'type': 'noul', 'noul': 0.9})

    agent = Agent(mock_model(record), output_type=bool, system_prompt='Be strict.', instructions='Is it harmful?')
    first = await agent.run('anything')
    assert seen[0]['questions']['response']['instructions'] == 'Is it harmful?'
    assert seen[0]['state'] == snapshot({'history': [{'system': 'Be strict.'}], 'text': 'anything'})

    # One arriving later in the history is judged the same way, not treated as a new instruction.
    history = [*first.all_messages(), ModelRequest(parts=[SystemPromptPart('Now be lenient.')])]
    await agent.run('again', message_history=history)
    assert seen[1]['questions']['response']['instructions'] == 'Is it harmful?'
    assert seen[1]['state'] == snapshot(
        {
            'history': [
                {'system': 'Be strict.'},
                {'user': 'anything'},
                {'tool_call': {'name': 'final_result', 'args': {'response': True}}},
                {'tool_return': {'name': 'final_result', 'content': 'Final result processed.'}},
                {'system': 'Now be lenient.'},
            ],
            'text': 'again',
        }
    )


async def test_a_judged_agents_persona_stays_out_of_the_question(allow_model_requests: None):
    """The case that motivated it: judging a run whose system prompt someone else wrote."""
    seen: list[dict[str, Any]] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return answers(response={'type': 'noul', 'noul': 0.9})

    judged = Agent(TestModel(custom_output_text='Arrr!'), system_prompt='You are a pirate. Always answer in rhyme.')
    conversation = await judged.run('hello')

    judge = Agent(mock_model(record), output_type=bool, instructions='Was the assistant polite?')
    await judge.run('Judge the conversation above.', message_history=conversation.all_messages())

    assert seen[0]['questions']['response']['instructions'] == 'Was the assistant polite?'
    assert {'system': 'You are a pirate. Always answer in rhyme.'} in seen[0]['state']['history']


async def test_direct_request_without_prompt(allow_model_requests: None):
    """A request whose latest message has no user text still has something to judge: the history."""
    seen: list[dict[str, Any]] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return answers(ok={'type': 'noul', 'noul': 0.9})

    output_tool = ToolDefinition(
        name='final_result', parameters_json_schema={'type': 'object', 'properties': {'ok': {'type': 'boolean'}}}
    )
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('anything')]),
        ModelResponse(parts=[ToolCallPart('final_result', {'ok': True}, tool_call_id='call_1')]),
        ModelRequest(parts=[ToolReturnPart('final_result', 'Final result processed.', tool_call_id='call_1')]),
    ]
    await model_request(
        mock_model(record),
        messages,
        model_request_parameters=ModelRequestParameters(
            output_mode='tool', output_tools=[output_tool], allow_text_output=False
        ),
    )
    assert 'prompt' not in seen[0]['state']
    assert seen[0]['state']['history'][-1] == {
        'tool_return': {'name': 'final_result', 'content': 'Final result processed.'}
    }


async def test_streaming_rejected(allow_model_requests: None, typesafe_model: TypeSafeModel):
    agent = Agent(typesafe_model, output_type=bool, instructions='Is this fine?')
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
        await Agent(model, output_type=bool, instructions='Is this fine?').run('anything')

    agent = Agent(FallbackModel(model, TestModel()), output_type=bool, instructions='Is this fine?')
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
        await Agent(model, output_type=bool, instructions='Is this fine?').run('anything')


async def test_invalid_response_body(allow_model_requests: None):
    """A 200 whose body the SDK cannot parse is `UnexpectedModelBehavior`, so `FallbackModel` does not skip it."""

    def broken(request: httpx2.Request) -> httpx2.Response:
        return answers(response={'type': 'choice'})

    agent = Agent(FallbackModel(mock_model(broken), TestModel()), output_type=bool, instructions='Is this fine?')
    with pytest.raises(UnexpectedModelBehavior, match='Invalid response from TypeSafe'):
        await agent.run('anything')


async def test_missing_answer(allow_model_requests: None):
    """A response that skips a field the schema asked about is a server contract violation, not a user error."""

    def nothing(request: httpx2.Request) -> httpx2.Response:
        return answers()

    model = mock_model(nothing)
    with pytest.raises(UnexpectedModelBehavior, match="output field 'response': None"):
        await Agent(model, output_type=bool, instructions='Is this fine?').run('anything')


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
        instructions='Is this fine?',
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


@pytest.mark.skipif(not evals_imports_successful(), reason='pydantic-evals not installed')
async def test_evals_classifier(allow_model_requests: None):
    """`Classifier` grades every case of a dataset with one Jev request each; the confidence is the reason."""
    seen: list[dict[str, Any]] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        seen.append(body)
        if 'verdict' in body['questions']:
            return answers(
                verdict={'type': 'choice', 'choice': 'ask', 'confidence': 0.69, 'probabilities': {'ask': 0.69}},
                irreversible={'type': 'noul', 'noul': 0.55},
            )
        return answers(response={'type': 'noul', 'noul': 0.93})

    model = mock_model(record)
    dataset = Dataset(
        name='commands',
        cases=[Case(name='build', inputs='clean the build', expected_output='rm -rf ./build')],
        evaluators=[
            Classifier('Is this a safe command?', model=model, include_input=True, evaluation_name='safe'),
            Classifier(output_type=Handling, model=model),
        ],
    )

    report = await dataset.evaluate(lambda command: 'rm -rf ./build')

    [case] = report.cases
    assert {name: (result.value, result.reason) for name, result in case.assertions.items()} == snapshot(
        {'safe': (True, 'confidence 0.93'), 'irreversible': (True, 'confidence 0.55')}
    )
    assert {name: (result.value, result.reason) for name, result in case.labels.items()} == snapshot(
        {'verdict': ('ask', 'confidence 0.69')}
    )
    assert seen == snapshot(
        [
            {
                'state': {
                    'prompt': """\
<Input>
clean the build
</Input>
<Output>
rm -rf ./build
</Output>\
"""
                },
                'model': 'jev-latest',
                'questions': {
                    'response': {'type': 'noul', 'instructions': {'instructions': 'Is this a safe command?'}}
                },
            },
            {
                'state': {
                    'prompt': """\
<Output>
rm -rf ./build
</Output>\
"""
                },
                'model': 'jev-latest',
                'questions': {
                    'verdict': {
                        'type': 'choice',
                        'criteria': {
                            'run': 'Reads, builds, tests or edits inside the project. Reversible.',
                            'reject': 'Destroys data, rewrites shared history, or sends secrets over the network.',
                            'ask': 'Legitimate but consequential enough that a human should confirm.',
                        },
                        'instructions': {
                            'question': 'How to handle this command.',
                            'goal': "Decide how a coding agent's shell command should be handled before it runs.",
                        },
                    },
                    'irreversible': {
                        'type': 'noul',
                        'instructions': {
                            'question': 'Would running this destroy data or leak secrets?',
                            'goal': "Decide how a coding agent's shell command should be handled before it runs.",
                        },
                    },
                },
            },
        ]
    )
