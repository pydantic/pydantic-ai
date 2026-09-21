from __future__ import annotations as _annotations

import json
import pickle
from collections.abc import Callable
from enum import Enum
from typing import Annotated, Any, Literal, cast

import httpx2
import pytest
from pydantic import BaseModel, Field, WithJsonSchema

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
    RunContext,
    SystemPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UseEnumMemberDocstrings,
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

    from pydantic_ai.models.typesafe import ToolCallProposed, TypeSafeModel, TypeSafeModelSettings
    from pydantic_ai.providers.typesafe import TypeSafeProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='typesafe-sdk not installed'),
    pytest.mark.anyio,
]


# Opted in, and the cassettes were recorded with the described options in the request, so Jev answered
# knowing what each verdict means.
class Verdict(UseEnumMemberDocstrings, str, Enum):
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


def rubric(*levels: tuple[int, str]) -> WithJsonSchema:
    """A rubric's levels with a description each, as the schema an `IntEnum` with member docstrings will render."""
    return WithJsonSchema(
        {'type': 'integer', 'anyOf': [{'const': level, 'description': meaning} for level, meaning in levels]}
    )


Clarity = Annotated[
    Literal[0, 1, 2],
    rubric(
        (0, 'Leaves a reader who did not already know none the wiser.'),
        (1, 'Explains some of it, and leaves an obvious question unanswered.'),
        (2, 'A reader who did not already know could act on it.'),
    ),
]


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
    assert result.output == snapshot(Review(clarity=0))
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
        pytest.param(
            [Handling, EnumAndProbability],
            "'final_result_EnumAndProbability' says nothing about itself",
            id='a union member with no docstring',
        ),
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


class WithOptional(BaseModel):
    ok: bool | None


class WithIntOptions(BaseModel):
    level: Literal[1, 2, 3]


class WithUndescribedLevels(BaseModel):
    level: Literal[0, 1, 2]


# Declared out of level order; the numbers are what count.
OutOfOrder = Annotated[
    Literal[2, 0, 1], rubric((2, 'Top of the rubric.'), (0, 'Bottom of the rubric.'), (1, 'The middle.'))
]


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
    assert result.output.level == 2
    assert seen[0]['questions']['level']['criteria'] == snapshot(
        ['Bottom of the rubric.', 'The middle.', 'Top of the rubric.']
    )


@pytest.mark.parametrize(
    'score,level',
    [pytest.param(0.5, 1, id='a half goes up'), pytest.param(2.4, 2, id='past the last level stays on it')],
)
async def test_a_score_between_levels_lands_on_the_nearest(allow_model_requests: None, score: float, level: int):
    jev = mock_model(
        lambda _: answers(
            level={
                'type': 'score',
                'score': score,
                'confidence': 0.9,
                'legend': {},
                'probabilities': {'0': 0.3, '1': 0.4, '2': 0.3},
            }
        )
    )
    result = await Agent(jev, output_type=WithOutOfOrderRubric).run('anything')
    assert result.output.level == level


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


class Ticket(BaseModel):
    """Triage a support ticket."""

    urgent: bool = Field(description='Does this need a reply within the hour?')


def refund(amount: float) -> str:
    """Return a payment to the customer."""
    return f'Refunded {amount}'


def tool_answers(choice: str, probability: float) -> httpx2.Response:
    """Jev's answers to a `Ticket` with `refund` attached: a sure `urgent`, and the tool question as given."""
    rest = round(1 - probability, 2)
    other = 'refund' if choice == 'final_result' else 'final_result'
    return answers(
        urgent={'type': 'noul', 'noul': 0.9},
        tool={
            'type': 'choice',
            'choice': choice,
            'confidence': 0.7,
            'probabilities': {choice: probability, other: rest},
        },
    )


@pytest.mark.vcr
async def test_a_tool_is_proposed_not_called(
    allow_model_requests: None, typesafe_model: TypeSafeModel, request_capture: RequestCapture
):
    """Jev proposes a selected tool whose unbounded numeric argument it cannot fill."""
    agent = Agent(typesafe_model, output_type=Ticket, tools=[refund])
    with pytest.raises(ToolCallProposed) as exc_info:
        await agent.run('You charged my card twice for the same month. Put the second one back.')
    assert exc_info.value.tool_name == 'refund'
    assert exc_info.value.probability == snapshot(1.0)
    assert str(exc_info.value) == snapshot(
        "Jev proposed calling 'refund' (probability 1.00) and cannot call tools itself. Put a model that can behind it: `FallbackModel(jev, llm)` hands it this request."
    )
    assert cast(dict[str, Any], request_capture.body('/v1/systemone')['questions'])['tool'] == snapshot(
        {
            'type': 'choice',
            'criteria': {'final_result': 'Triage a support ticket.', 'refund': 'Return a payment to the customer.'},
            'instructions': 'Which of these does this call for?',
        }
    )


async def test_a_fallback_model_takes_the_proposed_step(allow_model_requests: None):
    """`ToolCallProposed` is a `ModelAPIError`, so the default `FallbackModel` hands the step to the next model."""
    jev = mock_model(lambda _: tool_answers('refund', 0.95))
    called: list[float] = []

    def refund(amount: float) -> str:
        """Return a payment to the customer."""
        called.append(amount)
        return 'Refunded'

    agent = Agent(FallbackModel(jev, TestModel()), output_type=Ticket, tools=[refund])
    result = await agent.run('Charged twice.')
    # The next model took the refund step; with its result in the turn, Jev is not offered `refund` again and
    # fills the output itself.
    assert called == [0]
    assert [message.model_name for message in result.all_messages() if isinstance(message, ModelResponse)] == [
        'test',
        'jev-latest',
    ]
    assert result.output == Ticket(urgent=True)


class ContactPreference(BaseModel):
    method: Literal['email', 'phone']
    urgent_only: bool = Field(description='Should contact be limited to urgent updates?')


async def test_a_tool_with_supported_arguments_is_chosen_then_filled(allow_model_requests: None):
    """The first request picks the tool and the second asks only its arguments using the output-field mapping."""
    seen: list[dict[str, Any]] = []
    called: list[dict[str, Any]] = []

    def configure_contact(
        team: Literal['billing', 'technical'],
        urgent: bool,
        risk: Annotated[float, Field(ge=0, le=1)],
        channels: list[Literal['email', 'sms']],
        window: Literal['morning', 'evening'] | None,
        contact: ContactPreference,
    ) -> str:
        """Configure how the support team should handle this ticket.

        Args:
            team: Which team should handle this ticket?
            urgent: Does this ticket need urgent handling?
            risk: Is this ticket likely to cause customer harm?
            channels: Which channels should receive updates?
            window: Which contact window did the customer request, if any?
            contact: The customer's contact preference.
        """
        called.append(
            {
                'team': team,
                'urgent': urgent,
                'risk': risk,
                'channels': channels,
                'window': window,
                'contact': contact,
            }
        )
        return 'Configured.'

    def record(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        seen.append(body)
        if len(seen) == 1:
            return answers(
                urgent={'type': 'noul', 'noul': 0.9},
                tool={
                    'type': 'choice',
                    'choice': 'configure_contact',
                    'confidence': 0.9,
                    'probabilities': {'final_result': 0.05, 'configure_contact': 0.95},
                },
            )
        if len(seen) == 2:
            nested: dict[str, dict[str, object]] = {
                'channels.email': {'type': 'noul', 'noul': 0.9},
                'channels.sms': {'type': 'noul', 'noul': 0.1},
                'window': {
                    'type': 'choice',
                    'choice': 'none',
                    'confidence': 0.8,
                    'probabilities': {'morning': 0.1, 'evening': 0.1, 'none': 0.8},
                },
                'contact.method': {
                    'type': 'choice',
                    'choice': 'email',
                    'confidence': 0.9,
                    'probabilities': {'email': 0.9, 'phone': 0.1},
                },
                'contact.urgent_only': {'type': 'noul', 'noul': 0.9},
            }
            return answers(
                team={
                    'type': 'choice',
                    'choice': 'technical',
                    'confidence': 0.9,
                    'probabilities': {'billing': 0.1, 'technical': 0.9},
                },
                urgent={'type': 'noul', 'noul': 0.95},
                risk={'type': 'noul', 'noul': 0.8},
                **nested,
            )
        return answers(urgent={'type': 'noul', 'noul': 0.9})

    agent = Agent(
        mock_model(record),
        output_type=Ticket,
        tools=[configure_contact],
        instructions='Handle the customer request as written.',
    )
    result = await agent.run('Technical support should email me about urgent updates. No contact window specified.')

    assert result.output == Ticket(urgent=True)
    assert called == snapshot(
        [
            {
                'team': 'technical',
                'urgent': True,
                'risk': 0.8,
                'channels': ['email'],
                'window': None,
                'contact': ContactPreference(method='email', urgent_only=True),
            }
        ]
    )
    responses = [message for message in result.all_messages() if isinstance(message, ModelResponse)]
    assert responses[0].usage == RequestUsage(input_tokens=20)
    assert responses[0].provider_details == snapshot(
        {
            'confidence': {
                'team': 0.9,
                'urgent': 0.8999999999999999,
                'channels': 0.8,
                'window': 0.8,
                'contact.method': 0.9,
                'contact.urgent_only': 0.8,
            },
            'probabilities': {
                'team': {'billing': 0.1, 'technical': 0.9},
                'channels': {'email': 0.9, 'sms': 0.1},
                'window': {'morning': 0.1, 'evening': 0.1, 'none': 0.8},
                'contact.method': {'email': 0.9, 'phone': 0.1},
            },
            'scores': {},
            'tool': {
                'choice': 'configure_contact',
                'probabilities': {'final_result': 0.05, 'configure_contact': 0.95},
                'offered': ['configure_contact'],
            },
            'requests': 2,
        }
    )
    assert list(seen[0]['questions']) == ['urgent', 'tool']
    assert seen[1]['questions'] == snapshot(
        {
            'team': {
                'type': 'choice',
                'criteria': {'billing': None, 'technical': None},
                'instructions': {
                    'field': 'team',
                    'question': 'Which team should handle this ticket?',
                    'goal': 'Configure how the support team should handle this ticket.',
                    'instructions': 'Handle the customer request as written.',
                },
            },
            'urgent': {
                'type': 'noul',
                'instructions': {
                    'field': 'urgent',
                    'question': 'Does this ticket need urgent handling?',
                    'goal': 'Configure how the support team should handle this ticket.',
                    'instructions': 'Handle the customer request as written.',
                },
            },
            'risk': {
                'type': 'noul',
                'instructions': {
                    'field': 'risk',
                    'question': 'Is this ticket likely to cause customer harm?',
                    'goal': 'Configure how the support team should handle this ticket.',
                    'instructions': 'Handle the customer request as written.',
                },
            },
            'channels.email': {
                'type': 'noul',
                'instructions': {
                    'field': 'channels',
                    'question': 'Which channels should receive updates?',
                    'goal': 'Configure how the support team should handle this ticket.',
                    'instructions': 'Handle the customer request as written.',
                    'option': 'email',
                },
            },
            'channels.sms': {
                'type': 'noul',
                'instructions': {
                    'field': 'channels',
                    'question': 'Which channels should receive updates?',
                    'goal': 'Configure how the support team should handle this ticket.',
                    'instructions': 'Handle the customer request as written.',
                    'option': 'sms',
                },
            },
            'window': {
                'type': 'choice',
                'criteria': {'morning': None, 'evening': None, 'none': 'None of these.'},
                'instructions': {
                    'field': 'window',
                    'question': 'Which contact window did the customer request, if any?',
                    'goal': 'Configure how the support team should handle this ticket.',
                    'instructions': 'Handle the customer request as written.',
                },
            },
            'contact.method': {
                'type': 'choice',
                'criteria': {'email': None, 'phone': None},
                'instructions': {
                    'field': 'contact.method',
                    'goal': 'Configure how the support team should handle this ticket.',
                    'instructions': 'Handle the customer request as written.',
                },
            },
            'contact.urgent_only': {
                'type': 'noul',
                'instructions': {
                    'field': 'contact.urgent_only',
                    'question': 'Should contact be limited to urgent updates?',
                    'goal': 'Configure how the support team should handle this ticket.',
                    'instructions': 'Handle the customer request as written.',
                },
            },
        }
    )
    assert list(seen[2]['questions']) == ['urgent']


async def test_a_selected_tool_fill_failure_does_not_fall_back_to_another_route(allow_model_requests: None):
    """Once Jev selected a tool, a failed fill is terminal rather than replaying the whole step on the fallback."""
    seen = 0

    def record(request: httpx2.Request) -> httpx2.Response:
        nonlocal seen
        seen += 1
        if seen == 1:
            return answers(
                urgent={'type': 'noul', 'noul': 0.9},
                tool={
                    'type': 'choice',
                    'choice': 'set_direction',
                    'confidence': 0.9,
                    'probabilities': {'final_result': 0.05, 'set_direction': 0.95},
                },
            )
        return httpx2.Response(503, json={'detail': 'temporarily unavailable'})

    def set_direction(direction: Literal['left', 'right']) -> None:
        """Set the direction to take.

        Args:
            direction: Which direction should be taken?
        """

    agent = Agent(FallbackModel(mock_model(record), TestModel()), output_type=Ticket, tools=[set_direction])
    with pytest.raises(UnexpectedModelBehavior, match=r"selected 'set_direction'.*failed while filling"):
        await agent.run('Go left.')
    assert seen == 2


@pytest.mark.vcr
async def test_tool_arguments_live(
    allow_model_requests: None, typesafe_model: TypeSafeModel, request_capture: RequestCapture
):
    """The live API selects an argument-taking tool, then fills its supported argument."""
    output_tool = ToolDefinition(
        name='final_result',
        description='Classify a message that does not request a navigation action.',
        kind='output',
        parameters_json_schema={
            'type': 'object',
            'properties': {'urgent': {'type': 'boolean', 'description': 'Is this message urgent?'}},
            'required': ['urgent'],
        },
    )
    function_tool = ToolDefinition(
        name='set_direction',
        description='Set the navigation direction requested in the message.',
        parameters_json_schema={
            'type': 'object',
            'properties': {
                'direction': {
                    'type': 'string',
                    'enum': ['left', 'right'],
                    'description': 'Which direction should be taken?',
                }
            },
            'required': ['direction'],
        },
    )
    response = await model_request(
        typesafe_model,
        [ModelRequest(parts=[UserPromptPart('At the fork, take the left path. Set our direction accordingly.')])],
        model_request_parameters=ModelRequestParameters(
            output_mode='tool',
            output_tools=[output_tool],
            function_tools=[function_tool],
            allow_text_output=False,
        ),
    )

    assert response.parts == [ToolCallPart('set_direction', {'direction': 'left'}, tool_call_id=IsStr())]
    assert response.provider_details == snapshot(
        {
            'confidence': {'direction': 1.0},
            'probabilities': {'direction': {'left': 1.0, 'right': 0.0}},
            'scores': {},
            'tool': {
                'choice': 'set_direction',
                'probabilities': {'final_result': 0.0, 'set_direction': 1.0},
                'offered': ['set_direction'],
            },
            'requests': 2,
        }
    )
    first, second = request_capture.bodies('/v1/systemone')
    assert list(cast(dict[str, Any], first['questions'])) == ['urgent', 'tool']
    assert list(cast(dict[str, Any], second['questions'])) == ['direction']


@pytest.mark.parametrize(
    'probability,settings',
    [
        pytest.param(0.59, None, id='below the default threshold'),
        pytest.param(0.9, {'typesafe_tool_call_threshold': 0.95}, id='below a raised threshold'),
    ],
)
async def test_a_tool_below_the_threshold_is_a_lean(
    allow_model_requests: None, probability: float, settings: dict[str, float] | None
):
    """A tool picked below the threshold does not end the request; the output is filled and the lean is reported."""
    jev = mock_model(lambda _: tool_answers('refund', probability))
    agent = Agent(jev, output_type=Ticket, tools=[refund], model_settings=settings)  # type: ignore[arg-type]
    result = await agent.run('Charged twice.')
    assert result.output == Ticket(urgent=True)
    assert result.response.provider_details == snapshot(
        {
            'confidence': {'urgent': 0.8},
            'probabilities': {},
            'scores': {},
            'tool': {
                'choice': 'refund',
                'probabilities': {'refund': probability, 'final_result': round(1 - probability, 2)},
                'offered': ['refund'],
            },
        }
    )


async def test_the_output_tool_is_one_of_the_options(allow_model_requests: None):
    jev = mock_model(lambda _: tool_answers('final_result', 0.9))
    result = await Agent(jev, output_type=Ticket, tools=[refund]).run('Is my invoice due?')
    assert result.output == Ticket(urgent=True)
    assert (result.response.provider_details or {})['tool'] == {
        'choice': 'final_result',
        'probabilities': {'final_result': 0.9, 'refund': 0.1},
        'offered': ['refund'],
    }


async def test_the_tool_question_stays_clear_of_a_field_named_tool(allow_model_requests: None):
    seen: list[dict[str, Any]] = []

    class Uses(BaseModel):
        """Say what a text is about."""

        tool: bool = Field(description='Does it mention a tool?')

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return answers(
            tool={'type': 'noul', 'noul': 0.9},
            tool_={
                'type': 'choice',
                'choice': 'final_result',
                'confidence': 0.9,
                'probabilities': {'final_result': 0.9, 'refund': 0.1},
            },
        )

    result = await Agent(mock_model(record), output_type=Uses, tools=[refund]).run('A hammer.')
    assert result.output == Uses(tool=True)
    assert list(seen[0]['questions']) == ['tool', 'tool_']


@pytest.mark.parametrize(
    'tool,match',
    [
        pytest.param(
            {'type': 'noul', 'noul': 0.9}, 'Unexpected answer from TypeSafe for the tool question', id='not a choice'
        ),
        pytest.param(
            {'type': 'choice', 'choice': 'refund', 'confidence': 0.8, 'probabilities': {'final_result': 0.1}},
            'Unexpected answer from TypeSafe for the tool question',
            id='no probability for the choice',
        ),
        pytest.param(
            {'type': 'choice', 'choice': 'cancel', 'confidence': 0.8, 'probabilities': {'cancel': 0.9}},
            "TypeSafe picked a tool it was not offered: 'cancel'",
            id='a tool that was not offered',
        ),
        pytest.param(
            {
                'type': 'choice',
                'choice': 'refund',
                'confidence': 0.8,
                'probabilities': {'refund': 1.7, 'final_result': -0.7},
            },
            'Unexpected answer from TypeSafe for the tool question',
            id='a probability outside 0 to 1',
        ),
    ],
)
async def test_an_unexpected_tool_answer(allow_model_requests: None, tool: dict[str, object], match: str):
    jev = mock_model(lambda _: answers(urgent={'type': 'noul', 'noul': 0.9}, tool=tool))
    with pytest.raises(UnexpectedModelBehavior, match=match):
        await Agent(jev, output_type=Ticket, tools=[refund]).run('anything')


async def test_a_withheld_tool_is_not_offered(allow_model_requests: None):
    """A tool hidden until revealed is not on any wire, so it is not among Jev's options either."""
    seen: list[dict[str, Any]] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return tool_answers('final_result', 0.9)

    agent = Agent(mock_model(record), output_type=Ticket, tools=[approve])
    agent.tool_plain(defer_loading=True)(reject)
    await agent.run('Fine by me.')
    criteria = seen[0]['questions']['tool']['criteria']
    assert 'approve' in criteria and 'reject' not in criteria


@pytest.mark.parametrize('threshold', [-0.1, 1.5, float('nan')])
async def test_a_threshold_outside_zero_to_one_is_refused_before_the_request(
    allow_model_requests: None, typesafe_model: TypeSafeModel, threshold: float
):
    agent = Agent(typesafe_model, output_type=Ticket, tools=[refund])
    with pytest.raises(UserError, match='`typesafe_tool_call_threshold` must be between 0 and 1'):
        await agent.run('anything', model_settings=TypeSafeModelSettings(typesafe_tool_call_threshold=threshold))


@pytest.mark.parametrize('threshold', [-0.1, 1.5, float('nan')])
async def test_a_boolean_threshold_outside_zero_to_one_is_refused_before_the_request(
    allow_model_requests: None, typesafe_model: TypeSafeModel, threshold: float
):
    agent = Agent(typesafe_model, output_type=Ticket)
    with pytest.raises(UserError, match='`typesafe_boolean_threshold` must be between 0 and 1'):
        await agent.run('anything', model_settings=TypeSafeModelSettings(typesafe_boolean_threshold=threshold))


@pytest.mark.parametrize(
    'threshold,expected,expected_confidence',
    [
        pytest.param(None, True, 0.3999999999999999, id='the default rounds a 0.7 to yes'),
        pytest.param(0.5, True, 0.3999999999999999, id='the default, passed explicitly'),
        pytest.param(0.75, False, 0.06666666666666672, id='a raised bar turns the same answer into a no'),
        pytest.param(0.7, True, 0.0, id='an answer exactly at the bar is a yes, and the least sure one'),
        pytest.param(0.0, True, 0.7, id='a bar of zero takes every answer as a yes'),
        pytest.param(1.0, False, 0.30000000000000004, id='a bar of one takes nothing short of certainty'),
    ],
)
async def test_the_boolean_threshold_decides_what_a_probability_of_yes_rounds_to(
    allow_model_requests: None, threshold: float | None, expected: bool, expected_confidence: float
):
    """What `True` has to mean is the user's to choose, and confidence is the distance from their bar."""

    def record(request: httpx2.Request) -> httpx2.Response:
        return answers(urgent={'type': 'noul', 'noul': 0.7})

    settings = None if threshold is None else TypeSafeModelSettings(typesafe_boolean_threshold=threshold)
    agent = Agent(mock_model(record), output_type=Ticket)
    result = await agent.run('Is this urgent?', model_settings=settings)

    assert result.output == Ticket(urgent=expected)
    assert result.response.provider_details == {
        'confidence': {'urgent': expected_confidence},
        'probabilities': {},
        'scores': {},
    }


async def test_the_boolean_threshold_applies_to_each_option_of_a_list(allow_model_requests: None):
    """A list of options is one yes/no per option, so the same bar decides each of them."""

    class Routing(BaseModel):
        """Route a support ticket."""

        channels: list[Literal['email', 'sms']] = Field(description='Which channels should receive updates?')

    def record(request: httpx2.Request) -> httpx2.Response:
        options: dict[str, dict[str, object]] = {
            'channels.email': {'type': 'noul', 'noul': 0.7},
            'channels.sms': {'type': 'noul', 'noul': 0.6},
        }
        return answers(**options)

    agent = Agent(mock_model(record), output_type=Routing)
    assert (await agent.run('x')).output == Routing(channels=['email', 'sms'])

    result = await agent.run('x', model_settings=TypeSafeModelSettings(typesafe_boolean_threshold=0.65))
    assert result.output == Routing(channels=['email'])
    # The field is as sure as its least sure option, which is the `sms` that only just missed the bar.
    assert (result.response.provider_details or {})['confidence'] == {'channels': snapshot(0.07692307692307698)}


async def test_the_boolean_threshold_leaves_a_probability_field_alone(allow_model_requests: None):
    """A `float` bounded 0 to 1 asks for the probability itself, so there is nothing to round."""

    class Scored(BaseModel):
        """Score a support ticket."""

        risk: float = Field(ge=0, le=1, description='Is this ticket likely to cause customer harm?')

    def record(request: httpx2.Request) -> httpx2.Response:
        return answers(risk={'type': 'noul', 'noul': 0.7})

    agent = Agent(mock_model(record), output_type=Scored)
    result = await agent.run('x', model_settings=TypeSafeModelSettings(typesafe_boolean_threshold=0.95))

    assert result.output == Scored(risk=0.7)
    assert (result.response.provider_details or {})['confidence'] == {}


async def test_an_output_type_with_nothing_said_about_it_cannot_be_weighed_against_tools(
    allow_model_requests: None, typesafe_model: TypeSafeModel
):
    with pytest.raises(UserError, match='Give the output type a docstring'):
        await Agent(typesafe_model, output_type=Undescribed, tools=[refund]).run('anything')


async def test_the_instructions_describe_an_output_type_without_a_docstring(allow_model_requests: None):
    """The stock output tool description never goes to Jev; what the user wrote does."""
    seen: list[dict[str, Any]] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return tool_answers('final_result', 0.9)

    agent = Agent(mock_model(record), output_type=Undescribed, tools=[refund], instructions='Triage the ticket.')
    await agent.run('anything')
    assert seen[0]['questions']['tool']['criteria'] == snapshot(
        {'final_result': 'Triage the ticket.', 'refund': 'Return a payment to the customer.'}
    )


async def test_a_tool_that_asked_for_a_retry_stays_on_offer(allow_model_requests: None):
    """A call with no result is not a call made: `ModelRetry` from the tool leaves it on offer."""
    seen: list[dict[str, Any]] = []
    attempts = 0

    def flaky() -> str:
        """Try the flaky thing."""
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ModelRetry('Busy, try again.')
        return 'Done on the second try.'

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        if len(seen) < 3:
            return answers(urgent={'type': 'noul', 'noul': 0.9}, tool=tool_answers_for('flaky'))
        return answers(urgent={'type': 'noul', 'noul': 0.9}, tool=tool_answers_for('final_result'))

    result = await Agent(mock_model(record), output_type=Ticket, tools=[flaky], retries=2).run('Try it.')
    assert attempts == 2 and result.output == Ticket(urgent=True)
    assert [list(request['questions'].get('tool', {}).get('criteria', {})) for request in seen] == snapshot(
        [['final_result', 'flaky'], ['final_result', 'flaky'], []]
    )


async def test_the_last_route_left_is_taken_without_asking(allow_model_requests: None):
    """Output functions only: once every other option has returned, the one left is the answer, with no request."""
    seen: list[dict[str, Any]] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return answers(tool=tool_answers_for('approve'))

    result = await Agent(mock_model(record), output_type=[reject], tools=[approve]).run('Decide.')
    assert result.output == 'rejected'
    assert len(seen) == 1
    assert (result.response.provider_details or {})['tool'] == snapshot(
        {'choice': 'final_result', 'probabilities': {'final_result': 1.0}, 'offered': ['final_result']}
    )


async def test_a_tool_that_returned_is_not_proposed_again(allow_model_requests: None):
    """A model behind Jev took the refund; with its result in the turn, Jev is not asked about `refund` again."""
    seen: list[dict[str, Any]] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return answers(urgent={'type': 'noul', 'noul': 0.9})

    history = [
        ModelRequest(parts=[UserPromptPart('Charged twice.')]),
        ModelResponse(parts=[ToolCallPart('refund', {'amount': 10}, 'call_1')]),
        ModelRequest(parts=[ToolReturnPart('refund', 'Refunded', 'call_1')]),
    ]
    await Agent(mock_model(record), output_type=Ticket, tools=[refund]).run(message_history=history)
    assert 'tool' not in seen[0]['questions']


class Undescribed(BaseModel):
    urgent: bool = Field(description='Does this need a reply within the hour?')


async def test_the_composed_stock_description_is_no_description_either(
    allow_model_requests: None, typesafe_model: TypeSafeModel
):
    """With several output types, the framework composes `Name: <stock description>`; that is still nothing said."""
    with pytest.raises(UserError, match='Give the output type a docstring'):
        await Agent(typesafe_model, output_type=[Undescribed, approve]).run('anything')


async def test_below_the_threshold_with_nothing_to_fill_the_likeliest_hand_off_is_taken(
    allow_model_requests: None,
):
    probabilities = {'refund': 0.4, 'final_result_approve': 0.35, 'final_result_reject': 0.25}
    jev = mock_model(
        lambda _: answers(
            tool={'type': 'choice', 'choice': 'refund', 'confidence': 0.1, 'probabilities': probabilities}
        )
    )
    result = await Agent(jev, output_type=[approve, reject], tools=[refund]).run('Looks fine.')
    assert result.output == 'approved'
    assert (result.response.provider_details or {})['tool']['taken'] == 'final_result_approve'


async def test_a_streamed_run_can_be_cancelled_early(allow_model_requests: None):
    jev = mock_model(lambda _: answers(urgent={'type': 'noul', 'noul': 0.9}))
    async with Agent(jev, output_type=Ticket).run_stream('Cancel me.') as stream:
        await stream.cancel()


def test_tool_call_proposed_pickles():
    exc = pickle.loads(pickle.dumps(ToolCallProposed('jev-latest', 'refund', 0.9)))
    assert (exc.model_name, exc.tool_name, exc.probability) == ('jev-latest', 'refund', 0.9)


def approve() -> str:
    """Approve the request as it stands."""
    return 'approved'


def reject() -> str:
    """Turn the request down."""
    return 'rejected'


async def escalate(ctx: RunContext[None]) -> str:
    """Hand the ticket to a person on the support team."""
    return f'escalated after {len(ctx.messages)} messages'


@pytest.mark.vcr
async def test_an_output_function_is_a_hand_off_jev_picks(
    allow_model_requests: None, typesafe_model: TypeSafeModel, request_capture: RequestCapture
):
    """An output function that takes only the run context is an option beside the output type, and Jev can pick it."""
    agent = Agent(typesafe_model, output_type=[Ticket, escalate])
    result = await agent.run('I have explained this to your bot four times. I want a person to call me back today.')
    assert result.output == snapshot('escalated after 2 messages')
    assert (result.response.provider_details or {})['tool'] == snapshot(
        {
            'choice': 'final_result_escalate',
            'probabilities': {'final_result_escalate': 1.0, 'final_result_Ticket': 0.0},
            'offered': ['final_result_escalate'],
        }
    )
    assert cast(dict[str, Any], request_capture.body('/v1/systemone')['questions'])['tool']['criteria'] == snapshot(
        {
            'final_result_Ticket': 'Triage a support ticket.',
            'final_result_escalate': 'Hand the ticket to a person on the support team.',
        }
    )


async def test_an_arg_less_tool_is_called_by_jev_itself(allow_model_requests: None):
    """A tool with no arguments has nothing for Jev to write, so Jev calls it and judges the result next request."""
    seen: list[dict[str, Any]] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        # Jev would pick the tool again with its result in view; it is not offered again, so this is ignored.
        return answers(
            urgent={'type': 'noul', 'noul': 0.9},
            tool={'type': 'choice', 'choice': 'approve', 'confidence': 0.8, 'probabilities': {'approve': 0.9}},
        )

    result = await Agent(mock_model(record), output_type=Ticket, tools=[approve]).run('Fine by me.')
    assert result.output == Ticket(urgent=True)
    assert 'tool' in seen[0]['questions'] and 'tool' not in seen[1]['questions']
    assert seen[1]['state'] == snapshot(
        {
            'history': [
                {'user': 'Fine by me.'},
                {'tool_call': {'name': 'approve', 'args': {}}},
                {'tool_return': {'name': 'approve', 'content': 'approved'}},
            ]
        }
    )


async def test_a_tool_called_in_an_earlier_turn_is_offered_again(allow_model_requests: None):
    """Once per turn: a new user prompt is a new turn, and a call in another agent's run is not this one's."""
    seen: list[dict[str, Any]] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return answers(urgent={'type': 'noul', 'noul': 0.9}, tool=tool_answers_for('final_result'))

    agent = Agent(mock_model(record), output_type=Ticket, tools=[approve])
    first = await agent.run('Fine by me.')
    earlier = [
        *first.all_messages(),
        ModelResponse(parts=[ToolCallPart('approve', {}, 'call_1')]),
        ModelRequest(parts=[ToolReturnPart('approve', 'approved', 'call_1')]),
    ]
    await agent.run('And this one?', message_history=earlier)
    assert 'approve' in seen[-1]['questions']['tool']['criteria']


def tool_answers_for(choice: str) -> dict[str, object]:
    return {'type': 'choice', 'choice': choice, 'confidence': 0.8, 'probabilities': {choice: 0.9}}


async def test_with_nothing_to_fill_the_pick_is_the_answer(allow_model_requests: None):
    """Output functions and no output type: the tool question is the whole question, taken at any probability."""
    probabilities = {'final_result_approve': 0.55, 'final_result_reject': 0.45}
    jev = mock_model(
        lambda _: answers(
            tool={'type': 'choice', 'choice': 'final_result_approve', 'confidence': 0.1, 'probabilities': probabilities}
        )
    )
    result = await Agent(jev, output_type=[approve, reject]).run('Looks fine.')
    assert result.output == 'approved'


async def test_the_last_route_left_is_proposed_when_its_arguments_are_unsupported(allow_model_requests: None):
    """The forced route still becomes a proposal when its argument schema cannot be expressed."""

    def unasked(request: httpx2.Request) -> httpx2.Response:  # pragma: no cover
        raise AssertionError('Jev was asked a question when there was nothing left to ask about.')

    history = [
        ModelRequest(parts=[UserPromptPart('Charged twice.')]),
        ModelResponse(parts=[ToolCallPart('approve', {}, 'call_1')]),
        ModelRequest(parts=[ToolReturnPart('approve', 'approved', 'call_1')]),
        ModelResponse(parts=[ToolCallPart('final_result', {}, 'call_2')]),
        ModelRequest(parts=[ToolReturnPart('final_result', 'rejected', 'call_2')]),
    ]
    agent = Agent(mock_model(unasked), output_type=[reject], tools=[approve, refund])
    with pytest.raises(ToolCallProposed) as exc_info:
        await agent.run(message_history=history)
    assert (exc_info.value.tool_name, exc_info.value.probability) == ('refund', 1.0)


async def test_the_last_route_left_has_its_supported_arguments_filled(allow_model_requests: None):
    """Skipping a choice request still leaves one argument request for the forced route."""
    seen: list[dict[str, Any]] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return answers(
            direction={
                'type': 'choice',
                'choice': 'left',
                'confidence': 0.9,
                'probabilities': {'left': 0.9, 'right': 0.1},
            }
        )

    output_tool = ToolDefinition(name='final_result', description='Finish.', kind='output')
    function_tool = ToolDefinition(
        name='set_direction',
        description='Set the direction to take.',
        parameters_json_schema={
            'type': 'object',
            'properties': {
                'direction': {
                    'type': 'string',
                    'enum': ['left', 'right'],
                    'description': 'Which direction should be taken?',
                }
            },
            'required': ['direction'],
        },
    )
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('Go left.')]),
        ModelResponse(parts=[ToolCallPart('final_result', {}, 'call_1')]),
        ModelRequest(parts=[ToolReturnPart('final_result', 'Returned.', 'call_1')]),
    ]

    response = await model_request(
        mock_model(record),
        messages,
        model_request_parameters=ModelRequestParameters(
            output_mode='tool',
            output_tools=[output_tool],
            function_tools=[function_tool],
            allow_text_output=False,
        ),
    )

    assert response.parts == [ToolCallPart('set_direction', {'direction': 'left'}, tool_call_id=IsStr())]
    assert response.usage == RequestUsage(input_tokens=10)
    assert response.provider_details == snapshot(
        {
            'tool': {
                'choice': 'set_direction',
                'probabilities': {'set_direction': 1.0},
                'offered': ['set_direction'],
            },
            'confidence': {'direction': 0.9},
            'probabilities': {'direction': {'left': 0.9, 'right': 0.1}},
            'scores': {},
        }
    )
    assert len(seen) == 1
    assert seen[0]['questions'] == snapshot(
        {
            'direction': {
                'type': 'choice',
                'criteria': {'left': None, 'right': None},
                'instructions': {
                    'field': 'direction',
                    'question': 'Which direction should be taken?',
                    'goal': 'Set the direction to take.',
                },
            }
        }
    )


async def test_below_the_threshold_with_no_hand_off_left_the_pick_stands(allow_model_requests: None):
    """Below the threshold, with nothing to fill and every output function returned, the lean is taken anyway."""
    jev = mock_model(
        lambda _: answers(
            tool={
                'type': 'choice',
                'choice': 'refund',
                'confidence': 0.2,
                'probabilities': {'refund': 0.55, 'approve': 0.45},
            }
        )
    )
    history = [
        ModelRequest(parts=[UserPromptPart('Charged twice.')]),
        ModelResponse(parts=[ToolCallPart('final_result', {}, 'call_1')]),
        ModelRequest(parts=[ToolReturnPart('final_result', 'rejected', 'call_1')]),
    ]
    agent = Agent(jev, output_type=[reject], tools=[approve, refund])
    with pytest.raises(ToolCallProposed) as exc_info:
        await agent.run(message_history=history)
    assert (exc_info.value.tool_name, exc_info.value.probability) == ('refund', 0.55)


async def test_a_field_with_more_options_than_jev_picks_from_is_refused(
    allow_model_requests: None, typesafe_model: TypeSafeModel
):
    """Jev takes at most 255 options in one question; a 256th is a 400, so it is refused before the request."""

    class Routed(BaseModel):
        """Route the ticket."""

        area: Literal[tuple(f'area_{i:03d}' for i in range(256))] = Field(description='Which team owns it?')  # type: ignore[valid-type]

    with pytest.raises(UserError, match='picks from at most 255 options, and this one has 256'):
        await Agent(typesafe_model, output_type=Routed).run('anything')


async def test_more_routes_than_jev_picks_from_are_refused(allow_model_requests: None, typesafe_model: TypeSafeModel):
    """The output type is one route beside the tools, so 255 tools is already one too many."""
    tools = [_named_tool(f'tool_{i:03d}') for i in range(255)]
    with pytest.raises(UserError, match='being offered 256 routes'):
        await Agent(typesafe_model, output_type=Ticket, tools=tools).run('anything')


def _named_tool(name: str) -> Callable[[], str]:
    def tool() -> str:
        return 'done'  # pragma: no cover

    tool.__name__ = name
    tool.__doc__ = f'Handle {name}.'
    return tool


async def test_a_tool_with_unsupported_arguments_is_proposed_even_with_nothing_to_fill(allow_model_requests: None):
    jev = mock_model(
        lambda _: answers(
            tool={
                'type': 'choice',
                'choice': 'refund',
                'confidence': 0.2,
                'probabilities': {'refund': 0.6, 'final_result_approve': 0.4},
            }
        )
    )
    with pytest.raises(ToolCallProposed) as exc_info:
        await Agent(jev, output_type=[approve], tools=[refund]).run('Give me my money back.')
    assert exc_info.value.probability == 0.6


class Customer(BaseModel):
    """About the customer."""

    angry: bool = Field(description='Is the customer angry?')


# Opted in, and the cassette was recorded with the description on the `billing` option.
class Area(UseEnumMemberDocstrings, str, Enum):
    billing = 'billing'
    """Money already owed, charged or refunded."""
    account = 'account'
    bug = 'bug'


class Triage(BaseModel):
    """Triage a support ticket."""

    customer: Customer
    areas: list[Area] = Field(description='Which teams does this touch?')
    plan: Literal['free', 'pro', 'enterprise'] | None = Field(description='Which plan does the customer name, if any?')


@pytest.mark.vcr
async def test_nested_fields_lists_and_optionals(
    allow_model_requests: None, typesafe_model: TypeSafeModel, request_capture: RequestCapture
):
    """A nested model is its fields under dotted names, a list is one yes/no per option, and `| None` is one more option."""
    agent = Agent(typesafe_model, output_type=Triage)
    result = await agent.run('Third time our pro plan has been charged twice this year. I am furious. Refund it.')
    assert result.output == snapshot(
        Triage(customer=Customer(angry=True), areas=[Area.billing, Area.account, Area.bug], plan='pro')
    )
    assert result.response.provider_details == snapshot(
        {
            'confidence': {'customer.angry': 0.98, 'areas': 0.19999999999999996, 'plan': 1.0},
            'probabilities': {
                'areas': {'billing': 0.97, 'account': 0.76, 'bug': 0.6},
                'plan': {'pro': 1.0, 'free': 0.0, 'enterprise': 0.0, 'none': 0.0},
            },
            'scores': {},
        }
    )
    assert cast(dict[str, Any], request_capture.body('/v1/systemone')['questions']) == snapshot(
        {
            'customer.angry': {
                'type': 'noul',
                'instructions': {
                    'field': 'customer.angry',
                    'question': 'Is the customer angry?',
                    'goal': 'Triage a support ticket.',
                },
            },
            'areas.billing': {
                'type': 'noul',
                'instructions': {
                    'field': 'areas',
                    'question': 'Which teams does this touch?',
                    'goal': 'Triage a support ticket.',
                    'option': 'billing: Money already owed, charged or refunded.',
                },
            },
            'areas.account': {
                'type': 'noul',
                'instructions': {
                    'field': 'areas',
                    'question': 'Which teams does this touch?',
                    'goal': 'Triage a support ticket.',
                    'option': 'account',
                },
            },
            'areas.bug': {
                'type': 'noul',
                'instructions': {
                    'field': 'areas',
                    'question': 'Which teams does this touch?',
                    'goal': 'Triage a support ticket.',
                    'option': 'bug',
                },
            },
            'plan': {
                'type': 'choice',
                'criteria': {'free': None, 'pro': None, 'enterprise': None, 'none': 'None of these.'},
                'instructions': {
                    'field': 'plan',
                    'question': 'Which plan does the customer name, if any?',
                    'goal': 'Triage a support ticket.',
                },
            },
        }
    )


async def test_an_optional_pick_one_answers_none(allow_model_requests: None):
    class Named(BaseModel):
        plan: Literal['free', 'pro'] | None = Field(description='Which plan, if any?')

    jev = mock_model(
        lambda _: answers(
            plan={
                'type': 'choice',
                'choice': 'none',
                'confidence': 0.9,
                'probabilities': {'none': 0.9, 'free': 0.05, 'pro': 0.05},
            }
        )
    )
    result = await Agent(jev, output_type=Named).run('Hello.')
    assert result.output == Named(plan=None)


async def test_the_none_option_stays_clear_of_an_option_named_none(allow_model_requests: None):
    seen: list[dict[str, Any]] = []

    class Named(BaseModel):
        plan: Literal['none', 'some'] | None = Field(description='Which plan, if any is named?')

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return answers(
            plan={
                'type': 'choice',
                'choice': 'none_',
                'confidence': 0.9,
                'probabilities': {'none_': 0.9, 'none': 0.05, 'some': 0.05},
            }
        )

    result = await Agent(mock_model(record), output_type=Named).run('Hello.')
    assert result.output == Named(plan=None)
    assert list(seen[0]['questions']['plan']['criteria']) == ['none', 'some', 'none_']


async def test_a_list_answer_of_the_wrong_kind(allow_model_requests: None):
    class Touches(BaseModel):
        areas: list[Literal['billing', 'bug']] = Field(description='Which teams?')

    payload: dict[str, dict[str, object]] = {
        'areas.billing': {'type': 'choice', 'choice': 'yes', 'confidence': 0.9, 'probabilities': {'yes': 0.9}},
        'areas.bug': {'type': 'noul', 'noul': 0.1},
    }
    jev = mock_model(lambda _: answers(**payload))
    with pytest.raises(UnexpectedModelBehavior, match="output field 'areas', option 'billing'"):
        await Agent(jev, output_type=Touches).run('Hello.')


@pytest.mark.parametrize(
    'annotation,match',
    [
        pytest.param('list[str]', 'a list must be of two or more string options', id='list of text'),
        pytest.param('bool | None', 'only a `Literal` or `Enum` of strings can be optional', id='optional yes/no'),
        pytest.param('Customer | None', 'is not supported by this model', id='optional model'),
    ],
)
async def test_unsupported_richer_fields(
    allow_model_requests: None, typesafe_model: TypeSafeModel, annotation: str, match: str
):
    Richer = type('Richer', (BaseModel,), {'__annotations__': {'value': eval(annotation)}})
    with pytest.raises(UserError, match=match):
        await Agent(typesafe_model, output_type=Richer, instructions='Judge it.').run('anything')


async def test_a_model_that_refers_to_itself_is_refused_on_that_field(
    allow_model_requests: None, typesafe_model: TypeSafeModel
):
    class Comment(BaseModel):
        spam: bool
        replies: list[Comment] = []

    with pytest.raises(UserError, match="Output field 'replies' is not supported"):
        await Agent(typesafe_model, output_type=Comment).run('anything')


async def test_a_dot_in_a_field_name_is_refused(allow_model_requests: None, typesafe_model: TypeSafeModel):
    class Dotted(BaseModel):
        urgent: bool = Field(alias='is.urgent')

    with pytest.raises(UserError, match=r"Output field 'is\.urgent' is not supported by this model: a dot"):
        await Agent(typesafe_model, output_type=Dotted).run('anything')


async def test_a_nested_field_jev_cannot_answer_is_named_in_full(
    allow_model_requests: None, typesafe_model: TypeSafeModel
):
    class Inner(BaseModel):
        note: str

    class Outer(BaseModel):
        inner: Inner

    with pytest.raises(UserError, match=r"Output field 'inner\.note' is not supported"):
        await Agent(typesafe_model, output_type=Outer).run('anything')


async def test_one_output_function_alone_leaves_nothing_to_ask(
    allow_model_requests: None, typesafe_model: TypeSafeModel
):
    with pytest.raises(UserError, match='nothing to ask Jev'):
        await Agent(typesafe_model, output_type=[approve]).run('anything')


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


async def test_streaming_gives_the_whole_answer_as_one_event(allow_model_requests: None):
    """Jev answers in one piece, so a streamed run gets the answer as a single event rather than failing."""
    jev = mock_model(lambda _: answers(response={'type': 'noul', 'noul': 0.9}))
    agent = Agent(jev, output_type=bool, instructions='Is this fine?')
    async with agent.run_stream('anything') as stream:
        assert await stream.get_output() is True
    response = stream.response
    assert response.model_name == 'jev-latest'
    assert response.provider_details == {'confidence': {'response': 0.8}, 'probabilities': {}, 'scores': {}}
    assert response.usage == RequestUsage(input_tokens=10)


async def test_a_streamed_fallback_takes_the_proposed_step(allow_model_requests: None):
    jev = mock_model(lambda _: tool_answers('refund', 0.95))
    agent = Agent(FallbackModel(jev, TestModel()), output_type=Ticket, tools=[refund])
    async with agent.run_stream('Charged twice.') as stream:
        await stream.get_output()
    # The next model took the refund step, then Jev filled the output with its result in the turn, as in `run`.
    assert [message.model_name for message in stream.all_messages() if isinstance(message, ModelResponse)] == [
        'test',
        'jev-latest',
    ]


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


class Escalation(BaseModel):
    """Hand the ticket to a human specialist."""

    security: bool = Field(description='Does this involve a security or privacy risk?')


class DraftedReply(BaseModel):
    """Write the customer a reply."""

    ok: bool
    body: str


def _route(choice: str, probabilities: dict[str, float]) -> dict[str, object]:
    return {'type': 'choice', 'choice': choice, 'confidence': 0.9, 'probabilities': probabilities}


async def test_a_union_picks_the_type_then_fills_only_that_one(allow_model_requests: None):
    """Several output types are routes: one request picks, a second asks only the chosen type's fields."""
    seen: list[dict[str, Any]] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        if len(seen) == 1:
            return answers(
                tool=_route(
                    'final_result_Escalation',
                    {'final_result_Ticket': 0.2, 'final_result_Escalation': 0.8},
                )
            )
        return answers(security={'type': 'noul', 'noul': 0.9})

    agent = Agent(mock_model(record), output_type=[Ticket, Escalation])
    result = await agent.run('Someone else can see my invoices.')

    assert result.output == Escalation(security=True)
    # The first request asks nothing but the route; the second asks nothing but the chosen type's fields.
    assert list(seen[0]['questions']) == ['tool']
    assert list(seen[1]['questions']) == ['security']
    assert result.response.provider_details == snapshot(
        {
            'confidence': {'security': 0.8},
            'probabilities': {},
            'scores': {},
            'tool': {
                'choice': 'final_result_Escalation',
                'probabilities': {'final_result_Ticket': 0.2, 'final_result_Escalation': 0.8},
                'offered': [],
            },
            'requests': 2,
        }
    )


async def test_a_union_member_jev_cannot_express_is_offered_and_hands_off_when_picked(allow_model_requests: None):
    """A union is the route set, so a member beyond Jev is a hand-off rather than a refusal.

    With one output type there is no other route the run could take, so an unfillable one still raises up front.
    """
    seen: list[dict[str, Any]] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return answers(
            tool=_route(
                'final_result_DraftedReply',
                {'final_result_Ticket': 0.1, 'final_result_DraftedReply': 0.9},
            )
        )

    agent = Agent(mock_model(record), output_type=[Ticket, DraftedReply])
    with pytest.raises(ToolCallProposed) as exc_info:
        await agent.run('Write back and say sorry.')

    assert (exc_info.value.tool_name, exc_info.value.probability) == ('final_result_DraftedReply', 0.9)
    # The choice call happened; only the fill was beyond Jev.
    assert len(seen) == 1


async def test_a_union_member_jev_cannot_express_is_filled_by_the_model_behind_it(allow_model_requests: None):
    """`ToolCallProposed` is a `ModelAPIError`, so `FallbackModel` gives the whole step to a language model."""

    def record(request: httpx2.Request) -> httpx2.Response:
        return answers(
            tool=_route(
                'final_result_DraftedReply',
                {'final_result_Ticket': 0.1, 'final_result_DraftedReply': 0.9},
            )
        )

    agent = Agent(FallbackModel(mock_model(record), TestModel()), output_type=[Ticket, DraftedReply])
    result = await agent.run('Write back and say sorry.')

    # The step is handed over whole, so the model behind Jev picks the route and fills it.
    assert result.response.model_name == 'test'
    assert isinstance(result.output, (Ticket, DraftedReply))


async def test_one_output_type_jev_cannot_express_is_still_refused_before_any_request(allow_model_requests: None):
    """Alone, an unfillable output type can only ever fail, so it is a coding error rather than a hand-off."""

    def unreachable(request: httpx2.Request) -> httpx2.Response:  # pragma: no cover
        raise AssertionError('a lone unfillable output type must not reach a request')

    with pytest.raises(UserError, match="Output field 'summary' is not supported"):
        await Agent(mock_model(unreachable), output_type=WithText).run('anything')


async def test_a_union_below_the_threshold_fills_the_likeliest_output_type(allow_model_requests: None):
    """The threshold gates tools, not output types: an unsure tool pick falls back to the likeliest result."""
    seen: list[dict[str, Any]] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        if len(seen) == 1:
            return answers(
                tool=_route(
                    'refund',
                    {'final_result_Ticket': 0.2, 'final_result_Escalation': 0.5, 'refund': 0.3},
                )
            )
        return answers(security={'type': 'noul', 'noul': 0.9})

    agent = Agent(mock_model(record), output_type=[Ticket, Escalation], tools=[refund])
    result = await agent.run('Someone else can see my invoices.')

    assert result.output == Escalation(security=True)
    assert list(seen[1]['questions']) == ['security']


async def test_a_union_member_needs_its_own_docstring(allow_model_requests: None):
    """One instruction cannot describe two different routes, so each member says what it is for itself."""

    def unreachable(request: httpx2.Request) -> httpx2.Response:  # pragma: no cover
        raise AssertionError('a union member without a docstring must be refused before any request')

    agent = Agent(mock_model(unreachable), output_type=[Ticket, WithOptional], instructions='Handle the ticket.')
    with pytest.raises(UserError, match="'final_result_WithOptional' says nothing about itself"):
        await agent.run('anything')


async def test_a_route_jev_did_not_price_is_still_filled(allow_model_requests: None):
    """`_tool_call` falls back to the likeliest output type without requiring Jev to have priced it.

    Jev is not obliged to report a probability for every option it was offered, so the route that comes back
    is not necessarily one that appears in `probabilities`. Reading it as a plain index raised `KeyError`.
    """
    seen: list[dict[str, Any]] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        if len(seen) == 1:
            # A below-threshold tool pick, with neither output type priced.
            return answers(tool=_route('refund', {'refund': 0.3}))
        return answers(urgent={'type': 'noul', 'noul': 0.9})

    agent = Agent(mock_model(record), output_type=[Ticket, Escalation], tools=[refund])
    result = await agent.run('You charged me twice.')

    assert result.output == Ticket(urgent=True)
    details = result.response.provider_details or {}
    assert details['tool']['probabilities'] == {'refund': 0.3}
    assert details['requests'] == 2


async def test_a_union_no_member_of_which_jev_can_fill_is_refused_before_any_request(allow_model_requests: None):
    """A hand-off is worth building only while some other route is a real alternative.

    With every member beyond Jev the choice is decided before it is asked: each answer hands off, so the
    request that asks which one buys nothing and every run pays for Jev on top of the model behind it.
    """

    class DraftedReply(BaseModel):
        """Write the customer a reply."""

        body: str

    class Summary(BaseModel):
        """Summarise the thread."""

        text: str

    def unreachable(request: httpx2.Request) -> httpx2.Response:  # pragma: no cover
        raise AssertionError('a union with nothing fillable must be refused before any request')

    agent = Agent(mock_model(unreachable), output_type=[DraftedReply, Summary])
    with pytest.raises(UserError, match='None of the output types can be filled by this model'):
        await agent.run('Write back.')


async def test_a_union_with_one_fillable_member_is_still_offered(allow_model_requests: None):
    """One real alternative is enough: the hand-off then depends on the text rather than on the types."""

    class DraftedReply(BaseModel):
        """Write the customer a reply."""

        body: str

    seen: list[dict[str, Any]] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        if len(seen) == 1:
            return answers(
                tool=_route('final_result_Ticket', {'final_result_Ticket': 0.9, 'final_result_DraftedReply': 0.1})
            )
        return answers(urgent={'type': 'noul', 'noul': 0.9})

    agent = Agent(mock_model(record), output_type=[Ticket, DraftedReply])
    result = await agent.run('Is this urgent?')

    assert result.output == Ticket(urgent=True)
    # Both were offered: the hand-off now depends on which one the text calls for.
    assert set(seen[0]['questions']['tool']['criteria']) == {'final_result_Ticket', 'final_result_DraftedReply'}


@pytest.mark.parametrize('setting', ['typesafe_tool_call_threshold', 'typesafe_boolean_threshold'])
async def test_a_bad_threshold_is_refused_before_the_forced_route_spends_a_request(
    allow_model_requests: None, setting: str
):
    """The forced-fill path takes no choice question, so its bars are checked before it sends anything.

    Reading them only while handling the answer would mean paying for the request that carried the prompt and
    the whole history before saying the settings were wrong.
    """

    def unreachable(request: httpx2.Request) -> httpx2.Response:  # pragma: no cover
        raise AssertionError('a threshold outside 0 to 1 must be refused before any request')

    def set_direction(direction: Literal['left', 'right']) -> str:
        """Set the direction to take.

        Args:
            direction: Which direction should be taken?
        """
        return direction  # pragma: no cover

    # No output type to fill, and `approve` already returned this turn, so `set_direction` is the one route
    # left: it is filled without a choice question, which is the path that used to validate too late.
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('Go left.')]),
        ModelResponse(parts=[ToolCallPart('final_result', {}, 'call_1')]),
        ModelRequest(parts=[ToolReturnPart('final_result', 'approved', 'call_1')]),
    ]
    agent = Agent(mock_model(unreachable), output_type=[approve], tools=[set_direction])
    with pytest.raises(UserError, match=f'`{setting}` must be between 0 and 1'):
        await agent.run(message_history=history, model_settings=cast(TypeSafeModelSettings, {setting: 1.5}))


@pytest.mark.parametrize('noul', [-0.1, 1.5])
async def test_a_probability_outside_zero_to_one_is_a_model_error_not_a_crash(allow_model_requests: None, noul: float):
    """Both confidence scalings divide by the room left on their side of the bar, which can be zero.

    At a threshold of 0 a negative probability used to reach `(0 - noul) / 0`. A malformed answer is
    something the model reports, like every other unexpected answer, rather than a `ZeroDivisionError`.
    """
    agent = Agent(mock_model(lambda _: answers(urgent={'type': 'noul', 'noul': noul})), output_type=Ticket)
    with pytest.raises(UnexpectedModelBehavior, match=f'Unexpected probability from TypeSafe: {noul}'):
        await agent.run('x', model_settings=TypeSafeModelSettings(typesafe_boolean_threshold=0))
