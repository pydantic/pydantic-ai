"""Tests for `OpenAIDecisionsModel`.

Tests marked `vcr` run against recordings of the live API, and assert what the code sends with `request_capture`,
which sees the request on replay too. The rest send no request, or mock the transport and say why it can't be a
recording.
"""

from __future__ import annotations as _annotations

import json
from collections.abc import Callable, Mapping
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal
from unittest.mock import AsyncMock, PropertyMock, patch

import anyio
import httpx2
import pytest
from cassetter import Cassette
from dirty_equals import IsJson
from pydantic import BaseModel, Field, JsonValue, ValidationError, WithJsonSchema
from pydantic.json_schema import JsonSchemaValue

from pydantic_ai import (
    Agent,
    BinaryContent,
    BoolCriteria,
    ModelAPIError,
    ModelHTTPError,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.exceptions import ContentFilterError, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import (
    AudioUrl,
    FilePart,
    ImageUrl,
    ModelMessagesTypeAdapter,
    NativeToolCallPart,
    NativeToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters, get_user_agent, infer_model
from pydantic_ai.models.decision import (
    ChoiceQuestion,
    DecisionQuestion,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    NoulCriteria,
    NoulQuestion,
    ScoreQuestion,
    UnfillableRoute,
)
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles.decision import DecisionModelProfile
from pydantic_ai.tools import ObjectJsonSchema, ToolDefinition
from pydantic_ai.usage import RequestUsage

from .._inline_snapshot import snapshot
from ..conftest import IsDatetime, IsList, IsStr, RequestCapture, TestEnv, try_import
from .test_system_one import Captured, Frustration, Handler, Ticket

with try_import() as imports_successful:
    from openai import APIConnectionError, APIError, APIStatusError, AsyncOpenAI
    from openai.types.decision_create_params import DecisionCreateParams

    from pydantic_ai.models.openai_decisions import OpenAIDecisionsModel, OpenAIDecisionsModelSettings
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openai_decisions import OpenAIDecisionsProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')

READINESS_WAIT_TIMEOUT = 10


@pytest.fixture
def capture_model(openai_api_key: str, request_capture: RequestCapture) -> OpenAIDecisionsModel:
    provider = OpenAIDecisionsProvider(api_key=openai_api_key, http_client=request_capture.client)
    return OpenAIDecisionsModel('gpt-6-luna', provider=provider)


def mock_model(handler: Handler) -> OpenAIDecisionsModel:
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    client = AsyncOpenAI(api_key='test', max_retries=0, http_client=http_client)
    return OpenAIDecisionsModel('gpt-6-luna', provider=OpenAIDecisionsProvider(openai_client=client))


def decisions(*answers: Mapping[str, object]) -> httpx2.Response:
    """A `/v1/decisions` response, in the shape the API answers in.

    Encoded with `json.dumps`, which writes `NaN` as a server written in Python can, where `json=` refuses to.
    """
    usage = {
        'input_tokens': 396,
        'input_tokens_details': {'cached_tokens': 128, 'cache_write_tokens': 0},
        'output_tokens': 0,
        'output_tokens_details': {'reasoning_tokens': 0},
        'total_tokens': 396,
    }
    return httpx2.Response(
        200,
        content=json.dumps({'model': 'gpt-6-luna', 'answers': list(answers), 'usage': usage}),
        headers={'content-type': 'application/json', 'x-request-id': 'req_123'},
    )


URGENT = {'type': 'predicate', 'name': 'urgent', 'probability': 0.91}
AREA_PROBABILITIES = [{'value': 'billing', 'probability': 0.94}, {'value': 'bug', 'probability': 0.06}]
AREA = {
    'type': 'choice',
    'name': 'area',
    'choice': 'billing',
    'probabilities': AREA_PROBABILITIES,
    'confidence': 0.88,
}
REFUND = {'type': 'predicate', 'name': 'refund', 'probability': 0.2}
FRUSTRATION_PROBABILITIES = [
    {'value': 0, 'label': '0', 'probability': 0.05},
    {'value': 1, 'label': '1', 'probability': 0.2},
    {'value': 2, 'label': '2', 'probability': 0.75},
]
FRUSTRATION = {
    'type': 'score',
    'name': 'frustration',
    'score': 1.7,
    'probabilities': FRUSTRATION_PROBABILITIES,
    'confidence': 0.55,
}


def ticket_answers(request: httpx2.Request) -> httpx2.Response:
    return decisions(URGENT, AREA)


def boolean_answers(request: httpx2.Request) -> httpx2.Response:
    questions: list[dict[str, object]] = json.loads(request.content)['questions']
    answers: list[Mapping[str, object]] = []
    for question in questions:
        name = question['name']
        assert isinstance(name, str)
        answers.append({'type': 'predicate', 'name': name, 'probability': 0.9})
    return decisions(*answers)


def test_init(env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    model = OpenAIDecisionsModel('gpt-6-luna')
    assert model.model_name == 'gpt-6-luna'
    assert model.system == 'openai-decisions'
    assert model.base_url == 'https://api.openai.com/v1/'
    assert isinstance(model.client, AsyncOpenAI)
    # The ID round-trips, where `openai:gpt-6-luna` would be a Responses API model.
    assert model.model_id == 'openai-decisions:gpt-6-luna'
    assert isinstance(infer_model(model.model_id), OpenAIDecisionsModel)


def test_missing_api_key_names_this_provider(env: TestEnv):
    env.remove('OPENAI_API_KEY')
    env.remove('OPENAI_BASE_URL')
    with pytest.raises(UserError, match=r'`OpenAIDecisionsProvider\(api_key=\.\.\.\)`'):
        OpenAIDecisionsProvider()


def test_infer_model_refuses_another_provider():
    with pytest.raises(UserError, match='require an `OpenAIDecisionsProvider`'):
        infer_model('openai-decisions:gpt-6-luna', provider_factory=lambda _: OpenAIProvider(api_key='test'))


@pytest.mark.vcr
async def test_output_type(
    allow_model_requests: None, capture_model: OpenAIDecisionsModel, request_capture: RequestCapture
):
    agent = Agent(capture_model, output_type=Ticket)

    result = await agent.run('My invoice was charged twice and nobody answers the phone!')

    assert result.response == snapshot(
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='final_result',
                    args={'urgent': True, 'area': 'billing'},
                    tool_call_id=IsStr(),
                )
            ],
            usage=RequestUsage(input_tokens=312, output_reasoning_tokens=0, cost=Decimal('0.0000312')),
            model_name='gpt-6-luna',
            timestamp=IsDatetime(),
            provider_name='openai-decisions',
            provider_url='https://api.openai.com/v1/',
            provider_details={
                'confidence': {'urgent': 0.06, 'area': 1.0},
                'probabilities': {'area': {'billing': 1.0, 'bug': 0.0}},
                'scores': {},
            },
            finish_reason='tool_call',
            run_id=IsStr(),
            conversation_id=IsStr(),
        )
    )
    assert request_capture.paths == ['/v1/decisions']
    assert request_capture.headers[0]['user-agent'] == get_user_agent()
    assert request_capture.body('/decisions') == snapshot(
        {
            'model': 'gpt-6-luna',
            'input': 'My invoice was charged twice and nobody answers the phone!',
            'questions': [
                {
                    'type': 'predicate',
                    'name': 'urgent',
                    'instructions': '{"field": "urgent", "question": "Does this need a reply within the hour?", "goal": "Triage a support ticket."}',
                },
                {
                    'type': 'choice',
                    'name': 'area',
                    'instructions': '{"field": "area", "question": "Which team owns it?", "goal": "Triage a support ticket."}',
                    'choices': [{'value': 'billing'}, {'value': 'bug'}],
                },
            ],
        }
    )


class Mood(BaseModel):
    """Read the customer's mood."""

    refund: Annotated[bool, BoolCriteria(true='They ask for their money back.', false='They do not.')] = Field(
        description='Do they want a refund?'
    )
    frustration: Frustration = Field(description='How frustrated is the customer?')


@pytest.mark.vcr
async def test_yes_no_meanings_and_rubric(
    allow_model_requests: None, capture_model: OpenAIDecisionsModel, request_capture: RequestCapture
):
    """A predicate has no field for what yes and no mean, so they go into its instructions; a rubric is `levels`."""
    result = await Agent(capture_model, output_type=Mood).run('This is the third time I am asking. Fix it NOW.')

    assert result.output == snapshot(Mood(refund=False, frustration=2))
    assert result.response.provider_details == snapshot(
        {
            'confidence': {'refund': 1.0, 'frustration': 0.52},
            'probabilities': {'frustration': {'0': 0.0, '1': 0.32, '2': 0.68}},
            'scores': {'frustration': 1.68},
        }
    )
    assert request_capture.body('/decisions')['questions'] == snapshot(
        [
            {
                'type': 'predicate',
                'name': 'refund',
                'instructions': '{"field": "refund", "question": "Do they want a refund?", "goal": "Read the customer\'s mood.", "yes": "They ask for their money back.", "no": "They do not."}',
            },
            {
                'type': 'score',
                'name': 'frustration',
                'instructions': '{"field": "frustration", "question": "How frustrated is the customer?", "goal": "Read the customer\'s mood."}',
                'levels': [
                    {'label': '0', 'description': 'Calm'},
                    {'label': '1', 'description': 'Frustrated'},
                    {'label': '2', 'description': 'Very angry'},
                ],
            },
        ]
    )


class Team(StrEnum):
    billing = 'billing'
    bug = 'bug'


class Customer(BaseModel):
    vip: bool = Field(description='Are they on an enterprise plan?')


class Assignment(BaseModel):
    """Assign a support ticket."""

    team: Team = Field(description='Which team owns it?')
    customer: Customer = Field(description='Who is writing in.')


@pytest.mark.vcr
async def test_enum_and_nested_fields(
    allow_model_requests: None, capture_model: OpenAIDecisionsModel, request_capture: RequestCapture
):
    """An `Enum` or nested model field with a description is asked like on any other decision model.

    The schema puts a `$ref` beside the description, which the Responses API's profile for the same model ID would
    rewrite into a shape no question is built from, so the provider gives the decision model profile instead.
    """
    result = await Agent(capture_model, output_type=Assignment).run('I was charged twice on my personal card.')

    assert result.output == snapshot(Assignment(team=Team.billing, customer=Customer(vip=False)))
    assert request_capture.body('/decisions')['questions'] == snapshot(
        [
            {
                'type': 'choice',
                'name': 'team',
                'instructions': '{"field": "team", "question": "Which team owns it?", "goal": "Assign a support ticket."}',
                'choices': [{'value': 'billing'}, {'value': 'bug'}],
            },
            {
                'type': 'predicate',
                'name': 'customer.vip',
                'instructions': '{"field": "customer.vip", "context": ["customer: Who is writing in."], "question": "Are they on an enterprise plan?", "goal": "Assign a support ticket."}',
            },
        ]
    )


@pytest.mark.vcr
async def test_conversation_and_route(
    allow_model_requests: None, capture_model: OpenAIDecisionsModel, request_capture: RequestCapture
):
    """A conversation is JSON, sent as the `input` text, and the route between a tool and the output is a `choice`."""

    def escalate() -> None:
        """Hand the ticket to a human."""

    agent = Agent(capture_model, output_type=Ticket, tools=[escalate])
    history: list[ModelMessage] = [
        ModelRequest.user_text_prompt('I was charged twice.'),
        ModelResponse(parts=[TextPart('Sorry to hear that, we are looking into it.')]),
    ]

    result = await agent.run('Still no refund!', message_history=history)

    assert result.output == snapshot(Ticket(urgent=False, area='billing'))
    assert result.response.provider_details == snapshot(
        {
            'confidence': {'urgent': 0.1, 'area': 1.0},
            'probabilities': {'area': {'billing': 1.0, 'bug': 0.0}},
            'scores': {},
            'route': {
                'choice': 'Ticket',
                'probabilities': {'Ticket': 0.89, 'escalate': 0.11},
                'offered': ['Ticket', 'escalate'],
            },
        }
    )
    assert request_capture.body('/decisions') == snapshot(
        {
            'model': 'gpt-6-luna',
            'input': '{"history": [{"user": "I was charged twice."}, {"assistant": "Sorry to hear that, we are looking into it."}], "text": "Still no refund!"}',
            'questions': [
                {
                    'type': 'predicate',
                    'name': 'Ticket.urgent',
                    'instructions': '{"field": "urgent", "premise": "If the user\'s request calls for Ticket: Triage a support ticket.", "question": "Does this need a reply within the hour?"}',
                },
                {
                    'type': 'choice',
                    'name': 'Ticket.area',
                    'instructions': '{"field": "area", "premise": "If the user\'s request calls for Ticket: Triage a support ticket.", "question": "Which team owns it?"}',
                    'choices': [{'value': 'billing'}, {'value': 'bug'}],
                },
                {
                    'type': 'choice',
                    'name': 'route',
                    'instructions': 'Which of these does this call for?',
                    'choices': [
                        {'value': 'Ticket', 'description': 'Triage a support ticket.'},
                        {'value': 'escalate', 'description': 'Hand the ticket to a human.'},
                    ],
                },
            ],
        }
    )


@pytest.mark.vcr
async def test_refusal_on_a_route_not_taken(allow_model_requests: None, capture_model: OpenAIDecisionsModel):
    """Every route's fields are asked beside the route question, so a refusal fails the step whichever route is picked."""

    def record_profile(disabled: bool) -> None:
        """Record the customer's profile.

        Args:
            disabled: Does the customer have a disability?
        """

    agent = Agent(capture_model, output_type=Ticket, tools=[record_profile])

    with pytest.raises(ContentFilterError) as exc_info:
        await agent.run('My invoice was charged twice and nobody answers the phone!')
    assert exc_info.value.message == snapshot(
        "Content filter triggered. The OpenAI Decisions API declined to answer: 'record_profile.disabled'"
    )
    assert json.loads(exc_info.value.body or '')['answers'] == snapshot(
        [
            {'type': 'predicate', 'name': 'Ticket.urgent', 'probability': 0.76},
            {
                'type': 'choice',
                'name': 'Ticket.area',
                'choice': 'billing',
                'probabilities': [{'value': 'billing', 'probability': 1.0}, {'value': 'bug', 'probability': 0.0}],
                'confidence': 1.0,
            },
            {'type': 'refusal', 'name': 'record_profile.disabled'},
            {
                'type': 'choice',
                'name': 'route',
                'choice': 'Ticket',
                'probabilities': [
                    {'value': 'Ticket', 'probability': 1.0},
                    {'value': 'record_profile', 'probability': 0.0},
                ],
                'confidence': 1.0,
            },
        ]
    )


@pytest.mark.vcr
async def test_refusal_while_filling_a_picked_route(allow_model_requests: None, capture_model: OpenAIDecisionsModel):
    """The one route left is filled in a request of its own, and a refusal there fails naming the route.

    `request` is called directly: an agent run always offers its output as a route too, so it never has one route left.
    """
    record_profile = ToolDefinition(
        name='record_profile',
        parameters_json_schema={
            'type': 'object',
            'properties': {'white': {'type': 'boolean', 'description': "Is the customer's race white?"}},
            'required': ['white'],
        },
    )
    escalate = ToolDefinition(name='escalate', parameters_json_schema={'type': 'object'})
    messages: list[ModelMessage] = [
        ModelRequest.user_text_prompt('My invoice was charged twice and nobody answers the phone!'),
        ModelResponse(parts=[ToolCallPart('escalate', {}, tool_call_id='call_1')]),
        ModelRequest(parts=[ToolReturnPart('escalate', 'Escalated.', tool_call_id='call_1')]),
    ]
    parameters = ModelRequestParameters(function_tools=[record_profile, escalate], allow_text_output=False)

    with pytest.raises(
        UnexpectedModelBehavior, match="selected 'record_profile', but failed while filling its fields"
    ) as exc_info:
        await capture_model.request(messages, None, parameters)
    assert isinstance(exc_info.value.__cause__, ContentFilterError)


Severity = Annotated[
    Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    WithJsonSchema(
        {'type': 'integer', 'anyOf': [{'const': level, 'description': f'{level} of 10'} for level in range(11)]}
    ),
]


class Rating(BaseModel):
    """Rate a support ticket."""

    severity: Severity = Field(description='How severe is the problem?')


@pytest.mark.vcr
async def test_rubric_over_the_limit(
    allow_model_requests: None, capture_model: OpenAIDecisionsModel, request_capture: RequestCapture
):
    """Eleven levels are one more than a rubric takes, so they are asked as a pick-one."""
    result = await Agent(capture_model, output_type=Rating).run('Checkout is down for every customer.')

    assert result.output == snapshot(Rating(severity=10))
    assert request_capture.body('/decisions')['questions'] == snapshot(
        [
            {
                'type': 'choice',
                'name': 'severity',
                'instructions': '{"field": "severity", "question": "How severe is the problem?", "goal": "Rate a support ticket."}',
                'choices': [
                    {'value': '0', 'description': '0 of 10'},
                    {'value': '1', 'description': '1 of 10'},
                    {'value': '2', 'description': '2 of 10'},
                    {'value': '3', 'description': '3 of 10'},
                    {'value': '4', 'description': '4 of 10'},
                    {'value': '5', 'description': '5 of 10'},
                    {'value': '6', 'description': '6 of 10'},
                    {'value': '7', 'description': '7 of 10'},
                    {'value': '8', 'description': '8 of 10'},
                    {'value': '9', 'description': '9 of 10'},
                    {'value': '10', 'description': '10 of 10'},
                ],
            }
        ]
    )


@pytest.mark.vcr
@pytest.mark.parametrize(
    ('limit', 'question', 'expected'),
    [
        pytest.param(
            lambda: OpenAIDecisionsModel.max_choice_options,
            ChoiceQuestion(criteria={str(option): None for option in range(256)}),
            snapshot(
                {
                    'model': 'gpt-6-luna',
                    'input': 'Charged twice.',
                    'questions': [{'type': 'choice', 'name': 'q', 'instructions': '', 'choices': IsList(length=256)}],
                }
            ),
            id='options',
        ),
        pytest.param(
            lambda: OpenAIDecisionsModel.max_score_levels,
            ScoreQuestion(criteria=[None] * 11),
            snapshot(
                {
                    'model': 'gpt-6-luna',
                    'input': 'Charged twice.',
                    'questions': [{'type': 'score', 'name': 'q', 'instructions': '', 'levels': IsList(length=11)}],
                }
            ),
            id='levels',
        ),
    ],
)
async def test_api_limits(
    limit: Callable[[], int | None],
    question: DecisionQuestion,
    expected: JsonValue,
    allow_model_requests: None,
    capture_model: OpenAIDecisionsModel,
    request_capture: RequestCapture,
):
    """`max_choice_options` and `max_score_levels` are the API's own limits: one more is a 400 naming them.

    `decide` is called directly, since an agent run refuses these before sending.
    """
    with pytest.raises(ModelHTTPError, match=f'maximum length {limit()},') as exc_info:
        await capture_model.decide(DecisionRequest(state='Charged twice.', questions={'q': question}), {})
    assert exc_info.value.status_code == 400

    assert request_capture.body('/v1/decisions') == expected


@pytest.mark.vcr
async def test_http_error(allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture):
    """A model the API does not serve is an error response, raised for a `FallbackModel` to take over."""
    model = OpenAIDecisionsModel(
        'gpt-5', provider=OpenAIDecisionsProvider(api_key=openai_api_key, http_client=request_capture.client)
    )

    with pytest.raises(ModelHTTPError) as exc_info:
        await Agent(model, output_type=Ticket).run('Charged twice.')
    assert exc_info.value.status_code == 404
    assert exc_info.value.model_name == 'gpt-5'
    assert exc_info.value.body == snapshot(
        {
            'message': 'The model `gpt-5` does not exist or you do not have access to it.',
            'type': 'invalid_request_error',
            'param': None,
            'code': 'model_not_found',
        }
    )
    assert request_capture.body('/v1/decisions')['model'] == 'gpt-5'


async def test_http_error_preserves_sdk_response_details_and_authorization(allow_model_requests: None):
    """A fixed 403 pins the authenticated request and the error details retained from the SDK response."""
    captured_headers: list[str] = []
    body = {'message': 'Access denied.', 'code': 'permission_denied'}

    def reject(request: httpx2.Request) -> httpx2.Response:
        captured_headers.append(request.headers['authorization'])
        return httpx2.Response(403, json=body, headers={'x-request-id': 'req_403', 'x-team': 'support'})

    model = mock_model(reject)
    with pytest.raises(ModelHTTPError) as exc_info:
        await model.decide(DecisionRequest(state='Review this.', questions={'q': NoulQuestion()}), {})

    assert captured_headers == ['Bearer test']
    assert exc_info.value.status_code == 403
    assert exc_info.value.body == body
    assert exc_info.value.headers is not None
    assert {name: exc_info.value.headers[name] for name in ('content-type', 'x-request-id', 'x-team')} == {
        'content-type': 'application/json',
        'x-request-id': 'req_403',
        'x-team': 'support',
    }
    assert isinstance(exc_info.value.__cause__, APIStatusError)


@pytest.mark.parametrize('error_kind', ['connection', 'api'])
async def test_sdk_errors_map_to_model_api_error_with_cause(allow_model_requests: None, error_kind: str):
    """A transport failure is wrapped by the SDK; a synthetic `APIError` covers its protocol-error branch."""
    sdk_errors: list[APIError] = []
    transport_errors: list[httpx2.ConnectError] = []

    def fail(request: httpx2.Request) -> httpx2.Response:
        if error_kind == 'connection':
            transport_error = httpx2.ConnectError('connection failed')
            transport_errors.append(transport_error)
            raise transport_error
        sdk_error = APIError('API protocol failed', request, body={'code': 'protocol_error'})
        sdk_errors.append(sdk_error)
        raise sdk_error

    model = mock_model(fail)
    with pytest.raises(ModelAPIError) as exc_info:
        await model.decide(DecisionRequest(state='Review this.', questions={'q': NoulQuestion()}), {})

    assert exc_info.value.model_name == 'gpt-6-luna'
    assert exc_info.value.message == 'OpenAI Decisions request failed'
    if error_kind == 'connection':
        [transport_error] = transport_errors
        assert isinstance(exc_info.value.__cause__, APIConnectionError)
        assert exc_info.value.__cause__.__cause__ is transport_error
    else:
        [sdk_error] = sdk_errors
        assert exc_info.value.__cause__ is sdk_error


async def test_sdk_status_error_below_400_maps_to_model_api_error(allow_model_requests: None):
    """A redirect without a `Location` reaches the adapter as a status error below 400."""
    model = mock_model(lambda _: httpx2.Response(302, headers={'x-request-id': 'req_redirect'}))
    with pytest.raises(ModelAPIError) as exc_info:
        await model.decide(DecisionRequest(state='Review this.', questions={'q': NoulQuestion()}), {})

    assert exc_info.value.message == 'OpenAI Decisions request failed'
    assert isinstance(exc_info.value.__cause__, APIStatusError)
    assert exc_info.value.__cause__.status_code == 302


async def test_too_many_routes(allow_model_requests: None):
    """More routes than the API's 255 options are refused before a request is sent, so there is nothing to record."""
    captured = Captured(ticket_answers)
    tools = [ToolDefinition(name=f'tool_{index}', parameters_json_schema={'type': 'object'}) for index in range(256)]

    with pytest.raises(UserError, match='255'):
        await mock_model(captured).request(
            [ModelRequest.user_text_prompt('Pick a tool.')],
            None,
            ModelRequestParameters(function_tools=tools, allow_text_output=False),
        )
    assert captured.requests == []


@pytest.mark.parametrize(
    ('question', 'sent'),
    [
        pytest.param(
            NoulQuestion(instructions='Is this urgent?'),
            {'type': 'predicate', 'name': 'q', 'instructions': 'Is this urgent?'},
            id='text',
        ),
        pytest.param(
            NoulQuestion(instructions='Is this urgent?', criteria=NoulCriteria(true='Today — before noon.')),
            {
                'type': 'predicate',
                'name': 'q',
                'instructions': '{"question": "Is this urgent?", "yes": "Today — before noon."}',
            },
            id='text and yes',
        ),
        pytest.param(
            NoulQuestion(criteria=NoulCriteria(false='Not today.')),
            {'type': 'predicate', 'name': 'q', 'instructions': '{"no": "Not today."}'},
            id='only no',
        ),
        pytest.param(NoulQuestion(), {'type': 'predicate', 'name': 'q', 'instructions': ''}, id='nothing'),
        pytest.param(
            ScoreQuestion(criteria=['Calm', None]),
            {
                'type': 'score',
                'name': 'q',
                'instructions': '',
                'levels': [{'label': '0', 'description': 'Calm'}, {'label': '1'}],
            },
            id='score',
        ),
    ],
)
async def test_question_shapes(question: DecisionQuestion, sent: dict[str, object], allow_model_requests: None):
    """The yes/no and rubric shapes `decide` sends, with `instructions`, which the API requires, empty where unset.

    The recordings send the pick-one shapes. The base class only sends some of these from an agent run, so `decide`
    is called directly. Not recorded: the answer doesn't matter here, and asking about nothing gets a refusal.
    """
    captured = Captured(lambda request: httpx2.Response(400))
    with pytest.raises(ModelHTTPError):
        await mock_model(captured).decide(DecisionRequest(state='Down since 9am.', questions={'q': question}), {})

    assert captured.body['questions'] == [sent]


async def test_request_id(allow_model_requests: None):
    """`decide` is the only place the request ID reaches: the run's response is built from the answers.

    Not recorded: the cassettes strip the `x-request-id` header it comes from.
    """
    response = await mock_model(lambda request: decisions(URGENT)).decide(
        DecisionRequest(state='Down since 9am.', questions={'urgent': NoulQuestion(instructions='Is this urgent?')}), {}
    )

    assert response == snapshot(
        DecisionResponse(
            answers={'urgent': NoulAnswer(noul=0.91)},
            model_name='gpt-6-luna',
            usage=RequestUsage(input_tokens=396, cache_read_tokens=128, output_reasoning_tokens=0),
            provider_response_id='req_123',
        )
    )


@pytest.mark.parametrize('user_agent_header', ['User-Agent', 'user-agent', 'USER-AGENT'])
async def test_user_agent_and_default_timeout(allow_model_requests: None, user_agent_header: str):
    """A `User-Agent` in `extra_headers` replaces ours, and with no `timeout` set the SDK's default applies.

    `test_model_settings_support.py` checks the settings are forwarded. Not recorded: the timeout is set on the
    request's `extensions`, which neither a cassette nor a capture keeps.
    """
    captured = Captured(ticket_answers)
    agent = Agent(mock_model(captured), output_type=Ticket)

    settings: OpenAIDecisionsModelSettings = {'extra_headers': {user_agent_header: 'support-bot'}}
    await agent.run('Charged twice.', model_settings=settings)

    request = captured.requests[0]
    assert request.headers['user-agent'] == 'support-bot'
    assert None not in request.extensions['timeout'].values()


@pytest.mark.parametrize(
    'response',
    [
        pytest.param(httpx2.Response(200, text='not json'), id='not json'),
        pytest.param(decisions({**URGENT, 'type': 'noul'}, AREA), id='unknown type'),
    ],
)
async def test_invalid_response(response: httpx2.Response, allow_model_requests: None):
    """The SDK doesn't validate what it parses, so `decide` does. Not recorded: no live model answers like this."""
    agent = Agent(mock_model(lambda request: response), output_type=Ticket)
    with pytest.raises(UnexpectedModelBehavior) as exc_info:
        await agent.run('Charged twice.')
    assert exc_info.value.message == 'Invalid response from the OpenAI Decisions API'
    assert exc_info.value.body is not None
    assert exc_info.value.__cause__ is not None
    assert str(exc_info.value.__cause__)
    if response.text == 'not json':
        assert exc_info.value.body == 'not json'
    else:
        assert 'answers' in exc_info.value.body


@pytest.mark.parametrize(
    ('decision_request', 'answers'),
    [
        pytest.param(
            DecisionRequest(state='Review this.', questions={'q': NoulQuestion()}),
            ({'type': 'predicate', 'name': 'q', 'probability': True},),
            id='predicate boolean probability',
        ),
        pytest.param(
            DecisionRequest(state='Review this.', questions={'q': NoulQuestion()}),
            ({'type': 'predicate', 'name': 'q', 'probability': '0.9'},),
            id='predicate numeric-string probability',
        ),
        pytest.param(
            DecisionRequest(state='Review this.', questions={'area': ChoiceQuestion(criteria={'billing': None})}),
            ({**AREA, 'confidence': True},),
            id='choice boolean confidence',
        ),
        pytest.param(
            DecisionRequest(state='Review this.', questions={'area': ChoiceQuestion(criteria={'billing': None})}),
            ({**AREA, 'confidence': '0.88'},),
            id='choice numeric-string confidence',
        ),
        pytest.param(
            DecisionRequest(
                state='Review this.', questions={'score': ScoreQuestion(criteria=['low', 'medium', 'high'])}
            ),
            ({**FRUSTRATION, 'name': 'score', 'score': True},),
            id='score boolean value',
        ),
        pytest.param(
            DecisionRequest(
                state='Review this.', questions={'score': ScoreQuestion(criteria=['low', 'medium', 'high'])}
            ),
            ({**FRUSTRATION, 'name': 'score', 'score': '1.7'},),
            id='score numeric-string value',
        ),
    ],
)
async def test_decide_rejects_coerced_numeric_response_fields(
    decision_request: DecisionRequest, answers: tuple[Mapping[str, object], ...], allow_model_requests: None
):
    """Response numbers need their documented JSON numeric type; this exercises the SDK parsing boundary directly."""
    response = decisions(*answers)

    with pytest.raises(UnexpectedModelBehavior) as exc_info:
        await mock_model(lambda _: response).decide(decision_request, {})

    assert exc_info.value.message == 'Invalid response from the OpenAI Decisions API'
    assert exc_info.value.body is not None
    assert json.loads(exc_info.value.body) == json.loads(response.text)
    assert isinstance(exc_info.value.__cause__, ValidationError)


@pytest.mark.parametrize(
    'answers',
    [
        pytest.param((URGENT,), id='missing'),
        pytest.param((URGENT, AREA, {**URGENT, 'name': 'extra'}), id='extra'),
        pytest.param((URGENT, {**URGENT, 'probability': 0.1}, AREA), id='twice'),
        pytest.param((URGENT, {'type': 'refusal', 'name': 'extra'}), id='refusal of another'),
    ],
)
async def test_answer_names_match_questions(answers: tuple[Mapping[str, object], ...], allow_model_requests: None):
    """Not recorded: no live model answers like this."""
    agent = Agent(mock_model(lambda request: decisions(*answers)), output_type=Ticket)
    with pytest.raises(UnexpectedModelBehavior, match='answer names do not match the questions'):
        await agent.run('Charged twice.')


@pytest.mark.parametrize(
    ('output_type', 'answers'),
    [
        pytest.param(Ticket, (URGENT, {**URGENT, 'name': 'area'}), id='other kind'),
        pytest.param(Ticket, ({**URGENT, 'probability': -0.1}, AREA), id='probability below 0'),
        pytest.param(Ticket, (URGENT, {**AREA, 'confidence': -0.1}), id='confidence below 0'),
        pytest.param(Ticket, (URGENT, {**AREA, 'confidence': float('nan')}), id='confidence not a number'),
        pytest.param(
            Ticket,
            (
                URGENT,
                {
                    **AREA,
                    'probabilities': [*AREA_PROBABILITIES, {'value': True, 'probability': 0.0}],
                },
            ),
            id='boolean option',
        ),
        pytest.param(
            Ticket,
            (URGENT, {**AREA, 'probabilities': [*AREA_PROBABILITIES, {'value': 'bug', 'probability': 0.0}]}),
            id='option twice',
        ),
        pytest.param(
            Ticket,
            (
                URGENT,
                {
                    **AREA,
                    'probabilities': [{'value': 'billing', 'probability': 1.0}, {'value': 'bug', 'probability': 1.0}],
                },
            ),
            id='probabilities past one',
        ),
        pytest.param(Mood, (REFUND, {**FRUSTRATION, 'score': -0.5}), id='score below the rubric'),
        pytest.param(Mood, (REFUND, {**FRUSTRATION, 'score': 2.5}), id='score past the rubric'),
        pytest.param(
            Mood,
            (
                REFUND,
                {
                    **FRUSTRATION,
                    'score': 1.0,
                    'probabilities': [
                        {'value': 0, 'label': '0', 'probability': 0.9},
                        {'value': 1, 'label': '1', 'probability': 0.1},
                        {'value': 2, 'label': '2', 'probability': 0.0},
                    ],
                },
            ),
            id='score disagrees with probabilities',
        ),
        pytest.param(Mood, (REFUND, {**FRUSTRATION, 'probabilities': []}), id='no levels'),
        pytest.param(Mood, (REFUND, {**FRUSTRATION, 'score': float('nan')}), id='score not a number'),
        pytest.param(
            Mood,
            (
                REFUND,
                {
                    **FRUSTRATION,
                    'probabilities': [
                        {'value': 0, 'label': '0', 'probability': 0.0},
                        {'value': 1, 'label': '1', 'probability': 0.0},
                        {'value': 2, 'label': '2', 'probability': 1.01},
                    ],
                },
            ),
            id='level probability past 1',
        ),
        pytest.param(
            Mood,
            (
                REFUND,
                {
                    **FRUSTRATION,
                    'probabilities': [*FRUSTRATION_PROBABILITIES, {'value': 2, 'label': '2', 'probability': 0.0}],
                },
            ),
            id='level twice',
        ),
        pytest.param(
            Mood,
            (
                REFUND,
                {
                    **FRUSTRATION,
                    'probabilities': [*FRUSTRATION_PROBABILITIES[:2], {'value': 2, 'label': '2', 'probability': 0.5}],
                },
            ),
            id='level probabilities short of one',
        ),
    ],
)
async def test_answers_match_questions(
    output_type: type[BaseModel], answers: tuple[Mapping[str, object], ...], allow_model_requests: None
):
    """An answer its question does not allow fails the request, rather than reaching the output or a retry.

    Not recorded: no live model answers like this.
    """
    agent = Agent(mock_model(lambda request: decisions(*answers)), output_type=output_type)
    with pytest.raises(UnexpectedModelBehavior, match='does not match its question'):
        await agent.run('Charged twice.')


async def test_rounded_score_can_match_rounded_probabilities(allow_model_requests: None):
    """The score and probabilities may be rounded separately while still admitting a consistent distribution.

    Not recorded: a fixed transport response pins the rounding boundary.
    """
    answer: dict[str, object] = {
        **FRUSTRATION,
        'score': 1.0,
        'probabilities': [
            {'value': 0, 'label': '0', 'probability': 0.33},
            {'value': 1, 'label': '1', 'probability': 0.33},
            {'value': 2, 'label': '2', 'probability': 0.34},
        ],
    }
    result = await Agent(mock_model(lambda _: decisions(REFUND, answer)), output_type=Mood).run('How frustrated?')

    assert result.output == Mood(refund=False, frustration=1)


@pytest.mark.parametrize(
    ('option_count', 'probability', 'positive_count', 'matches'),
    [
        pytest.param(255, 0.01, 205, False, id='sum cannot round to one'),
        pytest.param(255, 0.01, 199, True, id='sum can round to one'),
        pytest.param(200, 0.0, 0, False, id='all-zero 200-way distribution'),
        pytest.param(200, 0.005, 200, True, id='uniform rounded 200-way distribution'),
    ],
)
async def test_choice_probability_sum_respects_rounding_bounds(
    option_count: int, probability: float, positive_count: int, matches: bool, allow_model_requests: None
):
    """Rounded choice probabilities must admit a normalized distribution, including a positive 200-way split."""
    criteria: dict[str, JsonValue] = {str(option): None for option in range(option_count)}
    probabilities: list[dict[str, object]] = [
        {'value': str(option), 'probability': probability if option < positive_count else 0.0}
        for option in range(option_count)
    ]
    answer: dict[str, object] = {
        'type': 'choice',
        'name': 'q',
        'choice': '0',
        'confidence': probability,
        'probabilities': probabilities,
    }
    model = mock_model(lambda _: decisions(answer))
    request = DecisionRequest(state='Choose an option.', questions={'q': ChoiceQuestion(criteria=criteria)})

    if matches:
        response = await model.decide(request, {})
        assert set(response.answers) == {'q'}
    else:
        with pytest.raises(UnexpectedModelBehavior, match='does not match its question'):
            await model.decide(request, {})


async def test_every_refusal_is_named(allow_model_requests: None):
    """Not recorded: no recording refuses more than one question."""
    refusals = ({'type': 'refusal', 'name': 'urgent'}, {'type': 'refusal', 'name': 'area'})
    agent = Agent(mock_model(lambda request: decisions(*refusals)), output_type=Ticket)
    with pytest.raises(ContentFilterError) as exc_info:
        await agent.run('Charged twice.')
    assert exc_info.value.message == snapshot(
        "Content filter triggered. The OpenAI Decisions API declined to answer: 'urgent', 'area'"
    )


@pytest.mark.vcr
async def test_image_only_prompt_streams(
    allow_model_requests: None,
    capture_model: OpenAIDecisionsModel,
    image_content: BinaryContent,
    request_capture: RequestCapture,
):
    """An image-only prompt can produce a typed answer through the streaming API."""
    agent = Agent(
        capture_model, output_type=bool, instructions='Does the pictured fruit have green flesh and black seeds?'
    )

    async with agent.run_stream([image_content]) as result:
        assert await result.get_output() is True

    assert request_capture.paths == ['/v1/decisions']
    request_body: JsonValue = json.loads(request_capture.raw_bodies[0])
    assert request_body == snapshot(
        {
            'model': 'gpt-6-luna',
            'input': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'input_text', 'text': '<image 1>'},
                        {'type': 'input_text', 'text': '<image 1>:'},
                        {'type': 'input_image', 'image_url': IsStr(regex=r'^data:image/jpeg;base64,.+$')},
                    ],
                }
            ],
            'questions': [
                {
                    'type': 'predicate',
                    'name': 'response',
                    'instructions': 'Does the pictured fruit have green flesh and black seeds?',
                }
            ],
        }
    )


@pytest.mark.vcr
async def test_image_in_history_and_text_in_current_prompt(
    allow_model_requests: None,
    capture_model: OpenAIDecisionsModel,
    disable_ssrf_protection_for_vcr: None,
    request_capture: RequestCapture,
    vcr: Cassette,
):
    """A prior image is downloaded and labeled in history, while the current text stays under judgement."""
    image_url = 'https://raw.githubusercontent.com/pydantic/pydantic-ai/main/tests/assets/kiwi.jpg'
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content=[ImageUrl(image_url)])]),
        ModelResponse(parts=[TextPart('This image was attached earlier.')]),
    ]

    result = await Agent(
        capture_model,
        output_type=bool,
        instructions='Does the pictured fruit have green flesh and black seeds?',
    ).run('Does the pictured fruit have green flesh and black seeds?', message_history=history)

    assert result.output is True
    request_body: JsonValue = json.loads(request_capture.raw_bodies[0])
    assert request_body == snapshot(
        {
            'model': 'gpt-6-luna',
            'input': [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'input_text',
                            'text': IsJson(
                                {
                                    'history': [
                                        {'user': '<image 1>'},
                                        {'assistant': 'This image was attached earlier.'},
                                    ],
                                    'text': 'Does the pictured fruit have green flesh and black seeds?',
                                }
                            ),
                        },
                        {'type': 'input_text', 'text': '<image 1>:'},
                        {'type': 'input_image', 'image_url': IsStr(regex=r'^data:image/jpeg;base64,.+$')},
                    ],
                }
            ],
            'questions': [
                {
                    'type': 'predicate',
                    'name': 'response',
                    'instructions': 'Does the pictured fruit have green flesh and black seeds?',
                }
            ],
        }
    )
    assert [(request.method, request.uri) for request in vcr.requests] == [
        ('GET', image_url),
        ('POST', 'https://api.openai.com/v1/decisions'),
    ]


async def test_images_from_assistant_and_tool_returns_keep_their_labels_and_order(allow_model_requests: None):
    """Assistant files and both tool-return forms stay beside their own images in the rendered conversation."""
    assistant_image = BinaryContent(b'assistant-image', media_type='image/png')
    tool_image = BinaryContent(b'tool-image', media_type='image/png')
    native_image = BinaryContent(b'native-image', media_type='image/png')
    history: list[ModelMessage] = [
        ModelRequest.user_text_prompt('Earlier image question.'),
        ModelResponse(
            parts=[
                TextPart('assistant before'),
                FilePart(content=assistant_image),
                TextPart('assistant after'),
            ]
        ),
        ModelRequest.user_text_prompt('Look up the image.'),
        ModelResponse(parts=[ToolCallPart('lookup', {}, tool_call_id='tool-1')]),
        ModelRequest(
            parts=[ToolReturnPart('lookup', ['tool before', tool_image, 'tool after'], tool_call_id='tool-1')]
        ),
        ModelResponse(
            parts=[
                NativeToolReturnPart(
                    'native_lookup',
                    ['native before', native_image, 'native after'],
                    provider_name='openai',
                )
            ]
        ),
    ]
    original_history = ModelMessagesTypeAdapter.dump_json(history)

    captured = Captured(boolean_answers)

    response = await mock_model(captured).request(
        history,
        None,
        ModelRequestParameters(
            output_tools=[
                ToolDefinition(
                    name='answer',
                    kind='output',
                    parameters_json_schema={
                        'type': 'object',
                        'properties': {'value': {'type': 'boolean', 'description': 'Does the input include an image?'}},
                        'required': ['value'],
                    },
                )
            ],
            output_mode='tool',
            allow_text_output=False,
        ),
    )

    assert len(response.parts) == 1
    answer_call = response.parts[0]
    assert isinstance(answer_call, ToolCallPart)
    assert answer_call.args == {'value': True}
    expected_state: dict[str, JsonValue] = {
        'history': [
            {'user': 'Earlier image question.'},
            {'assistant': 'assistant before'},
            {'assistant': '<image 1>'},
            {'assistant': 'assistant after'},
        ],
        'text': 'Look up the image.',
        'done': [
            {'tool_call': {'name': 'lookup', 'args': {}}},
            {'tool_return': {'name': 'lookup', 'content': '["tool before","<image 2>","tool after"]'}},
            {'tool_return': {'name': 'native_lookup', 'content': '["native before","<image 3>","native after"]'}},
        ],
    }
    request_body: JsonValue = json.loads(captured.requests[0].content)
    assert request_body == snapshot(
        {
            'model': 'gpt-6-luna',
            'input': [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'input_text',
                            'text': IsJson(expected_state),
                        },
                        {'type': 'input_text', 'text': '<image 1>:'},
                        {'type': 'input_image', 'image_url': 'data:image/png;base64,YXNzaXN0YW50LWltYWdl'},
                        {'type': 'input_text', 'text': '<image 2>:'},
                        {'type': 'input_image', 'image_url': 'data:image/png;base64,dG9vbC1pbWFnZQ=='},
                        {'type': 'input_text', 'text': '<image 3>:'},
                        {'type': 'input_image', 'image_url': 'data:image/png;base64,bmF0aXZlLWltYWdl'},
                    ],
                }
            ],
            'questions': [
                {
                    'type': 'predicate',
                    'name': 'value',
                    'instructions': '{"field": "value", "question": "Does the input include an image?"}',
                }
            ],
        }
    )
    assert ModelMessagesTypeAdapter.dump_json(history) == original_history


@pytest.mark.parametrize(
    'native',
    [pytest.param(False, id='tool-return'), pytest.param(True, id='native-tool-return')],
)
async def test_image_urls_in_tool_returns_are_downloaded_and_labeled(allow_model_requests: None, native: bool):
    """Image URLs in either tool-return form are downloaded and labeled in their original position."""
    image_url = ImageUrl('https://example.com/tool-image.png')
    history: list[ModelMessage] = [ModelRequest.user_text_prompt('Look up the image.')]
    if native:
        history.append(
            ModelResponse(
                parts=[NativeToolReturnPart('native_lookup', ['before', image_url, 'after'], provider_name='openai')]
            )
        )
    else:
        history.extend(
            [
                ModelResponse(parts=[ToolCallPart('lookup', {}, tool_call_id='tool-1')]),
                ModelRequest(parts=[ToolReturnPart('lookup', ['before', image_url, 'after'], tool_call_id='tool-1')]),
            ]
        )
    captured = Captured(boolean_answers)
    agent = Agent(mock_model(captured), output_type=bool, instructions='Does the lookup contain an image?')

    with patch(
        'pydantic_ai.models.decision.download_item',
        new_callable=AsyncMock,
        return_value={'data': b'tool-image', 'data_type': 'image/png'},
    ) as download:
        result = await agent.run('Does the lookup contain an image?', message_history=history)

    assert result.output is True
    download.assert_awaited_once_with(image_url, data_format='bytes')
    expected_state: dict[str, JsonValue]
    if native:
        expected_state = {
            'history': [
                {'user': 'Look up the image.'},
                {'tool_return': {'name': 'native_lookup', 'content': '["before","<image 1>","after"]'}},
            ],
            'text': 'Does the lookup contain an image?',
        }
    else:
        expected_state = {
            'history': [
                {'user': 'Look up the image.'},
                {'tool_call': {'name': 'lookup', 'args': {}}},
                {'tool_return': {'name': 'lookup', 'content': '["before","<image 1>","after"]'}},
            ],
            'text': 'Does the lookup contain an image?',
        }
    request_body: JsonValue = json.loads(captured.requests[0].content)
    assert request_body == snapshot(
        {
            'model': 'gpt-6-luna',
            'input': [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'input_text',
                            'text': IsJson(expected_state),
                        },
                        {'type': 'input_text', 'text': '<image 1>:'},
                        {'type': 'input_image', 'image_url': 'data:image/png;base64,dG9vbC1pbWFnZQ=='},
                    ],
                }
            ],
            'questions': [
                {'type': 'predicate', 'name': 'response', 'instructions': 'Does the lookup contain an image?'}
            ],
        }
    )


@pytest.mark.parametrize(
    'native',
    [pytest.param(False, id='tool-return'), pytest.param(True, id='native-tool-return')],
)
async def test_failed_tool_return_image_keeps_order_and_one_error_wrapper(allow_model_requests: None, native: bool):
    """A failed multimodal return keeps its text around the image and receives one error wrapper in history."""
    image = BinaryContent(b'tool-image', media_type='image/png')
    failed_content: list[str | BinaryContent] = ['before', image, 'after']
    history: list[ModelMessage] = [ModelRequest.user_text_prompt('Look up this image.')]
    if native:
        history.extend(
            [
                ModelResponse(
                    parts=[NativeToolCallPart('native_lookup', {}, tool_call_id='native-1', provider_name='openai')]
                ),
                ModelResponse(
                    parts=[
                        NativeToolReturnPart(
                            'native_lookup',
                            failed_content,
                            tool_call_id='native-1',
                            provider_name='openai',
                            outcome='failed',
                        )
                    ]
                ),
            ]
        )
    else:
        history.extend(
            [
                ModelResponse(parts=[ToolCallPart('lookup', {}, tool_call_id='tool-1')]),
                ModelRequest(parts=[ToolReturnPart('lookup', failed_content, tool_call_id='tool-1', outcome='failed')]),
            ]
        )
    original_history = ModelMessagesTypeAdapter.dump_json(history)
    captured = Captured(boolean_answers)

    result = await Agent(
        mock_model(captured), output_type=bool, instructions='Does the failed lookup retain its image?'
    ).run('Does the failed lookup retain its image?', message_history=history)

    assert result.output is True
    expected_state: dict[str, JsonValue]
    if native:
        expected_state = {
            'history': [
                {'user': 'Look up this image.'},
                {'tool_call': {'name': 'native_lookup', 'args': {}}},
                {
                    'tool_return': {
                        'name': 'native_lookup',
                        'content': '{"error":"[\\"before\\",\\"<image 1>\\",\\"after\\"]"}',
                    }
                },
            ],
            'text': 'Does the failed lookup retain its image?',
        }
    else:
        expected_state = {
            'history': [
                {'user': 'Look up this image.'},
                {'tool_call': {'name': 'lookup', 'args': {}}},
                {
                    'tool_return': {
                        'name': 'lookup',
                        'content': '{"error":"[\\"before\\",\\"<image 1>\\",\\"after\\"]"}',
                    }
                },
            ],
            'text': 'Does the failed lookup retain its image?',
        }
    request_body: JsonValue = json.loads(captured.requests[0].content)
    assert request_body == snapshot(
        {
            'model': 'gpt-6-luna',
            'input': [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'input_text',
                            'text': IsJson(expected_state),
                        },
                        {'type': 'input_text', 'text': '<image 1>:'},
                        {'type': 'input_image', 'image_url': 'data:image/png;base64,dG9vbC1pbWFnZQ=='},
                    ],
                }
            ],
            'questions': [
                {'type': 'predicate', 'name': 'response', 'instructions': 'Does the failed lookup retain its image?'}
            ],
        }
    )
    assert ModelMessagesTypeAdapter.dump_json(history) == original_history


def pick_first_route(body: DecisionCreateParams) -> httpx2.Response | None:
    for question in body['questions']:
        if question.get('name') == 'route' and question['type'] == 'choice':
            labels: list[str] = []
            for choice in question['choices']:
                value = choice['value']
                assert isinstance(value, str)
                labels.append(value)
            picked = labels[0]
            return decisions(
                {
                    'type': 'choice',
                    'name': 'route',
                    'choice': picked,
                    'probabilities': [
                        {'value': label, 'probability': 1.0 if label == picked else 0.0} for label in labels
                    ],
                    'confidence': 1.0,
                }
            )
    return None


async def test_image_is_prepared_once_for_route_then_fill(allow_model_requests: None):
    """The route and fill receive one identical prepared image input and aggregate their usage."""

    def route_and_fill(request: httpx2.Request) -> httpx2.Response:
        request_body: DecisionCreateParams = json.loads(request.content)
        route_response = pick_first_route(request_body)
        if route_response is not None:
            return route_response
        return decisions(
            {'type': 'predicate', 'name': 'urgent', 'probability': 0.9},
            {
                'type': 'choice',
                'name': 'area',
                'choice': 'billing',
                'probabilities': [{'value': 'billing', 'probability': 1.0}, {'value': 'bug', 'probability': 0.0}],
                'confidence': 1.0,
            },
        )

    captured = Captured(route_and_fill)
    image_url = ImageUrl('https://example.com/receipt.png', vendor_metadata={'detail': 'high'})
    long_prompt = 'The receipt is attached. ' + 'Some detail nobody asked about. ' * 3000
    agent = Agent(mock_model(captured), output_type=[Ticket, Mood])

    with patch('pydantic_ai.models.decision.download_item', new_callable=AsyncMock) as download:
        download.return_value = {'data': b'picture', 'data_type': 'image/png'}
        result = await agent.run([long_prompt, image_url])

    download.assert_awaited_once()
    assert result.output == Ticket(urgent=True, area='billing')
    expected_input: list[dict[str, object]] = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'input_text',
                    'text': IsStr(
                        regex=r'(?s)^The receipt is attached\. (?:Some detail nobody asked about\. ){3000}\s*<image 1>$'
                    ),
                },
                {'type': 'input_text', 'text': '<image 1>:'},
                {
                    'type': 'input_image',
                    'image_url': BinaryContent(b'picture', media_type='image/png').data_uri,
                    'detail': 'high',
                },
            ],
        }
    ]
    request_bodies: list[DecisionCreateParams] = [json.loads(request.content) for request in captured.requests]
    assert request_bodies == snapshot(
        [
            {
                'model': 'gpt-6-luna',
                'input': expected_input,
                'questions': [
                    {
                        'type': 'choice',
                        'name': 'route',
                        'instructions': 'Which of these does this call for?',
                        'choices': [
                            {'value': 'Ticket', 'description': 'Triage a support ticket.'},
                            {'value': 'Mood', 'description': "Read the customer's mood."},
                        ],
                    }
                ],
            },
            {
                'model': 'gpt-6-luna',
                'input': expected_input,
                'questions': [
                    {
                        'type': 'predicate',
                        'name': 'urgent',
                        'instructions': '{"field": "urgent", "premise": "If the user\'s request calls for Ticket: Triage a support ticket.", "question": "Does this need a reply within the hour?"}',
                    },
                    {
                        'type': 'choice',
                        'name': 'area',
                        'instructions': '{"field": "area", "premise": "If the user\'s request calls for Ticket: Triage a support ticket.", "question": "Which team owns it?"}',
                        'choices': [{'value': 'billing'}, {'value': 'bug'}],
                    },
                ],
            },
        ]
    )
    assert result.response.provider_details == {
        'confidence': {'urgent': 0.8, 'area': 1.0},
        'probabilities': {'area': {'billing': 1.0, 'bug': 0.0}},
        'scores': {},
        'route': {'choice': 'Ticket', 'probabilities': {'Ticket': 1.0, 'Mood': 0.0}, 'offered': ['Ticket', 'Mood']},
        'requests': 2,
    }
    assert result.usage.input_tokens == 792


@pytest.mark.parametrize(
    ('source', 'unsupported'),
    [
        pytest.param('prompt', BinaryContent(b'audio', media_type='audio/mpeg'), id='audio-prompt'),
        pytest.param('assistant-file', BinaryContent(b'%PDF', media_type='application/pdf'), id='document-assistant'),
        pytest.param('tool-return', BinaryContent(b'video', media_type='video/mp4'), id='video-tool-return'),
    ],
)
async def test_unsupported_files_fail_before_a_decisions_request(
    source: Literal['prompt', 'assistant-file', 'tool-return'],
    unsupported: BinaryContent,
    allow_model_requests: None,
):
    """Audio, documents, and video remain unsupported across user, assistant, and tool history."""
    history: list[ModelMessage] = []
    prompt: str | list[BinaryContent] = 'Classify this content.'
    if source == 'prompt':
        prompt = [unsupported]
    elif source == 'assistant-file':
        history = [
            ModelRequest.user_text_prompt('Earlier prompt.'),
            ModelResponse(parts=[FilePart(content=unsupported)]),
        ]
    else:
        history = [
            ModelRequest.user_text_prompt('Look up the attachment.'),
            ModelResponse(parts=[ToolCallPart('lookup', {}, tool_call_id='tool-1')]),
            ModelRequest(parts=[ToolReturnPart('lookup', unsupported, tool_call_id='tool-1')]),
        ]
    captured = Captured(boolean_answers)

    with pytest.raises(UserError, match='unsupported file: this model accepts text and images only'):
        await Agent(mock_model(captured), output_type=bool, instructions='Does this contain an image?').run(
            prompt, message_history=history
        )
    assert captured.requests == []


@pytest.mark.parametrize('source', ['prompt', 'tool-return'])
async def test_unsupported_audio_urls_fail_before_a_decisions_request(
    source: Literal['prompt', 'tool-return'], allow_model_requests: None
):
    """Audio URLs are rejected in user prompts and tool returns before a Decisions request is sent."""
    audio_url = AudioUrl('https://example.com/audio.mp3')
    history: list[ModelMessage] = []
    prompt: str | list[AudioUrl] = 'Classify this content.'
    if source == 'prompt':
        prompt = [audio_url]
    else:
        history = [
            ModelRequest.user_text_prompt('Look up the attachment.'),
            ModelResponse(parts=[ToolCallPart('lookup', {}, tool_call_id='tool-1')]),
            ModelRequest(parts=[ToolReturnPart('lookup', audio_url, tool_call_id='tool-1')]),
        ]
    captured = Captured(boolean_answers)
    agent = Agent(mock_model(captured), output_type=bool, instructions='Does this contain an image?')

    with pytest.raises(UserError, match='unsupported file: this model accepts text and images only'):
        await agent.run(prompt, message_history=history)
    assert captured.requests == []


async def test_image_download_error_prevents_a_decisions_request(allow_model_requests: None):
    captured = Captured(boolean_answers)
    agent = Agent(mock_model(captured), output_type=bool, instructions='Does the input contain an image?')

    with (
        patch(
            'pydantic_ai.models.decision.download_item',
            new_callable=AsyncMock,
            side_effect=httpx2.ConnectError('image download failed'),
        ),
        pytest.raises(httpx2.ConnectError, match='image download failed'),
    ):
        await agent.run([ImageUrl('https://example.com/missing.png')])

    assert captured.requests == []


async def test_image_url_with_non_image_content_is_rejected_before_a_decisions_request(
    allow_model_requests: None,
):
    image_url = ImageUrl('https://example.com/not-an-image')
    captured = Captured(boolean_answers)
    agent = Agent(mock_model(captured), output_type=bool, instructions='Does the input contain an image?')

    with (
        patch(
            'pydantic_ai.models.decision.download_item',
            new_callable=AsyncMock,
            return_value={'data': b'%PDF', 'data_type': 'application/pdf'},
        ) as download,
        pytest.raises(UserError, match='returned content that is not an image'),
    ):
        await agent.run([image_url])

    download.assert_awaited_once_with(image_url, data_format='bytes')
    assert captured.requests == []


@pytest.mark.parametrize(
    ('model_settings', 'threshold_name'),
    [
        pytest.param(
            OpenAIDecisionsModelSettings(decision_boolean_threshold=2.0),
            'decision_boolean_threshold',
            id='boolean-threshold',
        ),
        pytest.param(
            OpenAIDecisionsModelSettings(decision_route_threshold=2.0),
            'decision_route_threshold',
            id='route-threshold',
        ),
    ],
)
async def test_invalid_threshold_prevents_image_download(
    allow_model_requests: None,
    model_settings: OpenAIDecisionsModelSettings,
    threshold_name: str,
):
    """Invalid decision thresholds are rejected before downloading an image prompt."""
    captured = Captured(boolean_answers)
    agent = Agent(
        mock_model(captured),
        output_type=bool,
        instructions='Does the input contain an image?',
        model_settings=model_settings,
    )

    with (
        patch(
            'pydantic_ai.models.decision.download_item',
            new_callable=AsyncMock,
            side_effect=httpx2.ConnectError('image download failed'),
        ) as download,
        pytest.raises(UserError, match=f'`{threshold_name}` must be between 0 and 1'),
    ):
        await agent.run([ImageUrl('https://example.com/missing.png')])

    download.assert_not_awaited()
    assert captured.requests == []


async def test_unfillable_output_prevents_image_download(allow_model_requests: None):
    """An unsupported output field is rejected before downloading an image prompt."""

    class FreeTextOutput(BaseModel):
        free_text: str = Field(description='A free-form explanation.')

    captured = Captured(boolean_answers)
    agent = Agent(mock_model(captured), output_type=FreeTextOutput)

    with (
        patch(
            'pydantic_ai.models.decision.download_item',
            new_callable=AsyncMock,
            side_effect=httpx2.ConnectError('image download failed'),
        ) as download,
        pytest.raises(UserError, match="Output field 'free_text' is not supported by this model"),
    ):
        await agent.run([ImageUrl('https://example.com/missing.png')])

    download.assert_not_awaited()
    assert captured.requests == []


async def test_image_limit_prevents_url_download(allow_model_requests: None):
    """The provider's image limit is checked before resolving evidence URLs or sending a request."""
    captured = Captured(boolean_answers)
    image_urls: list[str | ImageUrl] = ['Review these images.']
    image_urls.extend(ImageUrl(f'https://example.com/{index}.png') for index in range(129))
    agent = Agent(mock_model(captured), output_type=bool, instructions='Are these images receipts?')

    with (
        patch(
            'pydantic_ai.models.decision.download_item',
            new_callable=AsyncMock,
            side_effect=httpx2.ConnectError('image download should not start'),
        ) as download,
        pytest.raises(ModelAPIError, match='accepts at most 128 images; got 129'),
    ):
        await agent.run(image_urls)

    download.assert_not_awaited()
    assert captured.requests == []


async def test_route_limit_prevents_image_download(allow_model_requests: None):
    """An overfull route question is rejected before downloading an image prompt."""

    def inspect_ticket() -> None:
        """Inspect the ticket."""

    captured = Captured(boolean_answers)
    client = AsyncOpenAI(
        api_key='test', max_retries=0, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(captured))
    )
    model = OpenAIDecisionsModel(
        'gpt-6-luna',
        provider=OpenAIDecisionsProvider(openai_client=client),
        profile=DecisionModelProfile(decision_max_choice_options=1),
    )
    agent = Agent(model, output_type=bool, tools=[inspect_ticket], instructions='Is this safe?')

    with (
        patch(
            'pydantic_ai.models.decision.download_item',
            new_callable=AsyncMock,
            side_effect=httpx2.ConnectError('image download failed'),
        ) as download,
        pytest.raises(UserError, match='being offered 2 routes'),
    ):
        await agent.run([ImageUrl('https://example.com/missing.png')])

    download.assert_not_awaited()
    assert captured.requests == []


async def test_overfull_speculation_picks_then_fills_under_question_limit(allow_model_requests: None):
    """When the speculative questions exceed the cap, a route and its one-field fill can still succeed."""

    class Approve(BaseModel):
        """Approve this request."""

        approved: bool = Field(description='Can this be approved?')

    class Notify(BaseModel):
        """Notify the owner."""

        notify: bool = Field(description='Should the owner be notified?')

    def route_and_fill(request: httpx2.Request) -> httpx2.Response:
        body: DecisionCreateParams = json.loads(request.content)
        route_response = pick_first_route(body)
        if route_response is not None:
            return route_response
        question = next(iter(body['questions']))
        question_name = question.get('name')
        assert question_name is not None
        return decisions({'type': 'predicate', 'name': question_name, 'probability': 0.9})

    captured = Captured(route_and_fill)
    model = mock_model(captured)
    image_url = ImageUrl('https://example.com/ticket.png')
    agent = Agent(model, output_type=[Approve, Notify], instructions='Choose an action for this request.')

    with (
        patch.object(OpenAIDecisionsModel, 'max_questions', 1),
        patch(
            'pydantic_ai.models.decision.download_item',
            new_callable=AsyncMock,
            return_value={'data': b'picture', 'data_type': 'image/png'},
        ) as download,
    ):
        result = await agent.run(['Review the ticket.', image_url])

    download.assert_awaited_once_with(image_url, data_format='bytes')
    assert result.output == Approve(approved=True)
    expected_input: list[dict[str, object]] = [
        {
            'role': 'user',
            'content': [
                {'type': 'input_text', 'text': 'Review the ticket.\n\n<image 1>'},
                {'type': 'input_text', 'text': '<image 1>:'},
                {
                    'type': 'input_image',
                    'image_url': BinaryContent(b'picture', media_type='image/png').data_uri,
                },
            ],
        }
    ]
    request_bodies: list[DecisionCreateParams] = [json.loads(request.content) for request in captured.requests]
    assert request_bodies == snapshot(
        [
            {
                'model': 'gpt-6-luna',
                'input': expected_input,
                'questions': [
                    {
                        'type': 'choice',
                        'name': 'route',
                        'instructions': '{"question": "Which of these does this call for?", "background": "Choose an action for this request."}',
                        'choices': [
                            {'value': 'Approve', 'description': 'Approve this request.'},
                            {'value': 'Notify', 'description': 'Notify the owner.'},
                        ],
                    }
                ],
            },
            {
                'model': 'gpt-6-luna',
                'input': expected_input,
                'questions': [
                    {
                        'type': 'predicate',
                        'name': 'approved',
                        'instructions': '{"field": "approved", "premise": "If the user\'s request calls for Approve: Approve this request.", "question": "Can this be approved?", "background": "Choose an action for this request."}',
                    }
                ],
            },
        ]
    )


@pytest.mark.parametrize('selected_route', ['Big', 'Small'])
async def test_selected_route_over_question_limit_hands_off_to_fallback(
    allow_model_requests: None, selected_route: Literal['Big', 'Small']
):
    """An unfillable picked route hands off to the fallback, while a feasible route is filled by Decisions."""
    big_field_names: list[str] = [f'field_{index}' for index in range(201)]
    big_properties: dict[str, JsonSchemaValue] = {
        name: {'type': 'boolean', 'description': 'Does this apply?'} for name in big_field_names
    }
    big_schema: ObjectJsonSchema = {'type': 'object', 'properties': big_properties, 'required': big_field_names}
    small_schema: ObjectJsonSchema = {
        'type': 'object',
        'properties': {'ready': {'type': 'boolean', 'description': 'Is the small route ready?'}},
        'required': ['ready'],
    }
    output_tools: list[ToolDefinition] = [
        ToolDefinition(
            name='Big', description='Return the large result.', kind='output', parameters_json_schema=big_schema
        ),
        ToolDefinition(
            name='Small', description='Return the small result.', kind='output', parameters_json_schema=small_schema
        ),
    ]
    question_names_by_request: list[list[str]] = []

    def route_and_fill(request: httpx2.Request) -> httpx2.Response:
        request_body: JsonValue = json.loads(request.content)
        assert isinstance(request_body, dict)
        questions = request_body['questions']
        assert isinstance(questions, list)
        question_names: list[str] = []
        for question in questions:
            assert isinstance(question, dict)
            name = question['name']
            assert isinstance(name, str)
            question_names.append(name)
        question_names_by_request.append(question_names)

        route_question = next(
            (question for question in questions if isinstance(question, dict) and question.get('name') == 'route'), None
        )
        if route_question is None:
            return boolean_answers(request)

        assert isinstance(route_question, dict)
        route_labels: list[str] = []
        choices = route_question['choices']
        assert isinstance(choices, list)
        for choice in choices:
            assert isinstance(choice, dict)
            label = choice['value']
            assert isinstance(label, str)
            route_labels.append(label)
        assert selected_route in route_labels
        return decisions(
            {
                'type': 'choice',
                'name': 'route',
                'choice': selected_route,
                'probabilities': [
                    {'value': label, 'probability': 1.0 if label == selected_route else 0.0} for label in route_labels
                ],
                'confidence': 1.0,
            }
        )

    captured = Captured(route_and_fill)
    primary = mock_model(captured)
    fallback = TestModel()
    model = FallbackModel(primary, fallback)

    with patch.object(OpenAIDecisionsModel, 'max_questions', 200):
        response = await model.request(
            [ModelRequest.user_text_prompt('Classify this request.')],
            None,
            ModelRequestParameters(output_tools=output_tools, output_mode='tool', allow_text_output=False),
        )

    assert len(response.parts) == 1
    output_call = response.parts[0]
    assert isinstance(output_call, ToolCallPart)
    assert output_call.tool_name == selected_route
    if selected_route == 'Big':
        assert fallback.last_model_request_parameters is not None
        assert question_names_by_request == [['route']]
    else:
        assert fallback.last_model_request_parameters is None
        assert question_names_by_request == [['route'], ['ready']]
    assert len(captured.requests) == len(question_names_by_request)


@pytest.mark.parametrize(
    ('single_output_with_tool', 'fields_per_output'),
    [
        pytest.param(False, 100, id='two-feasible-routes-at-200'),
        pytest.param(True, 200, id='single-output-with-tool-at-200'),
    ],
)
async def test_exact_question_limit_route_then_fill_stays_under_cap(
    allow_model_requests: None, single_output_with_tool: bool, fields_per_output: int
):
    """Overfull route speculation falls back to a route pick and a feasible fill at the exact question limit.

    Two output routes have enough evidence to fit the existing token heuristic, so the question cap alone splits
    their 201-question speculative request.
    """
    field_names: list[str] = [f'field_{index}' for index in range(fields_per_output)]
    properties: dict[str, JsonSchemaValue] = {
        name: {'type': 'boolean', 'description': 'Does this apply?'} for name in field_names
    }
    output_schema: ObjectJsonSchema = {'type': 'object', 'properties': properties, 'required': field_names}
    selected_output = ToolDefinition(
        name='selected_output',
        description='Return the answers.',
        kind='output',
        parameters_json_schema=output_schema,
    )
    output_tools: list[ToolDefinition] = [selected_output]
    function_tools: list[ToolDefinition] = []
    if not single_output_with_tool:
        other_output = ToolDefinition(
            name='other_output',
            description='Return the other answers.',
            kind='output',
            parameters_json_schema=output_schema,
        )
        output_tools.append(other_output)
    else:
        function_tools.append(
            ToolDefinition(
                name='inspect_ticket',
                description='Inspect the ticket.',
                kind='function',
                parameters_json_schema={'type': 'object', 'properties': {}, 'required': []},
            )
        )

    question_names_by_request: list[list[str]] = []

    def route_then_fill(request: httpx2.Request) -> httpx2.Response:
        request_body: JsonValue = json.loads(request.content)
        assert isinstance(request_body, dict)
        questions = request_body['questions']
        assert isinstance(questions, list)
        question_names: list[str] = []
        for question in questions:
            assert isinstance(question, dict)
            name = question['name']
            assert isinstance(name, str)
            question_names.append(name)
        question_names_by_request.append(question_names)

        route_question = next(
            (question for question in questions if isinstance(question, dict) and question['name'] == 'route'), None
        )
        if route_question is None:
            return boolean_answers(request)

        assert isinstance(route_question, dict)
        choices = route_question['choices']
        assert isinstance(choices, list)
        route_labels: list[str] = []
        for choice in choices:
            assert isinstance(choice, dict)
            label = choice['value']
            assert isinstance(label, str)
            route_labels.append(label)
        picked = selected_output.name
        assert picked in route_labels
        return decisions(
            {
                'type': 'choice',
                'name': 'route',
                'choice': picked,
                'probabilities': [
                    {'value': label, 'probability': 1.0 if label == picked else 0.0} for label in route_labels
                ],
                'confidence': 1.0,
            }
        )

    captured = Captured(route_then_fill)
    model = mock_model(captured)
    image_url = ImageUrl('https://example.com/ticket.png')
    review_text = 'Review the ticket.'
    if not single_output_with_tool:
        review_text += ' This record includes a payment of $42 and needs careful review.' * 470
    with (
        patch.object(OpenAIDecisionsModel, 'max_questions', 200),
        patch(
            'pydantic_ai.models.decision.download_item',
            new_callable=AsyncMock,
            return_value={'data': b'picture', 'data_type': 'image/png'},
        ) as download,
    ):
        response = await model.request(
            [ModelRequest(parts=[UserPromptPart([review_text, image_url])])],
            None,
            ModelRequestParameters(
                output_tools=output_tools,
                function_tools=function_tools,
                output_mode='tool',
                allow_text_output=False,
            ),
        )

    download.assert_awaited_once_with(image_url, data_format='bytes')
    [output_call] = [part for part in response.parts if isinstance(part, ToolCallPart)]
    assert output_call.tool_name == selected_output.name
    assert output_call.args == {name: True for name in field_names}
    assert len(captured.requests) == 2
    request_inputs: list[JsonValue] = []
    for request in captured.requests:
        request_body: JsonValue = json.loads(request.content)
        assert isinstance(request_body, dict)
        request_inputs.append(request_body['input'])
    assert request_inputs[0] == request_inputs[1]
    assert question_names_by_request == [
        ['route'],
        field_names,
    ]


async def test_201_questions_prevent_image_download(allow_model_requests: None):
    """A single-output request with 201 fields is rejected before resolving image URLs or sending it."""
    captured = Captured(boolean_answers)
    model = mock_model(captured)
    properties: dict[str, JsonSchemaValue] = {
        f'q{index}': {'type': 'boolean', 'description': 'Does this apply?'} for index in range(201)
    }
    required: list[str] = list(properties)
    output_schema: ObjectJsonSchema = {'type': 'object', 'properties': properties, 'required': required}
    output_tool = ToolDefinition(
        name='final_result',
        description='Answer the questions.',
        kind='output',
        parameters_json_schema=output_schema,
    )
    image_url = ImageUrl('https://example.com/ticket.png')

    with (
        patch(
            'pydantic_ai.models.decision.download_item',
            new_callable=AsyncMock,
            side_effect=httpx2.ConnectError('image download should not start'),
        ) as download,
        pytest.raises(ModelAPIError, match='accepts at most 200 questions; got 201'),
    ):
        await model.request(
            [ModelRequest(parts=[UserPromptPart(['Review the ticket.', image_url])])],
            None,
            ModelRequestParameters(output_tools=[output_tool], output_mode='tool', allow_text_output=False),
        )

    download.assert_not_awaited()
    assert captured.requests == []


async def test_unfillable_forced_tool_prevents_image_download(allow_model_requests: None):
    """A forced tool with an unsupported argument is handed off before downloading a history image."""
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content=[ImageUrl('https://example.com/missing.png')])]),
        ModelResponse(parts=[ToolCallPart('inspect_ticket', {}, 'call_1')]),
        ModelRequest(parts=[ToolReturnPart('inspect_ticket', 'Inspected.', 'call_1')]),
    ]
    function_tools: list[ToolDefinition] = [
        ToolDefinition(name='inspect_ticket', description='Inspect the ticket.'),
        ToolDefinition(
            name='write_note',
            description='Write a note.',
            parameters_json_schema={
                'type': 'object',
                'properties': {'note': {'type': 'string', 'description': 'A note to write.'}},
                'required': ['note'],
            },
        ),
    ]
    captured = Captured(boolean_answers)
    model = mock_model(captured)

    with (
        patch(
            'pydantic_ai.models.decision.download_item',
            new_callable=AsyncMock,
            side_effect=httpx2.ConnectError('image download failed'),
        ) as download,
        pytest.raises(UnfillableRoute) as exc_info,
    ):
        await model.request(
            history,
            None,
            ModelRequestParameters(function_tools=function_tools, allow_text_output=False),
        )

    assert exc_info.value.route == 'write_note'
    download.assert_not_awaited()
    assert captured.requests == []


@pytest.mark.parametrize(
    'model_settings',
    [
        pytest.param(None, id='without-settings'),
        pytest.param(OpenAIDecisionsModelSettings(extra_body=['ignored']), id='ignored-extra-body'),
    ],
)
async def test_forced_argumentless_tool_skips_image_preparation(
    allow_model_requests: None, model_settings: OpenAIDecisionsModelSettings | None
):
    """A sole argumentless route skips image preparation even with irrelevant request settings."""
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content=[ImageUrl('https://example.com/missing.png')])]),
        ModelResponse(parts=[ToolCallPart('inspect_ticket', {}, 'call_1')]),
        ModelRequest(parts=[ToolReturnPart('inspect_ticket', 'Inspected.', 'call_1')]),
    ]
    function_tools: list[ToolDefinition] = [
        ToolDefinition(name='inspect_ticket', description='Inspect the ticket.'),
        ToolDefinition(name='finish', description='Finish the task.'),
    ]
    captured = Captured(boolean_answers)
    model = mock_model(captured)

    with patch(
        'pydantic_ai.models.decision.download_item',
        new_callable=AsyncMock,
        side_effect=httpx2.ConnectError('image download failed'),
    ) as download:
        response = await model.request(
            history,
            model_settings,
            ModelRequestParameters(function_tools=function_tools, allow_text_output=False),
        )

    [tool_call] = [part for part in response.parts if isinstance(part, ToolCallPart)]
    assert tool_call.tool_name == 'finish'
    assert tool_call.args == {}
    download.assert_not_awaited()
    assert captured.requests == []


async def test_concurrent_image_requests_keep_their_own_inputs(allow_model_requests: None):
    """Concurrent runs on one model retain the image that belongs to each prompt."""
    first_url = 'https://example.com/first.png'
    second_url = 'https://example.com/second.png'
    first_data_uri = 'data:image/png;base64,Zmlyc3Q='
    second_data_uri = 'data:image/png;base64,c2Vjb25k'
    downloads: list[str] = []
    both_downloads_started = anyio.Event()

    async def download(item: ImageUrl, *, data_format: str) -> dict[str, bytes | str]:
        assert data_format == 'bytes'
        downloads.append(item.url)
        if len(downloads) == 2:
            both_downloads_started.set()
        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            await both_downloads_started.wait()
        return {'data': b'first' if item.url == first_url else b'second', 'data_type': 'image/png'}

    captured = Captured(boolean_answers)
    agent = Agent(mock_model(captured), output_type=bool, instructions='Does the input contain an image?')
    outputs: list[bool] = []

    async def run(prompt: str, url: str) -> None:
        result = await agent.run([prompt, ImageUrl(url)])
        outputs.append(result.output)

    with patch('pydantic_ai.models.decision.download_item', new_callable=AsyncMock, side_effect=download):
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(run, 'first concurrent prompt', first_url)
            task_group.start_soon(run, 'second concurrent prompt', second_url)

    assert sorted(downloads) == sorted([first_url, second_url])
    assert outputs == [True, True]
    assert len(captured.requests) == 2
    sent: dict[str, str] = {}
    for request in captured.requests:
        body = json.loads(request.content)
        input_message = body['input'][0]
        content = input_message['content']
        state_text = content[0]['text']
        image_url = content[-1]['image_url']
        if state_text.startswith('first concurrent prompt'):
            sent['first concurrent prompt'] = image_url
        else:
            sent['second concurrent prompt'] = image_url
    assert sent == {
        'first concurrent prompt': first_data_uri,
        'second concurrent prompt': second_data_uri,
    }


async def test_direct_decide_keeps_json_state_as_text(allow_model_requests: None):
    """An API-shaped JSON array is a decision state, not a native Decisions input message."""
    captured = Captured(lambda request: decisions({'type': 'predicate', 'name': 'q', 'probability': 0.9}))
    state: JsonValue = [{'type': 'input_text', 'text': 'a JSON state value'}]

    await mock_model(captured).decide(
        DecisionRequest(state=state, questions={'q': NoulQuestion(instructions='Is this true?')}), {}
    )

    request_body: JsonValue = json.loads(captured.requests[0].content)
    assert isinstance(request_body, dict)
    sent_input = request_body['input']
    assert isinstance(sent_input, str)
    assert json.loads(sent_input) == state


@pytest.mark.parametrize(
    ('vendor_metadata', 'detail_field'),
    [
        pytest.param(None, {}, id='without-metadata'),
        pytest.param({'detail': 'low'}, {'detail': 'low'}, id='low-detail'),
        pytest.param({'detail': 'high'}, {'detail': 'high'}, id='high-detail'),
        pytest.param({'detail': 'auto'}, {'detail': 'auto'}, id='auto-detail'),
        pytest.param({'detail': 'original'}, {'detail': 'original'}, id='original-detail'),
        pytest.param({'source': 'test'}, {'detail': 'auto'}, id='metadata-defaults-to-auto'),
        pytest.param({'detail': None}, {'detail': None}, id='explicit-none-detail'),
    ],
)
async def test_direct_decide_sends_image_evidence_with_ordered_labels(
    allow_model_requests: None,
    vendor_metadata: dict[str, str | None] | None,
    detail_field: dict[str, str | None],
):
    first_image = BinaryContent(b'first', media_type='image/png', vendor_metadata=vendor_metadata)
    second_image = BinaryContent(b'second', media_type='image/jpeg')
    state: JsonValue = {'history': [{'user': 'Receipt <image 1>'}], 'text': 'Compare <image 2>.'}
    captured = Captured(lambda _: decisions({'type': 'predicate', 'name': 'q', 'probability': 0.9}))

    response = await mock_model(captured).decide(
        DecisionRequest(
            state=state,
            questions={'q': NoulQuestion(instructions='Is it a receipt?')},
            images=(first_image, second_image),
        ),
        {},
    )

    assert response.answers == {'q': NoulAnswer(noul=0.9)}
    request_body: JsonValue = json.loads(captured.requests[0].content)
    assert request_body == snapshot(
        {
            'model': 'gpt-6-luna',
            'input': [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'input_text',
                            'text': IsJson({'history': [{'user': 'Receipt <image 1>'}], 'text': 'Compare <image 2>.'}),
                        },
                        {'type': 'input_text', 'text': '<image 1>:'},
                        {
                            'type': 'input_image',
                            'image_url': 'data:image/png;base64,Zmlyc3Q=',
                            **detail_field,
                        },
                        {'type': 'input_text', 'text': '<image 2>:'},
                        {'type': 'input_image', 'image_url': 'data:image/jpeg;base64,c2Vjb25k'},
                    ],
                }
            ],
            'questions': [{'type': 'predicate', 'name': 'q', 'instructions': 'Is it a receipt?'}],
        }
    )


async def test_direct_decide_rejects_non_image_evidence_before_a_request(allow_model_requests: None):
    captured = Captured(lambda _: decisions(URGENT))
    model = mock_model(captured)

    with pytest.raises(UserError, match=r'`request\.images` contains a non-image'):
        await model.decide(
            DecisionRequest(
                state='Review this document.',
                questions={'q': NoulQuestion()},
                images=(BinaryContent(b'%PDF', media_type='application/pdf'),),
            ),
            {},
        )

    assert captured.requests == []


@pytest.mark.parametrize(
    ('kind', 'count', 'limit'),
    [pytest.param('images', 129, 128, id='images'), pytest.param('questions', 201, 200, id='questions')],
)
async def test_direct_decide_rejects_requests_over_provider_limits_before_encoding(
    allow_model_requests: None, kind: str, count: int, limit: int
):
    captured = Captured(boolean_answers)
    model = mock_model(captured)
    images: tuple[BinaryContent, ...] = ()
    questions: dict[str, DecisionQuestion] = {'q': NoulQuestion()}
    if kind == 'images':
        images = tuple(BinaryContent(b'image', media_type='image/png') for _ in range(count))
    else:
        questions = {f'q{index}': NoulQuestion() for index in range(count)}

    with (
        patch.object(BinaryContent, 'data_uri', new_callable=PropertyMock) as data_uri,
        pytest.raises(ModelAPIError, match=f'accepts at most {limit} {kind}; got {count}'),
    ):
        await model.decide(DecisionRequest(state='Review this.', questions=questions, images=images), {})

    data_uri.assert_not_called()
    assert captured.requests == []


@pytest.mark.parametrize(
    ('kind', 'count'),
    [pytest.param('images', 128, id='images'), pytest.param('questions', 200, id='questions')],
)
async def test_direct_decide_accepts_requests_at_provider_limits(allow_model_requests: None, kind: str, count: int):
    captured = Captured(boolean_answers)
    images: tuple[BinaryContent, ...] = ()
    questions: dict[str, DecisionQuestion] = {'q': NoulQuestion()}
    if kind == 'images':
        images = tuple(BinaryContent(b'image', media_type='image/png') for _ in range(count))
    else:
        questions = {f'q{index}': NoulQuestion() for index in range(count)}

    response = await mock_model(captured).decide(
        DecisionRequest(state='Review this.', questions=questions, images=images), {}
    )

    assert response.answers.keys() == questions.keys()
    assert len(captured.requests) == 1
