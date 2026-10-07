"""Tests for `OpenAIDecisionsModel`.

Tests marked `vcr` run against recordings of the live API, and assert what the code sends with `request_capture`,
which sees the request on replay too. The rest send no request, or mock the transport and say why it can't be a
recording.
"""

from __future__ import annotations as _annotations

import json
from collections.abc import Mapping
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal

import httpx2
import pytest
from pydantic import BaseModel, Field, WithJsonSchema

from pydantic_ai import (
    Agent,
    BoolCriteria,
    ModelHTTPError,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.exceptions import ContentFilterError, UnexpectedModelBehavior, UserError
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
)
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

from .._inline_snapshot import snapshot
from ..conftest import IsDatetime, IsStr, RequestCapture, TestEnv, try_import
from .test_system_one import Captured, Frustration, Handler, Ticket

with try_import() as imports_successful:
    from openai import AsyncOpenAI

    from pydantic_ai.models.openai_decisions import OpenAIDecisionsModel, OpenAIDecisionsModelSettings
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openai_decisions import OpenAIDecisionsProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


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
AREA = {
    'type': 'choice',
    'name': 'area',
    'choice': 'billing',
    'probabilities': [{'value': 'billing', 'probability': 0.94}, {'value': 'bug', 'probability': 0.06}],
    'confidence': 0.88,
}
REFUND = {'type': 'predicate', 'name': 'refund', 'probability': 0.2}
FRUSTRATION = {
    'type': 'score',
    'name': 'frustration',
    'score': 1.7,
    'probabilities': [
        {'value': 0, 'label': '0', 'probability': 0.05},
        {'value': 1, 'label': '1', 'probability': 0.2},
        {'value': 2, 'label': '2', 'probability': 0.75},
    ],
    'confidence': 0.55,
}


def ticket_answers(request: httpx2.Request) -> httpx2.Response:
    return decisions(URGENT, AREA)


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

    assert result.output == Ticket(urgent=True, area='billing')
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


class Profile(BaseModel):
    """Triage a support ticket."""

    urgent: bool = Field(description='Does this need a reply within the hour?')
    disabled: bool = Field(description='Does the customer have a disability?')


@pytest.mark.vcr
async def test_refusal(allow_model_requests: None, capture_model: OpenAIDecisionsModel):
    """The model declines to infer a sensitive trait, and answers the other questions of the request."""
    agent = Agent(capture_model, output_type=Profile)

    with pytest.raises(ContentFilterError, match="declined to answer: 'disabled'") as exc_info:
        await agent.run('My invoice was charged twice and nobody answers the phone!')
    assert json.loads(exc_info.value.body or '')['answers'] == snapshot(
        [{'type': 'predicate', 'name': 'urgent', 'probability': 0.53}, {'type': 'refusal', 'name': 'disabled'}]
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
    """The one route left is filled in a request of its own, and a refusal there fails naming the route."""
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
    ('questions', 'limit'),
    [
        pytest.param({'q': ChoiceQuestion(criteria={str(option): None for option in range(256)})}, 255, id='options'),
        pytest.param({'q': ScoreQuestion(criteria=[None] * 11)}, 10, id='levels'),
        pytest.param(
            {f'q{index}': NoulQuestion(instructions='Is this urgent?') for index in range(201)}, 200, id='questions'
        ),
    ],
)
async def test_api_limits(
    questions: dict[str, DecisionQuestion], limit: int, allow_model_requests: None, capture_model: OpenAIDecisionsModel
):
    """The limits `max_choice_options` and `max_score_levels` keep to, and the question count, which the API checks.

    `decide` is called directly, since an agent run refuses the first two before sending.
    """
    with pytest.raises(ModelHTTPError, match=f'maximum length {limit},') as exc_info:
        await capture_model.decide(DecisionRequest(state='Charged twice.', questions=questions), {})
    assert exc_info.value.status_code == 400


@pytest.mark.vcr
async def test_http_error(allow_model_requests: None, openai_api_key: str):
    """A model the API does not serve is an error response, raised for a `FallbackModel` to take over."""
    model = OpenAIDecisionsModel('gpt-5', provider=OpenAIDecisionsProvider(api_key=openai_api_key))

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


async def test_limits(allow_model_requests: None):
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
            NoulQuestion(instructions='Is this urgent?', criteria=NoulCriteria(true='Today.')),
            {'type': 'predicate', 'name': 'q', 'instructions': '{"question": "Is this urgent?", "yes": "Today."}'},
            id='text and yes',
        ),
        pytest.param(
            NoulQuestion(criteria=NoulCriteria(false='Not today.')),
            {'type': 'predicate', 'name': 'q', 'instructions': '{"no": "Not today."}'},
            id='only no',
        ),
        pytest.param(NoulQuestion(), {'type': 'predicate', 'name': 'q', 'instructions': ''}, id='nothing'),
        pytest.param(
            ChoiceQuestion(criteria={'billing': None, 'bug': 'Something is broken.'}),
            {
                'type': 'choice',
                'name': 'q',
                'instructions': '',
                'choices': [{'value': 'billing'}, {'value': 'bug', 'description': 'Something is broken.'}],
            },
            id='choice',
        ),
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
    """Each shape `decide` can send a question in, with `instructions`, which the API requires, empty where unset.

    The base class only sends some of these from an agent run, so `decide` is called directly. Not recorded: the
    answer doesn't matter here, and asking about nothing gets a refusal.
    """
    captured = Captured(lambda request: httpx2.Response(400))
    with pytest.raises(ModelHTTPError):
        await mock_model(captured).decide(DecisionRequest(state='Down since 9am.', questions={'q': question}), {})

    assert captured.body['questions'] == [sent]


async def test_request_id_and_cached_tokens(allow_model_requests: None):
    """`decide` is the only place the request ID reaches: the run's response is built from the answers.

    Not recorded: the cassettes strip the `x-request-id` header it comes from, and none read cached tokens.
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


async def test_settings_are_forwarded(allow_model_requests: None):
    """Not recorded: the timeout is set on the request's `extensions`, which neither a cassette nor a capture keeps."""
    captured = Captured(ticket_answers)
    agent = Agent(mock_model(captured), output_type=Ticket)
    settings: OpenAIDecisionsModelSettings = {
        'timeout': 3,
        'extra_headers': {'X-Team': 'support'},
        'extra_body': {'safety_identifier': 'user_123'},
    }

    await agent.run('Charged twice.', model_settings=settings)

    request = captured.requests[0]
    assert request.headers['x-team'] == 'support'
    assert request.extensions['timeout'] == {'connect': 3, 'read': 3, 'write': 3, 'pool': 3}
    assert captured.body['safety_identifier'] == 'user_123'


async def test_extra_body_must_be_a_mapping(allow_model_requests: None):
    """Not recorded: refused before a request is sent."""
    captured = Captured(ticket_answers)
    agent = Agent(mock_model(captured), output_type=Ticket)
    with pytest.raises(UserError, match='`extra_body` must be a mapping'):
        await agent.run('Charged twice.', model_settings={'extra_body': ['not', 'a', 'mapping']})
    assert captured.requests == []


@pytest.mark.parametrize(
    'response',
    [
        pytest.param(httpx2.Response(200, text='not json'), id='not json'),
        pytest.param(decisions({**URGENT, 'probability': 1.2}, AREA), id='probability out of range'),
        pytest.param(decisions(URGENT, {**AREA, 'confidence': None}), id='no confidence'),
        pytest.param(decisions({**URGENT, 'type': 'noul'}, AREA), id='unknown type'),
        pytest.param(decisions({**URGENT, 'name': None}, AREA), id='unnamed'),
        pytest.param(decisions(URGENT, {**AREA, 'choice': True}), id='boolean choice'),
    ],
)
async def test_invalid_response(response: httpx2.Response, allow_model_requests: None):
    """Not recorded: no live model answers like this."""
    agent = Agent(mock_model(lambda request: response), output_type=Ticket)
    with pytest.raises(UnexpectedModelBehavior, match='Invalid response from the OpenAI Decisions API'):
        await agent.run('Charged twice.')


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
        pytest.param(Ticket, (URGENT, {**AREA, 'choice': 'other'}), id='option not offered'),
        pytest.param(
            Ticket,
            (URGENT, {**AREA, 'probabilities': [{'value': 'billing', 'probability': 1.0}]}),
            id='option left out',
        ),
        pytest.param(Mood, (REFUND, {**FRUSTRATION, 'score': 2.5}), id='score past the rubric'),
        pytest.param(Mood, (REFUND, {**FRUSTRATION, 'score': float('nan')}), id='score not a number'),
        pytest.param(Mood, (REFUND, {**FRUSTRATION, 'probabilities': []}), id='no levels'),
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
