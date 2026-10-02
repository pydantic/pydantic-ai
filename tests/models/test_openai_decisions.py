"""Tests for `OpenAIDecisionsModel`.

Not VCR tests: the Decisions API is in an invite-only preview, so there is no access to record cassettes with. The
transport is mocked instead, answering in the shape a preview user recorded live in
https://github.com/crmne/ruby_llm/pull/1008, and the tests go through the real `openai` client, as a user's run does.
"""

# TODO: Once the API is open, record cassettes with `pytestmark = pytest.mark.vcr`, move these tests onto them, and
# assert the outgoing body with the `request_capture` fixture, keeping the mock only for answers no live model gives.

from __future__ import annotations as _annotations

import json
from collections.abc import Mapping
from decimal import Decimal
from enum import Enum
from typing import Annotated

import httpx2
import pytest
from pydantic import BaseModel, Field

from pydantic_ai import (
    Agent,
    BoolCriteria,
    ModelHTTPError,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.exceptions import UnexpectedModelBehavior, UserError
from pydantic_ai.models import infer_model
from pydantic_ai.models.decision import DecisionRequest, NoulCriteria, NoulQuestion
from pydantic_ai.usage import RequestUsage

from .._inline_snapshot import snapshot
from ..conftest import IsStr, TestEnv, try_import
from .test_system_one import Captured, Frustration, Handler, Ticket

with try_import() as imports_successful:
    from openai import AsyncOpenAI

    from pydantic_ai.models.openai_decisions import OpenAIDecisionsModel, OpenAIDecisionsModelSettings
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openai_decisions import OpenAIDecisionsProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


class Mood(BaseModel):
    """Read the customer's mood."""

    refund: Annotated[bool, BoolCriteria(true='They ask for their money back.', false='They do not.')] = Field(
        description='Do they want a refund?'
    )
    frustration: Frustration = Field(description='How frustrated is the customer?')


def mock_model(handler: Handler) -> OpenAIDecisionsModel:
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    client = AsyncOpenAI(api_key='test', max_retries=0, http_client=http_client)
    return OpenAIDecisionsModel('gpt-6-luna', provider=OpenAIDecisionsProvider(openai_client=client))


def decisions(*answers: Mapping[str, object]) -> httpx2.Response:
    """A `/v1/decisions` response, in the shape the API was recorded answering in.

    Encoded with `json.dumps`, which writes `NaN` as a server written in Python can, where `json=` refuses to.
    """
    usage = {
        'input_tokens': 396,
        'input_tokens_details': {'cached_tokens': 128, 'cache_write_tokens': 0},
        'output_tokens': len(answers),
        'output_tokens_details': {'reasoning_tokens': 0},
        'total_tokens': 396 + len(answers),
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
    assert model.system == 'openai'
    assert model.base_url == 'https://api.openai.com/v1/'
    assert isinstance(model.client, AsyncOpenAI)
    # The ID round-trips, where `openai:gpt-6-luna` would be a Responses API model.
    assert model.model_id == 'openai-decisions:gpt-6-luna'
    assert isinstance(infer_model(model.model_id), OpenAIDecisionsModel)


def test_infer_model_refuses_another_provider():
    with pytest.raises(UserError, match='require an `OpenAIDecisionsProvider`'):
        infer_model('openai-decisions:gpt-6-luna', provider_factory=lambda _: OpenAIProvider(api_key='test'))


class Team(str, Enum):
    billing = 'billing'
    bug = 'bug'


class Customer(BaseModel):
    vip: bool = Field(description='Are they on an enterprise plan?')


class Assignment(BaseModel):
    """Assign a support ticket."""

    team: Team = Field(description='Which team owns it?')
    customer: Customer = Field(description='Who is writing in.')


async def test_enum_and_nested_fields(allow_model_requests: None):
    """An `Enum` or nested model field with a description is asked like on any other decision model.

    The schema puts a `$ref` beside the description, which the Responses API's profile for the same model ID would
    rewrite into a shape no question is built from, so the provider gives the decision model profile instead.
    """
    captured = Captured(
        lambda request: decisions(
            {**AREA, 'name': 'team'}, {'type': 'predicate', 'name': 'customer.vip', 'probability': 0.1}
        )
    )

    result = await Agent(mock_model(captured), output_type=Assignment).run('Charged twice.')

    assert result.output == Assignment(team=Team.billing, customer=Customer(vip=False))


async def test_output_type(allow_model_requests: None):
    captured = Captured(ticket_answers)
    agent = Agent(mock_model(captured), output_type=Ticket)

    result = await agent.run('My invoice was charged twice and nobody answers the phone!')

    assert result.output == Ticket(urgent=True, area='billing')
    assert result.response.parts == [ToolCallPart('final_result', result.output.model_dump(), tool_call_id=IsStr())]
    assert result.response.model_name == 'gpt-6-luna'
    assert result.response.provider_name == 'openai'
    assert result.response.usage == snapshot(
        RequestUsage(
            input_tokens=396,
            cache_read_tokens=128,
            output_reasoning_tokens=0,
            output_tokens=2,
            details={},
            cost=Decimal('0.00002908'),
        )
    )
    assert result.response.provider_details == snapshot(
        {
            'confidence': {'urgent': 0.82, 'area': 0.88},
            'probabilities': {'area': {'billing': 0.94, 'bug': 0.06}},
            'scores': {},
        }
    )
    request = captured.requests[0]
    assert str(request.url) == 'https://api.openai.com/v1/decisions'
    assert request.headers['authorization'] == 'Bearer test'
    assert captured.body == snapshot(
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
                    'choices': [{'value': 'billing'}, {'value': 'bug'}],
                    'instructions': '{"field": "area", "question": "Which team owns it?", "goal": "Triage a support ticket."}',
                },
            ],
        }
    )


async def test_yes_no_meanings_and_rubric(allow_model_requests: None):
    """A predicate has no field for what yes and no mean, so they go into its instructions; a rubric is `levels`."""
    captured = Captured(lambda request: decisions(REFUND, FRUSTRATION))
    result = await Agent(mock_model(captured), output_type=Mood).run('This is the third time I am asking. Fix it NOW.')

    assert result.output == Mood(refund=False, frustration=2)
    assert result.response.provider_details == snapshot(
        {
            'confidence': {'refund': 0.6, 'frustration': 0.55},
            'probabilities': {'frustration': {'0': 0.05, '1': 0.2, '2': 0.75}},
            'scores': {'frustration': 1.7},
        }
    )
    assert captured.body['questions'] == snapshot(
        [
            {
                'type': 'predicate',
                'name': 'refund',
                'instructions': '{"field": "refund", "question": "Do they want a refund?", "goal": "Read the customer\'s mood.", "yes": "They ask for their money back.", "no": "They do not."}',
            },
            {
                'type': 'score',
                'name': 'frustration',
                'levels': [
                    {'label': '0', 'description': 'Calm'},
                    {'label': '1', 'description': 'Frustrated'},
                    {'label': '2', 'description': 'Very angry'},
                ],
                'instructions': '{"field": "frustration", "question": "How frustrated is the customer?", "goal": "Read the customer\'s mood."}',
            },
        ]
    )


async def test_conversation_and_route(allow_model_requests: None):
    """A conversation is JSON, sent as the `input` text, and the route between a tool and the output is a `choice`."""

    def escalate() -> None:
        """Hand the ticket to a human."""

    route = {
        'type': 'choice',
        'name': 'route',
        'choice': 'Ticket',
        'probabilities': [{'value': 'Ticket', 'probability': 0.97}, {'value': 'escalate', 'probability': 0.03}],
        'confidence': 0.97,
    }
    captured = Captured(
        lambda request: decisions({**URGENT, 'name': 'Ticket.urgent'}, {**AREA, 'name': 'Ticket.area'}, route)
    )
    agent = Agent(mock_model(captured), output_type=Ticket, tools=[escalate])
    history: list[ModelMessage] = [
        ModelRequest.user_text_prompt('I was charged twice.'),
        ModelResponse(parts=[TextPart('Sorry to hear that, we are looking into it.')]),
    ]

    result = await agent.run('Still no refund!', message_history=history)

    assert result.output == Ticket(urgent=True, area='billing')
    assert captured.body == snapshot(
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
                    'choices': [{'value': 'billing'}, {'value': 'bug'}],
                    'instructions': '{"field": "area", "premise": "If the user\'s request calls for Ticket: Triage a support ticket.", "question": "Which team owns it?"}',
                },
                {
                    'type': 'choice',
                    'name': 'route',
                    'choices': [
                        {'value': 'Ticket', 'description': 'Triage a support ticket.'},
                        {'value': 'escalate', 'description': 'Hand the ticket to a human.'},
                    ],
                    'instructions': 'Which of these does this call for?',
                },
            ],
        }
    )


@pytest.mark.parametrize(
    ('question', 'sent'),
    [
        pytest.param(
            NoulQuestion(instructions='Is this urgent?'),
            {'type': 'predicate', 'name': 'urgent', 'instructions': 'Is this urgent?'},
            id='text',
        ),
        pytest.param(
            NoulQuestion(instructions='Is this urgent?', criteria=NoulCriteria(true='Today.')),
            {'type': 'predicate', 'name': 'urgent', 'instructions': '{"question": "Is this urgent?", "yes": "Today."}'},
            id='text and yes',
        ),
        pytest.param(
            NoulQuestion(criteria=NoulCriteria(false='Not today.')),
            {'type': 'predicate', 'name': 'urgent', 'instructions': '{"no": "Not today."}'},
            id='only no',
        ),
        pytest.param(NoulQuestion(), {'type': 'predicate', 'name': 'urgent'}, id='nothing'),
    ],
)
async def test_decide(question: NoulQuestion, sent: dict[str, str], allow_model_requests: None):
    """`decide` is public, and is the only place the request ID reaches: the run's response is built from the answers.

    The base class only sends a yes/no with plain-text instructions, or none, from a bare output, so the predicate is
    pinned here for each shape it can take.
    """
    captured = Captured(lambda request: decisions({'type': 'predicate', 'name': 'urgent', 'probability': 1.0}))
    response = await mock_model(captured).decide(
        DecisionRequest(state='Down since 9am.', questions={'urgent': question}), {}
    )

    assert response.provider_response_id == 'req_123'
    assert response.model_name == 'gpt-6-luna'
    assert captured.body['questions'] == [sent]


async def test_settings_are_forwarded(allow_model_requests: None):
    captured = Captured(ticket_answers)
    agent = Agent(mock_model(captured), output_type=Ticket)
    settings: OpenAIDecisionsModelSettings = {
        'timeout': 3,
        'extra_headers': {'X-Team': 'support'},
        'extra_body': {'trace': True},
    }

    await agent.run('Charged twice.', model_settings=settings)

    request = captured.requests[0]
    assert request.headers['x-team'] == 'support'
    assert request.extensions['timeout'] == {'connect': 3, 'read': 3, 'write': 3, 'pool': 3}
    assert captured.body['trace'] is True


async def test_extra_body_must_be_a_mapping(allow_model_requests: None):
    captured = Captured(ticket_answers)
    agent = Agent(mock_model(captured), output_type=Ticket)
    with pytest.raises(UserError, match='`extra_body` must be a mapping'):
        await agent.run('Charged twice.', model_settings={'extra_body': ['not', 'a', 'mapping']})
    assert captured.requests == []


async def test_http_error(allow_model_requests: None):
    """The API answers anyone outside the preview with this 403."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        error = {'message': 'Decision API is not enabled for this user.', 'type': 'invalid_request_error'}
        return httpx2.Response(403, json={'error': error})

    agent = Agent(mock_model(handler), output_type=Ticket)
    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('Charged twice.')
    assert exc_info.value.status_code == 403
    assert exc_info.value.model_name == 'gpt-6-luna'
    assert exc_info.value.body == snapshot(
        {'message': 'Decision API is not enabled for this user.', 'type': 'invalid_request_error'}
    )


@pytest.mark.parametrize(
    'response',
    [
        pytest.param(httpx2.Response(200, text='not json'), id='not json'),
        pytest.param(decisions({**URGENT, 'probability': 1.2}, AREA), id='probability out of range'),
        pytest.param(decisions(URGENT, {**AREA, 'confidence': None}), id='no confidence'),
        pytest.param(decisions({**URGENT, 'type': 'noul'}, AREA), id='unknown type'),
    ],
)
async def test_invalid_response(response: httpx2.Response, allow_model_requests: None):
    agent = Agent(mock_model(lambda request: response), output_type=Ticket)
    with pytest.raises(UnexpectedModelBehavior, match='Invalid response from the OpenAI Decisions API'):
        await agent.run('Charged twice.')


@pytest.mark.parametrize(
    'answers',
    [
        pytest.param((URGENT,), id='missing'),
        pytest.param((URGENT, AREA, {**URGENT, 'name': 'extra'}), id='extra'),
        pytest.param((URGENT, {**URGENT, 'probability': 0.1}, AREA), id='twice'),
    ],
)
async def test_answer_names_match_questions(answers: tuple[Mapping[str, object], ...], allow_model_requests: None):
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
    """An answer its question does not allow fails the request, rather than reaching the output or a retry."""
    agent = Agent(mock_model(lambda request: decisions(*answers)), output_type=output_type)
    with pytest.raises(UnexpectedModelBehavior, match='does not match its question'):
        await agent.run('Charged twice.')
