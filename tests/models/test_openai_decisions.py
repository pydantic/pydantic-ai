from __future__ import annotations as _annotations

import json
from collections.abc import Callable
from typing import Annotated, Literal
from unittest.mock import Mock

import httpx
import httpx2
import pytest
from pydantic import BaseModel, Field, WithJsonSchema

from pydantic_ai import Agent, ModelHTTPError
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior, UserError
from pydantic_ai.models.decision import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionModelSettings,
    DecisionRequest,
    NoulAnswer,
    NoulCriteria,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage

from .._inline_snapshot import snapshot
from ..conftest import RequestCapture, TestEnv, try_import

with try_import() as imports_successful:
    from openai import AsyncOpenAI

    from pydantic_ai.models.openai_decisions import OpenAIDecisionsModel
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


# This is the published preview user's recording, converted to cassetter's serialization without
# changing the request or response. It is not a recording made with our credentials.
# https://github.com/crmne/ruby_llm/pull/1008 (head fcea120022d9abbbd9904fc7dcf16f28f484851b)
@pytest.mark.vcr(additional_matchers=['json_body'])
async def test_recorded_questions(openai_api_key: str, request_capture: RequestCapture):
    model = OpenAIDecisionsModel(
        'gpt-6-luna', provider=OpenAIProvider(api_key=openai_api_key, http_client=request_capture.client)
    )
    response = await model.decide(
        DecisionRequest(
            state={'message': 'I was charged twice. Please refund the duplicate charge today.'},
            questions={
                'urgent': NoulQuestion(
                    instructions='Does the customer explicitly need action today?',
                    criteria=NoulCriteria(
                        true='Explicitly asks for action today', false='No deadline or a later deadline'
                    ),
                ),
                'department': ChoiceQuestion(
                    instructions='Which team should handle this message?',
                    criteria={'billing': 'Payments and refunds', 'technical': 'Bugs and integrations', 'other': None},
                ),
                'frustration': ScoreQuestion(
                    instructions='How frustrated is the customer?',
                    criteria=['Calm and polite', 'Expresses frustration', 'Angry or hostile'],
                ),
            },
        ),
        {},
    )
    assert response.model_name == 'gpt-6-luna'
    assert response.provider_response_id == '<X_REQUEST_ID>'
    assert response.usage == RequestUsage(input_tokens=396, output_tokens=3, details={'reasoning_tokens': 0})
    assert response.answers == snapshot(
        {
            'urgent': NoulAnswer(noul=1.0),
            'department': ChoiceAnswer(
                choice='billing', confidence=1.0, probabilities={'billing': 1.0, 'technical': 0.0, 'other': 0.0}
            ),
            'frustration': ScoreAnswer(score=1.0, confidence=1.0, probabilities={0: 0.0, 1: 1.0, 2: 0.0}),
        }
    )
    assert request_capture.body('/decisions') == snapshot(
        {
            'model': 'gpt-6-luna',
            'input': '{"message":"I was charged twice. Please refund the duplicate charge today."}',
            'questions': [
                {
                    'name': 'urgent',
                    'type': 'predicate',
                    'instructions': 'Does the customer explicitly need action today?\n'
                    'Yes: Explicitly asks for action today\nNo: No deadline or a later deadline',
                },
                {
                    'name': 'department',
                    'type': 'choice',
                    'choices': [
                        {'value': 'billing', 'description': 'Payments and refunds'},
                        {'value': 'technical', 'description': 'Bugs and integrations'},
                        {'value': 'other'},
                    ],
                    'instructions': 'Which team should handle this message?',
                },
                {
                    'name': 'frustration',
                    'type': 'score',
                    'levels': [
                        {'label': '0', 'description': 'Calm and polite'},
                        {'label': '1', 'description': 'Expresses frustration'},
                        {'label': '2', 'description': 'Angry or hostile'},
                    ],
                    'instructions': 'How frustrated is the customer?',
                },
            ],
        }
    )


class Ticket(BaseModel):
    urgent: bool = Field(description='Does this need attention today?')
    department: Literal['billing', 'technical'] = Field(description='Which team owns this?')
    frustration: Annotated[
        Literal[0, 1, 2],
        WithJsonSchema(
            {
                'type': 'integer',
                'anyOf': [
                    {'const': 0, 'description': 'Calm'},
                    {'const': 1, 'description': 'Frustrated'},
                    {'const': 2, 'description': 'Angry'},
                ],
            }
        ),
    ]


@pytest.mark.vcr
async def test_access_denied(allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture):
    """Recorded with our API key: authentication succeeds, but Decisions access is not enabled."""
    model = OpenAIDecisionsModel(
        'gpt-6-luna', provider=OpenAIProvider(api_key=openai_api_key, http_client=request_capture.client)
    )
    with pytest.raises(ModelHTTPError) as exc_info:
        await Agent(model, output_type=Ticket).run('Charged twice.')
    assert exc_info.value.status_code == 403
    assert exc_info.value.body == {
        'message': 'Decision API is not enabled for this user.',
        'type': 'invalid_request_error',
        'param': None,
        'code': None,
    }


# Synthetic responses exercise agent-generated questions and failures unavailable in the published recording.
def ticket_response() -> dict[str, object]:
    return {
        'model': 'gpt-6-luna',
        'answers': [
            {
                'name': 'frustration',
                'type': 'score',
                'score': 1.0,
                'confidence': 0.8,
                'probabilities': [
                    {'label': '2', 'value': 2, 'probability': 0.1},
                    {'label': '0', 'value': 0, 'probability': 0.1},
                    {'label': '1', 'value': 1, 'probability': 0.8},
                ],
            },
            {'name': 'urgent', 'type': 'predicate', 'probability': 0.9},
            {
                'name': 'department',
                'type': 'choice',
                'choice': 'billing',
                'confidence': 0.8,
                'probabilities': [
                    {'value': 'technical', 'probability': 0.2},
                    {'value': 'billing', 'probability': 0.8},
                ],
            },
        ],
        'usage': {
            'input_tokens': 100,
            'output_tokens': 8,
            'input_tokens_details': {'cached_tokens': 10, 'cache_write_tokens': 20},
            'output_tokens_details': {'reasoning_tokens': 5},
        },
    }


def mock_model(handler: Callable[[httpx2.Request], httpx2.Response]) -> OpenAIDecisionsModel:
    client = AsyncOpenAI(
        api_key='test-key',
        base_url='https://decisions.example.com/v1/',
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    return OpenAIDecisionsModel('gpt-6-luna', provider=OpenAIProvider(openai_client=client))


def test_init(env: TestEnv):
    env.set('OPENAI_API_KEY', 'test-key')
    model = OpenAIDecisionsModel('gpt-6-luna', settings={'timeout': 30})
    assert model.model_name == 'gpt-6-luna'
    assert model.system == 'openai'
    assert model.base_url == 'https://api.openai.com/v1/'
    assert isinstance(model.client, AsyncOpenAI)
    assert model.settings == {'timeout': 30}
    assert model.profile.get('supports_text_output') is False
    assert model.profile.get('default_structured_output_mode') == 'tool'


@pytest.mark.parametrize('stream', [False, True])
async def test_agent_ticket(allow_model_requests: None, stream: bool):
    requests: list[httpx2.Request] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json=ticket_response(), headers={'x-request-id': 'req-fixture'})

    model = mock_model(respond)
    settings = DecisionModelSettings(
        timeout=httpx.Timeout(12, connect=3),
        extra_headers={'x-custom': 'custom-value'},
        extra_body={'metadata': {'source': 'triage'}},
        temperature=0.7,
        decision_boolean_threshold=0.95,
    )
    async with Agent(model, output_type=Ticket) as agent:
        if stream:
            async with agent.run_stream('Charged twice.', model_settings=settings) as streamed:
                output = await streamed.get_output()
                usage = streamed.usage
                response = streamed.response
        else:
            result = await agent.run('Charged twice.', model_settings=settings)
            output, usage, response = result.output, result.usage, result.response
    assert output == Ticket(urgent=False, department='billing', frustration=1)
    assert (usage.input_tokens, usage.output_tokens, usage.cache_read_tokens, usage.cache_write_tokens) == (
        100,
        8,
        10,
        20,
    )
    assert usage.details == {'reasoning_tokens': 5}
    # DecisionModel can combine several decisions into one model response; IDs belong to each decision span.
    assert response.provider_response_id is None
    assert response.provider_name == 'openai'
    assert requests[0].url == 'https://decisions.example.com/v1/decisions'
    assert requests[0].headers['authorization'] == 'Bearer test-key'
    assert requests[0].headers['x-custom'] == 'custom-value'
    assert requests[0].extensions['timeout'] == {'connect': 3, 'read': 12, 'write': 12, 'pool': 12}
    assert json.loads(requests[0].content) == snapshot(
        {
            'model': 'gpt-6-luna',
            'input': 'Charged twice.',
            'questions': [
                {
                    'name': 'urgent',
                    'type': 'predicate',
                    'instructions': '{"field":"urgent","question":"Does this need attention today?"}',
                },
                {
                    'name': 'department',
                    'type': 'choice',
                    'choices': [{'value': 'billing'}, {'value': 'technical'}],
                    'instructions': '{"field":"department","question":"Which team owns this?"}',
                },
                {
                    'name': 'frustration',
                    'type': 'score',
                    'levels': [
                        {'label': '0', 'description': 'Calm'},
                        {'label': '1', 'description': 'Frustrated'},
                        {'label': '2', 'description': 'Angry'},
                    ],
                    'instructions': 'frustration',
                },
            ],
            'metadata': {'source': 'triage'},
        }
    )
    assert response.provider_details == snapshot(
        {
            'confidence': {'urgent': 0.052632, 'department': 0.8, 'frustration': 0.8},
            'probabilities': {
                'department': {'technical': 0.2, 'billing': 0.8},
                'frustration': {'2': 0.1, '0': 0.1, '1': 0.8},
            },
            'scores': {'frustration': 1.0},
        }
    )


async def test_json_descriptions_and_missing_usage():
    def respond(request: httpx2.Request) -> httpx2.Response:
        assert json.loads(request.content) == snapshot(
            {
                'model': 'gpt-6-luna',
                'input': 'null',
                'questions': [
                    {
                        'name': 'predicate',
                        'type': 'predicate',
                        'instructions': """\
{"goal":"Judge"}
Yes: {"label":"Yes"}\
""",
                    },
                    {'name': 'empty', 'type': 'predicate'},
                    {
                        'name': 'score',
                        'type': 'score',
                        'levels': [
                            {'label': '0', 'description': None},
                            {'label': '1', 'description': '{"label":"Very good"}'},
                        ],
                    },
                ],
            }
        )
        return httpx2.Response(
            200,
            json={
                'model': 'gpt-6-luna',
                'answers': [
                    {'name': 'predicate', 'type': 'predicate', 'probability': 0.5},
                    {'name': 'empty', 'type': 'predicate', 'probability': 0.5},
                    {
                        'name': 'score',
                        'type': 'score',
                        'score': 0,
                        'confidence': 1,
                        'probabilities': [{'label': '0', 'probability': 1}, {'label': '1', 'probability': 0}],
                    },
                ],
            },
        )

    response = await mock_model(respond).decide(
        DecisionRequest(
            state=None,
            questions={
                'predicate': NoulQuestion(instructions={'goal': 'Judge'}, criteria=NoulCriteria(true={'label': 'Yes'})),
                'empty': NoulQuestion(),
                'score': ScoreQuestion(criteria=[None, {'label': 'Very good'}]),
            },
        ),
        {},
    )
    assert response.usage == RequestUsage()
    assert response.provider_response_id is None


@pytest.mark.parametrize(
    'usage',
    [
        {'input_tokens': 5, 'output_tokens': 1},
        {'input_tokens': 5, 'output_tokens': 1, 'input_tokens_details': {}, 'output_tokens_details': {}},
    ],
)
async def test_optional_token_details(usage: dict[str, object]):
    model = mock_model(lambda _: httpx2.Response(200, json={'model': 'gpt-6-luna', 'answers': [], 'usage': usage}))
    response = await model.decide(DecisionRequest(state='', questions={}), {})
    assert response.usage.input_tokens == 5
    assert response.usage.output_tokens == 1
    assert response.usage.cache_read_tokens == response.usage.cache_write_tokens == 0


@pytest.mark.parametrize('status_code', [403, 429, 500])
async def test_http_errors(allow_model_requests: None, status_code: int):
    body = {'error': {'message': 'Decision API is not enabled for this user.'}}
    model = mock_model(lambda _: httpx2.Response(status_code, json=body, headers={'retry-after': '1'}))
    with pytest.raises(ModelHTTPError) as exc_info:
        await Agent(model, output_type=Ticket).run('Charged twice.')
    assert exc_info.value.status_code == status_code
    assert exc_info.value.model_name == 'gpt-6-luna'
    assert exc_info.value.body == body['error']
    assert exc_info.value.headers is not None and exc_info.value.headers['retry-after'] == '1'


async def test_transport_error_and_fallback(allow_model_requests: None):
    def fail(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError('offline', request=request)

    model = mock_model(fail)
    with pytest.raises(ModelAPIError, match='Connection error'):
        await Agent(model, output_type=Ticket).run('Charged twice.')
    result = await Agent(FallbackModel(model, TestModel()), output_type=Ticket).run('Charged twice.')
    assert isinstance(result.output, Ticket)


@pytest.mark.parametrize('extra_body', [['not', 'a', 'mapping'], {'bad': object()}])
async def test_invalid_request_body(extra_body: object):
    handler = Mock(side_effect=AssertionError('Invalid JSON must not be sent'))
    model = mock_model(handler)
    with pytest.raises(UserError, match='Could not send this request'):
        await model.decide(DecisionRequest(state='x', questions={}), {'extra_body': extra_body})
    handler.assert_not_called()


MALFORMED_RESPONSES: list[bytes | dict[str, object]] = [
    b'not json',
    {'answers': []},
    {'model': 'gpt-6-luna', 'answers': []},
    {'model': 'gpt-6-luna', 'answers': [{'name': 'other', 'type': 'predicate', 'probability': 1}]},
    {'model': 'gpt-6-luna', 'answers': [{'name': 'urgent', 'type': 'predicate'}]},
    {'model': 'gpt-6-luna', 'answers': [{'name': 'urgent', 'type': 'predicate', 'probability': 2}]},
    {'model': 'gpt-6-luna', 'answers': [{'name': 'urgent', 'type': 'predicate', 'probability': float('nan')}]},
    {'model': 'gpt-6-luna', 'answers': [{'name': 'urgent', 'type': 'unexpected', 'probability': 1}]},
    {
        'model': 'gpt-6-luna',
        'answers': [{'name': 'urgent', 'type': 'predicate', 'probability': 1}] * 2,
    },
    {
        'model': 'gpt-6-luna',
        'answers': [{'name': 'urgent', 'type': 'choice', 'choice': 'yes', 'confidence': 1, 'probabilities': []}],
    },
    {
        'model': 'gpt-6-luna',
        'answers': [{'name': 'urgent', 'type': 'predicate', 'probability': 1}],
        'usage': {'input_tokens': -1, 'output_tokens': 0},
    },
]


@pytest.mark.parametrize('body', MALFORMED_RESPONSES)
async def test_malformed_response(body: bytes | dict[str, object]):
    content = body if isinstance(body, bytes) else json.dumps(body).encode()
    model = mock_model(lambda _: httpx2.Response(200, content=content, headers={'content-type': 'application/json'}))
    with pytest.raises(UnexpectedModelBehavior, match='Invalid response from the OpenAI Decisions API'):
        await model.decide(DecisionRequest(state='x', questions={'urgent': NoulQuestion()}), {})


@pytest.mark.parametrize(
    'answer',
    [
        {'type': 'choice', 'choice': 'a', 'confidence': 1, 'probabilities': [{'value': 'a', 'probability': 1}]},
        {
            'type': 'choice',
            'choice': 'c',
            'confidence': 1,
            'probabilities': [{'value': 'a', 'probability': 1}, {'value': 'b', 'probability': 0}],
        },
        {
            'type': 'choice',
            'choice': 'a',
            'confidence': 1,
            'probabilities': [
                {'value': 'a', 'probability': 1},
                {'value': 'a', 'probability': 0},
                {'value': 'b', 'probability': 0},
            ],
        },
        {'type': 'score', 'score': 0, 'confidence': 1, 'probabilities': [{'label': '0', 'probability': 1}]},
        {
            'type': 'score',
            'score': 0,
            'confidence': 1,
            'probabilities': [{'label': '0', 'probability': 1}, {'label': 'wrong', 'probability': 0}],
        },
        {
            'type': 'score',
            'score': 2,
            'confidence': 1,
            'probabilities': [{'label': '0', 'probability': 1}, {'label': '1', 'probability': 0}],
        },
        {
            'type': 'choice',
            'choice': 'a',
            'confidence': 0.5,
            'probabilities': [{'value': 'a', 'probability': 0.5}, {'value': 'b', 'probability': 0}],
        },
        {
            'type': 'score',
            'score': 0,
            'confidence': 1,
            'probabilities': [{'label': '0', 'probability': 0}, {'label': '1', 'probability': 1}],
        },
    ],
)
async def test_invalid_distribution(answer: dict[str, object]):
    question = (
        ChoiceQuestion(criteria={'a': None, 'b': None})
        if answer['type'] == 'choice'
        else ScoreQuestion(criteria=['a', 'b'])
    )
    model = mock_model(
        lambda _: httpx2.Response(200, json={'model': 'gpt-6-luna', 'answers': [{'name': 'value', **answer}]})
    )
    with pytest.raises(UnexpectedModelBehavior, match=r'invalid .* probabilities'):
        await model.decide(DecisionRequest(state='x', questions={'value': question}), {})


async def test_tool_routing(allow_model_requests: None):
    class Status(BaseModel):
        """Report whether the customer's payment was refunded."""

        refunded: bool = Field(description='Was the payment refunded?')

    calls: list[str] = []

    def look_up_order() -> str:
        """Look up the customer's payment."""
        calls.append('looked up')
        return 'Payment refunded yesterday.'

    responses = iter(
        [
            {
                'model': 'gpt-6-luna',
                'answers': [
                    {'name': 'Status.refunded', 'type': 'predicate', 'probability': 0.5},
                    {
                        'name': 'route',
                        'type': 'choice',
                        'choice': 'look_up_order',
                        'confidence': 0.9,
                        'probabilities': [
                            {'value': 'Status', 'probability': 0.1},
                            {'value': 'look_up_order', 'probability': 0.9},
                        ],
                    },
                ],
            },
            {
                'model': 'gpt-6-luna',
                'answers': [{'name': 'refunded', 'type': 'predicate', 'probability': 1}],
            },
        ]
    )
    requests: list[httpx2.Request] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json=next(responses))

    result = await Agent(mock_model(respond), output_type=Status, tools=[look_up_order]).run('Was I refunded?')
    assert result.output == Status(refunded=True)
    assert calls == ['looked up']
    assert len(requests) == 2
    assert 'Payment refunded yesterday.' in json.loads(requests[1].content)['input']
