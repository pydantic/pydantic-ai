from __future__ import annotations as _annotations

import json
from collections.abc import Callable, Mapping
from typing import Annotated, Literal

import httpx2
import pytest
from pydantic import BaseModel, Field, WithJsonSchema

from pydantic_ai import Agent, ModelHTTPError, ToolCallPart
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior, UserError
from pydantic_ai.models import infer_model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.system_one import SystemOneModel, SystemOneModelSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles.decision import decision_model_profile
from pydantic_ai.profiles.typesafe import typesafe_model_profile
from pydantic_ai.providers import Provider, infer_provider
from pydantic_ai.providers.system_one import SystemOneProvider
from pydantic_ai.usage import RequestUsage

from .._inline_snapshot import snapshot
from ..conftest import IsStr, TestEnv


class Ticket(BaseModel):
    """Triage a support ticket."""

    urgent: bool = Field(description='Does this need a reply within the hour?')
    area: Literal['billing', 'bug'] = Field(description='Which team owns it?')


Frustration = Annotated[
    Literal[0, 1, 2],
    WithJsonSchema(
        {
            'type': 'integer',
            'anyOf': [
                {'const': 0, 'description': 'Calm'},
                {'const': 1, 'description': 'Frustrated'},
                {'const': 2, 'description': 'Very angry'},
            ],
        }
    ),
]


class Mood(BaseModel):
    """Read the customer's mood."""

    frustration: Frustration = Field(description='How frustrated is the customer?')


Handler = Callable[[httpx2.Request], httpx2.Response]

BASE_URL = 'http://localhost:8700'


def mock_model(handler: Handler, *, api_key: str | None = None) -> SystemOneModel:
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return SystemOneModel(
        'clm-latest', provider=SystemOneProvider(base_url=BASE_URL, api_key=api_key, http_client=http_client)
    )


def answers(**answers: Mapping[str, object]) -> httpx2.Response:
    """A `/v1/systemone` body as `clm-serve` sends it."""
    usage = {'billing_units': len(answers), 'input_tokens': 42, 'output_tokens': 0}
    return httpx2.Response(200, json={'model': 'clm-latest', 'answers': answers, 'usage': usage})


def ticket_answers(request: httpx2.Request) -> httpx2.Response:
    return answers(
        urgent={'type': 'noul', 'noul': 0.91},
        area={
            'type': 'choice',
            'choice': 'billing',
            'confidence': 0.88,
            'probabilities': {'billing': 0.94, 'bug': 0.06},
        },
    )


class Captured:
    def __init__(self, response: Handler):
        self.requests: list[httpx2.Request] = []
        self._response = response

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self._response(request)

    @property
    def body(self) -> dict[str, object]:
        return json.loads(self.requests[-1].content)


def test_init(env: TestEnv):
    env.set('SYSTEM_ONE_BASE_URL', 'http://gpu-box:8700/')
    env.remove('SYSTEM_ONE_API_KEY')
    model = SystemOneModel('clm-latest')
    assert model.model_name == 'clm-latest'
    assert model.system == 'system-one'
    assert model.base_url == 'http://gpu-box:8700'
    assert isinstance(model.client, httpx2.AsyncClient)
    assert model._api_key is None  # pyright: ignore[reportPrivateUsage]


def test_provider_reads_the_environment(env: TestEnv):
    env.set('SYSTEM_ONE_BASE_URL', 'http://gpu-box:8700/')
    env.set('SYSTEM_ONE_API_KEY', 'secret')
    provider = SystemOneProvider()
    assert provider.base_url == 'http://gpu-box:8700'
    assert provider.api_key == 'secret'
    assert provider.name == 'system-one'


def test_provider_needs_a_base_url(env: TestEnv):
    env.remove('SYSTEM_ONE_BASE_URL')
    with pytest.raises(UserError, match='SYSTEM_ONE_BASE_URL'):
        SystemOneProvider()


def test_infer_model(env: TestEnv):
    env.set('SYSTEM_ONE_BASE_URL', BASE_URL)
    model = infer_model('system-one:clm-latest')
    assert isinstance(model, SystemOneModel)
    assert model.model_name == 'clm-latest'
    assert isinstance(infer_provider('system-one'), SystemOneProvider)


class OtherProvider(Provider[httpx2.AsyncClient]):
    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def base_url(self) -> str:
        raise NotImplementedError

    @property
    def client(self) -> httpx2.AsyncClient:
        raise NotImplementedError


def test_infer_model_refuses_another_provider():
    with pytest.raises(UserError, match='require a `SystemOneProvider`'):
        infer_model('system-one:clm-latest', provider_factory=lambda _: OtherProvider())


def test_profile():
    model = SystemOneModel(
        'clm-latest', provider=SystemOneProvider(base_url=BASE_URL, http_client=httpx2.AsyncClient())
    )
    assert model.profile.get('supports_text_output') is False
    assert model.profile.get('supports_json_schema_output') is False
    assert model.profile.get('default_structured_output_mode') == 'tool'
    # TypeSafe's profile is the shared decision model profile, unchanged.
    assert typesafe_model_profile('jev-latest') == decision_model_profile('jev-latest')


async def test_output_type(allow_model_requests: None):
    captured = Captured(ticket_answers)
    model = mock_model(captured)
    agent = Agent(model, output_type=Ticket)

    result = await agent.run('My invoice was charged twice and nobody answers the phone!')

    assert result.output == Ticket(urgent=True, area='billing')
    assert result.response.parts == [ToolCallPart('final_result', result.output.model_dump(), tool_call_id=IsStr())]
    assert result.response.model_name == 'clm-latest'
    assert result.response.usage == RequestUsage(input_tokens=42)
    assert result.response.provider_name == 'system-one'
    assert result.response.provider_details == snapshot(
        {
            'confidence': {'urgent': 0.82, 'area': 0.88},
            'probabilities': {'area': {'billing': 0.94, 'bug': 0.06}},
            'scores': {},
        }
    )
    request = captured.requests[0]
    assert str(request.url) == 'http://localhost:8700/v1/systemone'
    assert 'authorization' not in request.headers
    assert captured.body == snapshot(
        {
            'state': 'My invoice was charged twice and nobody answers the phone!',
            'model': 'clm-latest',
            'questions': {
                'urgent': {
                    'type': 'noul',
                    'instructions': {
                        'field': 'urgent',
                        'question': 'Does this need a reply within the hour?',
                        'goal': 'Triage a support ticket.',
                    },
                },
                'area': {
                    'type': 'choice',
                    'criteria': {'billing': None, 'bug': None},
                    'instructions': {
                        'field': 'area',
                        'question': 'Which team owns it?',
                        'goal': 'Triage a support ticket.',
                    },
                },
            },
        }
    )


async def test_score_levels_come_back_as_numbers(allow_model_requests: None):
    """A server keys a rubric's probabilities and legend by level as strings, as JSON has to."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        return answers(
            frustration={
                'type': 'score',
                'score': 1.7,
                'confidence': 0.55,
                'probabilities': {'0': 0.05, '1': 0.2, '2': 0.75},
                'legend': {'0': 'Calm', '1': 'Frustrated', '2': 'Very angry'},
            }
        )

    agent = Agent(mock_model(handler), output_type=Mood)
    result = await agent.run('This is the third time I am asking. Fix it NOW.')
    assert result.output == Mood(frustration=2)
    assert result.response.provider_details == snapshot(
        {
            'confidence': {'frustration': 0.55},
            'probabilities': {'frustration': {'0': 0.05, '1': 0.2, '2': 0.75}},
            'scores': {'frustration': 1.7},
        }
    )


async def test_laya_response(allow_model_requests: None):
    """`laya-serve` adds keys of its own to the body and each answer, and leaves `billing_units` out of the usage."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        action = {'act_probability': 1.0}
        return httpx2.Response(
            200,
            json={
                'model': 'laya-rl-agent',
                'answers': {
                    'urgent': {
                        'type': 'noul',
                        'noul': 0.91,
                        'confidence': 0.91,
                        'answer_confidence': 0.91,
                        'action': action,
                    },
                    'area': {
                        'type': 'choice',
                        'choice': 'billing',
                        'probabilities': {'billing': 0.93, 'bug': 0.07},
                        'confidence': 0.63,
                        'answer_confidence': 0.93,
                        'action': action,
                    },
                },
                'usage': {'input_tokens': 74, 'output_tokens': 0},
                'routing': {'model': 'english', 'repo': 'convaiinnovations/laya', 'reason': 'English Latin text'},
            },
        )

    agent = Agent(mock_model(handler), output_type=Ticket)
    result = await agent.run('I was charged twice this month.')
    assert result.output == Ticket(urgent=True, area='billing')
    assert result.response.model_name == 'laya-rl-agent'
    assert result.usage.input_tokens == 74


async def test_settings_are_forwarded(allow_model_requests: None):
    captured = Captured(ticket_answers)
    agent = Agent(mock_model(captured, api_key='secret'), output_type=Ticket)
    settings: SystemOneModelSettings = {
        'temperature': 0.5,
        'timeout': 3,
        'extra_headers': {'X-Team': 'support'},
        'extra_body': {'trace': True},
    }

    await agent.run('Charged twice.', model_settings=settings)

    request = captured.requests[0]
    assert request.headers['authorization'] == 'Bearer secret'
    assert request.headers['x-team'] == 'support'
    assert request.extensions['timeout'] == {'connect': 3, 'read': 3, 'write': 3, 'pool': 3}
    assert captured.body['temperature'] == 0.5
    assert captured.body['trace'] is True


async def test_extra_body_must_be_a_mapping(allow_model_requests: None):
    agent = Agent(mock_model(ticket_answers), output_type=Ticket)
    with pytest.raises(UserError, match='`extra_body` must be a mapping'):
        await agent.run('Charged twice.', model_settings={'extra_body': ['not', 'a', 'mapping']})


async def test_extra_body_that_will_not_encode(allow_model_requests: None):
    agent = Agent(mock_model(ticket_answers), output_type=Ticket)
    with pytest.raises(UserError, match='Could not send this request to the System One server'):
        await agent.run('Charged twice.', model_settings={'extra_body': {'when': object()}})


async def test_http_error(allow_model_requests: None):
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(422, json={'detail': "unknown model 'clm-nope'"})

    agent = Agent(mock_model(handler), output_type=Ticket)
    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('Charged twice.')
    assert exc_info.value.status_code == 422
    assert exc_info.value.body == {'detail': "unknown model 'clm-nope'"}
    assert exc_info.value.model_name == 'clm-latest'


async def test_http_error_with_a_text_body(allow_model_requests: None):
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(502, text='embedder unreachable')

    agent = Agent(mock_model(handler), output_type=Ticket)
    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('Charged twice.')
    assert exc_info.value.status_code == 502
    assert exc_info.value.body == 'embedder unreachable'


async def test_connection_error(allow_model_requests: None):
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError('Connection refused')

    agent = Agent(mock_model(handler), output_type=Ticket)
    with pytest.raises(ModelAPIError, match='ConnectError: Connection refused'):
        await agent.run('Charged twice.')


async def test_server_error_falls_back(allow_model_requests: None):
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(503, text='warming up')

    agent = Agent(FallbackModel(mock_model(handler), TestModel()), output_type=Ticket)
    result = await agent.run('Charged twice.')
    assert result.response.model_name == 'test'


async def test_invalid_response(allow_model_requests: None):
    def handler(request: httpx2.Request) -> httpx2.Response:
        return answers(urgent={'type': 'noul'})

    agent = Agent(mock_model(handler), output_type=Ticket)
    with pytest.raises(UnexpectedModelBehavior, match='Invalid response from the System One server'):
        await agent.run('Charged twice.')


async def test_provider_recreates_its_client():
    provider = SystemOneProvider(base_url=BASE_URL)
    first = provider.client
    async with provider:
        pass
    assert first.is_closed
    async with provider:
        assert provider.client is not first
        assert not provider.client.is_closed
