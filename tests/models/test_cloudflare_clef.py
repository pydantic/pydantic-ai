"""Unit transport tests pin exact requests and synthetic responses that VCR cannot reliably reproduce."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Literal
from unittest.mock import AsyncMock, patch

import httpx2
import pytest
from pydantic import BaseModel, Field, WithJsonSchema

from pydantic_ai import Agent, BinaryContent, ImageUrl
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelConnectionError,
    ModelHTTPError,
    ModelRateLimitError,
    ModelServerError,
    ModelTimeoutError,
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai.models import infer_model
from pydantic_ai.models.cloudflare_clef import CloudflareClefModel
from pydantic_ai.models.decision import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionModelSettings,
    DecisionQuestion,
    DecisionRequest,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)
from pydantic_ai.providers import infer_provider_class
from pydantic_ai.providers.cloudflare import CloudflareProvider
from pydantic_ai.providers.system_one import SystemOneProvider
from pydantic_ai.usage import RequestUsage

pytestmark = pytest.mark.usefixtures('allow_model_requests')


@pytest.mark.vcr
@pytest.mark.skipif(not os.getenv('CLOUDFLARE_API_TOKEN'), reason='requires Cloudflare credentials to record')
async def test_clef_agent_run():
    """Recorded end-to-end run of the Clef path through `Agent`."""
    provider = CloudflareProvider(
        api_key=os.environ['CLOUDFLARE_API_TOKEN'], account_id=os.environ['CLOUDFLARE_ACCOUNT_ID']
    )
    agent = Agent(CloudflareClefModel('@cf/cloudflare/clef', provider=provider), instructions='Answer briefly.')
    result = await agent.run('What is the capital of France?')
    assert result.output


ACCOUNT_URL = 'https://api.cloudflare.com/client/v4/accounts/test-account'
MODEL_IDS: list[str] = ['@cf/cloudflare/clef', '@cf/cloudflare/clef-flash']

Impact = Annotated[
    Literal[0, 1, 2],
    WithJsonSchema(
        {
            'type': 'integer',
            'anyOf': [
                {'const': 0, 'description': 'No service impact'},
                {'const': 1, 'description': 'Some users affected'},
                {'const': 2, 'description': 'All production users affected'},
            ],
        }
    ),
]


class Triage(BaseModel):
    """Triage a production incident."""

    urgent: bool = Field(description='Does the request describe an urgent production incident?')
    route: Literal['support', 'sales'] = Field(description='Which team should handle the request?')
    impact: Impact = Field(description='Rate the incident impact.')


TRIAGE_ANSWERS: dict[str, object] = {
    'urgent': {'type': 'noul', 'noul': 0.9667},
    'route': {
        'type': 'choice',
        'choice': 'support',
        'probabilities': {'support': 0.9589, 'sales': 0.0411},
        'confidence': 0.8423,
    },
    'impact': {
        'type': 'score',
        'score': 1.9141,
        'legend': {'0': 'No service impact', '1': 'Some users affected', '2': 'All production users affected'},
        'probabilities': {'0': 0.0236, '1': 0.0387, '2': 0.9377},
        'confidence': 0.822,
    },
}


def response_body(answers: dict[str, object], *, model: str = 'clef-flash') -> dict[str, object]:
    return {
        'result': {'model': model, 'answers': answers, 'usage': {'input_tokens': 314, 'output_tokens': 0}},
        'success': True,
        'errors': [],
        'messages': [],
    }


@dataclass
class Captured:
    model: CloudflareClefModel
    requests: list[httpx2.Request]


def mock_model(
    body: object,
    *,
    model_name: str = '@cf/cloudflare/clef-flash',
    status: int = 200,
    headers: dict[str, str] | None = None,
) -> Captured:
    requests: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        request.read()
        requests.append(request)
        if isinstance(body, bytes):
            return httpx2.Response(status, content=body, headers=headers)
        return httpx2.Response(status, json=body, headers=headers)

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handle))
    provider = CloudflareProvider(api_key='test-token', base_url=ACCOUNT_URL, http_client=client)
    return Captured(CloudflareClefModel(model_name, provider=provider), requests)


@pytest.mark.parametrize('model_name', MODEL_IDS)
async def test_decide(model_name: str):
    selector = model_name.rsplit('/', 1)[-1]
    captured = mock_model(
        response_body(TRIAGE_ANSWERS, model=selector), model_name=model_name, headers={'cf-ai-req-id': 'request-123'}
    )
    questions: dict[str, DecisionQuestion] = {
        'urgent': NoulQuestion(instructions='Is this urgent?'),
        'route': ChoiceQuestion(
            instructions='Which team?', criteria={'support': 'Technical support', 'sales': 'Sales'}
        ),
        'impact': ScoreQuestion(
            instructions='Rate the impact.',
            criteria=['No service impact', 'Some users affected', 'All production users affected'],
        ),
    }
    result = await captured.model.decide(DecisionRequest(state='A production outage', questions=questions), {})
    assert result.model_name == selector
    assert result.provider_response_id == 'request-123'
    assert result.usage == RequestUsage(input_tokens=314, output_tokens=0)
    assert result.answers['urgent'] == NoulAnswer(noul=0.9667)
    assert result.answers['route'] == ChoiceAnswer(
        choice='support', probabilities={'support': 0.9589, 'sales': 0.0411}, confidence=0.8423
    )
    assert result.answers['impact'] == ScoreAnswer(
        score=1.9141,
        legend={0: 'No service impact', 1: 'Some users affected', 2: 'All production users affected'},
        probabilities={0: 0.0236, 1: 0.0387, 2: 0.9377},
        confidence=0.822,
    )
    sent = captured.requests[0]
    assert sent.method == 'POST'
    assert str(sent.url) == f'{ACCOUNT_URL}/ai/run/{model_name}'
    assert sent.headers['authorization'] == 'Bearer test-token'
    assert json.loads(sent.content) == {
        'model': selector,
        'state': 'A production outage',
        'questions': {
            'urgent': {'type': 'noul', 'instructions': 'Is this urgent?'},
            'route': {
                'type': 'choice',
                'instructions': 'Which team?',
                'criteria': {'support': 'Technical support', 'sales': 'Sales'},
            },
            'impact': {
                'type': 'score',
                'instructions': 'Rate the impact.',
                'criteria': ['No service impact', 'Some users affected', 'All production users affected'],
            },
        },
    }


@pytest.mark.parametrize('model_name', MODEL_IDS)
@pytest.mark.parametrize('streamed', [False, True])
async def test_agent_decisions(model_name: str, streamed: bool):
    captured = mock_model(response_body(TRIAGE_ANSWERS, model=model_name.rsplit('/', 1)[-1]), model_name=model_name)
    agent = Agent(captured.model, output_type=Triage)
    if streamed:
        async with agent.run_stream('All production users are affected by an outage.') as stream:
            output = await stream.get_output()
    else:
        output = (await agent.run('All production users are affected by an outage.')).output
    assert output == Triage(urgent=True, route='support', impact=2)
    assert len(captured.requests) == 1
    questions: dict[str, dict[str, object]] = json.loads(captured.requests[0].content)['questions']
    assert {name: question['type'] for name, question in questions.items()} == {
        'urgent': 'noul',
        'route': 'choice',
        'impact': 'score',
    }


async def test_infer_model_with_connection_factory(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv('CLOUDFLARE_ACCOUNT_ID', raising=False)
    monkeypatch.delenv('CLOUDFLARE_API_TOKEN', raising=False)
    captured = mock_model(response_body({'response': {'type': 'noul', 'noul': 0.9}}))
    prefixes: list[str] = []

    def factory(prefix: str) -> CloudflareProvider:
        prefixes.append(prefix)
        provider_class = infer_provider_class(prefix)
        assert issubclass(provider_class, CloudflareProvider)
        return provider_class(api_key='connection-token', base_url=ACCOUNT_URL, http_client=captured.model.client)

    model = infer_model('cloudflare:@cf/cloudflare/clef-flash', provider_factory=factory)
    assert isinstance(model, CloudflareClefModel)
    result = await Agent(model, output_type=bool, instructions='Is this urgent?').run('Production is down.')
    assert result.output is True
    assert prefixes == ['cloudflare']
    assert captured.requests[0].headers['authorization'] == 'Bearer connection-token'


def test_infer_model_rejects_wrong_provider():
    with pytest.raises(UserError, match='Cloudflare models require a `CloudflareProvider`'):
        infer_model(
            'cloudflare:@cf/cloudflare/clef',
            provider_factory=lambda _: SystemOneProvider(base_url='https://example.com'),
        )


@pytest.mark.parametrize('question_name', ['overridden', 'bad key'])
async def test_settings_and_question_override(question_name: str):
    wire_name = 'q0' if question_name == 'bad key' else question_name
    captured = mock_model(response_body({wire_name: {'type': 'noul', 'noul': 0.8}}))
    settings: DecisionModelSettings = {
        'temperature': 0.2,
        'top_p': 0.5,
        'timeout': 7,
        'extra_headers': {'authorization': 'Bearer override', 'x-extra': 'extra'},
        'extra_body': MappingProxyType(
            {
                'state': {'custom': 'state'},
                'questions': {question_name: {'type': 'noul', 'instructions': 'An override?'}},
            }
        ),
    }
    result = await captured.model.decide(
        DecisionRequest(state='original', questions={'original': NoulQuestion(instructions='Original?')}), settings
    )
    assert result.answers == {question_name: NoulAnswer(noul=0.8)}
    assert result.provider_response_id is None
    sent = captured.requests[0]
    assert sent.headers['authorization'] == 'Bearer override'
    assert sent.headers['x-extra'] == 'extra'
    assert sent.extensions['timeout'] == {'connect': 7, 'read': 7, 'write': 7, 'pool': 7}
    assert json.loads(sent.content) == {
        'model': 'clef-flash',
        'state': {'custom': 'state'},
        'questions': {wire_name: {'type': 'noul', 'instructions': 'An override?'}},
    }


@pytest.mark.parametrize('body', [[], {'questions': 'invalid'}, {'images': 'invalid'}, {1: 'non-string key'}])
async def test_invalid_extra_body(body: object):
    captured = mock_model({})
    with pytest.raises(UserError, match=r'Cloudflare.*extra_body'):
        await captured.model.decide(
            DecisionRequest(state='state', questions={'q': NoulQuestion(instructions='Question?')}),
            {'extra_body': body},
        )
    assert captured.requests == []


async def test_unserializable_body():
    captured = mock_model({})
    with pytest.raises(UserError, match='Could not send this request to Cloudflare Workers AI'):
        await captured.model.decide(
            DecisionRequest(state='state', questions={'q': NoulQuestion(instructions='Question?')}),
            {'extra_body': {'unsupported': object()}},
        )
    assert captured.requests == []


@pytest.mark.parametrize(
    'body', [b'not JSON', {}, {'result': None}, {'result': {'model': 'clef-flash', 'answers': {}}}]
)
async def test_invalid_envelope(body: object):
    captured = mock_model(body)
    with pytest.raises(UnexpectedModelBehavior, match='Invalid response from Cloudflare Workers AI'):
        await captured.model.decide(DecisionRequest(state='state', questions={'q': NoulQuestion()}), {})


@pytest.mark.parametrize('question_name', ['q', 'bad key'])
async def test_answer_names_must_match(question_name: str):
    captured = mock_model(response_body({'other': {'type': 'noul', 'noul': 0.8}}))
    with pytest.raises(UnexpectedModelBehavior, match='answer names do not match'):
        await captured.model.decide(DecisionRequest(state='state', questions={question_name: NoulQuestion()}), {})


@pytest.mark.parametrize('alias', ['', 'bad key', 'évidence', 'q' * 101])
async def test_field_aliases_are_mapped_to_safe_question_ids(alias: str):
    class AliasedOutput(BaseModel):
        original: bool = Field(alias='q0', description='Is the incident resolved?')
        aliased: bool = Field(alias=alias, description='Is there a production incident?')

    captured = mock_model(response_body({'q0': {'type': 'noul', 'noul': 0.1}, 'q1': {'type': 'noul', 'noul': 0.9}}))
    output = (await Agent(captured.model, output_type=AliasedOutput).run('Production is down.')).output
    assert output.model_dump(by_alias=True) == {'q0': False, alias: True}
    assert json.loads(captured.requests[0].content)['questions'] == {
        'q0': {'type': 'noul', 'instructions': {'field': 'q0', 'question': 'Is the incident resolved?'}},
        'q1': {'type': 'noul', 'instructions': {'field': alias, 'question': 'Is there a production incident?'}},
    }


@pytest.mark.parametrize('question_name', ['valid.id_1-with-dashes', 'q' * 100])
async def test_supported_question_ids_are_preserved(question_name: str):
    captured = mock_model(response_body({question_name: {'type': 'noul', 'noul': 0.9}}))
    result = await captured.model.decide(
        DecisionRequest(state='state', questions={question_name: NoulQuestion(instructions='Is this urgent?')}), {}
    )
    assert result.answers == {question_name: NoulAnswer(noul=0.9)}
    assert list(json.loads(captured.requests[0].content)['questions']) == [question_name]


@pytest.mark.parametrize(
    ('status', 'body', 'category'),
    [
        (401, {'success': False, 'errors': [{'code': 10000, 'message': 'Authentication error'}]}, ModelHTTPError),
        (429, {'errors': [{'message': 'Rate limit exceeded'}]}, ModelRateLimitError),
        (500, b'upstream unavailable', ModelServerError),
    ],
)
async def test_http_errors(status: int, body: object, category: type[ModelAPIError]):
    captured = mock_model(body, status=status, headers={'cf-ray': 'test-ray'})
    with pytest.raises(ModelHTTPError) as exc:
        await captured.model.decide(DecisionRequest(state='state', questions={'q': NoulQuestion()}), {})
    assert exc.value.status_code == status
    assert isinstance(exc.value, category)
    assert exc.value.model_name == '@cf/cloudflare/clef-flash'
    assert exc.value.body == (body.decode() if isinstance(body, bytes) else body)
    assert exc.value.headers is not None and exc.value.headers['cf-ray'] == 'test-ray'


@pytest.mark.parametrize(
    ('transport_error', 'category'),
    [(httpx2.ReadTimeout('timed out'), ModelTimeoutError), (httpx2.ConnectError('disconnected'), ModelConnectionError)],
)
async def test_transport_error(transport_error: httpx2.TransportError, category: type[ModelAPIError]):
    def handle(request: httpx2.Request) -> httpx2.Response:
        raise transport_error

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        model = CloudflareClefModel(
            '@cf/cloudflare/clef',
            provider=CloudflareProvider(api_key='token', account_id='account', http_client=client),
        )
        with pytest.raises(category, match=str(transport_error)):
            await model.decide(DecisionRequest(state='state', questions={'q': NoulQuestion()}), {})


@pytest.mark.parametrize('media_type', ['image/png', 'image/jpeg', 'image/webp'])
async def test_images_are_inline_data_uris(media_type: str):
    captured = mock_model(response_body({'response': {'type': 'noul', 'noul': 0.9}}))
    image = BinaryContent(data=b'image-bytes', media_type=media_type)
    agent = Agent(captured.model, output_type=bool, instructions='Is this a fruit?')
    assert (await agent.run(['Identify this image.', image])).output is True
    body: dict[str, object] = json.loads(captured.requests[0].content)
    assert body['images'] == [image.data_uri]
    assert '<image 1>' in str(body['state'])


async def test_image_url_is_downloaded_and_sent_inline():
    captured = mock_model(response_body({'response': {'type': 'noul', 'noul': 0.9}}))
    image = ImageUrl('https://example.com/fruit.png')
    agent = Agent(captured.model, output_type=bool, instructions='Is this a fruit?')
    with patch(
        'pydantic_ai.models.decision.download_item',
        new_callable=AsyncMock,
        return_value={'data': b'fruit', 'data_type': 'image/png'},
    ) as download:
        assert (await agent.run(['Identify this image.', image])).output is True
    download.assert_awaited_once_with(image, data_format='bytes')
    body: dict[str, object] = json.loads(captured.requests[0].content)
    assert body['images'] == ['data:image/png;base64,ZnJ1aXQ=']
    assert '<image 1>' in str(body['state'])


async def test_extra_body_images_at_limit():
    captured = mock_model(response_body({'q': {'type': 'noul', 'noul': 0.9}}))
    images = ['data:image/png;base64,ZnJ1aXQ='] * 4
    await captured.model.decide(
        DecisionRequest(state='state', questions={'q': NoulQuestion(instructions='Is this fruit?')}),
        {'extra_body': {'images': images}},
    )
    assert json.loads(captured.requests[0].content)['images'] == images


@pytest.mark.parametrize('media_type', ['image/gif', 'audio/wav'])
async def test_direct_unsupported_media(media_type: str):
    captured = mock_model({})
    image = BinaryContent(data=b'bytes', media_type=media_type)
    with pytest.raises(UserError, match='PNG, JPEG, and WebP'):
        await captured.model.decide(
            DecisionRequest(state='state', questions={'q': NoulQuestion()}, images=(image,)), {}
        )
    assert captured.requests == []


@pytest.mark.parametrize('kind', ['questions', 'images', 'overridden-questions', 'overridden-images'])
async def test_direct_transport_limits(kind: str):
    captured = mock_model({})
    questions: dict[str, DecisionQuestion] = {'q': NoulQuestion(instructions='Question?')}
    images: tuple[BinaryContent, ...] = ()
    settings: DecisionModelSettings = {}
    if kind == 'questions':
        questions = {str(i): NoulQuestion(instructions='Question?') for i in range(65)}
    elif kind == 'images':
        images = (BinaryContent(data=b'bytes', media_type='image/jpeg'),) * 5
    elif kind == 'overridden-questions':
        settings['extra_body'] = {
            'questions': {str(i): {'type': 'noul', 'instructions': 'Question?'} for i in range(65)}
        }
    else:
        settings['extra_body'] = {'images': ['data:image/png;base64,ZnJ1aXQ='] * 5}
    with pytest.raises(ModelAPIError, match='accepts at most'):
        await captured.model.decide(DecisionRequest(state='state', questions=questions, images=images), settings)
    assert captured.requests == []
