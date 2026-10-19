from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Annotated
from unittest.mock import AsyncMock, patch

import httpx2
import pytest
from pydantic import BaseModel, Field, JsonValue, TypeAdapter

from pydantic_ai import Agent, BinaryContent, ImageUrl, NativeOutput, PromptedOutput, ToolOutput
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters, infer_model
from pydantic_ai.models.cloudflare import CloudflareClefModel
from pydantic_ai.models.cloudflare_clairic import (
    ClairicDateEstimate,
    ClairicField,
    ClairicForecast,
    ClairicNumberEstimate,
    CloudflareClairicModel,
    CloudflareClairicModelSettings,
)
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.output import OutputObjectDefinition
from pydantic_ai.providers.cloudflare import CloudflareProvider
from pydantic_ai.providers.system_one import SystemOneProvider
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

# MockTransport fixtures verify the wire contract; these are not HTTP recordings.
pytestmark = pytest.mark.usefixtures('allow_model_requests')
ACCOUNT_URL = 'https://api.cloudflare.com/client/v4/accounts/test-account'


class Extraction(BaseModel):
    target: str | None
    article: int
    effective_date: date
    publication_date: date | None


@dataclass
class Captured:
    model: CloudflareClairicModel
    requests: list[httpx2.Request]


def mock_model(answers: dict[str, object], *, status: int = 200, raw: object = None) -> Captured:
    requests: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        request.read()
        requests.append(request)
        state: str = json.loads(request.content)['state']
        body = (
            raw
            if raw is not None
            else {
                'success': True,
                'result': {
                    'model': 'clairic',
                    'answers': answers,
                    'truncated': False,
                    'state_text': state,
                    'usage': {'input_tokens': 321, 'output_tokens': 0, 'state_tokens': 20, 'passes': 2},
                },
            }
        )
        return httpx2.Response(
            status,
            content=body if isinstance(body, bytes) else None,
            json=None if isinstance(body, bytes) else body,
            headers={'cf-ai-req-id': 'req-123'},
        )

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handle))
    return Captured(
        CloudflareClairicModel(provider=CloudflareProvider(api_key='token', base_url=ACCOUNT_URL, http_client=client)),
        requests,
    )


def params(schema: dict[str, JsonValue]) -> ModelRequestParameters:
    return ModelRequestParameters(
        output_mode='native', allow_text_output=False, output_object=OutputObjectDefinition(json_schema=schema)
    )


async def test_agent_native_output_with_textless_profile():
    answers: dict[str, object] = {
        'target': {
            'type': 'text',
            'value': 'article 7',
            'span': [6, 15],
            'guess': 'article 7',
            'p_null': 0.01,
            'confidence': 0.99,
        },
        'article': {'type': 'number', 'value': 7, 'guess': 7, 'confidence': 0.99, 'p_null': 0.01},
        'effective_date': {'type': 'date', 'value': '2027-03-01', 'p_null': 0.02},
        'publication_date': {'type': 'date', 'value': None, 'guess': '2027-03-01', 'p_null': 1.0, 'confidence': 0.99},
    }
    captured = mock_model(answers)
    assert captured.model.profile.get('supports_text_output') is False
    agent = Agent(captured.model, output_type=Extraction, instructions='Extract the amended article and dates.')
    result = await agent.run('Amend article 7 effective 2027-03-01.')
    assert result.output == Extraction(
        target='article 7', article=7, effective_date=date(2027, 3, 1), publication_date=None
    )
    assert result.response.usage == RequestUsage(
        input_tokens=321, output_tokens=0, details={'state_tokens': 20, 'passes': 2}
    )
    assert result.response.provider_response_id == 'req-123'
    details = result.response.provider_details
    assert details is not None
    assert details['answers'] == answers
    assert details['state_text'] == details['request_state'] == 'Amend article 7 effective 2027-03-01.'
    assert details['usage'] == {'input_tokens': 321, 'output_tokens': 0, 'state_tokens': 20, 'passes': 2}
    assert details['truncated'] is False
    sent = captured.requests[0]
    assert str(sent.url) == f'{ACCOUNT_URL}/ai/run/@cf/cloudflare/clairic'
    assert sent.headers['authorization'] == 'Bearer token'
    body = json.loads(sent.content)
    assert body['fields']['article'] == {
        'type': 'number',
        'integer': True,
        'instructions': 'Extract the amended article and dates.\n\nField: article',
    }
    assert body['options'] == {'return_state_text': True, 'include_quantiles': False}
    assert 'tools' not in body


@pytest.mark.parametrize('explicit_native', [False, True])
async def test_scalar_and_list_fields(explicit_native: bool):
    class Fields(BaseModel):
        text: str
        number: float
        integer: int
        day: date
        clock: time
        instant: datetime
        texts: list[str]
        numbers: list[float]
        days: list[date]
        missing: str | None
        optional_list: list[str] | None

    values: dict[str, tuple[str, JsonValue]] = {
        'text': ('text', 'Invoice'),
        'number': ('number', 1080.5),
        'integer': ('number', 7),
        'day': ('date', '2027-03-01'),
        'clock': ('time', '12:30:00'),
        'instant': ('datetime', '2027-03-01T12:30:00'),
        'texts': ('list', ['Dana', 'Priya']),
        'numbers': ('list', [1.5, 2]),
        'days': ('list', ['2027-03-01']),
        'missing': ('text', None),
        'optional_list': ('list', []),
    }
    captured = mock_model({name: {'type': field_type, 'value': value} for name, (field_type, value) in values.items()})
    output_type = NativeOutput(Fields) if explicit_native else Fields
    result = await Agent(captured.model, output_type=output_type).run('Invoice details')
    assert result.output.model_dump(mode='json') == {name: value for name, (_, value) in values.items()}
    specs = json.loads(captured.requests[0].content)['fields']
    assert specs['integer']['integer'] is True
    assert specs['numbers']['items'] == 'number'
    assert 'integer' not in specs['numbers']
    assert specs['clock']['type'] == 'time'
    assert specs['instant']['type'] == 'datetime'
    assert specs['days']['items'] == 'date'
    assert specs['missing']['type'] == 'text'


@pytest.mark.parametrize('alias', ['', 'bad key', 'évidence', '9start', 'x' * 65])
async def test_aliases_map_without_collisions(alias: str):
    class Aliased(BaseModel):
        first: str = Field(alias='f0')
        second: str | None = Field(alias=alias, description='Missing reference')

    captured = mock_model(
        {
            'f0': {'type': 'text', 'value': 'present'},
            'f1': {'type': 'text', 'value': None, 'guess': 'wrong', 'p_null': 1},
        }
    )
    result = await Agent(captured.model, output_type=Aliased).run('present')
    assert result.output.model_dump(by_alias=True) == {'f0': 'present', alias: None}
    assert result.response.provider_details is not None
    assert result.response.provider_details['field_names'] == {'f0': 'f0', 'f1': alias}
    assert (
        json.loads(captured.requests[0].content)['fields']['f1']['instructions']
        == f'Field: {alias}\n\nMissing reference'
    )


@pytest.mark.parametrize(
    'schema',
    [
        {'type': 'boolean'},
        {'type': 'object', 'properties': {'nested': {'type': 'string'}}},
        {'type': 'string', 'enum': ['a', 'b']},
        {'type': 'integer', 'minimum': 0},
        {'type': 'string', 'pattern': 'a+'},
        {'type': 'string', 'format': 'uuid'},
        {'type': 'array', 'items': {'type': 'integer'}},
        {'type': 'array', 'items': {'type': 'boolean'}},
        {'type': 'array', 'items': {'type': 'string', 'format': 'time'}},
        {'type': 'array'},
        {'type': 'array', 'items': {'anyOf': [{'type': 'string'}, {'type': 'null'}]}},
        {'anyOf': [{'type': 'string'}, {'type': 'number'}]},
        {'anyOf': []},
        {'$ref': '#/$defs/Thing'},
    ],
)
async def test_unsupported_fields_rejected_before_http(schema: dict[str, JsonValue]):
    captured = mock_model({})
    with pytest.raises(UserError, match='Cloudflare Clairic'):
        await captured.model.request(
            [ModelRequest.user_text_prompt('state')], None, params({'type': 'object', 'properties': {'field': schema}})
        )
    assert captured.requests == []


@pytest.mark.parametrize(
    'schema',
    [
        {'type': 'object', 'properties': {}},
        {'type': 'array', 'items': {'type': 'string'}},
        {'type': 'object', 'properties': {'x': {'type': 'string'}}, 'additionalProperties': {'type': 'string'}},
    ],
)
async def test_unsupported_root_schema(schema: dict[str, JsonValue]):
    captured = mock_model({})
    with pytest.raises(UserError, match='flat output object'):
        await captured.model.request([ModelRequest.user_text_prompt('state')], None, params(schema))
    assert captured.requests == []


async def test_unsupported_agent_output_and_tools():
    captured = mock_model({})
    for output_type in (str, ToolOutput(Extraction), PromptedOutput(Extraction), [Extraction, int]):
        with pytest.raises(UserError):
            await Agent(captured.model, output_type=output_type).run('state')
    for request_params in (
        ModelRequestParameters(function_tools=[ToolDefinition(name='search')]),
        ModelRequestParameters(native_tools=[WebSearchTool()]),
    ):
        with pytest.raises(UserError, match='does not support function or native tools'):
            await captured.model.request([ModelRequest.user_text_prompt('state')], None, request_params)
    assert captured.requests == []


async def test_history_and_source_spans():
    entries: list[dict[str, str]] = [
        {'role': 'system', 'content': 'Read literally.'},
        {'role': 'user', 'content': 'first café'},
        {'role': 'assistant', 'content': '{"target":"café"}'},
        {'role': 'user', 'content': 'Validation feedback:\nTry again\n\nFix the errors and try again.'},
        {'role': 'user', 'content': 'second café'},
    ]
    expected_state = json.dumps(entries, ensure_ascii=False)
    start = expected_state.index('café')
    answer: dict[str, object] = {'type': 'text', 'value': 'café', 'span': [start, start + 4]}
    captured = mock_model({'target': answer})
    request_params = params({'type': 'object', 'properties': {'target': {'type': 'string'}}})
    history = [
        ModelRequest(parts=[SystemPromptPart('Read literally.'), UserPromptPart('first café')]),
        ModelResponse(parts=[TextPart('{"target":"café"}')]),
        ModelRequest(parts=[RetryPromptPart('Try again'), UserPromptPart('second café')]),
    ]
    response = await captured.model.request(history, None, request_params)
    sent_state = json.loads(captured.requests[0].content)['state']
    assert sent_state == expected_state
    assert response.provider_details is not None
    assert response.provider_details['request_state'] == response.provider_details['state_text'] == sent_state
    assert response.provider_details['answers'] == {'target': answer}
    assert sent_state[start : start + 4] == 'café'


@pytest.mark.parametrize(
    'part', [ThinkingPart('thinking'), ToolCallPart('search', {}), ToolReturnPart('search', 'found')]
)
async def test_unsupported_history(part: ThinkingPart | ToolCallPart | ToolReturnPart):
    captured = mock_model({})
    history: list[ModelMessage] = (
        [ModelRequest(parts=[part])] if isinstance(part, ToolReturnPart) else [ModelResponse(parts=[part])]
    )
    with pytest.raises(UserError, match='history part'):
        await captured.model.request(history, None, params({'type': 'object', 'properties': {'x': {'type': 'string'}}}))
    assert captured.requests == []


async def test_retry_keeps_null_instead_of_guess():
    captured = mock_model({'article': {'type': 'number', 'value': None, 'guess': 7, 'p_null': 1}})

    class Required(BaseModel):
        article: int

    with pytest.raises(UnexpectedModelBehavior, match='Exceeded maximum'):
        await Agent(captured.model, output_type=Required, retries=1).run('No article is given.')
    assert len(captured.requests) == 2
    history = json.loads(json.loads(captured.requests[1].content)['state'])
    assert history[1] == {'role': 'assistant', 'content': '{"article": null}'}
    assert 'Input should be a valid integer' in history[2]['content']


async def test_images_inline_and_url_download():
    captured = mock_model({'text': {'type': 'text', 'value': 'Invoice'}})

    class Output(BaseModel):
        text: str

    image_url = ImageUrl('https://example.com/invoice.png')
    with patch(
        'pydantic_ai.models._cloudflare_clairic.download_item',
        new_callable=AsyncMock,
        return_value={'data': b'image', 'data_type': 'image/png'},
    ) as download:
        result = await Agent(captured.model, output_type=Output).run(
            ['Read the invoice.', BinaryContent(data=b'image', media_type='image/jpeg'), image_url]
        )
    assert result.output.text == 'Invoice'
    download.assert_awaited_once_with(image_url, data_format='bytes')
    body = json.loads(captured.requests[0].content)
    assert body['images'] == ['data:image/jpeg;base64,aW1hZ2U=', 'data:image/png;base64,aW1hZ2U=']
    assert json.loads(body['state'])[-1] == {'role': 'user', 'content': '<image 2>'}


@pytest.mark.parametrize(
    'prompt',
    [
        '',
        [BinaryContent(data=b'image', media_type='image/gif')],
        [BinaryContent(data=b'image', media_type='image/png')] * 5,
    ],
)
async def test_invalid_input(prompt: str | list[BinaryContent]):
    captured = mock_model({})
    with pytest.raises(UserError, match='Cloudflare Clairic'):
        await captured.model.request(
            [ModelRequest(parts=[UserPromptPart(prompt)])],
            None,
            params({'type': 'object', 'properties': {'x': {'type': 'string'}}}),
        )
    assert captured.requests == []


async def test_settings_and_state_text_fallback():
    raw: dict[str, object] = {
        'result': {
            'model': 'clairic',
            'answers': {'x': {'type': 'text', 'value': 'state'}},
            'truncated': True,
            'usage': {'input_tokens': 2, 'output_tokens': 0},
        }
    }
    captured = mock_model({}, raw=raw)
    response = await captured.model.request(
        [ModelRequest.user_text_prompt('state')],
        {
            'temperature': 0.2,
            'timeout': 7,
            'extra_headers': {'Authorization': 'Bearer override'},
            'extra_body': {'reference_date': '2026-10-19'},
        },
        params({'type': 'object', 'properties': {'x': {'type': 'string'}}}),
    )
    sent = captured.requests[0]
    assert sent.headers['authorization'] == 'Bearer override'
    assert sent.extensions['timeout'] == {'connect': 7, 'read': 7, 'write': 7, 'pool': 7}
    body = json.loads(sent.content)
    assert body['reference_date'] == '2026-10-19'
    assert 'temperature' not in body
    assert response.finish_reason == 'length'
    assert response.provider_details is not None and response.provider_details['state_text'] == 'state'


@pytest.mark.parametrize('key', ['model', 'state', 'fields', 'images', 'options'])
async def test_extra_body_cannot_bypass_schema_or_state(key: str):
    captured = mock_model({})
    with pytest.raises(UserError, match='cannot override'):
        await captured.model.request(
            [ModelRequest.user_text_prompt('state')],
            {'extra_body': {key: 'override'}},
            params({'type': 'object', 'properties': {'x': {'type': 'string'}}}),
        )
    assert captured.requests == []


@pytest.mark.parametrize(
    ('status', 'raw'), [(401, {'errors': [{'message': 'unauthorized'}]}), (500, b'upstream unavailable')]
)
async def test_http_errors(status: int, raw: object):
    captured = mock_model({}, status=status, raw=raw)
    with pytest.raises(ModelHTTPError) as exc:
        await captured.model.request(
            [ModelRequest.user_text_prompt('state')],
            None,
            params({'type': 'object', 'properties': {'x': {'type': 'string'}}}),
        )
    assert exc.value.status_code == status
    assert exc.value.body == (raw.decode() if isinstance(raw, bytes) else raw)


@pytest.mark.parametrize('raw', [b'not JSON', {}, {'result': None}, {'result': {'model': 'clairic'}}])
async def test_invalid_response(raw: object):
    captured = mock_model({}, raw=raw)
    with pytest.raises(UnexpectedModelBehavior, match='Invalid response from Cloudflare Clairic'):
        await captured.model.request(
            [ModelRequest.user_text_prompt('state')],
            None,
            params({'type': 'object', 'properties': {'x': {'type': 'string'}}}),
        )


@pytest.mark.parametrize(
    'answers', [{}, {'other': {'type': 'text', 'value': 'x'}}, {'x': {'type': 'number', 'value': 3}}]
)
async def test_mismatched_answers(answers: dict[str, object]):
    captured = mock_model(answers)
    with pytest.raises(UnexpectedModelBehavior, match='Cloudflare Clairic'):
        await captured.model.request(
            [ModelRequest.user_text_prompt('state')],
            None,
            params({'type': 'object', 'properties': {'x': {'type': 'string'}}}),
        )


async def test_transport_error():
    def handle(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout('timed out')

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        model = CloudflareClairicModel(
            provider=CloudflareProvider(api_key='token', account_id='account', http_client=client)
        )
        with pytest.raises(ModelAPIError, match='ReadTimeout: timed out'):
            await model.request(
                [ModelRequest.user_text_prompt('state')],
                None,
                params({'type': 'object', 'properties': {'x': {'type': 'string'}}}),
            )


async def test_inference_preserves_clef(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('CLOUDFLARE_API_TOKEN', 'token')
    monkeypatch.setenv('CLOUDFLARE_ACCOUNT_ID', 'account')
    model = infer_model('cloudflare:@cf/cloudflare/clairic')
    assert isinstance(model, CloudflareClairicModel)
    assert isinstance(infer_model('cloudflare:@cf/cloudflare/clef'), CloudflareClefModel)
    assert isinstance(infer_model('cloudflare:@cf/cloudflare/clef-flash'), CloudflareClefModel)
    captured = mock_model({})
    provider = captured.model.provider
    assert provider is not None
    connected = infer_model('cloudflare:@cf/cloudflare/clairic', provider_factory=lambda _: provider)
    assert isinstance(connected, CloudflareClairicModel)
    assert connected.client is captured.model.client
    with pytest.raises(UserError, match='Clairic models require a `CloudflareProvider`'):
        infer_model(
            'cloudflare:@cf/cloudflare/clairic',
            provider_factory=lambda _: SystemOneProvider(base_url='https://example.com'),
        )
    with pytest.raises(NotImplementedError, match='Streamed requests not supported'):
        async with model.request_stream([], None, ModelRequestParameters()):
            pytest.fail('Clairic has no streaming API')


@pytest.mark.parametrize('extra_body', [[], {1: 'not a string key'}, {'unsupported': object()}])
async def test_invalid_extra_body(extra_body: object):
    captured = mock_model({})
    with pytest.raises(UserError, match=r'extra_body|Could not send'):
        await captured.model.request(
            [ModelRequest.user_text_prompt('state')],
            {'extra_body': extra_body},
            params({'type': 'object', 'properties': {'x': {'type': 'string'}}}),
        )
    assert captured.requests == []


async def test_returned_state_is_preserved_independently_of_request():
    raw: dict[str, object] = {
        'result': {
            'model': 'clairic',
            'answers': {'x': {'type': 'text', 'value': 'café', 'span': [7, 11], 'p_null': 0.1, 'confidence': 0.8}},
            'state_text': 'Prefix café',
            'truncated': True,
            'usage': {'input_tokens': 2, 'output_tokens': 0, 'image_tokens': 10, 'schema_tokens': 5, 'passes': 1},
            'additional_metadata': {'retained': True},
        },
    }
    captured = mock_model({}, raw=raw)
    response = await captured.model.request(
        [ModelRequest.user_text_prompt('café')],
        None,
        params({'type': 'object', 'properties': {'x': {'type': 'string'}}}),
    )
    details = response.provider_details
    assert details is not None
    assert details['state_text'] == 'Prefix café'
    assert details['request_state'] == 'café'
    assert details['additional_metadata'] == {'retained': True}
    assert details['state_text'][7:11] == 'café'
    assert response.usage.details == {'image_tokens': 10, 'schema_tokens': 5, 'passes': 1}


@pytest.fixture
def clairic_evidence() -> dict[str, dict[str, JsonValue]]:
    """Executed MCP results, explicitly labelled fixtures in assets/clairic/README.md."""
    adapter = TypeAdapter(dict[str, JsonValue])
    directory = Path(__file__).parents[1] / 'assets' / 'clairic'
    return {
        name: adapter.validate_json((directory / f'{name}.json').read_bytes())
        for name in ('scalar', 'prediction', 'image')
    }


async def test_complete_native_prediction_output(clairic_evidence: dict[str, dict[str, JsonValue]]):
    class Predictions(BaseModel):
        payment_days: Annotated[ClairicNumberEstimate, ClairicField(min=0, max=120, step=1)]
        payment_date: Annotated[
            ClairicDateEstimate,
            ClairicField(min=date(2026, 10, 1), max=date(2027, 1, 31), step=1),
        ]
        collections: Annotated[
            ClairicForecast,
            ClairicField(min=0, max=1000, step=10, horizon=3, interval='week'),
        ]
        service_duration: timedelta

    native = clairic_evidence['prediction']
    captured = mock_model({}, raw={'result': native})
    settings = CloudflareClairicModelSettings(
        cloudflare_clairic_reference_date=date(2026, 10, 19),
        cloudflare_clairic_include_quantiles=True,
    )
    result = await Agent(captured.model, output_type=Predictions, model_settings=settings).run('Invoice history')
    output = result.output
    assert output.payment_days.value == 32
    assert output.payment_days.mean == 31.581952982005706
    assert output.payment_days.p_below == 0.000004675732270698063
    assert output.payment_days.p_above == 0.0000049770628720580135
    assert output.payment_days.quantiles is not None and len(output.payment_days.quantiles) == 99
    assert output.payment_date.value == output.payment_date.mean == date(2026, 11, 4)
    assert output.payment_date.median == date(2026, 11, 1)
    assert output.payment_date.p_below is None and output.payment_date.p_above is None
    assert output.payment_date.quantiles is not None and len(output.payment_date.quantiles) == 99
    assert output.collections.value == [170, 190, 220]
    assert output.collections.median == [170, 190, 210]
    assert output.collections.p10 == [130, 130, 140]
    assert output.collections.p90 == [230, 260, 300]
    assert output.collections.steps[2].value == 220
    assert output.collections.steps[2].median == 210
    for step in output.collections.steps:
        assert 'type' not in step.model_dump(exclude_unset=True)
        assert step.quantiles is not None and len(step.quantiles) == 99
        assert step.p_below is not None and step.p_above is not None
    assert output.service_duration == timedelta(minutes=90)
    assert result.response.provider_details is not None
    assert result.response.provider_details['answers'] == native['answers']
    body = json.loads(captured.requests[0].content)
    assert body['options'] == {'include_quantiles': True, 'return_state_text': True}
    assert body['reference_date'] == '2026-10-19'
    assert body['fields']['payment_date']['min'] == '2026-10-01'
    assert body['fields']['collections']['horizon'] == 3
    assert body['fields']['collections']['interval'] == 'week'
    assert body['fields']['service_duration']['unit'] == 'seconds'
    # Each distribution retains every returned field, including unknown native fields.
    answers = native['answers']
    assert isinstance(answers, dict)
    assert output.payment_days.model_dump(mode='json', exclude_unset=True) == answers['payment_days']
    assert output.payment_date.model_dump(mode='json', exclude_unset=True) == answers['payment_date']
    assert output.collections.model_dump(mode='json', exclude_unset=True) == answers['collections']


async def test_executed_scalar_uncertainty_is_preserved(clairic_evidence: dict[str, dict[str, JsonValue]]):
    class Scalars(BaseModel):
        customer: str
        amount: Annotated[float, ClairicField(unit='USD', decimals=2, null_threshold=0.5)]
        day: date
        clock: time
        instant: datetime
        reviewers: list[str]
        amounts: Annotated[list[float], ClairicField(decimals=2)]
        days: list[date]

    native = clairic_evidence['scalar']
    captured = mock_model({}, raw={'result': native})
    result = await Agent(captured.model, output_type=Scalars).run('Lab record')
    assert result.output.instant == datetime(2026, 10, 20, 14, 30)
    assert result.output.instant.tzinfo is None
    # Well-typed model answers can be wrong: do not silently repair or drop them.
    assert result.output.reviewers == ['Mara S', 'Ada Fox', 'Ravi Shah']
    assert result.response.provider_details is not None
    assert result.response.provider_details['answers'] == native['answers']
    answers = native['answers']
    assert isinstance(answers, dict) and isinstance(answers['reviewers'], dict)
    assert answers['reviewers']['confidence'] == 0.5534043557186195
    assert result.output.amounts == [10.25, 20.5]
    body = json.loads(captured.requests[0].content)
    assert body['fields']['amount']['decimals'] == body['fields']['amounts']['decimals'] == 2
    assert body['fields']['amount']['null_threshold'] == 0.5


async def test_image_only_native_output(clairic_evidence: dict[str, dict[str, JsonValue]]):
    class Count(BaseModel):
        fruit_count: int

    captured = mock_model({}, raw={'result': clairic_evidence['image']})
    image = (Path(__file__).parents[1] / 'assets' / 'kiwi.jpg').read_bytes()
    result = await Agent(captured.model, output_type=Count).run([BinaryContent(data=image, media_type='image/jpeg')])
    assert result.output.fruit_count == 1
    assert result.response.usage.details['image_tokens'] == 342
    body = json.loads(captured.requests[0].content)
    assert body['state'] == ''
    assert body['images'][0].startswith('data:image/jpeg;base64,')
    assert result.response.provider_details is not None
    assert result.response.provider_details['state_text'] == ''


async def test_metadata_survives_nullable_aliases_and_reused_refs():
    class Output(BaseModel):
        outer: Annotated[float | None, ClairicField(unit='USD', decimals=2)] = Field(alias='amount')
        inner: Annotated[float, ClairicField(unit='EUR')] | None
        first: Annotated[ClairicNumberEstimate, ClairicField(min=1, max=10, scale='log')]
        second: Annotated[ClairicNumberEstimate | None, ClairicField(min=0, max=100, step=2)]
        dates: Annotated[ClairicDateEstimate, ClairicField(min=date(2026, 1, 1), max=date(2027, 1, 1), step='month')]

    estimate = {'type': 'estimate', 'value': 5, 'mean': 5, 'median': 5, 'p10': 3, 'p90': 7, 'future_metadata': 123}
    captured = mock_model(
        {
            'amount': {'type': 'number', 'value': None},
            'inner': {'type': 'number', 'value': 1.5},
            'first': estimate,
            'second': estimate,
            'dates': {
                'type': 'estimate',
                'value': '2026-06-01',
                'mean': '2026-06-01',
                'median': '2026-05-01',
                'p10': '2026-02-01',
                'p90': '2026-11-01',
            },
        }
    )
    settings = CloudflareClairicModelSettings(cloudflare_clairic_return_state_text=False)
    provider = captured.model.provider
    assert isinstance(provider, CloudflareProvider)
    captured.model = CloudflareClairicModel(provider=provider, settings=settings)
    result = await Agent(captured.model, output_type=Output).run('state')
    assert result.output.outer is None and result.output.inner == 1.5
    assert result.output.first.model_dump()['future_metadata'] == 123
    assert result.output.second is not None and result.output.second.value == 5
    specs = json.loads(captured.requests[0].content)['fields']
    assert specs['amount']['unit'] == 'USD' and specs['amount']['decimals'] == 2
    assert specs['inner']['unit'] == 'EUR'
    assert specs['first']['min'] == 1 and specs['first']['scale'] == 'log'
    assert specs['second']['min'] == 0 and specs['second']['step'] == 2
    assert specs['dates']['step'] == 'month'
    assert json.loads(captured.requests[0].content)['options']['return_state_text'] is False


@pytest.mark.parametrize('unit', ['seconds', 'minutes', 'hours', 'days', 'weeks', 'months', 'years'])
async def test_numeric_duration_preserves_units(unit: str):
    class Duration(BaseModel):
        elapsed: Annotated[float, ClairicField(kind='duration', unit=unit, decimals=2)]
        whole: Annotated[int, ClairicField(kind='duration', unit=unit)]

    captured = mock_model(
        {
            'elapsed': {'type': 'duration', 'value': 1.5, 'unit': unit},
            'whole': {'type': 'duration', 'value': 2, 'unit': unit},
        }
    )
    result = await Agent(captured.model, output_type=Duration).run('Duration')
    assert result.output.elapsed == 1.5 and result.output.whole == 2
    specs = json.loads(captured.requests[0].content)['fields']
    assert specs['elapsed']['unit'] == specs['whole']['unit'] == unit
    assert specs['elapsed']['decimals'] == 2 and specs['whole']['integer'] is True


@pytest.mark.parametrize(
    ('kind', 'controls'),
    [
        ('text', {'null_threshold': 1}),
        ('number', {'unit': 'USD', 'integer': True, 'decimals': 0}),
        ('list', {'unit': 'USD', 'decimals': 6}),
    ],
)
async def test_native_field_controls(kind: str, controls: dict[str, JsonValue]):
    schemas: dict[str, dict[str, JsonValue]] = {
        'text': {'type': 'string'},
        'number': {'type': 'number'},
        'list': {'type': 'array', 'items': {'type': 'number'}},
    }
    captured = mock_model({'x': {'type': kind, 'value': [] if kind == 'list' else 1 if kind == 'number' else 'text'}})
    await captured.model.request(
        [ModelRequest.user_text_prompt('state')],
        None,
        params({'type': 'object', 'properties': {'x': {**schemas[kind], 'x-cloudflare-clairic': controls}}}),
    )
    field = json.loads(captured.requests[0].content)['fields']['x']
    assert all(field[key] == value for key, value in controls.items())


@pytest.mark.parametrize(
    ('schema', 'controls', 'error'),
    [
        ({'type': 'string'}, {'unit': 'USD'}, 'Incompatible `ClairicField` controls'),
        ({'type': 'string'}, {'kind': 'number'}, '`ClairicField.kind` is incompatible with the result type'),
        ({'type': 'number'}, {'unknown': True}, 'Incompatible `ClairicField` controls'),
        ({'type': 'number'}, {'integer': 'yes'}, '`ClairicField.integer` must be a boolean'),
        ({'type': 'integer'}, {'integer': False}, '`ClairicField.integer=False` conflicts with `int` field'),
        ({'type': 'integer'}, {'decimals': 2}, '`ClairicField.decimals` conflicts with integer precision'),
        ({'type': 'number'}, {'decimals': -1}, '`ClairicField.decimals` must be an integer from zero to six'),
        ({'type': 'number'}, {'decimals': 7}, '`ClairicField.decimals` must be an integer from zero to six'),
        ({'type': 'number'}, {'decimals': True}, '`ClairicField.decimals` must be an integer from zero to six'),
        ({'type': 'number'}, {'decimals': 1.5}, '`ClairicField.decimals` must be an integer from zero to six'),
        (
            {'type': 'number'},
            {'null_threshold': 0},
            '`ClairicField.null_threshold` must be greater than zero and at most one',
        ),
        (
            {'type': 'number'},
            {'null_threshold': 1.1},
            '`ClairicField.null_threshold` must be greater than zero and at most one',
        ),
        (
            {'type': 'number'},
            {'null_threshold': True},
            '`ClairicField.null_threshold` must be greater than zero and at most one',
        ),
        ({'type': 'number'}, {'unit': 1}, '`ClairicField.unit` must be a string'),
        ({'type': 'number'}, {'kind': 'duration', 'unit': 'fortnights'}, 'Invalid duration unit'),
        ({'type': 'string', 'format': 'duration'}, {'unit': 'months'}, 'Use seconds for `timedelta` field'),
        ({'type': 'array', 'items': {'type': 'number'}}, {'integer': True}, 'Incompatible `ClairicField` controls'),
        ({'type': 'array', 'items': {'type': 'string'}}, {'decimals': 2}, 'Incompatible `ClairicField` controls'),
        (
            {'type': 'array', 'items': {'type': 'number'}},
            {'null_threshold': 0.5},
            'Incompatible `ClairicField` controls',
        ),
        (
            {'type': 'array', 'items': {'type': 'number', 'x-cloudflare-clairic': {'unit': 'USD'}}},
            {},
            'requires text, float or date list items',
        ),
        ({'type': 'number'}, {'min': 0, 'max': 100}, 'Incompatible `ClairicField` controls'),
        ({'type': 'number'}, {'horizon': 3}, 'Incompatible `ClairicField` controls'),
    ],
)
async def test_incompatible_controls_fail_before_http(
    schema: dict[str, JsonValue], controls: dict[str, JsonValue], error: str
):
    captured = mock_model({})
    with pytest.raises(UserError, match=error):
        await captured.model.request(
            [ModelRequest.user_text_prompt('state')],
            None,
            params({'type': 'object', 'properties': {'x': {**schema, 'x-cloudflare-clairic': controls}}}),
        )
    assert captured.requests == []


@pytest.mark.parametrize(
    ('result_type', 'controls', 'error'),
    [
        ('number-estimate', {}, 'requires finite numeric `min` less than `max`'),
        ('number-estimate', {'min': 10, 'max': 0}, 'requires finite numeric `min` less than `max`'),
        ('number-estimate', {'min': 0, 'max': 0}, 'requires finite numeric `min` less than `max`'),
        ('number-estimate', {'min': 0, 'max': 10, 'step': 0}, 'Invalid `ClairicField.step`'),
        ('number-estimate', {'min': 0, 'max': 10, 'step': 'month'}, 'Invalid `ClairicField.step`'),
        ('number-estimate', {'min': 0, 'max': 10, 'scale': 'log'}, 'Log scale requires positive numeric bounds'),
        ('number-estimate', {'min': 0, 'max': 10, 'scale': 'invalid'}, 'Invalid `ClairicField.scale`'),
        ('number-estimate', {'min': False, 'max': 10}, 'requires finite numeric `min` less than `max`'),
        ('number-estimate', {'min': 0, 'max': True}, 'requires finite numeric `min` less than `max`'),
        ('number-estimate', {'min': 0, 'max': '10'}, 'requires finite numeric `min` less than `max`'),
        ('date-estimate', {'min': 0, 'max': 86400}, 'requires date `min` and `max`'),
        ('date-estimate', {'min': 'bad', 'max': '2027-01-01'}, 'requires date `min` and `max`'),
        ('date-estimate', {'min': '2027-01-01', 'max': '2026-01-01'}, '`ClairicField.min` must be less than `max`'),
        (
            'date-estimate',
            {'min': '2026-01-01', 'max': '2027-01-01', 'scale': 'log'},
            'Log scale requires positive numeric bounds',
        ),
        ('forecast', {'min': 0, 'max': 10}, 'requires an integer `horizon` from one to 64'),
        (
            'forecast',
            {'min': 0, 'max': 10, 'horizon': 0, 'interval': 'week'},
            'requires an integer `horizon` from one to 64',
        ),
        (
            'forecast',
            {'min': 0, 'max': 10, 'horizon': 65, 'interval': 'week'},
            'requires an integer `horizon` from one to 64',
        ),
        (
            'forecast',
            {'min': 0, 'max': 10, 'horizon': True, 'interval': 'week'},
            'requires an integer `horizon` from one to 64',
        ),
        ('forecast', {'min': 0, 'max': 10, 'horizon': 3, 'interval': 'invalid'}, 'requires a valid `interval`'),
        ('forecast', {'min': 0, 'max': 10, 'horizon': 3}, 'requires a valid `interval`'),
    ],
)
async def test_invalid_prediction_controls(result_type: str, controls: dict[str, JsonValue], error: str):
    captured = mock_model({})
    schema: dict[str, JsonValue] = {
        'type': 'object',
        'x-cloudflare-clairic-result': result_type,
        'x-cloudflare-clairic': controls,
    }
    with pytest.raises(UserError, match=error):
        await captured.model.request(
            [ModelRequest.user_text_prompt('state')], None, params({'type': 'object', 'properties': {'x': schema}})
        )
    assert captured.requests == []


async def test_duplicate_field_markers():
    class Invalid(BaseModel):
        value: Annotated[float, ClairicField(unit='USD'), ClairicField(decimals=2)]

    captured = mock_model({})
    with pytest.raises(UserError, match='one `ClairicField`'):
        await Agent(captured.model, output_type=Invalid).run('state')
    assert captured.requests == []


async def test_request_settings_override_model_defaults():
    class Output(BaseModel):
        x: str

    captured = mock_model({'x': {'type': 'text', 'value': 'state'}})
    provider = captured.model.provider
    assert isinstance(provider, CloudflareProvider)
    defaults = CloudflareClairicModelSettings(
        cloudflare_clairic_reference_date=date(2026, 1, 1),
        cloudflare_clairic_include_quantiles=True,
        cloudflare_clairic_return_state_text=True,
    )
    captured.model = CloudflareClairicModel(provider=provider, settings=defaults)
    overrides = CloudflareClairicModelSettings(
        cloudflare_clairic_reference_date='2026-10-19',
        cloudflare_clairic_include_quantiles=False,
        cloudflare_clairic_return_state_text=False,
    )
    await Agent(captured.model, output_type=Output).run('state', model_settings=overrides)
    body = json.loads(captured.requests[0].content)
    assert body['reference_date'] == '2026-10-19'
    assert body['options'] == {'include_quantiles': False, 'return_state_text': False}


@pytest.mark.parametrize('extension', [None, 'wrong', []])
async def test_invalid_field_extension(extension: JsonValue):
    captured = mock_model({})
    schema: dict[str, JsonValue] = {'type': 'string', 'x-cloudflare-clairic': extension}
    with pytest.raises(UserError, match='ClairicField'):
        await captured.model.request(
            [ModelRequest.user_text_prompt('state')],
            None,
            params({'type': 'object', 'properties': {'x': schema}}),
        )
    assert captured.requests == []


async def test_duplicate_markers_across_nullable_boundary():
    class Invalid(BaseModel):
        value: Annotated[Annotated[float, ClairicField(unit='USD')] | None, ClairicField(decimals=2)]

    captured = mock_model({})
    with pytest.raises(UserError, match='one `ClairicField`'):
        await Agent(captured.model, output_type=Invalid).run('state')
    assert captured.requests == []


async def test_nullable_list_items_are_rejected():
    class Invalid(BaseModel):
        values: list[str | None]

    captured = mock_model({})
    with pytest.raises(UserError, match='list items'):
        await Agent(captured.model, output_type=Invalid).run('state')
    assert captured.requests == []
