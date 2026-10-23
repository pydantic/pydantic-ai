from __future__ import annotations

import json

import pytest
from cassetter import Cassette
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.capabilities import BatchModelRequests
from pydantic_ai.direct import model_request_batch
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, ModelServerError
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart
from pydantic_ai.models import Model
from pydantic_ai.models.batch import BatchRequest
from pydantic_ai.models.instrumented import InstrumentedModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from tests.batch_vcr_utils import batch_poll_interval, wait_for_batch
from tests.conftest import IsStr, RequestCapture, try_import
from tests.models.conftest import AnthropicModelFactory

with try_import() as imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='anthropic not installed')


def _request(custom_id: str, settings: ModelSettings | None = None) -> BatchRequest:
    return BatchRequest(
        custom_id=custom_id,
        messages=[ModelRequest(parts=[UserPromptPart('Reply with OK.')])],
        model_settings=settings or {'max_tokens': 8},
    )


@pytest.mark.vcr
@pytest.mark.parametrize('wrapper', [None, WrapperModel, InstrumentedModel])
async def test_anthropic_batch_roundtrip(
    allow_model_requests: None, anthropic_api_key: str, vcr: Cassette, wrapper: type[WrapperModel] | None
) -> None:
    model: Model = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    if wrapper is not None:
        model = wrapper(model)
    requests = [_request('pydantic-ai-batch-one'), _request('pydantic-ai-batch-two')]

    batch = await model.batch_create([await model.batch_encode(request) for request in requests])
    assert batch.status == 'in_progress'
    assert batch.request_count == 2
    with pytest.raises(Exception, match='has not ended yet'):
        await model.batch_results(batch)

    batch = await wait_for_batch(model, batch, vcr)
    results = {result.custom_id: result for result in await model.batch_results(batch)}
    assert set(results) == {'pydantic-ai-batch-one', 'pydantic-ai-batch-two'}

    for request in requests:
        response = model.batch_decode(results[request.custom_id], request)
        assert response.text == 'OK.'
        assert response.state == 'complete'
        assert response.provider_details is not None and response.provider_details['processing'] == 'batch'
        # Batch prices are half the standard ones.
        assert (
            response.cost().total_price * 2
            == ModelResponse(
                parts=response.parts,
                usage=response.usage,
                model_name=response.model_name,
                provider_name=response.provider_name,
                timestamp=response.timestamp,
            )
            .cost()
            .total_price
        )


@pytest.mark.vcr
async def test_anthropic_batch_item_error(allow_model_requests: None, anthropic_api_key: str, vcr: Cassette) -> None:
    model = AnthropicModel('not-a-real-anthropic-model', provider=AnthropicProvider(api_key=anthropic_api_key))
    request = _request('batch-error')
    batch = await wait_for_batch(model, await model.batch_create([await model.batch_encode(request)]), vcr)
    [result] = await model.batch_results(batch)
    assert result.status == 'errored'
    with pytest.raises(ModelHTTPError) as exc_info:
        model.batch_decode(result, request)
    assert exc_info.value.status_code == 404
    body = exc_info.value.body
    assert isinstance(body, dict) and isinstance(body['error'], dict)
    assert body['error']['type'] == 'not_found_error'


@pytest.mark.vcr
async def test_anthropic_batch_cancellation(allow_model_requests: None, anthropic_api_key: str, vcr: Cassette) -> None:
    model = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    request = _request('batch-cancel')
    batch = await model.batch_create([await model.batch_encode(request)])
    batch = await model.batch_cancel(batch)
    assert batch.status == 'canceling'
    [result] = await model.batch_results(await wait_for_batch(model, batch, vcr))
    assert result.status == 'canceled'
    with pytest.raises(ModelAPIError, match='canceled before the model ran the request'):
        model.batch_decode(result, request)


class CityInfo(BaseModel):
    city: str
    country: str


@pytest.mark.vcr
async def test_anthropic_batch_model_agent_run(
    allow_model_requests: None, anthropic_api_key: str, vcr: Cassette
) -> None:
    """An agent with structured output runs through the batch API: its request waits in the run's message history.

    Concurrent runs sharing a batch is covered with the in-memory batch API in `test_batch.py`: how concurrent checks
    of one batch overlap depends on network latency, which a recording can't replay.
    """
    batching = BatchModelRequests(
        max_requests=1, poll_interval=batch_poll_interval(vcr), max_poll_interval=batch_poll_interval(vcr)
    )
    agent = Agent(
        AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(api_key=anthropic_api_key)),
        output_type=CityInfo,
        capabilities=[batching],
    )
    result = await agent.run('Which city is the Eiffel Tower in?')

    assert result.output == CityInfo(city='Paris', country='France')
    assert result.usage.requests == 1
    response = result.all_messages()[1]
    assert isinstance(response, ModelResponse)
    assert response.state == 'complete'
    assert response.provider_details is not None and response.provider_details['processing'] == 'batch'
    assert response.provider_details['batch'] == {'id': IsStr(), 'custom_id': IsStr()}


@pytest.mark.vcr
async def test_anthropic_model_request_batch(allow_model_requests: None, anthropic_api_key: str, vcr: Cassette) -> None:
    model = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    responses = await model_request_batch(
        model,
        [_request('one'), _request('two', {'max_tokens': 8, 'stop_sequences': ['OK']})],
        poll_interval=batch_poll_interval(vcr),
        max_poll_interval=batch_poll_interval(vcr),
    )
    # The second request stops at its stop sequence, before generating any text.
    assert [response.text if isinstance(response, ModelResponse) else response for response in responses] == [
        'OK.',
        None,
    ]


LONG_INSTRUCTIONS = ' '.join(f'Fact {i}: {i} squared is {i * i}.' for i in range(600))
"""Instructions longer than the 4,096 tokens Claude Haiku 4.5 caches at least."""


def _batch_priced(response: ModelResponse) -> bool:
    """Whether `response` costs half what the same usage costs outside a batch."""
    standard = ModelResponse(
        parts=response.parts,
        usage=response.usage,
        model_name=response.model_name,
        provider_name=response.provider_name,
        timestamp=response.timestamp,
    )
    return response.cost().total_price * 2 == standard.cost().total_price


@pytest.mark.vcr
async def test_anthropic_batch_mixes_successes_and_errors(
    allow_model_requests: None, anthropic_api_key: str, vcr: Cassette
) -> None:
    """Requests of one batch fail on their own, each with the error it would have raised outside a batch."""
    provider = AnthropicProvider(api_key=anthropic_api_key)
    model = AnthropicModel('claude-haiku-4-5', provider=provider)
    unknown = AnthropicModel('not-a-real-anthropic-model', provider=provider)
    ok, too_long = _request('batch-ok'), _request('batch-too-many-tokens', {'max_tokens': 10_000_000})
    missing = _request('batch-unknown-model')
    encoded = [await model.batch_encode(ok), await model.batch_encode(too_long), await unknown.batch_encode(missing)]
    batch = await wait_for_batch(model, await model.batch_create(encoded), vcr)
    results = {result.custom_id: result for result in await model.batch_results(batch)}
    assert {custom_id: result.status for custom_id, result in results.items()} == {
        'batch-ok': 'succeeded',
        'batch-too-many-tokens': 'errored',
        'batch-unknown-model': 'errored',
    }
    assert model.batch_decode(results['batch-ok'], ok).text == 'OK.'
    with pytest.raises(ModelHTTPError) as exc_info:
        model.batch_decode(results['batch-too-many-tokens'], too_long)
    assert exc_info.value.status_code == 400
    assert not isinstance(exc_info.value, ModelServerError)
    assert exc_info.value.provider_error_type == 'invalid_request_error'
    with pytest.raises(ModelHTTPError) as exc_info:
        unknown.batch_decode(results['batch-unknown-model'], missing)
    assert (exc_info.value.status_code, exc_info.value.provider_error_type) == (404, 'not_found_error')


@pytest.mark.vcr
async def test_anthropic_batches_split_by_beta_flags(
    allow_model_requests: None, anthropic_model: AnthropicModelFactory, request_capture: RequestCapture, vcr: Cassette
) -> None:
    """Beta flags apply to a whole batch, so requests with different ones are submitted as separate batches."""
    model = anthropic_model('claude-haiku-4-5', capture=True)
    responses = await model_request_batch(
        model,
        [
            _request('no-betas'),
            _request(
                'with-betas', AnthropicModelSettings(max_tokens=8, anthropic_betas=['interleaved-thinking-2025-05-14'])
            ),
        ],
        poll_interval=batch_poll_interval(vcr),
        max_poll_interval=batch_poll_interval(vcr),
    )
    assert [response.text if isinstance(response, ModelResponse) else response for response in responses] == [
        'OK.',
        'OK.',
    ]
    creates = [
        ([item['custom_id'] for item in json.loads(raw)['requests']], headers.get('anthropic-beta'))
        for path, raw, headers in zip(request_capture.paths, request_capture.raw_bodies, request_capture.headers)
        if path == '/v1/messages/batches' and raw
    ]
    assert creates == snapshot(
        [
            (['no-betas'], 'message-batches-2024-09-24'),
            (['with-betas'], 'interleaved-thinking-2025-05-14,message-batches-2024-09-24'),
        ]
    )


@pytest.mark.vcr
async def test_anthropic_batch_usage_has_cache_and_thinking_tokens(
    allow_model_requests: None, anthropic_api_key: str, vcr: Cassette
) -> None:
    """A batched response reports cache and thinking tokens as the request would on its own, priced at batch rates.

    The first batch writes the long instructions to the cache, with a one-hour lifetime so they outlast a batch's
    queueing, and the second reads them.
    """
    model = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    responses: list[ModelResponse] = []
    for custom_id in ('usage-first', 'usage-second'):
        request = BatchRequest(
            custom_id=custom_id,
            messages=[
                ModelRequest(
                    parts=[UserPromptPart('What is 17 squared? Reply with the number.')],
                    instructions=LONG_INSTRUCTIONS,
                )
            ],
            model_settings=AnthropicModelSettings(
                max_tokens=2048,
                anthropic_thinking={'type': 'enabled', 'budget_tokens': 1024},
                anthropic_cache_instructions='1h',
            ),
        )
        batch = await wait_for_batch(model, await model.batch_create([await model.batch_encode(request)]), vcr)
        [result] = await model.batch_results(batch)
        responses.append(model.batch_decode(result, request))

    first, second = responses
    assert first.usage.cache_write_tokens > 0
    assert second.usage.cache_read_tokens == first.usage.cache_write_tokens
    assert all(response.usage.output_reasoning_tokens > 0 for response in responses)
    assert all(_batch_priced(response) for response in responses)
