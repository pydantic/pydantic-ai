"""Gemini API Batch Mode against the live API: submission, status, results, cancellation, and `BatchModelRequests` runs."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from cassetter import Cassette
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.capabilities import BatchModelRequests
from pydantic_ai.direct import model_request_batch
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart
from pydantic_ai.models.batch import Batch, BatchRequest
from pydantic_ai.settings import ModelSettings
from tests.batch_vcr_utils import batch_poll_interval, wait_for_batch
from tests.conftest import IsStr, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
    from pydantic_ai.providers.google import GoogleProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='google-genai not installed')

MODEL_NAME = 'gemini-2.5-flash-lite'


def _request(custom_id: str, settings: ModelSettings | None = None) -> BatchRequest:
    return BatchRequest(
        custom_id=custom_id,
        messages=[ModelRequest(parts=[UserPromptPart('Reply with just OK.')])],
        model_settings=settings or {'max_tokens': 8},
    )


@pytest.mark.vcr
async def test_gemini_batch_roundtrip(allow_model_requests: None, gemini_api_key: str, vcr: Cassette) -> None:
    model = GoogleModel(MODEL_NAME, provider=GoogleProvider(api_key=gemini_api_key))
    requests = [_request('pydantic-ai-batch-one'), _request('pydantic-ai-batch-two')]

    batch = await model.batch_create([await model.batch_encode(request) for request in requests])
    assert batch.status == 'in_progress'
    assert batch.request_count == 2
    assert batch.id.startswith('batches/')
    with pytest.raises(Exception, match='has not ended yet'):
        await model.batch_results(batch)

    batch = await wait_for_batch(model, batch, vcr)
    assert batch.provider_details is not None and batch.provider_details['state'] == 'JOB_STATE_SUCCEEDED'
    results = {result.custom_id: result for result in await model.batch_results(batch)}
    assert set(results) == {'pydantic-ai-batch-one', 'pydantic-ai-batch-two'}

    for request in requests:
        response = model.batch_decode(results[request.custom_id], request)
        assert response.text == 'OK'
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
async def test_gemini_batch_item_error(allow_model_requests: None, gemini_api_key: str, vcr: Cassette) -> None:
    """A request that fails on its own is an errored result, raising the error `generateContent` would have."""
    model = GoogleModel(MODEL_NAME, provider=GoogleProvider(api_key=gemini_api_key))
    request = _request('batch-error', GoogleModelSettings(google_cached_content='cachedContents/does-not-exist'))
    batch = await wait_for_batch(model, await model.batch_create([await model.batch_encode(request)]), vcr)
    [result] = await model.batch_results(batch)
    assert result.status == 'errored'
    with pytest.raises(ModelHTTPError) as exc_info:
        model.batch_decode(result, request)
    assert exc_info.value.status_code == 403
    body = exc_info.value.body
    assert isinstance(body, dict) and isinstance(body['error'], dict)


@pytest.mark.vcr
async def test_gemini_batch_cancellation(allow_model_requests: None, gemini_api_key: str, vcr: Cassette) -> None:
    model = GoogleModel(MODEL_NAME, provider=GoogleProvider(api_key=gemini_api_key))
    request = _request('batch-cancel')
    batch = await model.batch_create([await model.batch_encode(request)])
    batch = await model.batch_cancel(batch)
    assert batch.status in ('canceling', 'ended')
    batch = await wait_for_batch(model, batch, vcr)
    assert batch.provider_details is not None and batch.provider_details['state'] == 'JOB_STATE_CANCELLED'
    assert await model.batch_results(batch) == []


class CityInfo(BaseModel):
    city: str
    country: str


@pytest.mark.vcr
async def test_gemini_batch_model_agent_run(allow_model_requests: None, gemini_api_key: str, vcr: Cassette) -> None:
    """An agent with structured output runs through the batch API: its request waits in the run's message history.

    Concurrent runs sharing a batch is covered with the in-memory batch API in `test_batch.py`: how concurrent checks
    of one batch overlap depends on network latency, which a recording can't replay.
    """
    batching = BatchModelRequests(
        max_requests=1, poll_interval=batch_poll_interval(vcr), max_poll_interval=batch_poll_interval(vcr)
    )
    agent = Agent(
        GoogleModel(MODEL_NAME, provider=GoogleProvider(api_key=gemini_api_key)),
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
async def test_gemini_model_request_batch(allow_model_requests: None, gemini_api_key: str, vcr: Cassette) -> None:
    model = GoogleModel(MODEL_NAME, provider=GoogleProvider(api_key=gemini_api_key))
    responses = await model_request_batch(
        model,
        [_request('one'), _request('two', {'max_tokens': 8, 'stop_sequences': ['OK']})],
        poll_interval=batch_poll_interval(vcr),
        max_poll_interval=batch_poll_interval(vcr),
    )
    assert [response.text if isinstance(response, ModelResponse) else response for response in responses] == [
        'OK',
        None,
    ]


@pytest.mark.vcr
async def test_gemini_batch_api_error(allow_model_requests: None, gemini_api_key: str) -> None:
    """An API error on a batch call raises the same `ModelHTTPError` as one on `generateContent`."""
    model = GoogleModel(MODEL_NAME, provider=GoogleProvider(api_key=gemini_api_key))
    batch = Batch(
        id='batches/does-not-exist',
        model_name=MODEL_NAME,
        provider_name=model.system,
        group=(await model.batch_encode(_request('missing'))).group,
        status='in_progress',
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        request_count=1,
    )
    with pytest.raises(ModelHTTPError) as exc_info:
        await model.batch_status(batch)
    assert exc_info.value.status_code == 400
    assert exc_info.value.body == {
        'error': {'code': 400, 'message': 'Could not parse the batch name', 'status': 'INVALID_ARGUMENT'}
    }
