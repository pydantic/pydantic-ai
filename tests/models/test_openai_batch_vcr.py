from __future__ import annotations

import pytest
from cassetter import Cassette
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart
from pydantic_ai.models.batch import BatchModel, BatchRequest
from pydantic_ai.settings import ModelSettings
from tests.batch_vcr_utils import batch_poll_interval, wait_for_batch
from tests.conftest import IsStr, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def _request(custom_id: str, settings: ModelSettings | None = None) -> BatchRequest:
    return BatchRequest(
        custom_id=custom_id,
        messages=[ModelRequest(parts=[UserPromptPart('Reply with OK.')])],
        model_settings=settings or {'max_tokens': 16},
    )


@pytest.mark.vcr
async def test_openai_batch_roundtrip(allow_model_requests: None, openai_api_key: str, vcr: Cassette) -> None:
    model = OpenAIResponsesModel('gpt-4.1-nano', provider=OpenAIProvider(api_key=openai_api_key))
    requests = [_request('pydantic-ai-batch-one'), _request('pydantic-ai-batch-two')]

    batch = await model.batch_create([await model.batch_encode(request) for request in requests])
    assert batch.status == 'in_progress'
    assert batch.request_count == 2
    assert batch.provider_details is not None and batch.provider_details['status'] == 'validating'
    with pytest.raises(Exception, match='has not ended yet'):
        await model.batch_results(batch)

    batch = await wait_for_batch(model, batch, vcr)
    assert batch.provider_details is not None and batch.provider_details['status'] == 'completed'
    results = {result.custom_id: result for result in await model.batch_results(batch)}
    assert set(results) == {'pydantic-ai-batch-one', 'pydantic-ai-batch-two'}

    for request in requests:
        response = model.batch_decode(results[request.custom_id], request)
        assert response.text == 'OK'
        assert response.state == 'complete'
        assert response.provider_response_id is not None
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
async def test_openai_batch_item_error(allow_model_requests: None, openai_api_key: str, vcr: Cassette) -> None:
    """A request the API rejects fails on its own, with the error the request would have raised."""
    model = OpenAIResponsesModel('gpt-4.1-nano', provider=OpenAIProvider(api_key=openai_api_key))
    ok, invalid = _request('batch-ok'), _request('batch-error', {'max_tokens': 1})
    batch = await model.batch_create([await model.batch_encode(ok), await model.batch_encode(invalid)])
    results = {
        result.custom_id: result for result in await model.batch_results(await wait_for_batch(model, batch, vcr))
    }
    assert {custom_id: result.status for custom_id, result in results.items()} == {
        'batch-ok': 'succeeded',
        'batch-error': 'errored',
    }
    assert model.batch_decode(results['batch-ok'], ok).text == 'OK'
    with pytest.raises(ModelHTTPError) as exc_info:
        model.batch_decode(results['batch-error'], invalid)
    assert exc_info.value.status_code == 400
    assert exc_info.value.body == snapshot(
        {
            'error': {
                'message': "Invalid 'max_output_tokens': integer below minimum value. Expected a value >= 16, but got 1 instead.",
                'type': 'invalid_request_error',
                'param': 'max_output_tokens',
                'code': 'integer_below_min_value',
            }
        }
    )


@pytest.mark.vcr
async def test_openai_batch_unknown_model(allow_model_requests: None, openai_api_key: str, vcr: Cassette) -> None:
    """A batch whose input file fails validation runs none of its requests, and each raises the batch's errors."""
    model = OpenAIResponsesModel('not-a-real-openai-model', provider=OpenAIProvider(api_key=openai_api_key))
    request = _request('batch-error')
    batch = await wait_for_batch(model, await model.batch_create([await model.batch_encode(request)]), vcr)
    assert batch.provider_details is not None and batch.provider_details['status'] == 'failed'
    [result] = await model.batch_results(batch)
    assert result.status == 'errored'
    with pytest.raises(ModelHTTPError) as exc_info:
        model.batch_decode(result, request)
    assert exc_info.value.status_code == 400
    assert exc_info.value.body == snapshot(
        {
            'errors': [
                {
                    'code': 'model_not_found',
                    'line': 1,
                    'message': "The provided model 'not-a-real-openai-model' is not supported by the Batch API.",
                    'param': 'body.model',
                }
            ]
        }
    )


@pytest.mark.vcr
async def test_openai_batch_cancellation(allow_model_requests: None, openai_api_key: str, vcr: Cassette) -> None:
    model = OpenAIResponsesModel('gpt-4.1-nano', provider=OpenAIProvider(api_key=openai_api_key))
    request = _request('batch-cancel')
    batch = await model.batch_create([await model.batch_encode(request)])
    batch = await model.batch_cancel(batch)
    assert batch.status == 'canceling'
    [result] = await model.batch_results(await wait_for_batch(model, batch, vcr))
    assert result.status == 'canceled'
    with pytest.raises(ModelAPIError, match='canceled before the model ran it'):
        model.batch_decode(result, request)


class CityInfo(BaseModel):
    city: str
    country: str


@pytest.mark.vcr
async def test_openai_batch_model_agent_run(allow_model_requests: None, openai_api_key: str, vcr: Cassette) -> None:
    """An agent with structured output runs through the batch API: its request waits in the run's message history.

    Concurrent runs sharing a batch is covered with the in-memory batch API in `test_batch.py`: how concurrent checks
    of one batch overlap depends on network latency, which a recording can't replay.
    """
    model = BatchModel(
        OpenAIResponsesModel('gpt-4.1-nano', provider=OpenAIProvider(api_key=openai_api_key)),
        max_requests=1,
        poll_interval=batch_poll_interval(vcr),
        max_poll_interval=batch_poll_interval(vcr),
    )
    result = await Agent(model, output_type=CityInfo).run('Which city is the Eiffel Tower in?')

    assert result.output == CityInfo(city='Paris', country='France')
    assert result.usage.requests == 1
    response = result.all_messages()[1]
    assert isinstance(response, ModelResponse)
    assert response.state == 'complete'
    assert response.provider_details is not None and response.provider_details['processing'] == 'batch'
    assert response.provider_details['batch'] == {'id': IsStr(), 'custom_id': IsStr()}
