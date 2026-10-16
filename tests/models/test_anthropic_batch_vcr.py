from __future__ import annotations

import pytest
from cassetter import Cassette
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.direct import model_request_batch
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart
from pydantic_ai.models import Model
from pydantic_ai.models.batch import BatchModel, BatchRequest
from pydantic_ai.models.instrumented import InstrumentedModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from tests.batch_vcr_utils import batch_poll_interval, wait_for_batch
from tests.conftest import IsStr, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel
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
    with pytest.raises(ModelAPIError, match='canceled before the model ran it'):
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
    model = BatchModel(
        AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(api_key=anthropic_api_key)),
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
