from __future__ import annotations

from decimal import Decimal

import pytest

from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from tests.conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='anthropic not installed')


@pytest.mark.vcr('test_anthropic_batch_roundtrip.yaml')
@pytest.mark.parametrize('check_status', [False, True])
async def test_anthropic_batch_roundtrip(
    allow_model_requests: None, monkeypatch: pytest.MonkeyPatch, check_status: bool
) -> None:
    model = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(api_key='test'))
    provider = model
    assert provider is not None
    parameters = ModelRequestParameters()
    settings: ModelSettings = {'max_tokens': 8}
    requests = [
        await provider.batch_encode(
            [ModelRequest(parts=[UserPromptPart('Reply with OK.')])],
            settings,
            parameters,
            custom_id=custom_id,
        )
        for custom_id in ('pydantic-ai-batch-one', 'pydantic-ai-batch-two')
    ]

    batch_id = await provider.batch_create(requests)
    if check_status:
        assert await provider.batch_status(batch_id) == 'in_progress'
    else:
        assert await provider.batch_results(batch_id) is None
    results = await provider.batch_results(batch_id)
    assert results is not None
    assert [item.custom_id for item in results] == ['pydantic-ai-batch-two', 'pydantic-ai-batch-one']
    assert all(item.status == 'succeeded' for item in results)

    for item in results:
        response = provider.batch_decode(item, settings, parameters)
        assert response.text == 'OK.'
        assert response.state == 'complete'
        assert response.provider_response_id is not None
        assert response.usage.input_tokens == 11
        assert response.usage.output_tokens == 5
        assert response.usage.cost == Decimal('0.000018')
        assert response.cost().total_price == Decimal('0.000036')

    def unavailable_price(*args: object, **kwargs: object) -> None:
        raise ValueError('Price data unavailable')

    monkeypatch.setattr('pydantic_ai._genai_prices.calc_price', unavailable_price)
    response = provider.batch_decode(results[0], settings, parameters)
    assert response.text == 'OK.' and response.usage.cost is None


@pytest.mark.vcr
async def test_anthropic_batch_item_error(allow_model_requests: None) -> None:
    model = AnthropicModel('not-a-real-anthropic-model', provider=AnthropicProvider(api_key='test'))
    provider = model
    assert provider is not None
    settings: ModelSettings = {'max_tokens': 8}
    parameters = ModelRequestParameters()
    request = await provider.batch_encode(
        [ModelRequest(parts=[UserPromptPart('Reply with OK.')])], settings, parameters, custom_id='batch-error'
    )
    batch_id = await provider.batch_create([request])
    results = await provider.batch_results(batch_id)
    assert results is not None and len(results) == 1
    assert results[0].status == 'errored'
    with pytest.raises(ModelHTTPError) as exc_info:
        provider.batch_decode(results[0], settings, parameters)
    assert exc_info.value.status_code == 404
    body = exc_info.value.body
    assert isinstance(body, dict) and isinstance(body['error'], dict)
    assert body['error']['type'] == 'not_found_error'


@pytest.mark.vcr
async def test_anthropic_batch_cancellation(allow_model_requests: None) -> None:
    model = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(api_key='test'))
    provider = model
    assert provider is not None
    settings: ModelSettings = {'max_tokens': 8}
    parameters = ModelRequestParameters()
    request = await provider.batch_encode(
        [ModelRequest(parts=[UserPromptPart('Reply with OK.')])], settings, parameters, custom_id='batch-cancel'
    )
    batch_id = await provider.batch_create([request])
    await provider.batch_cancel(batch_id)
    results = await provider.batch_results(batch_id)
    assert results is not None and len(results) == 1
    assert results[0].status == 'canceled'
    with pytest.raises(ModelAPIError, match='canceled before the model ran it'):
        provider.batch_decode(results[0], settings, parameters)
