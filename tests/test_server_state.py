"""Which server-side state each model sends back, by the endpoint that created it.

Unit tests of `Model.accepts_server_state` and `Provider.shares_server_state_with`: one `provider_name` can cover
several endpoints (`OpenAIProvider` with any `base_url`), and a stored response, item, conversation or container ID
from one can't be looked up at another. The request each adapter renders is tested next to its other request tests,
in `tests/models/test_openai_responses.py`, `tests/models/test_anthropic.py` and
`tests/models/anthropic/test_cache_diagnostics.py`. https://github.com/pydantic/pydantic-ai/issues/11508
"""

from __future__ import annotations as _annotations

from typing import Any, cast

import httpx2
import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import ModelFallback
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.instrumented import InstrumentedModel
from pydantic_ai.models.test import TestModel

from .conftest import try_import

with try_import() as imports_successful:
    from openai import APIStatusError
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText

    from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
    from pydantic_ai.providers.openai import OpenAIProvider

    from .models.mock_openai import MockOpenAIResponses, get_mock_responses_kwargs, response_message

    class AliasedOpenAIProvider(OpenAIProvider):
        """A provider that knows another URL reaches the same stored state, like a proxy alias."""

        def shares_server_state_with(self, provider_url: str) -> bool:
            return provider_url == 'https://alias.example/v1' or super().shares_server_state_with(provider_url)


pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')

GATEWAY_URL = 'https://gateway-a.example/v1'


def response(provider_name: str | None = 'openai', provider_url: str | None = GATEWAY_URL) -> ModelResponse:
    return ModelResponse(parts=[TextPart('hi')], provider_name=provider_name, provider_url=provider_url)


def gateway_model(provider_cls: type[OpenAIProvider] | None = None) -> OpenAIResponsesModel:
    return OpenAIResponsesModel('gpt-5', provider=(provider_cls or OpenAIProvider)(api_key='k', base_url=GATEWAY_URL))


@pytest.mark.parametrize(
    ('stored', 'accepted'),
    [
        pytest.param(response(), True, id='same-endpoint'),
        pytest.param(response(provider_url='HTTPS://Gateway-A.example:443/v1/'), True, id='same-endpoint-spelling'),
        pytest.param(response(provider_url='https://gateway-b.example/v1'), False, id='other-host'),
        pytest.param(response(provider_url='https://gateway-a.example/v2'), False, id='other-path'),
        pytest.param(response(provider_url='https://gateway-a.example:8443/v1'), False, id='other-port'),
        pytest.param(response(provider_url='https://gateway-a.example:99999/v1'), False, id='invalid-port'),
        pytest.param(response(provider_url='https://[broken/v1'), False, id='malformed'),
        pytest.param(response(provider_url='gateway-a.example/v1'), False, id='no-scheme'),
        pytest.param(response(provider_url='https://gateway-a.example/v1?tenant=b'), False, id='other-query'),
        pytest.param(response(provider_url=None), True, id='no-url-legacy'),
        pytest.param(response(provider_name='anthropic'), False, id='other-provider'),
        pytest.param(response(provider_name=None, provider_url=None), False, id='no-provider'),
    ],
)
def test_accepts_server_state(stored: ModelResponse, accepted: bool):
    assert gateway_model().accepts_server_state(stored) is accepted


def test_shares_server_state_with_can_be_overridden():
    model = gateway_model(AliasedOpenAIProvider)
    assert model.accepts_server_state(response(provider_url='https://alias.example/v1'))
    assert model.accepts_server_state(response())
    assert not model.accepts_server_state(response(provider_url='https://gateway-b.example/v1'))


def test_model_without_provider_judges_by_name():
    model = TestModel()
    assert model.accepts_server_state(response(provider_name='test'))
    assert not model.accepts_server_state(response())


def test_wrapper_and_fallback_models_delegate():
    gateway = gateway_model()
    other_endpoint = response(provider_url='https://gateway-b.example/v1')
    assert InstrumentedModel(gateway).accepts_server_state(response())
    assert not InstrumentedModel(gateway).accepts_server_state(other_endpoint)
    assert FallbackModel(gateway, TestModel()).accepts_server_state(response())
    assert not FallbackModel(TestModel(), gateway).accepts_server_state(response())


def mock_endpoint_model(base_url: str, response: Any) -> tuple[OpenAIResponsesModel, MockOpenAIResponses]:
    mock = cast(MockOpenAIResponses, MockOpenAIResponses.create_mock(response))
    mock.base_url = base_url
    return OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(openai_client=cast(Any, mock))), mock


def text_response(text: str) -> Any:
    return response_message(
        [
            ResponseOutputMessage(
                id=f'msg_{text}',
                content=[ResponseOutputText(text=text, type='output_text', annotations=[])],
                role='assistant',
                status='completed',
                type='message',
            )
        ]
    )


@pytest.mark.parametrize('route', ['fallback-model', 'model-fallback-capability'])
async def test_fallback_to_another_endpoint_drops_server_state(allow_model_requests: None, route: str):
    """A fallback attempt at another endpoint with the same `provider_name` gets none of the first endpoint's IDs."""
    producer, _ = mock_endpoint_model(GATEWAY_URL, text_response('first'))
    history: list[ModelMessage] = (await Agent(producer).run('q')).all_messages()

    unavailable = APIStatusError(
        'unavailable',
        response=httpx2.Response(503, request=httpx2.Request('POST', f'{GATEWAY_URL}/responses')),
        body=None,
    )
    primary, primary_mock = mock_endpoint_model(GATEWAY_URL, unavailable)
    fallback, fallback_mock = mock_endpoint_model('https://gateway-b.example/v1', text_response('second'))
    agent = (
        Agent(FallbackModel(primary, fallback))
        if route == 'fallback-model'
        else Agent(primary, capabilities=[ModelFallback(fallback)])
    )
    settings = OpenAIResponsesModelSettings(openai_previous_response_id='auto', openai_send_reasoning_ids=True)
    result = await agent.run('again', message_history=history, model_settings=settings)

    assert result.output == 'second'
    [primary_kwargs] = get_mock_responses_kwargs(cast(Any, primary_mock))
    assert primary_kwargs['previous_response_id'] == '123'
    [fallback_kwargs] = get_mock_responses_kwargs(cast(Any, fallback_mock))
    assert 'previous_response_id' not in fallback_kwargs
    assert fallback_kwargs['input'] == [
        {'role': 'user', 'content': 'q'},
        {'role': 'assistant', 'content': 'first'},
        {'role': 'user', 'content': 'again'},
    ]
