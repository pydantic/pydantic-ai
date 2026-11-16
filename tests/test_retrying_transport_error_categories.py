"""A status a retrying transport gave up on raises the error category the provider's recorded error body names.

The provider SDK never sees such a response: the transport's `validate_response` raises `HTTPStatusError` for it, so
each model classifies the body itself, as its SDK error mapping does. `tests/models/test_error_categories.py` checks
that both paths agree on fixed wire shapes; these cases record the real error responses the providers send for
requests they reject before running a model: an invalid API key and an unknown model.
"""

from __future__ import annotations as _annotations

from collections.abc import Callable
from dataclasses import dataclass

import httpx2
import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, ModelHTTPError
from pydantic_ai.models import Model

from .conftest import try_import

with try_import() as imports_successful:
    from anthropic import APIConnectionError as AnthropicAPIConnectionError, AsyncAnthropic
    from openai import AsyncOpenAI
    from tenacity import stop_after_attempt

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.mistral import MistralProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.retries import AsyncHTTPX2TenacityTransport, RetryConfig

pytestmark = [
    pytest.mark.vcr,
    pytest.mark.skipif(not imports_successful(), reason='provider SDKs not installed'),
]

ModelFactory = Callable[[httpx2.AsyncClient, str], Model]
"""Builds the model under test from the retrying client and the provider's API key."""


def _retrying_client() -> httpx2.AsyncClient:
    """A client whose retrying transport rejects every error status, giving up after one attempt.

    It's mounted rather than passed as `transport`, so the cassette replays beneath it, where the network answers.
    """
    transport = AsyncHTTPX2TenacityTransport(
        RetryConfig(stop=stop_after_attempt(1), reraise=True),
        wrapped=httpx2.AsyncHTTPTransport(),
        validate_response=lambda response: response.raise_for_status(),
    )
    return httpx2.AsyncClient(mounts={'all://': transport})


def _openai(model_name: str) -> ModelFactory:
    def factory(http_client: httpx2.AsyncClient, api_key: str) -> Model:
        client = AsyncOpenAI(api_key=api_key, max_retries=0, http_client=http_client)
        return OpenAIChatModel(model_name, provider=OpenAIProvider(openai_client=client))

    return factory


def _google(model_name: str) -> ModelFactory:
    def factory(http_client: httpx2.AsyncClient, api_key: str) -> Model:
        return GoogleModel(model_name, provider=GoogleProvider(api_key=api_key, http_client=http_client))

    return factory


def _mistral(model_name: str) -> ModelFactory:
    def factory(http_client: httpx2.AsyncClient, api_key: str) -> Model:
        return MistralModel(model_name, provider=MistralProvider(api_key=api_key, http_client=http_client))

    return factory


def _anthropic(model_name: str) -> ModelFactory:
    def factory(http_client: httpx2.AsyncClient, api_key: str) -> Model:
        client = AsyncAnthropic(api_key=api_key, max_retries=0, http_client=http_client)
        return AnthropicModel(model_name, provider=AnthropicProvider(anthropic_client=client))

    return factory


@dataclass(frozen=True)
class Case:
    id: str
    model: ModelFactory
    api_key: str | None
    """The key to send, or `None` for the provider's real key, which the request needs to reach the model lookup."""
    key_fixture: str
    expected: dict[str, object]
    """The error's category and the provider error fields it carries."""


CASES = [
    Case(
        id='openai-invalid-api-key',
        model=_openai('gpt-5.2'),
        api_key='sk-invalid-key',
        key_fixture='openai_api_key',
        expected=snapshot(
            {
                'category': 'ModelAuthenticationError',
                'status_code': 401,
                'provider_error_code': 'invalid_api_key',
                'provider_error_type': 'invalid_request_error',
                'suggested_model_id': None,
            }
        ),
    ),
    Case(
        id='openai-model-not-found',
        model=_openai('gpt-4o-mni'),
        api_key=None,
        key_fixture='openai_api_key',
        expected=snapshot(
            {
                'category': 'ModelNotFoundError',
                'status_code': 404,
                'provider_error_code': 'model_not_found',
                'provider_error_type': 'invalid_request_error',
                'suggested_model_id': None,
            }
        ),
    ),
    Case(
        # Gemini answers an invalid key with a 400 `INVALID_ARGUMENT`; only its `API_KEY_INVALID` reason says why.
        id='google-api-key-invalid',
        model=_google('gemini-2.5-flash'),
        api_key='AIzaInvalidKey000000000000000000000000',
        key_fixture='gemini_api_key',
        expected=snapshot(
            {
                'category': 'ModelAuthenticationError',
                'status_code': 400,
                'provider_error_code': 'INVALID_ARGUMENT',
                'provider_error_type': None,
                'suggested_model_id': None,
            }
        ),
    ),
    Case(
        id='google-model-not-found',
        model=_google('gemini-2.5-flsh'),
        api_key=None,
        key_fixture='gemini_api_key',
        expected=snapshot(
            {
                'category': 'ModelNotFoundError',
                'status_code': 404,
                'provider_error_code': 'NOT_FOUND',
                'provider_error_type': None,
                'suggested_model_id': 'google:gemini-3.8-flash',
            }
        ),
    ),
    Case(
        id='mistral-invalid-model',
        model=_mistral('mistral-smal-latest'),
        api_key=None,
        key_fixture='mistral_api_key',
        expected=snapshot(
            {
                'category': 'ModelRequestRejectedError',
                'status_code': 400,
                'provider_error_code': '1500',
                'provider_error_type': 'invalid_model',
                'suggested_model_id': None,
            }
        ),
    ),
    Case(
        id='anthropic-authentication',
        model=_anthropic('claude-sonnet-4-5'),
        api_key='sk-ant-invalid-key',
        key_fixture='anthropic_api_key',
        expected=snapshot(
            {
                'category': 'ModelAuthenticationError',
                'status_code': 401,
                'provider_error_code': None,
                'provider_error_type': 'authentication_error',
                'suggested_model_id': None,
            }
        ),
    ),
]


@pytest.mark.parametrize('case', CASES, ids=[case.id for case in CASES])
async def test_retrying_transport_gave_up_recorded_error_category(
    allow_model_requests: None, case: Case, request: pytest.FixtureRequest
):
    api_key = case.api_key if case.api_key is not None else request.getfixturevalue(case.key_fixture)
    async with _retrying_client() as http_client:
        with pytest.raises(ModelHTTPError) as exc_info:
            await Agent(case.model(http_client, api_key)).run('hello')

    exc = exc_info.value
    # Raised from the transport's `HTTPStatusError`, which `anthropic` wraps, not from an SDK status error.
    cause = exc.__cause__
    if isinstance(cause, AnthropicAPIConnectionError):
        cause = cause.__cause__
    assert isinstance(cause, httpx2.HTTPStatusError)
    assert {
        'category': type(exc).__name__,
        'status_code': exc.status_code,
        'provider_error_code': exc.provider_error_code,
        'provider_error_type': exc.provider_error_type,
        'suggested_model_id': exc.suggested_model_id,
    } == case.expected
