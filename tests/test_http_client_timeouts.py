"""Connect and pool timeouts on the HTTP clients Pydantic AI creates.

These tests replace the transport's `handle_async_request` instead of replaying a cassette: the
per-phase timeouts travel in `request.extensions`, which a cassette doesn't record, and the transport
is the layer HTTPX hands them to.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import httpx2
import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai._http import ConnectPoolTimeoutCap, create_async_httpx2_client
from pydantic_ai.models import create_async_http_client
from pydantic_ai.settings import ModelSettings

from .conftest import try_import

with try_import() as openai_imports_successful:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as google_imports_successful:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

_OPENAI_RESPONSE: dict[str, Any] = {
    'id': 'chatcmpl-1',
    'object': 'chat.completion',
    'created': 0,
    'model': 'gpt-5.2',
    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'Paris'}, 'finish_reason': 'stop'}],
    'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
}

_GOOGLE_RESPONSE: dict[str, Any] = {
    'candidates': [{'content': {'role': 'model', 'parts': [{'text': 'Paris'}]}, 'finishReason': 'STOP'}],
    'usageMetadata': {'promptTokenCount': 1, 'candidatesTokenCount': 1, 'totalTokenCount': 2},
    'modelVersion': 'gemini-2.5-flash',
}


@pytest.fixture
def sent_timeouts(monkeypatch: pytest.MonkeyPatch) -> Callable[[dict[str, Any]], list[dict[str, float | None]]]:
    """Make both HTTPX families' transports record each request's timeouts and answer with `body`."""

    def install(body: dict[str, Any]) -> list[dict[str, float | None]]:
        sent: list[dict[str, float | None]] = []

        async def handle_httpx2(self: httpx2.AsyncHTTPTransport, request: httpx2.Request) -> httpx2.Response:
            sent.append(request.extensions['timeout'])
            return httpx2.Response(200, json=body)

        async def handle_httpx(self: httpx.AsyncHTTPTransport, request: httpx.Request) -> httpx.Response:
            sent.append(request.extensions['timeout'])
            return httpx.Response(200, json=body)

        monkeypatch.setattr(httpx2.AsyncHTTPTransport, 'handle_async_request', handle_httpx2)
        monkeypatch.setattr(httpx.AsyncHTTPTransport, 'handle_async_request', handle_httpx)
        return sent

    return install


class _ClientDefault:
    """No per-request timeout: the client's own applies."""


_DEFAULT = _ClientDefault()


@pytest.mark.parametrize('family', ['httpx2', 'httpx'])
@pytest.mark.parametrize(
    'requested,expected',
    [
        pytest.param(
            _DEFAULT,
            snapshot({'connect': 5, 'read': 600, 'write': 600, 'pool': 600}),
            id='client-default',
        ),
        pytest.param(
            30.0,
            snapshot({'connect': 5, 'read': 30.0, 'write': 30.0, 'pool': 30.0}),
            id='shorter-scalar',
        ),
        pytest.param(
            1200.0,
            snapshot({'connect': 5, 'read': 1200.0, 'write': 1200.0, 'pool': 600}),
            id='longer-scalar',
        ),
        pytest.param(
            None,
            snapshot({'connect': 5, 'read': None, 'write': None, 'pool': 600}),
            id='no-timeout',
        ),
        pytest.param(
            (30.0, 2.0),
            snapshot({'connect': 2.0, 'read': 30.0, 'write': 30.0, 'pool': 30.0}),
            id='shorter-connect-kept',
        ),
        pytest.param(
            (60.0, 30.0),
            snapshot({'connect': 30.0, 'read': 60.0, 'write': 60.0, 'pool': 60.0}),
            id='explicit-longer-connect-kept',
        ),
    ],
)
async def test_created_client_caps_scalar_connect_and_pool_timeouts(
    family: str,
    requested: float | tuple[float, float] | _ClientDefault | None,
    expected: dict[str, float | None],
    sent_timeouts: Callable[[dict[str, Any]], list[dict[str, float | None]]],
):
    sent = sent_timeouts({})
    request_kwargs: dict[str, Any] = {}
    if isinstance(requested, tuple):
        timeout_type = httpx2.Timeout if family == 'httpx2' else httpx.Timeout
        request_kwargs['timeout'] = timeout_type(requested[0], connect=requested[1])
    elif not isinstance(requested, _ClientDefault):
        request_kwargs['timeout'] = requested

    async with create_async_httpx2_client() if family == 'httpx2' else create_async_http_client() as client:
        await client.get('https://example.com', **request_kwargs)

    assert sent == [expected]


async def test_client_phase_without_timeout_caps_nothing(
    sent_timeouts: Callable[[dict[str, Any]], list[dict[str, float | None]]],
):
    """A client built without a connect timeout has nothing to cap it at, so the request's applies."""
    sent = sent_timeouts({})
    async with create_async_httpx2_client(timeout=httpx2.Timeout(30, connect=None)) as client:
        await client.get('https://example.com', timeout=60)

    assert sent == snapshot([{'connect': 60, 'read': 60, 'write': 60, 'pool': 30}])


async def test_cap_ignores_request_without_timeout():
    """Only `AsyncClient.send` attaches a timeout, so a request handed to the hook directly has none to cap."""
    request = httpx2.Request('GET', 'https://example.com')
    await ConnectPoolTimeoutCap(connect=5, pool=600)(request)
    assert 'timeout' not in request.extensions


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
async def test_model_settings_timeout_keeps_provider_client_connect_timeout(
    allow_model_requests: None,
    sent_timeouts: Callable[[dict[str, Any]], list[dict[str, float | None]]],
):
    """A numeric `ModelSettings['timeout']` becomes a scalar in the OpenAI SDK, which would set connect too."""
    sent = sent_timeouts(_OPENAI_RESPONSE)
    agent = Agent(OpenAIChatModel('gpt-5.2', provider=OpenAIProvider(api_key='test')))

    async with agent:
        result = await agent.run('What is the capital of France?', model_settings={'timeout': 30})

    assert result.output == 'Paris'
    assert sent == snapshot([{'connect': 5, 'read': 30.0, 'write': 30.0, 'pool': 30.0}])


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
async def test_model_settings_timeout_leaves_user_client_untouched(
    allow_model_requests: None,
    sent_timeouts: Callable[[dict[str, Any]], list[dict[str, float | None]]],
):
    sent = sent_timeouts(_OPENAI_RESPONSE)
    async with httpx2.AsyncClient() as http_client:
        agent = Agent(OpenAIChatModel('gpt-5.2', provider=OpenAIProvider(api_key='test', http_client=http_client)))
        await agent.run('What is the capital of France?', model_settings={'timeout': 30})

    assert sent == snapshot([{'connect': 30.0, 'read': 30.0, 'write': 30.0, 'pool': 30.0}])


@pytest.mark.skipif(not google_imports_successful(), reason='google-genai not installed')
@pytest.mark.parametrize(
    'model_settings,expected',
    [
        pytest.param(
            {},
            snapshot({'connect': 5, 'read': 600.0, 'write': 600.0, 'pool': 600}),
            id='provider-default',
        ),
        pytest.param(
            {'timeout': 30},
            snapshot({'connect': 5, 'read': 30.0, 'write': 30.0, 'pool': 30.0}),
            id='model-settings',
        ),
    ],
)
async def test_google_provider_client_keeps_connect_timeout(
    allow_model_requests: None,
    model_settings: ModelSettings,
    expected: dict[str, float | None],
    sent_timeouts: Callable[[dict[str, Any]], list[dict[str, float | None]]],
):
    """google-genai sends the provider's pinned 600-second `HttpOptions.timeout` as a scalar on every request."""
    sent = sent_timeouts(_GOOGLE_RESPONSE)
    agent = Agent(GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key='test')))

    async with agent:
        result = await agent.run('What is the capital of France?', model_settings=model_settings)

    assert result.output == 'Paris'
    assert sent == [expected]
