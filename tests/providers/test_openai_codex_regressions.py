"""Credential boundary and recovery regressions through the real HTTP auth flow."""

import asyncio
import traceback

import httpx2
import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.providers.openai_codex import (
    CredentialsPersistenceError,
    CredentialsRefreshError,
    OpenAICodexCredentials,
    OpenAICodexProvider,
    _post_token_request,  # pyright: ignore[reportPrivateUsage]
)

from .test_openai_codex import CODEX_URL, TOKEN_RESPONSE, FakeCredentialSource, make_credentials

pytestmark = pytest.mark.anyio


async def test_refresh_failure_shared_then_later_request_recovers():
    token_hits = 0
    initial_requests = 0
    all_sent = asyncio.Event()

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal token_hits, initial_requests
        if request.url.host == 'auth.openai.com':
            token_hits += 1
            if token_hits == 1:
                return httpx2.Response(503, json={'error': 'temporarily_unavailable'})
            return httpx2.Response(200, json=TOKEN_RESPONSE)
        if request.headers['authorization'] == 'Bearer access-new':
            return httpx2.Response(200)
        initial_requests += 1
        if initial_requests == 5:
            all_sent.set()
        await all_sent.wait()
        return httpx2.Response(401)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        provider = OpenAICodexProvider(credentials=make_credentials(), http_client=client)
        results = await asyncio.gather(*(client.get(CODEX_URL) for _ in range(5)), return_exceptions=True)
        assert all(isinstance(result, CredentialsRefreshError) for result in results)
        assert token_hits == 1
        assert (await client.get(CODEX_URL)).status_code == 200
        assert token_hits == 2
        assert provider.credentials.access_token == 'access-new'


@pytest.mark.parametrize('stale', [False, True])
async def test_sdk_does_not_hide_persistence_failure(stale: bool):
    source = FakeCredentialSource(make_credentials(exp=0 if stale else None))
    saves = 0
    requests = 0

    async def save(credentials: OpenAICodexCredentials) -> None:
        nonlocal saves
        saves += 1
        raise RuntimeError('storage unavailable')

    source.save = save

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal requests
        if request.url.host == 'auth.openai.com':
            return httpx2.Response(200, json=TOKEN_RESPONSE)
        requests += 1
        if request.headers['authorization'] == 'Bearer access-new':
            return httpx2.Response(200, headers={'content-type': 'text/event-stream'}, content=b'data: [DONE]\n\n')
        return httpx2.Response(401)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        provider = OpenAICodexProvider(credential_source=source, http_client=client)
        with pytest.raises(CredentialsPersistenceError, match='saving'):
            await provider.client.responses.create(model='gpt-5.6-luna', input='hi', stream=True, store=False)
        assert provider.credentials.access_token == 'access-new'
    assert saves == 1
    assert requests == (0 if stale else 1)


@pytest.mark.parametrize('field', ['access_token', 'refresh_token', 'account_id'])
@pytest.mark.parametrize('value', ['', None])
def test_cli_rejects_malformed_fields_without_leaking(field: str, value: str | None):
    tokens: dict[str, object] = {
        'access_token': 'SENTINEL_ACCESS',
        'refresh_token': 'SENTINEL_REFRESH',
        'account_id': 'acc',
    }
    if value is None:
        del tokens[field]
    else:
        tokens[field] = value
    with pytest.raises(UserError) as exc:
        OpenAICodexCredentials.from_codex_cli_auth({'tokens': tokens})
    formatted = ''.join(traceback.format_exception(exc.value))
    assert 'SENTINEL_ACCESS' not in formatted
    assert 'SENTINEL_REFRESH' not in formatted
    assert field in str(exc.value)


@pytest.mark.parametrize('field', ['access_token', 'refresh_token'])
@pytest.mark.parametrize('value', ['', None])
async def test_token_endpoint_rejects_malformed_fields_without_leaking(field: str, value: str | None):
    payload: dict[str, object] = {
        'access_token': 'SENTINEL_ACCESS',
        'refresh_token': 'SENTINEL_REFRESH',
        'id_token': 'SENTINEL_ID',
    }
    if value is None:
        del payload[field]
    else:
        payload[field] = value

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=payload)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        with pytest.raises(CredentialsRefreshError) as exc:
            await _post_token_request('https://auth.openai.com/oauth/token', {}, http_client=client)
    formatted = ''.join(traceback.format_exception(exc.value))
    for token in ('SENTINEL_ACCESS', 'SENTINEL_REFRESH', 'SENTINEL_ID'):
        assert token not in formatted
    assert field in str(exc.value)
