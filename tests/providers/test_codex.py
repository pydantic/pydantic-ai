from __future__ import annotations as _annotations

from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone

import anyio
import httpx
import pytest
from pydantic import SecretStr

from pydantic_ai.auth.codex import CodexCredentials
from pydantic_ai.providers.codex import CODEX_BASE_URL, CodexProvider

pytestmark = pytest.mark.anyio


def _credentials(*, revision: str = 'revision-1', fedramp: bool = False) -> CodexCredentials:
    return CodexCredentials(
        access_token=SecretStr(f'access-{revision}'),
        refresh_token=SecretStr(f'refresh-{revision}'),
        id_token=SecretStr(f'id-{revision}'),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        account_id=SecretStr(f'account-{revision}'),
        account_is_fedramp=fedramp,
        revision=revision,
    )


class OneShotAsyncStream(httpx.AsyncByteStream):
    def __init__(self, content: bytes, *, fail: bool = False) -> None:
        self.content = content
        self.fail = fail
        self.iterations = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.iterations += 1
        if self.iterations > 1:
            raise AssertionError('stream was consumed more than once')
        if self.fail:
            raise httpx.ReadError('simulated stream failure')
        yield self.content


async def test_one_shot_stream_rejects_second_consumption() -> None:
    stream = OneShotAsyncStream(b'content')

    assert b''.join([chunk async for chunk in stream]) == b'content'
    with pytest.raises(AssertionError, match='more than once'):
        _ = [chunk async for chunk in stream]


class CredentialSource:
    def __init__(self, credentials: CodexCredentials | None = None) -> None:
        self.credentials = credentials or _credentials()
        self.calls: list[tuple[bool, str | None]] = []

    async def get_credentials(
        self, *, force_refresh: bool = False, rejected_revision: str | None = None
    ) -> CodexCredentials:
        self.calls.append((force_refresh, rejected_revision))
        return self.credentials


def test_provider_construction_is_lazy() -> None:
    source = CredentialSource()
    provider = CodexProvider(credential_source=source)

    assert provider.name == 'codex'
    assert provider.base_url == f'{CODEX_BASE_URL}/'
    assert source.calls == []


async def test_provider_adds_one_coherent_credential_snapshot() -> None:
    source = CredentialSource(_credentials(fedramp=True))
    captured_headers: list[dict[str, str]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        captured_headers.append(dict(request.headers))
        return httpx.Response(200, json={'ok': True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        CodexProvider(credential_source=source, http_client=client)
        response = await client.post(f'{CODEX_BASE_URL}/responses', json={'model': 'gpt-5.5'})

    assert response.status_code == 200
    assert source.calls == [(False, None)]
    assert captured_headers[0]['authorization'] == 'Bearer access-revision-1'
    assert captured_headers[0]['chatgpt-account-id'] == 'account-revision-1'
    assert captured_headers[0]['originator'] == 'pydantic-ai'
    assert captured_headers[0]['x-openai-fedramp'] == 'true'


async def test_provider_replays_one_responses_401_after_forced_refresh() -> None:
    source = CredentialSource()
    request_count = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(401)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        CodexProvider(credential_source=source, http_client=client)
        response = await client.post(f'{CODEX_BASE_URL}/responses', json={'model': 'gpt-5.5'})

    assert response.status_code == 401
    assert request_count == 2
    assert source.calls == [(False, None), (True, 'revision-1')]


async def test_provider_replays_buffered_request_and_drains_401_response() -> None:
    source = CredentialSource()
    response_stream = OneShotAsyncStream(b'unauthorized')
    request_bodies: list[bytes] = []

    def handle(request: httpx.Request) -> httpx.Response:
        request_bodies.append(request.content)
        if len(request_bodies) == 1:
            return httpx.Response(401, stream=response_stream)
        return httpx.Response(200, json={'ok': True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        CodexProvider(credential_source=source, http_client=client)
        response = await client.post(f'{CODEX_BASE_URL}/responses', content=b'{"model":"gpt-5.5"}')

    assert response.status_code == 200
    assert request_bodies == [b'{"model":"gpt-5.5"}'] * 2
    assert response_stream.iterations == 1


async def test_provider_does_not_replay_unbuffered_request_body() -> None:
    source = CredentialSource()
    request_stream = OneShotAsyncStream(b'{"model":"gpt-5.5"}')
    request_count = 0

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        assert await request.aread() == b'{"model":"gpt-5.5"}'
        return httpx.Response(401)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        CodexProvider(credential_source=source, http_client=client)
        response = await client.post(f'{CODEX_BASE_URL}/responses', content=request_stream)

    assert response.status_code == 401
    assert request_count == 1
    assert request_stream.iterations == 1
    assert source.calls == [(False, None)]


async def test_provider_does_not_replay_when_401_body_cannot_be_drained() -> None:
    source = CredentialSource()
    request_count = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(401, stream=OneShotAsyncStream(b'', fail=True))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        CodexProvider(credential_source=source, http_client=client)
        with pytest.raises(httpx.ReadError, match='simulated stream failure'):
            await client.post(f'{CODEX_BASE_URL}/responses', json={'model': 'gpt-5.5'})

    assert request_count == 1
    assert source.calls == [(False, None)]


async def test_concurrent_401_recovery_rotates_rejected_revision_once() -> None:
    class RotatingSource(CredentialSource):
        def __init__(self) -> None:
            super().__init__(_credentials(revision='revision-a'))
            self.lock = anyio.Lock()
            self.refreshes = 0

        async def get_credentials(
            self, *, force_refresh: bool = False, rejected_revision: str | None = None
        ) -> CodexCredentials:
            self.calls.append((force_refresh, rejected_revision))
            if force_refresh:
                async with self.lock:
                    if self.credentials.revision == rejected_revision:
                        await anyio.sleep(0)
                        self.credentials = _credentials(revision='revision-b')
                        self.refreshes += 1
            return self.credentials

    source = RotatingSource()
    both_initial_requests = anyio.Event()
    initial_requests = 0
    replay_authorizations: list[str] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal initial_requests
        authorization = request.headers['authorization']
        if authorization == 'Bearer access-revision-a':
            initial_requests += 1
            if initial_requests == 2:
                both_initial_requests.set()
            await both_initial_requests.wait()
            return httpx.Response(401)
        replay_authorizations.append(authorization)
        return httpx.Response(200, json={'ok': True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        CodexProvider(credential_source=source, http_client=client)
        statuses: list[int] = []

        async def send() -> None:
            response = await client.post(f'{CODEX_BASE_URL}/responses', json={'model': 'gpt-5.5'})
            statuses.append(response.status_code)

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(send)
            task_group.start_soon(send)

    assert statuses == [200, 200]
    assert source.refreshes == 1
    assert replay_authorizations == ['Bearer access-revision-b'] * 2
    assert source.calls.count((True, 'revision-a')) == 2


async def test_provider_does_not_replay_other_endpoints() -> None:
    source = CredentialSource()
    request_count = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(401)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        CodexProvider(credential_source=source, http_client=client)
        response = await client.get(f'{CODEX_BASE_URL}/models')

    assert response.status_code == 401
    assert request_count == 1
    assert source.calls == [(False, None)]


async def test_provider_does_not_close_caller_owned_client() -> None:
    source = CredentialSource()
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    provider = CodexProvider(credential_source=source, http_client=client)

    async with provider:
        assert not client.is_closed
    assert not client.is_closed
    await client.aclose()


async def test_provider_owned_client_closes_and_recreates_with_auth() -> None:
    source = CredentialSource()
    provider = CodexProvider(credential_source=source)
    first_client = provider.client._client  # pyright: ignore[reportPrivateUsage]

    async with provider:
        assert not first_client.is_closed
    assert first_client.is_closed

    async with provider:
        second_client = provider.client._client  # pyright: ignore[reportPrivateUsage]
        assert second_client is not first_client
        assert not second_client.is_closed
    assert second_client.is_closed


@pytest.mark.parametrize(
    'url',
    [
        'https://example.com/backend-api/codex/responses',
        'http://chatgpt.com/backend-api/codex/responses',
        'https://chatgpt.com:444/backend-api/codex/responses',
        'https://chatgpt.com/backend-api/other/responses',
        'https://chatgpt.com/backend-api/codex/%2e%2e/other',
        'https://chatgpt.com/backend-api/codex/.%2e/other',
        'https://chatgpt.com/backend-api/codex/a%2Fb',
    ],
)
async def test_provider_never_adds_credentials_off_trusted_origin_or_path(url: str) -> None:
    source = CredentialSource()
    captured_headers: list[httpx.Headers] = []

    def handle(request: httpx.Request) -> httpx.Response:
        captured_headers.append(request.headers)
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        CodexProvider(credential_source=source, http_client=client)
        await client.post(url, json={'model': 'gpt-5.5'})

    assert source.calls == []
    assert 'authorization' not in captured_headers[0]
    assert 'chatgpt-account-id' not in captured_headers[0]
    assert 'originator' not in captured_headers[0]


@pytest.mark.parametrize(
    'location',
    [
        'https://chatgpt.com/backend-api/other/collect',
        'https://example.com/collect',
    ],
)
async def test_provider_does_not_follow_initial_redirects(location: str) -> None:
    source = CredentialSource()
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(307, headers={'location': location})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        CodexProvider(credential_source=source, http_client=client)
        response = await client.post(f'{CODEX_BASE_URL}/responses', content=b'sensitive prompt')

    assert response.status_code == 307
    assert len(requests) == 1
    assert requests[0].content == b'sensitive prompt'


async def test_provider_does_not_follow_redirect_after_401_replay() -> None:
    source = CredentialSource()
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(401)
        return httpx.Response(307, headers={'location': 'https://example.com/collect'})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        CodexProvider(credential_source=source, http_client=client)
        response = await client.post(f'{CODEX_BASE_URL}/responses', content=b'sensitive prompt')

    assert response.status_code == 307
    assert len(requests) == 2
    assert all(request.content == b'sensitive prompt' for request in requests)


async def test_provider_rejects_caller_client_with_existing_auth() -> None:
    client = httpx.AsyncClient(auth=('user', 'password'))
    try:
        with pytest.raises(ValueError, match='must not already have authentication'):
            CodexProvider(credential_source=CredentialSource(), http_client=client)
    finally:
        await client.aclose()


async def test_provider_rejects_caller_client_that_follows_redirects() -> None:
    client = httpx.AsyncClient(follow_redirects=True)
    try:
        with pytest.raises(ValueError, match='follow_redirects=False'):
            CodexProvider(credential_source=CredentialSource(), http_client=client)
    finally:
        await client.aclose()


async def test_provider_preserves_falsey_credential_source() -> None:
    class FalseySource(CredentialSource):
        def __bool__(self) -> bool:
            return False  # pragma: no cover - the provider must use an explicit `None` check

    source = FalseySource()
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client:
        CodexProvider(credential_source=source, http_client=client)
        await client.post(f'{CODEX_BASE_URL}/responses', json={'model': 'gpt-5.5'})

    assert source.calls == [(False, None)]


def test_provider_profile_requires_non_stored_responses() -> None:
    profile = CodexProvider.model_profile('gpt-5.5')
    assert profile.get('openai_responses_requires_store_false') is True
    assert profile.get('supports_thinking') is True
