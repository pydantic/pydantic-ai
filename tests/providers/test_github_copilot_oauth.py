"""Public device-flow tests; synthetic transports cover approval and otherwise interactive failures."""

from __future__ import annotations

import json
from dataclasses import asdict
from urllib.parse import parse_qs

import anyio
import httpx2
import pytest
from typing_extensions import TypedDict

from pydantic_ai.exceptions import UserError

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.providers.github_copilot import (
        GitHubCopilotCredentials,
        GitHubCopilotOAuthFlow,
        GitHubCopilotProvider,
    )

pytestmark = [pytest.mark.anyio, pytest.mark.skipif(not imports_successful(), reason='openai not installed')]

DEVICE = {
    'device_code': 'secret-device-code',
    'user_code': 'ABCD-EFGH',
    'verification_uri': 'https://github.com/login/device',
    'expires_in': 900,
    'interval': 5,
}
TOKEN = {'access_token': 'secret-access-token', 'token_type': 'bearer', 'scope': 'read:user'}


class RecordedBody(TypedDict):
    string: bytes | str


class RecordedResponse(TypedDict):
    body: RecordedBody


@pytest.fixture(scope='module')
def vcr_config() -> dict[str, object]:
    def scrub_response(response: RecordedResponse) -> RecordedResponse:
        body = response['body']['string']
        data: dict[str, object] = json.loads(body)
        for key in ('device_code', 'user_code', 'access_token', 'refresh_token'):
            if key in data:
                data[key] = 'scrubbed'
        response['body']['string'] = json.dumps(data).encode()
        return response

    return {
        'filter_headers': ['authorization', 'cookie'],
        'filter_post_data_parameters': ['device_code', 'refresh_token'],
        'before_record_response': scrub_response,
        'decode_compressed_response': True,
    }


@pytest.mark.vcr
async def test_start_against_github() -> None:
    """Record only initiation, using the public GitHub CLI client; no user approves this grant."""
    flow = GitHubCopilotOAuthFlow(client_id='178c6fc778ccc68e1d6a')
    authorization = await flow.start()
    assert authorization.verification_uri == 'https://github.com/login/device'
    assert 0 < authorization.expires_in <= 900
    assert authorization.interval == 5
    assert authorization.user_code
    assert authorization.device_code


@pytest.mark.vcr
async def test_github_rejects_unknown_client() -> None:
    flow = GitHubCopilotOAuthFlow(client_id='pydantic-ai-invalid-client')
    with pytest.raises(UserError, match='HTTP 404'):
        await flow.start()


@pytest.mark.vcr
async def test_github_pending_authorization_can_be_cancelled() -> None:
    pending = anyio.Event()

    async def receive(response: httpx2.Response) -> None:
        await response.aread()
        if response.request.url.path == '/login/oauth/access_token':
            assert response.json()['error'] == 'authorization_pending'
            pending.set()

    async with httpx2.AsyncClient(event_hooks={'response': [receive]}) as client:
        flow = GitHubCopilotOAuthFlow(client_id='178c6fc778ccc68e1d6a', http_client=client)
        await flow.start()
        with anyio.fail_after(30):
            async with anyio.create_task_group() as group:
                group.start_soon(flow.wait_for_authorization)
                await pending.wait()
                group.cancel_scope.cancel()
        assert not client.is_closed


@pytest.fixture
def delays(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Advance polling without wall-clock sleeps, while retaining cancellation checkpoints."""
    delays: list[float] = []
    sleep = anyio.sleep

    async def record_sleep(delay: float) -> None:
        delays.append(delay)
        await sleep(0)

    monkeypatch.setattr(anyio, 'sleep', record_sleep)
    return delays


async def test_approval_and_provider_handoff(delays: list[float]) -> None:
    requests: list[httpx2.Request] = []
    responses = iter(
        [
            DEVICE,
            {'error': 'authorization_pending'},
            {'error': 'slow_down'},
            {'error': 'slow_down', 'interval': 1},
            {'error': 'slow_down', 'interval': 30},
            TOKEN,
        ]
    )

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json=next(responses))

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        flow = GitHubCopilotOAuthFlow(client_id='my-client', scope='read:user', http_client=client)
        authorization = await flow.start()
        assert 'secret-device-code' not in repr(authorization)
        credentials = await flow.wait_for_authorization()
        assert credentials == GitHubCopilotCredentials(
            access_token='secret-access-token', token_type='bearer', scope='read:user'
        )
        assert 'secret-access-token' not in repr(credentials)
        assert GitHubCopilotProvider(api_key=credentials.access_token).client.api_key == 'secret-access-token'
        assert not client.is_closed
        with pytest.raises(UserError, match='Call `start'):
            await flow.wait_for_authorization()

    assert delays == [5, 5, 10, 15, 30]
    assert [request.url.path for request in requests] == ['/login/device/code'] + ['/login/oauth/access_token'] * 5
    assert all(request.headers['accept'] == 'application/json' for request in requests)
    assert parse_qs(requests[0].content.decode()) == {'client_id': ['my-client'], 'scope': ['read:user']}
    assert parse_qs(requests[-1].content.decode()) == {
        'client_id': ['my-client'],
        'device_code': ['secret-device-code'],
        'grant_type': ['urn:ietf:params:oauth:grant-type:device_code'],
    }


async def test_expiring_credentials_preserve_renewal_metadata(delays: list[float]) -> None:
    token = {
        **TOKEN,
        'expires_in': 28800,
        'refresh_token': 'secret-refresh-token',
        'refresh_token_expires_in': 15897600,
    }
    responses = iter([DEVICE, token])
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda _: httpx2.Response(200, json=next(responses)))
    ) as client:
        flow = GitHubCopilotOAuthFlow(client_id='my-client', http_client=client)
        await flow.start()
        credentials = await flow.wait_for_authorization()
    assert asdict(credentials) == token
    assert 'secret-refresh-token' not in repr(credentials)


@pytest.mark.parametrize('client_id', ['', '   '])
def test_empty_client_id(client_id: str) -> None:
    with pytest.raises(UserError, match='non-empty'):
        GitHubCopilotOAuthFlow(client_id=client_id)


async def test_poll_without_start() -> None:
    with pytest.raises(UserError, match='Call `start'):
        await GitHubCopilotOAuthFlow(client_id='my-client').wait_for_authorization()


@pytest.mark.parametrize('error', ['access_denied', 'expired_token', 'incorrect_device_code', 'device_flow_disabled'])
async def test_oauth_rejection(error: str, delays: list[float]) -> None:
    responses = iter([DEVICE, {'error': error, 'error_description': 'secret-server-detail'}])
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda _: httpx2.Response(200, json=next(responses)))
    ) as client:
        flow = GitHubCopilotOAuthFlow(client_id='my-client', http_client=client)
        await flow.start()
        with pytest.raises(UserError, match=error) as exc:
            await flow.wait_for_authorization()
        assert 'secret-server-detail' not in str(exc.value)


@pytest.mark.parametrize(
    'body',
    [
        b'not json secret',
        b'[]',
        b'null',
        b'{"device_code": "secret-device-code"}',
        json.dumps({**DEVICE, 'verification_uri': 'file:///secret'}).encode(),
        json.dumps({**DEVICE, 'verification_uri': 'https://other.example/login/device'}).encode(),
        json.dumps({**DEVICE, 'expires_in': 0}).encode(),
        json.dumps({**DEVICE, 'interval': -1}).encode(),
        json.dumps({**DEVICE, 'interval': True}).encode(),
    ],
)
async def test_invalid_device_response(body: bytes) -> None:
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda _: httpx2.Response(200, content=body))
    ) as client:
        flow = GitHubCopilotOAuthFlow(client_id='my-client', http_client=client)
        with pytest.raises(UserError, match='invalid device authorization response') as exc:
            await flow.start()
        assert 'secret' not in str(exc.value)


@pytest.mark.parametrize(
    'body',
    [
        b'not json secret',
        b'[]',
        b'null',
        b'{}',
        json.dumps({**TOKEN, 'access_token': ''}).encode(),
        json.dumps({**TOKEN, 'token_type': 'mac'}).encode(),
        json.dumps({'error': 'slow_down', 'interval': 0}).encode(),
        json.dumps({'error': 'secret-server-detail'}).encode(),
    ],
)
async def test_invalid_token_response(body: bytes, delays: list[float]) -> None:
    responses = iter([httpx2.Response(200, json=DEVICE), httpx2.Response(200, content=body)])
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda _: next(responses))) as client:
        flow = GitHubCopilotOAuthFlow(client_id='my-client', http_client=client)
        await flow.start()
        with pytest.raises(UserError, match='invalid device token response') as exc:
            await flow.wait_for_authorization()
        assert 'secret' not in str(exc.value)


@pytest.mark.parametrize('status', [302, 400, 500])
@pytest.mark.parametrize('during_poll', [False, True])
async def test_http_failure_does_not_follow_redirects(status: int, during_poll: bool, delays: list[float]) -> None:
    responses = [httpx2.Response(status, text='secret-body', headers={'location': 'https://other.example'})]
    if during_poll:
        responses.insert(0, httpx2.Response(200, json=DEVICE))
    pending = iter(responses)
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda _: next(pending)), follow_redirects=True
    ) as client:
        flow = GitHubCopilotOAuthFlow(client_id='my-client', http_client=client)
        if during_poll:
            await flow.start()
        with pytest.raises(UserError, match=f'HTTP {status}') as exc:
            await (flow.wait_for_authorization() if during_poll else flow.start())
        assert 'secret-body' not in str(exc.value)


async def test_start_rejection_clears_previous_challenge() -> None:
    responses = iter([DEVICE, {'error': 'device_flow_disabled'}])
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda _: httpx2.Response(200, json=next(responses)))
    ) as client:
        flow = GitHubCopilotOAuthFlow(client_id='my-client', http_client=client)
        await flow.start()
        with pytest.raises(UserError, match='device_flow_disabled'):
            await flow.start()
        with pytest.raises(UserError, match='Call `start'):
            await flow.wait_for_authorization()


async def test_default_interval_and_restart(delays: list[float]) -> None:
    device = {key: value for key, value in DEVICE.items() if key != 'interval'}
    responses = iter([device, {'error': 'access_denied'}, DEVICE, TOKEN])
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda _: httpx2.Response(200, json=next(responses)))
    ) as client:
        flow = GitHubCopilotOAuthFlow(client_id='my-client', http_client=client)
        assert (await flow.start()).interval == 5
        with pytest.raises(UserError, match='access_denied'):
            await flow.wait_for_authorization()
        await flow.start()
        assert (await flow.wait_for_authorization()).access_token == TOKEN['access_token']


async def test_slow_initiation_preserves_challenge_lifetime(
    monkeypatch: pytest.MonkeyPatch, delays: list[float]
) -> None:
    now = 0
    monkeypatch.setattr('pydantic_ai.providers._github_copilot_oauth.monotonic', lambda: now)
    requests: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal now
        requests.append(request.url.path)
        if request.url.path == '/login/device/code':
            now += 10
            return httpx2.Response(200, json=DEVICE)
        return httpx2.Response(200, json=TOKEN)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        flow = GitHubCopilotOAuthFlow(client_id='my-client', http_client=client)
        await flow.start()
        now += 894
        assert (await flow.wait_for_authorization()).access_token == TOKEN['access_token']
    assert requests == ['/login/device/code', '/login/oauth/access_token']
    assert delays == [5]


async def test_expired_challenge_does_not_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([0, 901])
    monkeypatch.setattr('pydantic_ai.providers._github_copilot_oauth.monotonic', lambda: next(times))
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json=DEVICE)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        flow = GitHubCopilotOAuthFlow(client_id='my-client', http_client=client)
        await flow.start()
        with pytest.raises(UserError, match='expired'):
            await flow.wait_for_authorization()
    assert len(requests) == 1


@pytest.mark.parametrize('expires', [False, True])
async def test_cancellation_stops_inflight_request(
    delays: list[float], monkeypatch: pytest.MonkeyPatch, expires: bool
) -> None:
    if expires:
        times = iter([0, 899])
        monkeypatch.setattr('pydantic_ai.providers._github_copilot_oauth.monotonic', lambda: next(times))
    entered, exited = anyio.Event(), anyio.Event()

    async def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == '/login/device/code':
            return httpx2.Response(200, json=DEVICE)
        entered.set()
        try:
            await anyio.sleep_forever()
        finally:
            exited.set()
        raise AssertionError('The token request must be cancelled')  # pragma: no cover

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        flow = GitHubCopilotOAuthFlow(client_id='my-client', http_client=client)
        await flow.start()
        if expires:
            with pytest.raises(UserError, match='expired'):
                await flow.wait_for_authorization()
        else:
            with anyio.fail_after(10):
                async with anyio.create_task_group() as group:
                    group.start_soon(flow.wait_for_authorization)
                    await entered.wait()
                    group.cancel_scope.cancel()
        assert entered.is_set()
        assert exited.is_set()
        assert not client.is_closed
        with pytest.raises(UserError, match='Call `start'):
            await flow.wait_for_authorization()


async def test_transport_error_propagates() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError('offline', request=request)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        with pytest.raises(httpx2.ConnectError, match='offline'):
            await GitHubCopilotOAuthFlow(client_id='my-client', http_client=client).start()
