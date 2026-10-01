"""Synthetic issuer tests: OAuth failures/rotation cannot safely be replayed from live credentials."""

from __future__ import annotations

import base64
import hashlib
import json
import socket
import threading
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from http.server import HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

import anyio
import httpx2
import pytest

from pydantic_ai.exceptions import ModelAPIError, UserError
from pydantic_ai.models import infer_model
from pydantic_ai.providers import infer_provider_class

from ..conftest import try_import

with try_import() as imports_successful:
    import jwt
    from cryptography.hazmat.primitives.asymmetric import rsa
    from jwt.algorithms import RSAAlgorithm

    from pydantic_ai.models.openai_chatgpt import OpenAIChatGPTModel
    from pydantic_ai.providers.openai_chatgpt import (
        OpenAIChatGPTClient,
        OpenAIChatGPTCredentials,
        OpenAIChatGPTOAuthFlow,
        OpenAIChatGPTProvider,
    )

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai/pyjwt crypto not installed')

ISSUER = 'https://auth.openai.com'
RESOURCE = 'https://api.openai.com/v1'
REDIRECT = 'http://127.0.0.1:1455/auth/callback'
SCOPES = ('openid', 'profile', 'email', 'offline_access', 'resource.invoke', 'chatgpt.tokens.use.direct')


def credentials(expires: int = 3600) -> OpenAIChatGPTCredentials:
    return OpenAIChatGPTCredentials(
        subject='subject',
        client_id='oaiapp_test',
        ext_agent_host_id='host',
        redirect_uri=REDIRECT,
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=expires),
        scopes=SCOPES,
        access_token='synthetic-access',
        refresh_token='synthetic-refresh',
        id_token='synthetic-id',
    )


def callback(flow: OpenAIChatGPTOAuthFlow, **changes: str) -> str:
    values = {'state': flow.state, 'code': 'synthetic-code', 'client_id': 'oaiapp_test', **changes}
    return flow.redirect_uri + '?' + urlencode(values)


@pytest.fixture(scope='module')
def signing_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


class Issuer:
    def __init__(self, key: rsa.RSAPrivateKey):
        self.key = key
        self.nonce: str | None = None
        self.claims: dict[str, Any] = {}
        self.token_changes: dict[str, Any] = {}
        self.omit: set[str] = set()
        self.forms: list[dict[str, list[str]]] = []
        self.headers: list[httpx2.Headers] = []
        self.status = 200
        self.discovery_status = 200
        self.keys_status = 200
        self.discovery_changes: dict[str, Any] = {}
        self.bad_signature = False
        self.responses = 0
        self.always_401 = False
        self.refresh_started = anyio.Event()
        self.release_refresh: anyio.Event | None = None

    async def __call__(self, request: httpx2.Request) -> httpx2.Response:
        if request.url.path == '/.well-known/openid-configuration':
            return httpx2.Response(
                self.discovery_status,
                json={
                    'issuer': ISSUER,
                    'jwks_uri': ISSUER + '/.well-known/jwks.json',
                    'id_token_signing_alg_values_supported': ['RS256'],
                    **self.discovery_changes,
                },
            )
        if request.url.path == '/.well-known/jwks.json':
            key = json.loads(RSAAlgorithm.to_jwk(self.key.public_key()))
            return httpx2.Response(self.keys_status, json={'keys': [{**key, 'kid': 'test', 'alg': 'RS256'}]})
        if request.url.host == 'auth.openai.com':
            self.forms.append(parse_qs(request.content.decode()))
            self.headers.append(request.headers)
            self.refresh_started.set()
            if self.release_refresh is not None:
                await self.release_refresh.wait()
            if self.status != 200:
                return httpx2.Response(self.status, json={'error': 'invalid_grant', 'description': 'synthetic-secret'})
            now = datetime.now(timezone.utc).timestamp()
            claims = {
                'iss': ISSUER,
                'aud': 'oaiapp_test',
                'sub': 'subject',
                'iat': now,
                'exp': now + 3600,
                'email': 'test@example.com',
                **self.claims,
            }
            if self.nonce:
                claims.setdefault('nonce', self.nonce)
            token = jwt.encode(claims, self.key, algorithm='RS256', headers={'kid': 'test'})
            if self.bad_signature:
                parts = token.split('.')
                parts[2] = 'AAAA'
                token = '.'.join(parts)
            token_response = {
                'access_token': 'new-access',
                'refresh_token': 'new-refresh',
                'id_token': token,
                'token_type': 'Bearer',
                'expires_in': 3600,
                'scope': ' '.join(SCOPES),
                **self.token_changes,
            }
            for name in self.omit:
                token_response.pop(name)
            return httpx2.Response(200, json=token_response)
        if request.url.path.endswith('/responses'):
            self.responses += 1
            if self.always_401 or request.headers.get('authorization') != 'Bearer new-access':
                return httpx2.Response(401, json={'error': {'message': 'Expired'}})
        return httpx2.Response(200, json={'ok': True})


async def test_dynamic_registration(signing_key: rsa.RSAPrivateKey):
    issuer = Issuer(signing_key)
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(issuer)) as client:
        flow = OpenAIChatGPTOAuthFlow(ext_agent_host_id='host', agent_name='Test App', http_client=client)
        issuer.nonce = flow.nonce
        params = parse_qs(urlsplit(flow.authorization_url()).query)
        assert params == {
            'client_id': ['dynamic_agent_client'],
            'agent_name_hint': ['Test App'],
            'ext_agent_host_id': ['host'],
            'response_type': ['code'],
            'redirect_uri': [REDIRECT],
            'scope': [' '.join(SCOPES)],
            'resource': [RESOURCE],
            'state': [flow.state],
            'nonce': [flow.nonce],
            'code_challenge_method': ['S256'],
            'code_challenge': [
                base64.urlsafe_b64encode(hashlib.sha256(flow.code_verifier.encode()).digest()).rstrip(b'=').decode()
            ],
        }
        result = await flow.exchange_callback(callback(flow))
        assert (result.subject, result.client_id, result.email) == ('subject', 'oaiapp_test', 'test@example.com')
        assert result.scopes == SCOPES
        assert issuer.forms == [
            {
                'grant_type': ['authorization_code'],
                'code': ['synthetic-code'],
                'client_id': ['oaiapp_test'],
                'code_verifier': [flow.code_verifier],
                'redirect_uri': [REDIRECT],
                'resource': [RESOURCE],
            }
        ]
        assert 'synthetic-access' not in repr(credentials())
        with pytest.raises(UserError, match='consumed'):
            await flow.exchange_callback(callback(flow))
        with pytest.raises(UserError, match='issued client ID'):
            await OpenAIChatGPTOAuthFlow(ext_agent_host_id='host', agent_name='App').exchange_code('code')


@pytest.mark.parametrize(
    'changes',
    [
        {'state': 'wrong'},
        {'client_id': ''},
        {'client_id': 'dynamic_agent_client'},
        {'code': ''},
        {'error': 'access_denied'},
    ],
)
async def test_callback_rejected_before_network(changes: dict[str, str]):
    flow = OpenAIChatGPTOAuthFlow(ext_agent_host_id='host', agent_name='App')
    with pytest.raises(UserError):
        await flow.exchange_callback(callback(flow, **changes))


@pytest.mark.parametrize('suffix', ['#fragment', '&state=second', '&code=second', '&invalid', '&client_id=second'])
async def test_callback_ambiguity(suffix: str):
    flow = OpenAIChatGPTOAuthFlow(ext_agent_host_id='host', agent_name='App')
    with pytest.raises(UserError, match='does not match'):
        await flow.exchange_callback(callback(flow) + suffix)


@pytest.mark.parametrize(
    'redirect',
    [
        'http://localhost:1455/auth/callback',
        'https://127.0.0.1:1455/auth/callback',
        'http://127.0.0.1/auth/callback',
        'http://user@127.0.0.1:1455/auth/callback',
    ],
)
def test_oss_redirect_restriction(redirect: str):
    with pytest.raises(UserError, match='HTTP callback'):
        OpenAIChatGPTOAuthFlow(ext_agent_host_id='host', agent_name='App', redirect_uri=redirect)


def test_returning_registration():
    old = credentials()
    flow = OpenAIChatGPTOAuthFlow(
        ext_agent_host_id='host', agent_name='App', credentials=old, redirect_uri='http://127.0.0.1:15432/auth/callback'
    )
    params = parse_qs(urlsplit(flow.authorization_url()).query)
    assert params['client_id'] == ['oaiapp_test']
    assert params['id_token_hint'] == ['synthetic-id']
    assert 'agent_name_hint' not in params
    for field in ('nonce', 'resource', 'state', 'client_id', 'code_challenge_method', 'ext_agent_host_id'):
        with pytest.raises(UserError, match='cannot override'):
            flow.authorization_url(extra_params={field: 'wrong'})
    with pytest.raises(UserError, match='another host'):
        OpenAIChatGPTOAuthFlow(ext_agent_host_id='other', agent_name='App', credentials=old)
    with pytest.raises(UserError, match='Only the callback port'):
        OpenAIChatGPTOAuthFlow(
            ext_agent_host_id='host', agent_name='App', credentials=old, redirect_uri='http://127.0.0.1:1455/other'
        )


@pytest.mark.parametrize('method', ['none', 'client_secret_basic', 'client_secret_post'])
async def test_provisioned_client(signing_key: rsa.RSAPrivateKey, method: Any):
    issuer = Issuer(signing_key)
    config = OpenAIChatGPTClient(
        client_id='oaiapp_test',
        redirect_uri='https://app.example/callback',
        token_endpoint_auth_method=method,
        client_secret=None if method == 'none' else 'secret',
    )
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(issuer)) as client:
        flow = OpenAIChatGPTOAuthFlow(ext_agent_host_id='host', agent_name='App', client=config, http_client=client)
        params = parse_qs(urlsplit(flow.authorization_url()).query)
        assert params['client_id'] == [config.client_id]
        assert params['redirect_uri'] == [config.redirect_uri]
        assert {'agent_name_hint', 'id_token_hint'}.isdisjoint(params)
        issuer.nonce = flow.nonce
        result = await flow.exchange_code('code')
        assert result.redirect_uri == config.redirect_uri
        assert issuer.forms[0]['redirect_uri'] == [config.redirect_uri]
        if method == 'client_secret_basic':
            assert issuer.headers[0]['authorization'].startswith('Basic ')
            assert 'client_secret' not in issuer.forms[0]
        elif method == 'client_secret_post':
            assert issuer.forms[0]['client_secret'] == ['secret']
        else:
            assert 'authorization' not in issuer.headers[0]
            assert 'client_secret' not in issuer.forms[0]
        with pytest.raises(UserError, match='application server'):
            await flow.exchange_code_from_callback()
    assert "client_secret='secret'" not in repr(config)


@pytest.mark.parametrize(
    'claims',
    [
        {'iss': 'https://other.example'},
        {'aud': 'wrong'},
        {'nonce': 'wrong'},
        {'azp': 'wrong'},
        {'exp': 1},
        {'sub': ''},
        {'sub': 'different'},
    ],
)
async def test_identity_validation(signing_key: rsa.RSAPrivateKey, claims: dict[str, Any]):
    issuer = Issuer(signing_key)
    issuer.claims = claims
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(issuer)) as client:
        flow = OpenAIChatGPTOAuthFlow(
            ext_agent_host_id='host', agent_name='App', http_client=client, credentials=credentials()
        )
        issuer.nonce = flow.nonce
        with pytest.raises(ModelAPIError, match='ID token could not be verified'):
            await flow.exchange_callback(callback(flow))


@pytest.mark.parametrize(
    'changes',
    [
        {'scope': 'openid'},
        {'token_type': 'MAC'},
        {'expires_in': 0},
        {'refresh_token': ''},
        {'id_token': ''},
        {'expires_in': 1e300},
    ],
)
async def test_invalid_grant_response(signing_key: rsa.RSAPrivateKey, changes: dict[str, Any]):
    issuer = Issuer(signing_key)
    issuer.token_changes = changes
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(issuer)) as client:
        flow = OpenAIChatGPTOAuthFlow(ext_agent_host_id='host', agent_name='App', http_client=client)
        issuer.nonce = flow.nonce
        with pytest.raises(ModelAPIError):
            await flow.exchange_callback(callback(flow))


async def test_invalid_signing_key_and_discovery(signing_key: rsa.RSAPrivateKey):
    issuer = Issuer(signing_key)
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(issuer)) as client:
        for mode in ('bad_signature', 'issuer', 'jwks', 'discovery_status', 'keys_status'):
            issuer.bad_signature = mode == 'bad_signature'
            issuer.discovery_status = 503 if mode == 'discovery_status' else 200
            issuer.keys_status = 503 if mode == 'keys_status' else 200
            issuer.discovery_changes = (
                {'issuer': 'wrong'}
                if mode == 'issuer'
                else {'jwks_uri': 'https://evil.example/keys'}
                if mode == 'jwks'
                else {}
            )
            flow = OpenAIChatGPTOAuthFlow(ext_agent_host_id='host', agent_name='App', http_client=client)
            issuer.nonce = flow.nonce
            with pytest.raises(ModelAPIError):
                await flow.exchange_callback(callback(flow))


class MemorySource:
    def __init__(self, value: OpenAIChatGPTCredentials):
        self.value = value
        self.lock = anyio.Lock()
        self.saved: list[OpenAIChatGPTCredentials] = []
        self.fail_save = False

    async def load(self) -> OpenAIChatGPTCredentials:
        return self.value

    async def rotate(
        self,
        expected: OpenAIChatGPTCredentials,
        refresh: Callable[[OpenAIChatGPTCredentials], Awaitable[OpenAIChatGPTCredentials]],
    ) -> OpenAIChatGPTCredentials:
        async with self.lock:
            if self.value != expected:
                return self.value
            updated = await refresh(expected)
            if self.fail_save:
                raise RuntimeError('Could not persist rotation')
            self.saved.append(updated)
            self.value = updated
            return updated


@pytest.mark.parametrize('mode', ['memory', 'source'])
@pytest.mark.parametrize('expired', [False, True])
async def test_single_flight_refresh(signing_key: rsa.RSAPrivateKey, mode: str, expired: bool):
    issuer = Issuer(signing_key)
    issuer.release_refresh = anyio.Event()
    old = credentials(-1 if expired else 3600)
    source = MemorySource(old)
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(issuer)) as client:
        provider = OpenAIChatGPTProvider(
            old if mode == 'memory' else None,
            credential_source=source if mode == 'source' else None,
            http_client=client,
        )
        async with anyio.create_task_group() as group:

            async def request() -> None:
                response = await client.post(RESOURCE + '/responses', json={'input': []})
                assert response.status_code == 200

            for _ in range(5):
                group.start_soon(request)
            with anyio.fail_after(10):
                await issuer.refresh_started.wait()
            issuer.release_refresh.set()
        assert len(issuer.forms) == 1
        assert issuer.forms[0] == {
            'grant_type': ['refresh_token'],
            'client_id': ['oaiapp_test'],
            'refresh_token': ['synthetic-refresh'],
            'resource': [RESOURCE],
        }
        assert provider.credentials.refresh_token == 'new-refresh'
        assert len(source.saved) == (1 if mode == 'source' else 0)


async def test_multiple_providers_serialize_source(signing_key: rsa.RSAPrivateKey):
    issuer = Issuer(signing_key)
    source = MemorySource(credentials(-1))
    async with (
        httpx2.AsyncClient(transport=httpx2.MockTransport(issuer)) as first,
        httpx2.AsyncClient(transport=httpx2.MockTransport(issuer)) as second,
    ):
        one = OpenAIChatGPTProvider(credential_source=source, http_client=first)
        two = OpenAIChatGPTProvider(credential_source=source, http_client=second)
        with pytest.raises(UserError, match='not been loaded'):
            _ = one.credentials
        async with anyio.create_task_group() as group:
            group.start_soon(first.get, RESOURCE + '/models')
            group.start_soon(second.get, RESOURCE + '/models')
        assert len(issuer.forms) == 1
        assert one.credentials == two.credentials == source.value


@pytest.mark.parametrize('mode', ['exchange', 'persist', 'always_401'])
async def test_refresh_failures_are_not_replayed(signing_key: rsa.RSAPrivateKey, mode: str):
    issuer = Issuer(signing_key)
    issuer.status = 400 if mode == 'exchange' else 200
    issuer.always_401 = mode == 'always_401'
    source = MemorySource(credentials())
    source.fail_save = mode == 'persist'
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(issuer)) as client:
        OpenAIChatGPTProvider(credential_source=source, http_client=client)
        if mode == 'always_401':
            assert (await client.post(RESOURCE + '/responses')).status_code == 401
            assert issuer.responses == 2
        else:
            for _ in range(2):
                with pytest.raises((ModelAPIError, RuntimeError)) as error:
                    await client.post(RESOURCE + '/responses')
                assert 'synthetic-secret' not in str(error.value)
        assert len(issuer.forms) == 1


async def test_earliest_refresh_time(signing_key: rsa.RSAPrivateKey):
    issuer = Issuer(signing_key)
    value = replace(credentials(10), earliest_refresh_at=datetime.now(timezone.utc) + timedelta(hours=1))
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(issuer)) as client:
        OpenAIChatGPTProvider(value, http_client=client)
        assert (await client.get(RESOURCE + '/models')).status_code == 200
        with pytest.raises(ModelAPIError, match='not yet allowed'):
            await client.post(RESOURCE + '/responses')
        assert not issuer.forms


@pytest.mark.parametrize('url', ['https://other.example/models', 'http://api.openai.com/v1/models'])
async def test_token_destination_boundary(url: str):
    captured: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        captured.append(request)
        return httpx2.Response(200)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        OpenAIChatGPTProvider(credentials(), http_client=client)
        await client.get(url)
    assert 'authorization' not in captured[0].headers


def test_explicit_credentials_and_prefix():
    with pytest.raises(UserError, match='exactly one'):
        OpenAIChatGPTProvider()
    with pytest.raises(UserError, match='exactly one'):
        OpenAIChatGPTProvider(credentials(), credential_source=MemorySource(credentials()))
    assert infer_provider_class('openai-chatgpt') is OpenAIChatGPTProvider
    provider = OpenAIChatGPTProvider(credentials())
    assert isinstance(
        infer_model('openai-chatgpt:gpt-6.1-sol', provider_factory=lambda _: provider), OpenAIChatGPTModel
    )
    assert provider.name == 'openai-chatgpt'
    assert provider.base_url == RESOURCE


async def test_provider_client_lifecycle():
    provider = OpenAIChatGPTProvider(credentials())
    async with provider:
        first = provider.client._client  # pyright: ignore[reportPrivateUsage]
        assert first.auth is not None
    assert first.is_closed
    async with provider:
        second = provider.client._client  # pyright: ignore[reportPrivateUsage]
        assert second is not first and second.auth is not None


async def test_real_loopback_callback(signing_key: rsa.RSAPrivateKey, monkeypatch: pytest.MonkeyPatch):
    ready = threading.Event()

    class ReadyServer(HTTPServer):
        def server_activate(self) -> None:
            super().server_activate()
            ready.set()

    monkeypatch.setattr('pydantic_ai.providers._oauth.HTTPServer', ReadyServer)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    issuer = Issuer(signing_key)
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(issuer)) as client:
        flow = OpenAIChatGPTOAuthFlow(
            ext_agent_host_id='host',
            agent_name='App',
            http_client=client,
            redirect_uri=f'http://127.0.0.1:{port}/auth/callback',
        )
        issuer.nonce = flow.nonce
        results: list[OpenAIChatGPTCredentials] = []

        async def exchange() -> None:
            results.append(await flow.exchange_code_from_callback())

        async with anyio.create_task_group() as group:
            group.start_soon(exchange)
            assert await anyio.to_thread.run_sync(ready.wait, 10)
            async with httpx2.AsyncClient() as browser:
                await browser.get(callback(flow, state='stray'))
                await browser.get(callback(flow))
        assert results[0].client_id == 'oaiapp_test'


@pytest.mark.parametrize('omit', [{'id_token'}, {'scope'}, {'id_token', 'scope'}])
async def test_refresh_optional_identity_and_scopes(signing_key: rsa.RSAPrivateKey, omit: set[str]):
    issuer = Issuer(signing_key)
    issuer.omit = omit
    old = credentials(-1)
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(issuer)) as client:
        provider = OpenAIChatGPTProvider(old, http_client=client)
        await client.get(RESOURCE + '/models')
        assert provider.credentials.subject == old.subject
        assert provider.credentials.scopes == old.scopes
        assert provider.credentials.refresh_token == 'new-refresh'
        if 'id_token' in omit:
            assert provider.credentials.id_token == old.id_token


async def test_cancelled_refresh_is_not_spent_twice(signing_key: rsa.RSAPrivateKey):
    issuer = Issuer(signing_key)
    issuer.release_refresh = anyio.Event()
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(issuer)) as client:
        OpenAIChatGPTProvider(credentials(-1), http_client=client)
        async with anyio.create_task_group() as group:
            group.start_soon(client.get, RESOURCE + '/models')
            with anyio.fail_after(10):
                await issuer.refresh_started.wait()
            group.cancel_scope.cancel()
        with pytest.raises(ModelAPIError, match='outcome is unknown'):
            await client.get(RESOURCE + '/models')
        assert len(issuer.forms) == 1


@pytest.mark.parametrize('nonce', [None, 'original', 'wrong'])
async def test_refresh_nonce(signing_key: rsa.RSAPrivateKey, nonce: str | None):
    """OIDC permits omission on refresh, but a returned nonce must match the original sign-in."""
    issuer = Issuer(signing_key)
    if nonce is not None:
        issuer.claims['nonce'] = nonce
    old = replace(credentials(-1), nonce='original')
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(issuer)) as client:
        provider = OpenAIChatGPTProvider(old, http_client=client)
        if nonce == 'wrong':
            with pytest.raises(ModelAPIError, match='ID token could not be verified'):
                await client.get(RESOURCE + '/models')
        else:
            await client.get(RESOURCE + '/models')
            assert provider.credentials.nonce == 'original'


@pytest.mark.parametrize('missing', ['id_token', 'scope'])
async def test_initial_grant_requires_identity_and_scopes(signing_key: rsa.RSAPrivateKey, missing: str):
    issuer = Issuer(signing_key)
    issuer.omit.add(missing)
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(issuer)) as client:
        flow = OpenAIChatGPTOAuthFlow(ext_agent_host_id='host', agent_name='App', http_client=client)
        issuer.nonce = flow.nonce
        with pytest.raises(ModelAPIError, match='did not return'):
            await flow.exchange_callback(callback(flow))


@pytest.mark.parametrize('client_id', [None, 'oaiapp_test', 'other'])
async def test_returning_callback_client_id(signing_key: rsa.RSAPrivateKey, client_id: str | None):
    issuer = Issuer(signing_key)
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(issuer)) as client:
        flow = OpenAIChatGPTOAuthFlow(
            ext_agent_host_id='host', agent_name='App', credentials=credentials(), http_client=client
        )
        issuer.nonce = flow.nonce
        if client_id is None:
            result = await flow.exchange_code('code')
        elif client_id == 'other':
            with pytest.raises(UserError, match='changed the selected'):
                await flow.exchange_callback(callback(flow, client_id=client_id))
            assert not issuer.forms
            return
        else:
            result = await flow.exchange_callback(callback(flow, client_id=client_id))
        assert result.client_id == 'oaiapp_test'
        assert result.nonce == flow.nonce


@pytest.mark.parametrize('field', ['ext_agent_host_id', 'agent_name'])
def test_empty_registration_config(field: str):
    with pytest.raises(UserError, match='must not be empty'):
        OpenAIChatGPTOAuthFlow(
            ext_agent_host_id='' if field == 'ext_agent_host_id' else 'host',
            agent_name='' if field == 'agent_name' else 'App',
        )


@pytest.mark.parametrize(
    'config',
    [
        {'client_id': ''},
        {'client_id': 'dynamic_agent_client'},
        {'redirect_uri': 'http://app.example/callback'},
        {'redirect_uri': 'https://user@app.example/callback'},
        {'redirect_uri': 'https://app.example/callback#fragment'},
        {'client_secret': 'secret'},
        {'token_endpoint_auth_method': 'client_secret_basic'},
    ],
)
def test_invalid_provisioned_config(config: dict[str, Any]):
    values: dict[str, Any] = {'client_id': 'oaiapp_test', 'redirect_uri': 'https://app.example/callback', **config}
    with pytest.raises(UserError):
        OpenAIChatGPTClient(**values)


async def test_callback_target_and_expiry():
    flow = OpenAIChatGPTOAuthFlow(ext_agent_host_id='host', agent_name='App')
    with pytest.raises(UserError, match='does not match'):
        await flow.exchange_callback(callback(flow).replace('127.0.0.1', 'localhost'))
    flow._expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(UserError, match='expired'):
        await flow.exchange_callback(callback(flow))


async def test_provider_rejects_invalid_grant_and_client():
    config = OpenAIChatGPTClient(client_id='other', redirect_uri='https://app.example/callback')
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda _: httpx2.Response(200))) as client:
        OpenAIChatGPTProvider(credentials(), client=config, http_client=client)
        with pytest.raises(UserError, match='another ChatGPT client'):
            await client.get(RESOURCE + '/models')
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda _: httpx2.Response(200))) as client:
        OpenAIChatGPTProvider(replace(credentials(), scopes=('openid',)), http_client=client)
        with pytest.raises(UserError, match='do not grant plan usage'):
            await client.get(RESOURCE + '/models')
    with pytest.raises(UserError, match='another ChatGPT client'):
        OpenAIChatGPTOAuthFlow(ext_agent_host_id='host', agent_name='App', credentials=credentials(), client=config)


async def test_dedicated_async_client_required():
    async with httpx2.AsyncClient(auth=httpx2.BasicAuth('user', 'password')) as client:
        with pytest.raises(UserError, match='without existing auth'):
            OpenAIChatGPTProvider(credentials(), http_client=client)
    provider = OpenAIChatGPTProvider(credentials())
    async with provider:
        auth = provider.client._client.auth  # pyright: ignore[reportPrivateUsage]
        assert isinstance(auth, httpx2.Auth)
        with pytest.raises(UserError, match='async HTTP client'):
            next(auth.sync_auth_flow(httpx2.Request('GET', RESOURCE + '/models')))


async def test_source_cannot_switch_registration(signing_key: rsa.RSAPrivateKey):
    issuer = Issuer(signing_key)
    source = MemorySource(credentials(-1))
    source.value = replace(source.value, subject='other')
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(issuer)) as client:
        provider = OpenAIChatGPTProvider(credential_source=source, http_client=client)
        # Simulate a second process replacing the selected account while rotation is pending.
        provider._credentials = credentials(-1)  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(UserError, match='changed the selected registration'):
            await client.get(RESOURCE + '/models')
        assert not issuer.forms


@pytest.mark.parametrize('azp', [None, 'oaiapp_test'])
async def test_multiple_id_token_audiences(signing_key: rsa.RSAPrivateKey, azp: str | None):
    issuer = Issuer(signing_key)
    issuer.claims = {'aud': ['oaiapp_test', 'other'], 'azp': azp}
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(issuer)) as client:
        flow = OpenAIChatGPTOAuthFlow(ext_agent_host_id='host', agent_name='App', http_client=client)
        issuer.nonce = flow.nonce
        if azp is None:
            with pytest.raises(ModelAPIError, match='ID token could not be verified'):
                await flow.exchange_callback(callback(flow))
        else:
            assert (await flow.exchange_callback(callback(flow))).subject == 'subject'


def test_authorization_custom_scope_and_account_hint():
    flow = OpenAIChatGPTOAuthFlow(
        ext_agent_host_id='host', agent_name='App', credentials=replace(credentials(), email='test@example.com')
    )
    params = parse_qs(urlsplit(flow.authorization_url(scope='openid', extra_params={'prompt': 'login'})).query)
    assert params['login_hint'] == ['test@example.com']
    assert params['scope'] == ['openid']
    assert params['prompt'] == ['login']


async def test_oauth_owned_http_client(signing_key: rsa.RSAPrivateKey, monkeypatch: pytest.MonkeyPatch):
    issuer = Issuer(signing_key)
    clients: list[httpx2.AsyncClient] = []

    def create_client() -> httpx2.AsyncClient:
        client = httpx2.AsyncClient(transport=httpx2.MockTransport(issuer))
        clients.append(client)
        return client

    monkeypatch.setattr('pydantic_ai.providers._openai_chatgpt_oauth.create_async_httpx2_client', create_client)
    flow = OpenAIChatGPTOAuthFlow(ext_agent_host_id='host', agent_name='App')
    issuer.nonce = flow.nonce
    assert (await flow.exchange_callback(callback(flow))).client_id == 'oaiapp_test'
    assert len(clients) == 3
    assert all(client.is_closed for client in clients)


async def test_source_recovers_before_refresh_dispatch(signing_key: rsa.RSAPrivateKey, monkeypatch: pytest.MonkeyPatch):
    issuer = Issuer(signing_key)
    source = MemorySource(credentials(-1))
    rotate = source.rotate
    calls = 0

    async def unavailable_once(
        expected: OpenAIChatGPTCredentials,
        refresh: Callable[[OpenAIChatGPTCredentials], Awaitable[OpenAIChatGPTCredentials]],
    ) -> OpenAIChatGPTCredentials:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError('Storage temporarily unavailable')
        return await rotate(expected, refresh)

    monkeypatch.setattr(source, 'rotate', unavailable_once)
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(issuer)) as client:
        provider = OpenAIChatGPTProvider(credential_source=source, http_client=client)
        with pytest.raises(RuntimeError, match='temporarily unavailable'):
            await client.get(RESOURCE + '/models')
        assert not issuer.forms
        await client.get(RESOURCE + '/models')
        assert calls == 2
        assert len(issuer.forms) == 1
        assert provider.credentials == source.value


@pytest.mark.parametrize('mode', ['code', 'callback', 'changed_query'])
async def test_provisioned_callback_query(signing_key: rsa.RSAPrivateKey, mode: str):
    issuer = Issuer(signing_key)
    config = OpenAIChatGPTClient(client_id='oaiapp_test', redirect_uri='https://app.example/callback?tenant=demo')
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(issuer)) as client:
        flow = OpenAIChatGPTOAuthFlow(ext_agent_host_id='host', agent_name='App', client=config, http_client=client)
        issuer.nonce = flow.nonce
        callback_url = config.redirect_uri + '&' + urlencode({'state': flow.state, 'code': 'code'})
        if mode == 'changed_query':
            with pytest.raises(UserError, match='does not match'):
                await flow.exchange_callback(callback_url.replace('tenant=demo', 'tenant=other'))
            assert not issuer.forms
            return
        result = await flow.exchange_code('code') if mode == 'code' else await flow.exchange_callback(callback_url)
        assert result.redirect_uri == config.redirect_uri
        assert issuer.forms[0]['redirect_uri'] == [config.redirect_uri]
