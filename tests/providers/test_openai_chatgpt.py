"""OpenAI ChatGPT (Sign in with ChatGPT) provider tests: OAuth flow, credential lifecycle, and wire dialect.

These are unit tests by necessity, not omission: signing in needs a browser and a real ChatGPT account, and
recording against the token endpoint would spend (and rotate) a live grant. The shared refresh lifecycle
(single flight, 401 replay, credential sources) is covered through the Codex provider in
`tests/providers/codex/`; these tests cover what differs for ChatGPT plan usage.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable
from functools import partial
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import httpx2
import pytest
from pydantic import JsonValue

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import infer_model
from pydantic_ai.providers import infer_provider_class

from .._inline_snapshot import snapshot
from ..conftest import try_import
from .oauth_helpers import free_port, get_callback, make_jwt

with try_import() as imports_successful:
    from openai import AsyncOpenAI

    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers._oauth_provider import post_token_request
    from pydantic_ai.providers.openai_chatgpt import (
        CredentialsPersistenceError,
        CredentialsRefreshError,
        OpenAIChatGPTCredentials,
        OpenAIChatGPTOAuthFlow,
        OpenAIChatGPTProvider,
    )

pytestmark = [pytest.mark.skipif(not imports_successful(), reason='OpenAI client not installed')]

CLIENT_ID = 'oaiapp_issued'
HOST_ID = 'urn:uuid:6f1c3a52-2b0e-4b0f-9a35-6d2e4f1a7b9c'
PLAN_SCOPE = 'chatgpt.tokens.use.direct email offline_access openid profile resource.invoke'
TOKEN_URL = 'https://auth.openai.com/api/accounts/oauth/token'


def id_token(nonce: str, /, **overrides: JsonValue) -> str:
    claims: dict[str, JsonValue] = {
        'iss': 'https://auth.openai.com',
        'aud': CLIENT_ID,
        'exp': time.time() + 3600,
        'sub': 'user-1',
        'nonce': nonce,
        **overrides,
    }
    return make_jwt({key: value for key, value in claims.items() if value is not None})


def make_credentials(*, scope: str = PLAN_SCOPE, access_token: str = 'access-1') -> OpenAIChatGPTCredentials:
    return OpenAIChatGPTCredentials(
        client_id=CLIENT_ID, access_token=access_token, refresh_token='refresh-1', id_token='id-1', scope=scope
    )


class TokenEndpoint:
    """A token endpoint answering through the real `post_token_request`, recording the forms it receives."""

    def __init__(self) -> None:
        self.forms: list[dict[str, str]] = []
        self.respond: Callable[[dict[str, str]], httpx2.Response] | None = None

    def __call__(self, respond: Callable[[dict[str, str]], httpx2.Response]) -> TokenEndpoint:
        self.respond = respond
        return self

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        assert str(request.url) == TOKEN_URL
        form = dict(parse_qsl(request.content.decode()))
        self.forms.append(form)
        assert self.respond is not None, 'call the `token_endpoint` fixture with a responder first'
        return self.respond(form)


@pytest.fixture
async def token_endpoint(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[TokenEndpoint]:
    endpoint = TokenEndpoint()
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(endpoint.handle)) as client:
        monkeypatch.setattr(
            'pydantic_ai.providers.openai_chatgpt._post_token_request',
            partial(post_token_request, http_client=client),
        )
        yield endpoint


def token_response(
    flow: OpenAIChatGPTOAuthFlow, *, granted_scope: str | None = PLAN_SCOPE, claims: dict[str, JsonValue] | None = None
):
    def respond(form: dict[str, str]) -> httpx2.Response:
        body: dict[str, JsonValue] = {
            'access_token': 'access-1',
            'refresh_token': 'refresh-1',
            'id_token': id_token(flow.nonce, **(claims or {})),
            'token_type': 'Bearer',
            'expires_in': 3600,
        }
        if granted_scope is not None:
            body['scope'] = granted_scope
        return httpx2.Response(200, json=body)

    return respond


# --- Authorization URL ---


def test_authorization_url_registers_a_new_client():
    flow = OpenAIChatGPTOAuthFlow(app_name='My App', host_id=HOST_ID)
    url = urlsplit(flow.authorization_url())

    assert url._replace(query='').geturl() == 'https://auth.openai.com/api/accounts/authorize'
    params = dict(parse_qsl(url.query))
    assert (params.pop('state'), params.pop('nonce'), params.pop('code_challenge')) == (
        flow.state,
        flow.nonce,
        flow.code_challenge,
    )
    assert params == snapshot(
        {
            'response_type': 'code',
            'client_id': 'dynamic_agent_client',
            'redirect_uri': 'http://127.0.0.1:1455/auth/callback',
            'scope': 'openid profile email offline_access resource.invoke chatgpt.tokens.use.direct',
            'resource': 'https://api.openai.com/v1',
            'code_challenge_method': 'S256',
            'agent_name_hint': 'My App',
            'ext_agent_host_id': 'urn:uuid:6f1c3a52-2b0e-4b0f-9a35-6d2e4f1a7b9c',
        }
    )


def test_authorization_url_reuses_a_registration():
    """Signing in again sends the issued client id and the ID token hint, not the registration's name."""
    hint = make_jwt({'sub': 'user-1'})
    flow = OpenAIChatGPTOAuthFlow(app_name='My App', client_id=CLIENT_ID, id_token_hint=hint)
    params = dict(parse_qsl(urlsplit(flow.authorization_url(extra_params={'prompt': 'consent'})).query))

    assert params['client_id'] == CLIENT_ID
    assert params['id_token_hint'] == hint
    assert params['prompt'] == 'consent'
    assert 'agent_name_hint' not in params
    assert 'ext_agent_host_id' not in params


@pytest.mark.parametrize('param', ['nonce', 'resource'])
def test_extra_params_cannot_override_what_the_exchange_relies_on(param: str):
    flow = OpenAIChatGPTOAuthFlow(app_name='My App')
    with pytest.raises(UserError, match=f'cannot override {param}'):
        flow.authorization_url(extra_params={param: 'other'})


@pytest.mark.parametrize(
    'redirect_uri',
    [
        'http://localhost:1455/auth/callback',
        'https://127.0.0.1:1455/auth/callback',
        'http://127.0.0.1:1455/callback',
        'http://127.0.0.1/auth/callback',
        'http://127.0.0.1:0/auth/callback',
        'http://127.0.0.1:abc/auth/callback',
        'http://127.0.0.1:99999/auth/callback',
        'http://user:pass@127.0.0.1:1455/auth/callback',
        'http://127.0.0.1:1455/auth/callback?x=1',
        'http://127.0.0.1:1455/auth/callback#x',
        'HTTP://127.0.0.1:1455/auth/callback',
    ],
)
def test_redirect_uri_must_be_the_loopback_callback(redirect_uri: str):
    with pytest.raises(UserError, match='only lets the port vary'):
        OpenAIChatGPTOAuthFlow(app_name='My App', redirect_uri=redirect_uri)


def test_redirect_uri_port_may_vary():
    flow = OpenAIChatGPTOAuthFlow(app_name='My App', redirect_uri='http://127.0.0.1:54321/auth/callback')
    assert flow.redirect_uri == 'http://127.0.0.1:54321/auth/callback'


def test_host_id_must_be_opaque():
    with pytest.raises(UserError, match='`host_id` must be an opaque'):
        OpenAIChatGPTOAuthFlow(app_name='My App', host_id='user@example.com')


def test_id_token_hint_must_identify_an_account():
    with pytest.raises(UserError, match='`id_token_hint` is not an ID token'):
        OpenAIChatGPTOAuthFlow(app_name='My App', id_token_hint='not-a-jwt')


# --- Code exchange and ID token validation ---


async def test_sign_in_registers_through_the_loopback_callback(token_endpoint: TokenEndpoint):
    """The callback's issued `client_id` is the one exchanged and kept, never `dynamic_agent_client`."""
    url = f'http://127.0.0.1:{free_port()}/auth/callback'
    flow = OpenAIChatGPTOAuthFlow(app_name='My App', host_id=HOST_ID, redirect_uri=url)
    endpoint = token_endpoint(token_response(flow))
    exchange = asyncio.create_task(flow.exchange_code_from_callback())

    accepted = await get_callback(url, {'state': flow.state, 'code': 'the-code', 'client_id': CLIENT_ID})
    assert 'close this tab' in accepted.text
    credentials = await exchange

    assert credentials == OpenAIChatGPTCredentials(
        client_id=CLIENT_ID,
        access_token='access-1',
        refresh_token='refresh-1',
        id_token=credentials.id_token,
        scope=PLAN_SCOPE,
    )
    assert credentials.plan_usage_granted
    assert repr(credentials) == snapshot(
        "OpenAIChatGPTCredentials(client_id='oaiapp_issued', scope='chatgpt.tokens.use.direct email offline_access openid profile resource.invoke')"
    )
    assert endpoint.forms == [
        {
            'grant_type': 'authorization_code',
            'code': 'the-code',
            'code_verifier': flow.code_verifier,
            'redirect_uri': url,
            'client_id': CLIENT_ID,
            'resource': 'https://api.openai.com/v1',
        }
    ]


async def test_failed_exchange_keeps_the_issued_client(token_endpoint: TokenEndpoint):
    """SIWC: after a failed exchange, start again with the client id issued for this attempt."""
    flow = OpenAIChatGPTOAuthFlow(app_name='My App')
    token_endpoint(lambda form: httpx2.Response(400, json={'error': 'invalid_grant'}))

    with pytest.raises(CredentialsRefreshError, match='the grant was rejected'):
        await flow.exchange_code('the-code', client_id=CLIENT_ID)

    assert flow.client_id == CLIENT_ID


async def test_sign_in_again_keeps_the_registration(token_endpoint: TokenEndpoint):
    """A returning sign-in's callback may omit the client id; the saved one is exchanged."""
    flow = OpenAIChatGPTOAuthFlow(app_name='My App', client_id=CLIENT_ID, id_token_hint=make_jwt({'sub': 'user-1'}))
    endpoint = token_endpoint(token_response(flow))

    credentials = await flow.exchange_code('the-code')

    assert credentials.client_id == CLIENT_ID
    assert endpoint.forms[0]['client_id'] == CLIENT_ID


async def test_sign_in_without_plan_usage_keeps_the_registration(token_endpoint: TokenEndpoint):
    """Declining plan usage still returns credentials, so the issued client id can be saved and reused."""
    flow = OpenAIChatGPTOAuthFlow(app_name='My App')
    token_endpoint(token_response(flow, granted_scope='openid profile email offline_access'))

    credentials = await flow.exchange_code('the-code', client_id=CLIENT_ID)

    assert credentials.client_id == CLIENT_ID
    assert not credentials.plan_usage_granted
    with pytest.raises(UserError, match=r"OpenAIChatGPTOAuthFlow\(client_id='oaiapp_issued', \.\.\.\)"):
        OpenAIChatGPTProvider(credentials=credentials)


async def test_sign_in_without_granted_scopes_has_no_plan_usage(token_endpoint: TokenEndpoint):
    flow = OpenAIChatGPTOAuthFlow(app_name='My App')
    token_endpoint(token_response(flow, granted_scope=None))

    credentials = await flow.exchange_code('the-code', client_id=CLIENT_ID)

    assert not credentials.plan_usage_granted


@pytest.mark.parametrize(
    'flow_client_id,callback_client_id,message',
    [
        pytest.param(None, None, 'registration is incomplete', id='registration-without-issued-id'),
        pytest.param(None, 'dynamic_agent_client', 'registration is incomplete', id='registration-entrypoint-id'),
        pytest.param(CLIENT_ID, 'oaiapp_other', "carried client 'oaiapp_other'", id='different-registration'),
    ],
)
async def test_exchange_rejects_an_unusable_client_id(
    flow_client_id: str | None, callback_client_id: str | None, message: str
):
    flow = OpenAIChatGPTOAuthFlow(app_name='My App', client_id=flow_client_id)
    with pytest.raises(UserError, match=message):
        await flow.exchange_code('the-code', client_id=callback_client_id)


@pytest.mark.parametrize(
    'claims,message',
    [
        pytest.param({'iss': 'https://evil.example'}, "issued by 'https://evil.example'", id='issuer'),
        pytest.param({'aud': 'oaiapp_other'}, "not issued for client 'oaiapp_issued'", id='audience'),
        pytest.param({'aud': None}, "not issued for client 'oaiapp_issued'", id='no-audience'),
        pytest.param({'exp': time.time() - 3600}, 'has expired', id='expired'),
        pytest.param({'exp': time.time() - 120}, 'has expired', id='expired-beyond-clock-skew'),
        pytest.param({'exp': None}, 'has expired', id='no-expiry'),
        pytest.param({'exp': 10**14}, 'has expired', id='absurd-expiry'),
        pytest.param({'nonce': 'other-attempt'}, 'nonce', id='nonce'),
        pytest.param({'sub': None}, 'does not identify an account', id='no-subject'),
    ],
)
async def test_exchange_validates_the_id_token(
    token_endpoint: TokenEndpoint, claims: dict[str, JsonValue], message: str
):
    """An ID token that fails validation is a failed sign-in, reported like any token endpoint failure."""
    flow = OpenAIChatGPTOAuthFlow(app_name='My App', client_id=CLIENT_ID)
    token_endpoint(token_response(flow, claims=claims))

    with pytest.raises(CredentialsRefreshError, match=message):
        await flow.exchange_code('the-code')


async def test_exchange_rejects_a_different_account(token_endpoint: TokenEndpoint):
    flow = OpenAIChatGPTOAuthFlow(app_name='My App', client_id=CLIENT_ID, id_token_hint=make_jwt({'sub': 'user-1'}))
    token_endpoint(token_response(flow, claims={'sub': 'user-2'}))

    with pytest.raises(UserError, match='different ChatGPT account'):
        await flow.exchange_code('the-code')


async def test_exchange_accepts_an_audience_list(token_endpoint: TokenEndpoint):
    flow = OpenAIChatGPTOAuthFlow(app_name='My App', client_id=CLIENT_ID)
    token_endpoint(token_response(flow, claims={'aud': [CLIENT_ID, 'https://api.openai.com/v1']}))

    assert (await flow.exchange_code('the-code')).client_id == CLIENT_ID


async def test_exchange_tolerates_small_clock_skew(token_endpoint: TokenEndpoint):
    flow = OpenAIChatGPTOAuthFlow(app_name='My App', client_id=CLIENT_ID)
    token_endpoint(token_response(flow, claims={'exp': time.time() - 30}))

    assert (await flow.exchange_code('the-code')).client_id == CLIENT_ID


@pytest.mark.parametrize(
    'body,message',
    [
        pytest.param({'id_token': None}, 'carried no ID token', id='missing'),
        pytest.param({'id_token': 'not-a-jwt'}, 'malformed ID token', id='malformed'),
    ],
)
async def test_exchange_requires_an_id_token(token_endpoint: TokenEndpoint, body: dict[str, Any], message: str):
    payload = {'access_token': 'access-1', 'refresh_token': 'refresh-1', 'scope': PLAN_SCOPE, **body}
    token_endpoint(lambda form: httpx2.Response(200, json=payload))
    flow = OpenAIChatGPTOAuthFlow(app_name='My App', client_id=CLIENT_ID)

    with pytest.raises(CredentialsRefreshError, match=message):
        await flow.exchange_code('the-code')


# --- Provider ---


def test_provider_requires_credentials():
    with pytest.raises(UserError, match='sign in with `OpenAIChatGPTOAuthFlow` first'):
        OpenAIChatGPTProvider()


def test_provider_wraps_an_existing_client():
    client = AsyncOpenAI(api_key='unused')
    provider = OpenAIChatGPTProvider(openai_client=client)
    assert provider.client is client
    assert (provider.name, provider.base_url) == ('openai-chatgpt', 'https://api.openai.com/v1')


class CredentialSource:
    def __init__(self, credentials: OpenAIChatGPTCredentials):
        self.credentials = credentials
        self.saved: list[OpenAIChatGPTCredentials] = []

    async def load(self) -> OpenAIChatGPTCredentials:
        return self.credentials

    async def save(self, credentials: OpenAIChatGPTCredentials) -> None:
        self.credentials = credentials
        self.saved.append(credentials)


async def test_provider_refuses_stored_credentials_without_plan_usage():
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: httpx2.Response(200))) as client:
        OpenAIChatGPTProvider(credential_source=CredentialSource(make_credentials(scope='openid')), http_client=client)
        with pytest.raises(UserError, match='do not grant ChatGPT plan usage'):
            await client.get('https://api.openai.com/v1/models')


@pytest.mark.parametrize(
    'refreshed,expected_id_token,expected_scope',
    [
        pytest.param({}, 'id-1', PLAN_SCOPE, id='grant-kept'),
        pytest.param(
            {'id_token': 'id-2', 'scope': 'chatgpt.tokens.use.direct'},
            'id-2',
            'chatgpt.tokens.use.direct',
            id='grant-replaced',
        ),
    ],
)
async def test_provider_refreshes_against_the_issued_client(
    refreshed: dict[str, str], expected_id_token: str, expected_scope: str
):
    """A 401 refreshes with the issued client id and resource, and the rotated set is saved."""
    forms: list[dict[str, str]] = []
    seen: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.host == 'auth.openai.com':
            assert str(request.url) == TOKEN_URL
            forms.append(dict(parse_qsl(request.content.decode())))
            return httpx2.Response(200, json={'access_token': 'access-2', 'refresh_token': 'refresh-2', **refreshed})
        seen.append(request.headers['authorization'])
        return httpx2.Response(401 if request.headers['authorization'] == 'Bearer access-1' else 200)

    source = CredentialSource(make_credentials())
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        OpenAIChatGPTProvider(credential_source=source, http_client=client)
        response = await client.get('https://api.openai.com/v1/models')

    assert response.status_code == 200
    assert seen == ['Bearer access-1', 'Bearer access-2']
    assert forms == [
        {
            'grant_type': 'refresh_token',
            'refresh_token': 'refresh-1',
            'client_id': CLIENT_ID,
            'resource': 'https://api.openai.com/v1',
        }
    ]
    assert source.saved == [
        OpenAIChatGPTCredentials(
            client_id=CLIENT_ID,
            access_token='access-2',
            refresh_token='refresh-2',
            id_token=expected_id_token,
            scope=expected_scope,
        )
    ]


@pytest.mark.parametrize('stored_elsewhere', [False, True], ids=['refreshed', 'adopted-from-source'])
async def test_provider_refuses_credentials_that_lose_plan_usage(stored_elsewhere: bool):
    """A refresh that no longer grants plan usage is still saved, since its refresh token is the only usable one."""
    without_plan_usage = make_credentials(scope='openid', access_token='access-2')

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.host == 'auth.openai.com':
            return httpx2.Response(
                200, json={'access_token': 'access-2', 'refresh_token': 'refresh-1', 'scope': 'openid'}
            )
        return httpx2.Response(401)

    class Source(CredentialSource):
        async def load(self) -> OpenAIChatGPTCredentials:
            loaded = self.credentials
            if stored_elsewhere:
                self.credentials = without_plan_usage  # another process refreshes after this load
            return loaded

    source = Source(make_credentials())
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        OpenAIChatGPTProvider(credential_source=source, http_client=client)
        with pytest.raises(UserError, match='do not grant ChatGPT plan usage'):
            await client.get('https://api.openai.com/v1/models')

    assert source.saved == ([] if stored_elsewhere else [without_plan_usage])


@pytest.mark.parametrize(
    'error',
    [
        'invalid_grant',
        'invalid_refresh_token',
        'token_expired',
        'refresh_token_expired',
        'refresh_token_invalidated',
        'refresh_token_reused',
    ],
)
async def test_unusable_refresh_tokens_say_to_sign_in_again(error: str):
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.host == 'auth.openai.com':
            return httpx2.Response(400, json={'error': error})
        return httpx2.Response(401)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        OpenAIChatGPTProvider(credentials=make_credentials(), http_client=client)
        with pytest.raises(CredentialsRefreshError, match='the grant was rejected, rerun the authorization flow'):
            await client.get('https://api.openai.com/v1/models')


async def test_provider_stops_using_credentials_that_lost_plan_usage():
    """A proactive refresh that loses plan usage is refused before the API, and later requests spend nothing."""
    token_posts: list[str] = []
    api_requests: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.host == 'auth.openai.com':
            token_posts.append(str(request.url))
            return httpx2.Response(
                200, json={'access_token': 'access-2', 'refresh_token': 'refresh-2', 'scope': 'openid'}
            )
        api_requests.append(str(request.url))  # pragma: no cover
        return httpx2.Response(200)  # pragma: no cover

    expired = make_credentials(access_token=make_jwt({'exp': time.time() - 3600}))
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        OpenAIChatGPTProvider(credentials=expired, http_client=client)
        for _ in range(2):
            with pytest.raises(UserError, match='do not grant ChatGPT plan usage'):
                await client.get('https://api.openai.com/v1/models')

    assert len(token_posts) == 1
    assert api_requests == []


async def test_provider_keeps_an_unsaved_rotation_over_the_set_it_spent():
    """A refresh whose save failed keeps its replacement over the stored set that refresh spent."""
    grants: list[str] = []
    rotations = iter(['2', '3'])

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.host == 'auth.openai.com':
            grants.append(dict(parse_qsl(request.content.decode()))['refresh_token'])
            n = next(rotations)
            return httpx2.Response(200, json={'access_token': f'access-{n}', 'refresh_token': f'refresh-{n}'})
        return httpx2.Response(200 if request.headers['authorization'] == 'Bearer access-3' else 401)

    class FlakySource(CredentialSource):
        failing = True

        async def save(self, credentials: OpenAIChatGPTCredentials) -> None:
            if self.failing:
                raise OSError('disk full')
            await super().save(credentials)

    source = FlakySource(make_credentials(access_token=make_jwt({'exp': time.time() - 3600})))
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        provider = OpenAIChatGPTProvider(credential_source=source, http_client=client)
        with pytest.raises(CredentialsPersistenceError):
            await client.get('https://api.openai.com/v1/models')
        # Storage recovers while it still holds the spent `refresh-1`; the next 401 refreshes `refresh-2`.
        source.failing = False
        assert (await client.get('https://api.openai.com/v1/models')).status_code == 200

    assert grants == ['refresh-1', 'refresh-2']
    assert provider.credentials.refresh_token == source.credentials.refresh_token == 'refresh-3'


async def test_provider_sends_credentials_only_to_the_openai_api():
    seen: dict[str, str | None] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen[str(request.url)] = request.headers.get('authorization')
        return httpx2.Response(200)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        OpenAIChatGPTProvider(credentials=make_credentials(), http_client=client)
        await client.get('https://api.openai.com/v1/models')
        await client.get('https://example.com/')
        await client.get('http://api.openai.com/v1/models')

    assert seen == {
        'https://api.openai.com/v1/models': 'Bearer access-1',
        'https://example.com/': None,
        'http://api.openai.com/v1/models': None,
    }


# --- Prefix inference and wire dialect ---


def test_provider_class_inference():
    assert infer_provider_class('openai-chatgpt') is OpenAIChatGPTProvider


def test_prefix_infers_a_responses_model():
    provider = OpenAIChatGPTProvider(credentials=make_credentials())
    model = infer_model('openai-chatgpt:gpt-6.1-sol', provider_factory=lambda name: provider)
    assert isinstance(model, OpenAIResponsesModel)
    assert model.system == 'openai-chatgpt'


def test_prefix_without_credentials_explains_how_to_sign_in():
    with pytest.raises(UserError, match='sign in with `OpenAIChatGPTOAuthFlow` first'):
        Agent('openai-chatgpt:gpt-6.1-sol')


def _sse(*events: dict[str, Any]) -> bytes:
    return b''.join(
        f'event: {event["type"]}\ndata: {json.dumps({**event, "sequence_number": number})}\n\n'.encode()
        for number, event in enumerate(events)
    )


def _completed(response_id: str, output: list[dict[str, Any]]) -> bytes:
    response = {
        'id': response_id,
        'object': 'response',
        'created_at': 0,
        'model': 'gpt-6.1-sol',
        'status': 'completed',
        'output': output,
        'parallel_tool_calls': True,
        'tool_choice': 'auto',
        'tools': [],
        'usage': {
            'input_tokens': 10,
            'output_tokens': 5,
            'total_tokens': 15,
            'input_tokens_details': {'cached_tokens': 0},
            'output_tokens_details': {'reasoning_tokens': 0},
        },
    }
    items: list[dict[str, Any]] = []
    for index, item in enumerate(output):
        items.append({'type': 'response.output_item.added', 'output_index': index, 'item': item})
        if item['type'] == 'message':
            delta: dict[str, Any] = {'item_id': item['id'], 'output_index': index, 'content_index': 0, 'logprobs': []}
            items.append({'type': 'response.output_text.delta', 'delta': item['content'][0]['text'], **delta})
        items.append({'type': 'response.output_item.done', 'output_index': index, 'item': item})
    return _sse(
        {'type': 'response.created', 'response': {**response, 'status': 'in_progress', 'output': []}},
        *items,
        {'type': 'response.completed', 'response': response},
    )


# Reasoning models warn about sampling settings before the profile drops them.
@pytest.mark.filterwarnings('ignore:Sampling parameters.*:UserWarning')
async def test_agent_run_uses_the_plan_usage_dialect(allow_model_requests: None):
    """A tool round trip streams with `store=false`, sends instructions as developer messages, and drops
    the generic settings the route rejects."""
    bodies: list[dict[str, Any]] = []
    responses = [
        _completed(
            'resp_1',
            [
                {
                    'type': 'function_call',
                    'id': 'fc_1',
                    'call_id': 'call_1',
                    'name': 'moo',
                    'arguments': '{}',
                    'status': 'completed',
                }
            ],
        ),
        _completed(
            'resp_2',
            [
                {
                    'type': 'message',
                    'id': 'msg_1',
                    'role': 'assistant',
                    'status': 'completed',
                    'content': [{'type': 'output_text', 'text': 'Moo!', 'annotations': []}],
                }
            ],
        ),
    ]

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.headers['authorization'] == 'Bearer access-1'
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, headers={'content-type': 'text/event-stream'}, content=responses[len(bodies) - 1])

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        provider = OpenAIChatGPTProvider(credentials=make_credentials(), http_client=client)
        agent = Agent(OpenAIResponsesModel('gpt-6.1-sol', provider=provider), system_prompt='Be a cow.')

        @agent.tool_plain
        def moo() -> str:
            """Return a cow sound."""
            return 'Moo!'

        result = await agent.run('Moo once.', model_settings={'temperature': 0.5, 'max_tokens': 100, 'top_p': 0.5})

    assert result.output == 'Moo!'
    dialect = [
        {
            'stream': body['stream'],
            'store': body['store'],
            'sent_settings': [key for key in ('temperature', 'max_output_tokens', 'top_p') if key in body],
            'tools': body['tools'],
            'instructions': body['input'][0],
        }
        for body in bodies
    ]
    assert (
        dialect[0]
        == dialect[1]
        == snapshot(
            {
                'stream': True,
                'store': False,
                'sent_settings': [],
                'tools': [
                    {
                        'name': 'moo',
                        'parameters': {'additionalProperties': False, 'properties': {}, 'type': 'object'},
                        'type': 'function',
                        'description': 'Return a cow sound.',
                        'strict': False,
                    }
                ],
                'instructions': {'role': 'developer', 'content': 'Be a cow.'},
            }
        )
    )
