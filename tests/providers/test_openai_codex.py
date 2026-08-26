import asyncio
import base64
import hashlib
import json
import time
from pathlib import Path
from typing import Any, cast

import anyio
import httpx2
import pytest
from pydantic import SecretStr

from pydantic_ai import ModelRequest
from pydantic_ai.exceptions import UnexpectedModelBehavior, UserError
from pydantic_ai.messages import TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters, infer_model, infer_model_profile
from pydantic_ai.providers import infer_provider_class

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from openai.types import responses

    from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
    from pydantic_ai.providers.openai_codex import (
        CredentialsPersistenceError,
        CredentialsRefreshError,
        OpenAICodexAuth,
        OpenAICodexCredentials,
        OpenAICodexOAuthFlow,
        OpenAICodexProvider,
        _account_id_from_id_token,  # pyright: ignore[reportPrivateUsage]
        _credentials_from_token_response,  # pyright: ignore[reportPrivateUsage]
        _jwt_expires_at,  # pyright: ignore[reportPrivateUsage]
        _post_json,  # pyright: ignore[reportPrivateUsage]
    )

    from ..models.mock_openai import MockOpenAIResponses

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='OpenAI client not installed'),
    pytest.mark.anyio,
]

PUBLIC_CLIENT_ID = 'app_EMoamEEZ73f0CkXaXp7hrann'


def make_jwt(payload: dict[str, Any]) -> str:
    def encode(part: dict[str, Any]) -> str:
        return base64.urlsafe_b64encode(json.dumps(part).encode()).rstrip(b'=').decode()

    return f'{encode({"alg": "none"})}.{encode(payload)}.signature'


def make_credentials(*, exp: float | None = None, access_token: str = 'access-old') -> OpenAICodexCredentials:
    token = access_token if exp is None else make_jwt({'exp': exp})
    return OpenAICodexCredentials(
        access_token=SecretStr(token), refresh_token=SecretStr('refresh-1'), account_id='acc-1'
    )


def make_provider(credentials: OpenAICodexCredentials | None = None) -> OpenAICodexProvider:
    return OpenAICodexProvider(credentials=credentials or make_credentials(exp=time.time() + 3600))


class TokenEndpointMock:
    """Stands in for `_post_json`, recording forms and returning queued payloads/exceptions."""

    def __init__(self, *results: dict[str, Any] | Exception):
        self.results = list(results)
        self.forms: list[dict[str, Any]] = []

    async def __call__(self, url: str, form: dict[str, Any]) -> dict[str, Any]:
        self.forms.append(form)
        await asyncio.sleep(0.001)  # widen race windows for single-flight assertions
        result = self.results[min(len(self.forms), len(self.results)) - 1]
        if isinstance(result, Exception):
            raise result
        return result


TOKEN_RESPONSE: dict[str, Any] = {
    'access_token': 'access-new',
    'refresh_token': 'refresh-2',
    'id_token': make_jwt({'https://api.openai.com/auth': {'chatgpt_account_id': 'acc-9'}}),
}


def authed_client(provider: OpenAICodexProvider, handler: Any) -> httpx2.AsyncClient:
    transport = httpx2.MockTransport(handler)
    return httpx2.AsyncClient(transport=transport, auth=OpenAICodexAuth(provider))


# --- Credentials parsing and CLI loading ---


def test_credentials_from_codex_cli_auth():
    creds = OpenAICodexCredentials.from_codex_cli_auth(
        {
            'OPENAI_API_KEY': None,
            'tokens': {
                'access_token': 'super-secret-access',
                'refresh_token': 'super-secret-refresh',
                'account_id': 'acc',
            },
            'last_refresh': 'whenever',
            'some_future_field': {'nested': 1},
        }
    )
    assert creds.account_id == 'acc'
    assert creds.access_token.get_secret_value() == 'super-secret-access'
    # Unknown top-level fields are ignored; secrets never leak through repr.
    rendered = repr(creds)
    assert 'super-secret' not in rendered


def test_credentials_missing_tokens_object():
    with pytest.raises(UserError, match="expected an object with a 'tokens' entry"):
        OpenAICodexCredentials.from_codex_cli_auth({'nope': {}})


def test_credentials_missing_fields():
    with pytest.raises(UserError, match=r'missing access_token'):
        OpenAICodexCredentials.from_codex_cli_auth({'tokens': {'refresh_token': 'r', 'account_id': 'acc'}})


def test_from_codex_cli_honors_code_home(env: TestEnv, tmp_path: Path):
    auth_json = tmp_path / 'auth.json'
    original = json.dumps(
        {
            'OPENAI_API_KEY': None,
            'last_refresh': 'x',
            'tokens': {'access_token': 'a', 'refresh_token': 'r', 'account_id': 'acc'},
        }
    )
    auth_json.write_text(original)
    env.set('CODEX_HOME', str(tmp_path))

    provider = OpenAICodexProvider.from_codex_cli()

    assert provider.credentials.account_id == 'acc'
    assert provider.name == 'openai-codex'
    assert provider.base_url == 'https://chatgpt.com/backend-api/codex'
    # Read-only contract: byte-for-byte unchanged after construction.
    assert auth_json.read_text() == original


def test_from_codex_cli_missing_file(env: TestEnv, tmp_path: Path):
    env.set('CODEX_HOME', str(tmp_path))
    with pytest.raises(UserError, match=r'codex login'):
        OpenAICodexProvider()


def test_from_codex_cli_unreadable_file(env: TestEnv, tmp_path: Path):
    (tmp_path / 'auth.json').mkdir()  # a directory: `read_text` raises an `OSError` subclass
    env.set('CODEX_HOME', str(tmp_path))
    with pytest.raises(UserError, match='Could not read'):
        OpenAICodexProvider()


def test_from_codex_cli_malformed_json(env: TestEnv, tmp_path: Path):
    (tmp_path / 'auth.json').write_text('not json')
    env.set('CODEX_HOME', str(tmp_path))
    with pytest.raises(UserError, match='Malformed'):
        OpenAICodexProvider()


def test_no_openai_api_key_fallback(env: TestEnv, tmp_path: Path):
    env.set('CODEX_HOME', str(tmp_path))
    env.set('OPENAI_API_KEY', 'sk-fake')
    with pytest.raises(UserError, match=r'codex login'):
        OpenAICodexProvider()


# --- JWT expiry hint ---


def test_jwt_expiry_hint():
    now = time.time()
    assert _jwt_expires_at(make_jwt({'exp': now - 100})) is not None
    assert _jwt_expires_at('garbage') is None


def test_account_id_claim_fallbacks():
    assert _account_id_from_id_token('garbage') is None  # unparsable id_token
    # Top-level claims are consulted when the nested claim is absent or empty.
    assert _account_id_from_id_token(make_jwt({'chatgpt_account_id': 'acc-top'})) == 'acc-top'
    assert _account_id_from_id_token(make_jwt({'account_id': 'acc-legacy'})) == 'acc-legacy'
    assert _account_id_from_id_token(make_jwt({})) is None


def test_token_response_validation_errors():
    with pytest.raises(CredentialsRefreshError, match='access_token'):
        _credentials_from_token_response({})
    with pytest.raises(CredentialsRefreshError, match='refresh_token'):
        _credentials_from_token_response({'access_token': 'a'})
    with pytest.raises(CredentialsRefreshError, match='account id'):
        _credentials_from_token_response({'access_token': 'a', 'refresh_token': 'r'})


async def test_post_json_success_and_error_shapes(monkeypatch: pytest.MonkeyPatch):
    """The OAuth POST helper: success, JSON error with `invalid_grant` hint, JSON error without
    a description, and a non-JSON error body."""
    real_client = httpx2.AsyncClient
    queue = [
        httpx2.Response(200, json={'ok': True}),
        httpx2.Response(400, json={'error': 'invalid_grant', 'error_description': 'expired'}),
        httpx2.Response(403, json={'error': 'access_denied'}),
        httpx2.Response(500, text='gateway exploded'),
    ]

    def handler(request: httpx2.Request) -> httpx2.Response:
        return queue.pop(0)

    def client_factory(**kwargs: Any) -> httpx2.AsyncClient:
        return real_client(transport=httpx2.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx2, 'AsyncClient', client_factory)

    url = 'https://auth.openai.com/oauth/token'
    assert await _post_json(url, {'grant_type': 'refresh_token'}) == {'ok': True}
    with pytest.raises(CredentialsRefreshError, match='expired; the grant was rejected'):
        await _post_json(url, {})
    with pytest.raises(CredentialsRefreshError, match='access_denied'):
        await _post_json(url, {})
    with pytest.raises(CredentialsRefreshError, match='gateway exploded'):
        await _post_json(url, {})
    assert _jwt_expires_at('a.b') is None
    assert _jwt_expires_at(f'a.{base64.urlsafe_b64encode(b"not json").decode()}.c') is None
    assert _jwt_expires_at(make_jwt({'exp': 'soon'})) is None
    assert _jwt_expires_at(make_jwt({'exp': True})) is None
    assert _jwt_expires_at(make_jwt({'exp': 10**14})) is None  # absurd values degrade to None
    assert _jwt_expires_at(make_jwt({})) is None


# --- Proactive (expiry-hint) refresh: single flight under concurrency ---


async def test_simultaneous_expiry_performs_one_refresh(monkeypatch: pytest.MonkeyPatch):
    mock = TokenEndpointMock(TOKEN_RESPONSE)
    monkeypatch.setattr('pydantic_ai.providers.openai_codex._post_json', mock)
    provider = make_provider(make_credentials(exp=time.time() - 10))

    async def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.headers['authorization'] == 'Bearer access-new'
        return httpx2.Response(200)

    async with authed_client(provider, handler) as client:
        responses = await asyncio.gather(*(client.get('https://chatgpt.com/backend-api/codex/x') for _ in range(5)))

    assert all(r.status_code == 200 for r in responses)
    assert len(mock.forms) == 1  # five waiters, one network refresh
    assert mock.forms[0] == {'grant_type': 'refresh_token', 'refresh_token': 'refresh-1', 'client_id': PUBLIC_CLIENT_ID}
    assert provider.credentials.refresh_token.get_secret_value() == 'refresh-2'


async def test_fresh_credentials_skip_proactive_refresh(monkeypatch: pytest.MonkeyPatch):
    mock = TokenEndpointMock(TOKEN_RESPONSE)
    monkeypatch.setattr('pydantic_ai.providers.openai_codex._post_json', mock)
    provider = make_provider()  # healthy JWT

    old_bearer = f'Bearer {provider.credentials.access_token.get_secret_value()}'

    async def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.headers['authorization'] == old_bearer
        assert request.headers['chatgpt-account-id'] == 'acc-1'
        assert request.headers['originator'] == 'pydantic-ai'
        return httpx2.Response(200)

    async with authed_client(provider, handler) as client:
        response = await client.get('https://chatgpt.com/backend-api/codex/x')

    assert response.status_code == 200
    assert mock.forms == []


async def test_malformed_jwt_degrades_to_401_path(monkeypatch: pytest.MonkeyPatch):
    mock = TokenEndpointMock(TOKEN_RESPONSE)
    monkeypatch.setattr('pydantic_ai.providers.openai_codex._post_json', mock)
    provider = make_provider(make_credentials(access_token='not-a-jwt'))
    old_bearer = f'Bearer {provider.credentials.access_token.get_secret_value()}'
    requests_seen: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests_seen.append(request.headers['authorization'])
        if len(requests_seen) == 1:
            return httpx2.Response(401)
        return httpx2.Response(200)

    async with authed_client(provider, handler) as client:
        response = await client.get('https://chatgpt.com/backend-api/codex/x')

    assert response.status_code == 200
    assert len(mock.forms) == 1  # exactly one refresh — from the 401, not the unparsable hint
    assert requests_seen == [old_bearer, 'Bearer access-new']  # original + one replay


# --- 401-triggered refresh-and-replay ---


async def test_simultaneous_401s_single_flight_recheck(monkeypatch: pytest.MonkeyPatch):
    mock = TokenEndpointMock(TOKEN_RESPONSE)
    monkeypatch.setattr('pydantic_ai.providers.openai_codex._post_json', mock)
    provider = make_provider(make_credentials())  # no expiry hint: only the 401 can trigger refresh
    old_bearer = f'Bearer {provider.credentials.access_token.get_secret_value()}'
    sends: list[str] = []
    lock = anyio.Lock()

    async def handler(request: httpx2.Request) -> httpx2.Response:
        async with lock:
            bearer = request.headers['authorization']
            is_replay = bearer != old_bearer
            sends.append(bearer)
        if is_replay:
            return httpx2.Response(200)
        return httpx2.Response(401)

    async with authed_client(provider, handler) as client:
        responses = await asyncio.gather(*(client.get('https://chatgpt.com/backend-api/codex/x') for _ in range(5)))

    assert all(r.status_code == 200 for r in responses)
    assert len(mock.forms) == 1  # five simultaneous 401s must not mean five refreshes
    assert provider.credentials.access_token.get_secret_value() == 'access-new'
    assert len(sends) == 10  # every logical request was sent exactly twice (original + replay)
    assert sorted(set(sends)) == sorted({'Bearer access-new', old_bearer})


async def test_401_after_inflight_rotation_replays_without_second_refresh(monkeypatch: pytest.MonkeyPatch):
    mock = TokenEndpointMock(TOKEN_RESPONSE)
    monkeypatch.setattr('pydantic_ai.providers.openai_codex._post_json', mock)
    provider = make_provider(make_credentials())
    calls = 0

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            # Another task rotates the credentials while this request is in flight, so its 401
            # must replay with the fresh set directly instead of refreshing a second time.
            await provider._refresh_for_401(0)  # pyright: ignore[reportPrivateUsage]
            return httpx2.Response(401)
        return httpx2.Response(200)

    async with authed_client(provider, handler) as client:
        response = await client.get('https://chatgpt.com/backend-api/codex/x')

    assert response.status_code == 200
    assert len(mock.forms) == 1  # only the in-flight rotation refreshed; the 401 did not


async def test_non_expiry_401_does_not_loop(monkeypatch: pytest.MonkeyPatch):
    mock = TokenEndpointMock(TOKEN_RESPONSE, TOKEN_RESPONSE)
    monkeypatch.setattr('pydantic_ai.providers.openai_codex._post_json', mock)
    provider = make_provider()
    sends: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sends.append(request.headers['authorization'])
        return httpx2.Response(401, json={'error': 'insufficient_quota'})

    async with authed_client(provider, handler) as client:
        first = await client.get('https://chatgpt.com/backend-api/codex/x')
        second = await client.get('https://chatgpt.com/backend-api/codex/x')

    assert first.status_code == second.status_code == 401
    assert len(sends) == 4  # exactly two sends per request: original plus a single replay
    assert len(mock.forms) == 2  # at most one refresh per request — never a loop


async def test_refresh_failure_surfaces_and_keeps_old_credentials(monkeypatch: pytest.MonkeyPatch):
    error = CredentialsRefreshError('Token request failed with status 400: invalid_grant; rerun the authorization flow')
    mock = TokenEndpointMock(error)
    monkeypatch.setattr('pydantic_ai.providers.openai_codex._post_json', mock)
    provider = make_provider()

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(401)

    async with authed_client(provider, handler) as client:
        with pytest.raises(CredentialsRefreshError, match='invalid_grant'):
            await client.get('https://chatgpt.com/backend-api/codex/x')

    assert mock.forms == [{'grant_type': 'refresh_token', 'refresh_token': 'refresh-1', 'client_id': PUBLIC_CLIENT_ID}]


async def test_callback_failure_updates_memory_but_raises_persistence_error(monkeypatch: pytest.MonkeyPatch):
    persisted: list[OpenAICodexCredentials] = []
    monkeypatch.setattr('pydantic_ai.providers.openai_codex._post_json', TokenEndpointMock(TOKEN_RESPONSE))

    async def failing_callback(credentials: OpenAICodexCredentials) -> None:
        persisted.append(credentials)
        raise RuntimeError('db down')

    provider = OpenAICodexProvider(
        credentials=make_credentials(exp=time.time() + 3600), on_credentials_refresh=failing_callback
    )

    old_bearer = f'Bearer {provider.credentials.access_token.get_secret_value()}'

    def handler(request: httpx2.Request) -> httpx2.Response:
        # The persistence error surfaces during the refresh, so the replay never goes out.
        assert request.headers['authorization'] == old_bearer
        return httpx2.Response(401)

    async with authed_client(provider, handler) as client:
        with pytest.raises(CredentialsPersistenceError, match='persistence callback raised'):
            await client.get('https://chatgpt.com/backend-api/codex/x')

    # In-memory credentials are current even though persistence failed.
    assert len(persisted) == 1
    assert provider.credentials.access_token.get_secret_value() == 'access-new'
    assert provider.credentials.refresh_token.get_secret_value() == 'refresh-2'


async def test_account_id_falls_back_to_previous_on_rotation(monkeypatch: pytest.MonkeyPatch):
    mock = TokenEndpointMock({**TOKEN_RESPONSE, 'id_token': make_jwt({'sub': 'user'})})
    monkeypatch.setattr('pydantic_ai.providers.openai_codex._post_json', mock)
    provider = make_provider()
    await provider._refresh_for_401(provider._revision)  # pyright: ignore[reportPrivateUsage]
    assert provider.credentials.account_id == 'acc-1'  # carried over when id_token lacks the claim


def test_sync_auth_flow_is_rejected():
    auth = OpenAICodexAuth(make_provider())
    with pytest.raises(RuntimeError, match='async'):
        auth.sync_auth_flow(httpx2.Request('GET', 'https://example.com'))


async def test_auth_never_sent_to_foreign_hosts():
    """A caller-supplied client may be reused for other destinations; credentials stay home."""
    provider = make_provider()
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200)

    async with authed_client(provider, handler) as client:
        response = await client.get('https://example.com/unrelated')

    assert response.status_code == 200
    assert 'authorization' not in seen[0].headers
    assert 'chatgpt-account-id' not in seen[0].headers
    assert 'originator' not in seen[0].headers


def test_openai_client_passthrough():
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key='irrelevant', base_url='https://chatgpt.com/backend-api/codex')
    provider = OpenAICodexProvider(openai_client=client)
    assert provider.client is client  # used as-is: no credential injection, no auth wrapping


async def test_caller_supplied_http_client_gets_scoped_auth():
    http_client = httpx2.AsyncClient()
    try:
        OpenAICodexProvider(credentials=make_credentials(), http_client=http_client)
        assert isinstance(http_client.auth, OpenAICodexAuth)
    finally:
        await http_client.aclose()


async def test_reopen_after_close_reattaches_auth():
    """Exiting the provider context closes its owned client; re-entering rebuilds one with auth."""
    provider = make_provider()
    async with provider:
        pass
    async with provider:
        http_client = provider.client._client  # pyright: ignore[reportPrivateUsage]
        assert not http_client.is_closed
        assert isinstance(http_client.auth, OpenAICodexAuth)


# --- Flow primitives ---


def test_authorization_url_shape():
    flow = OpenAICodexOAuthFlow(state='my-state')
    url = flow.authorization_url()
    assert url.startswith('https://auth.openai.com/oauth/authorize?')
    assert 'response_type=code' in url
    assert f'client_id={PUBLIC_CLIENT_ID}' in url
    assert 'state=my-state' in url
    assert 'code_challenge_method=S256' in url
    challenge = url.split('code_challenge=')[1].split('&')[0]
    expected = base64.urlsafe_b64encode(hashlib.sha256(flow.code_verifier.encode()).digest()).rstrip(b'=').decode()
    assert challenge == expected
    assert 'redirect_uri=http%3A%2F%2Flocalhost%3A1455%2Fauth%2Fcallback' in url
    # Production-parity params (live-verified 2026-08-25): without `id_token_add_organizations`,
    # the id_token can omit the account id for multi-org accounts.
    assert 'id_token_add_organizations=true' in url
    assert 'codex_cli_simplified_flow=true' in url


def test_authorization_url_extra_params_add_and_override():
    flow = OpenAICodexOAuthFlow(state='my-state')
    url = flow.authorization_url(extra_params={'prompt': 'login', 'codex_cli_simplified_flow': 'false'})
    assert 'prompt=login' in url  # added
    assert 'codex_cli_simplified_flow=false' in url  # overridden
    assert 'codex_cli_simplified_flow=true' not in url
    assert 'id_token_add_organizations=true' in url  # untouched default survives


async def test_exchange_code_posts_pkce_form(monkeypatch: pytest.MonkeyPatch):
    mock = TokenEndpointMock(TOKEN_RESPONSE)
    monkeypatch.setattr('pydantic_ai.providers.openai_codex._post_json', mock)
    flow = OpenAICodexOAuthFlow()
    credentials = await flow.exchange_code('the-code')

    assert mock.forms[0]['grant_type'] == 'authorization_code'
    assert mock.forms[0]['code'] == 'the-code'
    assert mock.forms[0]['code_verifier'] == flow.code_verifier
    assert credentials.account_id == 'acc-9'  # extracted from the nested id_token claim


# --- Prefix inference and profile dialect ---


def test_provider_class_inference():
    assert infer_provider_class('openai-codex') is OpenAICodexProvider


def test_openai_codex_prefix_infers_responses_model(env: TestEnv, tmp_path: Path):
    (tmp_path / 'auth.json').write_text(
        json.dumps({'tokens': {'access_token': 'a', 'refresh_token': 'r', 'account_id': 'acc'}})
    )
    env.set('CODEX_HOME', str(tmp_path))
    model = infer_model('openai-codex:gpt-5.6-luna')

    assert isinstance(model, OpenAIResponsesModel)
    assert model.profile.get('openai_responses_requires_streaming') is True
    assert model.profile.get('openai_responses_requires_store_false') is True
    assert model.profile.get('openai_supports_input_token_counting') is False
    unsupported = model.profile.get('openai_unsupported_model_settings', ())
    assert {'max_tokens', 'temperature', 'top_p', 'openai_top_logprobs', 'openai_truncation', 'openai_user'} <= set(
        unsupported
    )


def test_standard_openai_profile_untouched():
    profile = infer_model_profile('openai:gpt-5')
    assert profile.get('openai_responses_requires_streaming', False) is False
    assert profile.get('openai_supports_input_token_counting', True) is True


async def test_count_tokens_raises_user_error(allow_model_requests: None):
    model = OpenAIResponsesModel('gpt-5.6-luna', provider=make_provider())
    with pytest.raises(UserError, match='Server-side token counting is not available'):
        await model.count_tokens([ModelRequest(parts=[UserPromptPart('hi')])], None, ModelRequestParameters())


# --- Wire dialect through the model's request path ---


_MINIMAL_RESPONSE: dict[str, Any] = {
    'id': 'resp_123',
    'object': 'response',
    'created_at': 0,
    'status': 'completed',
    'model': 'gpt-5.6-luna',
    'output': [
        {
            'type': 'message',
            'id': 'm1',
            'status': 'completed',
            'role': 'assistant',
            'content': [{'type': 'output_text', 'text': 'hi there', 'annotations': []}],
        }
    ],
    'usage': {
        'input_tokens': 3,
        'input_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 0},
        'output_tokens': 2,
        'output_tokens_details': {'reasoning_tokens': 0},
        'total_tokens': 5,
    },
    'parallel_tool_calls': False,
    'tool_choice': 'none',
    'tools': [],
}


def _codex_stream(*, slim_completed: bool) -> list[responses.ResponseStreamEvent]:
    """The SSE sequence observed live against the Codex backend (2026-08-25).

    With `slim_completed=True` this reproduces the real Codex shape: `response.completed` carries an
    EMPTY `output` array, and content exists only in the incremental events. `slim_completed=False`
    is the api.openai.com shape, where the terminal event repeats the full output.
    """
    completed = responses.Response.model_validate(_MINIMAL_RESPONSE)
    in_progress = completed.model_copy(update={'status': 'in_progress', 'usage': None, 'output': []})
    if slim_completed:
        completed = completed.model_copy(update={'output': []})
    message_done = responses.ResponseOutputMessage(
        id='m1',
        type='message',
        role='assistant',
        status='completed',
        content=[responses.ResponseOutputText(type='output_text', text='hi there', annotations=[])],
    )
    return [
        responses.ResponseCreatedEvent(type='response.created', response=in_progress, sequence_number=0),
        responses.ResponseInProgressEvent(type='response.in_progress', response=in_progress, sequence_number=1),
        responses.ResponseOutputItemAddedEvent(
            type='response.output_item.added',
            item=message_done.model_copy(update={'status': 'in_progress', 'content': []}),
            output_index=0,
            sequence_number=2,
        ),
        responses.ResponseContentPartAddedEvent(
            type='response.content_part.added',
            part=responses.ResponseOutputText(type='output_text', text='', annotations=[]),
            item_id='m1',
            output_index=0,
            content_index=0,
            sequence_number=3,
        ),
        responses.ResponseTextDeltaEvent(
            type='response.output_text.delta',
            delta='hi ',
            item_id='m1',
            output_index=0,
            content_index=0,
            logprobs=[],
            sequence_number=4,
        ),
        responses.ResponseTextDeltaEvent(
            type='response.output_text.delta',
            delta='there',
            item_id='m1',
            output_index=0,
            content_index=0,
            logprobs=[],
            sequence_number=5,
        ),
        responses.ResponseContentPartDoneEvent(
            type='response.content_part.done',
            part=responses.ResponseOutputText(type='output_text', text='hi there', annotations=[]),
            item_id='m1',
            output_index=0,
            content_index=0,
            sequence_number=6,
        ),
        responses.ResponseOutputItemDoneEvent(
            type='response.output_item.done', item=message_done, output_index=0, sequence_number=7
        ),
        responses.ResponseCompletedEvent(type='response.completed', response=completed, sequence_number=8),
    ]


def _codex_model_with_stream(
    events: list[responses.ResponseStreamEvent],
) -> tuple[OpenAIResponsesModel, MockOpenAIResponses]:
    mock_client = MockOpenAIResponses.create_mock_stream(events)
    model = OpenAIResponsesModel('gpt-5.6-luna', provider=OpenAICodexProvider(openai_client=mock_client))
    return model, cast(MockOpenAIResponses, mock_client)


def _assert_aggregated_response(response: Any, kwargs: dict[str, Any]) -> None:
    part = response.parts[0]
    assert isinstance(part, TextPart)
    assert part.content == 'hi there'
    assert response.usage.input_tokens == 3
    assert response.usage.output_tokens == 2
    assert response.provider_response_id == 'resp_123'
    # `stream=True` itself is proven by the mock: it refuses to serve a non-streaming create call
    # when only stream events are configured.
    assert kwargs['store'] is False  # cannot be omitted under Codex subscription auth


async def test_forced_stream_aggregates_codex_slim_completed(allow_model_requests: None):
    """REGRESSION (live-verified 2026-08-25): Codex sends `response.completed` with an EMPTY `output`.

    Content exists only in the incremental events, so trusting the terminal event's `response`
    produced `ModelResponse(parts=[])` with billed tokens. The forced stream must be drained through
    the streamed-response machinery, which builds parts from the incremental events.
    """
    model, mock = _codex_model_with_stream(_codex_stream(slim_completed=True))
    response = await model.request([ModelRequest(parts=[UserPromptPart('hi')])], None, ModelRequestParameters())
    _assert_aggregated_response(response, mock.response_kwargs[0])


async def test_forced_stream_aggregates_full_completed_output(allow_model_requests: None):
    # api.openai.com repeats the full output on `response.completed`; the profile flag must keep
    # working there too if some other streaming-only endpoint ever sets it.
    model, mock = _codex_model_with_stream(_codex_stream(slim_completed=False))
    response = await model.request([ModelRequest(parts=[UserPromptPart('hi')])], None, ModelRequestParameters())
    _assert_aggregated_response(response, mock.response_kwargs[0])


async def test_forced_stream_drops_unsupported_settings(allow_model_requests: None):
    model, mock = _codex_model_with_stream(_codex_stream(slim_completed=True))
    settings = OpenAIResponsesModelSettings(
        max_tokens=128,
        temperature=0.5,
        top_p=0.9,
        openai_top_logprobs=3,
        openai_truncation='auto',
        openai_user='user-1',
        openai_store=True,
    )

    # The Codex backend rejects tuning fields outright, and the generic reasoning seam also warns
    # about sampling params on GPT-5.6-family models; both land as UserWarnings here.
    with pytest.warns(UserWarning):
        response = await model.request([ModelRequest(parts=[UserPromptPart('hi')])], settings, ModelRequestParameters())

    _assert_aggregated_response(response, mock.response_kwargs[0])
    kwargs = mock.response_kwargs[0]
    for wire_name in ('max_output_tokens', 'temperature', 'top_p', 'top_logprobs', 'user', 'truncation'):
        assert wire_name not in kwargs  # dropped before anything reached the wire


async def test_forced_stream_without_events_raises(allow_model_requests: None):
    # The nested-list form is how the shared mock represents a single, empty stream.
    mock_client = MockOpenAIResponses.create_mock_stream([[]])
    model = OpenAIResponsesModel('gpt-5.6-luna', provider=OpenAICodexProvider(openai_client=mock_client))
    with pytest.raises(UnexpectedModelBehavior, match='without content'):
        await model.request([ModelRequest(parts=[UserPromptPart('hi')])], None, ModelRequestParameters())


# --- Session affinity (mirrors the official Codex client's session/thread identifiers) ---


def _codex_model_with_streams(count: int) -> tuple[OpenAIResponsesModel, MockOpenAIResponses]:
    events = [_codex_stream(slim_completed=True) for _ in range(count)]
    mock_client = MockOpenAIResponses.create_mock_stream(events)
    model = OpenAIResponsesModel('gpt-5.6-luna', provider=OpenAICodexProvider(openai_client=mock_client))
    return model, cast(MockOpenAIResponses, mock_client)


def _turn(conversation_id: str | None, run_id: str | None) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart('hi')], conversation_id=conversation_id, run_id=run_id)


async def test_session_affinity_stable_within_conversation(allow_model_requests: None):
    """Runs sharing a conversation share `session-id` and `prompt_cache_key`; each run is its own thread.

    Mirrors the official client's semantics: threads (root and subagents) differ in `thread-id` but
    share the session id and cache key.
    """
    model, mock = _codex_model_with_streams(2)
    await model.request([_turn('conv-1', 'run-1')], None, ModelRequestParameters())
    await model.request([_turn('conv-1', 'run-1'), _turn('conv-1', 'run-2')], None, ModelRequestParameters())

    first, second = mock.response_kwargs
    assert first['extra_headers']['session-id'] == 'conv-1'
    assert first['extra_headers']['thread-id'] == 'run-1'
    assert first['extra_headers']['x-client-request-id'] == 'run-1'
    assert first['prompt_cache_key'] == 'conv-1'
    assert second['extra_headers']['session-id'] == 'conv-1'
    assert second['extra_headers']['thread-id'] == 'run-2'
    assert second['prompt_cache_key'] == 'conv-1'


async def test_session_affinity_isolated_between_conversations(allow_model_requests: None):
    model, mock = _codex_model_with_streams(2)
    await model.request([_turn('conv-1', 'run-1')], None, ModelRequestParameters())
    await model.request([_turn('conv-2', 'run-2')], None, ModelRequestParameters())

    first, second = mock.response_kwargs
    assert first['extra_headers']['session-id'] == 'conv-1'
    assert second['extra_headers']['session-id'] == 'conv-2'
    assert first['prompt_cache_key'] != second['prompt_cache_key']


async def test_session_affinity_explicit_overrides_win(allow_model_requests: None):
    model, mock = _codex_model_with_stream(_codex_stream(slim_completed=True))
    settings = OpenAIResponsesModelSettings(
        openai_prompt_cache_key='my-key', extra_headers={'session-id': 'my-session'}
    )
    await model.request([_turn('conv-1', 'run-1')], settings, ModelRequestParameters())

    kwargs = mock.response_kwargs[0]
    assert kwargs['extra_headers']['session-id'] == 'my-session'  # the explicit header wins
    assert kwargs['extra_headers']['thread-id'] == 'run-1'  # unspecified headers are still derived
    assert kwargs['prompt_cache_key'] == 'my-key'  # the explicit cache key wins


async def test_session_affinity_thread_falls_back_to_session(allow_model_requests: None):
    # A request stamped with a conversation but no run (e.g. hand-built history) is one thread.
    model, mock = _codex_model_with_stream(_codex_stream(slim_completed=True))
    await model.request([_turn('conv-1', None)], None, ModelRequestParameters())

    kwargs = mock.response_kwargs[0]
    assert kwargs['extra_headers']['thread-id'] == 'conv-1'
    assert kwargs['extra_headers']['x-client-request-id'] == 'conv-1'


async def test_no_affinity_without_conversation_identity(allow_model_requests: None):
    """Direct model use outside an agent run leaves the wire shape unchanged."""
    model, mock = _codex_model_with_stream(_codex_stream(slim_completed=True))
    await model.request([ModelRequest(parts=[UserPromptPart('hi')])], None, ModelRequestParameters())

    kwargs = mock.response_kwargs[0]
    assert 'prompt_cache_key' not in kwargs
    for header in ('session-id', 'thread-id', 'x-client-request-id'):
        assert header not in kwargs['extra_headers']
