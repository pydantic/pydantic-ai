"""Browser sign-in for a public OAuth client, through core's real localhost callback and a mocked token endpoint."""

import re
import socket
import threading
import time
import webbrowser
from collections.abc import Callable, Generator
from contextlib import contextmanager
from urllib.parse import parse_qs, parse_qsl, urlencode, urlsplit

import anyio
import httpx
import pytest
from anyio import to_thread
from pydantic import SecretStr

from pydantic_ai.exceptions import UserError
from pydantic_clai2 import pkce
from pydantic_clai2.config.credential_store import delete_credentials, load_codex_credentials, save_codex_credentials
from pydantic_clai2.pkce import REFRESH_MARGIN, PKCEFlow, PKCESignIn, PublicClient, Tokens, refresh

pytestmark = pytest.mark.anyio

ACCOUNT = 'test-oauth'


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def client(*, client_id: str = 'app-1', scopes: tuple[str, ...] = ('read', 'write')) -> PublicClient:
    return PublicClient(
        authorize_url='https://auth.example/authorize',
        token_url='https://auth.example/token',
        client_id=client_id,
        redirect_uri=f'http://127.0.0.1:{free_port()}/callback',
        scopes=scopes,
        scope_separator=',',
    )


class TokenEndpoint:
    """Answers token requests in order and records each form, so tests can check what was proven."""

    def __init__(self, *answers: httpx.Response) -> None:
        self.answers = list(answers)
        self.forms: list[dict[str, str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.forms.append(dict(parse_qsl(request.content.decode())))
        return self.answers.pop(0)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)


def granted(access: str = 'at-1', refresh_token: str | None = 'rt-1', scope: str | None = None) -> httpx.Response:
    body: dict[str, object] = {'access_token': access, 'token_type': 'user', 'expires_in': 3600}
    if refresh_token:
        body['refresh_token'] = refresh_token
    if scope is not None:
        body['scope'] = scope
    return httpx.Response(200, json=body)


def browser(answer: Callable[[dict[str, str]], dict[str, str]], *, opens: bool = True) -> Callable[[str], bool]:
    """A browser that, once the callback server listens, lands on the redirect with `answer(authorize params)`."""

    def open_browser(url: str) -> bool:
        params = dict(parse_qsl(urlsplit(url).query))

        def land() -> None:
            callback = f'{params["redirect_uri"]}?{urlencode(answer(params))}'
            for _ in range(500):  # pragma: no branch
                try:
                    httpx.get(callback, timeout=1)
                    return
                except httpx.ConnectError:  # pragma: lax no cover
                    # Only when the browser lands before the callback server listens.
                    time.sleep(0.01)

        threading.Thread(target=land, daemon=True).start()
        return opens

    return open_browser


def approve(params: dict[str, str]) -> dict[str, str]:
    return {'code': 'the-code', 'state': params['state']}


def opened(url: str) -> bool:
    return True


def session(
    endpoint: TokenEndpoint,
    *,
    app: PublicClient | None = None,
    open_browser: Callable[[str], bool] = opened,
    timeout: float = 300,
) -> PKCESignIn:
    return PKCESignIn(
        client=app or client(),
        account=ACCOUNT,
        service='Example',
        setup='/plugins configure example',
        open_browser=open_browser,
        transport=endpoint.transport,
        timeout=timeout,
    )


def store(tokens: Tokens) -> None:
    save_codex_credentials(account=ACCOUNT, value=tokens.stored())


def tokens(
    *,
    expires_in: float | None,
    refresh_token: str | None = 'rt-old',
    client_id: str = 'app-1',
    scopes: tuple[str, ...] = ('read', 'write'),
) -> Tokens:
    return Tokens(
        client_id=client_id,
        access_token=SecretStr('at-old'),
        refresh_token=None if refresh_token is None else SecretStr(refresh_token),
        expires_at=None if expires_in is None else time.time() + expires_in,
        scopes=scopes,
    )


def test_authorization_url_carries_pkce_and_the_clients_scopes() -> None:
    app = client()
    flow = PKCEFlow(app, service='Example')
    url = urlsplit(flow.authorization_url())
    params = dict(parse_qsl(url.query))
    assert f'{url.scheme}://{url.netloc}{url.path}' == app.authorize_url
    assert params == {
        'response_type': 'code',
        'client_id': 'app-1',
        'redirect_uri': app.redirect_uri,
        'state': flow.state,
        'code_challenge': flow.code_challenge,
        'code_challenge_method': 'S256',
        'scope': 'read,write',
    }
    assert 'scope' not in flow.authorization_url(scope='')
    assert 'scope=other' in flow.authorization_url(scope='other')
    with pytest.raises(UserError, match='cannot override state'):
        flow.authorization_url(extra_params={'state': 'forged'})


def test_authorization_url_keeps_a_query_the_authorize_url_already_has() -> None:
    app = client().model_copy(update={'authorize_url': 'https://auth.example/authorize?team=T1&empty='})
    url = PKCEFlow(app, service='Example').authorization_url()
    assert url.count('?') == 1
    params = parse_qs(urlsplit(url).query, keep_blank_values=True)
    assert params['team'] == ['T1'] and params['empty'] == [''] and params['client_id'] == ['app-1']


async def test_a_flow_started_for_another_app_is_refused() -> None:
    other = PKCEFlow(client(client_id='app-2'), service='Example')
    with pytest.raises(ValueError, match='another Example app'):
        await session(TokenEndpoint()).sign_in(other)


async def test_sign_in_proves_the_verifier_without_a_secret_and_stores_the_tokens() -> None:
    endpoint = TokenEndpoint(granted())
    shown: list[str] = []
    sign_in = session(endpoint, open_browser=browser(approve))
    result = await sign_in.sign_in(show=shown.append)
    [form] = endpoint.forms
    assert set(form) == {'grant_type', 'code', 'redirect_uri', 'client_id', 'code_verifier'}
    assert form['grant_type'] == 'authorization_code' and form['code'] == 'the-code'
    assert result.refresh_token == SecretStr('rt-1')
    assert result.expires_at is not None and abs(result.expires_at - (time.time() + 3600)) < 60
    assert result.scopes == ('read', 'write'), 'a response without `scope` granted what was asked'
    assert shown == []
    assert sign_in.signed_in()
    assert await sign_in.token() == 'at-1'


async def test_an_already_approved_app_redirecting_at_once_is_not_lost() -> None:
    def redirects_at_once(url: str) -> bool:  # One visit, before `open_browser` even returns: no retries.
        params = dict(parse_qsl(urlsplit(url).query))
        assert httpx.get(f'{params["redirect_uri"]}?{urlencode(approve(params))}', timeout=5).status_code == 200
        return True

    endpoint = TokenEndpoint(granted())
    signed = await session(endpoint, open_browser=redirects_at_once, timeout=5).sign_in()
    assert signed.access_token == SecretStr('at-1')


async def test_cancelling_while_the_tokens_are_saved_waits_for_the_save(monkeypatch: pytest.MonkeyPatch) -> None:
    saving, release = threading.Event(), threading.Event()
    order: list[str] = []
    real_save = pkce.save_codex_credentials

    def slow_save(*, account: str, value: str) -> None:
        saving.set()
        release.wait(5)  # As if another CLAI held the credential lock.
        real_save(account=account, value=value)
        order.append('saved')

    monkeypatch.setattr(pkce, 'save_codex_credentials', slow_save)
    sign_in = session(TokenEndpoint(granted()), open_browser=browser(approve))
    async with anyio.create_task_group() as group:
        group.start_soon(sign_in.sign_in)
        await to_thread.run_sync(saving.wait, 5)
        group.cancel_scope.cancel()
        threading.Timer(0.2, release.set).start()
    order.append('returned')
    assert order == ['saved', 'returned']
    assert sign_in.signed_in()


def test_stored_tokens_hold_the_secrets_and_nothing_else_reveals_them() -> None:
    # `model_dump_json` masks `SecretStr`, which once stored asterisks in place of the tokens.
    signed = tokens(expires_in=60)
    assert 'at-old' in signed.stored() and 'rt-old' in signed.stored()
    assert Tokens.model_validate_json(signed.stored()) == signed
    assert 'at-old' not in repr(signed) + signed.model_dump_json()


@pytest.mark.parametrize('failure', ['closed', 'raises'])
async def test_no_browser_shows_the_url_instead(failure: str) -> None:
    endpoint = TokenEndpoint(granted())
    lands = browser(approve, opens=False)

    def open_browser(url: str) -> bool:
        lands(url)
        if failure == 'raises':
            raise webbrowser.Error('no browser')
        return False

    shown: list[str] = []
    await session(endpoint, open_browser=open_browser).sign_in(show=shown.append)
    [message] = shown
    assert message.startswith('Open this URL to sign in to Example: https://auth.example/authorize?')


async def test_denial_timeout_and_a_busy_port_are_explained_and_keep_the_earlier_sign_in() -> None:
    store(tokens(expires_in=None))

    def deny(params: dict[str, str]) -> dict[str, str]:
        return {'error': 'access_denied', 'state': params['state']}

    with pytest.raises(UserError, match='Authorization failed: access_denied'):
        await session(TokenEndpoint(), open_browser=browser(deny)).sign_in()
    with pytest.raises(UserError, match=r'sign-in timed out\. Run /plugins configure example'):
        await session(TokenEndpoint(), timeout=0.2).sign_in()
    app = client()
    with socket.socket() as busy:
        busy.bind(('127.0.0.1', urlsplit(app.redirect_uri).port or 0))
        busy.listen()
        with pytest.raises(UserError, match=re.escape('Could not listen on http://127.0.0.1')):
            await session(TokenEndpoint(), app=app).sign_in()
    assert Tokens.model_validate_json(load_codex_credentials(account=ACCOUNT) or '').access_token == SecretStr('at-old')


async def test_waiting_for_the_port_gives_up_when_something_keeps_it() -> None:
    with socket.socket() as busy:
        busy.bind(('127.0.0.1', 0))
        busy.listen()
        started = time.monotonic()
        await pkce._until_released(f'http://127.0.0.1:{busy.getsockname()[1]}/callback', limit=0.2)  # pyright: ignore[reportPrivateUsage]
        assert 0.2 <= time.monotonic() - started < 2


async def test_signing_in_again_right_after_giving_up_finds_the_port_free() -> None:
    endpoint, app = TokenEndpoint(granted()), client()
    with pytest.raises(UserError, match='sign-in timed out'):
        await session(endpoint, app=app, timeout=0.2).sign_in()
    signed = await session(endpoint, app=app, open_browser=browser(approve), timeout=5).sign_in()
    assert signed.access_token == SecretStr('at-1')


@pytest.mark.parametrize(
    ('answer', 'message'),
    [
        (httpx.Response(400, json={'error': 'invalid_grant'}), 'Example refused the sign-in: invalid_grant.'),
        (
            httpx.Response(200, json={'ok': False, 'error': 'invalid_code'}),
            'Example refused the sign-in: invalid_code.',
        ),
        (httpx.Response(503, json={}), 'Example refused the sign-in: HTTP 503.'),
        (httpx.Response(502, text='<html>'), 'Example did not answer the sign-in request.'),
    ],
)
async def test_token_errors_in_either_style_are_reported(answer: httpx.Response, message: str) -> None:
    flow = PKCEFlow(client(), service='Example', transport=TokenEndpoint(answer).transport)
    with pytest.raises(UserError) as error:
        await flow.exchange_code('the-code')
    assert str(error.value).startswith(message)


@pytest.mark.parametrize(('rotated', 'kept'), [('rt-new', 'rt-new'), (None, 'rt-old')])
async def test_refresh_uses_a_rotated_token_or_keeps_the_old_one(rotated: str | None, kept: str) -> None:
    endpoint = TokenEndpoint(granted(access='at-new', refresh_token=rotated))
    renewed = await refresh(client(), tokens(expires_in=0), service='Example', transport=endpoint.transport)
    assert endpoint.forms == [{'grant_type': 'refresh_token', 'refresh_token': 'rt-old', 'client_id': 'app-1'}]
    assert renewed.access_token == SecretStr('at-new') and renewed.refresh_token == SecretStr(kept)


async def test_the_granted_scopes_are_what_the_service_reports() -> None:
    endpoint = TokenEndpoint(granted(scope='read, extra'), granted(access='at-2'))
    signed = await session(endpoint, open_browser=browser(approve)).sign_in()
    assert signed.scopes == ('read', 'extra')
    renewed = await refresh(client(), signed, service='Example', transport=endpoint.transport)
    assert renewed.scopes == ('read', 'extra'), 'a refresh without `scope` keeps what was granted'


async def test_a_sign_in_without_the_scopes_the_client_now_asks_for_is_not_used() -> None:
    store(tokens(expires_in=3600, scopes=('read',)))
    narrow = session(TokenEndpoint(), app=client(scopes=('read',)))
    assert narrow.signed_in() and await narrow.token() == 'at-old'
    wider = session(TokenEndpoint())
    assert not wider.signed_in()
    with pytest.raises(
        UserError, match=r'does not grant what this setup needs\. Run /plugins configure example to sign in again'
    ):
        await wider.token()


async def test_a_refresh_that_narrows_the_scopes_is_saved_but_not_used() -> None:
    endpoint = TokenEndpoint(granted(access='at-narrow', refresh_token='rt-new', scope='read'))
    store(tokens(expires_in=0))
    with pytest.raises(UserError, match='does not grant what this setup needs'):
        await session(endpoint).token()
    saved = Tokens.model_validate_json(load_codex_credentials(account=ACCOUNT) or '')
    assert saved.refresh_token == SecretStr('rt-new'), 'the rotated refresh token is kept, the old one is spent'


async def test_token_is_reused_until_close_to_expiry_then_refreshed_and_saved() -> None:
    endpoint = TokenEndpoint(granted(access='at-new', refresh_token='rt-new'))
    sign_in = session(endpoint)
    store(tokens(expires_in=REFRESH_MARGIN + 60))
    assert await sign_in.token() == 'at-old'
    store(tokens(expires_in=REFRESH_MARGIN - 60))
    assert await sign_in.token() == 'at-new'
    saved = Tokens.model_validate_json(load_codex_credentials(account=ACCOUNT) or '')
    assert saved.refresh_token == SecretStr('rt-new')
    assert await sign_in.token() == 'at-new'
    assert len(endpoint.forms) == 1


def lock_after(other_session: Callable[[], None], monkeypatch: pytest.MonkeyPatch) -> None:
    """Let another CLAI act while this one waits for the refresh lock, then hand over the real lock."""
    real_lock = pkce.credential_lock

    @contextmanager
    def lock(*, account: str, busy: str) -> Generator[None]:
        other_session()
        with real_lock(account=account, busy=busy):
            yield

    monkeypatch.setattr(pkce, 'credential_lock', lock)


async def test_a_refresh_by_another_session_while_waiting_for_the_lock_is_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = TokenEndpoint()
    store(tokens(expires_in=0))
    theirs = tokens(expires_in=3600, refresh_token='rt-theirs').model_copy(update={'access_token': SecretStr('at-t')})
    lock_after(lambda: store(theirs), monkeypatch)
    assert await session(endpoint).token() == 'at-t'
    assert endpoint.forms == []


async def test_a_live_sign_in_without_a_refresh_token_saved_while_waiting_for_the_lock_is_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store(tokens(expires_in=0))
    theirs = tokens(expires_in=3600, refresh_token=None).model_copy(update={'access_token': SecretStr('at-live')})
    lock_after(lambda: store(theirs), monkeypatch)
    assert await session(TokenEndpoint()).token() == 'at-live'


@pytest.mark.parametrize(
    ('saved', 'message'),
    [
        (None, r'Not signed in to Example\. Run /plugins configure example to sign in\.'),
        ('{broken', r'Not signed in to Example\.'),
        ('clai-chunks-v1:broken', r'Not signed in to Example\. Run /plugins configure example'),
        (tokens(expires_in=3600, client_id='other-app').stored(), r'Not signed in to Example\.'),
        (tokens(expires_in=0, refresh_token=None).stored(), r'The Example sign-in expired\. Run /plugins configure'),
    ],
    ids=['nothing-saved', 'unreadable', 'corrupt-chunk-manifest', 'other-client', 'expired-no-refresh'],
)
async def test_token_fails_closed_when_there_is_nothing_usable(saved: str | None, message: str) -> None:
    if saved is not None:
        save_codex_credentials(account=ACCOUNT, value=saved)
    sign_in = session(TokenEndpoint())
    assert not sign_in.signed_in()
    with pytest.raises(UserError, match=message):
        await sign_in.token()


async def test_a_refused_refresh_asks_to_sign_in_again_and_keeps_the_tokens() -> None:
    store(tokens(expires_in=0))
    sign_in = session(TokenEndpoint(httpx.Response(200, json={'ok': False, 'error': 'invalid_refresh_token'})))
    with pytest.raises(UserError, match=r'could not be renewed\. Run /plugins configure example to sign in again'):
        await sign_in.token()
    assert load_codex_credentials(account=ACCOUNT) is not None


async def test_signing_out_while_a_refresh_waits_leaves_nothing_to_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    store(tokens(expires_in=0))
    lock_after(lambda: delete_credentials(account=ACCOUNT), monkeypatch)
    with pytest.raises(UserError, match=r'Not signed in to Example\.'):
        await session(TokenEndpoint()).token()


async def test_an_expired_sign_in_without_a_refresh_token_saved_while_a_refresh_waits_is_not_refreshed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = TokenEndpoint()
    store(tokens(expires_in=0))
    lock_after(lambda: store(tokens(expires_in=0, refresh_token=None)), monkeypatch)
    with pytest.raises(UserError, match=r'Not signed in to Example\. Run /plugins configure example to sign in\.'):
        await session(endpoint).token()
    assert endpoint.forms == []


def test_signed_in_counts_a_refreshable_or_live_sign_in_for_this_client() -> None:
    sign_in = session(TokenEndpoint())
    store(tokens(expires_in=0))
    assert sign_in.signed_in()
    store(tokens(expires_in=None, refresh_token=None))
    assert sign_in.signed_in()
    sign_in.sign_out()
    assert not sign_in.signed_in()
    assert load_codex_credentials(account=ACCOUNT) is None
