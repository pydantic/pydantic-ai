"""Logfire browser sign-in (device flow) against a fake Logfire; no request leaves the process."""

import json
import re
import time
import webbrowser
from urllib.parse import parse_qs

import httpx
import keyring
import pytest
from keyring.errors import KeyringError
from pydantic import JsonValue

from pydantic_clai2 import logfire_oauth
from pydantic_clai2.config.credential_store import save_codex_credentials
from pydantic_clai2.logfire_oauth import (
    DeviceAuth,
    LogfireSignIn,
    SignInError,
    Tokens,
    forget,
    load,
    sign_in,
    status,
)
from pydantic_clai2.plugins.sign_in import SignInRequired

ORIGIN = 'https://logfire.test'
RESOURCE = f'{ORIGIN}/mcp'
LINK = f'{ORIGIN}/auth/oauth-device?code=ABCD-EFGH'
OFFERED = ['project:read', 'project:write', 'organization:create_project']

Reply = tuple[int, JsonValue]


METADATA: dict[str, JsonValue] = {
    'issuer': ORIGIN,
    'device_authorization_endpoint': f'{ORIGIN}/api/oauth/device/code',
    'token_endpoint': f'{ORIGIN}/api/oauth/token',
    'registration_endpoint': f'{ORIGIN}/api/oauth/register',
}
DEVICE: dict[str, JsonValue] = {
    'device_code': 'device',
    'user_code': 'ABCD-EFGH',
    'verification_uri': f'{ORIGIN}/auth/oauth-device',
    'verification_uri_complete': LINK,
    'expires_in': 600,
    'interval': 0,
}


class Logfire:
    """Just enough of Logfire's OAuth server and MCP endpoint, recording what CLAI sends."""

    def __init__(self) -> None:
        offered: list[JsonValue] = [*OFFERED]
        self.resource: Reply = (
            200,
            {'resource': RESOURCE, 'authorization_servers': [ORIGIN], 'scopes_supported': offered},
        )
        self.metadata: JsonValue = METADATA
        self.registration: Reply = (201, {'client_id': 'client-1'})
        self.device: Reply = (200, DEVICE)
        self.polls: list[Reply] = [granted('access-1', refresh='refresh-1')]
        self.refreshes: list[Reply] = []
        self.valid = {'access-1'}
        self.registered: list[JsonValue] = []
        self.forms: list[dict[str, str]] = []
        self.bearers: list[str] = []
        self.discovered: list[str] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.startswith('/.well-known/oauth-protected-resource'):
            self.discovered.append(path)
            return httpx.Response(self.resource[0], json=self.resource[1])
        if path.startswith('/.well-known/oauth-authorization-server'):
            self.discovered.append(path)
            return httpx.Response(200, json=self.metadata)
        if path == '/api/oauth/register':
            self.registered.append(json.loads(request.content))
            return httpx.Response(self.registration[0], json=self.registration[1])
        if path == '/mcp':
            bearer = request.headers['Authorization'].removeprefix('Bearer ')
            self.bearers.append(bearer)
            return httpx.Response(200 if bearer in self.valid else 401)
        form = {key: value for key, [value] in parse_qs(request.content.decode()).items()}
        self.forms.append(form)
        if path == '/api/oauth/device/code':
            return httpx.Response(self.device[0], json=self.device[1])
        code, body = (self.refreshes if form['grant_type'] == 'refresh_token' else self.polls).pop(0)
        return httpx.Response(code, json=body)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handle))


def refuse(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError('refused', request=request)


def granted(access: str, *, refresh: str | None = None, **extra: JsonValue) -> Reply:
    body: dict[str, JsonValue] = {'access_token': access, 'token_type': 'Bearer', **extra}
    if refresh is not None:
        body['refresh_token'] = refresh
    return 200, body


def pending(error: str = 'authorization_pending') -> Reply:
    return 400, {'error': error, 'error_description': error.replace('_', ' ')}


@pytest.fixture(autouse=True)
def opened(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the links CLAI opens instead of opening a browser."""
    links: list[str] = []

    def open_link(url: str) -> bool:
        links.append(url)
        return True

    monkeypatch.setattr(webbrowser, 'open', open_link)
    return links


async def run_sign_in(logfire: Logfire, *, read_only: bool = True, sleeps: list[float] | None = None) -> list[str]:
    lines: list[str] = []

    async def sleep(seconds: float) -> None:
        (sleeps if sleeps is not None else []).append(seconds)

    async with logfire.client() as http:
        await sign_in(resource=RESOURCE, read_only=read_only, announce=lines.append, http=http, sleep=sleep)
    return lines


def stored(*, expires_in: float = 3600, refresh: str | None = 'refresh-1', writable: bool = False) -> Tokens:
    return Tokens(
        client_id='client-1',
        token_endpoint=f'{ORIGIN}/api/oauth/token',
        access_token='access-1',
        refresh_token=refresh,
        expires_at=time.time() + expires_in,
        writable=writable,
    )


def remember(tokens: Tokens) -> None:
    save_codex_credentials(value=json.dumps({RESOURCE: tokens.model_dump(mode='json')}), account=logfire_oauth.ACCOUNT)


async def no_wait(seconds: float) -> None:
    pass


class TestSignIn:
    async def test_announces_the_link_and_code_opens_the_browser_and_saves_tokens(self, opened: list[str]) -> None:
        logfire = Logfire()
        logfire.polls = [pending(), pending('slow_down'), *logfire.polls]
        sleeps: list[float] = []
        lines = await run_sign_in(logfire, sleeps=sleeps)
        assert lines == [
            f'Sign in to Logfire (new users can sign up there): open {LINK}',
            'Enter code: ABCD-EFGH',
            'Approve only the code shown here. You can open the link on another device.',
            'Signed in to Logfire.',
        ]
        assert opened == [LINK]
        assert sleeps == [1, 1, 6]  # Logfire's 0 is raised to 1 second; slow_down adds 5.
        [registered] = logfire.registered
        assert registered == {
            'client_name': 'CLAI',
            'client_uri': 'https://github.com/pydantic/pydantic-ai',
            'grant_types': ['urn:ietf:params:oauth:grant-type:device_code', 'refresh_token'],
            'token_endpoint_auth_method': 'none',
            'application_type': 'native',
            'scope': 'project:read',
        }
        device, *polls = logfire.forms
        assert (device['scope'], device['code_challenge_method']) == ('project:read', 'S256')
        assert {form['resource'] for form in logfire.forms} == {RESOURCE}
        assert logfire.discovered == [
            '/.well-known/oauth-protected-resource/mcp',
            '/.well-known/oauth-authorization-server',
        ]
        assert {poll['code_verifier'] for poll in polls} != {''}
        tokens = load(RESOURCE)
        assert tokens is not None
        assert (tokens.access_token, tokens.refresh_token, tokens.writable) == ('access-1', 'refresh-1', False)
        assert tokens.fresh()

    async def test_write_access_asks_for_every_offered_scope_with_a_new_client(self) -> None:
        remember(stored())
        logfire = Logfire()
        await run_sign_in(logfire, read_only=False)
        assert logfire.forms[0]['scope'] == ' '.join(OFFERED)
        assert len(logfire.registered) == 1
        tokens = load(RESOURCE)
        assert tokens is not None and tokens.writable

    async def test_a_second_read_only_sign_in_reuses_the_registered_client(self) -> None:
        remember(stored(expires_in=-10))
        logfire = Logfire()
        await run_sign_in(logfire)
        assert logfire.registered == []
        assert logfire.forms[0]['client_id'] == 'client-1'

    async def test_a_client_the_server_forgot_is_registered_again(self) -> None:
        remember(stored(expires_in=-10))
        logfire = Logfire()
        handle = logfire.handle

        def forgot_client_1(request: httpx.Request) -> httpx.Response:
            if request.url.path == '/api/oauth/device/code' and b'client_id=client-1' in request.content:
                return httpx.Response(401, json={'error': 'invalid_client', 'error_description': 'Unknown client_id'})
            return handle(request)

        logfire.handle = forgot_client_1
        logfire.registration = (201, {'client_id': 'client-2'})
        await run_sign_in(logfire)
        assert len(logfire.registered) == 1
        tokens = load(RESOURCE)
        assert tokens is not None and tokens.client_id == 'client-2'

    async def test_the_link_is_shown_when_no_browser_opens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def no_browser(url: str) -> bool:
            raise webbrowser.Error('could not locate runnable browser')

        monkeypatch.setattr(webbrowser, 'open', no_browser)
        logfire = Logfire()
        logfire.device = (200, {**DEVICE, 'verification_uri_complete': None})
        lines = await run_sign_in(logfire)
        assert lines[0] == f'Sign in to Logfire (new users can sign up there): open {ORIGIN}/auth/oauth-device'
        assert 'No browser opened; open the link above yourself.' in lines

    async def test_fewer_scopes_than_asked_are_reported_once_not_asked_for_again(self) -> None:
        logfire = Logfire()
        logfire.polls = [granted('access-1', scope='project:read')]
        lines = await run_sign_in(logfire, read_only=False)
        assert 'Logfire granted fewer scopes than asked; some write tools may be refused.' in lines
        tokens = load(RESOURCE)
        assert tokens is not None and tokens.serves(read_only=False)  # Asking again would loop on the same grant.
        logfire.polls = [granted('access-1', scope=' '.join(OFFERED))]
        assert not any('fewer scopes' in line for line in await run_sign_in(logfire, read_only=False))

    async def test_logging_out_while_waiting_for_approval_discards_the_sign_in(self) -> None:
        logfire = Logfire()

        async def log_out(seconds: float) -> None:
            forget()

        async with logfire.client() as http:
            with pytest.raises(SignInError, match='Signed out of Logfire while signing in'):
                await sign_in(resource=RESOURCE, read_only=True, announce=lambda line: None, http=http, sleep=log_out)
        assert load(RESOURCE) is None
        logfire.polls = [granted('access-1')]
        await run_sign_in(logfire)  # A sign-in started after the logout is kept.
        assert load(RESOURCE) is not None

    async def test_a_negative_interval_still_pauses_between_polls(self) -> None:
        logfire = Logfire()
        logfire.device = (200, {**DEVICE, 'interval': -3})
        sleeps: list[float] = []
        await run_sign_in(logfire, sleeps=sleeps)
        assert sleeps == [1]

    async def test_a_huge_interval_waits_no_longer_than_the_code_lasts(self) -> None:
        logfire = Logfire()
        logfire.device = (200, {**DEVICE, 'interval': 999_999_999})
        sleeps: list[float] = []
        await run_sign_in(logfire, sleeps=sleeps)
        [waited] = sleeps
        assert 590 < waited <= 600

    async def test_a_dropped_poll_keeps_waiting_for_approval(self) -> None:
        logfire = Logfire()
        handle = logfire.handle
        dropped: list[bool] = []

        def drop_first_poll(request: httpx.Request) -> httpx.Response:
            if b'grant_type=urn' in request.content and not dropped:
                dropped.append(True)
                raise httpx.ReadTimeout('slow', request=request)
            return handle(request)

        logfire.handle = drop_first_poll
        sleeps: list[float] = []
        assert (await run_sign_in(logfire, sleeps=sleeps))[-1] == 'Signed in to Logfire.'
        assert dropped == [True]
        assert sleeps == [1, 6]  # A dropped poll backs off like slow_down.

    @pytest.mark.parametrize(
        ('resource', 'metadata_path'),
        [
            (f'{ORIGIN}/tenant/mcp/', '/.well-known/oauth-protected-resource/tenant/mcp/'),
            (f'{ORIGIN}/', '/.well-known/oauth-protected-resource'),
        ],
    )
    async def test_metadata_is_found_at_the_exact_resource_path(self, resource: str, metadata_path: str) -> None:
        logfire = Logfire()
        logfire.resource = (200, {'resource': resource, 'authorization_servers': [ORIGIN]})
        async with logfire.client() as http:
            await sign_in(resource=resource, read_only=True, announce=lambda line: None, http=http, sleep=no_wait)
        assert logfire.discovered == [metadata_path, '/.well-known/oauth-authorization-server']
        assert load(resource) is not None

    async def test_servers_without_resource_metadata_are_their_own_issuer(self) -> None:
        logfire = Logfire()
        logfire.resource = (404, None)
        await run_sign_in(logfire, read_only=False)
        assert logfire.forms[0]['scope'] == 'project:read'  # Nothing offered beyond what MCP requires.
        tokens = load(RESOURCE)
        assert tokens is not None and tokens.serves(read_only=False)  # So write tools do not sign in on every request.

    async def test_a_path_issuer_is_discovered_the_rfc_8414_way(self) -> None:
        logfire = Logfire()
        logfire.resource = (200, {'resource': RESOURCE, 'authorization_servers': [f'{ORIGIN}/tenant/']})
        logfire.metadata = {**METADATA, 'issuer': f'{ORIGIN}/tenant'}
        await run_sign_in(logfire)
        assert logfire.discovered[1] == '/.well-known/oauth-authorization-server/tenant'

    @pytest.mark.parametrize(
        ('broken', 'value', 'message'),
        [
            ('polls', [pending('access_denied')], 'Logfire sign-in was denied. Run /logfire_mcp login to retry.'),
            ('polls', [pending('invalid_grant')], 'Logfire sign-in failed: invalid grant. Run /logfire_mcp login'),
            ('polls', [(200, {'access_token': ''})], 'Logfire sign-in failed: ValidationError.'),
            ('polls', [granted('access-1', token_type='DPoP')], 'Logfire sign-in failed: ValidationError.'),
            ('polls', [granted('access-1', scope='project:write')], 'Logfire did not grant project:read'),
            ('device', (200, {**DEVICE, 'verification_uri_complete': 'http://x.test/d'}), 'failed: ValidationError.'),
            ('device', (200, {**DEVICE, 'expires_in': 0}), 'The Logfire sign-in code expired before it was approved.'),
            ('device', (503, 'unavailable'), 'Logfire refused browser sign-in: HTTP 503'),
            (
                'device',
                (400, {'detail': {'error': 'invalid_client', 'error_description': 'PKCE is required'}}),
                'Logfire refused browser sign-in: PKCE is required',
            ),
            ('device', (400, ['not', 'an', 'object']), 'Logfire refused browser sign-in: HTTP 400'),
            (
                'metadata',
                {**METADATA, 'registration_endpoint': None},
                'This Logfire server does not let CLAI register for browser sign-in.',
            ),
            ('metadata', {**METADATA, 'token_endpoint': 'http://logfire.test/token'}, 'failed: ValidationError.'),
            (
                'registration',
                (400, {'error': 'invalid_client_metadata', 'error_description': 'nope'}),
                'Logfire refused to register CLAI: nope',
            ),
            (
                'resource',
                (200, {'resource': RESOURCE, 'authorization_servers': ['http://logfire.test']}),
                'Logfire sign-in failed: ValidationError.',
            ),
            (
                'resource',
                (200, {'resource': 'https://logfire-us.pydantic.dev/mcp', 'authorization_servers': [ORIGIN]}),
                f'{RESOURCE} described itself as https://logfire-us.pydantic.dev/mcp; not signing in.',
            ),
            (
                'metadata',
                {**METADATA, 'issuer': 'https://elsewhere.test'},
                f'{ORIGIN} described itself as https://elsewhere.test; not signing in.',
            ),
            ('handle', refuse, 'Logfire sign-in failed: ConnectError. Run /logfire_mcp login to retry.'),
        ],
        ids=[
            'denied',
            'failed',
            'malformed',
            'not-bearer',
            'no-read-scope',
            'plain-http-link',
            'expired',
            'device-down',
            'nested-error',
            'odd-error',
            'no-registration',
            'plain-http',
            'registration',
            'plain-http-issuer',
            'other-resource',
            'other-issuer',
            'unreachable',
        ],
    )
    async def test_failures_explain_how_to_retry_and_save_nothing(
        self, broken: str, value: object, message: str
    ) -> None:
        logfire = Logfire()
        setattr(logfire, broken, value)
        with pytest.raises(SignInError, match=re.escape(message)):
            await run_sign_in(logfire)
        assert load(RESOURCE) is None

    async def test_non_json_errors_report_the_status(self) -> None:
        logfire = Logfire()
        handle = logfire.handle

        def html_error(request: httpx.Request) -> httpx.Response:
            if request.url.path == '/api/oauth/device/code':
                return httpx.Response(502, text='<html>bad gateway</html>')
            return handle(request)

        logfire.handle = html_error
        with pytest.raises(SignInError, match='Logfire refused browser sign-in: HTTP 502'):
            await run_sign_in(logfire)


class TestDeviceAuth:
    async def mcp(self, logfire: Logfire, *, read_only: bool = True) -> int:
        auth = DeviceAuth(resource=RESOURCE, read_only=read_only, http=logfire.client)
        async with httpx.AsyncClient(transport=httpx.MockTransport(logfire.handle), auth=auth) as client:
            return (await client.post(RESOURCE)).status_code

    async def test_a_connection_without_a_sign_in_never_starts_one(self, opened: list[str]) -> None:
        logfire = Logfire()
        with pytest.raises(SignInRequired, match=r'Not signed in to Logfire\. Run /logfire_mcp login to sign in\.'):
            await self.mcp(logfire)
        assert (opened, logfire.forms, logfire.bearers) == ([], [], [])

    async def test_a_stored_sign_in_is_used_without_asking_again(self, opened: list[str]) -> None:
        remember(stored())
        logfire = Logfire()
        assert await self.mcp(logfire) == 200
        assert (opened, logfire.forms) == ([], [])

    async def test_expired_or_rejected_tokens_are_refreshed(self) -> None:
        remember(stored(expires_in=-10))
        logfire = Logfire()
        logfire.valid = {'access-2', 'access-3'}
        logfire.refreshes = [granted('access-2'), granted('access-3')]
        assert await self.mcp(logfire) == 200
        tokens = load(RESOURCE)
        assert tokens is not None
        assert (tokens.access_token, tokens.refresh_token) == ('access-2', 'refresh-1')
        logfire.valid = {'access-3'}  # The server revokes access-2 before it expires.
        assert await self.mcp(logfire) == 200
        assert logfire.bearers == ['access-2', 'access-2', 'access-3']
        assert [form['refresh_token'] for form in logfire.forms] == ['refresh-1', 'refresh-1']
        assert {form['resource'] for form in logfire.forms} == {RESOURCE}

    @pytest.mark.parametrize('refresh', [None, 'broken', 'unreachable', 'rejected', 'no-read-scope'])
    async def test_a_failed_refresh_asks_for_a_sign_in(self, refresh: str | None, opened: list[str]) -> None:
        remember(stored(expires_in=-10, refresh=None if refresh is None else 'refresh-1'))
        logfire = Logfire()
        if refresh == 'broken':
            logfire.refreshes = [granted('access-2', token_type='MAC')]
        elif refresh == 'rejected':
            logfire.refreshes = [pending('invalid_grant')]
        elif refresh == 'no-read-scope':
            logfire.refreshes = [granted('access-2', scope='project:write')]
        elif refresh == 'unreachable':

            def drop_refresh(request: httpx.Request) -> httpx.Response:
                assert b'grant_type=refresh_token' in request.content, 'the refresh comes first'
                raise httpx.ReadTimeout('slow', request=request)

            logfire.handle = drop_refresh
        with pytest.raises(SignInRequired):
            await self.mcp(logfire)
        assert opened == []

    async def test_a_read_only_sign_in_does_not_serve_write_access(self, opened: list[str]) -> None:
        remember(stored())
        logfire = Logfire()
        with pytest.raises(SignInRequired):
            await self.mcp(logfire, read_only=False)
        assert (opened, logfire.forms) == ([], [])

    async def test_logging_out_during_a_refresh_discards_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        remember(stored(expires_in=-10))
        logfire = Logfire()
        logfire.valid = {'access-2'}
        logfire.refreshes = [granted('access-2')]
        handle = logfire.handle

        def log_out_first(request: httpx.Request) -> httpx.Response:
            # The expired token is refreshed before anything else is sent.
            assert b'grant_type=refresh_token' in request.content
            assert forget()
            return handle(request)

        logfire.handle = log_out_first
        with pytest.raises(SignInError, match='Signed out of Logfire while signing in'):
            await self.mcp(logfire)
        assert load(RESOURCE) is None

    def test_sync_clients_are_refused(self) -> None:
        auth = DeviceAuth(resource=RESOURCE, read_only=True)
        with httpx.Client(transport=httpx.MockTransport(Logfire().handle), auth=auth) as client:
            with pytest.raises(RuntimeError, match='Logfire sign-in needs an async client'):
                client.post(RESOURCE)


class TestLogfireSignIn:
    def method(self, logfire: Logfire, *, read_only: bool = True) -> LogfireSignIn:
        return LogfireSignIn(resource=RESOURCE, read_only=read_only, http=logfire.client, sleep=no_wait)

    async def test_signs_in_on_request_showing_every_line_and_then_runs_connect(self, opened: list[str]) -> None:
        logfire = Logfire()
        method = self.method(logfire)
        assert (method.service, method.setup) == ('Logfire', '/logfire_mcp login')
        assert not method.signed_in()
        shown: list[str] = []
        await method.sign_in(show=shown.append)
        assert opened == [LINK]
        assert shown[-1].splitlines() == [
            f'Sign in to Logfire (new users can sign up there): open {LINK}',
            'Enter code: ABCD-EFGH',
            'Approve only the code shown here. You can open the link on another device.',
            'Signed in to Logfire.',
        ]
        assert method.signed_in() and not self.method(logfire, read_only=False).signed_in()
        assert await TestDeviceAuth().mcp(logfire) == 200
        method.sign_out()
        assert not method.signed_in()

    async def test_text_from_the_server_cannot_drive_the_terminal(self) -> None:
        logfire = Logfire()
        logfire.device = (200, {**DEVICE, 'user_code': '\x1b]52;c;eA==\x07'})
        shown: list[str] = []
        await self.method(logfire).sign_in(show=shown.append)
        assert 'Enter code: \\x1b]52;c;eA==\\x07' in shown[-1].splitlines()

    async def test_a_sign_in_the_store_refuses_lasts_the_session(
        self, monkeypatch: pytest.MonkeyPatch, opened: list[str]
    ) -> None:
        def locked(service: str, account: str, value: str) -> None:
            raise KeyringError('locked')

        monkeypatch.setattr(keyring, 'set_password', locked)
        logfire = Logfire()
        shown: list[str] = []
        await self.method(logfire).sign_in(show=shown.append)
        assert shown[-1].endswith(
            'Signed in to Logfire for this session only: saving the sign-in failed (KeyringError).'
        )
        auth = DeviceAuth(resource=RESOURCE, read_only=True, http=logfire.client)
        async with httpx.AsyncClient(transport=httpx.MockTransport(logfire.handle), auth=auth) as client:
            for _ in range(2):
                assert (await client.post(RESOURCE)).status_code == 200
            assert (opened, logfire.bearers, status(resource=RESOURCE, read_only=True)) == (
                [LINK],
                ['access-1', 'access-1'],
                'signed in',
            )
            assert forget()  # Logout drops the in-memory sign-in too, so the next request needs a new one.
            assert status(resource=RESOURCE, read_only=True) == 'signed out'
            with pytest.raises(SignInRequired):
                await client.post(RESOURCE)
        assert opened == [LINK]
        assert not forget()


def test_an_unreadable_sign_in_does_not_discard_the_others() -> None:
    other = 'https://other.test/mcp'
    save_codex_credentials(
        value=json.dumps({RESOURCE: stored().model_dump(mode='json'), other: {'access_token': 'partial'}}),
        account=logfire_oauth.ACCOUNT,
    )
    kept = load(RESOURCE)
    assert kept is not None and kept.access_token == 'access-1'
    assert load(other) is None
    assert logfire_oauth._save(other, stored(), logfire_oauth._logouts) is None  # pyright: ignore[reportPrivateUsage]
    assert load(RESOURCE) is not None
    assert load(other) is not None


def test_status_forget_and_unreadable_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    assert status(resource=RESOURCE, read_only=True) == 'signed out'
    remember(stored(writable=False))
    assert status(resource=RESOURCE, read_only=True) == 'signed in'
    assert status(resource=RESOURCE, read_only=False) == 'signed out'
    remember(stored(expires_in=-10))
    assert status(resource=RESOURCE, read_only=True) == 'signed in'  # It refreshes on the next run.
    remember(stored(expires_in=-10, refresh=None))
    assert status(resource=RESOURCE, read_only=True) == 'expired'
    assert forget() is True
    assert forget() is False
    save_codex_credentials(value='not json', account=logfire_oauth.ACCOUNT)
    assert load(RESOURCE) is None
    assert forget() is True  # Unparsable tokens are deleted too, so they cannot come back.
    assert forget() is False

    def unreadable(service: str, account: str) -> str:
        raise KeyringError('locked')

    monkeypatch.setattr(keyring, 'get_password', unreadable)
    deleted: list[str] = []

    def delete_credentials(*, account: str) -> None:
        deleted.append(account)

    monkeypatch.setattr(logfire_oauth, 'delete_credentials', delete_credentials)
    assert load(RESOURCE) is None
    assert forget() is True  # Unreadable is not gone, so it is deleted.
    assert deleted == [logfire_oauth.ACCOUNT]
