"""Exercise PKCE with real loopback callbacks and a mocked exchange endpoint."""

import asyncio
import base64
import hashlib
import threading
import webbrowser
from io import StringIO
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import anyio
import httpx
import pytest
from menu_script import make_context
from pydantic import SecretStr, TypeAdapter
from pydantic_ai.exceptions import UserError
from rich.console import Console

from pydantic_clai2 import openrouter
from pydantic_clai2.openrouter_auth import OpenRouterAuth, authorization_code


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.mark.parametrize('text', ['code', 'http://127.0.0.1:123/callback?code=code', '/callback?code=code'])
def test_parse_code(text: str) -> None:
    assert authorization_code(text=text) == 'code'


@pytest.mark.parametrize(
    'text', ['', 'https://localhost/callback', '/callback?error=secret', '/callback?code=a&code=b']
)
def test_invalid_code(text: str) -> None:
    with pytest.raises(UserError) as error:
        authorization_code(text=text)
    assert 'secret' not in str(error.value)


@pytest.mark.parametrize('mode', ['callback', 'paste', 'manual', 'browser_error', 'denied', 'eof', 'timeout', 'cancel'])
async def test_login(mode: str) -> None:
    output = StringIO()
    console = Console(file=output)
    browser_ready = asyncio.Event()
    prompt_closed = asyncio.Event()
    prompt_started = asyncio.Event()
    urls: list[str] = []
    requests: list[httpx.Request] = []
    loop = asyncio.get_running_loop()

    def browser(url: str) -> bool:
        urls.append(url)
        loop.call_soon_threadsafe(browser_ready.set)
        if mode == 'browser_error':
            raise webbrowser.Error('unavailable')
        return mode != 'manual'

    lines = iter(['', 'http://127.0.0.1/callback?code=secret-code'])

    async def read_line(message: str) -> str:
        prompt_started.set()
        try:
            if mode in ('paste', 'manual', 'browser_error'):
                return next(lines)
            if mode == 'eof':
                raise EOFError
            await asyncio.Future[None]()
            raise AssertionError('unreachable')
        finally:
            prompt_closed.set()

    def exchange(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert str(request.url) == 'https://openrouter.ai/api/v1/auth/keys'
        data = TypeAdapter(dict[str, str]).validate_json(request.content)
        assert data['code'] == 'secret-code'
        verifier = data['code_verifier']
        assert 43 <= len(verifier) <= 128
        expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b'=').decode()
        params = parse_qs(urlparse(urls[0]).query)
        assert params['code_challenge'] == [expected]
        assert params['code_challenge_method'] == [data['code_challenge_method']] == ['S256']
        return httpx.Response(200, json={'key': 'private-api-key'})

    auth = OpenRouterAuth(
        console=console,
        read_line=read_line,
        open_browser=browser,
        transport=httpx.MockTransport(exchange),
        timeout=0.2 if mode == 'timeout' else 5,
    )
    login = asyncio.create_task(auth.login())
    await browser_ready.wait()
    callback = parse_qs(urlparse(urls[0]).query)['callback_url'][0]
    if mode in ('callback', 'denied'):
        async with httpx.AsyncClient() as client:
            response = await client.get(callback.replace('/callback', '/favicon.ico'))
            assert response.status_code == 404
            response = await client.get(callback)
            assert response.status_code == 400
            response = await client.get(
                callback + ('?error=private-error' if mode == 'denied' else '?code=secret-code')
            )
            assert response.status_code == (400 if mode == 'denied' else 200)
    if mode == 'cancel':
        await prompt_started.wait()
        login.cancel()
        with pytest.raises(asyncio.CancelledError):
            await login
    elif mode in ('denied', 'eof', 'timeout'):
        with pytest.raises(UserError, match={'denied': 'denied', 'eof': 'cancelled', 'timeout': 'timed out'}[mode]):
            await login
    else:
        assert (await login).get_secret_value() == 'private-api-key'
        assert len(requests) == 1
    assert prompt_closed.is_set()
    with pytest.raises(httpx.ConnectError):
        async with httpx.AsyncClient() as client:
            await client.get(callback)
    assert 'private-api-key' not in output.getvalue()
    assert 'secret-code' not in output.getvalue()
    if mode in ('manual', 'browser_error'):
        assert 'manually' in output.getvalue()


@pytest.mark.parametrize('status,body', [(401, {'key': 'secret'}), (302, {}), (200, {}), (200, {'key': ''})])
async def test_exchange_failure(status: int, body: dict[str, str]) -> None:
    auth = OpenRouterAuth(
        console=Console(file=StringIO()),
        transport=httpx.MockTransport(lambda request: httpx.Response(status, json=body)),
    )
    with pytest.raises(UserError, match='key exchange failed') as error:
        await auth.exchange(code='secret-code', verifier='secret-verifier')
    assert 'secret' not in str(error.value)


async def test_connect_browser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context, _ = make_context(tmp_path)
    keys = iter(['enter'])
    monkeypatch.setattr(openrouter, 'menu_key', lambda: next(keys))

    async def login(self: OpenRouterAuth) -> SecretStr:
        return SecretStr('browser-key')

    async def discover(connection: openrouter.Connection) -> list[str]:
        assert isinstance(connection.token, SecretStr)
        assert connection.token.get_secret_value() == 'browser-key'
        return ['my/model']

    monkeypatch.setattr(OpenRouterAuth, 'login', login)
    monkeypatch.setattr(openrouter, 'discover', discover)

    def choose(names: list[str]) -> str:
        return names[0]

    monkeypatch.setattr(openrouter, 'choose', choose)
    assert await openrouter.connect(context, []) == 'Saved model. Applied.'
    assert openrouter.model('openrouter:my/model').model_name == 'my/model'


async def test_cancel_authentication_menu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(openrouter, 'menu_key', lambda: 'escape')
    assert await openrouter.prompt_connection() is None


async def test_listener_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def unavailable(*args: object, **kwargs: object) -> None:
        raise OSError('unavailable')

    monkeypatch.setattr(asyncio, 'start_server', unavailable)
    with pytest.raises(UserError, match='callback listener'):
        await OpenRouterAuth(console=Console(file=StringIO())).login()


async def test_network_failure() -> None:
    def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError('secret', request=request)

    auth = OpenRouterAuth(console=Console(file=StringIO()), transport=httpx.MockTransport(unavailable))
    with pytest.raises(UserError, match='key exchange failed') as error:
        await auth.exchange(code='secret', verifier='secret')
    assert 'secret' not in str(error.value)


@pytest.mark.parametrize('cancel_model', [True, False])
async def test_browser_failure_preserves_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel_model: bool
) -> None:
    context, _ = make_context(tmp_path)
    original = openrouter.Connection(token=SecretStr('original'))
    openrouter.save_connection(original)
    monkeypatch.setattr(openrouter, 'connection_action', lambda: 'configure')
    monkeypatch.setattr(openrouter, 'menu_key', lambda: 'enter')

    async def login(self: OpenRouterAuth) -> SecretStr:
        if not cancel_model:
            raise UserError('Login failed')
        return SecretStr('replacement')

    async def discover(connection: openrouter.Connection) -> list[str]:
        return ['my/model']

    def choose(names: list[str]) -> None:
        return None

    monkeypatch.setattr(OpenRouterAuth, 'login', login)
    monkeypatch.setattr(openrouter, 'discover', discover)
    monkeypatch.setattr(openrouter, 'choose', choose)
    if cancel_model:
        assert await openrouter.connect(context, []) == 'Connection cancelled.'
    else:
        with pytest.raises(UserError, match='Login failed'):
            await openrouter.connect(context, [])
    raw = openrouter.load_codex_credentials(account='openrouter')
    assert raw is not None
    assert openrouter.Connection.model_validate_json(raw) == original
    assert context.settings.model != 'openrouter:my/model'


async def test_cancel_before_browser_returns() -> None:
    browser_ready = asyncio.Event()
    release = threading.Event()
    browser_done = asyncio.Event()
    loop = asyncio.get_running_loop()

    def browser(url: str) -> bool:
        loop.call_soon_threadsafe(browser_ready.set)
        release.wait()
        loop.call_soon_threadsafe(browser_done.set)
        return True

    async def unexpected_prompt(message: str) -> str:
        raise AssertionError('Cancelled before opening the prompt')

    auth = OpenRouterAuth(console=Console(file=StringIO()), open_browser=browser, read_line=unexpected_prompt)
    login = asyncio.create_task(auth.login())
    try:
        await browser_ready.wait()
        login.cancel()
        with pytest.raises(asyncio.CancelledError):
            await login
    finally:
        release.set()
        await browser_done.wait()


async def test_callback_clients_during_exchange(monkeypatch: pytest.MonkeyPatch) -> None:
    browser_ready = asyncio.Event()
    exchange_started = asyncio.Event()
    release_exchange = asyncio.Event()
    loop = asyncio.get_running_loop()
    urls: list[str] = []

    def browser(url: str) -> bool:
        urls.append(url)
        loop.call_soon_threadsafe(browser_ready.set)
        return True

    async def prompt(message: str) -> str:
        await asyncio.Future[None]()
        raise AssertionError('unreachable')

    async def exchange(request: httpx.Request) -> httpx.Response:
        exchange_started.set()
        await release_exchange.wait()
        return httpx.Response(200, json={'key': 'key'})

    original = asyncio.StreamWriter.wait_closed

    async def disconnected(writer: asyncio.StreamWriter) -> None:
        await original(writer)
        raise ConnectionError('client disconnected during close')

    monkeypatch.setattr(asyncio.StreamWriter, 'wait_closed', disconnected)
    auth = OpenRouterAuth(
        console=Console(file=StringIO()),
        open_browser=browser,
        read_line=prompt,
        transport=httpx.MockTransport(exchange),
    )
    login = asyncio.create_task(auth.login())
    try:
        await browser_ready.wait()
        callback = parse_qs(urlparse(urls[0]).query)['callback_url'][0]
        async with httpx.AsyncClient() as client:
            port = urlparse(callback).port
            assert port is not None
            async with await anyio.connect_tcp('127.0.0.1', port) as stream:
                await stream.send(b'GET /callback?code=' + b'x' * 70000 + b' HTTP/1.1\r\n\r\n')
                with pytest.raises((anyio.EndOfStream, anyio.BrokenResourceError)):
                    await stream.receive()
            assert (await client.post(callback)).status_code == 404
            assert (await client.get(callback + '?code=first')).status_code == 200
            await exchange_started.wait()
            assert (await client.get(callback + '?code=duplicate')).status_code == 200
            assert (await client.get(callback + '?error=late-denial')).status_code == 400
            release_exchange.set()
        assert (await login).get_secret_value() == 'key'
    finally:
        login.cancel()
        await asyncio.gather(login, return_exceptions=True)
