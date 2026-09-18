"""OpenRouter's browser authorization-code flow with S256 PKCE.

Protocol: https://openrouter.ai/docs/use-cases/oauth-pkce. OpenRouter returns
an API key, not refresh tokens, and uses PKCE rather than an OAuth state field.
"""

import asyncio
import base64
import hashlib
import secrets
import webbrowser
from collections.abc import Callable
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
from pydantic import BaseModel, Field, SecretStr, ValidationError
from pydantic_ai.exceptions import UserError
from rich.console import Console

from . import theme
from .auth import ReadLine, read_line


class KeyResponse(BaseModel):
    """Only accept a nonempty key from the exchange endpoint."""

    key: SecretStr = Field(min_length=1)


def authorization_code(*, text: str) -> str:
    """Accept a pasted callback URL or a bare authorization code without echoing it."""
    parsed = urlparse(text)
    if parsed.scheme or parsed.netloc or text.startswith('/'):
        params = parse_qs(parsed.query)
        if 'error' in params:
            raise UserError('OpenRouter authorization was denied. Try connecting again.')
        codes = params.get('code', [])
        if len(codes) != 1 or not codes[0].strip():
            raise UserError('The callback URL must contain one authorization code.')
        return codes[0]
    if not text.strip():
        raise UserError('An authorization code is required.')
    return text.strip()


class OpenRouterAuth:
    """Own one browser login, including its listener and terminal prompt."""

    def __init__(
        self,
        *,
        console: Console,
        read_line: ReadLine = read_line,
        open_browser: Callable[[str], bool] = webbrowser.open,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 300,
    ) -> None:
        """Inject terminal and HTTP boundaries without opening a listener yet."""
        self.console = console
        self.read_line = read_line
        self.open_browser = open_browser
        self.transport = transport
        self.timeout = timeout

    async def login(self) -> SecretStr:
        """Return a key without saving it; cancellation leaves existing credentials alone."""
        verifier = secrets.token_urlsafe(64)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode('ascii')).digest()).rstrip(b'=').decode()
        code: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        handlers: set[asyncio.Task[None]] = set()

        def connected(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            task = asyncio.create_task(self._callback(reader=reader, writer=writer, code=code))
            handlers.add(task)
            task.add_done_callback(handlers.discard)

        try:
            server = await asyncio.start_server(connected, '127.0.0.1', 0)
        except OSError:
            raise UserError(
                'Could not start the OpenRouter callback listener. Try again or enter an API key.'
            ) from None
        port = server.sockets[0].getsockname()[1]
        url = 'https://openrouter.ai/auth?' + urlencode(
            {
                'callback_url': f'http://127.0.0.1:{port}/callback',
                'code_challenge': challenge,
                'code_challenge_method': 'S256',
            }
        )
        paste: asyncio.Task[str] | None = None
        try:
            async with server, asyncio.timeout(self.timeout):
                self.console.print(
                    'Sign in to OpenRouter in your browser. Waiting up to five minutes.', style=theme.INFO
                )
                self.console.print(url, markup=False, highlight=False)
                try:
                    opened = await asyncio.to_thread(self.open_browser, url)
                except webbrowser.Error:
                    opened = False
                if not opened:
                    self.console.print('Open the URL above manually.', style=theme.WARNING)
                paste = asyncio.create_task(self._paste())
                done, _ = await asyncio.wait({code, paste}, return_when=asyncio.FIRST_COMPLETED)
                received = code.result() if code in done else paste.result()
                return await self.exchange(code=received, verifier=verifier)
        except TimeoutError:
            raise UserError('OpenRouter login timed out. Connect again through /add_model > openrouter.') from None
        finally:
            code.cancel()
            tasks: list[asyncio.Task[None] | asyncio.Task[str]] = [*handlers]
            if paste is not None:
                tasks.append(paste)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*handlers, return_exceptions=True)
            if paste is not None:
                await asyncio.gather(paste, return_exceptions=True)
            await asyncio.gather(code, return_exceptions=True)

    async def _paste(self) -> str:
        try:
            while not (
                text := (await self.read_line('Finish in the browser, or paste its callback URL or code: ')).strip()
            ):
                pass
        except (EOFError, KeyboardInterrupt):
            raise UserError('OpenRouter login cancelled.') from None
        return authorization_code(text=text)

    async def _callback(
        self, *, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, code: asyncio.Future[str]
    ) -> None:
        try:
            async with asyncio.timeout(10):
                line = (await reader.readline()).decode('ascii', errors='replace').split()
                status, message = '404 Not Found', 'Callback endpoint not found.'
                if len(line) == 3 and line[0] == 'GET' and urlparse(line[1]).path == '/callback':
                    try:
                        received = authorization_code(text=line[1])
                    except UserError as exc:
                        if 'error' in parse_qs(urlparse(line[1]).query) and not code.done():
                            code.set_exception(exc)
                        status, message = '400 Bad Request', 'Authorization failed. Return to CLAI and try again.'
                    else:
                        status, message = '200 OK', 'Authorization received. Return to CLAI to finish connecting.'
                        if not code.done():
                            code.set_result(received)
                body = message.encode()
                writer.write(
                    (
                        f'HTTP/1.1 {status}\r\nContent-Type: text/plain\r\nContent-Length: {len(body)}\r\n'
                        'Connection: close\r\n\r\n'
                    ).encode()
                    + body
                )
                await writer.drain()
        except (TimeoutError, ConnectionError, ValueError):
            pass  # Malformed or disconnected local clients must not end a login.
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass

    async def exchange(self, *, code: str, verifier: str) -> SecretStr:
        """Exchange only at OpenRouter; never expose response bodies or follow redirects."""
        async with httpx.AsyncClient(transport=self.transport, timeout=30, follow_redirects=False) as client:
            try:
                response = await client.post(
                    'https://openrouter.ai/api/v1/auth/keys',
                    json={'code': code, 'code_verifier': verifier, 'code_challenge_method': 'S256'},
                )
                response.raise_for_status()
                return KeyResponse.model_validate_json(response.content).key
            except (httpx.HTTPError, ValidationError):
                raise UserError(
                    'OpenRouter key exchange failed. Connect again through /add_model > openrouter.'
                ) from None
