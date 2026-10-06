"""The shared sign-in API: sign in only on request, behind a waiting screen; runs never open a browser."""

import gc
import io
import socket
import threading
import webbrowser
from collections.abc import AsyncIterator, Callable

import anyio
import httpx
import pytest
import uvicorn
from fastmcp import FastMCP
from fastmcp.client.auth.oauth import TokenStorageAdapter
from fastmcp.client.transports import SSETransport
from fastmcp.server.auth.providers.in_memory import InMemoryOAuthProvider
from mcp.server.auth.settings import ClientRegistrationOptions
from mcp.shared.auth import OAuthToken
from rich.console import Console
from termflow.tui import MenuItem
from termflow.tui.menu import Menu, MenuResult

from pydantic_clai2.commands import Commands
from pydantic_clai2.mcp import OAuthSignIn, TokenStore
from pydantic_clai2.plugins import sign_in
from pydantic_clai2.plugins.sign_in import (
    SignInRequired,
    run_subcommand,
    sign_in_command,
    sign_in_now,
    sign_out_now,
    status,
    wait_for_sign_in,
    warn_if_signed_out,
)
from pydantic_clai2.ui.menus.field_menu import Runners
from tests.clai2.menu_script import UNTIL_CLOSED, Script

pytestmark = pytest.mark.anyio

CLOSE = MenuResult(cancelled=True)


class Service:
    """A `SignInMethod` whose sign-in waits for the test, or fails, instead of a browser."""

    service = 'Example'
    setup = '/example login'

    def __init__(self, *, state: bool | None = False, error: Exception | None = None, hang: bool = False) -> None:
        self.state = state
        self.error = error
        self.hang = hang
        self.shown: Callable[[str], object] | None = None
        self.started = threading.Event()
        """Set once the sign-in has shown its first lines, for a waiting screen that runs on another thread."""

    def signed_in(self) -> bool | None:
        return self.state

    async def sign_in(self, *, show: Callable[[str], object]) -> None:
        self.shown = show
        show('Open https://example.test/one')
        show('Open https://example.test/two')
        self.started.set()
        if self.error is not None:
            raise self.error
        if self.hang:
            await anyio.sleep_forever()
        self.state = True

    def sign_out(self) -> None:
        self.state = False


def runners(*choices: MenuResult) -> Runners:
    return Script(lists=[], choices=list(choices), texts=[]).runners


async def test_status_has_one_wording_for_each_state() -> None:
    assert status(Service(state=True)) == 'Signed in to Example.'
    assert status(Service(state=False)) == 'Not signed in to Example. Run /example login to sign in.'
    assert status(Service(state=None)) == (
        'Could not tell whether Example is signed in: the keyring could not be read.'
    )


async def test_sign_in_now_reports_success_failure_and_cancellation() -> None:
    service = Service()
    assert await sign_in_now(service, runners(UNTIL_CLOSED)) == 'Signed in to Example.'
    assert service.state is True
    failing = Service(error=RuntimeError('Client failed to connect: access_denied.'))
    assert await sign_in_now(failing, runners(UNTIL_CLOSED)) == (
        'Could not sign in to Example: Client failed to connect: access_denied. Run /example login to try again.'
    )
    waiting = Service(hang=True)
    with anyio.fail_after(5):
        message = await sign_in_now(waiting, runners(CLOSE))
    assert message == 'Example sign-in cancelled. Run /example login to try again.'
    assert waiting.state is False


async def test_the_waiting_screen_shows_the_latest_text_and_repaints_for_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sign_in, 'menu_key', lambda: 'esc')
    service = Service(hang=True)
    seen: list[tuple[str, str]] = []

    def run_choice(menu: Menu) -> MenuResult:
        assert service.started.wait(timeout=5)
        read_key: Callable[[], str] = menu._read_key  # pyright: ignore[reportPrivateUsage]
        preview = menu._preview  # pyright: ignore[reportPrivateUsage]
        assert preview is not None and service.shown is not None
        item = MenuItem('Cancel sign-in', value=None)
        seen.append((read_key(), preview(item)))
        service.shown('Enter code: ABCD')
        seen.append((read_key(), preview(item)))
        assert read_key() == 'esc', 'with nothing new, keys come from the terminal'
        return CLOSE  # Esc

    scripted = Runners(run_list=lambda menu: CLOSE, run_choice=run_choice, run_text=lambda widget: CLOSE)  # pyright: ignore[reportArgumentType]
    with anyio.fail_after(5):
        assert not await wait_for_sign_in(service, scripted)
    [(first_key, first), (second_key, second)] = seen
    assert first_key == second_key == sign_in._REDRAW  # pyright: ignore[reportPrivateUsage]
    assert first.splitlines() == [
        'Finish signing in to Example in your browser.',
        '',
        'Esc cancels; CLAI keeps working without Example.',
        '',
        'Open https://example.test/two',
    ]
    assert second.endswith('Esc cancels; CLAI keeps working without Example.\n\nEnter code: ABCD')


async def test_sign_out_and_the_session_start_notice() -> None:
    service = Service(state=True)
    output = io.StringIO()
    console = Console(file=output, width=200)
    assert await warn_if_signed_out(service, console)
    assert output.getvalue() == ''
    assert await sign_out_now(service) == 'Signed out of Example. Run /example login to sign in again.'
    assert not await warn_if_signed_out(service, console)
    assert output.getvalue() == 'Not signed in to Example. Run /example login to sign in.\n'


async def test_the_shared_command(monkeypatch: pytest.MonkeyPatch) -> None:
    service = Service()
    monkeypatch.setattr(sign_in, 'RUNNERS', runners(UNTIL_CLOSED))
    command = sign_in_command('example', service)
    commands = Commands()
    commands.register_many([command])
    assert list(command.complete([])) == ['login', 'logout', 'status']
    assert list(command.complete(['login', ''])) == []
    assert await commands.execute_async('/example') == 'Not signed in to Example. Run /example login to sign in.'
    assert await commands.execute_async('/example login') == 'Signed in to Example.'
    assert await commands.execute_async('/example status') == 'Signed in to Example.'
    assert await commands.execute_async('/example logout') == (
        'Signed out of Example. Run /example login to sign in again.'
    )
    assert await run_subcommand(service, ['key']) is None
    with pytest.raises(ValueError, match=r'Usage: /example \[login \| logout \| status\]'):
        await commands.execute_async('/example key')


@pytest.fixture
async def oauth_server() -> AsyncIterator[str]:
    """A local MCP server behind OAuth; its authorization endpoint approves at once, as a signed-in user would."""
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    server = FastMCP(
        'signed',
        auth=InMemoryOAuthProvider(base_url=base, client_registration_options=ClientRegistrationOptions(enabled=True)),
    )

    @server.tool
    def ping() -> str:
        return 'pong'  # pragma: no cover -- listed, never called

    running = uvicorn.Server(uvicorn.Config(server.http_app(), host='127.0.0.1', port=port, log_level='error'))
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(running.serve)
        while not running.started:
            await anyio.sleep(0.01)
        yield f'{base}/mcp'
        running.should_exit = True
    # Collect the stream the server leaves now, under this test's warning filter, not during a later test.
    gc.collect()


@pytest.fixture
def browser(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Approve in a 'browser' by following the authorization redirects to CLAI's callback server."""
    opened: list[str] = []

    def open_browser(url: str, *args: object, **kwargs: object) -> bool:
        opened.append(url)
        threading.Thread(target=lambda: httpx.get(url, follow_redirects=True, timeout=10), daemon=True).start()
        return True

    monkeypatch.setattr(webbrowser, 'open', open_browser)
    return opened


# The in-process MCP server, not CLAI's client, leaves a session stream for the collector (`mcp.server.streamable_http`).
@pytest.mark.filterwarnings(
    # Python 3.14 says "while calling deallocator"; earlier versions say "in:".
    r'ignore:Exception ignored .*MemoryObjectReceiveStream\.__del__:pytest.PytestUnraisableExceptionWarning'
)
async def test_oauth_runs_never_open_a_browser_and_sign_in_is_only_on_request(
    oauth_server: str, browser: list[str]
) -> None:
    method = OAuthSignIn(name='plugin_example', service='Example', setup='/example login', url=oauth_server)
    assert not await anyio.to_thread.run_sync(method.signed_in)
    with anyio.fail_after(10), pytest.raises(RuntimeError) as raised:
        async with method.client():
            pass  # pragma: no cover -- never connects
    assert 'Not signed in to Example. Run /example login to sign in.' in str(raised.value)
    assert isinstance(raised.value.__cause__, SignInRequired)
    assert browser == []

    shown: list[str] = []
    with anyio.fail_after(30):
        await method.sign_in(show=shown.append)
    assert len(browser) == 1
    assert shown[-1] == f'If no browser opened, open this link:\n{browser[0]}'
    assert await anyio.to_thread.run_sync(method.signed_in)

    async with method.client() as client:
        assert [tool.name for tool in await client.list_tools()] == ['ping']
    assert len(browser) == 1, 'a run with stored tokens opens nothing'

    await anyio.to_thread.run_sync(method.sign_out)
    assert not await anyio.to_thread.run_sync(method.signed_in)


async def test_a_sign_in_belongs_to_the_url_it_was_made_for() -> None:
    """A plugin whose URL follows its settings is signed out after the URL changes, so its menu offers to sign in."""
    read_only = 'https://mcp.example/mcp?read_only=true'
    await TokenStorageAdapter(TokenStore('plugin_example'), server_url=read_only).set_tokens(
        OAuthToken(access_token='a', token_type='Bearer', expires_in=3600)
    )

    def method(url: str) -> OAuthSignIn:
        return OAuthSignIn(name='plugin_example', service='Example', setup='/x', url=url)

    assert method(read_only).signed_in()
    assert method(f'{read_only}/').signed_in(), 'FastMCP keys tokens without a trailing slash'
    assert not method('https://mcp.example/mcp').signed_in()
    assert TokenStore('plugin_example').signed_in(), 'without a URL, any stored sign-in counts'


async def test_oauth_over_sse_and_with_headers() -> None:
    method = OAuthSignIn(
        name='plugin_example',
        service='Example',
        setup='/x',
        url='https://mcp.example/sse',
        headers={'X-A': 'b'},
        sse=True,
    )
    transport = method.transport()
    assert isinstance(transport, SSETransport)
    assert transport.headers == {'X-A': 'b'}
    assert transport.url == 'https://mcp.example/sse'
