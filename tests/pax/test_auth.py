"""OAuth orchestration with fake browser, exchange, and credential storage."""

import asyncio
import io
import re
import threading
import webbrowser
from collections.abc import Callable
from pathlib import Path

import anyio
import httpx2
import keyring
import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from termflow.tui import MenuItem
from termflow.tui.menu import Menu, MenuResult

from pydantic_ai.exceptions import UserError
from pydantic_ai.providers.openai_codex import (
    CredentialsPersistenceError,
    OpenAICodexCredentials,
    OpenAICodexOAuthFlow,
    OpenAICodexProvider,
)
from pydantic_ai_pax.auth import CodexAuth, CodexCredentials, code_from_paste, login_command
from pydantic_ai_pax.commands import Command, Commands
from pydantic_ai_pax.config import Settings
from pydantic_ai_pax.config.api_keys import forget_connection
from pydantic_ai_pax.config.credential_store import has_credentials, load_codex_credentials
from pydantic_ai_pax.config.settings_store import SettingsStore
from pydantic_ai_pax.models import codex as codex_models
from pydantic_ai_pax.models.model_catalog import CODEX_MODELS
from pydantic_ai_pax.plugins import PluginLogin
from pydantic_ai_pax.plugins.menus import Dialogs
from pydantic_ai_pax.ui.menus.dialogs import TerminalDialogs
from pydantic_ai_pax.ui.menus.field_menu import Runners
from pydantic_ai_pax.ui.menus.menu_worker import holding_output
from pydantic_ai_pax.ui.prompt.prompt_surface import PromptSurface
from tests._timeouts import READINESS_WAIT_TIMEOUT
from tests.pax.dialog_script import Back, ScriptedDialogs
from tests.pax.surface_terminal import SurfaceTerminal

CREDENTIALS = OpenAICodexCredentials(
    access_token='fake-access', refresh_token='fake-refresh', account_id='fake-account'
)


def fake_browser(url: str) -> bool:
    return True


def fixed_state(nbytes: int) -> str:
    return 'fixed-state'


async def never_pasted(message: str) -> str:
    await asyncio.Event().wait()
    raise AssertionError('unreachable')  # pragma: no cover


async def never_called_back(self: OpenAICodexOAuthFlow) -> OpenAICodexCredentials:
    await asyncio.Event().wait()
    raise AssertionError('unreachable')  # pragma: no cover


def scripted(values: list[str | None]) -> tuple[list[str], CodexAuth]:
    """An auth whose paste prompt pops scripted answers, `None` for a cancelled prompt; the callback never fires."""
    prompts: list[str] = []

    async def paste(message: str) -> str | None:
        prompts.append(message)
        return values.pop(0)

    return prompts, CodexAuth(Console(file=io.StringIO()), read_line=paste)


async def test_credentials_round_trip() -> None:
    source = CodexCredentials()
    with pytest.raises(UserError, match='/login'):
        await source.load()
    credentials = OpenAICodexCredentials(
        access_token='fake-access', refresh_token='fake-refresh', account_id='fake-account'
    )
    with pytest.raises(UserError, match='openai-codex was signed out'):
        await source.save(credentials)  # a refresh never creates a login
    await source.save_login(credentials)
    assert await source.load() == credentials
    refreshed = OpenAICodexCredentials(access_token='new', refresh_token='new-refresh', account_id='fake-account')
    await source.save(refreshed)
    assert await source.load() == refreshed
    forget_connection(account='openai-codex')
    with pytest.raises(UserError, match=r'signed out\. Run /login openai-codex to use it again\.'):
        await source.save(credentials)
    assert not has_credentials(account='openai-codex')


async def test_a_refresh_finishing_after_sign_out_does_not_sign_the_account_back_in() -> None:
    account = 'openai-codex@work'
    source = CodexCredentials(account=account)
    await source.save_login(OpenAICodexCredentials(access_token='old', refresh_token='old-refresh', account_id='acct'))
    sent: list[str] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        sent.append(request.url.host)
        if request.url.host == 'auth.openai.com':
            forget_connection(account=account)  # signed out while the token exchange is in flight
            return httpx2.Response(
                200, json={'access_token': 'new', 'refresh_token': 'new-refresh', 'account_id': 'acct'}
            )
        return httpx2.Response(401, json={'error': {'message': 'expired'}})

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
    provider = OpenAICodexProvider(credential_source=source, http_client=client)
    with pytest.raises(CredentialsPersistenceError):
        await provider.client.with_options(max_retries=0).get(
            'https://chatgpt.com/backend-api/wham/usage', cast_to=object
        )
    assert sent == ['chatgpt.com', 'auth.openai.com']
    assert not has_credentials(account=account), 'the refreshed tokens were not saved back'


@pytest.mark.parametrize('command', ['/login codex', '/login openai-codex'])
async def test_login_uses_core_flow(monkeypatch: pytest.MonkeyPatch, command: str) -> None:
    async def exchange(self: OpenAICodexOAuthFlow) -> OpenAICodexCredentials:
        assert self.redirect_uri == 'http://localhost:1455/auth/callback'
        return CREDENTIALS

    monkeypatch.setattr(OpenAICodexOAuthFlow, 'exchange_code_from_callback', exchange)
    monkeypatch.setattr('pydantic_ai_pax.auth.open_browser', fake_browser)
    output = io.StringIO()
    auth = CodexAuth(Console(file=output), read_line=never_pasted)
    commands = Commands()
    commands.register(Command(name='login', description='Login', handler=lambda args: login_command(args, codex=auth)))
    assert 'connected' in await commands.execute_async(command)
    assert await auth.source.load() == CREDENTIALS
    assert 'fake-access' not in output.getvalue()
    assert 'fake-refresh' not in output.getvalue()
    assert 'code_challenge=' in output.getvalue().replace('\n', '')
    assert 'over SSH' in output.getvalue()


async def test_login_dispatches_copilot_by_short_and_provider_name(monkeypatch: pytest.MonkeyPatch) -> None:
    async def copilot(*, console: Console, account: str, dialogs: Dialogs | None) -> str:
        assert dialogs is None
        return f'Copilot connected as {account}.'

    monkeypatch.setattr('pydantic_ai_pax.models.github_copilot.login', copilot)
    auth = CodexAuth(Console(file=io.StringIO()), read_line=never_pasted)
    assert await login_command(['copilot'], codex=auth) == 'Copilot connected as github-copilot.'
    assert await login_command(['github-copilot'], codex=auth) == 'Copilot connected as github-copilot.'
    assert await login_command(['copilot@work'], codex=auth) == 'Copilot connected as github-copilot@work.'


async def test_login_runs_a_plugin_sign_in_and_lists_it_in_usage() -> None:
    async def claude() -> str:
        return 'Signed in to Claude Code.'

    auth = CodexAuth(Console(file=io.StringIO()), read_line=never_pasted)
    plugins = {'claude': PluginLogin(name='claude', handler=claude)}
    assert await login_command(['claude'], codex=auth, plugins=plugins) == 'Signed in to Claude Code.'
    with pytest.raises(ValueError, match=r'^Usage: /login \[openai-codex\|github-copilot\|claude\]\[@PROFILE\]$'):
        await login_command(['grok'], codex=auth, plugins=plugins)
    with pytest.raises(ValueError, match=r'^Usage: /login \[openai-codex\|github-copilot\]\[@PROFILE\]$'):
        await login_command(['codex', 'extra'], codex=auth)


async def test_bare_login_asks_which_sign_in() -> None:
    async def claude() -> str:
        return 'Signed in to Claude Code.'

    offered: list[object] = []

    def pick(value: object) -> Runners:
        def run_list(menu: Menu) -> MenuResult:
            assert menu.highlighted is not None
            offered.append(menu.highlighted.value)
            return MenuResult(cancelled=True) if value is None else MenuResult(item=MenuItem('picked', value=value))

        return Runners(run_list=run_list)

    auth = CodexAuth(Console(file=io.StringIO()), read_line=never_pasted)
    plugins = {'claude-code': PluginLogin(name='claude-code', handler=claude)}
    picked = await login_command([], codex=auth, plugins=plugins, runners=pick('claude-code'))
    assert picked == 'Signed in to Claude Code.'
    assert offered == ['openai-codex']
    assert await login_command([], codex=auth, plugins=plugins, runners=pick(None)) == ''
    assert await login_command([], codex=auth, plugins=plugins, runners=pick(42)) == ''


async def test_failed_login_does_not_save(monkeypatch: pytest.MonkeyPatch) -> None:
    async def exchange(self: OpenAICodexOAuthFlow) -> OpenAICodexCredentials:
        raise UserError('Authorization denied')

    monkeypatch.setattr(OpenAICodexOAuthFlow, 'exchange_code_from_callback', exchange)
    monkeypatch.setattr('pydantic_ai_pax.auth.open_browser', fake_browser)
    auth = CodexAuth(Console(file=io.StringIO()), read_line=never_pasted)
    with pytest.raises(UserError, match='denied'):
        await auth.login([])
    assert load_codex_credentials() is None


@pytest.mark.parametrize('bare', [False, True])
async def test_pasted_redirect_wins_over_callback(monkeypatch: pytest.MonkeyPatch, *, bare: bool) -> None:
    exchanged: list[str] = []
    flows: list[OpenAICodexOAuthFlow] = []

    async def exchange_code(self: OpenAICodexOAuthFlow, code: str) -> OpenAICodexCredentials:
        flows.append(self)
        exchanged.append(code)
        return CREDENTIALS

    monkeypatch.setattr(OpenAICodexOAuthFlow, 'exchange_code_from_callback', never_called_back)
    monkeypatch.setattr(OpenAICodexOAuthFlow, 'exchange_code', exchange_code)
    monkeypatch.setattr('pydantic_ai_pax.auth.open_browser', fake_browser)
    monkeypatch.setattr('secrets.token_urlsafe', fixed_state)
    pasted = 'the-code' if bare else '  http://localhost:1455/auth/callback?code=the-code&state=fixed-state \n'
    prompts, auth = scripted(['', '   ', pasted])
    assert 'connected' in await auth.login(['openai-codex'])
    assert exchanged == ['the-code']
    assert flows[0].state == 'fixed-state'
    assert len(prompts) == 3
    assert 'Paste the URL' in prompts[0]
    assert await auth.source.load() == CREDENTIALS


async def test_paste_prompt_shows_under_the_sign_in_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """The prompt drew wherever the live panel left the cursor, over the URL it asks about."""
    screens: list[list[str]] = []
    terminal = SurfaceTerminal(width=100, height=30)

    async def exchange_code(self: OpenAICodexOAuthFlow, code: str) -> OpenAICodexCredentials:
        screens.append([line for line in terminal.lines() if line])
        return CREDENTIALS

    monkeypatch.setattr(OpenAICodexOAuthFlow, 'exchange_code_from_callback', never_called_back)
    monkeypatch.setattr(OpenAICodexOAuthFlow, 'exchange_code', exchange_code)
    monkeypatch.setattr('pydantic_ai_pax.auth.open_browser', fake_browser)
    surface = PromptSurface(output=terminal, size=lambda: (100, 30))
    surface.paint(('> /login openai-codex',))
    surface.release()
    console = Console(file=surface, width=100, height=30, color_system=None)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        monkeypatch.setattr('pydantic_ai_pax.ui.prompt.question_input.create_input', lambda: pipe)
        pipe.send_text('the-code\r')
        with holding_output(surface.held):
            assert 'connected' in await CodexAuth(console).login([])
    (screen,) = screens
    assert screen[-3:] == [
        'If the browser cannot reach this machine (for example over SSH), paste the URL it ends up on.',
        'Paste the URL the browser lands on (or finish there):',
        '[8 characters]',
    ]
    assert any(line.startswith('https://auth.openai.com/oauth/authorize?') for line in screen)
    assert 'the-code' not in terminal.getvalue(), 'the pasted authorization code is not shown'


async def port_in_use(self: OpenAICodexOAuthFlow) -> OpenAICodexCredentials:
    raise OSError(48, 'Address already in use')


@pytest.mark.parametrize('same_tick', [False, True])
async def test_paste_survives_a_failed_callback(monkeypatch: pytest.MonkeyPatch, *, same_tick: bool) -> None:
    """A busy port loses the race; a callback that fails in the same tick as a good paste loses too."""

    async def exchange_code(self: OpenAICodexOAuthFlow, code: str) -> OpenAICodexCredentials:
        return CREDENTIALS

    async def exchanged_elsewhere(self: OpenAICodexOAuthFlow) -> OpenAICodexCredentials:
        raise UserError('invalid_grant')

    output = io.StringIO()

    async def paste(message: str) -> str:
        while not same_tick and 'Address already in use' not in output.getvalue():
            await asyncio.sleep(0)  # the listener fails, and is reported, before anything is pasted
        return 'the-code'

    monkeypatch.setattr(
        OpenAICodexOAuthFlow, 'exchange_code_from_callback', exchanged_elsewhere if same_tick else port_in_use
    )
    monkeypatch.setattr(OpenAICodexOAuthFlow, 'exchange_code', exchange_code)
    monkeypatch.setattr('pydantic_ai_pax.auth.open_browser', fake_browser)
    auth = CodexAuth(Console(file=output), read_line=paste)
    assert 'connected' in await auth.login([])
    assert await auth.source.load() == CREDENTIALS
    assert ('Address already in use' in output.getvalue()) is not same_tick


async def test_lost_race_then_rejected_paste(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(OpenAICodexOAuthFlow, 'exchange_code_from_callback', port_in_use)
    monkeypatch.setattr('pydantic_ai_pax.auth.open_browser', fake_browser)
    _, auth = scripted(['', None])
    with pytest.raises(UserError, match='cancelled'):
        await auth.login([])
    assert load_codex_credentials() is None


@pytest.mark.parametrize(
    ('pasted', 'message'),
    [
        ('http://localhost:1455/auth/callback?code=x&state=other', 'different login'),
        ('http://localhost:1455/auth/callback?error=access_denied&state=fixed-state', 'access_denied'),
        (None, 'cancelled'),
    ],
)
async def test_rejected_paste(monkeypatch: pytest.MonkeyPatch, pasted: str | None, message: str) -> None:
    monkeypatch.setattr(OpenAICodexOAuthFlow, 'exchange_code_from_callback', never_called_back)
    monkeypatch.setattr('pydantic_ai_pax.auth.open_browser', fake_browser)
    monkeypatch.setattr('secrets.token_urlsafe', fixed_state)
    _, auth = scripted([pasted])
    with pytest.raises(UserError, match=message):
        await auth.login([])
    assert load_codex_credentials() is None


def test_code_from_paste_state_binding() -> None:
    assert code_from_paste(text='bare', state='s') == 'bare'
    assert code_from_paste(text='https://x/cb?state=s&code=c', state='s') == 'c'
    with pytest.raises(UserError, match='different login'):
        code_from_paste(text='https://x/cb?code=c', state='s')


async def test_auth_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    source = CodexCredentials()
    original_set = keyring.set_password

    def discard(service: str, account: str, value: str) -> None:
        pass

    monkeypatch.setattr(keyring, 'set_password', discard)
    await source.save_login(OpenAICodexCredentials(access_token='test', refresh_token='test', account_id='test'))
    assert await source.load() == OpenAICodexCredentials(access_token='test', refresh_token='test', account_id='test')
    monkeypatch.setattr(keyring, 'set_password', original_set)
    keyring.set_password('pydantic-clai2', 'openai-codex', 'not json')
    with pytest.raises(UserError, match='invalid'):
        await source.load()
    auth = CodexAuth(Console(file=io.StringIO()), read_line=never_pasted, login_timeout=0)
    with pytest.raises(ValueError, match='Usage'):
        await auth.login(['invalid'])

    monkeypatch.setattr(OpenAICodexOAuthFlow, 'exchange_code_from_callback', never_called_back)
    monkeypatch.setattr('pydantic_ai_pax.auth.open_browser', fake_browser)
    with pytest.raises(UserError, match='timed out'):
        await auth.login([])
    assert auth.model('openai-codex:test').model_name == 'test'
    provider = auth.provider
    auth.model('openai-codex:test')
    assert auth.provider is provider


def test_default_model() -> None:
    assert Settings().model == 'openai-codex:gpt-6-astra'


def mock_provider(account: str, respond: Callable[[httpx2.Request], httpx2.Response]) -> OpenAICodexProvider:
    """The account's core provider, answered by `respond` instead of the Codex backend."""
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
    return OpenAICodexProvider(credential_source=CodexCredentials(account=account), http_client=client)


def serve_codex_models(monkeypatch: pytest.MonkeyPatch, status: int = 200) -> list[httpx2.Request]:
    """Answer the Codex catalog `GET {base_url}/models` through the real discovery, recording each request."""
    requests: list[httpx2.Request] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        served = [
            {'slug': 'gpt-6.2-sol', 'display_name': 'GPT-6.2-Sol', 'visibility': 'list'},
            {'slug': 'codex-auto-review', 'display_name': 'Codex Auto Review', 'visibility': 'hide'},
            {'slug': 'gpt-6.1-terra', 'display_name': 'GPT-6.1 Terra', 'visibility': 'list'},
        ]
        return httpx2.Response(status, json={'models': served})

    def account_provider(self: CodexAuth, account: str, base_url: str | None = None) -> OpenAICodexProvider:
        return mock_provider(account, respond)

    monkeypatch.setattr(CodexAuth, 'account_provider', account_provider)
    return requests


def codex_signs_in(monkeypatch: pytest.MonkeyPatch) -> CodexAuth:
    async def exchange(self: OpenAICodexOAuthFlow) -> OpenAICodexCredentials:
        return CREDENTIALS

    monkeypatch.setattr(OpenAICodexOAuthFlow, 'exchange_code_from_callback', exchange)
    monkeypatch.setattr('pydantic_ai_pax.auth.open_browser', fake_browser)
    return CodexAuth(Console(file=io.StringIO()), read_line=never_pasted)


@pytest.mark.parametrize(('target', 'prefix'), [('codex', 'openai-codex'), ('openai-codex@work', 'openai-codex@work')])
async def test_codex_login_adds_the_account_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, target: str, prefix: str
) -> None:
    requests = serve_codex_models(monkeypatch)
    store = SettingsStore(tmp_path / 'config.db')
    message = await login_command([target], codex=codex_signs_in(monkeypatch), store=store)
    assert message.endswith(' Added 2 Codex models.')
    assert store.models() == [f'{prefix}:gpt-6.1-terra', f'{prefix}:gpt-6.2-sol']
    assert [
        (
            request.url.host,
            request.url.path,
            request.url.params['client_version'],
            request.headers['authorization'],
            request.headers['chatgpt-account-id'],
        )
        for request in requests
    ] == [('chatgpt.com', '/backend-api/codex/models', '0.170.0', 'Bearer fake-access', 'fake-account')]


async def test_codex_login_falls_back_to_the_static_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    serve_codex_models(monkeypatch, status=500)
    store = SettingsStore(tmp_path / 'config.db')
    message = await login_command(['codex@work'], codex=codex_signs_in(monkeypatch), store=store)
    assert 'Codex model discovery failed' in message
    assert f'Added the {len(CODEX_MODELS)} built-in Codex models.' in message
    assert store.models() == sorted(name.replace('openai-codex:', 'openai-codex@work:') for name in CODEX_MODELS)


async def test_codex_discovery_rejects_an_unexpected_catalog() -> None:
    def respond(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={'data': []})

    await CodexCredentials().save_login(CREDENTIALS)
    with pytest.raises(UserError, match='Codex returned an invalid model list'):
        await codex_models.discover(mock_provider('openai-codex', respond))

    def hidden(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={'models': [{'slug': 'codex-auto-review', 'visibility': 'hide'}]})

    with pytest.raises(UserError, match='Codex returned no models for this account'):
        await codex_models.discover(mock_provider('openai-codex', hidden))


async def test_cancelled_codex_login_adds_no_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    requests = serve_codex_models(monkeypatch)
    await CodexCredentials().save_login(CREDENTIALS)
    store = SettingsStore(tmp_path / 'config.db')
    message = await login_command(
        ['codex'], codex=CodexAuth(Console(file=io.StringIO())), store=store, dialogs=ScriptedDialogs([Back()])
    )
    assert message == 'ChatGPT/Codex sign-in cancelled.'
    assert requests == []
    assert store.models() == []


def hyperlinks(text: str) -> list[str]:
    """The targets of the OSC 8 hyperlinks in `text`, in order."""
    return [target for target in re.findall(r'\x1b\]8;[^;\x1b]*;([^\x1b]*)\x1b\\', text) if target]


async def test_terminal_sign_in_links_the_url_and_opens_it_with_webbrowser(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sign-in's own opener is mocked at `webbrowser`, so this pins the whole path from the login to it."""
    opened: list[str] = []
    urls: list[str] = []
    launched = asyncio.Event()
    loop = asyncio.get_running_loop()

    def open_url(url: str, *args: object, **kwargs: object) -> bool:
        opened.append(url)
        loop.call_soon_threadsafe(launched.set)  # The browser opens in a thread.
        return True

    def default_browser(using: str | None = None) -> webbrowser.BaseBrowser:
        return webbrowser.BaseBrowser('fake')  # Not a command-line opener, so `webbrowser.open` opens it.

    async def exchange(self: OpenAICodexOAuthFlow) -> OpenAICodexCredentials:
        urls.append(self.authorization_url())
        await launched.wait()
        return CREDENTIALS

    monkeypatch.setattr(OpenAICodexOAuthFlow, 'exchange_code_from_callback', exchange)
    monkeypatch.setattr(webbrowser, 'get', default_browser)
    monkeypatch.setattr(webbrowser, 'open', open_url)
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, color_system='truecolor', width=80)
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        assert 'connected' in await CodexAuth(console, read_line=never_pasted).login([])
    [url] = urls
    assert opened == [url]
    rows = [row for row in output.getvalue().splitlines() if hyperlinks(row)]
    assert len(rows) > 1, 'Rich wraps the URL'
    assert all(hyperlinks(row) == [url] for row in rows), 'every wrapped row links to the whole URL'
    assert ''.join(re.sub(r'\x1b\]8;[^\x1b]*\x1b\\', '', row) for row in rows) == url


async def test_terminal_waiting_screen_links_the_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """The URL is folded over the preview pane's lines, each a link to all of it."""
    previews: list[str] = []
    urls: list[str] = []
    shown = threading.Event()

    def run_choice(menu: Menu) -> MenuResult:
        render = menu._preview  # pyright: ignore[reportPrivateUsage]
        assert render is not None and shown.wait(timeout=READINESS_WAIT_TIMEOUT)
        previews.append(render(menu._items[0]))  # pyright: ignore[reportPrivateUsage]
        return MenuResult(cancelled=True)

    async def exchange(self: OpenAICodexOAuthFlow) -> OpenAICodexCredentials:
        urls.append(self.authorization_url())  # The screen shows the link before the callback is awaited.
        shown.set()
        return await never_called_back(self)

    monkeypatch.setattr(OpenAICodexOAuthFlow, 'exchange_code_from_callback', exchange)
    dialogs = TerminalDialogs(Runners(run_choice=run_choice))
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        assert await CodexAuth(Console(file=io.StringIO())).login([], dialogs=dialogs) == (
            'ChatGPT/Codex sign-in cancelled.'
        )
    [preview], [url] = previews, urls
    assert set(hyperlinks(preview)) == {url}
    assert url in re.sub(r'\x1b\]8;[^\x1b]*\x1b\\|\n', '', preview), 'the whole URL shows, over several lines'
