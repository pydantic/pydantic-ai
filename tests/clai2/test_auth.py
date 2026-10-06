"""OAuth orchestration with fake browser, exchange, and credential storage."""

import asyncio
import io

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
from pydantic_clai2.auth import CodexAuth, CodexCredentials, code_from_paste, login_command, read_line
from pydantic_clai2.commands import Command, Commands
from pydantic_clai2.config import Settings
from pydantic_clai2.config.api_keys import forget_connection
from pydantic_clai2.config.credential_store import has_credentials, load_codex_credentials
from pydantic_clai2.plugins import PluginLogin
from pydantic_clai2.ui.menus.field_menu import Runners

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


def scripted(values: list[str | BaseException]) -> tuple[list[str], CodexAuth]:
    """An auth whose paste prompt pops scripted answers; the callback never fires."""
    prompts: list[str] = []

    async def paste(message: str) -> str:
        prompts.append(message)
        value = values.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

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
    monkeypatch.setattr('webbrowser.open', fake_browser)
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
    async def copilot(*, console: Console, account: str) -> str:
        return f'Copilot connected as {account}.'

    monkeypatch.setattr('pydantic_clai2.models.github_copilot.login', copilot)
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
    monkeypatch.setattr('webbrowser.open', fake_browser)
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
    monkeypatch.setattr('webbrowser.open', fake_browser)
    monkeypatch.setattr('secrets.token_urlsafe', fixed_state)
    pasted = 'the-code' if bare else '  http://localhost:1455/auth/callback?code=the-code&state=fixed-state \n'
    prompts, auth = scripted(['', '   ', pasted])
    assert 'connected' in await auth.login(['openai-codex'])
    assert exchanged == ['the-code']
    assert flows[0].state == 'fixed-state'
    assert len(prompts) == 3
    assert 'Paste the URL' in prompts[0]
    assert await auth.source.load() == CREDENTIALS


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
    monkeypatch.setattr('webbrowser.open', fake_browser)
    auth = CodexAuth(Console(file=output), read_line=paste)
    assert 'connected' in await auth.login([])
    assert await auth.source.load() == CREDENTIALS
    assert ('Address already in use' in output.getvalue()) is not same_tick


async def test_lost_race_then_rejected_paste(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(OpenAICodexOAuthFlow, 'exchange_code_from_callback', port_in_use)
    monkeypatch.setattr('webbrowser.open', fake_browser)
    _, auth = scripted(['', EOFError()])
    with pytest.raises(UserError, match='cancelled'):
        await auth.login([])
    assert load_codex_credentials() is None


@pytest.mark.parametrize(
    ('pasted', 'message'),
    [
        ('http://localhost:1455/auth/callback?code=x&state=other', 'different login'),
        ('http://localhost:1455/auth/callback?error=access_denied&state=fixed-state', 'access_denied'),
        (KeyboardInterrupt(), 'cancelled'),
        (EOFError(), 'cancelled'),
    ],
)
async def test_rejected_paste(monkeypatch: pytest.MonkeyPatch, pasted: str | BaseException, message: str) -> None:
    monkeypatch.setattr(OpenAICodexOAuthFlow, 'exchange_code_from_callback', never_called_back)
    monkeypatch.setattr('webbrowser.open', fake_browser)
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


async def test_default_read_line_uses_prompt_toolkit() -> None:
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text('pasted value\n')
        assert await read_line('> ') == 'pasted value'


async def test_auth_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    source = CodexCredentials()
    original_set = keyring.set_password

    def discard(service: str, account: str, value: str) -> None:
        pass

    monkeypatch.setattr(keyring, 'set_password', discard)
    with pytest.raises(UserError, match='did not retain'):
        await source.save_login(OpenAICodexCredentials(access_token='test', refresh_token='test', account_id='test'))
    monkeypatch.setattr(keyring, 'set_password', original_set)
    keyring.set_password('pydantic-clai2', 'openai-codex', 'not json')
    with pytest.raises(UserError, match='invalid'):
        await source.load()
    auth = CodexAuth(Console(file=io.StringIO()), read_line=never_pasted, login_timeout=0)
    with pytest.raises(ValueError, match='Usage'):
        await auth.login(['invalid'])

    monkeypatch.setattr(OpenAICodexOAuthFlow, 'exchange_code_from_callback', never_called_back)
    monkeypatch.setattr('webbrowser.open', fake_browser)
    with pytest.raises(UserError, match='timed out'):
        await auth.login([])
    assert auth.model('openai-codex:test').model_name == 'test'
    provider = auth.provider
    auth.model('openai-codex:test')
    assert auth.provider is provider


def test_default_model() -> None:
    assert Settings().model == 'openai-codex:gpt-6-astra'
