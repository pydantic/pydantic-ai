"""OAuth orchestration with fake browser, exchange, and credential storage."""

import asyncio
import io

import keyring
import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic_ai.exceptions import UserError
from pydantic_ai.providers.openai_codex import OpenAICodexCredentials, OpenAICodexOAuthFlow
from rich.console import Console

from pydantic_clai2.auth import CodexAuth, CodexCredentials, code_from_paste, read_line
from pydantic_clai2.commands import Command, Commands
from pydantic_clai2.config import Settings

CREDENTIALS = OpenAICodexCredentials(
    access_token='fake-access', refresh_token='fake-refresh', account_id='fake-account'
)


def fake_browser(url: str) -> bool:
    return True


def fixed_state(nbytes: int) -> str:
    return 'fixed-state'


async def never_pasted(message: str) -> str:
    await asyncio.Event().wait()
    raise AssertionError('unreachable')


async def never_called_back(self: OpenAICodexOAuthFlow) -> OpenAICodexCredentials:
    await asyncio.Event().wait()
    raise AssertionError('unreachable')


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


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


async def test_credentials_round_trip() -> None:
    source = CodexCredentials()
    with pytest.raises(UserError, match='/login'):
        await source.load()
    credentials = OpenAICodexCredentials(
        access_token='fake-access', refresh_token='fake-refresh', account_id='fake-account'
    )
    await source.save(credentials)
    assert await source.load() == credentials


async def test_login_uses_core_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    async def exchange(self: OpenAICodexOAuthFlow) -> OpenAICodexCredentials:
        assert self.redirect_uri == 'http://localhost:1455/auth/callback'
        return CREDENTIALS

    monkeypatch.setattr(OpenAICodexOAuthFlow, 'exchange_code_from_callback', exchange)
    monkeypatch.setattr('webbrowser.open', fake_browser)
    output = io.StringIO()
    auth = CodexAuth(Console(file=output), read_line=never_pasted)
    commands = Commands()
    commands.register(Command(name='login', description='Login', handler=auth.login))
    assert 'connected' in await commands.execute_async('/login openai-codex')
    assert await auth.source.load() == CREDENTIALS
    assert 'fake-access' not in output.getvalue()
    assert 'fake-refresh' not in output.getvalue()
    assert 'code_challenge=' in output.getvalue().replace('\n', '')
    assert 'over SSH' in output.getvalue()


async def test_failed_login_does_not_save(monkeypatch: pytest.MonkeyPatch) -> None:
    async def exchange(self: OpenAICodexOAuthFlow) -> OpenAICodexCredentials:
        raise UserError('Authorization denied')

    monkeypatch.setattr(OpenAICodexOAuthFlow, 'exchange_code_from_callback', exchange)
    monkeypatch.setattr('webbrowser.open', fake_browser)
    auth = CodexAuth(Console(file=io.StringIO()), read_line=never_pasted)
    with pytest.raises(UserError, match='denied'):
        await auth.login([])
    assert keyring.get_password('pydantic-clai2', 'openai-codex') is None


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
    assert keyring.get_password('pydantic-clai2', 'openai-codex') is None


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
    assert keyring.get_password('pydantic-clai2', 'openai-codex') is None


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
    keyring.set_password('pydantic-clai2', 'openai-codex', 'not json')
    source = CodexCredentials()
    with pytest.raises(UserError, match='invalid'):
        await source.load()

    def discard(service: str, account: str, value: str) -> None:
        pass

    monkeypatch.setattr(keyring, 'set_password', discard)
    with pytest.raises(UserError, match='did not retain'):
        await source.save(OpenAICodexCredentials(access_token='test', refresh_token='test', account_id='test'))
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
