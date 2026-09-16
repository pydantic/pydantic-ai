"""OAuth orchestration with fake browser, exchange, and credential storage."""

import io

import keyring
import pytest
from pydantic_ai.exceptions import UserError
from pydantic_ai.providers.openai_codex import OpenAICodexCredentials, OpenAICodexOAuthFlow
from rich.console import Console

from pydantic_clai2.auth import CodexAuth, CodexCredentials
from pydantic_clai2.commands import Command, Commands
from pydantic_clai2.config import Settings


def fake_browser(url: str) -> bool:
    return True


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
    credentials = OpenAICodexCredentials(
        access_token='fake-access', refresh_token='fake-refresh', account_id='fake-account'
    )

    async def exchange(self: OpenAICodexOAuthFlow) -> OpenAICodexCredentials:
        assert self.redirect_uri == 'http://localhost:1455/auth/callback'
        return credentials

    monkeypatch.setattr(OpenAICodexOAuthFlow, 'exchange_code_from_callback', exchange)
    monkeypatch.setattr('webbrowser.open', fake_browser)
    output = io.StringIO()
    auth = CodexAuth(Console(file=output))
    commands = Commands()
    commands.register(Command(name='login', description='Login', handler=auth.login))
    assert 'connected' in await commands.execute_async('/login openai-codex')
    assert await auth.source.load() == credentials
    assert 'fake-access' not in output.getvalue()
    assert 'fake-refresh' not in output.getvalue()
    assert 'code_challenge=' in output.getvalue().replace('\n', '')


async def test_failed_login_does_not_save(monkeypatch: pytest.MonkeyPatch) -> None:
    async def exchange(self: OpenAICodexOAuthFlow) -> OpenAICodexCredentials:
        raise UserError('Authorization denied')

    monkeypatch.setattr(OpenAICodexOAuthFlow, 'exchange_code_from_callback', exchange)
    monkeypatch.setattr('webbrowser.open', fake_browser)
    auth = CodexAuth(Console(file=io.StringIO()))
    with pytest.raises(UserError, match='denied'):
        await auth.login([])
    assert keyring.get_password('pydantic-clai2', 'openai-codex') is None


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
    auth = CodexAuth(Console(file=io.StringIO()))
    with pytest.raises(ValueError, match='Usage'):
        await auth.login(['invalid'])

    async def timeout(self: OpenAICodexOAuthFlow) -> OpenAICodexCredentials:
        raise TimeoutError

    monkeypatch.setattr(OpenAICodexOAuthFlow, 'exchange_code_from_callback', timeout)
    monkeypatch.setattr('webbrowser.open', fake_browser)
    with pytest.raises(UserError, match='timed out'):
        await auth.login([])
    assert auth.model('openai-codex:test').model_name == 'test'
    provider = auth.provider
    auth.model('openai-codex:test')
    assert auth.provider is provider


def test_default_model() -> None:
    assert Settings().model == 'openai-codex:gpt-6-astra'
