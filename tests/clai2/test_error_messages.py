"""Provider failures remain recoverable without exposing nested authentication payloads."""

import io
from pathlib import Path

import pytest
from httpx2 import Request
from openai import APIConnectionError
from rich.console import Console

from pydantic_ai import Agent, ModelRequestContext, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.openai_codex import CredentialsPersistenceError, CredentialsRefreshError
from pydantic_ai_harness.step_persistence.conversations import SqliteConversationStore
from pydantic_clai2 import chat
from pydantic_clai2.cli import headless
from pydantic_clai2.config import Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.errors import error_message
from tests.clai2.test_app_edges import inputs


@pytest.mark.parametrize('mode', ['interactive', 'headless'])
@pytest.mark.parametrize('chain', ['direct', 'cause', 'context', 'suppressed', 'cycle', 'network', 'persistence'])
async def test_provider_error_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], mode: str, chain: str
) -> None:
    refresh = CredentialsRefreshError('refresh_token_expired: private-token-response')
    error: Exception = ModelAPIError(model_name='test', message='Connection error.')
    if chain == 'direct':
        error = refresh
    elif chain == 'persistence':
        error = CredentialsPersistenceError('Credential storage is locked')
    elif chain == 'cycle':
        error.__cause__ = error
    else:
        connection = APIConnectionError(request=Request('POST', 'https://chatgpt.com/backend-api/codex/responses'))
        error.__cause__ = connection
        if chain == 'cause':
            connection.__cause__ = refresh
        elif chain in ('context', 'suppressed'):
            connection.__context__ = refresh
            connection.__suppress_context__ = chain == 'suppressed'
        else:
            connection.__cause__ = OSError('Network unreachable')

    class Failure(AbstractCapability[None]):
        async def before_model_request(
            self, ctx: RunContext[None], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            raise error

    agent = Agent(TestModel(), deps_type=type(None), capabilities=[Failure()])
    store = SettingsStore(tmp_path / 'config.db')
    if mode == 'headless':
        monkeypatch.setattr(headless, 'create_agent', lambda: agent)
        monkeypatch.setattr(headless, 'STOCK_PLUGINS', ())
        assert (
            await headless.run_headless(
                text='hi there', settings=Settings(model='test'), store=store, project=ProjectSettings()
            )
            == 1
        )
        captured = capsys.readouterr()
        assert captured.out == ''
        output = captured.err
    else:
        inputs(monkeypatch, ['hi there', '/exit'])
        stream = io.StringIO()
        await chat(agent, deps=None, store=store, console=Console(file=stream, width=160))
        output = stream.getvalue()
        assert 'Goodbye.' in output
        assert 'Retained history may include partial progress' in output

    if chain in ('direct', 'cause', 'context'):
        assert 'Could not refresh your Codex login.' in output
        assert '/login openai-codex' in output
        assert 'Connection error.' not in output
    else:
        assert '/login openai-codex' not in output
        assert ('Credential storage is locked' if chain == 'persistence' else 'Connection error.') in output
    assert 'private-token-response' not in output
    saved = await SqliteConversationStore(database=tmp_path / 'sessions.db').listing()
    assert len(saved) == 1
    assert saved[0].outcome == 'failed'


ADMIN = 'Ask your Logfire admin.'


@pytest.mark.parametrize(
    ('status', 'body', 'message'),
    [
        (
            429,
            'Spending policy `per-dev` monthly limit of $50 exhausted',
            f"Your organization's AI budget for this month is used up (limit $50, policy per-dev). {ADMIN}",
        ),
        (
            429,
            'Spending policy daily limit of $5.5 for model `gpt-5` exhausted',
            f"Your organization's AI budget for today is used up (limit $5.5, for gpt-5). {ADMIN}",
        ),
        (429, 'Spending policy total limit exhausted', f"Your organization's total AI budget is used up. {ADMIN}"),
        (429, 'User limit exceeded', f"Your organization's AI budget is used up (your spending limit). {ADMIN}"),
        (
            429,
            'Project limit exceeded',
            f"Your organization's AI budget is used up (the project spending limit). {ADMIN}",
        ),
        (403, 'Forbidden - Spending limit exceeded', f"Your organization's AI budget is used up. {ADMIN}"),
    ],
)
def test_gateway_budget_rejections_are_said_plainly(status: int, body: str, message: str) -> None:
    """The Pydantic AI Gateway's plain-text spend-limit rejections, as `ModelHTTPError` carries them."""
    error = ModelHTTPError(status_code=status, model_name='anthropic:claude-sonnet-5-5', body=body)
    assert error_message(error) == message
    # Wrapped, as a fallback chain or a run error delivers it.
    try:
        raise RuntimeError('model request failed') from error
    except RuntimeError as wrapped:
        assert error_message(wrapped) == message


def test_other_gateway_errors_are_unchanged() -> None:
    error = ModelHTTPError(status_code=429, model_name='m', body='Rate limit reached for requests')
    assert error_message(error) == str(error)


@pytest.mark.parametrize('mode', ['interactive', 'headless'])
async def test_a_used_up_budget_ends_the_turn_with_one_plain_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], mode: str
) -> None:
    error = ModelHTTPError(
        status_code=429, model_name='m', body='Spending policy `per-dev` monthly limit of $50 exhausted'
    )

    class Failure(AbstractCapability[None]):
        async def before_model_request(
            self, ctx: RunContext[None], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            raise error

    agent = Agent(TestModel(), deps_type=type(None), capabilities=[Failure()])
    store = SettingsStore(tmp_path / 'config.db')
    if mode == 'headless':
        monkeypatch.setattr(headless, 'create_agent', lambda: agent)
        monkeypatch.setattr(headless, 'STOCK_PLUGINS', ())
        await headless.run_headless(text='hi', settings=Settings(model='test'), store=store, project=ProjectSettings())
        output = capsys.readouterr().err
    else:
        inputs(monkeypatch, ['hi', '/exit'])
        stream = io.StringIO()
        await chat(agent, deps=None, store=store, console=Console(file=stream, width=200))
        output = stream.getvalue()
    assert "Your organization's AI budget for this month is used up (limit $50, policy per-dev)." in ' '.join(
        output.split()
    )
    assert 'Traceback' not in output and 'status_code' not in output
