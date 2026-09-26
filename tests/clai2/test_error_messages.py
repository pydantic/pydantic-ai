"""Provider failures remain recoverable without exposing nested authentication payloads."""

import io
from pathlib import Path

import pytest
from httpx2 import Request
from openai import APIConnectionError
from pydantic_ai import Agent, ModelRequestContext, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.openai_codex import CredentialsPersistenceError, CredentialsRefreshError
from pydantic_ai_harness.step_persistence.conversations import SqliteConversationStore
from rich.console import Console
from test_app_edges import inputs

from pydantic_clai2 import chat, headless
from pydantic_clai2.config import Settings
from pydantic_clai2.project_settings import ProjectSettings
from pydantic_clai2.settings_store import SettingsStore


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
        monkeypatch.setattr(headless, 'DEFAULT_PLUGINS', ())
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
