"""`clai2 --web`: the stock agent in core's browser UI, on loopback, behind the link CLAI prints.

The server is real down to the socket CLAI binds; only uvicorn's accept loop is replaced, by a client
that sends its requests straight to the app uvicorn was given.
"""

import re
import socket
from collections.abc import AsyncIterator
from io import StringIO
from pathlib import Path

import httpx2
import pytest
import uvicorn
from rich.console import Console
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_clai2 import _app
from pydantic_clai2.cli import web
from pydantic_clai2.config import Settings
from tests.clai2.cli_runner import CliRunner

CHAT = {
    'trigger': 'submit-message',
    'id': 'chat',
    'messages': [{'id': 'm1', 'role': 'user', 'parts': [{'type': 'text', 'text': 'Hello'}]}],
}


async def test_serves_the_stock_agent_to_the_browser_that_opened_the_printed_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A provider core cannot infer, so only CLAI's resolution can build it, and no provider SDK is imported.
    resolved: list[str] = []

    async def resolve(self: object, name: str) -> Model | str:
        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            yield 'Hello from CLAI'

        resolved.append(name)
        return FunctionModel(stream_function=stream, model_name=name)

    monkeypatch.setattr(_app._ModelResolver, 'resolve', resolve)  # pyright: ignore[reportPrivateUsage]
    output = StringIO()
    responses: dict[str, httpx2.Response] = {}

    async def serve(self: uvicorn.Server, sockets: list[socket.socket]) -> None:
        [listener] = sockets
        host, port = listener.getsockname()
        assert host == '127.0.0.1'
        assert port != 0
        printed = output.getvalue()
        match = re.search(rf'Open http://127\.0\.0\.1:{port}/\?token=(\S+)', printed)
        assert match is not None, printed
        token = match[1]
        app = self.config.app
        assert isinstance(app, web.TokenGate)
        transport = httpx2.ASGITransport(app=app)
        async with httpx2.AsyncClient(transport=transport, base_url=f'http://127.0.0.1:{port}') as browser:
            responses['without link'] = await browser.get('/api/health')
            responses['wrong token'] = await browser.get('/api/health', params={'token': 'guess'})
            responses['wrong method'] = await browser.post('/api/chat', params={'token': token}, json=CHAT)
            responses['link'] = await browser.get('/api/health', params={'token': token, 'keep': '1'})
            responses['health'] = await browser.get('/api/health')
            responses['configure'] = await browser.get('/api/configure')
            responses['chat'] = await browser.post('/api/chat', json=CHAT)
        assert responses['link'].headers['set-cookie'] == (
            f'clai2-web-{port}={token}; HttpOnly; Path=/; SameSite=strict'
        )

    monkeypatch.setattr(uvicorn.Server, 'serve', serve)
    await web.serve_web(
        settings=Settings(model='clai-test:echo'),
        port=0,
        workspace=tmp_path,
        console=Console(file=output, width=40),
    )

    assert f'Serving {tmp_path} with clai-test:echo. Press Ctrl-C to stop.' in output.getvalue()
    for refused in ('without link', 'wrong token', 'wrong method'):
        assert responses[refused].status_code == 403
        assert 'Open the link `clai2 --web` printed' in responses[refused].text
    assert responses['link'].status_code == 303
    assert responses['link'].headers['location'] == '/api/health?keep=1'
    assert responses['health'].json() == {'ok': True}
    assert [model['id'] for model in responses['configure'].json()['models']] == ['clai-test:echo']
    assert responses['chat'].status_code == 200
    assert '"delta":"Hello from CLAI"' in responses['chat'].text
    assert resolved == ['clai-test:echo']


def test_refuses_a_websocket_without_the_cookie() -> None:
    gate = web.TokenGate(app=Agent(TestModel()).to_web(), token='secret', cookie='clai2-web-7932')
    # As a context manager, so the app's lifespan runs through the gate too.
    with TestClient(gate, base_url='http://127.0.0.1:7932') as client:
        with pytest.raises(WebSocketDisconnect) as refused, client.websocket_connect('/?token=secret'):
            pass  # pragma: no cover
    assert refused.value.code == 1008


async def test_a_busy_port_fails_before_the_agent_opens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def unreachable(*args: object, **kwargs: object) -> None:
        raise AssertionError('the agent must not open')  # pragma: no cover

    monkeypatch.setattr(web, 'open_stock_agent', unreachable)
    with socket.create_server(('127.0.0.1', 0)) as taken:
        with pytest.raises(OSError, match='Cannot serve the browser UI: Address already in use'):
            await web.serve_web(settings=Settings(model='test'), port=taken.getsockname()[1], workspace=tmp_path)


async def test_needs_a_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='Choose a model'):
        await web.serve_web(settings=Settings(model=None), port=0, workspace=tmp_path)


@pytest.mark.parametrize(
    'args, error',
    [
        (['--port', '8000'], '--port requires --web'),
        (['--web', '--port', '65536'], '--port must be between 0 and 65535'),
        (['--web', 'config', 'list'], '--web cannot be combined with config'),
        (
            ['--web', '-p', 'hi', '--resume-codex', 'abc', '--request-limit', '5'],
            '--web cannot be combined with --prompt, --resume-codex, --request-limit',
        ),
        (
            ['--web', '--session-id', '0b5a8f7e-3c1d-4e2a-9f6b-7d8c9e0a1b2c', '--fork-session', '-w', '-a', 'm:a'],
            '--web cannot be combined with --session-id, --fork-session, --worktree, --agent',
        ),
        (['--web', '--resume'], '--web cannot be combined with --resume'),
    ],
)
def test_rejects_options_the_browser_cannot_honor(run_cli: CliRunner, args: list[str], error: str) -> None:
    result = run_cli(*args)
    assert result.returncode == 2
    assert f'error: {error}\n' in result.stderr


@pytest.mark.parametrize(
    'args, port, interrupted',
    [(['-m', 'explicit:model'], 7932, False), (['--port', '0'], 0, True)],
)
def test_cli_serves_from_the_working_directory(
    run_cli: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    port: int,
    interrupted: bool,
) -> None:
    monkeypatch.setenv('CLAI_MODEL', 'env:model')
    served: list[tuple[str | None, int, Path]] = []

    async def serve_web(*, settings: Settings, port: int, workspace: Path) -> None:
        served.append((settings.model, port, workspace))
        if interrupted:
            raise KeyboardInterrupt

    monkeypatch.setattr(web, 'serve_web', serve_web)
    result = run_cli('--web', *args, '--database', str(tmp_path / 'config.db'), cwd=tmp_path)
    assert result.returncode == 0
    assert served == [('explicit:model' if '-m' in args else 'env:model', port, tmp_path)]


def test_cli_reports_a_busy_port(run_cli: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def serve_web(**kwargs: object) -> None:
        raise OSError('Cannot serve the browser UI: Address already in use')

    monkeypatch.setattr(web, 'serve_web', serve_web)
    result = run_cli('--web', '--database', str(tmp_path / 'config.db'), cwd=tmp_path)
    assert result.returncode == 2
    assert 'error: Cannot serve the browser UI: Address already in use\n' in result.stderr
