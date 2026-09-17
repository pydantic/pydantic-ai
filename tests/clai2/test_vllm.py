"""Private-server discovery without network or real credentials."""

from pathlib import Path

import httpx
import pytest
from menu_script import make_context
from pydantic import SecretStr
from pydantic_ai.exceptions import UserError

from pydantic_clai2 import vllm
from pydantic_clai2.model_menu import open_model_menu


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.mark.parametrize('token', ['', 'test-secret'])
async def test_discovery(token: str) -> None:
    def response(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == 'http://localhost:8000/v1/models'
        assert request.headers.get('authorization') == (f'Bearer {token}' if token else None)
        return httpx.Response(200, json={'data': [{'id': 'b'}, {'id': 'a'}, {'id': 'a'}]})

    assert await vllm.discover(
        vllm.Connection(url='http://localhost:8000', token=SecretStr(token)), transport=httpx.MockTransport(response)
    ) == ['a', 'b']


@pytest.mark.parametrize('status', [401, 302, 200])
async def test_discovery_failure(status: int) -> None:
    with pytest.raises(UserError):
        await vllm.discover(
            vllm.Connection(url='http://localhost:8000/v1'),
            transport=httpx.MockTransport(lambda request: httpx.Response(status, json={'data': []})),
        )


def test_url_and_credentials() -> None:
    with pytest.raises(UserError, match='valid HTTP'):
        vllm.api_url('not-a-url')
    for url in ('http://user:pass@host', 'https://host?token=x', 'https://host#fragment'):
        with pytest.raises(ValueError):
            vllm.api_url(url)
    with pytest.raises(UserError, match='Connect first'):
        vllm.model('vllm:test')
    vllm.save_connection(vllm.Connection(url='http://localhost:8000', token=SecretStr('test-token')))
    assert vllm.model('vllm:my/model').model_name == 'my/model'
    vllm.save_connection(vllm.Connection(url='http://localhost:8000'))
    assert vllm.model('vllm:anonymous').model_name == 'anonymous'


@pytest.mark.parametrize('outcome', ['ok', 'cancel', 'eof', 'args'])
async def test_connect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str) -> None:
    context, _ = make_context(tmp_path)
    values = iter(['http://localhost:8000', 'secret'])

    class Prompt:
        async def prompt_async(self, label: str, *, is_password: bool = False) -> str:
            if outcome == 'eof':
                raise EOFError
            if 'Token' in label:
                assert is_password
            return next(values)

    async def discovery(connection: vllm.Connection) -> list[str]:
        return ['my/model']

    monkeypatch.setattr(vllm, 'PromptSession', Prompt)
    monkeypatch.setattr(vllm, 'discover', discovery)

    def choose(names: list[str]) -> str | None:
        return None if outcome == 'cancel' else names[0]

    monkeypatch.setattr(vllm, 'choose', choose)
    if outcome == 'args':
        with pytest.raises(ValueError, match='Usage'):
            await vllm.connect(context, ['secret'])
    else:
        result = await vllm.connect(context, [])
        assert result == ('Saved model. Applied.' if outcome == 'ok' else 'Connection cancelled.')
        if outcome == 'ok':
            assert context.settings.model == 'vllm:my/model'


@pytest.mark.parametrize('action', ['browse', 'configure', None])
async def test_saved_connection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str | None) -> None:
    context, _ = make_context(tmp_path)
    connection = vllm.Connection(url='http://localhost:8000', token=SecretStr('test'))
    vllm.save_connection(connection)
    monkeypatch.setattr(vllm, 'connection_action', lambda: action)

    async def prompt() -> vllm.Connection:
        assert action == 'configure'
        return connection

    async def discovery(saved: vllm.Connection) -> list[str]:
        assert saved == connection
        return ['my/model']

    def choose(names: list[str]) -> str:
        return names[0]

    monkeypatch.setattr(vllm, 'prompt_connection', prompt)
    monkeypatch.setattr(vllm, 'discover', discovery)
    monkeypatch.setattr(vllm, 'choose', choose)
    assert await vllm.connect(context, []) == ('Connection cancelled.' if action is None else 'Saved model. Applied.')


async def test_provider_menu_connection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context, _ = make_context(tmp_path)
    keys = iter([*'vllm', 'enter', *'vllm', 'enter'])
    monkeypatch.setattr('pydantic_clai2.model_menu.menu_key', lambda: next(keys))
    results = iter(['Connection cancelled.', 'Saved model. Applied.'])

    async def connect(context: object, args: list[str]) -> str:
        return next(results)

    monkeypatch.setattr(vllm, 'connect', connect)
    assert await open_model_menu(context) == 'Saved model. Applied.'


async def test_corrupt_connection_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context, _ = make_context(tmp_path)

    def invalid(*, account: str) -> str:
        return 'invalid'

    monkeypatch.setattr(vllm, 'load_codex_credentials', invalid)
    with pytest.raises(UserError, match='Stored connection'):
        vllm.model('vllm:test')

    async def prompt() -> None:
        return None

    monkeypatch.setattr(vllm, 'prompt_connection', prompt)
    assert await vllm.connect(context, []) == 'Connection cancelled.'


async def test_invalid_discovery() -> None:
    with pytest.raises(UserError, match='invalid model list'):
        await vllm.discover(
            vllm.Connection(url='http://localhost:8000', token=SecretStr('test')),
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={'broken': True})),
        )


async def test_broken_manifest_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context, _ = make_context(tmp_path)

    def broken(*, account: str) -> str:
        raise UserError('Broken manifest')

    async def prompt() -> None:
        return None

    monkeypatch.setattr(vllm, 'load_codex_credentials', broken)
    monkeypatch.setattr(vllm, 'prompt_connection', prompt)
    assert await vllm.connect(context, []) == 'Connection cancelled.'
