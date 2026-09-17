"""Private-server discovery without network or real credentials."""

from pathlib import Path

import httpx
import pytest
from menu_script import make_context
from pydantic import SecretStr
from pydantic_ai.exceptions import UserError

from pydantic_clai2 import openrouter
from pydantic_clai2.model_menu import open_model_menu


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.mark.parametrize('token', ['test-secret'])
async def test_discovery(token: str) -> None:
    def response(request: httpx.Request) -> httpx.Response:
        assert str(request.url) in ('https://openrouter.ai/api/v1/models', 'https://openrouter.ai/api/v1/key')
        assert request.headers.get('authorization') == (f'Bearer {token}' if token else None)
        return httpx.Response(200, json={'data': [{'id': 'b'}, {'id': 'a'}, {'id': 'a'}]})

    assert await openrouter.discover(
        openrouter.Connection(token=SecretStr(token)), transport=httpx.MockTransport(response)
    ) == ['a', 'b']


@pytest.mark.parametrize('status', [401, 302, 200])
async def test_discovery_failure(status: int) -> None:
    with pytest.raises(UserError):
        await openrouter.discover(
            openrouter.Connection(token=SecretStr('test-token')),
            transport=httpx.MockTransport(lambda request: httpx.Response(status, json={'data': []})),
        )


def test_url_and_credentials() -> None:
    with pytest.raises(UserError, match='Connect first'):
        openrouter.model('openrouter:test')
    openrouter.save_connection(openrouter.Connection(token=SecretStr('test-token')))
    assert openrouter.model('openrouter:my/model').model_name == 'my/model'


@pytest.mark.parametrize('outcome', ['ok', 'cancel', 'eof', 'args', 'empty'])
async def test_connect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str) -> None:
    context, _ = make_context(tmp_path)
    values = iter(['' if outcome == 'empty' else 'secret'])

    class Prompt:
        async def prompt_async(self, label: str, *, is_password: bool = False) -> str:
            if outcome == 'eof':
                raise EOFError
            if 'API key' in label:
                assert is_password
            return next(values)

    async def discovery(connection: openrouter.Connection) -> list[str]:
        return ['my/model']

    keys = iter(['down', 'enter'])
    monkeypatch.setattr(openrouter, 'menu_key', lambda: next(keys))
    monkeypatch.setattr(openrouter, 'PromptSession', Prompt)
    monkeypatch.setattr(openrouter, 'discover', discovery)

    def choose(names: list[str]) -> str | None:
        return None if outcome == 'cancel' else names[0]

    monkeypatch.setattr(openrouter, 'choose', choose)
    if outcome == 'args':
        with pytest.raises(ValueError, match='Usage'):
            await openrouter.connect(context, ['secret'])
    elif outcome == 'empty':
        with pytest.raises(ValueError, match='required'):
            await openrouter.connect(context, [])
    else:
        result = await openrouter.connect(context, [])
        assert result == ('Saved model. Applied.' if outcome == 'ok' else 'Connection cancelled.')
        if outcome == 'ok':
            assert context.settings.model == 'openrouter:my/model'


@pytest.mark.parametrize('action', ['browse', 'configure', None])
async def test_saved_connection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str | None) -> None:
    context, _ = make_context(tmp_path)
    connection = openrouter.Connection(token=SecretStr('test'))
    openrouter.save_connection(connection)
    monkeypatch.setattr(openrouter, 'connection_action', lambda: action)

    async def prompt() -> openrouter.Connection:
        assert action == 'configure'
        return connection

    async def discovery(saved: openrouter.Connection) -> list[str]:
        assert saved == connection
        return ['my/model']

    def choose(names: list[str]) -> str:
        return names[0]

    monkeypatch.setattr(openrouter, 'prompt_connection', prompt)
    monkeypatch.setattr(openrouter, 'discover', discovery)
    monkeypatch.setattr(openrouter, 'choose', choose)
    assert await openrouter.connect(context, []) == (
        'Connection cancelled.' if action is None else 'Saved model. Applied.'
    )


async def test_provider_menu_connection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context, _ = make_context(tmp_path)
    keys = iter([*'openrouter', 'enter', *'openrouter', 'enter'])
    monkeypatch.setattr('pydantic_clai2.model_menu.menu_key', lambda: next(keys))
    results = iter(['Connection cancelled.', 'Saved model. Applied.'])

    async def connect(context: object, args: list[str]) -> str:
        return next(results)

    monkeypatch.setattr(openrouter, 'connect', connect)
    assert await open_model_menu(context) == 'Saved model. Applied.'


async def test_corrupt_connection_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context, _ = make_context(tmp_path)

    def invalid(*, account: str) -> str:
        return 'invalid'

    monkeypatch.setattr(openrouter, 'load_codex_credentials', invalid)
    with pytest.raises(UserError, match='Stored connection'):
        openrouter.model('openrouter:test')

    async def prompt() -> None:
        return None

    monkeypatch.setattr(openrouter, 'prompt_connection', prompt)
    assert await openrouter.connect(context, []) == 'Connection cancelled.'


async def test_invalid_discovery() -> None:
    with pytest.raises(UserError, match='invalid model list'):
        await openrouter.discover(
            openrouter.Connection(token=SecretStr('test')),
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={'broken': True})),
        )


async def test_broken_manifest_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context, _ = make_context(tmp_path)

    def broken(*, account: str) -> str:
        raise UserError('Broken manifest')

    async def prompt() -> None:
        return None

    monkeypatch.setattr(openrouter, 'load_codex_credentials', broken)
    monkeypatch.setattr(openrouter, 'prompt_connection', prompt)
    assert await openrouter.connect(context, []) == 'Connection cancelled.'
