"""Private-server discovery without network or real credentials."""

from pathlib import Path

import httpx
import httpx2
import pytest
from pydantic import SecretStr

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError, UserError
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.vllm import VLLMProvider
from pydantic_clai2.models import vllm
from pydantic_clai2.ui.menus.model_menu import open_add_model_menu
from tests.clai2.menu_script import QUIET, make_context


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
    monkeypatch.setattr('pydantic_clai2.ui.menus.model_menu.menu_key', lambda: next(keys))
    results = iter(['Connection cancelled.', 'Saved model. Applied.'])

    async def connect(context: object, args: list[str]) -> str:
        return next(results)

    monkeypatch.setattr(vllm, 'connect', connect)
    assert await open_add_model_menu(context, console=QUIET) == 'Saved model. Applied.'


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


COMPLETION = {
    'id': 'chatcmpl-1',
    'object': 'chat.completion',
    'created': 0,
    'model': 'my/model',
    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'hello'}, 'finish_reason': 'stop'}],
}


def gateway_model(
    monkeypatch: pytest.MonkeyPatch, statuses: list[int], delays: list[float], headers: dict[str, str] | None = None
) -> OpenAIChatModel:
    """The saved vLLM model's client, answering with `statuses` in turn and recording backoff without sleeping."""
    vllm.save_connection(vllm.Connection(url='http://localhost:8000', token=SecretStr('test-token')))
    calculate = vllm.RetryingClient._calculate_retry_timeout  # pyright: ignore[reportPrivateUsage]

    def no_wait(self: vllm.RetryingClient, *args: object, **kwargs: object) -> float:
        delays.append(calculate(self, *args, **kwargs))  # pyright: ignore[reportArgumentType]
        return 0

    monkeypatch.setattr(vllm.RetryingClient, '_calculate_retry_timeout', no_wait)
    replies = iter(statuses)

    def respond(request: httpx2.Request) -> httpx2.Response:
        status = next(replies)
        body = COMPLETION if status == 200 else {'error': f'http_{status}', 'detail': {'error': 'invalid_api_key'}}
        return httpx2.Response(status, json=body, headers=headers)

    client = vllm.model('vllm:my/model').client
    assert isinstance(client, vllm.RetryingClient)
    assert client.max_retries == vllm.MAX_RETRIES
    mocked = client.with_options(http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)))
    return OpenAIChatModel('my/model', provider=VLLMProvider(openai_client=mocked))


@pytest.mark.usefixtures('allow_model_requests')
async def test_gateway_403_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gateway 403 under load cannot be recorded on demand, so a mock transport plays the status sequence."""
    delays: list[float] = []
    model = gateway_model(monkeypatch, [403, 500, 200], delays)
    assert (await Agent(model).run('hi')).output == 'hello'
    assert len(delays) == 2
    assert 0.375 <= delays[0] <= 0.5 < 0.75 <= delays[1] <= 1


@pytest.mark.parametrize(('status', 'headers'), [(401, None), (403, {'x-should-retry': 'false'})])
@pytest.mark.usefixtures('allow_model_requests')
async def test_not_retried(monkeypatch: pytest.MonkeyPatch, status: int, headers: dict[str, str] | None) -> None:
    """Mocked like the test above: a 401, or a 403 the server marks as final, fails on the first response."""
    delays: list[float] = []
    model = gateway_model(monkeypatch, [status], delays, headers)
    with pytest.raises(ModelHTTPError) as error:
        await Agent(model).run('hi')
    assert error.value.status_code == status
    assert delays == []


@pytest.mark.usefixtures('allow_model_requests')
async def test_403_retries_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mocked like the tests above: a key that keeps getting 403 fails after `MAX_RETRIES` retries."""
    delays: list[float] = []
    model = gateway_model(monkeypatch, [403] * (vllm.MAX_RETRIES + 1), delays)
    with pytest.raises(ModelHTTPError) as error:
        await Agent(model).run('hi')
    assert error.value.status_code == 403
    assert len(delays) == vllm.MAX_RETRIES


async def test_provider_owns_its_http_client() -> None:
    """Checks the client lifecycle, without sending a request."""
    vllm.save_connection(vllm.Connection(url='http://localhost:8000'))
    model = vllm.model('vllm:my/model')
    async with model:
        pass
    assert model.client.is_closed()
