"""Named secrets stay in credentials and are picked by label only."""

from pathlib import Path

import keyring
import pytest
from keyring.errors import NoKeyringError
from menu_script import make_context
from pydantic_ai.exceptions import UserError
from termflow.tui.menu import Menu, MenuResult  # pyright: ignore[reportMissingTypeStubs]

from pydantic_clai2 import api_keys, openrouter, set_menu, vllm
from pydantic_clai2.commands import set_completions
from pydantic_clai2.credential_store import credentials_path, save_codex_credentials


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


class Prompt:
    def __init__(self, *, values: list[str | BaseException]) -> None:
        self.values = iter(values)
        self.labels: list[tuple[str, bool]] = []

    async def prompt_async(self, label: str, *, is_password: bool = False) -> str:
        self.labels.append((label, is_password))
        value = next(self.values)
        if isinstance(value, BaseException):
            raise value
        return value


def test_storage() -> None:
    assert api_keys.load_keys() == {}
    assert api_keys.save_key(name=' my_key ', value=' secret ') == 'Saved MY_KEY in the OS keyring.'
    api_keys.save_key(name='other', value='second')
    api_keys.save_key(name='my_key', value='replacement')
    keys = api_keys.load_keys()
    assert keys['MY_KEY'].get_secret_value() == 'replacement'
    assert keys['OTHER'].get_secret_value() == 'second'
    assert 'replacement' not in repr(keys)
    assert not credentials_path(account='api-keys').exists()
    assert 'api_key' in set_completions([])


@pytest.mark.parametrize('name', ['', '1KEY', 'bad-name', '../key', 'A B'])
def test_invalid_name(name: str) -> None:
    with pytest.raises(ValueError, match='Use letters'):
        api_keys.save_key(name=name, value='secret')
    assert not api_keys.load_keys()


def test_empty_and_corrupt() -> None:
    with pytest.raises(ValueError, match='required'):
        api_keys.save_key(name='KEY', value=' ')
    save_codex_credentials(account='api-keys', value='{"KEY": ["do-not-leak"]}')
    with pytest.raises(UserError, match='Stored API keys') as error:
        api_keys.load_keys()
    assert 'do-not-leak' not in str(error.value)


def test_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(service: str, account: str) -> None:
        raise NoKeyringError

    monkeypatch.setattr(keyring, 'get_password', unavailable)
    result = api_keys.save_key(name='key', value='secret')
    path = credentials_path(account='api-keys')
    assert 'plaintext' in result and str(path) in result and 'secret' not in result
    assert path.stat().st_mode & 0o777 == 0o600
    assert api_keys.load_keys()['KEY'].get_secret_value() == 'secret'


@pytest.mark.parametrize('answer', [None, 'y', 'n'])
async def test_set_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: str | None) -> None:
    context, _ = make_context(tmp_path)
    values: list[str | BaseException] = [' my_key ']
    if answer is not None:
        api_keys.save_key(name='MY_KEY', value='old')
        values.append(answer)
    values.append('new-secret')
    prompt = Prompt(values=values)
    monkeypatch.setattr(api_keys, 'PromptSession', lambda: prompt)
    result = await set_menu.set_command(context, ['api_key'])
    assert 'new-secret' not in result
    assert api_keys.load_keys()['MY_KEY'].get_secret_value() == ('old' if answer == 'n' else 'new-secret')
    if answer != 'n':
        assert prompt.labels[-1] == ('API key value for MY_KEY: ', True)
    assert await set_menu.set_command(context, ['display.thinking', 'false']) == 'Saved display.thinking. Applied.'
    with pytest.raises(ValueError, match='Usage'):
        await set_menu.set_command(context, ['api_key', 'secret'])


@pytest.mark.parametrize('values', [[EOFError()], ['key', KeyboardInterrupt()], ['key', EOFError()]])
async def test_entry_cancel(monkeypatch: pytest.MonkeyPatch, values: list[str | BaseException]) -> None:
    prompt = Prompt(values=values)
    monkeypatch.setattr(api_keys, 'PromptSession', lambda: prompt)
    assert await api_keys.set_api_key(args=[]) == 'API key entry cancelled.'
    assert not api_keys.load_keys()


@pytest.mark.parametrize('value', ['manual', EOFError(), KeyboardInterrupt()])
async def test_direct_prompt(value: str | BaseException) -> None:
    prompt = Prompt(values=[value])
    assert await api_keys.prompt_api_key(prompt=prompt, label='Key: ') == (value if isinstance(value, str) else None)
    assert prompt.labels == [('Key: ', True)]


@pytest.mark.parametrize(
    ('keys', 'optional', 'expected'),
    [
        (['enter'], False, 'saved-secret'),
        (['down', 'enter'], False, 'manual'),
        (['down', 'down', 'enter'], True, ''),
        (['escape'], False, None),
        (['ctrl-c'], True, None),
    ],
)
async def test_picker(monkeypatch: pytest.MonkeyPatch, keys: list[str], optional: bool, expected: str | None) -> None:
    api_keys.save_key(name='KEY', value='saved-secret')
    pressed = iter(keys)
    monkeypatch.setattr(api_keys, 'menu_key', lambda: next(pressed))
    prompt = Prompt(values=['manual'])
    result = await api_keys.prompt_api_key(prompt=prompt, label='Key: ', optional=optional)
    if isinstance(result, api_keys.KeyReference):
        assert result.name == 'KEY'
        result = api_keys.resolve_key(token=result)
    assert result == expected
    assert bool(prompt.labels) == (expected == 'manual')


async def test_empty_menu_result(monkeypatch: pytest.MonkeyPatch) -> None:
    api_keys.save_key(name='KEY', value='secret')

    def empty(menu: Menu) -> MenuResult:
        return MenuResult(item=None)

    monkeypatch.setattr(Menu, 'run', empty)
    assert await api_keys.prompt_api_key(prompt=Prompt(values=[]), label='Key: ') is None


@pytest.mark.parametrize('provider', ['vllm', 'openrouter'])
@pytest.mark.parametrize('cancel', [False, True])
async def test_provider_saved_key(monkeypatch: pytest.MonkeyPatch, provider: str, cancel: bool) -> None:
    api_keys.save_key(name='MY_KEY', value='saved-secret')
    pressed = iter(['escape' if cancel else 'enter'])
    monkeypatch.setattr(api_keys, 'menu_key', lambda: next(pressed))
    if provider == 'vllm':
        prompt = Prompt(values=['http://localhost:8000'])
        monkeypatch.setattr(vllm, 'PromptSession', lambda: prompt)
        connection = await vllm.prompt_connection()
    else:
        method = iter(['down', 'enter'])
        monkeypatch.setattr(openrouter, 'menu_key', lambda: next(method))
        monkeypatch.setattr(openrouter, 'PromptSession', lambda: Prompt(values=[]))
        connection = await openrouter.prompt_connection()
    if cancel:
        assert connection is None
    else:
        assert connection is not None
        assert connection.token == api_keys.KeyReference(name='MY_KEY')
        assert api_keys.resolve_key(token=connection.token) == 'saved-secret'
