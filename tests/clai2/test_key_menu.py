"""Drive credential management without a terminal or real secrets."""

import keyring
import pytest
from keyring.errors import KeyringError, NoKeyringError
from menu_script import Script, pick, typed
from pydantic_ai.exceptions import UserError
from termflow.tui.menu import MenuResult  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.textinput import TextInputResult  # pyright: ignore[reportMissingTypeStubs]

from pydantic_clai2 import api_keys, key_menu
from pydantic_clai2.credential_store import save_codex_credentials
from pydantic_clai2.field_menu import FieldMenu
from pydantic_clai2.key_menu import KeyAction, KeysSource, build_keys_menu, keys_command, run_keys_flow


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def test_management_flow() -> None:
    script = Script(
        lists=[
            pick(KeyAction(action='add')),
            pick('FIRST'),
            pick(KeyAction(action='rename', name='FIRST')),
            pick(KeyAction(action='delete', name='RENAMED')),
            MenuResult(cancelled=True),
        ],
        choices=[pick(True)],
        texts=[typed(' first '), typed('secret'), typed('replacement'), typed('renamed')],
    )
    run_keys_flow(runners=script.runners)
    assert api_keys.load_keys() == {}
    assert script.opened == ['list', 'text', 'text', 'list', 'text', 'list', 'text', 'list', 'choice', 'list']


@pytest.mark.parametrize('cancel', [MenuResult(cancelled=True), MenuResult(item=None), pick(False)])
def test_delete_cancel(cancel: MenuResult) -> None:
    api_keys.save_key(name='KEY', value='secret')
    script = Script(
        lists=[pick(KeyAction(action='delete', name='KEY')), MenuResult(item=None)], choices=[cancel], texts=[]
    )
    run_keys_flow(runners=script.runners)
    assert api_keys.resolve_key(token=api_keys.KeyReference(name='KEY')) == 'secret'


@pytest.mark.parametrize('result', [TextInputResult(cancelled=True), TextInputResult(value=None)])
@pytest.mark.parametrize('action', [KeyAction(action='add'), 'KEY'])
def test_input_cancel(result: TextInputResult, action: KeyAction | str) -> None:
    api_keys.save_key(name='KEY', value='secret')
    script = Script(lists=[pick(action), MenuResult(cancelled=True)], choices=[], texts=[result])
    run_keys_flow(runners=script.runners)
    assert list(api_keys.load_keys()) == ['KEY']


@pytest.mark.parametrize(
    ('action', 'texts'),
    [(KeyAction(action='add'), ['KEY']), (KeyAction(action='add'), ['bad-name']), ('KEY', [''])],
)
def test_validation_keeps_key(action: KeyAction | str, texts: list[str]) -> None:
    api_keys.save_key(name='KEY', value='secret')
    script = Script(lists=[pick(action), MenuResult(cancelled=True)], choices=[], texts=[typed(t) for t in texts])
    run_keys_flow(runners=script.runners)
    assert api_keys.load_keys()['KEY'].get_secret_value() == 'secret'


def test_unknown_action() -> None:
    script = Script(lists=[pick(42), MenuResult(cancelled=True)], choices=[], texts=[])
    run_keys_flow(runners=script.runners)
    assert not api_keys.load_keys()


@pytest.mark.parametrize('key', ['a', 'r', 'd', 'enter', 'escape', 'ctrl-c'])
def test_menu_keys(monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    pressed = iter([key])
    monkeypatch.setattr(key_menu, 'menu_key', lambda: next(pressed))
    result = build_keys_menu(names=['KEY']).run()
    if key in ('escape', 'ctrl-c'):
        assert result.cancelled
    else:
        assert result.item is not None
        expected = {
            'a': KeyAction(action='add'),
            'r': KeyAction(action='rename', name='KEY'),
            'd': KeyAction(action='delete', name='KEY'),
            'enter': 'KEY',
        }
        assert result.item.value == expected[key]


def test_empty_menu_action(monkeypatch: pytest.MonkeyPatch) -> None:
    pressed = iter(['r', 'd', 'a'])
    monkeypatch.setattr(key_menu, 'menu_key', lambda: next(pressed))
    result = build_keys_menu(names=[]).run()
    assert result.item is not None and result.item.value == KeyAction(action='add')


def test_masked_editor(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    api_keys.save_key(name='KEY', value='existing-secret')
    source = KeysSource()
    row = source.rows()[0]
    assert source.current(row) == '(hidden)'
    assert source.problem(row, '') is not None
    assert source.problem(row, 'new-secret') is None
    pressed = iter([*'new-secret', 'enter'])
    monkeypatch.setattr('pydantic_clai2.field_menu.menu_key', lambda: next(pressed))
    widget = FieldMenu(source).build_editor(row)
    assert widget.run().value == 'new-secret'
    output = capsys.readouterr().out
    assert 'existing-secret' not in output and 'new-secret' not in output


@pytest.mark.parametrize('failure', [OSError('secret'), KeyringError('secret'), UserError('invalid bundle')])
def test_backend_failure(monkeypatch: pytest.MonkeyPatch, failure: Exception) -> None:
    def broken() -> dict[str, object]:
        raise failure

    monkeypatch.setattr(api_keys, 'load_keys', broken)
    script = Script(lists=[pick(KeyAction(action='add')), MenuResult(cancelled=True)], choices=[], texts=[typed('KEY')])
    run_keys_flow(runners=script.runners)


def test_plaintext_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(service: str, account: str) -> None:
        raise NoKeyringError

    monkeypatch.setattr(keyring, 'get_password', unavailable)
    api_keys.save_key(name='KEY', value='secret')
    script = Script(lists=[MenuResult(cancelled=True)], choices=[], texts=[])
    run_keys_flow(runners=script.runners)


async def test_command(monkeypatch: pytest.MonkeyPatch) -> None:
    pressed = iter(['escape'])
    monkeypatch.setattr(key_menu, 'menu_key', lambda: next(pressed))
    assert await keys_command([]) == ''
    with pytest.raises(ValueError, match='Usage'):
        await keys_command(['secret'])


def test_storage_rename_and_delete() -> None:
    api_keys.save_key(name='KEY', value='secret')
    assert api_keys.rename_key(name='KEY', new_name='key') == 'API key unchanged.'
    with pytest.raises(ValueError, match='no longer exists'):
        api_keys.rename_key(name='MISSING', new_name='NEW')
    api_keys.save_key(name='OTHER', value='other')
    with pytest.raises(ValueError, match='already exists'):
        api_keys.rename_key(name='KEY', new_name='OTHER')
    save_codex_credentials(account='vllm', value='{"token":{"name":"KEY"}}')
    save_codex_credentials(account='openrouter', value='{"token":"inline"}')
    assert api_keys.key_users(name='KEY') == ['vllm']
    with pytest.raises(ValueError, match='used by vllm'):
        api_keys.rename_key(name='KEY', new_name='NEW')
    save_codex_credentials(account='vllm', value='broken')
    with pytest.raises(UserError, match='invalid vllm'):
        api_keys.key_users(name='KEY')
    api_keys.delete_key(name='MISSING')
    assert set(api_keys.load_keys()) == {'KEY', 'OTHER'}


def test_rename_preserves_other_keys() -> None:
    api_keys.save_key(name='KEY', value='secret')
    api_keys.save_key(name='OTHER', value='other-secret')
    save_codex_credentials(account='vllm', value='{"token":{"name":"OTHER"}}')
    assert api_keys.rename_key(name='KEY', new_name=' new ') == 'Renamed KEY to NEW.'
    assert set(api_keys.load_keys()) == {'NEW', 'OTHER'}
    assert api_keys.resolve_key(token=api_keys.KeyReference(name='NEW')) == 'secret'
    assert api_keys.resolve_key(token=api_keys.KeyReference(name='OTHER')) == 'other-secret'


def test_disabled_row_has_no_key_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    # Filtering can focus a disabled status row, which is not a credential.
    pressed = iter(['z', 'r', 'd', 'escape'])
    monkeypatch.setattr(key_menu, 'menu_key', lambda: next(pressed))
    assert build_keys_menu(names=['KEY'], message='zzz status').run().cancelled
