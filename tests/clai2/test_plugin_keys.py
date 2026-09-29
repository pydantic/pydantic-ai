"""Choosing a plugin's `/keys` entry: a masked new value saved under the plugin's name, or a saved key's name."""

import asyncio

import pytest
from pydantic import SecretStr
from termflow.tui.menu import MenuResult
from termflow.tui.textinput import TextInputResult

from pydantic_clai2 import api_keys
from pydantic_clai2.api_keys import KeyReference
from pydantic_clai2.plugin_keys import choose_key, pick_key_from_menu
from tests.clai2.menu_script import Script, pick, typed

CLOSE = MenuResult(cancelled=True)


async def choose(script: Script) -> KeyReference | None:
    return await choose_key(name='DEMO_TOKEN', label='Demo token', placeholder='Paste it', runners=script.runners)


def answer(monkeypatch: pytest.MonkeyPatch, choice: str | KeyReference | None) -> None:
    """Answer `prompt_api_key` directly; with saved keys its list needs a real terminal."""

    async def prompt_api_key(*, prompt: object, label: str, optional: bool = False) -> str | KeyReference | None:
        return choice

    monkeypatch.setattr('pydantic_clai2.plugin_keys.prompt_api_key', prompt_api_key)


async def test_a_masked_new_value_is_saved_under_the_plugins_name() -> None:
    script = Script(lists=[], choices=[], texts=[typed(' secret ')])
    assert await choose(script) == KeyReference(name='DEMO_TOKEN')
    assert script.opened == ['text']
    assert api_keys.load_keys()['DEMO_TOKEN'].get_secret_value() == 'secret'


@pytest.mark.parametrize('result', [TextInputResult(cancelled=True), typed('   ')])
async def test_escape_or_an_empty_value_saves_nothing(result: TextInputResult) -> None:
    assert await choose(Script(lists=[], choices=[], texts=[result])) is None
    assert api_keys.load_keys() == {}


async def test_a_key_saved_by_another_session_meanwhile_is_not_overwritten(monkeypatch: pytest.MonkeyPatch) -> None:
    answer(monkeypatch, 'mine')
    real_load_keys = api_keys.load_keys

    def load_keys_then_another_session_saves() -> dict[str, SecretStr]:
        keys = real_load_keys()
        api_keys.save_key(name='DEMO_TOKEN', value='theirs')
        return keys

    monkeypatch.setattr('pydantic_clai2.plugin_keys.load_keys', load_keys_then_another_session_saves)
    with pytest.raises(api_keys.KeyExistsError, match='DEMO_TOKEN is already saved'):
        await choose(Script(lists=[], choices=[], texts=[]))
    assert api_keys.load_keys()['DEMO_TOKEN'].get_secret_value() == 'theirs'


async def test_a_saved_key_is_returned_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    answer(monkeypatch, KeyReference(name='OTHER'))
    assert await choose(Script(lists=[], choices=[], texts=[])) == KeyReference(name='OTHER')


async def test_a_menu_thread_waits_for_a_slow_picker(monkeypatch: pytest.MonkeyPatch) -> None:
    async def prompt_api_key(*, prompt: object, label: str, optional: bool = False) -> str | KeyReference | None:
        await asyncio.sleep(0.2)
        return KeyReference(name='SLOW')

    monkeypatch.setattr('pydantic_clai2.plugin_keys.prompt_api_key', prompt_api_key)
    runners = Script(lists=[], choices=[], texts=[]).runners
    loop = asyncio.get_running_loop()
    picked = await asyncio.to_thread(
        lambda: pick_key_from_menu(loop, name='DEMO_TOKEN', label='Demo token', placeholder='Paste it', runners=runners)
    )
    assert picked == KeyReference(name='SLOW')


@pytest.mark.parametrize(('confirm', 'kept'), [(CLOSE, 'old'), (pick(False), 'old'), (pick(True), 'new')])
async def test_replacing_a_shared_key_asks_first(
    monkeypatch: pytest.MonkeyPatch, confirm: MenuResult, kept: str
) -> None:
    api_keys.save_key(name='DEMO_TOKEN', value='old')
    answer(monkeypatch, 'new')
    result = await choose(Script(lists=[], choices=[confirm], texts=[]))
    assert result == (KeyReference(name='DEMO_TOKEN') if kept == 'new' else None)
    assert api_keys.load_keys()['DEMO_TOKEN'].get_secret_value() == kept
