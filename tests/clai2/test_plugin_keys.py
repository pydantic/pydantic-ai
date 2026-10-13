"""A plugin settings menu's credential rows: a masked new value saved under the plugin's name, or a saved key's name."""

import asyncio

import pytest
from anyio import to_thread
from pydantic import SecretStr
from termflow.tui.menu import MenuResult
from termflow.tui.textinput import TextInputResult

from pydantic_clai2.config import api_keys
from pydantic_clai2.config.api_keys import KeyReference
from pydantic_clai2.plugins.keys import choose_key, on_loop
from tests.clai2.menu_script import Script, pick, typed

CLOSE = MenuResult(cancelled=True)


async def choose(script: Script) -> KeyReference | None:
    return await choose_key(name='DEMO_TOKEN', label='Demo token', runners=script.runners)


def answer(monkeypatch: pytest.MonkeyPatch, choice: str | KeyReference | None) -> None:
    """Answer `prompt_api_key` directly; with saved keys its list needs a real terminal."""

    async def prompt_api_key(*, prompt: object, label: str, optional: bool = False) -> str | KeyReference | None:
        return choice

    monkeypatch.setattr('pydantic_clai2.plugins.keys.prompt_api_key', prompt_api_key)


async def test_a_masked_new_value_is_saved_under_the_plugins_name() -> None:
    script = Script(lists=[], choices=[], texts=[typed(' secret ')])
    assert await choose(script) == KeyReference(name='DEMO_TOKEN')
    assert script.opened == ['text']
    assert api_keys.load_keys()['DEMO_TOKEN'].get_secret_value() == 'secret'


@pytest.mark.parametrize('result', [TextInputResult(cancelled=True), typed('   ')])
async def test_escape_or_an_empty_value_saves_nothing(result: TextInputResult) -> None:
    assert await choose(Script(lists=[], choices=[], texts=[result])) is None
    assert api_keys.load_keys() == {}


@pytest.mark.parametrize(('confirm', 'kept'), [(CLOSE, 'theirs'), (pick(True), 'mine')])
async def test_a_key_saved_by_another_session_meanwhile_is_only_replaced_after_asking(
    monkeypatch: pytest.MonkeyPatch, confirm: MenuResult, kept: str
) -> None:
    answer(monkeypatch, 'mine')
    real_load_keys = api_keys.load_keys

    def load_keys_then_another_session_saves() -> dict[str, SecretStr]:
        keys = real_load_keys()
        api_keys.save_key(name='DEMO_TOKEN', value='theirs')
        return keys

    monkeypatch.setattr('pydantic_clai2.plugins.keys.load_keys', load_keys_then_another_session_saves)
    result = await choose(Script(lists=[], choices=[confirm], texts=[]))
    assert result == (KeyReference(name='DEMO_TOKEN') if kept == 'mine' else None)
    assert api_keys.load_keys()['DEMO_TOKEN'].get_secret_value() == kept


async def test_a_saved_key_is_returned_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    answer(monkeypatch, KeyReference(name='OTHER'))
    assert await choose(Script(lists=[], choices=[], texts=[])) == KeyReference(name='OTHER')


@pytest.mark.parametrize(('confirm', 'kept'), [(CLOSE, 'old'), (pick(False), 'old'), (pick(True), 'new')])
async def test_replacing_a_shared_key_asks_first(
    monkeypatch: pytest.MonkeyPatch, confirm: MenuResult, kept: str
) -> None:
    api_keys.save_key(name='DEMO_TOKEN', value='old')
    answer(monkeypatch, 'new')
    result = await choose(Script(lists=[], choices=[confirm], texts=[]))
    assert result == (KeyReference(name='DEMO_TOKEN') if kept == 'new' else None)
    assert api_keys.load_keys()['DEMO_TOKEN'].get_secret_value() == kept


async def test_a_flows_own_timeout_reaches_the_menu_instead_of_being_polled_forever() -> None:
    # `concurrent.futures.TimeoutError` is the builtin since Python 3.11, so it must not double as "still waiting".
    async def timed_out() -> str:
        raise TimeoutError('the service did not answer')

    loop = asyncio.get_running_loop()
    with pytest.raises(TimeoutError, match='the service did not answer'):
        await to_thread.run_sync(on_loop, timed_out, loop)


async def test_the_flows_result_is_returned() -> None:
    async def answered() -> str:
        await asyncio.sleep(0.1)  # Past one polling interval.
        return 'done'

    assert await to_thread.run_sync(on_loop, answered, asyncio.get_running_loop()) == 'done'
