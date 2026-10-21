"""A plugin settings menu's credential rows: a masked new value saved under the plugin's name, or a saved key's name."""

import asyncio

import pytest
from anyio import to_thread
from pydantic import SecretStr
from termflow.tui.menu import Menu, MenuResult
from termflow.tui.textinput import TextInput, TextInputResult

from pydantic_clai2.config import api_keys
from pydantic_clai2.config.api_keys import KeyReference
from pydantic_clai2.plugins.keys import choose_key, on_loop
from pydantic_clai2.plugins.menus import Option
from pydantic_clai2.ui.menus.dialogs import TerminalDialogs
from pydantic_clai2.ui.menus.field_menu import Runners
from tests.clai2.dialog_script import Answer, Asked, Back, Confirm, Pick, ScriptedDialogs, Type
from tests.clai2.menu_script import pick, typed

TYPING = Asked(
    kind='text',
    title='Demo token',
    label='Token',
    placeholder='Paste the token; it is saved in /keys as DEMO_TOKEN',
    secret=True,
)


async def choose(dialogs: ScriptedDialogs) -> KeyReference | None:
    reference = await choose_key(name='DEMO_TOKEN', label='Demo token', dialogs=dialogs)
    assert dialogs.unused == []
    return reference


def reject_bots(value: str) -> None:
    if value.startswith('bot-'):
        raise ValueError('Bot tokens are refused.')


async def test_with_no_saved_keys_a_masked_value_is_typed_and_saved_under_the_plugins_name() -> None:
    dialogs = ScriptedDialogs([Type(text=' secret ')])
    assert await choose(dialogs) == KeyReference(name='DEMO_TOKEN')
    assert dialogs.asked == [TYPING]
    assert api_keys.load_keys()['DEMO_TOKEN'].get_secret_value() == 'secret'


@pytest.mark.parametrize('answer', [Back(), Type(text='   ')])
async def test_backing_out_or_an_empty_value_saves_nothing(answer: Answer) -> None:
    assert await choose(ScriptedDialogs([answer])) is None
    assert api_keys.load_keys() == {}


async def test_saved_keys_are_offered_by_name_before_typing_a_new_one() -> None:
    api_keys.save_key(name='OTHER', value='other-secret')
    api_keys.save_key(name='ALPHA', value='alpha-secret')
    dialogs = ScriptedDialogs([Pick(value='OTHER')])
    assert await choose(dialogs) == KeyReference(name='OTHER')
    assert dialogs.asked == [
        Asked(
            kind='choose',
            title='Demo token',
            options=(
                Option(value='ALPHA', label='ALPHA'),
                Option(value='OTHER', label='OTHER'),
                Option(value='enter', label='Enter a different API key'),
            ),
            description='Choose a saved key for this connection. Only key names are displayed.',
        )
    ]
    assert await choose(ScriptedDialogs([Back()])) is None
    dialogs = ScriptedDialogs([Pick(value='enter'), Type(text='typed-secret')])
    assert await choose(dialogs) == KeyReference(name='DEMO_TOKEN')
    assert dialogs.asked[1] == TYPING
    assert api_keys.load_keys()['DEMO_TOKEN'].get_secret_value() == 'typed-secret'


@pytest.mark.parametrize(
    ('confirm', 'kept'), [(Back(), 'old'), (Confirm(yes=False), 'old'), (Confirm(yes=True), 'new')]
)
async def test_replacing_a_shared_key_asks_first(confirm: Answer, kept: str) -> None:
    api_keys.save_key(name='DEMO_TOKEN', value='old')
    dialogs = ScriptedDialogs([Pick(value='enter'), Type(text='new'), confirm])
    assert await choose(dialogs) == (KeyReference(name='DEMO_TOKEN') if kept == 'new' else None)
    assert dialogs.asked[-1] == Asked(
        kind='confirm',
        title='DEMO_TOKEN is already in /keys',
        yes='Replace DEMO_TOKEN for every plugin and connection that uses it',
        no='Keep the saved token',
    )
    assert api_keys.load_keys()['DEMO_TOKEN'].get_secret_value() == kept


@pytest.mark.parametrize(('confirm', 'kept'), [(Back(), 'theirs'), (Confirm(yes=True), 'mine')])
async def test_a_key_saved_by_another_session_meanwhile_is_only_replaced_after_asking(
    monkeypatch: pytest.MonkeyPatch, confirm: Answer, kept: str
) -> None:
    dialogs = ScriptedDialogs([Type(text='mine'), confirm])
    real_load_keys = api_keys.load_keys

    def load_keys_then_another_session_saves() -> dict[str, SecretStr]:
        keys = real_load_keys()
        if dialogs.asked:  # The name was free when the typed token was checked against /keys.
            api_keys.save_key(name='DEMO_TOKEN', value='theirs')
        return keys

    monkeypatch.setattr('pydantic_clai2.plugins.keys.load_keys', load_keys_then_another_session_saves)
    assert await choose(dialogs) == (KeyReference(name='DEMO_TOKEN') if kept == 'mine' else None)
    assert [asked.kind for asked in dialogs.asked] == ['text', 'confirm']
    assert api_keys.load_keys()['DEMO_TOKEN'].get_secret_value() == kept


async def test_check_refuses_a_picked_or_typed_value_the_service_would_reject(monkeypatch: pytest.MonkeyPatch) -> None:
    api_keys.save_key(name='BOT', value='bot-picked')
    for answers in ([Pick(value='BOT')], [Pick(value='enter'), Type(text='bot-typed')]):
        with pytest.raises(ValueError, match='Bot tokens are refused'):
            await choose_key(name='DEMO_TOKEN', label='Demo token', dialogs=ScriptedDialogs(answers), check=reject_bots)
    assert 'DEMO_TOKEN' not in api_keys.load_keys()
    # A value `check` accepts is saved as usual.
    accepted = ScriptedDialogs([Pick(value='enter'), Type(text='user-typed')])
    assert await choose_key(name='DEMO_TOKEN', label='Demo token', dialogs=accepted, check=reject_bots) == KeyReference(
        name='DEMO_TOKEN'
    )
    api_keys.delete_key(name='DEMO_TOKEN')

    real_load_keys = api_keys.load_keys
    dialogs = ScriptedDialogs([Pick(value='BOT')])

    def load_keys_after_another_session_deleted_it() -> dict[str, SecretStr]:
        if dialogs.asked:
            api_keys.delete_key(name='BOT')
        return real_load_keys()

    monkeypatch.setattr('pydantic_clai2.plugins.keys.load_keys', load_keys_after_another_session_deleted_it)
    # Nothing to check: saving the reference reports the missing key.
    reference = await choose_key(name='DEMO_TOKEN', label='Demo token', dialogs=dialogs, check=reject_bots)
    assert reference == KeyReference(name='BOT')


async def test_the_terminal_shows_the_key_list_the_masked_input_and_the_replace_question() -> None:
    api_keys.save_key(name='DEMO_TOKEN', value='old')
    menus: list[Menu] = []
    widgets: list[TextInput] = []
    answers = [pick('enter'), pick(True)]

    def run_choice(menu: Menu) -> MenuResult:
        menus.append(menu)
        return answers.pop(0)

    def run_text(widget: TextInput) -> TextInputResult:
        widgets.append(widget)
        return typed(' new ')

    dialogs = TerminalDialogs(Runners(run_choice=run_choice, run_text=run_text))
    assert await choose_key(name='DEMO_TOKEN', label='Demo token', dialogs=dialogs) == KeyReference(name='DEMO_TOKEN')
    assert api_keys.load_keys()['DEMO_TOKEN'].get_secret_value() == 'new'
    [keys, replace] = menus
    assert keys._title == 'Demo token'  # pyright: ignore[reportPrivateUsage]
    assert [item.label for item in keys._items] == ['DEMO_TOKEN', 'Enter a different API key']  # pyright: ignore[reportPrivateUsage]
    preview = keys._preview  # pyright: ignore[reportPrivateUsage]
    assert preview is not None
    assert preview(keys._items[0]) == 'Choose a saved key for this connection. Only key names are displayed.'  # pyright: ignore[reportPrivateUsage]
    [widget] = widgets
    assert (widget._prompt, widget._placeholder, widget._mask) == (  # pyright: ignore[reportPrivateUsage]
        'Token: ',
        'Paste the token; it is saved in /keys as DEMO_TOKEN',
        '*',
    )
    assert replace._title == 'DEMO_TOKEN is already in /keys'  # pyright: ignore[reportPrivateUsage]
    assert [item.label for item in replace._items] == [  # pyright: ignore[reportPrivateUsage]
        'Keep the saved token',
        'Replace DEMO_TOKEN for every plugin and connection that uses it',
    ]


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
