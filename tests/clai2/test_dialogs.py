"""`Dialogs` in the terminal, and settings menus declared with `Plugin.settings_menu`."""

import asyncio
import io
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager

import anyio
import pytest
from pydantic import BaseModel
from rich.console import Console
from termflow.tui import MenuItem
from termflow.tui.menu import Menu, MenuResult
from termflow.tui.pager import Pager
from termflow.tui.textinput import TextInput, TextInputResult

from pydantic_clai2.plugins import Plugin, PluginHost, load_plugin
from pydantic_clai2.plugins.menus import Dialogs, FieldRow, Option, SettingsMenu
from pydantic_clai2.ui.menus import dialogs
from pydantic_clai2.ui.menus.dialogs import TerminalDialogs, build_pager, run_settings_menu, wrap
from pydantic_clai2.ui.menus.field_menu import Runners
from pydantic_clai2.ui.menus.menu_worker import holding_output
from tests.clai2.dialog_script import Back, Pick, ScriptedDialogs, Type
from tests.clai2.menu_script import UNTIL_CLOSED, Script, pick, typed

CLOSED = MenuResult(cancelled=True)


def shows(menus: list[Menu], result: MenuResult) -> Runners:
    """Runners that remember each menu and answer `result`."""

    def run(menu: Menu) -> MenuResult:
        menus.append(menu)
        return result

    return Runners(run_list=run, run_choice=run)


def labels(menu: Menu) -> list[str]:
    return [item.label for item in menu._items]  # pyright: ignore[reportPrivateUsage]


def preview(menu: Menu, item: MenuItem) -> str:
    render = menu._preview  # pyright: ignore[reportPrivateUsage]
    assert render is not None
    return render(item)


async def test_choose_marks_the_current_option_and_shows_descriptions() -> None:
    menus: list[Menu] = []
    options = [Option(value='a', label='Apple', description='Red or green.'), Option(value='b', label='Banana')]
    chosen = await TerminalDialogs(shows(menus, pick('b'))).choose(
        title='Fruit', options=options, current='b', description='Pick one to eat.'
    )
    assert chosen == 'b'
    [menu] = menus
    assert labels(menu) == ['Apple', 'Banana (current)']
    assert menu._cursor == 1  # pyright: ignore[reportPrivateUsage]
    assert not menu._searchable  # pyright: ignore[reportPrivateUsage]
    assert preview(menu, menu._items[0]) == 'Red or green.\n\nPick one to eat.'  # pyright: ignore[reportPrivateUsage]
    assert preview(menu, menu._items[1]) == 'Pick one to eat.'  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize('result', [CLOSED, MenuResult(item=None)])
async def test_choose_returns_none_when_backed_out(result: MenuResult) -> None:
    menus: list[Menu] = []
    options = [Option(value=str(number), label=str(number)) for number in range(9)]
    assert await TerminalDialogs(shows(menus, result)).choose(title='Number', options=options, current='x') is None
    [menu] = menus
    assert menu._searchable and menu._cursor == 0  # pyright: ignore[reportPrivateUsage]
    assert menu._preview is None, 'nothing to describe'  # pyright: ignore[reportPrivateUsage]


async def test_text_checks_only_non_empty_text_and_masks_secrets() -> None:
    widgets: list[TextInput] = []

    def run_text(widget: TextInput) -> TextInputResult:
        widgets.append(widget)
        return typed('  hunter2  ')

    checked: list[str] = []

    def check(text: str) -> str | None:
        checked.append(text)
        return None if text.startswith('hunter') else 'Not a hunter.'

    typed_text = await TerminalDialogs(Runners(run_text=run_text)).text(
        title='Password',
        label='Token',
        placeholder='Paste it',
        initial='hun',
        secret=True,
        check=check,
        description='Find it at https://example.com.',
    )
    assert typed_text == 'hunter2'
    [widget] = widgets
    assert (widget._prompt, widget._placeholder, widget.text, widget._mask) == ('Token: ', 'Paste it', 'hun', '*')  # pyright: ignore[reportPrivateUsage]
    assert widget._title == 'Password\n\nFind it at https://example.com.'  # pyright: ignore[reportPrivateUsage]
    validator = widget._validator  # pyright: ignore[reportPrivateUsage]
    assert validator is not None
    assert [validator(' x '), validator('   '), validator('hunter')] == ['Not a hunter.', None, None]
    assert checked == ['x', 'hunter']


@pytest.mark.parametrize('result', [TextInputResult(cancelled=True), TextInputResult(value=None)])
async def test_text_returns_none_when_backed_out(result: TextInputResult) -> None:
    widgets: list[TextInput] = []

    def run_text(widget: TextInput) -> TextInputResult:
        widgets.append(widget)
        return result

    assert await TerminalDialogs(Runners(run_text=run_text)).text(title='Name') is None
    [widget] = widgets
    assert (widget._title, widget._prompt, widget._mask, widget._validator) == ('Name', 'Value: ', None, None)  # pyright: ignore[reportPrivateUsage]


async def test_multiline_text_opens_the_editor_until_check_accepts_it() -> None:
    opened: list[tuple[str, str]] = []
    replies = ['  bad\n', ' Two\nlines \n', None]

    def edit(text: str, /, *, title: str) -> str | None:
        opened.append((title, text))
        return replies.pop(0)

    def check(text: str) -> str | None:
        return 'Not that.' if text == 'bad' else None

    terminal = TerminalDialogs(Runners(edit=edit))
    assert await terminal.text(title='Notes', initial='Old', check=check, multiline=True) == 'Two\nlines'
    assert opened == [('Notes', 'Old'), ('Notes: Not that.', 'bad')], 'reopened on what was typed, saying why'
    assert await terminal.text(title='Notes', multiline=True) is None, 'cancelled'


async def test_view_pages_the_text_wrapped_and_disarmed() -> None:
    pagers: list[Pager] = []
    await TerminalDialogs(Runners(run_pager=pagers.append)).view(title='Prompt', text='short\n\n' + 'word ' * 100)
    [pager] = pagers
    assert pager.line_count > 3
    assert build_pager('Prompt', 'x').line_count >= 1
    assert wrap('a\x1b]52;c;x\x07\n\nb', 80) == ['a\\x1b]52;c;x\\x07', '', 'b']


@pytest.mark.parametrize(('result', 'answer'), [(pick(True), True), (pick(False), False), (CLOSED, False)])
async def test_confirm_puts_no_first(result: MenuResult, answer: bool) -> None:
    menus: list[Menu] = []
    confirmed = await TerminalDialogs(shows(menus, result)).confirm(
        title='Replace it?', yes='Replace', no='Keep', description='Others share it.'
    )
    assert confirmed is answer
    [menu] = menus
    assert labels(menu) == ['Keep', 'Replace']
    assert preview(menu, menu._items[0]) == 'Others share it.'  # pyright: ignore[reportPrivateUsage]


async def test_confirm_without_a_description_has_no_preview() -> None:
    menus: list[Menu] = []
    assert await TerminalDialogs(shows(menus, pick(True))).confirm(title='Sure?', yes='Yes', no='No')
    assert menus[0]._preview is None  # pyright: ignore[reportPrivateUsage]


async def test_wait_closes_when_the_task_finishes_and_raises_its_error() -> None:
    script = Script([], [UNTIL_CLOSED, UNTIL_CLOSED], [])

    async def done(show: Callable[[str], None]) -> None:
        show('Code: 1234')

    async def fails(show: Callable[[str], None]) -> None:
        raise ValueError('denied')

    with anyio.fail_after(5):
        assert await TerminalDialogs(script.runners).wait(title='Signing in', description='Finish it.', task=done)
        with pytest.raises(ValueError, match='denied'):
            await TerminalDialogs(script.runners).wait(title='Signing in', description='Finish it.', task=fails)


async def test_wait_raises_when_its_screen_fails_rather_than_reading_as_a_cancel() -> None:
    def broken(menu: Menu) -> MenuResult:
        raise OSError('no terminal')

    async def forever(show: Callable[[str], None]) -> None:
        await anyio.sleep_forever()

    with anyio.fail_after(5), pytest.raises(OSError, match='no terminal'):
        await TerminalDialogs(Runners(run_choice=broken)).wait(
            title='Signing in', description='Finish it.', task=forever
        )


def _on_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


class Fields:
    """One row, `name`, kept in memory."""

    title = 'Sample settings'

    def __init__(self) -> None:
        self.value = 'ada'
        self.read_on_loop: list[bool] = []

    def rows(self) -> Sequence[FieldRow]:
        self.read_on_loop.append(_on_loop())
        return (
            FieldRow(key='name', label='Name', description='Who.', default='ada'),
            FieldRow(key='token', label='Token', description='A /keys entry.', default='(not set)'),
        )

    def current(self, row: FieldRow) -> str:
        return self.value if row.key == 'name' else '(not set)'

    def problem(self, row: FieldRow, text: str) -> str | None:
        return None

    def apply(self, row: FieldRow, raw: str) -> str:
        self.value = raw
        return f'Saved {row.label}.'

    def reset(self, row: FieldRow) -> str:
        self.value = 'ada'
        return f'Reset {row.label}.'


async def test_settings_menu_runs_actions_with_terminal_dialogs_under_one_hold() -> None:
    held: list[str] = []

    @contextmanager
    def hold(*, leave_screen: bool) -> Generator[None]:
        held.append(f'hold leave_screen={leave_screen}')
        yield
        held.append('release')

    asked: list[Dialogs] = []

    async def pick_token(dialogs: Dialogs) -> Sequence[str]:
        asked.append(dialogs)
        return ['Picked a token.'] if await dialogs.confirm(title='Use it?', yes='Use', no='Skip') else []

    fields = Fields()
    menu = SettingsMenu(source=fields, actions={'token': pick_token}, searchable=False)
    script = Script([pick('name'), pick('token'), CLOSED], [pick(True)], [typed('grace')])
    with holding_output(hold):
        messages = await run_settings_menu(menu, script.runners)
    assert messages == ['Saved Name.', 'Picked a token.']
    # The rows may read the keyring or `/keys`, so the terminal menu never reads them on the event loop.
    assert fields.read_on_loop and not any(fields.read_on_loop)
    assert fields.value == 'grace'
    [name, _] = fields.rows()
    assert (fields.problem(name, 'x'), fields.reset(name), fields.value) == (None, 'Reset Name.', 'ada')
    assert [type(dialog) for dialog in asked] == [TerminalDialogs]
    assert script.opened == ['list', 'text', 'list', 'choice', 'list']
    # The outer hold spans the whole menu; each widget's own hold nests inside it.
    assert held[0] == 'hold leave_screen=True' and held[-1] == 'release'
    assert held.count('release') == len(script.opened) + 1


class Settings(BaseModel):
    name: str = 'ada'


class Declared(Plugin[Settings]):
    def settings_menu(self) -> SettingsMenu:
        return SettingsMenu(source=Fields())


class Bare(Plugin[Settings]):
    pass


class Custom(Plugin[Settings]):
    async def configure(self) -> str:
        return 'custom'


def plugin(cls: type[Plugin[Settings]], *, terminal: bool) -> Plugin[BaseModel]:
    console = Console(file=io.StringIO(), force_terminal=terminal)
    return load_plugin(cls, PluginHost[None](name='sample', console=console, settings={})).plugin


async def test_a_declared_menu_is_what_configure_shows(monkeypatch: pytest.MonkeyPatch) -> None:
    declared = plugin(Declared, terminal=True)
    assert declared.has_configure
    monkeypatch.setattr(dialogs, 'RUNNERS', Script([pick('name'), CLOSED], [], [typed('grace')]).runners)
    assert await declared.configure() == 'Saved Name.'
    monkeypatch.setattr(dialogs, 'RUNNERS', Script([CLOSED], [], []).runners)
    assert await declared.configure() == 'Sample settings: nothing changed.'


async def test_a_declared_menu_needs_a_terminal_to_configure_there() -> None:
    assert await plugin(Declared, terminal=False).configure() == (
        'Configure Sample settings from a terminal: /plugins configure sample'
    )


async def test_without_a_menu_there_is_nothing_to_configure() -> None:
    custom, bare = plugin(Custom, terminal=True), plugin(Bare, terminal=True)
    assert custom.has_configure and custom.settings_menu() is None
    assert await custom.configure() == 'custom'
    assert not bare.has_configure
    with pytest.raises(NotImplementedError, match='has no settings menu'):
        await bare.configure()


async def test_scripted_dialogs_refuse_answers_the_terminal_could_not_give() -> None:
    options = [Option(value='a', label='Apple')]
    with pytest.raises(AssertionError, match="no answer left for choose 'Fruit'"):
        await ScriptedDialogs([]).choose(title='Fruit', options=options)
    with pytest.raises(AssertionError, match="cannot answer choose 'Fruit'"):
        await ScriptedDialogs([Type(text='a')]).choose(title='Fruit', options=options)
    with pytest.raises(AssertionError, match="'b' is not offered in 'Fruit'"):
        await ScriptedDialogs([Pick(value='b')]).choose(title='Fruit', options=options)
    with pytest.raises(AssertionError, match=r"'Name' would not accept 'x': Too short\."):
        await ScriptedDialogs([Type(text=' x ')]).text(title='Name', check=lambda text: 'Too short.')


async def test_backing_out_of_a_scripted_wait_cancels_the_task_once_it_shows_something() -> None:
    async def quiet(show: Callable[[str], None]) -> None:
        pass

    async def fails(show: Callable[[str], None]) -> None:
        raise ValueError('denied')

    async def hangs(show: Callable[[str], None]) -> None:
        show('Code: 1234')
        await anyio.sleep_forever()

    dialogs = ScriptedDialogs([Back(), Back(), Back()])
    assert await dialogs.wait(title='Signing in', description='Finish it.', task=quiet), 'it ended first'
    with pytest.raises(ValueError, match='denied'):
        await dialogs.wait(title='Signing in', description='Finish it.', task=fails)
    with anyio.fail_after(5):
        assert not await dialogs.wait(title='Signing in', description='Finish it.', task=hangs)
    assert [asked.shown for asked in dialogs.asked] == [[], [], ['Code: 1234']]
