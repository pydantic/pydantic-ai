"""Fallback chains in the `/model` picker: create, edit, rename, and delete them, and `/model chains`."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from termflow.tui.menu import Menu, MenuResult
from termflow.tui.textinput import TextInput, TextInputResult

from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.ui.menus import chain_menu
from pydantic_clai2.ui.menus.chain_menu import chain_details
from pydantic_clai2.ui.menus.field_menu import TERMINAL, Runners
from pydantic_clai2.ui.menus.model_picker import (
    DeleteModel,
    EditChain,
    ModelPickerAction,
    RenameChain,
    build_model_picker,
    model_command,
)
from tests.clai2.menu_script import Script, make_context, pick, typed

ADD = chain_menu._Row.ADD  # pyright: ignore[reportPrivateUsage]
SAVE = chain_menu._Row.SAVE  # pyright: ignore[reportPrivateUsage]
TYPE = chain_menu._Row.TYPE  # pyright: ignore[reportPrivateUsage]
Member = chain_menu._Member  # pyright: ignore[reportPrivateUsage]
Move = chain_menu._Move  # pyright: ignore[reportPrivateUsage]
Remove = chain_menu._Remove  # pyright: ignore[reportPrivateUsage]
CLOSE = MenuResult(cancelled=True)


def keys(monkeypatch: pytest.MonkeyPatch, module: str, pressed: list[str]) -> None:
    """Feed `pressed` to the menus `module` builds, as if typed."""
    sequence: Iterator[str] = iter(pressed)
    monkeypatch.setattr(f'pydantic_clai2.ui.menus.{module}.menu_key', lambda: next(sequence))


def chain_context(tmp_path: Path) -> CommandContext:
    context, _ = make_context(tmp_path)
    context.store.add_model(name='openai:gpt-5')
    context.store.save_chain(name='best', models=['openai:gpt-5', 'anthropic:claude-sonnet-4-5'])
    return context


async def test_create_a_chain_and_select_it_in_the_model_picker(tmp_path: Path) -> None:
    context, applied = make_context(tmp_path)
    context.store.add_model(name='openai:gpt-5')
    script = Script(
        lists=[
            pick(ModelPickerAction.NEW_CHAIN),
            pick(SAVE),  # refused: no models yet
            pick(ADD),
            pick('openai:gpt-5'),
            pick(ADD),
            pick(TYPE),
            pick(SAVE),
            pick('chain:best'),
        ],
        choices=[],
        texts=[typed('best '), typed(' openai@work:gpt-5')],
    )
    assert await model_command(context, [], runners=script.runners) == (
        'Saved chain:best (openai:gpt-5 -> openai@work:gpt-5).\nSaved model. Applied.'
    )
    assert context.store.chains() == {'best': ['openai:gpt-5', 'openai@work:gpt-5']}
    assert context.settings.model == 'chain:best' and applied == ['model']


async def test_leaving_a_new_chain_saves_nothing(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    unnamed = Script(
        lists=[pick(ModelPickerAction.NEW_CHAIN), CLOSE], choices=[], texts=[TextInputResult(cancelled=True)]
    )
    assert await model_command(context, [], runners=unnamed.runners) == 'No changes.'
    # An add that is cancelled, at the list or at the typed name, adds nothing.
    unsaved = Script(
        lists=[pick(ModelPickerAction.NEW_CHAIN), pick(ADD), CLOSE, pick(ADD), pick(TYPE), CLOSE, CLOSE],
        choices=[],
        texts=[typed('draft'), TextInputResult(cancelled=True)],
    )
    assert await model_command(context, [], runners=unsaved.runners) == 'No changes.'
    assert context.store.chains() == {}


async def test_edit_a_chains_models_and_their_order(tmp_path: Path) -> None:
    context = chain_context(tmp_path)
    context.store.save_chain(name='best', models=['a:1', 'b:2', 'c:3'])
    script = Script(
        lists=[pick(EditChain(name='best')), pick(Move(0, 1)), pick(Remove(2)), pick(Member(0)), pick(SAVE), CLOSE],
        choices=[],
        texts=[],
    )
    assert await model_command(context, [], runners=script.runners) == 'Saved chain:best (b:2 -> a:1).'
    assert context.store.chains()['best'] == ['b:2', 'a:1']
    # Esc leaves the editor without saving its changes.
    discarded = Script(
        lists=[pick(EditChain(name='best')), pick(Remove(0)), pick(Remove(0)), CLOSE, CLOSE], choices=[], texts=[]
    )
    assert await model_command(context, [], runners=discarded.runners) == 'No changes.'
    assert context.store.chains()['best'] == ['b:2', 'a:1']


async def test_rename_a_chain_with_its_settings(tmp_path: Path) -> None:
    context = chain_context(tmp_path)
    context.store.save_model_settings('chain:best', {'max_tokens': 10})
    script = Script(lists=[pick(RenameChain(name='best')), CLOSE], choices=[], texts=[typed('top')])
    assert await model_command(context, [], runners=script.runners) == 'Renamed chain:best to chain:top.'
    assert context.store.chains() == {'top': ['openai:gpt-5', 'anthropic:claude-sonnet-4-5']}
    assert 'chain:top' in context.store.models() and 'chain:best' not in context.store.models()
    assert context.store.model_settings('chain:top') == {'max_tokens': 10}
    assert context.store.model_settings('chain:best') == {}
    unchanged = Script(lists=[pick(RenameChain(name='top')), CLOSE], choices=[], texts=[typed('top')])
    assert await model_command(context, [], runners=unchanged.runners) == 'No changes.'


async def test_a_chain_in_use_is_not_renamed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    context = chain_context(tmp_path)
    await model_command(context, ['chain:best'])
    keys(monkeypatch, 'model_picker', ['ctrl-r', 'escape'])
    assert await model_command(context, []) == 'No changes.'
    assert 'Select another model before renaming' in capsys.readouterr().out

    # It also stays when another session makes it the saved default while the name is typed.
    await model_command(context, ['openai:gpt-5'])

    def rename_after_default_changes(widget: TextInput) -> TextInputResult:
        context.store.set('model', 'chain:best')
        return typed('top')

    script = Script(lists=[pick(RenameChain(name='best')), CLOSE], choices=[], texts=[])
    runners = Runners(run_list=script.run_list, run_choice=script.run_choice, run_text=rename_after_default_changes)
    assert await model_command(context, [], runners=runners) == 'No changes.'
    assert 'best' in context.store.chains()


async def test_delete_a_chain_from_the_model_picker(tmp_path: Path) -> None:
    context = chain_context(tmp_path)
    script = Script(lists=[pick(DeleteModel(name='chain:best')), CLOSE], choices=[pick(True)], texts=[])
    assert await model_command(context, [], runners=script.runners) == 'Deleted chain:best.'
    assert context.store.chains() == {} and 'chain:best' not in context.store.models()


async def test_model_chains_opens_the_picker_on_its_chains(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    highlighted: list[object] = []

    def run_list(menu: Menu) -> MenuResult:
        highlighted.append(menu.highlighted.value if menu.highlighted is not None else None)
        return CLOSE

    runners = Runners(run_list=run_list, run_choice=TERMINAL.run_choice, run_text=TERMINAL.run_text)
    assert await model_command(context, ['chains'], runners=runners) == 'No changes.'
    context.store.save_chain(name='best', models=['openai:gpt-5', 'openai@work:gpt-5'])
    assert await model_command(context, ['chains'], runners=runners) == 'No changes.'
    assert highlighted == [ModelPickerAction.NEW_CHAIN, 'chain:best']
    with pytest.raises(ValueError, match=r'^Usage: /model chains\. Create, edit, rename, and delete'):
        await model_command(context, ['chains', 'best'])


def test_picker_lists_chains_after_models_with_their_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    context = chain_context(tmp_path)
    keys(monkeypatch, 'model_picker', ['escape'])
    assert build_model_picker(context).run().cancelled
    shown = capsys.readouterr().out
    rows = ['openai:gpt-5', 'Add a model...', 'Fallback chains', 'chain:best', 'New fallback chain...']
    assert [shown.index(row) for row in rows] == sorted(shown.index(row) for row in rows)
    for key, action in [('ctrl-e', EditChain), ('ctrl-r', RenameChain)]:
        # The key does nothing on a model that is not a chain, then acts on the chain.
        keys(monkeypatch, 'model_picker', [key, *'best', key])
        result = build_model_picker(context).run()
        assert result.item is not None and result.item.value == action(name='best')
    shown = capsys.readouterr().out
    assert all(line in shown for line in ['Tries in order:', '1. openai:gpt-5', '2. anthropic:claude-sonnet-4-5'])
    keys(monkeypatch, 'model_picker', ['escape'])
    assert build_model_picker(context, focus=ModelPickerAction.NEW_CHAIN).run().cancelled
    assert 'Create a fallback chain' in capsys.readouterr().out


def test_chain_editor_keys_and_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    build = chain_menu._build_editor  # pyright: ignore[reportPrivateUsage]
    models = ['a:1', 'b:2']
    expected: list[tuple[int, list[str], object]] = [
        (0, [']'], Move(0, 1)),
        (1, ['['], Move(1, -1)),
        (1, ['d'], Remove(1)),
        (2, ['a'], ADD),
        # Keys that cannot apply to a row do nothing: the first model up, the last down, or a row that is not a model.
        (0, ['[', 'escape'], None),
        (1, [']', 'escape'], None),
        (2, ['d', ']', 'escape'], None),
    ]
    for cursor, pressed, value in expected:
        keys(monkeypatch, 'chain_menu', pressed)
        result = build('best', models, cursor=cursor, notice='').run()
        assert (None if result.cancelled or result.item is None else result.item.value) == value
    keys(monkeypatch, 'chain_menu', ['down', 'down', 'escape'])
    assert build('best', models, cursor=1, notice='A chain needs at least two models.').run().cancelled
    shown = capsys.readouterr().out
    for text in [
        'Tried number 2 of 2.',
        'or type any PROVIDER[@PROFILE]:MODEL',
        'Save chain:best.',
        'at least two models',
    ]:
        assert text in shown


def test_chain_names_and_typed_models_are_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = chain_context(tmp_path)
    ask_name = chain_menu._ask_name  # pyright: ignore[reportPrivateUsage]
    # Taken, then invalid, then a free name: each refused one stays on screen until it is fixed.
    keys(
        monkeypatch,
        'chain_menu',
        [*'best', 'enter', *['backspace'] * 4, *'Bad!', 'enter', *['backspace'] * 4, *'top', 'enter'],
    )
    assert ask_name(context, TERMINAL, title='Name the new fallback chain', initial='') == 'top'
    # The chain's own name is not taken when renaming it.
    keys(monkeypatch, 'chain_menu', ['enter'])
    assert ask_name(context, TERMINAL, title='Rename chain:best', initial='best') == 'best'

    pick_member = chain_menu._pick_member  # pyright: ignore[reportPrivateUsage]
    keys(monkeypatch, 'chain_menu', ['enter'])
    # Saved models already in the chain are not offered.
    assert pick_member(context, [str(context.settings.model)], TERMINAL) == 'openai:gpt-5'
    # Nor are saved models a chain cannot run, such as one without a provider: only typing is left.
    context.store.add_model(name='test')
    keys(monkeypatch, 'chain_menu', ['enter', 'escape'])
    assert pick_member(context, [str(context.settings.model), 'openai:gpt-5'], TERMINAL) is None
    pressed = ['end', 'enter']
    for refused in ['openai:gpt-5', 'chain:best', 'bare']:
        pressed += [*refused, 'enter', *['backspace'] * len(refused)]
    keys(monkeypatch, 'chain_menu', [*pressed, *'openai@*:gpt-5', 'enter'])
    assert pick_member(context, ['openai:gpt-5'], TERMINAL) == 'openai@*:gpt-5'


def test_store_renames_a_chain_unless_it_is_the_saved_default(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    store.save_chain(name='pool', models=['openai:gpt-5', 'openai@work:gpt-5'])
    store.set('model', 'chain:pool')
    assert not store.rename_chain(old='pool', new='top')
    assert 'pool' in store.chains()
    store.set('model', 'openai:gpt-5')
    # A name another session took after it was checked is refused, not overwritten.
    store.save_chain(name='taken', models=['openai:gpt-5', 'anthropic:claude-sonnet-4-5'])
    with pytest.raises(ValueError, match=r'^chain:taken already exists\.$'):
        store.rename_chain(old='pool', new='taken')
    store.add_model(name='chain:orphan')  # a `/model` entry left by an older build, with no chain behind it
    with pytest.raises(ValueError, match='chain:orphan already exists'):
        store.rename_chain(old='pool', new='orphan')
    assert store.rename_chain(old='pool', new='top')
    assert store.chains()['top'] == ['openai:gpt-5', 'openai@work:gpt-5'] and 'pool' not in store.chains()
    # Deleting it from the `/model` list removes the chain too, unless it is the saved default.
    store.set('model', 'chain:top')
    assert not store.remove_model(name='chain:top')
    store.set('model', 'openai:gpt-5')
    assert store.remove_model(name='chain:top')
    assert 'top' not in store.chains()


def test_chain_details() -> None:
    assert chain_details(['a:1', 'b:2']) == ['Tries in order:', '1. a:1', '2. b:2']
    assert chain_details(None) == ['Its models could not be read.']
