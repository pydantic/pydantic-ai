"""Fallback chains in the `/model` picker: create one, change its models and their order, or rename it.

A chain is listed as `chain:NAME` beside the other models, and Enter selects it like any of them.
The editors here run on the picker's worker thread and return to it when they close.
"""

from dataclasses import dataclass
from enum import Enum

from termflow.tui import MenuBuilder, MenuItem, TextInputBuilder
from termflow.tui.menu import Menu, MenuResult

from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.models.chains import check_member
from pydantic_clai2.models.profiles import check_name
from pydantic_clai2.ui.menus.field_menu import Runners
from pydantic_clai2.ui.menus.menu_worker import menu_key
from pydantic_clai2.ui.menus.slash_search import KeyHandler, slash_search
from pydantic_clai2.ui.rendering._rendering import markdown_style

_HINT = 'a add · d remove · [ ] reorder · esc discard'


@dataclass(frozen=True, kw_only=True)
class ChainChange:
    """A saved change: the chain's name now, and what to report."""

    name: str
    message: str


class _Row(Enum):
    ADD = 'add'
    SAVE = 'save'
    TYPE = 'type'


@dataclass(frozen=True)
class _Member:
    index: int


@dataclass(frozen=True)
class _Remove:
    index: int


@dataclass(frozen=True)
class _Move:
    index: int
    offset: int


def chain_details(models: list[str] | None) -> list[str]:
    """The `/model` picker's lines for a chain: its models in the order they are tried."""
    if not models:
        return ['Its models could not be read.']
    return ['Tries in order:', *(f'{number}. {model}' for number, model in enumerate(models, start=1))]


def edit_chain(context: CommandContext, name: str | None, runners: Runners) -> ChainChange | None:
    """Create a chain (`name` is `None`) or change an existing one's models; `None` when left unsaved."""
    if name is None:
        name = _ask_name(context, runners, title='Name the new fallback chain', initial='')
        if name is None:
            return None
    models = list(context.store.chains().get(name, []))
    cursor = 0
    notice = ''
    while True:
        result = runners.run_list(_build_editor(name, models, cursor=cursor, notice=notice))
        notice = ''
        value = None if result.cancelled or result.item is None else result.item.value
        if value is None:
            return None
        if isinstance(value, _Move):
            models.insert(value.index + value.offset, models.pop(value.index))
            cursor = value.index + value.offset
        elif isinstance(value, _Remove):
            models.pop(value.index)
            cursor = min(value.index, max(len(models) - 1, 0))
        elif value is _Row.ADD:
            cursor = len(models)  # back on the add row, or on the model it added
            if (picked := _pick_member(context, models, runners)) is not None:
                models.append(picked)
        elif value is _Row.SAVE:
            if len(models) < 2:
                notice = 'A chain needs at least two models.'
                cursor = len(models)
                continue
            context.store.save_chain(name=name, models=models)
            return ChainChange(name=name, message=f'Saved chain:{name} ({" -> ".join(models)}).')
        else:
            # Enter on a model does nothing but keep the cursor on it.
            assert isinstance(value, _Member)
            cursor = value.index


def rename_chain(context: CommandContext, name: str, runners: Runners) -> ChainChange | None:
    """Ask for a new name and rename the chain; `None` when cancelled or unchanged."""
    new = _ask_name(context, runners, title=f'Rename chain:{name}', initial=name)
    if new is None or new == name:
        return None
    if not context.store.rename_chain(old=name, new=new):
        raise ValueError(f'Kept chain:{name}: it became the saved default.')
    return ChainChange(name=new, message=f'Renamed chain:{name} to chain:{new}.')


def _ask_name(context: CommandContext, runners: Runners, *, title: str, initial: str) -> str | None:
    taken = set(context.store.chains()) - {initial}
    typed = runners.run_text(
        TextInputBuilder(title)
        .style(markdown_style())
        .prompt('Name: ')
        .initial(initial)
        .validator(lambda text: _name_problem(text.strip(), taken))
        .footer_hint('Enter save name - Esc back')
        .key_source(menu_key)
        .build()
    )
    return None if typed.cancelled or typed.value is None else typed.value.strip()


def _name_problem(name: str, taken: set[str]) -> str | None:
    if name in taken:
        return f'chain:{name} already exists.'
    try:
        check_name(name, kind='Chain name')
    except ValueError as exc:
        return str(exc)
    return None


def _member_problem(model: str, models: list[str]) -> str | None:
    if model in models:
        return f'{model} is already in this chain.'
    try:
        check_member(model)
    except ValueError as exc:
        return str(exc)
    return None


def _editor_details(name: str, models: list[str], item: MenuItem) -> str:
    if isinstance(item.value, _Member):
        return f'{models[item.value.index]}\n\nTried number {item.value.index + 1} of {len(models)}.'
    if item.value is _Row.ADD:
        return 'Add a saved model,\nor type any PROVIDER[@PROFILE]:MODEL,\nsuch as PROVIDER@*:MODEL.'
    return f'Save chain:{name}.\nA chain needs at least two models.'


def _build_editor(name: str, models: list[str], *, cursor: int, notice: str) -> Menu:
    rows = [MenuItem(f'{number}. {model}', value=_Member(number - 1)) for number, model in enumerate(models, start=1)]
    rows += [MenuItem('+ Add a model...', value=_Row.ADD), MenuItem('Save chain', value=_Row.SAVE)]
    if notice:
        rows.append(MenuItem(notice, disabled=True))

    def member(item: MenuItem) -> int | None:
        return item.value.index if isinstance(item.value, _Member) else None

    def move(offset: int) -> KeyHandler:
        def handler(menu: Menu, item: MenuItem) -> MenuResult | None:
            index = member(item)
            if index is None or not 0 <= index + offset < len(models):
                return None
            return MenuResult(item=MenuItem('move', value=_Move(index, offset)))

        return handler

    def add(menu: Menu, item: MenuItem) -> MenuResult:
        return MenuResult(item=MenuItem('add', value=_Row.ADD))

    def remove(menu: Menu, item: MenuItem) -> MenuResult | None:
        index = member(item)
        return None if index is None else MenuResult(item=MenuItem('remove', value=_Remove(index)))

    builder = (
        MenuBuilder(f'chain:{name} - models in the order they are tried')
        .style(markdown_style())
        .items(rows)
        .initial_index(cursor)
        .preview(lambda item: _editor_details(name, models, item))
    )

    hotkeys = {'a': add, 'd': remove, '[': move(-1), ']': move(1)}
    return slash_search(builder, footer=_HINT, key_source=menu_key, hotkeys=hotkeys)


def _pick_member(context: CommandContext, models: list[str], runners: Runners) -> str | None:
    # Only models a chain can run: provider-qualified, installed, not chains, and not in it already.
    saved = [model for model in context.store.models() if _member_problem(model, models) is None]
    rows = [MenuItem(model, value=model) for model in saved]
    picked = runners.run_list(
        MenuBuilder('Add a model to the chain')
        .style(markdown_style())
        .items([*rows, MenuItem('Type a model name...', value=_Row.TYPE)])
        .searchable()
        .preview(lambda item: 'Saved models come from /model add.' if item.value is _Row.TYPE else str(item.value))
        .footer_hint('type to filter - Enter add - Esc back')
        .key_source(menu_key)
        .build()
    )
    value = None if picked.cancelled or picked.item is None else picked.item.value
    if value is not _Row.TYPE:
        return value if isinstance(value, str) else None

    typed = runners.run_text(
        TextInputBuilder('Add a model to the chain')
        .style(markdown_style())
        .prompt('Model: ')
        .validator(lambda text: _member_problem(text.strip(), models))
        .footer_hint('Enter add - Esc back')
        .key_source(menu_key)
        .build()
    )
    return None if typed.cancelled or typed.value is None else typed.value.strip()
