"""A full-screen editor for a set of named, validated fields. `/set`, `/model add`, and plugin settings use it."""

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Protocol

from pydantic import JsonValue, ValidationError
from termflow.tui import MenuBuilder, MenuItem, TextInputBuilder
from termflow.tui.menu import Menu, MenuResult
from termflow.tui.terminal import terminal_size
from termflow.tui.textinput import TextInput, TextInputResult

from pydantic_clai2.ui import telemetry
from pydantic_clai2.ui.menus.menu_worker import menu_key, run_worker
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering._rendering import markdown_style

CUSTOM = 'Type a value...'
KEEP = 'Keep current'
SAVE_AND_CLOSE = 'Save & close'
SAVE_AND_CLOSE_DETAILS = 'Leave this menu. Each change was saved as you made it.'
_LIST_HINT = 'type to filter - Enter edit - R reset - Esc close'
_PREVIEW_LIST_WIDTH = 30


class _SaveAndClose:
    """The value of the Save & close row; never a field key or a key name."""


_CLOSE = _SaveAndClose()


def save_and_close_item() -> MenuItem:
    """The last row of every settings menu. Edits save as they happen, so choosing it only leaves."""
    return MenuItem(SAVE_AND_CLOSE, value=_CLOSE)


def is_save_and_close(item: MenuItem) -> bool:
    """Whether `item` is the row built by `save_and_close_item`."""
    return item.value is _CLOSE


def picked(result: MenuResult) -> MenuItem | None:
    """The chosen row, or `None` when the user left with Esc or the Save & close row."""
    if result.cancelled or result.item is None or is_save_and_close(result.item):
        return None
    return result.item


@dataclass(frozen=True)
class _Reset:
    """What the `r` key hands back to the loop instead of a row."""

    key: str


class ChoicePreview(Protocol):
    """Render a sample of `choice` for the choice picker's right-hand panel, `width` cells wide."""

    def __call__(self, choice: str, /, *, width: int) -> str: ...


@dataclass(frozen=True, kw_only=True)
class FieldRow:
    """One editable field as the menu sees it."""

    key: str
    description: str
    default: str
    choices: tuple[str, ...] = ()
    label: str = ''
    choice_labels: Mapping[str, str] = field(default_factory=dict[str, str])
    allow_custom: bool = True
    secret: bool = False
    note: str = ''
    """Where the value comes from when not from the user; shown muted after the value."""
    preview: ChoicePreview | None = None
    """Renders a sample of each choice in the choice picker."""

    def display(self, value: str) -> str:
        """Label a choice without changing its stored or validated value."""
        return self.choice_labels.get(value, value)


class FieldSource(Protocol):
    """Where the rows come from and where edits go. Every method is synchronous."""

    @property
    def title(self) -> str:
        """Menu title."""
        ...

    def rows(self) -> Sequence[FieldRow]:
        """Every editable field, in display order."""
        ...

    def current(self, row: FieldRow) -> str:
        """The active value as the user would type it."""
        ...

    def problem(self, row: FieldRow, text: str) -> str | None:
        """Why `text` is not valid for the row, or `None` when it is."""
        ...

    def apply(self, row: FieldRow, raw: str) -> str:
        """Save and apply a non-empty value; return the message to show."""
        ...

    def reset(self, row: FieldRow) -> str:
        """Forget the override; return the message to show."""
        ...


def shown(value: JsonValue) -> str:
    """Display a stored value the way the user would type it back."""
    if value is None:
        return '(not set)'
    return value if isinstance(value, str) else json.dumps(value)


def first_error(exc: ValueError) -> str:
    """The first validation message, which is all a one-line hint has room for."""
    return exc.errors()[0]['msg'] if isinstance(exc, ValidationError) else str(exc)


class FieldMenu:
    """Rows, details, and the widgets that edit them."""

    def __init__(self, source: FieldSource, *, searchable: bool = True) -> None:
        """Everything the menu shows or saves goes through `source`."""
        self._searchable = searchable
        self._source = source
        self.rows = list(source.rows())

    def items(self) -> list[MenuItem]:
        """One row per field with its current value, then Save & close."""
        fields = [
            MenuItem(
                f'{row.label or row.key:<24} {row.display(self._source.current(row))}',
                value=row.key,
                description=f'{theme.sgr(theme.MUTED)}{row.note}' if row.note else '',
            )
            for row in self.rows
        ]
        return [*fields, save_and_close_item()]

    def details(self, item: MenuItem) -> str:
        """The right-hand panel: current value, default, choices, description."""
        if is_save_and_close(item):
            return SAVE_AND_CLOSE_DETAILS
        row = self.row_for(item.value)
        if row is None:
            return ''
        current = self._source.current(row)
        lines = [
            row.label or row.key,
            '',
            f'current  {row.display(current)}' + (' (default)' if current == row.default else ''),
            f'default  {row.display(row.default)}',
        ]
        if row.note:
            lines.append(f'origin   {row.note}')
        if row.choices and len(row.choices) <= 8:
            lines.append(f'choices  {", ".join(row.display(choice) for choice in row.choices)}')
        elif row.choices:
            lines.append(f'choices  {len(row.choices)} options; Enter opens a searchable list')
        lines += ['', row.description]
        return '\n'.join(lines)

    def build(self, initial: int = 0) -> Menu:
        """The field list. Searchable lists use uppercase `R` so typing still filters."""
        self.rows = list(self._source.rows())
        builder = (
            MenuBuilder(self._source.title)
            .style(markdown_style())
            .items(self.items())
            .searchable(self._searchable)
            .initial_index(min(initial, len(self.rows) - 1))
            .preview(self.details)
            .on_key('R' if self._searchable else 'r', self._reset_key)
            .footer_hint(_LIST_HINT if self._searchable else 'Enter edit - r reset - Esc back')
            .key_source(menu_key)
        )
        if not self._searchable:
            builder.list_width(46)
        return builder.build()

    def reset_marker(self, menu: object, item: MenuItem) -> MenuResult:
        """R: hand the row back to the loop tagged for reset."""
        return MenuResult(item=MenuItem(item.label, value=_Reset(str(item.value))))

    def _reset_key(self, menu: object, item: MenuItem) -> MenuResult | None:
        """R on Save & close has nothing to reset, so the list stays open."""
        return None if is_save_and_close(item) else self.reset_marker(menu, item)

    def build_choices(self, row: FieldRow) -> Menu:
        """A picker for fields with a fixed set of values, plus typing your own."""
        current = self._source.current(row)
        items = [
            MenuItem(f'{row.display(choice)}{" (current)" if choice == current else ""}', value=choice)
            for choice in row.choices
        ]
        if row.allow_custom:
            items += [MenuItem(CUSTOM, value=CUSTOM), MenuItem(KEEP, value=KEEP)]
        initial = row.choices.index(current) if current in row.choices else 0
        builder = (
            MenuBuilder(f'Choose {row.label or row.key}')
            .style(markdown_style())
            .items(items)
            .searchable(len(row.choices) > 8)
            .initial_index(initial)
            .footer_hint('Enter select - Esc keep current')
            .key_source(menu_key)
        )
        if row.preview is not None:
            builder.list_width(_PREVIEW_LIST_WIDTH).preview(partial(self._preview, row.preview))
        return builder.build()

    @staticmethod
    def _preview(render: ChoicePreview, item: MenuItem) -> str:
        """Size the sample to the panel; the typed-value and keep rows have none."""
        if not isinstance(item.value, str) or item.value in (CUSTOM, KEEP):
            return ''
        return render(item.value, width=max(20, terminal_size()[0] - _PREVIEW_LIST_WIDTH - 4))

    def build_editor(self, row: FieldRow) -> TextInput:
        """A typed input that validates as you go; empty resets."""
        builder = (
            TextInputBuilder(f'New value for {row.label or row.key}')
            .style(markdown_style())
            .prompt('Value: ')
            .placeholder(
                'Enter a new secret'
                if row.secret
                else f'current: {row.display(self._source.current(row))} (empty resets)'
            )
            .validator(lambda text: None if not text.strip() else self._source.problem(row, text.strip()))
            .footer_hint('Enter save - Esc cancel')
            .key_source(menu_key)
        )
        if row.secret:
            builder.mask()
        return builder.build()

    def apply(self, row: FieldRow, raw: str) -> str:
        """Save and apply, or reset on empty input."""
        raw = raw.strip()
        if not raw:
            return self.reset(row)
        message = self._source.apply(row, raw)
        # Only a listed choice is recorded: typed text can be a secret or a path.
        chosen = {'choice': raw} if raw in row.choices and not row.secret else {}
        telemetry.record('{menu} field {field} set', menu=self._source.title, field=row.key, **chosen)
        return message

    def reset(self, row: FieldRow) -> str:
        """Forget the override and apply the default."""
        message = self._source.reset(row)
        telemetry.record('{menu} field {field} reset', menu=self._source.title, field=row.key)
        return message

    def row_for(self, key: object) -> FieldRow | None:
        """Look a row up by its key."""
        return next((row for row in self.rows if row.key == key), None)


ListRunner = Callable[[Menu], MenuResult]
TextRunner = Callable[[TextInput], TextInputResult]


def run_menu(menu: Menu) -> MenuResult:
    """Show a termflow menu on the real terminal."""
    return menu.run()


def run_text(widget: TextInput) -> TextInputResult:  # pragma: no cover -- needs a real terminal.
    """Show a termflow text input on the real terminal."""
    return widget.run()


@dataclass(frozen=True, kw_only=True)
class Runners:
    """How widgets get shown; tests swap these for scripted results."""

    run_list: ListRunner = run_menu
    run_choice: ListRunner = run_menu
    run_text: TextRunner = run_text


TERMINAL = Runners()
"""The real terminal."""


def run_flow(
    menu: FieldMenu, runners: Runners = TERMINAL, *, submenus: dict[str, Callable[[], list[str]]] | None = None
) -> list[str]:
    """List, edit, back to the list, until Esc or Save & close. Returns the messages to show afterwards."""
    messages: list[str] = []
    cursor = 0
    while True:
        step = _next_step(menu, runners, cursor, messages)
        if step is None:
            return messages
        row, cursor = step
        if row is None:
            continue
        if submenus and row.key in submenus:
            messages.extend(submenus[row.key]())
            continue
        message = _edit(menu, row, runners)
        if message is not None:
            messages.append(message)


async def run_flow_async(
    menu: FieldMenu,
    runners: Runners = TERMINAL,
    *,
    submenus: Mapping[str, Callable[[], Awaitable[list[str]]]] | None = None,
) -> list[str]:
    """`run_flow` driven from the event loop, for submenus that are themselves async, such as `prompt_api_key`.

    Each widget runs in its own `run_worker`, so a submenu can open its own worker without nesting one
    terminal owner inside another.
    """
    messages: list[str] = []
    cursor = 0
    while True:
        step = await run_worker(partial(_next_step, menu, runners, cursor, messages))
        if step is None:
            return messages
        row, cursor = step
        if row is None:
            continue
        if submenus and row.key in submenus:
            messages.extend(await submenus[row.key]())
            continue
        message = await run_worker(partial(_edit, menu, row, runners))
        if message is not None:
            messages.append(message)


def _next_step(
    menu: FieldMenu, runners: Runners, cursor: int, messages: list[str]
) -> tuple[FieldRow | None, int] | None:
    """Show the list and act on the pick in one step, so sources that touch the keyring stay off the event loop."""
    return _selected(menu, runners.run_list(menu.build(cursor)), messages)


def _selected(menu: FieldMenu, result: MenuResult, messages: list[str]) -> tuple[FieldRow | None, int] | None:
    """`None` when the list closed; else the row to edit (or `None` after a reset) and the cursor to return to."""
    item = picked(result)
    if item is None:
        return None
    value = item.value
    reset = isinstance(value, _Reset)
    row = menu.row_for(value.key if isinstance(value, _Reset) else value)
    if row is None:
        return (None, 0) if reset else None
    if reset:
        messages.append(menu.reset(row))
        return None, menu.rows.index(row)
    return row, menu.rows.index(row)


def _edit(menu: FieldMenu, row: FieldRow, runners: Runners) -> str | None:
    if row.choices:
        pick = runners.run_choice(menu.build_choices(row))
        if pick.cancelled or pick.item is None or pick.item.value == KEEP:
            return None
        if isinstance(pick.item.value, str) and pick.item.value != CUSTOM:
            return menu.apply(row, pick.item.value)
    typed = runners.run_text(menu.build_editor(row))
    if typed.cancelled or not isinstance(typed.value, str):
        return None
    return menu.apply(row, typed.value)
