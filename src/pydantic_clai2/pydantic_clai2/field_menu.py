"""A full-screen editor for a set of named, validated fields. `/set` and `/model` both use it."""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic import JsonValue, ValidationError
from termflow.tui import MenuBuilder, MenuItem, TextInputBuilder  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.menu import Menu, MenuResult  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.textinput import TextInput, TextInputResult  # pyright: ignore[reportMissingTypeStubs]

from . import theme
from ._rendering import markdown_style
from .menu_worker import menu_key

CUSTOM = 'Type a value...'
KEEP = 'Keep current'
_LIST_HINT = 'type to filter - Enter edit - R reset - Esc close'


@dataclass(frozen=True)
class _Reset:
    """What the `r` key hands back to the loop instead of a row."""

    key: str


@dataclass(frozen=True, kw_only=True)
class FieldRow:
    """One editable field as the menu sees it."""

    key: str
    description: str
    default: str
    choices: tuple[str, ...] = ()
    note: str = ''
    """Where the value comes from when not from the user; shown muted after the value."""


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


def first_error(exc: ValidationError) -> str:
    """The first validation message, which is all a one-line hint has room for."""
    return exc.errors()[0]['msg']


class FieldMenu:
    """Rows, details, and the widgets that edit them."""

    def __init__(self, source: FieldSource) -> None:
        """Everything the menu shows or saves goes through `source`."""
        self._source = source
        self.rows = list(source.rows())

    def items(self) -> list[MenuItem]:
        """One row per field with its current value."""
        return [
            MenuItem(
                f'{row.key:<24} {self._source.current(row)}',
                value=row.key,
                description=f'{theme.sgr(theme.MUTED)}{row.note}' if row.note else '',
            )
            for row in self.rows
        ]

    def details(self, item: MenuItem) -> str:
        """The right-hand panel: current value, default, choices, description."""
        row = self.row_for(item.value)
        if row is None:
            return ''
        current = self._source.current(row)
        lines = [
            row.key,
            '',
            f'current  {current}' + (' (default)' if current == row.default else ''),
            f'default  {row.default}',
        ]
        if row.note:
            lines.append(f'origin   {row.note}')
        if row.choices and len(row.choices) <= 8:
            lines.append(f'choices  {", ".join(row.choices)}')
        elif row.choices:
            lines.append(f'choices  {len(row.choices)} options; Enter opens a searchable list')
        lines += ['', row.description]
        return '\n'.join(lines)

    def build(self, initial: int = 0) -> Menu:
        """The field list. `r` returns a reset marker instead of a row."""
        return (
            MenuBuilder(self._source.title)
            .style(markdown_style())
            .items(self.items())
            .searchable()
            .initial_index(min(initial, len(self.rows) - 1))
            .preview(self.details)
            .on_key('r', self.reset_marker)
            .footer_hint(_LIST_HINT)
            .key_source(menu_key)
            .build()
        )

    def reset_marker(self, menu: object, item: MenuItem) -> MenuResult:
        """R: hand the row back to the loop tagged for reset."""
        return MenuResult(item=MenuItem(item.label, value=_Reset(str(item.value))))

    def build_choices(self, row: FieldRow) -> Menu:
        """A picker for fields with a fixed set of values, plus typing your own."""
        current = self._source.current(row)
        items = [
            MenuItem(f'{choice}{" (current)" if choice == current else ""}', value=choice) for choice in row.choices
        ]
        items += [MenuItem(CUSTOM, value=CUSTOM), MenuItem(KEEP, value=KEEP)]
        initial = row.choices.index(current) if current in row.choices else 0
        return (
            MenuBuilder(f'Choose {row.key}')
            .style(markdown_style())
            .items(items)
            .searchable(len(row.choices) > 8)
            .initial_index(initial)
            .footer_hint('Enter select - Esc keep current')
            .key_source(menu_key)
            .build()
        )

    def build_editor(self, row: FieldRow) -> TextInput:
        """A typed input that validates as you go; empty resets."""
        return (
            TextInputBuilder(f'New value for {row.key}')
            .style(markdown_style())
            .prompt('Value: ')
            .placeholder(f'current: {self._source.current(row)} (empty resets)')
            .validator(lambda text: None if not text.strip() else self._source.problem(row, text.strip()))
            .footer_hint('Enter save - Esc cancel')
            .key_source(menu_key)
            .build()
        )

    def apply(self, row: FieldRow, raw: str) -> str:
        """Save and apply, or reset on empty input."""
        raw = raw.strip()
        return self._source.apply(row, raw) if raw else self._source.reset(row)

    def reset(self, row: FieldRow) -> str:
        """Forget the override and apply the default."""
        return self._source.reset(row)

    def row_for(self, key: object) -> FieldRow | None:
        """Look a row up by its key."""
        return next((row for row in self.rows if row.key == key), None)


ListRunner = Callable[[Menu], MenuResult]
TextRunner = Callable[[TextInput], TextInputResult]


def run_menu(menu: Menu) -> MenuResult:  # pragma: no cover -- needs a real terminal.
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


def run_flow(menu: FieldMenu, runners: Runners = TERMINAL) -> list[str]:
    """List, edit, back to the list, until Esc. Returns the messages to show afterwards."""
    messages: list[str] = []
    cursor = 0
    while True:
        result = runners.run_list(menu.build(cursor))
        if result.cancelled or result.item is None:
            return messages
        value = result.item.value
        if isinstance(value, _Reset):
            row = menu.row_for(value.key)
            if row is not None:
                cursor = menu.rows.index(row)
                messages.append(menu.reset(row))
            continue
        row = menu.row_for(value)
        if row is None:
            return messages
        cursor = menu.rows.index(row)
        message = _edit(menu, row, runners)
        if message is not None:
            messages.append(message)


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
