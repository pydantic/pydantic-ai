"""Scripted widget runners so menus can be driven without a terminal."""

from collections.abc import Iterator
from pathlib import Path

from termflow.tui import MenuItem  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.menu import Menu, MenuResult  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.textinput import TextInput, TextInputResult  # pyright: ignore[reportMissingTypeStubs]

from pydantic_clai2.command_context import CommandContext
from pydantic_clai2.field_menu import Runners
from pydantic_clai2.settings_store import SettingsStore


def make_context(tmp_path: Path) -> tuple[CommandContext, list[str]]:
    applied: list[str] = []
    store = SettingsStore(tmp_path / 'config.db')
    context = CommandContext(
        settings=store.load(),
        store=store,
        clear_history=lambda: None,
        apply_setting=lambda key, settings: applied.append(key),
    )
    return context, applied


def pick(value: object) -> MenuResult:
    return MenuResult(item=MenuItem(str(value), value=value))


def typed(text: str) -> TextInputResult:
    return TextInputResult(value=text)


class Script:
    """Feed scripted results to the runners, in order, and remember what was shown."""

    def __init__(self, lists: list[MenuResult], choices: list[MenuResult], texts: list[TextInputResult]) -> None:
        self._lists: Iterator[MenuResult] = iter(lists)
        self._choices: Iterator[MenuResult] = iter(choices)
        self._texts: Iterator[TextInputResult] = iter(texts)
        self.opened: list[str] = []

    def run_list(self, menu: Menu) -> MenuResult:
        self.opened.append('list')
        return next(self._lists)

    def run_choice(self, menu: Menu) -> MenuResult:
        self.opened.append('choice')
        return next(self._choices)

    def run_text(self, widget: TextInput) -> TextInputResult:
        self.opened.append('text')
        return next(self._texts)

    @property
    def runners(self) -> Runners:
        return Runners(run_list=self.run_list, run_choice=self.run_choice, run_text=self.run_text)
