"""The `/plugins` full-screen menu, built on termflow like Code Puppy's menus."""

import asyncio
import textwrap
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol

from termflow.tui import MenuBuilder, MenuItem
from termflow.tui.menu import Menu, MenuResult
from termflow.tui.terminal import terminal_size

from pydantic_clai2.plugins import DepsT
from pydantic_clai2.plugins.describe import describe
from pydantic_clai2.plugins.loader import TURN_NOTICE, PluginEntry, PluginError, PluginLoader
from pydantic_clai2.ui.menus.field_menu import SAVE_AND_CLOSE_DETAILS, is_save_and_close, save_and_close_item
from pydantic_clai2.ui.menus.menu_worker import menu_key, run_worker
from pydantic_clai2.ui.menus.slash_search import slash_search
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering._rendering import markdown_style

Apply = Callable[[Coroutine[object, object, object]], None]
_HINT = 'space on/off · c configure · r reload · d remove · esc close'
_RESET = '\x1b[0m'
_UNDIM = '\x1b[22m'
"""Termflow dims row descriptions; the status word cancels that so its colour reads clearly."""
_STATUS_WIDTH = len('failed')


def _status(entry: PluginEntry[DepsT]) -> tuple[str, str]:
    """A one-word status and the theme role that colours it."""
    if entry.loaded is not None:
        return 'on', theme.SUCCESS
    if not entry.declaration.enabled:
        return 'off', theme.MUTED
    return ('failed', theme.ERROR) if entry.error else ('idle', theme.WARNING)


def _origin(entry: PluginEntry[DepsT]) -> str:
    """Where the declaration came from, short enough for the list."""
    if entry.project:
        return 'project'
    if entry.builtin:
        return 'built-in'
    return 'drop-in' if entry.path is not None else 'installed'


def _paint(role: str, text: str, *, bold: bool = False) -> str:
    return f'{theme.sgr(role, bold=bold)}{text}{_RESET}'


@dataclass(frozen=True)
class Configure:
    """What the menu hands back to open a plugin's settings menu, which needs the screen to itself."""

    name: str


class Redrawable(Protocol):
    """The part of a termflow menu a key handler needs."""

    def replace_items(self, items: Sequence[MenuItem]) -> None:
        """Redraw with new rows."""
        ...


class PluginMenu(Generic[DepsT]):
    """Rows, details, and key actions; `build()` wires them into a termflow menu."""

    def __init__(self, loader: PluginLoader[DepsT], *, apply: Apply) -> None:
        """`apply` runs a loader coroutine to completion from the menu's thread."""
        self._loader = loader
        self._apply = apply
        self.notice: str | None = None
        self._descriptions: dict[str, str] = {}
        """Read once per plugin, since the panel repaints on every key; reload reads again."""

    def items(self) -> list[MenuItem]:
        """One row per plugin, `●` when loaded, with a coloured status and origin; then Save & close.

        A plugin another loaded plugin includes is a greyed-out row naming that plugin.
        """
        entries = self._loader.entries()
        if not entries:
            hint = f'No plugins. Use /plugins add, or drop a file in {self._loader.plugins_dir}'
            return [MenuItem(hint, disabled=True), save_and_close_item()]
        width = self._name_width()
        rows: list[MenuItem] = []
        for entry in entries:
            if entry.included_in is not None:
                # Greyed out and skipped by the cursor: the plugin that includes it already provides it.
                label = f'○ {entry.name:<{width}}  {"off":<{_STATUS_WIDTH}} in {entry.included_in}'
                rows.append(MenuItem(label, value=entry.name, disabled=True))
                continue
            word, role = _status(entry)
            status = f'{_UNDIM}{theme.sgr(role)}{word:<{_STATUS_WIDTH}}{_RESET}'
            rows.append(
                MenuItem(
                    f'{"●" if entry.loaded else "○"} {entry.name:<{width}}',
                    value=entry.name,
                    description=f'{status} {_paint(theme.MUTED, _origin(entry))}',
                )
            )
        return [*rows, save_and_close_item()]

    def details(self, item: MenuItem) -> str:
        """The right-hand panel for the highlighted row."""
        entry = self._find(item)
        if entry is None:
            done = SAVE_AND_CLOSE_DETAILS if is_save_and_close(item) else ''
            return '\n'.join(line for line in (self._notice(), done) if line)
        word, role = _status(entry)
        lines = [
            _paint(theme.ACCENT, entry.name),
            f'{_paint(role, word)}{_paint(theme.MUTED, f" · {_origin(entry)}")}',
            '',
            *self._description(entry),
            self._field('source', entry.declaration.factory if entry.path is None else str(entry.path)),
            self._field('provides', entry.loaded.summary() if entry.loaded else 'nothing while off'),
            self._field('settings', self._settings_hint(entry)),
        ]
        if entry.ignored:
            lines += ['', _paint(theme.WARNING, 'notice', bold=True), *self._wrap(entry.ignored, theme.WARNING)]
        if entry.error:
            lines += ['', _paint(theme.ERROR, 'error', bold=True), *self._wrap(entry.error, theme.ERROR)]
        if self.notice:
            lines += ['', self._notice()]
        if self._loader.in_turn:
            lines += ['', *self._wrap(TURN_NOTICE, theme.MUTED)]
        return '\n'.join(lines)

    def toggle(self, menu: Redrawable, item: MenuItem) -> MenuResult | None:
        """Space: enable or disable, saved immediately. Enabling opens the plugin's settings menu, if it has one."""
        entry = self._find(item)
        if entry is not None:
            enabling = entry.loaded is None
            self._run((self._loader.enable if enabling else self._loader.disable)(entry.name))
            if enabling and self.notice is None and self._loader.configurable(entry.name):
                return MenuResult(item=MenuItem(item.label, value=Configure(entry.name)))
        menu.replace_items(self.items())
        return None

    def reload(self, menu: Redrawable, item: MenuItem) -> None:
        """R: re-import and load again."""
        entry = self._find(item)
        if entry is not None:
            self._descriptions.pop(entry.name, None)
            self._run(self._loader.reload(entry.name))
        menu.replace_items(self.items())

    def remove(self, menu: Redrawable, item: MenuItem) -> None:
        """D: unload and forget."""
        entry = self._find(item)
        if entry is not None:
            self._descriptions.pop(entry.name, None)
            self._run(self._loader.remove(entry.name))
        menu.replace_items(self.items())

    def configure(self, menu: Redrawable, item: MenuItem) -> MenuResult | None:
        """C: open the highlighted plugin's settings menu; stay here when it has none."""
        entry = self._find(item)
        if entry is None:
            return None
        if self._loader.configurable(entry.name):
            return MenuResult(item=MenuItem(item.label, value=Configure(entry.name)))
        self.notice = f'{entry.name} has no settings menu.' if entry.loaded else f'Enable {entry.name} to configure it.'
        menu.replace_items(self.items())
        return None

    def close(self, menu: Redrawable, item: MenuItem) -> MenuResult:
        """Q: close; every change was already applied."""
        return MenuResult(item=item)

    def build(self) -> Menu:
        """Wire rows, details, and keys into a termflow menu."""
        builder = (
            MenuBuilder('Plugins')
            .style(markdown_style())
            .items(self.items())
            .list_width(self._list_width())
            .preview(self.details)
        )
        hotkeys = {' ': self.toggle, 'c': self.configure, 'r': self.reload, 'd': self.remove, 'q': self.close}
        return slash_search(builder, footer=_HINT, key_source=menu_key, hotkeys=hotkeys)

    def _name_width(self) -> int:
        return max((len(entry.name) for entry in self._loader.entries()), default=0)

    def _list_width(self) -> int:
        """Pointer, glyph, name, status, and origin, with a little air before the divider."""
        return max(30, 2 + 2 + self._name_width() + 2 + _STATUS_WIDTH + 1 + 10)

    def _wrap(self, text: str, role: str) -> list[str]:
        """Wrap long text to the details pane so errors and notices are read, not clipped."""
        width = max(20, terminal_size()[0] - 1 - self._list_width() - 3)
        return [_paint(role, line) for line in textwrap.wrap(text, width)]

    def _description(self, entry: PluginEntry[DepsT]) -> list[str]:
        """The plugin's docstring summary and a blank line, or nothing when it has none."""
        if entry.name not in self._descriptions:
            self._descriptions[entry.name] = describe(entry)
        text = self._descriptions[entry.name]
        return [*self._wrap(text, theme.SUGAR), ''] if text else []

    def _notice(self) -> str:
        return '\n'.join(self._wrap(self.notice, theme.WARNING)) if self.notice else ''

    def _settings_hint(self, entry: PluginEntry[DepsT]) -> str:
        if self._loader.configurable(entry.name):
            return 'press c to configure'
        return 'none' if entry.loaded else 'turn on to see'

    @staticmethod
    def _field(label: str, value: str) -> str:
        return f'{_paint(theme.MUTED, f"{label:<9}")}{value}'

    def _find(self, item: MenuItem) -> PluginEntry[DepsT] | None:
        name = item.value
        if not isinstance(name, str):
            return None
        return next((entry for entry in self._loader.entries() if entry.name == name), None)

    def _run(self, action: Coroutine[object, object, object]) -> None:
        self.notice = None
        try:
            self._apply(action)
        except (PluginError, ValueError) as exc:
            self.notice = str(exc)


async def open_plugins_menu(
    loader: PluginLoader[DepsT], *, run: Callable[[PluginMenu[DepsT]], MenuResult] | None = None
) -> str:
    """Show the menu in a thread; key actions hop back to the event loop to apply.

    A plugin's settings menu opens once this one has closed, and this one reopens after it,
    showing the settings menu's message. Returns those messages.
    """
    loop = asyncio.get_running_loop()

    def apply(action: Coroutine[object, object, object]) -> None:
        asyncio.run_coroutine_threadsafe(action, loop).result()

    menu = PluginMenu(loader, apply=apply)
    messages: list[str] = []
    while True:
        result = await run_worker(lambda: (run or _run_menu)(menu))
        chosen = result.item.value if result.item is not None else None
        if not isinstance(chosen, Configure):
            return '\n'.join(messages)
        try:
            menu.notice = await loader.configure(chosen.name)
        except (PluginError, ValueError) as exc:
            menu.notice = str(exc)
        messages.append(menu.notice)


def _run_menu(menu: PluginMenu[DepsT]) -> MenuResult:  # pragma: no cover -- needs a real terminal.
    return menu.build().run()
