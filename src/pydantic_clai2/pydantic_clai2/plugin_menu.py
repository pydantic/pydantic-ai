"""The `/plugins` full-screen menu, built on termflow like Code Puppy's menus."""

import asyncio
from collections.abc import Callable, Coroutine, Sequence
from typing import Generic, Protocol

from termflow.tui import MenuBuilder, MenuItem  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.menu import Menu, MenuResult  # pyright: ignore[reportMissingTypeStubs]

from ._rendering import markdown_style
from .menu_worker import menu_key, run_worker
from .plugin_loader import PluginEntry, PluginError, PluginLoader
from .plugins import DepsT

Apply = Callable[[Coroutine[object, object, object]], None]
_HINT = 'Up/Down move - Space enable/disable - R reload - D remove - Enter/Q close'


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

    def items(self) -> list[MenuItem]:
        """One row per plugin, `[x]` when loaded."""
        entries = self._loader.entries()
        if not entries:
            hint = f'No plugins. Use /plugins add, or drop a file in {self._loader.plugins_dir}'
            return [MenuItem(hint, disabled=True)]
        return [
            MenuItem(f'[{"x" if entry.host else " "}] {entry.name:<16} {entry.source}', value=entry.name)
            for entry in entries
        ]

    def details(self, item: MenuItem) -> str:
        """The right-hand panel for the highlighted row."""
        entry = self._find(item)
        if entry is None:
            return self.notice or ''
        lines = [
            f'source  {entry.source}',
            f'state   {entry.state}',
            f'adds    {entry.host.summary() if entry.host else "-"}',
            f'error   {entry.error or "none"}',
        ]
        if self.notice:
            lines.append(f'notice  {self.notice}')
        return '\n'.join(lines)

    def toggle(self, menu: Redrawable, item: MenuItem) -> None:
        """Space: enable or disable, saved immediately."""
        entry = self._find(item)
        if entry is not None:
            action = self._loader.disable if entry.host else self._loader.enable
            self._run(action(entry.name))
        menu.replace_items(self.items())

    def reload(self, menu: Redrawable, item: MenuItem) -> None:
        """R: re-import and load again."""
        entry = self._find(item)
        if entry is not None:
            self._run(self._loader.reload(entry.name))
        menu.replace_items(self.items())

    def remove(self, menu: Redrawable, item: MenuItem) -> None:
        """D: unload and forget."""
        entry = self._find(item)
        if entry is not None:
            self._run(self._loader.remove(entry.name))
        menu.replace_items(self.items())

    def close(self, menu: Redrawable, item: MenuItem) -> MenuResult:
        """Q: close; every change was already applied."""
        return MenuResult(item=item)

    def build(self) -> Menu:
        """Wire rows, details, and keys into a termflow menu."""
        return (
            MenuBuilder('Plugins')
            .style(markdown_style())
            .items(self.items())
            .preview(self.details)
            .on_key(' ', self.toggle)
            .on_key('r', self.reload)
            .on_key('d', self.remove)
            .on_key('q', self.close)
            .footer_hint(_HINT)
            .key_source(menu_key)
            .build()
        )

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
    loader: PluginLoader[DepsT], *, run: Callable[[PluginMenu[DepsT]], object] | None = None
) -> str:
    """Show the menu in a thread; key actions hop back to the event loop to apply."""
    loop = asyncio.get_running_loop()

    def apply(action: Coroutine[object, object, object]) -> None:
        asyncio.run_coroutine_threadsafe(action, loop).result()

    menu = PluginMenu(loader, apply=apply)
    await run_worker(lambda: (run or _run_menu)(menu))
    return ''


def _run_menu(menu: PluginMenu[DepsT]) -> None:  # pragma: no cover -- needs a real terminal.
    menu.build().run()
