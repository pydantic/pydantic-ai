"""Slash-to-search for termflow menus: plain keys stay hotkeys until `/` starts a search."""

from collections.abc import Callable, Mapping
from typing import Literal

from termflow.tui import MenuBuilder, MenuItem
from termflow.tui.keys import Key
from termflow.tui.menu import Menu, MenuResult

KeyHandler = Callable[[Menu, MenuItem], MenuResult | None]
_HOTKEY = 'slash-search:'
"""Hotkeys are bound under this prefix, so a typed character reaches them only outside a search."""


def _typed(key: str) -> bool:
    return len(key) == 1 and key.isprintable()


class _SlashSearch:
    """Translates raw keys for termflow's search, which otherwise filters on every typed key."""

    def __init__(self, *, key_source: Callable[[], str], hotkeys: Mapping[str, KeyHandler]) -> None:
        self.key_source = key_source
        self.hotkeys = hotkeys
        self.mode: Literal['browse', 'search', 'filtered'] = 'browse'
        self.query = ''
        """Mirrors termflow's query, which its public API does not expose."""
        self.edited = False
        """Whether the query was edited; only then has the cursor lost its place in the full list."""

    def build(self, builder: MenuBuilder) -> Menu:
        for key, handler in self.hotkeys.items():
            builder.on_key(_HOTKEY + key, handler)
        self.menu = builder.searchable().key_source(self.read_key).build()
        return self.menu

    def read_key(self) -> str:
        key = self.key_source()
        return self.searching(key) if self.mode == 'search' else self.browsing(key)

    def searching(self, key: str) -> str:
        if key == Key.ESCAPE:
            return self.clear()
        if key == Key.ENTER and self.hotkeys:
            if not self.query:
                self.mode, self.edited = 'browse', False
            elif self.menu.highlighted is not None:
                self.mode = 'filtered'
            return ''
        if key in self.hotkeys and not _typed(key):
            return _HOTKEY + key
        before = self.query
        if _typed(key):
            self.query += key
        elif key == Key.BACKSPACE:
            self.query = self.query[:-1]
        self.edited = self.edited or self.query != before
        return key

    def browsing(self, key: str) -> str:
        if key == Key.ESCAPE and self.mode == 'filtered':
            return self.clear()
        if key == '/':
            self.mode = 'search'
            return ''
        if key in self.hotkeys:
            return _HOTKEY + key
        # Termflow's search consumes typed characters; outside a search they do nothing.
        return '' if _typed(key) or key == Key.BACKSPACE else key

    def clear(self) -> str:
        self.menu.clear_search()
        moved = self.edited
        self.mode, self.query, self.edited = 'browse', '', False
        return Key.HOME if moved else ''


def slash_search(
    builder: MenuBuilder,
    *,
    footer: str,
    key_source: Callable[[], str],
    hotkeys: Mapping[str, KeyHandler] | None = None,
) -> Menu:
    """Build a menu where plain keys are hotkeys and `/` starts a search.

    While searching, typed characters (hotkeys included) edit the query, Backspace deletes, and
    the arrows move. Esc leaves the search and clears the query. Enter picks the highlighted row,
    except in a menu with hotkeys: there it ends the search and keeps the matches, so the hotkeys
    act on them, and the next Esc clears the filter. Otherwise Esc closes.
    The footer starts with `/ search`, followed by `footer`.
    """
    search = _SlashSearch(key_source=key_source, hotkeys=dict(hotkeys or {}))
    return search.build(builder.footer_hint(f'/ search · {footer}'))
