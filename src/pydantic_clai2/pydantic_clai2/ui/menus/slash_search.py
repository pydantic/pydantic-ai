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


def slash_search(
    builder: MenuBuilder,
    *,
    footer: str,
    key_source: Callable[[], str],
    hotkeys: Mapping[str, KeyHandler] | None = None,
    close_key: str | None = None,
) -> Menu:
    """Build a menu where plain keys are hotkeys and `/` starts a search.

    While searching, typed characters (hotkeys included) edit the query, Backspace deletes, and
    the arrows move. Esc leaves the search and clears the query. Enter picks the highlighted row,
    except in a menu with hotkeys: there it ends the search and keeps the matches, so the hotkeys
    act on them, and the next Esc clears the filter. Otherwise Esc, like `close_key`, closes.
    The footer starts with `/ search`, followed by `footer`.
    """
    actions = dict(hotkeys or {})
    mode: Literal['browse', 'search', 'filtered'] = 'browse'

    def clear() -> str:
        nonlocal mode
        mode = 'browse'
        menu.clear_search()
        return Key.HOME

    def read_key() -> str:
        nonlocal mode
        key = key_source()
        if mode == 'search':
            if key == Key.ESCAPE:
                return clear()
            if key == Key.ENTER and actions:
                if menu.highlighted is not None:
                    mode = 'filtered'
                return ''
            return _HOTKEY + key if key in actions and not _typed(key) else key
        if key == Key.ESCAPE and mode == 'filtered':
            return clear()
        if key == '/':
            mode = 'search'
            return ''
        if key == close_key:
            return Key.ESCAPE
        if key in actions:
            return _HOTKEY + key
        # Termflow's search consumes typed characters; outside a search they do nothing.
        return '' if _typed(key) or key == Key.BACKSPACE else key

    for key, handler in actions.items():
        builder.on_key(_HOTKEY + key, handler)
    menu = builder.searchable().footer_hint(f'/ search · {footer}').key_source(read_key).build()
    return menu
