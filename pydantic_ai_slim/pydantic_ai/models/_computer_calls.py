"""The X11 key names that Claude's and Gemini's computer tools send, and the `computer` tool's names for them."""

from __future__ import annotations as _annotations

from ..computer_use import Key

X11_KEY_MAP: dict[str, Key] = {
    'return': 'enter',
    'control_l': 'ctrl',
    'shift_l': 'shift',
    'alt_l': 'alt',
    'super_l': 'super',
    'meta_l': 'super',
    'page_up': 'pageup',
    'page_down': 'pagedown',
    'print': 'printscreen',
    'caps_lock': 'capslock',
    'num_lock': 'numlock',
    'scroll_lock': 'scrolllock',
    **{f'kp_{digit}': str(digit) for digit in range(10)},
    'kp_enter': 'enter',
    'kp_add': '+',
    'kp_subtract': '-',
    'kp_multiply': '*',
    'kp_divide': '/',
    'kp_decimal': '.',
    'plus': '+',
    'minus': '-',
    'equal': '=',
    'comma': ',',
    'period': '.',
    'slash': '/',
    'backslash': '\\',
    'semicolon': ';',
    'apostrophe': "'",
    'grave': '`',
    'bracketleft': '[',
    'bracketright': ']',
}
"""X11 key names (keysyms), lower-cased, that the `computer` tool spells differently. Claude and Gemini send these.

The keypad keys become the keys they type. A name not listed here, such as `Escape` or `F5`, is only lower-cased, and the
tool refuses one it doesn't have, such as `XF86AudioRaiseVolume`.
"""


def key_from_x11(name: str) -> str:
    """The `computer` tool's name for an X11 key name."""
    lower = name.lower()
    return X11_KEY_MAP.get(lower, lower)
