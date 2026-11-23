"""Helpers the model adapters' computer tools share: X11 key names, and keeping a response's calls from running."""

from __future__ import annotations as _annotations

from collections.abc import Sequence
from dataclasses import replace

from ..computer_use import COMPUTER_KIND, Key
from ..messages import ModelResponsePart, ToolCallPart
from . import ModelRequestParameters

X11_KEY_MAP: dict[str, Key] = {
    'return': 'enter',
    'control': 'ctrl',
    'control_l': 'ctrl',
    'shift_l': 'shift',
    'alt_l': 'alt',
    'super_l': 'super',
    'meta_l': 'super',
    'prior': 'pageup',
    'page_up': 'pageup',
    'next': 'pagedown',
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


def x11_key(name: str) -> str:
    """The `computer` tool's name for an X11 key name."""
    lower = name.lower()
    return X11_KEY_MAP.get(lower, lower)


UNFINISHED_COMPUTER_ARGS = '{"actions": ['
"""Arguments for a `computer` call that must not run: invalid JSON, like a call cut off by the token limit."""


def block_computer_calls(
    parts: Sequence[ModelResponsePart], model_request_parameters: ModelRequestParameters
) -> list[ModelResponsePart]:
    """Give every `computer` call in `parts` arguments that aren't valid JSON, so none of them runs.

    A blocked call keeps its place. Calls to a deferred `computer` tool are blocked too, because tool search can reveal
    it in the same response.

    The agent treats a blocked call like any call whose arguments don't parse: it asks the model again, and raises once
    the tool's retries are used up (`IncompleteToolCall` when the call is the last part of a response cut off by the
    token limit).
    """
    tool_names = {tool.name for tool in model_request_parameters.function_tools if tool.tool_kind == COMPUTER_KIND.id}
    return [
        replace(part, args=UNFINISHED_COMPUTER_ARGS)
        if isinstance(part, ToolCallPart) and part.tool_name in tool_names
        else part
        for part in parts
    ]
