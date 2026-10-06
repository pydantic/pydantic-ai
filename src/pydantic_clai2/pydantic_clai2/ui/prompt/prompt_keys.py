"""Keyboard decoding only, with no prompt-toolkit application or renderer."""

import asyncio
import re
from collections.abc import Callable
from contextlib import ExitStack

from prompt_toolkit.input import Input
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.key_binding import KeyPress
from prompt_toolkit.keys import Keys

_MODIFIED_KEY = re.compile(r'\x1b\[(?:(\d+)(?::\d*)?(?::(\d+))?(?:;(\d+))?u|27;(\d+);(\d+)~)')
_CSI_KEY = re.compile(r'\x1b\[(\d+);(\d+)([A-Z~])')
_LOCK_MODIFIERS = 64 | 128  # Caps Lock, Num Lock.
_SUPPORTED_MODIFIERS = 1 | 2 | 4  # Shift, Alt, Ctrl.
_KEY_ALIASES = {
    's-tab': 'backtab',
    'shift-tab': 'backtab',
    'shift-backspace': 'backspace',
    'ctrl-backspace': 'backspace',
    'ctrl-h': 'backspace',
    'ctrl-i': 'tab',
    'ctrl-m': 'enter',
    'ctrl-[': 'escape',
    'ctrl-enter': 'enter',
    'ctrl-shift-enter': 'enter',
    '<bracketed-paste>': 'paste',
    '<vt100-mouse-event>': 'mouse',
}
_NAMED_KEYS = {
    8: 'backspace',
    9: 'tab',
    13: 'enter',
    27: 'escape',
    127: 'backspace',
    # Kitty disambiguation also gives non-text keypad keys distinct codes.
    57414: 'enter',
    57417: 'left',
    57418: 'right',
    57419: 'up',
    57420: 'down',
    57421: 'pageup',
    57422: 'pagedown',
    57423: 'home',
    57424: 'end',
    57425: 'insert',
    57426: 'delete',
    57427: 'begin',
}


def _key_name(key: Keys | str) -> str:
    name = key.value if isinstance(key, Keys) else key
    if name.startswith('c-'):
        name = 'ctrl-' + name[2:]
    return _KEY_ALIASES.get(name, name)


def _modifiers(value: str) -> int | None:
    """Ignore lock state without dropping unsupported modifiers."""
    modifiers = (int(value) - 1) & ~_LOCK_MODIFIERS
    return None if modifiers & ~_SUPPORTED_MODIFIERS else modifiers


def _modified_key(sequence: str) -> str | None:
    """Normalize the CSI-u and xterm reports requested by the editor."""
    # Known Alt reports become two decoder tokens; leave both on the legacy path.
    if (match := _CSI_KEY.fullmatch(sequence)) and sequence not in ANSI_SEQUENCES:
        code, modifier, suffix = match.groups()
        modifiers = _modifiers(modifier)
        if modifiers is None:
            return None
        params = f'{code};{modifiers + 1}' if modifiers else code
        if params == '1' and suffix != '~':
            params = ''
        key = ANSI_SEQUENCES.get(f'\x1b[{params}{suffix}')
        if isinstance(key, tuple):
            return 'alt-' + _key_name(key[-1])
        return _key_name(key) if key is not None else None
    match = _MODIFIED_KEY.fullmatch(sequence)
    if match is None:
        return None
    code, base_code, modifier, xterm_modifier, xterm_code = match.groups()
    codepoint = int(code or xterm_code)
    modifiers = _modifiers(modifier or xterm_modifier or '1')
    if modifiers is None:
        return None
    if modifiers & 4 and codepoint > 127 and base_code:
        # Kitty supplies the layout-independent identity for non-Latin Ctrl keys.
        codepoint = int(base_code)
    name = _NAMED_KEYS.get(codepoint)
    if name is None:
        if not 32 <= codepoint <= 126 or not modifiers:
            return None
        name = chr(codepoint)
    if name == ' ' and modifiers == 1:
        return name
    prefix = ''.join(label for bit, label in ((4, 'ctrl-'), (2, 'alt-'), (1, 'shift-')) if modifiers & bit)
    name = prefix + name
    return _KEY_ALIASES.get(name, name)


class PromptKeys:
    """Attach prompt-toolkit input and normalize paste, CSI-u, Kitty alternate-key and xterm reports.

    `PromptSurface` scopes xterm's `CSI >4;1m`, Kitty's `CSI >5u`, and SGR mouse reporting.
    No prompt-toolkit application or renderer is started.
    """

    def __init__(
        self,
        *,
        source: Input,
        feed: Callable[[str, str], None],
        eof: Callable[[], None],
    ) -> None:
        """Keep decoder and callbacks local to this editor."""
        self.source = source
        self.feed = feed
        self.eof = eof
        self._stack: ExitStack | None = None
        self._timer: asyncio.TimerHandle | None = None
        self._escape = False
        self._csi = ''

    def start(self) -> None:
        """Attach one input reader, with raw mode owned by its lifetime."""
        stack = ExitStack()
        try:
            stack.enter_context(self.source.raw_mode())
            stack.enter_context(self.source.attach(self.read))
        except BaseException:
            stack.close()
            raise
        self._stack = stack
        self.read()

    def stop(self) -> None:
        """Detach before a menu reads input, and cancel pending escape decoding."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._escape = False
        self._csi = ''
        if self._stack is not None:
            self._stack.close()
            self._stack = None

    def read(self) -> None:
        """Consume available decoded keys without awaiting or starting a thread."""
        if self._timer is not None:
            self._timer.cancel()
        for key in self.source.read_keys():
            self.dispatch(key)
        if self.source.closed:
            self.eof()
        else:
            self._timer = asyncio.get_running_loop().call_later(0.05, self.flush)

    def flush(self) -> None:
        """Resolve a lone Escape without treating an Alt chord as cancellation."""
        self._timer = None
        for key in self.source.flush_keys():
            self.dispatch(key)
        if self._escape:
            self._escape = False
            self.feed('escape', '')
        self._csi = ''

    def dispatch(self, key: KeyPress) -> None:
        """Translate decoder tokens into editor actions and literal paste payloads."""
        if key.key == Keys.CPRResponse:
            # Ignore a late response left over from a previous renderer/menu.
            return
        if key.key != Keys.BracketedPaste and (name := _modified_key(key.data)) is not None:
            self._escape = False
            self._csi = ''
            self.feed(name, key.data)
            return
        if self._csi:
            # The installed decoder splits unrecognized modified keys into individual
            # keys. Reassemble it here, without modifying its global key table.
            self._csi += key.data
            if len(self._csi) > 32 or len(key.data) != 1 or '@' <= key.data <= '~':
                sequence, self._csi = self._csi, ''
                if (name := _modified_key(sequence)) is not None:
                    self.feed(name, sequence)
            return
        if key.key == Keys.Escape:
            if self._escape:
                self.feed('escape', '')
            self._escape = True
            return
        if self._escape and key.data == '[':
            self._escape = False
            self._csi = '\x1b['
            return
        name = _key_name(key.key)
        if self._escape:
            name = 'alt-' + name
            self._escape = False
        self.feed(name, key.data)
