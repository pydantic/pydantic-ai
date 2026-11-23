"""What each terminal sends for the editor's Enter chords, and what the editor does with it.

Terminals disagree on the bytes for a modified Enter, and some keep a chord for themselves. A row records one
terminal's bytes in one keyboard mode; `test_chords.py` sends them through the editor's decoder and `LivePrompt`
and checks the action. A row whose chord the terminal keeps (`sends=None`) runs nothing, so its action must be
reachable through another row of the same terminal. A new report of a terminal's bytes adds a row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Terminal = Literal['xterm', 'windows-terminal']
"""`windows-terminal` is Windows itself or Windows Terminal (`WT_SESSION`, also in WSL): Ctrl+Enter steers there."""

Action = Literal['submit', 'newline', 'steer']


@dataclass(frozen=True, kw_only=True)
class Chord:
    terminal: Terminal
    mode: Literal['legacy', 'kitty', 'modify-other-keys']
    """`legacy` reports no modified keys; the other two are the modified-key reports the editor turns on."""
    chord: str
    sends: str | None
    """The bytes the terminal sends, or `None` when it keeps the chord for itself."""
    action: Action


CHORDS = (
    Chord(terminal='xterm', mode='legacy', chord='Enter', sends='\r', action='submit'),
    Chord(terminal='xterm', mode='legacy', chord='Ctrl+J', sends='\n', action='newline'),
    Chord(terminal='xterm', mode='legacy', chord='Alt+Enter', sends='\x1b\r', action='steer'),
    Chord(terminal='xterm', mode='kitty', chord='Shift+Enter', sends='\x1b[13;2u', action='newline'),
    Chord(terminal='xterm', mode='kitty', chord='Ctrl+Enter', sends='\x1b[13;5u', action='submit'),
    Chord(terminal='xterm', mode='modify-other-keys', chord='Shift+Enter', sends='\x1b[27;2;13~', action='newline'),
    Chord(terminal='windows-terminal', mode='legacy', chord='Enter', sends='\r', action='submit'),
    # Without modified-key reporting Windows Terminal sends Ctrl+Enter as LF, the same byte as Ctrl+J.
    Chord(terminal='windows-terminal', mode='legacy', chord='Ctrl+Enter', sends='\n', action='steer'),
    # Windows Terminal takes Alt+Enter for full screen.
    Chord(terminal='windows-terminal', mode='legacy', chord='Alt+Enter', sends=None, action='steer'),
    Chord(terminal='windows-terminal', mode='kitty', chord='Ctrl+Enter', sends='\x1b[13;5u', action='steer'),
    Chord(terminal='windows-terminal', mode='kitty', chord='Shift+Enter', sends='\x1b[13;2u', action='newline'),
)
