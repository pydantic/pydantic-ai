"""Multi-line text editing in menus: the user's `$VISUAL` or `$EDITOR`, else a built-in editor.

Both run in a menu worker thread between widgets, while the live panel is released, so the editor
owns the terminal. Termflow has no multi-line input, so the built-in editor is a prompt_toolkit text area.
"""

import os
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from contextlib import suppress
from pathlib import Path

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.widgets import TextArea

EditorApp = Application[str | None]
"""The built-in editor: its result is the edited text, or `None` when cancelled."""
TextAreaRunner = Callable[[EditorApp], str | None]
BUILT_IN_KEYS = 'ctrl-s save · esc cancel'


def editor_command(*, default: str = '') -> list[str]:
    """`$VISUAL`, else `$EDITOR`, else `default`, split as a shell would; empty when none is set.

    Raises `ValueError` for a value a shell could not split either, such as one with an unclosed quote.
    """
    return shlex.split(os.environ.get('VISUAL') or os.environ.get('EDITOR') or default)


def run_editor(command: Sequence[str], text: str, *, prefix: str, suffix: str) -> str | None:
    """Open `command` on a temporary file holding `text`; what it saved, or `None` when it exited with an error.

    Raises `OSError` when the editor cannot start. The screen is cleared first, so a terminal editor
    does not open over the menu it was started from.
    """
    handle, name = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    path = Path(name)
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as file:
            file.write(text)
        print('\x1b[2J\x1b[H', end='', flush=True, file=sys.__stdout__)
        if subprocess.call([*command, name]) != 0:
            return None
        return path.read_text(encoding='utf-8')
    finally:
        path.unlink(missing_ok=True)


def external_editor() -> list[str] | None:
    """The command `edit_text` opens, or `None` for the built-in editor.

    That is `$VISUAL` or `$EDITOR`, except on Windows, where the built-in editor always opens.
    A value that cannot be split into a command also leaves the built-in editor.
    """
    if sys.platform == 'win32':
        return None
    try:
        return editor_command() or None
    except ValueError:
        return None


def build_text_area(*, title: str, text: str) -> EditorApp:
    """A full-screen editor for `text`. Ctrl-S returns the edited text; Esc and Ctrl-C return `None`."""
    area = TextArea(text=text, multiline=True, scrollbar=True)
    bindings = KeyBindings()

    @bindings.add('c-s')
    def save(event: KeyPressEvent) -> None:
        event.app.exit(result=area.text)

    @bindings.add('escape', eager=True)
    @bindings.add('c-c')
    def cancel(event: KeyPressEvent) -> None:
        event.app.exit(result=None)

    header = Window(FormattedTextControl([('bold', title)]), height=1)
    footer = Window(FormattedTextControl(BUILT_IN_KEYS), height=1, style='reverse')
    return Application(
        layout=Layout(HSplit([header, area, footer]), focused_element=area),
        key_bindings=bindings,
        full_screen=True,
    )


def run_text_area(app: EditorApp) -> str | None:  # pragma: no cover -- needs a real terminal.
    """Show the built-in editor on the real terminal."""
    return app.run()


def edit_text(text: str, *, title: str, run_area: TextAreaRunner = run_text_area) -> str | None:
    """Edit `text` in `external_editor()`, else in the built-in editor; `None` when the user cancelled.

    An editor that cannot start falls back to the built-in editor. One that exits with an error,
    such as Vim's `:cq`, counts as cancelled.
    """
    command = external_editor()
    if command is not None:
        with suppress(OSError):
            return run_editor(command, text, prefix='clai_', suffix='.md')
    return run_area(build_text_area(title=title, text=text))
