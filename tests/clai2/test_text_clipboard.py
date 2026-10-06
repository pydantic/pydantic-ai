"""Copying text out picks the local clipboard command, or OSC 52 for the terminal.

Unit tests: the real clipboard commands replace the developer's clipboard, so these run stand-in
commands; `test_image_clipboard_native.py` round-trips the real ones on desktop CI runners.
"""

import io
import sys
import threading
from pathlib import Path

import pytest
from termflow.ansi import make_clipboard_copy

from pydantic_clai2.ui.prompt import text_clipboard
from pydantic_clai2.ui.prompt.text_clipboard import ClipboardCommand, LatestCopy, copy_command, copy_text, run_copy


@pytest.fixture
def installed(monkeypatch: pytest.MonkeyPatch) -> set[str]:
    """Commands `shutil.which` finds, each under `/bin`."""
    names: set[str] = set()

    def which(name: str) -> str | None:
        return f'/bin/{name}' if name in names else None

    monkeypatch.setattr(text_clipboard.shutil, 'which', which)
    monkeypatch.delenv('WAYLAND_DISPLAY', raising=False)
    monkeypatch.delenv('DISPLAY', raising=False)
    return names


def test_macos_uses_pbcopy_in_a_utf8_locale(installed: set[str]) -> None:
    assert copy_command(platform='darwin') is None
    installed.add('pbcopy')
    assert copy_command(platform='darwin') == ClipboardCommand(argv=('/bin/pbcopy',), env={'LC_CTYPE': 'UTF-8'})


def test_windows_uses_the_system_powershell_never_one_found_on_path(
    installed: set[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A `PATH` search on Windows tries the working directory first, where a repository could plant a program."""
    installed.update({'clip', 'powershell'})
    monkeypatch.setenv('SystemRoot', str(tmp_path))
    assert copy_command(platform='win32') is None
    powershell = tmp_path / 'System32' / 'WindowsPowerShell' / 'v1.0' / 'powershell.exe'
    powershell.parent.mkdir(parents=True)
    powershell.write_bytes(b'')
    command = copy_command(platform='win32')
    assert command is not None
    assert command.argv[0] == str(powershell)
    assert 'Set-Clipboard' in command.argv[-1]


def test_linux_prefers_wayland_then_x11_tools(installed: set[str], monkeypatch: pytest.MonkeyPatch) -> None:
    installed.update({'wl-copy', 'xclip', 'xsel'})
    assert copy_command(platform='linux') is None, 'no display to own a clipboard'
    monkeypatch.setenv('DISPLAY', ':0')
    assert copy_command(platform='linux') == ClipboardCommand(argv=('/bin/xclip', '-selection', 'clipboard'))
    installed.discard('xclip')
    assert copy_command(platform='linux') == ClipboardCommand(argv=('/bin/xsel', '--clipboard', '--input'))
    monkeypatch.setenv('WAYLAND_DISPLAY', 'wayland-0')
    assert copy_command(platform='linux') == ClipboardCommand(argv=('/bin/wl-copy',))


def test_run_copy_feeds_utf8_text_and_the_command_environment(tmp_path: Path) -> None:
    path = tmp_path / 'clipboard'
    script = f'import os, sys; open({str(path)!r}, "wb").write(os.environ["MARK"].encode() + sys.stdin.buffer.read())'
    command = ClipboardCommand(argv=(sys.executable, '-c', script), env={'MARK': '>'})
    run_copy(command=command, text='héllo')
    assert path.read_bytes()[1:].decode() == 'héllo'
    assert path.read_bytes()[:1] == b'>', 'the command gets its own variables'


def test_a_failing_clipboard_command_loses_only_that_copy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_copy(command=ClipboardCommand(argv=(str(tmp_path / 'missing'),)), text='lost')
    monkeypatch.setattr(text_clipboard, 'COPY_TIMEOUT', 0.01)
    run_copy(command=ClipboardCommand(argv=(sys.executable, '-c', 'import time; time.sleep(30)')), text='hung')


def test_without_a_local_command_the_terminal_copies_through_osc_52() -> None:
    output = io.StringIO()
    copy_text('copied', output=output)
    assert output.getvalue() == make_clipboard_copy('copied')


def test_locally_the_clipboard_command_copies_in_the_background(monkeypatch: pytest.MonkeyPatch) -> None:
    command = ClipboardCommand(argv=('/bin/pbcopy',))
    monkeypatch.setattr(text_clipboard, 'copy_command', lambda: command)
    monkeypatch.delenv('SSH_CONNECTION', raising=False)
    monkeypatch.delenv('SSH_TTY', raising=False)
    copied: list[tuple[ClipboardCommand, str, bool]] = []
    done = threading.Event()

    def record(*, command: ClipboardCommand, text: str) -> None:
        copied.append((command, text, threading.current_thread().daemon))
        done.set()

    monkeypatch.setattr(text_clipboard, 'run_copy', record)
    output = io.StringIO()
    copy_text('copied', output=output)
    assert done.wait(5)
    assert copied == [(command, 'copied', False)], 'not a daemon: exiting right after a copy must not lose it'
    assert output.getvalue() == '', 'no OSC 52 as well: some terminals ask before honouring it'


def test_copies_run_one_at_a_time_and_the_newest_waiting_one_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """A slow clipboard command must not leave an older selection on the clipboard, or pile up threads."""
    command = ClipboardCommand(argv=('/bin/pbcopy',))
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    copied: list[str] = []
    workers: set[int] = set()

    def slow(*, command: ClipboardCommand, text: str) -> None:
        copied.append(text)
        workers.add(threading.get_ident())
        started.set()
        assert release.wait(5)
        if text == 'third':
            finished.set()

    monkeypatch.setattr(text_clipboard, 'run_copy', slow)
    copies = LatestCopy()
    copies.submit(command=command, text='first')
    assert started.wait(5)
    copies.submit(command=command, text='second')
    copies.submit(command=command, text='third')
    release.set()
    assert finished.wait(5)
    assert copied == ['first', 'third']
    assert len(workers) == 1, 'one worker thread ran every copy'


@pytest.mark.parametrize('variable', ['SSH_CONNECTION', 'SSH_TTY'])
def test_over_ssh_the_terminal_copies_to_the_users_own_clipboard(
    monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    monkeypatch.setattr(text_clipboard, 'copy_command', lambda: ClipboardCommand(argv=('/bin/pbcopy',)))
    monkeypatch.delenv('SSH_CONNECTION', raising=False)
    monkeypatch.delenv('SSH_TTY', raising=False)
    monkeypatch.setenv(variable, 'remote')
    output = io.StringIO()
    copy_text('remote text', output=output)
    assert output.getvalue() == make_clipboard_copy('remote text')
