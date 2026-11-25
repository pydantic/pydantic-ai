"""While the interactive session runs on a terminal, only what Pax displays reaches it.

Everything written to standard error during the session, by Pax, a library's logging, or a child process, goes to
`telemetry.log` instead. A child that draws on the terminal itself, such as an editor, still reaches it.
"""

import logging
import os
import re
import subprocess
import sys
import threading
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path

import anyio
import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel
from pydantic_ai_pax import chat
from pydantic_ai_pax.cli.command_context import CommandContext
from pydantic_ai_pax.commands import Command
from pydantic_ai_pax.config import Settings
from pydantic_ai_pax.config.settings_store import SettingsStore
from pydantic_ai_pax.ui import stderr_relay, telemetry_log as telemetry_log_module
from pydantic_ai_pax.ui.menus.text_editor import run_editor
from pydantic_ai_pax.ui.stderr_relay import relay_stderr, terminal_stderr
from pydantic_ai_pax.ui.telemetry_log import telemetry_log
from tests._timeouts import READINESS_WAIT_TIMEOUT
from tests.pax.test_telemetry_log import unconfigured_logging

pytestmark = pytest.mark.skipif(sys.platform == 'win32', reason='needs a POSIX pseudo-terminal')

_FORK_NOTICE = b"python(4242) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.\n"
_WRITTEN = ('a library warning', 'a library handler', 'a print', 'a raw write', 'a child')


def _write_to_stderr() -> None:
    """Write to standard error each way a session's code and the programs it starts do."""
    # No handler on the record's path, so `logging.lastResort` writes it to `sys.stderr`.
    logging.getLogger('a.library').warning('a library warning')
    # A library that sets up its own handler on `sys.stderr`, as FastMCP does.
    own = logging.getLogger('a.library.with.handler')
    handler = logging.StreamHandler(sys.stderr)
    own.addHandler(handler)
    try:
        own.warning('a library handler')
    finally:
        own.removeHandler(handler)
    print('a print', file=sys.stderr, flush=True)
    os.write(2, b'a raw write\n' + _FORK_NOTICE)
    subprocess.run(['sh', '-c', 'echo a child >&2'], check=True)


class _Noise(AbstractCapability[None]):
    """`/noise` writes to standard error as the session's code and children do."""

    def get_commands(self, context: CommandContext) -> Sequence[Command]:
        async def noise(args: list[str]) -> str:
            await anyio.to_thread.run_sync(_write_to_stderr)
            return 'Noisy.'

        return [Command(name='noise', description='Write to standard error', handler=noise)]


@contextmanager
def _terminal() -> Generator[tuple[Console, list[bytes]]]:
    """A pseudo-terminal as standard error and as the console, with what it showed collected until the block ends.

    `sys.stderr` writes to file descriptor 2, as outside pytest, which keeps it for itself. Not a fixture: pytest
    points file descriptor 2 at its capture file again when the test body starts.
    """
    screen, terminal = os.openpty()
    shown: list[bytes] = []

    def collect() -> None:
        chunk = b'start'
        while chunk:
            try:
                chunk = os.read(screen, 65536)
            except OSError:  # The terminal's last open descriptor closed: `EIO` on Linux, an empty read on macOS.
                chunk = b''
            shown.append(chunk)

    reader = threading.Thread(target=collect, daemon=True)
    reader.start()
    saved = os.dup(2)
    os.dup2(terminal, 2)
    try:
        with open(terminal, 'w', encoding='utf-8') as file, open(2, 'w', encoding='utf-8', closefd=False) as stderr:
            pytest_stderr, sys.stderr = sys.stderr, stderr
            try:
                yield Console(file=file, width=1000), shown
            finally:
                sys.stderr = pytest_stderr
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        reader.join(READINESS_WAIT_TIMEOUT)
        os.close(screen)


async def test_only_what_pax_displays_reaches_the_terminal(tmp_path: Path) -> None:
    with (
        unconfigured_logging(),
        _terminal() as (console, shown),
        create_pipe_input() as keys,
        create_app_session(input=keys, output=DummyOutput()),
        anyio.fail_after(READINESS_WAIT_TIMEOUT),
    ):
        keys.send_text('/noise\r/exit\r')
        await chat(
            Agent(TestModel()),
            deps=None,
            plugins=[_Noise()],
            settings=Settings(model='test'),
            console=console,
            store=SettingsStore(tmp_path / 'config.db'),
        )
        assert terminal_stderr() is None
    log = tmp_path / 'telemetry.log'
    screen = re.sub(
        r'\x1b(\[[0-9;?<>=]*[ -/]*[@-~]|\][^\x07\x1b]*(\x07|\x1b\\))', '', b''.join(shown).decode(errors='replace')
    )
    assert [text for text in _WRITTEN if text in screen] == []
    assert 'Noisy.' in screen
    assert f'Warnings and errors from this session are in {log}.' in screen
    written = log.read_text()
    assert [text for text in _WRITTEN if text not in written] == []
    assert 'MallocStackLogging' not in written
    assert log.stat().st_mode & 0o777 == 0o600


def test_an_editor_still_draws_on_the_terminal(tmp_path: Path) -> None:
    """An editor gets the terminal as its standard error; some draw their screen there."""
    with _terminal() as (console, shown):
        with telemetry_log(tmp_path / 'telemetry.log', console=console):
            edited = run_editor(
                ['sh', '-c', 'echo drawn >&2; echo saved > "$0"'], 'text', prefix='edit-', suffix='.txt'
            )
    assert edited == 'saved\n'
    assert b'drawn' in b''.join(shown)
    assert not (tmp_path / 'telemetry.log').exists()


def test_standard_error_rotates_the_log_like_its_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """At the log's size limit the file rotates, keeping one previous file, as it does for the records written to it.

    With the limit and the read size patched down, a write of nearly twice the limit rotates the log: the two files
    hold the end of what was written, each under the limit. Where the reads split the write does not matter.
    """
    monkeypatch.setattr(telemetry_log_module, '_MAX_BYTES', 1000)
    monkeypatch.setattr(stderr_relay, '_CHUNK', 600)
    written = b''.join(f'{index:03} '.encode() + b'x' * 95 + b'\n' for index in range(18))
    log = tmp_path / 'telemetry.log'
    with _terminal() as (console, shown):
        with telemetry_log(log, console=console):
            os.write(2, written)
    previous, current = log.with_name('telemetry.log.1').read_bytes(), log.read_bytes()
    assert written.endswith(previous + current)
    assert 0 < len(previous) < 1000 and 0 < len(current) < 1000
    assert not log.with_name('telemetry.log.2').exists()
    assert b'Warnings and errors from this session are in' in b''.join(shown)


def test_an_unwritable_log_loses_standard_error_quietly(tmp_path: Path) -> None:
    """Reporting the failure would write to standard error, which is the pipe the log could not take."""
    with _terminal() as (console, shown):
        with telemetry_log(tmp_path / 'missing' / 'telemetry.log', console=console):
            os.write(2, b'lost\n')
    assert b''.join(shown) == b''


def test_under_the_macos_relay_the_terminal_is_the_one_it_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Around the session, the CLI relays standard error on macOS; inside it, standard error still goes to the log."""
    monkeypatch.setattr(stderr_relay.sys, 'platform', 'darwin')
    with _terminal() as (console, shown):
        with relay_stderr():
            held = terminal_stderr()
            with telemetry_log(tmp_path / 'telemetry.log', console=console):
                assert terminal_stderr() is not None and terminal_stderr() != held
                assert os.isatty(terminal_stderr() or 2)
                os.write(2, _FORK_NOTICE)
            assert terminal_stderr() == held
    assert b''.join(shown) == b''
    assert not (tmp_path / 'telemetry.log').exists()
