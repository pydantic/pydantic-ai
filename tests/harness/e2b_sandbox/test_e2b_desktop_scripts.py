"""Runs the desktop start script `E2BSandboxBackend(desktop=True)` sends, captured from the fake SDK, in a real shell.

The other E2B desktop tests answer the scripts by matching their text, so these check that the start script parses and
does what it should, with stub programs for the sandbox's own.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from pydantic_ai_harness.e2b_sandbox import E2BSandboxBackend

from ..display_fakes import screen
from .fake_e2b import FakeE2B

pytestmark = pytest.mark.skipif(os.name != 'posix', reason='needs a POSIX host')

_STUBS = {
    'setsid': 'exec "$@"',  # macOS has no `setsid`
    'Xvfb': ':',  # the display answers through `xdotool`
    # The start script checks these are installed; the display answers, so they don't run.
    **dict.fromkeys(['xwininfo', 'scrot'], 'exit 1'),
    # The start script runs as root and makes the X socket directory, which these keep off the host's `/tmp`.
    'id': 'echo 0',
    **dict.fromkeys(['mkdir', 'chmod'], ':'),
    'xdotool': '[ "$1" != search ] || [ -e "$bin/xfce-started" ]',  # the display answers; XFCE's window once started
    # Shows its window, then stays up past the script's next 0.1 second check, so the start succeeds without leaving a
    # process behind; `startxfce4-fails` makes it exit with an error instead.
    'startxfce4': '[ -e "$bin/startxfce4-fails" ] && { echo no session bus; exit 17; }; '
    'echo started >>"$bin/xfce-starts"; touch "$bin/xfce-started"; exec sleep 1',
}


async def _start_script(fake_e2b: FakeE2B) -> str:
    """The desktop start script, as the backend sends it."""
    scripts: list[str] = []

    def respond(command: str, timeout: float | None) -> tuple[str, str, int]:
        args = shlex.split(command)
        if args[:2] == ['sh', '-c']:
            scripts.append(args[2])
        return ('  Width: 1280\n  Height: 800\n' if 'xwininfo' in command else ''), '', 0

    fake_e2b.responder = respond
    await screen(E2BSandboxBackend(desktop=True)).display_size()
    (start,) = scripts
    assert 'Xvfb' in start
    return start


_TOOLS = ('cat', 'rm', 'sleep', 'tail', 'touch', 'tr')
"""The host programs the script and the stubs use; every other program is a stub or missing, never the host's."""


@pytest.fixture
def stubs(tmp_path: Path) -> Path:
    """The directory of stub programs, which a test's flag files sit beside: the only one on the script's `PATH`."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    for name, body in _STUBS.items():
        (bin_dir / name).write_text(f'#!/bin/sh\nbin={bin_dir}\n{body}\n')
        (bin_dir / name).chmod(0o755)
    for name in _TOOLS:
        tool = shutil.which(name)
        assert tool is not None, name
        (bin_dir / name).symlink_to(tool)
    return bin_dir


def _run(script: str, stubs: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Runs `script` under `/bin/sh` with only the stubs on `PATH` and its fixed paths moved beside them."""
    root = stubs.parent
    for old, new in [('/tmp/', f'{root}/'), ('/proc/', f'{root}/proc/')]:
        script = script.replace(old, new)
    env = {'PATH': str(stubs)}
    return subprocess.run(['/bin/sh', '-c', script, 'sh', *args], env=env, capture_output=True, text=True, timeout=30)


async def test_xfce_starts_unless_its_desktop_shows(fake_e2b: FakeE2B, stubs: Path) -> None:
    start = await _start_script(fake_e2b)

    for _ in range(2):
        result = _run(start, stubs, '1280x800')
        assert result.returncode == 0, result.stderr

    assert (stubs / 'xfce-starts').read_text() == 'started\n'


async def test_xfce_exiting_fails_the_start_with_its_output(fake_e2b: FakeE2B, stubs: Path) -> None:
    start = await _start_script(fake_e2b)
    (stubs / 'startxfce4-fails').touch()

    result = _run(start, stubs, '1280x800')

    assert result.returncode == 1
    assert result.stderr == 'the XFCE desktop did not start:\nno session bus\n'


async def test_a_template_without_xfce_starts_with_no_desktop(fake_e2b: FakeE2B, stubs: Path) -> None:
    start = await _start_script(fake_e2b)
    (stubs / 'startxfce4').unlink()

    result = _run(start, stubs, '1280x800')

    assert result.returncode == 0, result.stderr
    assert not (stubs / 'xfce-starts').exists()
