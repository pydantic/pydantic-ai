"""Runs the desktop start script `E2BSandboxBackend(desktop=True)` sends, captured from the fake SDK, in a real shell.

The other E2B desktop tests answer the scripts by matching their text, so these check that the start script parses and
does what it should, with stub programs for the sandbox's own.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from pydantic_ai_harness.e2b_sandbox import E2BSandboxBackend

from ..display_fakes import screen
from .fake_e2b import FakeE2B

_STUBS = {
    'setsid': 'exec "$@"',  # macOS has no `setsid`
    'xdotool': '[ "$1" != search ] || [ -e "$bin/xfce-started" ]',  # the display answers; XFCE's window once started
    # `-x NAME` prints the PIDs the test listed.
    'pgrep': 'cat "$bin/pgrep-$2" 2>/dev/null',
    # Stays up, as XFCE's session does; `startxfce4-fails` makes it exit with an error instead.
    'startxfce4': '[ -e "$bin/startxfce4-fails" ] && { echo no session bus; exit 17; }; '
    'touch "$bin/xfce-started"; exec sleep 1',
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


@pytest.fixture
def stubs(tmp_path: Path) -> Path:
    """The directory of stub programs, which a test's flag files sit beside."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    for name, body in _STUBS.items():
        (bin_dir / name).write_text(f'#!/bin/sh\nbin={bin_dir}\n{body}\n')
        (bin_dir / name).chmod(0o755)
    return bin_dir


def _run(script: str, stubs: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Runs `script` under `/bin/sh` with the stubs first on `PATH` and its fixed paths moved beside them."""
    root = stubs.parent
    for old, new in [('/tmp/', f'{root}/'), ('/proc/', f'{root}/proc/')]:
        script = script.replace(old, new)
    env = {'PATH': f'{stubs}:/usr/bin:/bin'}
    return subprocess.run(['/bin/sh', '-c', script, 'sh', *args], env=env, capture_output=True, text=True, timeout=30)


@pytest.mark.skipif(not shutil.which('shellcheck'), reason='shellcheck not installed (the CI runner image has it)')
async def test_the_start_script_passes_shellcheck(fake_e2b: FakeE2B) -> None:
    # SC2046 and SC2086 flag the unquoted `/proc/$(...)` and `/proc/$p` paths, which hold a PID or nothing.
    check = ['shellcheck', '-s', 'sh', '-e', 'SC2046,SC2086', '-']
    result = subprocess.run(check, input=await _start_script(fake_e2b), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize(('session_display', 'starts'), [(':1', False), (':0', True)])
async def test_xfce_starts_unless_its_session_runs_on_the_display(
    fake_e2b: FakeE2B, tmp_path: Path, stubs: Path, session_display: str, starts: bool
) -> None:
    start = await _start_script(fake_e2b)
    (tmp_path / 'proc' / '42').mkdir(parents=True)
    (tmp_path / 'proc' / '42' / 'environ').write_bytes(f'HOME=/root\0DISPLAY={session_display}\0'.encode())
    (stubs / 'pgrep-xfce4-session').write_text('42\n')

    result = _run(start, stubs, '1280x800')

    assert result.returncode == 0, result.stderr
    assert (stubs / 'xfce-started').exists() == starts


async def test_xfce_exiting_fails_the_start_with_its_output(fake_e2b: FakeE2B, stubs: Path) -> None:
    start = await _start_script(fake_e2b)
    (stubs / 'startxfce4-fails').touch()

    result = _run(start, stubs, '1280x800')

    assert result.returncode == 1
    assert result.stderr == 'the XFCE desktop did not start:\nno session bus\n'
