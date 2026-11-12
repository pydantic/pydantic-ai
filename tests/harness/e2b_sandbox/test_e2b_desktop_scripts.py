"""Runs the shell scripts `E2BSandboxBackend(desktop=True)` sends, captured from the fake SDK, in a real shell.

The other E2B desktop tests answer the scripts by matching their text, so these check that the scripts parse and do
what they should, with stub programs for the sandbox's own.
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
    'pgrep': 'cat "$bin/pgrep-$2" 2>/dev/null',  # prints the PIDs the test listed for `-x NAME` or `-f PATTERN`
    'startxfce4': 'touch "$bin/xfce-started"',
    # The start script checks these are installed; the display already answers, so they don't run.
    **dict.fromkeys(['Xvfb', 'xwininfo', 'scrot'], 'exit 1'),
    # The viewer's two programs: each lists its PID for `pgrep`, logs its start, prints its ready line and stays up.
    'x11vnc': '[ "$1" = -storepasswd ] && { echo "$2" > "$3"; exit; }; '
    'echo $$ > "$bin/pgrep-x11vnc"; echo x11vnc >> "$bin/starts"; echo PORT=5900; exec sleep 1',
    'novnc_proxy': 'echo $$ > "$bin/pgrep-[w]ebsockify"; echo novnc >> "$bin/starts"; '
    'echo Navigate to this URL; exec sleep 1',
}


async def _scripts(fake_e2b: FakeE2B) -> tuple[str, str]:
    """The desktop start script and the viewer script, as the backend sends them."""
    scripts: list[str] = []

    def respond(command: str, timeout: float | None) -> tuple[str, str, int]:
        args = shlex.split(command)
        if args[:2] == ['sh', '-c']:
            scripts.append(args[2])
        return ('  Width: 1280\n  Height: 800\n' if 'xwininfo' in command else 'password'), '', 0

    fake_e2b.responder = respond
    backend = E2BSandboxBackend(desktop=True)
    await screen(backend).display_size()
    await backend.live_view_url()
    # The viewer starts the display again first, with the same script.
    start, again, viewer = scripts
    assert again == start and 'Xvfb' in start and 'novnc_proxy' in viewer
    return start, viewer


def _run(script: str, tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Runs `script` under `/bin/sh` with the stubs first on `PATH` and its fixed paths moved under `tmp_path`."""
    bin_dir = tmp_path / 'bin'
    if not bin_dir.exists():
        bin_dir.mkdir()
        for name, body in _STUBS.items():
            (bin_dir / name).write_text(f'#!/bin/sh\nbin={bin_dir}\n{body}\n')
            (bin_dir / name).chmod(0o755)
    for old, new in [('/tmp/', f'{tmp_path}/'), ('/proc/', f'{tmp_path}/proc/'), ('/opt/noVNC/utils/', f'{bin_dir}/')]:
        script = script.replace(old, new)
    env = {'PATH': f'{bin_dir}:/usr/bin:/bin', 'LC_ALL': 'C'}  # `C`, so `tr` takes `/dev/urandom`'s bytes
    return subprocess.run(['/bin/sh', '-c', script, 'sh', *args], env=env, capture_output=True, text=True, timeout=30)


async def test_the_scripts_are_valid_shell(fake_e2b: FakeE2B) -> None:
    for script in await _scripts(fake_e2b):
        if shutil.which('dash'):  # pragma: no branch - CI has it; a laptop may not
            assert subprocess.run(['dash', '-n', '-c', script], capture_output=True, text=True).stderr == ''
        if shutil.which('shellcheck'):  # pragma: no branch - CI has it; a laptop may not
            # SC2046 and SC2086 flag the unquoted `/proc/$(...)` and `/proc/$p` paths, which hold a PID or nothing.
            check = ['shellcheck', '-s', 'sh', '-e', 'SC2046,SC2086', '-']
            result = subprocess.run(check, input=script, capture_output=True, text=True)
            assert result.returncode == 0, result.stdout


async def test_the_viewer_keeps_its_password_and_starts_once(fake_e2b: FakeE2B, tmp_path: Path) -> None:
    _, viewer = await _scripts(fake_e2b)

    first, second = _run(viewer, tmp_path), _run(viewer, tmp_path)

    assert (first.returncode, second.returncode) == (0, 0), first.stderr + second.stderr
    password = (tmp_path / 'pydantic-ai-viewer' / 'password').read_text()
    assert len(password) == 8 and password.isalnum()
    assert first.stdout == second.stdout == password
    assert (tmp_path / 'pydantic-ai-viewer' / 'passwd').read_text() == f'{password}\n'
    assert (tmp_path / 'bin' / 'starts').read_text() == 'x11vnc\nnovnc\n'


@pytest.mark.parametrize(('session_display', 'starts'), [(':1', False), (':0', True)])
async def test_xfce_starts_unless_its_session_runs_on_the_display(
    fake_e2b: FakeE2B, tmp_path: Path, session_display: str, starts: bool
) -> None:
    start, _ = await _scripts(fake_e2b)
    (tmp_path / 'proc' / '42').mkdir(parents=True)
    (tmp_path / 'proc' / '42' / 'environ').write_bytes(f'HOME=/root\0DISPLAY={session_display}\0'.encode())
    _run('true', tmp_path)  # writes the stubs
    (tmp_path / 'bin' / 'pgrep-xfce4-session').write_text('42\n')

    result = _run(start, tmp_path, '1280x800')

    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'bin' / 'xfce-started').exists() == starts
