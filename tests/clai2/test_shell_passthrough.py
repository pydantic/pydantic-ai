"""`!command` input runs in the system shell and never starts an agent turn."""

import asyncio
import contextlib
import io
import os
import shlex
import signal
import sys
import time
from pathlib import Path

import anyio
import pytest
from rich.console import Console

from pydantic_ai import Agent, ModelRequestContext, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel
from pydantic_clai2 import chat
from pydantic_clai2.settings_store import SettingsStore
from pydantic_clai2.shell_passthrough import _taskkill_path, shell_command  # pyright: ignore[reportPrivateUsage]
from tests.clai2.test_app_edges import inputs


@pytest.mark.parametrize(
    ('text', 'command'),
    [
        ('!ls -lh', 'ls -lh'),
        ('  !git status  ', 'git status'),
        ('!  echo hi', 'echo hi'),
        ('!', None),
        ('!   ', None),
        ('  !  ', None),
        ('ls !', None),
        ('/help', None),
        ('hello', None),
    ],
)
def test_shell_command_detection(text: str, command: str | None) -> None:
    assert shell_command(text) == command


async def shell_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, values: list[str | BaseException], *, agent_turns: int = 0
) -> str:
    """Run the interactive loop, checking how many inputs reached the model."""
    requests: list[ModelRequestContext] = []

    class CountRequests(AbstractCapability[None]):
        async def before_model_request(
            self, ctx: RunContext[None], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            requests.append(request_context)
            return request_context

    monkeypatch.chdir(tmp_path)
    inputs(monkeypatch, values)
    output = io.StringIO()
    await chat(
        Agent(TestModel(custom_output_text='agent reply'), deps_type=type(None), capabilities=[CountRequests()]),
        deps=None,
        console=Console(file=output, width=120),
        store=SettingsStore(tmp_path / 'config.db'),
    )
    assert len(requests) == agent_turns
    return output.getvalue()


def test_taskkill_is_resolved_from_system_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `taskkill.exe` planted in the working directory must never be the one that runs."""
    monkeypatch.setenv('SystemRoot', r'D:\Win')
    assert _taskkill_path() == r'D:\Win\System32\taskkill.exe'
    monkeypatch.delenv('SystemRoot')
    assert _taskkill_path() == r'C:\Windows\System32\taskkill.exe'


class TestShellPassthrough:
    async def test_runs_in_cwd_without_agent_turn(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        text = await shell_session(tmp_path, monkeypatch, ['  !printf hi > marker.txt  ', '/exit'])
        assert (tmp_path / 'marker.txt').read_text() == 'hi'
        assert '$ printf hi > marker.txt' in text
        assert 'Shell passthrough, not sent to the agent' in text
        assert 'Done (' in text
        assert 'agent reply' not in text

    async def test_reports_exit_code(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        text = await shell_session(tmp_path, monkeypatch, ['!exit 3', '/exit'])
        assert 'Exit code 3 (' in text

    async def test_ctrl_c_interrupts_command_not_clai(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # The shell signals CLAI as the terminal would on Ctrl-C, and CLAI forwards it to the command.
        started = time.monotonic()
        text = await shell_session(
            tmp_path, monkeypatch, ['!kill -INT $PPID; exec sleep 30', '!printf after > marker.txt', '/exit']
        )
        assert time.monotonic() - started < 10
        assert 'Interrupted (' in text
        assert (tmp_path / 'marker.txt').read_text() == 'after'

    @pytest.mark.skipif(sys.platform == 'win32', reason='uses POSIX shell process-group signalling')
    async def test_ctrl_c_is_forwarded_before_kill(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A command in its own session still gets Ctrl-C, so it can clean up before the grace kill."""
        spawn = asyncio.create_subprocess_shell

        async def spawn_then_ctrl_c(command: str, *, start_new_session: bool) -> asyncio.subprocess.Process:
            # Press Ctrl-C only once the spawn has returned and the shell has installed its trap.
            process = await spawn(command, start_new_session=start_new_session)
            with anyio.fail_after(5):
                while not (tmp_path / 'ready').exists():
                    await anyio.sleep(0.01)  # pragma: lax no cover -- the shell may already be ready.
            os.kill(os.getpid(), signal.SIGINT)
            return process

        monkeypatch.setattr('pydantic_clai2.shell_passthrough.asyncio.create_subprocess_shell', spawn_then_ctrl_c)
        command = "trap 'printf cleaned > cleanup.txt; exit 130' INT; : > ready; while :; do :; done"
        text = await shell_session(tmp_path, monkeypatch, [f'!{command}', '/exit'])
        assert 'Interrupted (' in text
        assert (tmp_path / 'cleanup.txt').read_text() == 'cleaned'

    @pytest.mark.skipif(sys.platform == 'win32', reason='uses POSIX shell process-group signalling')
    async def test_ctrl_c_kills_shell_descendants(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cancelling a shell command must also terminate a background child."""
        child_code = "import os, pathlib, time; pathlib.Path('child.pid').write_text(str(os.getpid())); time.sleep(30)"
        python = shlex.quote(sys.executable)
        command = f'{python} -c {shlex.quote(child_code)} & while [ ! -f child.pid ]; do :; done; kill -INT $PPID; wait'

        text = await shell_session(tmp_path, monkeypatch, [f'!{command}', '/exit'])
        assert 'Interrupted (' in text
        child_pid = int((tmp_path / 'child.pid').read_text())
        try:
            with anyio.fail_after(2):
                while True:
                    try:
                        os.kill(child_pid, 0)
                    except ProcessLookupError:
                        break
                    await anyio.sleep(0.01)
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.kill(child_pid, signal.SIGKILL)

    async def test_second_ctrl_c_exits(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        text = await shell_session(tmp_path, monkeypatch, [KeyboardInterrupt(), '!kill -INT $PPID; exec sleep 30'])
        assert 'Input cleared' in text
        assert 'Interrupted (' in text

    async def test_spawn_failure_is_reported(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        async def unavailable(command: str, **kwargs: object) -> None:
            raise FileNotFoundError('no shell')

        monkeypatch.setattr('pydantic_clai2.shell_passthrough.asyncio.create_subprocess_shell', unavailable)
        text = await shell_session(tmp_path, monkeypatch, ['!ls', '/exit'])
        assert 'Shell error: no shell' in text

    async def test_nul_byte_is_reported(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        text = await shell_session(tmp_path, monkeypatch, ['!echo a\x00b', '/exit'])
        assert 'Shell error: embedded null byte' in text

    async def test_bare_bang_is_a_prompt(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        text = await shell_session(tmp_path, monkeypatch, ['!', '/exit'], agent_turns=1)
        assert 'agent reply' in text

    async def test_help_mentions_passthrough(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        text = await shell_session(tmp_path, monkeypatch, ['/help', '/exit'])
        assert '!COMMAND: Run COMMAND with the system shell' in text
