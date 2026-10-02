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
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.step_persistence.conversations import SqliteConversationStore
from pydantic_clai2 import chat
from pydantic_clai2.cli.shell_passthrough import (
    _taskkill_path,  # pyright: ignore[reportPrivateUsage]
    run_shell_command,
    shell_command,
)
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.ui.prompt.interrupts import Interrupts
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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    values: list[str | BaseException],
    *,
    agent_turns: int = 0,
    requests: list[ModelRequestContext] | None = None,
    resume: str | None = None,
) -> str:
    """Run the interactive loop, checking how many inputs reached the model."""
    captured = requests if requests is not None else []

    class CountRequests(AbstractCapability[None]):
        async def before_model_request(
            self, ctx: RunContext[None], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            captured.append(request_context)
            return request_context

    monkeypatch.chdir(tmp_path)
    inputs(monkeypatch, values)
    output = io.StringIO()
    await chat(
        Agent(TestModel(custom_output_text='agent reply'), deps_type=type(None), capabilities=[CountRequests()]),
        deps=None,
        console=Console(file=output, force_terminal=False, width=120),
        store=SettingsStore(tmp_path / 'config.db'),
        resume=resume,
    )
    assert len(captured) == agent_turns
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
        assert 'Shell command and output saved for the next prompt' in text
        assert 'Done (' in text
        assert 'agent reply' not in text

    async def test_reports_exit_code(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        text = await shell_session(tmp_path, monkeypatch, ['!exit 3', '/exit'])
        assert 'Exit code 3 (' in text

    async def test_ctrl_c_during_process_spawn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        spawn_started = asyncio.Event()
        release_spawn = asyncio.Event()
        wait_calls = 0
        cleanup_calls: list[str] = []

        class Process:
            def __init__(self) -> None:
                self.stdout = asyncio.StreamReader()
                self.stderr = asyncio.StreamReader()
                self.stdout.feed_eof()
                self.stderr.feed_eof()

            async def wait(self) -> int:
                nonlocal wait_calls
                wait_calls += 1
                return 0

        async def delayed_spawn(command: str, *, start_new_session: bool, stdout: int, stderr: int) -> Process:
            assert command == 'sleep forever'
            assert start_new_session is True
            assert stdout == stderr == asyncio.subprocess.PIPE
            spawn_started.set()
            await release_spawn.wait()
            return Process()

        def interrupt(_process: Process) -> None:
            cleanup_calls.append('interrupt')

        async def kill_process_tree(_process: Process) -> None:
            cleanup_calls.append('kill')

        monkeypatch.setattr('pydantic_clai2.cli.shell_passthrough.asyncio.create_subprocess_shell', delayed_spawn)
        monkeypatch.setattr('pydantic_clai2.cli.shell_passthrough._interrupt', interrupt)
        monkeypatch.setattr('pydantic_clai2.cli.shell_passthrough._kill_process_tree', kill_process_tree)
        output = io.StringIO()
        interrupts = Interrupts()
        command = asyncio.create_task(
            run_shell_command('sleep forever', console=Console(file=output), interrupts=interrupts)
        )
        await spawn_started.wait()

        assert interrupts.cancel()
        asyncio.get_running_loop().call_soon(release_spawn.set)
        await command

        assert wait_calls == 3
        assert cleanup_calls == ['interrupt', 'kill']
        assert 'Interrupted (' in output.getvalue()

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

        async def spawn_then_ctrl_c(
            command: str, *, start_new_session: bool, stdout: int, stderr: int
        ) -> asyncio.subprocess.Process:
            # Press Ctrl-C only once the spawn has returned and the shell has installed its trap.
            process = await spawn(command, start_new_session=start_new_session, stdout=stdout, stderr=stderr)
            with anyio.fail_after(5):
                while not (tmp_path / 'ready').exists():
                    await anyio.sleep(0.01)  # pragma: lax no cover -- the shell may already be ready.
            os.kill(os.getpid(), signal.SIGINT)
            return process

        monkeypatch.setattr('pydantic_clai2.cli.shell_passthrough.asyncio.create_subprocess_shell', spawn_then_ctrl_c)
        command = "trap 'printf cleaned > cleanup.txt; exit 130' INT; : > ready; while :; do :; done"
        text = await shell_session(tmp_path, monkeypatch, [f'!{command}', '/exit'])
        assert 'Interrupted (' in text
        assert (tmp_path / 'cleanup.txt').read_text() == 'cleaned'

    @pytest.mark.skipif(sys.platform == 'win32', reason='uses POSIX shell process-group signalling')
    async def test_ctrl_c_kills_shell_descendants(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cancelling a shell command must also terminate a background child."""
        child_code = (
            "import os, pathlib, time; pathlib.Path('child.pid.tmp').write_text(str(os.getpid())); "
            "os.replace('child.pid.tmp', 'child.pid'); time.sleep(30)"
        )
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

        monkeypatch.setattr('pydantic_clai2.cli.shell_passthrough.asyncio.create_subprocess_shell', unavailable)
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
        assert 'save command and output for the next prompt' in ' '.join(text.split())


@pytest.mark.parametrize(
    ('command', 'status', 'stdout', 'stderr'),
    [
        ('true', 'Exit code 0', '', ''),
        ("printf '[bold]out[/bold]\\n'; printf 'err\\n' >&2", 'Exit code 0', '[bold]out[/bold]\n', 'err\n'),
        ("printf 'failed' >&2; exit 7", 'Exit code 7', '', 'failed'),
        ("printf 'café \\nnext line'", 'Exit code 0', 'café \nnext line', ''),
        ('echo a\x00b', 'Shell error: embedded null byte', '', ''),
    ],
)
async def test_shell_context_reaches_next_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str, status: str, stdout: str, stderr: str
) -> None:
    requests: list[ModelRequestContext] = []
    text = await shell_session(
        tmp_path, monkeypatch, ['before', f'!{command}', 'after', '/exit'], agent_turns=2, requests=requests
    )
    prompts = [
        part.content
        for message in requests[-1].messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ]
    assert prompts == [
        'before',
        f'The user ran a local shell command (not an agent tool call):\n'
        f'$ {command}\n{status}\n\nstdout:\n{stdout}\n\nstderr:\n{stderr}',
        'after',
    ]
    assert stdout in text
    assert stderr in text
    if stdout and not stderr and not stdout.endswith('\n'):
        assert f'{stdout}\nDone (' in text


async def test_shell_context_survives_exit_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    await shell_session(
        tmp_path,
        monkeypatch,
        ['!printf once >> marker; printf output', '!printf error >&2; exit 2', '/exit'],
    )
    conversations = SqliteConversationStore(database=tmp_path / 'sessions.db')
    [summary] = await conversations.listing()
    saved = await conversations.get(conversation_id=summary.id)
    assert len(saved.messages) == 2
    assert all(isinstance(message, ModelRequest) for message in saved.messages)
    requests: list[ModelRequestContext] = []
    await shell_session(
        tmp_path, monkeypatch, ['explain', '/exit'], agent_turns=1, requests=requests, resume=summary.id
    )
    assert [
        part.content
        for message in requests[0].messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ] == [
        part.content
        for message in saved.messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ] + ['explain']
    assert (tmp_path / 'marker').read_text() == 'once'


async def test_new_session_drops_shell_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[ModelRequestContext] = []
    await shell_session(
        tmp_path, monkeypatch, ['!printf old', '/new', 'fresh', '/exit'], agent_turns=1, requests=requests
    )
    assert [
        part.content
        for message in requests[0].messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ] == ['fresh']


async def test_interrupted_shell_output_reaches_next_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[ModelRequestContext] = []
    await shell_session(
        tmp_path,
        monkeypatch,
        ['!printf partial; printf problem >&2; kill -INT $PPID; exec sleep 30', 'explain', '/exit'],
        agent_turns=1,
        requests=requests,
    )
    message = requests[0].messages[0]
    assert isinstance(message, ModelRequest)
    part = message.parts[0]
    assert isinstance(part, UserPromptPart)
    assert isinstance(part.content, str)
    assert '\nInterrupted\n\nstdout:\npartial\n\nstderr:\nproblem' in part.content


async def test_large_stdout_and_stderr_are_drained_concurrently() -> None:
    output = io.StringIO()
    command = "i=0; while [ $i -lt 5000 ]; do printf 'stdout\\n'; printf 'stderr\\n' >&2; i=$((i + 1)); done"
    with anyio.fail_after(10):
        context = await run_shell_command(command, console=Console(file=output), interrupts=Interrupts())
    stdout = 'stdout\n' * 5000
    stderr = 'stderr\n' * 5000
    assert f'\nstdout:\n{stdout}\n\nstderr:\n{stderr}' in context
    assert output.getvalue().count('stdout\n') == 5000
    assert output.getvalue().count('stderr\n') == 5000


@pytest.mark.parametrize('external_cancel', [False, True])
async def test_output_streams_before_exit_and_readers_are_drained(external_cancel: bool) -> None:
    printed = asyncio.Event()

    class Output(io.StringIO):
        def write(self, text: str) -> int:
            if text == 'ready':
                printed.set()
            return super().write(text)

    output = Output()
    interrupts = Interrupts()
    tasks_before = asyncio.all_tasks()
    task = asyncio.create_task(
        run_shell_command('printf ready; exec sleep 30', console=Console(file=output), interrupts=interrupts)
    )
    with anyio.fail_after(10):
        await printed.wait()
        assert not task.done()
        if external_cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            assert interrupts.cancel()
            context = await task
            assert '\nInterrupted\n\nstdout:\nready\n\nstderr:\n' in context
    assert asyncio.all_tasks() == tasks_before


async def test_decodes_split_and_invalid_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    streams = [asyncio.StreamReader(), asyncio.StreamReader()]
    first_read = asyncio.Event()
    printed = asyncio.Event()

    class Output(io.StringIO):
        def write(self, text: str) -> int:
            if text == 'start':
                first_read.set()
            if '€' in text:
                printed.set()
            return super().write(text)

    class Process:
        stdout, stderr = streams

        async def wait(self) -> int:
            return 0

    async def spawn(command: str, **kwargs: object) -> Process:
        return Process()

    monkeypatch.setattr('pydantic_clai2.cli.shell_passthrough.asyncio.create_subprocess_shell', spawn)
    task = asyncio.create_task(run_shell_command('test', console=Console(file=Output()), interrupts=Interrupts()))
    with anyio.fail_after(10):
        streams[0].feed_data(b'start\xe2\x82')
        await first_read.wait()
        streams[0].feed_data(b'\xac')
        await printed.wait()
        streams[0].feed_eof()
        streams[1].feed_data(b'\xff\xe2')
        streams[1].feed_eof()
        context = await task
    assert '\nstdout:\nstart€\n\nstderr:\n��' in context
