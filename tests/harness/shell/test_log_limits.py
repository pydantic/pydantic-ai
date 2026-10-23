"""Real subprocess regressions for `Shell.max_log_bytes`, which bounds background and `shell` job logs."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import sys
from collections.abc import Mapping
from pathlib import Path

import anyio
import pytest

from pydantic_ai.workspaces import CommandResult, LocalWorkspaceBackend, WorkspaceCommand
from pydantic_ai_harness.shell import Shell

from .._tool_calls import call_tool
from ._context import run_context
from .test_persistent import Recorder, shell

pytestmark = [pytest.mark.skipif(os.name == 'nt', reason='POSIX shell commands and process groups')]

NOTICE = '[Log exceeded max_log_bytes=2000; earlier output was cleared.]'
CHATTY = "while :; do printf '%01000d\\n' 0{redirect}; sleep 0.01; done"
"""Writes about 100 KB a second until stopped."""


@pytest.fixture(autouse=True)
def fast_log_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('pydantic_ai_harness.shell._jobs._LOG_CHECK_INTERVAL', 0.05)


def _handle(output: str, label: str) -> Path:
    return Path(output.split(f'{label}: ', 1)[1].split('\n', 1)[0])


async def _wait_for_notice(log: Path) -> None:
    with anyio.fail_after(30):
        while NOTICE not in log.read_text(encoding='utf-8', errors='replace'):
            await anyio.sleep(0.05)


class _ReadsHeldUntilWritten(LocalWorkspaceBackend):
    """Marks the first read of the start of a job log with a `read` file, and holds later reads until `written` exists."""

    def __init__(self, working_dir: Path) -> None:
        super().__init__(working_dir)
        self.markers = working_dir

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        assert isinstance(command, str)
        from_start = 'tail -c +1 ' in command
        if 'tail -c +' in command and not from_start:
            with anyio.fail_after(10):
                while not (self.markers / 'written').exists():
                    await anyio.sleep(0.01)
        result = await super().run(command, shell=shell, env=env, timeout=timeout)
        if from_start and result.stdout.strip():
            (self.markers / 'read').touch()
        return result


class _ClearedOnEveryRead(LocalWorkspaceBackend):
    """Reports a new `max_log_bytes` clear of every job's log each time a log is read."""

    def __init__(self, working_dir: Path) -> None:
        super().__init__(working_dir)
        self.jobs = working_dir / '.pydantic-ai-harness' / 'shell'

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        assert isinstance(command, str)
        result = await super().run(command, shell=shell, env=env, timeout=timeout)
        if 'tail -c +' in command:
            for job in self.jobs.iterdir():
                with (job / 'log_clears').open('ab') as clears:
                    clears.write(b'xx')
        return result


class _ClearsAroundReads(LocalWorkspaceBackend):
    """Plays the watcher by hand: the first read past the log's start lands while a clear is under way.

    That read only starts the clear (the first `log_clears` byte); the next one empties the log, leaves the notice,
    finishes the clear, and creates `cleared`, which the command waits for.
    """

    def __init__(self, working_dir: Path) -> None:
        super().__init__(working_dir)
        self.markers = working_dir
        self.steps = 0

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        assert isinstance(command, str)
        if 'tail -c +' in command and 'tail -c +1 ' not in command and self.steps < 2:
            [job] = (self.markers / '.pydantic-ai-harness' / 'shell').iterdir()
            if self.steps:
                (job / 'output.log').write_text(f'{NOTICE}\n')
            with (job / 'log_clears').open('ab') as clears:
                clears.write(b'x')
            if self.steps:
                (self.markers / 'cleared').touch()
            self.steps += 1
        return await super().run(command, shell=shell, env=env, timeout=timeout)


class _ClearedAgainDuringTheReread(LocalWorkspaceBackend):
    """Clears the log, as the watcher would, before the first read past its start and again before the read after it."""

    def __init__(self, working_dir: Path) -> None:
        super().__init__(working_dir)
        self.markers = working_dir
        self.clears = 0

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        assert isinstance(command, str)
        if 'tail -c +' in command and (self.clears or 'tail -c +1 ' not in command) and self.clears < 2:
            [job] = (self.markers / '.pydantic-ai-harness' / 'shell').iterdir()
            (job / 'output.log').write_text(f'{NOTICE}\n')
            with (job / 'log_clears').open('ab') as clears:
                clears.write(b'xx')
            self.clears += 1
            if self.clears == 2:
                (self.markers / 'cleared').touch()
        return await super().run(command, shell=shell, env=env, timeout=timeout)


class _TermIgnored(LocalWorkspaceBackend):
    """Launches jobs with `SIGTERM` ignored, which a shell can't trap, so no `TERM` stops the log watcher."""

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        if isinstance(command, str) and 'wrapper=' in command:
            command = f"trap '' TERM\n{command}"
        return await super().run(command, shell=shell, env=env, timeout=timeout)


class _ClearedWhileCounted(LocalWorkspaceBackend):
    """Clears the log, as the watcher would, at the first read from its start once the command has exited."""

    def __init__(self, working_dir: Path) -> None:
        super().__init__(working_dir)
        self.markers = working_dir
        self.cleared = False

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        assert isinstance(command, str)
        if 'tail -c +1 ' in command and not self.cleared:
            [job] = (self.markers / '.pydantic-ai-harness' / 'shell').iterdir()
            status = job / 'status.json'
            if status.exists() and json.loads(status.read_text())['exit_code'] is not None:
                (job / 'output.log').write_text(f'{NOTICE}\n')
                (job / 'log_clears').write_bytes(b'xx')
                self.cleared = True
        return await super().run(command, shell=shell, env=env, timeout=timeout)


class TestShellLogLimits:
    @pytest.mark.parametrize('limit', [0, -1, True, sys.maxsize + 1])
    def test_invalid(self, limit: int) -> None:
        with pytest.raises(ValueError, match='max_log_bytes'):
            Shell(max_log_bytes=limit).get_toolset()

    async def test_command_reading_its_own_log_finishes(self, tmp_path: Path) -> None:
        # Each line read from the log is appended to it again; without a limit this never reaches the end.
        command = (
            'echo seed; for log in .pydantic-ai-harness/shell/*/output.log; do '
            'while IFS= read -r line; do echo "$line"; done < "$log"; done'
        )
        output = await shell(tmp_path, {'command': command, 'timeout': 60}, max_log_bytes=2000)
        assert '"exit_code": 0' in output
        log = _handle(output, 'Output')
        assert NOTICE in log.read_text(encoding='utf-8')
        assert log.stat().st_size < 1_000_000

    async def test_events_report_a_cleared_log(self, tmp_path: Path) -> None:
        # The first write is streamed before the second pushes the log past the limit and it is emptied.
        command = "printf '%01500d\\n' 0; sleep 0.3; printf '%01500d\\n' 0; sleep 0.5; echo after"
        recorder = Recorder()
        output = await shell(tmp_path, {'command': command, 'timeout': 60}, capabilities=[recorder], max_log_bytes=2000)
        assert '"exit_code": 0' in output
        assert recorder.output.startswith('0' * 1500)
        assert recorder.output.endswith(f'{NOTICE}\nafter\n')
        assert recorder.finished.truncated
        assert recorder.finished.total_lines is None

    async def test_events_report_a_log_cleared_and_regrown_between_reads(self, tmp_path: Path) -> None:
        # The command overflows the log only once its start has been read, and writes past that point again
        # once the log has been emptied; the read after the first is held until then, so it lands in the
        # regrown log, and the events must start again from the notice.
        command = (
            "printf '%01500d\\n' 0; until [ -e read ]; do sleep 0.01; done; printf '%01500d\\n' 0; "
            'until [ "$(cat .pydantic-ai-harness/shell/*/log_clears 2> /dev/null | wc -c)" -ge 2 ]; do sleep 0.01; done; '
            "printf '%01800d\\n' 0; echo after; : > written"
        )
        recorder = Recorder()
        output = await call_tool(
            [Shell[None](denied_commands=[], tools=['shell'], max_log_bytes=2000), recorder],
            'shell',
            {'command': command, 'timeout': 30},
            workspace=_ReadsHeldUntilWritten(tmp_path),
        )
        assert '"exit_code": 0' in output
        assert recorder.output.endswith(f'{NOTICE}\n{"0" * 1800}\nafter\n')
        assert recorder.finished.truncated
        assert recorder.finished.total_lines is None

    async def test_events_skip_a_read_while_a_clear_is_under_way(self, tmp_path: Path) -> None:
        # A chunk read while a clear is under way is dropped, and the log is read again from its notice once it's done.
        recorder = Recorder()
        output = await call_tool(
            [Shell[None](denied_commands=[], tools=['shell'], max_log_bytes=2000), recorder],
            'shell',
            {'command': 'echo first; until [ -e cleared ]; do sleep 0.01; done; echo after', 'timeout': 30},
            workspace=_ClearsAroundReads(tmp_path),
        )
        assert '"exit_code": 0' in output
        assert recorder.output == f'first\n{NOTICE}\nafter\n'

    async def test_events_restart_after_a_clear_during_the_reread(self, tmp_path: Path) -> None:
        # The read from the start that follows one clear lands in the log of the next: it is dropped too.
        recorder = Recorder()
        output = await call_tool(
            [Shell[None](denied_commands=[], tools=['shell'], max_log_bytes=2000), recorder],
            'shell',
            {'command': 'echo first; until [ -e cleared ]; do sleep 0.01; done; echo after', 'timeout': 30},
            workspace=_ClearedAgainDuringTheReread(tmp_path),
        )
        assert '"exit_code": 0' in output
        assert recorder.output == f'first\n{NOTICE}\nafter\n'

    async def test_a_log_holding_only_the_notice_is_left_alone(self, tmp_path: Path) -> None:
        # The notice alone is over this log limit; clearing it again would loop at every check.
        output = await shell(
            tmp_path, {'command': "printf '%0100d' 0; sleep 30", 'mode': 'background'}, max_log_bytes=10
        )
        group = int(output.split('kill -- -', 1)[1].split('`', 1)[0])
        clears = _handle(output, 'Output').parent / 'log_clears'
        try:
            with anyio.fail_after(10):
                while not clears.exists() or clears.stat().st_size < 2:
                    await anyio.sleep(0.01)
            await anyio.sleep(0.3)  # several checks
            assert clears.read_bytes() == b'xx'
        finally:
            os.killpg(group, signal.SIGTERM)

    async def test_finished_event_reports_a_clear_while_the_log_is_measured(self, tmp_path: Path) -> None:
        recorder = Recorder()
        output = await call_tool(
            [Shell[None](denied_commands=[], tools=['shell'], max_log_bytes=2000), recorder],
            'shell',
            {'command': 'echo first; sleep 0.3', 'timeout': 30},
            workspace=_ClearedWhileCounted(tmp_path),
        )
        assert '"exit_code": 0' in output
        assert recorder.finished.truncated
        assert recorder.finished.total_lines is None

    async def test_logs_are_still_cleared_under_a_file_limit(self, tmp_path: Path) -> None:
        # The command fills `log_clears` up to the file limit (one `ulimit -f` block) before its output passes the
        # log limit, so the watcher's next write to it is past that limit, which applies to the command alone.
        # Linux stops a write that starts at the limit; macOS lets it through, so only Linux catches a limited watcher.
        command = (
            "for job in .pydantic-ai-harness/shell/*/; do ( trap '' XFSZ; printf '%01100d' 0 >> \"${job}log_clears\" ) "
            '2> /dev/null; wc -c < "${job}log_clears" > seeded.tmp; done; mv seeded.tmp seeded; '
            "printf '%0100d' 0; sleep 30"
        )
        toolset = Shell(max_file_bytes=20, max_log_bytes=10).get_toolset()
        async with toolset:
            started = await toolset.start_command(run_context(tmp_path), command)
            command_id = started.rsplit('ID: ', 1)[1]
            clears = tmp_path / '.pydantic-ai-harness' / 'shell' / command_id / 'log_clears'
            try:
                with anyio.fail_after(10):
                    while not (tmp_path / 'seeded').exists():
                        await anyio.sleep(0.01)
                    seeded = int((tmp_path / 'seeded').read_text())
                    while clears.stat().st_size < seeded + 2:
                        await anyio.sleep(0.01)
            finally:
                await toolset.stop_command(run_context(tmp_path), command_id)

    async def test_watcher_stops_without_its_signal(self, tmp_path: Path) -> None:
        # A `TERM` sent as the watcher starts can be lost before its trap is set; here every one is.
        with anyio.fail_after(10):
            output = await call_tool(
                [Shell[None](denied_commands=[], tools=['shell'], max_log_bytes=2000)],
                'shell',
                {'command': 'true', 'timeout': 30},
                workspace=_TermIgnored(tmp_path),
            )
        assert '"exit_code": 0' in output

    async def test_watcher_exits_once_the_wrapper_is_killed(self, tmp_path: Path) -> None:
        output = await shell(tmp_path, {'command': 'sleep 0.5', 'mode': 'background'}, max_log_bytes=2000)
        group = int(output.split('kill -- -', 1)[1].split('`', 1)[0])
        try:
            os.kill(int(output.split('PID: ', 1)[1].split(' ', 1)[0]), signal.SIGKILL)
            with anyio.fail_after(10):
                while True:
                    try:
                        os.killpg(group, 0)
                    except (ProcessLookupError, PermissionError):  # macOS: only zombies left
                        break
                    await anyio.sleep(0.05)
        finally:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(group, signal.SIGKILL)

    async def test_events_end_while_clears_keep_coming(self, tmp_path: Path) -> None:
        # Each restart after a clear reads the log again, so the events' byte budget still bounds the reads.
        recorder = Recorder()
        with anyio.fail_after(10):
            output = await call_tool(
                [Shell[None](denied_commands=[], tools=['shell']), recorder],
                'shell',
                {'command': "printf '%04000d\\n' 0", 'timeout': 30},
                workspace=_ClearedOnEveryRead(tmp_path),
            )
        assert '"exit_code": 0' in output
        assert recorder.finished.truncated
        assert recorder.finished.total_lines is None

    async def test_events_without_a_limit(self, tmp_path: Path) -> None:
        recorder = Recorder()
        output = await shell(
            tmp_path, {'command': 'echo hi', 'timeout': 60}, capabilities=[recorder], max_log_bytes=None
        )
        assert '"exit_code": 0' in output
        assert recorder.output == 'hi\n'
        assert not recorder.finished.truncated
        assert recorder.finished.total_lines == 1

    async def test_chatty_shell_command_keeps_running(self, tmp_path: Path) -> None:
        output = await shell(
            tmp_path, {'command': CHATTY.format(redirect=''), 'mode': 'background'}, max_log_bytes=2000
        )
        group = int(output.split('kill -- -', 1)[1].split('`', 1)[0])
        try:
            await _wait_for_notice(_handle(output, 'Output'))
            assert '"exit_code": null' in _handle(output, 'Status').read_text(encoding='utf-8')
        finally:
            # Even when an assertion fails, so a failure does not leave the loop writing behind.
            os.killpg(group, signal.SIGTERM)

    @pytest.mark.parametrize(('redirect', 'log_name'), [('', 'stdout.log'), (' >&2', 'stderr.log')])
    async def test_background_logs_are_bounded(self, tmp_path: Path, redirect: str, log_name: str) -> None:
        toolset = Shell(max_log_bytes=2000).get_toolset()
        async with toolset:
            started = await toolset.start_command(run_context(tmp_path), CHATTY.format(redirect=redirect))
            command_id = started.rsplit('ID: ', 1)[1]
            try:
                await _wait_for_notice(tmp_path / '.pydantic-ai-harness' / 'shell' / command_id / log_name)
                assert 'exit code' not in await toolset.check_command(run_context(tmp_path), command_id)
            finally:
                stopped = await toolset.stop_command(run_context(tmp_path), command_id)
            assert '[exit code: 143]' in stopped
