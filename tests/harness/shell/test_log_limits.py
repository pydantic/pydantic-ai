"""Real subprocess regressions for `Shell.max_log_bytes`, which bounds background and `shell` job logs."""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

import anyio
import pytest

from pydantic_ai_harness.shell import Shell

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
        # The reader backs off while the command is quiet, so the log is emptied and grows past what was
        # already read before its next read; it must still start again from the notice.
        command = (
            "printf '%01500d\\n' 0; sleep 1.5; printf '%01500d\\n' 0; sleep 0.2; "
            "printf '%01800d\\n' 0; sleep 1; echo after"
        )
        recorder = Recorder()
        output = await shell(tmp_path, {'command': command, 'timeout': 60}, capabilities=[recorder], max_log_bytes=2000)
        assert '"exit_code": 0' in output
        assert recorder.output.startswith('0' * 1500)
        assert recorder.output.endswith(f'{NOTICE}\n{"0" * 1800}\nafter\n')
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
