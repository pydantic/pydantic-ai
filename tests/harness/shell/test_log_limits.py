"""Real subprocess regressions for `Shell.max_log_bytes`, which bounds background and `shell` job logs."""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

import anyio
import pytest

from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import LocalWorkspaceBackend, Workspace
from pydantic_ai_harness.shell import Shell

from .._tool_calls import call_tool

pytestmark = [pytest.mark.skipif(os.name == 'nt', reason='POSIX shell commands and process groups')]

NOTICE = '[Log exceeded max_log_bytes=2000; earlier output was cleared.]'
CHATTY = "while :; do printf '%01000d\\n' 0{redirect}; sleep 0.01; done"
"""Writes about 100 KB a second until stopped."""


def _ctx(working_dir: Path) -> RunContext[None]:
    return RunContext[None](
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
        workspace=Workspace(LocalWorkspaceBackend(working_dir)),
    )


async def _shell(cwd: Path, arguments: dict[str, object]) -> str:
    capability = Shell[None](denied_commands=[], tools=['shell'], max_log_bytes=2000)
    return await call_tool([capability], 'shell', arguments, workspace=LocalWorkspaceBackend(cwd))


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
        output = await _shell(tmp_path, {'command': command, 'timeout': 60})
        assert '"exit_code": 0' in output
        log = Path(output.split('Output: ', 1)[1].split('\n', 1)[0])
        assert NOTICE in log.read_text(encoding='utf-8')
        assert log.stat().st_size < 1_000_000

    async def test_chatty_shell_command_keeps_running(self, tmp_path: Path) -> None:
        output = await _shell(tmp_path, {'command': CHATTY.format(redirect=''), 'mode': 'background'})
        log = Path(output.split('Output: ', 1)[1].split('\n', 1)[0])
        await _wait_for_notice(log)
        assert '"exit_code": null' in Path(output.split('Status: ', 1)[1].split('\n', 1)[0]).read_text(encoding='utf-8')
        os.killpg(int(output.split('kill -- -', 1)[1].split('`', 1)[0]), signal.SIGTERM)

    @pytest.mark.parametrize(('redirect', 'log_name'), [('', 'stdout.log'), (' >&2', 'stderr.log')])
    async def test_background_logs_are_bounded(self, tmp_path: Path, redirect: str, log_name: str) -> None:
        toolset = Shell(max_log_bytes=2000).get_toolset()
        async with toolset:
            started = await toolset.start_command(_ctx(tmp_path), CHATTY.format(redirect=redirect))
            command_id = started.rsplit('ID: ', 1)[1]
            await _wait_for_notice(tmp_path / '.pydantic-ai-harness' / 'shell' / command_id / log_name)
            assert 'exit code' not in await toolset.check_command(_ctx(tmp_path), command_id)
            assert '[exit code: 143]' in await toolset.stop_command(_ctx(tmp_path), command_id)
