"""`Shell` background jobs under each POSIX shell CI provides, against commands that work against their own supervisor.

A job's launcher, supervisor and log watcher are `sh` scripts, and `Job` finds `sh` on the workspace's `PATH`, so
a `PATH` whose `sh` is `dash` or `bash` runs the whole job under that shell. Each case asserts what a model relies
on: the supervisor exits, the exit code is published, and nothing the job started is left running.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import anyio
import pytest

from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import LocalWorkspaceBackend, Workspace
from pydantic_ai_harness.shell import Shell, ShellToolset
from tests._timeouts import READINESS_WAIT_TIMEOUT

from .._tool_calls import call_tool

pytestmark = [pytest.mark.skipif(os.name == 'nt', reason='POSIX shell commands and process groups')]

SHELLS = ('dash', 'bash')
"""Each shell that is `/bin/sh` on a CI runner: `dash` on Ubuntu, `bash` (3.2) on macOS."""


@dataclass(frozen=True)
class Host:
    shell: str
    working_dir: Path
    backend: LocalWorkspaceBackend

    def context(self) -> RunContext[None]:
        return RunContext[None](
            deps=None,
            model=TestModel(),
            usage=RunUsage(),
            prompt=None,
            messages=[],
            run_step=0,
            workspace=Workspace(self.backend),
        )


@pytest.fixture(params=SHELLS)
def host(request: pytest.FixtureRequest, tmp_path: Path) -> Host:
    shell: str = request.param
    path = shutil.which(shell)
    if path is None:  # pragma: no cover - CI runners have both
        pytest.skip(f'`{shell}` is not installed')
    shim = tmp_path / 'shim'
    shim.mkdir()
    (shim / 'sh').symlink_to(path)
    working_dir = tmp_path / 'work'
    working_dir.mkdir()
    backend = LocalWorkspaceBackend(working_dir, env={'PATH': f'{shim}{os.pathsep}{os.environ["PATH"]}'})
    return Host(shell, working_dir, backend)


@pytest.fixture
def writable_after(host: Host) -> Iterator[Host]:
    """`host` for a command that takes write permission away, given back afterwards so it can be removed."""
    if os.geteuid() == 0:  # pragma: no cover
        pytest.skip('root writes regardless of mode bits')
    yield host
    for path in (host.working_dir, *host.working_dir.rglob('*')):
        path.chmod(path.stat().st_mode | 0o200)


def _toolset() -> ShellToolset[None]:
    return ShellToolset(
        allowed_commands=[],
        denied_commands=[],
        denied_operators=[],
        default_timeout=10.0,
        max_output_chars=50_000,
        persist_cwd=False,
        allow_interactive=False,
    )


def _command_id(started: str) -> str:
    return started.split('ID: ')[1].strip()


def _job_directory(host: Host, command_id: str) -> Path:
    return host.working_dir / '.pydantic-ai-harness' / 'shell' / command_id


async def _wait_for_exit(pid: int) -> None:
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        # real-wait: nothing announces the exit of a process this test did not start
        while True:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            await anyio.sleep(0.01)  # pragma: lax no cover


async def _wait_for_file(path: Path) -> str:
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        # real-wait: the job writes the file from another process, which nothing in this one observes
        while not (text := path.read_text(encoding='utf-8') if path.exists() else ''):
            await anyio.sleep(0.01)  # pragma: lax no cover
    return text


@pytest.mark.parametrize('term', ['handled', 'ignored'])
async def test_supervisor_exits_when_its_job_directory_is_removed(host: Host, term: str) -> None:
    """With nowhere to create `watch.stop`, the supervisor still stops its log watcher and exits.

    TERM is then the watcher's only stop. A job launched with TERM ignored never receives it, the loss `dash` also
    causes when the signal arrives before the watcher resets its inherited trap.
    """
    previous = signal.signal(signal.SIGTERM, signal.SIG_IGN if term == 'ignored' else signal.SIG_DFL)
    try:
        output = await call_tool(
            [Shell[None](denied_commands=[], allow_interactive=True, tools=['shell'])],
            'shell',
            {'command': 'rm -rf .pydantic-ai-harness', 'timeout': 0.2},
            workspace=host.backend,
        )
    finally:
        signal.signal(signal.SIGTERM, previous)
    pid = int(output.split('PID: ')[1].split(' ')[0])
    group = int(output.split('kill -- -')[1].split('`')[0])
    try:
        await _wait_for_exit(pid)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(group, signal.SIGKILL)


async def test_exit_code_is_published_when_the_job_directory_is_read_only(writable_after: Host) -> None:
    host, toolset = writable_after, _toolset()
    command_id = _command_id(await toolset.start_command(host.context(), 'chmod 500 .pydantic-ai-harness/shell/*'))
    status = json.loads(await _wait_for_file(_job_directory(host, command_id) / 'status.json'))
    await _wait_for_exit(status['pid'])
    result = await toolset.check_command(host.context(), command_id)
    assert result.splitlines()[-2:] == ['[status: finished]', '[exit code: 0]']


async def test_stop_kills_a_job_that_ignores_term(host: Host, monkeypatch: pytest.MonkeyPatch) -> None:
    toolset = _toolset()
    ready = host.working_dir / 'ready'
    started = await toolset.start_command(
        host.context(), f"trap '' TERM; echo $$ > {ready.name}; while :; do sleep 1; done"
    )
    pid = int(await _wait_for_file(ready))
    monkeypatch.setattr('pydantic_ai_harness.shell._jobs._KILL_GRACE_PERIOD', 0.2)
    result = await toolset.stop_command(host.context(), _command_id(started))
    assert result.endswith('[stopped]')
    await _wait_for_exit(pid)


async def test_a_job_that_exits_at_once_is_reported_and_stopped(host: Host) -> None:
    """The job runs under the `sh` on the workspace's `PATH`, and a stop after it has exited reports its exit code."""
    toolset = _toolset()
    command_id = _command_id(
        await toolset.start_command(
            host.context(), 'if [ -n "${BASH_VERSION-}" ]; then printf bash; else printf dash; fi; exit 3'
        )
    )
    status = json.loads(await _wait_for_file(_job_directory(host, command_id) / 'status.json'))
    await _wait_for_exit(status['pid'])
    assert (await toolset.check_command(host.context(), command_id)).splitlines() == [
        '[stdout]',
        host.shell,
        '[status: finished]',
        '[exit code: 3]',
    ]
    result = await toolset.stop_command(host.context(), command_id)
    assert result.splitlines()[-2:] == ['[stopped]', '[exit code: 3]']
    assert not _job_directory(host, command_id).exists()
