"""Bounded observation of the original background command through the public shell tools."""

import json
from collections.abc import AsyncIterator
from pathlib import Path

import anyio
import pytest
from anyio.abc import TaskStatus

from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import FileEntry, LocalWorkspaceBackend, Workspace
from pydantic_ai_harness.shell import Shell, ShellToolset

from .._tool_calls import call_tool

READINESS_WAIT_TIMEOUT = 60


class RecordingWorkspace(LocalWorkspaceBackend):
    """Record observation boundaries while keeping real processes and filesystem I/O."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.status_reads = 0
        self.output_stats: list[str] = []
        self.observed = anyio.Event()

    async def read_bytes(self, path: str) -> bytes:
        result = await super().read_bytes(path)
        if path.endswith('/status.json'):
            self.status_reads += 1
            if self.status_reads >= 2 and json.loads(result)['exit_code'] is None:
                self.observed.set()
        return result

    async def stat(self, path: str) -> FileEntry:
        if path.endswith(('/stdout.log', '/stderr.log')):
            self.output_stats.append(path)
        return await super().stat(path)


class GlitchingWorkspace(LocalWorkspaceBackend):
    """Fail the second read of a status file, as a rename in flight could, then carry on."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.status_reads = 0
        self.resumed = anyio.Event()

    async def read_bytes(self, path: str) -> bytes:
        if path.endswith('/status.json'):
            self.status_reads += 1
            if self.status_reads == 2:
                raise FileNotFoundError(path)
            if self.status_reads == 3:
                self.resumed.set()
        return await super().read_bytes(path)


class StoppingWorkspace(LocalWorkspaceBackend):
    """Hold a real stop before output collection removes the command's files."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.collecting = anyio.Event()
        self.release = anyio.Event()

    async def stat(self, path: str) -> FileEntry:
        if path.endswith('/stdout.log'):
            self.collecting.set()
            await self.release.wait()
        return await super().stat(path)


def context(backend: LocalWorkspaceBackend) -> RunContext[None]:
    return RunContext(
        deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], workspace=Workspace(backend)
    )


@pytest.fixture
async def running_command(
    tmp_path: Path,
) -> AsyncIterator[tuple[ShellToolset[None], RunContext[None], RecordingWorkspace, str]]:
    backend = RecordingWorkspace(tmp_path)
    ctx = context(backend)
    toolset = Shell[None](denied_commands=[]).get_toolset()
    start = await toolset.start_command(
        ctx,
        'echo launch >> launches; echo before; touch ready; '
        'while [ ! -f release ]; do sleep 0.05; done; echo after; echo error >&2',
    )
    command_id = start.split('ID: ')[1].strip()
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        while True:
            try:
                await ctx.workspace.stat(str(tmp_path / 'ready'))
            except FileNotFoundError:  # pragma: lax no cover
                await anyio.sleep(0.01)
            else:
                break
    backend.status_reads = 0
    backend.observed = anyio.Event()
    backend.output_stats.clear()
    try:
        yield toolset, ctx, backend, command_id
    finally:
        await toolset.stop_command(ctx, command_id)


class TestCheckCommandWait:
    async def test_command_finishing_during_the_wait_is_collected_once(
        self,
        running_command: tuple[ShellToolset[None], RunContext[None], RecordingWorkspace, str],
        tmp_path: Path,
    ) -> None:
        toolset, ctx, backend, command_id = running_command
        results: list[str] = []

        async def wait() -> None:
            results.append(
                await call_tool(
                    [Shell(tools=['check_command'])],
                    'check_command',
                    {'command_id': command_id, 'wait_seconds': 60},
                    workspace=backend,
                )
            )

        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(wait)
                await backend.observed.wait()
                await ctx.workspace.write_bytes(str(tmp_path / 'release'), b'')
        result = results[0]
        assert result.splitlines()[-2:] == ['[status: finished]', '[exit code: 0]']
        assert 'before' in result and 'after' in result and '[stderr]\nerror' in result
        assert len(backend.output_stats) == 2
        assert await ctx.workspace.read_bytes(str(tmp_path / 'launches')) == b'launch\n'
        # A command that has exited is reported after one read of its status, whatever the wait.
        backend.status_reads = 0
        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            assert await toolset.check_command(ctx, command_id, wait_seconds=270) == result
        assert backend.status_reads == 1

    @pytest.mark.parametrize('explicit_zero', [False, True])
    async def test_zero_checks_once(
        self,
        running_command: tuple[ShellToolset[None], RunContext[None], RecordingWorkspace, str],
        explicit_zero: bool,
    ) -> None:
        toolset, ctx, backend, command_id = running_command
        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            if explicit_zero:
                result = await toolset.check_command(ctx, command_id, wait_seconds=0)
            else:
                result = await toolset.check_command(ctx, command_id)
        assert result.endswith('[status: running]')
        assert backend.status_reads == 1
        assert len(backend.output_stats) == 2

    async def test_timeout_keeps_job_and_collects_output_once(
        self,
        running_command: tuple[ShellToolset[None], RunContext[None], RecordingWorkspace, str],
        tmp_path: Path,
    ) -> None:
        toolset, ctx, backend, command_id = running_command
        # Too short for the first poll interval, so the wait ends at its first sleep.
        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            result = await toolset.check_command(ctx, command_id, wait_seconds=0.001)
        assert result.endswith('[status: running]')
        assert 'before' in result
        assert backend.status_reads == 2
        assert len(backend.output_stats) == 2
        await ctx.workspace.write_bytes(str(tmp_path / 'release'), b'')
        result = await toolset.check_command(ctx, command_id, wait_seconds=READINESS_WAIT_TIMEOUT)
        assert result.splitlines()[-2:] == ['[status: finished]', '[exit code: 0]']
        assert 'after' in result and '[stderr]\nerror' in result
        assert await ctx.workspace.read_bytes(str(tmp_path / 'launches')) == b'launch\n'

    async def test_concurrent_stop_can_be_observed_before_cleanup(
        self,
        running_command: tuple[ShellToolset[None], RunContext[None], RecordingWorkspace, str],
        tmp_path: Path,
    ) -> None:
        toolset, ctx, backend, command_id = running_command
        stopping = StoppingWorkspace(tmp_path)
        results: list[str] = []
        stopped: list[str] = []
        observed = anyio.Event()

        async def wait() -> None:
            results.append(await toolset.check_command(ctx, command_id, wait_seconds=270))
            observed.set()

        async def stop() -> None:
            stopped.append(await toolset.stop_command(context(stopping), command_id))

        try:
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                async with anyio.create_task_group() as tasks:
                    tasks.start_soon(wait)
                    await backend.observed.wait()
                    tasks.start_soon(stop)
                    await stopping.collecting.wait()
                    await observed.wait()
                    assert results[0].splitlines()[-2:] == ['[status: finished]', '[exit code: 143]']
                    assert 'before' in results[0]
                    assert len(backend.output_stats) == 2
                    assert await ctx.workspace.read_bytes(str(tmp_path / 'launches')) == b'launch\n'
                    stopping.release.set()
        finally:
            stopping.release.set()
        assert stopped[0].splitlines()[-2:] == ['[stopped]', '[exit code: 143]']
        assert await toolset.check_command(ctx, command_id, wait_seconds=270) == (
            f'[Error: unknown command ID {command_id!r}]'
        )

    async def test_waited_output_remains_bounded_through_agent(self, tmp_path: Path) -> None:
        backend = LocalWorkspaceBackend(tmp_path)
        toolset = Shell[None](max_output_chars=200, denied_commands=[]).get_toolset()
        ctx = context(backend)
        start = await toolset.start_command(ctx, "printf '%0500d' 0")
        command_id = start.split('ID: ')[1].strip()
        try:
            result = await call_tool(
                [Shell(tools=['check_command'], max_output_chars=200)],
                'check_command',
                {'command_id': command_id, 'wait_seconds': 60},
                workspace=backend,
            )
            assert len(result) == 200
            assert 'output truncated' in result
            assert result.splitlines()[-2:] == ['[status: finished]', '[exit code: 0]']
        finally:
            await toolset.stop_command(ctx, command_id)

    async def test_cancelling_observation_keeps_job_and_files(
        self,
        running_command: tuple[ShellToolset[None], RunContext[None], RecordingWorkspace, str],
        tmp_path: Path,
    ) -> None:
        toolset, ctx, backend, command_id = running_command
        results: list[str] = []

        async def observe(*, task_status: TaskStatus[anyio.CancelScope]) -> None:
            with anyio.CancelScope() as scope:
                task_status.started(scope)
                results.append(await toolset.check_command(ctx, command_id, wait_seconds=270))

        async with anyio.create_task_group() as tasks:
            scope = await tasks.start(observe)
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                await backend.observed.wait()
            scope.cancel()
        assert results == []
        assert backend.output_stats == []
        assert (await toolset.check_command(ctx, command_id)).endswith('[status: running]')
        await ctx.workspace.write_bytes(str(tmp_path / 'release'), b'')
        result = await toolset.check_command(ctx, command_id, wait_seconds=READINESS_WAIT_TIMEOUT)
        assert 'before' in result and 'after' in result
        assert result.endswith('[exit code: 0]')
        assert await ctx.workspace.read_bytes(str(tmp_path / 'launches')) == b'launch\n'

    async def test_wait_ends_when_the_command_is_removed(self, tmp_path: Path) -> None:
        """`stop_command` from another call removes a command's files while a wait is polling them."""
        backend = RecordingWorkspace(tmp_path)
        ctx = context(backend)
        toolset = Shell[None](denied_commands=[]).get_toolset()
        # The command removes its own job directory, as `stop_command` does, once the test says so.
        start = await toolset.start_command(
            ctx, 'while [ ! -f remove ]; do sleep 0.05; done; rm -rf .pydantic-ai-harness/shell/*'
        )
        command_id = start.split('ID: ')[1].strip()
        results: list[str] = []

        async def wait() -> None:
            results.append(await toolset.check_command(ctx, command_id, wait_seconds=270))

        try:
            # A wait that polled on until its deadline would run past this guard.
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                async with anyio.create_task_group() as tasks:
                    tasks.start_soon(wait)
                    await backend.observed.wait()
                    await ctx.workspace.write_bytes(str(tmp_path / 'remove'), b'')
        finally:
            await toolset.stop_command(ctx, command_id)
        assert results == [f'[Error: unknown command ID {command_id!r}]']

    async def test_wait_continues_through_a_status_read_that_briefly_fails(self, tmp_path: Path) -> None:
        backend = GlitchingWorkspace(tmp_path)
        ctx = context(backend)
        toolset = Shell[None](denied_commands=[]).get_toolset()
        start = await toolset.start_command(ctx, 'while [ ! -f release ]; do sleep 0.05; done; echo after')
        command_id = start.split('ID: ')[1].strip()
        results: list[str] = []

        async def wait() -> None:
            results.append(await toolset.check_command(ctx, command_id, wait_seconds=270))

        try:
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                async with anyio.create_task_group() as tasks:
                    tasks.start_soon(wait)
                    # The command's handle is still there, so the missing status is not taken for removal.
                    await backend.resumed.wait()
                    await ctx.workspace.write_bytes(str(tmp_path / 'release'), b'')
        finally:
            await toolset.stop_command(ctx, command_id)
        assert results[0].splitlines()[-2:] == ['[status: finished]', '[exit code: 0]']
        assert 'after' in results[0]

    @pytest.mark.parametrize('wait_seconds', [-1, 270.1, float('nan'), float('inf'), float('-inf')])
    async def test_out_of_range_wait_is_rejected_before_the_lookup(self, tmp_path: Path, wait_seconds: float) -> None:
        backend = LocalWorkspaceBackend(tmp_path)
        message = 'wait_seconds must be between 0 and 270 seconds.'
        result = await call_tool(
            [Shell(tools=['check_command'])],
            'check_command',
            {'command_id': 'unknown', 'wait_seconds': wait_seconds},
            workspace=backend,
        )
        assert result == message
        with pytest.raises(ModelRetry, match=message):
            await Shell[None]().get_toolset().check_command(context(backend), 'unknown', wait_seconds=wait_seconds)

    async def test_public_schema_and_unknown_id(self, tmp_path: Path) -> None:
        toolset = Shell[None]().get_toolset()
        ctx = context(LocalWorkspaceBackend(tmp_path))
        tools = await toolset.get_tools(ctx)
        schema = tools['check_command'].tool_def.parameters_json_schema
        parameter = schema['properties']['wait_seconds']
        assert parameter['type'] == 'number'
        assert parameter['default'] == 0
        assert 'from 0 to 270' in parameter['description']
        assert schema['required'] == ['command_id']
        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            assert 'unknown command ID' in await toolset.check_command(ctx, 'unknown', wait_seconds=270)
