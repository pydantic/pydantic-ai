"""Real subprocess regressions for per-file size limits applied by the workspace shell."""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

import anyio
import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import LocalWorkspaceBackend, Workspace
from pydantic_ai_harness.shell import Shell


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


if os.name == 'posix':  # pragma: no branch
    import resource


def writer(size: int) -> str:
    code = f"with open('output', 'wb') as output: output.write(b'x' * {size})"
    return f'{shlex.quote(sys.executable)} -c {shlex.quote(code)}'


class TestShellFileLimits:
    @pytest.mark.parametrize('limit', [0, -1, True, sys.maxsize + 1])
    def test_invalid(self, limit: int) -> None:
        with pytest.raises(ValueError, match='max_file_bytes'):
            Shell(max_file_bytes=limit).get_toolset()

    @pytest.mark.skipif(os.name != 'posix', reason='Requires POSIX resource limits')
    def test_persistent_rejected(self) -> None:
        with pytest.raises(ValueError, match='persistent shell'):
            Shell(max_file_bytes=1024, tools=['shell']).get_toolset()

    @pytest.mark.skipif(os.name != 'posix', reason='Requires POSIX resource limits')
    def test_persist_cwd_rejected(self) -> None:
        with pytest.raises(ValueError, match='persist_cwd'):
            Shell(max_file_bytes=1, persist_cwd=True).get_toolset()

    @pytest.mark.skipif(os.name != 'posix', reason='Requires POSIX resource limits')
    @pytest.mark.parametrize('background', [False, True])
    @pytest.mark.parametrize('size', [128, 8192])
    async def test_file_bound_and_recovery(self, tmp_path: Path, background: bool, size: int) -> None:
        parent_limits = resource.getrlimit(resource.RLIMIT_FSIZE)
        toolset = Shell(max_file_bytes=1024, env={'CUSTOM': 'kept'}).get_toolset()
        async with toolset:
            if background:
                started = await toolset.start_command(_ctx(tmp_path), writer(size))
                command_id = started.rsplit('ID: ', 1)[1]
                with anyio.fail_after(10):
                    while True:
                        result = await toolset.check_command(_ctx(tmp_path), command_id)
                        if '[exit code:' in result:
                            break
                        await anyio.sleep(0.01)  # pragma: lax no cover
            else:
                result = await toolset.run_command(_ctx(tmp_path), writer(size))
            assert (tmp_path / 'output').stat().st_size == min(size, 1024)
            if size > 1024:
                assert 'max_file_bytes=1024' in result
            else:
                assert 'max_file_bytes' not in result
            assert 'kept' in await toolset.run_command(_ctx(tmp_path), 'printf "$CUSTOM"')
        assert resource.getrlimit(resource.RLIMIT_FSIZE) == parent_limits
        (tmp_path / 'parent-output').write_bytes(b'x' * 8192)

    async def test_default_unlimited(self, tmp_path: Path) -> None:
        toolset = Shell().get_toolset()
        assert 'exit code' not in await toolset.run_command(_ctx(tmp_path), writer(8192))
        assert (tmp_path / 'output').stat().st_size == 8192

    @pytest.mark.skipif(os.name != 'posix', reason='Requires POSIX resource limits')
    async def test_child_hard_limit(self, tmp_path: Path) -> None:
        toolset = Shell(max_file_bytes=1024).get_toolset()
        code = 'import resource; print(resource.getrlimit(resource.RLIMIT_FSIZE))'
        result = await toolset.run_command(_ctx(tmp_path), f'{shlex.quote(sys.executable)} -c {shlex.quote(code)}')
        assert '(1024, 1024)' in result

    @pytest.mark.skipif(os.name != 'posix', reason='Requires POSIX resource limits')
    async def test_signal_diagnosis(self, tmp_path: Path) -> None:
        toolset = Shell(max_file_bytes=1024).get_toolset()
        result = await toolset.run_command(_ctx(tmp_path), 'exec yes x > output')
        assert 'File-size limit exceeded' in result
        assert (tmp_path / 'output').stat().st_size == 1024

    @pytest.mark.skipif(os.name != 'posix', reason='Requires POSIX resource limits')
    async def test_agent_run_clone(self, tmp_path: Path, anyio_backend: str) -> None:
        if anyio_backend != 'asyncio':  # pragma: no cover -- only asyncio runs here
            pytest.skip('Agent lifecycle requires asyncio')

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(parts=[ToolCallPart('run_command', {'command': writer(8192)})])
            assert 'max_file_bytes' in str(messages[-1])
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(
            FunctionModel(model),
            capabilities=[Shell(max_file_bytes=1024), LocalWorkspace(tmp_path)],
        )
        result = await agent.run('Write a file')
        assert result.output == 'done'
        assert (tmp_path / 'output').stat().st_size == 1024
