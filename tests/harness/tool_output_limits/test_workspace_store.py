"""Spilled tool results live in the run's workspace by default."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.exceptions import ToolFailed, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.workspaces import (
    FileEntry,
    LocalWorkspaceBackend,
    Workspace,
    WorkspaceError,
    WorkspaceRef,
)
from pydantic_ai_harness.tool_output_limits import (
    READ_TOOL_NAME,
    Band,
    LocalFileStore,
    Spill,
    ToolOutputLimits,
    Truncate,
    WorkspaceStore,
)


def _returns(messages: Sequence[ModelMessage], tool_name: str) -> list[ToolReturnPart]:
    return [
        part
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == tool_name
    ]


class _FilesystemOnly:
    """A workspace backend with file operations but no command execution, like a mounted bucket."""

    def __init__(self, working_dir: Path) -> None:
        self._local = LocalWorkspaceBackend(working_dir)

    @property
    def ref(self) -> WorkspaceRef | None:
        return self._local.ref  # pragma: no cover - not used by the store

    async def working_dir(self) -> str:
        return await self._local.working_dir()

    async def read_bytes(self, path: str) -> bytes:
        return await self._local.read_bytes(path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        await self._local.write_bytes(path, data)

    async def stat(self, path: str) -> FileEntry:
        return await self._local.stat(path)  # pragma: no cover - not used by the store

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        return await self._local.list_dir(path)  # pragma: no cover - not used by the store

    async def make_dir(self, path: str) -> None:
        await self._local.make_dir(path)

    async def remove(self, path: str) -> None:
        await self._local.remove(path)  # pragma: no cover - not used by the store

    async def exists(self, path: str) -> bool:
        return await self._local.exists(path)


class _FailingRead(LocalWorkspaceBackend):
    async def read_bytes(self, path: str) -> bytes:
        raise WorkspaceError('sandbox refused the read')


def _call_big_tool(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Call `big_tool` once, then finish."""
    return ModelResponse(parts=[TextPart('done') if _returns(messages, 'big_tool') else ToolCallPart('big_tool', {})])


@dataclasses.dataclass
class _Ctx:
    workspace: Workspace


class TestDefaultStore:
    async def test_spill_and_read_back_through_the_workspace(self, tmp_path: Path):
        work = tmp_path / 'work'
        work.mkdir()
        payload = '\n'.join(f'line {i}' for i in range(500))

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if read := _returns(messages, READ_TOOL_NAME):
                return ModelResponse(parts=[TextPart(str(read[0].content))])
            if spilled := _returns(messages, 'big_tool'):
                assert spilled[0].metadata is not None
                handle = spilled[0].metadata['overflow_handle']
                return ModelResponse(parts=[ToolCallPart(READ_TOOL_NAME, {'handle': handle, 'limit': 2})])
            return ModelResponse(parts=[ToolCallPart('big_tool', {})])

        agent = Agent(
            FunctionModel(respond),
            capabilities=[ToolOutputLimits(bands=[Band(over=100, action=Spill())]), LocalWorkspace(work)],
        )

        @agent.tool_plain
        def big_tool() -> str:
            return payload

        result = await agent.run('go')

        [spilled] = _returns(result.all_messages(), 'big_tool')
        assert spilled.metadata is not None
        handle = spilled.metadata['overflow_handle']
        assert handle.startswith(f'{work}/.pydantic-ai-harness/tool-output/')
        assert Path(handle).read_text(encoding='utf-8') == payload
        assert (work / '.pydantic-ai-harness' / '.gitignore').read_text() == '*\n'
        assert result.output == f'[handle {handle!r}: 500 matching line(s); showing 2]\nline 0\nline 1'

    async def test_host_store_spills_without_a_workspace(self, tmp_path: Path):
        payload = '\n'.join(f'line {i}' for i in range(500))

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if read := _returns(messages, READ_TOOL_NAME):
                return ModelResponse(parts=[TextPart(str(read[0].content))])
            if spilled := _returns(messages, 'big_tool'):
                assert spilled[0].metadata is not None
                handle = spilled[0].metadata['overflow_handle']
                return ModelResponse(parts=[ToolCallPart(READ_TOOL_NAME, {'handle': handle, 'limit': 2})])
            return ModelResponse(parts=[ToolCallPart('big_tool', {})])

        limits = ToolOutputLimits(bands=[Band(over=100, action=Spill())], store=LocalFileStore(base_dir=tmp_path))
        agent = Agent(FunctionModel(respond), capabilities=[limits])

        @agent.tool_plain
        def big_tool() -> str:
            return payload

        result = await agent.run('go')

        [spilled] = _returns(result.all_messages(), 'big_tool')
        assert spilled.metadata is not None
        handle = spilled.metadata['overflow_handle']
        assert (tmp_path / handle).read_text() == payload
        assert result.output == f'[handle {handle!r}: 500 matching line(s); showing 2]\nline 0\nline 1'

    @pytest.mark.parametrize('store', [None, WorkspaceStore()], ids=['default', 'explicit'])
    async def test_spilling_without_a_workspace_fails_the_run(self, store: WorkspaceStore | None):
        agent = Agent(FunctionModel(_call_big_tool), capabilities=[ToolOutputLimits(store=store)])
        with pytest.raises(UserError, match=r'none is attached to this run.*store=LocalFileStore\(\)'):
            await agent.run('go')

    async def test_store_with_its_own_workspace_needs_none_from_the_run(self, tmp_path: Path):
        store = WorkspaceStore(workspace=LocalWorkspaceBackend(tmp_path))
        agent = Agent(FunctionModel(_call_big_tool), capabilities=[ToolOutputLimits(store=store)])

        @agent.tool_plain
        def big_tool() -> str:
            return 'x' * 20_000

        result = await agent.run('go')

        [spilled] = _returns(result.all_messages(), 'big_tool')
        assert spilled.metadata is not None
        assert spilled.metadata['overflow_handle'].startswith(f'{tmp_path}/.pydantic-ai-harness/tool-output/')

    def test_store_workspace_must_be_a_backend(self, tmp_path: Path):
        with pytest.raises(TypeError, match=r'takes a workspace backend.*LocalWorkspaceBackend\('):
            WorkspaceStore(workspace=LocalWorkspace(tmp_path))  # pyright: ignore[reportArgumentType]

    async def test_read_only_workspace_warns_and_falls_back(self, tmp_path: Path):
        limits = ToolOutputLimits(bands=[Band(over=100, action=Spill(then=Truncate(max_chars=150)))])
        agent = Agent(FunctionModel(_call_big_tool), capabilities=[limits, LocalWorkspace(tmp_path, read_only=True)])

        @agent.tool_plain
        def big_tool() -> str:
            return 'x' * 1_000

        with pytest.warns(UserWarning, match="could not spill a 'big_tool' result"):
            result = await agent.run('go')

        [part] = _returns(result.all_messages(), 'big_tool')
        assert isinstance(part.content, str) and 'truncated' in part.content

    async def test_spill_without_a_workspace_warns_and_falls_back(self):
        # A deferred-loaded capability skips `before_run`, so the spill itself meets the missing workspace.
        class Deferred(ToolOutputLimits):
            async def before_run(self, ctx: RunContext[Any]) -> None:
                pass

        limits = Deferred(bands=[Band(over=100, action=Spill(then=Truncate(max_chars=150)))])
        agent = Agent(FunctionModel(_call_big_tool), capabilities=[limits])

        @agent.tool_plain
        def big_tool() -> str:
            return 'x' * 1_000

        with pytest.warns(UserWarning, match=r"could not spill a 'big_tool' result: No workspace is attached"):
            result = await agent.run('go')

        [part] = _returns(result.all_messages(), 'big_tool')
        assert isinstance(part.content, str) and 'truncated' in part.content

    async def test_read_without_workspace_guides_the_model(self):
        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if read := _returns(messages, READ_TOOL_NAME):
                return ModelResponse(parts=[TextPart(str(read[0].content))])
            return ModelResponse(parts=[ToolCallPart(READ_TOOL_NAME, {'handle': 'call-1'})])

        limits = ToolOutputLimits(bands=[Band(over=100, action=Truncate())])
        result = await Agent(FunctionModel(respond), capabilities=[limits]).run('go')
        assert result.output.startswith("[No stored tool result for handle 'call-1'.")

    async def test_workspace_failure_on_read_is_a_failed_tool_call(self, tmp_path: Path):
        toolset = ToolOutputLimits(store=WorkspaceStore()).get_toolset()
        assert toolset is not None
        tool = toolset.tools[READ_TOOL_NAME]  # type: ignore[union-attr]
        ctx = _Ctx(workspace=Workspace(_FailingRead(tmp_path)))
        with pytest.raises(ToolFailed, match='sandbox refused the read'):
            await tool.function(ctx, 'run/call.0')  # type: ignore[attr-defined]


class TestWorkspaceStore:
    async def test_keys_become_paths_in_the_store_directory(self, tmp_path: Path):
        workspace = Workspace(LocalWorkspaceBackend(tmp_path))
        store = WorkspaceStore()
        directory = f'{tmp_path.resolve()}/.pydantic-ai-harness/tool-output'
        handle = await store.write(workspace, 'run-1/../call 1.0', b'payload')
        assert handle == f'{directory}/run-1/_/call_1.0'
        assert await store.read(workspace, handle) == b'payload'
        assert await store.read(workspace, 'run-1/_/call_1.0') == b'payload'
        assert await store.write(workspace, '', b'empty') == f'{directory}/_'

    @pytest.mark.parametrize('handle', ['../secret.txt', '/etc/passwd', '.'])
    async def test_read_outside_directory_is_refused(self, tmp_path: Path, handle: str):
        (tmp_path / 'secret.txt').write_text('secret')
        workspace = Workspace(LocalWorkspaceBackend(tmp_path))
        with pytest.raises(PermissionError, match='outside the store directory'):
            await WorkspaceStore().read(workspace, handle)

    async def test_symlink_in_the_store_directory_is_refused(self, tmp_path: Path):
        (tmp_path / 'secret.txt').write_text('secret')
        workspace = Workspace(LocalWorkspaceBackend(tmp_path))
        store = WorkspaceStore()
        await store.write(workspace, 'run/call.0', b'payload')
        (tmp_path / '.pydantic-ai-harness' / 'tool-output' / 'leak').symlink_to(tmp_path / 'secret.txt')
        with pytest.raises(PermissionError, match='outside the store directory'):
            await store.read(workspace, 'leak')

    @pytest.mark.parametrize('link', ['.pydantic-ai-harness', '.pydantic-ai-harness/tool-output'])
    async def test_symlinked_store_directory_is_refused_before_anything_is_created(self, tmp_path: Path, link: str):
        outside = tmp_path / 'outside'
        (outside / 'run').mkdir(parents=True)
        (outside / 'run' / 'call.0').write_bytes(b'secret')
        (tmp_path / 'work' / link).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / 'work' / link).symlink_to(outside)
        workspace = Workspace(LocalWorkspaceBackend(tmp_path / 'work'))
        store = WorkspaceStore()
        with pytest.raises(WorkspaceError, match='is a symlink'):
            await store.read(workspace, 'run/call.0')
        with pytest.raises(WorkspaceError, match='is a symlink'):
            await store.write(workspace, 'run/new.0', b'payload')
        assert sorted(path.name for path in outside.rglob('*')) == ['call.0', 'run']

    async def test_dangling_gitignore_symlink_is_not_written_through(self, tmp_path: Path):
        outside = tmp_path / 'outside.txt'
        (tmp_path / 'work' / '.pydantic-ai-harness').mkdir(parents=True)
        (tmp_path / 'work' / '.pydantic-ai-harness' / '.gitignore').symlink_to(outside)
        workspace = Workspace(LocalWorkspaceBackend(tmp_path / 'work'))
        with pytest.raises(WorkspaceError, match='is a symlink'):
            await WorkspaceStore().write(workspace, 'run/new.0', b'payload')
        assert not outside.exists()

    async def test_missing_handle_raises_file_not_found(self, tmp_path: Path):
        workspace = Workspace(LocalWorkspaceBackend(tmp_path))
        with pytest.raises(FileNotFoundError):
            await WorkspaceStore().read(workspace, 'run/missing.0')

    async def test_filesystem_only_workspace_uses_working_dir(self, tmp_path: Path):
        workspace = Workspace(_FilesystemOnly(tmp_path))
        store = WorkspaceStore()
        handle = await store.write(workspace, 'run/call.0', b'data')
        assert handle == f'{tmp_path.resolve()}/.pydantic-ai-harness/tool-output/run/call.0'
        assert await store.read(workspace, handle) == b'data'
