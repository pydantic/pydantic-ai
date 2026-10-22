"""`read_memory` steps aside for `FileSystem`'s `read_file` when a `FileStore` is in the run's workspace."""

from __future__ import annotations

from pathlib import Path

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, LocalWorkspace, PrefixTools
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.memory import FileStore, InMemoryStore, Memory, MemoryStore


class _Recorder:
    def __init__(self, call: ToolCallPart | None = None) -> None:
        self.call = call
        self.offered: list[set[str]] = []
        self.instructions: list[str | None] = []
        self.user_text: list[str] = []

    def respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.offered.append({tool.name for tool in info.function_tools})
        self.instructions.append(info.instructions)
        self.user_text.extend(
            str(part.content)
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, UserPromptPart)
        )
        returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if returns:
            return ModelResponse(parts=[TextPart(str(returns[0].content))])
        if self.call is not None:
            return ModelResponse(parts=[self.call])
        return ModelResponse(parts=[TextPart('done')])


async def _run(
    workspace: Path, *capabilities: AbstractCapability[None], call: ToolCallPart | None = None
) -> tuple[_Recorder, str]:
    recorder = _Recorder(call)
    agent = Agent(
        FunctionModel(recorder.respond), deps_type=type(None), capabilities=[LocalWorkspace(workspace), *capabilities]
    )
    result = await agent.run('go')
    return recorder, result.output


class TestReaderStepsAside:
    async def test_read_file_replaces_read_memory(self, tmp_path: Path) -> None:
        notes = tmp_path / '.agent-memory' / 'main' / 'notes.md'
        notes.parent.mkdir(parents=True)
        notes.write_text('the deploy key rotates on Mondays\n')
        recorder, output = await _run(
            tmp_path,
            Memory(store=FileStore('.agent-memory')),
            FileSystem(),
            call=ToolCallPart('read_file', {'path': '.agent-memory/main/notes.md'}),
        )
        assert all('read_memory' not in tools and 'read_file' in tools for tools in recorder.offered)
        assert any('These files are in `.agent-memory/main/` in the workspace.' in text for text in recorder.user_text)
        assert 'the deploy key rotates on Mondays' in output

    async def test_prefixed_read_file_replaces_read_memory(self, tmp_path: Path) -> None:
        recorder, _ = await _run(
            tmp_path, Memory(store=FileStore('.agent-memory')), PrefixTools(FileSystem(), prefix='repo')
        )
        assert all('read_memory' not in tools and 'repo_read_file' in tools for tools in recorder.offered)

    async def test_dropped_notebook_lines_point_at_read_file(self, tmp_path: Path) -> None:
        notebook = tmp_path / '.agent-memory' / 'main' / 'MEMORY.md'
        notebook.parent.mkdir(parents=True)
        notebook.write_text('\n'.join(f'- fact {i}' for i in range(10)) + '\n')
        recorder, _ = await _run(tmp_path, Memory(store=FileStore('.agent-memory'), max_lines=2), FileSystem())
        assert any(
            'use read_file(path=".agent-memory/main/MEMORY.md") for the full notebook' in text
            for text in recorder.user_text
        )

    async def test_store_resolver_returning_a_run_workspace_file_store(self, tmp_path: Path) -> None:
        def resolve(ctx: RunContext[None]) -> MemoryStore:
            return FileStore('.agent-memory')

        notebook = tmp_path / '.agent-memory' / 'tenant-a' / 'main' / 'MEMORY.md'
        notebook.parent.mkdir(parents=True)
        notebook.write_text('- a fact\n')
        recorder, _ = await _run(tmp_path, Memory(store_resolver=resolve, namespace='tenant-a'), FileSystem())
        assert all('read_memory' not in tools for tools in recorder.offered)
        assert any('`.agent-memory/tenant-a/main/`' in text for text in recorder.user_text)


class TestGuidanceStaysStatic:
    def test_guidance_is_static_for_every_store(self) -> None:
        def resolve(ctx: RunContext[None]) -> MemoryStore:
            return FileStore('.agent-memory')  # pragma: no cover

        default = Memory[None](store=InMemoryStore()).get_instructions()
        assert isinstance(default, str)
        assert Memory[None](store=FileStore('.agent-memory')).get_instructions() == default
        assert Memory[None](store_resolver=resolve).get_instructions() == default

    async def test_instructions_match_with_and_without_file_system(self, tmp_path: Path) -> None:
        without, _ = await _run(tmp_path, Memory(store=FileStore('.agent-memory')))
        with_file_system, _ = await _run(tmp_path, Memory(store=FileStore('.agent-memory')), FileSystem())
        assert without.instructions[0] is not None
        assert with_file_system.instructions == without.instructions


class TestReaderKept:
    async def test_without_file_system(self, tmp_path: Path) -> None:
        notebook = tmp_path / '.agent-memory' / 'main' / 'MEMORY.md'
        notebook.parent.mkdir(parents=True)
        notebook.write_text('- a fact\n')
        recorder, _ = await _run(tmp_path, Memory(store=FileStore('.agent-memory')))
        assert all('read_memory' in tools for tools in recorder.offered)
        assert recorder.user_text and not any('in the workspace.' in text for text in recorder.user_text)

    async def test_without_injection(self, tmp_path: Path) -> None:
        recorder, _ = await _run(tmp_path, Memory(store=FileStore('.agent-memory'), inject_memory=False), FileSystem())
        assert all('read_memory' in tools and 'read_file' in tools for tools in recorder.offered)

    async def test_when_the_guidance_leaves_no_room_for_the_snapshot(self, tmp_path: Path) -> None:
        recorder, _ = await _run(tmp_path, Memory(store=FileStore('.agent-memory'), max_tokens=100), FileSystem())
        assert all('read_memory' in tools and 'read_file' in tools for tools in recorder.offered)

    @pytest.mark.parametrize('store_kind', ['own_workspace', 'in_memory', 'outside_working_dir'])
    async def test_store_the_file_tools_cannot_see(self, tmp_path: Path, store_kind: str) -> None:
        elsewhere = tmp_path / 'elsewhere'
        elsewhere.mkdir()
        work = tmp_path / 'work'
        work.mkdir()
        store: MemoryStore
        if store_kind == 'own_workspace':
            store = FileStore('.', workspace=LocalWorkspaceBackend(elsewhere))
        elif store_kind == 'in_memory':
            store = InMemoryStore()
        else:
            store = FileStore('../elsewhere')
        recorder, _ = await _run(work, Memory(store=store), FileSystem())
        assert all('read_memory' in tools and 'read_file' in tools for tools in recorder.offered)
