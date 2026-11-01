"""`read_memory` steps aside for `FileSystem`'s `read_file` when a `FileStore` is in the run's workspace."""

from __future__ import annotations

import warnings
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, LocalWorkspace, PrefixTools, Toolset
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.realtime import RealtimeModel, RealtimeModelProfile, RealtimeModelSettings
from pydantic_ai.realtime.codec import RealtimeCodecEvent, RealtimeConnection, RealtimeInput
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.memory import FileStore, InMemoryStore, Memory, MemoryStore
from tests.function_model_prefix import TOOLS_CHANGE_MID_RUN
from tests.realtime_connections import InferredConnection


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

    async def test_file_system_adds_only_the_directory_block(self, tmp_path: Path) -> None:
        without, _ = await _run(tmp_path, Memory(store=FileStore('.agent-memory')))
        with_file_system, _ = await _run(tmp_path, Memory(store=FileStore('.agent-memory')), FileSystem())
        assert without.instructions[0] is not None
        directory = 'Memory files are in `.agent-memory/main/` in the workspace; read them with the file-reading tool.'
        assert directory not in without.instructions[0]
        [instructions] = with_file_system.instructions
        assert instructions is not None
        assert instructions.startswith(without.instructions[0])
        assert '<context id="toolset:memory:directory">\n' + directory in instructions


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


class TestDirectoryBlock:
    async def test_tools_and_instructions_are_the_same_on_every_request_of_a_promptless_run(
        self, tmp_path: Path
    ) -> None:
        offered: list[set[str]] = []
        instructions: list[str | None] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            offered.append({tool.name for tool in info.function_tools})
            instructions.append(info.instructions)
            if len(offered) == 1:
                return ModelResponse(parts=[ToolCallPart('list_directory', {'path': '.'}, tool_call_id='list')])
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(
            FunctionModel(respond),
            deps_type=type(None),
            capabilities=[LocalWorkspace(tmp_path), FileSystem(), Memory(store=FileStore('.agent-memory'))],
        )
        await agent.run(message_history=[ModelRequest(parts=[UserPromptPart('go')])])

        assert len(offered) == 2
        assert offered[0] == offered[1]
        assert 'read_memory' not in offered[0]
        assert instructions[0] == instructions[1]
        assert instructions[0] is not None and '`.agent-memory/main/`' in instructions[0]

    async def test_memory_loaded_mid_run_names_its_directory(self, tmp_path: Path) -> None:
        offered: list[set[str]] = []
        seen: list[str] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            offered.append({tool.name for tool in info.function_tools})
            # A block that appears mid-run is appended to the history rather than the instructions.
            seen.append(f'{info.instructions}\n{messages!r}')
            if len(offered) == 1:
                return ModelResponse(parts=[ToolCallPart('load_capability', {'id': 'memory'}, tool_call_id='load')])
            return ModelResponse(parts=[TextPart('done')])

        memory = Memory(
            store=FileStore('.agent-memory'), namespace='acme', description='Long-term memory.', defer_loading=True
        )
        agent = Agent(
            FunctionModel(respond),
            deps_type=type(None),
            capabilities=[LocalWorkspace(tmp_path), FileSystem(), memory],
        )
        await agent.run('go')

        # Loaded mid-run, Memory records no snapshot, so its instructions are what give the directory.
        assert len(offered) == 2
        assert 'read_memory' not in offered[1]
        assert (
            'Memory files are in `.agent-memory/acme/main/` in the workspace; read them with the file-reading tool.'
            in seen[1]
        )


class TestSeveralFileMemories:
    async def test_each_memory_names_its_own_directory_across_runs(self, tmp_path: Path) -> None:
        instructions: list[str | None] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            instructions.append(info.instructions)
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(
            FunctionModel(respond),
            deps_type=type(None),
            capabilities=[
                LocalWorkspace(tmp_path),
                FileSystem(),
                Memory(store=FileStore('.agent-memory'), heading='Personal notes'),
                Memory(
                    store=FileStore('.agent-memory'), agent_name='org', heading='Org notes', id='org_memory'
                ).prefix_tools('org'),
            ],
        )
        first = await agent.run('first')
        await agent.run('second', message_history=first.all_messages())

        assert len(instructions) == 2
        assert instructions[0] == instructions[1]
        assert instructions[0] is not None
        assert '<context id="toolset:memory:directory">\nMemory files are in `.agent-memory/main/`' in instructions[0]
        assert (
            '<context id="toolset:org_memory:directory">\nMemory files are in `.agent-memory/org/`' in instructions[0]
        )


class TestDirectoryBlockShape:
    @pytest.mark.moves_cache_prefix(reason=TOOLS_CHANGE_MID_RUN)
    async def test_block_is_present_from_the_first_request_when_the_reader_joins_mid_run(self, tmp_path: Path) -> None:
        instructions: list[str | None] = []
        seen: list[str] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            instructions.append(info.instructions)
            seen.append(repr(messages))
            if len(instructions) == 1:
                return ModelResponse(parts=[ToolCallPart('list_directory', {'path': '.'}, tool_call_id='list')])
            return ModelResponse(parts=[TextPart('done')])

        file_tools = FileSystem[None]().get_toolset()
        assert file_tools is not None
        # The file reader is only offered from the second request on.
        late_reader = file_tools.filtered(lambda ctx, tool_def: ctx.run_step > 1 or tool_def.name != 'read_file')
        agent = Agent(
            FunctionModel(respond),
            deps_type=type(None),
            capabilities=[LocalWorkspace(tmp_path), Toolset(late_reader), Memory(store=FileStore('.agent-memory'))],
        )
        await agent.run('go')

        # The block is empty until the reader joins, then its text is appended: the instructions stay as sent.
        assert len(instructions) == 2
        assert instructions[0] == instructions[1]
        assert instructions[0] is not None and 'Memory files are in' not in instructions[0]
        assert 'Memory files are in `.agent-memory/main/` in the workspace' in seen[1]

    async def test_deferred_memory_says_where_its_files_are_when_the_reader_joins_after_the_load(
        self, tmp_path: Path
    ) -> None:
        """A deferred `Memory` says where its files are when it loads, before the file reader joins.

        Its instructions are only collected at the load, so a reader that joins later couldn't be announced, and
        `read_memory` steps aside for it.
        """
        seen: list[str] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(repr(messages))
            if len(seen) == 1:
                return ModelResponse(parts=[ToolCallPart('load_capability', {'id': 'memory'}, tool_call_id='load')])
            return ModelResponse(parts=[TextPart('done')])

        file_tools = FileSystem[None]().get_toolset()
        assert file_tools is not None
        # The file reader is only offered after the load.
        late_reader = file_tools.filtered(lambda ctx, tool_def: ctx.run_step > 2 or tool_def.name != 'read_file')
        memory = Memory(
            store=FileStore('.agent-memory'), id='memory', description='Long-term memory.', defer_loading=True
        )
        agent = Agent(
            FunctionModel(respond),
            deps_type=type(None),
            capabilities=[LocalWorkspace(tmp_path), Toolset(late_reader), memory],
        )
        await agent.run('go')

        assert 'Memory files are in `.agent-memory/main/` in the workspace' in seen[1]

    async def test_memory_instructions_and_snapshot_share_max_tokens(self, tmp_path: Path) -> None:
        notebook = tmp_path / '.agent-memory' / 'main' / 'MEMORY.md'
        notebook.parent.mkdir(parents=True)
        notebook.write_text('x' * 5_000)
        memory = Memory(store=FileStore('.agent-memory'), max_tokens=400)
        recorder, _ = await _run(tmp_path, memory, FileSystem())

        [instructions] = recorder.instructions
        assert instructions is not None
        guidance = memory.get_instructions()
        assert isinstance(guidance, str)
        directory = 'Memory files are in `.agent-memory/main/` in the workspace; read them with the file-reading tool.'
        assert directory in instructions
        [snapshot] = [text for text in recorder.user_text if text.startswith("[TextContent(content='<memory>")]
        snapshot_text = snapshot.split("content='", 1)[1].rsplit("', metadata", 1)[0]
        assert len(guidance) + len(directory) + len(snapshot_text) <= 400 * 4

    async def test_reader_guidance_that_does_not_fit_keeps_read_memory(self, tmp_path: Path) -> None:
        # Room for the guidance but not the guidance and the directory sentence together.
        memory = Memory(store=FileStore('.agent-memory'), guidance='', max_tokens=20)
        recorder, _ = await _run(tmp_path, memory, FileSystem())
        assert all('read_memory' in tools and 'read_file' in tools for tools in recorder.offered)
        assert recorder.instructions[0] is None or 'Memory files are in' not in recorder.instructions[0]


class _SilentConnection(InferredConnection):
    async def send_input(self, content: RealtimeInput) -> None:
        pass  # pragma: no cover

    async def codec_events(self) -> AsyncIterator[RealtimeCodecEvent]:  # pragma: no cover - the session closes first
        return
        yield


class _RecordingRealtimeModel(RealtimeModel):
    def __init__(self) -> None:
        self.instructions: list[str] = []

    @property
    def model_name(self) -> str:
        return 'fake-realtime'

    @property
    def system(self) -> str:
        return 'fake'

    @property
    def profile(self) -> RealtimeModelProfile:
        return RealtimeModelProfile(supports_text_output=True)

    @asynccontextmanager
    async def connect(
        self,
        *,
        messages: Sequence[ModelMessage],
        model_settings: RealtimeModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncGenerator[RealtimeConnection]:
        self.instructions.extend(part.content for part in model_request_parameters.instruction_parts or ())
        yield _SilentConnection()


class TestRealtime:
    async def test_session_opens_without_an_append_warning(self) -> None:
        # The `directory` block is always there, empty here: a session sends instructions once, so it's not
        # asked to append, which a realtime session would warn about.
        model = _RecordingRealtimeModel()
        agent = Agent(capabilities=[Memory(store=InMemoryStore())])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            async with agent.realtime(model).session():
                pass

        assert not [warning for warning in caught if "`on_change='append'`" in str(warning.message)]

        assert model.instructions
        assert not any('Memory files are in' in text for text in model.instructions)
