"""Saved sessions restore context, never tool execution or executable configuration."""

from collections.abc import Sequence
from dataclasses import replace
from io import StringIO
from pathlib import Path

import anyio
import pytest
from rich.console import Console

from pydantic_ai import Agent, AgentStreamEvent, FunctionToolCallEvent, RunContext
from pydantic_ai.agent import WrapperAgent
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, WrapperCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import LocalWorkspaceBackend, WorkspaceBackend, WorkspaceRef
from pydantic_ai_harness import Coder
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, StepPersistence
from pydantic_ai_harness.step_persistence.conversations import (
    ConversationConflict,
    ConversationSummary,
    SqliteConversationStore,
)
from pydantic_clai2._session import Session
from pydantic_clai2.plugins import PluginHost
from pydantic_clai2.sessions import activate


def saved_session(tmp_path: Path) -> Session[None, str]:
    return Session(
        Agent(TestModel(custom_output_text='answer')),
        deps=None,
        conversations=SqliteConversationStore(database=tmp_path / 'sessions.db'),
        workspace=tmp_path,
    )


async def test_multi_turn_restart_new_and_compaction(tmp_path: Path) -> None:
    session = saved_session(tmp_path)
    result = await session.prompt('first')
    first_id = session.summary.id
    await session.prompt('second')
    store = session.conversations
    assert store is not None
    assert (await store.get(conversation_id=first_id)).messages == session.messages
    restored = saved_session(tmp_path)
    notice = await restored.resume(first_id)
    assert 'Resumed' in notice
    assert restored.messages == session.messages
    assert restored.summary.id == first_id
    third = await restored.prompt('third')
    assert third.all_messages()[0] == result.all_messages()[0]
    compacted = [ModelRequest(parts=[UserPromptPart('summary')])]
    await restored.commit_messages(compacted)
    restarted = saved_session(tmp_path)
    await restarted.resume(first_id)
    assert restarted.messages == compacted
    restarted.clear()
    assert restarted.summary.id != first_id
    assert restarted.messages == []
    assert (await store.get(conversation_id=first_id)).messages == compacted


async def test_failed_first_request_keeps_accepted_prompt(tmp_path: Path) -> None:
    session = saved_session(tmp_path)

    async def broken(name: str) -> str:
        raise ValueError('model unavailable')

    session.model = 'missing:model'
    session.resolve_model = broken
    with pytest.raises(ValueError, match='unavailable'):
        await session.prompt('do not lose this')
    store = session.conversations
    assert store is not None
    record = await store.get(conversation_id=session.summary.id)
    assert record.summary.outcome == 'failed'
    assert record.messages == session.messages
    part = record.messages[0].parts[0]
    assert isinstance(part, UserPromptPart) and part.content == 'do not lose this'
    restored = saved_session(tmp_path)
    assert 'Interrupted session' in await restored.resume(session.summary.id)


async def test_stale_writer_and_failed_compaction_leave_memory_unchanged(tmp_path: Path) -> None:
    first, second = saved_session(tmp_path), saved_session(tmp_path)
    await first.prompt('first')
    await second.resume(first.summary.id)
    previous = second.messages
    await first.prompt('second')
    with pytest.raises(ConversationConflict):
        await second.prompt('stale')
    assert second.messages == previous
    with pytest.raises(ConversationConflict):
        await second.commit_messages([])
    assert second.messages == previous


async def test_cross_workspace_and_bad_id_leave_active_session_unchanged(tmp_path: Path) -> None:
    first = saved_session(tmp_path)
    await first.prompt('first')
    second = Session(Agent(TestModel()), deps=None, conversations=first.conversations, workspace=tmp_path / 'elsewhere')
    with pytest.raises(ValueError, match='belongs to'):
        await second.resume(first.summary.id)
    assert second.messages == []
    await second.resume(first.summary.id, allow_other_workspace=True)
    assert second.workspace == str((tmp_path / 'elsewhere').resolve())
    assert second.summary.workspace == first.workspace
    previous = second.messages
    with pytest.raises(LookupError):
        await second.resume('not-a-session')
    assert second.messages == previous
    with pytest.raises(ValueError, match='not configured'):
        await Session(Agent(TestModel()), deps=None).resume('missing')


async def test_recover_latest_frontier_without_replaying_tools(tmp_path: Path) -> None:
    first = saved_session(tmp_path)
    await first.prompt('first')
    store, steps = first.conversations, first.step_store
    assert store is not None and steps is not None
    head = await store.save(
        summary=replace(first.summary, outcome='running', run_id='interrupted', owner_pid=None), messages=first.messages
    )
    frontier = [*first.messages, ModelResponse(parts=[ToolCallPart('dangerous', {'x': 1}, tool_call_id='t1')])]
    await steps.save_snapshot(
        ContinuableSnapshot(
            run_id='interrupted', step_index=2, messages=frontier, state='interrupted', conversation_id=head.id
        )
    )
    second = saved_session(tmp_path)
    notice = await second.resume(head.id)
    assert second.messages == [*frontier[:-1], replace(frontier[-1], state='interrupted')]
    await second.prompt('continue without replaying tools')
    assert 'No tools were replayed' in notice


async def test_cancellation_persists_and_live_session_cannot_resume(tmp_path: Path) -> None:
    entered = anyio.Event()

    class Pause(AbstractCapability[None]):
        async def before_run(self, ctx: RunContext[None]) -> None:
            entered.set()
            await anyio.sleep_forever()

    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    session = Session(
        Agent(TestModel(), deps_type=type(None), capabilities=[Pause()]),
        deps=None,
        conversations=store,
        workspace=tmp_path,
    )
    other = saved_session(tmp_path)
    with anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            group.start_soon(session.prompt, 'interrupted prompt')
            await entered.wait()
            with pytest.raises(ConversationConflict, match='busy'):
                await other.resume(session.summary.id)
            with pytest.raises(RuntimeError):
                await session.resume(session.summary.id)
            with pytest.raises(RuntimeError):
                await session.commit_messages([])
            group.cancel_scope.cancel()
    loaded = await store.get(conversation_id=session.summary.id)
    assert loaded.summary.outcome == 'cancelled'
    assert loaded.messages == session.messages
    assert loaded.summary.owner_pid is None


async def test_persistence_plugin_uses_session_store(tmp_path: Path) -> None:
    session = saved_session(tmp_path)
    host = PluginHost(name='persistence', console=Console(file=StringIO()), settings={}, conversation=session)
    activate(host)
    assert any(isinstance(cap, StepPersistence) for cap in host.capabilities)
    session.plugins = host.capabilities
    await session.prompt('hello')
    assert session.step_store is not None and session.summary.run_id is not None
    snapshot = await session.step_store.latest_snapshot(run_id=session.summary.run_id)
    assert snapshot is not None
    assert snapshot.messages == session.messages
    assert snapshot.conversation_id == session.summary.id
    bare = PluginHost(name='persistence', console=Console(file=StringIO()), settings={})
    activate(bare)
    assert not bare.capabilities


async def test_cancellation_still_propagates_when_saving_fails(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    entered = anyio.Event()

    class BrokenCleanup(SqliteConversationStore):
        async def save(self, *, summary: ConversationSummary, messages: Sequence[ModelMessage]) -> ConversationSummary:
            if summary.outcome == 'cancelled':
                raise OSError('disk full')
            return await super().save(summary=summary, messages=messages)

    class Pause(AbstractCapability[None]):
        async def before_run(self, ctx: RunContext[None]) -> None:
            entered.set()
            await anyio.sleep_forever()

    store = BrokenCleanup(database=tmp_path / 'sessions.db')
    session = Session(Agent(TestModel(), deps_type=type(None), capabilities=[Pause()]), deps=None, conversations=store)
    with anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            group.start_soon(session.prompt, 'preserve cancellation')
            await entered.wait()
            group.cancel_scope.cancel()
    assert session.messages
    assert 'Could not save cancelled turn: disk full' in caplog.text


async def test_memory_only_commit_and_missing_recovery_snapshot(tmp_path: Path) -> None:
    bare = Session(Agent(TestModel()), deps=None)
    await bare.commit_messages([ModelRequest(parts=[UserPromptPart('memory')])])
    assert bare.messages
    session = saved_session(tmp_path)
    assert session.conversations is not None
    saved = await session.conversations.save(
        summary=replace(session.summary, outcome='running', run_id='no-checkpoint'), messages=bare.messages
    )
    await session.resume(saved.id)
    assert session.messages == [replace(bare.messages[0], state='interrupted')]


@pytest.mark.parametrize('cancel', [False, True])
async def test_interrupted_tool_frontier_accepts_followup(tmp_path: Path, *, cancel: bool) -> None:
    entered = anyio.Event()
    calls: list[str] = []
    agent = Agent(TestModel(call_tools=['effect'], custom_output_text='answer'))

    @agent.tool_plain
    def effect() -> str:
        calls.append('executed')  # pragma: no cover
        return 'done'  # pragma: no cover

    async def interrupt(event: AgentStreamEvent) -> None:
        if isinstance(event, FunctionToolCallEvent):
            entered.set()
            if cancel:
                await anyio.sleep_forever()
            raise ValueError('interrupted before result')

    session = Session(
        agent,
        deps=None,
        conversations=SqliteConversationStore(database=tmp_path / 'sessions.db'),
        on_stream_event=interrupt,
    )
    if cancel:
        with anyio.fail_after(10):
            async with anyio.create_task_group() as group:
                group.start_soon(session.prompt, 'start')
                await entered.wait()
                group.cancel_scope.cancel()
    else:
        with pytest.raises(ValueError, match='interrupted before result'):
            await session.prompt('start')
    previous_calls = list(calls)
    session.on_stream_event = None
    with agent.override(model=TestModel(call_tools=[], custom_output_text='recovered')):
        result = await session.prompt('clarification')
    assert result.output == 'recovered'
    assert calls == previous_calls
    assert any(
        isinstance(part, ToolReturnPart)
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
    )


@pytest.mark.parametrize('frontier', ['empty', 'suspended', 'partial'])
async def test_restore_interrupted_frontier_preserves_existing_results(tmp_path: Path, frontier: str) -> None:
    session = saved_session(tmp_path)
    messages: list[ModelMessage] = []
    if frontier == 'suspended':
        messages = [ModelResponse(parts=[], state='suspended')]
    elif frontier == 'partial':
        messages = [
            ModelResponse(
                parts=[ToolCallPart('effect', {}, tool_call_id='a'), ToolCallPart('effect', {}, tool_call_id='b')]
            ),
            ModelRequest(parts=[ToolReturnPart('effect', 'already completed', tool_call_id='a')]),
        ]
    assert session.conversations is not None
    head = await session.conversations.save(summary=replace(session.summary, outcome='failed'), messages=messages)
    await session.resume(head.id)
    if frontier != 'partial':
        assert session.messages == messages
        return
    result = await session.prompt('continue')
    returns = [
        part
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]
    assert len(returns) == 2
    assert returns[0].content == 'already completed'
    assert {part.tool_call_id for part in returns} == {'a', 'b'}


async def test_resume_from_another_directory_runs_in_this_session_directory(tmp_path: Path) -> None:
    working_dirs: list[str] = []

    class RecordWorkingDir(AbstractCapability[None]):
        async def before_run(self, ctx: RunContext[None]) -> None:
            working_dirs.append(await ctx.workspace.working_dir())

    agent = Agent(TestModel(custom_output_text='answer'), deps_type=type(None), capabilities=[RecordWorkingDir()])
    conversations = SqliteConversationStore(database=tmp_path / 'sessions.db')
    original, elsewhere = tmp_path / 'original', tmp_path / 'elsewhere'
    original.mkdir()
    elsewhere.mkdir()
    first = Session(agent, deps=None, conversations=conversations, workspace=original)
    await first.prompt('first')
    second = Session(agent, deps=None, conversations=conversations, workspace=elsewhere)
    await second.resume(first.summary.id, allow_other_workspace=True)
    await second.prompt('second')
    await second.prompt('third')
    assert working_dirs == [str(original.resolve()), str(elsewhere.resolve()), str(elsewhere.resolve())]


@pytest.mark.parametrize(
    'shape', ['bare', 'wrapped', 'wrapped twice', 'wrapped group', 'supplied by the wrapper', 'capability function']
)
async def test_a_sandbox_on_the_agent_itself_replaces_the_session_directory(tmp_path: Path, shape: str) -> None:
    working_dirs: list[str] = []
    sandbox = tmp_path / 'sandbox'
    sandbox.mkdir()

    class Probe(AbstractCapability[None]):
        async def before_run(self, ctx: RunContext[None]) -> None:
            working_dirs.append(await ctx.workspace.working_dir())

    class Sandbox(Probe):
        def get_workspace(self, ctx: RunContext[None], *, ref: WorkspaceRef | None) -> WorkspaceBackend:
            return LocalWorkspaceBackend(sandbox)

    class SandboxWrapper(WrapperCapability[None]):
        def get_workspace(self, ctx: RunContext[None], *, ref: WorkspaceRef | None) -> WorkspaceBackend:
            return LocalWorkspaceBackend(sandbox)

    def capability_function(ctx: RunContext[None]) -> AbstractCapability[None]:
        return Sandbox()

    capability = {
        'bare': Sandbox(),
        'wrapped': WrapperCapability(Sandbox()),
        'wrapped twice': WrapperCapability(WrapperCapability(Sandbox())),
        'wrapped group': WrapperCapability(CombinedCapability([Sandbox()])),
        'supplied by the wrapper': SandboxWrapper(Probe()),
        'capability function': capability_function,
    }[shape]
    agent = Agent(TestModel(custom_output_text='answer'), deps_type=type(None), capabilities=[capability])
    await Session(agent, deps=None, workspace=tmp_path).prompt('go')
    assert working_dirs == [str(sandbox.resolve())]


async def test_a_capability_function_without_a_workspace_fails_the_coder_run(tmp_path: Path) -> None:
    # The function may have picked a sandbox, so clai adds no directory; without one, `Coder` says so.
    def capability_function(ctx: RunContext[None]) -> None:
        return None

    agent = Agent(TestModel(), deps_type=type(None), capabilities=[Coder(), capability_function])
    with pytest.raises(UserError, match='`Coder` needs a workspace'):
        await Session(agent, deps=None, workspace=tmp_path).prompt('go')


async def test_an_agent_without_a_capability_tree_gets_the_session_directory(tmp_path: Path) -> None:
    working_dirs: list[str] = []

    class Probe(AbstractCapability[None]):
        async def before_run(self, ctx: RunContext[None]) -> None:
            working_dirs.append(await ctx.workspace.working_dir())

    class OpaqueAgent(WrapperAgent[None, str]):
        @property
        def root_capability(self) -> CombinedCapability[None]:
            raise NotImplementedError

    agent = OpaqueAgent(Agent(TestModel(custom_output_text='answer'), deps_type=type(None)))
    await Session(agent, deps=None, workspace=tmp_path, plugins=[Probe()]).prompt('go')
    assert working_dirs == [str(tmp_path.resolve())]


async def test_a_plugin_that_supplies_the_workspace_replaces_the_session_directory(tmp_path: Path) -> None:
    working_dirs: list[str] = []
    sandbox = tmp_path / 'sandbox'
    sandbox.mkdir()

    class SandboxPlugin(AbstractCapability[None]):
        def get_workspace(self, ctx: RunContext[None], *, ref: WorkspaceRef | None) -> WorkspaceBackend:
            return LocalWorkspaceBackend(sandbox)

        async def before_run(self, ctx: RunContext[None]) -> None:
            working_dirs.append(await ctx.workspace.working_dir())

    def no_capability(ctx: RunContext[None]) -> None:
        return None

    session = Session(
        Agent(TestModel(custom_output_text='answer'), deps_type=type(None)),
        deps=None,
        workspace=tmp_path,
        plugins=[no_capability, SandboxPlugin()],
    )
    await session.prompt('go')
    assert working_dirs == [str(sandbox.resolve())]


async def test_a_group_plugin_that_supplies_the_workspace_replaces_the_session_directory(tmp_path: Path) -> None:
    working_dirs: list[str] = []
    sandbox = tmp_path / 'sandbox'
    sandbox.mkdir()

    class Probe(AbstractCapability[None]):
        async def before_run(self, ctx: RunContext[None]) -> None:
            working_dirs.append(await ctx.workspace.working_dir())

    class SandboxGroup(CombinedCapability[None]):
        # The group supplies the workspace itself, not through any of its members.
        def get_workspace(self, ctx: RunContext[None], *, ref: WorkspaceRef | None) -> WorkspaceBackend:
            return LocalWorkspaceBackend(sandbox)

    session = Session(
        Agent(TestModel(custom_output_text='answer'), deps_type=type(None)),
        deps=None,
        workspace=tmp_path,
        plugins=[SandboxGroup([Probe()])],
    )
    await session.prompt('go')
    assert working_dirs == [str(sandbox.resolve())]
