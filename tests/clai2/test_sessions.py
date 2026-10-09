"""Saved sessions restore context, never tool execution or executable configuration."""

import os
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from io import StringIO
from pathlib import Path

import anyio
import pytest
from rich.console import Console

from pydantic_ai import Agent, AgentStreamEvent, FunctionToolCallEvent, ModelRetry, RunContext
from pydantic_ai.agent import WrapperAgent
from pydantic_ai.capabilities import (
    AbstractCapability,
    Capability,
    CombinedCapability,
    Hooks,
    LocalWorkspace,
    ValidatedToolArgs,
    WrapperCapability,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.workspaces import LocalWorkspaceBackend, WorkspaceBackend, WorkspaceRef
from pydantic_ai_harness import Coder
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, StepPersistence
from pydantic_ai_harness.step_persistence.conversations import (
    ConversationConflict,
    ConversationSummary,
    SqliteConversationStore,
)
from pydantic_clai2 import DEFAULT_PLUGINS, create_agent
from pydantic_clai2._app import STOCK_PLUGINS, create_stock_agent
from pydantic_clai2.plugins import PluginHost, load_plugin
from pydantic_clai2.runtime._session import Session
from pydantic_clai2.runtime.sessions import PersistencePlugin


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


async def test_busy_session_resumes_as_a_fork_of_its_latest_step(tmp_path: Path) -> None:
    first = saved_session(tmp_path)
    await first.prompt('first')
    store, steps = first.conversations, first.step_store
    assert store is not None and steps is not None
    # This process is alive, so the session is busy in it.
    head = await store.save(
        summary=replace(first.summary, outcome='running', run_id='live', owner_pid=os.getpid()), messages=first.messages
    )
    frontier = [*first.messages, ModelResponse(parts=[ToolCallPart('slow', {'x': 1}, tool_call_id='t1')])]
    await steps.save_snapshot(
        ContinuableSnapshot(
            run_id='live', step_index=2, messages=frontier, state='interrupted', conversation_id=head.id
        )
    )
    second = saved_session(tmp_path)
    notice = await second.resume(head.id)
    fork = second.summary
    assert notice == (
        f'Session is busy in process {os.getpid()}. Resumed a fork of it instead: first (fork) ({fork.id}). '
        'The original keeps running there and may still change files. No tools were replayed.'
    )
    assert fork.id != head.id
    assert (fork.workspace, fork.outcome, fork.owner_pid, fork.run_id) == (head.workspace, 'ready', None, None)
    assert second.messages == [*frontier[:-1], replace(frontier[-1], state='interrupted')]
    assert (await store.get(conversation_id=fork.id)).messages == second.messages
    await second.prompt('continue in the fork')
    original = await store.get(conversation_id=head.id)
    assert original.summary == head
    assert original.messages == first.messages


async def test_failed_fork_save_leaves_session_unchanged(tmp_path: Path) -> None:
    class FullDisk(SqliteConversationStore):
        async def save(self, *, summary: ConversationSummary, messages: Sequence[ModelMessage]) -> ConversationSummary:
            if summary.title.endswith('(fork)'):
                raise OSError('disk full')
            return await super().save(summary=summary, messages=messages)

    store = FullDisk(database=tmp_path / 'sessions.db')
    first = Session(Agent(TestModel()), deps=None, conversations=store, workspace=tmp_path)
    await first.prompt('first')
    head = await store.save(summary=replace(first.summary, outcome='running', owner_pid=os.getpid()), messages=[])
    second = Session(Agent(TestModel()), deps=None, conversations=store, workspace=tmp_path)
    await second.prompt('mine')
    previous, summary = second.messages, second.summary
    with pytest.raises(OSError, match='disk full'):
        await second.resume(head.id)
    assert (second.messages, second.summary) == (previous, summary)
    assert not second.running


async def test_cancellation_persists_and_live_session_resumes_as_a_fork(tmp_path: Path) -> None:
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
            assert 'Resumed a fork of it instead' in await other.resume(session.summary.id)
            assert other.summary.id != session.summary.id
            assert other.messages == [replace(session.messages[0], state='interrupted')]
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
    plugin = load_plugin(PersistencePlugin, host)
    assert any(isinstance(cap, StepPersistence) for cap in plugin.capabilities)
    session.plugins = list(plugin.capabilities)
    await session.prompt('hello')
    assert session.step_store is not None and session.summary.run_id is not None
    snapshot = await session.step_store.latest_snapshot(run_id=session.summary.run_id)
    assert snapshot is not None
    assert snapshot.messages == session.messages
    assert snapshot.conversation_id == session.summary.id
    bare = PluginHost(name='persistence', console=Console(file=StringIO()), settings={})
    assert not load_plugin(PersistencePlugin, bare).capabilities


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


@pytest.mark.parametrize('deny', [False, True])
async def test_stock_agent_delegates_with_plugin_tools_instructions_and_guardrails(tmp_path: Path, deny: bool) -> None:
    hooks = Hooks[None]()
    tools = Capability[None](instructions='Follow the plugin guardrail.')
    executed: list[str] = []
    guarded: list[str] = []
    runs: list[tuple[str | None, str | None, str]] = []

    @tools.tool_plain
    def plugin_tool() -> str:
        executed.append('plugin_tool')
        return 'plugin result'

    bindings: list[RunContext[None]] = []

    def dynamic(ctx: RunContext[None]) -> Capability[None]:
        bindings.append(ctx)
        return Capability(instructions='Dynamic plugin instructions.')

    @hooks.on.before_run
    async def record(ctx: RunContext[None]) -> None:
        runs.append((ctx.run_id, ctx.conversation_id, await ctx.workspace.working_dir()))

    @hooks.on.before_tool_execute
    async def guard(
        ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs
    ) -> ValidatedToolArgs:
        guarded.append(call.tool_name)
        if deny and call.tool_name == 'plugin_tool':
            raise ModelRetry('Denied by plugin')
        return args

    (tmp_path / 'note.txt').write_text('workspace content')
    observed: list[AgentInfo] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        observed.append(info)
        assert 'Follow the plugin guardrail.' in (info.instructions or '')
        assert (info.instructions or '').count('Dynamic plugin instructions.') == 1
        assert {'delegate_task', 'plugin_tool', 'read_file'} <= {tool.name for tool in info.function_tools}
        first = messages[0]
        assert isinstance(first, ModelRequest)
        [prompt] = first.parts
        assert isinstance(prompt, UserPromptPart)
        if len(messages) == 1:
            if prompt.content == 'parent task':
                return ModelResponse(
                    parts=[ToolCallPart('delegate_task', {'agent_name': 'self', 'task': 'child task'})]
                )
            assert prompt.content == 'child task'
            return ModelResponse(
                parts=[ToolCallPart('plugin_tool', {}), ToolCallPart('read_file', {'path': 'note.txt'})]
            )
        if prompt.content == 'child task':
            returns = [
                part
                for message in messages
                if isinstance(message, ModelRequest)
                for part in message.parts
                if isinstance(part, (ToolReturnPart, RetryPromptPart))
            ]
            assert any('workspace content' in str(part.content) for part in returns)
            assert any(('Denied by plugin' if deny else 'plugin result') in str(part.content) for part in returns)
        return ModelResponse(parts=[TextPart('done')])

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        for index, part in enumerate(respond(messages, info).parts):
            if isinstance(part, TextPart):
                yield part.content
            else:
                assert isinstance(part, ToolCallPart)
                yield {index: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str())}

    model = FunctionModel(respond, stream_function=stream)
    agent = create_stock_agent(model)
    session = Session(
        agent,
        deps=None,
        workspace=tmp_path,
        conversations=SqliteConversationStore(database=tmp_path / 'sessions.db'),
        plugins=[Coder(repo_context=False), tools, dynamic, hooks, LocalWorkspace(tmp_path)],
    )
    persistence = PluginHost[None](
        name='persistence', console=Console(file=StringIO()), settings={}, conversation=session
    )
    session.plugins = [*session.plugins, *load_plugin(PersistencePlugin, persistence).capabilities]
    assert (await session.prompt('parent task')).output == 'done'
    assert executed == ([] if deny else ['plugin_tool'])
    assert sorted(guarded) == ['delegate_task', 'plugin_tool', 'read_file']
    assert len(observed) == 4
    assert len(runs) == 2
    assert len(bindings) == 2
    assert bindings[0] is not bindings[1]
    assert runs[0][0] != runs[1][0]
    assert runs[0][1] == session.summary.id
    assert runs[1][1] != session.summary.id
    assert {workspace for _, _, workspace in runs} == {str(tmp_path.resolve())}
    assert session.step_store is not None and session.summary.run_id is not None
    snapshot = await session.step_store.latest_snapshot(run_id=session.summary.run_id)
    assert snapshot is not None
    assert snapshot.messages == session.messages
    assert all(
        part.content != 'child task'
        for message in session.messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    )
    assert agent is not session.agent


async def test_stock_agent_rebuilds_only_when_plugin_snapshot_changes(tmp_path: Path) -> None:
    model = TestModel(call_tools=[], custom_output_text='answer')
    agent = create_stock_agent(model)
    session = Session(agent, deps=None, workspace=tmp_path)
    await session.prompt('no plugins')
    assert session.agent is agent
    coder = Coder[None](repo_context=False)
    session.plugins = [coder]
    await session.prompt('enabled')
    bound = session.agent
    assert bound is not agent
    assert model.last_model_request_parameters is not None
    assert 'delegate_task' in {tool.name for tool in model.last_model_request_parameters.function_tools}
    session.plugins = [coder]
    await session.prompt('unchanged')
    assert session.agent is bound
    session.plugins = [Coder(repo_context=False, sub_agents=False)]
    await session.prompt('reconfigured')
    assert session.agent is not bound
    assert 'delegate_task' not in {tool.name for tool in model.last_model_request_parameters.function_tools}
    session.plugins = []
    await session.prompt('disabled')
    assert {tool.name for tool in model.last_model_request_parameters.function_tools} == {
        'read_clai_customization_guide'
    }
    assert session.summary.id
    assert (
        sum(
            isinstance(part, UserPromptPart)
            for message in session.messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        )
        == 5
    )
    await agent.run('original remains unchanged')
    assert {tool.name for tool in model.last_model_request_parameters.function_tools} == {
        'read_clai_customization_guide'
    }


@pytest.mark.parametrize('base_clai_agent', [False, True])
async def test_supplied_agent_is_not_rebuilt_for_plugins(tmp_path: Path, base_clai_agent: bool) -> None:
    model = TestModel(call_tools=[])
    agent = create_agent('test') if base_clai_agent else Agent(model, deps_type=type(None))
    session = Session(agent, deps=None, workspace=tmp_path, plugins=[Coder(repo_context=False, sub_agents=False)])
    with agent.override(model=model):
        await session.prompt('custom')
    assert session.agent is agent
    session.plugins = [Coder(repo_context=False)]
    with pytest.raises(UserError, match='bound to the `Agent`'):
        await session.prompt('delegation still requires agent-bound capabilities')
    assert session.agent is agent


async def test_running_stock_sessions_keep_independent_plugin_snapshots(tmp_path: Path) -> None:
    entered, release = anyio.Event(), anyio.Event()
    seen: dict[str, str] = {}

    class Hold(Capability[None]):
        async def before_run(self, ctx: RunContext[None]) -> None:
            entered.set()
            await release.wait()

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        [request] = messages
        assert isinstance(request, ModelRequest)
        [prompt] = request.parts
        assert isinstance(prompt, UserPromptPart) and isinstance(prompt.content, str)
        seen[prompt.content] = info.instructions or ''
        yield 'done'

    template = create_stock_agent(FunctionModel(stream_function=stream))
    old = Session(template, deps=None, workspace=tmp_path, plugins=[Hold(instructions='OLD_SNAPSHOT_MARKER')])
    new = Session(template, deps=None, workspace=tmp_path, plugins=[Capability(instructions='NEW_SNAPSHOT_MARKER')])
    with anyio.fail_after(10):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(old.prompt, 'old')
            await entered.wait()
            await new.prompt('new')
            assert old.agent is not new.agent
            release.set()
    assert 'OLD_SNAPSHOT_MARKER' in seen['old'] and 'NEW_SNAPSHOT_MARKER' not in seen['old']
    assert 'NEW_SNAPSHOT_MARKER' in seen['new'] and 'OLD_SNAPSHOT_MARKER' not in seen['new']


def test_stock_coder_enables_delegation() -> None:
    assert next(plugin for plugin in STOCK_PLUGINS if plugin.id == 'coder').settings['sub_agents'] is True
    assert next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'coder').settings['sub_agents'] is False
