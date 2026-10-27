"""Per-delegation workspaces use the selected backend without changing the parent run."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal

import anyio
import pytest
from pydantic import TypeAdapter
from typing_extensions import TypeIs

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import LocalWorkspaceBackend, ReadOnlyWorkspace, Workspace, WorkspaceReadOnlyError
from pydantic_ai_harness.subagents import (
    DelegationTask,
    DelegationTasks,
    DelegationWorkspaceRequest,
    SubAgent,
    SubAgents,
    SubAgentToolset,
    TaskControl,
)


def is_toolset(value: object) -> TypeIs[SubAgentToolset[object]]:
    return isinstance(value, SubAgentToolset)


@pytest.mark.parametrize('enabled', [False, True])
async def test_directory_schema_is_opt_in(tmp_path: Path, enabled: bool) -> None:
    async def resolve(ctx: RunContext[object], request: DelegationWorkspaceRequest) -> Workspace:
        return Workspace(LocalWorkspaceBackend(tmp_path / request.directory))  # pragma: no cover

    def inspect(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        schema = info.function_tools[0].parameters_json_schema
        assert ('working_directory' in schema['properties']) is enabled
        return ModelResponse(parts=[TextPart('checked')])

    agent = Agent(
        FunctionModel(inspect),
        capabilities=[
            SubAgents(
                agents=[SubAgent(Agent(TestModel(), name='worker'))],
                workspace_resolver=resolve if enabled else None,
            )
        ],
    )
    assert (await agent.run('inspect schema')).output == 'checked'


@pytest.mark.parametrize('mode', ['plain', 'owned', 'controlled'])
@pytest.mark.parametrize('parent_read_only,child_read_only', [(False, False), (True, False), (False, True)])
async def test_selected_directory_and_read_only(
    tmp_path: Path, mode: Literal['plain', 'owned', 'controlled'], parent_read_only: bool, child_read_only: bool
) -> None:
    parent_dir, child_dir = tmp_path / 'parent', tmp_path / 'policy'
    parent_dir.mkdir()
    child_dir.mkdir()
    (parent_dir / 'marker').write_text('parent')
    (child_dir / 'marker').write_text('policy')
    workspace = Workspace(LocalWorkspaceBackend(parent_dir))
    if parent_read_only:
        workspace = ReadOnlyWorkspace(workspace)
    child: Agent[object, str] = Agent(TestModel(call_tools=['inspect']), name='worker')

    @child.tool
    async def inspect(ctx: RunContext[object]) -> str:
        assert await ctx.workspace.working_dir() == str(child_dir)
        assert await ctx.workspace.read_text('marker') == 'policy'
        assert ctx.workspace.read_only is (parent_read_only or child_read_only)
        if ctx.workspace.read_only:
            with pytest.raises(WorkspaceReadOnlyError):
                await ctx.workspace.write_text('marker', 'changed')
            with pytest.raises(WorkspaceReadOnlyError):
                await ctx.workspace.run(['touch', 'changed'])
        return 'inspected'

    async def resolve(ctx: RunContext[object], request: DelegationWorkspaceRequest) -> Workspace:
        assert ctx.workspace is workspace and request.agent_name == 'worker' and request.directory == '../policy'
        return Workspace(LocalWorkspaceBackend(await ctx.workspace.resolve(request.directory)))

    capability = SubAgents[object](
        agents=[SubAgent(child, read_only=child_read_only)],
        workspace_resolver=resolve,
        task_control=TaskControl() if mode == 'controlled' else None,
    )
    toolset = capability.get_toolset()
    assert is_toolset(toolset)
    ctx = RunContext(deps=object(), model=TestModel(), usage=RunUsage(), workspace=workspace, conversation_id='parent')
    if mode == 'plain':
        output = await toolset.delegate_task(ctx, 'worker', 'inspect', working_directory='../policy')
    else:
        async with DelegationTasks().opened() as owner:
            with owner.bind():
                output = await toolset.delegate_task(ctx, 'worker', 'inspect', working_directory='../policy')
                (record,) = owner.records.values()
                assert record.working_directory == str(child_dir)
    assert 'inspected' in output
    assert await workspace.working_dir() == str(parent_dir)
    assert (parent_dir / 'marker').read_text() == 'parent'
    assert (child_dir / 'marker').read_text() == 'policy'


@pytest.mark.parametrize('controlled', [False, True])
async def test_background_resume_keeps_directory(tmp_path: Path, controlled: bool) -> None:
    selected = tmp_path / 'policy'
    selected.mkdir()
    seen: list[str] = []
    started, finish = anyio.Event(), anyio.Event()

    async def inspect_each_prompt(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if any(isinstance(part, UserPromptPart) for part in messages[-1].parts):
            yield {0: DeltaToolCall(name='inspect', json_args='{}')}
        else:
            yield 'inspected'

    child: Agent[object, str] = Agent(FunctionModel(stream_function=inspect_each_prompt), name='worker')

    @child.tool
    async def inspect(ctx: RunContext[object]) -> str:
        seen.append(await ctx.workspace.working_dir())
        started.set()
        await finish.wait()
        return 'inspected'

    async def resolve(ctx: RunContext[object], request: DelegationWorkspaceRequest) -> Workspace:
        return Workspace(LocalWorkspaceBackend(tmp_path / request.directory))

    capability = SubAgents[object](
        agents=[SubAgent(child)], workspace_resolver=resolve, task_control=TaskControl() if controlled else None
    )
    toolset = capability.get_toolset()
    assert is_toolset(toolset)
    ctx = RunContext(deps=object(), model=TestModel(), usage=RunUsage(), conversation_id='parent')
    with anyio.fail_after(10):
        async with DelegationTasks(directory=tmp_path / 'tasks').opened() as owner:
            with owner.bind():
                receipt = await toolset.delegate_task(
                    ctx, 'worker', 'inspect', background=True, working_directory='policy'
                )
                await started.wait()
                (record,) = owner.records.values()
                assert record.id in receipt
                finish.set()
                await owner.wait_children(None)
                await toolset.delegate_task(ctx, 'worker', 'again', resume=record.id)
                with pytest.raises(ModelRetry, match='cannot change its working directory'):
                    await toolset.delegate_task(
                        ctx, 'worker', 'elsewhere', resume=record.id, working_directory=str(tmp_path)
                    )
                assert record.generation == 2
    async with DelegationTasks(directory=tmp_path / 'tasks').opened() as restored:
        assert restored.records[record.id].working_directory == str(selected)
    assert seen == [str(selected), str(selected)]


async def test_owner_rejects_resume_directory_change_before_mutation(tmp_path: Path) -> None:
    async def run(record: DelegationTask) -> str:
        assert record.working_directory == str(tmp_path)
        return 'finished'

    adapter = TypeAdapter(DelegationTask)
    async with DelegationTasks().opened() as owner:
        await owner.delegate(
            agent_name='worker',
            prompt='inspect',
            conversation_id='parent',
            model=None,
            background=False,
            resume=None,
            run=run,
            working_directory=str(tmp_path),
        )
        (record,) = owner.records.values()
        saved = adapter.dump_python(record, mode='json')
        with pytest.raises(ModelRetry, match='cannot change its working directory'):
            await owner.delegate(
                agent_name='worker',
                prompt='changed prompt',
                conversation_id='parent',
                model='changed-model',
                background=False,
                resume=record.id,
                run=run,
                working_directory=str(tmp_path / 'elsewhere'),
            )
        assert adapter.dump_python(record, mode='json') == saved


async def test_rejected_resume_directory_does_not_consume_call_budget(tmp_path: Path) -> None:
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()

    async def resolve(ctx: RunContext[object], request: DelegationWorkspaceRequest) -> Workspace:
        return Workspace(LocalWorkspaceBackend(request.directory))

    capability = SubAgents[object](
        agents=[SubAgent(Agent(TestModel(custom_output_text='finished'), name='worker'), max_calls=2)],
        workspace_resolver=resolve,
    )
    toolset = capability.get_toolset()
    assert is_toolset(toolset)
    ctx = RunContext(deps=object(), model=TestModel(), usage=RunUsage(), conversation_id='parent', run_id='one-run')
    async with DelegationTasks().opened() as owner:
        with owner.bind():
            await toolset.delegate_task(ctx, 'worker', 'first', working_directory=str(tmp_path))
            (record,) = owner.records.values()
            with pytest.raises(ModelRetry, match='cannot change its working directory'):
                await toolset.delegate_task(
                    ctx, 'worker', 'invalid', resume=record.id, working_directory=str(elsewhere)
                )
            result = await toolset.delegate_task(ctx, 'worker', 'valid', resume=record.id)
            assert 'finished' in result
            assert record.generation == 2


async def test_directory_requires_resolver_and_old_records_keep_defaults(tmp_path: Path) -> None:
    child = Agent(TestModel(custom_output_text='done'), name='worker')
    capability = SubAgents[object](agents=[SubAgent(child)])
    toolset = capability.get_toolset()
    assert is_toolset(toolset)
    ctx = RunContext(deps=object(), model=TestModel(), usage=RunUsage(), conversation_id='parent')
    with pytest.raises(ModelRetry, match='requires'):
        await toolset.delegate_task(ctx, 'worker', 'inspect', working_directory=str(tmp_path))
    async with DelegationTasks().opened() as owner:
        with owner.bind():
            await toolset.delegate_task(ctx, 'worker', 'inspect')
            (record,) = owner.records.values()
            with pytest.raises(ModelRetry, match='cannot change its working directory'):
                await toolset.delegate_task(ctx, 'worker', 'again', resume=record.id, working_directory=str(tmp_path))
    old = TypeAdapter(DelegationTask).validate_json(
        '{"id":"old","agent_name":"worker","prompt":"inspect","conversation_id":"parent"}'
    )
    assert old.working_directory is None


async def test_resolver_errors_and_composition(tmp_path: Path) -> None:
    async def resolve(ctx: RunContext[object], request: DelegationWorkspaceRequest) -> Workspace:
        raise ModelRetry('directory refused')

    async def other(ctx: RunContext[object], request: DelegationWorkspaceRequest) -> Workspace:
        return Workspace(LocalWorkspaceBackend(tmp_path))  # pragma: no cover

    capability = SubAgents[object](agents=[SubAgent(Agent(TestModel(), name='worker'))], workspace_resolver=resolve)
    toolset = capability.get_toolset()
    assert is_toolset(toolset)
    ctx = RunContext(deps=object(), model=TestModel(), usage=RunUsage())
    with pytest.raises(ModelRetry, match='directory refused'):
        await toolset.delegate_task(ctx, 'worker', 'inspect', working_directory='missing')
    with pytest.raises(UserError, match='workspace_resolver'):
        Agent(TestModel(), capabilities=[capability, SubAgents(workspace_resolver=other)])


async def test_durable_directory_override_is_rejected(tmp_path: Path) -> None:
    pytest.importorskip('temporalio')
    from pydantic_ai.durable_exec.temporal import TemporalRunContext

    async def resolve(ctx: RunContext[object], request: DelegationWorkspaceRequest) -> Workspace:
        raise AssertionError('resolver must not run')  # pragma: no cover

    capability = SubAgents[object](agents=[SubAgent(Agent(TestModel(), name='worker'))], workspace_resolver=resolve)
    toolset = capability.get_toolset()
    assert is_toolset(toolset)
    with pytest.raises(ModelRetry, match='durable or remote'):
        await toolset.delegate_task(
            TemporalRunContext[object](deps=None, tool_name='delegate_task'),
            'worker',
            'inspect',
            working_directory='policy',
        )


async def test_remote_directory_override_is_rejected(tmp_path: Path) -> None:
    from pydantic_ai_harness.subagents import InMemoryTransport, RemoteSubAgent

    async def resolve(ctx: RunContext[object], request: DelegationWorkspaceRequest) -> Workspace:
        raise AssertionError('resolver must not run')  # pragma: no cover

    capability = SubAgents[object](
        task_control=TaskControl(remote_agents=[RemoteSubAgent(name='remote')], transport=InMemoryTransport()),
        workspace_resolver=resolve,
    )
    toolset = capability.get_toolset()
    assert is_toolset(toolset)
    ctx = RunContext(deps=object(), model=TestModel(), usage=RunUsage())
    async with DelegationTasks().opened() as owner:
        with owner.bind(), pytest.raises(ModelRetry, match='durable or remote'):
            await toolset.delegate_task(ctx, 'remote', 'inspect', working_directory=str(tmp_path))


async def test_nested_plain_task_resume_uses_the_parent_conversation() -> None:
    capability = SubAgents[object](agents=[SubAgent(Agent(TestModel(custom_output_text='child'), name='worker'))])
    toolset = capability.get_toolset()
    assert is_toolset(toolset)
    async with DelegationTasks().opened() as owner:
        with owner.bind():

            async def nested(parent: DelegationTask) -> str:
                ctx = RunContext(deps=object(), model=TestModel(), usage=RunUsage(), conversation_id=parent.id)
                await toolset.delegate_task(ctx, 'worker', 'first')
                child = next(record for record in owner.records.values() if record.parent_id == parent.id)
                await toolset.delegate_task(ctx, 'worker', 'resume', resume=child.id)
                assert child.generation == 2 and child.conversation_id == 'root'
                return 'nested complete'

            await owner.delegate(
                agent_name='parent',
                prompt='delegate',
                conversation_id='root',
                model=None,
                background=False,
                resume=None,
                run=nested,
            )
