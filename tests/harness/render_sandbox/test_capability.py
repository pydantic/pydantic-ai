"""Compose the public workspace provider with real Agent and Harness capabilities."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import anyio
import pytest

from ...conftest import try_import

with try_import() as imports_successful:
    from render.client.errors import ClientError

    from pydantic_ai import Agent, RunContext
    from pydantic_ai.capabilities import AbstractCapability, WrapperCapability
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
    from pydantic_ai.models.function import AgentInfo, FunctionModel
    from pydantic_ai.workspaces import (
        LocalWorkspaceBackend,
        Workspace,
        WorkspaceBackend,
        WorkspaceRef,
        WorkspaceUnavailableError,
        WrapperWorkspace,
    )
    from pydantic_ai_harness import Coder, FileSystem, Shell
    from pydantic_ai_harness.render_sandbox import (
        RenderSandbox,
        RenderSandboxBackend,
        RenderSandboxCleanupError,
        RenderSandboxConfig,
        RenderSandboxEvent,
    )

    from .workspace_helpers import WorkspaceHarness

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='render not installed'),
    pytest.mark.subprocess(reason='Exercise real POSIX workspace commands and file helper processes', conditional=True),
]


def parts(messages: list[ModelMessage]) -> list[ToolReturnPart | RetryPromptPart]:
    return [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, (ToolReturnPart, RetryPromptPart))
    ]


def tool_model(name: str = 'run_command', args: Mapping[str, object] | None = None) -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        start = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if isinstance(message := messages[index], ModelRequest)
                and any(isinstance(part, UserPromptPart) for part in message.parts)
            ),
            0,
        )
        if parts(messages[start:]):
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(
            parts=[ToolCallPart(name, dict(args) if args is not None else {'command': 'printf hello'})]
        )

    return FunctionModel(respond)


def capability(harness: WorkspaceHarness, *, auto_cleanup: bool = True) -> RenderSandbox[None]:
    return RenderSandbox[None](client=harness.client, config=harness.config, auto_cleanup=auto_cleanup)


async def test_standard_shell_files_seed_finish_and_cleanup(
    workspace_harness: WorkspaceHarness,
) -> None:
    returns: list[ToolReturnPart | RetryPromptPart] = []
    events: list[RenderSandboxEvent] = []
    captured: list[bytes] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        names = {tool.name for tool in info.function_tools}
        assert {'run_command', 'read_file', 'write_file'} <= names
        assert not any(name.startswith('sandbox_') for name in names)
        completed = parts(messages)
        if not completed:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        'run_command',
                        {'command': 'cat seed.txt; printf repaired > answer.txt; exit 7'},
                    )
                ]
            )
        if len(completed) == 1:
            returns.extend(completed)
            return ModelResponse(parts=[ToolCallPart('read_file', {'path': 'answer.txt'})])
        returns.append(completed[-1])
        return ModelResponse(parts=[TextPart('done')])

    async def finish(workspace: Workspace) -> None:
        assert workspace_harness.terminated == []
        captured.append(await workspace.read_bytes('answer.txt'))
        assert await workspace.read_bytes('raw.bin') == b'\x00\xff'

    cap = RenderSandbox[None](
        client=workspace_harness.client,
        config=workspace_harness.config,
        files={'seed.txt': 'seed data', 'raw.bin': b'\x00\xff'},
        on_finish=finish,
        on_sandbox_event=events.append,
    )
    agent = Agent(
        FunctionModel(respond),
        deps_type=type(None),
        capabilities=[
            cap,
            Shell[None](tools=['run_command']),
            FileSystem[None](content_hashes=False),
        ],
    )
    result = await agent.run('repair')
    assert result.output == 'done'
    assert result.workspace.ref == WorkspaceRef(provider='render', id='sb-1')
    assert captured == [b'repaired']
    assert 'seed data' in str(returns[0].content)
    assert 'exit code: 7' in str(returns[0].content)
    assert 'repaired' in str(returns[1].content)
    assert workspace_harness.terminated == ['sb-1']
    assert events[0].event == 'creating'
    assert events[-1].event == 'terminated'


async def test_coder_uses_ordinary_file_tools(workspace_harness: WorkspaceHarness) -> None:
    captured: list[bytes] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        names = {tool.name for tool in info.function_tools}
        assert {'read_file', 'write_file', 'shell'} <= names
        assert not any(name.startswith('sandbox_') for name in names)
        if not parts(messages):
            return ModelResponse(parts=[ToolCallPart('write_file', {'path': 'coder.txt', 'content': 'standard coder'})])
        return ModelResponse(parts=[TextPart('done')])

    async def finish(workspace: Workspace) -> None:
        captured.append(await workspace.read_bytes('coder.txt'))

    cap = capability(workspace_harness)
    cap.on_finish = finish
    agent = Agent(
        FunctionModel(respond),
        deps_type=type(None),
        capabilities=[cap, Coder[None](repo_context=False, sub_agents=False)],
    )
    assert (await agent.run('write')).output == 'done'
    assert captured == [b'standard coder']
    assert workspace_harness.terminated == ['sb-1']


async def test_repeated_and_concurrent_runs_use_distinct_backends(
    workspace_harness: WorkspaceHarness,
) -> None:
    captured: list[Workspace] = []

    async def finish(workspace: Workspace) -> None:
        captured.append(workspace)

    cap = capability(workspace_harness)
    cap.on_finish = finish
    agent = Agent(tool_model(), deps_type=type(None), capabilities=[cap, Shell[None](tools=['run_command'])])
    first = await agent.run('first')
    rest = await asyncio.gather(agent.run('second'), agent.run('third'))
    assert [first.output, *(result.output for result in rest)] == ['done'] * 3
    assert len({id(workspace.backend) for workspace in captured}) == 3
    assert {workspace.ref.id for workspace in captured if workspace.ref is not None} == {
        'sb-1',
        'sb-2',
        'sb-3',
    }
    assert sorted(workspace_harness.terminated) == ['sb-1', 'sb-2', 'sb-3']


async def test_history_attachment_preserves_ownership_and_never_recreates(
    workspace_harness: WorkspaceHarness,
) -> None:
    cap = capability(workspace_harness, auto_cleanup=False)
    agent = Agent(tool_model(), deps_type=type(None), capabilities=[cap, Shell[None](tools=['run_command'])])
    first = await agent.run('first')
    ref = first.workspace.ref
    assert ref is not None
    cap.auto_cleanup = True
    second = await agent.run('again', message_history=first.all_messages())
    assert second.workspace.ref == ref
    assert workspace_harness.created == [ref.id]
    assert workspace_harness.terminated == []
    await cap.destroy(ref)
    with pytest.raises(WorkspaceUnavailableError):
        await cap.backend(ref).run(['true'])
    assert workspace_harness.created == [ref.id]


async def test_caller_backend_never_auto_destroyed(workspace_harness: WorkspaceHarness) -> None:
    backend = workspace_harness.backend()
    cap = capability(workspace_harness)
    agent = Agent(tool_model(), deps_type=type(None), capabilities=[cap, Shell[None](tools=['run_command'])])
    result = await agent.run('run', workspace=backend)
    assert result.workspace.backend is backend
    assert workspace_harness.terminated == []
    await backend.destroy()
    assert workspace_harness.terminated == ['sb-1']


async def test_caller_local_workspace_does_not_claim_render_isolation(
    workspace_harness: WorkspaceHarness, tmp_path: Path
) -> None:
    def done(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert 'isolated Render sandbox' not in (info.instructions or '')
        return ModelResponse(parts=[TextPart('done')])

    backend = LocalWorkspaceBackend(tmp_path)
    agent = Agent(FunctionModel(done), deps_type=type(None), capabilities=[capability(workspace_harness)])
    result = await agent.run('hello', workspace=backend)
    assert result.workspace.backend is backend
    assert workspace_harness.created == workspace_harness.terminated == []


async def test_lazy_run_with_no_workspace_use_creates_nothing(
    workspace_harness: WorkspaceHarness,
) -> None:
    def done(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(done), deps_type=type(None), capabilities=[capability(workspace_harness)])
    result = await agent.run('hello')
    assert result.output == 'done'
    assert result.workspace.ref is None
    assert workspace_harness.created == workspace_harness.terminated == []


@dataclass
class FailingBeforeRun(AbstractCapability[None]):
    async def before_run(self, ctx: RunContext[None]) -> None:
        raise RuntimeError('before run failed')


@pytest.mark.parametrize('phase', ['model', 'before_run', 'finish', 'seed'])
async def test_failure_paths_cleanup_owned_workspace(workspace_harness: WorkspaceHarness, phase: str) -> None:
    finish_called = False

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if phase == 'model':
            raise RuntimeError('model failed')
        return ModelResponse(parts=[TextPart('done')])

    async def finish(workspace: Workspace) -> None:
        nonlocal finish_called
        finish_called = True
        raise RuntimeError('finish failed')

    cap = capability(workspace_harness)
    cap.files = {'seed.txt': 'create before handler'}
    cap.on_finish = finish
    caps: list[AbstractCapability[None]] = [cap]
    if phase == 'before_run':
        caps.append(FailingBeforeRun())
    if phase == 'seed':
        workspace_harness.command_error = ClientError('access rejected')
    agent = Agent(FunctionModel(respond), deps_type=type(None), capabilities=caps)
    with pytest.raises((RuntimeError, WorkspaceUnavailableError)):
        await agent.run('fail')
    assert finish_called == (phase == 'finish')
    assert workspace_harness.terminated == ['sb-1']


@pytest.mark.parametrize('cancel_mode', ['native', 'anyio'])
async def test_agent_cancellation_interrupts_detached_tool_and_cleans(
    workspace_harness: WorkspaceHarness, cancel_mode: str
) -> None:
    workspace_harness.block_commands = True
    agent = Agent(
        tool_model(),
        deps_type=type(None),
        capabilities=[capability(workspace_harness), Shell[None](tools=['run_command'])],
    )
    if cancel_mode == 'native':
        task = asyncio.create_task(agent.run('wait'))
        await workspace_harness.command_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
    else:
        async with anyio.create_task_group() as group:
            group.start_soon(agent.run, 'wait')
            await workspace_harness.command_started.wait()
            group.cancel_scope.cancel()
    assert workspace_harness.terminated == ['sb-1']


@pytest.mark.parametrize('already_ready', [False, True])
async def test_handled_tool_cancellation_keeps_workspace(
    workspace_harness: WorkspaceHarness, already_ready: bool
) -> None:
    # A tool may abandon one operation and continue the run. Its local cancellation
    # must not acquire ownership of run cleanup, either before or after preparation.
    agent = Agent(tool_model('check_workspace', {}), deps_type=type(None), capabilities=[capability(workspace_harness)])

    @agent.tool
    async def check_workspace(ctx: RunContext[None]) -> str:
        if already_ready:
            await ctx.workspace.working_dir()
        with anyio.CancelScope() as scope:
            scope.cancel()
            await ctx.workspace.working_dir()
        assert workspace_harness.terminated == []
        command = await ctx.workspace.run(['printf', 'still usable'])
        return command.stdout

    result = await agent.run('check workspace')
    assert result.output == 'done'
    assert parts(result.all_messages())[0].content == 'still usable'
    # The enclosing run still owns termination once the tool has finished.
    assert workspace_harness.created == workspace_harness.terminated == ['sb-1']


async def test_terminal_tool_failure_is_not_a_model_retry(
    workspace_harness: WorkspaceHarness,
) -> None:
    requests = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal requests
        requests += 1
        return ModelResponse(parts=[ToolCallPart('run_command', {'command': 'true'})])

    workspace_harness.command_error = ClientError('auth denied')
    agent = Agent(
        FunctionModel(respond),
        deps_type=type(None),
        capabilities=[capability(workspace_harness), Shell[None](tools=['run_command'])],
    )
    with pytest.raises(WorkspaceUnavailableError):
        await agent.run('run')
    assert requests == 1
    assert workspace_harness.terminated == ['sb-1']


async def test_missing_file_is_reported_by_standard_filesystem(
    workspace_harness: WorkspaceHarness,
) -> None:
    returned: list[ToolReturnPart | RetryPromptPart] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if completed := parts(messages):
            returned.extend(completed)
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart('read_file', {'path': 'missing.txt'})])

    agent = Agent(
        FunctionModel(respond),
        deps_type=type(None),
        capabilities=[capability(workspace_harness), FileSystem[None]()],
    )
    assert (await agent.run('read')).output == 'done'
    assert isinstance(returned[0], ToolReturnPart)
    assert 'Path not found' in str(returned[0].content)
    assert workspace_harness.terminated == ['sb-1']


async def test_cleanup_failure_remains_visible_after_model_error(
    workspace_harness: WorkspaceHarness,
) -> None:
    original = RuntimeError('model failed')

    def fail(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise original

    cap = capability(workspace_harness)
    cap.files = {'seed.txt': 'seed'}
    workspace_harness.terminate_error = ClientError('denied')
    agent = Agent(FunctionModel(fail), deps_type=type(None), capabilities=[cap])
    with pytest.raises(RuntimeError, match='model failed') as caught:
        await agent.run('fail')
    assert caught.value is original
    assert isinstance(caught.value.__cause__, RenderSandboxCleanupError)
    assert any('sb-1' in note for note in caught.value.__notes__)


async def test_cleanup_failure_after_success_is_not_hidden(
    workspace_harness: WorkspaceHarness,
) -> None:
    workspace_harness.terminate_error = ClientError('denied')
    agent = Agent(
        tool_model(),
        deps_type=type(None),
        capabilities=[capability(workspace_harness), Shell[None](tools=['run_command'])],
    )
    with pytest.raises(RenderSandboxCleanupError):
        await agent.run('run')


def test_configuration_is_copied_and_repr_contains_no_secrets(
    workspace_harness: WorkspaceHarness,
) -> None:
    files = {'secret.txt': 'secret content'}
    cap = RenderSandbox[None](
        client=workspace_harness.client,
        config=RenderSandboxConfig(env={'TOKEN': 'secret env'}),
        files=files,
    )
    files['secret.txt'] = 'changed'
    assert cap.files == {'secret.txt': 'secret content'}
    assert 'secret' not in repr(cap)
    assert 'test-only' not in repr(cap)
    assert cap.get_serialization_name() is None


class WorkspaceWrappingCapability(WrapperCapability[None]):
    """Exercise public capability rebinding and a public workspace facade together."""

    def get_workspace(self, ctx: RunContext[None], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        backend = super().get_workspace(ctx, ref=ref)
        return WrapperWorkspace(Workspace(backend)) if backend is not None else None


@dataclass(frozen=True)
class IsolationSnapshot:
    ref: WorkspaceRef
    backend: WorkspaceBackend
    seed: bytes
    binary: bytes
    result: str


async def exercise_wrapped_run_isolation(
    harness: WorkspaceHarness, *, caller_model_error: Exception | None = None
) -> None:
    """Run the same observable scenario for the positive and deliberate fault cases."""
    names = ('alpha', 'beta', 'caller')
    configs = {
        name: replace(harness.config, env={'RUN_LABEL': name}, lifetime_seconds=300 + index)
        for index, name in enumerate(names)
    }
    captures: dict[str, IsolationSnapshot] = {}
    completed_tools: set[str] = set()
    all_tools_done = asyncio.Event()
    release_caller = asyncio.Event()
    providers: dict[str, RenderSandbox[None]] = {}

    def wrapped_provider(name: str) -> WrapperCapability[None]:
        async def finish(workspace: Workspace) -> None:
            assert isinstance(workspace, WrapperWorkspace)
            assert workspace.ref is not None
            captures[name] = IsolationSnapshot(
                ref=workspace.ref,
                backend=workspace.backend,
                seed=await workspace.read_bytes('seed.txt'),
                binary=await workspace.read_bytes('seed.bin'),
                result=await workspace.read_text('result.txt'),
            )

        provider = RenderSandbox[None](
            client=harness.client,
            config=configs[name],
            files={'seed.txt': f'seed:{name}', 'seed.bin': b'\x00\xff' + name.encode()},
            on_finish=finish,
        )
        providers[name] = provider
        # Transparent wrappers must preserve each provider's ownership across runs.
        return WrapperCapability(WorkspaceWrappingCapability(provider))

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        name = next(
            part.content
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, UserPromptPart)
        )
        assert isinstance(name, str)
        if name == 'caller' and caller_model_error is not None:
            raise caller_model_error
        if not parts(messages):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        'run_command',
                        {'command': 'printf "%s|%s" "$(cat seed.txt)" "$RUN_LABEL" > result.txt'},
                    )
                ]
            )
        completed_tools.add(name)
        if completed_tools == set(names):
            all_tools_done.set()
        # All writes overlap before any finish callback reads. Sharing either a
        # file root or command environment must now change an observable result.
        await all_tools_done.wait()
        if name == 'caller':
            await release_caller.wait()
        return ModelResponse(parts=[TextPart(f'done:{name}')])

    agent = Agent(
        FunctionModel(respond),
        deps_type=type(None),
        capabilities=[Shell[None](tools=['run_command'])],
    )
    caller_backend = RenderSandboxBackend(harness.client, config=configs['caller'])
    await caller_backend.working_dir()
    assert caller_backend.ref is not None
    caller_ref = caller_backend.ref
    tasks = [asyncio.create_task(agent.run(name, capabilities=[wrapped_provider(name)])) for name in names[:2]]
    caller_task = asyncio.create_task(
        agent.run(
            'caller',
            capabilities=[wrapped_provider('caller')],
            workspace=WrapperWorkspace(Workspace(caller_backend)),
        )
    )
    owned_results = asyncio.gather(*tasks)
    try:
        # This is a generous hang guard, not a latency assertion. Observe the caller
        # too: if it fails before the barrier, waiting only for the owned runs hides
        # that original failure behind a timeout while they await their sibling.
        async with asyncio.timeout(30):
            await asyncio.wait((owned_results, caller_task), return_when=asyncio.FIRST_COMPLETED)
            if caller_task.done():
                caller_task.result()  # Re-raise an early caller failure with its cause intact.
                raise AssertionError('caller completed before the ownership check released it')  # pragma: no cover
            results = await owned_results
        assert [result.output for result in results] == ['done:alpha', 'done:beta']
        assert completed_tools == set(names), 'runs must overlap before checking ownership'
        assert not caller_task.done(), 'caller-owned run must remain active during owned cleanup'
        owned_refs = {captures[name].ref.id for name in names[:2]}
        assert set(harness.terminated) == owned_refs
        assert caller_ref.id not in harness.terminated
        assert (await caller_backend.run(['printf', 'still usable'])).stdout == 'still usable'
        release_caller.set()
        assert (await asyncio.wait_for(caller_task, 15)).output == 'done:caller'
        assert caller_ref.id not in harness.terminated

        assert len({capture.ref.id for capture in captures.values()}) == 3
        assert len({id(capture.backend) for capture in captures.values()}) == 3
        # Check every run's seed files before command results: a shared result file
        # can also differ, but the file-sharing control should identify its root cause.
        for name in names:
            capture = captures[name]
            assert capture.seed == f'seed:{name}'.encode(), f'file content leaked for {name}'
            assert capture.binary == b'\x00\xff' + name.encode(), f'binary content leaked for {name}'
        for name in names:
            capture = captures[name]
            assert capture.result == f'seed:{name}|{name}', f'command configuration leaked for {name}'
            created = harness.create_calls[capture.ref.id]
            assert created.env == {'RUN_LABEL': name}, f'creation configuration leaked for {name}'
            assert created.timeout_seconds == configs[name].lifetime_seconds
            assert providers[name].config is configs[name]
            assert providers[name].files == {
                'seed.txt': f'seed:{name}',
                'seed.bin': b'\x00\xff' + name.encode(),
            }
    finally:
        release_caller.set()
        for task in [*tasks, caller_task]:
            if not task.done():
                task.cancel()
        owned_results.cancel()
        await asyncio.gather(*tasks, caller_task, return_exceptions=True)
        await asyncio.gather(owned_results, return_exceptions=True)
        await caller_backend.destroy()
    assert set(harness.terminated) == set(harness.created)


async def test_wrapped_concurrent_runs_isolate_files_config_and_cleanup_ownership(
    isolated_workspace_harness: WorkspaceHarness,
) -> None:
    await exercise_wrapped_run_isolation(isolated_workspace_harness)


@pytest.mark.parametrize('fault', ['files', 'environment'])
async def test_wrapped_isolation_oracle_rejects_shared_state(
    isolated_workspace_harness: WorkspaceHarness,
    fault: str,
) -> None:
    # Negative controls execute the real Agent and filesystem flow, not fabricated
    # snapshots. Distinct IDs still exist, so an ID-only assertion cannot pass these.
    isolated_workspace_harness.share_filesystem = fault == 'files'
    isolated_workspace_harness.share_environment = fault == 'environment'
    # Concurrent writes may first reveal the shared text file or the shared binary file.
    message = '(?:file|binary) content leaked' if fault == 'files' else 'command configuration leaked'
    with pytest.raises(AssertionError, match=message):
        await exercise_wrapped_run_isolation(isolated_workspace_harness)
    assert set(isolated_workspace_harness.terminated) == set(isolated_workspace_harness.created)


async def test_wrapped_caller_failure_surfaces_without_waiting_for_other_runs(
    isolated_workspace_harness: WorkspaceHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Hold both owned runs in an upload before the caller fails. This reproduces
    # cancellation during setup without relying on which local process finishes first.
    uploads_ready = asyncio.Event()
    owned_uploads: set[str] = set()
    sdk = isolated_workspace_harness.client.experimental.sandboxes
    copy_to = sdk.copy_to

    async def hold_owned_uploads(identifier: str, local: Path, remote: str) -> None:
        if isolated_workspace_harness.create_calls[identifier].env['RUN_LABEL'] != 'caller':
            owned_uploads.add(identifier)
            if len(owned_uploads) == 2:
                uploads_ready.set()
            # The test's sibling cancellation releases this deliberately pending operation.
            await asyncio.Future[None]()
        await uploads_ready.wait()
        await copy_to(identifier, local, remote)

    monkeypatch.setattr(sdk, 'copy_to', hold_owned_uploads)
    original = RuntimeError('caller model failed before the barrier')
    with pytest.raises(RuntimeError, match='caller model failed') as caught:
        await exercise_wrapped_run_isolation(isolated_workspace_harness, caller_model_error=original)
    assert caught.value is original
    # Both owned uploads were interrupted, so the cleanup assertion covers both siblings.
    assert len(owned_uploads) == 2
    assert set(isolated_workspace_harness.terminated) == set(isolated_workspace_harness.created)


async def test_durable_run_rejected_before_resource_creation(workspace_harness: WorkspaceHarness) -> None:
    from collections.abc import Awaitable, Callable
    from typing import NoReturn

    from pydantic_ai.durable_exec import (
        JSON_CODEC,
        BaseDurabilityCapability,
        CallableOperationBackend,
        DurabilityEngineSpec,
        DurableOperationId,
        JournalOperationNamer,
        RoleBasedOperationConfig,
    )
    from pydantic_ai.exceptions import UserError

    class UnsupportedBackend(CallableOperationBackend[None]):
        """Fail if the rejected engine reaches a model, tool, or workspace operation."""

        async def execute(
            self,
            *,
            operation_id: DurableOperationId,
            name: str,
            body: Callable[[], Awaitable[object]],
            cache_key: tuple[object, ...],
            config: None,
        ) -> NoReturn:
            raise AssertionError(
                'The unsupported engine must be rejected before executing operations'
            )  # pragma: no cover

    class UnsupportedDurability(BaseDurabilityCapability[None]):
        """Advertise an active durable run through the public capability interface."""

        engine_spec = DurabilityEngineSpec(
            engine_name='Unsupported', durable_unit_noun='unit', durable_container_noun='run', codec=JSON_CODEC
        )

        @property
        def in_durable_context(self) -> bool:
            return True

        def get_durable_operation_backend(self) -> UnsupportedBackend:
            # Register operations normally, so the test still uses the real agent setup path.
            return UnsupportedBackend(
                namer=JournalOperationNamer(self.name),
                config=RoleBasedOperationConfig(model=None, event=None, capability=None, tool=None),
            )

    agent = Agent(
        tool_model(),
        name='render_durable_rejection',
        deps_type=type(None),
        capabilities=[
            UnsupportedDurability(),
            RenderSandbox(id='sandbox', client=workspace_harness.client, config=workspace_harness.config),
        ],
    )
    with pytest.raises(UserError, match='only supports durable execution with RenderWorkflows'):
        await agent.run('Run a command')
    assert workspace_harness.created == []
    assert workspace_harness.commands == []


async def test_second_provider_cannot_destroy_first_providers_retained_workspace(
    workspace_harness: WorkspaceHarness,
) -> None:
    first = capability(workspace_harness, auto_cleanup=False)
    second = capability(workspace_harness, auto_cleanup=True)
    agent = Agent(tool_model(), deps_type=type(None), capabilities=[first, second, Shell()])
    result = await agent.run('Run a command')
    assert workspace_harness.created == ['sb-1']
    assert workspace_harness.terminated == []
    assert result.workspace.ref is not None
    await first.destroy(result.workspace.ref)
    assert workspace_harness.terminated == ['sb-1']


async def test_documented_quickstart(workspace_harness: WorkspaceHarness, capsys: pytest.CaptureFixture[str]) -> None:
    import re
    from collections.abc import Awaitable, Callable
    from pathlib import Path
    from types import ModuleType

    from pydantic_ai.settings import ModelSettings

    page = Path(__file__).resolve().parents[3] / 'docs/harness/render-sandbox.md'
    snippet = re.search(r'```python[^\n]*\n(.*?)```', page.read_text(), re.DOTALL)
    assert snippet is not None
    example = ModuleType('render_guide_example')
    exec(compile(snippet[1], str(page), 'exec'), example.__dict__)

    def configured_provider(*, on_finish: Callable[[Workspace], Awaitable[None]]) -> RenderSandbox[None]:
        return RenderSandbox(client=workspace_harness.client, config=workspace_harness.config, on_finish=on_finish)

    def configured_agent(model: str, *, capabilities: list[AbstractCapability[None]], model_settings: ModelSettings):
        assert model == 'anthropic:claude-haiku-5-5'
        assert model_settings == {'parallel_tool_calls': False}
        return Agent(
            tool_model('write_file', {'path': 'answer.txt', 'content': '42'}),
            deps_type=type(None),
            capabilities=capabilities,
            model_settings=model_settings,
        )

    setattr(example, 'RenderSandbox', configured_provider)
    setattr(example, 'Agent', configured_agent)
    await example.main()
    assert capsys.readouterr().out == 'Verified answer.txt: 42\n'
    assert workspace_harness.created == workspace_harness.terminated == ['sb-1']


async def test_context_closes_sdk_pool_without_destroying_retained_workspace(
    workspace_harness: WorkspaceHarness,
) -> None:
    client = workspace_harness.client.experimental.sandboxes.client.get_async_httpx_client()
    async with capability(workspace_harness, auto_cleanup=False) as provider:
        agent = Agent(tool_model(), deps_type=type(None), capabilities=[provider, Shell()])
        await agent.run('Use the workspace')
        assert not client.is_closed
    assert client.is_closed
    assert workspace_harness.terminated == []


def test_deferred_provider_is_rejected_before_creation(workspace_harness: WorkspaceHarness) -> None:
    from pydantic_ai.exceptions import UserError

    # The workspace must be chosen before model tools load deferred capabilities.
    with pytest.raises(UserError, match='cannot be deferred'):
        RenderSandbox(client=workspace_harness.client, defer_loading=True)
    assert workspace_harness.created == []


async def test_foreign_reference_is_resolved_by_its_own_provider(
    workspace_harness: WorkspaceHarness, tmp_path: Path
) -> None:
    from pydantic_ai.models.test import TestModel

    reference = WorkspaceRef(provider='other', id='external')

    class OtherBackend:
        ref = reference

        async def working_dir(self) -> str:
            return str(tmp_path)

    class OtherProvider(AbstractCapability[None]):
        def get_workspace(self, ctx: RunContext[None], *, ref: WorkspaceRef | None) -> WorkspaceBackend:
            assert ref == reference
            return OtherBackend()

    # Render declines the foreign reference, allowing the next provider to handle it.
    agent = Agent(
        TestModel(custom_output_text='done'),
        deps_type=type(None),
        capabilities=[RenderSandbox(client=workspace_harness.client), OtherProvider()],
    )
    result = await agent.run('use the supplied workspace', workspace=reference)
    assert result.output == 'done'
    assert result.workspace.ref == reference
    assert await result.workspace.working_dir() == str(tmp_path)
    assert workspace_harness.created == []
    assert workspace_harness.terminated == []


def test_public_import_explains_missing_optional_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import sys

    import pydantic_ai_harness.render_sandbox as public_provider

    with monkeypatch.context() as missing_sdk:
        missing_sdk.setitem(sys.modules, 'render', None)
        missing_sdk.delitem(sys.modules, 'pydantic_ai_harness.render_sandbox._backend')
        with pytest.raises(ImportError, match=r'pydantic-ai-harness\[render-sandbox\]'):
            importlib.reload(public_provider)
    # Restore the public module after the optional-dependency probe.
    importlib.reload(public_provider)
