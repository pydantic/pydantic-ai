"""Exercise Sandbox ownership across the real Workflows JSON task boundary.

Render's HTTP calls are replaced with local files and processes. These tests check
our task wiring and cleanup rules without creating paid cloud resources.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NoReturn

import pytest

from ...conftest import try_import

with try_import() as imports_successful:
    from render import TaskContext, Workflows

    from pydantic_ai import Agent, RunContext
    from pydantic_ai.capabilities import AbstractCapability
    from pydantic_ai.exceptions import UserError
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.workspaces import (
        ReadOnlyWorkspace,
        Workspace,
        WorkspaceBackend,
        WorkspaceReadOnlyError,
        WorkspaceRef,
        WorkspaceUnavailableError,
    )
    from pydantic_ai_harness import RenderSandbox, RenderWorkflows
    from pydantic_ai_harness.render_sandbox import RenderSandboxCleanupError, RenderSandboxEvent

    from ..render.conftest import WorkspaceTaskContext
    from .workspace_helpers import WorkspaceHarness

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='render not installed'),
    pytest.mark.subprocess(reason='Exercise real POSIX workspace commands and file helper processes', conditional=True),
]


class TestRenderSandboxWorkflows:
    async def test_seed_tool_finish_and_cleanup(self, workspace_harness: WorkspaceHarness) -> None:
        # One agent run crosses separate model, tool, workspace, and cleanup tasks.
        runtime = RenderWorkflows[None](Workflows())
        context = WorkspaceTaskContext()
        finished: list[str] = []
        lifecycle_tasks: list[tuple[str, bool]] = []

        def record(event: RenderSandboxEvent) -> None:
            # Creation and termination must both happen inside child tasks.
            if event.event in {'created', 'terminated'}:
                lifecycle_tasks.append((event.event, context.in_child_task.get()))

        async def verify(workspace: Workspace) -> None:
            # The callback reads through the public workspace before deletion.
            finished.append(await workspace.read_text('answer.txt'))
            assert workspace_harness.terminated == []

        sandbox = RenderSandbox[None](
            id='sandbox',
            client=workspace_harness.client,
            config=workspace_harness.config,
            files={'input.txt': '42'},
            on_finish=verify,
            on_sandbox_event=record,
        )
        agent = Agent(
            TestModel(call_tools=['save_answer'], custom_output_text='done'),
            name='sandbox_workflows',
            deps_type=type(None),
            capabilities=[runtime, sandbox],
        )

        @agent.tool
        async def save_answer(ctx: RunContext[None]) -> str:
            # A tool task reconstructs the same sandbox from its portable reference.
            assert context.in_child_task.get()
            answer = await ctx.workspace.read_text('input.txt')
            await ctx.workspace.write_text('answer.txt', answer)
            command = await ctx.workspace.run(['cat', 'answer.txt'])
            assert command.stdout == '42'
            return answer

        @runtime.task
        async def root(ctx: TaskContext) -> str:
            del ctx
            return (await agent.run('save the answer')).output

        # Invoke the SDK task definition, retaining real JSON serialization between tasks.
        pending = root.func(context)
        assert inspect.isawaitable(pending)
        assert await pending == 'done'
        assert finished == ['42']
        assert lifecycle_tasks == [('created', True), ('terminated', True)]
        assert workspace_harness.created == ['sb-1']
        assert workspace_harness.terminated == ['sb-1']
        assert any(name.endswith('workspace.call') for name in context.task_names)
        assert any(name.endswith('.destroy') for name in context.task_names)

    @pytest.mark.parametrize('attachment', ['ref', 'history', 'backend'])
    async def test_caller_keeps_attached_sandbox(self, workspace_harness: WorkspaceHarness, attachment: str) -> None:
        # A later run may borrow an existing environment, but must not delete it.
        runtime = RenderWorkflows[None](Workflows())
        sandbox = RenderSandbox[None](
            id='sandbox', client=workspace_harness.client, config=workspace_harness.config, auto_cleanup=False
        )
        agent = Agent(TestModel(), name='retained', deps_type=type(None), capabilities=[runtime, sandbox])
        context = WorkspaceTaskContext()

        @runtime.task
        async def root(ctx: TaskContext) -> str:
            del ctx
            first = await agent.run('first')
            ref = first.workspace.ref
            assert ref is not None
            await first.workspace.write_text('retained.txt', 'keep me')
            sandbox.auto_cleanup = True
            if attachment == 'history':
                second = await agent.run('again', message_history=first.all_messages())
            else:
                workspace = sandbox.backend(ref) if attachment == 'backend' else ref
                second = await agent.run('again', workspace=workspace)
            assert second.workspace.ref == ref
            assert workspace_harness.terminated == []
            return await second.workspace.read_text('retained.txt')

        pending = root.func(context)
        assert inspect.isawaitable(pending)
        assert await pending == 'keep me'
        assert workspace_harness.created == ['sb-1']
        assert not any(name.endswith('.destroy') for name in context.task_names)
        await sandbox.destroy(WorkspaceRef(provider='render', id='sb-1'))
        assert workspace_harness.terminated == ['sb-1']

    @pytest.mark.parametrize('phase', ['model', 'seed', 'finish'])
    async def test_failure_cleans_owned_sandbox(self, workspace_harness: WorkspaceHarness, phase: str) -> None:
        # Errors at each stage must still dispatch cleanup for the created environment.
        runtime = RenderWorkflows[None](Workflows())

        def fail_model(*args: object) -> NoReturn:
            raise RuntimeError('model failed')

        async def finish(workspace: Workspace) -> None:
            del workspace
            # Model and seed failures occur before this callback; the remaining case fails here.
            raise RuntimeError('finish failed')

        if phase == 'seed':
            workspace_harness.transfer_error = RuntimeError('seed failed')
        sandbox = RenderSandbox[None](
            id='sandbox',
            client=workspace_harness.client,
            config=workspace_harness.config,
            files={'input.txt': 'data'},
            on_finish=finish,
        )
        agent = Agent(
            FunctionModel(fail_model) if phase == 'model' else TestModel(),
            name='failure',
            deps_type=type(None),
            capabilities=[runtime, sandbox],
        )

        @runtime.task
        async def root(ctx: TaskContext) -> str:
            del ctx
            return (await agent.run('go')).output

        context = WorkspaceTaskContext()
        pending = root.func(context)
        assert inspect.isawaitable(pending)
        with pytest.raises(RuntimeError, match=f'{phase} failed'):
            await pending
        assert workspace_harness.created == workspace_harness.terminated == ['sb-1']
        assert any(name.endswith('.destroy') for name in context.task_names)

    @pytest.mark.parametrize('failure', ['timeout', 'errored'])
    @pytest.mark.parametrize('ownership', ['owned', 'retained', 'attached'])
    async def test_preparation_failure_respects_ownership(
        self, workspace_harness: WorkspaceHarness, failure: str, ownership: str
    ) -> None:
        # Preparation happens before the run hook. Only a newly owned sandbox should
        # be removed when it never becomes ready; retained and borrowed IDs survive.
        harness = workspace_harness
        harness.create_status = 'starting' if failure == 'timeout' else 'errored'
        runtime = RenderWorkflows[None](Workflows())
        sandbox = RenderSandbox[None](
            id='sandbox',
            client=harness.client,
            config=replace(harness.config, readiness_timeout=0.01),
            auto_cleanup=ownership != 'retained',
        )
        ref: WorkspaceRef | None = None
        if ownership == 'attached':
            existing = await harness.client.experimental.sandboxes.create(
                timeout_seconds=300, network_policy='deny-all', env=None
            )
            ref = WorkspaceRef(provider='render', id=existing.id)
        agent = Agent(TestModel(), name='preparation_failure', deps_type=type(None), capabilities=[runtime, sandbox])

        @runtime.task
        async def root(ctx: TaskContext) -> str:
            del ctx
            return (await agent.run('go', workspace=ref)).output

        pending = root.func(WorkspaceTaskContext())
        assert inspect.isawaitable(pending)
        message = 'readiness exceeded' if failure == 'timeout' else 'sandbox is errored'
        with pytest.raises(WorkspaceUnavailableError, match=message):
            await pending
        assert harness.created == ['sb-1']
        assert harness.terminated == (['sb-1'] if ownership == 'owned' else [])
        # The test owns retained and attached resources, just as an application would.
        if ownership != 'owned':
            await sandbox.destroy(WorkspaceRef(provider='render', id='sb-1'))

    async def test_cancelled_preparation_waits_for_cleanup(self, workspace_harness: WorkspaceHarness) -> None:
        # Cancellation can arrive before create returns its ID. Wait for that result,
        # then finish termination even if cancellation is requested again during it.
        harness = workspace_harness
        harness.create_gate = asyncio.Event()
        harness.terminate_gate = asyncio.Event()
        runtime = RenderWorkflows[None](Workflows())
        sandbox = RenderSandbox[None](id='sandbox', client=harness.client, config=harness.config)
        agent = Agent(TestModel(), name='cancelled_preparation', deps_type=type(None), capabilities=[runtime, sandbox])

        @runtime.task
        async def root(ctx: TaskContext) -> str:
            del ctx
            return (await agent.run('go')).output

        async def run() -> None:
            pending = root.func(WorkspaceTaskContext())
            assert inspect.isawaitable(pending)
            await pending

        task = asyncio.create_task(run())
        await harness.create_started.wait()
        task.cancel()
        harness.create_gate.set()
        await harness.terminate_started.wait()
        task.cancel()
        harness.terminate_gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert harness.created == harness.terminated == ['sb-1']

    async def test_preparation_and_cleanup_failures_remain_visible(self, workspace_harness: WorkspaceHarness) -> None:
        # A cleanup failure must not disappear when Workflows serializes an expected
        # workspace error. Both errors and the known sandbox ID are needed for recovery.
        harness = workspace_harness
        harness.create_status = 'errored'
        harness.terminate_error = RuntimeError('termination failed')
        runtime = RenderWorkflows[None](Workflows())
        sandbox = RenderSandbox[None](id='sandbox', client=harness.client, config=harness.config)
        agent = Agent(
            TestModel(), name='failed_preparation_cleanup', deps_type=type(None), capabilities=[runtime, sandbox]
        )

        @runtime.task
        async def root(ctx: TaskContext) -> str:
            del ctx
            return (await agent.run('go')).output

        pending = root.func(WorkspaceTaskContext())
        assert inspect.isawaitable(pending)
        with pytest.raises(ExceptionGroup, match='sb-1') as caught:
            await pending
        assert len(caught.value.exceptions) == 2
        assert isinstance(caught.value.exceptions[0], WorkspaceUnavailableError)
        assert isinstance(caught.value.exceptions[1], RenderSandboxCleanupError)
        assert harness.created == ['sb-1']
        assert harness.terminated == []
        harness.terminate_error = None
        await sandbox.destroy(WorkspaceRef(provider='render', id='sb-1'))

    async def test_concurrent_runs_and_root_retry(self, isolated_workspace_harness: WorkspaceHarness) -> None:
        # Overlapping root runs must keep distinct files and terminate only their own ID.
        harness = isolated_workspace_harness
        runtime = RenderWorkflows[None](Workflows())
        ready = asyncio.Event()
        refs: list[WorkspaceRef] = []

        async def finish(workspace: Workspace) -> None:
            ref = workspace.ref
            assert ref is not None
            refs.append(ref)
            await workspace.write_text('identity.txt', ref.id)
            if len(refs) == 2:
                ready.set()
            await ready.wait()
            assert await workspace.read_text('identity.txt') == ref.id

        sandbox = RenderSandbox[None](id='sandbox', client=harness.client, config=harness.config, on_finish=finish)
        agent = Agent(TestModel(), name='concurrent', deps_type=type(None), capabilities=[runtime, sandbox])

        @runtime.task
        async def root(ctx: TaskContext) -> str:
            del ctx
            return (await agent.run('go')).output

        async def run() -> None:
            pending = root.func(WorkspaceTaskContext())
            assert inspect.isawaitable(pending)
            await pending

        await asyncio.gather(run(), run())
        # Re-running the entry task starts a fresh environment, as Render root retries do.
        await run()
        assert len(set(refs)) == 3
        assert sorted(harness.terminated) == harness.created == ['sb-1', 'sb-2', 'sb-3']

    async def test_cancellation_waits_for_cleanup(self, workspace_harness: WorkspaceHarness) -> None:
        # Stop a run while its finish callback waits, then cancel it again during cleanup.
        runtime = RenderWorkflows[None](Workflows())
        finishing = asyncio.Event()
        release = asyncio.Event()
        workspace_harness.terminate_gate = release

        async def finish(workspace: Workspace) -> None:
            del workspace
            finishing.set()
            await asyncio.Event().wait()

        sandbox = RenderSandbox[None](
            id='sandbox', client=workspace_harness.client, config=workspace_harness.config, on_finish=finish
        )
        agent = Agent(TestModel(), name='cancelled', deps_type=type(None), capabilities=[runtime, sandbox])

        @runtime.task
        async def root(ctx: TaskContext) -> str:
            del ctx
            return (await agent.run('go')).output

        async def run() -> None:
            pending = root.func(WorkspaceTaskContext())
            assert inspect.isawaitable(pending)
            await pending

        task = asyncio.create_task(run())
        await finishing.wait()
        task.cancel()
        await workspace_harness.terminate_started.wait()
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert workspace_harness.created == workspace_harness.terminated == ['sb-1']

    async def test_cleanup_failure_preserves_run_error(self, workspace_harness: WorkspaceHarness) -> None:
        # Keep both failures visible so the caller knows that termination needs attention.
        runtime = RenderWorkflows[None](Workflows())
        workspace_harness.terminate_error = RuntimeError('termination failed')

        async def finish(workspace: Workspace) -> None:
            del workspace
            raise ValueError('verification failed')

        sandbox = RenderSandbox[None](
            id='sandbox', client=workspace_harness.client, config=workspace_harness.config, on_finish=finish
        )
        agent = Agent(TestModel(), name='cleanup_failure', deps_type=type(None), capabilities=[runtime, sandbox])

        @runtime.task
        async def root(ctx: TaskContext) -> str:
            del ctx
            return (await agent.run('go')).output

        pending = root.func(WorkspaceTaskContext())
        assert inspect.isawaitable(pending)
        with pytest.raises(ValueError, match='verification failed') as caught:
            await pending
        assert isinstance(caught.value.__cause__, RenderSandboxCleanupError)
        assert 'sb-1' in str(caught.value.__notes__)
        assert workspace_harness.terminated == []
        workspace_harness.terminate_error = None
        await sandbox.destroy(WorkspaceRef(provider='render', id='sb-1'))

    async def test_other_provider_and_read_only_policy(self, workspace_harness: WorkspaceHarness) -> None:
        # A wrapper may restrict file changes; ownership inspection must not bypass it.
        @dataclass(kw_only=True)
        class ReadOnlySandbox(RenderSandbox[None]):
            def get_workspace(self, ctx: RunContext[None], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
                backend = super().get_workspace(ctx, ref=ref)
                return ReadOnlyWorkspace(Workspace(backend)) if backend is not None else None

        runtime = RenderWorkflows[None](Workflows())

        async def check_policy(workspace: Workspace) -> None:
            # Reject an actual write, not just the descriptive read_only flag.
            with pytest.raises(WorkspaceReadOnlyError):
                await workspace.write_text('forbidden.txt', 'do not write')

        first = ReadOnlySandbox(
            id='first', client=workspace_harness.client, config=workspace_harness.config, on_finish=check_policy
        )
        second = RenderSandbox[None](id='second', client=workspace_harness.client, config=workspace_harness.config)
        agent = Agent(TestModel(), name='read_only', deps_type=type(None), capabilities=[first, second, runtime])

        @runtime.task
        async def root(ctx: TaskContext) -> str:
            del ctx
            result = await agent.run('go')
            assert result.workspace.read_only
            return result.output

        context = WorkspaceTaskContext()
        pending = root.func(context)
        assert inspect.isawaitable(pending)
        await pending
        assert workspace_harness.created == workspace_harness.terminated == ['sb-1']
        assert sum(name.endswith('.destroy') for name in context.task_names) == 1

    def test_stable_id_required(self, workspace_harness: WorkspaceHarness) -> None:
        # Stored cleanup task names need an explicit, stable capability ID.
        with pytest.raises(UserError, match='needs an explicit `id`'):
            Agent(
                TestModel(),
                name='missing_id',
                capabilities=[
                    RenderWorkflows(Workflows()),
                    RenderSandbox(client=workspace_harness.client, config=workspace_harness.config),
                ],
            )

    async def test_documented_workflows_example(
        self, workspace_harness: WorkspaceHarness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Execute the published wiring unchanged, replacing only external services.
        import pydantic_ai
        import pydantic_ai_harness

        def example_sandbox(*, id: str) -> RenderSandbox[None]:
            return RenderSandbox(id=id, client=workspace_harness.client, config=workspace_harness.config)

        def example_agent(model: str, *, name: str, capabilities: list[AbstractCapability[None]]) -> Agent[None, str]:
            del model
            return Agent(
                TestModel(call_tools=[], custom_output_text='verified'),
                name=name,
                deps_type=type(None),
                capabilities=capabilities,
            )

        monkeypatch.setattr(pydantic_ai_harness, 'RenderSandbox', example_sandbox)
        monkeypatch.setattr(pydantic_ai, 'Agent', example_agent)
        path = Path(__file__).parents[3] / 'src/pydantic_ai_harness/pydantic_ai_harness/render_sandbox/README.md'
        section = path.read_text().split('## Use with Render Workflows', 1)[1]
        snippet = section.split('```python\n', 1)[1].split('```', 1)[0]
        # Execute the call in the snippet's namespace, then check the returned value.
        namespace: dict[str, object] = {'test_context': WorkspaceTaskContext()}
        invocation = "\nexample_result = solve.func(test_context, 'verify the documented wiring')\n"
        exec(compile(snippet + invocation, str(path), 'exec'), namespace)
        pending = namespace['example_result']
        assert inspect.isawaitable(pending)
        assert await pending == 'verified'
        assert workspace_harness.created == workspace_harness.terminated == ['sb-1']
