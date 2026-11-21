from __future__ import annotations

import json
from collections.abc import Generator
from contextlib import ExitStack
from pathlib import Path

import pytest
from pydantic import JsonValue, TypeAdapter

from pydantic_ai import Agent, UserError
from pydantic_ai.models.test import TestModel

from ...conftest import try_import

with try_import() as imports_successful:
    from durabletask import task
    from durabletask.client import OrchestrationFailedError, OrchestrationStatus
    from durabletask.internal.exceptions import AbandonOrchestrationError
    from durabletask.worker import VersionFailureStrategy, VersioningOptions, VersionMatchStrategy

    from pydantic_ai_harness.azure_durable_task import AzureDurableTaskDurability, run_durable

    from .generation_example import README, python_examples
    from .worker import InMemoryWorker, completed, registered_agent

pytestmark = pytest.mark.skipif(not imports_successful(), reason='durabletask not installed')


class TestNativeVersioning:
    def test_documented_stacked_registration(
        self, durable_worker: InMemoryWorker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        starts: list[bool] = []
        results: list[JsonValue] = []
        old = Agent(
            TestModel(custom_output_text='old code'), name='support-1.0', capabilities=[AzureDurableTaskDurability()]
        )
        new = Agent(
            TestModel(custom_output_text='new code'), name='support-2.0', capabilities=[AzureDurableTaskDurability()]
        )
        agents = {'1.0': old, '2.0': new}

        def support(ctx: task.OrchestrationContext, prompt: str) -> Generator[task.Task[object], object, str]:
            version = ctx.version
            assert version is not None
            result = yield from run_durable(ctx, lambda: agents[version].run(prompt))
            return result.output

        durable_worker.add_orchestrator(support)

        def drain(prompt: str) -> str:
            assert prompt == 'Press Enter after stopping traffic and draining the worker.\n'
            for version in ['1.0', '2.0']:
                instance_id = durable_worker.schedule(support, input='Hi', version=version)
                results.append(completed(durable_worker, instance_id))
            return ''

        source = next(source for source in python_examples() if 'with old_durability.register(worker)' in source)
        monkeypatch.setattr(durable_worker, 'start', lambda: starts.append(True))
        monkeypatch.setattr(durable_worker, 'stop', lambda: starts.append(False))
        monkeypatch.setattr('builtins.input', drain)
        exec(
            compile(source, f'{README}:versioning', 'exec', dont_inherit=True),
            {'old_agent': old, 'new_agent': new, 'worker': durable_worker},
        )
        assert starts == [True, False]
        assert results == ['old code', 'new code']

    @pytest.mark.parametrize('names', [('agent-1.0', 'agent-2.0'), ('version_1', 'version<1')])
    def test_retained_activity_dispatch(self, durable_worker: InMemoryWorker, names: tuple[str, str]) -> None:
        old = Agent(
            TestModel(custom_output_text='old code'), name=names[0], capabilities=[AzureDurableTaskDurability()]
        )
        new = Agent(
            TestModel(custom_output_text='new code'), name=names[1], capabilities=[AzureDurableTaskDurability()]
        )
        versions = {'1.0': old, '2.0': new}

        def workflow(ctx: task.OrchestrationContext, prompt: str) -> Generator[task.Task[object], object, str]:
            version = ctx.version
            assert version is not None
            result = yield from run_durable(ctx, lambda: versions[version].run(prompt))
            return result.output

        durable_worker.use_versioning(
            VersioningOptions(
                version='2.0',
                match_strategy=VersionMatchStrategy.CURRENT_OR_OLDER,
                failure_strategy=VersionFailureStrategy.REJECT,
            )
        )
        durable_worker.add_orchestrator(workflow)
        old_capability = AzureDurableTaskDurability.from_agent(old)
        capability = AzureDurableTaskDurability.from_agent(new)
        assert old_capability is not None and capability is not None
        with old_capability.register(durable_worker), capability.register(durable_worker):
            old_id = durable_worker.schedule(workflow, input='old', version='1.0')
            new_id = durable_worker.schedule(workflow, input='new', version='2.0')
            assert completed(durable_worker, old_id) == 'old code'
            assert completed(durable_worker, new_id) == 'new code'
        assert durable_worker.activity_calls == (
            ['agent--1.0__model.request', 'agent--2.0__model.request']
            if names[0] == 'agent-1.0'
            else ['version_1__model.request', 'version-x3C-1__model.request']
        )

    @pytest.mark.parametrize('failure', ['REJECT', 'FAIL'])
    @pytest.mark.parametrize('match', ['STRICT', 'CURRENT_OR_OLDER'])
    def test_native_version_admission(self, durable_worker: InMemoryWorker, failure: str, match: str) -> None:
        agent = Agent(TestModel(), name='admission', capabilities=[AzureDurableTaskDurability()])

        def workflow(ctx: task.OrchestrationContext, prompt: str) -> Generator[task.Task[object], object, str]:
            result = yield from run_durable(ctx, lambda: agent.run(prompt))
            return result.output

        durable_worker.add_orchestrator(workflow)
        durable_worker.use_versioning(
            VersioningOptions(
                version='1.0',
                match_strategy=VersionMatchStrategy[match],
                failure_strategy=VersionFailureStrategy[failure],
            )
        )
        instance_id = durable_worker.schedule(workflow, input='future', version='2.0')
        with registered_agent(durable_worker, agent):
            if failure == 'REJECT':
                with pytest.raises(AbandonOrchestrationError):
                    durable_worker.step()
                state = durable_worker.state(instance_id)
                assert state is not None and state.runtime_status == OrchestrationStatus.RUNNING
                assert durable_worker.activity_calls == []
                durable_worker.use_versioning(
                    VersioningOptions(
                        version='2.0',
                        match_strategy=VersionMatchStrategy[match],
                        failure_strategy=VersionFailureStrategy.REJECT,
                    )
                )
                assert isinstance(completed(durable_worker, instance_id), str)
            else:
                with pytest.raises(OrchestrationFailedError, match='version'):
                    completed(durable_worker, instance_id)
        assert durable_worker.activity_calls == (['admission__model.request'] if failure == 'REJECT' else [])

    def test_sdk_rejects_duplicate_activity_names(self, durable_worker: InMemoryWorker) -> None:
        old = Agent(TestModel(), name='version', capabilities=[AzureDurableTaskDurability()])
        new = Agent(TestModel(), name='version', capabilities=[AzureDurableTaskDurability()])
        old_capability = AzureDurableTaskDurability.from_agent(old)
        capability = AzureDurableTaskDurability.from_agent(new)
        assert old_capability is not None and capability is not None
        with pytest.raises(ValueError, match=r"A 'version__[^']+' activity already exists"):
            with ExitStack() as stack:
                stack.enter_context(old_capability.register(durable_worker))
                stack.enter_context(capability.register(durable_worker))
        with old_capability.register(InMemoryWorker()), capability.register(InMemoryWorker()):
            assert old_capability.get_durable_operation_backend().registrations()
            assert capability.get_durable_operation_backend().registrations()

    def test_registration_scope_rolls_back_after_failure(self, durable_worker: InMemoryWorker) -> None:
        old = Agent(TestModel(), name='scope-old', capabilities=[AzureDurableTaskDurability()])
        new = Agent(TestModel(), name='scope-new', capabilities=[AzureDurableTaskDurability()])
        old_capability = AzureDurableTaskDurability.from_agent(old)
        capability = AzureDurableTaskDurability.from_agent(new)
        assert old_capability is not None and capability is not None
        replacement = InMemoryWorker()
        with old_capability.register(durable_worker):
            with pytest.raises(UserError, match='already registered'):
                with ExitStack() as stack:
                    stack.enter_context(capability.register(replacement))
                    stack.enter_context(old_capability.register(replacement))
            with capability.register(durable_worker):
                assert capability.get_durable_operation_backend().registrations()

    def test_renamed_activity_breaks_recorded_replay(self, durable_worker: InMemoryWorker) -> None:
        recording = json.loads(Path(__file__).with_name('model-tool-replay.json').read_text(encoding='utf-8'))
        agent = Agent(TestModel(), name='renamed-agent', capabilities=[AzureDurableTaskDurability()])

        @agent.tool_plain
        async def baseline_tool() -> str:
            return 'not called'  # pragma: no cover

        def workflow(ctx: task.OrchestrationContext, prompt: str) -> Generator[task.Task[object], object, JsonValue]:
            result = yield from run_durable(ctx, lambda: agent.run(prompt))
            return result.output  # pragma: no cover

        workflow.__name__ = recording['name']
        durable_worker.add_orchestrator(workflow)
        with registered_agent(durable_worker, agent):
            actions = durable_worker.replay_history(recording)
        assert len(actions) == 1
        failure = actions[0].completeOrchestration.failureDetails
        assert failure.errorType == 'durabletask.task.NonDeterminismError'
        assert 'renamed--agent__model.request' in failure.errorMessage
        assert durable_worker.activity_calls == []

    def test_native_migration_preserves_unprocessed_events(self, durable_worker: InMemoryWorker) -> None:
        def workflow(
            ctx: task.OrchestrationContext, payload: object
        ) -> Generator[task.Task[object], object, JsonValue]:
            if ctx.version == '1.0':
                migration = yield ctx.wait_for_external_event('migrate')
                ctx.continue_as_new({'migrated': migration}, save_events=True, new_version='2.0')
                return None
            assert ctx.version == '2.0'
            decision = yield ctx.wait_for_external_event('review:1')
            return TypeAdapter[JsonValue](JsonValue).validate_python({'payload': payload, 'decision': decision})

        durable_worker.add_orchestrator(workflow)
        instance_id = durable_worker.schedule(workflow, version='1.0')
        durable_worker.run_until_idle()
        durable_worker.raise_event(instance_id, 'review:1', data={'call-A': 'deny'})
        durable_worker.raise_event(instance_id, 'migrate', data={'messages': ['retained']})
        assert completed(durable_worker, instance_id) == {
            'payload': {'migrated': {'messages': ['retained']}},
            'decision': {'call-A': 'deny'},
        }
        assert len(durable_worker.instances[instance_id].generations) == 1
