from __future__ import annotations

from collections.abc import Generator
from unittest.mock import MagicMock

import pytest

from ...conftest import try_import

with try_import() as imports_successful:
    from durabletask import task

    from .worker import InMemoryWorker

pytestmark = pytest.mark.skipif(not imports_successful(), reason='durabletask not installed')


def test_worker_completes_synchronous_orchestrators(durable_worker: InMemoryWorker) -> None:
    def orchestrator(context: task.OrchestrationContext, payload: object) -> str:
        return 'done'

    durable_worker.add_orchestrator(orchestrator)
    instance_id = durable_worker.client.schedule_new_orchestration(orchestrator)
    state = durable_worker.wait(instance_id)
    assert state is not None
    state.raise_if_failed()
    assert state.get_output() == 'done'


def test_worker_bounds_busy_dispatches(durable_worker: InMemoryWorker, monkeypatch: pytest.MonkeyPatch) -> None:
    step = MagicMock(return_value=True)
    monkeypatch.setattr(durable_worker, 'step', step)
    with pytest.raises(pytest.fail.Exception, match='did not become idle after 200 turns'):
        durable_worker.run_until_idle()
    assert step.call_count == 200


def test_worker_rejects_unsupported_sdk_actions(durable_worker: InMemoryWorker) -> None:
    def orchestrator(context: task.OrchestrationContext, payload: object) -> str:
        context.send_event(context.instance_id, 'unsupported')
        return 'done'

    durable_worker.add_orchestrator(orchestrator)
    instance_id = durable_worker.client.schedule_new_orchestration(orchestrator)
    with pytest.raises(pytest.fail.Exception, match='Unsupported SDK action in fixture') as exc:
        durable_worker.wait(instance_id)
    assert 'sendEvent' in str(exc.value)


def test_worker_waits_for_an_external_event(durable_worker: InMemoryWorker) -> None:
    def orchestrator(context: task.OrchestrationContext, payload: object) -> Generator[task.Task[object], object, str]:
        response = yield context.wait_for_external_event('resume')
        assert isinstance(response, str)
        return response

    durable_worker.add_orchestrator(orchestrator)
    instance_id = durable_worker.client.schedule_new_orchestration(orchestrator)
    with pytest.raises(TimeoutError, match='has not completed'):
        durable_worker.client.wait_for_orchestration_completion(instance_id)
    durable_worker.client.raise_orchestration_event(instance_id, 'resume', data='resumed')
    state = durable_worker.client.wait_for_orchestration_completion(instance_id)
    assert state is not None
    state.raise_if_failed()
    assert state.get_output() == 'resumed'
