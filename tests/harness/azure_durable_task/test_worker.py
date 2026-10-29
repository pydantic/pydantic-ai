from __future__ import annotations

from collections.abc import Generator

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
    instance_id = durable_worker.schedule(orchestrator)
    state = durable_worker.wait(instance_id)
    state.raise_if_failed()
    assert state.get_output() == 'done'


def test_worker_waits_for_an_external_event(durable_worker: InMemoryWorker) -> None:
    def orchestrator(context: task.OrchestrationContext, payload: object) -> Generator[task.Task[object], object, str]:
        response = yield context.wait_for_external_event('resume')
        assert isinstance(response, str)
        return response

    durable_worker.add_orchestrator(orchestrator)
    instance_id = durable_worker.schedule(orchestrator)
    with pytest.raises(TimeoutError, match='has not completed'):
        durable_worker.wait(instance_id)
    durable_worker.raise_event(instance_id, 'resume', data='resumed')
    state = durable_worker.wait(instance_id)
    state.raise_if_failed()
    assert state.get_output() == 'resumed'
