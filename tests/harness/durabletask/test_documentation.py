from __future__ import annotations

import asyncio
import inspect
import re
from pathlib import Path
from typing import Any

import pytest

from pydantic_ai.models.test import TestModel

from ...conftest import try_import

with try_import() as imports_successful:
    from durabletask.azuremanaged.client import DurableTaskSchedulerClient
    from durabletask.azuremanaged.worker import DurableTaskSchedulerWorker

    from .worker import InMemoryWorker

pytestmark = pytest.mark.skipif(not imports_successful(), reason='durabletask-azuremanaged not installed')


def test_documented_examples(
    durable_worker: InMemoryWorker, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Execute the published examples unchanged, substituting only the external services."""
    worker_signature = inspect.signature(DurableTaskSchedulerWorker)
    client_signature = inspect.signature(DurableTaskSchedulerClient)
    connections: list[dict[str, object]] = []
    credential = object()
    azure_worker = InMemoryWorker()

    def worker(**kwargs: object) -> InMemoryWorker:
        worker_signature.bind(**kwargs)
        connections.append(kwargs)
        return azure_worker if kwargs['token_credential'] is credential else durable_worker

    def client(**kwargs: object) -> object:
        client_signature.bind(**kwargs)
        connections.append(kwargs)
        return azure_worker.client if kwargs['token_credential'] is credential else durable_worker.client

    def model(*args: object, **kwargs: object) -> TestModel:
        return TestModel(custom_output_text='documented result')

    monkeypatch.setattr('durabletask.azuremanaged.worker.DurableTaskSchedulerWorker', worker)
    monkeypatch.setattr('durabletask.azuremanaged.client.DurableTaskSchedulerClient', client)
    monkeypatch.setattr('azure.identity.DefaultAzureCredential', lambda: credential)
    monkeypatch.setattr('pydantic_ai.models.infer_model', model)
    monkeypatch.setattr('pydantic_ai.durable_exec._base.infer_model', model)
    starts: list[bool] = []
    for example_worker in (durable_worker, azure_worker):
        monkeypatch.setattr(example_worker, 'start', lambda: starts.append(True))
        monkeypatch.setattr(example_worker, 'stop', lambda: None)
        example_worker.client_mock.__enter__.return_value = example_worker.client
    readme = (
        Path(__file__).parents[3] / 'src' / 'pydantic_ai_harness' / 'pydantic_ai_harness' / 'durabletask' / 'README.md'
    )
    examples = re.findall(r'```python[^\n]*\n(.*?)\n```', readme.read_text(encoding='utf-8'), flags=re.DOTALL)
    assert len(examples) == 5
    namespaces: list[dict[str, Any]] = []
    for index, source in enumerate(examples):
        namespace: dict[str, Any] = {'__name__': __name__}
        exec(compile(source, f'{readme}:example-{index}', 'exec', dont_inherit=True), namespace)
        namespaces.append(namespace)
    assert capsys.readouterr().out.splitlines() == ['documented result', 'documented result']
    assert starts == [True, True]
    assert (
        connections
        == [
            {
                'host_address': 'localhost:8080',
                'taskhub': 'default',
                'token_credential': None,
                'secure_channel': False,
            },
        ]
        * 2
        + [
            {
                'host_address': '<scheduler>.<region>.durabletask.io',
                'taskhub': '<hub>',
                'token_credential': credential,
            },
        ]
        * 2
    )
    assert worker_signature.parameters['secure_channel'].default is True
    assert client_signature.parameters['secure_channel'].default is True

    by_name = {
        name: namespace
        for namespace in namespaces
        for name in ('build_dependencies', 'resolve_reviewed_tools', 'register_existing')
        if name in namespace
    }
    assert set(by_name) == {'build_dependencies', 'resolve_reviewed_tools', 'register_existing'}

    async def check_dependencies() -> None:
        async with by_name['build_dependencies']['build_dependencies']('tenant-a') as deps:
            assert deps.tenant == 'tenant-a'
            assert not deps.client.is_closed
        assert deps.client.is_closed

    asyncio.run(check_dependencies())

    reviewed = by_name['resolve_reviewed_tools']
    for approved in (True, False):
        approval_worker = InMemoryWorker()
        with reviewed['durable'].register(approval_worker):
            run = reviewed['durable'].start(approval_worker.client, 'Review the order')
            approval_worker.run_until_idle()
            pending = run.pending_tools()
            assert len(pending.approvals) == len(pending.calls) == 1
            reviewed['resolve_reviewed_tools'](approval_worker.client, run.instance_id, approved=approved)
            assert run.result().output == 'documented result'
        assert [
            event.kwargs['data'] for event in approval_worker.client_mock.raise_orchestration_event.call_args_list
        ] == [approved, {'status': 'complete'}]

    existing = by_name['register_existing']
    existing_worker = InMemoryWorker()
    with existing['register_existing'](existing_worker):
        instance_id = existing_worker.client.schedule_new_orchestration(existing['orchestrator'], input='Hi')
        state = existing_worker.wait(instance_id)
    assert state is not None
    state.raise_if_failed()
    assert state.get_output() == 'documented result'
