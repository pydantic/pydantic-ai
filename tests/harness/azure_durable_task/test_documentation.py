from __future__ import annotations

import inspect
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from types import FunctionType
from unittest.mock import patch

import pytest
from pydantic import InstanceOf, JsonValue, TypeAdapter, ValidationError

from pydantic_ai import Agent, UserError
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from ...conftest import try_import

with try_import() as imports_successful:
    from durabletask.azuremanaged.client import DurableTaskSchedulerClient
    from durabletask.azuremanaged.worker import DurableTaskSchedulerWorker
    from durabletask.client import OrchestrationFailedError, TaskHubGrpcClient
    from durabletask.worker import TaskHubGrpcWorker

    from pydantic_ai_harness.azure_durable_task import AzureDurableTaskDurability

    from .examples import README, load_review_example, python_examples
    from .worker import InMemoryWorker, completed

pytestmark = pytest.mark.skipif(not imports_successful(), reason='durabletask-azuremanaged not installed')


def test_documented_setup_examples(
    durable_worker: InMemoryWorker, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Execute the native and Azure setup and serializable-dependency examples through public boundaries."""
    connections: list[dict[str, object]] = []
    credential = object()
    starts: list[bool] = []
    azure_worker = InMemoryWorker()
    deps_worker = InMemoryWorker()

    def make_worker(**kwargs: object) -> InMemoryWorker:
        connections.append(kwargs)
        if 'token_credential' in kwargs:
            inspect.signature(DurableTaskSchedulerWorker).bind(**kwargs)
            return azure_worker
        inspect.signature(TaskHubGrpcWorker).bind(**kwargs)
        return durable_worker

    @contextmanager
    def make_client(**kwargs: object) -> Generator[TaskHubGrpcClient]:
        connections.append(kwargs)
        if 'token_credential' in kwargs:
            inspect.signature(DurableTaskSchedulerClient).bind(**kwargs)
            yield azure_worker.client
        else:
            inspect.signature(TaskHubGrpcClient).bind(**kwargs)
            yield durable_worker.client

    @contextmanager
    def make_credential() -> Generator[object]:
        yield credential

    def make_agent(
        model: str,
        *,
        name: str,
        capabilities: Sequence[AbstractCapability[object]],
        deps_type: type[object] = type(None),
    ) -> Agent[object, str]:
        assert model == 'openai:gpt-6.1-sol'
        return Agent[object, str](
            TestModel(custom_output_text='documented result'), name=name, deps_type=deps_type, capabilities=capabilities
        )

    monkeypatch.setattr('durabletask.worker.TaskHubGrpcWorker', make_worker)
    monkeypatch.setattr('durabletask.client.TaskHubGrpcClient', make_client)
    monkeypatch.setattr('durabletask.azuremanaged.worker.DurableTaskSchedulerWorker', make_worker)
    monkeypatch.setattr('durabletask.azuremanaged.client.DurableTaskSchedulerClient', make_client)
    monkeypatch.setattr('azure.identity.DefaultAzureCredential', make_credential)
    for example_worker in (durable_worker, azure_worker):
        monkeypatch.setattr(example_worker, 'start', lambda: starts.append(True))
        monkeypatch.setattr(example_worker, 'stop', lambda: None)
    examples = python_examples()
    assert len(examples) == 6
    setup_examples = [examples[0], examples[1], next(source for source in examples if 'class Dependencies:' in source)]
    namespaces: list[dict[str, object]] = []
    with patch('pydantic_ai.Agent', make_agent):
        for index, source in enumerate(setup_examples):
            namespace: dict[str, object] = {'__name__': __name__}
            exec(compile(source, f'{README}:setup-{index}', 'exec', dont_inherit=True), namespace)
            namespaces.append(namespace)
    assert capsys.readouterr().out.splitlines() == ['documented result', 'documented result']
    assert starts == [True, True]
    assert connections == [
        {'host_address': 'localhost:8080', 'metadata': [('taskhub', 'default')], 'secure_channel': False},
        {'host_address': 'localhost:8080', 'metadata': [('taskhub', 'default')], 'secure_channel': False},
        {'host_address': '<scheduler>.<region>.durabletask.io', 'taskhub': '<hub>', 'token_credential': credential},
        {'host_address': '<scheduler>.<region>.durabletask.io', 'taskhub': '<hub>', 'token_credential': credential},
    ]
    assert inspect.signature(DurableTaskSchedulerWorker).parameters['secure_channel'].default is True
    assert inspect.signature(DurableTaskSchedulerClient).parameters['secure_channel'].default is True

    deps_example = namespaces[2]
    agent = TypeAdapter[Agent[object, str]](InstanceOf[Agent[object, str]]).validate_python(deps_example['agent'])
    orchestrator = deps_example['support']
    assert isinstance(orchestrator, FunctionType)
    capability = AzureDurableTaskDurability.from_agent(agent)
    assert capability is not None
    with capability.register(deps_worker):
        deps_worker.add_orchestrator(orchestrator)
        instance_id = deps_worker.client.schedule_new_orchestration(
            orchestrator, input={'prompt': 'Hi', 'tenant': 'acme'}
        )
        assert completed(deps_worker, instance_id) == 'documented result'


def test_documented_retained_version_registration(
    durable_worker: InMemoryWorker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execute SDK worker setup and both version dispatches from the published example."""
    starts: list[bool] = []
    results: list[JsonValue] = []

    def make_worker(**kwargs: object) -> InMemoryWorker:
        inspect.signature(TaskHubGrpcWorker).bind(**kwargs)
        return durable_worker

    def make_agent(model: str, *, name: str, capabilities: Sequence[AbstractCapability[None]]) -> Agent[None, str]:
        assert model == 'openai:gpt-6.1-sol'
        return Agent(TestModel(custom_output_text=name), name=name, deps_type=type(None), capabilities=capabilities)

    def drain(prompt: str) -> str:
        assert prompt == 'Press Enter after stopping traffic and draining the worker.\n'
        for version in ['1.0', '2.0']:
            instance_id = durable_worker.schedule('support', input='Hi', version=version)
            results.append(completed(durable_worker, instance_id))
        return ''

    source = next(source for source in python_examples() if 'previous_versions=' in source)
    monkeypatch.setattr('durabletask.worker.TaskHubGrpcWorker', make_worker)
    monkeypatch.setattr(durable_worker, 'start', lambda: starts.append(True))
    monkeypatch.setattr(durable_worker, 'stop', lambda: starts.append(False))
    monkeypatch.setattr('builtins.input', drain)
    with patch('pydantic_ai.Agent', make_agent):
        exec(compile(source, f'{README}:versioning', 'exec', dont_inherit=True), {'__name__': __name__})
    assert starts == [True, False]
    assert results == ['support-1.0', 'support-2.0']


@pytest.mark.parametrize('decision', ['approve', 'deny', 'retry', 'case-sensitive'])
def test_documented_review_pattern(durable_worker: InMemoryWorker, decision: str) -> None:
    requests = 0

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal requests
        requests += 1
        if requests == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart('approve_payment', {'amount': 1}, tool_call_id='call_X'),
                    ToolCallPart(
                        'approve_payment' if decision == 'case-sensitive' else 'external_lookup',
                        {'amount': 2} if decision == 'case-sensitive' else {},
                        tool_call_id='call_x',
                    ),
                ]
            )
        if decision == 'retry' and requests == 2:
            assert any(
                isinstance(part, ToolReturnPart) and part.outcome == 'retried' and 'Try again' in str(part.content)
                for message in messages
                if isinstance(message, ModelRequest)
                for part in message.parts
            )
            return ModelResponse(parts=[ToolCallPart('external_lookup', {}, tool_call_id='call_x')])
        content = [
            str(part.content)
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        return ModelResponse(parts=[TextPart('|'.join(content))])

    example = load_review_example(FunctionModel(model), agent_name='documented_review')
    capability = AzureDurableTaskDurability.from_agent(example.agent)
    assert capability is not None
    with capability.register(durable_worker):
        durable_worker.add_orchestrator(example.orchestrator)
        instance_id = durable_worker.client.schedule_new_orchestration(example.orchestrator, input='Review')
        durable_worker.run_until_idle()
        state = durable_worker.client.get_orchestration_state(instance_id)
        assert state is not None
        pending = state.get_custom_status()
        assert isinstance(pending, dict)
        assert pending['pause'] == 0 and pending['event'] == 'review:0'
        decisions: dict[str, JsonValue] = {
            'call_X': {'action': 'approve', 'override_args': {'amount': 7}},
            'call_x': {'action': 'return', 'value': {'status': 'complete'}},
        }
        if decision == 'deny':
            decisions['call_X'] = {'action': 'deny', 'message': 'Payment blocked'}
        elif decision == 'case-sensitive':
            decisions['call_x'] = {'action': 'deny', 'message': 'Other payment blocked'}
        elif decision == 'retry':
            decisions['call_x'] = {'action': 'retry', 'message': 'Try again'}
        example.send_review(durable_worker.client, instance_id, pause=0, decisions=decisions)
        if decision == 'retry':
            durable_worker.run_until_idle()
            state = durable_worker.client.get_orchestration_state(instance_id)
            assert state is not None
            pending = state.get_custom_status()
            assert isinstance(pending, dict) and pending['pause'] == 1
            with pytest.raises(UserError, match='no longer pending'):
                example.send_review(durable_worker.client, instance_id, pause=0, decisions=decisions)
            durable_worker.raise_event(instance_id, 'review:0', data={'pause': 0, 'decisions': decisions})
            durable_worker.run_until_idle()
            state = durable_worker.client.get_orchestration_state(instance_id)
            assert state is not None
            assert state.get_custom_status() == pending
            example.send_review(
                durable_worker.client,
                instance_id,
                pause=1,
                decisions={'call_x': {'action': 'return', 'value': {'status': 'complete'}}},
            )
        output = completed(durable_worker, instance_id)
        assert isinstance(output, dict)
        assert output['requests'] == (3 if decision == 'retry' else 2)
        text = output['output']
        assert isinstance(text, str)
        assert ('Paid 7' in text) is (decision != 'deny')
        if decision == 'case-sensitive':
            assert 'Other payment blocked' in text and 'Paid 2' not in text
        else:
            assert 'complete' in text
        if decision == 'deny':
            assert 'Payment blocked' in text
        with pytest.raises(UserError, match='no longer pending'):
            example.send_review(durable_worker.client, instance_id, pause=0, decisions=decisions)
        with pytest.raises(UserError, match='no longer pending'):
            example.send_review(durable_worker.client, 'missing', pause=0, decisions=decisions)
        with pytest.raises(ValidationError, match='at most 100 characters'):
            example.send_review(durable_worker.client, instance_id, pause=10**100, decisions=decisions)


@pytest.mark.parametrize(
    ('tool_name', 'pause', 'decisions', 'message'),
    [
        ('approve_payment', 1, {'call': {'action': 'approve'}}, 'does not match the reviewed pause'),
        ('approve_payment', 0, {'other': {'action': 'approve'}}, 'tool-call IDs exactly'),
        ('approve_payment', 0, {'call': {'action': 'return', 'value': 'not approval'}}, 'approve or deny'),
        ('approve_payment', 0, {'call': {'message': 'Not an approval'}}, 'Field required'),
        ('external_lookup', 0, {'call': {'action': 'approve'}}, 'retry or return'),
    ],
)
def test_documented_review_rejects_invalid_decisions(
    durable_worker: InMemoryWorker, tool_name: str, pause: int, decisions: dict[str, JsonValue], message: str
) -> None:
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart(tool_name, {'amount': 1} if tool_name == 'approve_payment' else {}, tool_call_id='call')
            ]
        )

    example = load_review_example(FunctionModel(model), agent_name='invalid_review')
    capability = AzureDurableTaskDurability.from_agent(example.agent)
    assert capability is not None
    with capability.register(durable_worker):
        durable_worker.add_orchestrator(example.orchestrator)
        instance_id = durable_worker.client.schedule_new_orchestration(example.orchestrator, input='Review')
        durable_worker.run_until_idle()
        durable_worker.raise_event(instance_id, 'review:0', data={'pause': pause, 'decisions': decisions})
        with pytest.raises(OrchestrationFailedError, match=message):
            completed(durable_worker, instance_id)
