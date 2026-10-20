from __future__ import annotations

import json

import pytest
from inline_snapshot import snapshot
from pydantic import ValidationError

from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from ...conftest import try_import

with try_import() as imports_successful:
    from durabletask.client import OrchestrationFailedError, OrchestrationStatus

    from pydantic_ai_harness.durabletask import DurableTaskAgent, DurableTaskDurability

    from .worker import InMemoryWorker

pytestmark = pytest.mark.skipif(not imports_successful(), reason='durabletask not installed')


def test_unknown_run_has_no_pending_tools(durable_worker: InMemoryWorker) -> None:
    agent = Agent(TestModel(), name='unknown', capabilities=[DurableTaskDurability()])
    handle = DurableTaskAgent(agent).get_run(durable_worker.client, 'missing')
    with pytest.raises(UserError, match='no pending tool requests'):
        handle.pending_tools()


def test_reconnect_and_terminal_status(durable_worker: InMemoryWorker) -> None:
    agent = Agent(TestModel(custom_output_text='done'), name='reconnect', capabilities=[DurableTaskDurability()])
    durable = DurableTaskAgent(agent)
    with durable.register(durable_worker):
        handle = durable.start(durable_worker.client, 'Hi')
        assert durable.get_run(durable_worker.client, handle.instance_id).result().output == 'done'
        state = durable_worker.state(handle.instance_id)
        assert state is not None
        state.runtime_status = OrchestrationStatus.TERMINATED
        durable_worker.client_mock.wait_for_orchestration_completion.side_effect = None
        durable_worker.client_mock.wait_for_orchestration_completion.return_value = state
        with pytest.raises(UserError, match='status TERMINATED'):
            handle.result()


def test_corrupt_persisted_result_is_rejected(durable_worker: InMemoryWorker) -> None:
    agent = Agent(TestModel(custom_output_text='done'), name='corrupt', capabilities=[DurableTaskDurability()])
    durable = DurableTaskAgent(agent)
    with durable.register(durable_worker):
        handle = durable.start(durable_worker.client, 'Hi')
        assert handle.result().output == 'done'
        durable_worker.instances[handle.instance_id].output = '{"output":42}'
        with pytest.raises(ValidationError, match='valid string'):
            handle.result()


def test_legacy_result_envelope_preserves_run_state(durable_worker: InMemoryWorker) -> None:
    agent = Agent(TestModel(custom_output_text='done'), name='legacy', capabilities=[DurableTaskDurability()])
    durable = DurableTaskAgent(agent)
    with durable.register(durable_worker):
        handle = durable.start(durable_worker.client, 'Hi')
        original = handle.result()
        encoded = durable_worker.instances[handle.instance_id].output
        assert encoded is not None
        data = json.loads(encoded)
        legacy = {
            'output': data['output'],
            '_state': {
                'message_history': data['messages'],
                'usage': data['usage'],
                'run_id': data['run_id'],
                'conversation_id': data['conversation_id'],
                'metadata': data['metadata'],
            },
            '_new_message_index': data['new_message_index'],
            '_output_tool_name': data['output_tool_name'],
            '_traceparent_value': data['traceparent'],
        }
        durable_worker.instances[handle.instance_id].output = json.dumps(legacy)
        restored = handle.result()
    assert restored.output == original.output
    assert restored.all_messages() == original.all_messages()
    assert restored.new_messages() == original.new_messages()
    assert restored.usage == original.usage
    assert restored.run_id == original.run_id
    assert restored.conversation_id == original.conversation_id
    assert restored.metadata == original.metadata


def test_invalid_tool_resolution_and_malformed_approval(durable_worker: InMemoryWorker) -> None:
    agent = Agent(
        TestModel(custom_output_text='done'),
        output_type=[str, DeferredToolRequests],
        name='approval-input',
        capabilities=[DurableTaskDurability()],
    )

    @agent.tool_plain(requires_approval=True)
    def submit() -> str:
        return 'submitted'

    durable = DurableTaskAgent(agent)
    with durable.register(durable_worker):
        handle = durable.start(durable_worker.client, 'Hi')
        durable_worker.run_until_idle()
        call = handle.pending_tools().approvals[0]
        encoded = durable_worker.instances[handle.instance_id].custom_status
        assert encoded is not None
        assert json.loads(encoded) == snapshot(
            {
                'requests': {
                    'calls': [],
                    'approvals': [
                        {
                            'tool_name': 'submit',
                            'args': {},
                            'tool_call_id': 'pyd_ai_tool_call_id__submit',
                            'tool_kind': None,
                            'id': None,
                            'provider_name': None,
                            'provider_details': None,
                            'part_kind': 'tool-call',
                        }
                    ],
                    'metadata': {},
                },
                'turn': 0,
            }
        )
        with pytest.raises(UserError, match='not awaiting a result'):
            handle.resolve(call.tool_call_id, 'wrong kind')
        durable_worker.raise_event(handle.instance_id, f'pydantic_ai.tool:0:{call.tool_call_id}', data='true')
        with pytest.raises(OrchestrationFailedError, match='valid boolean'):
            handle.result()


def test_approval_event_and_result_contract(durable_worker: InMemoryWorker) -> None:
    agent = Agent(
        TestModel(custom_output_text='done'),
        name='wire',
        output_type=[str, DeferredToolRequests],
        capabilities=[DurableTaskDurability()],
    )

    @agent.tool_plain(requires_approval=True)
    def submit() -> str:
        return 'submitted'

    durable = DurableTaskAgent(agent)
    with durable.register(durable_worker):
        run = durable.start(durable_worker.client, 'Hi', instance_id='wire')
        durable_worker.run_until_idle()
        run.approve(run.pending_tools().approvals[0].tool_call_id)
        event = durable_worker.client_mock.raise_orchestration_event.call_args
        assert {'args': list(event.args), 'data': event.kwargs['data']} == snapshot(
            {'args': ['wire', 'pydantic_ai.tool:0:pyd_ai_tool_call_id__submit'], 'data': True}
        )
        assert run.result().output == 'done'
    encoded = durable_worker.instances['wire'].output
    assert encoded is not None
    assert sorted(json.loads(encoded)) == snapshot(
        [
            'conversation_id',
            'messages',
            'metadata',
            'new_message_index',
            'output',
            'output_tool_name',
            'run_id',
            'traceparent',
            'usage',
        ]
    )


def test_repeated_tool_call_ids_need_a_new_approval(durable_worker: InMemoryWorker) -> None:
    calls: list[str] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if len(returns) < 2:
            return ModelResponse(parts=[ToolCallPart('submit', {}, tool_call_id='same-call')])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(
        FunctionModel(model),
        output_type=[str, DeferredToolRequests],
        name='repeat',
        capabilities=[DurableTaskDurability()],
    )

    @agent.tool_plain(requires_approval=True)
    def submit() -> str:
        calls.append('submitted')
        return 'submitted'

    durable = DurableTaskAgent(agent)
    with durable.register(durable_worker):
        handle = durable.start(durable_worker.client, 'Hi')
        durable_worker.run_until_idle()
        assert handle.pending_tools().approvals[0].tool_call_id == 'same-call'
        handle.approve('same-call')
        durable_worker.run_until_idle()
        assert calls == ['submitted']
        assert handle.pending_tools().approvals[0].tool_call_id == 'same-call'
        handle.approve('same-call')
        result = handle.result()
    assert result.output == 'done'
    assert calls == ['submitted', 'submitted']
    assert len([message for message in result.all_messages() if isinstance(message, ModelResponse)]) == 3
