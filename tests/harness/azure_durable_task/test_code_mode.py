"""`CodeMode` under Azure Durable Task: Monty runs in orchestrator code, nested tool calls are activities."""

from __future__ import annotations

import threading

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from ...conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai_harness import CodeMode
    from pydantic_ai_harness.azure_durable_task import AzureDurableTaskDurability

    from .worker import InMemoryWorker, agent_orchestrator, completed, registered_agent, schedule_agent

pytestmark = pytest.mark.skipif(not imports_successful(), reason='durabletask or pydantic-monty not installed')

# Parallel nested calls, a durable timer, then a call that depends on both results.
_CODE = """\
import asyncio
a, b = await asyncio.gather(add(a=1, b=2), add(a=3, b=4))
await asyncio.sleep(30)
await add(a=a, b=b)
"""


def _model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
    if not returns:
        return ModelResponse(parts=[ToolCallPart('run_code', {'code': _CODE}, tool_call_id='code')])
    return ModelResponse(parts=[TextPart(str(returns[-1].content))])


def test_run_code_replays_nested_calls_from_history(durable_worker: InMemoryWorker) -> None:
    """Monty's wakeup is bridged into the orchestrator loop, and a new worker resumes without repeating a call."""
    calls: list[tuple[int, int]] = []

    def make_agent() -> Agent[None, str]:
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            calls.append((a, b))
            return a + b

        return Agent(
            FunctionModel(_model),
            name='code_mode',
            tools=[add],
            capabilities=[CodeMode(), AzureDurableTaskDurability()],
        )

    threads = threading.active_count()
    agent = make_agent()
    with registered_agent(durable_worker, agent):
        instance_id = schedule_agent(durable_worker, agent, deps=None)
        durable_worker.run_until_idle()
    # The orchestration is parked on the sandbox's sleep, a durable timer, after both gathered calls.
    assert sorted(calls) == [(1, 2), (3, 4)]
    assert len(durable_worker.timers) == 1
    # Each dispatch closes its Monty worker and portal thread.
    assert threading.active_count() == threads

    replacement = InMemoryWorker()
    replacement.instances = durable_worker.instances
    replacement.timers = durable_worker.timers
    replacement.now = durable_worker.now
    new_agent = make_agent()
    replacement.add_orchestrator(agent_orchestrator(new_agent, deps=None))
    with registered_agent(replacement, new_agent):
        replacement.fire_timers()
        assert completed(replacement, instance_id) == '10'
    assert sorted(calls) == [(1, 2), (3, 4), (3, 7)]
    assert replacement.activity_calls == [
        'code_mode__function_toolset__-x3C-agent-x3E-.call_tool',
        'code_mode__model.request',
    ]
    assert threading.active_count() == threads
