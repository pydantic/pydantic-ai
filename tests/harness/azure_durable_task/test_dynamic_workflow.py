"""`DynamicWorkflow` under Azure Durable Task: the script runs in orchestrator code, sub-agents make activities."""

from __future__ import annotations

import pytest

from pydantic_ai import Agent, Tool

from ...conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai_harness.azure_durable_task import AzureDurableTaskDurability
    from pydantic_ai_harness.dynamic_workflow import DynamicWorkflow, WorkflowResourceLimits

    from ..dynamic_workflow._concurrent_sub_agents import orchestrator_model, summarize_model
    from .worker import InMemoryWorker, agent_orchestrator, completed, portal_threads, registered_agent, schedule_agent

pytestmark = pytest.mark.skipif(not imports_successful(), reason='durabletask or pydantic-monty not installed')

# Two gathered sub-agents, a durable timer, then a sub-agent that depends on both results.
_SCRIPT = """\
import asyncio
a, b = await asyncio.gather(summarize(task='alpha'), summarize(task='beta'))
await asyncio.sleep(30)
c = await summarize(task=a + ' + ' + b)
[a, b, c]
"""


def _agents(
    model_calls: list[str],
    tool_calls: list[str],
    *,
    script: str = _SCRIPT,
    resource_limits: WorkflowResourceLimits | None = None,
) -> tuple[Agent[None, str], Agent[None, str]]:
    def noop() -> str:
        tool_calls.append('noop')
        return 'ok'

    summarize = Agent(
        summarize_model(model_calls, alpha_waits_for_beta=False),
        name='summarize',
        description='Summarize a text.',
        tools=[Tool(noop, name='noop')],
        capabilities=[AzureDurableTaskDurability()],
    )
    orchestrator = Agent(
        orchestrator_model(script),
        name='orchestrator',
        capabilities=[
            DynamicWorkflow(agents=[summarize], resource_limits=resource_limits),
            AzureDurableTaskDurability(),
        ],
    )
    return orchestrator, summarize


def test_workflow_script_resumes_on_a_new_worker_without_repeating_work(durable_worker: InMemoryWorker) -> None:
    """Monty is called through a portal in orchestrator code, and a new worker resumes from history.

    Every orchestration turn replays the whole history, and the SDK fails the orchestration with a
    `NonDeterminismError` if a replay schedules a different activity or timer at any sequence number.
    """
    model_calls: list[str] = []
    tool_calls: list[str] = []

    orchestrator, summarize = _agents(model_calls, tool_calls)
    with registered_agent(durable_worker, orchestrator), registered_agent(durable_worker, summarize):
        instance_id = schedule_agent(durable_worker, orchestrator, deps=None)
        durable_worker.run_until_idle()
    # Parked on the script's sleep, a durable timer, after both gathered sub-agents finished.
    assert sorted(model_calls) == ['alpha 1', 'alpha 3', 'beta 1', 'beta 3']
    assert tool_calls == ['noop', 'noop']
    assert len(durable_worker.timers) == 1
    # The gathered sub-agents schedule their activities in the same turns, so they run concurrently.
    model_request, call_tool = 'summarize__model.request', 'summarize__function_toolset__-x3C-agent-x3E-.call_tool'
    assert durable_worker.activity_calls == [
        'orchestrator__model.request',
        *[model_request] * 2,
        *[call_tool] * 2,
        *[model_request] * 2,
    ]
    # Each dispatch closes its Monty worker and portal thread.
    assert portal_threads() == []

    replacement = InMemoryWorker()
    replacement.instances = durable_worker.instances
    replacement.timers = durable_worker.timers
    replacement.now = durable_worker.now
    model_calls.clear()
    tool_calls.clear()
    new_orchestrator, new_summarize = _agents(model_calls, tool_calls)
    replacement.add_orchestrator(agent_orchestrator(new_orchestrator, deps=None))
    with registered_agent(replacement, new_orchestrator), registered_agent(replacement, new_summarize):
        replacement.fire_timers()
        output = completed(replacement, instance_id)
        # Re-executing the finished history on a fresh executor schedules no new activity or timer.
        replayed = replacement.replay(instance_id)
    assert [action.WhichOneof('orchestratorActionType') for action in replayed] == ['completeOrchestration']
    assert replayed[0].completeOrchestration.result.value == replacement.instances[instance_id].output
    assert output == "done: ['summary of alpha', 'summary of beta', 'summary of summary of alpha + summary of beta']"
    # Only the sub-agent after the timer runs: no gathered sub-agent, model request or tool call repeats,
    # and the replay of the finished history reran none either.
    assert model_calls == ['summary of alpha + summary of beta 1', 'summary of alpha + summary of beta 3']
    assert tool_calls == ['noop']
    assert replacement.activity_calls == [
        'summarize__model.request',
        'summarize__function_toolset__-x3C-agent-x3E-.call_tool',
        'summarize__model.request',
        'orchestrator__model.request',
    ]
    assert portal_threads() == []


# Pure computation: far longer than the limit below, with no sub-agent call to wait on.
_BUSY_SCRIPT = """\
total = 0
for i in range(200000):
    total += i
total
"""


def test_elapsed_time_limit_is_off_in_orchestrator_code(durable_worker: InMemoryWorker) -> None:
    """A run and its replay must take the same branch, so the script's time limit is off in orchestrator code."""
    limits: WorkflowResourceLimits = {'max_duration_seconds': 1e-6}
    orchestrator, summarize = _agents([], [], script=_BUSY_SCRIPT, resource_limits=limits)

    # Outside an orchestration, the limit stops the script.
    outside = Agent[None, str](
        orchestrator_model(_BUSY_SCRIPT),
        deps_type=type(None),
        capabilities=[DynamicWorkflow[None](agents=[summarize], resource_limits=limits)],
    )
    assert 'time limit exceeded' in outside.run_sync('Hi').output.lower()

    with registered_agent(durable_worker, orchestrator), registered_agent(durable_worker, summarize):
        instance_id = schedule_agent(durable_worker, orchestrator, deps=None)
        assert completed(durable_worker, instance_id) == 'done: 19999900000'
