"""`CodeMode` under Azure Durable Task: each Monty turn is an activity, and configs that can't do that are refused."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, DynamicCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from ...conftest import try_import

with try_import() as imports_successful:
    from durabletask.client import OrchestrationFailedError
    from pydantic_monty import MountDir, OSAccess

    from pydantic_ai_harness import CodeMode
    from pydantic_ai_harness.azure_durable_task import AzureDurableTaskDurability
    from pydantic_ai_harness.dynamic_workflow import DynamicWorkflow

    from ..code_mode.stored_sessions import SERVER, stored_session_pool
    from .worker import InMemoryWorker, agent_orchestrator, completed, portal_threads, registered_agent, schedule_agent

pytestmark = pytest.mark.skipif(not imports_successful(), reason='durabletask or pydantic-monty not installed')

# Parallel nested calls, a durable timer, then a call that depends on both results.
_CODE = """\
import asyncio
a, b = await asyncio.gather(add(a=1, b=2), add(a=3, b=4))
await asyncio.sleep(30)
await add(a=a, b=b)
"""

_REFUSED = "`CodeMode` can't run snippets in this Azure Durable Task orchestration"
_HERE = str(Path(__file__).parent)


@pytest.fixture(autouse=True)
def server() -> Iterator[None]:
    SERVER.reset()
    yield
    SERVER.reset()


def _model(*snippets: str) -> FunctionModel:
    """Call `run_code` with each snippet in turn, then answer with every result, retries included."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        results = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart | RetryPromptPart) and part.tool_name == 'run_code'
        ]
        if len(results) < len(snippets):
            code = snippets[len(results)]
            return ModelResponse(parts=[ToolCallPart('run_code', {'code': code}, tool_call_id=f'code-{len(results)}')])
        return ModelResponse(parts=[TextPart('\n'.join(str(part.content) for part in results))])

    return FunctionModel(model)


def _make_agent(calls: list[tuple[int, int]], code_mode: AbstractCapability[None], *snippets: str) -> Agent[None, str]:
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        calls.append((a, b))
        return a + b

    return Agent(
        _model(*snippets or (_CODE,)),
        name='code_mode',
        tools=[add],
        capabilities=[code_mode, AzureDurableTaskDurability()],
    )


def _refusal(worker: InMemoryWorker, agent: Agent[None, str]) -> str:
    """Run `agent` to its failure and return the failure's message."""
    with registered_agent(worker, agent):
        instance_id = schedule_agent(worker, agent, deps=None)
        with pytest.raises(OrchestrationFailedError) as exc_info:
            completed(worker, instance_id)
    return str(exc_info.value)


def test_remote_turns_run_as_activities_and_resume_from_history(durable_worker: InMemoryWorker) -> None:
    """Each turn is an activity, the sandbox's sleep is a durable timer, and a new worker resumes without repeats."""
    calls: list[tuple[int, int]] = []
    agent = _make_agent(calls, CodeMode(monty_sandbox=stored_session_pool))
    with registered_agent(durable_worker, agent):
        instance_id = schedule_agent(durable_worker, agent, deps=None)
        durable_worker.run_until_idle()
    assert sorted(calls) == [(1, 2), (3, 4)]
    assert len(durable_worker.timers) == 1
    turn = 'code_mode__capability__code_mode.monty_turn'
    add = 'code_mode__function_toolset__-x3C-agent-x3E-.call_tool'
    # A turn pauses at each call and at the sleep; the host schedules the calls as activities between turns.
    assert durable_worker.activity_calls == ['code_mode__model.request', turn, turn, add, turn, add, turn, turn]

    replacement = InMemoryWorker()
    replacement.instances = durable_worker.instances
    replacement.timers = durable_worker.timers
    replacement.now = durable_worker.now
    agent = _make_agent(calls, CodeMode(monty_sandbox=stored_session_pool))
    replacement.add_orchestrator(agent_orchestrator(agent, deps=None))
    with registered_agent(replacement, agent):
        replacement.fire_timers()
        assert completed(replacement, instance_id) == '10'
    assert sorted(calls) == [(1, 2), (3, 4), (3, 7)]
    # The recorded turns and calls replay from history: only the work after the timer runs.
    assert replacement.activity_calls == [turn, turn, add, turn, 'code_mode__model.request']
    # No Monty call ran in orchestrator code, so no portal was opened there.
    assert portal_threads() == []


def test_turn_activity_stops_a_snippet_that_computes_forever(durable_worker: InMemoryWorker) -> None:
    """`max_duration_seconds` stops a CPU-bound turn inside its activity, and the model sees the reset session."""
    agent = _make_agent(
        [],
        CodeMode(monty_sandbox=stored_session_pool, resource_limits={'max_duration_seconds': 0.1}),
        'x = 41',
        'while True:\n    pass',
        'x + 1',
    )
    with registered_agent(durable_worker, agent):
        output = completed(durable_worker, schedule_agent(durable_worker, agent, deps=None))
    assert isinstance(output, str)
    assert 'ran longer than its time limit and was stopped, so the session was reset' in output
    # The reset session no longer defines `x`.
    assert 'Name `x` used when not defined' in output


@pytest.mark.parametrize(
    'code_mode,change',
    [
        pytest.param(CodeMode(), 'set `monty_sandbox` to a Monty server that stores sessions', id='local'),
        pytest.param(
            CodeMode(monty_sandbox=stored_session_pool, os_access=OSAccess()), 'remove `os_access`', id='os_access'
        ),
        pytest.param(
            CodeMode(monty_sandbox=stored_session_pool, mount=MountDir(virtual_path='/data', host_path=_HERE)),
            "use only mounts without `mode='overlay'` or `write_bytes_limit`",
            id='overlay',
        ),
        pytest.param(
            CodeMode(
                monty_sandbox=stored_session_pool,
                mount=[MountDir(virtual_path='/data', host_path=_HERE, mode='read-write', write_bytes_limit=6)],
            ),
            "use only mounts without `mode='overlay'` or `write_bytes_limit`",
            id='write_bytes_limit',
        ),
        pytest.param(
            CodeMode(id=None, monty_sandbox=stored_session_pool),
            'give `CodeMode` an `id` and add it to the agent directly, not from a `DynamicCapability`',
            id='no_id',
        ),
        pytest.param(
            DynamicCapability(lambda ctx: CodeMode(monty_sandbox=stored_session_pool), id='dynamic_code'),
            'give `CodeMode` an `id` and add it to the agent directly, not from a `DynamicCapability`',
            id='dynamic',
        ),
    ],
)
def test_configs_whose_turns_cannot_be_activities_are_refused(
    durable_worker: InMemoryWorker, code_mode: AbstractCapability[None], change: str
) -> None:
    """A snippet that would run in orchestrator code fails the run before it starts, saying what to change."""
    calls: list[tuple[int, int]] = []
    message = _refusal(durable_worker, _make_agent(calls, code_mode))
    assert _REFUSED in message
    assert change in message
    assert calls == []
    assert not [name for name in durable_worker.activity_calls if name.endswith(('.monty_turn', '.call_tool'))]


def test_refusal_names_every_change(durable_worker: InMemoryWorker) -> None:
    """A config with several problems names each change it needs."""
    message = _refusal(durable_worker, _make_agent([], CodeMode(id=None, os_access=OSAccess())))
    assert (
        f'{_REFUSED}: Azure Durable Task replays orchestrator code, so a snippet there has no time limit, and '
        'nothing stops one that computes forever. To run each turn in its own activity, where '
        '`max_duration_seconds` applies, set `monty_sandbox` to a Monty server that stores sessions; remove '
        '`os_access`; give `CodeMode` an `id` and add it to the agent directly, not from a `DynamicCapability`.'
    ) in message


def test_server_without_a_session_store_is_refused(durable_worker: InMemoryWorker) -> None:
    """A server that stores no sessions can't run turns as activities, so the snippet isn't re-run in orchestrator code."""
    calls: list[tuple[int, int]] = []
    message = _refusal(durable_worker, _make_agent(calls, CodeMode(monty_sandbox=lambda: stored_session_pool(False))))
    assert (
        "The `monty_sandbox` server doesn't store sessions, which `CodeMode` needs to run each turn in its own "
        'activity in this Azure Durable Task orchestration. Use a Monty server with a session store.'
    ) in message
    assert calls == []
    assert durable_worker.activity_calls == ['code_mode__model.request', 'code_mode__capability__code_mode.monty_turn']


def test_agent_without_durability_in_an_orchestration_is_refused(durable_worker: InMemoryWorker) -> None:
    """An agent an inline tool runs has no durability capability of its own, but its snippets would still replay."""
    inner = Agent(_model('1 + 1'), name='inner', capabilities=[CodeMode(monty_sandbox=stored_session_pool)])

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('delegate', {})])

    outer = Agent(FunctionModel(model), name='outer', capabilities=[AzureDurableTaskDurability()])

    @outer.tool_plain(metadata={'azure_durable_task': False})
    async def delegate() -> str:
        """Ask the inner agent."""
        return (await inner.run('Hi')).output

    assert _REFUSED in _refusal(durable_worker, outer)


def test_dynamic_workflow_is_refused(durable_worker: InMemoryWorker) -> None:
    """`DynamicWorkflow` scripts run in orchestrator code, with nothing to stop one that computes forever."""
    summarize = Agent(_model(), name='summarize', description='Summarize a text.')

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('run_workflow', {'code': "await summarize(task='alpha')"})])

    agent = Agent(
        FunctionModel(model),
        name='dynamic_workflow',
        capabilities=[DynamicWorkflow(agents=[summarize]), AzureDurableTaskDurability()],
    )
    message = _refusal(durable_worker, agent)
    assert "`DynamicWorkflow` can't run scripts in this Azure Durable Task orchestration" in message
    assert 'https://github.com/pydantic/pydantic-ai/issues/14974' in message
