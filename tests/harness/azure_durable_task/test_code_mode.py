"""`CodeMode` under Azure Durable Task: Monty runs in orchestrator code, nested tool calls are activities."""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from ...conftest import try_import

with try_import() as imports_successful:
    from durabletask.client import OrchestrationFailedError
    from pydantic_monty import NOT_HANDLED, AsyncMonty, OsFunction

    from pydantic_ai_harness import CodeMode
    from pydantic_ai_harness.azure_durable_task import AzureDurableTaskDurability

    from .worker import (
        InMemoryWorker,
        agent_orchestrator,
        completed,
        portal_threads,
        registered_agent,
        schedule_agent,
    )

pytestmark = pytest.mark.skipif(not imports_successful(), reason='durabletask or pydantic-monty not installed')

# Parallel nested calls, a durable timer, then a call that depends on both results. `MODE` comes from
# the host's `os_access`, which orchestrator code asks again on every replay.
_CODE = """\
import asyncio
import os
if os.getenv('MODE') == 'wait':
    await asyncio.sleep(1)
a, b = await asyncio.gather(add(a=1, b=2), add(a=3, b=4))
await asyncio.sleep(30)
await add(a=a, b=b)
"""


@dataclass
class _Pools:
    """Counts the Monty worker pools Code Mode opens and closes; each pool owns a worker process."""

    opened: int = 0
    closed: int = 0

    def __call__(self) -> _CountedPool:
        return _CountedPool(self, AsyncMonty())


@dataclass
class _CountedPool:
    pools: _Pools
    pool: AsyncMonty

    async def __aenter__(self) -> AsyncMonty:
        entered = await self.pool.__aenter__()
        self.pools.opened += 1
        return entered

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        try:
            await self.pool.__aexit__(exc_type, exc, tb)
        finally:
            self.pools.closed += 1


@pytest.fixture
def pools(monkeypatch: pytest.MonkeyPatch) -> _Pools:
    pools = _Pools()
    monkeypatch.setattr('pydantic_ai_harness._monty_exec.AsyncMonty', pools)
    return pools


def _make_agent(calls: list[tuple[int, int]], mode: str = 'go') -> Agent[None, str]:
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if not returns:
            return ModelResponse(parts=[ToolCallPart('run_code', {'code': _CODE}, tool_call_id='code')])
        return ModelResponse(parts=[TextPart(str(returns[-1].content))])

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        calls.append((a, b))
        return a + b

    def os_access(*, name: OsFunction, args: tuple[object, ...], kwargs: dict[str, object], **_: object) -> object:
        return mode if name == 'os.getenv' and args[0] == 'MODE' else NOT_HANDLED

    return Agent(
        FunctionModel(model),
        name='code_mode',
        tools=[add],
        capabilities=[CodeMode(os_access=os_access), AzureDurableTaskDurability()],
    )


def _park_on_sleep(worker: InMemoryWorker, calls: list[tuple[int, int]]) -> tuple[str, InMemoryWorker]:
    """Run until the sandbox's sleep, a durable timer, and return a new worker over the same history."""
    agent = _make_agent(calls)
    with registered_agent(worker, agent):
        instance_id = schedule_agent(worker, agent, deps=None)
        worker.run_until_idle()
    assert sorted(calls) == [(1, 2), (3, 4)]
    assert len(worker.timers) == 1
    replacement = InMemoryWorker()
    replacement.instances = worker.instances
    replacement.timers = worker.timers
    replacement.now = worker.now
    return instance_id, replacement


def test_run_code_replays_nested_calls_from_history(durable_worker: InMemoryWorker, pools: _Pools) -> None:
    """Monty's wakeup is bridged into the orchestrator loop, and a new worker resumes without repeating a call."""
    calls: list[tuple[int, int]] = []
    instance_id, replacement = _park_on_sleep(durable_worker, calls)
    # Each dispatch replays the snippet in a Monty worker of its own, then closes it and its portal thread.
    assert pools.opened > 1
    assert pools.closed == pools.opened
    assert portal_threads() == []

    agent = _make_agent(calls)
    replacement.add_orchestrator(agent_orchestrator(agent, deps=None))
    with registered_agent(replacement, agent):
        replacement.fire_timers()
        assert completed(replacement, instance_id) == '10'
    assert sorted(calls) == [(1, 2), (3, 4), (3, 7)]
    assert replacement.activity_calls == [
        'code_mode__function_toolset__-x3C-agent-x3E-.call_tool',
        'code_mode__model.request',
    ]
    assert pools.closed == pools.opened
    assert portal_threads() == []


def test_changed_snippet_fails_replay_with_nondeterminism_error(durable_worker: InMemoryWorker, pools: _Pools) -> None:
    """A snippet whose host answer changes on replay fails the orchestration instead of hanging or diverging."""
    calls: list[tuple[int, int]] = []
    instance_id, replacement = _park_on_sleep(durable_worker, calls)

    # The recorded history continues with the gathered activities; the replayed snippet now starts a timer.
    agent = _make_agent(calls, mode='wait')
    replacement.add_orchestrator(agent_orchestrator(agent, deps=None))
    with registered_agent(replacement, agent):
        replacement.fire_timers()
        with pytest.raises(OrchestrationFailedError, match='non-deterministic logic'):
            completed(replacement, instance_id)
    assert sorted(calls) == [(1, 2), (3, 4)]
    assert replacement.activity_calls == []
    assert pools.closed == pools.opened
    assert portal_threads() == []
