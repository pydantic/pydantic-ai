from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, TypeVar

import pytest
from durabletask import client, task, worker
from durabletask.serialization import DEFAULT_DATA_CONVERTER
from pydantic import JsonValue, TypeAdapter
from pydantic_core import to_jsonable_python

from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.run import AgentRunResult
from pydantic_ai_harness.azure_durable_task import AzureDurableTaskDurability, run_durable

from .sdk import ActivityExecutor, OrchestrationExecutor, Registry, pb, ph

DepsT = TypeVar('DepsT')
OutputT = TypeVar('OutputT')
_JSON = TypeAdapter[JsonValue](JsonValue)


@dataclass
class _Instance:
    name: str
    input: str | None
    history: list[pb.HistoryEvent] = field(default_factory=list[pb.HistoryEvent])
    new_events: list[pb.HistoryEvent] = field(default_factory=list[pb.HistoryEvent])
    status: client.OrchestrationStatus = client.OrchestrationStatus.RUNNING
    output: str | None = None
    custom_status: str | None = None
    failure: task.FailureDetails | None = None


class InMemoryWorker(worker.TaskHubGrpcWorker):
    """Replay real SDK histories in process, without gRPC, service threads or wall-clock waits."""

    def __init__(self) -> None:
        self._registry = Registry()
        self._logger = logging.getLogger(__name__)
        self._data_converter = DEFAULT_DATA_CONVERTER
        self._is_running = False
        self.now = datetime(2026, 1, 1)
        self.instances: dict[str, _Instance] = {}
        self.activity_queue: deque[tuple[str, str, int, str | None]] = deque()
        self.activity_calls: list[str] = []
        self.timers: list[tuple[str, int, pb.CreateTimerAction]] = []

    def schedule(
        self, orchestrator: Callable[..., object] | str, *, input: object = None, instance_id: str | None = None
    ) -> str:
        name = orchestrator if isinstance(orchestrator, str) else orchestrator.__name__
        instance_id = instance_id or f'run-{len(self.instances)}'
        encoded = self._data_converter.serialize(input)
        record = _Instance(name=name, input=encoded)
        record.new_events.append(ph.new_execution_started_event(name, instance_id, encoded))
        self.instances[instance_id] = record
        return instance_id

    def event(self, **kwargs: Any) -> pb.HistoryEvent:
        event = pb.HistoryEvent(eventId=-1, **kwargs)
        event.timestamp.FromDatetime(self.now)
        return event

    def step(self) -> bool:
        for instance_id, record in self.instances.items():
            if record.new_events:
                self.orchestration_turn(instance_id)
                return True
        if self.activity_queue:
            self.activity_turn(*self.activity_queue.popleft())
            return True
        return False

    def run_until_idle(self) -> None:
        for _ in range(200):
            if not self.step():
                return
        pytest.fail('Durable Task fixture did not become idle after 200 turns')  # pragma: no cover

    def orchestration_turn(self, instance_id: str) -> None:
        record = self.instances[instance_id]
        events = [self.event(orchestratorStarted=pb.OrchestratorStartedEvent()), *record.new_events]
        record.new_events.clear()
        executor = OrchestrationExecutor(self._registry, self._logger, data_converter=self._data_converter)
        try:
            result = executor.execute(instance_id, record.history, events)
        finally:
            executor.close()
        record.history.extend(events)
        record.custom_status = result.encoded_custom_status
        for action in result.actions:
            self.apply_action(instance_id, action)

    def apply_action(self, instance_id: str, action: pb.OrchestratorAction) -> None:
        record = self.instances[instance_id]
        if action.HasField('scheduleTask'):
            scheduled = action.scheduleTask
            event = self.event(taskScheduled=pb.TaskScheduledEvent(name=scheduled.name, input=scheduled.input))
            event.eventId = action.id
            record.history.append(event)
            self.activity_queue.append((instance_id, scheduled.name, action.id, scheduled.input.value))
        elif action.HasField('createTimer'):
            event = self.event(timerCreated=pb.TimerCreatedEvent(fireAt=action.createTimer.fireAt))
            event.eventId = action.id
            record.history.append(event)
            self.timers.append((instance_id, action.id, action.createTimer))
        elif action.HasField('completeOrchestration'):
            completed = action.completeOrchestration
            if completed.orchestrationStatus == pb.ORCHESTRATION_STATUS_CONTINUED_AS_NEW:
                self.schedule(
                    record.name, input=self._data_converter.deserialize(completed.result.value), instance_id=instance_id
                )
                self.instances[instance_id].new_events.extend(completed.carryoverEvents)
                return
            record.output = completed.result.value
            if completed.orchestrationStatus == pb.ORCHESTRATION_STATUS_COMPLETED:
                record.status = client.OrchestrationStatus.COMPLETED
            else:
                record.status = client.OrchestrationStatus.FAILED
                details = completed.failureDetails
                record.failure = task.FailureDetails(details.errorMessage, details.errorType, None)
        else:  # pragma: no cover
            pytest.fail(f'Unsupported SDK action in fixture: {action}')

    def activity_turn(self, instance_id: str, name: str, task_id: int, encoded: str | None) -> None:
        self.activity_calls.append(name)
        executor = ActivityExecutor(self._registry, self._logger, self._data_converter)
        try:
            result = executor.execute(instance_id, name, task_id, encoded)
        except Exception as exc:
            event = self.event(
                taskFailed=pb.TaskFailedEvent(taskScheduledId=task_id, failureDetails=ph.new_failure_details(exc))
            )
        else:
            event = self.event(
                taskCompleted=pb.TaskCompletedEvent(taskScheduledId=task_id, result=ph.get_string_value(result))
            )
        self.instances[instance_id].new_events.append(event)

    def fire_timers(self) -> None:
        deadline = min(timer.fireAt.ToDatetime() for _, _, timer in self.timers)
        timers = [item for item in self.timers if item[2].fireAt.ToDatetime() == deadline]
        self.timers = [item for item in self.timers if item[2].fireAt.ToDatetime() != deadline]
        for instance_id, timer_id, timer in timers:
            self.now = max(self.now, timer.fireAt.ToDatetime())
            self.instances[instance_id].new_events.append(
                self.event(timerFired=pb.TimerFiredEvent(timerId=timer_id, fireAt=timer.fireAt))
            )

    def state(self, instance_id: str) -> client.OrchestrationState:
        record = self.instances[instance_id]
        return client.OrchestrationState(
            instance_id=instance_id,
            name=record.name,
            runtime_status=record.status,
            created_at=self.now.replace(tzinfo=UTC),
            last_updated_at=self.now.replace(tzinfo=UTC),
            serialized_input=record.input,
            serialized_output=record.output,
            serialized_custom_status=record.custom_status,
            failure_details=record.failure,
        )

    def wait(self, instance_id: str) -> client.OrchestrationState:
        self.run_until_idle()
        state = self.state(instance_id)
        if state.runtime_status not in (
            client.OrchestrationStatus.COMPLETED,
            client.OrchestrationStatus.FAILED,
            client.OrchestrationStatus.TERMINATED,
        ):
            raise TimeoutError(f'Orchestration {instance_id!r} has not completed.')
        return state

    def raise_event(self, instance_id: str, event_name: str, *, data: object = None) -> None:
        self.instances[instance_id].new_events.append(
            ph.new_event_raised_event(event_name, self._data_converter.serialize(data))
        )


@contextmanager
def registered_agent(worker: InMemoryWorker, agent: AbstractAgent[DepsT, OutputT]) -> Generator[None]:
    capability = AzureDurableTaskDurability.from_agent(agent)
    assert capability is not None
    with capability.register(worker):
        yield


def agent_orchestrator(
    agent: AbstractAgent[DepsT, OutputT],
    prompt: str = 'Hi',
    *,
    deps: DepsT,
    timeout: float | None = None,
    results: list[AgentRunResult[OutputT]] | None = None,
) -> Callable[[task.OrchestrationContext, object], Generator[task.Task[object], object, JsonValue]]:
    def orchestrator(
        context: task.OrchestrationContext, payload: object
    ) -> Generator[task.Task[object], object, JsonValue]:
        result = yield from run_durable(context, lambda: agent.run(prompt, deps=deps, timeout=timeout))
        if results is not None:
            results.append(result)
        return _JSON.validate_python(to_jsonable_python(result.output, round_trip=True))

    assert agent.name is not None
    orchestrator.__name__ = f'{agent.name}_workflow'
    return orchestrator


def schedule_agent(
    worker: InMemoryWorker,
    agent: AbstractAgent[DepsT, OutputT],
    prompt: str = 'Hi',
    *,
    deps: DepsT,
    instance_id: str | None = None,
    timeout: float | None = None,
    results: list[AgentRunResult[OutputT]] | None = None,
) -> str:
    orchestrator = agent_orchestrator(agent, prompt, deps=deps, timeout=timeout, results=results)
    worker.add_orchestrator(orchestrator)
    return worker.schedule(orchestrator, instance_id=instance_id)


def completed(worker: InMemoryWorker, instance_id: str) -> JsonValue:
    state = worker.wait(instance_id)
    state.raise_if_failed()
    return _JSON.validate_python(state.get_output())


def run_agent(
    worker: InMemoryWorker,
    agent: AbstractAgent[DepsT, OutputT],
    prompt: str = 'Hi',
    *,
    deps: DepsT,
    instance_id: str | None = None,
    timeout: float | None = None,
) -> AgentRunResult[OutputT]:
    """Observe the actual agent result in the orchestrator, without reconstructing it on a client."""
    results: list[AgentRunResult[OutputT]] = []
    instance_id = schedule_agent(
        worker, agent, prompt, deps=deps, instance_id=instance_id, timeout=timeout, results=results
    )
    completed(worker, instance_id)
    assert len(results) == 1
    return results[0]
