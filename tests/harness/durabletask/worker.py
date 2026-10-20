from __future__ import annotations

# The SDK exposes no public in-process executor. Keep this coupling confined to the test fixture.
# pyright: reportPrivateUsage=false
import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
from durabletask import client, task, worker
from durabletask.internal import helpers as ph, orchestrator_service_pb2 as pb
from durabletask.serialization import DEFAULT_DATA_CONVERTER


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


class _Executor(worker._OrchestrationExecutor):
    context: worker._RuntimeOrchestrationContext | None = None

    def process_event(self, ctx: worker._RuntimeOrchestrationContext, event: pb.HistoryEvent) -> None:
        self.context = ctx
        super().process_event(ctx, event)

    def close(self) -> None:
        if self.context is not None and self.context._generator is not None:
            self.context._generator.close()


class InMemoryWorker(worker.TaskHubGrpcWorker):
    """Replay real SDK histories in process, without gRPC, service threads or wall-clock waits."""

    def __init__(self) -> None:
        self._registry = worker._Registry()
        self._logger = logging.getLogger(__name__)
        self._data_converter = DEFAULT_DATA_CONVERTER
        self._is_running = False
        self.now = datetime(2026, 1, 1)
        self.instances: dict[str, _Instance] = {}
        self.activity_queue: deque[tuple[str, str, int, str | None]] = deque()
        self.activity_calls: list[str] = []
        self.timers: list[tuple[str, int, pb.CreateTimerAction]] = []
        mock_client = MagicMock(spec=client.TaskHubGrpcClient)
        mock_client.schedule_new_orchestration.side_effect = self.schedule
        mock_client.get_orchestration_state.side_effect = self.state
        mock_client.wait_for_orchestration_completion.side_effect = self.wait
        mock_client.raise_orchestration_event.side_effect = self.raise_event
        self.client_mock = mock_client
        self.client: client.TaskHubGrpcClient = mock_client

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
        pytest.fail('Durable Task fixture did not become idle after 200 turns')

    def orchestration_turn(self, instance_id: str) -> None:
        record = self.instances[instance_id]
        events = [self.event(orchestratorStarted=pb.OrchestratorStartedEvent()), *record.new_events]
        record.new_events.clear()
        executor = _Executor(self._registry, self._logger, data_converter=self._data_converter)
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
            record.output = completed.result.value
            if completed.orchestrationStatus == pb.ORCHESTRATION_STATUS_COMPLETED:
                record.status = client.OrchestrationStatus.COMPLETED
            else:
                record.status = client.OrchestrationStatus.FAILED
                details = completed.failureDetails
                record.failure = task.FailureDetails(details.errorMessage, details.errorType, None)
        else:
            pytest.fail(f'Unsupported SDK action in fixture: {action}')

    def activity_turn(self, instance_id: str, name: str, task_id: int, encoded: str | None) -> None:
        self.activity_calls.append(name)
        executor = worker._ActivityExecutor(self._registry, self._logger, self._data_converter)
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

    def state(self, instance_id: str, **kwargs: object) -> client.OrchestrationState | None:
        if (record := self.instances.get(instance_id)) is None:
            return None
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

    def wait(self, instance_id: str, **kwargs: object) -> client.OrchestrationState | None:
        self.run_until_idle()
        state = self.state(instance_id)
        if state is not None and state.runtime_status not in (
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
