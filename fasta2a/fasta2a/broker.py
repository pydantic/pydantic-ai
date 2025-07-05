from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Annotated, Any, Generic, Literal, TypeVar

import anyio
from anyio.streams.memory import MemoryObjectSendStream
from opentelemetry.trace import Span, get_current_span, get_tracer
from pydantic import Discriminator
from typing_extensions import Self, TypedDict

from .schema import StreamEvent, TaskIdParams, TaskSendParams

tracer = get_tracer(__name__)


@dataclass
class Broker(ABC):
    """The broker class is in charge of scheduling the tasks.

    The HTTP server uses the broker to schedule tasks.

    The simple implementation is the `InMemoryBroker`, which is the broker that
    runs the tasks in the same process as the HTTP server. That said, this class can be
    extended to support remote workers.
    """

    @abstractmethod
    async def run_task(self, params: TaskSendParams) -> None:
        """Send a task to be executed by the worker."""
        raise NotImplementedError('send_run_task is not implemented yet.')

    @abstractmethod
    async def cancel_task(self, params: TaskIdParams) -> None:
        """Cancel a task."""
        raise NotImplementedError('send_cancel_task is not implemented yet.')

    @abstractmethod
    async def __aenter__(self) -> Self: ...

    @abstractmethod
    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any): ...

    @abstractmethod
    def receive_task_operations(self) -> AsyncIterator[TaskOperation]:
        """Receive task operations from the broker.

        On a multi-worker setup, the broker will need to round-robin the task operations
        between the workers.
        """

    @abstractmethod
    async def send_stream_event(self, task_id: str, event: StreamEvent) -> None:
        """Send a streaming event from worker to subscribers.

        This is used by workers to publish status updates, messages, and artifacts
        during task execution. Events are forwarded to all active subscribers of
        the given task_id.
        """
        raise NotImplementedError('send_stream_event is not implemented yet.')

    @abstractmethod
    def subscribe_to_stream(self, task_id: str) -> AsyncIterator[StreamEvent]:
        """Subscribe to streaming events for a specific task.

        Returns an async iterator that yields events published by workers for the
        given task_id. The iterator completes when a TaskStatusUpdateEvent with
        final=True is received or the subscription is cancelled.
        """
        raise NotImplementedError('subscribe_to_stream is not implemented yet.')


OperationT = TypeVar('OperationT')
ParamsT = TypeVar('ParamsT')


class _TaskOperation(TypedDict, Generic[OperationT, ParamsT]):
    """A task operation."""

    operation: OperationT
    params: ParamsT
    _current_span: Span


_RunTask = _TaskOperation[Literal['run'], TaskSendParams]
_CancelTask = _TaskOperation[Literal['cancel'], TaskIdParams]

TaskOperation = Annotated['_RunTask | _CancelTask', Discriminator('operation')]


class InMemoryBroker(Broker):
    """A broker that schedules tasks in memory."""

    def __init__(self):
        # Event streams per task_id for pub/sub
        self._event_subscribers: dict[str, list[MemoryObjectSendStream[StreamEvent]]] = {}
        # Lock for thread-safe subscriber management
        self._subscriber_lock = anyio.Lock()

    async def __aenter__(self):
        self.aexit_stack = AsyncExitStack()
        await self.aexit_stack.__aenter__()

        self._write_stream, self._read_stream = anyio.create_memory_object_stream[TaskOperation]()
        await self.aexit_stack.enter_async_context(self._read_stream)
        await self.aexit_stack.enter_async_context(self._write_stream)

        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any):
        await self.aexit_stack.__aexit__(exc_type, exc_value, traceback)

    async def run_task(self, params: TaskSendParams) -> None:
        await self._write_stream.send(_RunTask(operation='run', params=params, _current_span=get_current_span()))

    async def cancel_task(self, params: TaskIdParams) -> None:
        await self._write_stream.send(_CancelTask(operation='cancel', params=params, _current_span=get_current_span()))

    async def receive_task_operations(self) -> AsyncIterator[TaskOperation]:
        """Receive task operations from the broker."""
        async for task_operation in self._read_stream:
            yield task_operation

    async def send_stream_event(self, task_id: str, event: StreamEvent) -> None:
        """Send a streaming event to all subscribers of a task."""
        async with self._subscriber_lock:
            subscribers = self._event_subscribers.get(task_id, [])
            # Send to all active subscribers, removing any that are closed
            active_subscribers: list[MemoryObjectSendStream[StreamEvent]] = []
            for send_stream in subscribers:
                try:
                    await send_stream.send(event)
                    active_subscribers.append(send_stream)
                except anyio.ClosedResourceError:
                    # Subscriber disconnected, remove it
                    pass

            # Update subscriber list with only active ones
            if active_subscribers:
                self._event_subscribers[task_id] = active_subscribers
            elif task_id in self._event_subscribers:
                # No active subscribers, clean up
                del self._event_subscribers[task_id]

    async def subscribe_to_stream(self, task_id: str) -> AsyncIterator[StreamEvent]:
        """Subscribe to events for a specific task."""
        # Create a new stream for this subscriber
        send_stream, receive_stream = anyio.create_memory_object_stream[StreamEvent](max_buffer_size=100)

        # Register the subscriber
        async with self._subscriber_lock:
            if task_id not in self._event_subscribers:
                self._event_subscribers[task_id] = []
            self._event_subscribers[task_id].append(send_stream)

        try:
            # Yield events as they arrive
            async with receive_stream:
                async for event in receive_stream:
                    yield event
                    # Check if this is a final event
                    if isinstance(event, dict) and event.get('kind') == 'status-update' and event.get('final', False):
                        break
        finally:
            # Clean up subscription on exit
            async with self._subscriber_lock:
                if task_id in self._event_subscribers:
                    try:
                        self._event_subscribers[task_id].remove(send_stream)
                        if not self._event_subscribers[task_id]:
                            del self._event_subscribers[task_id]
                    except ValueError:
                        # Already removed
                        pass
            await send_stream.aclose()
