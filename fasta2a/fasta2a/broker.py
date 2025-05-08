from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Annotated, Any, Generic, Literal, TypeVar

import anyio
from opentelemetry.trace import Span, get_current_span, get_tracer
from pydantic import Discriminator
from typing_extensions import Self, TypedDict

from .schema import TaskIdParams, TaskSendParams

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
