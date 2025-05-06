from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Annotated, Any, Generic, Literal, TypeVar

import anyio
from opentelemetry.trace import Span, get_current_span, get_tracer, use_span
from pydantic import Discriminator
from typing_extensions import Self, TypedDict

from .runner import Runner
from .schema import TaskIdParams, TaskSendParams
from .storage import Storage

tracer = get_tracer(__name__)


class WorkerClient(ABC):
    """A client for the worker."""

    @abstractmethod
    async def send_run_task(self, params: TaskSendParams) -> None:
        """Send a task to be executed by the worker."""
        raise NotImplementedError('send_run_task is not implemented yet.')

    @abstractmethod
    async def send_cancel_task(self, params: TaskIdParams) -> None:
        """Cancel a task."""
        raise NotImplementedError('send_cancel_task is not implemented yet.')


class WorkerServer(ABC):
    """A server for the worker."""

    @abstractmethod
    async def run_task(self, params: TaskSendParams) -> None:
        """Run a task."""
        raise NotImplementedError('run_task is not implemented yet.')

    @abstractmethod
    async def cancel_task(self, params: TaskIdParams) -> None:
        """Cancel a task."""
        raise NotImplementedError('cancel_task is not implemented yet.')


@dataclass
class Worker(WorkerClient, WorkerServer, ABC):
    """The worker class is in charge of scheduling and running tasks.

    The HTTP server is in charge of scheduling tasks, but you can have a remote worker
    that is in charge of running the tasks.

    The simple implementation is the `InMemoryWorker`, which is the worker that
    runs the tasks in the same process as the HTTP server. That said, this class can be
    extended to support remote workers.
    """

    storage: Storage
    """The storage to save and load tasks."""

    runner: Runner
    """The runner to execute tasks."""

    @abstractmethod
    async def __aenter__(self) -> Self: ...

    @abstractmethod
    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any): ...


class InMemoryWorker(Worker):
    """A worker that executes tasks in memory.

    The worker keeps a loop that waits for tasks to be scheduled.
    """

    async def __aenter__(self):
        self.aexit_stack = AsyncExitStack()
        await self.aexit_stack.__aenter__()

        self._write_stream, self._read_stream = anyio.create_memory_object_stream[TaskOperations]()
        await self.aexit_stack.enter_async_context(self._read_stream)
        await self.aexit_stack.enter_async_context(self._write_stream)

        self._task_group = await self.aexit_stack.enter_async_context(anyio.create_task_group())
        self._task_group.start_soon(self._run_task_operations)

        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any):
        self._task_group.cancel_scope.cancel()
        await self.aexit_stack.__aexit__(exc_type, exc_value, traceback)

    async def _run_task_operations(self):
        async for task_operation in self._read_stream:
            with use_span(task_operation['_current_span']):
                with tracer.start_as_current_span(
                    f'{task_operation["operation"]} task', attributes={'logfire.tags': ['fasta2a']}
                ):
                    if task_operation['operation'] == 'run':
                        self._task_group.start_soon(self._handle_run_task, task_operation['params'])

    async def _handle_run_task(self, params: TaskSendParams):
        task_context = TaskContext(storage=self.storage, params=params)
        await self.runner.run(task_context)

    async def run_task(self, params: TaskSendParams) -> None:
        await self._write_stream.send(_RunTask(operation='run', params=params, _current_span=get_current_span()))

    async def cancel_task(self, params: TaskIdParams) -> None:
        await self._write_stream.send(_CancelTask(operation='cancel', params=params, _current_span=get_current_span()))

    async def send_run_task(self, params: TaskSendParams) -> None:
        await self.run_task(params)

    async def send_cancel_task(self, params: TaskIdParams) -> None:
        await self.cancel_task(params)


OperationT = TypeVar('OperationT')
ParamsT = TypeVar('ParamsT')


class _TaskOperation(TypedDict, Generic[OperationT, ParamsT]):
    """A task operation."""

    operation: OperationT
    params: ParamsT
    _current_span: Span


_RunTask = _TaskOperation[Literal['run'], TaskSendParams]
_CancelTask = _TaskOperation[Literal['cancel'], TaskIdParams]

TaskOperations = Annotated['_RunTask | _CancelTask', Discriminator('operation')]

TaskParamsT = TypeVar('TaskParamsT')


@dataclass
class TaskContext(Generic[TaskParamsT]):
    """A context for a task."""

    storage: Storage
    """The storage to save and load tasks."""

    params: TaskParamsT
    """The parameters of the task."""
