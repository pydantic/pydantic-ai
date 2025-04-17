from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from typing import Any, Generic, Literal, Self, TypeVar

import anyio
from typing_extensions import TypedDict

from .runner import Runner
from .schema import TaskIdParams, TaskSendParams
from .storage import Storage


class Worker(ABC):
    """In charge of executing tasks."""

    storage: Storage
    """The storage to save and load tasks."""

    runner: Runner
    """The runner to execute tasks."""

    @abstractmethod
    async def __aenter__(self) -> Self: ...

    @abstractmethod
    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any): ...

    @abstractmethod
    async def schedule_task(self, params: TaskSendParams) -> None: ...

    @abstractmethod
    async def cancel_task(self, params: TaskIdParams) -> None: ...


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
        await self.aexit_stack.__aexit__(exc_type, exc_value, traceback)

    async def _run_task_operations(self):
        async for task_operation in self._read_stream:
            # TODO(Marcelo): Build the TaskContext to pass to the runner. The object can have a
            # `save_task` or `save_artifact` or `save_task_history` method.
            if task_operation['operation'] == 'run':
                self._task_group.start_soon(self._handle_send_task, task_operation['params'])

    async def _handle_send_task(self, params: TaskSendParams):
        task = await self.storage.load_task(params['id'])
        if task is None:
            raise ValueError(f'Task {params["id"]} not found')
        await self.runner.run(params, task)

    async def schedule_task(self, params: TaskSendParams) -> None:
        await self._write_stream.send(RunTask(operation='run', params=params))

    async def cancel_task(self, params: TaskIdParams) -> None:
        await self._write_stream.send(CancelTask(operation='cancel', params=params))


OperationT = TypeVar('OperationT')
ParamsT = TypeVar('ParamsT')


class TaskOperation(TypedDict, Generic[OperationT, ParamsT]):
    operation: OperationT
    params: ParamsT


RunTask = TaskOperation[Literal['run'], TaskSendParams]
CancelTask = TaskOperation[Literal['cancel'], TaskIdParams]

TaskOperations = RunTask | CancelTask
