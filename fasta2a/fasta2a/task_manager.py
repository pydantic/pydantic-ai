"""This module defines the TaskManager class, which is responsible for managing tasks.

In our structure, we have the following components:

- TaskManager: A class that manages tasks.
- Scheduler: A class that schedules tasks to be sent to the worker.
- Worker: A class that executes tasks.
- Runner: A class that defines how tasks run and how history is structured.
- Storage: A class that stores tasks and artifacts.

Architecture:
```
  +-----------------+
  |   HTTP Server   |
  +-------+---------+
          ^
          | Sends Requests/
          | Receives Results
          v
  +-------+---------+
  |                 |
  |   TaskManager   |<-----------------+
  |  (coordinates)  |                  |
  |                 |                  |
  +-------+---------+                  |
          |                            |
          |  Schedules Tasks           |
          v                            v
  +------------------+         +----------------+
  |                  |         |                |
  |      Broker      | .       |    Storage     |
  |     (queues) .   |         | (persistence)  |
  |                  |         |                |
  +------------------+         +----------------+
          ^                            ^
          |                            |
          | Delegates Execution        |
          v                            |
  +------------------+                 |
  |                  |                 |
  |      Worker      |                 |
  | (implementation) |-----------------+
  |                  |
  +------------------+
```

The flow:
1. The HTTP server sends a task to TaskManager
2. TaskManager stores initial task state in Storage
3. TaskManager passes task to Scheduler
4. Scheduler determines when to send tasks to Worker
5. Worker delegates to Runner for task execution
6. Runner defines how tasks run and how history is structured
7. Worker processes task results from Runner
8. Worker reads from and writes to Storage directly
9. Worker updates task status in Storage as execution progresses
10. TaskManager can also read/write from Storage for task management
11. Client queries TaskManager for results, which reads from Storage
"""

from __future__ import annotations as _annotations

import uuid
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from .broker import Broker
from .schema import (
    CancelTaskRequest,
    CancelTaskResponse,
    GetTaskPushNotificationRequest,
    GetTaskPushNotificationResponse,
    GetTaskRequest,
    GetTaskResponse,
    ResubscribeTaskRequest,
    SendTaskRequest,
    SendTaskResponse,
    SendTaskStreamingRequest,
    SendTaskStreamingResponse,
    SetTaskPushNotificationRequest,
    SetTaskPushNotificationResponse,
    TaskNotFoundError,
)
from .storage import Storage


@dataclass
class TaskManager:
    """A task manager responsible for managing tasks."""

    broker: Broker
    storage: Storage

    _aexit_stack: AsyncExitStack | None = field(default=None, init=False)

    async def __aenter__(self):
        self._aexit_stack = AsyncExitStack()
        await self._aexit_stack.__aenter__()
        await self._aexit_stack.enter_async_context(self.broker)

        return self

    @property
    def is_running(self) -> bool:
        return self._aexit_stack is not None

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any):
        if self._aexit_stack is None:
            raise RuntimeError('TaskManager was not properly initialized.')
        await self._aexit_stack.__aexit__(exc_type, exc_value, traceback)
        self._aexit_stack = None

    async def send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """Send a task to the worker."""
        request_id = str(uuid.uuid4())
        task_id = request['params']['id']
        task = await self.storage.load_task(task_id)

        if task is None:
            session_id = request['params'].get('session_id', str(uuid.uuid4()))
            message = request['params']['message']
            task = await self.storage.submit_task(task_id, session_id, message)

        await self.broker.run_task(request['params'])
        return SendTaskResponse(jsonrpc='2.0', id=request_id, result=task)

    async def get_task(self, request: GetTaskRequest) -> GetTaskResponse:
        """Get a task, and return it to the client.

        No further actions are needed here.
        """
        task_id = request['params']['id']
        history_length = request['params'].get('history_length')
        task = await self.storage.load_task(task_id, history_length)
        if task is None:
            return GetTaskResponse(
                jsonrpc='2.0',
                id=request['id'],
                error=TaskNotFoundError(code=-32001, message='Task not found'),
            )
        return GetTaskResponse(jsonrpc='2.0', id=request['id'], result=task)

    async def cancel_task(self, request: CancelTaskRequest) -> CancelTaskResponse:
        await self.broker.cancel_task(request['params'])
        task = await self.storage.load_task(request['params']['id'])
        if task is None:
            return CancelTaskResponse(
                jsonrpc='2.0',
                id=request['id'],
                error=TaskNotFoundError(code=-32001, message='Task not found'),
            )
        return CancelTaskResponse(jsonrpc='2.0', id=request['id'], result=task)

    async def send_task_streaming(self, request: SendTaskStreamingRequest) -> SendTaskStreamingResponse:
        raise NotImplementedError('SendTaskStreaming is not implemented yet.')

    async def set_task_push_notification(
        self, request: SetTaskPushNotificationRequest
    ) -> SetTaskPushNotificationResponse:
        raise NotImplementedError('SetTaskPushNotification is not implemented yet.')

    async def get_task_push_notification(
        self, request: GetTaskPushNotificationRequest
    ) -> GetTaskPushNotificationResponse:
        raise NotImplementedError('GetTaskPushNotification is not implemented yet.')

    async def resubscribe_task(self, request: ResubscribeTaskRequest) -> SendTaskStreamingResponse:
        raise NotImplementedError('Resubscribe is not implemented yet.')
