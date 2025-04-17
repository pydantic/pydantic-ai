"""This module defines the TaskManager class, which is responsible for managing tasks.

In our structure, we have the following components:

- TaskManager: A class that manages tasks.
- Worker: A class that executes tasks.
- Runner: A class that defines how tasks run and how history is structured.
- Storage: A class that stores tasks and artifacts.

Architecture:
```
  +-----------------+
  |   HTTP Server   |
  +-------+---------+
          |
          | Sends Requests/
          | Receives Results
          v
  +-------+---------+        +----------------+
  |                 |        |                |
  |   TaskManager   |<------>|    Storage     |
  |  (coordinates)  |        | (persistence)  |
  |                 |        |                |
  +-------+---------+        +----------------+
          |                         ^
          |                         |
          | Triggers Tasks          |
          v                         |
  +------------------+              |
  |      Worker      |--------------┘
  | (executes tasks) |
  +------------------+
          ^
          |
          | Delegates Execution
          v
  +------------------+
  |      Runner      |
  | (implementation) |
  +------------------+
```

The flow:
1. The HTTP server sends a task to TaskManager
2. TaskManager stores initial task state in Storage
3. TaskManager triggers Worker to execute tasks
4. Worker delegates to Runner for task execution
5. Runner defines how tasks run and how history is structured
6. Worker processes task results from Runner
7. Worker reads from and writes to Storage directly
8. Worker updates task status in Storage as execution progresses
9. TaskManager can also read/write from Storage for task management
10. Client queries TaskManager for results, which reads from Storage
"""

import datetime
import uuid
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from pydantic_ai.a2a.schema import (
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
    Task,
    TaskNotFoundError,
    TaskStatus,
)
from pydantic_ai.a2a.storage import Storage
from pydantic_ai.a2a.worker import Worker


@dataclass
class TaskManager:
    """A task manager responsible for managing tasks."""

    worker: Worker
    storage: Storage

    async def __aenter__(self):
        self.aexit_stack = AsyncExitStack()
        await self.aexit_stack.__aenter__()
        await self.aexit_stack.enter_async_context(self.worker)

        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any):
        await self.aexit_stack.__aexit__(exc_type, exc_value, traceback)

    async def send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """Send a task to the worker."""
        request_id = str(uuid.uuid4())
        task_id = request['params']['id']
        task = await self.storage.load_task(task_id)

        if task is None:
            session_id = request['params'].get('session_id', str(uuid.uuid4()))
            task = Task(
                id=task_id,
                session_id=session_id,
                status=TaskStatus(state='submitted', timestamp=datetime.datetime.now().isoformat()),
            )
            await self.storage.save_task(task)

        await self.worker.schedule_task(request['params'])
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
        await self.worker.cancel_task(request['params'])
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
