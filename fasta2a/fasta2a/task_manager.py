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
    SendMessageRequest,
    SendMessageResponse,
    SetTaskPushNotificationRequest,
    SetTaskPushNotificationResponse,
    StreamMessageRequest,
    TaskNotFoundError,
    TaskSendParams,
)
from .storage import Storage


@dataclass
class TaskManager:
    """A task manager responsible for managing tasks."""

    broker: Broker
    storage: Storage[Any]

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

    async def send_message(self, request: SendMessageRequest) -> SendMessageResponse:
        """Send a message using the A2A v0.2.3 protocol."""
        request_id = request['id']
        message = request['params']['message']
        context_id = message.get('context_id', str(uuid.uuid4()))

        task = await self.storage.submit_task(context_id, message)

        broker_params: TaskSendParams = {'id': task['id'], 'context_id': context_id, 'message': message}
        config = request['params'].get('configuration', {})
        history_length = config.get('history_length')
        if history_length is not None:
            broker_params['history_length'] = history_length

        await self.broker.run_task(broker_params)
        return SendMessageResponse(jsonrpc='2.0', id=request_id, result=task)

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

    async def stream_message(self, request: StreamMessageRequest) -> None:
        """Stream messages using Server-Sent Events."""
        raise NotImplementedError('message/stream method is not implemented yet.')

    async def set_task_push_notification(
        self, request: SetTaskPushNotificationRequest
    ) -> SetTaskPushNotificationResponse:
        raise NotImplementedError('SetTaskPushNotification is not implemented yet.')

    async def get_task_push_notification(
        self, request: GetTaskPushNotificationRequest
    ) -> GetTaskPushNotificationResponse:
        raise NotImplementedError('GetTaskPushNotification is not implemented yet.')

    async def resubscribe_task(self, request: ResubscribeTaskRequest) -> None:
        raise NotImplementedError('Resubscribe is not implemented yet.')
