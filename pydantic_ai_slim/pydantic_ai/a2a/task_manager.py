import datetime
import uuid
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from typing import Any

import anyio

from pydantic_ai.a2a.schema import (
    CancelTaskRequest,
    CancelTaskResponse,
    GetTaskPushNotificationRequest,
    GetTaskPushNotificationResponse,
    GetTaskRequest,
    GetTaskResponse,
    Message,
    Part,
    ResubscribeTaskRequest,
    SendTaskRequest,
    SendTaskResponse,
    SendTaskStreamingRequest,
    SendTaskStreamingResponse,
    SetTaskPushNotificationRequest,
    SetTaskPushNotificationResponse,
    Task,
    TaskSendParams,
    TaskStatus,
)
from pydantic_ai.agent import Agent
from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    UserPromptPart,
    VideoUrl,
)


class TaskManager(ABC):
    """A task manager responsible for managing tasks.

    Only `send_task` and `get_task` are required.
    """

    def __init__(self): ...

    @abstractmethod
    async def send_task(self, request: SendTaskRequest) -> SendTaskResponse: ...

    @abstractmethod
    async def get_task(self, request: GetTaskRequest) -> GetTaskResponse: ...

    async def cancel_task(self, request: CancelTaskRequest) -> CancelTaskResponse: ...

    async def send_task_streaming(self, request: SendTaskStreamingRequest) -> SendTaskStreamingResponse: ...

    async def set_task_push_notification(
        self, request: SetTaskPushNotificationRequest
    ) -> SetTaskPushNotificationResponse: ...

    async def get_task_push_notification(
        self, request: GetTaskPushNotificationRequest
    ) -> GetTaskPushNotificationResponse: ...

    async def resubscribe_task(self, request: ResubscribeTaskRequest) -> SendTaskStreamingResponse:
        raise NotImplementedError('Resubscribe is not implemented yet.')


# class TaskSerializer: ...


# class Storage:
#     artifacts: ...
#     tasks: ...


class InMemoryTaskManager(TaskManager):
    """A task manager that stores tasks in memory."""

    def __init__(self, agent: Agent):
        self.agent = agent
        # This could be Postgres.
        self.tasks: dict[str, Task] = {}
        self._write_stream, self._read_stream = anyio.create_memory_object_stream[Task]()

    async def __aenter__(self):
        self.aexit_stack = AsyncExitStack()
        await self.aexit_stack.enter_async_context(self._read_stream)
        await self.aexit_stack.enter_async_context(self._write_stream)

        self.task_handler = anyio.create_task_group()
        self.task_handler.start_soon(self._handle_tasks)

        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any):
        await self.aexit_stack.aclose()

    async def send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        request_id = str(uuid.uuid4())
        task_id = request['params']['id']
        session_id = request['params'].get('session_id', str(uuid.uuid4()))

        task = Task(
            id=task_id,
            session_id=session_id,
            status=TaskStatus(state='submitted', timestamp=datetime.datetime.now().isoformat()),
        )
        await self._write_stream.send(task)
        return SendTaskResponse(jsonrpc='2.0', id=request_id, result=task)

    async def get_task(self, request: GetTaskRequest) -> GetTaskResponse:
        task_id = request['params']['id']
        return GetTaskResponse(jsonrpc='2.0', id=request['id'], result=self.tasks[task_id])

    async def _handle_tasks(self):
        while True:
            task = await self._read_stream.receive()
            self.tasks[task['id']] = task

            if task['status']['state'] == 'submitted':
                self.task_handler.start_soon(self._handle_submitted_task, task)

    @staticmethod
    async def execute_task(send_task_params: TaskSendParams, task: Task, storage: Storage): ...

    async def _handle_submitted_task(self, request: SendTaskRequest):
        task = request['params']
        history: list[Message] = task.get('history', [])
        result = await self.agent.run(message_history=self._map_history(history))
        task['status']['state'] = 'completed'
        # return result

    def _map_history(self, history: list[Message]) -> list[ModelMessage]:
        model_messages: list[ModelMessage] = []
        for message in history:
            if message['role'] == 'user':
                # NOTE: This is the user input message.
                model_messages.append(ModelRequest(parts=self._map_request_parts(message['parts'])))
            else:
                model_messages.append(ModelResponse(parts=self._map_response_parts(message['parts'])))
        return model_messages

    def _map_request_parts(self, parts: list[Part]) -> list[ModelRequestPart]:
        model_parts: list[ModelRequestPart] = []
        for part in parts:
            if part['type'] == 'text':
                model_parts.append(UserPromptPart(content=part['text']))
            elif part['type'] == 'file':
                file = part['file']
                if 'data' in file:
                    data = file['data'].encode('utf-8')
                    content = BinaryContent(data=data, media_type=file['mime_type'])
                    model_parts.append(UserPromptPart(content=[content]))
                else:
                    url = file['url']
                    for url_cls in (DocumentUrl, AudioUrl, ImageUrl, VideoUrl):
                        content = url_cls(url=url)
                        try:
                            content.media_type
                        except ValueError:
                            continue
                        else:
                            break
                    else:
                        raise ValueError(f'Unknown file type: {file["mime_type"]}')
                    model_parts.append(UserPromptPart(content=[content]))
            elif part['type'] == 'data':
                # TODO(Marcelo): Maybe we should use this for `ToolReturnPart`, and `RetryPromptPart`.
                raise NotImplementedError('Data parts are not supported yet.')
        return model_parts

    def _map_response_parts(self, parts: list[Part]) -> list[ModelResponsePart]: ...
