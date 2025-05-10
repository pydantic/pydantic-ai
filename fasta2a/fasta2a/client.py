from __future__ import annotations as _annotations

import uuid
from typing import Any

import pydantic

from .schema import (
    GetTaskRequest,
    GetTaskResponse,
    Message,
    PushNotificationConfig,
    SendTaskRequest,
    SendTaskResponse,
    TaskSendParams,
    a2a_request_ta,
)

send_task_response_ta = pydantic.TypeAdapter(SendTaskResponse)
get_task_response_ta = pydantic.TypeAdapter(GetTaskResponse)

try:
    import httpx
except ImportError as _import_error:
    raise ImportError(
        'httpx is required to use the A2AClient. Please install it with `pip install httpx`.',
    ) from _import_error


class A2AClient:
    """A client for the A2A protocol."""

    def __init__(self, base_url: str = 'http://localhost:8000', http_client: httpx.AsyncClient | None = None) -> None:
        if http_client is None:
            self.http_client = httpx.AsyncClient(base_url=base_url)
        else:
            self.http_client = http_client
            self.http_client.base_url = base_url

    async def send_task(
        self,
        message: Message,
        history_length: int | None = None,
        push_notification: PushNotificationConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SendTaskResponse:
        task = TaskSendParams(message=message, id=str(uuid.uuid4()))
        if history_length is not None:
            task['history_length'] = history_length
        if push_notification is not None:
            task['push_notification'] = push_notification
        if metadata is not None:
            task['metadata'] = metadata

        payload = SendTaskRequest(jsonrpc='2.0', id=None, method='tasks/send', params=task)
        content = a2a_request_ta.dump_json(payload, by_alias=True)
        response = await self.http_client.post('/', content=content, headers={'Content-Type': 'application/json'})
        self._raise_for_status(response)
        return send_task_response_ta.validate_json(response.content)

    async def get_task(self, task_id: str) -> GetTaskResponse:
        payload = GetTaskRequest(jsonrpc='2.0', id=None, method='tasks/get', params={'id': task_id})
        content = a2a_request_ta.dump_json(payload, by_alias=True)
        response = await self.http_client.post('/', content=content, headers={'Content-Type': 'application/json'})
        self._raise_for_status(response)
        return get_task_response_ta.validate_json(response.content)

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code >= 400:
            raise UnexpectedResponseError(response.status_code, response.text)


class UnexpectedResponseError(Exception):
    """An error raised when an unexpected response is received from the server."""

    def __init__(self, status_code: int, content: str) -> None:
        self.status_code = status_code
        self.content = content
