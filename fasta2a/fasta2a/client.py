from __future__ import annotations as _annotations

import uuid
from typing import Any

from a2a.client.client import A2AClient as BaseA2AClient
from a2a.client.errors import A2AClientHTTPError, A2AClientJSONError
from a2a.types import (
    GetTaskRequest,
    GetTaskResponse,
    Message,
    MessageSendParams,
    PushNotificationConfig,
    SendMessageRequest,
    SendMessageResponse,
)

try:
    import httpx
except ImportError as _import_error:
    raise ImportError(
        "httpx is required to use the A2AClient. Please install it with `pip install httpx`.",
    ) from _import_error

UnexpectedResponseError = A2AClientHTTPError


class A2AClient:
    """A client for the A2A protocol, built on the Google A2A SDK."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if http_client is None:
            self.http_client = httpx.AsyncClient()
        else:
            self.http_client = http_client

        # The SDK client requires a URL on the agent card, but we can initialize it with a dummy
        # and then set the URL directly for the internal calls.
        self._sdk_client = BaseA2AClient(httpx_client=self.http_client, url=base_url)

    async def send_task(
        self,
        message: Message,
        history_length: int | None = None,
        push_notification: PushNotificationConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SendMessageResponse:
        """Sends a task to the A2A server."""
        if not message.taskId:
            message.taskId = str(uuid.uuid4())

        params = MessageSendParams(
            message=message,
            configuration={
                "historyLength": history_length,
                "pushNotificationConfig": push_notification,
            },
            metadata=metadata,
        )
        payload = SendMessageRequest(
            id=str(uuid.uuid4()), method="message/send", params=params
        )

        try:
            response = await self._sdk_client.send_message(payload)
            return response
        except (A2AClientHTTPError, A2AClientJSONError) as e:
            raise UnexpectedResponseError(getattr(e, "status_code", 500), str(e)) from e

    async def get_task(self, task_id: str) -> GetTaskResponse:
        """Retrieves a task from the A2A server."""
        payload = GetTaskRequest(
            id=str(uuid.uuid4()), method="tasks/get", params={"id": task_id}
        )
        try:
            response = await self._sdk_client.get_task(payload)
            return response
        except (A2AClientHTTPError, A2AClientJSONError) as e:
            raise UnexpectedResponseError(getattr(e, "status_code", 500), str(e)) from e
