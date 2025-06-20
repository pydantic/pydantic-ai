from __future__ import annotations as _annotations

import uuid
from typing import Any

import httpx
from a2a.client import A2AClient as SDKA2AClient
from a2a.types import (
    GetTaskRequest,
    GetTaskResponse,
    Message,
    MessageSendConfiguration,
    SendMessageRequest,
    SendMessageResponse,
    TaskQueryParams,
)

from .schema import PushNotificationConfig


class A2AClient:
    """A client for the A2A protocol."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if http_client is None:
            http_client = httpx.AsyncClient()
        # The SDK's client will be initialized with a URL that points to the JSON-RPC endpoint.
        # fasta2a server sets this at root '/', so we append it.
        self.sdk_client = SDKA2AClient(http_client, url=f'{base_url.rstrip("/")}/')

    async def send_task(
        self,
        message: Message,
        history_length: int | None = None,
        push_notification: PushNotificationConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SendMessageResponse:
        """Sends a task to the agent.

        This now maps to the 'message/send' A2A method.
        """
        if metadata:
            message.metadata = (message.metadata or {}) | metadata

        configuration = MessageSendConfiguration(
            historyLength=history_length,
            pushNotificationConfig=push_notification,
        )

        request = SendMessageRequest(
            id=str(uuid.uuid4()),
            params={"message": message, "configuration": configuration},
        )
        return await self.sdk_client.send_message(request)

    async def get_task(self, task_id: str) -> GetTaskResponse:
        """Retrieves a task from the agent."""
        request = GetTaskRequest(
            id=str(uuid.uuid4()), params=TaskQueryParams(id=task_id)
        )
        return await self.sdk_client.get_task(request)
