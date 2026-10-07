from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Self

import anyio
from httpx2 import Timeout
from openai import AsyncOpenAI, Omit
from openai.resources.responses.responses import AsyncResponsesConnection
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponseStreamEvent,
)
from openai.types.responses.response_steer_accepted_event import ResponseSteerAcceptedEvent
from openai.types.responses.response_steer_failed_event import ResponseSteerFailedEvent
from openai.types.responses.response_steer_pending_event import ResponseSteerPendingEvent
from openai.types.responses.responses_server_event import ResponseWsError
from openai.types.websocket_connection_options import WebSocketConnectionOptions
from pydantic_core import to_json
from websockets.exceptions import InvalidStatus, WebSocketException

from ..exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior, UserError


@dataclass
class ResponsesWebSocket:
    """A socket owned by one model connection context, shared by sequential requests."""

    connection: AsyncResponsesConnection
    model_name: str
    headers: Mapping[str, str]
    active: bool = False
    closed: bool = False

    @classmethod
    async def connect(
        cls,
        client: AsyncOpenAI,
        model_name: str,
        headers: Mapping[str, str],
        timeout: Timeout,
        options: WebSocketConnectionOptions,
    ) -> Self:
        try:
            with anyio.fail_after(timeout.connect):
                connection = await client.responses.connect(
                    extra_headers=headers, websocket_connection_options=options
                ).enter()
        except InvalidStatus as exc:
            raise ModelHTTPError(
                status_code=exc.response.status_code,
                model_name=model_name,
                body=bytes(exc.response.body).decode(errors='replace'),
                headers=dict(exc.response.headers),
            ) from exc
        except (OSError, WebSocketException, TimeoutError) as exc:
            raise ModelAPIError(model_name=model_name, message=f'WebSocket connection failed: {exc}') from exc

        return cls(connection, model_name, cls.effective_headers(client, headers))

    @staticmethod
    def effective_headers(client: AsyncOpenAI, extra_headers: Mapping[str, str]) -> dict[str, str]:
        headers: dict[str, str | Omit] = {
            key.lower(): value
            for header_set in (client.auth_headers, client.default_headers, extra_headers)
            for key, value in header_set.items()
        }
        return {key: value for key, value in headers.items() if isinstance(value, str)}

    async def request(self, body: Mapping[str, object], timeout: Timeout) -> ResponsesWebSocketStream:
        if self.closed:
            raise UserError('This Responses WebSocket is closed. Open a new `model.connect()` context.')
        if self.active:
            raise UserError(
                'A Responses WebSocket supports one active response. Use a separate `model.connect()` context.'
            )

        # No checkpoint separates the overlap check from reserving the socket.
        self.active = True
        sent = False
        try:
            with anyio.fail_after(timeout.write):
                await self.connection.send_raw(to_json({'type': 'response.create', **body}).decode())
            sent = True
        except (OSError, WebSocketException, TimeoutError) as exc:
            raise ModelAPIError(model_name=self.model_name, message=f'WebSocket request failed: {exc}') from exc
        finally:
            if not sent:
                await self.close()
                self.active = False
        return ResponsesWebSocketStream(self, timeout)

    async def close(self) -> None:
        if not self.closed:
            self.closed = True
            with anyio.CancelScope(shield=True):
                await self.connection.close()


@dataclass
class ResponsesWebSocketStream:
    """Expose one response as a stream without closing a healthy shared socket."""

    websocket: ResponsesWebSocket
    timeout: Timeout
    completed: bool = False
    closed: bool = False

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> ResponseStreamEvent:
        if self.completed or self.closed:
            raise StopAsyncIteration
        try:
            with anyio.fail_after(self.timeout.read):
                event = await self.websocket.connection.recv()
        except (OSError, WebSocketException, TimeoutError) as exc:
            raise ModelAPIError(
                model_name=self.websocket.model_name,
                message=f'WebSocket response interrupted before completion: {exc}',
            ) from exc

        if isinstance(event, ResponseWsError):
            if event.status is not None:
                raise ModelHTTPError(
                    status_code=event.status,
                    model_name=self.websocket.model_name,
                    body=event.error.model_dump(exclude_none=True),
                    headers=event.error.headers,
                )
            message = f'{event.error.code}: {event.error.message}' if event.error.code else event.error.message
            raise ModelAPIError(model_name=self.websocket.model_name, message=message)
        if isinstance(event, (ResponseSteerAcceptedEvent, ResponseSteerPendingEvent, ResponseSteerFailedEvent)):
            raise UnexpectedModelBehavior(f'Unexpected Responses WebSocket event: {event.type!r}')

        if isinstance(event, (ResponseCompletedEvent, ResponseFailedEvent, ResponseIncompleteEvent)):
            self.completed = True
        return event

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        await self.close()

    async def close(self) -> None:
        if not self.closed:
            self.closed = True
            try:
                if not self.completed:
                    # Unread events cannot be attributed safely to a subsequent response.
                    await self.websocket.close()
            finally:
                self.websocket.active = False
