from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import TracebackType
from typing import Self

import anyio
from httpx2 import Timeout
from openai import APIError, AsyncOpenAI, Omit, OpenAIError
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
from openai.types.responses.responses_server_event import ResponsesServerEvent, ResponseWsError
from openai.types.websocket_connection_options import WebSocketConnectionOptions
from pydantic_core import to_json
from websockets.exceptions import InvalidStatus, WebSocketException

from ..exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior, UserError
from . import _suggest_known_model_id_from_provider_error  # pyright: ignore[reportPrivateUsage]


class ConnectionOptions(WebSocketConnectionOptions):
    """Include the opening-handshake timeout omitted by the SDK's options type."""

    open_timeout: None


class WebSocketTransportError(ModelAPIError):
    """A transport failure on a Responses WebSocket."""


# OpenAI accepts up to 32 distinct named `stream_id` values per connection and never releases one
# before the connection closes, so lane IDs are pooled rather than minted per request:
# https://developers.openai.com/api/docs/guides/websocket-mode
NAMED_LANE_LIMIT = 32


@dataclass(eq=False)
class ResponsesWebSocketLane:
    """One ordered `stream_id` lane: it runs one response at a time and caches that lane's latest response."""

    stream_id: str | None
    active: bool = False
    abandoned: bool = False
    """The response was closed before its terminal event, which the reader discards when it arrives."""
    latest_response_id: str | None = None
    last_used: int = 0
    events: deque[ResponsesServerEvent] = field(default_factory=deque[ResponsesServerEvent])
    receiving: anyio.CancelScope = field(default_factory=anyio.CancelScope)
    """The consumer's wait for this lane's next event, which closing the response interrupts."""


def _is_terminal(event: ResponsesServerEvent) -> bool:
    # The SDK's lane router also treats an `error` event as the end of its request.
    return isinstance(event, (ResponseCompletedEvent, ResponseFailedEvent, ResponseIncompleteEvent, ResponseWsError))


@dataclass
class ResponsesWebSocket:
    """A socket owned by one model connection context, multiplexing concurrent requests over lanes.

    The default lane, which sends no `stream_id`, serves the first request so a single conversation keeps
    the plain wire format. Whichever request is waiting reads the next event and routes it to its lane.
    """

    connection: AsyncResponsesConnection
    model_name: str
    model_id_namespace: str
    headers: Mapping[str, str]
    closed: bool = False
    """The owning `connect()` context exited."""
    connection_closed: bool = False
    failure: str | None = None
    """Why the socket failed. Later requests raise `WebSocketTransportError` so a fallback can take over."""
    lanes: list[ResponsesWebSocketLane] = field(default_factory=lambda: [ResponsesWebSocketLane(None)])
    reading: bool = False
    changed: anyio.Event = field(default_factory=anyio.Event)
    uses: int = 0

    @classmethod
    async def connect(
        cls,
        client: AsyncOpenAI,
        model_name: str,
        model_id_namespace: str,
        headers: Mapping[str, str],
        timeout: Timeout,
        options: WebSocketConnectionOptions,
    ) -> Self:
        # The enclosing scope owns the deadline, including an unlimited connect timeout.
        connection_options: ConnectionOptions = {**options, 'open_timeout': None}
        try:
            with anyio.fail_after(timeout.connect):
                connection = await client.responses.connect(
                    extra_headers=headers, websocket_connection_options=connection_options
                ).enter()
        except APIError:
            raise
        except OpenAIError as exc:
            raise UserError(
                'The OpenAI SDK could not configure this WebSocket connection: '
                f'{exc} Use HTTP or supply an SDK client with WebSocket-compatible authentication.'
            ) from exc
        except InvalidStatus as exc:
            if exc.response.status_code < 400:
                raise ModelAPIError(
                    model_name=model_name, message=f'WebSocket handshake failed with status {exc.response.status_code}'
                ) from exc
            raise ModelHTTPError(
                status_code=exc.response.status_code,
                model_name=model_name,
                body=bytes(exc.response.body).decode(errors='replace'),
                # `dict(exc.response.headers)` raises `MultipleValuesError` on a repeated header.
                headers={name: ', '.join(exc.response.headers.get_all(name)) for name in exc.response.headers},
            ) from exc
        except (OSError, WebSocketException, TimeoutError) as exc:
            raise ModelAPIError(model_name=model_name, message=f'WebSocket connection failed: {exc}') from exc

        return cls(connection, model_name, model_id_namespace, cls.effective_headers(client, headers))

    @staticmethod
    def effective_headers(client: AsyncOpenAI, extra_headers: Mapping[str, str]) -> dict[str, str]:
        # The client's credentials stay out: a shared client can refresh them during any handshake, so they don't
        # show what this socket sent.
        headers: dict[str, str | Omit] = {
            key.lower(): value
            for header_set in (client.default_headers, extra_headers)
            for key, value in header_set.items()
        }
        # Drop only `Omit`: a value of any other type has to reach the handshake comparison, so a request header
        # the socket wasn't opened with is rejected rather than silently ignored.
        return {key: value for key, value in headers.items() if not isinstance(value, Omit)}

    async def reserve(self, response_id: str | None, timeout: Timeout) -> ResponsesWebSocketLane:
        """Reserve the idle lane whose latest response is `response_id`, else a lane with the least cached state."""
        try:
            with anyio.fail_after(timeout.pool):
                while True:
                    if self.closed:
                        raise UserError('This Responses WebSocket is closed. Open a new `model.connect()` context.')
                    if self.failure is not None:
                        raise WebSocketTransportError(model_name=self.model_name, message=self.failure)
                    if (lane := self._idle_lane(response_id)) is not None:
                        self.uses += 1
                        lane.active = True
                        lane.last_used = self.uses
                        return lane
                    # Only the reader frees an abandoned lane, so drive it when nobody else is reading.
                    if not self.reading and any(lane.abandoned for lane in self.lanes):
                        await self._read()
                    else:
                        await self.changed.wait()
        except TimeoutError as exc:
            raise ModelAPIError(
                model_name=self.model_name, message='No Responses WebSocket lane became free within the pool timeout.'
            ) from exc

    def _idle_lane(self, response_id: str | None) -> ResponsesWebSocketLane | None:
        idle = [lane for lane in self.lanes if not lane.active]
        if response_id is not None:
            for lane in idle:
                if lane.latest_response_id == response_id:
                    return lane
        # Reusing a lane evicts the response it cached, which another conversation may still continue from.
        for lane in idle:
            if lane.latest_response_id is None:
                return lane
        if len(self.lanes) <= NAMED_LANE_LIMIT:
            lane = ResponsesWebSocketLane(f'lane-{len(self.lanes)}')
            self.lanes.append(lane)
            return lane
        return min(idle, key=lambda lane: lane.last_used, default=None)

    def release(self, lane: ResponsesWebSocketLane, *, finished: bool) -> None:
        """Free a lane, or hold an unfinished one until the reader discards the rest of its response."""
        if lane.abandoned:
            return
        if not finished:
            lane.receiving.cancel()
            lane.latest_response_id = None
            finished = self.closed or self.failure is not None or any(_is_terminal(event) for event in lane.events)
        lane.events.clear()
        if finished:
            lane.active = False
            self._notify()
        else:
            lane.abandoned = True

    async def request(
        self, lane: ResponsesWebSocketLane, body: Mapping[str, object], timeout: Timeout
    ) -> ResponsesWebSocketStream:
        frame = {'type': 'response.create', **body}
        if lane.stream_id is not None:
            frame['stream_id'] = lane.stream_id
        data = to_json(frame).decode()
        try:
            with anyio.fail_after(timeout.write):
                await self.connection.send_raw(data)
        except (OSError, WebSocketException, TimeoutError) as exc:
            failure = f'WebSocket request failed: {exc}'
            await self._fail(failure)
            raise ModelAPIError(model_name=self.model_name, message=failure) from exc
        except BaseException:
            # `websockets` writes an unfragmented frame before its only suspension point, so a cancelled send still
            # reaches the server: discard that response when it arrives instead of reusing the lane before then.
            self.release(lane, finished=False)
            raise
        return ResponsesWebSocketStream(self, lane, timeout)

    async def receive(self, lane: ResponsesWebSocketLane) -> ResponsesServerEvent:
        with anyio.CancelScope() as lane.receiving:
            while not lane.events:
                # A closed connection fails the next read, so that is where a closed context surfaces too.
                if self.failure is not None:
                    raise WebSocketTransportError(model_name=self.model_name, message=self.failure)
                if self.reading:
                    await self.changed.wait()
                else:
                    await self._read()
            return lane.events.popleft()
        raise WebSocketTransportError(
            model_name=self.model_name, message='WebSocket response interrupted before completion: the response closed.'
        )

    async def _read(self) -> None:
        self.reading = True
        try:
            try:
                event = await self.connection.recv()
            except (OSError, WebSocketException) as exc:
                await self._fail(f'WebSocket response interrupted before completion: {exc}')
                return
            lane = next(
                (lane for lane in self.lanes if lane.active and lane.stream_id == event.stream_id),
                None,
            )
            if isinstance(event, ResponseWsError) and (
                lane is None or event.error.code == 'websocket_connection_limit_reached'
            ):
                # An error no active lane owns leaves the connection's state unknown, and the lifetime limit ends it.
                error = event.error
                message = f'{error.code}: {error.message}' if error.code else error.message
                await self._fail(f'Responses WebSocket connection failed: {message}')
            elif lane is not None and lane.abandoned:
                if _is_terminal(event):
                    lane.abandoned = False
                    lane.active = False
            elif lane is not None:
                lane.events.append(event)
        finally:
            self.reading = False
            self._notify()

    def _notify(self) -> None:
        self.changed.set()
        self.changed = anyio.Event()

    async def _fail(self, failure: str) -> None:
        self.failure = failure
        await self._close_connection()

    async def close(self) -> None:
        self.closed = True
        await self._close_connection()

    async def _close_connection(self) -> None:
        if not self.connection_closed:
            self.connection_closed = True
            self._notify()
            with anyio.CancelScope(shield=True):
                await self.connection.close()


@dataclass
class ResponsesWebSocketStream:
    """Expose one response on its lane as a stream without closing the shared socket."""

    websocket: ResponsesWebSocket
    lane: ResponsesWebSocketLane
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
                event = await self.websocket.receive(self.lane)
        except TimeoutError as exc:
            raise WebSocketTransportError(
                model_name=self.websocket.model_name,
                message=f'WebSocket response interrupted before completion: {exc}',
            ) from exc

        if isinstance(event, ResponseWsError):
            # A 4xx or 5xx on a lane evicts the response it continued from, so nothing on the lane is cached.
            self.completed = True
            self.lane.latest_response_id = None
            if event.status is not None:
                raise ModelHTTPError(
                    status_code=event.status,
                    model_name=self.websocket.model_name,
                    body=event.error.model_dump(exclude_none=True),
                    headers=event.error.headers,
                    suggested_model_id=(
                        _suggest_known_model_id_from_provider_error(
                            self.websocket.model_id_namespace, self.websocket.model_name
                        )
                        if event.error.code == 'model_not_found'
                        else None
                    ),
                )
            message = f'{event.error.code}: {event.error.message}' if event.error.code else event.error.message
            raise ModelAPIError(model_name=self.websocket.model_name, message=message)
        # Steering isn't supported: a lane serves one response at a time, so its cancellation and cleanup have
        # one owner, and steering would need its own event routing.
        if isinstance(event, (ResponseSteerAcceptedEvent, ResponseSteerPendingEvent, ResponseSteerFailedEvent)):
            raise UnexpectedModelBehavior(f'Unexpected Responses WebSocket event: {event.type!r}')

        if _is_terminal(event):
            self.completed = True
            # With store=false the provider only promises the latest completed response in this lane's
            # connection-local cache. A failed or incomplete response cannot be the parent of another response.
            # The shared response parser still maps an incomplete result or a failed response without an error
            # payload to its normal finish reason.
            self.lane.latest_response_id = event.response.id if isinstance(event, ResponseCompletedEvent) else None
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
            self.websocket.release(self.lane, finished=self.completed)
