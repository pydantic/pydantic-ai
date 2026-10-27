from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import TracebackType
from typing import Protocol, Self

import anyio
from httpx2 import Timeout
from openai import APIError, AsyncOpenAI, Omit
from openai.resources.responses.responses import AsyncResponsesConnection
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponseStreamEvent,
)
from openai.types.responses.response_steer_accepted_event import ResponseSteerAcceptedEvent
from openai.types.responses.response_steer_failed_event import ResponseSteerFailedEvent
from openai.types.responses.response_steer_pending_event import ResponseSteerPendingEvent
from openai.types.responses.responses_server_event import ResponsesServerEvent, ResponseWsError
from openai.types.websocket_connection_options import WebSocketConnectionOptions
from pydantic_core import from_json, to_json
from websockets.exceptions import ConnectionClosed, InvalidStatus, WebSocketException

from .._utils import is_str_dict
from ..exceptions import ModelAPIError, ModelConnectionError, ModelHTTPError, ModelTimeoutError, UnexpectedModelBehavior


class ConnectionOptions(WebSocketConnectionOptions):
    """Include the opening-handshake timeout omitted by the SDK's options type."""

    open_timeout: None


class StatusError(Protocol):
    """Build the error a response with this status and provider error raises, as over HTTP."""

    def __call__(
        self,
        *,
        status_code: int,
        code: str | None,
        error_type: str | None,
        message: str,
        body: object,
        headers: Mapping[str, str] | None,
    ) -> ModelHTTPError: ...


# OpenAI's limits per connection: https://developers.openai.com/api/docs/guides/websocket-mode
# It accepts up to 32 distinct named `stream_id` values and never releases one before the connection closes, so lane
# IDs are pooled rather than minted per request.
NAMED_LANE_LIMIT = 32
# It runs up to 16 responses at once and queues the rest, so a request past that goes over HTTP instead of waiting.
ACTIVE_RESPONSE_LIMIT = 16
# It closes a connection after 60 minutes. New requests move to a new socket ten minutes before that, so a response
# that starts on the old one has time to finish.
RETIREMENT_AGE = 50 * 60


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


@dataclass(eq=False)
class ResponsesWebSocket:
    """One socket, multiplexing concurrent requests over lanes.

    The default lane, which sends no `stream_id`, serves the first request so a single conversation keeps
    the plain wire format. Whichever request is waiting reads the next event and routes it to its lane.
    """

    connection: AsyncResponsesConnection
    model_name: str
    headers: Mapping[str, str]
    """The handshake headers, without the client's credentials. A request with other headers can't use the socket."""
    named_lane_limit: int
    """How many named `stream_id` lanes concurrent requests may use; with none, only one request runs at a time."""
    status_error: StatusError
    loop: asyncio.AbstractEventLoop = field(default_factory=asyncio.get_running_loop)
    """The event loop the socket opened on, the only one that can use it."""
    opened_at: float = field(default_factory=time.monotonic)
    connection_closed: bool = False
    failure: str | None = None
    """Why the socket failed. Its requests still waiting for events raise `ModelConnectionError`."""
    lanes: list[ResponsesWebSocketLane] = field(default_factory=lambda: [ResponsesWebSocketLane(None)])
    reading: bool = False
    changed: anyio.Event = field(default_factory=anyio.Event)
    uses: int = 0

    @classmethod
    async def connect(
        cls,
        client: AsyncOpenAI,
        model_name: str,
        headers: Mapping[str, str],
        timeout: Timeout,
        options: WebSocketConnectionOptions,
        named_lane_limit: int,
        status_error: StatusError,
    ) -> Self:
        # Unless the caller sets them, lift the `websockets` limits on message size and on unread frames:
        # - OpenAI events can exceed the 1 MiB default message size; an image generation result does.
        # - Past 16 unread frames, `websockets` stops reading the socket. An abandoned response's frames wait unread
        #   until the next request, so the socket would miss the server's keepalive pings and close. Each lane's
        #   buffer is unbounded already, so lifting this adds no new kind of growth.
        # The enclosing scope owns the deadline, including an unlimited connect timeout.
        connection_options: ConnectionOptions = {'max_size': None, 'max_queue': None, **options, 'open_timeout': None}
        try:
            with anyio.fail_after(timeout.connect):
                connection = await client.responses.connect(
                    extra_headers=headers, websocket_connection_options=connection_options
                ).enter()
        except APIError:
            raise
        except InvalidStatus as exc:
            status_code = exc.response.status_code
            if status_code < 400:
                raise ModelAPIError(
                    model_name=model_name, message=f'WebSocket handshake failed with status {status_code}'
                ) from exc
            text = bytes(exc.response.body).decode(errors='replace')
            # Read like the SDK reads an HTTP error response, so the rejection gets the category it gets over HTTP.
            try:
                parsed: object = from_json(text)
            except ValueError:
                parsed = None
            error: object = parsed.get('error', parsed) if is_str_dict(parsed) else None
            fields = (
                {key: value for key, value in error.items() if isinstance(value, str)} if is_str_dict(error) else {}
            )
            raise status_error(
                status_code=status_code,
                code=fields.get('code'),
                error_type=fields.get('type'),
                message=fields.get('message', text),
                body=error if error is not None else text,
                # `dict(exc.response.headers)` raises `MultipleValuesError` on a repeated header.
                headers={name: ', '.join(exc.response.headers.get_all(name)) for name in exc.response.headers},
            ) from exc
        except TimeoutError as exc:
            raise ModelTimeoutError(
                model_name=model_name, message='WebSocket connection timed out.', phase='connect'
            ) from exc
        except (OSError, WebSocketException) as exc:
            raise ModelConnectionError(
                model_name=model_name, message=f'WebSocket connection failed: {exc}', phase='connect'
            ) from exc

        return cls(connection, model_name, cls.effective_headers(client, headers), named_lane_limit, status_error)

    @staticmethod
    def effective_headers(client: AsyncOpenAI, extra_headers: Mapping[str, str]) -> dict[str, str]:
        # The client's credentials stay out: a shared client can refresh them during any handshake, so they don't
        # show what this socket sent.
        headers: dict[str, str | Omit] = {
            key.lower(): value
            for header_set in (client.default_headers, extra_headers)
            for key, value in header_set.items()
        }
        # Drop only `Omit`: a value of any other type has to reach the handshake comparison, so a request with a
        # header the socket wasn't opened with goes over HTTP rather than silently dropping it.
        return {key: value for key, value in headers.items() if not isinstance(value, Omit)}

    @property
    def accepts_requests(self) -> bool:
        """Whether new requests may use the socket: it's open, hasn't failed, and isn't near OpenAI's time limit."""
        return (
            self.failure is None and not self.connection_closed and time.monotonic() - self.opened_at < RETIREMENT_AGE
        )

    def reserve(self, response_id: str | None) -> ResponsesWebSocketStream | None:
        """Reserve the idle lane whose latest response is `response_id`, else a lane with the least cached state.

        Returns `None` without waiting when every lane is busy or OpenAI's limit of responses at once is reached.
        """
        if not self.has_capacity():
            return None
        idle = [lane for lane in self.lanes if not lane.active]
        lane = next((lane for lane in idle if lane.latest_response_id == response_id), None)
        if lane is None:
            # Reusing a lane evicts the response it cached, which another conversation may still continue from.
            lane = next((lane for lane in idle if lane.latest_response_id is None), None)
        if lane is None and len(self.lanes) <= self.named_lane_limit:
            lane = ResponsesWebSocketLane(f'lane-{len(self.lanes)}')
            self.lanes.append(lane)
        if lane is None:
            lane = min(idle, key=lambda lane: lane.last_used)
        self.uses += 1
        lane.active = True
        lane.last_used = self.uses
        return ResponsesWebSocketStream(self, lane)

    def has_capacity(self) -> bool:
        active = sum(lane.active for lane in self.lanes)
        return active < ACTIVE_RESPONSE_LIMIT and (active < len(self.lanes) or len(self.lanes) <= self.named_lane_limit)

    async def release(self, lane: ResponsesWebSocketLane, *, finished: bool) -> None:
        """Free a lane, or hold an unfinished one until the reader discards the rest of its response."""
        if not finished:
            lane.receiving.cancel()
            lane.latest_response_id = None
            finished = (
                self.failure is not None or self.connection_closed or any(_is_terminal(event) for event in lane.events)
            )
        lane.events.clear()
        if finished:
            lane.active = False
            self._notify()
        else:
            lane.abandoned = True
        await self.close_if_idle()

    async def close_if_idle(self) -> None:
        """Close the socket once no response reads from it and no new request can use it.

        That's a socket past its age limit, or one whose lanes are all held by abandoned responses: nothing would read
        the rest of those, so the socket could never take a request again.
        """
        if not any(lane.active and not lane.abandoned for lane in self.lanes) and not (
            self.accepts_requests and self.has_capacity()
        ):
            await self.close()

    async def receive(self, lane: ResponsesWebSocketLane) -> ResponsesServerEvent:
        with anyio.CancelScope() as lane.receiving:
            while not lane.events:
                if self.failure is not None:
                    # The request was sent, so the provider may have acted on it.
                    raise ModelConnectionError(model_name=self.model_name, message=self.failure, phase='read')
                if self.reading:
                    await self.changed.wait()
                else:
                    await self._read()
            return lane.events.popleft()
        raise ModelConnectionError(
            model_name=self.model_name,
            message='WebSocket response interrupted before completion: the response closed.',
            phase='read',
        )

    async def _read(self) -> None:
        self.reading = True
        try:
            try:
                event = await self.connection.recv()
            except (OSError, WebSocketException) as exc:
                await self.fail(f'WebSocket response interrupted before completion: {exc}')
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
                await self.fail(f'Responses WebSocket connection failed: {message}')
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

    async def fail(self, failure: str) -> None:
        # Closing the socket interrupts any other request reading from it, which shouldn't hide what failed first.
        if self.failure is None:
            self.failure = failure
        await self.close()

    async def close(self) -> None:
        if not self.connection_closed:
            self.connection_closed = True
            self._notify()
            with anyio.CancelScope(shield=True):
                await self.connection.close()


@dataclass(eq=False)
class ResponsesWebSocketStream:
    """One request on a reserved lane, then its response as a stream, without closing the shared socket."""

    websocket: ResponsesWebSocket
    lane: ResponsesWebSocketLane
    retried: bool = False
    """Whether the request already moved to a new socket because the last one closed before it was sent."""
    read_timeout: float | None = None
    rejection_retries: int = 0
    pending_event: ResponseStreamEvent | None = None
    sent: bool = False
    completed: bool = False
    closed: bool = False

    async def send(self, body: Mapping[str, object], timeout: Timeout) -> bool:
        """Send the request, returning `False` when the socket had closed before any of it was written."""
        frame = {'type': 'response.create', **body}
        if self.lane.stream_id is not None:
            frame['stream_id'] = self.lane.stream_id
        data = to_json(frame).decode()
        self.read_timeout = timeout.read
        # `websockets` writes an unfragmented frame before its only suspension point, so a cancelled send still
        # reaches the server: closing this then holds the lane until that response arrives to be discarded.
        self.sent = True
        websocket = self.websocket
        try:
            with anyio.fail_after(timeout.write):
                await websocket.connection.send_raw(data)
        except (OSError, WebSocketException, TimeoutError) as exc:
            timed_out = isinstance(exc, TimeoutError)
            failure = 'WebSocket request timed out.' if timed_out else f'WebSocket request failed: {exc}'
            await websocket.fail(failure)
            # `websockets` raises `ConnectionClosed` without a cause when the connection had closed before anything
            # was written, as when the server closed an idle socket.
            if isinstance(exc, ConnectionClosed) and exc.__cause__ is None:
                self.sent = False
                return False
            error = ModelTimeoutError if timed_out else ModelConnectionError
            raise error(model_name=websocket.model_name, message=failure, phase='write') from exc
        return True

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> ResponseStreamEvent:
        if self.pending_event is not None:
            event, self.pending_event = self.pending_event, None
            return event
        if self.completed or self.closed:
            raise StopAsyncIteration
        try:
            with anyio.fail_after(self.read_timeout):
                event = await self.websocket.receive(self.lane)
        except TimeoutError as exc:
            raise ModelTimeoutError(
                model_name=self.websocket.model_name, message='WebSocket response timed out.', phase='read'
            ) from exc

        if isinstance(event, ResponseWsError):
            # A 4xx or 5xx on a lane evicts the response it continued from, so nothing on the lane is cached.
            self.completed = True
            self.lane.latest_response_id = None
            error = event.error
            if event.status is None:
                # The response parser raises this as it does an error in an HTTP stream.
                return ResponseErrorEvent(
                    type='error',
                    code=error.code,
                    message=error.message,
                    param=error.param,
                    sequence_number=event.sequence_number or 0,
                )
            raise self.websocket.status_error(
                status_code=event.status,
                code=error.code,
                error_type=error.type,
                message=error.message,
                body=error.model_dump(exclude_none=True),
                headers=error.headers,
            )
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
            await self.websocket.release(self.lane, finished=self.completed or not self.sent)
