"""Session-owned Responses WebSocket transport; response parsing stays in `openai.py`."""

from __future__ import annotations

from collections.abc import AsyncIterator, Generator, Mapping
from contextlib import contextmanager
from typing import Literal, Self, cast

import anyio
from httpx2 import Timeout
from openai import AsyncOpenAI, Omit
from openai.resources.responses.responses import AsyncResponsesConnection
from openai.types import responses
from openai.types.responses.response_create_params import ContextManagement, Moderation, PromptCacheOptions, ToolChoice
from openai.types.responses.responses_client_event_param import ResponseCreate
from openai.types.responses.responses_server_event import ResponseWsError
from openai.types.shared_params import Reasoning
from typing_extensions import TypedDict

from .._utils import is_str_dict
from ..exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior, UserError


class ResponsesCreateOptions(TypedDict):
    """The shared HTTP/WS create body, before the SDK removes omitted settings."""

    model: str
    input: list[responses.ResponseInputItemParam]
    instructions: str | Omit
    parallel_tool_calls: bool | Omit
    tools: list[responses.ToolParam] | Omit
    tool_choice: ToolChoice | Omit
    previous_response_id: str | Omit
    reasoning: Reasoning | Omit
    text: responses.ResponseTextConfigParam | Omit
    truncation: Literal['auto', 'disabled'] | Omit
    context_management: list[ContextManagement] | Omit
    max_output_tokens: int | Omit
    temperature: float | Omit
    top_p: float | Omit
    service_tier: responses.ServiceTier | Omit
    conversation: str | Omit
    top_logprobs: int | Omit
    store: bool | None | Omit
    user: str | Omit
    include: list[responses.ResponseIncludable] | Omit
    prompt_cache_key: str | Omit
    prompt_cache_retention: Literal['in_memory', '24h'] | Omit
    prompt_cache_options: PromptCacheOptions | Omit
    moderation: Moderation | Omit


class ResponsesWebSocket:
    """One isolated conversation's resource, closed by `Model.open_session()`."""

    def __init__(self, client: AsyncOpenAI, model_name: str) -> None:
        self.client = client
        self.model_name = model_name
        self.connection: AsyncResponsesConnection | None = None
        self.headers: Mapping[str, str] | None = None
        self.active = False
        self.closed = False
        self.last_response_id: str | None = None

    async def disconnect(self) -> None:
        connection, self.connection = self.connection, None
        self.last_response_id = None
        if connection is not None:
            with anyio.CancelScope(shield=True):
                await connection.close()

    async def close(self) -> None:
        self.closed = True
        await self.disconnect()

    async def create(
        self,
        options: ResponsesCreateOptions,
        *,
        headers: Mapping[str, str],
        timeout: Timeout | float | None,
        extra_body: object,
    ) -> ResponsesWebSocketStream:
        if self.closed:
            raise UserError('This Responses model session has closed.')
        if self.active:
            raise UserError('A Responses model session can execute only one request at a time.')
        # Reject malformed wire options before opening or invalidating a usable connection.
        body = {key: value for key, value in options.items() if not isinstance(value, Omit)}
        if extra_body is not None:
            if not is_str_dict(extra_body):
                raise UserError('Responses WebSocket `extra_body` must be a JSON object.')
            body.update(extra_body)
        body['type'] = 'response.create'
        if 'stream' in body or 'background' in body:
            raise UserError('Responses WebSocket requests do not accept `stream` or `background`.')
        self.active = True
        try:
            if self.connection is not None and headers != self.headers:
                await self.disconnect()
            if self.connection is None:
                # The SDK's WS handshake builds cached platform headers synchronously, unlike
                # its HTTP path. On Linux this can run `uname`; initialize them off the event loop.
                await anyio.to_thread.run_sync(self.client.platform_headers)
                connect_timeout = timeout.connect if isinstance(timeout, Timeout) else timeout
                with _map_websocket_errors(self.model_name), anyio.fail_after(connect_timeout):
                    self.connection = await self.client.responses.connect(
                        extra_headers=headers,
                        # Do not replay requests with unknown outcomes or lose connection-local input.
                        max_retries=0,
                    ).enter()
                self.headers = dict(headers)
            write_timeout = timeout.write if isinstance(timeout, Timeout) else timeout
            with _map_websocket_errors(self.model_name), anyio.fail_after(write_timeout):
                # The SDK accepts wire values only: omitted settings were removed above.
                await self.connection.send(cast(ResponseCreate, body))
            return ResponsesWebSocketStream(self, timeout.read if isinstance(timeout, Timeout) else timeout)
        except BaseException:
            self.active = False
            await self.disconnect()
            raise


class ResponsesWebSocketStream:
    """A bounded response, not an owner of a successfully completed connection."""

    def __init__(self, owner: ResponsesWebSocket, read_timeout: float | None) -> None:
        self.owner = owner
        self.read_timeout = read_timeout
        self.finished = False
        self.closed = False

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if not self.finished:
                # Early final output, cancellation and failed decoding cannot leave unread frames
                # for another request. Keep the conversation, but discard this physical connection.
                await self.owner.disconnect()
        finally:
            self.owner.active = False

    async def __aiter__(self) -> AsyncIterator[responses.ResponseStreamEvent]:
        connection = self.owner.connection
        assert connection is not None
        try:
            while not self.closed and not self.finished:
                with _map_websocket_errors(self.owner.model_name), anyio.fail_after(self.read_timeout):
                    event = await connection.recv()
                if isinstance(event, ResponseWsError):
                    error = event.error
                    if event.status is not None:
                        raise ModelHTTPError(event.status, self.owner.model_name, error.model_dump())
                    raise ModelAPIError(self.owner.model_name, f'{error.code}: {error.message}')
                if isinstance(
                    event,
                    (
                        responses.ResponseSteerAcceptedEvent,
                        responses.ResponseSteerPendingEvent,
                        responses.ResponseSteerFailedEvent,
                    ),
                ):
                    raise UnexpectedModelBehavior('Received steering events without an active steering operation.')
                if isinstance(
                    event,
                    (
                        responses.ResponseCompletedEvent,
                        responses.ResponseIncompleteEvent,
                        responses.ResponseFailedEvent,
                    ),
                ):
                    self.finished = True
                    self.owner.last_response_id = event.response.id
                yield event
        except BaseException:
            await self.close()
            raise


@contextmanager
def _map_websocket_errors(model_name: str) -> Generator[None]:
    # Keep the optional transport dependency out of the default HTTP import path.
    try:
        from websockets.exceptions import InvalidStatus, WebSocketException
    except ImportError as exc:
        raise ImportError('Install `openai[realtime]` to use Responses WebSocket transport.') from exc

    try:
        yield
    except InvalidStatus as exc:
        response = exc.response
        raise ModelHTTPError(
            response.status_code,
            model_name,
            response.body.decode('utf-8', errors='replace'),
            headers={key.lower(): value for key, value in response.headers.raw_items()},
        ) from exc
    except TimeoutError as exc:
        raise ModelAPIError(model_name, 'Responses WebSocket request timed out.') from exc
    except (OSError, WebSocketException) as exc:
        raise ModelAPIError(model_name, f'Responses WebSocket connection failed: {exc}') from exc
