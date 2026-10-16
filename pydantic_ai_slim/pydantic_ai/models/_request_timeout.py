"""Enforcement of `ModelSettings['request_timeout']`: one deadline over one request to one model.

The deadline is an absolute point on anyio's clock, fixed when the request starts. Every wait that belongs
to the request runs under [`RequestDeadline.enforce`][pydantic_ai.models._request_timeout.RequestDeadline.enforce]:
the whole non-streamed request, opening a stream, and each pull of the stream's next event. A scope is never
held across a `yield`, so a stream can be consumed from any task, and time spent between pulls counts towards
the deadline without anything having to watch it.

Under Temporal the clock is the workflow's, so the deadline is a durable workflow timer, and expiry cancels the
in-flight model activity, which also ends its retries.

A [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] starts a deadline for each model it tries, so the
agent graph starts none for it. When the model that answers suspends its response (Anthropic `pause_turn`, OpenAI
background mode), the graph's continuation loop calls the `FallbackModel` again for each segment, pinned to that
model. So the chain stays one request to that model, the `FallbackModel` carries its deadline on the suspended
response, as a UTC timestamp in the `__pydantic_ai__` metadata next to the continuation pin, and each later segment
gets what is left of it: a segment that starts after the deadline, for example after a long wait between polls, times
out at once, and the `FallbackModel` moves on to its next model. Wall-clock time is carried because a durable engine
runs each segment in its own activity or step, possibly in another process, where anyio's clock means nothing. The
timestamp is dropped once the response completes, and when a later run resumes a suspended response from history,
which is a new request.
"""

from __future__ import annotations as _annotations

from collections.abc import AsyncGenerator, AsyncIterator, Generator
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    AsyncExitStack,
    asynccontextmanager,
    contextmanager,
    nullcontext,
)
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, TypeVar

import anyio

from .._utils import is_str_dict, now_utc
from ..exceptions import ModelRequestTimeout
from ..settings import ModelSettings

if TYPE_CHECKING:
    from ..messages import ModelMessage, ModelResponse
    from . import Model, ModelRequestParameters, StreamedResponse

_PYDANTIC_AI_METADATA_KEY = '__pydantic_ai__'
_CARRIED_DEADLINE_KEY = 'request_deadline'

_T = TypeVar('_T')


@dataclass(frozen=True)
class RequestDeadline:
    """The deadline of one request to one model."""

    model_name: str
    timeout: float
    deadline: float
    """When the request times out, on anyio's clock."""

    @classmethod
    def start(cls, model_name: str, model_settings: ModelSettings | None) -> RequestDeadline | None:
        """Start the deadline for a request made now, or `None` if `request_timeout` isn't set."""
        timeout = (model_settings or {}).get('request_timeout')
        if timeout is None:
            return None
        return cls(model_name=model_name, timeout=timeout, deadline=anyio.current_time() + timeout)

    @contextmanager
    def enforce(self) -> Generator[None]:
        """Cancel the enclosed wait at the deadline and raise `ModelRequestTimeout` instead."""
        with anyio.CancelScope(deadline=self.deadline) as scope:
            yield
        if scope.cancelled_caught:
            raise ModelRequestTimeout(self.model_name, self.timeout)

    def expires_at(self) -> datetime:
        """When the request times out, in wall-clock UTC."""
        return now_utc() + timedelta(seconds=self.deadline - anyio.current_time())

    def until(self, expires_at: datetime) -> RequestDeadline:
        """This deadline moved to wall-clock UTC `expires_at`, e.g. one carried over from an earlier segment."""
        return replace(self, deadline=anyio.current_time() + (expires_at - now_utc()).total_seconds())

    async def iterate(self, iterator: AsyncIterator[_T]) -> AsyncIterator[_T]:
        """Pull from `iterator` under the deadline, one item at a time."""
        while True:
            with self.enforce():
                try:
                    item = await anext(iterator)
                except StopAsyncIteration:
                    return
            yield item


def carry_deadline(response: ModelResponse | StreamedResponse, deadline: RequestDeadline | None) -> None:
    """Carry `deadline` on a suspended response, for the segments that continue it, if there is one.

    It goes next to the continuation pin, which a `FallbackModel` stamps on the response first. A nested
    `FallbackModel` has no deadline of its own, and has already carried the one of the model it picked.
    """
    if deadline is None:
        return
    assert response.metadata is not None, 'The continuation pin is stamped before the deadline is carried'
    namespace = response.metadata.setdefault(_PYDANTIC_AI_METADATA_KEY, {})
    namespace[_CARRIED_DEADLINE_KEY] = deadline.expires_at().isoformat()


def carried_deadline(response: ModelResponse) -> datetime | None:
    """The deadline carried on a suspended response, in wall-clock UTC, if any."""
    namespace = (response.metadata or {}).get(_PYDANTIC_AI_METADATA_KEY)
    if is_str_dict(namespace) and isinstance(expires_at := namespace.get(_CARRIED_DEADLINE_KEY), str):
        return datetime.fromisoformat(expires_at)
    return None


def without_carried_deadline(response: ModelResponse) -> ModelResponse:
    """`response` without a carried deadline, copying its metadata rather than changing it."""
    metadata = response.metadata
    namespace = (metadata or {}).get(_PYDANTIC_AI_METADATA_KEY)
    if metadata is None or not (is_str_dict(namespace) and _CARRIED_DEADLINE_KEY in namespace):
        return response
    # The continuation pin stays next to it, so the namespace isn't left empty.
    namespace = {key: value for key, value in namespace.items() if key != _CARRIED_DEADLINE_KEY}
    return replace(response, metadata={**metadata, _PYDANTIC_AI_METADATA_KEY: namespace})


def start_request_deadline(model: Model, model_settings: ModelSettings | None) -> RequestDeadline | None:
    """Start the `request_timeout` deadline of a request to `model` made now, if it's set there or on the model.

    `None` for a [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel], also behind a wrapper: it starts a
    fresh one for each model it tries.
    """
    return model._start_request_deadline(model_settings)  # pyright: ignore[reportPrivateUsage]


def enforce_request_deadline(deadline: RequestDeadline | None) -> AbstractContextManager[None]:
    """Enforce `deadline` on the enclosed wait, if there is one."""
    return deadline.enforce() if deadline is not None else nullcontext()


@asynccontextmanager
async def open_request_stream(
    model: Model,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
) -> AsyncGenerator[StreamedResponse]:
    """Open a streamed request to `model` under its `request_timeout` deadline, if it's set.

    The deadline starts when the stream is opened, not when this is called, and covers opening the stream and each
    pull of its next event.
    """
    stream = model.request_stream(messages, model_settings, model_request_parameters)
    async with stream_under_deadline(start_request_deadline(model, model_settings), stream) as streamed_response:
        yield streamed_response


@asynccontextmanager
async def stream_under_deadline(
    deadline: RequestDeadline | None, stream: AbstractAsyncContextManager[StreamedResponse]
) -> AsyncGenerator[StreamedResponse]:
    """Open `stream` and pull each of its events under `deadline`, if there is one."""
    if deadline is None:
        async with stream as streamed_response:
            yield streamed_response
        return
    async with AsyncExitStack() as stack:
        with deadline.enforce():
            streamed_response = await stack.enter_async_context(stream)
        streamed_response._enforce_request_deadline(deadline)  # pyright: ignore[reportPrivateUsage]
        yield streamed_response
