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
model. A [`ContinuationChain`][pydantic_ai.models._request_timeout.ContinuationChain] carries the pinned model's
deadline from one segment to the next, so the chain stays one request to it: a segment that starts after the deadline,
for example after a long wait between polls, times out at once, and the `FallbackModel` moves on to its next model.
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
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import anyio

from ..exceptions import ModelRequestTimeout
from ..settings import ModelSettings

if TYPE_CHECKING:
    from .._run_context import RunContext
    from ..messages import ModelMessage
    from . import Model, ModelRequestParameters, StreamedResponse

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

    async def iterate(self, iterator: AsyncIterator[_T]) -> AsyncIterator[_T]:
        """Pull from `iterator` under the deadline, one item at a time."""
        while True:
            with self.enforce():
                try:
                    item = await anext(iterator)
                except StopAsyncIteration:
                    return
            yield item


@dataclass
class ContinuationChain:
    """The deadline of the model a continuation chain is pinned to, once a model that starts its own picked one."""

    pinned: RequestDeadline | None = None


_current_chain: ContextVar[ContinuationChain | None] = ContextVar('_current_continuation_chain', default=None)


@contextmanager
def use_continuation_chain(chain: ContinuationChain) -> Generator[None]:
    """Make `chain` the continuation chain of the requests made inside."""
    token = _current_chain.set(chain)
    try:
        yield
    finally:
        _current_chain.reset(token)


def current_continuation_chain() -> ContinuationChain | None:
    """The continuation chain of the request being made, if the agent graph is resolving one."""
    return _current_chain.get()


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
    run_context: RunContext[Any] | None = None,
) -> AsyncGenerator[StreamedResponse]:
    """Open a streamed request to `model` under its `request_timeout` deadline, if it's set.

    The deadline covers opening the stream and each pull of its next event.
    """
    stream = model.request_stream(messages, model_settings, model_request_parameters, run_context)
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
