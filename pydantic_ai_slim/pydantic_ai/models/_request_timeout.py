"""Enforcement of `ModelSettings['request_timeout']` and `ModelSettings['first_event_timeout']` on one request to one model.

Both are deadlines, absolute points on anyio's clock fixed when the request starts. `request_timeout`'s covers the whole
request; `first_event_timeout`'s, only on a streamed request, covers it until the stream's first event. Every wait that
belongs to the request runs under [`RequestDeadline.enforce`][pydantic_ai.models._request_timeout.RequestDeadline.enforce],
which cancels it at whichever of the two passes first: the whole non-streamed request, opening a stream, and each pull of
the stream's next event, after the first of which only `request_timeout`'s applies. A scope is never held across a
`yield`, so a stream can be consumed from any task, and time spent between pulls counts towards the deadline without
anything having to watch it.

Under Temporal the clock is the workflow's, so the deadline is a durable workflow timer, and expiry cancels the
in-flight model activity, which also ends its retries. The workflow only sees a streamed segment once the activity has
consumed it, so the first-event deadline is enforced inside the activity instead (`open_durable_unit_stream`), as it is
in the step or task of the other durable engines.

A [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] starts a deadline for each model it tries, so the
agent graph starts none for it. When the model that answers suspends its response (Anthropic `pause_turn`, OpenAI
background mode), the graph's continuation loop calls the `FallbackModel` again for each segment, pinned to that
model. So the chain stays one request to that model, the `FallbackModel` carries its deadline on the suspended
response, as a UTC timestamp in the `__pydantic_ai__` metadata next to the continuation pin, and each later segment
gets what is left of it: a segment that starts after the deadline, for example after a long wait between polls, times
out at once, and the `FallbackModel` moves on to its next model. The first-event deadline is carried the same way,
until a segment has produced an event, after which it's carried as `None`, so later segments don't get one. Wall-clock
time is carried because a durable engine runs each segment in its own activity or step, possibly in another process,
where anyio's clock means nothing. The timestamps are dropped once the response completes, and when a later run
resumes a suspended response from history, which is a new request.
"""

from __future__ import annotations as _annotations

import math
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
from typing import TYPE_CHECKING, Any, TypeVar

import anyio

from .._utils import is_str_dict, now_utc
from ..exceptions import ModelRequestTimeout
from ..messages import ModelMessage, ModelResponse
from ..settings import ModelSettings

if TYPE_CHECKING:
    from .._run_context import RunContext
    from . import Model, ModelRequestParameters, StreamedResponse

_PYDANTIC_AI_METADATA_KEY = '__pydantic_ai__'
_CARRIED_DEADLINE_KEY = 'request_deadline'
_CARRIED_FIRST_EVENT_DEADLINE_KEY = 'first_event_deadline'
_CARRIED_KEYS = (_CARRIED_DEADLINE_KEY, _CARRIED_FIRST_EVENT_DEADLINE_KEY)

_T = TypeVar('_T')


@dataclass(frozen=True)
class RequestDeadline:
    """The deadlines of one request to one model.

    At least one of `timeout` and `first_event_timeout` is set.
    """

    model_name: str
    timeout: float | None
    """The `request_timeout`, in seconds, if it's set."""
    deadline: float = math.inf
    """When the request times out, on anyio's clock."""
    first_event_timeout: float | None = None
    """The `first_event_timeout` of a streamed request, in seconds, if it's set."""
    first_event_deadline: float = math.inf
    """When the request times out if its stream hasn't produced an event, on anyio's clock: never once it has."""

    @classmethod
    def start(cls, model_name: str, model_settings: ModelSettings | None, *, stream: bool) -> RequestDeadline | None:
        """Start the deadlines for a request made now, or `None` if neither applies to it.

        `first_event_timeout` only applies to a streamed request.
        """
        settings = model_settings or {}
        timeout = settings.get('request_timeout')
        first_event_timeout = settings.get('first_event_timeout') if stream else None
        if timeout is None and first_event_timeout is None:
            return None
        now = anyio.current_time()
        return cls(
            model_name=model_name,
            timeout=timeout,
            deadline=math.inf if timeout is None else now + timeout,
            first_event_timeout=first_event_timeout,
            first_event_deadline=math.inf if first_event_timeout is None else now + first_event_timeout,
        )

    @contextmanager
    def enforce(self) -> Generator[None]:
        """Cancel the enclosed wait at the earlier deadline and raise `ModelRequestTimeout` instead."""
        with anyio.CancelScope(deadline=min(self.deadline, self.first_event_deadline)) as scope:
            yield
        if scope.cancelled_caught:
            if self.first_event_deadline < self.deadline:
                assert self.first_event_timeout is not None
                raise ModelRequestTimeout(self.model_name, self.first_event_timeout, setting='first_event_timeout')
            assert self.timeout is not None
            raise ModelRequestTimeout(self.model_name, self.timeout)

    def after_first_event(self) -> RequestDeadline:
        """What applies once the stream has produced its first event: the `request_timeout` deadline, if any."""
        return self if self.first_event_deadline == math.inf else replace(self, first_event_deadline=math.inf)

    def first_event_only(self) -> RequestDeadline | None:
        """Just the `first_event_timeout` deadline, if any, for when something else enforces `request_timeout`."""
        if self.first_event_timeout is None:
            return None
        return replace(self, timeout=None, deadline=math.inf)

    async def iterate(self, iterator: AsyncIterator[_T]) -> AsyncIterator[_T]:
        """Pull from `iterator` under the deadline, one item at a time, and the first under the first-event one too."""
        deadline = self
        while True:
            with deadline.enforce():
                try:
                    item = await anext(iterator)
                except StopAsyncIteration:
                    return
            deadline = deadline.after_first_event()
            yield item


def _expires_at(deadline: float) -> str:
    """`deadline` on anyio's clock, as a wall-clock UTC timestamp."""
    return (now_utc() + timedelta(seconds=deadline - anyio.current_time())).isoformat()


def _on_anyio_clock(expires_at: str) -> float:
    """Wall-clock UTC timestamp `expires_at`, on anyio's clock."""
    return anyio.current_time() + (datetime.fromisoformat(expires_at) - now_utc()).total_seconds()


def carry_deadline(response: ModelResponse | StreamedResponse, deadline: RequestDeadline | None) -> None:
    """Carry `deadline` on a suspended response, for the segments that continue it, if there is one.

    It goes next to the continuation pin, if there is one. A nested `FallbackModel` has no deadline of its own, and has
    already carried the one of the model it picked. Once a streamed response has produced an event, its first-event
    deadline no longer applies; a non-streamed response has none.
    """
    if deadline is None:
        return
    carried: dict[str, str | None] = {}
    if deadline.timeout is not None:
        carried[_CARRIED_DEADLINE_KEY] = _expires_at(deadline.deadline)
    if deadline.first_event_timeout is not None:
        # `None` once the first event has arrived, so a later segment doesn't start a deadline of its own.
        first_event_received = isinstance(response, ModelResponse) or bool(response.get().parts)
        pending = not first_event_received and deadline.first_event_deadline != math.inf
        carried[_CARRIED_FIRST_EVENT_DEADLINE_KEY] = _expires_at(deadline.first_event_deadline) if pending else None
    if response.metadata is None:
        response.metadata = {}
    namespace = response.metadata.get(_PYDANTIC_AI_METADATA_KEY)
    if not is_str_dict(namespace):
        namespace = response.metadata[_PYDANTIC_AI_METADATA_KEY] = {}
    namespace.update(carried)


def resume_deadline(deadline: RequestDeadline | None, response: ModelResponse) -> RequestDeadline | None:
    """`deadline` of a segment that continues the suspended `response`, moved to what's left of the one it carries.

    A response that carries none starts a new request, with `deadline` as it is.
    """
    namespace = response.metadata.get(_PYDANTIC_AI_METADATA_KEY) if response.metadata else None
    if deadline is None or not is_str_dict(namespace):
        return deadline
    if deadline.timeout is not None and isinstance(expires_at := namespace.get(_CARRIED_DEADLINE_KEY), str):
        deadline = replace(deadline, deadline=_on_anyio_clock(expires_at))
    if deadline.first_event_timeout is not None and _CARRIED_FIRST_EVENT_DEADLINE_KEY in namespace:
        if isinstance(expires_at := namespace[_CARRIED_FIRST_EVENT_DEADLINE_KEY], str):
            return replace(deadline, first_event_deadline=_on_anyio_clock(expires_at))
        # An earlier segment produced the first event.
        return deadline.after_first_event()
    return deadline


def without_carried_deadline(response: ModelResponse) -> ModelResponse:
    """`response` without a carried deadline, copying its metadata rather than changing it."""
    metadata = response.metadata
    namespace = (metadata or {}).get(_PYDANTIC_AI_METADATA_KEY)
    if metadata is None or not (is_str_dict(namespace) and any(key in namespace for key in _CARRIED_KEYS)):
        return response
    namespace = {key: value for key, value in namespace.items() if key not in _CARRIED_KEYS}
    if namespace:
        return replace(response, metadata={**metadata, _PYDANTIC_AI_METADATA_KEY: namespace})
    return replace(
        response, metadata={key: value for key, value in metadata.items() if key != _PYDANTIC_AI_METADATA_KEY}
    )


def start_request_deadline(
    model: Model, model_settings: ModelSettings | None, *, stream: bool
) -> RequestDeadline | None:
    """Start the deadlines of a request to `model` made now, if they're set there or on the model.

    `None` for a [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel], also behind a wrapper: it starts fresh
    ones for each model it tries.
    """
    return model._start_request_deadline(model_settings, stream=stream)  # pyright: ignore[reportPrivateUsage]


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
    """Open a streamed request to `model` under its deadlines, if they're set.

    They start when the stream is opened, not when this is called, and cover opening the stream and each pull of its
    next event.
    """
    stream = model.request_stream(messages, model_settings, model_request_parameters)
    async with stream_under_deadline(start_request_deadline(model, model_settings, stream=True), stream) as response:
        yield response


@asynccontextmanager
async def open_durable_unit_stream(
    model: Model,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
    run_context: RunContext[Any] | None,
) -> AsyncGenerator[StreamedResponse]:
    """Open a streamed request to `model` in a durable execution engine's unit, under its first-event deadline.

    The unit consumes the stream and hands the workflow its final response, so the workflow, which enforces
    `request_timeout` around the unit, can't see when the first event arrives. A segment that continues a suspended
    response gets what's left of the first-event deadline the unit before it carried on it.
    """
    deadline = start_request_deadline(model, model_settings, stream=True)
    if (
        deadline is not None
        and (deadline := deadline.first_event_only()) is not None
        and messages
        and isinstance(last := messages[-1], ModelResponse)
        and last.state == 'suspended'
    ):
        deadline = resume_deadline(deadline, last)
    stream = model.request_stream(messages, model_settings, model_request_parameters, run_context)
    async with stream_under_deadline(deadline, stream) as response:
        yield response
        if response.state == 'suspended':
            carry_deadline(response, deadline)


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
