"""Enforcement of `ModelSettings['request_timeout']`: one deadline over one request to one model.

The deadline is an absolute point on anyio's clock, fixed when the request starts. Every wait that belongs
to the request runs under [`RequestDeadline.enforce`][pydantic_ai.models._request_timeout.RequestDeadline.enforce]:
the whole non-streamed request, opening a stream, and each pull of the stream's next event. A scope is never
held across a `yield`, so a stream can be consumed from any task, and time spent between pulls counts towards
the deadline without anything having to watch it.

Under Temporal the clock is the workflow's, so the deadline is a durable workflow timer, and expiry cancels the
in-flight model activity, which also ends its retries.
"""

from __future__ import annotations as _annotations

from collections.abc import AsyncIterator, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TypeVar

import anyio

from ..exceptions import ModelRequestTimeout
from ..settings import ModelSettings

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
