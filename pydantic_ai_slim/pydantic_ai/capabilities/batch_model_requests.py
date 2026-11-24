from __future__ import annotations

import asyncio
import weakref
from dataclasses import KW_ONLY, dataclass, field
from datetime import timedelta
from types import TracebackType
from typing import Any, Self

from pydantic import TypeAdapter

from pydantic_ai.models import Model
from pydantic_ai.models._batch_model import BatchedModel, BatchPool
from pydantic_ai.models._batch_settings import (
    DEFAULT_MAX_POLL_INTERVAL,
    DEFAULT_MAX_REQUESTS,
    DEFAULT_MAX_RESUBMISSIONS,
    DEFAULT_POLL_INTERVAL,
)

from .abstract import AbstractCapability

_DEFAULT_MAX_WAIT = timedelta(seconds=30)

_DURATION_ADAPTER = TypeAdapter(timedelta)
"""Reads a spec's duration, given as a number of seconds or an ISO 8601 duration such as `PT5M`."""


@dataclass
class BatchModelRequests(AbstractCapability[Any]):
    """Send an agent's model requests through their provider's batch API.

    Batch APIs cost less, typically half, but a result can take up to a day. The agent runs as usual: each request is
    queued with other requests made through the same `BatchModelRequests` instance in the same process, submitted in a
    batch, and checked on until its result arrives. Add one instance to several agents, or pass it to many runs with
    `capabilities=`, and their requests share batches.

    ```python
    from datetime import timedelta

    from pydantic_ai import Agent
    from pydantic_ai.capabilities import BatchModelRequests

    batching = BatchModelRequests(max_requests=5_000, max_wait=timedelta(minutes=5))
    agent = Agent('anthropic:claude-sonnet-5-5', capabilities=[batching])
    ```

    A batch is submitted when it holds `max_requests` requests, when its oldest request has waited `max_wait`, or when
    [`flush()`][pydantic_ai.capabilities.BatchModelRequests.flush] is called, whichever comes first.

    Meanwhile the run's message history ends in a suspended response that records the batch, so a run resumed from
    that history, in this process or another, waits for the same batch instead of submitting the request again. To
    end the run instead of waiting, add [`PendingModelResponse`][pydantic_ai.output.PendingModelResponse] to the
    agent's `output_type`. See [Batch processing](../batch.md) for a guide.

    A request that expired before the provider ran it, is missing from its batch's results, or failed with a rate
    limit or server error, is submitted again in a new batch, up to `max_resubmissions` times, as a provider SDK
    retries a live request. Other errors, like an invalid request or a cancelled batch, reach the run as usual.

    The capability wraps whichever model serves each request, which must have a batch API. A streamed request waits
    for its batch too, and then streams each part of the result in one event. Batching needs an `asyncio` event loop. Submissions and checks run as tasks shared by the requests
    waiting on them; using the capability as an async context manager cancels any that are still running when it
    exits.
    """

    _: KW_ONLY

    max_requests: int = DEFAULT_MAX_REQUESTS
    """The most requests one batch holds. The provider's own limit caps it."""

    max_wait: timedelta | None = _DEFAULT_MAX_WAIT
    """How long the oldest queued request waits for others before its batch is submitted.

    Later requests never extend it. `None` waits until the batch is full or
    [`flush()`][pydantic_ai.capabilities.BatchModelRequests.flush] is called.
    """

    max_resubmissions: int = DEFAULT_MAX_RESUBMISSIONS
    """How many times a request is submitted again in a new batch after it expired or failed with a rate limit,
    timeout or server error. `0` raises the error instead. A cancelled request is never submitted again.
    """

    poll_interval: timedelta = DEFAULT_POLL_INTERVAL
    """How long to wait before first checking a submitted batch. The interval grows by half after every check.

    Checks of a batch made within `poll_interval` of each other, like those of runs resumed one after another from
    their message histories, share one status request.
    """

    max_poll_interval: timedelta = DEFAULT_MAX_POLL_INTERVAL
    """The longest wait between checks of a submitted batch."""

    _pool: BatchPool = field(init=False, repr=False, compare=False)
    _entered: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, int] = field(
        default_factory=weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, int], init=False, repr=False, compare=False
    )
    """How many `async with` blocks are open on each event loop, as exiting the last one closes that loop's batching."""

    def __post_init__(self) -> None:
        self._pool = BatchPool(
            max_requests=self.max_requests,
            max_wait=self.max_wait,
            max_resubmissions=self.max_resubmissions,
            poll_interval=self.poll_interval,
            max_poll_interval=self.max_poll_interval,
        )

    @classmethod
    def from_spec(
        cls,
        *,
        max_requests: int = DEFAULT_MAX_REQUESTS,
        max_wait: timedelta | float | None = _DEFAULT_MAX_WAIT,
        max_resubmissions: int = DEFAULT_MAX_RESUBMISSIONS,
        poll_interval: timedelta | float = DEFAULT_POLL_INTERVAL,
        max_poll_interval: timedelta | float = DEFAULT_MAX_POLL_INTERVAL,
    ) -> BatchModelRequests:
        """Build from an agent spec, which gives each duration as a number of seconds or an ISO 8601 duration.

        Each agent built from a spec gets an instance of its own, so the agent's runs share batches with each other,
        but not with other agents' runs.

        YAML form:

            capabilities:
              - BatchModelRequests: {}            # default settings
              - BatchModelRequests:
                  max_requests: 5000
                  max_wait: PT5M                  # or 300, in seconds
        """
        return cls(
            max_requests=max_requests,
            max_wait=None if max_wait is None else _DURATION_ADAPTER.validate_python(max_wait),
            max_resubmissions=max_resubmissions,
            poll_interval=_DURATION_ADAPTER.validate_python(poll_interval),
            max_poll_interval=_DURATION_ADAPTER.validate_python(max_poll_interval),
        )

    def get_wrapper_model(self, model: Model) -> Model | None:
        """Wrap `model` so each request it's given waits in this capability's batch pool, unless it's batched already."""
        if isinstance(model, BatchedModel):
            # Already batched, by this capability or another one.
            return None
        return self._pool.wrap(model)

    async def flush(self) -> None:
        """Submit every queued request now instead of waiting for a full batch or `max_wait`."""
        await self._pool.flush()

    async def __aenter__(self) -> Self:
        loop = asyncio.get_running_loop()
        self._entered[loop] = self._entered.get(loop, 0) + 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Once the outermost `async with` on this event loop exits, fail its queued requests and cancel its running submissions and checks."""
        loop = asyncio.get_running_loop()
        if (entered := self._entered[loop] - 1) == 0:
            del self._entered[loop]
            await self._pool.close()
        else:
            self._entered[loop] = entered
