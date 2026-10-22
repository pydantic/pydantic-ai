from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from datetime import timedelta
from types import TracebackType
from typing import Any, Self

from pydantic_ai.models import Model
from pydantic_ai.models._batch_model import BatchedModel, BatchPool
from pydantic_ai.models._batch_settings import (
    DEFAULT_MAX_POLL_INTERVAL,
    DEFAULT_MAX_REQUESTS,
    DEFAULT_MAX_RESUBMISSIONS,
    DEFAULT_POLL_INTERVAL,
)

from .abstract import AbstractCapability


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
    agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[batching])
    ```

    A batch is submitted when it holds `max_requests` requests, when its oldest request has waited `max_wait`, or when
    [`flush()`][pydantic_ai.capabilities.BatchModelRequests.flush] is called, whichever comes first.

    Meanwhile the run's message history ends in a suspended response that records the batch, so a run resumed from
    that history, in this process or another, waits for the same batch instead of submitting the request again. To
    end the run instead of waiting, add [`PendingModelResponse`][pydantic_ai.output.PendingModelResponse] to the
    agent's `output_type`. See [Batch processing](../batch.md) for a guide.

    A request that expired before the provider ran it, is missing from its batch's results, or failed with a rate
    limit or server error, is submitted again in a new batch, up to `max_resubmissions` times, as a provider SDK
    retries a live request. Other errors, like an invalid request or a canceled batch, reach the run as usual.

    The capability wraps whichever model serves each request, which must have a batch API. Requests can't be
    streamed, and batching needs an `asyncio` event loop. Submissions and checks run as tasks shared by the requests
    waiting on them; using the capability as an async context manager cancels any that are still running when it
    exits.
    """

    _: KW_ONLY

    max_requests: int = DEFAULT_MAX_REQUESTS
    """The most requests one batch holds. The provider's own limit caps it."""

    max_wait: timedelta | None = timedelta(seconds=30)
    """How long the oldest queued request waits for others before its batch is submitted.

    Later requests never extend it. `None` waits until the batch is full or
    [`flush()`][pydantic_ai.capabilities.BatchModelRequests.flush] is called.
    """

    max_resubmissions: int = DEFAULT_MAX_RESUBMISSIONS
    """How many times a request is submitted again in a new batch after it expired or failed with a rate limit,
    timeout or server error. `0` raises the error instead. A canceled request is never submitted again.
    """

    poll_interval: timedelta = DEFAULT_POLL_INTERVAL
    """How long to wait before first checking a submitted batch. The interval grows by half after every check.

    Checks of a batch made within `poll_interval` of each other, like those of runs resumed one after another from
    their message histories, share one status request.
    """

    max_poll_interval: timedelta = DEFAULT_MAX_POLL_INTERVAL
    """The longest wait between checks of a submitted batch."""

    _pool: BatchPool = field(init=False, repr=False, compare=False)
    _entered: int = field(default=0, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._pool = BatchPool(
            max_requests=self.max_requests,
            max_wait=self.max_wait,
            max_resubmissions=self.max_resubmissions,
            poll_interval=self.poll_interval,
            max_poll_interval=self.max_poll_interval,
        )

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None

    def get_wrapper_model(self, model: Model) -> Model | None:
        if isinstance(model, BatchedModel):
            # Already batched, by this capability or another one.
            return None
        return self._pool.wrap(model)

    async def flush(self) -> None:
        """Submit every queued request now instead of waiting for a full batch or `max_wait`."""
        await self._pool.flush()

    async def __aenter__(self) -> Self:
        self._entered += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Once the outermost `async with` exits, fail the queued requests and cancel the running submissions and checks."""
        self._entered -= 1
        if self._entered == 0:
            await self._pool.close()
