"""Concurrency limiting infrastructure with OpenTelemetry observability."""

from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import TypeAlias

import anyio
from opentelemetry.trace import Tracer, get_tracer
from typing_extensions import Self

__all__ = (
    'AbstractConcurrencyLimiter',
    'ConcurrencyLimiter',
    'ConcurrencyLimit',
    'AnyConcurrencyLimit',
)


class AbstractConcurrencyLimiter(ABC):
    """Abstract base class for concurrency limiters.

    Subclass this to create custom concurrency limiters
    (e.g., Redis-backed distributed limiters).

    Example:
    ```python
    from pydantic_ai.concurrency import AbstractConcurrencyLimiter


    class RedisConcurrencyLimiter(AbstractConcurrencyLimiter):
        def __init__(self, redis_client, key: str, max_running: int):
            self._redis = redis_client
            self._key = key
            self._max_running = max_running

        async def acquire(self, source: str) -> None:
            # Implement Redis-based distributed locking
            ...

        def release(self) -> None:
            # Release the Redis lock
            ...
    ```
    """

    @abstractmethod
    async def acquire(self, source: str) -> None:
        """Acquire a slot, waiting if necessary.

        Args:
            source: Identifier for observability (e.g., 'model:gpt-4o').
        """
        ...

    @abstractmethod
    def release(self) -> None:
        """Release a slot."""
        ...


@dataclass
class ConcurrencyLimit:
    """Configuration for concurrency limiting with optional backpressure.

    Args:
        max_running: Maximum number of concurrent operations allowed.
        max_queued: Maximum number of operations waiting in the queue.
            If None, the queue is unlimited. If exceeded, raises `ConcurrencyLimitExceeded`.
    """

    max_running: int
    max_queued: int | None = None


class ConcurrencyLimiter(AbstractConcurrencyLimiter):
    """A concurrency limiter that tracks waiting operations for observability.

    This class wraps an anyio.CapacityLimiter and tracks the number of waiting operations.
    When an operation has to wait to acquire a slot, a span is created for
    observability purposes.
    """

    def __init__(
        self,
        max_running: int,
        *,
        max_queued: int | None = None,
        name: str | None = None,
        tracer: Tracer | None = None,
    ):
        """Initialize the ConcurrencyLimiter.

        Args:
            max_running: Maximum number of concurrent operations.
            max_queued: Maximum queue depth before raising ConcurrencyLimitExceeded.
            name: Optional name for this limiter, used for observability when sharing
                a limiter across multiple models or agents.
            tracer: OpenTelemetry tracer for span creation.
        """
        self._limiter = anyio.CapacityLimiter(max_running)
        self._max_queued = max_queued
        self._name = name
        self._tracer = tracer
        # Lock and counter to atomically check and track waiting tasks for max_queued enforcement
        self._queue_lock = anyio.Lock()
        self._waiting_count = 0

    @classmethod
    def from_limit(
        cls,
        limit: int | ConcurrencyLimit,
        *,
        name: str | None = None,
        tracer: Tracer | None = None,
    ) -> Self:
        """Create a ConcurrencyLimiter from a ConcurrencyLimit configuration.

        Args:
            limit: Either an int for simple limiting or a ConcurrencyLimit for full config.
            name: Optional name for this limiter, used for observability.
            tracer: OpenTelemetry tracer for span creation.

        Returns:
            A configured ConcurrencyLimiter.
        """
        if isinstance(limit, int):
            return cls(max_running=limit, name=name, tracer=tracer)
        else:
            return cls(
                max_running=limit.max_running,
                max_queued=limit.max_queued,
                name=name,
                tracer=tracer,
            )

    @property
    def name(self) -> str | None:
        """Name of the limiter for observability."""
        return self._name

    @property
    def waiting_count(self) -> int:
        """Number of operations currently waiting to acquire a slot."""
        return self._waiting_count

    @property
    def running_count(self) -> int:
        """Number of operations currently running."""
        return self._limiter.statistics().borrowed_tokens

    @property
    def available_count(self) -> int:
        """Number of slots available."""
        return int(self._limiter.available_tokens)

    @property
    def max_running(self) -> int:
        """Maximum concurrent operations allowed."""
        return int(self._limiter.total_tokens)

    def _get_tracer(self) -> Tracer:
        """Get the tracer, falling back to global tracer if not set."""
        if self._tracer is not None:
            return self._tracer
        return get_tracer('pydantic-ai')

    async def acquire(self, source: str) -> None:
        """Acquire a slot, creating a span if waiting is required.

        Args:
            source: Identifier for the source of this acquisition (e.g., 'agent:my-agent' or 'model:gpt-4').
        """
        from .exceptions import ConcurrencyLimitExceeded

        # Try to acquire immediately without blocking
        try:
            self._limiter.acquire_nowait()
            return
        except anyio.WouldBlock:
            pass

        # We need to wait - atomically check queue limits and register ourselves as waiting
        # This prevents a race condition where multiple tasks could pass the check before
        # any of them actually start waiting on the limiter
        async with self._queue_lock:
            if self._max_queued is not None and self._waiting_count >= self._max_queued:
                # Use limiter name if set, otherwise use source for error messages
                display_name = self._name or source
                raise ConcurrencyLimitExceeded(
                    f'Concurrency queue depth ({self._waiting_count + 1}) exceeds max_queued ({self._max_queued})'
                    + (f' for {display_name}' if display_name else '')
                )
            # Register ourselves as waiting before releasing the lock
            self._waiting_count += 1

        # Now we're registered as waiting, proceed to wait on the limiter
        # Use try/finally to ensure we decrement the counter even on cancellation
        try:
            # Create a span for observability while waiting
            tracer = self._get_tracer()
            display_name = self._name or source
            attributes: dict[str, str | int] = {
                'source': source,
                'waiting_count': self._waiting_count,
                'max_running': int(self._limiter.total_tokens),
            }
            if self._name is not None:
                attributes['limiter_name'] = self._name
            if self._max_queued is not None:
                attributes['max_queued'] = self._max_queued

            # Span name uses limiter name if set, otherwise source
            span_name = f'waiting for {display_name} concurrency'
            with tracer.start_as_current_span(span_name, attributes=attributes):
                await self._limiter.acquire()
        finally:
            # We're no longer waiting (either we acquired or we were cancelled)
            self._waiting_count -= 1

    def release(self) -> None:
        """Release a slot."""
        self._limiter.release()


AnyConcurrencyLimit: TypeAlias = 'int | ConcurrencyLimit | AbstractConcurrencyLimiter | None'
"""Type alias for concurrency limit configuration.

Can be:
- An `int`: Simple limit on concurrent operations (unlimited queue).
- A `ConcurrencyLimit`: Full configuration with optional backpressure.
- An `AbstractConcurrencyLimiter`: A pre-created limiter instance for sharing across multiple models/agents.
- `None`: No concurrency limiting (default).
"""


@asynccontextmanager
async def _null_context() -> AsyncIterator[None]:
    """A no-op async context manager."""
    yield


@asynccontextmanager
async def _limiter_context(limiter: AbstractConcurrencyLimiter, source: str) -> AsyncIterator[None]:
    """Context manager that acquires and releases a limiter with the given source."""
    await limiter.acquire(source)
    try:
        yield
    finally:
        limiter.release()


def get_concurrency_context(
    limiter: AbstractConcurrencyLimiter | None,
    source: str = 'unnamed',
) -> AbstractAsyncContextManager[None]:
    """Get an async context manager for the concurrency limiter.

    If limiter is None, returns a no-op context manager.

    Args:
        limiter: The AbstractConcurrencyLimiter or None.
        source: Identifier for the source of this acquisition (e.g., 'agent:my-agent' or 'model:gpt-4').

    Returns:
        An async context manager.
    """
    if limiter is None:
        return _null_context()
    return _limiter_context(limiter, source)


def normalize_to_limiter(
    limit: AnyConcurrencyLimit,
    *,
    name: str | None = None,
) -> AbstractConcurrencyLimiter | None:
    """Normalize a concurrency limit configuration to an AbstractConcurrencyLimiter.

    Args:
        limit: The concurrency limit configuration.
        name: Optional name for the limiter if one is created.

    Returns:
        An AbstractConcurrencyLimiter if limit is not None, otherwise None.
    """
    if limit is None:
        return None
    elif isinstance(limit, AbstractConcurrencyLimiter):
        return limit
    else:
        return ConcurrencyLimiter.from_limit(limit, name=name)
