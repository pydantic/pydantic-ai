"""Concurrency limiting infrastructure with OpenTelemetry observability."""

from __future__ import annotations as _annotations

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import TypeAlias

import anyio
from opentelemetry.trace import Tracer, get_tracer
from typing_extensions import Self

__all__ = ('ConcurrencyLimiter', 'ConcurrencyLimit', 'RecordingSemaphore')


@dataclass
class ConcurrencyLimiter:
    """Configuration for concurrency limiting with optional backpressure.

    Args:
        max_running: Maximum number of concurrent operations allowed.
        max_queued: Maximum number of operations waiting in the queue.
            If None, the queue is unlimited. If exceeded, raises `ConcurrencyLimitExceeded`.
    """

    max_running: int
    max_queued: int | None = None


ConcurrencyLimit: TypeAlias = 'int | ConcurrencyLimiter | RecordingSemaphore | None'
"""Type alias for concurrency limit configuration.

Can be:
- An `int`: Simple limit on concurrent operations (unlimited queue).
- A `ConcurrencyLimiter`: Full configuration with optional backpressure.
- A `RecordingSemaphore`: A pre-created semaphore instance for sharing across multiple models/agents.
- `None`: No concurrency limiting (default).
"""


class RecordingSemaphore:
    """A semaphore wrapper that records waiting operations for observability.

    This class wraps an anyio.Semaphore and tracks the number of waiting operations.
    When an operation has to wait to acquire the semaphore, a span is created for
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
        """Initialize the RecordingSemaphore.

        Args:
            max_running: Maximum number of concurrent operations.
            max_queued: Maximum queue depth before raising ConcurrencyLimitExceeded.
            name: Optional name for this semaphore, used for observability when sharing
                a semaphore across multiple models or agents.
            tracer: OpenTelemetry tracer for span creation.
        """
        self._semaphore = anyio.Semaphore(max_running)
        self._max_running = max_running
        self._max_queued = max_queued
        self._name = name
        self._waiting_count = 0
        self._tracer = tracer

    @classmethod
    def from_limit(
        cls,
        limit: int | ConcurrencyLimiter,
        *,
        name: str | None = None,
        tracer: Tracer | None = None,
    ) -> Self:
        """Create a RecordingSemaphore from a ConcurrencyLimit configuration.

        Args:
            limit: Either an int for simple limiting or a ConcurrencyLimiter for full config.
            name: Optional name for this semaphore, used for observability.
            tracer: OpenTelemetry tracer for span creation.

        Returns:
            A configured RecordingSemaphore.
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
        """Name of the semaphore for observability."""
        return self._name

    @property
    def waiting_count(self) -> int:
        """Number of operations currently waiting to acquire the semaphore."""
        return self._waiting_count

    def _get_tracer(self) -> Tracer:
        """Get the tracer, falling back to global tracer if not set."""
        if self._tracer is not None:
            return self._tracer
        return get_tracer('pydantic-ai')

    async def acquire(self, source: str) -> None:
        """Acquire the semaphore, creating a span if waiting is required.

        Args:
            source: Identifier for the source of this acquisition (e.g., 'agent:my-agent' or 'model:gpt-4').
        """
        from .exceptions import ConcurrencyLimitExceeded

        # Try to acquire immediately without blocking
        try:
            self._semaphore.acquire_nowait()
            return
        except anyio.WouldBlock:
            pass

        # We need to wait - track and check queue limits
        self._waiting_count += 1
        try:
            # Use semaphore name if set, otherwise use source for error messages
            display_name = self._name or source
            if self._max_queued is not None and self._waiting_count > self._max_queued:
                raise ConcurrencyLimitExceeded(
                    f'Concurrency queue depth ({self._waiting_count}) exceeds max_queued ({self._max_queued})'
                    + (f' for {display_name}' if display_name else '')
                )

            # Create a span for observability while waiting
            tracer = self._get_tracer()
            attributes: dict[str, str | int] = {
                'source': source,
                'waiting_count': self._waiting_count,
                'max_running': self._max_running,
            }
            if self._name is not None:
                attributes['limiter_name'] = self._name
            if self._max_queued is not None:
                attributes['max_queued'] = self._max_queued

            # Span name uses semaphore name if set, otherwise source
            span_name = f'waiting for {display_name} concurrency'
            with tracer.start_as_current_span(span_name, attributes=attributes):
                await self._semaphore.acquire()
        finally:
            self._waiting_count -= 1

    def release(self) -> None:
        """Release the semaphore."""
        self._semaphore.release()


@asynccontextmanager
async def _null_context() -> AsyncIterator[None]:
    """A no-op async context manager."""
    yield


@asynccontextmanager
async def _semaphore_context(semaphore: RecordingSemaphore, source: str) -> AsyncIterator[None]:
    """Context manager that acquires and releases a semaphore with the given source."""
    await semaphore.acquire(source)
    try:
        yield
    finally:
        semaphore.release()


def get_concurrency_context(
    limiter: RecordingSemaphore | None,
    source: str = 'unnamed',
) -> AbstractAsyncContextManager[None]:
    """Get an async context manager for the concurrency limiter.

    If limiter is None, returns a no-op context manager.

    Args:
        limiter: The RecordingSemaphore or None.
        source: Identifier for the source of this acquisition (e.g., 'agent:my-agent' or 'model:gpt-4').

    Returns:
        An async context manager.
    """
    if limiter is None:
        return _null_context()
    return _semaphore_context(limiter, source)


def normalize_to_semaphore(
    limit: ConcurrencyLimit,
    *,
    name: str | None = None,
) -> RecordingSemaphore | None:
    """Normalize a concurrency limit configuration to a RecordingSemaphore.

    Args:
        limit: The concurrency limit configuration.
        name: Optional name for the semaphore if one is created.

    Returns:
        A RecordingSemaphore if limit is not None, otherwise None.
    """
    if limit is None:
        return None
    elif isinstance(limit, RecordingSemaphore):
        return limit
    else:
        return RecordingSemaphore.from_limit(limit, name=name)
