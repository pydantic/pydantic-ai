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


ConcurrencyLimit: TypeAlias = 'int | ConcurrencyLimiter | None'
"""Type alias for concurrency limit configuration.

Can be:
- An `int`: Simple limit on concurrent operations (unlimited queue).
- A `ConcurrencyLimiter`: Full configuration with optional backpressure.
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
        tracer: Tracer | None = None,
    ):
        """Initialize the RecordingSemaphore.

        Args:
            max_running: Maximum number of concurrent operations.
            max_queued: Maximum queue depth before raising ConcurrencyLimitExceeded.
            tracer: OpenTelemetry tracer for span creation.
        """
        self._semaphore = anyio.Semaphore(max_running)
        self._max_running = max_running
        self._max_queued = max_queued
        self._waiting_count = 0
        self._tracer = tracer

    @classmethod
    def from_limit(
        cls,
        limit: int | ConcurrencyLimiter,
        *,
        tracer: Tracer | None = None,
    ) -> Self:
        """Create a RecordingSemaphore from a ConcurrencyLimit configuration.

        Args:
            limit: Either an int for simple limiting or a ConcurrencyLimiter for full config.
            tracer: OpenTelemetry tracer for span creation.

        Returns:
            A configured RecordingSemaphore.
        """
        if isinstance(limit, int):
            return cls(max_running=limit, tracer=tracer)
        else:
            return cls(
                max_running=limit.max_running,
                max_queued=limit.max_queued,
                tracer=tracer,
            )

    @property
    def waiting_count(self) -> int:
        """Number of operations currently waiting to acquire the semaphore."""
        return self._waiting_count

    def _get_tracer(self) -> Tracer:
        """Get the tracer, falling back to global tracer if not set."""
        if self._tracer is not None:
            return self._tracer
        return get_tracer('pydantic-ai')

    async def acquire(self, name: str) -> None:
        """Acquire the semaphore, creating a span if waiting is required.

        Args:
            name: Identifier for observability (e.g., 'agent:my-agent' or 'model:gpt-4').
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
            if self._max_queued is not None and self._waiting_count > self._max_queued:
                raise ConcurrencyLimitExceeded(
                    f'Concurrency queue depth ({self._waiting_count}) exceeds max_queued ({self._max_queued})'
                    + (f' for {name}' if name else '')
                )

            # Create a span for observability while waiting
            tracer = self._get_tracer()
            attributes: dict[str, str | int] = {
                'limiter_name': name,
                'waiting_count': self._waiting_count,
                'max_running': self._max_running,
            }
            if self._max_queued is not None:
                attributes['max_queued'] = self._max_queued
            with tracer.start_as_current_span(
                f'waiting for {name} concurrency',
                attributes=attributes,
            ):
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
async def _semaphore_context(semaphore: RecordingSemaphore, name: str) -> AsyncIterator[None]:
    """Context manager that acquires and releases a semaphore with the given name."""
    await semaphore.acquire(name)
    try:
        yield
    finally:
        semaphore.release()


def get_concurrency_context(
    limiter: RecordingSemaphore | None,
    name: str = 'unnamed',
) -> AbstractAsyncContextManager[None]:
    """Get an async context manager for the concurrency limiter.

    If limiter is None, returns a no-op context manager.

    Args:
        limiter: The RecordingSemaphore or None.
        name: Identifier for observability (e.g., 'agent:my-agent' or 'model:gpt-4').

    Returns:
        An async context manager.
    """
    if limiter is None:
        return _null_context()
    return _semaphore_context(limiter, name)
