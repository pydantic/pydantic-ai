"""Retries utilities based on tenacity, especially for HTTP requests.

This module provides HTTP transport wrappers and wait strategies that integrate with
the tenacity library to add retry capabilities to HTTP requests. The transports can be
used with HTTP clients that support custom transports (such as httpx), while the wait
strategies can be used with any tenacity retry decorator.

The module includes:
- TenacityTransport: Synchronous HTTP transport with retry capabilities
- AsyncTenacityTransport: Asynchronous HTTP transport with retry capabilities
- wait_retry_after: Wait strategy that respects HTTP Retry-After headers
"""

from __future__ import annotations

from dataclasses import dataclass

from httpx import AsyncBaseTransport, AsyncHTTPTransport, BaseTransport, HTTPTransport, Request, Response
from pydantic_core import PydanticUndefinedType as Undefined

try:
    from tenacity import AsyncRetrying, Retrying, WrappedFn
except ImportError as _import_error:
    raise ImportError(
        'Please install `tenacity` to use the retries utilities, '
        'you can use the `retries` optional group — `pip install "pydantic-ai-slim[retries]"`'
    ) from _import_error

from collections.abc import Awaitable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any, Callable, cast

from httpx import HTTPStatusError
from tenacity import RetryCallState, RetryError, retry, wait_exponential

if TYPE_CHECKING:
    from tenacity.asyncio.retry import RetryBaseT
    from tenacity.retry import RetryBaseT as SyncRetryBaseT
    from tenacity.stop import StopBaseT
    from tenacity.wait import WaitBaseT

__all__ = ['RetryConfig', 'TenacityTransport', 'AsyncTenacityTransport', 'wait_retry_after']

UNDEFINED = Undefined()


@dataclass
class RetryConfig:
    """These are the arguments to the tenacity retry function and AsyncRetrying/Retrying classes."""

    # The following arguments cannot be None in tenacity but have private default values, so we use None as a sentinel
    sleep: Callable[[int | float], None | Awaitable[None]] | None = None
    stop: StopBaseT | None = None
    wait: WaitBaseT | None = None
    retry: SyncRetryBaseT | RetryBaseT | None = None
    before: Callable[[RetryCallState], None | Awaitable[None]] | None = None
    after: Callable[[RetryCallState], None | Awaitable[None]] | None = None

    # The following have public types and default values in tenacity, so we just repeat them verbatim here
    before_sleep: Callable[[RetryCallState], None | Awaitable[None]] | None = None
    reraise: bool = False
    retry_error_cls: type[RetryError] = RetryError
    retry_error_callback: Callable[[RetryCallState], Any | Awaitable[Any]] | None = None

    def tenacity_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            'before_sleep': self.before_sleep,
            'reraise': self.reraise,
            'retry_error_cls': self.retry_error_cls,
            'retry_error_callback': self.retry_error_callback,
        }
        if self.sleep is not None:
            kwargs['sleep'] = self.sleep
        if self.stop is not None:
            kwargs['stop'] = self.stop
        if self.wait is not None:
            kwargs['wait'] = self.wait
        if self.retry is not None:
            kwargs['retry'] = self.retry
        if self.before is not None:
            kwargs['before'] = self.before
        if self.after is not None:
            kwargs['after'] = self.after

        return kwargs

    def tenacity_decorator(self, function: WrappedFn) -> WrappedFn:
        """Wrap the provided function using this config to populate the tenacity `retry` decorator.

        Returns:
            A wrapped version of the function that will use this configuration for tenacity-based retrying when called.
        """
        return retry(**self.tenacity_kwargs())(function)


class TenacityTransport(BaseTransport):
    """Synchronous HTTP transport with tenacity-based retry functionality.

    This transport wraps another BaseTransport and adds retry capabilities using the tenacity library.
    It can be configured to retry requests based on various conditions such as specific exception types,
    response status codes, or custom validation logic.

    The transport works by intercepting HTTP requests and responses, allowing the tenacity controller
    to determine when and how to retry failed requests. The validate_response function can be used
    to convert HTTP responses into exceptions that trigger retries.

    Args:
        wrapped: The underlying transport to wrap and add retry functionality to.
        controller: The tenacity Retrying instance that defines the retry behavior
                   (retry conditions, wait strategy, stop conditions, etc.).
        validate_response: Optional callable that takes a Response and can raise an exception
            to be handled by the controller if the response should trigger a retry.
            Common use case is to raise exceptions for certain HTTP status codes.
            If None, no response validation is performed.

    Example:
        ```python
        from httpx import Client, HTTPTransport, HTTPStatusError
        from tenacity import Retrying, stop_after_attempt, retry_if_exception_type
        from pydantic_ai.retries import TenacityTransport, wait_retry_after

        transport = TenacityTransport(
            HTTPTransport(),
            Retrying(
                retry=retry_if_exception_type(HTTPStatusError),
                wait=wait_retry_after(max_wait=300),
                stop=stop_after_attempt(5),
                reraise=True
            ),
            validate_response=lambda r: r.raise_for_status()
        )
        client = Client(transport=transport)
        ```
    """

    def __init__(
        self,
        controller: RetryConfig | Retrying,
        wrapped: BaseTransport | None = None,
        validate_response: Callable[[Response], None] | None = None,
    ):
        self.controller = controller
        self.wrapped = wrapped or HTTPTransport()
        self.validate_response = validate_response

    def handle_request(self, request: Request) -> Response:
        """Handle an HTTP request with retry logic.

        Args:
            request: The HTTP request to handle.

        Returns:
            The HTTP response.

        Raises:
            RuntimeError: If the retry controller did not make any attempts.
            Exception: Any exception raised by the wrapped transport or validation function.
        """
        controller = (
            self.controller if isinstance(self.controller, Retrying) else Retrying(**self.controller.tenacity_kwargs())
        )
        for attempt in controller:
            with attempt:
                response = self.wrapped.handle_request(request)
                if self.validate_response:
                    self.validate_response(response)
                return response
        raise RuntimeError('The retry controller did not make any attempts')  # pragma: no cover


class AsyncTenacityTransport(AsyncBaseTransport):
    """Asynchronous HTTP transport with tenacity-based retry functionality.

    This transport wraps another AsyncBaseTransport and adds retry capabilities using the tenacity library.
    It can be configured to retry requests based on various conditions such as specific exception types,
    response status codes, or custom validation logic.

    The transport works by intercepting HTTP requests and responses, allowing the tenacity controller
    to determine when and how to retry failed requests. The validate_response function can be used
    to convert HTTP responses into exceptions that trigger retries.

    Args:
        wrapped: The underlying async transport to wrap and add retry functionality to.
        controller: The tenacity AsyncRetrying instance that defines the retry behavior
                   (retry conditions, wait strategy, stop conditions, etc.).
        validate_response: Optional callable that takes a Response and can raise an exception
            to be handled by the controller if the response should trigger a retry.
            Common use case is to raise exceptions for certain HTTP status codes.
            If None, no response validation is performed.

    Example:
        ```python
        from httpx import AsyncClient, HTTPStatusError
        from tenacity import AsyncRetrying, stop_after_attempt, retry_if_exception_type
        from pydantic_ai.retries import AsyncTenacityTransport, wait_retry_after

        transport = AsyncTenacityTransport(
            AsyncRetrying(
                retry=retry_if_exception_type(HTTPStatusError),
                wait=wait_retry_after(max_wait=300),
                stop=stop_after_attempt(5),
                reraise=True
            ),
            validate_response=lambda r: r.raise_for_status()
        )
        client = AsyncClient(transport=transport)
        ```
    """

    def __init__(
        self,
        controller: RetryConfig | AsyncRetrying,
        wrapped: AsyncBaseTransport | None = None,
        validate_response: Callable[[Response], None] | None = None,
    ):
        self.controller = controller
        self.wrapped = wrapped or AsyncHTTPTransport()
        self.validate_response = validate_response

    async def handle_async_request(self, request: Request) -> Response:
        """Handle an async HTTP request with retry logic.

        Args:
            request: The HTTP request to handle.

        Returns:
            The HTTP response.

        Raises:
            RuntimeError: If the retry controller did not make any attempts.
            Exception: Any exception raised by the wrapped transport or validation function.
        """
        controller = (
            self.controller
            if isinstance(self.controller, AsyncRetrying)
            else AsyncRetrying(**self.controller.tenacity_kwargs())
        )
        async for attempt in controller:
            with attempt:
                response = await self.wrapped.handle_async_request(request)
                if self.validate_response:
                    self.validate_response(response)
                return response
        raise RuntimeError('The retry controller did not make any attempts')  # pragma: no cover


def wait_retry_after(
    fallback_strategy: Callable[[RetryCallState], float] | None = None, max_wait: float = 300
) -> Callable[[RetryCallState], float]:
    """Create a tenacity-compatible wait strategy that respects HTTP Retry-After headers.

    This wait strategy checks if the exception contains an HTTPStatusError with a
    Retry-After header, and if so, waits for the time specified in the header.
    If no header is present or parsing fails, it falls back to the provided strategy.

    The Retry-After header can be in two formats:
    - An integer representing seconds to wait
    - An HTTP date string representing when to retry

    Args:
        fallback_strategy: Wait strategy to use when no Retry-After header is present
                          or parsing fails. Defaults to exponential backoff with max 60s.
        max_wait: Maximum time to wait in seconds, regardless of header value.
                 Defaults to 300 (5 minutes).

    Returns:
        A wait function that can be used with tenacity retry decorators.

    Example:
        ```python
        from httpx import AsyncClient, HTTPStatusError
        from tenacity import AsyncRetrying, stop_after_attempt, retry_if_exception_type
        from pydantic_ai.retries import AsyncTenacityTransport, wait_retry_after

        transport = AsyncTenacityTransport(
            AsyncRetrying(
                retry=retry_if_exception_type(HTTPStatusError),
                wait=wait_retry_after(max_wait=120),
                stop=stop_after_attempt(5),
                reraise=True
            ),
            validate_response=lambda r: r.raise_for_status()
        )
        client = AsyncClient(transport=transport)
        ```
    """
    if fallback_strategy is None:
        fallback_strategy = wait_exponential(multiplier=1, max=60)

    def wait_func(state: RetryCallState) -> float:
        exc = state.outcome.exception() if state.outcome else None
        if isinstance(exc, HTTPStatusError):
            retry_after = exc.response.headers.get('retry-after')
            if retry_after:
                try:
                    # Try parsing as seconds first
                    wait_seconds = int(retry_after)
                    return min(float(wait_seconds), max_wait)
                except ValueError:
                    # Try parsing as HTTP date
                    try:
                        retry_time = cast(datetime, parsedate_to_datetime(retry_after))
                        assert isinstance(retry_time, datetime)
                        now = datetime.now(timezone.utc)
                        wait_seconds = (retry_time - now).total_seconds()

                        if wait_seconds > 0:
                            return min(wait_seconds, max_wait)
                    except (ValueError, TypeError, AssertionError):
                        # If date parsing fails, fall back to fallback strategy
                        pass

        # Use fallback strategy
        return fallback_strategy(state)

    return wait_func
