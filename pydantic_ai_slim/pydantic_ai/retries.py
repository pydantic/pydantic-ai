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

from types import TracebackType

from httpx import (
    AsyncBaseTransport,
    AsyncHTTPTransport,
    BaseTransport,
    HTTPStatusError,
    HTTPTransport,
    Request,
    Response,
)

try:
    from tenacity import RetryCallState, RetryError, retry, wait_exponential
except ImportError as _import_error:
    raise ImportError(
        'Please install `tenacity` to use the retries utilities, '
        'you can use the `retries` optional group — `pip install "pydantic-ai-slim[retries]"`'
    ) from _import_error

from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from tenacity.asyncio.retry import RetryBaseT
    from tenacity.retry import RetryBaseT as SyncRetryBaseT
    from tenacity.stop import StopBaseT
    from tenacity.wait import WaitBaseT

__all__ = ['RetryConfig', 'TenacityTransport', 'AsyncTenacityTransport', 'wait_retry_after']


class RetryConfig(TypedDict, total=False):
    """The configuration for tenacity-based retrying.

    These are precisely the arguments to the tenacity `retry` decorator, and they are generally
    used internally by passing them to that decorator via `@retry(**config)` or similar.

    All fields are optional, and if not provided, the default values from the `tenacity.retry` decorator will be used.
    """

    sleep: Callable[[int | float], None | Awaitable[None]]
    """A sleep strategy to use for sleeping between retries.

    Tenacity's default for this argument is `tenacity.nap.sleep`."""

    stop: StopBaseT
    """
    A stop strategy to determine when to stop retrying.

    Tenacity's default for this argument is `tenacity.stop.stop_never`."""

    wait: WaitBaseT
    """
    A wait strategy to determine how long to wait between retries.

    Tenacity's default for this argument is `tenacity.wait.wait_none`."""

    retry: SyncRetryBaseT | RetryBaseT
    """A retry strategy to determine which exceptions should trigger a retry.

    Tenacity's default for this argument is `tenacity.retry.retry_if_exception_type()`."""

    before: Callable[[RetryCallState], None | Awaitable[None]]
    """
    A callable that is called before each retry attempt.

    Tenacity's default for this argument is `tenacity.before.before_nothing`."""

    after: Callable[[RetryCallState], None | Awaitable[None]]
    """
    A callable that is called after each retry attempt.

    Tenacity's default for this argument is `tenacity.after.after_nothing`."""

    before_sleep: Callable[[RetryCallState], None | Awaitable[None]] | None
    """
    An optional callable that is called before sleeping between retries.

    Tenacity's default for this argument is `None`."""

    reraise: bool
    """Whether to reraise the last exception if the retry attempts are exhausted, or raise a RetryError instead.

    Tenacity's default for this argument is `False`."""

    retry_error_cls: type[RetryError]
    """The exception class to raise when the retry attempts are exhausted and `reraise` is False.

    Tenacity's default for this argument is `tenacity.RetryError`."""

    retry_error_callback: Callable[[RetryCallState], Any | Awaitable[Any]] | None
    """An optional callable that is called when the retry attempts are exhausted and `reraise` is False.

    Tenacity's default for this argument is `None`."""


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
        config: The arguments to use for the tenacity `retry` decorator, including retry conditions,
            wait strategy, stop conditions, etc. See the tenacity docs for more info.
        validate_response: Optional callable that takes a Response and can raise an exception
            to be handled by the controller if the response should trigger a retry.
            Common use case is to raise exceptions for certain HTTP status codes.
            If None, no response validation is performed.

    Example:
        ```python
        from httpx import Client, HTTPStatusError, HTTPTransport
        from tenacity import retry_if_exception_type, stop_after_attempt

        from pydantic_ai.retries import RetryConfig, TenacityTransport, wait_retry_after

        transport = TenacityTransport(
            RetryConfig(
                retry=retry_if_exception_type(HTTPStatusError),
                wait=wait_retry_after(max_wait=300),
                stop=stop_after_attempt(5),
                reraise=True
            ),
            HTTPTransport(),
            validate_response=lambda r: r.raise_for_status()
        )
        client = Client(transport=transport)
        ```
    """

    def __init__(
        self,
        config: RetryConfig,
        wrapped: BaseTransport | None = None,
        validate_response: Callable[[Response], Any] | None = None,
    ):
        self.config = config
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

        @retry(**self.config)
        def handle_request(req: Request) -> Response:
            response = self.wrapped.handle_request(req)

            # this is normally set by httpx _after_ calling this function, but we want the request in the validator:
            response.request = req

            if self.validate_response:
                try:
                    self.validate_response(response)
                except Exception:
                    response.close()
                    raise
            return response

        return handle_request(request)

    def __enter__(self) -> TenacityTransport:
        self.wrapped.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        self.wrapped.__exit__(exc_type, exc_value, traceback)

    def close(self) -> None:
        self.wrapped.close()  # pragma: no cover


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
        config: The arguments to use for the tenacity `retry` decorator, including retry conditions,
            wait strategy, stop conditions, etc. See the tenacity docs for more info.
        validate_response: Optional callable that takes a Response and can raise an exception
            to be handled by the controller if the response should trigger a retry.
            Common use case is to raise exceptions for certain HTTP status codes.
            If None, no response validation is performed.

    Example:
        ```python
        from httpx import AsyncClient, HTTPStatusError
        from tenacity import retry_if_exception_type, stop_after_attempt

        from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after

        transport = AsyncTenacityTransport(
            RetryConfig(
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
        config: RetryConfig,
        wrapped: AsyncBaseTransport | None = None,
        validate_response: Callable[[Response], Any] | None = None,
    ):
        self.config = config
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

        @retry(**self.config)
        async def handle_async_request(req: Request) -> Response:
            response = await self.wrapped.handle_async_request(req)

            # this is normally set by httpx _after_ calling this function, but we want the request in the validator:
            response.request = req

            if self.validate_response:
                try:
                    self.validate_response(response)
                except Exception:
                    await response.aclose()
                    raise
            return response

        return await handle_async_request(request)

    async def __aenter__(self) -> AsyncTenacityTransport:
        await self.wrapped.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        await self.wrapped.__aexit__(exc_type, exc_value, traceback)

    async def aclose(self) -> None:
        await self.wrapped.aclose()


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
        from tenacity import retry_if_exception_type, stop_after_attempt

        from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after

        transport = AsyncTenacityTransport(
            RetryConfig(
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
