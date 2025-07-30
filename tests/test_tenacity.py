from __future__ import annotations as _annotations

import asyncio
import time
from datetime import datetime, timezone
from email.utils import formatdate
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from .conftest import try_import

with try_import() as imports_successful:
    from tenacity import (
        AsyncRetrying,
        RetryCallState,
        Retrying,
        retry_if_exception_type,
        stop_after_attempt,
        wait_fixed,
    )

    from pydantic_ai.retries import AsyncTenacityTransport, TenacityTransport, wait_retry_after

pytestmark = pytest.mark.skipif(not imports_successful(), reason='install tenacity to run tenacity tests')


class TestTenacityTransport:
    """Tests for the synchronous TenacityTransport."""

    def test_successful_request(self):
        """Test that successful requests pass through without retry."""
        mock_transport = Mock(spec=httpx.BaseTransport)
        mock_response = Mock(spec=httpx.Response)
        mock_transport.handle_request.return_value = mock_response

        controller = Retrying(stop=stop_after_attempt(3), reraise=True)
        transport = TenacityTransport(controller, mock_transport)

        request = httpx.Request('GET', 'https://example.com')
        result = transport.handle_request(request)

        assert result is mock_response
        mock_transport.handle_request.assert_called_once_with(request)

    def test_retry_on_exception(self):
        """Test that exceptions trigger retries."""
        mock_transport = Mock(spec=httpx.BaseTransport)
        mock_response = Mock(spec=httpx.Response)

        # Fail twice, succeed on third attempt
        mock_transport.handle_request.side_effect = [
            httpx.ConnectError('Connection failed'),
            httpx.ConnectError('Connection failed again'),
            mock_response,
        ]

        controller = Retrying(
            retry=retry_if_exception_type(httpx.ConnectError),
            stop=stop_after_attempt(3),
            wait=wait_fixed(0.001),  # Very short wait for tests
            reraise=True,
        )
        transport = TenacityTransport(controller, mock_transport)

        request = httpx.Request('GET', 'https://example.com')
        result = transport.handle_request(request)

        assert result is mock_response
        assert mock_transport.handle_request.call_count == 3

    def test_retry_exhausted(self):
        """Test that retry exhaustion re-raises the last exception."""
        mock_transport = Mock(spec=httpx.BaseTransport)
        mock_transport.handle_request.side_effect = httpx.ConnectError('Connection failed')

        controller = Retrying(
            retry=retry_if_exception_type(httpx.ConnectError),
            stop=stop_after_attempt(2),
            wait=wait_fixed(0.001),
            reraise=True,
        )
        transport = TenacityTransport(controller, mock_transport)

        request = httpx.Request('GET', 'https://example.com')
        with pytest.raises(httpx.ConnectError, match='Connection failed'):
            transport.handle_request(request)

        assert mock_transport.handle_request.call_count == 2

    def test_validate_response_success(self):
        """Test that validate_response is called and doesn't raise."""
        mock_transport = Mock(spec=httpx.BaseTransport)
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_transport.handle_request.return_value = mock_response

        validate_response = Mock()
        controller = Retrying(stop=stop_after_attempt(3), reraise=True)
        transport = TenacityTransport(controller, mock_transport, validate_response)

        request = httpx.Request('GET', 'https://example.com')
        result = transport.handle_request(request)

        assert result is mock_response
        validate_response.assert_called_once_with(mock_response)

    def test_validate_response_triggers_retry(self):
        """Test that validate_response can trigger retries."""
        mock_transport = Mock(spec=httpx.BaseTransport)
        mock_response_fail = Mock(spec=httpx.Response)
        mock_response_fail.status_code = 429
        mock_response_success = Mock(spec=httpx.Response)
        mock_response_success.status_code = 200

        mock_transport.handle_request.side_effect = [mock_response_fail, mock_response_success]

        def validate_response(response: httpx.Response):
            if response.status_code == 429:
                raise httpx.HTTPStatusError('Rate limited', request=request, response=response)

        controller = Retrying(
            retry=retry_if_exception_type(httpx.HTTPStatusError),
            stop=stop_after_attempt(3),
            wait=wait_fixed(0.001),
            reraise=True,
        )
        transport = TenacityTransport(controller, mock_transport, validate_response)

        request = httpx.Request('GET', 'https://example.com')
        result = transport.handle_request(request)

        assert result is mock_response_success
        assert mock_transport.handle_request.call_count == 2


class TestAsyncTenacityTransport:
    """Tests for the asynchronous AsyncTenacityTransport."""

    async def test_successful_request(self):
        """Test that successful requests pass through without retry."""
        mock_transport = AsyncMock(spec=httpx.AsyncBaseTransport)
        mock_response = Mock(spec=httpx.Response)
        mock_transport.handle_async_request.return_value = mock_response

        controller = AsyncRetrying(stop=stop_after_attempt(3), reraise=True)
        transport = AsyncTenacityTransport(controller, mock_transport)

        request = httpx.Request('GET', 'https://example.com')
        result = await transport.handle_async_request(request)

        assert result is mock_response
        mock_transport.handle_async_request.assert_called_once_with(request)

    async def test_retry_on_exception(self):
        """Test that exceptions trigger retries."""
        mock_transport = AsyncMock(spec=httpx.AsyncBaseTransport)
        mock_response = Mock(spec=httpx.Response)

        # Fail twice, succeed on third attempt
        mock_transport.handle_async_request.side_effect = [
            httpx.ConnectError('Connection failed'),
            httpx.ConnectError('Connection failed again'),
            mock_response,
        ]

        controller = AsyncRetrying(
            retry=retry_if_exception_type(httpx.ConnectError),
            stop=stop_after_attempt(3),
            wait=wait_fixed(0.001),
            reraise=True,
        )
        transport = AsyncTenacityTransport(controller, mock_transport)

        request = httpx.Request('GET', 'https://example.com')
        result = await transport.handle_async_request(request)

        assert result is mock_response
        assert mock_transport.handle_async_request.call_count == 3

    async def test_retry_exhausted(self):
        """Test that retry exhaustion re-raises the last exception."""
        mock_transport = AsyncMock(spec=httpx.AsyncBaseTransport)
        mock_transport.handle_async_request.side_effect = httpx.ConnectError('Connection failed')

        controller = AsyncRetrying(
            retry=retry_if_exception_type(httpx.ConnectError),
            stop=stop_after_attempt(2),
            wait=wait_fixed(0.001),
            reraise=True,
        )
        transport = AsyncTenacityTransport(controller, mock_transport)

        request = httpx.Request('GET', 'https://example.com')
        with pytest.raises(httpx.ConnectError, match='Connection failed'):
            await transport.handle_async_request(request)

        assert mock_transport.handle_async_request.call_count == 2

    async def test_validate_response_success(self):
        """Test that validate_response is called and doesn't raise."""
        mock_transport = AsyncMock(spec=httpx.AsyncBaseTransport)
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_transport.handle_async_request.return_value = mock_response

        validate_response = Mock()
        controller = AsyncRetrying(stop=stop_after_attempt(3), reraise=True)
        transport = AsyncTenacityTransport(controller, mock_transport, validate_response)

        request = httpx.Request('GET', 'https://example.com')
        result = await transport.handle_async_request(request)

        assert result is mock_response
        validate_response.assert_called_once_with(mock_response)

    async def test_validate_response_triggers_retry(self):
        """Test that validate_response can trigger retries."""
        mock_transport = AsyncMock(spec=httpx.AsyncBaseTransport)
        mock_response_fail = Mock(spec=httpx.Response)
        mock_response_fail.status_code = 429
        mock_response_success = Mock(spec=httpx.Response)
        mock_response_success.status_code = 200

        mock_transport.handle_async_request.side_effect = [mock_response_fail, mock_response_success]

        def validate_response(response: httpx.Response):
            if response.status_code == 429:
                raise httpx.HTTPStatusError('Rate limited', request=request, response=response)

        controller = AsyncRetrying(
            retry=retry_if_exception_type(httpx.HTTPStatusError),
            stop=stop_after_attempt(3),
            wait=wait_fixed(0.001),
            reraise=True,
        )
        transport = AsyncTenacityTransport(controller, mock_transport, validate_response)

        request = httpx.Request('GET', 'https://example.com')
        result = await transport.handle_async_request(request)

        assert result is mock_response_success
        assert mock_transport.handle_async_request.call_count == 2


class TestWaitRetryAfter:
    """Tests for the wait_retry_after wait strategy."""

    def test_no_exception_uses_fallback(self):
        """Test that fallback strategy is used when there's no exception."""
        fallback = Mock(return_value=5.0)
        wait_func = wait_retry_after(fallback_strategy=fallback, max_wait=300)

        # Create a retry state with no exception
        retry_state = Mock(spec=RetryCallState)
        retry_state.outcome = None

        result = wait_func(retry_state)

        assert result == 5.0
        fallback.assert_called_once_with(retry_state)

    def test_non_http_exception_uses_fallback(self):
        """Test that fallback strategy is used for non-HTTP exceptions."""
        fallback = Mock(return_value=3.0)
        wait_func = wait_retry_after(fallback_strategy=fallback, max_wait=300)

        # Create a retry state with a non-HTTP exception
        retry_state = Mock(spec=RetryCallState)
        retry_state.outcome = Mock()
        retry_state.outcome.failed = True
        retry_state.outcome.exception.return_value = ValueError('Some error')

        result = wait_func(retry_state)

        assert result == 3.0
        fallback.assert_called_once_with(retry_state)

    def test_http_exception_no_retry_after_uses_fallback(self):
        """Test that fallback strategy is used when there's no Retry-After header."""
        fallback = Mock(return_value=2.0)
        wait_func = wait_retry_after(fallback_strategy=fallback, max_wait=300)

        # Create HTTP status error without Retry-After header
        request = httpx.Request('GET', 'https://example.com')
        response = Mock(spec=httpx.Response)
        response.headers = {}
        http_error = httpx.HTTPStatusError('Rate limited', request=request, response=response)

        retry_state = Mock(spec=RetryCallState)
        retry_state.outcome = Mock()
        retry_state.outcome.failed = True
        retry_state.outcome.exception.return_value = http_error

        result = wait_func(retry_state)

        assert result == 2.0
        fallback.assert_called_once_with(retry_state)

    def test_retry_after_seconds_format(self):
        """Test parsing Retry-After header in seconds format."""
        fallback = Mock()
        wait_func = wait_retry_after(fallback_strategy=fallback, max_wait=300)

        # Create HTTP status error with Retry-After in seconds
        request = httpx.Request('GET', 'https://example.com')
        response = Mock(spec=httpx.Response)
        response.headers = {'retry-after': '30'}
        http_error = httpx.HTTPStatusError('Rate limited', request=request, response=response)

        retry_state = Mock(spec=RetryCallState)
        retry_state.outcome = Mock()
        retry_state.outcome.failed = True
        retry_state.outcome.exception.return_value = http_error

        result = wait_func(retry_state)

        assert result == 30.0
        fallback.assert_not_called()

    def test_retry_after_seconds_respects_max_wait(self):
        """Test that max_wait is respected for seconds format."""
        fallback = Mock()
        wait_func = wait_retry_after(fallback_strategy=fallback, max_wait=60)

        # Create HTTP status error with Retry-After > max_wait
        request = httpx.Request('GET', 'https://example.com')
        response = Mock(spec=httpx.Response)
        response.headers = {'retry-after': '120'}
        http_error = httpx.HTTPStatusError('Rate limited', request=request, response=response)

        retry_state = Mock(spec=RetryCallState)
        retry_state.outcome = Mock()
        retry_state.outcome.failed = True
        retry_state.outcome.exception.return_value = http_error

        result = wait_func(retry_state)

        assert result == 60.0  # Capped at max_wait
        fallback.assert_not_called()

    def test_retry_after_http_date_format(self):
        """Test parsing Retry-After header in HTTP date format."""
        fallback = Mock()
        wait_func = wait_retry_after(fallback_strategy=fallback, max_wait=300)

        # Create a future date (30 seconds from now)
        future_time = datetime.now(timezone.utc).timestamp() + 30
        http_date = formatdate(future_time, usegmt=True)

        # Create HTTP status error with Retry-After in HTTP date format
        request = httpx.Request('GET', 'https://example.com')
        response = Mock(spec=httpx.Response)
        response.headers = {'retry-after': http_date}
        http_error = httpx.HTTPStatusError('Rate limited', request=request, response=response)

        retry_state = Mock(spec=RetryCallState)
        retry_state.outcome = Mock()
        retry_state.outcome.failed = True
        retry_state.outcome.exception.return_value = http_error

        result = wait_func(retry_state)

        # Should be approximately 30 seconds (allow some tolerance for test timing)
        assert 25 <= result <= 35
        fallback.assert_not_called()

    def test_retry_after_http_date_past_time_uses_fallback(self):
        """Test that past dates in Retry-After fall back to fallback strategy."""
        fallback = Mock(return_value=1.0)
        wait_func = wait_retry_after(fallback_strategy=fallback, max_wait=300)

        # Create a past date
        past_time = datetime.now(timezone.utc).timestamp() - 30
        http_date = formatdate(past_time, usegmt=True)

        # Create HTTP status error with Retry-After in HTTP date format
        request = httpx.Request('GET', 'https://example.com')
        response = Mock(spec=httpx.Response)
        response.headers = {'retry-after': http_date}
        http_error = httpx.HTTPStatusError('Rate limited', request=request, response=response)

        retry_state = Mock(spec=RetryCallState)
        retry_state.outcome = Mock()
        retry_state.outcome.failed = True
        retry_state.outcome.exception.return_value = http_error

        result = wait_func(retry_state)

        assert result == 1.0
        fallback.assert_called_once_with(retry_state)

    def test_retry_after_http_date_respects_max_wait(self):
        """Test that max_wait is respected for HTTP date format."""
        fallback = Mock()
        wait_func = wait_retry_after(fallback_strategy=fallback, max_wait=60)

        # Create a future date (120 seconds from now, > max_wait)
        future_time = datetime.now(timezone.utc).timestamp() + 120
        http_date = formatdate(future_time, usegmt=True)

        # Create HTTP status error with Retry-After in HTTP date format
        request = httpx.Request('GET', 'https://example.com')
        response = Mock(spec=httpx.Response)
        response.headers = {'retry-after': http_date}
        http_error = httpx.HTTPStatusError('Rate limited', request=request, response=response)

        retry_state = Mock(spec=RetryCallState)
        retry_state.outcome = Mock()
        retry_state.outcome.failed = True
        retry_state.outcome.exception.return_value = http_error

        result = wait_func(retry_state)

        assert result == 60.0  # Capped at max_wait
        fallback.assert_not_called()

    def test_retry_after_invalid_format_uses_fallback(self):
        """Test that invalid Retry-After values fall back to fallback strategy."""
        fallback = Mock(return_value=4.0)
        wait_func = wait_retry_after(fallback_strategy=fallback, max_wait=300)

        # Create HTTP status error with invalid Retry-After
        request = httpx.Request('GET', 'https://example.com')
        response = Mock(spec=httpx.Response)
        response.headers = {'retry-after': 'invalid-value'}
        http_error = httpx.HTTPStatusError('Rate limited', request=request, response=response)

        retry_state = Mock(spec=RetryCallState)
        retry_state.outcome = Mock()
        retry_state.outcome.failed = True
        retry_state.outcome.exception.return_value = http_error

        result = wait_func(retry_state)

        assert result == 4.0
        fallback.assert_called_once_with(retry_state)

    def test_default_fallback_strategy(self):
        """Test that default fallback strategy is used when none is provided."""
        wait_func = wait_retry_after(max_wait=300)

        # Create a retry state with no exception to trigger fallback
        retry_state = Mock(spec=RetryCallState)
        retry_state.outcome = None
        retry_state.attempt_number = 1

        # Should use default exponential backoff, exact value depends on retry state
        result = wait_func(retry_state)

        assert result == 1  # first backoff

    def test_default_max_wait(self):
        """Test that default max_wait of 300 seconds is used."""
        wait_func = wait_retry_after()  # Use all defaults

        # Create HTTP status error with large Retry-After value
        request = httpx.Request('GET', 'https://example.com')
        response = Mock(spec=httpx.Response)
        response.headers = {'retry-after': '600'}  # 10 minutes
        http_error = httpx.HTTPStatusError('Rate limited', request=request, response=response)

        retry_state = Mock(spec=RetryCallState)
        retry_state.outcome = Mock()
        retry_state.outcome.failed = True
        retry_state.outcome.exception.return_value = http_error

        result = wait_func(retry_state)

        assert result == 300.0  # Capped at default max_wait

    def test_case_insensitive_header_access(self):
        """Test that Retry-After header access is case insensitive."""
        fallback = Mock()
        wait_func = wait_retry_after(fallback_strategy=fallback, max_wait=300)

        # Create HTTP status error with uppercase Retry-After header
        request = httpx.Request('GET', 'https://example.com')
        response = Mock(spec=httpx.Response)
        # httpx headers are case-insensitive, so this should work
        response.headers = httpx.Headers({'Retry-After': '45'})
        http_error = httpx.HTTPStatusError('Rate limited', request=request, response=response)

        retry_state = Mock(spec=RetryCallState)
        retry_state.outcome = Mock()
        retry_state.outcome.failed = True
        retry_state.outcome.exception.return_value = http_error

        result = wait_func(retry_state)

        assert result == 45.0
        fallback.assert_not_called()


class TestIntegration:
    """Integration tests combining transports with wait strategies."""

    async def test_async_transport_with_wait_retry_after(self):
        """Test AsyncTenacityTransport with wait_retry_after strategy."""
        mock_transport = AsyncMock(spec=httpx.AsyncBaseTransport)
        mock_response_fail = Mock(spec=httpx.Response)
        mock_response_fail.status_code = 429
        mock_response_fail.headers = {'retry-after': '1'}
        mock_response_success = Mock(spec=httpx.Response)
        mock_response_success.status_code = 200

        mock_transport.handle_async_request.side_effect = [mock_response_fail, mock_response_success]

        # Track validation calls
        validation_calls: list[int] = []

        def validate_response(response: httpx.Response):
            validation_calls.append(response.status_code)
            if response.status_code == 429:
                raise httpx.HTTPStatusError('Rate limited', request=request, response=response)

        controller = AsyncRetrying(
            retry=retry_if_exception_type(httpx.HTTPStatusError),
            wait=wait_retry_after(max_wait=5),  # Short max_wait for tests
            stop=stop_after_attempt(3),
            reraise=True,
        )
        transport = AsyncTenacityTransport(controller, mock_transport, validate_response)

        request = httpx.Request('GET', 'https://example.com')

        # Time the request to ensure retry-after wait was respected
        start_time = asyncio.get_event_loop().time()
        result = await transport.handle_async_request(request)
        end_time = asyncio.get_event_loop().time()

        assert result is mock_response_success
        assert mock_transport.handle_async_request.call_count == 2
        assert validation_calls == [429, 200]  # First call failed, second succeeded
        # Should have waited approximately 1 second (allow some tolerance)
        assert 0.8 <= (end_time - start_time) <= 2.0

    def test_sync_transport_with_wait_retry_after(self):
        """Test TenacityTransport with wait_retry_after strategy."""
        mock_transport = Mock(spec=httpx.BaseTransport)
        mock_response_fail = Mock(spec=httpx.Response)
        mock_response_fail.status_code = 429
        mock_response_fail.headers = {'retry-after': '30'}  # 30 seconds, will be capped
        mock_response_success = Mock(spec=httpx.Response)
        mock_response_success.status_code = 200

        mock_transport.handle_request.side_effect = [mock_response_fail, mock_response_success]

        def validate_response(response: httpx.Response):
            if response.status_code == 429:
                raise httpx.HTTPStatusError('Rate limited', request=request, response=response)

        controller = Retrying(
            retry=retry_if_exception_type(httpx.HTTPStatusError),
            wait=wait_retry_after(max_wait=2),  # Cap at 2 seconds for tests
            stop=stop_after_attempt(3),
            reraise=True,
        )
        transport = TenacityTransport(controller, mock_transport, validate_response)

        request = httpx.Request('GET', 'https://example.com')

        # Time the request to ensure max_wait was respected
        start_time = time.time()
        result = transport.handle_request(request)
        end_time = time.time()

        assert result is mock_response_success
        assert mock_transport.handle_request.call_count == 2
        # Should have waited approximately 2 seconds (capped by max_wait)
        assert 1.8 <= (end_time - start_time) <= 3.0
