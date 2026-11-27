from __future__ import annotations as _annotations

import threading
import time
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from datetime import UTC, datetime, timedelta, tzinfo
from email.utils import format_datetime
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import AsyncMock, Mock

import anyio
import httpcore2
import httpx
import httpx2
import pytest

from tests._timeouts import READINESS_WAIT_TIMEOUT

from .conftest import try_import

with try_import() as imports_successful:
    from tenacity import (
        RetryCallState,
        retry_if_exception_type,
        stop_after_attempt,
        wait_fixed,
    )

    from pydantic_ai import PydanticAIDeprecationWarning
    from pydantic_ai.retries import (
        AsyncHTTPX2TenacityTransport,
        AsyncTenacityTransport,
        HTTPX2TenacityTransport,
        RetryConfig,
        TenacityTransport,
        wait_retry_after,
    )

pytestmark = pytest.mark.skipif(not imports_successful(), reason='install tenacity to run tenacity tests')

# Only the classes that construct the deprecated transports may silence their warning; the `httpx2` classes
# run under the repo-wide `filterwarnings = ["error"]` so a deprecation escaping into them fails the test.
ignore_deprecation = pytest.mark.filterwarnings('ignore::pydantic_ai._warnings.PydanticAIDeprecationWarning')


class _NetworkStream(httpx2.SyncByteStream, httpx2.AsyncByteStream):
    """A response body that, like one from the network, is read only when iterated, and can fail mid-read."""

    def __init__(self, *, fail_read: bool):
        self.fail_read = fail_read
        self.iterated = False
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        self.iterated = True
        if self.fail_read:
            raise httpx2.ReadError('connection lost')
        yield b'{"error": "rejected"}'

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self:
            yield chunk

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.close()


class _StalledConnection(httpcore2.AsyncNetworkStream):
    """A connection that sends a 503's headers, then stalls its body until the reader is cancelled."""

    def __init__(self):
        self.reading_body = anyio.Event()
        self.closed = False
        self._head = [b'HTTP/1.1 503 Service Unavailable\r\nContent-Length: 64\r\n\r\n']

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        if self._head:
            return self._head.pop()
        self.reading_body.set()
        await anyio.sleep_forever()
        return b''  # pragma: no cover

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        pass

    async def aclose(self) -> None:
        self.closed = True


class _StalledConnectionBackend(httpcore2.AsyncNetworkBackend):
    def __init__(self, connection: _StalledConnection):
        self.connection = connection

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[object] | None = None,
    ) -> httpcore2.AsyncNetworkStream:
        return self.connection


def _reject(response: httpx2.Response) -> None:
    raise httpx2.HTTPStatusError(f'rejected {response.status_code}', request=response.request, response=response)


def _read_content(response: httpx2.Response) -> bytes | None:
    try:
        return response.content
    except httpx2.ResponseNotRead:
        return None


_REJECTED_RESPONSES = [
    pytest.param(429, False, True, b'{"error": "rejected"}', id='error-status-read'),
    pytest.param(200, False, False, None, id='success-status-closed-unread'),
    pytest.param(503, True, True, None, id='failed-read-keeps-validator-error'),
]
"""`(status_code, fail_read, iterated, content)`: a response `validate_response` rejected reads its body only for an
error status, and a failed read still raises the validator's exception."""


class TestHTTPX2TenacityTransport:
    """`httpx2` retry transports use the same Tenacity configuration as their `httpx` predecessors."""

    def test_default_wrapped_transports_use_pydantic_ai_pool_limits(self, transport_limits: list[httpx2.Limits]):
        """Without `wrapped`, each transport sends through one with the pool limits of Pydantic AI's own clients."""
        sync_transport = HTTPX2TenacityTransport(config=RetryConfig())
        async_transport = AsyncHTTPX2TenacityTransport(config=RetryConfig())

        assert isinstance(sync_transport.wrapped, httpx2.HTTPTransport)
        assert isinstance(async_transport.wrapped, httpx2.AsyncHTTPTransport)
        assert transport_limits == [httpx2.Limits(max_connections=1000, max_keepalive_connections=100)] * 2

    def test_sync_transport_passes_response_without_validator_and_closes_wrapped_transport(self):
        request = httpx2.Request('GET', 'https://example.com')
        response = httpx2.Response(200, request=request)
        wrapped = Mock(spec=httpx2.BaseTransport)
        wrapped.handle_request.return_value = response
        transport = HTTPX2TenacityTransport(config=RetryConfig(), wrapped=wrapped)

        assert transport.handle_request(request) is response

        transport.close()

        wrapped.handle_request.assert_called_once_with(request)
        wrapped.close.assert_called_once_with()

    def test_sync_transport_retries_response_validator(self):
        attempts = 0

        def handler(request: httpx2.Request) -> httpx2.Response:
            nonlocal attempts
            attempts += 1
            return httpx2.Response(503 if attempts == 1 else 200, request=request)

        transport = HTTPX2TenacityTransport(
            config=RetryConfig(
                retry=retry_if_exception_type(httpx2.HTTPStatusError),
                stop=stop_after_attempt(2),
                wait=wait_fixed(0),
                reraise=True,
            ),
            wrapped=httpx2.MockTransport(handler),
            validate_response=lambda response: response.raise_for_status(),
        )

        with httpx2.Client(transport=transport) as client:
            response = client.get('https://example.com')

        assert response.status_code == 200
        assert attempts == 2

    async def test_async_transport_exits_wrapped_transport_once(self):
        wrapped = AsyncMock(spec=httpx2.AsyncBaseTransport)
        transport = AsyncHTTPX2TenacityTransport(
            config=RetryConfig(),
            wrapped=wrapped,
        )

        async with transport:
            pass

        wrapped.__aenter__.assert_awaited_once_with()
        wrapped.__aexit__.assert_awaited_once_with(None, None, None)

    async def test_async_transport_retries_response_validator(self):
        attempts = 0

        async def handler(request: httpx2.Request) -> httpx2.Response:
            nonlocal attempts
            attempts += 1
            return httpx2.Response(503 if attempts == 1 else 200, request=request)

        transport = AsyncHTTPX2TenacityTransport(
            config=RetryConfig(
                retry=retry_if_exception_type(httpx2.HTTPStatusError),
                stop=stop_after_attempt(2),
                wait=wait_fixed(0),
                reraise=True,
            ),
            wrapped=httpx2.MockTransport(handler),
            validate_response=lambda response: response.raise_for_status(),
        )

        async with httpx2.AsyncClient(transport=transport) as client:
            response = await client.get('https://example.com')

        assert response.status_code == 200
        assert attempts == 2

    async def test_async_transport_passes_response_without_validator_and_closes_wrapped_transport(self):
        request = httpx2.Request('GET', 'https://example.com')
        response = httpx2.Response(200, request=request)
        wrapped = AsyncMock(spec=httpx2.AsyncBaseTransport)
        wrapped.handle_async_request.return_value = response
        transport = AsyncHTTPX2TenacityTransport(config=RetryConfig(), wrapped=wrapped)

        assert await transport.handle_async_request(request) is response

        await transport.aclose()

        wrapped.handle_async_request.assert_awaited_once_with(request)
        wrapped.aclose.assert_awaited_once_with()

    @pytest.mark.parametrize(('status_code', 'fail_read', 'iterated', 'content'), _REJECTED_RESPONSES)
    def test_sync_transport_reads_rejected_error_body_before_closing(
        self, status_code: int, fail_read: bool, iterated: bool, content: bytes | None
    ):
        """Below any model, so a test can see whether the body was read and the stream closed, which no model shows."""
        stream = _NetworkStream(fail_read=fail_read)
        transport = HTTPX2TenacityTransport(
            config=RetryConfig(stop=stop_after_attempt(1), reraise=True),
            wrapped=httpx2.MockTransport(lambda request: httpx2.Response(status_code, stream=stream)),
            validate_response=_reject,
        )

        with httpx2.Client(transport=transport) as client, pytest.raises(httpx2.HTTPStatusError) as exc_info:
            client.get('https://example.com')

        assert exc_info.value.response.status_code == status_code
        assert _read_content(exc_info.value.response) == content
        assert stream.iterated == iterated
        assert stream.closed

    @pytest.mark.parametrize(('status_code', 'fail_read', 'iterated', 'content'), _REJECTED_RESPONSES)
    async def test_async_transport_reads_rejected_error_body_before_closing(
        self, status_code: int, fail_read: bool, iterated: bool, content: bytes | None
    ):
        """The async counterpart; `test_retrying_transport_gave_up_keeps_streamed_error_body` covers it through a model."""
        stream = _NetworkStream(fail_read=fail_read)
        transport = AsyncHTTPX2TenacityTransport(
            config=RetryConfig(stop=stop_after_attempt(1), reraise=True),
            wrapped=httpx2.MockTransport(lambda request: httpx2.Response(status_code, stream=stream)),
            validate_response=_reject,
        )

        async with httpx2.AsyncClient(transport=transport) as client:
            with pytest.raises(httpx2.HTTPStatusError) as exc_info:
                await client.get('https://example.com')

        assert exc_info.value.response.status_code == status_code
        assert _read_content(exc_info.value.response) == content
        assert stream.iterated == iterated
        assert stream.closed

    @pytest.mark.parametrize('anyio_backend', ['asyncio', 'trio'])
    async def test_async_transport_closes_rejected_response_when_body_read_is_cancelled(self):
        """Cancelling the error body's read still closes the default transport's connection.

        Otherwise each cancelled attempt would hold a pooled connection
        (https://github.com/pydantic/pydantic-ai/issues/2724). `httpx2.AsyncHTTPTransport` takes no network backend,
        so its pool is replaced with one on a stalled connection; the transport's own response stream still sits
        between the read and that connection.
        """
        connection = _StalledConnection()
        pool = httpcore2.AsyncConnectionPool(network_backend=_StalledConnectionBackend(connection))
        wrapped = httpx2.AsyncHTTPTransport()
        wrapped._pool = pool  # pyright: ignore[reportPrivateUsage]
        transport = AsyncHTTPX2TenacityTransport(
            config=RetryConfig(stop=stop_after_attempt(1), reraise=True), wrapped=wrapped, validate_response=_reject
        )

        async with httpx2.AsyncClient(transport=transport) as client:
            async with anyio.create_task_group() as tg:
                tg.start_soon(client.get, 'http://example.com')
                with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                    await connection.reading_body.wait()
                tg.cancel_scope.cancel()

            # Checked before the client closes, since closing it closes the pool and every connection in it.
            assert connection.closed
            assert pool.connections == []


class TestLegacyHTTPXTransports:
    def test_sync_transport_warns_with_httpx2_replacement(self):
        with pytest.warns(
            PydanticAIDeprecationWarning,
            match=(
                r'`TenacityTransport` is deprecated and will be removed in v3; use `HTTPX2TenacityTransport` '
                r'with `httpx2.Client` instead\.'
            ),
        ):
            TenacityTransport(RetryConfig())

    def test_async_transport_warns_with_httpx2_replacement(self):
        with pytest.warns(
            PydanticAIDeprecationWarning,
            match=(
                r'`AsyncTenacityTransport` is deprecated and will be removed in v3; use `AsyncHTTPX2TenacityTransport` '
                r'with `httpx2.AsyncClient` instead\.'
            ),
        ):
            AsyncTenacityTransport(RetryConfig())


@ignore_deprecation
class TestTenacityTransport:
    """Tests for the synchronous TenacityTransport."""

    def test_successful_request(self):
        """Test that successful requests pass through without retry."""
        mock_transport = Mock(spec=httpx.BaseTransport)
        mock_transport.__enter__ = Mock(return_value=mock_transport)
        mock_transport.__exit__ = Mock(return_value=None)
        mock_response = Mock(spec=httpx.Response)
        mock_transport.handle_request.return_value = mock_response

        config = RetryConfig(stop=stop_after_attempt(3), reraise=True)
        transport = TenacityTransport(config, mock_transport)

        request = httpx.Request('GET', 'https://example.com')
        with transport:
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

        config = RetryConfig(
            retry=retry_if_exception_type(httpx.ConnectError),
            stop=stop_after_attempt(3),
            wait=wait_fixed(0.001),  # Very short wait for tests
            reraise=True,
        )
        transport = TenacityTransport(config, mock_transport)

        request = httpx.Request('GET', 'https://example.com')
        result = transport.handle_request(request)

        assert result is mock_response
        assert mock_transport.handle_request.call_count == 3

    def test_retry_exhausted(self):
        """Test that retry exhaustion re-raises the last exception."""
        mock_transport = Mock(spec=httpx.BaseTransport)
        mock_transport.handle_request.side_effect = httpx.ConnectError('Connection failed')

        config = RetryConfig(
            retry=retry_if_exception_type(httpx.ConnectError),
            stop=stop_after_attempt(2),
            wait=wait_fixed(0.001),
            reraise=True,
        )
        transport = TenacityTransport(config, mock_transport)

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
        config = RetryConfig(stop=stop_after_attempt(3), reraise=True)
        transport = TenacityTransport(config, mock_transport, validate_response)

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

        config = RetryConfig(
            retry=retry_if_exception_type(httpx.HTTPStatusError),
            stop=stop_after_attempt(3),
            wait=wait_fixed(0.001),
            reraise=True,
        )
        transport = TenacityTransport(config, mock_transport, validate_response)

        request = httpx.Request('GET', 'https://example.com')
        result = transport.handle_request(request)

        assert result is mock_response_success
        assert mock_transport.handle_request.call_count == 2

    def test_raise_for_status_in_validate_response(self):
        """Test that response.raise_for_status() works in validate_response callback."""
        mock_transport = Mock(spec=httpx.BaseTransport)
        mock_response_fail = Mock(spec=httpx.Response)
        mock_response_fail.status_code = 429
        mock_response_fail.is_success = False
        mock_response_fail.is_error = True
        mock_response_fail.request = None  # Initially None, will be set by transport

        # Mock raise_for_status to check if request is set
        def mock_raise_for_status():
            if mock_response_fail.request is None:
                raise RuntimeError(  # pragma: no cover
                    'Cannot call `raise_for_status` as the request instance has not been set on this response.'
                )
            raise httpx.HTTPStatusError(
                'Too Many Requests', request=mock_response_fail.request, response=mock_response_fail
            )

        mock_response_fail.raise_for_status = mock_raise_for_status

        mock_response_success = Mock(spec=httpx.Response)
        mock_response_success.status_code = 200
        mock_response_success.is_success = True
        mock_response_success.is_error = False
        mock_response_success.raise_for_status = Mock()  # Should not raise

        mock_transport.handle_request.side_effect = [mock_response_fail, mock_response_success]

        config = RetryConfig(
            retry=retry_if_exception_type(httpx.HTTPStatusError),
            stop=stop_after_attempt(3),
            wait=wait_fixed(0.001),
            reraise=True,
        )
        transport = TenacityTransport(
            config, mock_transport, validate_response=lambda response: response.raise_for_status()
        )

        request = httpx.Request('GET', 'https://example.com')
        result = transport.handle_request(request)

        assert result is mock_response_success
        assert mock_transport.handle_request.call_count == 2
        # Verify that the request was set on the failed response before raise_for_status was called
        assert mock_response_fail.request is request
        mock_response_success.raise_for_status.assert_called_once()


@ignore_deprecation
class TestAsyncTenacityTransport:
    """Tests for the asynchronous AsyncTenacityTransport."""

    async def test_successful_request(self):
        """Test that successful requests pass through without retry."""
        mock_transport = AsyncMock(spec=httpx.AsyncBaseTransport)
        mock_response = Mock(spec=httpx.Response)
        mock_transport.handle_async_request.return_value = mock_response

        config = RetryConfig(stop=stop_after_attempt(3), reraise=True)
        transport = AsyncTenacityTransport(config, mock_transport)

        request = httpx.Request('GET', 'https://example.com')
        async with transport:
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

        config = RetryConfig(
            retry=retry_if_exception_type(httpx.ConnectError),
            stop=stop_after_attempt(3),
            wait=wait_fixed(0.001),
            reraise=True,
        )
        transport = AsyncTenacityTransport(config, mock_transport)

        request = httpx.Request('GET', 'https://example.com')
        result = await transport.handle_async_request(request)

        assert result is mock_response
        assert mock_transport.handle_async_request.call_count == 3

    async def test_retry_exhausted(self):
        """Test that retry exhaustion re-raises the last exception."""
        mock_transport = AsyncMock(spec=httpx.AsyncBaseTransport)
        mock_transport.handle_async_request.side_effect = httpx.ConnectError('Connection failed')

        config = RetryConfig(
            retry=retry_if_exception_type(httpx.ConnectError),
            stop=stop_after_attempt(2),
            wait=wait_fixed(0.001),
            reraise=True,
        )
        transport = AsyncTenacityTransport(config, mock_transport)

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
        config = RetryConfig(stop=stop_after_attempt(3), reraise=True)
        transport = AsyncTenacityTransport(config, mock_transport, validate_response)

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

        config = RetryConfig(
            retry=retry_if_exception_type(httpx.HTTPStatusError),
            stop=stop_after_attempt(3),
            wait=wait_fixed(0.001),
            reraise=True,
        )
        transport = AsyncTenacityTransport(config, mock_transport, validate_response)

        request = httpx.Request('GET', 'https://example.com')
        result = await transport.handle_async_request(request)

        assert result is mock_response_success
        assert mock_transport.handle_async_request.call_count == 2

    async def test_raise_for_status_in_validate_response(self):
        """Test that response.raise_for_status() works in validate_response callback."""
        mock_transport = AsyncMock(spec=httpx.AsyncBaseTransport)
        mock_response_fail = Mock(spec=httpx.Response)
        mock_response_fail.status_code = 429
        mock_response_fail.is_success = False
        mock_response_fail.is_error = True
        mock_response_fail.request = None  # Initially None, will be set by transport

        # Mock raise_for_status to check if request is set
        def mock_raise_for_status():
            if mock_response_fail.request is None:
                raise RuntimeError(  # pragma: no cover
                    'Cannot call `raise_for_status` as the request instance has not been set on this response.'
                )
            raise httpx.HTTPStatusError(
                'Too Many Requests', request=mock_response_fail.request, response=mock_response_fail
            )

        mock_response_fail.raise_for_status = mock_raise_for_status

        mock_response_success = Mock(spec=httpx.Response)
        mock_response_success.status_code = 200
        mock_response_success.is_success = True
        mock_response_success.is_error = False
        mock_response_success.raise_for_status = Mock()  # Should not raise

        mock_transport.handle_async_request.side_effect = [mock_response_fail, mock_response_success]

        config = RetryConfig(
            retry=retry_if_exception_type(httpx.HTTPStatusError),
            stop=stop_after_attempt(3),
            wait=wait_fixed(0.001),
            reraise=True,
        )
        transport = AsyncTenacityTransport(
            config, mock_transport, validate_response=lambda response: response.raise_for_status()
        )

        request = httpx.Request('GET', 'https://example.com')
        result = await transport.handle_async_request(request)

        assert result is mock_response_success
        assert mock_transport.handle_async_request.call_count == 2
        # Verify that the request was set on the failed response before raise_for_status was called
        assert mock_response_fail.request is request
        mock_response_success.raise_for_status.assert_called_once()


_FROZEN_NOW = datetime(2095, 11, 6, 8, 49, 37, tzinfo=UTC)


class _FrozenDatetimeMeta(type):
    def __instancecheck__(cls, instance: object) -> bool:
        return isinstance(instance, datetime)


class _FrozenDatetime(datetime, metaclass=_FrozenDatetimeMeta):
    """`datetime` whose `now()` is `_FROZEN_NOW`.

    `wait_retry_after` checks the parsed date with `isinstance` against this patched name, and
    `parsedate_to_datetime` returns a real `datetime`, so the metaclass accepts real instances.
    """

    @classmethod
    def now(cls, tz: tzinfo | None = None) -> datetime:
        return _FROZEN_NOW


@pytest.fixture
def frozen_now(monkeypatch: pytest.MonkeyPatch) -> datetime:
    """Freeze the clock `wait_retry_after` measures `Retry-After` dates against, so waits are exact."""
    monkeypatch.setattr('pydantic_ai.retries.datetime', _FrozenDatetime)
    return _FROZEN_NOW


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

    @pytest.mark.parametrize(('retry_after', 'expected'), [('30', 30.0), pytest.param('0', 0.0, id='zero')])
    def test_retry_after_seconds_format(self, retry_after: str, expected: float):
        """Test parsing Retry-After header in seconds format, including `0` for an immediate retry."""
        fallback = Mock()
        wait_func = wait_retry_after(fallback_strategy=fallback, max_wait=300)

        # Create HTTP status error with Retry-After in seconds
        request = httpx.Request('GET', 'https://example.com')
        response = Mock(spec=httpx.Response)
        response.headers = {'retry-after': retry_after}
        http_error = httpx.HTTPStatusError('Rate limited', request=request, response=response)

        retry_state = Mock(spec=RetryCallState)
        retry_state.outcome = Mock()
        retry_state.outcome.failed = True
        retry_state.outcome.exception.return_value = http_error

        result = wait_func(retry_state)

        assert result == expected
        fallback.assert_not_called()

    def test_httpx2_retry_after_seconds_format(self):
        fallback = Mock()
        wait_func = wait_retry_after(fallback_strategy=fallback, max_wait=300)

        request = httpx2.Request('GET', 'https://example.com')
        response = httpx2.Response(429, headers={'retry-after': '30'}, request=request)
        with pytest.raises(httpx2.HTTPStatusError) as exc_info:
            response.raise_for_status()

        retry_state = Mock(spec=RetryCallState)
        retry_state.outcome = Mock()
        retry_state.outcome.failed = True
        retry_state.outcome.exception.return_value = exc_info.value

        assert wait_func(retry_state) == 30.0
        fallback.assert_not_called()

    @pytest.mark.parametrize('retry_after', ['120', pytest.param('1' + '0' * 309, id='huge')])
    def test_retry_after_seconds_respects_max_wait(self, retry_after: str):
        """Integer `Retry-After` values are capped locally without a provider request."""
        fallback = Mock()
        wait_func = wait_retry_after(fallback_strategy=fallback, max_wait=60)

        request = httpx.Request('GET', 'https://example.com')
        response = Mock(spec=httpx.Response)
        response.headers = {'retry-after': retry_after}
        http_error = httpx.HTTPStatusError('Rate limited', request=request, response=response)

        retry_state = Mock(spec=RetryCallState)
        retry_state.outcome = Mock()
        retry_state.outcome.failed = True
        retry_state.outcome.exception.return_value = http_error

        result = wait_func(retry_state)

        assert result == 60.0
        fallback.assert_not_called()

    @pytest.mark.parametrize(
        ('headers', 'max_wait', 'expected'),
        [
            pytest.param({'retry-after-ms': '1500'}, 300, 1.5, id='ms-only'),
            pytest.param({'retry-after-ms': '1500', 'retry-after': '2'}, 300, 1.5, id='ms-preferred'),
            pytest.param({'retry-after-ms': '90000'}, 60, 60.0, id='ms-capped'),
            pytest.param({'retry-after-ms': 'invalid', 'retry-after': '2'}, 300, 2.0, id='invalid-ms-falls-through'),
            pytest.param({'retry-after-ms': '-1', 'retry-after': '2'}, 300, 2.0, id='negative-ms-falls-through'),
            pytest.param({'retry-after-ms': 'inf', 'retry-after': '2'}, 300, 2.0, id='inf-ms-falls-through'),
            pytest.param({'retry-after-ms': 'nan', 'retry-after': '2'}, 300, 2.0, id='nan-ms-falls-through'),
        ],
    )
    def test_retry_after_ms(self, headers: dict[str, str], max_wait: float, expected: float):
        """`retry-after-ms`, as OpenAI and Anthropic send it, takes precedence over `Retry-After` like in their SDKs."""
        fallback = Mock(return_value=8.0)
        wait_func = wait_retry_after(fallback_strategy=fallback, max_wait=max_wait)

        request = httpx2.Request('POST', 'https://api.openai.com/v1/chat/completions')
        response = httpx2.Response(429, headers=headers, request=request)
        http_error = httpx2.HTTPStatusError('Rate limited', request=request, response=response)

        retry_state = Mock(spec=RetryCallState)
        retry_state.outcome = Mock()
        retry_state.outcome.exception.return_value = http_error

        assert wait_func(retry_state) == expected
        fallback.assert_not_called()

    @pytest.mark.parametrize(
        'format_date',
        [
            pytest.param(partial(format_datetime, usegmt=True), id='imf-fixdate'),
            pytest.param(datetime.ctime, id='asctime'),
        ],
    )
    @pytest.mark.parametrize(
        ('offset', 'max_wait', 'expected'),
        [
            pytest.param(30, 300, 30.0, id='within-max-wait'),
            pytest.param(120, 60, 60.0, id='capped'),
        ],
    )
    def test_retry_after_http_date(
        self,
        frozen_now: datetime,
        format_date: Callable[[datetime], str],
        offset: int,
        max_wait: float,
        expected: float,
    ):
        """An HTTP-date `Retry-After` waits until that date, capped at `max_wait`.

        Both formats are covered: IMF-fixdate, and asctime, which carries no timezone and is read as UTC.
        """
        fallback = Mock()
        wait_func = wait_retry_after(fallback_strategy=fallback, max_wait=max_wait)

        request = httpx2.Request('GET', 'https://example.com')
        response = httpx2.Response(
            429,
            headers={'retry-after': format_date(frozen_now + timedelta(seconds=offset))},
            request=request,
        )
        http_error = httpx2.HTTPStatusError('Rate limited', request=request, response=response)
        retry_state = Mock(spec=RetryCallState)
        retry_state.outcome = Mock()
        retry_state.outcome.exception.return_value = http_error

        assert wait_func(retry_state) == expected
        fallback.assert_not_called()

    @pytest.mark.parametrize('offset', [pytest.param(-30, id='past'), pytest.param(0, id='now')])
    def test_retry_after_http_date_not_in_future_uses_fallback(self, frozen_now: datetime, offset: int):
        """A `Retry-After` date that is not after now falls back to the fallback strategy."""
        fallback = Mock(return_value=1.0)
        wait_func = wait_retry_after(fallback_strategy=fallback, max_wait=300)

        http_date = format_datetime(frozen_now + timedelta(seconds=offset), usegmt=True)

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

    @pytest.mark.parametrize('retry_after', ['invalid-value', '-1', 'Wed, 21 Oct 999999999999 07:28:00 GMT'])
    def test_invalid_retry_after_uses_fallback(self, retry_after: str):
        """Invalid `Retry-After` values fall back locally without a provider request."""
        fallback = Mock(return_value=4.0)
        wait_func = wait_retry_after(fallback_strategy=fallback, max_wait=300)

        request = httpx.Request('GET', 'https://example.com')
        response = Mock(spec=httpx.Response)
        response.headers = {'retry-after': retry_after}
        http_error = httpx.HTTPStatusError('Rate limited', request=request, response=response)

        retry_state = Mock(spec=RetryCallState)
        retry_state.outcome = Mock()
        retry_state.outcome.failed = True
        retry_state.outcome.exception.return_value = http_error

        result = wait_func(retry_state)

        assert result == 4.0
        fallback.assert_called_once_with(retry_state)

    @pytest.mark.parametrize(
        ('headers', 'attempt_number', 'expected'),
        [
            pytest.param(None, 1, 1.0, id='first-backoff'),
            # 2 ** 6 = 64 s uncapped
            pytest.param(None, 7, 60.0, id='capped'),
            pytest.param({}, 7, 60.0, id='capped-no-retry-after'),
            pytest.param({'retry-after': 'invalid-value'}, 7, 60.0, id='capped-invalid-retry-after'),
        ],
    )
    def test_default_fallback_strategy(self, headers: dict[str, str] | None, attempt_number: int, expected: float):
        """The default fallback backs off exponentially from 1 s and never waits more than 60 s.

        `headers=None` retries with no exception; otherwise the retry follows a 429 carrying `headers`.
        """
        wait_func = wait_retry_after(max_wait=300)

        retry_state = Mock(spec=RetryCallState)
        retry_state.attempt_number = attempt_number
        if headers is None:
            retry_state.outcome = None
        else:
            request = httpx2.Request('GET', 'https://example.com')
            response = httpx2.Response(429, headers=headers, request=request)
            retry_state.outcome = Mock()
            retry_state.outcome.exception.return_value = httpx2.HTTPStatusError(
                'Rate limited', request=request, response=response
            )

        assert wait_func(retry_state) == expected

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

    @pytest.mark.parametrize(
        ('headers', 'expected'),
        [
            pytest.param({'Retry-After': '45'}, 45.0, id='retry-after'),
            pytest.param({'Retry-After-Ms': '1500'}, 1.5, id='retry-after-ms'),
        ],
    )
    def test_case_insensitive_header_access(self, headers: dict[str, str], expected: float):
        """`Retry-After` and `retry-after-ms` are read case-insensitively from a legacy `httpx` error."""
        fallback = Mock()
        wait_func = wait_retry_after(fallback_strategy=fallback, max_wait=300)

        # Create HTTP status error with uppercase headers
        request = httpx.Request('GET', 'https://example.com')
        response = Mock(spec=httpx.Response)
        # httpx headers are case-insensitive, so this should work
        response.headers = httpx.Headers(headers)
        http_error = httpx.HTTPStatusError('Rate limited', request=request, response=response)

        retry_state = Mock(spec=RetryCallState)
        retry_state.outcome = Mock()
        retry_state.outcome.failed = True
        retry_state.outcome.exception.return_value = http_error

        result = wait_func(retry_state)

        assert result == expected
        fallback.assert_not_called()


@ignore_deprecation
class TestIntegration:
    """Integration tests combining transports with wait strategies."""

    async def test_async_transport_with_wait_retry_after(self):
        """Test AsyncTenacityTransport with wait_retry_after strategy."""
        mock_transport = AsyncMock(spec=httpx.AsyncBaseTransport)
        mock_response_fail = Mock(spec=httpx.Response)
        mock_response_fail.status_code = 429
        mock_response_fail.headers = {'retry-after': '2'}
        mock_response_success = Mock(spec=httpx.Response)
        mock_response_success.status_code = 200

        mock_transport.handle_async_request.side_effect = [mock_response_fail, mock_response_success]

        # Track validation calls
        validation_calls: list[int] = []

        def validate_response(response: httpx.Response):
            validation_calls.append(response.status_code)
            if response.status_code == 429:
                raise httpx.HTTPStatusError('Rate limited', request=request, response=response)

        sleep_calls: list[float] = []

        async def sleep(seconds: int | float) -> None:
            sleep_calls.append(float(seconds))

        config = RetryConfig(
            retry=retry_if_exception_type(httpx.HTTPStatusError),
            wait=wait_retry_after(max_wait=5),  # Short max_wait for tests
            stop=stop_after_attempt(3),
            sleep=sleep,
            reraise=True,
        )
        transport = AsyncTenacityTransport(config, mock_transport, validate_response)

        request = httpx.Request('GET', 'https://example.com')

        result = await transport.handle_async_request(request)

        assert result is mock_response_success
        assert mock_transport.handle_async_request.call_count == 2
        assert validation_calls == [429, 200]  # First call failed, second succeeded
        # The header's 2 s, not the 1 s the exponential fallback would wait first.
        assert sleep_calls == [2.0]

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

        sleep_calls: list[float] = []

        def sleep(seconds: int | float) -> None:
            sleep_calls.append(float(seconds))

        config = RetryConfig(
            retry=retry_if_exception_type(httpx.HTTPStatusError),
            wait=wait_retry_after(max_wait=0.1),  # Cap at 0.1 seconds for tests
            stop=stop_after_attempt(3),
            sleep=sleep,
            reraise=True,
        )
        transport = TenacityTransport(config, mock_transport, validate_response)

        request = httpx.Request('GET', 'https://example.com')

        result = transport.handle_request(request)

        assert result is mock_response_success
        assert mock_transport.handle_request.call_count == 2
        assert sleep_calls == [0.1]


@ignore_deprecation
class TestConnectionPool:
    class AlwaysReturnHTTP429Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(429)
            self.send_header('Retry-After', '0')
            self.end_headers()
            self.wfile.write(b'Rate limited')

    def start_test_server(self, port: int = 8429) -> HTTPServer:
        server = HTTPServer(('localhost', port), self.AlwaysReturnHTTP429Handler)

        def run_server():
            server.serve_forever()

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        time.sleep(0.1)
        return server

    async def test_connection_pool(self):
        server = self.start_test_server(8429)
        test_url = 'http://localhost:8429/test'

        def validate_response(response: httpx.Response) -> None:
            response.raise_for_status()

        retry_strategy = RetryConfig(
            stop=stop_after_attempt(5),
            wait=wait_retry_after(max_wait=0, fallback_strategy=wait_fixed(0)),
            retry=retry_if_exception_type(httpx.HTTPStatusError),
            reraise=True,
        )

        transport = AsyncTenacityTransport(
            config=retry_strategy,
            validate_response=validate_response,
            wrapped=httpx.AsyncHTTPTransport(
                limits=httpx.Limits(max_connections=2, max_keepalive_connections=2, keepalive_expiry=30)
            ),
        )

        client = httpx.AsyncClient(transport=transport)

        with pytest.raises(httpx.HTTPStatusError, match='429 Too Many Requests'):
            try:
                await client.get(test_url)
            finally:
                await client.aclose()
                server.shutdown()
