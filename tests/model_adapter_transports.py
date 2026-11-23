"""The transports `tests/model_adapter_cases.py` builds each model class over, so a test controls what the provider sends.

Each transport takes the `build` of a model class and gives the model back over a client whose wire a test supplies:

- `Httpx2Transport` and `LegacyHttpxTransport` hand the provider an `httpx2` or legacy `httpx` client on a mock
  transport. `HuggingFaceTransport` does the same for the `httpx` client inside `AsyncInferenceClient`.
- `BotocoreTransport` puts a connection pool in front of botocore's own `URLLib3Session.send`, so botocore's error
  translation, its read of a non-streamed body and its retries all still run.
- `XaiTransport` replaces the gRPC stubs of an `xai_sdk` client, so `xai_sdk` itself still runs.

The HTTP transports all answer through one `Handler`, which takes and returns `httpx2` objects whatever client the
provider's SDK uses, and raises an `httpx2` transport error for a failed exchange: each transport converts them to the
objects its client deals in.
"""

from __future__ import annotations

import importlib
import io
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib.util import find_spec
from types import SimpleNamespace
from typing import TYPE_CHECKING, Protocol

import httpx
import httpx2
import pytest

from pydantic_ai.embeddings import EmbeddingModel
from pydantic_ai.images import ImageGenerationModel
from pydantic_ai.models import Model

if TYPE_CHECKING:
    from botocore.client import BaseClient
    from huggingface_hub import AsyncInferenceClient
    from urllib3 import HTTPResponse
    from xai_sdk import AsyncClient as AsyncXaiClient

AdapterModel = Model | EmbeddingModel | ImageGenerationModel
"""A model class the registry covers."""

Handler = Callable[[httpx2.Request], httpx2.Response]
"""Answers one request the way the provider would, or raises the `httpx2` transport error the exchange failed with."""


@dataclass(frozen=True)
class Httpx2Transport:
    """A provider that takes an `httpx2.AsyncClient`."""

    build: Callable[[httpx2.AsyncClient], AdapterModel]

    @asynccontextmanager
    async def model(self, handler: Handler) -> AsyncGenerator[AdapterModel]:
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
            yield self.build(client)


_LEGACY_ERRORS: dict[type[httpx2.TransportError], type[httpx.TransportError]] = {
    httpx2.ConnectError: httpx.ConnectError,
    httpx2.ConnectTimeout: httpx.ConnectTimeout,
    httpx2.ReadTimeout: httpx.ReadTimeout,
    httpx2.PoolTimeout: httpx.PoolTimeout,
    httpx2.ReadError: httpx.ReadError,
    httpx2.RemoteProtocolError: httpx.RemoteProtocolError,
}


def _legacy_error(error: httpx2.TransportError, request: httpx.Request) -> httpx.TransportError:
    return _LEGACY_ERRORS[type(error)](str(error), request=request)


class _LegacyStream(httpx.AsyncByteStream):
    """An `httpx2` response body as a legacy `httpx` one, raising the legacy twin of an error it breaks off with."""

    def __init__(self, response: httpx2.Response, request: httpx.Request) -> None:
        self._response = response
        self._request = request

    async def __aiter__(self) -> AsyncIterator[bytes]:
        stream = self._response.stream
        assert isinstance(stream, httpx2.AsyncByteStream)
        try:
            async for chunk in stream:
                yield chunk
        except httpx2.TransportError as error:
            raise _legacy_error(error, self._request) from None


def _legacy_response(response: httpx2.Response, request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        response.status_code,
        headers=response.headers.multi_items(),
        stream=_LegacyStream(response, request),
        request=request,
    )


def legacy_handler(handler: Handler) -> Callable[[httpx.Request], httpx.Response]:
    """`handler` as a legacy `httpx.MockTransport` handler."""

    def handle(request: httpx.Request) -> httpx.Response:
        converted = httpx2.Request(
            request.method, str(request.url), headers=request.headers.multi_items(), content=request.read()
        )
        try:
            response = handler(converted)
        except httpx2.HTTPStatusError as error:
            # What a legacy retrying transport raises once it gave up: it closes the response without reading it.
            raise httpx.HTTPStatusError(
                str(error), request=request, response=httpx.Response(error.response.status_code, request=request)
            ) from None
        except httpx2.TransportError as error:
            raise _legacy_error(error, request) from None
        return _legacy_response(response, request)

    return handle


@dataclass(frozen=True)
class LegacyHttpxTransport:
    """A provider whose SDK still takes only a legacy `httpx.AsyncClient` (Groq, Cohere)."""

    build: Callable[[httpx.AsyncClient], AdapterModel]

    @asynccontextmanager
    async def model(self, handler: Handler) -> AsyncGenerator[AdapterModel]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(legacy_handler(handler))) as client:
            yield self.build(client)


@dataclass(frozen=True)
class HuggingFaceTransport:
    """`HuggingFaceProvider` takes no HTTP client, only an `AsyncInferenceClient`, which sends through legacy `httpx`."""

    build: Callable[[AsyncInferenceClient], AdapterModel]

    @asynccontextmanager
    async def model(self, handler: Handler) -> AsyncGenerator[AdapterModel]:
        from huggingface_hub import AsyncInferenceClient

        hf_client = AsyncInferenceClient(token='hf_test')
        async with httpx.AsyncClient(transport=httpx.MockTransport(legacy_handler(handler))) as client:
            # `AsyncInferenceClient` opens its own client on first use unless it already holds one.
            hf_client._async_client = client  # pyright: ignore[reportPrivateUsage]
            yield self.build(hf_client)


class _BotocoreBody(io.RawIOBase):
    """An `httpx2` response body as the socket file urllib3 reads, failing the way a socket does where it breaks off."""

    def __init__(self, response: httpx2.Response) -> None:
        assert isinstance(response.stream, httpx2.SyncByteStream)
        self._chunks: Iterator[bytes] = iter(response.stream)
        self._pending = b''

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: memoryview) -> int:  # pyright: ignore[reportIncompatibleMethodOverride]
        if not self._pending:
            try:
                self._pending = next(self._chunks, b'')
            except httpx2.ReadTimeout:
                raise TimeoutError('timed out') from None
            except httpx2.TransportError:
                raise ConnectionResetError(104, 'Connection reset by peer') from None
        size = min(len(buffer), len(self._pending))
        buffer[:size] = self._pending[:size]
        self._pending = self._pending[size:]
        return size


class _BotocorePool:
    """The urllib3 connection pool botocore sends through, answering from a `Handler`."""

    proxy_headers: dict[str, str]

    def __init__(self, handler: Handler) -> None:
        self._handler = handler
        self.proxy_headers = {}

    def urlopen(
        self, method: str, url: str, body: bytes | None = None, headers: object = None, **_: object
    ) -> HTTPResponse:
        from urllib3 import HTTPResponse
        from urllib3.exceptions import ConnectTimeoutError, NewConnectionError, ReadTimeoutError

        request = httpx2.Request(method, f'https://bedrock.invalid{url}', content=body or b'')
        try:
            response = self._handler(request)
        except httpx2.ConnectTimeout:
            raise ConnectTimeoutError(None, 'Connection timed out') from None
        except httpx2.ConnectError:
            raise NewConnectionError(None, 'Failed to establish a new connection') from None  # pyright: ignore[reportArgumentType]
        except httpx2.ReadTimeout:
            raise ReadTimeoutError(None, url, 'Read timed out.') from None  # pyright: ignore[reportArgumentType]
        return HTTPResponse(
            body=_BotocoreBody(response),
            headers=dict(response.headers),
            status=response.status_code,
            preload_content=False,
            decode_content=False,
        )


@dataclass(frozen=True)
class _ConnectionManager:
    pool: _BotocorePool

    def connection_from_url(self, url: str) -> _BotocorePool:
        return self.pool


@dataclass(frozen=True)
class BotocoreTransport:
    """A provider on a boto3 client, which has no HTTP client to inject.

    botocore's `URLLib3Session.send` asks for a connection pool per request; the transport hands it one that answers
    from the handler. botocore retries a throttled, unavailable or dropped request itself, like an `httpx` retrying
    transport would, so no transport here gives up for it.
    """

    build: Callable[[], AdapterModel]
    client: Callable[[AdapterModel], BaseClient]

    @asynccontextmanager
    async def model(self, handler: Handler) -> AsyncGenerator[AdapterModel]:
        model = self.build()
        pool = _BotocorePool(handler)

        def connection_manager(url: str, proxy_url: str | None = None) -> _ConnectionManager:
            return _ConnectionManager(pool)

        endpoint: object = getattr(self.client(model), '_endpoint')
        setattr(getattr(endpoint, 'http_session'), '_get_connection_manager', connection_manager)
        yield model


class XaiStub(Protocol):
    """What the `xai_sdk` client's gRPC stubs answer: a unary call, or a server stream."""

    def unary(self) -> Awaitable[object]: ...

    def stream(self) -> AsyncIterator[object]: ...


class _XaiStubs:
    """Stands in for the generated `ChatStub`, `ImageStub` and `FilesStub`, answering every call from one `XaiStub`."""

    def __init__(self, stub: XaiStub) -> None:
        self._stub = stub

    async def GetCompletion(self, request: object, **_: object) -> object:
        return await self._stub.unary()

    def GetCompletionChunk(self, request: object, **_: object) -> AsyncIterator[object]:
        return self._stub.stream()

    async def GenerateImage(self, request: object, **_: object) -> object:
        return await self._stub.unary()

    async def UploadFile(self, requests: AsyncIterator[object], **_: object) -> object:
        async for _request in requests:
            pass
        return await self._stub.unary()


@dataclass(frozen=True)
class XaiTransport:
    """A provider on an `xai_sdk` client, which speaks gRPC, so the stubs stand in for the server."""

    build: Callable[[AsyncXaiClient], AdapterModel]

    @asynccontextmanager
    async def model(self, stub: XaiStub) -> AsyncGenerator[AdapterModel]:
        from xai_sdk import AsyncClient

        client = AsyncClient(api_key='test', api_host='xai.invalid:443')
        stubs = _XaiStubs(stub)
        for sub_client in (client.chat, client.image, client.files):
            setattr(sub_client, '_stub', stubs)
        try:
            yield self.build(client)
        finally:
            await client.close()


HttpTransport = Httpx2Transport | LegacyHttpxTransport | HuggingFaceTransport | BotocoreTransport
Transport = HttpTransport | XaiTransport


def _no_delay(*_: object, **__: object) -> float:
    return 0.0


def instant_sdk_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the SDKs that retry by default retry at once, rather than back off on the real clock."""
    for package in ('openai', 'anthropic', 'groq'):
        if find_spec(package) is not None:
            base_client = importlib.import_module(f'{package}._base_client').BaseClient
            monkeypatch.setattr(base_client, '_calculate_retry_timeout', _no_delay)
    if find_spec('cohere') is not None:
        monkeypatch.setattr(importlib.import_module('cohere.core.http_client'), '_retry_timeout', _no_delay)
    if find_spec('typesafe_sdk') is not None:
        monkeypatch.setattr(importlib.import_module('typesafe_sdk._core.retry'), '_backoff', _no_delay)
    if find_spec('botocore') is not None:
        # botocore's retry loop sleeps through `botocore.endpoint`'s own `time` import.
        monkeypatch.setattr(importlib.import_module('botocore.endpoint'), 'time', SimpleNamespace(sleep=_no_delay))
