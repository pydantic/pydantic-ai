"""Every model class's entry points raise a `ModelAPIError` for every way its provider's transport can fail.

The classes, their builds and their entry points come from `tests/model_adapter_cases.py`. Each entry point that calls
the provider's SDK is sent each fault below at the transport the SDK sends through: a refused connection, a read
timeout, an error status the provider sent, one a retrying `httpx` transport gave up on, a 200 whose body isn't JSON
or isn't JSON at all, a body that breaks off, and an error status whose body breaks off. A fault has one expected set
of error categories, so it is raised the same way from every entry point of every class.

A 200 that isn't JSON is sent only to the entry points that don't stream: a streamed request reads it as an event
stream, which several SDKs end quietly with no events.

Not VCR tests: a provider doesn't fail on demand, so each fault is replayed at the transport, under the provider's
real SDK.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AbstractAsyncContextManager, suppress
from dataclasses import dataclass

import httpx2
import pytest

from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelContextWindowExceededError,
    ModelHTTPError,
    ModelNotFoundError,
    ModelOverloadedError,
    ModelQuotaExceededError,
    ModelRateLimitError,
    ModelRequestRejectedError,
    ModelServerError,
    ModelTimeoutError,
    ModelUnavailableError,
    TransportPhase,
)

from .conftest import gave_up_on, try_import
from .manifests import assert_classified
from .model_adapter_cases import (
    CALLS_SDK,
    CASES,
    ENTRY_POINTS,
    ModelAdapterCase,
    NoIO,
    NotSupported,
    SdkCell,
    base_of,
    invoke,
    sdk_cells,
    shipped_model_classes,
)
from .model_adapter_transports import (
    AdapterModel,
    BotocoreTransport,
    Handler,
    Httpx2Transport,
    XaiStub,
    XaiTransport,
    instant_sdk_retries,
)

with try_import():
    import grpc
    import grpc.aio
    from xai_sdk.proto import chat_pb2


@pytest.fixture(autouse=True)
def _instant_sdk_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    instant_sdk_retries(monkeypatch)


_CATEGORIES: tuple[type[ModelAPIError], ...] = (
    ModelHTTPError,
    ModelRateLimitError,
    ModelQuotaExceededError,
    ModelUnavailableError,
    ModelOverloadedError,
    ModelServerError,
    ModelConnectionError,
    ModelTimeoutError,
    ModelContextWindowExceededError,
    ModelRequestRejectedError,
    ModelAuthenticationError,
    ModelNotFoundError,
)


def _categories(error: ModelAPIError) -> frozenset[type[ModelAPIError]]:
    """The error categories `error` belongs to, plus `ModelHTTPError` when it has a status."""
    return frozenset(cls for cls in _CATEGORIES if isinstance(error, cls))


def test_every_model_class_has_a_case():
    classes, skipped = shipped_model_classes()
    assert_classified(classes, CASES, complete_walk=not skipped, table_name='CASES')


def _adapter_cases() -> list[tuple[type[AdapterModel], ModelAdapterCase]]:
    classes, _ = shipped_model_classes()
    return [
        (model_class, case)
        for name, model_class in sorted(classes.items())
        if isinstance(case := CASES.get(name), ModelAdapterCase)
    ]


def _defining_class(model_class: type[AdapterModel], name: str) -> type:
    return next(cls for cls in model_class.__mro__ if name in vars(cls))


@pytest.mark.parametrize(
    ('model_class', 'case'), [pytest.param(*pair, id=pair[0].__name__) for pair in _adapter_cases()]
)
def test_each_case_classifies_the_entry_points_its_class_overrides(
    model_class: type[AdapterModel], case: ModelAdapterCase
):
    base = base_of(model_class)
    overridden = {name for name in ENTRY_POINTS[base] if _defining_class(model_class, name) is not base}
    assert set(case.entry_points) == overridden, (
        f'`CASES[{model_class.__name__!r}].entry_points` must classify exactly the entry points the class overrides: '
        f'unclassified {sorted(overridden - set(case.entry_points))}, '
        f'not overridden {sorted(set(case.entry_points) - overridden)}'
    )


def _rpc_error(status: str, details: str) -> grpc.aio.AioRpcError:
    return grpc.aio.AioRpcError(grpc.StatusCode[status], grpc.aio.Metadata(), grpc.aio.Metadata(), details=details)


@dataclass(frozen=True)
class _XaiAnswer:
    """An `XaiStub` whose calls send `chunks`, then fail with `error`, or else answer `response`.

    A `response` of `None` is what a gRPC call returns for a payload its deserializer couldn't read.
    """

    error: Callable[[], grpc.aio.AioRpcError] | None = None
    response: object = None
    chunks: tuple[Callable[[], object], ...] = ()

    async def unary(self) -> object:
        if self.error is not None:
            raise self.error()
        return self.response

    async def stream(self) -> AsyncIterator[object]:
        for chunk in self.chunks:
            yield chunk()
        if self.error is not None:
            raise self.error()
        yield self.response


def _open(case: ModelAdapterCase, handler: Handler, stub: XaiStub) -> AbstractAsyncContextManager[AdapterModel]:
    """The case's model, its HTTP transport answering from `handler`, or its gRPC stubs from `stub`."""
    transport = case.transport
    if isinstance(transport, XaiTransport):
        return transport.model(stub)
    return transport.model(handler)


@dataclass
class _Sent:
    """Counts the requests that reach the transport, and fails each one, so a call that reaches the provider stops."""

    requests: int = 0

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests += 1
        return httpx2.Response(400, json={'error': {'message': 'probe', 'type': 'probe', 'code': 'probe'}})

    def error(self) -> grpc.aio.AioRpcError:
        self.requests += 1
        return _rpc_error('INVALID_ARGUMENT', 'probe')


def _classified_cells() -> list[object]:
    cells: list[object] = []
    for model_class, case in _adapter_cases():
        for entry_point in ENTRY_POINTS[base_of(model_class)]:
            kind = case.kind(model_class, entry_point)
            if not isinstance(kind, NoIO) or kind.invoke:
                cells.append(pytest.param(model_class, entry_point, case, id=f'{model_class.__name__}.{entry_point}'))
    return cells


@pytest.mark.parametrize(('model_class', 'entry_point', 'case'), _classified_cells())
async def test_entry_point_classification_holds(
    allow_model_requests: None, model_class: type[AdapterModel], entry_point: str, case: ModelAdapterCase
):
    """A `CALLS_SDK` entry point reaches the transport; a `NoIO` one answers without it; a `NotSupported` one raises."""
    sent = _Sent()
    kind = case.kind(model_class, entry_point)
    async with _open(case, sent.handler, _XaiAnswer(error=sent.error)) as model:
        assert type(model) is model_class
        if kind == CALLS_SDK:
            # What the failed request raises is `test_transport_fault_raises_model_api_error`'s concern.
            with suppress(Exception):
                await invoke(model, entry_point, case.parameters)
            assert sent.requests > 0
        elif isinstance(kind, NoIO):
            await invoke(model, entry_point, case.parameters)
            assert sent.requests == 0
        else:
            assert isinstance(kind, NotSupported)
            with pytest.raises(kind.raises):
                await invoke(model, entry_point, case.parameters)
            assert sent.requests == 0


_ERROR_BODY = {'error': {'message': 'fault', 'type': 'fault', 'code': 'fault'}}
"""An error body that names no category, so every API classifies it by its status alone."""


def _status(status: int) -> Handler:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json=_ERROR_BODY)

    return handler


def _raise(error: type[httpx2.TransportError]) -> Handler:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise error('fault', request=request)

    return handler


def _body(content_type: str, content: bytes) -> Handler:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, headers={'content-type': content_type}, content=content)

    return handler


class _BreakingOff(httpx2.SyncByteStream, httpx2.AsyncByteStream):
    """A body that breaks off after its first chunk, as when the connection drops or the read times out."""

    def __init__(self, error: type[httpx2.TransportError]) -> None:
        self._error = error

    def __iter__(self) -> Iterator[bytes]:
        yield b'{"id": "'
        raise self._error('fault')

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b'{"id": "'
        raise self._error('fault')


def _breaking_off(status: int, content_type: str, error: type[httpx2.TransportError]) -> Handler:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, headers={'content-type': content_type}, stream=_BreakingOff(error))

    return handler


_RATE_LIMITED = frozenset({ModelHTTPError, ModelRateLimitError})
_UNAVAILABLE = frozenset({ModelHTTPError, ModelUnavailableError})
_REJECTED = frozenset({ModelHTTPError, ModelRequestRejectedError})
_CONNECTION = frozenset({ModelConnectionError})
_TIMEOUT = frozenset({ModelConnectionError, ModelTimeoutError})
_UNCLASSIFIED: frozenset[type[ModelAPIError]] = frozenset()


@dataclass(frozen=True)
class _HttpFault:
    id: str
    handler: Callable[[], Handler]
    categories: frozenset[type[ModelAPIError]]
    phase: TransportPhase | None = None
    gave_up: bool = False
    """Raised by a retrying `httpx` transport, which a user mounts on a provider that takes an HTTP client, or through
    `huggingface_hub.set_async_client_factory`. botocore retries itself and takes no transport."""
    streamed: bool | None = None
    """Sent only to `request_stream` (`True`), only to the other entry points (`False`), or to all of them (`None`)."""


HTTP_FAULTS = [
    _HttpFault('connect', lambda: _raise(httpx2.ConnectError), _CONNECTION, 'connect'),
    _HttpFault('read-timeout', lambda: _raise(httpx2.ReadTimeout), _TIMEOUT, 'read'),
    _HttpFault('status-429', lambda: _status(429), _RATE_LIMITED),
    _HttpFault('status-503', lambda: _status(503), _UNAVAILABLE),
    _HttpFault('status-400', lambda: _status(400), _REJECTED),
    _HttpFault('gave-up-429', lambda: gave_up_on(_status(429)), _RATE_LIMITED, gave_up=True),
    _HttpFault('gave-up-503', lambda: gave_up_on(_status(503)), _UNAVAILABLE, gave_up=True),
    _HttpFault('gave-up-400', lambda: gave_up_on(_status(400)), _REJECTED, gave_up=True),
    _HttpFault('non-json-200', lambda: _body('application/json', b'not json'), _UNCLASSIFIED, streamed=False),
    _HttpFault('text-plain-200', lambda: _body('text/plain', b'upstream error'), _UNCLASSIFIED, streamed=False),
    _HttpFault(
        'body-breaks-off',
        lambda: _breaking_off(200, 'text/event-stream', httpx2.RemoteProtocolError),
        _CONNECTION,
        'read',
    ),
    _HttpFault(
        'error-body-breaks-off',
        lambda: _breaking_off(503, 'application/json', httpx2.RemoteProtocolError),
        _CONNECTION,
        'read',
    ),
    _HttpFault(
        'error-body-times-out', lambda: _breaking_off(503, 'application/json', httpx2.ReadTimeout), _TIMEOUT, 'read'
    ),
]


def _xai_chunk() -> object:
    chunk = chat_pb2.GetChatCompletionChunk(id='response-1', model='grok-4')
    chunk.outputs.append(
        chat_pb2.CompletionOutputChunk(
            index=0, delta=chat_pb2.Delta(content='Hel', role=chat_pb2.MessageRole.ROLE_ASSISTANT)
        )
    )
    return chunk


def _socket_closed() -> grpc.aio.AioRpcError:
    return _rpc_error('UNAVAILABLE', 'Stream removed (Socket closed)')


@dataclass(frozen=True)
class _XaiFault:
    id: str
    stub: Callable[[], XaiStub]
    categories: frozenset[type[ModelAPIError]]
    phase: TransportPhase | None = None
    streamed: bool | None = None


XAI_FAULTS = [
    _XaiFault(
        'connect',
        lambda: _XaiAnswer(
            error=lambda: _rpc_error(
                'UNAVAILABLE',
                'failed to connect to all addresses; last error: UNKNOWN: ipv4:127.0.0.1:1: '
                'Failed to connect to remote host: Connection refused',
            )
        ),
        _CONNECTION,
        'connect',
    ),
    # The gRPC client can't tell whether a request it timed out was sent.
    _XaiFault(
        'read-timeout', lambda: _XaiAnswer(error=lambda: _rpc_error('DEADLINE_EXCEEDED', 'Deadline Exceeded')), _TIMEOUT
    ),
    _XaiFault('status-429', lambda: _XaiAnswer(error=lambda: _rpc_error('RESOURCE_EXHAUSTED', 'fault')), _RATE_LIMITED),
    _XaiFault('status-503', lambda: _XaiAnswer(error=lambda: _rpc_error('UNAVAILABLE', 'fault')), _UNAVAILABLE),
    _XaiFault('status-400', lambda: _XaiAnswer(error=lambda: _rpc_error('INVALID_ARGUMENT', 'fault')), _REJECTED),
    _XaiFault('undecodable-response', lambda: _XaiAnswer(), _UNCLASSIFIED),
    _XaiFault('connection-lost', lambda: _XaiAnswer(error=_socket_closed), _CONNECTION, streamed=False),
    _XaiFault(
        'stream-breaks-off',
        lambda: _XaiAnswer(error=_socket_closed, chunks=(_xai_chunk,)),
        _CONNECTION,
        'read',
        streamed=True,
    ),
]


_ANSWER_UNREAD = frozenset({'cancel_suspended_response'})
"""Entry points that return nothing from the provider's answer, so a 200 that isn't JSON is no failure for them."""


def _applies(fault: _HttpFault | _XaiFault, cell: SdkCell) -> bool:
    if fault.streamed is not None and fault.streamed != (cell.entry_point == 'request_stream'):
        return False
    if fault.streamed is False and fault.categories == _UNCLASSIFIED and cell.entry_point in _ANSWER_UNREAD:
        return False
    if isinstance(fault, _HttpFault):
        return not isinstance(cell.case.transport, XaiTransport) and (
            not fault.gave_up or not isinstance(cell.case.transport, BotocoreTransport)
        )
    return isinstance(cell.case.transport, XaiTransport)


_DECISIONS_INVALID_RESPONSE = (
    "a 200 that isn't JSON raises `UnexpectedModelBehavior`, as the decision models raise for any body that doesn't "
    'answer the questions, and as `test_invalid_response` in `tests/models/test_openai_decisions.py` and '
    '`tests/models/test_google_decisions.py` pins; raising a `ModelAPIError` instead changes the type a caller catches'
)
_STREAMED_INTERNALLY = (
    "the request is streamed under the hood, so a 200 that isn't an event stream ends as `UnexpectedModelBehavior`, "
    'or with no error at all'
)
_CHAT_COMPLETIONS_TEXT = (
    'a text 200 raises `UnexpectedModelBehavior`, as `test_text_response` in `tests/models/test_openai.py` pins; '
    'raising a `ModelAPIError` instead changes the type a caller catches'
)
_CHAT_COMPLETIONS = (
    'BedrockMantleChatModel',
    'CerebrasModel',
    'CrusoeModel',
    'GitHubCopilotModel',
    'OllamaModel',
    'OpenAIChatModel',
    'OpenRouterModel',
    'SnowflakeModel',
    'ZaiModel',
)
_NOT_JSON = ('non-json-200', 'text-plain-200')

KNOWN_FAULT_GAPS: dict[tuple[str, str], str] = {
    **{
        (f'{model}.request', fault): _DECISIONS_INVALID_RESPONSE
        for model in ('GoogleDecisionsModel', 'OpenAIDecisionsModel', 'SystemOneModel', 'TypeSafeModel')
        for fault in _NOT_JSON
    },
    **{
        (f'{model}.request', fault): _STREAMED_INTERNALLY
        for model in ('AnthropicModel', 'OpenAICodexModel')
        for fault in _NOT_JSON
    },
    ('BedrockConverseModel.request', 'non-json-200'): _STREAMED_INTERNALLY,
    **{(f'{model}.request', 'text-plain-200'): _CHAT_COMPLETIONS_TEXT for model in _CHAT_COMPLETIONS},
}
"""Faults an entry point doesn't raise as its `ModelAPIError` on `main` yet, each pinned as a strict expected failure."""


def _known_gap(cell: SdkCell, fault_id: str) -> tuple[pytest.MarkDecorator, ...]:
    reason = KNOWN_FAULT_GAPS.get((cell.id, fault_id))
    return () if reason is None else (pytest.mark.xfail(strict=True, reason=reason),)


def test_known_fault_gaps_name_real_cells():
    """An entry in `KNOWN_FAULT_GAPS` names a cell that's sent that fault, so a renamed cell can't leave it behind."""
    cells = {
        (cell.id, fault.id) for cell in sdk_cells() for fault in [*HTTP_FAULTS, *XAI_FAULTS] if _applies(fault, cell)
    }
    classes, skipped = shipped_model_classes()
    if skipped:
        installed = {f'{name}.' for name in classes}
        known = {key for key in KNOWN_FAULT_GAPS if any(key[0].startswith(prefix) for prefix in installed)}
    else:
        known = set(KNOWN_FAULT_GAPS)
    assert known <= cells, sorted(known - cells)


def _fault_cells() -> list[object]:
    cells: list[object] = []
    for cell in sdk_cells():
        for fault in [*HTTP_FAULTS, *XAI_FAULTS]:
            if _applies(fault, cell):
                cells.append(pytest.param(cell, fault, id=f'{cell.id}-{fault.id}', marks=_known_gap(cell, fault.id)))
    return cells


async def _raised(cell: SdkCell, handler: Handler, stub: XaiStub) -> ModelAPIError:
    async with _open(cell.case, handler, stub) as model:
        with pytest.raises(ModelAPIError) as exc_info:
            await invoke(model, cell.entry_point, cell.case.parameters)
    return exc_info.value


def _unreachable_handler(request: httpx2.Request) -> httpx2.Response:
    raise AssertionError('an xAI fault was sent to an HTTP transport')  # pragma: no cover


@pytest.mark.parametrize(('cell', 'fault'), _fault_cells())
async def test_transport_fault_raises_model_api_error(
    allow_model_requests: None, cell: SdkCell, fault: _HttpFault | _XaiFault
):
    if isinstance(fault, _HttpFault):
        error = await _raised(cell, fault.handler(), _XaiAnswer())
    else:
        error = await _raised(cell, _unreachable_handler, fault.stub())
    assert _categories(error) == fault.categories
    if fault.phase is not None:
        assert isinstance(error, ModelConnectionError)
        assert error.phase == fault.phase


def _summary(error: ModelAPIError) -> tuple[object, ...]:
    """What a status a retrying transport gave up on must share with the provider SDK's error for the same response."""
    assert isinstance(error, ModelHTTPError)
    return (
        _categories(error),
        error.status_code,
        error.provider_error_code,
        error.provider_error_type,
        error.suggested_model_id,
        error.hint,
        error.retry_after,
    )


def _gave_up_cells() -> list[object]:
    cells: list[object] = []
    for cell in sdk_cells():
        # A legacy `httpx` retrying transport closes the response it gave up on without reading its body, so only the
        # status is left to classify.
        if isinstance(cell.case.transport, Httpx2Transport):
            for status in (429, 503, 400):
                cells.append(
                    pytest.param(cell, status, id=f'{cell.id}-{status}', marks=_known_gap(cell, f'gave-up-{status}'))
                )
    return cells


@pytest.mark.parametrize(('cell', 'status'), _gave_up_cells())
async def test_gave_up_status_is_raised_like_the_sdk_error(allow_model_requests: None, cell: SdkCell, status: int):
    """A status a retrying transport gave up on raises what the provider SDK raises for the same response."""
    sdk_error = await _raised(cell, _status(status), _XaiAnswer())
    gave_up_error = await _raised(cell, gave_up_on(_status(status)), _XaiAnswer())
    assert _summary(gave_up_error) == _summary(sdk_error)
