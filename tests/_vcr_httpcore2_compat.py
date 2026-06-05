# pyright: basic
# Mirror of vcr/stubs/httpcore_stubs.py (untyped upstream); keep `basic` so the
# transcribed code typechecks without re-deriving types vcrpy doesn't ship.
"""Make vcrpy intercept httpcore2 requests (the transport layer underneath httpx2).

vcrpy 8.x patches `httpcore` (v1) via `vcr/stubs/httpcore_stubs.py`. After we
swapped internal HTTP from `httpx` to `httpx2` (which uses `httpcore2`), VCR
silently bypasses tests that go through `pydantic_ai._ssrf.safe_download` and
friends — cassettes don't replay, live network calls happen, tests fail.

This module monkey-patches `vcr.patch.CassettePatcherBuilder` to also patch
`httpcore2.AsyncConnectionPool.handle_async_request` and
`httpcore2.ConnectionPool.handle_request` with the same VCR cassette hooks the
stock builder applies to httpcore 1.x. The stub functions are lifted verbatim
from `vcr/stubs/httpcore_stubs.py` (vcrpy 8.1.0) with the imports re-pointed at
httpcore2; the httpcore2 model API is intentionally compatible at this layer.

Tracked upstream at https://github.com/kevin1024/vcrpy/issues/990. Drop this
module once vcrpy ships httpcore2_stubs.py natively.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections import defaultdict
from collections.abc import AsyncIterable, Iterable

import httpcore2
import vcr.patch
from httpcore2 import Response
from httpcore2._models import ByteStream
from vcr.errors import CannotOverwriteExistingCassetteException
from vcr.filters import decode_response
from vcr.request import Request as VcrRequest
from vcr.serializers.compat import convert_body_to_bytes

_logger = logging.getLogger(__name__)


async def _convert_byte_stream(stream):
    if isinstance(stream, Iterable):
        return list(stream)
    if isinstance(stream, AsyncIterable):
        return [part async for part in stream]
    raise TypeError(
        f'_convert_byte_stream: stream must be Iterable or AsyncIterable, got {type(stream).__name__}',
    )


def _serialize_headers(real_response):
    headers = defaultdict(list)
    for name, value in real_response.headers:
        headers[name.decode('ascii')].append(value.decode('ascii'))
    return dict(headers)


async def _serialize_response(real_response):
    try:
        reason_phrase = real_response.extensions['reason_phrase'].decode('ascii')
    except KeyError:
        reason_phrase = None
    content = b''.join(await _convert_byte_stream(real_response.stream))
    real_response.stream = ByteStream(content)
    return {
        'status': {'code': real_response.status, 'message': reason_phrase},
        'headers': _serialize_headers(real_response),
        'body': {'string': content},
    }


def _deserialize_headers(headers):
    return [(name.encode('ascii'), value.encode('ascii')) for name, values in headers.items() for value in values]


def _deserialize_response(vcr_response):
    if 'status_code' in vcr_response:
        vcr_response = decode_response(
            convert_body_to_bytes(
                {
                    'headers': vcr_response['headers'],
                    'body': {'string': vcr_response['content']},
                    'status': {'code': vcr_response['status_code']},
                },
            ),
        )
        extensions = None
    else:
        extensions = (
            {'reason_phrase': vcr_response['status']['message'].encode('ascii')}
            if vcr_response['status']['message']
            else None
        )
    return Response(
        vcr_response['status']['code'],
        headers=_deserialize_headers(vcr_response['headers']),
        content=vcr_response['body']['string'],
        extensions=extensions,
    )


async def _make_vcr_request(real_request):
    body = b''.join(await _convert_byte_stream(real_request.stream))
    real_request.stream = ByteStream(body)
    uri = bytes(real_request.url).decode('ascii')
    headers = defaultdict(list)
    for name, value in real_request.headers:
        headers[name.decode('ascii')].append(value.decode('ascii'))
    headers = {name: ', '.join(values) for name, values in headers.items()}
    return VcrRequest(real_request.method.decode('ascii'), uri, body, headers)


async def _vcr_request(cassette, real_request):
    vcr_request = await _make_vcr_request(real_request)
    if cassette.can_play_response_for(vcr_request):
        return vcr_request, _play_responses(cassette, vcr_request)
    if cassette.write_protected and cassette.filter_request(vcr_request):
        raise CannotOverwriteExistingCassetteException(cassette=cassette, failed_request=vcr_request)
    _logger.info('%s not in cassette, sending to real server', vcr_request)
    return vcr_request, None


async def _record_responses(cassette, vcr_request, real_response):
    cassette.append(vcr_request, await _serialize_response(real_response))


def _play_responses(cassette, vcr_request):
    vcr_response = cassette.play_response(vcr_request)
    return _deserialize_response(vcr_response)


async def _vcr_handle_async_request(cassette, real_handle_async_request, self, real_request):
    vcr_request, vcr_response = await _vcr_request(cassette, real_request)
    if vcr_response:
        return vcr_response
    real_response = await real_handle_async_request(self, real_request)
    await _record_responses(cassette, vcr_request, real_response)
    return real_response


def _vcr_handle_async_request_factory(cassette, real_handle_async_request):
    @functools.wraps(real_handle_async_request)
    def _inner(self, real_request):
        return _vcr_handle_async_request(cassette, real_handle_async_request, self, real_request)

    return _inner


def _run_async_function(sync_func, *args, **kwargs):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(sync_func(*args, **kwargs))
    else:
        return asyncio.ensure_future(sync_func(*args, **kwargs))


def _vcr_handle_request(cassette, real_handle_request, self, real_request):
    vcr_request, vcr_response = _run_async_function(_vcr_request, cassette, real_request)
    if vcr_response:
        return vcr_response
    real_response = real_handle_request(self, real_request)
    _run_async_function(_record_responses, cassette, vcr_request, real_response)
    return real_response


def _vcr_handle_request_factory(cassette, real_handle_request):
    @functools.wraps(real_handle_request)
    def _inner(self, real_request):
        return _vcr_handle_request(cassette, real_handle_request, self, real_request)

    return _inner


_ORIG_HANDLE_ASYNC = httpcore2.AsyncConnectionPool.handle_async_request
_ORIG_HANDLE_SYNC = httpcore2.ConnectionPool.handle_request


def _httpcore2(self):
    new_async = _vcr_handle_async_request_factory(self._cassette, _ORIG_HANDLE_ASYNC)
    new_sync = _vcr_handle_request_factory(self._cassette, _ORIG_HANDLE_SYNC)
    yield httpcore2.AsyncConnectionPool, 'handle_async_request', new_async
    yield httpcore2.ConnectionPool, 'handle_request', new_sync


_httpcore2_decorated = vcr.patch.CassettePatcherBuilder._build_patchers_from_mock_triples_decorator(_httpcore2)  # pyright: ignore[reportArgumentType]

_orig_build = vcr.patch.CassettePatcherBuilder.build


def _patched_build(self):
    yield from _orig_build(self)
    yield from _httpcore2_decorated(self)


vcr.patch.CassettePatcherBuilder.build = _patched_build  # pyright: ignore[reportAttributeAccessIssue]
