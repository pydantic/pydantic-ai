"""Shared plumbing for the harness's own JSON API clients over `httpx2`."""

from __future__ import annotations

import functools
import json
import os
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AsyncExitStack, suppress
from typing import ClassVar, ParamSpec, TypeVar

import anyio
import httpx2

from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai_harness._typeguards import is_object_dict

_P = ParamSpec('_P')
_R = TypeVar('_R')

_ACCOUNT_STATUSES = frozenset({401, 402, 403})


class MalformedResponse(Exception):
    """A 2xx response whose body is not the JSON the API documents."""


class JsonApiClient:
    """A JSON API client over `httpx2`, with a validated base URL, a request deadline, and a body size cap.

    Subclasses set `_DEFAULT_BASE_URL` and `_BASE_URL_ENV`, and pass the headers that authenticate them.
    """

    _DEFAULT_BASE_URL: ClassVar[str]
    _BASE_URL_ENV: ClassVar[str]
    # Both the httpx2 timeout on each connect, read, write, and pool wait, and
    # the deadline for the request as a whole.
    _TIMEOUT_SECONDS: ClassVar[float] = 30.0
    _MAX_RESPONSE_BYTES: ClassVar[int] = 10 * 1024 * 1024

    def __init__(
        self, *, base_url: str | None, http_client: httpx2.AsyncClient | None, headers: Mapping[str, str]
    ) -> None:
        base = (base_url if base_url is not None else os.environ.get(self._BASE_URL_ENV)) or self._DEFAULT_BASE_URL
        base = base.rstrip('/')
        source = '`base_url`' if base_url is not None else f'`{self._BASE_URL_ENV}`'
        parsed = httpx2.URL(base)
        # The base URL is developer-controlled, but a typo should fail here
        # rather than produce a plaintext request. Credentials are rejected
        # without echoing the URL: HTTPX puts the request URL in its error
        # messages, so they would otherwise reach logs and retry prompts.
        if parsed.userinfo:
            raise UserError(f'{source} must not contain credentials; pass the API key separately')
        loopback_http = parsed.scheme == 'http' and parsed.host in {'localhost', '127.0.0.1', '::1'}
        # The endpoint path is appended to the base URL, so a query or fragment,
        # even an empty one that parses as none at all, would land before it.
        if not parsed.host or '?' in base or '#' in base or not (parsed.scheme == 'https' or loopback_http):
            raise UserError(
                f'{source} must be an https:// URL, or http:// on loopback, with a host and no query or fragment, '
                f'got {base!r}'
            )
        self._base_url = base
        self._http_client = http_client
        self._headers = dict(headers)

    @staticmethod
    def _string_keyed(payload: object) -> dict[str, object]:
        """Narrow a decoded JSON value to a string-keyed mapping, or an empty one."""
        if not is_object_dict(payload):
            return {}
        return {str(key): value for key, value in payload.items()}

    async def _request(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, object] | None = None,
        params: Mapping[str, str] | None = None,
    ) -> object:
        """Send one request and decode its JSON body.

        The HTTPX timeout bounds each read, so a server that keeps sending a
        little at a time could hold the request open indefinitely; the deadline
        bounds the whole exchange. `base_url` can point at any server, so the
        body is read in chunks and abandoned once it passes `_MAX_RESPONSE_BYTES`.
        """
        # The response and the default client close in a shielded scope: an
        # expired anyio scope, this deadline or a caller's tool timeout, cancels
        # every await inside it, which would cut a transport's async close short.
        stack = AsyncExitStack()
        response: httpx2.Response | None = None
        body_complete = False
        try:
            # The default client lives for one request: nothing here would close
            # a shared one, and its connections belong to the event loop that
            # opened them, which each `run_sync` call replaces.
            client = (
                self._http_client
                if self._http_client is not None
                else await stack.enter_async_context(httpx2.AsyncClient(timeout=self._TIMEOUT_SECONDS))
            )
            with anyio.fail_after(self._TIMEOUT_SECONDS):
                # An injected client may follow redirects, and HTTPX keeps
                # credential headers on a redirect to another host, so the key
                # would reach wherever the server points.
                response = await stack.enter_async_context(
                    client.stream(
                        method,
                        f'{self._base_url}{path}',
                        json=payload,
                        params=params,
                        headers=self._headers,
                        follow_redirects=False,
                    )
                )
                response.raise_for_status()
                content = bytearray()
                async for chunk in response.aiter_bytes():
                    content += chunk
                    if len(content) > self._MAX_RESPONSE_BYTES:
                        raise httpx2.ReadError(
                            f'response body exceeds {self._MAX_RESPONSE_BYTES} bytes', request=response.request
                        )
                body_complete = True
        finally:
            with anyio.CancelScope(shield=True):
                # HTTPX2 closes the response from inside its body iterator, in whichever scope
                # cancelled that iteration, and marks it closed before awaiting. When that close
                # was cut short, the stack's close below is a no-op, so it is finished here.
                close_incomplete = response is not None and not body_complete and response.is_closed
                await stack.aclose()
                # TODO: drop this and its `suppress` once httpx2 completes an interrupted close
                # itself: https://github.com/pydantic/httpx2/issues/1326
                if close_incomplete and response is not None and isinstance(response.stream, httpx2.AsyncByteStream):
                    with suppress(Exception):
                        await response.stream.aclose()
        try:
            return json.loads(content)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            # `json.loads` decodes bytes before parsing them, so a body in a
            # broken encoding fails as a decode error, not a JSON one.
            raise MalformedResponse(str(error)) from error


def recoverable(service: str) -> Callable[[Callable[_P, Awaitable[_R]]], Callable[_P, Awaitable[_R]]]:
    """Convert transient failures of a `JsonApiClient` request into `ModelRetry`.

    Pydantic AI only feeds `ModelRetry` back to the model as a retry prompt;
    any other exception propagates and aborts the run. Rate limits, transient
    5xx, rejected parameters, a 404 on a dead URL, a request past its
    deadline, and a 2xx whose body is too large or malformed are things a model
    can recover from (wait, rephrase, pick another URL), so they become retries.
    A 401/402/403 means a bad key or an exhausted account, configuration the
    model cannot fix, so it propagates. The retry prompt names the status, not
    the request URL, which can carry a proxy's credentials.
    """

    def decorate(func: Callable[_P, Awaitable[_R]]) -> Callable[_P, Awaitable[_R]]:
        @functools.wraps(func)
        async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            try:
                return await func(*args, **kwargs)
            except httpx2.HTTPStatusError as error:
                response = error.response
                if response.status_code in _ACCOUNT_STATUSES:
                    raise
                raise ModelRetry(
                    f'{service} request failed with HTTP {response.status_code} {response.reason_phrase}.'
                ) from error
            except httpx2.HTTPError as error:
                raise ModelRetry(f'{service} request failed: {error}') from error
            except TimeoutError as error:
                raise ModelRetry(f'{service} request timed out.') from error
            except MalformedResponse as error:
                # Only `JsonApiClient` raises this, so any other error, such as
                # a custom client's bug or a key HTTPX cannot encode into a
                # header, propagates as configuration.
                raise ModelRetry(f'{service} returned a malformed response: {error}') from error

        return wrapper

    return decorate
