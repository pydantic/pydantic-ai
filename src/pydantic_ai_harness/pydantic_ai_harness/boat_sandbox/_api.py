"""Boat's v1 REST API over `httpx`: requests and typed answers, with no workspace semantics.

Boat answers every call with a JSON envelope, `{"ok": true, ...}` on success and
`{"ok": false, "code": ..., "message": ..., "error": {"details": ...}}` otherwise. Its documented
retry signals (`409 sandbox_starting`, `502 sandbox_direct_failed` while restoring, and
`error.details.retryable`) were checked against https://docs.boat.dev/api/v1 and the live API on
2026-10-21; the `boat_live` test tier re-checks them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, TypeVar

import anyio
import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic.alias_generators import to_camel

API_URL = 'https://boat.dev/api/v1'
_REQUEST_TIMEOUT = 30
# Between repeats of a call Boat answered with "not yet", and how long to keep repeating it.
_RETRY_DELAY = 1.0
_RETRY_WINDOW = 60.0

Retry = Literal['never', 'not_run', 'idempotent']


class BoatAPIError(Exception):
    """An error answer from Boat, or an answer that is not Boat's envelope (`code` `invalid_response`)."""

    def __init__(self, status: int, code: str, message: str, *, retryable: bool | None = None) -> None:
        super().__init__(f'Boat API returned HTTP {status} ({code}): {message}')
        self.status = status
        self.code = code
        self.message = message
        self.retryable = retryable


class _Model(BaseModel):
    model_config = ConfigDict(extra='ignore', alias_generator=to_camel, populate_by_name=True)


class Sandbox(_Model):
    id: str
    state: str


class SandboxResponse(_Model):
    sandbox: Sandbox


class CommandStarted(_Model):
    process_id: int


class CommandStatus(_Model):
    running: bool
    # Omitted from an answer read from byte offsets.
    known: bool = True
    status: str | None = None
    exit_code: int | None = None
    stdout: str = ''
    stderr: str = ''
    stdout_truncated: bool = False
    stderr_truncated: bool = False


class CommandFinished(_Model):
    exit_code: int
    stdout: str
    stderr: str
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    timed_out: bool = False


class FileRead(_Model):
    content: str


class TransferURL(_Model):
    url: str


class Empty(_Model):
    pass


class _ErrorDetails(_Model):
    retryable: bool | None = None


class _ErrorBody(_Model):
    details: _ErrorDetails = Field(default_factory=_ErrorDetails)


class _ErrorEnvelope(_Model):
    code: str
    message: str
    error: _ErrorBody = Field(default_factory=_ErrorBody)


class _TransferError(_Model):
    error: str
    message: str


_M = TypeVar('_M', bound=_Model)


def open_client(api_key: str) -> httpx.AsyncClient:
    """An API client that authenticates with `api_key`; the caller closes it."""
    return httpx.AsyncClient(
        base_url=API_URL,
        headers={'Authorization': f'Bearer {api_key}'},
        timeout=httpx.Timeout(_REQUEST_TIMEOUT),
    )


async def call(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    *,
    model: type[_M],
    json: Mapping[str, object] | None = None,
    params: Mapping[str, str | int] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float | None = None,
    retry: Retry = 'never',
) -> _M:
    """Send one API call and parse its answer into `model`.

    `retry` repeats the call only on answers Boat documents as having taken no effect: a sandbox
    that is still starting or being restored (`not_run`), and, for a call that is safe to repeat,
    any 502 Boat marks retryable (`idempotent`). Any other error answer raises `BoatAPIError`, and
    `httpx` errors propagate as they are.
    """
    deadline = anyio.current_time() + _RETRY_WINDOW
    while True:
        response = await client.request(
            method,
            path,
            json=json,
            params=params,
            headers=headers,
            timeout=_REQUEST_TIMEOUT if timeout is None else timeout,
        )
        if response.is_success:
            try:
                return model.model_validate_json(response.content)
            except ValidationError:
                raise BoatAPIError(response.status_code, 'invalid_response', 'Unexpected Boat response') from None
        error = _error(response)
        if not _retryable(error, retry) or anyio.current_time() >= deadline:
            raise error
        await anyio.sleep(_RETRY_DELAY)


async def transfer(
    client: httpx.AsyncClient, method: Literal['GET', 'PUT'], url: str, *, content: bytes | None = None
) -> bytes:
    """Send a request to a signed file-transfer URL and return the answer's body."""
    request = client.build_request(method, url, content=content, timeout=_REQUEST_TIMEOUT)
    # The URL carries its own signature and points at the sandbox's machine, not at Boat's API.
    request.headers.pop('Authorization', None)
    response = await client.send(request, auth=None)
    if response.is_success:
        return response.content
    try:
        body = _TransferError.model_validate_json(response.content)
    except ValidationError:
        raise BoatAPIError(response.status_code, 'invalid_response', 'Unexpected Boat response') from None
    raise BoatAPIError(response.status_code, body.error, body.message)


def _error(response: httpx.Response) -> BoatAPIError:
    try:
        envelope = _ErrorEnvelope.model_validate_json(response.content)
    except ValidationError:
        return BoatAPIError(response.status_code, 'invalid_response', 'Unexpected Boat response')
    return BoatAPIError(
        response.status_code, envelope.code, envelope.message, retryable=envelope.error.details.retryable
    )


def _retryable(error: BoatAPIError, retry: Retry) -> bool:
    if retry == 'never':
        return False
    if error.status == 409 and error.code == 'sandbox_starting':
        return True
    if error.status != 502:
        return False
    if error.code == 'sandbox_direct_failed' and 'sandbox_restoring' in error.message:
        return True
    return retry == 'idempotent' and error.retryable is True
