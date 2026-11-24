"""The parts of the Runta API the sandbox backend uses: runtime lifecycle over REST, commands over the exec WebSocket.

The published `runta-sdk` (0.2.7) returns a command's buffered output once it ends, and reports a
timeout as exit code 124, which a command can also exit with. The exec WebSocket it wraps streams
output, takes signals, and ends the command when the socket closes, so the backend speaks that
protocol here.

A runtime is addressed by its UUID: the API refuses a name in a runtime path.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Literal

import httpx
from pydantic import Base64Bytes, BaseModel, ValidationError
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

DEFAULT_ENDPOINT = 'https://api.runta.com'

_REQUEST_TIMEOUT = 30.0
_HANDSHAKE_TIMEOUT = 30.0


class RuntaError(Exception):
    """The Runta API refused a request or broke its protocol."""


class RuntaAPIError(RuntaError):
    """A REST call answered with an error status."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f'Runta API answered {status}: {message}')
        self.status = status


class ExecRefused(RuntaError):
    """The exec handshake was refused, so the command was not dispatched."""

    def __init__(self, status: int) -> None:
        super().__init__(f'Runta refused the exec connection with HTTP {status}')
        self.status = status


class ExecLost(RuntaError):
    """The exec connection ended before the command reported an exit code; the command may have run."""


class OutputLimitExceeded(RuntaError):
    """The command printed more than the backend keeps."""


@dataclass(frozen=True, kw_only=True)
class Runtime:
    id: str
    name: str
    status: str
    revision: int


class _RuntimeData(BaseModel):
    id: str
    display_name: str
    status: str
    revision: int


class _RuntimeEnvelope(BaseModel):
    data: _RuntimeData


class _Frame(BaseModel):
    type: str
    data_base64: Base64Bytes = b''
    code: int = 0
    message: str = ''


class RuntaExec:
    """One command on the exec WebSocket, from the start frame to its exit code."""

    def __init__(self, connection: ClientConnection) -> None:
        self._connection = connection
        self.stdout = bytearray()
        self.stderr = bytearray()

    async def wait(self, *, limit: int) -> int:
        """Collect output until the command exits, and return its exit code.

        Raises:
            OutputLimitExceeded: The command printed more than `limit` bytes.
            ExecLost: The connection ended without an exit code.
            RuntaError: Runta reported that the command could not be run.
        """
        try:
            async for message in self._connection:
                frame = _Frame.model_validate_json(message)
                if frame.type == 'stdout':
                    self.stdout += frame.data_base64
                elif frame.type == 'stderr':
                    self.stderr += frame.data_base64
                elif frame.type == 'exit':
                    return frame.code
                elif frame.type == 'error':
                    raise RuntaError(f'Runta could not run the command: {frame.message}')
                if len(self.stdout) + len(self.stderr) > limit:
                    raise OutputLimitExceeded
        except (ConnectionClosed, ValidationError) as error:
            raise ExecLost('The Runta exec connection broke before the command exited') from error
        raise ExecLost('The Runta exec connection closed before the command exited')

    async def close(self) -> None:
        """Close the socket. Runta ends a command that is still running when its socket closes."""
        await self._connection.close()


class RuntaClient:
    """An authenticated Runta API client, used as an async context manager for one operation."""

    def __init__(self, *, token: str, endpoint: str = DEFAULT_ENDPOINT) -> None:
        self._token = token
        self._endpoint = endpoint.rstrip('/')
        self._http = httpx.AsyncClient(
            base_url=self._endpoint,
            headers={'Authorization': f'Bearer {token}', 'User-Agent': 'pydantic-ai-harness'},
            timeout=_REQUEST_TIMEOUT,
        )

    async def __aenter__(self) -> RuntaClient:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        await self._http.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Mapping[str, object] | None = None,
        params: Mapping[str, int] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        response = await self._http.request(method, path, json=json, params=params, headers=headers)
        if response.is_error:
            raise RuntaAPIError(response.status_code, response.text[:500])
        return response

    async def create_runtime(
        self, name: str, *, image: str | None, vcpus: int | None, memory_mib: int | None
    ) -> Runtime:
        """Create a runtime called `name`. Repeating the call returns the same runtime instead of a second one."""
        requests = {key: value for key, value in (('vcpus', vcpus), ('memory_mib', memory_mib)) if value is not None}
        body: dict[str, object] = {'name': name}
        if image is not None:
            body['image'] = {'id': image}
        if requests:
            body['resources'] = {'requests': requests}
        # The key makes a retried create idempotent, so a lost response cannot leave a second runtime behind.
        response = await self._request('POST', '/v2/runtimes', json=body, headers={'Idempotency-Key': name})
        return _runtime(response)

    async def get_runtime(self, runtime_id: str) -> Runtime:
        return _runtime(await self._request('GET', f'/v2/runtimes/{runtime_id}'))

    async def set_running(self, runtime: Runtime, action: Literal['start', 'resume']) -> None:
        """Start a shut-down runtime, or resume a paused or suspended one. Returns before it is running."""
        await self._request(
            'POST', f'/v2/runtimes/{runtime.id}/{action}', params={'expected_revision': runtime.revision}
        )

    async def delete_runtime(self, runtime_id: str) -> None:
        """Delete a runtime. One that no longer exists is already deleted."""
        try:
            runtime = await self.get_runtime(runtime_id)
            await self._request('DELETE', f'/v2/runtimes/{runtime_id}', params={'expected_revision': runtime.revision})
        except RuntaAPIError as error:
            if error.status != 404:
                raise

    async def exec(self, runtime_id: str, argv: Sequence[str], env: Mapping[str, str]) -> RuntaExec:
        """Start `argv` with stdin at EOF.

        Raises:
            ExecRefused: Runta refused the handshake, so the command was not dispatched.
        """
        url = f'{self._endpoint.replace("http", "ws", 1)}/v2/runtimes/{runtime_id}/exec/stream'
        try:
            connection = await connect(
                url,
                additional_headers={'Authorization': f'Bearer {self._token}'},
                open_timeout=_HANDSHAKE_TIMEOUT,
                # Output frames are bounded by the caller's own limit, not by the message size.
                max_size=None,
            )
        except InvalidStatus as error:
            raise ExecRefused(error.response.status_code) from error
        start = {'type': 'start', 'command': argv[0], 'args': list(argv[1:]), 'env': dict(env), 'tty': False}
        await connection.send(json.dumps(start))
        await connection.send(json.dumps({'type': 'close_stdin'}))
        return RuntaExec(connection)


def _runtime(response: httpx.Response) -> Runtime:
    try:
        data = _RuntimeEnvelope.model_validate_json(response.content).data
    except ValidationError as error:
        raise RuntaError('Runta returned a runtime in an unexpected shape') from error
    return Runtime(id=data.id, name=data.display_name, status=data.status, revision=data.revision)
