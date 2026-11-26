"""A peer's reset during Windows Proactor socket teardown neither prints a traceback nor skips the cleanup.

CPython's own `_call_connection_lost` runs on a stand-in socket, since only Windows sockets raise `WinError 10054`.
"""

from __future__ import annotations

import asyncio
import socket
import sys
from asyncio import proactor_events
from typing import cast

import pytest

from pydantic_ai_pax.__main__ import quiet_proactor_resets


def _reset(winerror: int) -> ConnectionResetError:
    error = ConnectionResetError(winerror, 'An existing connection was forcibly closed by the remote host')
    setattr(error, 'winerror', winerror)
    return error


class _Pipe:
    """A subprocess pipe handle, which has no `shutdown`."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _Socket(socket.socket):
    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self.error = error
        self.closed = False

    def shutdown(self, how: int) -> None:
        raise self.error

    def close(self) -> None:
        self.closed = True
        super().close()


class _Protocol(asyncio.BaseProtocol):
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error

    def connection_lost(self, exc: Exception | None) -> None:
        if self.error is not None:
            raise self.error


class _Server:
    def __init__(self) -> None:
        self.detached = False

    def _detach(self, *transport: object) -> None:
        self.detached = True


def _teardown(
    sock: _Socket | _Pipe, protocol: _Protocol, *, calls: int = 1
) -> tuple[list[BaseException], dict[str, object]]:
    """Run the transport's connection-lost callback on an event loop; return the loop's errors and the end state."""
    transport_class = cast(type[object], vars(proactor_events)['_ProactorBasePipeTransport'])
    transport: object = object.__new__(transport_class)
    server = _Server()
    vars(transport).update(_called_connection_lost=False, _protocol=protocol, _sock=sock, _server=server)
    errors: list[BaseException] = []
    loop = asyncio.new_event_loop()
    loop.set_exception_handler(lambda _, context: errors.append(context['exception']))
    try:
        for _ in range(calls):
            loop.call_soon(getattr(transport, '_call_connection_lost'), None)
        loop.run_until_complete(asyncio.sleep(0))
    finally:
        loop.close()
    state: dict[str, object] = {**vars(transport), 'detached': server.detached, 'closed': sock.closed}
    vars(transport)['_sock'] = None
    sock.close()
    return errors, state


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, 'platform', 'win32')


@pytest.mark.usefixtures('windows')
def test_reset_on_shutdown_finishes_teardown_quietly() -> None:
    with quiet_proactor_resets():
        errors, state = _teardown(_Socket(_reset(10054)), _Protocol(), calls=2)
    assert errors == []
    assert state['closed']
    assert state['_sock'] is None
    assert state['detached']
    assert state['_called_connection_lost']


@pytest.mark.usefixtures('windows')
@pytest.mark.parametrize('error', [_reset(10053), OSError('boom')], ids=['other-winerror', 'other-error'])
def test_other_socket_errors_still_reach_the_loop(error: OSError) -> None:
    sock = _Socket(error)
    with quiet_proactor_resets():
        errors, state = _teardown(sock, _Protocol())
    assert errors == [error]
    assert state['_sock'] is sock


@pytest.mark.usefixtures('windows')
def test_protocol_reset_still_reaches_the_loop() -> None:
    error = _reset(10054)
    with quiet_proactor_resets():
        errors, state = _teardown(_Socket(_reset(10054)), _Protocol(error))
    assert errors == [error]
    assert state['closed']
    assert state['_called_connection_lost']


@pytest.mark.usefixtures('windows')
def test_pipes_still_close() -> None:
    with quiet_proactor_resets():
        errors, state = _teardown(_Pipe(), _Protocol())
    assert errors == []
    assert state['closed']
    assert state['_called_connection_lost']


def test_other_platforms_are_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, 'platform', 'linux')
    transport_class = cast(type[object], vars(proactor_events)['_ProactorBasePipeTransport'])
    original = vars(transport_class)['_call_connection_lost']
    with quiet_proactor_resets():
        assert vars(transport_class)['_call_connection_lost'] is original
