"""Windows ProactorEventLoop socket teardown suppresses WinError 10054 without hiding other errors."""

from __future__ import annotations

import sys
from asyncio import proactor_events
from collections.abc import Callable
from typing import cast

import pytest

from pydantic_ai_pax.__main__ import quiet_windows_proactor_reset


class _DummySock:
    def __init__(self, *, shutdown_error: BaseException | None = None) -> None:
        self.shutdown_error = shutdown_error
        self.closed = False

    def fileno(self) -> int:
        return 123

    def shutdown(self, _how: int) -> None:
        if self.shutdown_error is not None:
            raise self.shutdown_error

    def close(self) -> None:
        self.closed = True


class _DummyProtocol:
    def __init__(self, *, error: BaseException | None = None) -> None:
        self.error = error

    def connection_lost(self, _exc: BaseException | None) -> None:
        if self.error is not None:
            raise self.error


class _DummyServer:
    def __init__(self) -> None:
        self.detached: list[object] = []

    def _detach(self, transport: object) -> None:
        self.detached.append(transport)


class _FakeTransport:
    def __init__(
        self,
        *,
        sock: _DummySock | None,
        protocol: _DummyProtocol,
        server: _DummyServer | None = None,
    ) -> None:
        self._called_connection_lost = False
        self._sock = sock
        self._protocol = protocol
        self._server = server

    @property
    def sock(self) -> _DummySock | None:
        return self._sock

    @property
    def server(self) -> _DummyServer | None:
        return self._server

    @property
    def called_connection_lost(self) -> bool:
        return self._called_connection_lost

    def _call_connection_lost(self, exc: BaseException | None) -> None:
        if self._called_connection_lost:
            return
        try:
            self._protocol.connection_lost(exc)
        finally:
            if self._sock is not None and self._sock.fileno() != -1:
                self._sock.shutdown(2)
            if self._sock is not None:
                self._sock.close()
            self._sock = None
            if self._server is not None:
                detach = cast(Callable[[object], None], getattr(self._server, '_detach'))
                detach(self)
                self._server = None
            self._called_connection_lost = True


def _win_reset_error(winerror: int = 10054) -> ConnectionResetError:
    err = ConnectionResetError(winerror, f'simulated winerror {winerror}')
    err.winerror = winerror
    return err


def _invoke_call_connection_lost(transport: _FakeTransport, exc: BaseException | None = None) -> None:
    method = cast(
        Callable[[_FakeTransport, BaseException | None], None],
        getattr(_FakeTransport, '_call_connection_lost'),
    )
    method(transport, exc)


def test_suppresses_winerror_10054_on_shutdown_and_finishes_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(
        proactor_events,
        '_ProactorBasePipeTransport',
        _FakeTransport,
        raising=False,
    )
    sock = _DummySock(shutdown_error=_win_reset_error(10054))
    server = _DummyServer()
    transport = _FakeTransport(sock=sock, protocol=_DummyProtocol(), server=server)

    with quiet_windows_proactor_reset():
        _invoke_call_connection_lost(transport)

    assert sock.closed is True
    assert transport.sock is None
    assert server.detached == [transport]
    assert transport.server is None
    assert transport.called_connection_lost is True


def test_reraises_non_10054_connection_reset_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(
        proactor_events,
        '_ProactorBasePipeTransport',
        _FakeTransport,
        raising=False,
    )
    sock = _DummySock(shutdown_error=_win_reset_error(10053))
    transport = _FakeTransport(sock=sock, protocol=_DummyProtocol())

    with quiet_windows_proactor_reset(), pytest.raises(ConnectionResetError) as exc_info:
        _invoke_call_connection_lost(transport)

    assert exc_info.value.winerror == 10053


def test_reraises_winerror_10054_from_protocol_connection_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(
        proactor_events,
        '_ProactorBasePipeTransport',
        _FakeTransport,
        raising=False,
    )
    sock = _DummySock()
    transport = _FakeTransport(
        sock=sock,
        protocol=_DummyProtocol(error=_win_reset_error(10054)),
    )

    with quiet_windows_proactor_reset(), pytest.raises(ConnectionResetError) as exc_info:
        _invoke_call_connection_lost(transport)

    assert exc_info.value.winerror == 10054
    assert sock.closed is True
    assert transport.called_connection_lost is True


def test_restores_original_method_on_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(
        proactor_events,
        '_ProactorBasePipeTransport',
        _FakeTransport,
        raising=False,
    )
    original = cast(object, getattr(_FakeTransport, '_call_connection_lost'))

    with quiet_windows_proactor_reset():
        assert cast(object, getattr(_FakeTransport, '_call_connection_lost')) is not original

    assert cast(object, getattr(_FakeTransport, '_call_connection_lost')) is original


def test_noop_on_non_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr(
        proactor_events,
        '_ProactorBasePipeTransport',
        _FakeTransport,
        raising=False,
    )
    original = cast(object, getattr(_FakeTransport, '_call_connection_lost'))

    with quiet_windows_proactor_reset():
        assert cast(object, getattr(_FakeTransport, '_call_connection_lost')) is original
