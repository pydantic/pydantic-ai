"""Tests for WebSocket cassette utilities."""

from __future__ import annotations as _annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

    from .conftest import _RecordingConnect  # pyright: ignore[reportPrivateUsage]
    from .websocket_cassettes import (
        CassetteInteraction,
        RecordingWebSocket,
        ReplayWebSocket,
        WebSocketCassette,
        ws_cassette_plan,
    )

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='websockets/pyyaml not installed'),
]


def test_plan_none_mode_with_cassette() -> None:
    assert ws_cassette_plan(cassette_exists=True, record_mode=None) == 'replay'


def test_plan_none_mode_without_cassette() -> None:
    assert ws_cassette_plan(cassette_exists=False, record_mode=None) == 'error_missing'


def test_plan_once_mode_with_cassette() -> None:
    assert ws_cassette_plan(cassette_exists=True, record_mode='once') == 'replay'


def test_plan_once_mode_without_cassette() -> None:
    assert ws_cassette_plan(cassette_exists=False, record_mode='once') == 'record'


def test_plan_rewrite_mode() -> None:
    assert ws_cassette_plan(cassette_exists=True, record_mode='rewrite') == 'record'


def test_plan_all_mode() -> None:
    assert ws_cassette_plan(cassette_exists=False, record_mode='all') == 'record'


def test_plan_new_episodes_mode() -> None:
    assert ws_cassette_plan(cassette_exists=False, record_mode='new_episodes') == 'error_unsupported'


def test_plan_unknown_mode_with_cassette() -> None:
    assert ws_cassette_plan(cassette_exists=True, record_mode='unknown') == 'replay'


def test_plan_unknown_mode_without_cassette() -> None:
    assert ws_cassette_plan(cassette_exists=False, record_mode='unknown') == 'error_missing'


@pytest.mark.anyio
async def test_replay_recv_decode_false() -> None:
    cassette = WebSocketCassette(
        interactions=[
            CassetteInteraction(direction='received', data={'type': 'session.created'}),
        ]
    )
    ws = ReplayWebSocket(cassette)
    result = await ws.recv(decode=False)
    assert isinstance(result, bytes)


@pytest.mark.anyio
async def test_replay_send_matches_recorded_frame() -> None:
    cassette = WebSocketCassette(
        interactions=[
            CassetteInteraction(direction='sent', data={'type': 'response.create', 'model': 'gpt-4o-mini'}),
        ]
    )
    ws = ReplayWebSocket(cassette)
    await ws.send('{"type": "response.create", "model": "gpt-4o-mini"}')


@pytest.mark.anyio
async def test_replay_preserves_interaction_order() -> None:
    cassette = WebSocketCassette(
        interactions=[
            CassetteInteraction(direction='sent', data={'type': 'response.create'}),
            CassetteInteraction(direction='received', data={'type': 'response.completed'}),
        ]
    )
    ws = ReplayWebSocket(cassette)

    receive = asyncio.create_task(ws.recv())
    await asyncio.sleep(0)
    assert not receive.done()

    await ws.send('{"type": "response.create"}')
    assert await receive == b'{"type": "response.completed"}'
    assert ws.interactions_consumed


@pytest.mark.anyio
async def test_replay_send_mismatch_shows_expected_and_actual() -> None:
    cassette = WebSocketCassette(
        interactions=[
            CassetteInteraction(direction='sent', data={'type': 'response.create', 'model': 'gpt-4o-mini'}),
        ]
    )
    ws = ReplayWebSocket(cassette)

    with pytest.raises(AssertionError) as exc_info:
        await ws.send('{"type": "response.create", "model": "gpt-5"}')
    message = str(exc_info.value)
    assert 'Expected:' in message
    assert 'gpt-4o-mini' in message
    assert 'Actual:' in message
    assert 'gpt-5' in message


@pytest.mark.anyio
async def test_replay_rejects_send_before_recorded_receive() -> None:
    cassette = WebSocketCassette(
        interactions=[CassetteInteraction(direction='received', data={'type': 'response.created'})]
    )
    ws = ReplayWebSocket(cassette)

    with pytest.raises(AssertionError, match="before recorded 'received' interaction"):
        await ws.send('{"type": "response.create"}')


@pytest.mark.anyio
async def test_replay_send_unexpected_frame() -> None:
    cassette = WebSocketCassette(
        interactions=[
            CassetteInteraction(direction='sent', data={'type': 'response.create', 'model': 'gpt-4o-mini'}),
        ]
    )
    ws = ReplayWebSocket(cassette)

    await ws.send('{"type": "response.create", "model": "gpt-4o-mini"}')
    with pytest.raises(AssertionError, match='Unexpected WebSocket send'):
        await ws.send('{"type": "response.create", "model": "gpt-4o-mini"}')


@pytest.mark.anyio
async def test_replay_recv_connection_closed() -> None:
    cassette = WebSocketCassette(interactions=[])
    ws = ReplayWebSocket(cassette)
    with pytest.raises(ConnectionClosedOK):
        await ws.recv()


@pytest.mark.anyio
async def test_replay_close_unblocks_receive_waiting_for_send() -> None:
    cassette = WebSocketCassette(interactions=[CassetteInteraction(direction='sent', data={'type': 'response.create'})])
    ws = ReplayWebSocket(cassette)

    receive = asyncio.create_task(ws.recv())
    await asyncio.sleep(0)
    await ws.close()

    with pytest.raises(ConnectionClosedOK):
        await receive


@pytest.mark.anyio
async def test_replay_recv_abnormal_close() -> None:
    cassette = WebSocketCassette(
        interactions=[
            CassetteInteraction(direction='closed', data={'code': 1011, 'reason': 'upstream error'}),
        ]
    )
    ws = ReplayWebSocket(cassette)
    with pytest.raises(ConnectionClosedError):
        await ws.recv()


class _FakeRealWebSocket:
    transport_name = 'fake transport'

    def __init__(self, received: list[str | bytes]) -> None:
        self.received = iter(received)
        self.sent: list[str | bytes] = []
        self.close_args: tuple[int, str] | None = None

    async def send(self, message: str | bytes) -> None:
        self.sent.append(message)

    async def recv(self, **kwargs: Any) -> str | bytes:
        return next(self.received)

    async def close(self, *, code: int = 1000, reason: str = '') -> None:
        self.close_args = code, reason


@pytest.mark.anyio
async def test_recording_websocket_and_cassette_roundtrip(tmp_path: Path) -> None:
    cassette = WebSocketCassette(synthetic=True)
    real = _FakeRealWebSocket([b'{"type": "received_bytes"}', '{"type": "received_text"}'])
    ws = RecordingWebSocket(real, cassette)

    await ws.send('{"type": "sent_text"}')
    await ws.send(b'{"type": "sent_bytes"}')
    assert await ws.recv(decode=False) == b'{"type": "received_bytes"}'
    assert await ws.recv() == '{"type": "received_text"}'
    assert ws.transport_name == 'fake transport'
    await ws.close(code=1001, reason='done')

    assert real.sent == ['{"type": "sent_text"}', b'{"type": "sent_bytes"}']
    assert real.close_args == (1001, 'done')
    assert cassette.interactions == [
        CassetteInteraction(direction='sent', data={'type': 'sent_text'}),
        CassetteInteraction(direction='sent', data={'type': 'sent_bytes'}),
        CassetteInteraction(direction='received', data={'type': 'received_bytes'}),
        CassetteInteraction(direction='received', data={'type': 'received_text'}),
    ]

    path = tmp_path / 'nested' / 'cassette.yaml'
    cassette.dump(path)
    assert WebSocketCassette.load(path) == cassette


class _AwaitableWebSocket:
    def __init__(self, websocket: _FakeRealWebSocket) -> None:
        self.websocket = websocket

    def __await__(self) -> Any:
        async def resolve() -> _FakeRealWebSocket:
            return self.websocket

        return resolve().__await__()


@pytest.mark.anyio
async def test_recording_connect_wraps_and_closes_real_socket() -> None:
    cassette = WebSocketCassette()
    real = _FakeRealWebSocket([])

    with patch('tests.models.conftest._real_ws_connect', return_value=_AwaitableWebSocket(real)) as connect:
        manager = _RecordingConnect('wss://example.test', option=True).with_cassette(cassette)
        async with manager as ws:
            await ws.send('{"type": "sent"}')

    connect.assert_called_once_with('wss://example.test', option=True)
    assert real.close_args == (1000, '')
    assert cassette.interactions == [CassetteInteraction(direction='sent', data={'type': 'sent'})]
