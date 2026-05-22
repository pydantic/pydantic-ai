"""Tests for WebSocket cassette utilities."""

from __future__ import annotations as _annotations

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from websockets.exceptions import ConnectionClosedOK

    from .cassettes import (
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
    assert ws_cassette_plan(cassette_exists=False, record_mode='new_episodes') == 'record'


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
async def test_replay_recv_connection_closed() -> None:
    cassette = WebSocketCassette(interactions=[])
    ws = ReplayWebSocket(cassette)
    with pytest.raises(ConnectionClosedOK):
        await ws.recv()


@pytest.mark.anyio
async def test_replay_aiter_bytes_result() -> None:
    cassette = WebSocketCassette(
        interactions=[
            CassetteInteraction(direction='received', data={'type': 'test'}),
        ]
    )
    ws = ReplayWebSocket(cassette)
    original_recv = ws.recv

    async def bytes_recv(**kwargs: object) -> str | bytes:
        return await original_recv(decode=False)

    ws.recv = bytes_recv
    result = await ws.__anext__()
    assert isinstance(result, str)


@pytest.mark.anyio
async def test_replay_aiter_yields_str_frames() -> None:
    """Iterating a `ReplayWebSocket` yields each received frame as a JSON string, then stops."""
    cassette = WebSocketCassette(
        interactions=[
            CassetteInteraction(direction='received', data={'type': 'first'}),
            CassetteInteraction(direction='received', data={'type': 'second'}),
        ]
    )
    received = [frame async for frame in ReplayWebSocket(cassette)]
    assert received == ['{"type": "first"}', '{"type": "second"}']


class _FakeRealWebSocket:
    """Minimal real-WebSocket stand-in for exercising `RecordingWebSocket`."""

    transport_name = 'fake'  # reached via `RecordingWebSocket.__getattr__`

    def __init__(self, frames: list[str | bytes]) -> None:
        self._frames = iter(frames)
        self.closed = False

    async def send(self, message: str) -> None:
        pass

    async def recv(self, **kwargs: object) -> str | bytes:
        try:
            return next(self._frames)
        except StopIteration:
            raise StopAsyncIteration

    async def close(self, *, code: int = 1000, reason: str = '') -> None:
        self.closed = True


@pytest.mark.anyio
async def test_recording_websocket_records_and_delegates() -> None:
    """`RecordingWebSocket` records sent/received frames (bytes and str) and delegates unknown attributes."""
    cassette = WebSocketCassette()
    fake = _FakeRealWebSocket([b'{"type": "received_bytes"}', '{"type": "received_str"}'])
    ws = RecordingWebSocket(fake, cassette)

    await ws.send('{"type": "sent"}')
    frames = [frame async for frame in ws]
    await ws.close()

    assert frames == [b'{"type": "received_bytes"}', '{"type": "received_str"}']
    assert fake.closed is True
    assert ws.transport_name == 'fake'
    assert cassette.interactions == [
        CassetteInteraction(direction='sent', data={'type': 'sent'}),
        CassetteInteraction(direction='received', data={'type': 'received_bytes'}),
        CassetteInteraction(direction='received', data={'type': 'received_str'}),
    ]
