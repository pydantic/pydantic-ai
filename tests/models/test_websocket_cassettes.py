"""Tests for WebSocket cassette utilities."""

from __future__ import annotations as _annotations

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

    from .websocket_cassettes import (
        CassetteInteraction,
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
async def test_replay_send_matches_recorded_frame() -> None:
    cassette = WebSocketCassette(
        interactions=[
            CassetteInteraction(direction='sent', data={'type': 'response.create', 'model': 'gpt-4o-mini'}),
        ]
    )
    ws = ReplayWebSocket(cassette)
    await ws.send('{"type": "response.create", "model": "gpt-4o-mini"}')


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
async def test_replay_recv_connection_closed() -> None:
    cassette = WebSocketCassette(interactions=[])
    ws = ReplayWebSocket(cassette)
    with pytest.raises(ConnectionClosedOK):
        await ws.recv()


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
