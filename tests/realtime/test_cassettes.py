"""Tests for realtime cassette utilities."""

from __future__ import annotations as _annotations

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from .cassettes import CassetteInteraction, RealtimeCassette, ReplayWebSocket, realtime_cassette_plan

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='websockets/pyyaml not installed'),
]


# ---------------------------------------------------------------------------
# realtime_cassette_plan tests
# ---------------------------------------------------------------------------


def test_plan_none_mode_with_cassette() -> None:
    assert realtime_cassette_plan(cassette_exists=True, record_mode=None) == 'replay'


def test_plan_none_mode_without_cassette() -> None:
    assert realtime_cassette_plan(cassette_exists=False, record_mode=None) == 'error_missing'


def test_plan_once_mode_with_cassette() -> None:
    assert realtime_cassette_plan(cassette_exists=True, record_mode='once') == 'replay'


def test_plan_once_mode_without_cassette() -> None:
    assert realtime_cassette_plan(cassette_exists=False, record_mode='once') == 'record'


def test_plan_rewrite_mode() -> None:
    assert realtime_cassette_plan(cassette_exists=True, record_mode='rewrite') == 'record'


def test_plan_all_mode() -> None:
    assert realtime_cassette_plan(cassette_exists=False, record_mode='all') == 'record'


def test_plan_new_episodes_mode() -> None:
    assert realtime_cassette_plan(cassette_exists=False, record_mode='new_episodes') == 'record'


def test_plan_unknown_mode_with_cassette() -> None:
    assert realtime_cassette_plan(cassette_exists=True, record_mode='unknown') == 'replay'


def test_plan_unknown_mode_without_cassette() -> None:
    assert realtime_cassette_plan(cassette_exists=False, record_mode='unknown') == 'error_missing'


# ---------------------------------------------------------------------------
# ReplayWebSocket tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_replay_recv_decode_false() -> None:
    cassette = RealtimeCassette(interactions=[
        CassetteInteraction(direction='received', data={'type': 'session.created'}),
    ])
    ws = ReplayWebSocket(cassette)
    result = await ws.recv(decode=False)
    assert isinstance(result, bytes)


@pytest.mark.anyio
async def test_replay_recv_connection_closed() -> None:
    from websockets.exceptions import ConnectionClosedOK

    cassette = RealtimeCassette(interactions=[])
    ws = ReplayWebSocket(cassette)
    with pytest.raises(ConnectionClosedOK):
        await ws.recv()


@pytest.mark.anyio
async def test_replay_aiter_bytes_result() -> None:
    """Cover the bytes decode branch in __anext__."""
    cassette = RealtimeCassette(interactions=[
        CassetteInteraction(direction='received', data={'type': 'test'}),
    ])
    ws = ReplayWebSocket(cassette)
    # Force recv to return bytes by patching
    original_recv = ws.recv

    async def bytes_recv(**kwargs: object) -> str | bytes:
        return (await original_recv(decode=False))

    ws.recv = bytes_recv  # type: ignore[assignment]
    result = await ws.__anext__()
    assert isinstance(result, str)
