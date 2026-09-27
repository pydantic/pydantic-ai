from __future__ import annotations

import asyncio
import importlib.util
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

_HAS_SPRITES = importlib.util.find_spec('sprites') is not None
collect_ignore = (
    []
    if _HAS_SPRITES
    else ['fake_sprites.py', 'test_conformance.py', 'test_sprites_sandbox.py', 'test_sprites_live.py']
)

if TYPE_CHECKING or _HAS_SPRITES:  # pragma: no branch - installed and slim jobs take opposite branches
    from sprites import AsyncSprite, AsyncSpritesClient
    from sprites.async_filesystem import AsyncSpritePath
    from sprites.types import FileStat

    from .fake_sprites import SpriteTransport


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def live_token() -> str:
    """Return `SPRITE_TOKEN` for the live tier, or skip it.

    The tier runs only with `PYDANTIC_AI_HARNESS_SPRITES_LIVE=1` and a non-empty token. CI also
    sets `SPRITES_REQUIRE_LIVE`, which turns the skip into a failure, so a missing secret cannot
    pass as a green run that tested nothing.
    """
    token = os.getenv('SPRITE_TOKEN')
    if os.getenv('PYDANTIC_AI_HARNESS_SPRITES_LIVE') == '1' and token:
        return token
    reason = 'requires PYDANTIC_AI_HARNESS_SPRITES_LIVE=1 and a non-empty SPRITE_TOKEN'
    if os.getenv('SPRITES_REQUIRE_LIVE', '').lower() in {'1', 'true', 'yes'}:
        pytest.fail(reason)
    pytest.skip(reason)


@pytest.fixture(scope='session')
def sprites_token() -> str:
    """Session-scoped so it skips before any live fixture that would use the token."""
    return live_token()


if _HAS_SPRITES:  # pragma: no branch - the fixture requires the SDK-backed fake

    @pytest.fixture
    async def transport(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AsyncIterator[SpriteTransport]:
        transport = SpriteTransport(tmp_path)
        monkeypatch.setenv('SPRITE_TOKEN', 'test-token')
        monkeypatch.setattr('pydantic_ai_harness.sprites_sandbox._backend.AsyncSpritesClient', transport.client)
        # `WSCommand` opens the exec WebSocket through the `connect` it imports from `websockets`.
        monkeypatch.setattr('sprites.websocket.connect', transport.connect)

        async def get(client: AsyncSpritesClient, name: str) -> AsyncSprite:
            return await transport.get(client, name)

        async def create(client: AsyncSpritesClient, name: str, *, runtime: str | None) -> AsyncSprite:
            return await transport.create(client, name, runtime=runtime)

        monkeypatch.setattr(AsyncSpritesClient, 'get_sprite', get)
        monkeypatch.setattr(AsyncSpritesClient, 'create_sprite', create)

        async def close(client: AsyncSpritesClient) -> None:
            await transport.close(client)

        monkeypatch.setattr(AsyncSpritesClient, 'aclose', close)

        async def destroy(client: AsyncSpritesClient, name: str) -> None:
            await transport.destroy(client, name)

        monkeypatch.setattr(AsyncSpritesClient, 'destroy_sprite', destroy)

        async def fs_stat(path: AsyncSpritePath) -> FileStat:
            return await transport.fs_stat(path)

        async def fs_write(path: AsyncSpritePath, data: bytes, mode: int = 0o644, mkdir_parents: bool = True) -> None:
            await transport.fs_write(path, data, mode)

        monkeypatch.setattr(AsyncSpritePath, 'stat', fs_stat)
        monkeypatch.setattr(AsyncSpritePath, 'write_bytes', fs_write)
        yield transport
        for client in transport.clients:
            await client.aclose()
        # Each exec's thread posts its last frame to this loop, so it must finish while the loop runs.
        for socket in transport.execs:
            await asyncio.to_thread(socket.thread.join, 10)
