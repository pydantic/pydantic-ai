"""Pydantic AI's workspace backend conformance suite, run against `SpritesSandboxBackend`.

`TestFakeSpritesSandboxBackend` runs everywhere, over the fake Sprites SDK, whose commands run in
local subprocesses under a temporary host directory. `TestLiveSpritesSandboxBackend` runs the same
rules against a real Sprite and is gated like `test_sprites_live.py`.

The backend implements `SupportsCommands` and `SupportsFilesystem`, so the suite's command,
filesystem, reattach, and destroy rules all run here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable

import pytest

from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite
from pydantic_ai_harness.sprites_sandbox import SpritesSandboxBackend

from .fake_sprites import SpriteTransport


def _attach(ref: WorkspaceRef) -> WorkspaceBackend:
    return SpritesSandboxBackend(ref=ref)


async def _delete(backend: WorkspaceBackend) -> None:  # pragma: no cover - live tier only
    assert isinstance(backend, SpritesSandboxBackend)
    sprite = await backend.get_sandbox()
    await sprite.delete()


async def _teardown(backend: SpritesSandboxBackend) -> None:  # pragma: no cover - live tier only
    if backend.ref is not None:
        from sprites.exceptions import NotFoundError

        try:
            await _delete(backend)
        except NotFoundError:
            pass
    await backend.aclose()


class TestFakeSpritesSandboxBackend(WorkspaceBackendSuite):
    @pytest.fixture
    def backend(self, transport: SpriteTransport) -> SpritesSandboxBackend:
        del transport
        return SpritesSandboxBackend()

    @pytest.fixture
    def can_detect_exit_with_inherited_output_pipes(self) -> bool:
        # The Sprite exec socket withholds EXIT until descendants close inherited pipes.
        return False

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return _attach

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _delete


@pytest.mark.sprites_live
@pytest.mark.usefixtures('sprites_token')
class TestLiveSpritesSandboxBackend(WorkspaceBackendSuite):  # pragma: no cover - live tier runs without coverage
    # One event loop for the class: the backend's `AsyncSpritesClient` holds an `httpx.AsyncClient`
    # whose pooled connections are bound to the loop they were opened on. A class-scoped async
    # fixture is what holds that loop open between tests.
    @pytest.fixture(scope='class')
    @classmethod
    def anyio_backend(cls) -> str:
        return 'asyncio'

    @pytest.fixture
    def can_detect_exit_with_inherited_output_pipes(self) -> bool:
        # Live Sprite (2026-09-26): the exec socket sends EXIT only after a background child
        # holding stdout exits, so `run()` waits for it. Documented with a redirect workaround.
        return False

    @pytest.fixture
    def filesystem_honors_shell_permissions(self) -> bool:
        # Live Sprite (2026-09-26): uid 1001 reads chmod 000 files (`test -r` succeeds).
        # The Sprite filesystem does not enforce ordinary POSIX mode checks for this user.
        return False

    # Class-scoped so the rules share one Sprite instead of creating one each.
    @pytest.fixture(scope='class')
    @classmethod
    async def backend(cls) -> AsyncIterator[SpritesSandboxBackend]:
        backend = SpritesSandboxBackend()
        yield backend
        await _teardown(backend)

    # The suite drops the backend it attaches; each one opened its own SDK client, which is closed
    # here rather than left to the garbage collector with a socket still open.
    @pytest.fixture
    async def attach_backend(self) -> AsyncIterator[Callable[[WorkspaceRef], WorkspaceBackend]]:
        attached: list[SpritesSandboxBackend] = []

        def attach(ref: WorkspaceRef) -> WorkspaceBackend:
            backend = SpritesSandboxBackend(ref=ref)
            attached.append(backend)
            return backend

        yield attach
        for backend in attached:
            await backend.aclose()

    # Destructive rules need their own Sprite, not the shared class-scoped one. Sprites have no
    # lifetime of their own, so one a failed rule left behind is deleted here, and its client closed.
    @pytest.fixture
    async def destructive_backend(self) -> AsyncIterator[Callable[[], WorkspaceBackend]]:
        created: list[SpritesSandboxBackend] = []

        def create() -> WorkspaceBackend:
            backend = SpritesSandboxBackend()
            created.append(backend)
            return backend

        yield create
        for backend in created:
            await _teardown(backend)

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _delete
