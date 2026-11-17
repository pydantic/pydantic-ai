"""Pydantic AI's workspace backend conformance suite, run against `BoxLiteSandboxBackend`.

`TestFakeBoxLiteSandboxBackend` runs everywhere BoxLite installs, over the fake SDK, where commands
run as host processes. `TestLiveBoxLiteSandboxBackend` runs the same rules against real VMs and is
gated by `PYDANTIC_AI_HARNESS_BOXLITE_LIVE=1`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite
from pydantic_ai_harness.boxlite_sandbox import BoxLiteSandbox, BoxLiteSandboxBackend

from .fake_boxlite import FakeBoxLite


async def _destroy(backend: WorkspaceBackend) -> None:
    assert backend.ref is not None
    await BoxLiteSandbox[None]().destroy(backend.ref)


class TestFakeBoxLiteSandboxBackend(WorkspaceBackendSuite):
    @pytest.fixture
    def working_dir(self, fake_boxlite: FakeBoxLite, tmp_path: Path) -> str:
        # Resolved so the directory `pwd -P` reports matches it on hosts whose temp directory is a symlink.
        return str(tmp_path.resolve() / 'workspace')

    @pytest.fixture
    def backend(self, working_dir: str) -> BoxLiteSandboxBackend:
        return BoxLiteSandboxBackend(working_dir=working_dir)

    @pytest.mark.skip(reason='shell paging at a host process per 64 KiB; the live tier runs it in a VM')
    async def test_large_file_round_trip(self, backend: WorkspaceBackend) -> None: ...

    @pytest.fixture
    def fresh_backend(self, working_dir: str) -> Callable[[], WorkspaceBackend]:
        return lambda: BoxLiteSandboxBackend(working_dir=working_dir)

    @pytest.fixture
    def attach_backend(self, working_dir: str) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return lambda ref: BoxLiteSandboxBackend(ref=ref, working_dir=working_dir)

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _destroy


def _delete_now(ref: WorkspaceRef | None) -> None:  # pragma: no cover - live tier
    # Its own event loop, because a class-scoped fixture outlives each test's.
    import boxlite

    async def delete(name: str) -> None:
        try:
            await cast(Any, boxlite).Boxlite.default().remove(name, force=True)
        except boxlite.NotFoundError:
            pass

    if ref is not None:
        asyncio.run(delete(ref.id))


# The live tier runs without coverage; in CI only its gate runs, to skip it.
@pytest.mark.boxlite_live
class TestLiveBoxLiteSandboxBackend(WorkspaceBackendSuite):  # pragma: lax no cover
    # Class-scoped so the rules share one box instead of booting one each.
    @pytest.fixture(scope='class')
    @classmethod
    def backend(cls) -> Iterator[BoxLiteSandboxBackend]:
        backend = BoxLiteSandboxBackend(image='alpine:3.20')
        yield backend
        _delete_now(backend.ref)

    @pytest.fixture
    def fresh_backend(self) -> Iterator[Callable[[], WorkspaceBackend]]:
        created: list[BoxLiteSandboxBackend] = []

        def factory() -> BoxLiteSandboxBackend:
            backend = BoxLiteSandboxBackend(image='alpine:3.20')
            created.append(backend)
            return backend

        yield factory
        for backend in created:
            _delete_now(backend.ref)

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return lambda ref: BoxLiteSandboxBackend(ref=ref)

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _destroy
