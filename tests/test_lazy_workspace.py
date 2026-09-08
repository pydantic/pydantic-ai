"""The shared workspace authoring helper's acquisition contract."""

from collections.abc import Awaitable, Callable

import anyio
import pytest
from anyio.abc import TaskStatus
from typing_extensions import assert_type

from pydantic_ai.workspaces import LazyWorkspace

pytestmark = pytest.mark.anyio


class NativeWorkspace:
    pass


class Backend(LazyWorkspace[NativeWorkspace]):
    def __init__(self, acquire: Callable[[], Awaitable[NativeWorkspace]], workspace: NativeWorkspace | None = None) -> None:
        super().__init__(workspace)
        self.acquire = acquire

    async def create_or_attach(self) -> NativeWorkspace:
        return await self.acquire()


class TestLazyWorkspace:
    async def test_property_waits_for_await_and_reuses_native_handle(self) -> None:
        calls = 0
        native = NativeWorkspace()

        async def acquire() -> NativeWorkspace:
            nonlocal calls
            calls += 1
            return native

        backend = Backend(acquire)
        pending = backend.workspace
        assert_type(pending, Awaitable[NativeWorkspace])
        assert calls == 0
        assert assert_type(await pending, NativeWorkspace) is native
        assert await backend.workspace is native
        assert calls == 1

    async def test_supplied_native_handle_skips_acquisition(self) -> None:
        async def acquire() -> NativeWorkspace:
            pytest.fail('An existing handle must not be acquired again')  # pragma: no cover

        native = NativeWorkspace()
        assert await Backend(acquire, native).workspace is native

    async def test_concurrent_first_use_acquires_once(self) -> None:
        calls = 0
        entered = anyio.Event()
        release = anyio.Event()
        native = NativeWorkspace()
        results: list[NativeWorkspace] = []

        async def acquire() -> NativeWorkspace:
            nonlocal calls
            calls += 1
            entered.set()
            await release.wait()
            return native

        backend = Backend(acquire)

        async def use() -> None:
            results.append(await backend.workspace)

        async with anyio.create_task_group() as group:
            group.start_soon(use)
            await entered.wait()
            group.start_soon(use)
            await anyio.wait_all_tasks_blocked()
            release.set()
        assert calls == 1
        assert results == [native, native]

    async def test_failed_acquisition_can_be_retried(self) -> None:
        calls = 0
        native = NativeWorkspace()
        error = RuntimeError('provider unavailable')

        async def acquire() -> NativeWorkspace:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise error
            return native

        backend = Backend(acquire)
        with pytest.raises(RuntimeError) as caught:
            await backend.workspace
        assert caught.value is error
        assert await backend.workspace is native
        assert calls == 2

    async def test_cancelled_acquisition_releases_lock_for_retry(self) -> None:
        calls = 0
        entered = anyio.Event()
        native = NativeWorkspace()

        async def acquire() -> NativeWorkspace:
            nonlocal calls
            calls += 1
            if calls == 1:
                entered.set()
                await anyio.sleep_forever()
            return native

        backend = Backend(acquire)

        async def use(*, task_status: TaskStatus[anyio.CancelScope]) -> None:
            with anyio.CancelScope() as scope:
                task_status.started(scope)
                await backend.workspace

        async with anyio.create_task_group() as group:
            scope = await group.start(use)
            await entered.wait()
            scope.cancel()
        assert await backend.workspace is native
        assert calls == 2

    async def test_cancelling_waiter_preserves_acquisition(self) -> None:
        entered = anyio.Event()
        release = anyio.Event()
        native = NativeWorkspace()
        results: list[NativeWorkspace] = []

        async def acquire() -> NativeWorkspace:
            entered.set()
            await release.wait()
            return native

        backend = Backend(acquire)

        async def owner() -> None:
            results.append(await backend.workspace)

        async def waiter(*, task_status: TaskStatus[anyio.CancelScope]) -> None:
            with anyio.CancelScope() as scope:
                task_status.started(scope)
                await backend.workspace
            assert scope.cancelled_caught

        async with anyio.create_task_group() as group:
            group.start_soon(owner)
            await entered.wait()
            scope = await group.start(waiter)
            await anyio.wait_all_tasks_blocked()
            scope.cancel()
            release.set()
        assert results == [native]
        assert await backend.workspace is native
