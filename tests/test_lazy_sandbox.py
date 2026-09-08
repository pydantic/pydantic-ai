"""The shared sandbox authoring helper's acquisition contract."""

from collections.abc import Awaitable, Callable

import anyio
import pytest
from anyio.abc import TaskStatus
from typing_extensions import assert_type

from pydantic_ai.sandboxes import LazySandbox

pytestmark = pytest.mark.anyio


class NativeSandbox:
    pass


class Backend(LazySandbox[NativeSandbox]):
    def __init__(self, acquire: Callable[[], Awaitable[NativeSandbox]], sandbox: NativeSandbox | None = None) -> None:
        super().__init__(sandbox)
        self.acquire = acquire

    async def create_or_attach(self) -> NativeSandbox:
        return await self.acquire()


class TestLazySandbox:
    async def test_property_waits_for_await_and_reuses_native_handle(self) -> None:
        calls = 0
        native = NativeSandbox()

        async def acquire() -> NativeSandbox:
            nonlocal calls
            calls += 1
            return native

        backend = Backend(acquire)
        pending = backend.sandbox
        assert_type(pending, Awaitable[NativeSandbox])
        assert calls == 0
        assert assert_type(await pending, NativeSandbox) is native
        assert await backend.sandbox is native
        assert calls == 1

    async def test_supplied_native_handle_skips_acquisition(self) -> None:
        async def acquire() -> NativeSandbox:
            pytest.fail('An existing handle must not be acquired again')

        native = NativeSandbox()
        assert await Backend(acquire, native).sandbox is native

    async def test_concurrent_first_use_acquires_once(self) -> None:
        calls = 0
        entered = anyio.Event()
        release = anyio.Event()
        native = NativeSandbox()
        results: list[NativeSandbox] = []

        async def acquire() -> NativeSandbox:
            nonlocal calls
            calls += 1
            entered.set()
            await release.wait()
            return native

        backend = Backend(acquire)

        async def use() -> None:
            results.append(await backend.sandbox)

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
        native = NativeSandbox()
        error = RuntimeError('provider unavailable')

        async def acquire() -> NativeSandbox:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise error
            return native

        backend = Backend(acquire)
        with pytest.raises(RuntimeError) as caught:
            await backend.sandbox
        assert caught.value is error
        assert await backend.sandbox is native
        assert calls == 2

    async def test_cancelled_acquisition_releases_lock_for_retry(self) -> None:
        calls = 0
        entered = anyio.Event()
        native = NativeSandbox()

        async def acquire() -> NativeSandbox:
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
                await backend.sandbox

        async with anyio.create_task_group() as group:
            scope = await group.start(use)
            await entered.wait()
            scope.cancel()
        assert await backend.sandbox is native
        assert calls == 2

    async def test_cancelling_waiter_preserves_acquisition(self) -> None:
        entered = anyio.Event()
        release = anyio.Event()
        native = NativeSandbox()
        results: list[NativeSandbox] = []

        async def acquire() -> NativeSandbox:
            entered.set()
            await release.wait()
            return native

        backend = Backend(acquire)

        async def owner() -> None:
            results.append(await backend.sandbox)

        async def waiter(*, task_status: TaskStatus[anyio.CancelScope]) -> None:
            with anyio.CancelScope() as scope:
                task_status.started(scope)
                await backend.sandbox
            assert scope.cancelled_caught

        async with anyio.create_task_group() as group:
            group.start_soon(owner)
            await entered.wait()
            scope = await group.start(waiter)
            await anyio.wait_all_tasks_blocked()
            scope.cancel()
            release.set()
        assert results == [native]
        assert await backend.sandbox is native
