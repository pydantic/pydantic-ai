from __future__ import annotations as _annotations

import asyncio
import functools
from collections.abc import Callable
from functools import partial
from typing import Any

import anyio.from_thread
import anyio.to_thread
import pytest
import trio
from dirty_equals import HasRepr

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_evals._utils import (
        UNSET,
        Unset,
        get_event_loop,
        get_unwrapped_function_name,
        is_set,
        run_until_complete,
        running_on_asyncio,
        task_group_gather,
    )

pytestmark = [pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed')]


def test_run_until_complete_cleans_up_own_task_on_interrupt():
    """A `KeyboardInterrupt` during `run_until_complete` must cancel only its own task and drive its
    cleanup, leaving no pending task and not touching other tasks on the caller-owned loop.

    Simulated by patching the loop because a real interrupt mid-`run_until_complete` while the
    coroutine is suspended can't be triggered reliably through the public API.
    """
    cleaned: list[str] = []

    async def coro() -> None:
        try:
            await asyncio.Event().wait()  # suspends forever
        finally:
            cleaned.append('cleaned')

    loop = get_event_loop()
    tasks_before = asyncio.all_tasks(loop)
    real_run_until_complete = loop.run_until_complete
    calls = 0

    def interrupt_once(future: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            # Let our task start and suspend, then simulate Ctrl-C reaching the caller.
            loop.call_soon(loop.stop)
            loop.run_forever()
            raise KeyboardInterrupt
        return real_run_until_complete(future)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(loop, 'run_until_complete', interrupt_once)
        with pytest.raises(KeyboardInterrupt):
            run_until_complete(coro())

    assert cleaned == ['cleaned']  # our coroutine's cleanup ran
    assert asyncio.all_tasks(loop) == tasks_before  # our task didn't leak, nothing else was touched


def test_run_until_complete_propagates_coroutine_error():
    """When the coroutine itself raises, the task is already done so no cleanup is attempted and the
    exception propagates unchanged."""

    async def coro() -> None:
        raise ValueError('boom')

    with pytest.raises(ValueError, match='boom'):
        run_until_complete(coro())


def test_unset():
    """Test Unset singleton."""
    assert isinstance(UNSET, Unset)
    assert UNSET is not Unset()  # note: we might want to change this and make it a true singleton..


def test_is_set():
    """Test is_set function."""
    assert is_set(42) is True
    assert is_set(None) is True
    assert is_set(UNSET) is False


def test_get_unwrapped_function_name_basic():
    """Test get_unwrapped_function_name with basic function."""

    def test_func():
        pass

    assert get_unwrapped_function_name(test_func) == 'test_func'


def test_get_unwrapped_function_name_partial():
    """Test get_unwrapped_function_name with partial function."""

    def test_func(x: int, y: int):
        raise NotImplementedError

    partial_func = partial(test_func, y=42)
    assert get_unwrapped_function_name(partial_func) == 'test_func'


def test_get_unwrapped_function_name_decorated():
    """Test get_unwrapped_function_name with decorated function."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError

        return wrapper

    @decorator
    def test_func():
        pass

    assert get_unwrapped_function_name(test_func) == 'test_func'


def test_get_unwrapped_function_name_callable_class():
    """Test get_unwrapped_function_name with callable class."""

    class CallableClass:
        def __call__(self):
            pass

    assert (
        get_unwrapped_function_name(CallableClass())
        == 'test_get_unwrapped_function_name_callable_class.<locals>.CallableClass.__call__'
    )


def test_get_unwrapped_function_name_method():
    """Test get_unwrapped_function_name with method."""

    class TestClass:
        def test_method(self):
            pass

    assert get_unwrapped_function_name(TestClass().test_method) == 'test_method'


def test_get_unwrapped_function_name_error():
    """Test get_unwrapped_function_name with invalid input."""

    class InvalidCallable:
        pass

    with pytest.raises(AttributeError) as exc_info:
        get_unwrapped_function_name(InvalidCallable())  # pyright: ignore[reportArgumentType]

    assert str(exc_info.value) == "'InvalidCallable' object has no attribute '__name__'"


async def test_task_group_gather():
    """Test task_group_gather function."""

    async def task1():
        return 1

    async def task2():
        return 2

    async def task3():
        return 3

    tasks = [task1, task2, task3]
    results = await task_group_gather(tasks)
    assert results == [1, 2, 3]


async def test_task_group_gather_with_error():
    """Test task_group_gather function with error in one task."""

    async def task1():
        return 1

    async def task2():
        raise ValueError('Task 2 failed')

    async def task3():
        return 3

    tasks = [task1, task2, task3]
    with pytest.raises(ExceptionGroup) as exc_info:
        await task_group_gather(tasks)

    assert exc_info.value == HasRepr(
        repr(ExceptionGroup('unhandled errors in a TaskGroup', [ValueError('Task 2 failed')]))
    )


async def test_running_on_asyncio():
    assert running_on_asyncio()
    # `anyio.from_thread.run_sync` calls it on the loop, but outside any asyncio task.
    assert await anyio.to_thread.run_sync(anyio.from_thread.run_sync, running_on_asyncio)
    assert not await anyio.to_thread.run_sync(running_on_asyncio)


def test_trio_guest_mode_is_not_asyncio():
    # Trio guest mode runs Trio tasks on a thread whose asyncio loop is running.
    results: list[bool] = []

    async def guest() -> None:
        results.append(running_on_asyncio())

    async def host() -> None:
        done = asyncio.Event()
        trio.lowlevel.start_guest_run(
            guest,
            run_sync_soon_threadsafe=asyncio.get_running_loop().call_soon_threadsafe,
            done_callback=lambda _: done.set(),
        )
        await done.wait()

    asyncio.run(host())
    assert results == [False]


async def test_running_on_asyncio_without_sniffio(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr('pydantic_evals._utils._sniffio', None)
    assert running_on_asyncio()
    assert not await anyio.to_thread.run_sync(running_on_asyncio)
