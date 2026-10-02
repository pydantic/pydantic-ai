"""`running_on_asyncio`, kept in both `pydantic-evals` and `pydantic-ai-harness`, in each context a caller can be in.

A unit test: the helper is private and only decides which event-loop API its callers use.
"""

from __future__ import annotations as _annotations

import asyncio
import importlib
from collections.abc import Callable

import anyio.from_thread
import anyio.to_thread
import pytest
import trio

from .conftest import try_import

with try_import() as evals_available:
    import pydantic_evals._utils  # noqa: F401  # pyright: ignore[reportUnusedImport]
with try_import() as harness_available:
    import pydantic_ai_harness._workspace_provider  # noqa: F401  # pyright: ignore[reportUnusedImport]

Check = Callable[[], bool]


@pytest.fixture(
    params=[
        pytest.param(
            'pydantic_evals._utils',
            id='evals',
            marks=pytest.mark.skipif(not evals_available(), reason='pydantic-evals not installed'),
        ),
        pytest.param(
            'pydantic_ai_harness._workspace_provider',
            id='harness',
            marks=pytest.mark.skipif(not harness_available(), reason='pydantic-ai-harness not installed'),
        ),
    ]
)
def module(request: pytest.FixtureRequest) -> str:
    return request.param


def _in_asyncio_task(check: Check) -> bool:
    async def main() -> bool:
        return check()

    return asyncio.run(main())


def _in_asyncio_loop_callback(check: Check) -> bool:
    # `anyio.from_thread.run_sync` runs the function on the loop, but outside any asyncio task.
    async def main() -> bool:
        return await anyio.to_thread.run_sync(anyio.from_thread.run_sync, check)

    return asyncio.run(main())


def _in_worker_thread(check: Check) -> bool:
    async def main() -> bool:
        return await anyio.to_thread.run_sync(check)

    return asyncio.run(main())


def _in_trio_task(check: Check) -> bool:
    async def main() -> bool:
        return check()

    return trio.run(main)


def _in_trio_guest_on_asyncio(check: Check) -> bool:
    # Trio guest mode runs Trio tasks on a thread whose asyncio loop is running.
    results: list[bool] = []

    async def guest() -> None:
        results.append(check())

    async def host() -> None:
        done = asyncio.Event()
        trio.lowlevel.start_guest_run(
            guest,
            run_sync_soon_threadsafe=asyncio.get_running_loop().call_soon_threadsafe,
            done_callback=lambda _: done.set(),
        )
        await done.wait()

    asyncio.run(host())
    [result] = results
    return result


CONTEXTS = [
    pytest.param(_in_asyncio_task, True, id='asyncio-task'),
    pytest.param(_in_asyncio_loop_callback, True, id='asyncio-loop-callback'),
    pytest.param(_in_worker_thread, False, id='worker-thread'),
    pytest.param(_in_trio_task, False, id='trio-task'),
]


@pytest.mark.parametrize(
    ('context', 'expected'),
    [*CONTEXTS, pytest.param(_in_trio_guest_on_asyncio, False, id='trio-guest-on-asyncio')],
)
def test_running_on_asyncio(module: str, context: Callable[[Check], bool], expected: bool):
    check: Check = importlib.import_module(module).running_on_asyncio
    assert context(check) is expected


@pytest.mark.parametrize(('context', 'expected'), CONTEXTS)
def test_running_on_asyncio_without_sniffio(
    monkeypatch: pytest.MonkeyPatch, module: str, context: Callable[[Check], bool], expected: bool
):
    # A clean install has no `sniffio`. Trio guest mode needs it installed, so it has no row here.
    monkeypatch.setattr(f'{module}._sniffio', None)
    check: Check = importlib.import_module(module).running_on_asyncio
    assert context(check) is expected
