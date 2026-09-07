from __future__ import annotations

from collections.abc import AsyncIterator

import anyio
import pytest
import sniffio
from anyio.streams.memory import MemoryObjectReceiveStream

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel


@pytest.fixture(scope='class')
async def backend_stream(anyio_backend: str) -> AsyncIterator[MemoryObjectReceiveStream[str]]:
    send, receive = anyio.create_memory_object_stream[str](1)

    async def producer() -> None:
        async with send:
            await send.send(sniffio.current_async_library())

    async with anyio.create_task_group() as group, receive:
        group.start_soon(producer)
        yield receive
        assert sniffio.current_async_library() == anyio_backend


async def test_unmarked_async_test_uses_selected_backend(pytestconfig: pytest.Config) -> None:
    await anyio.sleep(0)
    assert sniffio.current_async_library() == pytestconfig.getoption('--anyio-backend')


class TestBackendFixtureLifecycle:
    async def test_producer_finishes(self, backend_stream: MemoryObjectReceiveStream[str], anyio_backend: str) -> None:
        assert await backend_stream.receive() == anyio_backend
        with pytest.raises(anyio.EndOfStream):
            await backend_stream.receive()


@pytest.mark.parametrize('anyio_backend', ['trio'])
def test_sync_test_does_not_inherit_trio_backend(anyio_backend: str) -> None:
    assert anyio_backend == 'trio'
    with pytest.raises(sniffio.AsyncLibraryNotFoundError):
        sniffio.current_async_library()
    assert Agent(TestModel(custom_output_text='sync works')).run_sync('Hello').output == 'sync works'
