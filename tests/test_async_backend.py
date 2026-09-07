from __future__ import annotations

from collections.abc import AsyncIterator

import anyio
import pytest
import sniffio
from anyio.streams.memory import MemoryObjectReceiveStream

pytestmark = pytest.mark.anyio


@pytest.fixture(scope='module')
async def backend_stream(anyio_backend: str) -> AsyncIterator[MemoryObjectReceiveStream[str]]:
    send, receive = anyio.create_memory_object_stream[str](1)

    async def producer() -> None:
        async with send:
            await send.send(sniffio.current_async_library())

    async with anyio.create_task_group() as group, receive:
        group.start_soon(producer)
        yield receive
        assert sniffio.current_async_library() == anyio_backend


async def test_selected_backend(anyio_backend: str) -> None:
    await anyio.sleep(0)
    assert sniffio.current_async_library() == anyio_backend


async def test_backend_fixture_lifecycle(backend_stream: MemoryObjectReceiveStream[str], anyio_backend: str) -> None:
    assert await backend_stream.receive() == anyio_backend
    with pytest.raises(anyio.EndOfStream):
        await backend_stream.receive()
