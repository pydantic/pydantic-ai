import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

import httpx
import pytest
from typing_extensions import TypeAlias

__all__ = 'IsNow', 'TestEnv'

try:
    import logfire
except ImportError:
    pass
else:
    logfire.configure(send_to_logfire=False)

if TYPE_CHECKING:

    def IsNow(*args: Any, **kwargs: Any) -> datetime: ...
else:
    from dirty_equals import IsNow


class TestEnv:
    __test__ = False

    def __init__(self):
        self.envars: set[str] = set()

    def set(self, name: str, value: str) -> None:
        self.envars.add(name)
        os.environ[name] = value

    def pop(self, name: str) -> None:  # pragma: no cover
        self.envars.remove(name)
        os.environ.pop(name)

    def clear(self) -> None:
        for n in self.envars:
            os.environ.pop(n)


@pytest.fixture
def env():
    test_env = TestEnv()

    yield test_env

    test_env.clear()


@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.fixture
async def client_with_handler():
    client: httpx.AsyncClient | None = None

    def create_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
        nonlocal client
        assert client is None, 'client_with_handler can only be called once'
        client = httpx.AsyncClient(mounts={'all://': httpx.MockTransport(handler)})
        return client

    try:
        yield create_client
    finally:
        if client:  # pragma: no cover
            await client.aclose()


ClientWithHandler: TypeAlias = Callable[[Callable[[httpx.Request], httpx.Response]], httpx.AsyncClient]
