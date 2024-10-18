import os
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pytest

__all__ = 'IsNow', 'TestEnv'

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
