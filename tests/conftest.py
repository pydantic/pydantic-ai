import os

import pytest


class TestEnv:
    __test__ = False

    def __init__(self):
        self.envars: set[str] = set()

    def set(self, name: str, value: str) -> None:
        self.envars.add(name)
        os.environ[name] = value

    def pop(self, name: str) -> None:
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
