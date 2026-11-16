from __future__ import annotations

import pytest

from ...conftest import try_import

with try_import():
    from .worker import InMemoryWorker


@pytest.fixture
def durable_worker() -> InMemoryWorker:
    return InMemoryWorker()
