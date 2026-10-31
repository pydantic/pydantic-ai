from __future__ import annotations

import pytest


@pytest.fixture
def anyio_backend() -> str:
    """Run live server tests once under asyncio, the only loop asyncpg supports."""
    return 'asyncio'
