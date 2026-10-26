"""Real-browser regressions run once in the dedicated CLAI 2 web CI job."""

import pytest


@pytest.fixture(scope='module')
def anyio_backend() -> str:
    """Keep the browser's shared asynchronous fixtures on one asyncio runner."""
    return 'asyncio'
