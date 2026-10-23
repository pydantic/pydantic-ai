"""Real-browser regressions run once in the dedicated CLAI 2 web CI job."""

import os

import pytest

collect_ignore: list[str] = ['test_web_browser.py'] if os.environ.get('CLAI2_BROWSER_TESTS') != '1' else []


@pytest.fixture(scope='module')
def anyio_backend() -> str:
    """Keep the browser's shared asynchronous fixtures on one asyncio runner."""
    return 'asyncio'
