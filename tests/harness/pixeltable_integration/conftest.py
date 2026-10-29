"""Shared setup for the Pixeltable integration tests.

The folder is not named `tests/pixeltable`: Pyright treats `tests` as a root, and a
`pixeltable` folder there would shadow the installed package.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest


@pytest.fixture(scope='session', autouse=True)
def pixeltable_home(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv('PIXELTABLE_HOME', str(tmp_path_factory.mktemp('pixeltable')))
        yield


@pytest.fixture
def root() -> Iterator[str]:
    import pixeltable as pxt

    name = f'harness_pxt_{uuid.uuid4().hex[:8]}'
    pxt.create_dir(name)
    yield name
    pxt.drop_dir(name, force=True)


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the asyncio backend (matching upstream pydantic-ai)."""
    return 'asyncio'
