"""Shared setup for the Pixeltable integration tests.

The folder is not named `tests/pixeltable`: Pyright treats `tests` as a root, and a
`pixeltable` folder there would shadow the installed package.
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
import uuid
from collections.abc import Iterator

import pytest

# Tests run in their own Pixeltable catalog, never `~/.pixeltable`. Set the home
# before any test module imports Pixeltable, and use a fresh directory so an
# interrupted local run cannot leave a partial PostgreSQL cluster for the next.
os.environ['PIXELTABLE_HOME'] = tempfile.mkdtemp(prefix='pydantic-ai-pixeltable-tests-')

# The `pixeltable` dependency is gated on the `pixeltable` extra (and needs Python 3.11+), so
# slim CI runs can't import these modules. Ignore them at collection. A conditional expression
# rather than an `if` statement: branch coverage traces statement arcs, and no single
# environment can take both arms of an install-dependent branch.
collect_ignore = (
    ['test_capability.py', 'test_memory.py', 'test_store.py', 'test_toolset.py']
    if importlib.util.find_spec('pixeltable') is None
    else []
)


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
