from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest


@pytest.fixture
def root() -> Iterator[str]:
    import pixeltable as pxt

    name = f'harness_pxt_{uuid.uuid4().hex[:8]}'
    pxt.create_dir(name)
    yield name
    pxt.drop_dir(name, force=True)
