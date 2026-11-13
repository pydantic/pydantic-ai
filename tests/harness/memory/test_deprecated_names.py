"""Tests for the deprecated `pydantic_ai_harness.memory` store names."""

from __future__ import annotations

import re

import pytest

import pydantic_ai_harness.memory as package
from pydantic_ai_harness import HarnessDeprecationWarning


@pytest.mark.parametrize(('old', 'new'), [('FileStore', 'FileMemoryStore'), ('InMemoryStore', 'InMemoryMemoryStore')])
def test_renamed_class_alias_warns_and_resolves(old: str, new: str) -> None:
    message = f'`pydantic_ai_harness.memory.{old}` has been renamed to `pydantic_ai_harness.memory.{new}`.'
    with pytest.warns(HarnessDeprecationWarning, match=re.escape(message)):
        alias = getattr(package, old)

    assert alias is getattr(package, new)
    assert new in package.__all__
    assert old not in package.__all__


def test_unknown_attribute_raises() -> None:
    with pytest.raises(AttributeError, match='has no attribute'):
        getattr(package, 'DoesNotExist')
