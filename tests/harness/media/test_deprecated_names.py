"""Tests for the deprecated `pydantic_ai_harness.media` store names."""

from __future__ import annotations

import importlib
import re
import warnings

import pytest

import pydantic_ai_harness.media as package
from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness.media import FileMediaStore


@pytest.mark.parametrize(('old', 'new'), [('DiskMediaStore', 'FileMediaStore')])
def test_renamed_class_alias_warns_and_resolves(old: str, new: str) -> None:
    message = f'`pydantic_ai_harness.media.{old}` has been renamed to `pydantic_ai_harness.media.{new}`.'
    with pytest.warns(HarnessDeprecationWarning, match=re.escape(message)):
        alias = getattr(package, old)

    assert alias is getattr(package, new)
    assert new in package.__all__
    assert old not in package.__all__


def test_experimental_media_shim_keeps_the_old_name() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', HarnessDeprecationWarning)
        shim = importlib.import_module('pydantic_ai_harness.experimental.media')

    assert 'DiskMediaStore' in shim.__all__
    assert shim.DiskMediaStore is FileMediaStore


def test_unknown_attribute_raises() -> None:
    with pytest.raises(AttributeError, match='has no attribute'):
        getattr(package, 'DoesNotExist')
