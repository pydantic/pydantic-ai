"""Tests for the deprecated `pydantic_ai_harness.filesystem` import shim."""

from __future__ import annotations

import importlib
import re
import sys
import warnings

import pytest

from pydantic_ai_harness import HarnessDeprecationWarning, file_system
from pydantic_ai_harness.file_system import FileChangeRequestedEvent


def test_old_module_warns_and_reexports() -> None:
    sys.modules.pop('pydantic_ai_harness.filesystem', None)
    message = '`pydantic_ai_harness.filesystem` has been renamed to `pydantic_ai_harness.file_system`.'
    with pytest.warns(HarnessDeprecationWarning, match=re.escape(message)):
        shim = importlib.import_module('pydantic_ai_harness.filesystem')

    assert shim.__all__ == file_system.__all__
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', HarnessDeprecationWarning)
        assert all(getattr(shim, name) is getattr(file_system, name) for name in file_system.__all__)
    with pytest.raises(AttributeError, match='has no attribute'):
        getattr(shim, 'DoesNotExist')


def test_old_module_points_forwarded_deprecations_at_the_caller() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', HarnessDeprecationWarning)
        shim = importlib.import_module('pydantic_ai_harness.filesystem')

    with pytest.warns(
        HarnessDeprecationWarning, match='renamed to `pydantic_ai_harness.file_system.FileChangeRequestedEvent`'
    ) as caught:
        assert getattr(shim, 'FileChangeRequestEvent') is FileChangeRequestedEvent
    assert [warning.filename for warning in caught] == [__file__]
