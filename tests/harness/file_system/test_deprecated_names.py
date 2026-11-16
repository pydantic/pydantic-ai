"""Tests for the deprecated `pydantic_ai_harness.filesystem` import shim."""

from __future__ import annotations

import importlib
import re
import sys
import warnings

import pytest

from pydantic_ai_harness import HarnessDeprecationWarning, file_system


def test_old_module_warns_and_reexports() -> None:
    sys.modules.pop('pydantic_ai_harness.filesystem', None)
    message = '`pydantic_ai_harness.filesystem` has been renamed to `pydantic_ai_harness.file_system`.'
    with pytest.warns(HarnessDeprecationWarning, match=re.escape(message)):
        shim = importlib.import_module('pydantic_ai_harness.filesystem')

    assert set(shim.__all__) - set(file_system.__all__) == set()
    assert all(getattr(shim, name) is getattr(file_system, name) for name in file_system.__all__)
    with pytest.raises(AttributeError, match='has no attribute'):
        getattr(shim, 'DoesNotExist')


def test_deprecated_names_stay_in_the_old_module() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', HarnessDeprecationWarning)
        shim = importlib.import_module('pydantic_ai_harness.filesystem')

    with pytest.warns(
        HarnessDeprecationWarning, match='renamed to `pydantic_ai_harness.file_system.FileChangeRequestedEvent`'
    ) as caught:
        getattr(shim, 'FileChangeRequestEvent')
    assert [warning.filename for warning in caught] == [__file__]
    assert not hasattr(file_system, 'FileChangeRequestEvent')
