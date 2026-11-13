"""Tests for the deprecated `pydantic_ai_harness.spend` import shim."""

from __future__ import annotations

import importlib
import re
import sys
import warnings

import pytest

from pydantic_ai_harness import HarnessDeprecationWarning, spend_limits


def test_old_module_warns_and_reexports() -> None:
    sys.modules.pop('pydantic_ai_harness.spend', None)
    message = '`pydantic_ai_harness.spend` has been renamed to `pydantic_ai_harness.spend_limits`.'
    with pytest.warns(HarnessDeprecationWarning, match=re.escape(message)):
        shim = importlib.import_module('pydantic_ai_harness.spend')

    assert shim.__all__ == spend_limits.__all__
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', HarnessDeprecationWarning)
        assert all(getattr(shim, name) is getattr(spend_limits, name) for name in spend_limits.__all__)
    with pytest.raises(AttributeError, match='has no attribute'):
        getattr(shim, 'DoesNotExist')


def test_old_module_points_forwarded_deprecations_at_the_caller() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', HarnessDeprecationWarning)
        shim = importlib.import_module('pydantic_ai_harness.spend')

    with pytest.warns(HarnessDeprecationWarning, match='SpendCompositionWarning` is deprecated') as caught:
        assert issubclass(getattr(shim, 'SpendCompositionWarning'), Warning)
    assert [warning.filename for warning in caught] == [__file__]
