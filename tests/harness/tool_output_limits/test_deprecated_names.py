"""Tests for the deprecated `pydantic_ai_harness.overflowing_tool_output` import shim and store names."""

from __future__ import annotations

import importlib
import re
import sys
import warnings

import pytest

import pydantic_ai_harness.tool_output_limits as tool_output_limits
from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness.tool_output_limits import LocalFileOverflowStore, ToolOutputLimits


def test_old_module_emits_deprecation_warning_and_aliases_class() -> None:
    sys.modules.pop('pydantic_ai_harness.overflowing_tool_output', None)
    with pytest.warns(HarnessDeprecationWarning, match='renamed to `pydantic_ai_harness.tool_output_limits`'):
        module = importlib.import_module('pydantic_ai_harness.overflowing_tool_output')
    assert module.OverflowingToolOutput is ToolOutputLimits


@pytest.mark.parametrize(
    ('old', 'new'), [('LocalFileStore', 'LocalFileOverflowStore'), ('WorkspaceStore', 'WorkspaceOverflowStore')]
)
def test_renamed_store_alias_warns_and_resolves(old: str, new: str) -> None:
    message = f'`pydantic_ai_harness.tool_output_limits.{old}` has been renamed to `pydantic_ai_harness.tool_output_limits.{new}`.'
    with pytest.warns(HarnessDeprecationWarning, match=re.escape(message)):
        alias = getattr(tool_output_limits, old)

    assert alias is getattr(tool_output_limits, new)
    assert new in tool_output_limits.__all__
    assert old not in tool_output_limits.__all__
    with pytest.raises(AttributeError, match='has no attribute'):
        getattr(tool_output_limits, 'DoesNotExist')


@pytest.mark.parametrize(
    'module', ['pydantic_ai_harness.overflowing_tool_output', 'pydantic_ai_harness.experimental.overflow']
)
def test_deprecated_modules_keep_the_old_store_name(module: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', HarnessDeprecationWarning)
        shim = importlib.import_module(module)

    assert 'LocalFileStore' in shim.__all__
    assert shim.LocalFileStore is LocalFileOverflowStore
