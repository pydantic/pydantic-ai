"""Tests for the deprecated `pydantic_ai_harness.overflowing_tool_output` import shim."""

from __future__ import annotations

import importlib
import sys

import pytest

from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness.tool_output_limits import ToolOutputLimits


def test_old_module_emits_deprecation_warning_and_aliases_class() -> None:
    sys.modules.pop('pydantic_ai_harness.overflowing_tool_output', None)
    with pytest.warns(HarnessDeprecationWarning, match='renamed to `pydantic_ai_harness.tool_output_limits`'):
        module = importlib.import_module('pydantic_ai_harness.overflowing_tool_output')
    assert module.OverflowingToolOutput is ToolOutputLimits
