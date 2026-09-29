"""Tests for the deprecated `pydantic_ai_harness.runtime_authoring` import shim."""

from __future__ import annotations

import importlib
import sys

import pytest

from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness.capability_creation import CapabilityCreation


def test_old_module_emits_deprecation_warning_and_aliases_class() -> None:
    sys.modules.pop('pydantic_ai_harness.runtime_authoring', None)
    with pytest.warns(HarnessDeprecationWarning, match='renamed to `pydantic_ai_harness.capability_creation`'):
        module = importlib.import_module('pydantic_ai_harness.runtime_authoring')
    assert module.RuntimeAuthoring is CapabilityCreation
