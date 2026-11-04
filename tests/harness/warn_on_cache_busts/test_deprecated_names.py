"""Tests for the deprecated `pydantic_ai_harness.cache_stability` import shim."""

from __future__ import annotations

import importlib
import sys

import pytest

from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness.warn_on_cache_busts import WarnOnCacheBusts


def test_old_module_emits_deprecation_warning_and_aliases_class() -> None:
    sys.modules.pop('pydantic_ai_harness.cache_stability', None)
    with pytest.warns(HarnessDeprecationWarning, match='renamed to `pydantic_ai_harness.warn_on_cache_busts`'):
        module = importlib.import_module('pydantic_ai_harness.cache_stability')
    assert module.CacheStabilityMonitor is WarnOnCacheBusts
