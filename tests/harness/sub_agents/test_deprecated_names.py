"""Tests for the deprecated `pydantic_ai_harness.subagents` import shim."""

from __future__ import annotations

import importlib
import re
import sys
import warnings

import pytest

from pydantic_ai_harness import HarnessDeprecationWarning, sub_agents
from pydantic_ai_harness.sub_agents import DelegationFinishedEvent


def test_old_module_warns_and_reexports() -> None:
    sys.modules.pop('pydantic_ai_harness.subagents', None)
    message = '`pydantic_ai_harness.subagents` has been renamed to `pydantic_ai_harness.sub_agents`.'
    with pytest.warns(HarnessDeprecationWarning, match=re.escape(message)):
        shim = importlib.import_module('pydantic_ai_harness.subagents')

    assert shim.__all__ == sub_agents.__all__
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', HarnessDeprecationWarning)
        assert all(getattr(shim, name) is getattr(sub_agents, name) for name in sub_agents.__all__)
    with pytest.raises(AttributeError, match='has no attribute'):
        getattr(shim, 'DoesNotExist')


def test_old_module_points_forwarded_deprecations_at_the_caller() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', HarnessDeprecationWarning)
        shim = importlib.import_module('pydantic_ai_harness.subagents')

    with pytest.warns(
        HarnessDeprecationWarning, match='renamed to `pydantic_ai_harness.sub_agents.DelegationFinishedEvent`'
    ) as caught:
        assert getattr(shim, 'DelegationEndEvent') is DelegationFinishedEvent
    assert [warning.filename for warning in caught] == [__file__]
