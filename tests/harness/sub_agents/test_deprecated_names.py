"""Tests for the deprecated `pydantic_ai_harness.subagents` import shim."""

from __future__ import annotations

import importlib
import re
import sys
import warnings

import pytest

from pydantic_ai_harness import HarnessDeprecationWarning, sub_agents


def test_old_module_warns_and_reexports() -> None:
    sys.modules.pop('pydantic_ai_harness.subagents', None)
    message = '`pydantic_ai_harness.subagents` has been renamed to `pydantic_ai_harness.sub_agents`.'
    with pytest.warns(HarnessDeprecationWarning, match=re.escape(message)):
        shim = importlib.import_module('pydantic_ai_harness.subagents')

    assert set(shim.__all__) - set(sub_agents.__all__) == {'MINIMUM_EFFORT_FLOOR', 'clamp_effort'}
    assert all(getattr(shim, name) is getattr(sub_agents, name) for name in sub_agents.__all__)
    with pytest.raises(AttributeError, match='has no attribute'):
        getattr(shim, 'DoesNotExist')


def test_deprecated_names_stay_in_the_old_module() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', HarnessDeprecationWarning)
        shim = importlib.import_module('pydantic_ai_harness.subagents')

    with pytest.warns(
        HarnessDeprecationWarning, match='renamed to `pydantic_ai_harness.sub_agents.DelegationFinishedEvent`'
    ) as caught:
        getattr(shim, 'DelegationEndEvent')
    assert [warning.filename for warning in caught] == [__file__]
    assert not hasattr(sub_agents, 'DelegationEndEvent')
