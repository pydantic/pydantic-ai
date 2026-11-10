"""Internal factories and closers renamed to `make_*` and `aclose` stay callable under their old names."""

from __future__ import annotations

import pytest

from pydantic_ai.models.test import TestModel
from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness._monty_exec import MontyRunState
from pydantic_ai_harness.smart_search._judge import build_agent, make_agent
from pydantic_ai_harness.spend import SpendLimits
from pydantic_ai_harness.spend._toolset import build_toolset, make_toolset


async def test_monty_run_state_close_is_a_deprecated_alias_of_aclose() -> None:
    state = MontyRunState()
    with pytest.warns(HarnessDeprecationWarning, match=r'`MontyRunState\.close\(\)` has been renamed to'):
        await state.close()
    await state.aclose()
    assert state.pool is None


def test_smart_search_build_agent_is_a_deprecated_alias_of_make_agent() -> None:
    with pytest.warns(HarnessDeprecationWarning, match=r'`build_agent\(\)` has been renamed to `make_agent\(\)`'):
        agent = build_agent(TestModel(), 'query')
    assert agent.name == make_agent(TestModel(), 'query').name == 'smart_search'


def test_spend_build_toolset_is_a_deprecated_alias_of_make_toolset() -> None:
    limits = SpendLimits[None]()
    with pytest.warns(HarnessDeprecationWarning, match=r'`build_toolset\(\)` has been renamed to `make_toolset\(\)`'):
        toolset = build_toolset(limits)
    assert toolset.id == make_toolset(limits).id == 'spend'
