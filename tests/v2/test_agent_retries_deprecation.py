"""1.x deprecation warnings for split agent retry kwargs.

`retries: int | AgentRetries` is the canonical retry configuration in v2. The
legacy `Agent(tool_retries=...)` / `Agent(output_retries=...)` kwargs still work
in 1.x as deprecation shims that fold into `retries`. v2 will remove them.
"""

from __future__ import annotations

import pytest

from pydantic_ai import Agent
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.models.test import TestModel


def test_agent_tool_retries_kwarg_emits_deprecation_warning():
    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`Agent\(tool_retries=\.\.\.\)` is deprecated and will be removed in v2\.0',
    ):
        agent = Agent(TestModel(), tool_retries=5)  # pyright: ignore[reportCallIssue]
    assert agent._max_tool_retries == 5  # pyright: ignore[reportPrivateUsage]


def test_agent_output_retries_kwarg_emits_deprecation_warning():
    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`Agent\(output_retries=\.\.\.\)` is deprecated and will be removed in v2\.0',
    ):
        agent = Agent(TestModel(), output_retries=7)  # pyright: ignore[reportCallIssue]
    assert agent._max_output_retries == 7  # pyright: ignore[reportPrivateUsage]


def test_agent_legacy_split_kwargs_win_over_retries():
    with pytest.warns(PydanticAIDeprecationWarning):
        agent = Agent(TestModel(), retries=2, tool_retries=4, output_retries=6)  # pyright: ignore[reportCallIssue]
    assert agent._max_tool_retries == 4  # pyright: ignore[reportPrivateUsage]
    assert agent._max_output_retries == 6  # pyright: ignore[reportPrivateUsage]
