"""Tests for Agent.to_ag_ui method."""

from __future__ import annotations

import contextlib
import logging
import sys
from dataclasses import dataclass, field
from typing import Final

import pytest

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

has_required_python: bool = sys.version_info >= (3, 10)
has_ag_ui: bool = False
if has_required_python:
    with contextlib.suppress(ImportError):
        from pydantic_ai_ag_ui.adapter import _LOGGER as adapter_logger, Adapter  # type: ignore[reportPrivateUsage]

        has_ag_ui = True


pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not has_required_python, reason='requires Python 3.10 or higher'),
    pytest.mark.skipif(has_required_python and not has_ag_ui, reason='pydantic-ai-ag-ui not installed'),
]

# Constants.
CUSTOM_LOGGER: Final[logging.Logger] = logging.getLogger('test_logger')


@pytest.fixture
async def agent() -> Agent[None, str]:
    """Create an Adapter instance for testing."""
    return Agent(model=TestModel())


@dataclass
class ToAGUITest:
    id: str
    logger: logging.Logger | None = None
    tool_prefix: str | None = None
    expected_logger: logging.Logger = field(
        default_factory=lambda: adapter_logger if has_ag_ui else logging.getLogger(__name__)  # type: ignore[reportPossiblyUnboundVariable]
    )
    expected_tool_prefix: str = ''


TEST_PARAMETERS = [
    ToAGUITest(
        id='defaults',
    ),
    ToAGUITest(
        id='custom_logger',
        logger=CUSTOM_LOGGER,
        expected_logger=CUSTOM_LOGGER,
    ),
    ToAGUITest(
        id='custom_tool_prefix',
        tool_prefix='test_prefix',
        expected_tool_prefix='test_prefix',
    ),
    ToAGUITest(
        id='custom_tool_timeout',
    ),
    ToAGUITest(
        id='custom_all',
        logger=CUSTOM_LOGGER,
        tool_prefix='test_prefix',
        expected_logger=CUSTOM_LOGGER,
        expected_tool_prefix='test_prefix',
    ),
]


@pytest.mark.parametrize('tc', TEST_PARAMETERS, ids=lambda tc: tc.id)
@pytest.mark.anyio
async def test_to_ag_ui(agent: Agent[None, str], tc: ToAGUITest) -> None:
    """Test the agent.to_ag_ui method.

    Args:
        agent: The agent instance to test.
        tc: Test case parameters including logger, tool prefix, and timeout.
    """

    adapter: Adapter[None, str] = agent.to_ag_ui(
        logger=tc.logger,
        tool_prefix=tc.tool_prefix,
    )
    assert adapter.logger == tc.expected_logger
    assert adapter.tool_prefix == tc.expected_tool_prefix
