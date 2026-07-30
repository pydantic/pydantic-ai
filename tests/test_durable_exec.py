from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from pydantic_ai import Agent, FunctionToolset, RunContext
from pydantic_ai.agent.wrapper import WrapperAgent
from pydantic_ai.durable_exec.temporal import TemporalAgent  # pyright: ignore[reportDeprecated]
from pydantic_ai.models.test import TestModel


class CustomAgentWithoutValidationContext(WrapperAgent[None, str]):
    """Custom agent using the default `AbstractAgent.validation_context` implementation."""

    @property
    def validation_context(self) -> Any | Callable[[RunContext[None]], Any]:
        return super(WrapperAgent, self).validation_context


@pytest.mark.anyio
@pytest.mark.filterwarnings('ignore:`TemporalAgent` is deprecated')
async def test_custom_agent_without_validation_context_runs_without_args_validator() -> None:
    toolset = FunctionToolset(id='tools')

    @toolset.tool_plain
    def answer() -> str:
        return '42'

    custom_agent = CustomAgentWithoutValidationContext(
        Agent(TestModel(call_tools=[], custom_output_text='success'), name='custom_agent', toolsets=[toolset])
    )
    temporal_agent = TemporalAgent(custom_agent)  # pyright: ignore[reportDeprecated]

    result = await temporal_agent.run('Hello')

    assert result.output == 'success'
