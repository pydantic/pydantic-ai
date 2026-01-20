"""Code mode comparison test - demonstrates real model generating and executing code."""

from __future__ import annotations

import os

import pytest

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.toolsets.code_mode import CodeModeToolset
from pydantic_ai.toolsets.function import FunctionToolset

from .conftest import get_weather

pytestmark = [pytest.mark.anyio]


@pytest.fixture
def anthropic_api_key() -> str:
    """Get Anthropic API key from environment."""
    key = os.getenv('ANTHROPIC_API_KEY')
    if not key:
        pytest.skip('ANTHROPIC_API_KEY not set')
    return key


async def test_code_mode_with_real_model(anthropic_api_key: str):
    """Test code mode with a real Claude model.

    This test demonstrates:
    1. Model receives the run_code tool with available functions
    2. Model generates Python code to call multiple tools
    3. Code executes and returns results
    4. All inner tool calls are traced via spans
    """
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))

    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(get_weather, takes_ctx=False)

    code_mode_toolset = CodeModeToolset(wrapped=toolset)

    agent: Agent[None, str] = Agent(model)

    async with code_mode_toolset:
        result = await agent.run(
            'Get the weather for London, Paris, and Tokyo. Return the average temperature.',
            toolsets=[code_mode_toolset],
        )

    print(f'\n=== Result ===\n{result.output}')
    print(f'\n=== Messages ===')
    for msg in result.all_messages():
        print(msg)

    # The model should have used run_code to fetch weather and compute average
    assert result.output is not None
