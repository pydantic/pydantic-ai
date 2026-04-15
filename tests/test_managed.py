"""Tests for `pydantic_ai.managed.*` capabilities."""

from __future__ import annotations

import pytest

# Skip the whole module when logfire (with managed variables) isn't installed —
# `pydantic-ai-slim` and `pydantic-evals` test jobs run without the `logfire` extra.
pytest.importorskip('logfire.variables')

import logfire

from pydantic_ai.agent import Agent
from pydantic_ai.managed.logfire import Managed
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

pytestmark = [pytest.mark.anyio]


def _echo_model() -> FunctionModel:
    """A `FunctionModel` that reflects the effective instructions and model settings back as text."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                TextPart(content=f'instructions={info.instructions!r};settings={info.model_settings!r}'),
            ]
        )

    return FunctionModel(respond)


# Module-level variables (logfire registers by name; redeclaring the same name in multiple tests errors).
_instructions_var = logfire.var('pai_test_managed_instructions', default='Default instructions.')
_settings_var = logfire.var(
    'pai_test_managed_settings',
    type=dict,
    default={'temperature': 0.2},
)


async def test_instructions_from_variable_default() -> None:
    agent = Agent(_echo_model(), capabilities=[Managed(instructions=_instructions_var)])

    result = await agent.run('hi')

    assert "instructions='Default instructions.'" in result.output


async def test_instructions_from_variable_override() -> None:
    agent = Agent(_echo_model(), capabilities=[Managed(instructions=_instructions_var)])

    with _instructions_var.override('Overridden instructions.'):
        result = await agent.run('hi')

    assert "instructions='Overridden instructions.'" in result.output


async def test_instructions_merged_with_agent_level_instructions() -> None:
    agent = Agent(
        _echo_model(),
        instructions='Agent-level instructions.',
        capabilities=[Managed(instructions=_instructions_var)],
    )

    with _instructions_var.override('From variable.'):
        result = await agent.run('hi')

    # Both layers should be present in the final prompt.
    assert 'Agent-level instructions.' in result.output
    assert 'From variable.' in result.output


async def test_model_settings_from_variable() -> None:
    agent = Agent(_echo_model(), capabilities=[Managed(model_settings=_settings_var)])

    with _settings_var.override({'temperature': 0.9, 'max_tokens': 123}):
        result = await agent.run('hi')

    assert "'temperature': 0.9" in result.output
    assert "'max_tokens': 123" in result.output


async def test_model_settings_merged_with_agent_level_settings() -> None:
    agent = Agent(
        _echo_model(),
        model_settings={'temperature': 0.0, 'max_tokens': 50},
        capabilities=[Managed(model_settings=_settings_var)],
    )

    with _settings_var.override({'temperature': 0.9}):
        result = await agent.run('hi')

    # Variable temperature replaces agent default; agent's max_tokens survives.
    assert "'temperature': 0.9" in result.output
    assert "'max_tokens': 50" in result.output


async def test_resolves_fresh_per_run() -> None:
    agent = Agent(_echo_model(), capabilities=[Managed(instructions=_instructions_var)])

    result_a = await agent.run('a')
    assert "instructions='Default instructions.'" in result_a.output

    with _instructions_var.override('mid-run value'):
        result_b = await agent.run('b')
    assert "instructions='mid-run value'" in result_b.output

    result_c = await agent.run('c')
    assert "instructions='Default instructions.'" in result_c.output


async def test_no_variables_is_noop() -> None:
    """A `Managed` with no variables set shouldn't crash and shouldn't change behavior."""
    agent = Agent(
        _echo_model(),
        instructions='Just the agent.',
        capabilities=[Managed()],
    )

    result = await agent.run('hi')

    assert "instructions='Just the agent.'" in result.output
