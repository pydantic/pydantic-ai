"""Tests for `pydantic_ai.managed.*` capabilities."""

from __future__ import annotations

import pytest

# Skip the whole module when logfire (with managed variables) isn't installed —
# `pydantic-ai-slim` and `pydantic-evals` test jobs run without the `logfire` extra.
pytest.importorskip('logfire.variables')

import logfire

from pydantic_ai import AgentSpec
from pydantic_ai.agent import Agent
from pydantic_ai.managed.logfire import Managed, ManagedAgentSpec
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from ._inline_snapshot import snapshot
from .conftest import IsDatetime, IsStr

pytestmark = [pytest.mark.anyio]


def _echo_model() -> FunctionModel:
    """A `FunctionModel` that reflects the effective instructions and model settings back as text.

    Model settings aren't otherwise visible from `result.all_messages()` (they live in
    `ModelRequestParameters`), so the echo model lets snapshot-based assertions confirm
    that managed values actually reach the model request.
    """

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
_agent_spec_var = logfire.var(
    'pai_test_managed_agent_spec',
    type=AgentSpec,
    default=AgentSpec.from_dict({'model': 'test'}),
)


async def test_instructions_from_variable_default() -> None:
    agent = Agent(_echo_model(), capabilities=[Managed(instructions=_instructions_var)])

    result = await agent.run('hi')

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='hi', timestamp=IsDatetime())],
                instructions='Default instructions.',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content="instructions='Default instructions.';settings=None")],
                usage=RequestUsage(input_tokens=51, output_tokens=3),
                model_name='function:respond:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_instructions_from_variable_override() -> None:
    agent = Agent(_echo_model(), capabilities=[Managed(instructions=_instructions_var)])

    with _instructions_var.override('Overridden instructions.'):
        result = await agent.run('hi')

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='hi', timestamp=IsDatetime())],
                instructions='Overridden instructions.',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content="instructions='Overridden instructions.';settings=None")],
                usage=RequestUsage(input_tokens=51, output_tokens=3),
                model_name='function:respond:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_instructions_merged_with_agent_level_instructions() -> None:
    agent = Agent(
        _echo_model(),
        instructions='Agent-level instructions.',
        capabilities=[Managed(instructions=_instructions_var)],
    )

    with _instructions_var.override('From variable.'):
        result = await agent.run('hi')

    # Both layers appear, concatenated in order: agent first, then managed.
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='hi', timestamp=IsDatetime())],
                instructions="""\
Agent-level instructions.
From variable.\
""",
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content="instructions='Agent-level instructions.\\nFrom variable.';settings=None")],
                usage=RequestUsage(input_tokens=51, output_tokens=5),
                model_name='function:respond:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_model_settings_from_variable() -> None:
    agent = Agent(_echo_model(), capabilities=[Managed(model_settings=_settings_var)])

    with _settings_var.override({'temperature': 0.9, 'max_tokens': 123}):
        result = await agent.run('hi')

    # Model settings aren't stored on `ModelRequest`, so we verify via the echo response.
    assert result.all_messages()[-1] == snapshot(
        ModelResponse(
            parts=[TextPart(content="instructions=None;settings={'temperature': 0.9, 'max_tokens': 123}")],
            usage=RequestUsage(input_tokens=51, output_tokens=5),
            model_name='function:respond:',
            timestamp=IsDatetime(),
            run_id=IsStr(),
            conversation_id=IsStr(),
        )
    )


async def test_model_settings_merged_with_agent_level_settings() -> None:
    agent = Agent(
        _echo_model(),
        model_settings={'temperature': 0.0, 'max_tokens': 50},
        capabilities=[Managed(model_settings=_settings_var)],
    )

    with _settings_var.override({'temperature': 0.9}):
        result = await agent.run('hi')

    # Managed temperature replaces agent default; agent's max_tokens survives.
    assert result.all_messages()[-1] == snapshot(
        ModelResponse(
            parts=[TextPart(content="instructions=None;settings={'temperature': 0.9, 'max_tokens': 50}")],
            usage=RequestUsage(input_tokens=51, output_tokens=5),
            model_name='function:respond:',
            timestamp=IsDatetime(),
            run_id=IsStr(),
            conversation_id=IsStr(),
        )
    )


async def test_resolves_fresh_per_run() -> None:
    agent = Agent(_echo_model(), capabilities=[Managed(instructions=_instructions_var)])

    result_a = await agent.run('a')
    assert result_a.all_messages()[0] == snapshot(
        ModelRequest(
            parts=[UserPromptPart(content='a', timestamp=IsDatetime())],
            instructions='Default instructions.',
            timestamp=IsDatetime(),
            run_id=IsStr(),
            conversation_id=IsStr(),
        )
    )

    with _instructions_var.override('mid-run value'):
        result_b = await agent.run('b')
    assert result_b.all_messages()[0] == snapshot(
        ModelRequest(
            parts=[UserPromptPart(content='b', timestamp=IsDatetime())],
            instructions='mid-run value',
            timestamp=IsDatetime(),
            run_id=IsStr(),
            conversation_id=IsStr(),
        )
    )

    result_c = await agent.run('c')
    assert result_c.all_messages()[0] == snapshot(
        ModelRequest(
            parts=[UserPromptPart(content='c', timestamp=IsDatetime())],
            instructions='Default instructions.',
            timestamp=IsDatetime(),
            run_id=IsStr(),
            conversation_id=IsStr(),
        )
    )


async def test_agent_spec_variable_overrides_model() -> None:
    """Characterization test: the managed spec's `model` overrides the agent's base model per-run.

    Pins the behavior the current model-swap implementation provides *before* it is refactored
    into a generic `contribute_run_spec` hook. The base agent uses the echo `FunctionModel`
    (`model_name == 'function:respond:'`); the managed spec resolves to `'test'` (→ `TestModel`,
    `model_name == 'test'`). If the swap works, the response's `model_name` flips to `'test'`,
    and the agent instance must be unchanged across the swap (no rebuild).
    """
    agent = Agent(_echo_model(), capabilities=[ManagedAgentSpec(agent_spec=_agent_spec_var)])
    agent_id_before = id(agent)

    # Default spec drives the model to `'test'`.
    result_default = await agent.run('hi')
    response_default = result_default.all_messages()[-1]
    assert isinstance(response_default, ModelResponse)
    assert response_default.model_name == 'test'

    # A spec with no `model` falls back to the agent's base model (the echo `FunctionModel`),
    # proving the swap is per-run and not a one-time rebuild.
    with _agent_spec_var.override(AgentSpec.from_dict({})):
        result_fallback = await agent.run('hi')
    response_fallback = result_fallback.all_messages()[-1]
    assert isinstance(response_fallback, ModelResponse)
    assert response_fallback.model_name == 'function:respond:'

    # Same agent object throughout — the spec resolves per-run, the agent is not rebuilt.
    assert id(agent) == agent_id_before


async def test_no_variables_is_noop() -> None:
    """A `Managed` with no variables set shouldn't crash and shouldn't change behavior."""
    agent = Agent(
        _echo_model(),
        instructions='Just the agent.',
        capabilities=[Managed()],
    )

    result = await agent.run('hi')

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='hi', timestamp=IsDatetime())],
                instructions='Just the agent.',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content="instructions='Just the agent.';settings=None")],
                usage=RequestUsage(input_tokens=51, output_tokens=4),
                model_name='function:respond:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
