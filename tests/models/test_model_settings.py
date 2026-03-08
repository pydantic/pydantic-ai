"""Tests for per-model settings functionality"""

from __future__ import annotations

import asyncio

from pydantic_ai import Agent
from pydantic_ai.direct import model_request as direct_model_request
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentedModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings


def test_model_settings_property():
    """Test that the Model base class settings property works correctly."""
    # Test with settings
    settings = ModelSettings(max_tokens=100, temperature=0.5)
    test_model = TestModel(settings=settings)
    assert test_model.settings == settings

    # Test without settings
    test_model_no_settings = TestModel()
    assert test_model_no_settings.settings is None


def test_function_model_settings():
    """Test that FunctionModel correctly stores and returns settings."""
    settings = ModelSettings(max_tokens=200, temperature=0.7)

    def simple_response(messages: list[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('response')])  # pragma: no cover

    # Test with settings
    function_model = FunctionModel(simple_response, settings=settings)
    assert function_model.settings == settings

    # Test without settings
    function_model_no_settings = FunctionModel(simple_response)
    assert function_model_no_settings.settings is None


def test_wrapper_model_settings_delegation():
    """Test that WrapperModel correctly delegates settings to wrapped model."""
    # Create a base model with settings
    base_settings = ModelSettings(max_tokens=150, temperature=0.6)
    base_model = TestModel(settings=base_settings)

    # Create wrapper - it should delegate to wrapped model's settings
    wrapper = WrapperModel(base_model)
    assert wrapper.settings == base_settings

    # Test with wrapped model without settings
    base_model_no_settings = TestModel()
    wrapper_no_settings = WrapperModel(base_model_no_settings)
    assert wrapper_no_settings.settings is None


def test_instrumented_model_settings_delegation():
    """Test that InstrumentedModel correctly delegates settings to wrapped model."""
    # Create a base model with settings
    base_settings = ModelSettings(max_tokens=100, temperature=0.5)
    base_model = TestModel(settings=base_settings)

    # InstrumentedModel should delegate settings to wrapped model
    instrumented = InstrumentedModel(base_model)
    assert instrumented.settings == base_settings

    # Test with wrapped model without settings
    base_model_no_settings = TestModel()
    instrumented_no_settings = InstrumentedModel(base_model_no_settings)
    assert instrumented_no_settings.settings is None


def test_settings_merge_hierarchy():
    """Test the complete settings merge hierarchy: model -> agent -> run."""
    # Create a function that captures the merged settings
    captured_settings = None

    def capture_settings(messages: list[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
        nonlocal captured_settings
        captured_settings = agent_info.model_settings
        return ModelResponse(parts=[TextPart('captured')])

    # Model settings (lowest priority)
    model_settings = ModelSettings(max_tokens=100, temperature=0.5, top_p=0.8, seed=123)
    model = FunctionModel(capture_settings, settings=model_settings)

    # Agent settings (medium priority)
    agent_settings = ModelSettings(
        max_tokens=200,  # overrides model
        temperature=0.6,  # overrides model
        frequency_penalty=0.1,  # new setting
    )
    agent = Agent(model=model, model_settings=agent_settings)

    # Run settings (highest priority)
    run_settings = ModelSettings(
        temperature=0.7,  # overrides agent and model
        presence_penalty=0.2,  # new setting
        seed=456,  # overrides model
    )

    # Run the agent
    result = agent.run_sync('test', model_settings=run_settings)
    assert result.output == 'captured'

    # Verify the merged settings follow the correct precedence
    assert captured_settings is not None
    assert captured_settings['temperature'] == 0.7  # from run_settings
    assert captured_settings['max_tokens'] == 200  # from agent_settings
    assert captured_settings['top_p'] == 0.8  # from model_settings
    assert captured_settings['seed'] == 456  # from run_settings
    assert captured_settings['frequency_penalty'] == 0.1  # from agent_settings
    assert captured_settings['presence_penalty'] == 0.2  # from run_settings


def test_none_settings_in_hierarchy():
    """Test that None settings at any level don't break the merge hierarchy."""
    captured_settings = None

    def capture_settings(messages: list[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
        nonlocal captured_settings
        captured_settings = agent_info.model_settings
        return ModelResponse(parts=[TextPart('captured')])

    # Model with no settings
    model = FunctionModel(capture_settings, settings=None)

    # Agent with settings
    agent_settings = ModelSettings(max_tokens=150, temperature=0.5)
    agent = Agent(model=model, model_settings=agent_settings)

    # Run with no additional settings
    result = agent.run_sync('test', model_settings=None)
    assert result.output == 'captured'

    # Should have agent settings
    assert captured_settings is not None
    assert captured_settings['max_tokens'] == 150
    assert captured_settings['temperature'] == 0.5


def test_empty_settings_objects():
    """Test that empty ModelSettings objects work correctly in the hierarchy."""
    captured_settings = None

    def capture_settings(messages: list[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
        nonlocal captured_settings
        captured_settings = agent_info.model_settings
        return ModelResponse(parts=[TextPart('captured')])

    # All levels have empty settings
    model = FunctionModel(capture_settings, settings=ModelSettings())
    agent = Agent(model=model, model_settings=ModelSettings())

    # Run with one actual setting
    run_settings = ModelSettings(temperature=0.75)
    result = agent.run_sync('test', model_settings=run_settings)
    assert result.output == 'captured'

    # Should only have the run setting
    assert captured_settings is not None
    assert captured_settings.get('temperature') == 0.75
    assert len(captured_settings) == 1  # Only one setting should be present


def test_direct_model_request_merges_model_settings():
    """Ensure direct requests merge model defaults with provided run settings."""

    captured_settings = None

    async def capture(messages: list[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
        nonlocal captured_settings
        captured_settings = agent_info.model_settings
        return ModelResponse(parts=[TextPart('ok')])

    model = FunctionModel(
        capture,
        settings=ModelSettings(max_tokens=50, temperature=0.3),
    )

    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('hi')]
    run_settings = ModelSettings(temperature=0.9, top_p=0.2)

    async def _run() -> ModelResponse:
        return await direct_model_request(
            model,
            messages,
            model_settings=run_settings,
            model_request_parameters=ModelRequestParameters(),
        )

    response = asyncio.run(_run())

    assert response.parts == [TextPart('ok')]
    assert captured_settings == {
        'max_tokens': 50,
        'temperature': 0.9,
        'top_p': 0.2,
    }
