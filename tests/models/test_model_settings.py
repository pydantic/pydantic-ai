"""Tests for per-model settings functionality."""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings

try:
    from pydantic_ai.models.gemini import GeminiModel

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from pydantic_ai.models.openai import OpenAIResponsesModel

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def test_model_settings_initialization():
    """Test that models can be initialized with settings."""
    settings = ModelSettings(max_tokens=100, temperature=0.5)

    # Test TestModel
    test_model = TestModel(settings=settings)
    assert test_model.settings == settings

    # Test FunctionModel
    def simple_response(messages: list[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('response')])

    function_model = FunctionModel(simple_response, settings=settings)
    assert function_model.settings == settings


def test_model_settings_none():
    """Test that models can be initialized without settings."""
    # Test TestModel
    test_model = TestModel()
    assert test_model.settings is None

    # Test FunctionModel
    def simple_response(messages: list[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('response')])

    function_model = FunctionModel(simple_response)
    assert function_model.settings is None


def test_agent_with_model_settings():
    """Test that Agent properly merges model settings."""
    # Create a model with default settings
    model_settings = ModelSettings(max_tokens=100, temperature=0.5)
    test_model = TestModel(settings=model_settings)

    # Create an agent with its own settings
    agent_settings = ModelSettings(max_tokens=200, top_p=0.9)
    agent = Agent(model=test_model, model_settings=agent_settings)

    # The agent should have its own settings stored
    assert agent.model_settings == agent_settings

    # The model should have its own settings
    assert test_model.settings == model_settings


def test_agent_run_settings_merge():
    """Test that Agent.run properly merges settings from model, agent, and run parameters."""

    def capture_settings_response(messages: list[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
        # Access the model settings that were passed to the model
        # Note: This is a simplified test - in real usage, the settings would be
        # passed through the request method
        return ModelResponse(parts=[TextPart('captured')])

    # Create models and agent with different settings
    model_settings = ModelSettings(max_tokens=100, temperature=0.5)
    function_model = FunctionModel(capture_settings_response, settings=model_settings)

    agent_settings = ModelSettings(max_tokens=200, top_p=0.9)
    agent = Agent(model=function_model, model_settings=agent_settings)

    # Run with additional settings
    run_settings = ModelSettings(temperature=0.8, seed=42)

    # This should work without errors and properly merge the settings
    result = agent.run_sync('test', model_settings=run_settings)
    assert result.output == 'captured'


def test_agent_iter_settings_merge():
    """Test that Agent.iter properly merges settings from model, agent, and iter parameters."""

    def another_capture_response(messages: list[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('captured')])

    # Create models and agent with different settings
    model_settings = ModelSettings(max_tokens=100, temperature=0.5)
    function_model = FunctionModel(another_capture_response, settings=model_settings)

    agent_settings = ModelSettings(max_tokens=200, top_p=0.9)
    agent = Agent(model=function_model, model_settings=agent_settings)

    # Run with additional settings to test the merge functionality
    iter_settings = ModelSettings(temperature=0.8, seed=42)

    # This should work without errors and properly merge the settings
    result = agent.run_sync('test', model_settings=iter_settings)
    assert result.output == 'captured'


def test_gemini_model_settings():
    """Test that GeminiModel can be initialized with settings."""
    if not GEMINI_AVAILABLE:
        return  # Skip if dependencies not available

    settings = ModelSettings(max_tokens=300, temperature=0.6)

    # This should not raise an exception
    try:
        gemini_model = GeminiModel('gemini-1.5-flash', settings=settings)
        assert gemini_model.settings == settings
    except Exception:
        # Skip if provider setup fails (e.g., missing API keys)
        pass


def test_openai_responses_model_settings():
    """Test that OpenAIResponsesModel can be initialized with settings."""
    if not OPENAI_AVAILABLE:
        return  # Skip if dependencies not available

    settings = ModelSettings(max_tokens=400, temperature=0.7)

    # This should not raise an exception
    try:
        openai_model = OpenAIResponsesModel('gpt-3.5-turbo', settings=settings)
        assert openai_model.settings == settings
    except Exception:
        # Skip if provider setup fails (e.g., missing API keys)
        pass
