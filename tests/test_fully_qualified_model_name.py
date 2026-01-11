"""Tests for fully_qualified_model_name property across all Model subclasses."""

from typing import Any

import pytest

from pydantic_ai.models.test import TestModel
from pydantic_ai.models.wrapper import WrapperModel


def test_test_model_fully_qualified_name():
    """Test TestModel.fully_qualified_model_name format."""
    model = TestModel()
    assert model.fully_qualified_model_name == 'test:test'
    assert ':' in model.fully_qualified_model_name


def test_test_model_has_provider_prefix():
    """Test that fully_qualified_model_name contains provider:name format."""
    model = TestModel()
    fqn = model.fully_qualified_model_name
    parts = fqn.split(':')
    assert len(parts) >= 2, f'Fully qualified name should contain provider:name format, got {fqn}'
    assert parts[0] == 'test'


def test_wrapper_model_delegates_fully_qualified_name():
    """Test WrapperModel delegates fully_qualified_model_name to wrapped model."""
    wrapped = TestModel()
    wrapper = WrapperModel(wrapped)
    assert wrapper.fully_qualified_model_name == wrapped.fully_qualified_model_name
    assert wrapper.fully_qualified_model_name == 'test:test'


def test_fully_qualified_name_matches_system_and_model_name():
    """Test that fully_qualified_model_name combines system and model_name correctly."""
    model = TestModel()
    fqn = model.fully_qualified_model_name
    assert model.system in fqn, f'System {model.system} should be in {fqn}'
    assert model.model_name in fqn, f'Model name {model.model_name} should be in {fqn}'


# Tests for production model classes with optional dependencies
def test_anthropic_model_fully_qualified_name():
    """Test AnthropicModel.fully_qualified_model_name."""
    try:
        from pydantic_ai.models.anthropic import AnthropicModel
    except ImportError:
        pytest.skip('anthropic SDK not installed')

    # Verify the property exists and is properly defined
    assert hasattr(AnthropicModel, 'fully_qualified_model_name')
    assert isinstance(AnthropicModel.fully_qualified_model_name, property)


def test_bedrock_model_fully_qualified_name():
    """Test BedrockConverseModel.fully_qualified_model_name."""
    try:
        from pydantic_ai.models.bedrock import BedrockConverseModel
    except ImportError:
        pytest.skip('bedrock SDK not installed')

    assert hasattr(BedrockConverseModel, 'fully_qualified_model_name')
    assert isinstance(BedrockConverseModel.fully_qualified_model_name, property)


def test_cohere_model_fully_qualified_name():
    """Test CohereModel.fully_qualified_model_name."""
    try:
        from pydantic_ai.models.cohere import CohereModel
    except ImportError:
        pytest.skip('cohere SDK not installed')

    assert hasattr(CohereModel, 'fully_qualified_model_name')
    assert isinstance(CohereModel.fully_qualified_model_name, property)


def test_fallback_model_fully_qualified_name():
    """Test FallbackModel.fully_qualified_model_name."""
    from pydantic_ai.models.fallback import FallbackModel

    model1 = TestModel()
    model2 = TestModel()
    fallback = FallbackModel(model1, model2)
    fqn = fallback.fully_qualified_model_name
    assert ':' in fqn
    assert fqn.startswith('fallback:')
    assert fallback.system.startswith('fallback:')


def test_function_model_fully_qualified_name():
    """Test FunctionModel.fully_qualified_model_name."""
    from pydantic_ai import ModelResponse
    from pydantic_ai.models.function import FunctionModel

    async def dummy_handler(messages: Any, model_settings: Any) -> ModelResponse:
        return ModelResponse(
            finish_reason='stop',
            model_name='test-function',
            parts=[],
        )

    model = FunctionModel(dummy_handler, model_name='test-function')
    fqn = model.fully_qualified_model_name
    assert ':' in fqn
    assert fqn.startswith('function:')
    assert model.system == 'function'
    assert model.model_name in fqn


def test_gemini_model_fully_qualified_name():
    """Test GoogleModel.fully_qualified_model_name (GeminiModel is deprecated)."""
    try:
        from pydantic_ai.models.google import GoogleModel
    except ImportError:
        pytest.skip('google SDK not installed')

    assert hasattr(GoogleModel, 'fully_qualified_model_name')
    assert isinstance(GoogleModel.fully_qualified_model_name, property)


def test_google_model_fully_qualified_name():
    """Test GoogleModel.fully_qualified_model_name."""
    try:
        from pydantic_ai.models.google import GoogleModel
    except ImportError:
        pytest.skip('google SDK not installed')

    assert hasattr(GoogleModel, 'fully_qualified_model_name')
    assert isinstance(GoogleModel.fully_qualified_model_name, property)


def test_groq_model_fully_qualified_name():
    """Test GroqModel.fully_qualified_model_name."""
    try:
        from pydantic_ai.models.groq import GroqModel
    except ImportError:
        pytest.skip('groq SDK not installed')

    assert hasattr(GroqModel, 'fully_qualified_model_name')
    assert isinstance(GroqModel.fully_qualified_model_name, property)


def test_huggingface_model_fully_qualified_name():
    """Test HuggingFaceModel.fully_qualified_model_name."""
    try:
        from pydantic_ai.models.huggingface import HuggingFaceModel
    except ImportError:
        pytest.skip('huggingface SDK not installed')

    assert hasattr(HuggingFaceModel, 'fully_qualified_model_name')
    assert isinstance(HuggingFaceModel.fully_qualified_model_name, property)


def test_mcp_sampling_model_fully_qualified_name():
    """Test MCPSamplingModel.fully_qualified_model_name."""
    try:
        from pydantic_ai.models.mcp_sampling import MCPSamplingModel
    except ImportError:
        pytest.skip('mcp SDK not installed')

    # MCPSamplingModel requires ServerSession, which is hard to mock
    # so we just verify the class implements the property
    assert hasattr(MCPSamplingModel, 'fully_qualified_model_name')
    assert isinstance(MCPSamplingModel.fully_qualified_model_name, property)


def test_mistral_model_fully_qualified_name():
    """Test MistralModel.fully_qualified_model_name."""
    try:
        from pydantic_ai.models.mistral import MistralModel
    except ImportError:
        pytest.skip('mistral SDK not installed')

    assert hasattr(MistralModel, 'fully_qualified_model_name')
    assert isinstance(MistralModel.fully_qualified_model_name, property)


def test_openai_chat_model_fully_qualified_name():
    """Test OpenAIChatModel.fully_qualified_model_name."""
    try:
        from pydantic_ai.models.openai import OpenAIChatModel
    except ImportError:
        pytest.skip('openai SDK not installed')

    assert hasattr(OpenAIChatModel, 'fully_qualified_model_name')
    assert isinstance(OpenAIChatModel.fully_qualified_model_name, property)


def test_openai_responses_model_fully_qualified_name():
    """Test OpenAIResponsesModel.fully_qualified_model_name."""
    try:
        from pydantic_ai.models.openai import OpenAIResponsesModel
    except ImportError:
        pytest.skip('openai SDK not installed')

    assert hasattr(OpenAIResponsesModel, 'fully_qualified_model_name')
    assert isinstance(OpenAIResponsesModel.fully_qualified_model_name, property)


def test_outlines_model_fully_qualified_name():
    """Test OutlinesModel.fully_qualified_model_name."""
    try:
        from pydantic_ai.models.outlines import OutlinesModel
    except ImportError:
        pytest.skip('outlines SDK not installed')

    # OutlinesModel requires more complex setup, so verify it has the property
    assert hasattr(OutlinesModel, 'fully_qualified_model_name')
    assert isinstance(OutlinesModel.fully_qualified_model_name, property)


def test_openrouter_model_fully_qualified_name():
    """Test OpenRouterModel.fully_qualified_model_name."""
    try:
        from pydantic_ai.models.openrouter import OpenRouterModel
    except ImportError:
        pytest.skip('openrouter SDK not installed')

    assert hasattr(OpenRouterModel, 'fully_qualified_model_name')
    assert isinstance(OpenRouterModel.fully_qualified_model_name, property)


def test_cerebras_model_fully_qualified_name():
    """Test CerebrasModel.fully_qualified_model_name."""
    try:
        from pydantic_ai.models.cerebras import CerebrasModel
    except ImportError:
        pytest.skip('cerebras SDK not installed')

    assert hasattr(CerebrasModel, 'fully_qualified_model_name')
    assert isinstance(CerebrasModel.fully_qualified_model_name, property)
