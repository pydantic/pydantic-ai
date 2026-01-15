"""Tests for fully_qualified_model_name property across all Model subclasses.

This module contains tests for the fully_qualified_model_name property
which is defined in the base Model class and inherited by all subclasses.
"""

from unittest.mock import Mock

import pytest

from pydantic_ai.models import Model
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.wrapper import WrapperModel


# -------------------- Base Model Tests --------------------
def test_base_model_has_fully_qualified_name_property():
    """Verify that the base Model class has the fully_qualified_model_name property."""
    assert hasattr(Model, 'fully_qualified_model_name')
    assert isinstance(Model.fully_qualified_model_name, property)


def test_fully_qualified_name_format():
    """Test that fully_qualified_model_name follows the 'system:model_name' format."""
    model = TestModel()
    fqn = model.fully_qualified_model_name
    assert ':' in fqn
    assert fqn == f"{model.system}:{model.model_name}"


def test_fully_qualified_name_components():
    """Test that fully_qualified_model_name contains both system and model_name."""
    model = TestModel()
    fqn = model.fully_qualified_model_name
    parts = fqn.split(':', 1)
    assert len(parts) == 2
    assert parts[0] == model.system
    assert parts[1] == model.model_name


# -------------------- TestModel Tests --------------------
def test_test_model_fully_qualified_name():
    """Test TestModel's fully_qualified_model_name."""
    model = TestModel()
    assert model.fully_qualified_model_name == 'test:test'


def test_test_model_inherits_property():
    """Verify TestModel inherits the property from base Model class."""
    model = TestModel()
    assert model.system == 'test'
    assert model.model_name == 'test'
    assert model.fully_qualified_model_name == 'test:test'


# -------------------- WrapperModel Tests --------------------
def test_wrapper_model_delegates_fully_qualified_name():
    """Test that WrapperModel correctly delegates to wrapped model."""
    wrapped = TestModel()
    wrapper = WrapperModel(wrapped)
    assert wrapper.fully_qualified_model_name == wrapped.fully_qualified_model_name
    assert wrapper.fully_qualified_model_name == 'test:test'


# -------------------- FunctionModel Tests --------------------
def test_function_model_fully_qualified_name():
    """Test FunctionModel's fully_qualified_model_name."""
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    def dummy_handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(model_name='test-function', parts=[TextPart(content='test')])

    model = FunctionModel(dummy_handler, model_name='test-function')
    fqn = model.fully_qualified_model_name
    assert fqn == 'function:test-function'
    assert fqn.startswith('function:')
    assert model.model_name in fqn

    # Verify it works in an agent
    agent = Agent(model)
    result = agent.run_sync('test')
    assert result.output == 'test'


# -------------------- Provider-specific Models Tests --------------------
# These tests verify that the property works correctly for all provider models


def test_openai_chat_model_fully_qualified_name():
    """Test OpenAIChatModel inherits fully_qualified_model_name correctly."""
    pytest.importorskip('openai')
    from pydantic_ai.models.openai import OpenAIChatModel

    fake_provider = Mock()
    fake_provider.name = 'openai'
    obj = OpenAIChatModel(provider=fake_provider, model_name='gpt-4')
    assert obj.fully_qualified_model_name == 'openai:gpt-4'


def test_openai_responses_model_fully_qualified_name():
    """Test OpenAIResponsesModel inherits fully_qualified_model_name correctly."""
    pytest.importorskip('openai')
    from pydantic_ai.models.openai import OpenAIResponsesModel

    fake_provider = Mock()
    fake_provider.name = 'openai'
    obj = OpenAIResponsesModel(provider=fake_provider, model_name='gpt-4')
    assert obj.fully_qualified_model_name == 'openai:gpt-4'


def test_anthropic_model_fully_qualified_name():
    """Test AnthropicModel inherits fully_qualified_model_name correctly."""
    pytest.importorskip('pydantic_ai.models.anthropic', exc_type=ImportError)
    from pydantic_ai.models.anthropic import AnthropicModel

    fake_provider = Mock()
    fake_provider.name = 'anthropic'
    obj = AnthropicModel(provider=fake_provider, model_name='claude-3-opus-20240229')
    assert obj.fully_qualified_model_name == 'anthropic:claude-3-opus-20240229'


def test_bedrock_model_fully_qualified_name():
    """Test BedrockConverseModel inherits fully_qualified_model_name correctly."""
    pytest.importorskip('pydantic_ai.models.bedrock', exc_type=ImportError)
    from pydantic_ai.models.bedrock import BedrockConverseModel

    fake_provider = Mock()
    fake_provider.name = 'bedrock'
    obj = BedrockConverseModel(provider=fake_provider, model_name='anthropic.claude-3')
    assert obj.fully_qualified_model_name == 'bedrock:anthropic.claude-3'


def test_cohere_model_fully_qualified_name():
    """Test CohereModel inherits fully_qualified_model_name correctly."""
    pytest.importorskip('pydantic_ai.models.cohere', exc_type=ImportError)
    from pydantic_ai.models.cohere import CohereModel

    fake_provider = Mock()
    fake_provider.name = 'cohere'
    obj = CohereModel(provider=fake_provider, model_name='command-r-plus')
    assert obj.fully_qualified_model_name == 'cohere:command-r-plus'


def test_google_model_fully_qualified_name():
    """Test GoogleModel inherits fully_qualified_model_name correctly."""
    pytest.importorskip('pydantic_ai.models.google', exc_type=ImportError)
    from pydantic_ai.models.google import GoogleModel

    fake_provider = Mock()
    fake_provider.name = 'google-vertex'
    obj = GoogleModel(provider=fake_provider, model_name='gemini-2.0-flash')
    assert obj.fully_qualified_model_name == 'google-vertex:gemini-2.0-flash'


def test_groq_model_fully_qualified_name():
    """Test GroqModel inherits fully_qualified_model_name correctly."""
    pytest.importorskip('pydantic_ai.models.groq', exc_type=ImportError)
    from pydantic_ai.models.groq import GroqModel

    fake_provider = Mock()
    fake_provider.name = 'groq'
    obj = GroqModel(provider=fake_provider, model_name='llama-3.3-70b')
    assert obj.fully_qualified_model_name == 'groq:llama-3.3-70b'


def test_huggingface_model_fully_qualified_name():
    """Test HuggingFaceModel inherits fully_qualified_model_name correctly."""
    pytest.importorskip('pydantic_ai.models.huggingface', exc_type=ImportError)
    from pydantic_ai.models.huggingface import HuggingFaceModel

    fake_provider = Mock()
    fake_provider.name = 'huggingface'
    obj = HuggingFaceModel(provider=fake_provider, model_name='meta-llama/Llama-3.3-70B')
    assert obj.fully_qualified_model_name == 'huggingface:meta-llama/Llama-3.3-70B'


def test_mcp_sampling_model_fully_qualified_name():
    """Test MCPSamplingModel inherits fully_qualified_model_name correctly."""
    pytest.importorskip('pydantic_ai.models.mcp_sampling', exc_type=ImportError)
    from pydantic_ai.models.mcp_sampling import MCPSamplingModel

    fake_session = Mock()
    obj = MCPSamplingModel(session=fake_session, default_max_tokens=16384)
    assert obj.fully_qualified_model_name == 'mcp-sampling:mcp-sampling-model'


def test_mistral_model_fully_qualified_name():
    """Test MistralModel inherits fully_qualified_model_name correctly."""
    pytest.importorskip('pydantic_ai.models.mistral', exc_type=ImportError)
    from pydantic_ai.models.mistral import MistralModel

    fake_provider = Mock()
    fake_provider.name = 'mistral'
    obj = MistralModel(provider=fake_provider, model_name='mistral-large-latest')
    assert obj.fully_qualified_model_name == 'mistral:mistral-large-latest'


def test_outlines_model_fully_qualified_name():
    """Test OutlinesModel inherits fully_qualified_model_name correctly."""
    pytest.importorskip('pydantic_ai.models.outlines', exc_type=ImportError)
    from pydantic_ai.models.outlines import OutlinesModel

    fake_model = Mock()
    fake_model.model_name = 'outlines-test'
    obj = OutlinesModel(model=fake_model)
    # Note: OutlinesModel might have its own model_name logic
    fqn = obj.fully_qualified_model_name
    assert fqn.startswith('outlines:')


def test_openrouter_model_fully_qualified_name():
    """Test OpenRouterModel inherits fully_qualified_model_name correctly."""
    pytest.importorskip('pydantic_ai.models.openrouter', exc_type=ImportError)
    from pydantic_ai.models.openrouter import OpenRouterModel

    fake_provider = Mock()
    fake_provider.name = 'openrouter'
    obj = OpenRouterModel(provider=fake_provider, model_name='meta-llama/llama-3.3-70b')
    assert obj.fully_qualified_model_name == 'openrouter:meta-llama/llama-3.3-70b'


def test_cerebras_model_fully_qualified_name():
    """Test CerebrasModel inherits fully_qualified_model_name correctly."""
    pytest.importorskip('pydantic_ai.models.cerebras', exc_type=ImportError)
    from pydantic_ai.models.cerebras import CerebrasModel

    fake_provider = Mock()
    fake_provider.name = 'cerebras'
    obj = CerebrasModel(provider=fake_provider, model_name='llama-3.3-70b')
    assert obj.fully_qualified_model_name == 'cerebras:llama-3.3-70b'


# -------------------- Edge Cases --------------------
def test_model_name_with_special_characters():
    """Test that special characters in model names are preserved."""
    from pydantic_ai.models.function import AgentInfo, FunctionModel
    from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart

    def dummy_handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(model_name='test-model-v1.2.3', parts=[TextPart(content='test')])

    model = FunctionModel(dummy_handler, model_name='test-model-v1.2.3')
    assert model.fully_qualified_model_name == 'function:test-model-v1.2.3'


def test_model_name_with_slashes():
    """Test that model names with slashes (like HuggingFace) work correctly."""
    from pydantic_ai.models.function import AgentInfo, FunctionModel
    from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart

    def dummy_handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(model_name='org/model-name', parts=[TextPart(content='test')])

    model = FunctionModel(dummy_handler, model_name='org/model-name')
    assert model.fully_qualified_model_name == 'function:org/model-name'