"""Tests for fully_qualified_model_name property across all Model subclasses.

This module contains tests for the fully_qualified_model_name property
across all model types in pydantic_ai.
"""

# All tests use the correct pytest.importorskip() pattern without deprecated parameters

import pytest

from pydantic_ai.models.test import TestModel
from pydantic_ai.models.wrapper import WrapperModel


def test_test_model_fully_qualified_name():
    model = TestModel()
    assert model.fully_qualified_model_name == 'test:test'
    assert ':' in model.fully_qualified_model_name


def test_test_model_has_provider_prefix():
    model = TestModel()
    fqn = model.fully_qualified_model_name
    parts = fqn.split(':')
    assert len(parts) >= 2
    assert parts[0] == 'test'


def test_wrapper_model_delegates_fully_qualified_name():
    wrapped = TestModel()
    wrapper = WrapperModel(wrapped)
    assert wrapper.fully_qualified_model_name == wrapped.fully_qualified_model_name
    assert wrapper.fully_qualified_model_name == 'test:test'


def test_fully_qualified_name_matches_system_and_model_name():
    model = TestModel()
    fqn = model.fully_qualified_model_name
    assert model.system in fqn
    assert model.model_name in fqn


def test_anthropic_model_fully_qualified_name():
    pytest.importorskip('pydantic_ai.models.anthropic', exc_type=ImportError)
    from pydantic_ai.models.anthropic import AnthropicModel

    assert isinstance(AnthropicModel.fully_qualified_model_name, property)


def test_bedrock_model_fully_qualified_name():
    pytest.importorskip('pydantic_ai.models.bedrock', exc_type=ImportError)
    from pydantic_ai.models.bedrock import BedrockConverseModel

    assert isinstance(BedrockConverseModel.fully_qualified_model_name, property)


def test_cohere_model_fully_qualified_name():
    pytest.importorskip('pydantic_ai.models.cohere', exc_type=ImportError)
    from pydantic_ai.models.cohere import CohereModel

    assert isinstance(CohereModel.fully_qualified_model_name, property)


def test_fallback_model_fully_qualified_name():
    from pydantic_ai.models.fallback import FallbackModel

    fallback = FallbackModel(TestModel(), TestModel())
    fqn = fallback.fully_qualified_model_name
    assert fqn.startswith('fallback:')
    assert ':' in fqn


def test_function_model_fully_qualified_name():
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    def dummy_handler(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        return ModelResponse(
            model_name='test-function',
            parts=[TextPart(content='test')],
        )

    model = FunctionModel(dummy_handler, model_name='test-function')
    fqn = model.fully_qualified_model_name
    assert fqn.startswith('function:')
    assert model.model_name in fqn
    
    # Execute the handler to get full coverage
    agent = Agent(model)
    result = agent.run_sync('test')
    assert result.output == 'test'


def test_google_model_fully_qualified_name():
    pytest.importorskip('pydantic_ai.models.google', exc_type=ImportError)
    from pydantic_ai.models.google import GoogleModel

    assert isinstance(GoogleModel.fully_qualified_model_name, property)


def test_groq_model_fully_qualified_name():
    pytest.importorskip('pydantic_ai.models.groq', exc_type=ImportError)
    from pydantic_ai.models.groq import GroqModel

    assert isinstance(GroqModel.fully_qualified_model_name, property)


def test_huggingface_model_fully_qualified_name():
    pytest.importorskip('pydantic_ai.models.huggingface', exc_type=ImportError)
    from pydantic_ai.models.huggingface import HuggingFaceModel

    assert isinstance(HuggingFaceModel.fully_qualified_model_name, property)


def test_mcp_sampling_model_fully_qualified_name():
    pytest.importorskip('pydantic_ai.models.mcp_sampling', exc_type=ImportError)
    from pydantic_ai.models.mcp_sampling import MCPSamplingModel

    assert isinstance(MCPSamplingModel.fully_qualified_model_name, property)


def test_mistral_model_fully_qualified_name():
    pytest.importorskip('pydantic_ai.models.mistral', exc_type=ImportError)
    from pydantic_ai.models.mistral import MistralModel

    assert isinstance(MistralModel.fully_qualified_model_name, property)


def test_openai_chat_model_fully_qualified_name():
    pytest.importorskip('pydantic_ai.models.openai', exc_type=ImportError)
    from pydantic_ai.models.openai import OpenAIChatModel

    assert isinstance(OpenAIChatModel.fully_qualified_model_name, property)


def test_openai_responses_model_fully_qualified_name():
    pytest.importorskip('pydantic_ai.models.openai', exc_type=ImportError)
    from pydantic_ai.models.openai import OpenAIResponsesModel

    assert isinstance(OpenAIResponsesModel.fully_qualified_model_name, property)


def test_outlines_model_fully_qualified_name():
    pytest.importorskip('pydantic_ai.models.outlines', exc_type=ImportError)
    from pydantic_ai.models.outlines import OutlinesModel

    assert isinstance(OutlinesModel.fully_qualified_model_name, property)


def test_openrouter_model_fully_qualified_name():
    pytest.importorskip('pydantic_ai.models.openrouter', exc_type=ImportError)
    from pydantic_ai.models.openrouter import OpenRouterModel

    assert isinstance(OpenRouterModel.fully_qualified_model_name, property)


def test_cerebras_model_fully_qualified_name():
    pytest.importorskip('pydantic_ai.models.cerebras', exc_type=ImportError)
    from pydantic_ai.models.cerebras import CerebrasModel

    assert isinstance(CerebrasModel.fully_qualified_model_name, property)
