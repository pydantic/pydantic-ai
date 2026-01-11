"""Tests for fully_qualified_model_name property across all Model subclasses."""

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
    AnthropicModel = pytest.importorskip(
        'pydantic_ai.models.anthropic',
        reason='anthropic SDK not installed',
    ).AnthropicModel

    assert isinstance(AnthropicModel.fully_qualified_model_name, property)


def test_bedrock_model_fully_qualified_name():
    BedrockConverseModel = pytest.importorskip(
        'pydantic_ai.models.bedrock',
        reason='bedrock SDK not installed',
    ).BedrockConverseModel

    assert isinstance(BedrockConverseModel.fully_qualified_model_name, property)


def test_cohere_model_fully_qualified_name():
    CohereModel = pytest.importorskip(
        'pydantic_ai.models.cohere',
        reason='cohere SDK not installed',
    ).CohereModel

    assert isinstance(CohereModel.fully_qualified_model_name, property)


def test_fallback_model_fully_qualified_name():
    from pydantic_ai.models.fallback import FallbackModel

    fallback = FallbackModel(TestModel(), TestModel())
    fqn = fallback.fully_qualified_model_name
    assert fqn.startswith('fallback:')
    assert ':' in fqn


def test_function_model_fully_qualified_name():
    from pydantic_ai.models.function import AgentInfo, FunctionModel
    from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart

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


def test_google_model_fully_qualified_name():
    GoogleModel = pytest.importorskip(
        'pydantic_ai.models.google',
        reason='google SDK not installed',
    ).GoogleModel

    assert isinstance(GoogleModel.fully_qualified_model_name, property)


def test_groq_model_fully_qualified_name():
    GroqModel = pytest.importorskip(
        'pydantic_ai.models.groq',
        reason='groq SDK not installed',
    ).GroqModel

    assert isinstance(GroqModel.fully_qualified_model_name, property)


def test_huggingface_model_fully_qualified_name():
    HuggingFaceModel = pytest.importorskip(
        'pydantic_ai.models.huggingface',
        reason='huggingface SDK not installed',
    ).HuggingFaceModel

    assert isinstance(HuggingFaceModel.fully_qualified_model_name, property)


def test_mcp_sampling_model_fully_qualified_name():
    MCPSamplingModel = pytest.importorskip(
        'pydantic_ai.models.mcp_sampling',
        reason='mcp SDK not installed',
    ).MCPSamplingModel

    assert isinstance(MCPSamplingModel.fully_qualified_model_name, property)


def test_mistral_model_fully_qualified_name():
    MistralModel = pytest.importorskip(
        'pydantic_ai.models.mistral',
        reason='mistral SDK not installed',
    ).MistralModel

    assert isinstance(MistralModel.fully_qualified_model_name, property)


def test_openai_chat_model_fully_qualified_name():
    OpenAIChatModel = pytest.importorskip(
        'pydantic_ai.models.openai',
        reason='openai SDK not installed',
    ).OpenAIChatModel

    assert isinstance(OpenAIChatModel.fully_qualified_model_name, property)


def test_openai_responses_model_fully_qualified_name():
    OpenAIResponsesModel = pytest.importorskip(
        'pydantic_ai.models.openai',
        reason='openai SDK not installed',
    ).OpenAIResponsesModel

    assert isinstance(OpenAIResponsesModel.fully_qualified_model_name, property)


def test_outlines_model_fully_qualified_name():
    OutlinesModel = pytest.importorskip(
        'pydantic_ai.models.outlines',
        reason='outlines SDK not installed',
        exc_type=ImportError,
    ).OutlinesModel

    assert isinstance(OutlinesModel.fully_qualified_model_name, property)


def test_openrouter_model_fully_qualified_name():
    OpenRouterModel = pytest.importorskip(
        'pydantic_ai.models.openrouter',
        reason='openrouter SDK not installed',
    ).OpenRouterModel

    assert isinstance(OpenRouterModel.fully_qualified_model_name, property)


def test_cerebras_model_fully_qualified_name():
    CerebrasModel = pytest.importorskip(
        'pydantic_ai.models.cerebras',
        reason='cerebras SDK not installed',
    ).CerebrasModel

    assert isinstance(CerebrasModel.fully_qualified_model_name, property)
