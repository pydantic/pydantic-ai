from importlib import import_module

import pytest

from pydantic_ai import UserError
from pydantic_ai.models import infer_model

from ..conftest import TestEnv

# TODO(Marcelo): We need to add Vertex AI to the test cases.

TEST_CASES = [
    ('OPENAI_API_KEY', 'openai:gpt-3.5-turbo', 'gpt-3.5-turbo', 'openai', 'openai', 'OpenAIModel'),
    ('OPENAI_API_KEY', 'gpt-3.5-turbo', 'gpt-3.5-turbo', 'openai', 'openai', 'OpenAIModel'),
    ('OPENAI_API_KEY', 'o1', 'o1', 'openai', 'openai', 'OpenAIModel'),
    ('AZURE_OPENAI_API_KEY', 'azure:gpt-3.5-turbo', 'gpt-3.5-turbo', 'azure', 'azure', 'OpenAIModel'),
    ('GEMINI_API_KEY', 'google-gla:gemini-1.5-flash', 'gemini-1.5-flash', 'google-gla', 'google', 'GoogleModel'),
    ('GEMINI_API_KEY', 'gemini-1.5-flash', 'gemini-1.5-flash', 'google-gla', 'google', 'GoogleModel'),
    (
        'ANTHROPIC_API_KEY',
        'anthropic:claude-3-5-haiku-latest',
        'claude-3-5-haiku-latest',
        'anthropic',
        'anthropic',
        'AnthropicModel',
    ),
    (
        'ANTHROPIC_API_KEY',
        'claude-3-5-haiku-latest',
        'claude-3-5-haiku-latest',
        'anthropic',
        'anthropic',
        'AnthropicModel',
    ),
    (
        'GROQ_API_KEY',
        'groq:llama-3.3-70b-versatile',
        'llama-3.3-70b-versatile',
        'groq',
        'groq',
        'GroqModel',
    ),
    (
        'MISTRAL_API_KEY',
        'mistral:mistral-small-latest',
        'mistral-small-latest',
        'mistral_ai',
        'mistral',
        'MistralModel',
    ),
    (
        'CO_API_KEY',
        'cohere:command',
        'command',
        'cohere',
        'cohere',
        'CohereModel',
    ),
    (
        'AWS_DEFAULT_REGION',
        'bedrock:bedrock-claude-3-5-haiku-latest',
        'bedrock-claude-3-5-haiku-latest',
        'bedrock',
        'bedrock',
        'BedrockConverseModel',
    ),
    (
        'GITHUB_API_KEY',
        'github:xai/grok-3-mini',
        'xai/grok-3-mini',
        'github',
        'github',
        'OpenAIModel',
    ),
    (
        'MOONSHOTAI_API_KEY',
        'moonshotai:kimi-k2-0711-preview',
        'kimi-k2-0711-preview',
        'moonshotai',
        'moonshotai',
        'OpenAIModel',
    ),
    (
        'GROK_API_KEY',
        'grok:grok-3',
        'grok-3',
        'grok',
        'grok',
        'OpenAIModel',
    ),
    (
        'GROK_API_KEY',
        'grok-4',  # Note that the provider and model name are both "grok", so the plain string grok with no prefix works because its also the provider name
        'grok-4',
        'grok',
        'grok',
        'OpenAIModel',
    ),
]


@pytest.mark.parametrize(
    'mock_api_key, model_name, expected_model_name, expected_system, module_name, model_class_name', TEST_CASES
)
def test_infer_model(
    env: TestEnv,
    mock_api_key: str,
    model_name: str,
    expected_model_name: str,
    expected_system: str,
    module_name: str,
    model_class_name: str,
):
    env.set(mock_api_key, 'via-env-var')

    try:
        model_module = import_module(f'pydantic_ai.models.{module_name}')
        expected_model = getattr(model_module, model_class_name)
        m = infer_model(model_name)
    except ImportError:
        pytest.skip(f'{model_name} dependencies not installed')

    assert isinstance(m, expected_model)
    assert m.model_name == expected_model_name
    assert m.system == expected_system

    m2 = infer_model(m)
    assert m2 is m


def test_infer_str_unknown():
    with pytest.raises(UserError, match='Unknown model: foobar'):
        infer_model('foobar')
