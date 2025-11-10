import os
import warnings
from importlib import import_module
from unittest.mock import patch

import pytest

from pydantic_ai import UserError
from pydantic_ai.models import Model, infer_model

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.models.cohere import CohereModel
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel


if not imports_successful():
    pytest.skip('model packages were not installed', allow_module_level=True)  # pragma: lax no cover


# TODO(Marcelo): We need to add Vertex AI to the test cases.

TEST_CASES = [
    pytest.param(
        {'PYDANTIC_AI_GATEWAY_API_KEY': 'gateway-api-key'},
        'gateway/chat:gpt-5',
        'gpt-5',
        'openai',
        'openai',
        OpenAIChatModel,
        id='gateway/chat:gpt-5',
    ),
    pytest.param(
        {'PYDANTIC_AI_GATEWAY_API_KEY': 'gateway-api-key'},
        'gateway/responses:gpt-5',
        'gpt-5',
        'openai',
        'openai',
        OpenAIResponsesModel,
        id='gateway/responses:gpt-5',
    ),
    pytest.param(
        {'PYDANTIC_AI_GATEWAY_API_KEY': 'gateway-api-key'},
        'gateway/groq:llama-3.3-70b-versatile',
        'llama-3.3-70b-versatile',
        'groq',
        'groq',
        GroqModel,
        id='gateway/groq:llama-3.3-70b-versatile',
    ),
    pytest.param(
        {'PYDANTIC_AI_GATEWAY_API_KEY': 'gateway-api-key'},
        'gateway/gemini:gemini-1.5-flash',
        'gemini-1.5-flash',
        'google-vertex',
        'google',
        GoogleModel,
        id='gateway/gemini:gemini-1.5-flash',
    ),
    pytest.param(
        {'PYDANTIC_AI_GATEWAY_API_KEY': 'gateway-api-key'},
        'gateway/anthropic:claude-sonnet-4-5',
        'claude-sonnet-4-5',
        'anthropic',
        'anthropic',
        AnthropicModel,
        id='gateway/anthropic:claude-sonnet-4-5',
    ),
    pytest.param(
        {'PYDANTIC_AI_GATEWAY_API_KEY': 'gateway-api-key'},
        'gateway/converse:amazon.nova-micro-v1:0',
        'amazon.nova-micro-v1:0',
        'bedrock',
        'bedrock',
        BedrockConverseModel,
        id='gateway/converse:amazon.nova-micro-v1:0',
    ),
    pytest.param(
        {'OPENAI_API_KEY': 'openai-api-key'},
        'openai:gpt-3.5-turbo',
        'gpt-3.5-turbo',
        'openai',
        'openai',
        OpenAIChatModel,
    ),
    pytest.param(
        {'OPENAI_API_KEY': 'openai-api-key'},
        'gpt-3.5-turbo',
        'gpt-3.5-turbo',
        'openai',
        'openai',
        OpenAIChatModel,
    ),
    pytest.param(
        {'OPENAI_API_KEY': 'openai-api-key'},
        'o1',
        'o1',
        'openai',
        'openai',
        OpenAIChatModel,
    ),
    pytest.param(
        {
            'AZURE_OPENAI_API_KEY': 'azure-openai-api-key',
            'AZURE_OPENAI_ENDPOINT': 'azure-openai-endpoint',
            'OPENAI_API_VERSION': '2024-12-01-preview',
        },
        'azure:gpt-3.5-turbo',
        'gpt-3.5-turbo',
        'azure',
        'openai',
        OpenAIChatModel,
    ),
    pytest.param(
        {'GEMINI_API_KEY': 'gemini-api-key'},
        'google-gla:gemini-1.5-flash',
        'gemini-1.5-flash',
        'google-gla',
        'google',
        GoogleModel,
    ),
    pytest.param(
        {'GEMINI_API_KEY': 'gemini-api-key'},
        'gemini-1.5-flash',
        'gemini-1.5-flash',
        'google-gla',
        'google',
        GoogleModel,
    ),
    pytest.param(
        {'ANTHROPIC_API_KEY': 'anthropic-api-key'},
        'anthropic:claude-haiku-4-5',
        'claude-haiku-4-5',
        'anthropic',
        'anthropic',
        AnthropicModel,
    ),
    pytest.param(
        {'ANTHROPIC_API_KEY': 'anthropic-api-key'},
        'claude-haiku-4-5',
        'claude-haiku-4-5',
        'anthropic',
        'anthropic',
        AnthropicModel,
    ),
    pytest.param(
        {'GROQ_API_KEY': 'groq-api-key'},
        'groq:llama-3.3-70b-versatile',
        'llama-3.3-70b-versatile',
        'groq',
        'groq',
        GroqModel,
    ),
    pytest.param(
        {'MISTRAL_API_KEY': 'mistral-api-key'},
        'mistral:mistral-small-latest',
        'mistral-small-latest',
        'mistral',
        'mistral',
        MistralModel,
    ),
    pytest.param(
        {'CO_API_KEY': 'co-api-key'},
        'cohere:command',
        'command',
        'cohere',
        'cohere',
        CohereModel,
    ),
    pytest.param(
        {'AWS_DEFAULT_REGION': 'aws-default-region'},
        'bedrock:bedrock-claude-haiku-4-5',
        'bedrock-claude-haiku-4-5',
        'bedrock',
        'bedrock',
        BedrockConverseModel,
    ),
    pytest.param(
        {'GITHUB_API_KEY': 'github-api-key'},
        'github:xai/grok-3-mini',
        'xai/grok-3-mini',
        'github',
        'openai',
        OpenAIChatModel,
    ),
    pytest.param(
        {'MOONSHOTAI_API_KEY': 'moonshotai-api-key'},
        'moonshotai:kimi-k2-0711-preview',
        'kimi-k2-0711-preview',
        'moonshotai',
        'openai',
        OpenAIChatModel,
    ),
    pytest.param(
        {'GROK_API_KEY': 'grok-api-key'},
        'grok:grok-3',
        'grok-3',
        'grok',
        'openai',
        OpenAIChatModel,
    ),
    pytest.param(
        {'OPENAI_API_KEY': 'openai-api-key'},
        'openai-responses:gpt-4o',
        'gpt-4o',
        'openai',
        'openai',
        OpenAIResponsesModel,
    ),
]


@pytest.mark.parametrize(
    'mock_env_vars, model_name, expected_model_name, expected_system, module_name, model_class', TEST_CASES
)
def test_infer_model(
    mock_env_vars: dict[str, str],
    model_name: str,
    expected_model_name: str,
    expected_system: str,
    module_name: str,
    model_class: type[Model],
):
    with patch.dict(os.environ, mock_env_vars):
        model_module = import_module(f'pydantic_ai.models.{module_name}')
        expected_model = getattr(model_module, model_class.__name__)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            m = infer_model(model_name)

        assert isinstance(m, expected_model)
        assert m.model_name == expected_model_name
        assert m.system == expected_system

        m2 = infer_model(m)
        assert m2 is m


def test_infer_model_with_provider():
    from pydantic_ai.providers import openai

    provider_class = openai.OpenAIProvider(api_key='1234', base_url='http://test')
    m = infer_model('openai:gpt-5', lambda x: provider_class)

    assert isinstance(m, OpenAIChatModel)
    assert m._provider is provider_class  # type: ignore
    assert m._provider.base_url == 'http://test'  # type: ignore


def test_infer_str_unknown():
    with pytest.raises(UserError, match='Unknown model: foobar'):
        infer_model('foobar')
