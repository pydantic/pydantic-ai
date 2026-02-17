import os
import warnings
from importlib import import_module
from unittest.mock import patch

import pytest

from pydantic_ai import UserError
from pydantic_ai.models import DEFAULT_PROFILE, Model, infer_model, infer_model_profile, parse_model_id

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.models.cohere import CohereModel
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.models.openrouter import OpenRouterModel

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
    pytest.param(
        {'OPENROUTER_API_KEY': 'openrouter-api-key'},
        'openrouter:anthropic/claude-3.5-sonnet',
        'anthropic/claude-3.5-sonnet',
        'openrouter',
        'openrouter',
        OpenRouterModel,
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

        # Test that model_id matches the provider:model string that was passed in
        assert m.model_id == f'{expected_system}:{expected_model_name}'

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


@pytest.mark.parametrize(
    ('model_id', 'expected'),
    [
        pytest.param('openai:gpt-5', ('openai', 'gpt-5'), id='provider:model'),
        pytest.param('anthropic:claude-3', ('anthropic', 'claude-3'), id='anthropic:model'),
        pytest.param('gpt-4', ('openai', 'gpt-4'), id='legacy-gpt'),
        pytest.param('o1-mini', ('openai', 'o1-mini'), id='legacy-o1'),
        pytest.param('o3-mini', ('openai', 'o3-mini'), id='legacy-o3'),
        pytest.param('claude-3-opus', ('anthropic', 'claude-3-opus'), id='legacy-claude'),
        pytest.param('gemini-1.5-flash', ('google-gla', 'gemini-1.5-flash'), id='legacy-gemini'),
        pytest.param('unknown-model', (None, 'unknown-model'), id='unknown'),
        pytest.param('custom:model:with:colons', ('custom', 'model:with:colons'), id='multiple-colons'),
        pytest.param('gateway/openai:gpt-5', ('gateway/openai', 'gpt-5'), id='gateway-prefix'),
    ],
)
def test_parse_model_id(model_id: str, expected: tuple[str | None, str]):
    assert parse_model_id(model_id) == expected


@pytest.mark.parametrize(
    ('model_id', 'is_default'),
    [
        pytest.param('openai:gpt-5', False, id='openai'),
        pytest.param('anthropic:claude-sonnet-4-5', False, id='anthropic'),
        pytest.param('gateway/openai:gpt-5', False, id='gateway-openai'),
        pytest.param('unknown-provider:some-model', True, id='unknown-provider'),
        pytest.param('unknown-model', True, id='unknown-no-prefix'),
        pytest.param('nebius:model-without-slash', False, id='provider-unknown-model'),
        pytest.param('google:gemini-2.0-flash', False, id='google-shorthand'),
        pytest.param('openrouter:model-without-slash', True, id='openrouter-no-slash'),
        pytest.param('together:model-without-slash', True, id='together-no-slash'),
    ],
)
def test_infer_model_profile(model_id: str, is_default: bool):
    profile = infer_model_profile(model_id)
    if is_default:
        assert profile is DEFAULT_PROFILE
    else:
        assert profile is not DEFAULT_PROFILE


@pytest.mark.parametrize(
    ('model_id', 'provider_path', 'model_name'),
    [
        pytest.param('openai:gpt-5', 'pydantic_ai.providers.openai.OpenAIProvider', 'gpt-5', id='openai'),
        pytest.param(
            'anthropic:claude-sonnet-4-5',
            'pydantic_ai.providers.anthropic.AnthropicProvider',
            'claude-sonnet-4-5',
            id='anthropic',
        ),
        pytest.param(
            'google-gla:gemini-2.0-flash',
            'pydantic_ai.providers.google.GoogleProvider',
            'gemini-2.0-flash',
            id='google-gla',
        ),
        pytest.param(
            'google:gemini-2.0-flash',
            'pydantic_ai.providers.google.GoogleProvider',
            'gemini-2.0-flash',
            id='google-shorthand',
        ),
    ],
)
def test_infer_model_profile_matches_provider(model_id: str, provider_path: str, model_name: str):
    """Verify infer_model_profile returns the same profile as the provider's model_profile."""
    module_path, class_name = provider_path.rsplit('.', 1)
    module = import_module(module_path)
    provider_class = getattr(module, class_name)

    profile = infer_model_profile(model_id)
    provider_profile = provider_class.model_profile(model_name)
    assert profile == provider_profile
