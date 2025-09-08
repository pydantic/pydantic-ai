import warnings
from importlib import import_module

import pytest

from pydantic_ai import UserError
from pydantic_ai.models import Model, infer_model

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.models.cohere import CohereModel
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='model packages were not installed'),
]


# TODO(Marcelo): We need to add Vertex AI to the test cases.

TEST_CASES = [
    pytest.param(
        'PYDANTIC_AI_GATEWAY_API_KEY',
        'gateway:openai/gpt-5',
        'gpt-5',
        'openai',
        'openai',
        OpenAIChatModel,
        id='gateway:openai/gpt-5',
    ),
    pytest.param(
        'PYDANTIC_AI_GATEWAY_API_KEY',
        'gateway:groq/llama-3.3-70b-versatile',
        'llama-3.3-70b-versatile',
        'groq',
        'groq',
        GroqModel,
        id='gateway:groq/llama-3.3-70b-versatile',
    ),
    pytest.param(
        'PYDANTIC_AI_GATEWAY_API_KEY',
        'gateway:google-vertex/gemini-1.5-flash',
        'gemini-1.5-flash',
        'google-vertex',
        'google',
        GoogleModel,
        id='gateway:google-vertex/gemini-1.5-flash',
    ),
    ('OPENAI_API_KEY', 'openai:gpt-3.5-turbo', 'gpt-3.5-turbo', 'openai', 'openai', OpenAIChatModel),
    ('OPENAI_API_KEY', 'gpt-3.5-turbo', 'gpt-3.5-turbo', 'openai', 'openai', OpenAIChatModel),
    ('OPENAI_API_KEY', 'o1', 'o1', 'openai', 'openai', OpenAIChatModel),
    ('AZURE_OPENAI_API_KEY', 'azure:gpt-3.5-turbo', 'gpt-3.5-turbo', 'azure', 'openai', OpenAIChatModel),
    ('GEMINI_API_KEY', 'google-gla:gemini-1.5-flash', 'gemini-1.5-flash', 'google-gla', 'google', GoogleModel),
    ('GEMINI_API_KEY', 'gemini-1.5-flash', 'gemini-1.5-flash', 'google-gla', 'google', GoogleModel),
    (
        'ANTHROPIC_API_KEY',
        'anthropic:claude-3-5-haiku-latest',
        'claude-3-5-haiku-latest',
        'anthropic',
        'anthropic',
        AnthropicModel,
    ),
    (
        'ANTHROPIC_API_KEY',
        'claude-3-5-haiku-latest',
        'claude-3-5-haiku-latest',
        'anthropic',
        'anthropic',
        AnthropicModel,
    ),
    (
        'GROQ_API_KEY',
        'groq:llama-3.3-70b-versatile',
        'llama-3.3-70b-versatile',
        'groq',
        'groq',
        GroqModel,
    ),
    (
        'MISTRAL_API_KEY',
        'mistral:mistral-small-latest',
        'mistral-small-latest',
        'mistral',
        'mistral',
        MistralModel,
    ),
    (
        'CO_API_KEY',
        'cohere:command',
        'command',
        'cohere',
        'cohere',
        CohereModel,
    ),
    (
        'AWS_DEFAULT_REGION',
        'bedrock:bedrock-claude-3-5-haiku-latest',
        'bedrock-claude-3-5-haiku-latest',
        'bedrock',
        'bedrock',
        BedrockConverseModel,
    ),
    (
        'GITHUB_API_KEY',
        'github:xai/grok-3-mini',
        'xai/grok-3-mini',
        'github',
        'openai',
        OpenAIChatModel,
    ),
    (
        'MOONSHOTAI_API_KEY',
        'moonshotai:kimi-k2-0711-preview',
        'kimi-k2-0711-preview',
        'moonshotai',
        'openai',
        OpenAIChatModel,
    ),
    (
        'GROK_API_KEY',
        'grok:grok-3',
        'grok-3',
        'grok',
        'openai',
        OpenAIChatModel,
    ),
    (
        'OPENAI_API_KEY',
        'openai-responses:gpt-4o',
        'gpt-4o',
        'openai',
        'openai',
        OpenAIResponsesModel,
    ),
]


@pytest.mark.parametrize(
    'mock_api_key, model_name, expected_model_name, expected_system, module_name, model_class', TEST_CASES
)
def test_infer_model(
    env: TestEnv,
    mock_api_key: str,
    model_name: str,
    expected_model_name: str,
    expected_system: str,
    module_name: str,
    model_class: type[Model],
):
    env.set(mock_api_key, 'via-env-var')

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


def test_infer_str_unknown():
    with pytest.raises(UserError, match='Unknown model: foobar'):
        infer_model('foobar')
