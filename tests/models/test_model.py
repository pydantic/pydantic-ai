from collections.abc import Iterator
from importlib import import_module
from typing import Any, get_args

import pytest

from pydantic_ai import UserError
from pydantic_ai.models import KnownModelName, infer_model
from pydantic_ai.models.anthropic import AnthropicModelName
from pydantic_ai.models.cohere import CohereModelName
from pydantic_ai.models.gemini import GeminiModelName
from pydantic_ai.models.groq import GroqModelName
from pydantic_ai.models.mistral import MistralModelName
from pydantic_ai.models.ollama import OllamaModelName
from pydantic_ai.models.openai import OpenAIModelName

from ..conftest import TestEnv

TEST_CASES = [
    ('OPENAI_API_KEY', 'openai:gpt-3.5-turbo', 'openai:gpt-3.5-turbo', 'openai', 'OpenAIModel'),
    ('OPENAI_API_KEY', 'gpt-3.5-turbo', 'openai:gpt-3.5-turbo', 'openai', 'OpenAIModel'),
    ('OPENAI_API_KEY', 'o1', 'openai:o1', 'openai', 'OpenAIModel'),
    ('GEMINI_API_KEY', 'google-gla:gemini-1.5-flash', 'google-gla:gemini-1.5-flash', 'gemini', 'GeminiModel'),
    ('GEMINI_API_KEY', 'gemini-1.5-flash', 'google-gla:gemini-1.5-flash', 'gemini', 'GeminiModel'),
    (
        'GEMINI_API_KEY',
        'google-vertex:gemini-1.5-flash',
        'google-vertex:gemini-1.5-flash',
        'vertexai',
        'VertexAIModel',
    ),
    (
        'GEMINI_API_KEY',
        'vertexai:gemini-1.5-flash',
        'google-vertex:gemini-1.5-flash',
        'vertexai',
        'VertexAIModel',
    ),
    (
        'ANTHROPIC_API_KEY',
        'anthropic:claude-3-5-haiku-latest',
        'anthropic:claude-3-5-haiku-latest',
        'anthropic',
        'AnthropicModel',
    ),
    (
        'ANTHROPIC_API_KEY',
        'claude-3-5-haiku-latest',
        'anthropic:claude-3-5-haiku-latest',
        'anthropic',
        'AnthropicModel',
    ),
    (
        'GROQ_API_KEY',
        'groq:llama-3.3-70b-versatile',
        'groq:llama-3.3-70b-versatile',
        'groq',
        'GroqModel',
    ),
    ('OLLAMA_API_KEY', 'ollama:llama3', 'ollama:llama3', 'ollama', 'OllamaModel'),
    (
        'MISTRAL_API_KEY',
        'mistral:mistral-small-latest',
        'mistral:mistral-small-latest',
        'mistral',
        'MistralModel',
    ),
    (
        'COHERE_API_KEY',
        'cohere:command',
        'cohere:command',
        'cohere',
        'CohereModel',
    ),
]


@pytest.mark.parametrize('mock_api_key, model_name, expected_model_name, module_name, model_class_name', TEST_CASES)
def test_infer_model(
    env: TestEnv, mock_api_key: str, model_name: str, expected_model_name: str, module_name: str, model_class_name: str
):
    try:
        model_module = import_module(f'pydantic_ai.models.{module_name}')
        expected_model = getattr(model_module, model_class_name)
    except ImportError:
        pytest.skip(f'{model_name} dependencies not installed')

    env.set(mock_api_key, 'via-env-var')

    m = infer_model(model_name)  # pyright: ignore[reportArgumentType]
    assert isinstance(m, expected_model)
    assert m.name() == expected_model_name

    m2 = infer_model(m)
    assert m2 is m


def test_infer_str_unknown():
    with pytest.raises(UserError, match='Unknown model: foobar'):
        infer_model('foobar')  # pyright: ignore[reportArgumentType]


def test_known_model_names():
    def get_model_names(model_name_type: Any) -> Iterator[str]:
        for arg in get_args(model_name_type):
            if isinstance(arg, str):
                yield arg
            else:
                yield from get_model_names(arg)

    anthropic_names = [f'anthropic:{n}' for n in get_model_names(AnthropicModelName)] + [
        n for n in get_model_names(AnthropicModelName) if n.startswith('claude')
    ]
    cohere_names = [f'cohere:{n}' for n in get_model_names(CohereModelName)]
    google_names = [f'google-gla:{n}' for n in get_model_names(GeminiModelName)] + [
        f'google-vertex:{n}' for n in get_model_names(GeminiModelName)
    ]
    groq_names = [f'groq:{n}' for n in get_model_names(GroqModelName)]
    mistral_names = [f'mistral:{n}' for n in get_model_names(MistralModelName)]
    ollama_names = [f'ollama:{n}' for n in get_model_names(OllamaModelName)]
    openai_names = [f'openai:{n}' for n in get_model_names(OpenAIModelName)] + [
        n for n in get_model_names(OpenAIModelName) if n.startswith('o1') or n.startswith('gpt')
    ]
    extra_names = ['test']

    generated_names = sorted(
        anthropic_names
        + cohere_names
        + google_names
        + groq_names
        + mistral_names
        + ollama_names
        + openai_names
        + extra_names
    )
    known_model_names = sorted(get_args(KnownModelName))

    assert generated_names == known_model_names
