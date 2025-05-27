from collections.abc import Iterator
from typing import Any

import pytest
from typing_extensions import get_args

from pydantic_ai.models import KnownModelName

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModelName
    from pydantic_ai.models.bedrock import BedrockModelName
    from pydantic_ai.models.cohere import CohereModelName
    from pydantic_ai.models.gemini import GeminiModelName
    from pydantic_ai.models.groq import GroqModelName
    from pydantic_ai.models.mistral import MistralModelName
    from pydantic_ai.models.openai import OpenAIModelName

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='some model package was not installed'),
]


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
    openai_names = [f'openai:{n}' for n in get_model_names(OpenAIModelName)] + [
        n for n in get_model_names(OpenAIModelName) if n.startswith('o1') or n.startswith('gpt') or n.startswith('o3')
    ]
    bedrock_names = [f'bedrock:{n}' for n in get_model_names(BedrockModelName)]
    deepseek_names = ['deepseek:deepseek-chat', 'deepseek:deepseek-reasoner']
    extra_names = ['test']

    generated_names = sorted(
        anthropic_names
        + cohere_names
        + google_names
        + groq_names
        + mistral_names
        + openai_names
        + bedrock_names
        + deepseek_names
        + extra_names
    )

    known_model_names = sorted(get_args(KnownModelName.__value__))
    assert generated_names == known_model_names
