import os
from collections.abc import Iterator
from functools import partial
from typing import Any, Literal, get_args

import httpx
import pytest
from typing_extensions import TypedDict

from pydantic_ai.models import KnownModelName

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModelName
    from pydantic_ai.models.bedrock import BedrockModelName
    from pydantic_ai.models.cohere import CohereModelName
    from pydantic_ai.models.google import GoogleModelName
    from pydantic_ai.models.groq import GroqModelName
    from pydantic_ai.models.huggingface import HuggingFaceModelName
    from pydantic_ai.models.mistral import MistralModelName
    from pydantic_ai.models.openai import OpenAIModelName
    from pydantic_ai.providers.grok import GrokModelName
    from pydantic_ai.providers.moonshotai import MoonshotAIModelName

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='some model package was not installed'),
    pytest.mark.vcr,
]


def modify_response(response: dict[str, Any], filter_headers: list[str]) -> dict[str, Any]:  # pragma: lax no cover
    for header in response['headers'].copy():
        assert isinstance(header, str)
        if header.lower() in filter_headers:
            del response['headers'][header]
    return response


@pytest.fixture(scope='module')
def vcr_config():  # pragma: lax no cover
    if os.getenv('CI') or not os.getenv('CEREBRAS_API_KEY'):
        return {'record_mode': 'none'}

    return {
        'record_mode': 'rewrite',
        'filter_headers': ['accept-encoding'],
        'before_record_response': partial(modify_response, filter_headers=['cache-control', 'connection']),
    }


def test_known_model_names():  # pragma: lax no cover
    # Coverage seems to be misbehaving..?
    def get_model_names(model_name_type: Any) -> Iterator[str]:
        for arg in get_args(model_name_type):
            if isinstance(arg, str):
                yield arg
            else:
                yield from get_model_names(arg)

    anthropic_names = [f'anthropic:{n}' for n in get_model_names(AnthropicModelName)]
    cohere_names = [f'cohere:{n}' for n in get_model_names(CohereModelName)]
    google_names = [f'google-gla:{n}' for n in get_model_names(GoogleModelName)] + [
        f'google-vertex:{n}' for n in get_model_names(GoogleModelName)
    ]
    grok_names = [f'grok:{n}' for n in get_model_names(GrokModelName)]
    groq_names = [f'groq:{n}' for n in get_model_names(GroqModelName)]
    moonshotai_names = [f'moonshotai:{n}' for n in get_model_names(MoonshotAIModelName)]
    mistral_names = [f'mistral:{n}' for n in get_model_names(MistralModelName)]
    openai_names = [f'openai:{n}' for n in get_model_names(OpenAIModelName)]
    bedrock_names = [f'bedrock:{n}' for n in get_model_names(BedrockModelName)]
    deepseek_names = ['deepseek:deepseek-chat', 'deepseek:deepseek-reasoner']
    huggingface_names = [f'huggingface:{n}' for n in get_model_names(HuggingFaceModelName)]
    heroku_names = get_heroku_model_names()
    cerebras_names = get_cerebras_model_names()
    extra_names = ['test']

    generated_names = sorted(
        anthropic_names
        + cohere_names
        + google_names
        + grok_names
        + groq_names
        + mistral_names
        + moonshotai_names
        + openai_names
        + bedrock_names
        + deepseek_names
        + huggingface_names
        + heroku_names
        + cerebras_names
        + extra_names
    )

    known_model_names = sorted(get_args(KnownModelName.__value__))
    assert generated_names == known_model_names


class HerokuModel(TypedDict):
    model_id: str
    regions: list[str]
    type: list[str]


def get_heroku_model_names():
    response = httpx.get('https://us.inference.heroku.com/available-models')

    if response.status_code != 200:
        pytest.skip(f'Heroku AI returned status code {response.status_code}')  # pragma: lax no cover

    heroku_models: list[HerokuModel] = response.json()

    models: list[str] = []
    for model in heroku_models:
        if 'text-to-text' in model['type']:
            models.append(f'heroku:{model["model_id"]}')
    return sorted(models)


class CerebrasModel(TypedDict):
    created: int
    id: str
    object: Literal['model']
    owned_by: Literal['Cerebras']


def get_cerebras_model_names():  # pragma: lax no cover
    api_key = os.getenv('CEREBRAS_API_KEY', 'testing')

    response = httpx.get(
        'https://api.cerebras.ai/v1/models',
        headers={'Authorization': f'Bearer {api_key}', 'Accept': 'application/json', 'Accept-Encoding': 'identity'},
    )

    if response.status_code != 200:
        pytest.skip(f'Cerebras returned status code {response.status_code}')  # pragma: lax no cover

    cerebras_models: list[CerebrasModel] = response.json()['data']
    return sorted(f'cerebras:{model["id"]}' for model in cerebras_models)
