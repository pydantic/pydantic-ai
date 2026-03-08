import os
from collections.abc import Iterator
from functools import partial
from typing import Any, Literal, get_args

import httpx
import pytest
from typing_extensions import TypedDict

from pydantic_ai.models import KnownModelName
from pydantic_ai.providers.gateway import ModelProvider as GatewayModelProvider

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
    from pydantic_ai.models.xai import XaiModelName
    from pydantic_ai.providers.deepseek import DeepSeekModelName
    from pydantic_ai.providers.grok import GrokModelName
    from pydantic_ai.providers.moonshotai import MoonshotAIModelName

if not imports_successful():  # pragma: lax no cover
    # Define placeholders so the module can be loaded for test collection
    AnthropicModelName = BedrockModelName = CohereModelName = GoogleModelName = None
    GroqModelName = HuggingFaceModelName = MistralModelName = OpenAIModelName = None
    DeepSeekModelName = GrokModelName = XaiModelName = MoonshotAIModelName = None

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


_PROVIDER_TO_MODEL_NAMES = {
    'anthropic': AnthropicModelName,
    'bedrock': BedrockModelName,
    'cohere': CohereModelName,
    'deepseek': DeepSeekModelName,
    'google-gla': GoogleModelName,
    'google-vertex': GoogleModelName,
    'grok': GrokModelName,
    'xai': XaiModelName,
    'groq': GroqModelName,
    'huggingface': HuggingFaceModelName,
    'mistral': MistralModelName,
    'moonshotai': MoonshotAIModelName,
    'openai': OpenAIModelName,
}


def test_known_model_names():  # pragma: lax no cover
    # Coverage seems to be misbehaving..?
    def get_model_names(model_name_type: Any) -> Iterator[str]:
        for arg in get_args(model_name_type):
            if isinstance(arg, str):
                yield arg
            else:
                yield from get_model_names(arg)

    all_generated_names = [
        f'{provider}:{n}'
        for provider, model_names in _PROVIDER_TO_MODEL_NAMES.items()
        for n in get_model_names(model_names)
    ]

    cerebras_names = get_cerebras_model_names()
    heroku_names = get_heroku_model_names()
    gateway_names = [
        f'gateway/{provider}:{model_name}'
        for provider in GatewayModelProvider.__args__
        for model_name in get_model_names(_PROVIDER_TO_MODEL_NAMES[provider])
    ]

    extra_names = ['test']

    generated_names = sorted(all_generated_names + gateway_names + heroku_names + cerebras_names + extra_names)

    known_model_names = sorted(get_args(KnownModelName.__value__))

    if generated_names != known_model_names:
        errors: list[str] = []
        missing_names = set(generated_names) - set(known_model_names)
        if missing_names:
            errors.append(f'Missing names: {missing_names}')
        extra_names = set(known_model_names) - set(generated_names)
        if extra_names:
            errors.append(f'Extra names: {extra_names}')
        raise AssertionError('\n'.join(errors))


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
