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
    from pydantic_ai.models.openai import DEPRECATED_OPENAI_MODELS, OpenAIModelName
    from pydantic_ai.models.xai import XaiModelName
    from pydantic_ai.providers.deepseek import DeepSeekModelName
    from pydantic_ai.providers.grok import GrokModelName
    from pydantic_ai.providers.moonshotai import MoonshotAIModelName

if not imports_successful():  # pragma: lax no cover
    # Define placeholders so the module can be loaded for test collection
    AnthropicModelName = BedrockModelName = CohereModelName = GoogleModelName = None
    GroqModelName = HuggingFaceModelName = MistralModelName = OpenAIModelName = None
    DEPRECATED_OPENAI_MODELS: frozenset[str] = frozenset()  # pyright: ignore[reportConstantRedefinition]
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

_PROVIDER_DEPRECATED_MODELS: dict[str, frozenset[str]] = {
    'openai': DEPRECATED_OPENAI_MODELS,
}

UNSUPPORTED_GATEWAY_MODEL_NAMES = frozenset(
    {
        'gateway/bedrock:amazon.titan-text-express-v1',
        'gateway/bedrock:amazon.titan-text-lite-v1',
        'gateway/bedrock:amazon.titan-tg1-large',
        'gateway/bedrock:anthropic.claude-3-5-haiku-20241022-v1:0',
        'gateway/bedrock:anthropic.claude-3-5-sonnet-20241022-v2:0',
        'gateway/bedrock:anthropic.claude-3-7-sonnet-20250219-v1:0',
        'gateway/bedrock:anthropic.claude-3-opus-20240229-v1:0',
        'gateway/bedrock:anthropic.claude-3-sonnet-20240229-v1:0',
        'gateway/bedrock:anthropic.claude-haiku-4-5-20251001-v1:0',
        'gateway/bedrock:anthropic.claude-instant-v1',
        'gateway/bedrock:anthropic.claude-opus-4-20250514-v1:0',
        'gateway/bedrock:anthropic.claude-sonnet-4-20250514-v1:0',
        'gateway/bedrock:anthropic.claude-sonnet-4-5-20250929-v1:0',
        'gateway/bedrock:anthropic.claude-sonnet-4-6',
        'gateway/bedrock:anthropic.claude-v2',
        'gateway/bedrock:anthropic.claude-v2:1',
        'gateway/bedrock:cohere.command-light-text-v14',
        'gateway/bedrock:cohere.command-r-plus-v1:0',
        'gateway/bedrock:cohere.command-r-v1:0',
        'gateway/bedrock:cohere.command-text-v14',
        'gateway/bedrock:meta.llama3-1-405b-instruct-v1:0',
        'gateway/bedrock:meta.llama3-1-70b-instruct-v1:0',
        'gateway/bedrock:meta.llama3-1-8b-instruct-v1:0',
        'gateway/bedrock:meta.llama3-70b-instruct-v1:0',
        'gateway/bedrock:meta.llama3-8b-instruct-v1:0',
        'gateway/bedrock:mistral.mistral-7b-instruct-v0:2',
        'gateway/bedrock:mistral.mistral-large-2402-v1:0',
        'gateway/bedrock:mistral.mistral-large-2407-v1:0',
        'gateway/bedrock:mistral.mixtral-8x7b-instruct-v0:1',
        'gateway/bedrock:us.amazon.nova-2-lite-v1:0',
        'gateway/bedrock:us.amazon.nova-lite-v1:0',
        'gateway/bedrock:us.amazon.nova-micro-v1:0',
        'gateway/bedrock:us.amazon.nova-pro-v1:0',
        'gateway/bedrock:us.anthropic.claude-3-5-haiku-20241022-v1:0',
        'gateway/bedrock:us.anthropic.claude-3-5-sonnet-20240620-v1:0',
        'gateway/bedrock:us.anthropic.claude-3-5-sonnet-20241022-v2:0',
        'gateway/bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0',
        'gateway/bedrock:us.anthropic.claude-3-haiku-20240307-v1:0',
        'gateway/bedrock:us.anthropic.claude-3-opus-20240229-v1:0',
        'gateway/bedrock:us.anthropic.claude-3-sonnet-20240229-v1:0',
        'gateway/bedrock:us.anthropic.claude-haiku-4-5-20251001-v1:0',
        'gateway/bedrock:us.anthropic.claude-opus-4-20250514-v1:0',
        'gateway/bedrock:us.anthropic.claude-sonnet-4-20250514-v1:0',
        'gateway/bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0',
        'gateway/bedrock:us.anthropic.claude-sonnet-4-6',
        'gateway/bedrock:us.meta.llama3-1-70b-instruct-v1:0',
        'gateway/bedrock:us.meta.llama3-1-8b-instruct-v1:0',
        'gateway/bedrock:us.meta.llama3-2-11b-instruct-v1:0',
        'gateway/bedrock:us.meta.llama3-2-1b-instruct-v1:0',
        'gateway/bedrock:us.meta.llama3-2-3b-instruct-v1:0',
        'gateway/bedrock:us.meta.llama3-2-90b-instruct-v1:0',
        'gateway/bedrock:us.meta.llama3-3-70b-instruct-v1:0',
        'gateway/google-vertex:gemini-2.0-flash',
        'gateway/google-vertex:gemini-2.0-flash-lite',
        'gateway/google-vertex:gemini-2.5-flash-preview-09-2025',
        'gateway/google-vertex:gemini-3-pro-preview',
        'gateway/google-vertex:gemini-flash-latest',
        'gateway/google-vertex:gemini-flash-lite-latest',
        'gateway/groq:meta-llama/llama-prompt-guard-2-22m',
        'gateway/groq:meta-llama/llama-prompt-guard-2-86m',
        'gateway/groq:meta-llama/llama-guard-4-12b',
        'gateway/groq:meta-llama/llama-4-maverick-17b-128e-instruct',
        'gateway/groq:playai-tts',
        'gateway/groq:playai-tts-arabic',
        'gateway/groq:qwen/qwen-3-32b',
        'gateway/groq:whisper-large-v3',
        'gateway/groq:whisper-large-v3-turbo',
        'gateway/openai:chatgpt-4o-latest',
        'gateway/openai:codex-mini-latest',
        'gateway/openai:computer-use-preview',
        'gateway/openai:computer-use-preview-2025-03-11',
        'gateway/openai:gpt-3.5-turbo-0301',
        'gateway/openai:gpt-3.5-turbo-0613',
        'gateway/openai:gpt-3.5-turbo-16k-0613',
        'gateway/openai:gpt-4-0125-preview',
        'gateway/openai:gpt-4-0314',
        'gateway/openai:gpt-4-1106-preview',
        'gateway/openai:gpt-4-32k',
        'gateway/openai:gpt-4-32k-0314',
        'gateway/openai:gpt-4-32k-0613',
        'gateway/openai:gpt-4-turbo-preview',
        'gateway/openai:gpt-4-vision-preview',
        'gateway/openai:gpt-4o-audio-preview',
        'gateway/openai:gpt-4o-audio-preview-2024-10-01',
        'gateway/openai:gpt-4o-audio-preview-2024-12-17',
        'gateway/openai:gpt-4o-audio-preview-2025-06-03',
        'gateway/openai:gpt-4o-mini-audio-preview',
        'gateway/openai:gpt-4o-mini-audio-preview-2024-12-17',
        'gateway/openai:gpt-5-codex',
        'gateway/openai:gpt-5-pro',
        'gateway/openai:gpt-5-pro-2025-10-06',
        'gateway/openai:gpt-5.1-codex',
        'gateway/openai:gpt-5.1-codex-max',
        'gateway/openai:gpt-5.1-mini',
        'gateway/openai:gpt-5.2-pro',
        'gateway/openai:gpt-5.2-pro-2025-12-11',
        'gateway/openai:gpt-5.3-chat-latest',
        'gateway/openai:o1-mini',
        'gateway/openai:o1-mini-2024-09-12',
        'gateway/openai:o1-preview',
        'gateway/openai:o1-preview-2024-09-12',
        'gateway/openai:o1-pro',
        'gateway/openai:o1-pro-2025-03-19',
        'gateway/openai:o3-deep-research',
        'gateway/openai:o3-deep-research-2025-06-26',
        'gateway/openai:o3-pro',
        'gateway/openai:o3-pro-2025-06-10',
        'gateway/openai:o4-mini-deep-research',
        'gateway/openai:o4-mini-deep-research-2025-06-26',
    }
)


def test_known_model_names():  # pragma: lax no cover
    # Coverage seems to be misbehaving..?
    def get_model_names(model_name_type: Any) -> Iterator[str]:
        for arg in get_args(model_name_type):
            if isinstance(arg, str):
                yield arg
            else:
                yield from get_model_names(arg)

    def is_deprecated(provider: str, model_name: str) -> bool:
        return model_name in _PROVIDER_DEPRECATED_MODELS.get(provider, frozenset())

    all_generated_names = [
        f'{provider}:{n}'
        for provider, model_names in _PROVIDER_TO_MODEL_NAMES.items()
        for n in get_model_names(model_names)
        if not is_deprecated(provider, n)
    ]

    cerebras_names = get_cerebras_model_names()
    heroku_names = get_heroku_model_names()
    gateway_names = [
        f'gateway/{provider}:{model_name}'
        for provider in GatewayModelProvider.__args__
        for model_name in get_model_names(_PROVIDER_TO_MODEL_NAMES[provider])
        if f'gateway/{provider}:{model_name}' not in UNSUPPORTED_GATEWAY_MODEL_NAMES
        if not is_deprecated(provider, model_name)
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
