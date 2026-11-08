import os
import subprocess
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, get_args

import httpx
import pytest
from inline_snapshot import snapshot
from typing_extensions import TypedDict

from pydantic_ai.models import KnownModelName, infer_model_profile, known_model_names
from pydantic_ai.providers.gateway import ModelProvider as GatewayModelProvider

from ..conftest import try_import

with try_import() as imports_successful:
    from anthropic.types import ModelParam
    from openai.types import AllModels, EmbeddingModel, ImageModel
    from xai_sdk.types import ImageGenerationModel
    from xai_sdk.types.model import ChatModel

    from pydantic_ai.embeddings.openai import LatestOpenAIEmbeddingModelNames
    from pydantic_ai.images.openai import LatestOpenAIImageModelNames
    from pydantic_ai.images.xai import LatestXaiImageGenerationModelNames
    from pydantic_ai.models.anthropic import DEPRECATED_ANTHROPIC_MODELS, AnthropicModelName
    from pydantic_ai.models.bedrock import BedrockModelName
    from pydantic_ai.models.bedrock_mantle import BedrockMantleModelName
    from pydantic_ai.models.cerebras import CerebrasModelName
    from pydantic_ai.models.cohere import CohereModelName
    from pydantic_ai.models.crusoe import CrusoeModelName
    from pydantic_ai.models.google import GoogleModelName
    from pydantic_ai.models.groq import GroqModelName
    from pydantic_ai.models.huggingface import HuggingFaceModelName
    from pydantic_ai.models.mistral import MistralModelName
    from pydantic_ai.models.openai import DEPRECATED_OPENAI_MODELS, OpenAIModelName
    from pydantic_ai.models.openai_decisions import OpenAIDecisionsModelName
    from pydantic_ai.models.snowflake import SnowflakeModelName
    from pydantic_ai.models.typesafe import TypeSafeModelName
    from pydantic_ai.models.xai import XaiModelName
    from pydantic_ai.models.zai import ZaiModelName
    from pydantic_ai.providers.deepseek import DeepSeekModelName
    from pydantic_ai.providers.meta import MetaModelName
    from pydantic_ai.providers.moonshotai import MoonshotAIModelName

if not imports_successful():  # pragma: lax no cover
    # Define placeholders so the module can be loaded for test collection
    ModelParam = AllModels = EmbeddingModel = ImageModel = ImageGenerationModel = ChatModel = None
    LatestOpenAIEmbeddingModelNames = LatestOpenAIImageModelNames = LatestXaiImageGenerationModelNames = None
    AnthropicModelName = BedrockModelName = BedrockMantleModelName = CerebrasModelName = CohereModelName = None
    GoogleModelName = GroqModelName = HuggingFaceModelName = MistralModelName = OpenAIModelName = None
    OpenAIDecisionsModelName = None
    DEPRECATED_ANTHROPIC_MODELS: frozenset[str] = frozenset()  # pyright: ignore[reportConstantRedefinition]
    DEPRECATED_OPENAI_MODELS: frozenset[str] = frozenset()  # pyright: ignore[reportConstantRedefinition]
    CrusoeModelName = None
    DeepSeekModelName = XaiModelName = MoonshotAIModelName = ZaiModelName = SnowflakeModelName = None
    MetaModelName = None
    TypeSafeModelName = None

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='some model package was not installed'),
    pytest.mark.vcr,
]


@pytest.fixture(scope='module')
def vcr_config(vcr_config: dict[str, Any]) -> dict[str, Any]:  # pragma: lax no cover
    """Re-record the provider model lists whenever a developer with the keys runs the module."""
    if os.getenv('CI') or not os.getenv('CEREBRAS_API_KEY'):
        return vcr_config
    return {**vcr_config, 'record_mode': 'rewrite'}


_PROVIDER_TO_MODEL_NAMES = {
    'anthropic': AnthropicModelName,
    'bedrock': BedrockModelName,
    'bedrock-mantle': BedrockMantleModelName,
    'cerebras': CerebrasModelName,
    'cohere': CohereModelName,
    'crusoe': CrusoeModelName,
    'deepseek': DeepSeekModelName,
    'google': GoogleModelName,
    'google-cloud': GoogleModelName,
    'xai': XaiModelName,
    'groq': GroqModelName,
    'huggingface': HuggingFaceModelName,
    'meta': MetaModelName,
    'mistral': MistralModelName,
    'moonshotai': MoonshotAIModelName,
    'openai': OpenAIModelName,
    'openai-chat': OpenAIModelName,
    'openai-decisions': OpenAIDecisionsModelName,
    'snowflake': SnowflakeModelName,
    'typesafe': TypeSafeModelName,
    'zai': ZaiModelName,
}

# Heroku has no model-name alias; its ids in `KnownModelName` are curated from its live model list.
_HEROKU_MODEL_NAMES = frozenset(
    {
        'claude-fable-5-1',
        'claude-opus-5-5',
        'claude-sonnet-5-5',
        'claude-haiku-5-5',
        'claude-opus-4-8',
        'claude-4-6-sonnet',
        'claude-4-5-haiku',
        'gpt-oss-120b',
        'nova-2-lite',
        'nova-pro',
    }
)

UNSUPPORTED_GATEWAY_MODEL_NAMES = frozenset(
    {
        'gateway/bedrock:us.amazon.nova-lite-v1:0',
        'gateway/bedrock:us.amazon.nova-micro-v1:0',
        'gateway/bedrock:us.amazon.nova-pro-v1:0',
        'gateway/bedrock:us.meta.llama3-3-70b-instruct-v1:0',
        # Gateway rejects it with "The provided model identifier is invalid."; native Bedrock serves it.
        'gateway/bedrock:us.deepseek.r1-v1:0',
        # Gateway has no price data for these yet, so it refuses them with "No cost data available".
        'gateway/bedrock:global.moonshotai.kimi-k3',
        'gateway/bedrock:global.moonshotai.kimi-k3.1',
        'gateway/bedrock:global.zai.glm-5.3',
    }
)


def get_model_names(model_name_type: Any) -> Iterator[str]:
    for arg in get_args(model_name_type):
        if isinstance(arg, str):
            yield arg
        else:
            yield from get_model_names(arg)


def generated_model_names() -> set[str]:
    """The `KnownModelName` members the curated per-provider aliases imply, gateway mirror included."""
    provider_names = {
        f'{provider}:{model_name}'
        for provider, model_names in _PROVIDER_TO_MODEL_NAMES.items()
        for model_name in get_model_names(model_names)
    }
    gateway_names = {
        f'gateway/{provider}:{model_name}'
        for provider in get_args(GatewayModelProvider)
        for model_name in get_model_names(_PROVIDER_TO_MODEL_NAMES[provider])
    } - UNSUPPORTED_GATEWAY_MODEL_NAMES
    heroku_names = {f'heroku:{model_name}' for model_name in _HEROKU_MODEL_NAMES}
    return provider_names | gateway_names | heroku_names | {'test'}


def test_known_model_names():
    """`KnownModelName` is exactly the curated per-provider aliases plus their gateway mirror, so the two can't drift."""
    generated_names = generated_model_names()
    known_names = set(known_model_names())
    assert known_names - generated_names == set(), 'in `KnownModelName` but no curated alias'
    assert generated_names - known_names == set(), 'in a curated alias but not in `KnownModelName`'


def test_unsupported_gateway_model_names_are_curated():
    """Every gateway exclusion still names a curated id, so the exclusion list can't go stale."""
    curated_gateway_names = {
        f'gateway/{provider}:{model_name}'
        for provider in get_args(GatewayModelProvider)
        for model_name in get_model_names(_PROVIDER_TO_MODEL_NAMES[provider])
    }
    assert UNSUPPORTED_GATEWAY_MODEL_NAMES - curated_gateway_names == set()


def test_curated_model_names_exclude_retired_sdk_ids():
    assert set(get_model_names(OpenAIModelName)) & DEPRECATED_OPENAI_MODELS == set()
    assert set(get_model_names(AnthropicModelName)) & DEPRECATED_ANTHROPIC_MODELS == set()


@pytest.mark.parametrize(
    'model_name',
    [
        'anthropic:claude-opus-4-5',
        'bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0',
        'google:gemini-2.5-pro',
        'moonshotai:kimi-k2.5',
        'openai:o3',
        'xai:grok-4.5',
    ],
)
def test_model_names_outside_the_literals_keep_their_profile(model_name: str):
    """Curation only changes autocomplete: an id the literals no longer list still resolves its family's profile."""
    assert model_name not in known_model_names()
    assert infer_model_profile(model_name).get('supports_thinking') is True


@dataclass(frozen=True)
class SdkModelLiteral:
    """An SDK's model-name `Literal` and the curated Pydantic AI alias that picks from it."""

    id: str
    package: str
    sdk_literal: object
    curated_alias: object
    bridges: frozenset[str] = frozenset()
    """Curated ids the locked SDK doesn't list yet. Drop one once the SDK in `uv.lock` lists it."""


SDK_MODEL_LITERALS = [
    SdkModelLiteral('openai', 'openai', AllModels, OpenAIModelName, frozenset({'gpt-6.1-terra'})),
    SdkModelLiteral('openai-image', 'openai', ImageModel, LatestOpenAIImageModelNames),
    SdkModelLiteral('openai-embedding', 'openai', EmbeddingModel, LatestOpenAIEmbeddingModelNames),
    SdkModelLiteral('anthropic', 'anthropic', ModelParam, AnthropicModelName),
    SdkModelLiteral('xai', 'xai-sdk', ChatModel, XaiModelName, frozenset({'grok-4.7'})),
    SdkModelLiteral(
        'xai-image',
        'xai-sdk',
        ImageGenerationModel,
        LatestXaiImageGenerationModelNames,
        frozenset({'grok-imagine-image-2.0'}),
    ),
]

SDK_MODEL_IDS: dict[str, list[str]] = {
    'openai': snapshot(
        [
            'chatgpt-4o-latest',
            'codex-mini-latest',
            'computer-use-preview',
            'computer-use-preview-2025-03-11',
            'gpt-3.5-turbo',
            'gpt-3.5-turbo-0125',
            'gpt-3.5-turbo-0301',
            'gpt-3.5-turbo-0613',
            'gpt-3.5-turbo-1106',
            'gpt-3.5-turbo-16k',
            'gpt-3.5-turbo-16k-0613',
            'gpt-4',
            'gpt-4-0125-preview',
            'gpt-4-0314',
            'gpt-4-0613',
            'gpt-4-1106-preview',
            'gpt-4-32k',
            'gpt-4-32k-0314',
            'gpt-4-32k-0613',
            'gpt-4-turbo',
            'gpt-4-turbo-2024-04-09',
            'gpt-4-turbo-preview',
            'gpt-4-vision-preview',
            'gpt-4.1',
            'gpt-4.1-2025-04-14',
            'gpt-4.1-mini',
            'gpt-4.1-mini-2025-04-14',
            'gpt-4.1-nano',
            'gpt-4.1-nano-2025-04-14',
            'gpt-4o',
            'gpt-4o-2024-05-13',
            'gpt-4o-2024-08-06',
            'gpt-4o-2024-11-20',
            'gpt-4o-audio-preview',
            'gpt-4o-audio-preview-2024-10-01',
            'gpt-4o-audio-preview-2024-12-17',
            'gpt-4o-audio-preview-2025-06-03',
            'gpt-4o-mini',
            'gpt-4o-mini-2024-07-18',
            'gpt-4o-mini-audio-preview',
            'gpt-4o-mini-audio-preview-2024-12-17',
            'gpt-4o-mini-search-preview',
            'gpt-4o-mini-search-preview-2025-03-11',
            'gpt-4o-search-preview',
            'gpt-4o-search-preview-2025-03-11',
            'gpt-5',
            'gpt-5-2025-08-07',
            'gpt-5-chat-latest',
            'gpt-5-codex',
            'gpt-5-mini',
            'gpt-5-mini-2025-08-07',
            'gpt-5-nano',
            'gpt-5-nano-2025-08-07',
            'gpt-5-pro',
            'gpt-5-pro-2025-10-06',
            'gpt-5.1',
            'gpt-5.1-2025-11-13',
            'gpt-5.1-chat-latest',
            'gpt-5.1-codex',
            'gpt-5.1-codex-max',
            'gpt-5.1-mini',
            'gpt-5.2',
            'gpt-5.2-2025-12-11',
            'gpt-5.2-chat-latest',
            'gpt-5.2-pro',
            'gpt-5.2-pro-2025-12-11',
            'gpt-5.3-chat-latest',
            'gpt-5.4',
            'gpt-5.4-mini',
            'gpt-5.4-mini-2026-03-17',
            'gpt-5.4-nano',
            'gpt-5.4-nano-2026-03-17',
            'gpt-5.5',
            'gpt-5.5-2026-04-23',
            'gpt-5.5-pro',
            'gpt-5.5-pro-2026-04-23',
            'gpt-5.6-cyber',
            'gpt-5.6-luna',
            'gpt-5.6-sol',
            'gpt-5.6-terra',
            'gpt-6-astra',
            'gpt-6-luna',
            'gpt-6-sol',
            'gpt-6.1-sol',
            'gpt-audio-mini',
            'gpt-audio-mini-2025-12-15',
            'gpt-daybreak-blue-latest',
            'gpt-daybreak-red-latest',
            'gpt-rosalind-research',
            'o1',
            'o1-2024-12-17',
            'o1-mini',
            'o1-mini-2024-09-12',
            'o1-preview',
            'o1-preview-2024-09-12',
            'o1-pro',
            'o1-pro-2025-03-19',
            'o3',
            'o3-2025-04-16',
            'o3-deep-research',
            'o3-deep-research-2025-06-26',
            'o3-mini',
            'o3-mini-2025-01-31',
            'o3-pro',
            'o3-pro-2025-06-10',
            'o4-mini',
            'o4-mini-2025-04-16',
            'o4-mini-deep-research',
            'o4-mini-deep-research-2025-06-26',
        ]
    ),
    'openai-image': snapshot(
        [
            'chatgpt-image-latest',
            'dall-e-2',
            'dall-e-3',
            'gpt-image-1',
            'gpt-image-1-mini',
            'gpt-image-1.5',
            'gpt-image-2',
            'gpt-image-2-2026-04-21',
            'gpt-image-2.5-flare',
            'gpt-image-2.5-flare-2026-09-08',
            'gpt-image-2.5-sunburst',
            'gpt-image-2.5-sunburst-2026-09-08',
        ]
    ),
    'openai-embedding': snapshot(['text-embedding-3-large', 'text-embedding-3-small', 'text-embedding-ada-002']),
    'anthropic': snapshot(
        [
            'claude-fable-5',
            'claude-fable-5-1',
            'claude-haiku-4-5',
            'claude-haiku-4-5-20251001',
            'claude-haiku-5-5',
            'claude-mythos-5',
            'claude-mythos-5-1',
            'claude-mythos-preview',
            'claude-opus-4-5',
            'claude-opus-4-5-20251101',
            'claude-opus-4-6',
            'claude-opus-4-7',
            'claude-opus-4-8',
            'claude-opus-5',
            'claude-opus-5-5',
            'claude-sonnet-4-5',
            'claude-sonnet-4-5-20250929',
            'claude-sonnet-4-6',
            'claude-sonnet-5',
            'claude-sonnet-5-5',
        ]
    ),
    'xai': snapshot(
        [
            'grok-3',
            'grok-3-fast',
            'grok-3-fast-latest',
            'grok-3-latest',
            'grok-3-mini',
            'grok-3-mini-fast',
            'grok-3-mini-fast-latest',
            'grok-4',
            'grok-4-0709',
            'grok-4-1-fast',
            'grok-4-1-fast-non-reasoning',
            'grok-4-1-fast-non-reasoning-latest',
            'grok-4-1-fast-reasoning',
            'grok-4-1-fast-reasoning-latest',
            'grok-4-fast',
            'grok-4-fast-non-reasoning',
            'grok-4-fast-non-reasoning-latest',
            'grok-4-fast-reasoning',
            'grok-4-fast-reasoning-latest',
            'grok-4-latest',
            'grok-4.20',
            'grok-4.20-0309',
            'grok-4.20-0309-non-reasoning',
            'grok-4.20-0309-reasoning',
            'grok-4.20-multi-agent',
            'grok-4.20-multi-agent-0309',
            'grok-4.20-multi-agent-latest',
            'grok-4.20-non-reasoning',
            'grok-4.20-non-reasoning-latest',
            'grok-4.20-reasoning-latest',
            'grok-4.3',
            'grok-4.3-latest',
            'grok-4.5',
            'grok-4.5-latest',
            'grok-4.6',
            'grok-build-0.1',
            'grok-code-fast-1',
        ]
    ),
    'xai-image': snapshot(['grok-imagine-image', 'grok-imagine-image-pro', 'grok-imagine-image-quality']),
}


@cache
def _committed_lock() -> dict[str, Any]:
    # Read from git, not the working tree: the lowest-versions job and the upgrade canary rewrite `uv.lock` before
    # they install, so the working-tree file would always match what they installed.
    try:
        lock = subprocess.run(
            ['git', 'show', 'HEAD:uv.lock'], cwd=Path(__file__).parent, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        pytest.skip('reading the committed `uv.lock` needs `git` and a git checkout')
    return tomllib.loads(lock)


def _skip_unless_locked(package: str) -> None:
    """Skip where the installed SDK isn't the committed `uv.lock` version, as on lowest-versions and the canary."""
    installed = version(package)
    locked = {entry['version'] for entry in _committed_lock()['package'] if entry['name'] == package}
    if installed not in locked:  # pragma: lax no cover
        pytest.skip(f'`{package}` {installed} is not the version in the committed `uv.lock`')


@pytest.mark.parametrize('sdk', [pytest.param(sdk, id=sdk.id) for sdk in SDK_MODEL_LITERALS])
def test_sdk_model_literal(sdk: SdkModelLiteral):
    """Fails when an SDK upgrade in `uv.lock` changes the model ids the SDK lists.

    Review the new ids against the curation rule in `.agents/skills/add-new-model/SKILL.md`, update the
    curated alias, then refresh this snapshot with `--inline-snapshot=fix`.
    """
    _skip_unless_locked(sdk.package)
    assert sorted(get_model_names(sdk.sdk_literal)) == SDK_MODEL_IDS[sdk.id]


@pytest.mark.parametrize('sdk', [pytest.param(sdk, id=sdk.id) for sdk in SDK_MODEL_LITERALS])
def test_curated_model_names_are_sdk_ids_or_bridges(sdk: SdkModelLiteral):
    """A curated id the SDK doesn't list is a typo unless it's an explicit bridge; a bridge the SDK lists is stale."""
    _skip_unless_locked(sdk.package)
    assert set(get_model_names(sdk.curated_alias)) - set(get_model_names(sdk.sdk_literal)) == sdk.bridges


class HerokuModel(TypedDict):
    model_id: str
    regions: list[str]
    type: list[str]


def get_heroku_model_names() -> list[str]:
    response = httpx.get('https://us.inference.heroku.com/available-models')

    if response.status_code != 200:
        pytest.skip(f'Heroku AI returned status code {response.status_code}')  # pragma: lax no cover

    heroku_models: list[HerokuModel] = response.json()
    return sorted(model['model_id'] for model in heroku_models if 'text-to-text' in model['type'])


def test_heroku_model_list():
    """Re-recording fails when Heroku's model list changes, so someone reviews the curated Heroku ids."""
    live_names = get_heroku_model_names()
    assert live_names == snapshot(
        [
            'claude-4-5-haiku',
            'claude-4-5-sonnet',
            'claude-4-6-sonnet',
            'claude-fable-5-1',
            'claude-haiku-5-5',
            'claude-opus-4-5',
            'claude-opus-4-6',
            'claude-opus-4-7',
            'claude-opus-4-8',
            'claude-opus-5-5',
            'claude-sonnet-5-5',
            'gpt-oss-120b',
            'nova-2-lite',
            'nova-lite',
            'nova-pro',
        ]
    )
    assert _HEROKU_MODEL_NAMES <= set(live_names)


class CerebrasModel(TypedDict):
    created: int
    id: str
    object: Literal['model']
    owned_by: Literal['Cerebras']


def get_cerebras_model_names() -> list[str]:  # pragma: lax no cover
    api_key = os.getenv('CEREBRAS_API_KEY', 'testing')

    response = httpx.get(
        'https://api.cerebras.ai/v1/models',
        headers={'Authorization': f'Bearer {api_key}', 'Accept': 'application/json', 'Accept-Encoding': 'identity'},
    )

    if response.status_code != 200:
        pytest.skip(f'Cerebras returned status code {response.status_code}')  # pragma: lax no cover

    cerebras_models: list[CerebrasModel] = response.json()['data']
    return sorted(model['id'] for model in cerebras_models)


def test_cerebras_model_list():
    """Re-recording fails when Cerebras's model list changes, so someone reviews `CerebrasModelName`."""
    live_names = get_cerebras_model_names()
    assert live_names == snapshot(['gpt-oss-120b', 'qwen-3.8-27b'])
    assert set(get_model_names(CerebrasModelName)) <= set(live_names)


def test_known_model_names_accessor():
    """`known_model_names()` exposes exactly the `KnownModelName` members, verbatim."""
    assert known_model_names() == get_args(KnownModelName.__value__)


def test_known_model_names_docstring_warns_not_a_validator():
    """The docstring states the list's purpose and warns against using it to validate user input."""
    # Whitespace-normalized so rewrapping the docstring does not break the assertions.
    doc = ' '.join((known_model_names.__doc__ or '').split())

    # The intended purpose is named: autocomplete and error suggestions.
    assert 'autocomplete' in doc
    assert 'error suggestions' in doc
    # The list is explicitly not exhaustive of the model ids providers accept.
    assert 'not exhaustive' in doc
    # Using it to validate user input is explicitly warned against.
    assert 'validate user input' in doc


def test_known_model_names_enumeration_contract_intact():
    """`known_model_names()` still returns a non-empty tuple of provider-prefixed ids."""
    names = known_model_names()
    assert isinstance(names, tuple)
    assert all(isinstance(name, str) for name in names)
    assert names
    assert any(name.startswith('openai:') for name in names)
    assert any(name.startswith('anthropic:') for name in names)
