"""A model's `base_url` setting is the endpoint its provider is built with."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr
from rich.console import Console

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai_codex import OpenAICodexModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers import Provider, infer_provider_class
from pydantic_ai.providers.github_copilot import GitHubCopilotCredentials
from pydantic_clai2.config.credential_store import save_codex_credentials
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.models import github_copilot, key_profiles, openrouter, vllm
from pydantic_clai2.models.model_settings import ModelSettingsForm
from pydantic_clai2.models.resolver import ModelResolver
from pydantic_clai2.plugins import ModelProvider
from pydantic_clai2.ui.menus.field_menu import FieldMenu
from pydantic_clai2.ui.menus.model_menu import ModelSettingsSource

URL = 'https://gateway.example/v1'
ENV_URL = 'https://env.example/v1'
KEYS = (
    'ALIBABA_API_KEY',
    'ANTHROPIC_API_KEY',
    'AWS_BEARER_TOKEN_BEDROCK',
    'AZURE_OPENAI_API_KEY',
    'CEREBRAS_API_KEY',
    'CO_API_KEY',
    'CRUSOE_API_KEY',
    'DEEPSEEK_API_KEY',
    'ELEVENLABS_API_KEY',
    'FIREWORKS_API_KEY',
    'GITHUB_API_KEY',
    'GITHUB_COPILOT_API_KEY',
    'GOOGLE_API_KEY',
    'GROQ_API_KEY',
    'HEROKU_INFERENCE_KEY',
    'HF_TOKEN',
    'META_API_KEY',
    'MISTRAL_API_KEY',
    'MOONSHOTAI_API_KEY',
    'NEBIUS_API_KEY',
    'OPENAI_API_KEY',
    'OPENROUTER_API_KEY',
    'OVHCLOUD_API_KEY',
    'PYDANTIC_AI_GATEWAY_API_KEY',
    'SAMBANOVA_API_KEY',
    'SNOWFLAKE_TOKEN',
    'SYSTEM_ONE_API_KEY',
    'TOGETHER_API_KEY',
    'TYPESAFE_API_KEY',
    'VERCEL_AI_GATEWAY_API_KEY',
    'VOYAGE_API_KEY',
    'XAI_API_KEY',
    'ZAI_API_KEY',
)
ENV_URLS = (
    'ANTHROPIC_BASE_URL',
    'AZURE_OPENAI_ENDPOINT',
    'CO_BASE_URL',
    'GITHUB_COPILOT_BASE_URL',
    'GROQ_BASE_URL',
    'HEROKU_INFERENCE_URL',
    'MOONSHOTAI_BASE_URL',
    'OLLAMA_BASE_URL',
    'OPENAI_BASE_URL',
    'PYDANTIC_AI_GATEWAY_BASE_URL',
    'SAMBANOVA_BASE_URL',
    'SYSTEM_ONE_BASE_URL',
    'TYPESAFE_BASE_URL',
    'VLLM_BASE_URL',
)
UNKEYED = ('snowflake',)
PROVIDERS = (
    'alibaba',
    'anthropic',
    'azure',
    'azure-responses',
    'bedrock',
    'bedrock-mantle',
    'cerebras',
    'cohere',
    'crusoe',
    'deepseek',
    'elevenlabs',
    'fireworks',
    'gateway/anthropic',
    'gateway/openai',
    'github',
    'github-copilot',
    'google',
    'google-cloud',
    'groq',
    'heroku',
    'huggingface',
    'litellm',
    'meta',
    'mistral',
    'moonshotai',
    'nebius',
    'ollama',
    'openai',
    'openai-chat',
    'openai-decisions',
    'openai-responses',
    'openrouter',
    'ovhcloud',
    'sambanova',
    'system-one',
    'together',
    'typesafe',
    'vercel',
    'vllm',
    'voyageai',
    'zai',
    *UNKEYED,
)


@pytest.fixture
def environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every provider's key, and an endpoint a `base_url` has to win over."""
    settings = {'AWS_DEFAULT_REGION': 'us-east-1', 'OPENAI_API_VERSION': '2025-04-01-preview'}
    for name, value in {**dict.fromkeys(KEYS, 'sk-env'), **dict.fromkeys(ENV_URLS, ENV_URL), **settings}.items():
        monkeypatch.setenv(name, value)


def host(model: Model | str) -> str:
    assert isinstance(model, Model)
    return httpx.URL(model.base_url or '').host


def resolver(store: SettingsStore, providers: dict[str, ModelProvider] | None = None) -> ModelResolver:
    return ModelResolver(console=Console(file=io.StringIO()), plugins=lambda: providers or {}, store=store)


@pytest.fixture(scope='module')
def missing_extras() -> dict[str, str]:
    """Import every provider's SDK once, outside the per-test budget, noting those this cell lacks."""
    missing: dict[str, str] = {}
    for provider in PROVIDERS:
        try:
            infer_provider_class(provider.removeprefix('gateway/'))
        except ImportError as exc:  # pragma: lax no cover
            missing[provider] = str(exc)
    return missing


@pytest.mark.usefixtures('environment')
@pytest.mark.filterwarnings('ignore:`GitHubProvider` is deprecated:pydantic_ai._warnings.PydanticAIDeprecationWarning')
@pytest.mark.parametrize('provider', PROVIDERS)
def test_every_provider_is_built_with_the_base_url(provider: str, missing_extras: dict[str, str]) -> None:
    if provider in missing_extras:  # pragma: lax no cover
        pytest.skip(missing_extras[provider])
    for key in (None,) if provider in UNKEYED else (None, 'sk-profile'):
        built = key_profiles.build_provider(provider, key=key, base_url=URL)
        assert httpx.URL(built.base_url).host == 'gateway.example'


def test_xai_takes_the_base_url_as_its_api_host() -> None:
    pytest.importorskip('xai_sdk', reason='needs the `xai` extra')
    from pydantic_ai.providers.xai import XaiProvider

    for base_url in ('gateway.example:8443', 'https://gateway.example:8443/v1'):
        built = key_profiles.build_provider('xai', key='sk-profile', base_url=base_url)
        assert isinstance(built, XaiProvider) and built.api_host == 'gateway.example:8443'


def test_a_provider_without_an_endpoint_fails_clearly() -> None:
    with pytest.raises(
        UserError, match=r'sentence-transformers has no endpoint to override\. Reset base_url in /model settings\.'
    ):
        key_profiles.build_provider('sentence-transformers', base_url=URL)


def test_a_known_provider_without_its_extra_still_offers_a_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_extra(provider: str) -> type[Provider[object]]:
        raise ImportError(f'install the `{provider}` extra')

    monkeypatch.setattr(key_profiles, 'infer_provider_class', missing_extra)
    assert key_profiles.takes_endpoint('mistral')


def test_openai_chatgpt_takes_the_base_url_and_still_needs_its_sign_in() -> None:
    with pytest.raises(UserError, match='`OpenAIChatGPTProvider` needs `credentials` or a `credential_source`'):
        key_profiles.build_provider('openai-chatgpt', base_url=URL)


async def test_models_of_one_provider_use_their_own_base_url(tmp_path: Path, environment: None) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    store.save_model_settings('openai:gpt-5', {'base_url': 'https://one.example/v1', 'max_tokens': 5})
    store.save_model_settings('openai:gpt-5-mini', {'base_url': 'https://two.example/v1'})
    store.save_chain(name='both', models=['openai:gpt-5', 'openai:gpt-5-mini', 'openai:gpt-4.1'])
    models = resolver(store)
    assert host(await models.resolve('openai:gpt-5')) == 'one.example'
    assert host(await models.resolve('openai:gpt-5-mini')) == 'two.example'
    assert await models.resolve('openai:gpt-4.1') == 'openai:gpt-4.1'  # core's own, on `OPENAI_BASE_URL`
    chain = await models.resolve('chain:both')
    assert isinstance(chain, FallbackModel)
    assert [host(model) for model in chain.models] == ['one.example', 'two.example', 'env.example']


async def test_every_account_of_a_model_uses_its_base_url(tmp_path: Path, environment: None) -> None:
    copilot = github_copilot.Connection(
        credentials=GitHubCopilotCredentials(access_token='work-token', token_type='bearer', scope=''),
        issued_at=time.time(),
    )
    save_codex_credentials(account='github-copilot@work', value=copilot.model_dump_json())
    codex = json.dumps({'access_token': 'a', 'refresh_token': 'r', 'account_id': 'x'})
    save_codex_credentials(account='openai-codex@work', value=codex)
    save_codex_credentials(account='openai@work', value=json.dumps({'token': 'sk-work'}))
    openrouter.save_connection(openrouter.Connection(token=SecretStr('or-key')))
    vllm.save_connection(vllm.Connection(url='http://localhost:8000', token=SecretStr('lab-key')))
    store = SettingsStore(tmp_path / 'config.db')
    models = resolver(store, {'single': ModelProvider(prefix='single', resolve=lambda name: TestModel())})
    for name in (
        'github-copilot@work:gpt-5',
        'openai-codex@work:gpt-5',
        'openai@work:gpt-5',
        'openai@default:gpt-5',
        'openai@*:gpt-5',
        'openrouter:openai/gpt-5',
        'vllm:my/model',
    ):
        store.save_model_settings(name, {'base_url': URL})
        assert host(await models.resolve(name)) == 'gateway.example'
    codex = await models.resolve('openai-codex@work:gpt-5')
    assert isinstance(codex, OpenAICodexModel)
    assert models.codex_providers == {f'openai-codex@work {URL}': codex.provider}
    models.codex_auth().forget('openai-codex@work')
    assert models.codex_providers == {}
    store.save_chain(name='both', models=['openai:gpt-5', 'openai:gpt-4.1'])
    for name, provider in (('single:x', 'single'), ('chain:both', 'chain:both')):
        store.save_model_settings(name, {'base_url': URL})
        with pytest.raises(UserError, match=f'{provider} has no endpoint to override'):
            await models.resolve(name)


async def test_a_pooled_account_falls_back_to_its_own_base_url(tmp_path: Path, environment: None) -> None:
    for account in ('openai@home', 'openai@work'):
        save_codex_credentials(account=account, value=json.dumps({'token': f'sk-{account}'}))
    store = SettingsStore(tmp_path / 'config.db')
    store.save_model_settings('openai@work:gpt-5', {'base_url': URL})
    pooled = await resolver(store).resolve('openai@*:gpt-5')
    assert isinstance(pooled, FallbackModel)
    assert [host(model) for model in pooled.models] == ['env.example', 'gateway.example']
    # A session resolves each pooled member on its own, so a member finds its plain model's endpoint.
    store.save_model_settings('openai:gpt-5', {'base_url': 'https://shared.example/v1'})
    models = resolver(store)
    assert [host(await models.resolve(name)) for name in await models.expand('openai@*:gpt-5')] == [
        'shared.example',
        'gateway.example',
    ]


def test_base_url_is_a_model_setting_that_no_request_carries(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    source = ModelSettingsSource(store, 'openai:gpt-5')
    row = FieldMenu(source).row_for('base_url')
    assert row is not None and (row.label, row.choices) == ('Base URL', ())
    assert source.apply(row, URL) == 'Saved base_url for openai:gpt-5. Applies when this model is selected.'
    assert store.model_settings('openai:gpt-5') == {'base_url': URL}
    assert ModelSettingsForm(base_url=URL).to_model_settings() is None
    assert source.problem(row, 'gateway.example/v1') == (
        'Value error, must be an http:// or https:// URL, such as https://gateway.example/v1'
    )
    store.save_chain(name='both', models=['openai:gpt-5', 'openai:gpt-4.1'])
    assert FieldMenu(ModelSettingsSource(store, 'chain:both', settings_as='openai:gpt-5')).row_for('base_url') is None
    assert FieldMenu(ModelSettingsSource(store, 'single:x')).row_for('base_url') is None
    assert FieldMenu(ModelSettingsSource(store, 'gateway/openai:gpt-5')).row_for('base_url') is not None
