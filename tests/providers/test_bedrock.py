from typing import Annotated, Literal, cast, get_args

import pytest
from pydantic import BaseModel, Field
from pytest_mock import MockerFixture

from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.native_tools import SUPPORTED_NATIVE_TOOLS, CodeExecutionTool
from pydantic_ai.profiles.amazon import amazon_model_profile
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.profiles.zai import zai_model_profile
from pydantic_ai.providers._bedrock_model_names import split_bedrock_model_id

from .._inline_snapshot import snapshot
from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient

    from pydantic_ai.models.bedrock import LatestBedrockModelNames
    from pydantic_ai.providers.bedrock import (
        BEDROCK_GEO_PREFIXES,
        BedrockJsonSchemaTransformer,
        BedrockProvider,
        remove_bedrock_geo_prefix,
    )

if not imports_successful():
    BEDROCK_GEO_PREFIXES: tuple[str, ...] = ()  # pragma: lax no cover  # type: ignore[no-redef]

pytestmark = pytest.mark.skipif(not imports_successful(), reason='bedrock not installed')


@pytest.fixture(autouse=True)
def bedrock_credentials(env: TestEnv):
    env.set('AWS_ACCESS_KEY_ID', 'test-access-key')
    env.set('AWS_SECRET_ACCESS_KEY', 'test-secret-key')


def test_bedrock_provider(env: TestEnv):
    env.set('AWS_DEFAULT_REGION', 'us-east-1')
    provider = BedrockProvider()
    assert isinstance(provider, BedrockProvider)
    assert provider.name == 'bedrock'
    assert provider.base_url == 'https://bedrock-runtime.us-east-1.amazonaws.com'


def test_bedrock_provider_client_setter(env: TestEnv):
    env.set('AWS_DEFAULT_REGION', 'us-east-1')
    provider = BedrockProvider()
    original_client = provider.client

    env.set('AWS_DEFAULT_REGION', 'us-west-2')
    new_client = BedrockProvider().client
    provider.client = new_client

    assert provider.client is new_client
    assert provider.client is not original_client
    assert provider.base_url == 'https://bedrock-runtime.us-west-2.amazonaws.com'


def test_bedrock_provider_bearer_token_env_var(env: TestEnv, mocker: MockerFixture):
    """Test that AWS_BEARER_TOKEN_BEDROCK env var is used for bearer token auth."""
    env.set('AWS_DEFAULT_REGION', 'us-east-1')
    env.set('AWS_BEARER_TOKEN_BEDROCK', 'test-bearer-token')

    mock_session = mocker.patch('pydantic_ai.providers.bedrock._BearerTokenSession')

    provider = BedrockProvider()

    mock_session.assert_called_once_with('test-bearer-token')
    assert provider.name == 'bedrock'


def test_bedrock_provider_timeout(env: TestEnv):
    env.set('AWS_DEFAULT_REGION', 'us-east-1')
    env.set('AWS_READ_TIMEOUT', '1')
    env.set('AWS_CONNECT_TIMEOUT', '1')
    provider = BedrockProvider()
    assert isinstance(provider, BedrockProvider)
    assert provider.name == 'bedrock'

    config = cast(BedrockRuntimeClient, provider.client).meta.config
    assert config.read_timeout == 1  # type: ignore
    assert config.connect_timeout == 1  # type: ignore


def test_bedrock_provider_model_profile(env: TestEnv, mocker: MockerFixture):
    env.set('AWS_DEFAULT_REGION', 'us-east-1')
    provider = BedrockProvider()

    ns = 'pydantic_ai.providers.bedrock'
    anthropic_model_profile_mock = mocker.patch(f'{ns}.anthropic_model_profile', wraps=anthropic_model_profile)
    mistral_model_profile_mock = mocker.patch(f'{ns}.mistral_model_profile', wraps=mistral_model_profile)
    meta_model_profile_mock = mocker.patch(f'{ns}.meta_model_profile', wraps=meta_model_profile)
    cohere_model_profile_mock = mocker.patch(f'{ns}.cohere_model_profile', wraps=cohere_model_profile)
    deepseek_model_profile_mock = mocker.patch(f'{ns}.deepseek_model_profile', wraps=deepseek_model_profile)
    amazon_model_profile_mock = mocker.patch(f'{ns}.amazon_model_profile', wraps=amazon_model_profile)
    qwen_model_profile_mock = mocker.patch(f'{ns}.qwen_model_profile', wraps=qwen_model_profile)
    google_model_profile_mock = mocker.patch(f'{ns}.google_model_profile', wraps=google_model_profile)
    zai_model_profile_mock = mocker.patch(f'{ns}.zai_model_profile', wraps=zai_model_profile)
    moonshotai_model_profile_mock = mocker.patch(f'{ns}.moonshotai_model_profile', wraps=moonshotai_model_profile)

    anthropic_profile = provider.model_profile('us.anthropic.claude-3-5-sonnet-20240620-v1:0')
    anthropic_model_profile_mock.assert_called_with('claude-3-5-sonnet-20240620')
    assert isinstance(anthropic_profile, dict)
    assert anthropic_profile.get('bedrock_supports_tool_choice', False) is True
    # claude-3-5-sonnet predates Anthropic's native structured output support
    assert anthropic_profile.get('supports_json_schema_output', False) is False
    assert anthropic_profile.get('bedrock_supports_strict_tool_definition', False) is False
    assert anthropic_profile.get('json_schema_transformer', None) is BedrockJsonSchemaTransformer
    assert anthropic_profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS) == frozenset()

    anthropic_profile = provider.model_profile('us.anthropic.claude-sonnet-4-5-20250929-v1:0')
    anthropic_model_profile_mock.assert_called_with('claude-sonnet-4-5-20250929')
    assert isinstance(anthropic_profile, dict)
    assert anthropic_profile.get('bedrock_supports_tool_choice', False) is True
    assert anthropic_profile.get('supports_json_schema_output', False) is True
    assert anthropic_profile.get('bedrock_supports_strict_tool_definition', False) is True
    # Anthropic on Bedrock accepts a leading assistant turn, so we skip the synthetic user prepend.
    assert anthropic_profile.get('bedrock_supports_leading_assistant_message', False) is True
    assert anthropic_profile.get('json_schema_transformer', None) is BedrockJsonSchemaTransformer
    assert anthropic_profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS) == frozenset()
    # Anthropic rejects documents/video co-located with a `toolResult`, but accepts text and images.
    assert anthropic_profile.get('bedrock_tool_result_colocatable_content') == frozenset({'text', 'image'})
    # Anthropic accepts images and documents inside a `toolResult`; no video support.
    assert anthropic_profile.get('bedrock_supported_media_kinds_in_tool_returns') == frozenset({'image', 'document'})

    anthropic_profile = provider.model_profile('anthropic.claude-instant-v1')
    anthropic_model_profile_mock.assert_called_with('claude-instant')
    assert isinstance(anthropic_profile, dict)
    assert anthropic_profile.get('bedrock_supports_tool_choice', False) is True
    assert anthropic_profile.get('supports_json_schema_output', False) is False
    assert anthropic_profile.get('bedrock_supports_strict_tool_definition', False) is False
    assert anthropic_profile.get('json_schema_transformer', None) is BedrockJsonSchemaTransformer
    assert anthropic_profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS) == frozenset()

    anthropic_profile = provider.model_profile('us.anthropic.claude-opus-4-1-20250805-v1:0')
    anthropic_model_profile_mock.assert_called_with('claude-opus-4-1-20250805')
    assert isinstance(anthropic_profile, dict)
    assert anthropic_profile.get('supports_json_schema_output', False) is False
    assert anthropic_profile.get('bedrock_supports_strict_tool_definition', False) is False
    # Pre-4.6 Claude on Bedrock keeps the legacy `enabled + budget_tokens` translation.
    assert anthropic_profile.get('bedrock_supports_adaptive_thinking', False) is False
    assert anthropic_profile.get('bedrock_supports_effort', False) is False

    anthropic_profile = provider.model_profile('us.anthropic.claude-opus-4-7-20260115-v1:0')
    anthropic_model_profile_mock.assert_called_with('claude-opus-4-7-20260115')
    assert isinstance(anthropic_profile, dict)
    assert anthropic_profile.get('supports_json_schema_output', False) is False
    assert anthropic_profile.get('bedrock_supports_strict_tool_definition', False) is False
    assert anthropic_profile.get('bedrock_supports_adaptive_thinking', False) is True
    assert anthropic_profile.get('bedrock_supports_effort', False) is True

    anthropic_profile = provider.model_profile('us.anthropic.claude-sonnet-4-5-20250929-v1:0')
    anthropic_model_profile_mock.assert_called_with('claude-sonnet-4-5-20250929')
    assert isinstance(anthropic_profile, dict)
    # Sonnet 4.5 is the most-recent non-adaptive model — the boundary case users compare
    # against Sonnet 4.6 when evaluating this fix.
    assert anthropic_profile.get('bedrock_supports_adaptive_thinking', False) is False
    assert anthropic_profile.get('bedrock_supports_effort', False) is False

    anthropic_profile = provider.model_profile('us.anthropic.claude-opus-4-5-20251101-v1:0')
    anthropic_model_profile_mock.assert_called_with('claude-opus-4-5-20251101')
    assert isinstance(anthropic_profile, dict)
    # Opus 4.5 supports `effort` on the direct Anthropic API but Bedrock only honors it
    # alongside adaptive thinking, so the Bedrock flag must stay False here.
    assert anthropic_profile.get('bedrock_supports_adaptive_thinking', False) is False
    assert anthropic_profile.get('bedrock_supports_effort', False) is False

    anthropic_profile = provider.model_profile('us.anthropic.claude-sonnet-4-6-20251015-v1:0')
    anthropic_model_profile_mock.assert_called_with('claude-sonnet-4-6-20251015')
    assert isinstance(anthropic_profile, dict)
    # Sonnet 4.6+ requires adaptive thinking on Bedrock — see issue #5304.
    assert anthropic_profile.get('bedrock_supports_adaptive_thinking', False) is True
    assert anthropic_profile.get('bedrock_supports_effort', False) is True

    anthropic_profile = provider.model_profile('us.anthropic.claude-opus-4-6-20251015-v1:0')
    anthropic_model_profile_mock.assert_called_with('claude-opus-4-6-20251015')
    assert isinstance(anthropic_profile, dict)
    assert anthropic_profile.get('bedrock_supports_adaptive_thinking', False) is True
    assert anthropic_profile.get('bedrock_supports_effort', False) is True

    mistral_profile = provider.model_profile('mistral.mistral-large-2407-v1:0')
    mistral_model_profile_mock.assert_called_with('mistral-large-2407')
    assert isinstance(mistral_profile, dict)
    assert mistral_profile.get('bedrock_tool_result_format', 'text') == 'json'
    assert mistral_profile.get('json_schema_transformer', None) is BedrockJsonSchemaTransformer
    assert mistral_profile.get('supports_json_schema_output', False) is False
    assert mistral_profile.get('bedrock_supports_strict_tool_definition', False) is False
    assert mistral_profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS) == frozenset()
    # Mistral requires a `toolResult` to be alone in its turn.
    assert mistral_profile.get('bedrock_tool_result_colocatable_content') == frozenset()
    # Mistral accepts documents inside a `toolResult` but rejects images there; no video support.
    assert mistral_profile.get('bedrock_supported_media_kinds_in_tool_returns') == frozenset({'document'})

    mistral_profile = provider.model_profile('mistral.mistral-large-3-675b-instruct')
    mistral_model_profile_mock.assert_called_with('mistral-large-3-675b-instruct')
    assert isinstance(mistral_profile, dict)
    assert mistral_profile.get('bedrock_tool_result_format', 'text') == 'json'
    assert mistral_profile.get('json_schema_transformer', None) is BedrockJsonSchemaTransformer
    assert mistral_profile.get('supports_json_schema_output', False) is True
    assert mistral_profile.get('bedrock_supports_strict_tool_definition', False) is True
    assert mistral_profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS) == frozenset()

    meta_profile = provider.model_profile('meta.llama3-8b-instruct-v1:0')
    meta_model_profile_mock.assert_called_with('llama3-8b-instruct')
    assert meta_profile is not None
    assert meta_profile.get('json_schema_transformer', None) == InlineDefsJsonSchemaTransformer
    # Meta Llama rejects a leading assistant turn, so the strict default applies (flag unset).
    assert meta_profile.get('bedrock_supports_leading_assistant_message', False) is False
    assert meta_profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS) == frozenset()
    # Llama requires a `toolResult` to be alone in its turn.
    assert meta_profile.get('bedrock_tool_result_colocatable_content') == frozenset()
    # Meta Llama accepts images and documents inside a `toolResult`; no video support.
    assert meta_profile.get('bedrock_supported_media_kinds_in_tool_returns') == frozenset({'image', 'document'})

    cohere_profile = provider.model_profile('cohere.command-text-v14')
    cohere_model_profile_mock.assert_called_with('command-text')
    assert cohere_profile is not None
    assert cohere_profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS) == frozenset()

    deepseek_profile = provider.model_profile('deepseek.deepseek-r1')
    deepseek_model_profile_mock.assert_called_with('deepseek-r1')
    assert deepseek_profile is not None
    assert deepseek_profile.get('ignore_streamed_leading_whitespace', False) is True
    assert deepseek_profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS) == frozenset()

    qwen_profile = provider.model_profile('qwen.qwen3-32b-v1:0')
    qwen_model_profile_mock.assert_called_with('qwen3-32b')
    assert isinstance(qwen_profile, dict)
    assert qwen_profile.get('json_schema_transformer', None) is BedrockJsonSchemaTransformer
    assert qwen_profile.get('supports_json_schema_output', False) is True
    assert qwen_profile.get('bedrock_supports_strict_tool_definition', False) is True
    # Qwen on Bedrock also accepts a leading assistant turn.
    assert qwen_profile.get('bedrock_supports_leading_assistant_message', False) is True
    assert qwen_profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS) == frozenset()
    # Qwen rejects every media kind inside a `toolResult`.
    assert qwen_profile.get('bedrock_supported_media_kinds_in_tool_returns') == frozenset()

    google_profile = provider.model_profile('google.gemma-3-27b-it')
    google_model_profile_mock.assert_called_with('gemma-3-27b-it')
    assert isinstance(google_profile, dict)
    assert google_profile.get('json_schema_transformer', None) is BedrockJsonSchemaTransformer
    assert google_profile.get('supports_json_schema_output', False) is True
    assert google_profile.get('bedrock_supports_strict_tool_definition', False) is True
    assert google_profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS) == frozenset()

    # gemma-3-4b-it is NOT in the structured output supported list
    google_profile = provider.model_profile('google.gemma-3-4b-it')
    google_model_profile_mock.assert_called_with('gemma-3-4b-it')
    assert isinstance(google_profile, dict)
    assert google_profile.get('json_schema_transformer', None) is BedrockJsonSchemaTransformer
    assert google_profile.get('supports_json_schema_output', False) is False
    assert google_profile.get('bedrock_supports_strict_tool_definition', False) is False
    assert google_profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS) == frozenset()

    minimax_profile = provider.model_profile('minimax.minimax-m2')
    assert isinstance(minimax_profile, dict)
    assert minimax_profile.get('json_schema_transformer', None) is BedrockJsonSchemaTransformer
    assert minimax_profile.get('supports_json_schema_output', False) is True
    assert minimax_profile.get('bedrock_supports_strict_tool_definition', False) is True
    assert minimax_profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS) == frozenset()

    nvidia_profile = provider.model_profile('nvidia.nemotron-nano-12b-v2')
    assert isinstance(nvidia_profile, dict)
    assert nvidia_profile.get('json_schema_transformer', None) is BedrockJsonSchemaTransformer
    assert nvidia_profile.get('supports_json_schema_output', False) is True
    assert nvidia_profile.get('bedrock_supports_strict_tool_definition', False) is True
    assert nvidia_profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS) == frozenset()

    # Z.AI GLM composes the upstream `zai_model_profile` (thinking support) with Bedrock overrides.
    zai_profile = provider.model_profile('zai.glm-5')
    zai_model_profile_mock.assert_called_with('glm-5')
    assert isinstance(zai_profile, dict)
    assert zai_profile.get('bedrock_supports_tool_choice', False) is True
    assert zai_profile.get('bedrock_supports_leading_assistant_message', False) is True
    assert zai_profile.get('json_schema_transformer', None) is BedrockJsonSchemaTransformer
    assert zai_profile.get('supports_json_schema_output', False) is True
    assert zai_profile.get('bedrock_supports_strict_tool_definition', False) is True
    assert zai_profile.get('supports_thinking', False) is True
    assert zai_profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS) == frozenset()

    # Moonshot AI's Kimi models are registered under both the `moonshot.` and `moonshotai.` prefixes.
    for kimi_model in ('moonshot.kimi-k2-thinking', 'moonshotai.kimi-k2.5'):
        kimi_profile = provider.model_profile(kimi_model)
        assert isinstance(kimi_profile, dict), kimi_model
        assert kimi_profile.get('bedrock_supports_tool_choice', False) is True
        assert kimi_profile.get('bedrock_supports_leading_assistant_message', False) is True
        assert kimi_profile.get('json_schema_transformer', None) is BedrockJsonSchemaTransformer
        assert kimi_profile.get('supports_json_schema_output', False) is True
        assert kimi_profile.get('bedrock_supports_strict_tool_definition', False) is True
        assert kimi_profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS) == frozenset()
        # Kimi rejects every media kind inside a `toolResult`, so media tool returns are deferred.
        assert kimi_profile.get('bedrock_supported_media_kinds_in_tool_returns') == frozenset()
    moonshotai_model_profile_mock.assert_any_call('kimi-k2-thinking')
    moonshotai_model_profile_mock.assert_any_call('kimi-k2.5')

    # Writer Palmyra has no upstream provider module, so its profile is built from scratch.
    writer_profile = provider.model_profile('us.writer.palmyra-x4-v1:0')
    assert isinstance(writer_profile, dict)
    assert writer_profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS) == frozenset()
    assert writer_profile.get('json_schema_transformer', None) is BedrockJsonSchemaTransformer
    # Writer requires a `toolResult` to be alone in its turn, and doesn't support tool choice forcing,
    # a leading assistant turn, or structured output — all of which stay at their (falsy) defaults.
    assert writer_profile.get('bedrock_tool_result_colocatable_content') == frozenset()
    assert writer_profile.get('bedrock_supports_tool_choice', False) is False
    assert writer_profile.get('bedrock_supports_leading_assistant_message', False) is False
    assert writer_profile.get('supports_json_schema_output', False) is False
    # Writer also rejects the `status` field on a `toolResult` block.
    assert writer_profile.get('bedrock_supports_tool_result_status', True) is False

    amazon_profile = provider.model_profile('us.amazon.nova-pro-v1:0')
    amazon_model_profile_mock.assert_called_with('nova-pro')
    assert isinstance(amazon_profile, dict)
    assert amazon_profile.get('json_schema_transformer', None) == InlineDefsJsonSchemaTransformer
    assert amazon_profile.get('bedrock_supports_tool_choice', False) is True
    assert amazon_profile.get('bedrock_supports_prompt_caching', False) is True
    # Nova rejects a leading assistant turn, so the strict default (synthesize a user prepend) applies.
    assert amazon_profile.get('bedrock_supports_leading_assistant_message', False) is False
    assert amazon_profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS) == frozenset()
    # Nova has no `toolResult` co-location constraint, so it keeps the permissive default (unset).
    assert amazon_profile.get('bedrock_tool_result_colocatable_content') is None
    # Nova stays at the default (unset) media kinds: its constraint is document-format-dependent
    # (accepts csv/txt but not pdf/docx inside a `toolResult`), which this kind-based flag can't express.
    assert amazon_profile.get('bedrock_supported_media_kinds_in_tool_returns') is None

    amazon_profile = provider.model_profile('us.amazon.nova-2-lite-v1:0')
    amazon_model_profile_mock.assert_called_with('nova-2-lite')
    assert isinstance(amazon_profile, dict)
    assert amazon_profile.get('json_schema_transformer', None) == InlineDefsJsonSchemaTransformer
    assert amazon_profile.get('bedrock_supports_tool_choice', False) is True
    assert amazon_profile.get('bedrock_supports_prompt_caching', False) is True
    assert amazon_profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS) == frozenset({CodeExecutionTool})

    amazon_profile = provider.model_profile('us.amazon.titan-text-express-v1:0')
    amazon_model_profile_mock.assert_called_with('titan-text-express')
    assert amazon_profile is not None
    assert amazon_profile.get('json_schema_transformer', None) == InlineDefsJsonSchemaTransformer
    assert amazon_profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS) == frozenset()

    unknown_model = provider.model_profile('unknown-model')
    assert unknown_model is None

    unknown_model = provider.model_profile('unknown.unknown-model')
    assert unknown_model is None


@pytest.mark.parametrize(
    ('model_name', 'expected'),
    [
        ('us.anthropic.claude-sonnet-4-20250514-v1:0', 'anthropic.claude-sonnet-4-20250514-v1:0'),
        ('eu.amazon.nova-micro-v1:0', 'amazon.nova-micro-v1:0'),
        ('apac.meta.llama3-8b-instruct-v1:0', 'meta.llama3-8b-instruct-v1:0'),
        ('anthropic.claude-3-7-sonnet-20250219-v1:0', 'anthropic.claude-3-7-sonnet-20250219-v1:0'),
    ],
)
def test_remove_inference_geo_prefix(model_name: str, expected: str):
    assert remove_bedrock_geo_prefix(model_name) == expected


@pytest.mark.parametrize(
    ('model_id', 'expected'),
    [
        ('us.anthropic.claude-haiku-4-5-20251001-v1:0', ('anthropic', 'claude-haiku-4-5-20251001')),
        ('anthropic.claude-haiku-4-5-20251001-v1:0', ('anthropic', 'claude-haiku-4-5-20251001')),
        ('anthropic.claude-haiku-4-5', ('anthropic', 'claude-haiku-4-5')),
        ('eu.amazon.nova-micro-v1:0', ('amazon', 'nova-micro')),
        ('cohere.command-r-v1:0', ('cohere', 'command-r')),
        ('meta.llama3-8b-instruct-v14', ('meta', 'llama3-8b-instruct')),
        # Not a `<provider>.<name>` shape — returned unchanged.
        ('claude-haiku-4-5', (None, 'claude-haiku-4-5')),
        ('claude-haiku-4-5@20251001', (None, 'claude-haiku-4-5@20251001')),
    ],
)
def test_split_bedrock_model_id(model_id: str, expected: tuple[str | None, str]):
    assert split_bedrock_model_id(model_id) == expected


@pytest.mark.parametrize('prefix', BEDROCK_GEO_PREFIXES)
def test_bedrock_provider_model_profile_all_geo_prefixes(env: TestEnv, prefix: str):
    """Test that all cross-region inference geo prefixes are correctly handled."""
    env.set('AWS_DEFAULT_REGION', 'us-east-1')
    provider = BedrockProvider()

    model_name = f'{prefix}.anthropic.claude-sonnet-4-5-20250929-v1:0'
    profile = provider.model_profile(model_name)

    assert profile is not None, f'model_profile returned None for {model_name}'


def test_bedrock_provider_model_profile_with_unknown_geo_prefix(env: TestEnv):
    env.set('AWS_DEFAULT_REGION', 'us-east-1')
    provider = BedrockProvider()

    model_name = 'narnia.anthropic.claude-sonnet-4-5-20250929-v1:0'
    profile = provider.model_profile(model_name)
    assert profile is None, f'model_profile returned {profile} for {model_name}'


def test_latest_bedrock_model_names_geo_prefixes_are_supported():
    """Ensure all geo prefixes used in LatestBedrockModelNames are in BEDROCK_GEO_PREFIXES.

    This test prevents adding new model names with geo prefixes that aren't handled
    by the provider's model_profile method.
    """
    model_names = get_args(LatestBedrockModelNames)

    missing_prefixes: set[str] = set()

    # Known provider prefixes that are not geo prefixes (e.g. 'minimax.minimax-m2.1' has 3 parts
    # but 'minimax' is a provider, not a geo prefix)
    known_providers = {
        'anthropic',
        'mistral',
        'cohere',
        'amazon',
        'meta',
        'deepseek',
        'qwen',
        'google',
        'minimax',
        'nvidia',
        'writer',
        'zai',
        'moonshot',
        'moonshotai',
    }

    for model_name in model_names:
        # Model names with geo prefixes have 3+ dot-separated parts:
        # - No prefix: "anthropic.claude-xxx" (2 parts)
        # - With prefix: "us.anthropic.claude-xxx" (3 parts)
        # - Provider with dot in model name: "minimax.minimax-m2.1" (3 parts, not a geo prefix)
        parts = model_name.split('.')
        if len(parts) >= 3:
            geo_prefix = parts[0]
            if geo_prefix not in BEDROCK_GEO_PREFIXES and geo_prefix not in known_providers:  # pragma: no cover
                missing_prefixes.add(geo_prefix)

    if missing_prefixes:  # pragma: no cover
        pytest.fail(
            f'Found geo prefixes in LatestBedrockModelNames that are not in BEDROCK_GEO_PREFIXES: {missing_prefixes}. '
            f'Please add them to BEDROCK_GEO_PREFIXES'
        )


def test_strict_true_simple_schema():
    """With strict=True, simple object schemas get Bedrock-required additionalProperties=false."""

    class Person(BaseModel):
        name: str
        age: int

    transformer = BedrockJsonSchemaTransformer(Person.model_json_schema(), strict=True)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is True
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}},
            'required': ['name', 'age'],
            'additionalProperties': False,
        }
    )


def test_strict_true_schema_with_constraints():
    """With strict=True, string constraints (minLength, pattern) are preserved — Bedrock accepts these."""

    class User(BaseModel):
        username: Annotated[str, Field(min_length=3)]
        email: Annotated[str, Field(pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')]

    original_schema = User.model_json_schema()
    transformer = BedrockJsonSchemaTransformer(original_schema, strict=True)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is True
    assert original_schema == snapshot(
        {
            'properties': {
                'username': {'minLength': 3, 'title': 'Username', 'type': 'string'},
                'email': {'pattern': '^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$', 'title': 'Email', 'type': 'string'},
            },
            'required': ['username', 'email'],
            'title': 'User',
            'type': 'object',
        }
    )
    # String constraints preserved (Bedrock accepts minLength/pattern), title removed
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'username': {'minLength': 3, 'type': 'string'},
                'email': {'pattern': '^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$', 'type': 'string'},
            },
            'required': ['username', 'email'],
            'additionalProperties': False,
        }
    )


def test_strict_true_nested_model():
    """With strict=True, nested models with $defs are preserved."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    transformer = BedrockJsonSchemaTransformer(Person.model_json_schema(), strict=True)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is True
    assert transformed == snapshot(
        {
            '$defs': {
                'Address': {
                    'type': 'object',
                    'properties': {
                        'street': {'type': 'string'},
                        'city': {'type': 'string'},
                    },
                    'required': ['street', 'city'],
                    'additionalProperties': False,
                }
            },
            'type': 'object',
            'additionalProperties': False,
            'properties': {'name': {'type': 'string'}, 'address': {'$ref': '#/$defs/Address'}},
            'required': ['name', 'address'],
        }
    )


def test_strict_false_preserves_schema():
    """With strict=False, schema is preserved as-is (no additionalProperties injection, no constraint stripping)."""

    class User(BaseModel):
        username: Annotated[str, Field(min_length=3)]
        age: int

    transformer = BedrockJsonSchemaTransformer(User.model_json_schema(), strict=False)
    transformed = transformer.walk()

    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'username': {'minLength': 3, 'type': 'string'},
                'age': {'type': 'integer'},
            },
            'required': ['username', 'age'],
        }
    )


def test_strict_none_preserves_schema():
    """With strict=None, strict-mode rewrites are skipped and is_strict_compatible=False.

    Mirrors Anthropic: strict=None never auto-promotes to True — the caller must opt in
    explicitly. See https://github.com/pydantic/pydantic-ai/issues/5579. `title` and
    `$schema` are still stripped (always-on transformer behavior).
    """

    class User(BaseModel):
        username: Annotated[str, Field(min_length=3)]
        age: int

    transformer = BedrockJsonSchemaTransformer(User.model_json_schema(), strict=None)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is False
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'username': {'minLength': 3, 'type': 'string'},
                'age': {'type': 'integer'},
            },
            'required': ['username', 'age'],
        }
    )


def test_strict_none_simple_schema():
    """With strict=None, even simple schemas are not strict-compatible — opt-in required."""

    class Person(BaseModel):
        name: str
        age: int

    transformer = BedrockJsonSchemaTransformer(Person.model_json_schema(), strict=None)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is False
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}},
            'required': ['name', 'age'],
        }
    )


def test_strict_none_never_strict_compatible():
    """With strict=None and constrained fields, is_strict_compatible=False and constraints survive.

    Mirrors the Anthropic transformer's stance — strict=None is never auto-promoted.
    """

    class ConstrainedInput(BaseModel):
        username: Annotated[str, Field(min_length=3)]
        count: Annotated[int, Field(ge=0)]

    transformer = BedrockJsonSchemaTransformer(ConstrainedInput.model_json_schema(), strict=None)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is False
    # Constraints are preserved (no stripping when strict is not True)
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'username': {'minLength': 3, 'type': 'string'},
                'count': {'minimum': 0, 'type': 'integer'},
            },
            'required': ['username', 'count'],
        }
    )


def test_strict_none_with_additional_properties_true():
    """With strict=None and explicit additionalProperties=True, is_strict_compatible=False and value preserved."""
    schema = {
        'type': 'object',
        'properties': {'name': {'type': 'string'}},
        'required': ['name'],
        'additionalProperties': True,
    }
    transformer = BedrockJsonSchemaTransformer(schema, strict=None)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is False
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {'name': {'type': 'string'}},
            'required': ['name'],
            'additionalProperties': True,
        }
    )


def test_strict_true_strips_numeric_constraints():
    """With strict=True, numeric constraints (minimum, maximum, multipleOf) are stripped and noted in description."""

    class Task(BaseModel):
        score: Annotated[float, Field(ge=0.0, le=100.0)]
        rating: Annotated[int, Field(multiple_of=5)]

    transformer = BedrockJsonSchemaTransformer(Task.model_json_schema(), strict=True)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is True
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'score': {'type': 'number', 'description': 'minimum=0.0, maximum=100.0'},
                'rating': {'type': 'integer', 'description': 'multipleOf=5'},
            },
            'required': ['score', 'rating'],
            'additionalProperties': False,
        }
    )


def test_strict_true_strips_exclusive_bounds():
    """With strict=True, exclusive bounds (gt, lt) are stripped and noted in description."""

    class Range(BaseModel):
        value: Annotated[int, Field(gt=0, lt=100)]

    transformer = BedrockJsonSchemaTransformer(Range.model_json_schema(), strict=True)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is True
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'value': {'type': 'integer', 'description': 'exclusiveMinimum=0, exclusiveMaximum=100'},
            },
            'required': ['value'],
            'additionalProperties': False,
        }
    )


def test_strict_true_strips_array_max_items():
    """With strict=True, maxItems is stripped and noted in description."""

    class Config(BaseModel):
        tags: Annotated[list[str], Field(max_length=5)]

    transformer = BedrockJsonSchemaTransformer(Config.model_json_schema(), strict=True)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is True
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'tags': {'type': 'array', 'items': {'type': 'string'}, 'description': 'maxItems=5'},
            },
            'required': ['tags'],
            'additionalProperties': False,
        }
    )


def test_strict_true_strips_array_min_items_gt1():
    """With strict=True, minItems > 1 is stripped and noted in description."""

    class Config(BaseModel):
        tags: Annotated[list[str], Field(min_length=3)]

    transformer = BedrockJsonSchemaTransformer(Config.model_json_schema(), strict=True)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is True
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'tags': {'type': 'array', 'items': {'type': 'string'}, 'description': 'minItems=3'},
            },
            'required': ['tags'],
            'additionalProperties': False,
        }
    )


def test_strict_true_preserves_array_min_items_0_and_1():
    """With strict=True, minItems=0 and minItems=1 are preserved — Bedrock accepts these."""

    class Config(BaseModel):
        optional_tags: Annotated[list[str], Field(min_length=0)]
        required_tags: Annotated[list[str], Field(min_length=1)]

    transformer = BedrockJsonSchemaTransformer(Config.model_json_schema(), strict=True)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is True
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'optional_tags': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 0},
                'required_tags': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1},
            },
            'required': ['optional_tags', 'required_tags'],
            'additionalProperties': False,
        }
    )


def test_strict_true_preserves_string_constraints():
    """With strict=True, string constraints (minLength, maxLength, pattern) are preserved."""

    class Input(BaseModel):
        name: Annotated[str, Field(min_length=1, max_length=100)]
        code: Annotated[str, Field(pattern=r'^[A-Z]{3}$')]

    transformer = BedrockJsonSchemaTransformer(Input.model_json_schema(), strict=True)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is True
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'name': {'type': 'string', 'minLength': 1, 'maxLength': 100},
                'code': {'type': 'string', 'pattern': '^[A-Z]{3}$'},
            },
            'required': ['name', 'code'],
            'additionalProperties': False,
        }
    )


def test_strict_true_mixed_constraints():
    """With strict=True, numeric constraints are stripped while string constraints on the same model are kept."""

    class MixedModel(BaseModel):
        name: Annotated[str, Field(min_length=1)]
        score: Annotated[float, Field(ge=0.0, le=100.0)]

    transformer = BedrockJsonSchemaTransformer(MixedModel.model_json_schema(), strict=True)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is True
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'name': {'type': 'string', 'minLength': 1},
                'score': {'type': 'number', 'description': 'minimum=0.0, maximum=100.0'},
            },
            'required': ['name', 'score'],
            'additionalProperties': False,
        }
    )


def test_strict_true_description_appended():
    """With strict=True, stripped constraint info is appended to existing description, not replacing it."""

    class Task(BaseModel):
        score: Annotated[float, Field(ge=0.0, le=100.0, description='The task score')]

    transformer = BedrockJsonSchemaTransformer(Task.model_json_schema(), strict=True)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is True
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'score': {
                    'type': 'number',
                    'description': 'The task score (minimum=0.0, maximum=100.0)',
                },
            },
            'required': ['score'],
            'additionalProperties': False,
        }
    )


def test_strict_true_preserves_default_values():
    """With strict=True, default values are preserved — Bedrock accepts these."""

    class CityWithDefaults(BaseModel):
        city: str
        country: str = 'Unknown'
        population: int = 0

    transformer = BedrockJsonSchemaTransformer(CityWithDefaults.model_json_schema(), strict=True)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is True
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'city': {'type': 'string'},
                'country': {'default': 'Unknown', 'type': 'string'},
                'population': {'default': 0, 'type': 'integer'},
            },
            'required': ['city'],
            'additionalProperties': False,
        }
    )


def test_strict_true_preserves_any_of_with_null():
    """With strict=True, anyOf with null type (optional fields) is preserved — Bedrock accepts these."""

    class PersonOptional(BaseModel):
        name: str
        nickname: str | None = None

    transformer = BedrockJsonSchemaTransformer(PersonOptional.model_json_schema(), strict=True)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is True
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'nickname': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None},
            },
            'required': ['name'],
            'additionalProperties': False,
        }
    )


def test_strict_true_preserves_literal_unions():
    """With strict=True, Literal union types are preserved via anyOf — Bedrock accepts these."""

    class StatusModel(BaseModel):
        status: Literal['active', 'inactive'] | int

    transformer = BedrockJsonSchemaTransformer(StatusModel.model_json_schema(), strict=True)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is True
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'status': {
                    'anyOf': [{'enum': ['active', 'inactive'], 'type': 'string'}, {'type': 'integer'}],
                },
            },
            'required': ['status'],
            'additionalProperties': False,
        }
    )


def test_strict_false_preserves_numeric_constraints():
    """With strict=False, numeric constraints are preserved — no stripping occurs."""

    class Task(BaseModel):
        score: Annotated[float, Field(ge=0.0, le=100.0)]
        rating: Annotated[int, Field(multiple_of=5)]

    transformer = BedrockJsonSchemaTransformer(Task.model_json_schema(), strict=False)
    transformed = transformer.walk()

    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'score': {'type': 'number', 'minimum': 0.0, 'maximum': 100.0},
                'rating': {'type': 'integer', 'multipleOf': 5},
            },
            'required': ['score', 'rating'],
        }
    )


def test_strict_none_preserves_numeric_constraints():
    """With strict=None, numeric constraints are preserved — no stripping, no auto-promotion."""

    class Task(BaseModel):
        score: Annotated[float, Field(ge=0.0, le=100.0)]

    transformer = BedrockJsonSchemaTransformer(Task.model_json_schema(), strict=None)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is False
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'score': {'type': 'number', 'minimum': 0.0, 'maximum': 100.0},
            },
            'required': ['score'],
        }
    )
