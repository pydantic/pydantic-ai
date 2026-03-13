from typing import Annotated, Literal, cast, get_args

import pytest
from pydantic import BaseModel, Field
from pytest_mock import MockerFixture

from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.builtin_tools import CodeExecutionTool
from pydantic_ai.profiles.amazon import amazon_model_profile
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.qwen import qwen_model_profile

from .._inline_snapshot import snapshot
from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient

    from pydantic_ai.models.bedrock import LatestBedrockModelNames
    from pydantic_ai.providers.bedrock import (
        BEDROCK_GEO_PREFIXES,
        BedrockJsonSchemaTransformer,
        BedrockModelProfile,
        BedrockProvider,
        remove_bedrock_geo_prefix,
    )

if not imports_successful():
    BEDROCK_GEO_PREFIXES: tuple[str, ...] = ()  # pragma: lax no cover  # type: ignore[no-redef]

pytestmark = pytest.mark.skipif(not imports_successful(), reason='bedrock not installed')


def test_bedrock_provider(env: TestEnv):
    env.set('AWS_DEFAULT_REGION', 'us-east-1')
    provider = BedrockProvider()
    assert isinstance(provider, BedrockProvider)
    assert provider.name == 'bedrock'
    assert provider.base_url == 'https://bedrock-runtime.us-east-1.amazonaws.com'


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

    anthropic_profile = provider.model_profile('us.anthropic.claude-3-5-sonnet-20240620-v1:0')
    anthropic_model_profile_mock.assert_called_with('claude-3-5-sonnet-20240620')
    assert isinstance(anthropic_profile, BedrockModelProfile)
    assert anthropic_profile.bedrock_supports_tool_choice is True
    # claude-3-5-sonnet predates Anthropic's native structured output support
    assert anthropic_profile.supports_json_schema_output is False
    assert anthropic_profile.json_schema_transformer is BedrockJsonSchemaTransformer
    assert anthropic_profile.supported_builtin_tools == frozenset()

    anthropic_profile = provider.model_profile('us.anthropic.claude-sonnet-4-5-20250929-v1:0')
    anthropic_model_profile_mock.assert_called_with('claude-sonnet-4-5-20250929')
    assert isinstance(anthropic_profile, BedrockModelProfile)
    assert anthropic_profile.bedrock_supports_tool_choice is True
    assert anthropic_profile.supports_json_schema_output is True
    assert anthropic_profile.json_schema_transformer is BedrockJsonSchemaTransformer
    assert anthropic_profile.supported_builtin_tools == frozenset()

    anthropic_profile = provider.model_profile('anthropic.claude-instant-v1')
    anthropic_model_profile_mock.assert_called_with('claude-instant')
    assert isinstance(anthropic_profile, BedrockModelProfile)
    assert anthropic_profile.bedrock_supports_tool_choice is True
    assert anthropic_profile.supports_json_schema_output is False
    assert anthropic_profile.json_schema_transformer is BedrockJsonSchemaTransformer
    assert anthropic_profile.supported_builtin_tools == frozenset()

    mistral_profile = provider.model_profile('mistral.mistral-large-2407-v1:0')
    mistral_model_profile_mock.assert_called_with('mistral-large-2407')
    assert isinstance(mistral_profile, BedrockModelProfile)
    assert mistral_profile.bedrock_tool_result_format == 'json'
    assert mistral_profile.json_schema_transformer is BedrockJsonSchemaTransformer
    assert mistral_profile.supports_json_schema_output is False
    assert mistral_profile.supported_builtin_tools == frozenset()

    mistral_profile = provider.model_profile('mistral.mistral-large-3-675b-instruct')
    mistral_model_profile_mock.assert_called_with('mistral-large-3-675b-instruct')
    assert isinstance(mistral_profile, BedrockModelProfile)
    assert mistral_profile.bedrock_tool_result_format == 'json'
    assert mistral_profile.json_schema_transformer is BedrockJsonSchemaTransformer
    assert mistral_profile.supports_json_schema_output is True
    assert mistral_profile.supported_builtin_tools == frozenset()

    meta_profile = provider.model_profile('meta.llama3-8b-instruct-v1:0')
    meta_model_profile_mock.assert_called_with('llama3-8b-instruct')
    assert meta_profile is not None
    assert meta_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer
    assert meta_profile.supported_builtin_tools == frozenset()

    cohere_profile = provider.model_profile('cohere.command-text-v14')
    cohere_model_profile_mock.assert_called_with('command-text')
    assert cohere_profile is not None
    assert cohere_profile.supported_builtin_tools == frozenset()

    deepseek_profile = provider.model_profile('deepseek.deepseek-r1')
    deepseek_model_profile_mock.assert_called_with('deepseek-r1')
    assert deepseek_profile is not None
    assert deepseek_profile.ignore_streamed_leading_whitespace is True
    assert deepseek_profile.supported_builtin_tools == frozenset()

    qwen_profile = provider.model_profile('qwen.qwen3-32b-v1:0')
    qwen_model_profile_mock.assert_called_with('qwen3-32b')
    assert isinstance(qwen_profile, BedrockModelProfile)
    assert qwen_profile.json_schema_transformer is BedrockJsonSchemaTransformer
    assert qwen_profile.supports_json_schema_output is True
    assert qwen_profile.supported_builtin_tools == frozenset()

    google_profile = provider.model_profile('google.gemma-3-27b-it')
    google_model_profile_mock.assert_called_with('gemma-3-27b-it')
    assert isinstance(google_profile, BedrockModelProfile)
    assert google_profile.json_schema_transformer is BedrockJsonSchemaTransformer
    assert google_profile.supports_json_schema_output is True
    assert google_profile.supported_builtin_tools == frozenset()

    # gemma-3-4b-it is NOT in the structured output supported list
    google_profile = provider.model_profile('google.gemma-3-4b-it')
    google_model_profile_mock.assert_called_with('gemma-3-4b-it')
    assert isinstance(google_profile, BedrockModelProfile)
    assert google_profile.json_schema_transformer is BedrockJsonSchemaTransformer
    assert google_profile.supports_json_schema_output is False
    assert google_profile.supported_builtin_tools == frozenset()

    minimax_profile = provider.model_profile('minimax.minimax-m2')
    assert isinstance(minimax_profile, BedrockModelProfile)
    assert minimax_profile.json_schema_transformer is BedrockJsonSchemaTransformer
    assert minimax_profile.supports_json_schema_output is True
    assert minimax_profile.supported_builtin_tools == frozenset()

    nvidia_profile = provider.model_profile('nvidia.nemotron-nano-12b-v2')
    assert isinstance(nvidia_profile, BedrockModelProfile)
    assert nvidia_profile.json_schema_transformer is BedrockJsonSchemaTransformer
    assert nvidia_profile.supports_json_schema_output is True
    assert nvidia_profile.supported_builtin_tools == frozenset()

    amazon_profile = provider.model_profile('us.amazon.nova-pro-v1:0')
    amazon_model_profile_mock.assert_called_with('nova-pro')
    assert isinstance(amazon_profile, BedrockModelProfile)
    assert amazon_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer
    assert amazon_profile.bedrock_supports_tool_choice is True
    assert amazon_profile.bedrock_supports_prompt_caching is True
    assert amazon_profile.supported_builtin_tools == frozenset()

    amazon_profile = provider.model_profile('us.amazon.nova-2-lite-v1:0')
    amazon_model_profile_mock.assert_called_with('nova-2-lite')
    assert isinstance(amazon_profile, BedrockModelProfile)
    assert amazon_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer
    assert amazon_profile.bedrock_supports_tool_choice is True
    assert amazon_profile.bedrock_supports_prompt_caching is True
    assert amazon_profile.supported_builtin_tools == frozenset({CodeExecutionTool})

    amazon_profile = provider.model_profile('us.amazon.titan-text-express-v1:0')
    amazon_model_profile_mock.assert_called_with('titan-text-express')
    assert amazon_profile is not None
    assert amazon_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer
    assert amazon_profile.supported_builtin_tools == frozenset()

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


# =============================================================================
# BedrockJsonSchemaTransformer Tests
# =============================================================================


def test_strict_true_simple_schema():
    """With strict=True, simple schemas are returned (no-op transform), is_strict_compatible=True."""

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
    """With strict=None (default), schema is preserved, is_strict_compatible=True when compatible."""

    class User(BaseModel):
        username: Annotated[str, Field(min_length=3)]
        age: int

    transformer = BedrockJsonSchemaTransformer(User.model_json_schema(), strict=None)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is True
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'username': {'minLength': 3, 'type': 'string'},
                'age': {'type': 'integer'},
            },
            'required': ['username', 'age'],
            'additionalProperties': False,
        }
    )


def test_strict_none_simple_schema():
    """With strict=None, simple schemas are strict-compatible."""

    class Person(BaseModel):
        name: str
        age: int

    transformer = BedrockJsonSchemaTransformer(Person.model_json_schema(), strict=None)
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


def test_strict_none_incompatible_schema_disables_auto_strict():
    """With strict=None and constrained fields, is_strict_compatible=False.

    This ensures strict is NOT auto-enabled for tools with constrained schemas,
    mirroring the Anthropic test_strict_tools_incompatible_schema_not_auto_enabled.
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
            'additionalProperties': False,
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
            'additionalProperties': False,
        }
    )
