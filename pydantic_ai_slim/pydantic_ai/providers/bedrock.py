from __future__ import annotations as _annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal, overload

from pydantic_ai import ModelProfile
from pydantic_ai._json_schema import JsonSchema, JsonSchemaTransformer
from pydantic_ai.exceptions import UserError
from pydantic_ai.native_tools import CodeExecutionTool
from pydantic_ai.profiles.amazon import amazon_model_profile
from pydantic_ai.profiles.anthropic import AnthropicModelProfile, anthropic_model_profile
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.providers import Provider
from pydantic_ai.providers._bedrock_model_names import (
    BEDROCK_GEO_PREFIXES as BEDROCK_GEO_PREFIXES,  # re-exported for backwards compatibility
    remove_bedrock_geo_prefix as remove_bedrock_geo_prefix,  # re-exported for backwards compatibility
    split_bedrock_model_id,
)

try:
    import boto3
    from botocore.client import BaseClient
    from botocore.config import Config
    from botocore.exceptions import NoRegionError
    from botocore.session import Session
    from botocore.tokens import FrozenAuthToken
except ImportError as _import_error:
    raise ImportError(
        'Please install the `boto3` package to use the Bedrock provider, '
        'you can use the `bedrock` optional group — `pip install "pydantic-ai-slim[bedrock]"`'
    ) from _import_error


# JSON Schema keys that Bedrock structured output rejects with a 400 under `strict=True`.
# Source: empirically verified against `us.anthropic.claude-sonnet-4-5` on 2026-05-19; AWS docs
# at https://docs.aws.amazon.com/bedrock/latest/userguide/structured-output.html disagree in two
# places so the wire response is the source of truth:
#   - the doc lists string constraints (`minLength`, `maxLength`) as unsupported, but Bedrock
#     accepts them — do NOT add them here.
#   - the doc lists only `minimum`/`maximum`/`multipleOf` for numerical types, but Bedrock also
#     rejects `exclusiveMinimum`/`exclusiveMaximum` — those are stripped too.
# `array.minItems` is conditionally unsupported (Bedrock allows 0 or 1, rejects >1), so it's not
# in this mapping — handled inline in `transform()`.
# Repro: see `tests/providers/test_bedrock.py::test_bedrock_strict_unsupported_keys_*` cassettes.
# Tuples (not sets) — iteration order shapes the synthesized description string and is asserted
# by tests; keep the order matching the JSON Schema spec's listing of each keyword family.
_BEDROCK_STRICT_UNSUPPORTED_KEYS_BY_TYPE: dict[str, tuple[str, ...]] = {
    'number': ('minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum', 'multipleOf'),
    'integer': ('minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum', 'multipleOf'),
    'array': ('maxItems',),
}


@dataclass(init=False)
class BedrockJsonSchemaTransformer(JsonSchemaTransformer):
    """Transforms schemas to the subset supported by Bedrock structured outputs.

    The transformer is applied to Bedrock tool and output schemas during request
    customization. Strict-mode rewrites are applied when:
    - `NativeOutput` is used as the `output_type` of the Agent. `BedrockConverseModel`
      forces native output schemas to `strict=True` before request customization.
    - `strict=True` is set explicitly on a Tool.

    Like `AnthropicJsonSchemaTransformer`, Bedrock does not infer strict tool mode
    from `strict=None`. Strict tool definitions are opt-in: callers must set
    `strict=True` explicitly. This avoids silently changing large toolsets into
    strict toolsets, which can exceed Anthropic/Bedrock's 20 strict-tools-per-request
    limit, and avoids applying potentially lossy strict-mode schema rewrites unless
    requested.

    When `strict=True`, `additionalProperties: false` is injected on objects and keys
    Bedrock rejects are removed from the schema and re-emitted into the field's
    `description` so the model still has the hint.
    """

    def walk(self) -> JsonSchema:
        schema = super().walk()

        # `_customize_tool_def()` and `_customize_output_object()` use this flag
        # to resolve `strict=None`. For Bedrock tools, strict mode is opt-in, so
        # only an explicit `strict=True` should resolve to strict-compatible.
        self.is_strict_compatible = self.strict is True

        return schema

    def transform(self, schema: JsonSchema) -> JsonSchema:
        schema.pop('title', None)
        schema.pop('$schema', None)

        if not self.strict:
            return schema

        schema_type = schema.get('type')

        if schema_type == 'object':
            schema['additionalProperties'] = False

        incompatible: dict[str, object] = {}
        if isinstance(schema_type, str):
            for key in _BEDROCK_STRICT_UNSUPPORTED_KEYS_BY_TYPE.get(schema_type, ()):
                if key in schema:
                    incompatible[key] = schema[key]
            if schema_type == 'array' and schema.get('minItems', 0) > 1:
                incompatible['minItems'] = schema['minItems']

        if incompatible:
            notes: list[str] = []
            for key, value in incompatible.items():
                schema.pop(key)
                notes.append(f'{key}={value}')
            notes_str = ', '.join(notes)
            desc = schema.get('description')
            schema['description'] = notes_str if not desc else f'{desc} ({notes_str})'

        return schema


@dataclass(kw_only=True)
class BedrockModelProfile(ModelProfile):
    """Profile for models used with BedrockModel.

    ALL FIELDS MUST BE `bedrock_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    bedrock_supports_tool_choice: bool = False
    bedrock_tool_result_format: Literal['text', 'json'] = 'text'
    bedrock_send_back_thinking_parts: bool = False
    bedrock_supports_prompt_caching: bool = False
    bedrock_supports_tool_caching: bool = False
    bedrock_supported_media_kinds_in_tool_returns: frozenset[str] = frozenset({'image'})
    bedrock_supports_strict_tool_definition: bool = False
    """Whether this model accepts `strict: true` on `toolSpec` in Bedrock's Converse API.

    Tracked separately from `supports_json_schema_output` (which gates `NativeOutput` /
    `outputConfig`) because AWS could in principle ship a model that supports one without the
    other; today both features track the same per-model allowlist per the Bedrock structured-output
    docs: https://docs.aws.amazon.com/bedrock/latest/userguide/structured-output.html.
    """

    bedrock_thinking_variant: Literal['anthropic', 'openai', 'qwen'] | None = None
    """Which thinking API shape to use for unified thinking translation.

    - `'anthropic'`: Uses `{'thinking': {'type': 'adaptive'}}` for 4.6+ models,
      or `{'thinking': {'type': 'enabled', 'budget_tokens': N}}` for older models.
    - `'openai'`: Uses `{'reasoning_effort': 'low'|'medium'|'high'}`
    - `'qwen'`: Uses `{'reasoning_config': 'low'|'high'}`
    - `None`: No unified thinking support.
    """

    bedrock_supports_adaptive_thinking: bool = False
    """Whether this model accepts `{'thinking': {'type': 'adaptive'}}` (Sonnet 4.6+, Opus 4.6+).

    Only meaningful for the `'anthropic'` variant. When False, the variant falls back to
    `{'type': 'enabled', 'budget_tokens': N}` for pre-4.6 models.
    """

    bedrock_supports_effort: bool = False
    """Whether this model emits `output_config.effort` on Bedrock Converse (Sonnet 4.6+, Opus 4.6+).

    Only meaningful for the `'anthropic'` variant AND only honored alongside
    `bedrock_supports_adaptive_thinking=True`. Bedrock has not been verified to accept
    `output_config.effort` on the legacy `{'type': 'enabled', 'budget_tokens': N}` path
    (e.g. Opus 4.5), so the translator skips it there even though the direct Anthropic
    API accepts it. Effort lives at `additionalModelRequestFields.output_config.effort`
    (a sibling of `thinking`, not inside it).
    """


def bedrock_anthropic_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for an Anthropic model used via Bedrock."""
    # These Opus models support structured output on the direct Anthropic API but are not listed
    # in the Bedrock Runtime structured-output docs:
    # https://docs.aws.amazon.com/bedrock/latest/userguide/structured-output.html
    bedrock_structured_output_unsupported = ('claude-opus-4-1', 'claude-opus-4-7', 'claude-opus-4-8')
    downstream = anthropic_model_profile(model_name)
    # Read anthropic_* capability flags before update() strips them: ModelProfile.update()
    # only copies fields that exist on self, so anthropic-prefixed fields would be lost.
    is_anthropic = isinstance(downstream, AnthropicModelProfile)
    supports_adaptive = is_anthropic and downstream.anthropic_supports_adaptive_thinking
    # Bedrock only honors effort inside the adaptive branch of `_translate_thinking`, so don't claim
    # support for non-adaptive models (e.g. Opus 4.5) even when the direct Anthropic API supports it.
    supports_effort = supports_adaptive and is_anthropic and downstream.anthropic_supports_effort
    profile = BedrockModelProfile(
        bedrock_supports_tool_choice=True,
        bedrock_send_back_thinking_parts=True,
        bedrock_supports_prompt_caching=True,
        bedrock_supports_tool_caching=True,
        bedrock_supported_media_kinds_in_tool_returns=frozenset({'image', 'document'}),
        bedrock_thinking_variant='anthropic',
        bedrock_supports_adaptive_thinking=supports_adaptive,
        bedrock_supports_effort=supports_effort,
    ).update(_without_builtin_tools(downstream))
    supports_structured_output = profile.supports_json_schema_output and not model_name.startswith(
        bedrock_structured_output_unsupported
    )
    return replace(
        profile,
        json_schema_transformer=BedrockJsonSchemaTransformer,
        supports_json_schema_output=supports_structured_output,
        bedrock_supports_strict_tool_definition=supports_structured_output,
    )


def bedrock_amazon_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for an Amazon model used via Bedrock."""
    profile = _without_builtin_tools(amazon_model_profile(model_name))
    if 'nova' in model_name:
        profile = BedrockModelProfile(
            bedrock_supports_tool_choice=True,
            bedrock_supports_prompt_caching=True,
        ).update(profile)

    if 'nova-2' in model_name:
        profile.supported_native_tools = frozenset({CodeExecutionTool})

    return profile


def bedrock_deepseek_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a DeepSeek model used via Bedrock."""
    profile = deepseek_model_profile(model_name)
    if 'r1' in model_name:
        return BedrockModelProfile(bedrock_send_back_thinking_parts=True).update(profile)
    return profile  # pragma: no cover


def bedrock_mistral_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Mistral model used via Bedrock."""
    models_that_support_structured_output = ('magistral-small', 'ministral-3', 'mistral-large-3', 'voxtral')
    supports_structured_output = model_name.startswith(models_that_support_structured_output)
    return replace(
        BedrockModelProfile(
            bedrock_tool_result_format='json',
        ).update(_without_builtin_tools(mistral_model_profile(model_name))),
        json_schema_transformer=BedrockJsonSchemaTransformer,
        supports_json_schema_output=supports_structured_output,
        bedrock_supports_strict_tool_definition=supports_structured_output,
    )


def bedrock_qwen_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Qwen model used via Bedrock."""
    models_that_support_structured_output = ('qwen3',)
    supports_structured_output = model_name.startswith(models_that_support_structured_output)
    # Bedrock-Converse exposes only `reasoning_config ∈ {low, high}` for Qwen3 — no disable value.
    supports_reasoning = 'qwq' in model_name or 'qwen3' in model_name
    return replace(
        BedrockModelProfile(
            bedrock_thinking_variant='qwen',
            supports_thinking=supports_reasoning,
            thinking_always_enabled=supports_reasoning,
        ).update(_without_builtin_tools(qwen_model_profile(model_name))),
        json_schema_transformer=BedrockJsonSchemaTransformer,
        supports_json_schema_output=supports_structured_output,
        bedrock_supports_strict_tool_definition=supports_structured_output,
        # Bedrock Converse API doesn't support JSON object mode
        supports_json_object_output=False,
    )


def bedrock_google_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Google model used via Bedrock."""
    models_that_support_structured_output = ('gemma-3-12b-it', 'gemma-3-27b-it')
    supports_structured_output = model_name.startswith(models_that_support_structured_output)
    return replace(
        BedrockModelProfile().update(_without_builtin_tools(google_model_profile(model_name))),
        json_schema_transformer=BedrockJsonSchemaTransformer,
        supports_json_schema_output=supports_structured_output,
        bedrock_supports_strict_tool_definition=supports_structured_output,
        # Bedrock Converse API doesn't support JSON object mode
        supports_json_object_output=False,
        # Bedrock Converse API doesn't support tool return schemas natively
        supports_tool_return_schema=False,
    )


# MiniMax and NVIDIA don't have non-Bedrock provider modules in `pydantic_ai/profiles/`, so
# these profile fns build a `BedrockModelProfile` from scratch instead of composing with an
# upstream profile via `_without_builtin_tools(<upstream>_model_profile(model_name))` like the
# other `bedrock_<vendor>_model_profile` fns do. The inline `'openai'` lambda in
# `BedrockProvider.model_profile` follows the same from-scratch pattern for the same reason.


def bedrock_minimax_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a MiniMax model used via Bedrock."""
    models_that_support_structured_output = ('minimax-m2',)
    supports_structured_output = model_name.startswith(models_that_support_structured_output)
    return replace(
        BedrockModelProfile(),
        supported_native_tools=frozenset(),
        json_schema_transformer=BedrockJsonSchemaTransformer,
        supports_json_schema_output=supports_structured_output,
        bedrock_supports_strict_tool_definition=supports_structured_output,
    )


def bedrock_nvidia_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for an NVIDIA model used via Bedrock."""
    models_that_support_structured_output = ('nemotron-nano',)
    supports_structured_output = model_name.startswith(models_that_support_structured_output)
    return replace(
        BedrockModelProfile(),
        supported_native_tools=frozenset(),
        json_schema_transformer=BedrockJsonSchemaTransformer,
        supports_json_schema_output=supports_structured_output,
        bedrock_supports_strict_tool_definition=supports_structured_output,
    )


def _without_builtin_tools(profile: ModelProfile | None) -> ModelProfile:
    return replace(profile or BedrockModelProfile(), supported_native_tools=frozenset())


class BedrockProvider(Provider[BaseClient]):
    """Provider for AWS Bedrock."""

    @property
    def name(self) -> str:
        return 'bedrock'

    @property
    def base_url(self) -> str:
        return self._client.meta.endpoint_url

    @property
    def client(self) -> BaseClient:
        return self._client

    @client.setter
    def client(self, client: BaseClient) -> None:
        """Replace the underlying boto3 client.

        Useful for rotating short-lived credentials (e.g. temporary STS credentials) in a long-running service:
        construct a fresh `bedrock-runtime` client and assign it here, and every [`BedrockConverseModel`]
        [pydantic_ai.models.bedrock.BedrockConverseModel] using this provider will pick it up.
        """
        self._client = client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        provider_to_profile: dict[str, Callable[[str], ModelProfile | None]] = {
            'anthropic': bedrock_anthropic_model_profile,
            'mistral': bedrock_mistral_model_profile,
            'cohere': lambda model_name: _without_builtin_tools(cohere_model_profile(model_name)),
            'amazon': bedrock_amazon_model_profile,
            'meta': lambda model_name: _without_builtin_tools(meta_model_profile(model_name)),
            'deepseek': lambda model_name: _without_builtin_tools(bedrock_deepseek_model_profile(model_name)),
            # Converse rejects `reasoning_effort='none'` — mark always-on.
            'openai': lambda _mn: BedrockModelProfile(
                bedrock_thinking_variant='openai',
                supports_thinking=True,
                thinking_always_enabled=True,
            ),
            'qwen': bedrock_qwen_model_profile,
            'google': bedrock_google_model_profile,
            'minimax': bedrock_minimax_model_profile,
            'nvidia': bedrock_nvidia_model_profile,
        }

        # Bedrock model IDs are `<provider>.<model-name>-v<n>(:<m>)?`, optionally with a
        # cross-region inference geo prefix (e.g. `us.anthropic.claude-haiku-4-5-20251001-v1:0`).
        provider, model_name = split_bedrock_model_id(model_name)
        if provider in provider_to_profile:
            return provider_to_profile[provider](model_name)

        return None

    @overload
    def __init__(self, *, bedrock_client: BaseClient) -> None: ...

    @overload
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        region_name: str | None = None,
        profile_name: str | None = None,
        aws_read_timeout: float | None = None,
        aws_connect_timeout: float | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        base_url: str | None = None,
        region_name: str | None = None,
        profile_name: str | None = None,
        aws_read_timeout: float | None = None,
        aws_connect_timeout: float | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        bedrock_client: BaseClient | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        base_url: str | None = None,
        region_name: str | None = None,
        profile_name: str | None = None,
        api_key: str | None = None,
        aws_read_timeout: float | None = None,
        aws_connect_timeout: float | None = None,
    ) -> None:
        """Initialize the Bedrock provider.

        Args:
            bedrock_client: A boto3 client for Bedrock Runtime. If provided, other arguments are ignored.
            aws_access_key_id: The AWS access key ID. If not set, the `AWS_ACCESS_KEY_ID` environment variable will be used if available.
            aws_secret_access_key: The AWS secret access key. If not set, the `AWS_SECRET_ACCESS_KEY` environment variable will be used if available.
            aws_session_token: The AWS session token. If not set, the `AWS_SESSION_TOKEN` environment variable will be used if available.
            api_key: The API key for Bedrock client. Can be used instead of `aws_access_key_id`, `aws_secret_access_key`, and `aws_session_token`. If not set, the `AWS_BEARER_TOKEN_BEDROCK` environment variable will be used if available.
            base_url: The base URL for the Bedrock client.
            region_name: The AWS region name. If not set, the `AWS_DEFAULT_REGION` environment variable will be used if available.
            profile_name: The AWS profile name.
            aws_read_timeout: The read timeout for Bedrock client.
            aws_connect_timeout: The connect timeout for Bedrock client.
        """
        if bedrock_client is not None:
            self._client = bedrock_client
        else:
            read_timeout = aws_read_timeout or float(os.getenv('AWS_READ_TIMEOUT', 300))
            connect_timeout = aws_connect_timeout or float(os.getenv('AWS_CONNECT_TIMEOUT', 60))
            config: dict[str, Any] = {
                'read_timeout': read_timeout,
                'connect_timeout': connect_timeout,
            }
            api_key = api_key or os.getenv('AWS_BEARER_TOKEN_BEDROCK')
            try:
                if api_key is not None:
                    session = boto3.Session(
                        botocore_session=_BearerTokenSession(api_key),
                        region_name=region_name,
                        profile_name=profile_name,
                    )
                    config['signature_version'] = 'bearer'
                else:  # pragma: lax no cover
                    session = boto3.Session(
                        aws_access_key_id=aws_access_key_id,
                        aws_secret_access_key=aws_secret_access_key,
                        aws_session_token=aws_session_token,
                        region_name=region_name,
                        profile_name=profile_name,
                    )
                self._client = session.client(  # type: ignore[reportUnknownMemberType]
                    'bedrock-runtime',
                    config=Config(**config),
                    endpoint_url=base_url,
                )
            except NoRegionError as exc:  # pragma: no cover
                raise UserError('You must provide a `region_name` or a boto3 client for Bedrock Runtime.') from exc


class _BearerTokenSession(Session):
    def __init__(self, token: str):
        super().__init__()
        self.token = token

    def get_auth_token(self, **_kwargs: Any) -> FrozenAuthToken:
        return FrozenAuthToken(self.token)

    def get_credentials(self) -> None:  # type: ignore[reportIncompatibleMethodOverride]
        return None
