"""Helpers for normalizing Amazon Bedrock model identifiers.

Bedrock model IDs carry a `<provider>.` segment (e.g. `anthropic.`, `amazon.`) and,
on the legacy `InvokeModel`/`Converse` APIs, a `-v<n>(:<m>)?` version suffix and an
optional cross-region inference geo prefix (`us.`, `eu.`, ...): e.g.
`us.anthropic.claude-haiku-4-5-20251001-v1:0`.

These live in a boto3-free module so both [`BedrockProvider`][pydantic_ai.providers.bedrock.BedrockProvider]
(which needs boto3) and [`AnthropicProvider`][pydantic_ai.providers.anthropic.AnthropicProvider]
(which talks to Bedrock via `AsyncAnthropicBedrock`/`AsyncAnthropicBedrockMantle` and doesn't)
can share them.
"""

from __future__ import annotations as _annotations

import re

from ..settings import CacheRetention

__all__ = (
    'BEDROCK_GEO_PREFIXES',
    'bedrock_claude_cache_retentions',
    'bedrock_claude_supports_automatic_caching',
    'bedrock_claude_supports_mid_conversation_effort',
    'bedrock_lacks_claude_structured_output',
    'remove_bedrock_geo_prefix',
    'split_bedrock_model_id',
)

# Known geo prefixes for cross-region inference profile IDs
BEDROCK_GEO_PREFIXES: tuple[str, ...] = ('us', 'eu', 'apac', 'jp', 'au', 'ca', 'in', 'global', 'us-gov')

_VERSION_SUFFIX_RE = re.compile(r'(.+)-v\d+(?::\d+)?$')

# An inference profile or foundation model ARN names the model ID after its resource type, e.g.
# `arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-opus-4-8`.
# https://platform.claude.com/docs/en/build-with-claude/claude-on-amazon-bedrock-legacy#api-model-ids
# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModel.html#API_runtime_InvokeModel_RequestSyntax
_MODEL_ARN_RE = re.compile(r'^arn:[^:]+:bedrock:[^:]*:[^:]*:(?:inference-profile|foundation-model)/(.+)$')


def remove_bedrock_geo_prefix(model_name: str) -> str:
    """Remove the cross-region inference geographic prefix from a model ID if present.

    Bedrock supports cross-region inference using geographic prefixes like
    `us.`, `eu.`, `apac.`, etc. This function strips those prefixes.

    Example:
        `us.amazon.titan-embed-text-v2:0` -> `amazon.titan-embed-text-v2:0`
        `amazon.titan-embed-text-v2:0` -> `amazon.titan-embed-text-v2:0`
    """
    for prefix in BEDROCK_GEO_PREFIXES:
        if model_name.startswith(f'{prefix}.'):
            return model_name.removeprefix(f'{prefix}.')
    return model_name


def split_bedrock_model_id(model_id: str) -> tuple[str | None, str]:
    """Split a Bedrock model ID into its `<provider>` segment and the bare model name.

    Strips any inference profile or foundation model ARN, cross-region inference geo prefix and
    `-v<n>(:<m>)?` version suffix.

    Example:
        `us.anthropic.claude-haiku-4-5-20251001-v1:0` -> `('anthropic', 'claude-haiku-4-5-20251001')`
        `arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-opus-4-8`
            -> `('anthropic', 'claude-opus-4-8')`
        `anthropic.claude-haiku-4-5` -> `('anthropic', 'claude-haiku-4-5')`
        `claude-haiku-4-5` -> `(None, 'claude-haiku-4-5')`
    """
    if arn_match := _MODEL_ARN_RE.match(model_id):
        model_id = arn_match.group(1)
    provider, _, name = remove_bedrock_geo_prefix(model_id).partition('.')
    if not name:  # no `<provider>.` segment
        return None, model_id
    if version_match := _VERSION_SUFFIX_RE.match(name):
        name = version_match.group(1)
    return provider, name


# The Claude models whose Bedrock entry lists both the 5-minute and the 1-hour cache TTL; the others
# (Claude 3.7 Sonnet, Claude 3.5 Sonnet v2) only take the default 5 minutes. Bare names, as
# `split_bedrock_model_id` returns them; a prefix also covers later point releases (`claude-opus-5-5`).
# https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html#prompt-caching-models
_ONE_HOUR_CACHE_CLAUDE_MODEL_PREFIXES = (
    'claude-fable-5',
    'claude-haiku-4-5',
    'claude-mythos-5',
    'claude-opus-4-5',
    'claude-opus-4-6',
    'claude-opus-4-7',
    'claude-opus-4-8',
    'claude-opus-5',
    'claude-sonnet-4-5',
    'claude-sonnet-4-6',
    'claude-sonnet-5',
)


def bedrock_claude_cache_retentions(model_name: str) -> tuple[CacheRetention, ...]:
    """The prompt-cache retention tiers Bedrock supports for a Claude model.

    Accepts a full Bedrock model ID (e.g. `us.anthropic.claude-sonnet-4-5-20250929-v1:0`) or a bare
    model name (e.g. `claude-sonnet-4-5-20250929`).
    """
    _, name = split_bedrock_model_id(model_name)
    return ('5m', '1h') if name.startswith(_ONE_HOUR_CACHE_CLAUDE_MODEL_PREFIXES) else ('5m',)


# The Claude models that the legacy InvokeModel API (`AsyncAnthropicBedrock`) serves from the same infrastructure as
# the Bedrock Messages API, which take Anthropic's top-level automatic caching parameter; the ARN-versioned models
# reject it with a 400 (`cache_control: Extra inputs are not permitted`). Verified live on 2026-10-16 for every
# prefix except `claude-fable-5`, which is documented in the same list. Bare names, as `split_bedrock_model_id`
# returns them; a prefix also covers later point releases (`claude-opus-5-5`).
# https://platform.claude.com/docs/en/build-with-claude/claude-on-amazon-bedrock-legacy#api-model-ids
_AUTOMATIC_CACHING_CLAUDE_MODEL_PREFIXES = (
    'claude-fable-5',
    'claude-haiku-5-5',
    'claude-opus-4-7',
    'claude-opus-4-8',
    'claude-opus-5',
    'claude-sonnet-5',
)


def bedrock_claude_supports_automatic_caching(model_name: str) -> bool:
    """Whether the legacy Bedrock InvokeModel API takes Anthropic's top-level automatic caching parameter for a model.

    Accepts a full Bedrock model ID (e.g. `us.anthropic.claude-opus-4-8`) or a bare model name
    (e.g. `claude-opus-4-8`).
    """
    _, name = split_bedrock_model_id(model_name)
    return name.startswith(_AUTOMATIC_CACHING_CLAUDE_MODEL_PREFIXES)


# The Claude models that Amazon Bedrock serves per-message effort changes for, under the
# `mid-conversation-output-config-2026-07-01` beta. The SDK sends the beta in the `anthropic_beta` body field on the
# legacy InvokeModel API (`AsyncAnthropicBedrock`), and in the `anthropic-beta` header on the Bedrock Messages API
# (`AsyncAnthropicBedrockMantle`). Bare names, as `split_bedrock_model_id` returns them.
# https://platform.claude.com/docs/en/build-with-claude/effort#per-message-effort-beta
_MID_CONVERSATION_EFFORT_CLAUDE_MODEL_PREFIXES = (
    'claude-fable-5-1',
    'claude-opus-5-5',
)


def bedrock_claude_supports_mid_conversation_effort(model_name: str) -> bool:
    """Whether Amazon Bedrock takes a per-message `output_config.effort` for a Claude model, on either Bedrock API.

    Accepts a full Bedrock model ID (e.g. `us.anthropic.claude-opus-5-5`) or a bare model name.
    """
    _, name = split_bedrock_model_id(model_name)
    return name.startswith(_MID_CONVERSATION_EFFORT_CLAUDE_MODEL_PREFIXES)


# The Claude models with structured output (JSON schema output and strict tool use) on the direct Anthropic
# API that Bedrock Runtime doesn't serve it for, on either Converse or InvokeModel. Bare names, as
# `split_bedrock_model_id` returns them; a prefix also covers later point releases.
# https://platform.claude.com/docs/en/build-with-claude/structured-outputs
# https://docs.aws.amazon.com/bedrock/latest/userguide/structured-output.html
_STRUCTURED_OUTPUT_UNSUPPORTED_CLAUDE_MODEL_PREFIXES = (
    'claude-opus-4-1',
    'claude-fable-5',
)


def bedrock_lacks_claude_structured_output(model_name: str) -> bool:
    """Whether Bedrock Runtime withholds the structured output a Claude model has on the direct Anthropic API.

    Accepts a full Bedrock model ID (e.g. `global.anthropic.claude-sonnet-5`) or a bare model name
    (e.g. `claude-sonnet-5`).
    """
    _, name = split_bedrock_model_id(model_name)
    return name.startswith(_STRUCTURED_OUTPUT_UNSUPPORTED_CLAUDE_MODEL_PREFIXES)
