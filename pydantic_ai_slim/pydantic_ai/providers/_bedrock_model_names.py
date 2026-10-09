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
    'remove_bedrock_geo_prefix',
    'split_bedrock_model_id',
)

# Known geo prefixes for cross-region inference profile IDs
BEDROCK_GEO_PREFIXES: tuple[str, ...] = ('us', 'eu', 'apac', 'jp', 'au', 'ca', 'in', 'global', 'us-gov')

_VERSION_SUFFIX_RE = re.compile(r'(.+)-v\d+(?::\d+)?$')


def _bedrock_arn_resource(model_id: str) -> str:
    """The resource segment of a Bedrock ARN, or the ID unchanged if it is not one.

    An inference-profile or foundation-model ARN carries the model ID after the last `/`:
    `arn:aws:bedrock:eu-central-1:<account>:inference-profile/eu.anthropic.claude-sonnet-5-5`.
    A provisioned-model or application-inference-profile ARN instead ends in an opaque ID,
    which has no `<provider>.` segment and so is left for the caller to reject.
    """
    if not model_id.startswith('arn:'):
        return model_id
    _, _, resource = model_id.rpartition('/')
    return resource or model_id


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

    Strips any cross-region inference geo prefix and `-v<n>(:<m>)?` version suffix.

    Also accepts an inference-profile or foundation-model ARN, whose resource segment holds
    the model ID. Without that, partitioning the ARN itself on `.` yields a provider of
    `arn:aws:bedrock:<region>:<account>:inference-profile/eu`, which matches no provider, and
    the caller falls back to default capabilities for a model whose real ones are known.

    Example:
        `us.anthropic.claude-haiku-4-5-20251001-v1:0` -> `('anthropic', 'claude-haiku-4-5-20251001')`
        `anthropic.claude-haiku-4-5` -> `('anthropic', 'claude-haiku-4-5')`
        `claude-haiku-4-5` -> `(None, 'claude-haiku-4-5')`
        `arn:aws:bedrock:eu-central-1:1:inference-profile/eu.anthropic.claude-sonnet-5-5`
            -> `('anthropic', 'claude-sonnet-5-5')`
    """
    provider, _, name = remove_bedrock_geo_prefix(_bedrock_arn_resource(model_id)).partition('.')
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
