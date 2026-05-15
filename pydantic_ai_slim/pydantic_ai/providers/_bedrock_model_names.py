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

__all__ = ('BEDROCK_GEO_PREFIXES', 'remove_bedrock_geo_prefix', 'split_bedrock_model_id')

# Known geo prefixes for cross-region inference profile IDs
BEDROCK_GEO_PREFIXES: tuple[str, ...] = ('us', 'eu', 'apac', 'jp', 'au', 'ca', 'global', 'us-gov')

_VERSION_SUFFIX_RE = re.compile(r'(.+)-v\d+(?::\d+)?$')


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

    Example:
        `us.anthropic.claude-haiku-4-5-20251001-v1:0` -> `('anthropic', 'claude-haiku-4-5-20251001')`
        `anthropic.claude-haiku-4-5` -> `('anthropic', 'claude-haiku-4-5')`
        `claude-haiku-4-5` -> `(None, 'claude-haiku-4-5')`
    """
    provider, _, name = remove_bedrock_geo_prefix(model_id).partition('.')
    if not name:  # no `<provider>.` segment
        return None, model_id
    if version_match := _VERSION_SUFFIX_RE.match(name):
        name = version_match.group(1)
    return provider, name
