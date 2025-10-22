"""Utilities for Vercel AI protocol.

Converted to Python from:
https://github.com/vercel/ai/blob/ai%405.0.34/packages/ai/src/ui/ui-messages.ts
"""

from abc import ABC
from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

__all__ = ('CamelBaseModel', 'ProviderMetadata', 'JSONValue')

# Technically this is recursive union of JSON types; for simplicity, we call it Any
JSONValue = Any
ProviderMetadata = dict[str, dict[str, JSONValue]]


class CamelBaseModel(BaseModel, ABC):
    """Base model with camelCase aliases."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra='forbid')
