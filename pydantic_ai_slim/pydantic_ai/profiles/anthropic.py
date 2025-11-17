from __future__ import annotations as _annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from .._json_schema import JsonSchema, JsonSchemaTransformer
from . import ModelProfile

TransformSchemaFunc = Callable[[Any], JsonSchema]

try:  # pragma: no cover
    _anthropic_module = importlib.import_module('anthropic')
except Exception:
    _anthropic_transform_schema: TransformSchemaFunc | None = None
else:
    _anthropic_transform_schema = cast(TransformSchemaFunc | None, getattr(_anthropic_module, 'transform_schema', None))


@dataclass(init=False)
class AnthropicJsonSchemaTransformer(JsonSchemaTransformer):
    """Transforms schemas to the subset supported by Anthropic structured outputs."""

    def walk(self) -> JsonSchema:
        schema = super().walk()
        helper = _anthropic_transform_schema
        if helper is None:
            return schema
        try:  # pragma: no branch
            # helper may raise if schema already transformed
            transformed = helper(schema)
        except Exception:
            return schema
        if isinstance(transformed, dict):
            return transformed
        return schema

    def transform(self, schema: JsonSchema) -> JsonSchema:
        schema.pop('title', None)
        schema.pop('$schema', None)
        return schema


def anthropic_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for an Anthropic model."""
    return ModelProfile(
        thinking_tags=('<thinking>', '</thinking>'),
        supports_json_schema_output=True,
        json_schema_transformer=AnthropicJsonSchemaTransformer,
    )
