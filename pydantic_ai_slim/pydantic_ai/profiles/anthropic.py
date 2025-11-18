from __future__ import annotations as _annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .._json_schema import JsonSchema, JsonSchemaTransformer
from . import ModelProfile

TransformSchemaFunc = Callable[[Any], JsonSchema]


@dataclass(init=False)
class AnthropicJsonSchemaTransformer(JsonSchemaTransformer):
    """Transforms schemas to the subset supported by Anthropic structured outputs."""

    def walk(self) -> JsonSchema:
        from anthropic import transform_schema

        schema = super().walk()
        transformed = transform_schema(schema)
        return transformed

    def transform(self, schema: JsonSchema) -> JsonSchema:
        # for consistency with other transformers (openai,google)
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
