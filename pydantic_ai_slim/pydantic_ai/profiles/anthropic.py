from __future__ import annotations as _annotations

from copy import deepcopy
from dataclasses import dataclass

from .._json_schema import JsonSchema, JsonSchemaTransformer
from . import ModelProfile


@dataclass(init=False)
class AnthropicJsonSchemaTransformer(JsonSchemaTransformer):
    """Transforms schemas to the subset supported by Anthropic structured outputs.

    Anthropic's SDK `transform_schema()` automatically:
    - Adds `additionalProperties: false` to all objects (required by API)
    - Removes unsupported constraints (minLength, pattern, etc.)
    - Moves removed constraints to description field
    - Removes title and $schema fields

    When `strict=None`, we compare before/after to detect if constraints were dropped.
    """

    def walk(self) -> JsonSchema:
        from anthropic import transform_schema

        schema = super().walk()

        if self.strict is False:
            # no transformation if specifically non-strict
            return schema

        if self.strict is None:
            before = deepcopy(schema)
            transformed = transform_schema(schema)
            if before != transformed:
                self.is_strict_compatible = False
            return transformed

        return transform_schema(schema)

    def transform(self, schema: JsonSchema) -> JsonSchema:
        schema.pop('title', None)
        schema.pop('$schema', None)
        return schema


def anthropic_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for an Anthropic model."""
    models_that_support_json_schema_output = ('claude-sonnet-4-5', 'claude-opus-4-1')
    # anthropic introduced support for both structured outputs and strict tool use
    # https://docs.claude.com/en/docs/build-with-claude/structured-outputs#example-usage
    supports_json_schema_output = model_name.startswith(models_that_support_json_schema_output)
    return ModelProfile(
        thinking_tags=('<thinking>', '</thinking>'),
        supports_json_schema_output=supports_json_schema_output,
        json_schema_transformer=AnthropicJsonSchemaTransformer,
    )
