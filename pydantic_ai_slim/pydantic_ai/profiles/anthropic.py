from __future__ import annotations as _annotations

from dataclasses import dataclass

from .._json_schema import JsonSchema, JsonSchemaTransformer
from . import ModelProfile

ANTHROPIC_MODELS_THAT_SUPPORT_JSON_SCHEMA_OUTPUT = ('claude-sonnet-4-5', 'claude-opus-4-1', 'claude-opus-4-5')
"""These models support both structured outputs and strict tool calling."""
# TODO update when new models are released that support structured outputs
# https://docs.claude.com/en/docs/build-with-claude/structured-outputs#example-usage


def anthropic_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for an Anthropic model."""
    supports_json_schema_output = model_name.startswith(ANTHROPIC_MODELS_THAT_SUPPORT_JSON_SCHEMA_OUTPUT)
    return ModelProfile(
        thinking_tags=('<thinking>', '</thinking>'),
        supports_json_schema_output=supports_json_schema_output,
        json_schema_transformer=AnthropicJsonSchemaTransformer,
    )


@dataclass(init=False)
class AnthropicJsonSchemaTransformer(JsonSchemaTransformer):
    """Transforms schemas to the subset supported by Anthropic structured outputs.

    The transformer is generally called by [AnthropicModel.prepare_request](../pydantic_ai_slim/pydantic_ai/models/anthropic.py).

    Anthropic's SDK `transform_schema()` automatically:
    - Adds `additionalProperties: false` to all objects (required by API)
    - Removes unsupported constraints (minLength, pattern, etc.)
    - Moves removed constraints to description field
    - Removes title and $schema fields
    """

    def walk(self) -> JsonSchema:
        from anthropic import transform_schema

        schema = super().walk()

        # NOTE: The caller (pydantic_ai.models._customize_tool_def or _customize_output_object) will coalesce
        # - tool_def.strict = self.is_strict_compatible
        # - output_object.strict = self.is_strict_compatible
        # we need to set it to False if we're not transforming, otherwise anthropic's API will reject the request
        # if you want this behavior to change, please comment in this issue:
        # https://github.com/pydantic/pydantic-ai/issues/3541
        self.is_strict_compatible = self.strict is True  # not compatible is strict is False/None

        return transform_schema(schema) if self.strict is True else schema

    def transform(self, schema: JsonSchema) -> JsonSchema:
        schema.pop('title', None)
        schema.pop('$schema', None)
        return schema
