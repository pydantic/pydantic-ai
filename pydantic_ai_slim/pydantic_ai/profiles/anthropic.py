from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Any

from .._json_schema import JsonSchema, JsonSchemaTransformer
from . import ModelProfile

try:
    from anthropic.lib.tools import transform_schema as anthropic_transform_schema  # type: ignore[import-not-found]
except ImportError:
    anthropic_transform_schema = None


@dataclass(init=False)
class AnthropicJsonSchemaTransformer(JsonSchemaTransformer):
    """Transform JSON schemas for Anthropic structured outputs.

    Anthropic requires schemas to not include 'title', '$schema', or 'discriminator' at root level.

    `additionalProperties` is set to false for objects to ensure strict mode compatibility.

    optionally use Anthropic's transform_schema helper for validation

    see: https://docs.claude.com/en/docs/build-with-claude/structured-outputs
    """

    def transform(self, schema: JsonSchema) -> JsonSchema:
        """Apply Anthropic-specific schema transformations.

        Removes 'title', '$schema', and 'discriminator' fields which are not supported by Anthropic API,
        and sets `additionalProperties` to false for objects to ensure strict mode compatibility.

        If available, also validates the schema using Anthropic's transform_schema helper from their SDK.

        Args:
            schema: The JSON schema to transform

        Returns:
            Transformed schema compatible with Anthropic's structured outputs API
        """
        # remove fields not supported by Anthropic
        schema.pop('title', None)
        schema.pop('$schema', None)
        schema.pop('discriminator', None)

        schema_type = schema.get('type')
        if schema_type == 'object':
            schema['additionalProperties'] = False
            if self.strict is True:
                if 'properties' not in schema:
                    schema['properties'] = dict[str, Any]()
                schema['required'] = list(schema['properties'].keys())
            elif self.strict is None:
                if schema.get('additionalProperties', None) not in (None, False):
                    self.is_strict_compatible = False
                else:
                    schema['additionalProperties'] = False

                if 'properties' not in schema or 'required' not in schema:
                    self.is_strict_compatible = False
                else:
                    required = schema['required']
                    for k in schema['properties'].keys():
                        if k not in required:
                            self.is_strict_compatible = False
            else:
                if 'additionalProperties' not in schema:
                    schema['additionalProperties'] = False

        if anthropic_transform_schema is not None:
            try:
                validated_schema = anthropic_transform_schema(schema)  # pyright: ignore[reportUnknownVariableType]
                if isinstance(validated_schema, dict):
                    schema = validated_schema
            except Exception:
                pass

        if self.strict is True:
            self.is_strict_compatible = True

        return schema


def anthropic_model_profile(model_name: str) -> ModelProfile:
    """Get the model profile for an Anthropic model."""
    return ModelProfile(
        json_schema_transformer=AnthropicJsonSchemaTransformer,
        supports_json_schema_output=True,
        supports_json_object_output=True,
        thinking_tags=('<thinking>', '</thinking>'),
    )
