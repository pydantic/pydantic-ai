from __future__ import annotations as _annotations

from .._json_schema import JsonSchema, JsonSchemaTransformer
from . import ModelProfile


def mistral_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Mistral model."""
    return ModelProfile(
        json_schema_transformer=MistralJsonSchemaTransformer,
        supports_json_schema_output=True,
        supports_json_object_output=True,
    )


class MistralJsonSchemaTransformer(JsonSchemaTransformer):
    """Transforms the JSON Schema from Pydantic to be suitable for Mistral.

    Mistral supports JSON schema for structured output.
    See https://docs.mistral.ai/capabilities/structured-output/custom_structured_output/ for more information.
    """

    def transform(self, schema: JsonSchema) -> JsonSchema:
        # Remove properties not supported by Mistral
        schema.pop('$schema', None)
        schema.pop('title', None)

        # Handle const by converting to enum
        if (const := schema.pop('const', None)) is not None:
            # Mistral doesn't support const, but it does support enum with a single value
            schema['enum'] = [const]

        return schema
