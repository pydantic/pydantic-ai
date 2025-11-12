from __future__ import annotations as _annotations

from .._json_schema import JsonSchema, JsonSchemaTransformer
from . import ModelProfile


def google_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Google model."""
    is_image_model = 'image' in model_name
    return ModelProfile(
        json_schema_transformer=GoogleJsonSchemaTransformer,
        supports_image_output=is_image_model,
        supports_json_schema_output=not is_image_model,
        supports_json_object_output=not is_image_model,
        supports_tools=not is_image_model,
    )


class GoogleJsonSchemaTransformer(JsonSchemaTransformer):
    """Transforms the JSON Schema from Pydantic to be suitable for Gemini.

    Gemini supports [a subset of OpenAPI v3.0.3](https://ai.google.dev/gemini-api/docs/function-calling#function_declarations).
    """

    def __init__(self, schema: JsonSchema, *, strict: bool | None = None):
        # prefer_inlined_defs defaults to False (native $ref/$defs support)
        # simplify_nullable_unions defaults to False (Google now supports type: 'null' natively per Nov 2025 announcement)
        super().__init__(schema, strict=strict)

    def transform(self, schema: JsonSchema) -> JsonSchema:
        # Remove properties not supported by Gemini
        schema.pop('$schema', None)
        if (const := schema.pop('const', None)) is not None:
            # Gemini doesn't support const, but it does support enum with a single value
            schema['enum'] = [const]
        schema.pop('discriminator', None)
        schema.pop('examples', None)

        type_ = schema.get('type')
        if type_ == 'string' and (fmt := schema.pop('format', None)):
            description = schema.get('description')
            if description:
                schema['description'] = f'{description} (format: {fmt})'
            else:
                schema['description'] = f'Format: {fmt}'

        # Note: exclusiveMinimum/exclusiveMaximum are NOT yet supported
        schema.pop('exclusiveMinimum', None)
        schema.pop('exclusiveMaximum', None)

        return schema
