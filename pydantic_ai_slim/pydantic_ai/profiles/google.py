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

    As of November 2025, Gemini 2.5+ models support enhanced JSON Schema features
    (see [announcement](https://blog.google/technology/developers/gemini-api-structured-outputs/)) including:
    * `title` for short property descriptions
    * `anyOf` and `oneOf` for conditional structures (unions)
    * `$ref` and `$defs` for recursive schemas and reusable definitions
    * `minimum` and `maximum` for numeric constraints
    * `additionalProperties` for dictionaries
    * `type: 'null'` for optional fields
    * `prefixItems` for tuple-like arrays

    Not supported (empirically tested as of November 2025):
    * `exclusiveMinimum` and `exclusiveMaximum` are not yet supported by the Google SDK
    * `discriminator` field causes validation errors with nested oneOf schemas
    """

    def __init__(self, schema: JsonSchema, *, strict: bool | None = None):
        super().__init__(schema, strict=strict, prefer_inlined_defs=False, simplify_nullable_unions=True)

    def transform(self, schema: JsonSchema) -> JsonSchema:
        # Remove properties not supported by Gemini
        schema.pop('$schema', None)
        if (const := schema.pop('const', None)) is not None:
            # Gemini doesn't support const, but it does support enum with a single value
            schema['enum'] = [const]
        schema.pop('discriminator', None)
        schema.pop('examples', None)

        # Gemini only supports string enums, so we need to convert any enum values to strings.
        # Pydantic will take care of transforming the transformed string values to the correct type.
        if enum := schema.get('enum'):
            schema['type'] = 'string'
            schema['enum'] = [str(val) for val in enum]

        type_ = schema.get('type')
        if type_ == 'string' and (fmt := schema.pop('format', None)):
            description = schema.get('description')
            if description:
                schema['description'] = f'{description} (format: {fmt})'
            else:
                schema['description'] = f'Format: {fmt}'

        # As of November 2025, Gemini 2.5+ models now support:
        # - additionalProperties (for dict types)
        # - $ref (for recursive schemas)
        # - prefixItems (for tuple-like arrays)
        # These are no longer stripped from the schema.

        # Note: exclusiveMinimum/exclusiveMaximum are NOT yet supported by Google SDK,
        # so we still need to strip them
        schema.pop('exclusiveMinimum', None)
        schema.pop('exclusiveMaximum', None)

        return schema
