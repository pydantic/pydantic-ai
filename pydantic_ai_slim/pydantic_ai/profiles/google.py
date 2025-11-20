from __future__ import annotations as _annotations

from dataclasses import dataclass

from .._json_schema import JsonSchema, JsonSchemaTransformer
from . import ModelProfile


@dataclass(kw_only=True)
class GoogleModelProfile(ModelProfile):
    """Profile for models used with `GoogleModel`.

    ALL FIELDS MUST BE `google_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    google_supports_native_output_with_builtin_tools: bool = False
    """Whether the model supports native output with builtin tools.
    See https://ai.google.dev/gemini-api/docs/structured-output?example=recipe#structured_outputs_with_tools"""


def google_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Google model."""
    is_image_model = 'image' in model_name
    is_3_or_newer = 'gemini-3' in model_name
    return GoogleModelProfile(
        json_schema_transformer=GoogleJsonSchemaTransformer,
        supports_image_output=is_image_model,
        supports_json_schema_output=not is_image_model,
        supports_json_object_output=not is_image_model,
        supports_tools=not is_image_model,
        google_supports_native_output_with_builtin_tools=is_3_or_newer,
    )


class GoogleJsonSchemaTransformer(JsonSchemaTransformer):
    """Transforms the JSON Schema from Pydantic to be suitable for Gemini.

    Gemini supports [a subset of OpenAPI v3.0.3](https://ai.google.dev/gemini-api/docs/function-calling#function_declarations).
    """

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
