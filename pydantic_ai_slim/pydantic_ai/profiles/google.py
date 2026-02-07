from __future__ import annotations as _annotations

from dataclasses import dataclass

from .._json_schema import JsonSchema, JsonSchemaTransformer
from . import ModelProfile

GOOGLE_NATIVE_TOOL_RETURN_MIME_TYPES: tuple[str, ...] = (
    'image/png',
    'image/jpeg',
    'image/webp',
    'application/pdf',
    'text/plain',
)
"""MIME types supported in native FunctionResponseDict.parts for Gemini 3+.
See https://ai.google.dev/gemini-api/docs/function-calling?example=meeting#multimodal"""


@dataclass(kw_only=True)
class GoogleModelProfile(ModelProfile):
    """Profile for models used with `GoogleModel`.

    ALL FIELDS MUST BE `google_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    google_supports_native_output_with_builtin_tools: bool = False
    """Whether the model supports native output with builtin tools.
    See https://ai.google.dev/gemini-api/docs/structured-output?example=recipe#structured_outputs_with_tools"""

    google_supported_mime_types_in_tool_returns: tuple[str, ...] = ()
    """MIME types supported in native FunctionResponseDict.parts.
    See https://ai.google.dev/gemini-api/docs/function-calling#multimodal-function-responses"""


def google_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Google model."""
    is_image_model = 'image' in model_name
    is_3_or_newer = 'gemini-3' in model_name
    return GoogleModelProfile(
        json_schema_transformer=GoogleJsonSchemaTransformer,
        supports_image_output=is_image_model,
        supports_json_schema_output=is_3_or_newer or not is_image_model,
        supports_json_object_output=is_3_or_newer or not is_image_model,
        supports_tools=not is_image_model,
        google_supports_native_output_with_builtin_tools=is_3_or_newer,
        google_supported_mime_types_in_tool_returns=GOOGLE_NATIVE_TOOL_RETURN_MIME_TYPES if is_3_or_newer else (),
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
            # If type is not present, infer it from the const value for Gemini API compatibility
            if 'type' not in schema:
                if isinstance(const, str):
                    schema['type'] = 'string'
                elif isinstance(const, bool):
                    # bool must be checked before int since bool is a subclass of int in Python
                    schema['type'] = 'boolean'
                elif isinstance(const, int):
                    schema['type'] = 'integer'
                elif isinstance(const, float):
                    schema['type'] = 'number'
        schema.pop('discriminator', None)
        schema.pop('examples', None)

        # Remove 'title' due to https://github.com/googleapis/python-genai/issues/1732
        schema.pop('title', None)

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
