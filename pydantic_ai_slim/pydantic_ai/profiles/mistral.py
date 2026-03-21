from __future__ import annotations as _annotations

from .._json_schema import JsonSchema, JsonSchemaTransformer
from . import ModelProfile


class MistralJsonSchemaTransformer(JsonSchemaTransformer):
    """Transforms JSON schema for Mistral strict mode.

    Mistral strict mode requires ``additionalProperties: false`` on all objects,
    but unlike OpenAI it does not require all properties to be listed as required
    or the removal of ``default`` values.
    """

    def __init__(self, schema: JsonSchema, *, strict: bool | None = None):
        super().__init__(schema, strict=strict)

    def transform(self, schema: JsonSchema) -> JsonSchema:
        if schema.get('type') == 'object':
            if self.strict is True:
                schema['additionalProperties'] = False
            elif self.strict is None:
                current = schema.get('additionalProperties')
                if current is not None and current is not False:
                    self.is_strict_compatible = False
                else:
                    schema['additionalProperties'] = False
        return schema


def mistral_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Mistral model.

    Note: ``json_schema_transformer`` is intentionally not set here because it
    depends on the provider API, not the model.  The native Mistral API needs
    ``MistralJsonSchemaTransformer`` while OpenAI-compatible providers
    (Fireworks, Azure, etc.) need ``OpenAIJsonSchemaTransformer``.  Each
    provider sets the appropriate transformer in its own ``model_profile()``.
    """
    return ModelProfile(
        supports_json_schema_output=True,
        default_structured_output_mode='native',
    )
