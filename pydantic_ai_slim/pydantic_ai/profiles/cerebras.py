from __future__ import annotations as _annotations

import warnings

from pydantic_ai.exceptions import UserError

from ._json_schema import JsonSchema, JsonSchemaTransformer


class CerebrasJsonSchemaTransformer(JsonSchemaTransformer):
    """Transforms the JSON Schema from Pydantic to be suitable for Cerebras.

    Cerebras supports a subset of OpenAI's structured output capabilities, which is documented here:
    - https://inference-docs.cerebras.ai/capabilities/structured-outputs#advanced-schema-features
    - https://inference-docs.cerebras.ai/capabilities/structured-outputs#variations-from-openai’s-structured-output-capabilities
    - https://inference-docs.cerebras.ai/capabilities/tool-use

    TODO: `transform` method is based on GoogleJsonSchemaTransformer, and it doesn't handle all cases mentioned in linkes above.
    """

    def __init__(self, schema: JsonSchema, *, strict: bool | None = None):
        super().__init__(schema, strict=strict, prefer_inlined_defs=True, simplify_nullable_unions=False)

    def transform(self, schema: JsonSchema) -> JsonSchema:
        additional_properties = schema.pop(
            'additionalProperties', None
        )  # don't pop yet so it's included in the warning
        if additional_properties:
            original_schema = {**schema, 'additionalProperties': additional_properties}
            warnings.warn(
                '`additionalProperties` is not supported by Cerebras; it will be removed from the tool JSON schema.'
                f' Full schema: {self.schema}\n\n'
                f'Source of additionalProperties within the full schema: {original_schema}\n\n'
                'If this came from a field with a type like `dict[str, MyType]`, that field will always be empty.\n\n'
                "If Google's APIs are updated to support this properly, please create an issue on the PydanticAI GitHub"
                ' and we will fix this behavior.',
                UserWarning,
            )

        schema.pop('title', None)
        schema.pop('default', None)
        schema.pop('$schema', None)
        if (const := schema.pop('const', None)) is not None:  # pragma: no cover
            schema['enum'] = [const]
        schema.pop('discriminator', None)
        schema.pop('examples', None)

        # TODO: Should we use the trick from pydantic_ai.models.openai._OpenAIJsonSchema
        #   where we add notes about these properties to the field description?
        schema.pop('exclusiveMaximum', None)
        schema.pop('exclusiveMinimum', None)

        # Pydantic will take care of transforming the transformed string values to the correct type.
        if enum := schema.get('enum'):
            schema['type'] = 'string'
            schema['enum'] = [str(val) for val in enum]

        type_ = schema.get('type')
        if 'oneOf' in schema and 'type' not in schema:  # pragma: no cover
            # This gets hit when we have a discriminated union
            # Changing the oneOf to an anyOf prevents the API error and I think is functionally equivalent
            schema['anyOf'] = schema.pop('oneOf')

        if type_ == 'string' and (fmt := schema.pop('format', None)):
            description = schema.get('description')
            if description:
                schema['description'] = f'{description} (format: {fmt})'
            else:
                schema['description'] = f'Format: {fmt}'

        if '$ref' in schema:
            raise UserError(f'Recursive `$ref`s in JSON Schema are not supported by Cerebras: {schema["$ref"]}')

        if 'prefixItems' in schema:
            # prefixItems is not currently supported in Cerebras, so we convert it to items for best compatibility
            prefix_items = schema.pop('prefixItems')
            items = schema.get('items')
            unique_items = [items] if items is not None else []
            for item in prefix_items:
                if item not in unique_items:
                    unique_items.append(item)
            if len(unique_items) > 1:  # pragma: no cover
                schema['items'] = {'anyOf': unique_items}
            elif len(unique_items) == 1:  # pragma: no branch
                schema['items'] = unique_items[0]
            schema.setdefault('minItems', len(prefix_items))
            if items is None:  # pragma: no branch
                schema.setdefault('maxItems', len(prefix_items))

        return schema
