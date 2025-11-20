from __future__ import annotations as _annotations

from dataclasses import dataclass

from .._json_schema import JsonSchema, JsonSchemaTransformer
from . import ModelProfile


def _schema_is_lossless(schema: JsonSchema) -> bool:  # noqa: C901
    """Return True when `anthropic.transform_schema` won't need to drop constraints.

    Checks are performed based on https://docs.claude.com/en/docs/build-with-claude/structured-outputs#how-sdk-transformation-works
    """
    from anthropic.lib._parse._transform import SupportedStringFormats

    def _walk(node: JsonSchema) -> bool:  # noqa: C901
        if not isinstance(node, dict):  # pragma: no cover
            return False

        node = dict(node)

        if '$ref' in node:
            node.pop('$ref')
            return not node

        defs = node.pop('$defs', None)
        if defs:
            for value in defs.values():
                if not _walk(value):
                    return False

        type_ = node.pop('type', None)
        any_of = node.pop('anyOf', None)
        one_of = node.pop('oneOf', None)
        all_of = node.pop('allOf', None)

        node.pop('description', None)
        node.pop('title', None)

        # every sub-schema in the list must itself be lossless -> `all(_walk(item) for item in any_of)`
        # the wrapper object must not have any other unsupported fields -> `and not node`
        if isinstance(any_of, list):
            return all(_walk(item) for item in any_of) and not node  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        if isinstance(one_of, list):  # pragma: no cover
            # pydantic generates anyOf for Union types, leaving this here for JSON schemas that don't come from pydantic.BaseModel
            return all(_walk(item) for item in one_of) and not node  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        if isinstance(all_of, list):
            return all(_walk(item) for item in all_of) and not node  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]

        if type_ is None:
            return False

        if type_ == 'object':
            properties = node.pop('properties', None)
            if properties:
                for value in properties.values():
                    if not _walk(value):
                        return False
            additional = node.pop('additionalProperties', None)
            if additional not in (None, False):
                return False
            node.pop('required', None)
        elif type_ == 'array':
            items = node.pop('items', None)
            if items and not _walk(items):
                return False
            min_items = node.pop('minItems', None)
            if min_items not in (None, 0, 1):
                return False
        elif type_ == 'string':
            format_ = node.pop('format', None)
            if format_ is not None and format_ not in SupportedStringFormats:  # pragma: no cover
                return False
        elif type_ in {'integer', 'number', 'boolean', 'null'}:
            pass
        else:
            return False

        return not node

    return _walk(schema)


@dataclass(init=False)
class AnthropicJsonSchemaTransformer(JsonSchemaTransformer):
    """Transforms schemas to the subset supported by Anthropic structured outputs."""

    def walk(self) -> JsonSchema:
        from anthropic import transform_schema

        schema = super().walk()
        if self.is_strict_compatible and not _schema_is_lossless(schema):
            # check compatibility before calling anthropic's transformer
            # so we don't auto-enable strict when the SDK would drop constraints
            self.is_strict_compatible = False
        transformed = transform_schema(schema)
        return transformed

    def transform(self, schema: JsonSchema) -> JsonSchema:
        # for consistency with other transformers (openai,google)
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
