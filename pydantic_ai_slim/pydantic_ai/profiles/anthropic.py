from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import assert_never

from .._json_schema import JsonSchema, JsonSchemaTransformer
from . import ModelProfile


def _schema_is_lossless(schema: JsonSchema) -> bool:  # noqa: C901
    """Return True when `anthropic.transform_schema` won't need to drop constraints.

    Anthropic's structured output API only supports a subset of JSON Schema features.
    This function detects whether a schema uses only supported features, allowing us
    to safely enable strict mode for guaranteed server-side validation.

    Checks are performed based on https://docs.claude.com/en/docs/build-with-claude/structured-outputs#how-sdk-transformation-works

    Args:
        schema: JSON Schema dictionary (typically from BaseModel.model_json_schema())

    Returns:
        True if schema is lossless (all constraints preserved), False if lossy

    Examples:
        Lossless schemas (constraints preserved):
        >>> _schema_is_lossless({'type': 'string'})
        True
        >>> _schema_is_lossless({'type': 'object', 'properties': {'name': {'type': 'string'}}})
        True

        Lossy schemas (constraints dropped):
        >>> _schema_is_lossless({'type': 'string', 'minLength': 5})
        False
        >>> _schema_is_lossless({'type': 'array', 'items': {'type': 'string'}, 'minItems': 2})
        False

    Note:
        Some checks handle edge cases that rarely occur with Pydantic-generated schemas:
        - oneOf: Pydantic generates anyOf for Union types
        - Custom formats: Pydantic doesn't expose custom format generation in normal API
    """
    from anthropic.lib._parse._transform import SupportedStringFormats

    def _walk(node: JsonSchema) -> bool:  # noqa: C901
        if not isinstance(node, dict):
            assert_never(False)

        node = node.copy()

        if '$ref' in node:
            node.pop('$ref')
            return not node

        defs = node.get('$defs')
        if defs:
            for value in defs.values():
                if not _walk(value):
                    return False
            node.pop('$defs')

        type_ = node.get('type')
        any_of = node.get('anyOf')
        one_of = node.get('oneOf')
        all_of = node.get('allOf')

        if any_of is not None:
            node.pop('anyOf')
        if one_of is not None:
            node.pop('oneOf')
        if all_of is not None:
            node.pop('allOf')
        if type_ is not None:
            node.pop('type')

        # every sub-schema in the list must itself be lossless -> `all(_walk(item) for item in any_of)`
        # the wrapper object must not have any other unsupported fields -> `and not node`
        if isinstance(any_of, list):
            return all(_walk(item) for item in any_of) and not node  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        if isinstance(one_of, list):  # pragma: no cover
            # pydantic generates anyOf for Union types, leaving this here for JSON schemas that don't come from pydantic.BaseModel
            return all(_walk(item) for item in one_of) and not node  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        if isinstance(all_of, list):  # pragma: no cover
            return all(_walk(item) for item in all_of) and not node  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]

        if type_ is None:  # pragma: no cover
            return False

        if type_ == 'object':
            properties = node.get('properties')
            if properties:
                for value in properties.values():
                    if not _walk(value):
                        return False
                node.pop('properties')
            additional = node.get('additionalProperties')
            if additional not in (None, False):
                return False
            if additional is not None:
                node.pop('additionalProperties')
            if 'required' in node:
                node.pop('required')
        elif type_ == 'array':
            items = node.get('items')
            if items:
                if not _walk(items):
                    return False
                node.pop('items')
            min_items = node.get('minItems')
            if min_items not in (None, 0, 1):
                return False
            if min_items is not None:
                node.pop('minItems')
        elif type_ == 'string':
            format_ = node.get('format')
            if format_ is not None and format_ not in SupportedStringFormats:  # pragma: no cover
                return False
            if format_ is not None:
                node.pop('format')
        elif type_ in {'integer', 'number', 'boolean', 'null'}:
            pass
        else:  # pragma: no cover
            return False

        return not node

    return _walk(schema)


@dataclass(init=False)
class AnthropicJsonSchemaTransformer(JsonSchemaTransformer):
    """Transforms schemas to the subset supported by Anthropic structured outputs."""

    def walk(self) -> JsonSchema:
        from anthropic import transform_schema

        schema = super().walk()
        if self.strict is None and not _schema_is_lossless(schema):
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
