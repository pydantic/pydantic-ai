from __future__ import annotations as _annotations

from copy import deepcopy
from dataclasses import dataclass

from typing_extensions import assert_never

from .._json_schema import JsonSchema, JsonSchemaTransformer
from . import ModelProfile


def anthropic_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for an Anthropic model.

    The profile is set as soon as the model is instantiated.
    """
    models_that_support_json_schema_output = ('claude-sonnet-4-5', 'claude-opus-4-1')
    # anthropic introduced support for both structured outputs and strict tool use
    # https://docs.claude.com/en/docs/build-with-claude/structured-outputs#example-usage
    supports_json_schema_output = model_name.startswith(models_that_support_json_schema_output)
    return ModelProfile(
        thinking_tags=('<thinking>', '</thinking>'),
        supports_json_schema_output=supports_json_schema_output,
        json_schema_transformer=AnthropicJsonSchemaTransformer,
    )


@dataclass(init=False)
class AnthropicJsonSchemaTransformer(JsonSchemaTransformer):
    """Transforms schemas to the subset supported by Anthropic structured outputs.

    The transformer is applied (if it is applied) when the [AnthropicModel.prepare_request](../pydantic_ai_slim/pydantic_ai/models/anthropic.py) is called.

    Anthropic's SDK `transform_schema()` automatically:
    - Adds `additionalProperties: false` to all objects (required by API)
    - Removes unsupported constraints (minLength, pattern, etc.)
    - Moves removed constraints to description field
    - Removes title and $schema fields

    When `strict=None`, we compare before/after to detect if constraints were dropped.
    """

    def walk(self) -> JsonSchema:
        from anthropic import transform_schema

        schema = super().walk()

        # no transformation if specifically non-strict
        if self.strict is False:
            return schema
        else:
            transformed = transform_schema(schema)
            has_lossy_changes = self._has_lossy_changes(schema, transformed)
            self.is_strict_compatible = not has_lossy_changes

            # this is the default
            if self.strict is None:
                # is_strict_compatible sets the ToolDefinition.strict value when tool_def.strict is None
                return transformed if self.is_strict_compatible else schema
            else:
                # strict=True
                # self.is_strict_compatible won't be used by caller bc strict is explicit
                return transformed

    def transform(self, schema: JsonSchema) -> JsonSchema:
        schema.pop('title', None)
        schema.pop('$schema', None)
        return schema

    @staticmethod
    def _has_lossy_changes(before: JsonSchema, after: JsonSchema) -> bool:  # noqa: C901
        """Check if transformation dropped validation constraints.

        Safe changes that don't count as lossy:
        - Adding additionalProperties: false
        - Removing title, $schema, or other metadata fields
        - Reordering keys

        Lossy changes:
        - Removing validation constraints (minLength, pattern, minimum, etc.)
        - Changing constraint values
        - Moving constraints to description field
        """

        def normalize(schema: JsonSchema) -> JsonSchema:
            """Remove fields that are safe to add/remove."""
            normalized = deepcopy(schema)
            normalized.pop('additionalProperties', None)
            normalized.pop('title', None)
            normalized.pop('$schema', None)
            return normalized

        def has_lossy_object_changes(before_obj: JsonSchema, after_obj: JsonSchema) -> bool:
            """Recursively check for lossy changes in object schemas.

            Returns:
                True if validation constraints were removed or modified (lossy changes detected).
                False if all validation constraints are preserved (no lossy changes).
            """
            validation_keys = {
                'minLength',
                'maxLength',
                'pattern',
                'format',
                'minimum',
                'maximum',
                'exclusiveMinimum',
                'exclusiveMaximum',
                'minItems',
                'maxItems',
                'uniqueItems',
                'minProperties',
                'maxProperties',
            }

            for key in validation_keys:
                if key in before_obj and key not in after_obj:
                    return True
                # should never happen that an sdk modifies a constraint value
                if key in before_obj and key in after_obj and before_obj[key] != after_obj[key]:
                    return True  # pragma: no cover

            before_props = before_obj.get('properties', {})
            after_props = after_obj.get('properties', {})
            for prop_name, before_prop in before_props.items():
                if prop_name in after_props:  # pragma: no branch
                    if has_lossy_schema_changes(before_prop, after_props[prop_name]):
                        return True

            if 'items' in before_obj and 'items' in after_obj:
                if has_lossy_schema_changes(before_obj['items'], after_obj['items']):
                    return True

            before_defs = before_obj.get('$defs', {})
            after_defs = after_obj.get('$defs', {})
            for def_name, before_def in before_defs.items():
                if def_name in after_defs:  # pragma: no branch
                    if has_lossy_schema_changes(before_def, after_defs[def_name]):  # pragma: no branch
                        return True

            return False

        def has_lossy_schema_changes(before_schema: JsonSchema, after_schema: JsonSchema) -> bool:
            """Check a single schema object for lossy changes.

            Returns:
                True if validation constraints were removed or modified (lossy changes detected).
                False if all validation constraints are preserved (no lossy changes).
            """
            if isinstance(before_schema, dict) and isinstance(after_schema, dict):
                return has_lossy_object_changes(before_schema, after_schema)
            # schemas should always be dicts
            assert_never(False)

        return has_lossy_schema_changes(normalize(before), normalize(after))
